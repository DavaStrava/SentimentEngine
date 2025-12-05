"""Stream Input Manager for multimedia stream ingestion and frame distribution"""

import asyncio
import logging
import time
import pickle
from pathlib import Path
from typing import Optional

import av
import numpy as np
import redis.asyncio as redis

from src.models.frames import AudioFrame, VideoFrame
from src.models.enums import StreamProtocol, StreamConnection
from src.config.config_loader import config


logger = logging.getLogger(__name__)


class StreamInputManager:
    """Manages multimedia stream ingestion, decoding, and frame distribution
    
    Handles connection to various stream sources (local files, RTMP, HLS, WebRTC),
    decodes audio and video streams, and publishes frames to Redis Streams for
    asynchronous consumption by analysis modules.
    
    Requirements:
        - Req 1.1: Begin processing within 2 seconds of stream initiation
        - Req 8.1: Support common streaming protocols (RTMP, HLS, WebRTC, local files)
        - Req 8.2: Decode audio and video streams using appropriate codecs
        - Req 9.1: End-to-end latency not exceeding 3 seconds
    """
    
    def __init__(self):
        self.connection: Optional[StreamConnection] = None
        self.redis_client: Optional[redis.Redis] = None
        self.container: Optional[av.container.InputContainer] = None
        self.audio_stream = None
        self.video_stream = None
        self.start_time: float = 0.0
        self.frame_count: int = 0
        self.is_streaming: bool = False
        
        # Load configuration
        self.audio_sample_rate = config.get('audio.sample_rate', 16000)
        self.frame_duration = config.get('audio.frame_duration', 0.5)
        self.buffer_size = config.get('stream.buffer_size', 100)
        self.redis_url = config.get('redis.url', 'redis://localhost:6379')
        self.audio_stream_name = config.get('redis.audio_stream', 'audio_frames')
        self.video_stream_name = config.get('redis.video_stream', 'video_frames')
        
        # Codec support configuration
        self.supported_audio_codecs = config.get('stream.supported_audio_codecs', [])
        self.supported_video_codecs = config.get('stream.supported_video_codecs', [])
        
        # Adaptive processing configuration
        self.adaptive_enabled = config.get('stream.adaptive_processing.enabled', True)
        self.quality_thresholds = config.get('stream.adaptive_processing.quality_thresholds', {})
        self.frame_skip_by_quality = config.get('stream.adaptive_processing.frame_skip_by_quality', {})
        self.min_resolution = config.get('stream.adaptive_processing.min_resolution', {})
        self.min_audio_bitrate = config.get('stream.adaptive_processing.min_audio_bitrate', 32000)
        self.target_audio_bitrate = config.get('stream.adaptive_processing.target_audio_bitrate', 128000)
        
        # Adaptive processing state
        self.current_video_quality: float = 1.0
        self.current_audio_quality: float = 1.0
        self.adaptive_frame_skip: int = 1
        self.video_frame_counter: int = 0
    
    def _validate_codec(self, codec_name: str, codec_type: str) -> bool:
        """Validate if codec is supported
        
        Args:
            codec_name: Name of the codec
            codec_type: Type of codec ('audio' or 'video')
            
        Returns:
            True if codec is supported, False otherwise
            
        Requirements:
            - Req 8.2: Decode audio and video streams using appropriate codecs
        """
        if codec_type == 'audio':
            supported = self.supported_audio_codecs
        elif codec_type == 'video':
            supported = self.supported_video_codecs
        else:
            return False
        
        # If no specific codecs configured, accept all
        if not supported:
            return True
        
        return codec_name in supported
    
    def _assess_audio_quality(self, audio_stream) -> float:
        """Assess audio stream quality based on bitrate and codec
        
        Args:
            audio_stream: PyAV audio stream
            
        Returns:
            Quality score from 0.0 to 1.0
            
        Requirements:
            - Req 8.3: Adapt processing parameters to maintain analysis accuracy
        """
        try:
            # Get bitrate (may be None for some streams)
            bitrate = audio_stream.bit_rate or audio_stream.codec_context.bit_rate
            
            if bitrate is None:
                # No bitrate info, assume medium quality
                return 0.7
            
            # Calculate quality based on bitrate
            if bitrate >= self.target_audio_bitrate:
                quality = 1.0
            elif bitrate >= self.min_audio_bitrate:
                # Linear interpolation between min and target
                quality = 0.5 + 0.5 * (bitrate - self.min_audio_bitrate) / (self.target_audio_bitrate - self.min_audio_bitrate)
            else:
                # Below minimum, scale down
                quality = 0.5 * (bitrate / self.min_audio_bitrate)
            
            return min(1.0, max(0.0, quality))
            
        except Exception as e:
            logger.warning(f"Failed to assess audio quality: {e}")
            return 0.7  # Default to medium quality
    
    def _assess_video_quality(self, video_stream) -> float:
        """Assess video stream quality based on resolution and bitrate
        
        Args:
            video_stream: PyAV video stream
            
        Returns:
            Quality score from 0.0 to 1.0
            
        Requirements:
            - Req 8.3: Adapt processing parameters to maintain analysis accuracy
        """
        try:
            width = video_stream.width
            height = video_stream.height
            
            min_width = self.min_resolution.get('width', 320)
            min_height = self.min_resolution.get('height', 240)
            
            # Base quality on resolution
            if width >= 1920 and height >= 1080:  # Full HD or better
                quality = 1.0
            elif width >= 1280 and height >= 720:  # HD
                quality = 0.9
            elif width >= 640 and height >= 480:  # SD
                quality = 0.7
            elif width >= min_width and height >= min_height:  # Minimum acceptable
                quality = 0.5
            else:  # Below minimum
                quality = 0.3
            
            # Adjust for bitrate if available
            bitrate = video_stream.bit_rate or video_stream.codec_context.bit_rate
            if bitrate:
                # Higher bitrate improves quality
                if bitrate < 500000:  # < 500 kbps
                    quality *= 0.8
                elif bitrate > 2000000:  # > 2 Mbps
                    quality = min(1.0, quality * 1.1)
            
            return min(1.0, max(0.0, quality))
            
        except Exception as e:
            logger.warning(f"Failed to assess video quality: {e}")
            return 0.7  # Default to medium quality
    
    def _update_adaptive_parameters(self) -> None:
        """Update adaptive processing parameters based on stream quality
        
        Adjusts frame skip rate based on current video quality to maintain
        analysis accuracy while managing computational load.
        
        Requirements:
            - Req 8.3: Adapt processing parameters to maintain analysis accuracy
        """
        if not self.adaptive_enabled:
            self.adaptive_frame_skip = 1
            return
        
        # Determine quality level
        quality = self.current_video_quality
        
        if quality >= self.quality_thresholds.get('high', 0.8):
            quality_level = 'high'
        elif quality >= self.quality_thresholds.get('medium', 0.5):
            quality_level = 'medium'
        else:
            quality_level = 'low'
        
        # Update frame skip
        self.adaptive_frame_skip = self.frame_skip_by_quality.get(quality_level, 1)
        
        logger.info(f"Adaptive processing: quality={quality:.2f} ({quality_level}), frame_skip={self.adaptive_frame_skip}")
    
    def connect(self, stream_url: str, protocol: StreamProtocol) -> StreamConnection:
        """Connect to a multimedia stream source
        
        Args:
            stream_url: URL or file path of the stream
            protocol: Streaming protocol to use
            
        Returns:
            StreamConnection object with connection details
            
        Raises:
            FileNotFoundError: If local file doesn't exist
            ConnectionError: If stream connection fails
        """
        logger.info(f"Connecting to stream: {stream_url} (protocol: {protocol.value})")
        
        # Validate file exists for FILE protocol
        if protocol == StreamProtocol.FILE:
            if not Path(stream_url).exists():
                raise FileNotFoundError(f"Video file not found: {stream_url}")
        
        try:
            # Open container using PyAV
            self.container = av.open(stream_url)
            
            # Find audio and video streams
            self.audio_stream = None
            self.video_stream = None
            
            for stream in self.container.streams:
                if stream.type == 'audio' and self.audio_stream is None:
                    self.audio_stream = stream
                elif stream.type == 'video' and self.video_stream is None:
                    self.video_stream = stream
            
            # Get codec information
            audio_codec = self.audio_stream.codec_context.name if self.audio_stream else ""
            video_codec = self.video_stream.codec_context.name if self.video_stream else ""
            
            # Validate codecs (Req 8.2)
            if audio_codec and not self._validate_codec(audio_codec, 'audio'):
                logger.warning(f"Audio codec '{audio_codec}' may not be fully supported")
            
            if video_codec and not self._validate_codec(video_codec, 'video'):
                logger.warning(f"Video codec '{video_codec}' may not be fully supported")
            
            # Assess stream quality (Req 8.3)
            if self.audio_stream:
                self.current_audio_quality = self._assess_audio_quality(self.audio_stream)
                logger.info(f"Audio quality score: {self.current_audio_quality:.2f}")
            
            if self.video_stream:
                self.current_video_quality = self._assess_video_quality(self.video_stream)
                logger.info(f"Video quality score: {self.current_video_quality:.2f}")
            
            # Update adaptive parameters based on quality
            self._update_adaptive_parameters()
            
            logger.info(f"Stream opened - Audio codec: {audio_codec}, Video codec: {video_codec}")
            
            # Create connection object
            self.connection = StreamConnection(
                url=stream_url,
                protocol=protocol,
                is_active=True,
                audio_codec=audio_codec,
                video_codec=video_codec
            )
            
            return self.connection
            
        except Exception as e:
            logger.error(f"Failed to connect to stream: {e}")
            raise ConnectionError(f"Stream connection failed: {e}")
    
    async def start_streaming(self) -> None:
        """Start streaming and publish frames to Redis
        
        Continuously reads frames from the stream, decodes them, and publishes
        to Redis Streams for asynchronous consumption by analysis modules.
        
        Requirements:
            - Req 1.1: Begin processing within 2 seconds
            - Req 9.1: Maintain low latency
        """
        if not self.connection or not self.connection.is_active:
            raise RuntimeError("No active stream connection")
        
        # Initialize Redis connection
        self.redis_client = redis.from_url(self.redis_url)
        
        logger.info("Starting stream processing...")
        self.start_time = time.time()
        self.frame_count = 0
        self.is_streaming = True
        
        try:
            # Process frames from container
            for packet in self.container.demux():
                if not self.is_streaming:
                    break
                
                # Decode audio packets
                if packet.stream.type == 'audio' and self.audio_stream:
                    for frame in packet.decode():
                        await self._process_audio_frame(frame)
                
                # Decode video packets
                elif packet.stream.type == 'video' and self.video_stream:
                    for frame in packet.decode():
                        await self._process_video_frame(frame)
        
        except Exception as e:
            logger.error(f"Error during streaming: {e}", exc_info=True)
            raise
        
        finally:
            logger.info("Stream processing stopped")
            self.is_streaming = False
    
    async def _process_audio_frame(self, av_frame: av.AudioFrame) -> None:
        """Process and publish audio frame
        
        Args:
            av_frame: PyAV audio frame
        """
        try:
            # Convert to numpy array and resample to target sample rate
            audio_array = av_frame.to_ndarray()
            
            # Handle multi-channel audio (convert to mono)
            if len(audio_array.shape) > 1:
                audio_array = audio_array.mean(axis=0)
            
            # Resample if needed (simplified - in production use librosa.resample)
            current_rate = av_frame.sample_rate
            if current_rate != self.audio_sample_rate:
                # For MVP, we'll accept the current rate and note it
                # Full resampling would use librosa.resample
                logger.debug(f"Audio sample rate: {current_rate} Hz (target: {self.audio_sample_rate} Hz)")
            
            # Calculate timestamp
            timestamp = time.time() - self.start_time
            
            # Get codec name
            codec_name = self.audio_stream.codec_context.name if self.audio_stream else "unknown"
            
            # Create AudioFrame with quality indicators (Req 8.3)
            audio_frame = AudioFrame(
                samples=audio_array.astype(np.float32),
                sample_rate=current_rate,
                timestamp=timestamp,
                duration=len(audio_array) / current_rate,
                quality_score=self.current_audio_quality,
                codec=codec_name
            )
            
            # Publish to Redis
            await self._publish_audio_frame(audio_frame)
            
        except Exception as e:
            logger.warning(f"Failed to process audio frame: {e}")
    
    async def _process_video_frame(self, av_frame: av.VideoFrame) -> None:
        """Process and publish video frame with adaptive frame skipping
        
        Args:
            av_frame: PyAV video frame
            
        Requirements:
            - Req 8.3: Adapt processing parameters to maintain analysis accuracy
        """
        try:
            # Increment frame counter
            self.video_frame_counter += 1
            
            # Apply adaptive frame skipping (Req 8.3)
            if self.adaptive_enabled and self.video_frame_counter % self.adaptive_frame_skip != 0:
                logger.debug(f"Skipping video frame {self.video_frame_counter} (adaptive skip={self.adaptive_frame_skip})")
                return
            
            # Convert to RGB numpy array
            rgb_frame = av_frame.to_ndarray(format='rgb24')
            
            # Calculate timestamp
            timestamp = time.time() - self.start_time
            
            # Get codec name and resolution
            codec_name = self.video_stream.codec_context.name if self.video_stream else "unknown"
            width = av_frame.width
            height = av_frame.height
            
            # Create VideoFrame with quality indicators (Req 8.3)
            video_frame = VideoFrame(
                image=rgb_frame,
                timestamp=timestamp,
                frame_number=self.frame_count,
                quality_score=self.current_video_quality,
                codec=codec_name,
                resolution=(width, height)
            )
            
            self.frame_count += 1
            
            # Publish to Redis
            await self._publish_video_frame(video_frame)
            
        except Exception as e:
            logger.warning(f"Failed to process video frame: {e}")
    
    async def _publish_audio_frame(self, frame: AudioFrame) -> None:
        """Publish audio frame to Redis Stream
        
        Args:
            frame: AudioFrame to publish
            
        Requirements:
            - Req 1.1: Continuous audio processing
            - Req 9.1: Low latency publishing
        """
        try:
            # Serialize frame data
            frame_data = {
                'samples': pickle.dumps(frame.samples),
                'sample_rate': frame.sample_rate,
                'timestamp': frame.timestamp,
                'duration': frame.duration,
                'quality_score': frame.quality_score,
                'codec': frame.codec
            }
            
            # Publish to Redis Stream
            await self.redis_client.xadd(
                self.audio_stream_name,
                frame_data,
                maxlen=self.buffer_size
            )
            
            logger.debug(f"Published audio frame at {frame.timestamp:.2f}s")
            
        except Exception as e:
            logger.error(f"Failed to publish audio frame: {e}")
    
    async def _publish_video_frame(self, frame: VideoFrame) -> None:
        """Publish video frame to Redis Stream
        
        Args:
            frame: VideoFrame to publish
            
        Requirements:
            - Req 1.3: Continuous video processing
            - Req 9.1: Low latency publishing
        """
        try:
            # Serialize frame data
            frame_data = {
                'image': pickle.dumps(frame.image),
                'timestamp': frame.timestamp,
                'frame_number': frame.frame_number,
                'quality_score': frame.quality_score,
                'codec': frame.codec,
                'resolution': f"{frame.resolution[0]}x{frame.resolution[1]}"
            }
            
            # Publish to Redis Stream
            await self.redis_client.xadd(
                self.video_stream_name,
                frame_data,
                maxlen=self.buffer_size
            )
            
            logger.debug(f"Published video frame #{frame.frame_number} at {frame.timestamp:.2f}s")
            
        except Exception as e:
            logger.error(f"Failed to publish video frame: {e}")
    
    def is_active(self) -> bool:
        """Check if stream is currently active
        
        Returns:
            True if streaming is active, False otherwise
        """
        return self.is_streaming and self.connection is not None and self.connection.is_active
    
    def disconnect(self) -> None:
        """Disconnect from stream and cleanup resources
        
        Requirements:
            - Req 8.4: Handle stream interruption gracefully
        """
        logger.info("Disconnecting stream...")
        
        self.is_streaming = False
        
        if self.container:
            try:
                self.container.close()
            except Exception as e:
                logger.warning(f"Error closing container: {e}")
            self.container = None
        
        if self.connection:
            self.connection.is_active = False
        
        self.audio_stream = None
        self.video_stream = None
        
        logger.info("Stream disconnected")
    
    async def close(self) -> None:
        """Close Redis connection and cleanup
        
        Should be called when shutting down the manager.
        """
        self.disconnect()
        
        if self.redis_client:
            await self.redis_client.close()
            self.redis_client = None
