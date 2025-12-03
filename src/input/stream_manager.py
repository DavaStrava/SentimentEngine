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
            
            # Create AudioFrame
            audio_frame = AudioFrame(
                samples=audio_array.astype(np.float32),
                sample_rate=current_rate,
                timestamp=timestamp,
                duration=len(audio_array) / current_rate
            )
            
            # Publish to Redis
            await self._publish_audio_frame(audio_frame)
            
        except Exception as e:
            logger.warning(f"Failed to process audio frame: {e}")
    
    async def _process_video_frame(self, av_frame: av.VideoFrame) -> None:
        """Process and publish video frame
        
        Args:
            av_frame: PyAV video frame
        """
        try:
            # Convert to RGB numpy array
            rgb_frame = av_frame.to_ndarray(format='rgb24')
            
            # Calculate timestamp
            timestamp = time.time() - self.start_time
            
            # Create VideoFrame
            video_frame = VideoFrame(
                image=rgb_frame,
                timestamp=timestamp,
                frame_number=self.frame_count
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
                'duration': frame.duration
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
                'frame_number': frame.frame_number
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
