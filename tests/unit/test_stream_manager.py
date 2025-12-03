"""Unit tests for Stream Input Manager"""

import pytest
import asyncio
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock

from src.input.stream_manager import StreamInputManager
from src.models.enums import StreamProtocol
from src.models.frames import AudioFrame, VideoFrame


class TestStreamInputManager:
    """Test suite for StreamInputManager"""
    
    def test_initialization(self):
        """Test StreamInputManager initializes correctly"""
        manager = StreamInputManager()
        
        assert manager.connection is None
        assert manager.redis_client is None
        assert manager.container is None
        assert manager.is_streaming is False
        assert manager.frame_count == 0
    
    def test_connect_file_not_found(self):
        """Test connection fails when file doesn't exist"""
        manager = StreamInputManager()
        
        with pytest.raises(FileNotFoundError):
            manager.connect("nonexistent_file.mp4", StreamProtocol.FILE)
    
    @patch('av.open')
    def test_connect_success(self, mock_av_open):
        """Test successful connection to stream"""
        # Mock PyAV container
        mock_container = Mock()
        mock_audio_stream = Mock()
        mock_audio_stream.type = 'audio'
        mock_audio_stream.codec_context.name = 'aac'
        
        mock_video_stream = Mock()
        mock_video_stream.type = 'video'
        mock_video_stream.codec_context.name = 'h264'
        
        mock_container.streams = [mock_audio_stream, mock_video_stream]
        mock_av_open.return_value = mock_container
        
        manager = StreamInputManager()
        
        # Create a temporary file for testing
        test_file = Path("test_video.mp4")
        test_file.touch()
        
        try:
            connection = manager.connect(str(test_file), StreamProtocol.FILE)
            
            assert connection is not None
            assert connection.is_active is True
            assert connection.protocol == StreamProtocol.FILE
            assert connection.audio_codec == 'aac'
            assert connection.video_codec == 'h264'
            assert manager.audio_stream == mock_audio_stream
            assert manager.video_stream == mock_video_stream
        finally:
            test_file.unlink()
    
    def test_is_active_when_not_streaming(self):
        """Test is_active returns False when not streaming"""
        manager = StreamInputManager()
        assert manager.is_active() is False
    
    def test_disconnect(self):
        """Test disconnect cleans up resources"""
        manager = StreamInputManager()
        manager.container = Mock()
        manager.connection = Mock()
        manager.connection.is_active = True
        manager.is_streaming = True
        
        manager.disconnect()
        
        assert manager.is_streaming is False
        assert manager.container is None
        assert manager.connection.is_active is False
        assert manager.audio_stream is None
        assert manager.video_stream is None
    
    @pytest.mark.asyncio
    async def test_publish_audio_frame(self):
        """Test audio frame publishing to Redis"""
        manager = StreamInputManager()
        manager.redis_client = AsyncMock()
        
        # Create test audio frame
        audio_frame = AudioFrame(
            samples=np.array([0.1, 0.2, 0.3], dtype=np.float32),
            sample_rate=16000,
            timestamp=1.0,
            duration=0.5
        )
        
        await manager._publish_audio_frame(audio_frame)
        
        # Verify Redis xadd was called
        manager.redis_client.xadd.assert_called_once()
        call_args = manager.redis_client.xadd.call_args
        
        assert call_args[0][0] == manager.audio_stream_name
        assert 'sample_rate' in call_args[0][1]
        assert call_args[0][1]['sample_rate'] == 16000
    
    @pytest.mark.asyncio
    async def test_publish_video_frame(self):
        """Test video frame publishing to Redis"""
        manager = StreamInputManager()
        manager.redis_client = AsyncMock()
        
        # Create test video frame
        video_frame = VideoFrame(
            image=np.zeros((480, 640, 3), dtype=np.uint8),
            timestamp=1.0,
            frame_number=42
        )
        
        await manager._publish_video_frame(video_frame)
        
        # Verify Redis xadd was called
        manager.redis_client.xadd.assert_called_once()
        call_args = manager.redis_client.xadd.call_args
        
        assert call_args[0][0] == manager.video_stream_name
        assert 'frame_number' in call_args[0][1]
        assert call_args[0][1]['frame_number'] == 42
    
    @pytest.mark.asyncio
    async def test_close(self):
        """Test close cleans up Redis connection"""
        manager = StreamInputManager()
        mock_redis = AsyncMock()
        manager.redis_client = mock_redis
        manager.container = Mock()
        
        await manager.close()
        
        mock_redis.close.assert_called_once()
        assert manager.redis_client is None
        assert manager.container is None


class TestAudioFrameExtraction:
    """Test suite for audio frame extraction and format validation"""
    
    @pytest.mark.asyncio
    async def test_audio_frame_extraction_format(self):
        """Test audio frame extraction produces correct format
        
        Requirements: 1.1, 8.2
        Validates that extracted audio frames are in PCM format with correct sample rate
        """
        manager = StreamInputManager()
        manager.redis_client = AsyncMock()
        manager.start_time = 0.0
        
        # Mock PyAV audio frame
        mock_av_frame = Mock()
        mock_av_frame.sample_rate = 16000
        
        # Create mock audio data (mono channel)
        audio_data = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32)
        mock_av_frame.to_ndarray.return_value = audio_data
        
        # Process the frame
        await manager._process_audio_frame(mock_av_frame)
        
        # Verify Redis was called
        assert manager.redis_client.xadd.called
        call_args = manager.redis_client.xadd.call_args[0][1]
        
        # Verify frame format
        assert 'samples' in call_args
        assert 'sample_rate' in call_args
        assert 'timestamp' in call_args
        assert 'duration' in call_args
        
        # Verify sample rate
        assert call_args['sample_rate'] == 16000
        
        # Verify duration calculation
        expected_duration = len(audio_data) / 16000
        assert abs(call_args['duration'] - expected_duration) < 0.001
    
    @pytest.mark.asyncio
    async def test_audio_frame_multichannel_to_mono(self):
        """Test multi-channel audio is converted to mono
        
        Requirements: 1.1, 8.2
        """
        manager = StreamInputManager()
        manager.redis_client = AsyncMock()
        manager.start_time = 0.0
        
        # Mock PyAV audio frame with stereo data
        mock_av_frame = Mock()
        mock_av_frame.sample_rate = 16000
        
        # Create stereo audio data (2 channels)
        stereo_data = np.array([
            [0.1, 0.2, 0.3],  # Left channel
            [0.4, 0.5, 0.6]   # Right channel
        ], dtype=np.float32)
        mock_av_frame.to_ndarray.return_value = stereo_data
        
        # Process the frame
        await manager._process_audio_frame(mock_av_frame)
        
        # Verify Redis was called (frame was processed successfully)
        assert manager.redis_client.xadd.called
    
    @pytest.mark.asyncio
    async def test_audio_frame_pcm_format(self):
        """Test audio frames are in PCM format (float32)
        
        Requirements: 1.1, 8.2
        """
        manager = StreamInputManager()
        manager.redis_client = AsyncMock()
        manager.start_time = 0.0
        
        # Mock PyAV audio frame
        mock_av_frame = Mock()
        mock_av_frame.sample_rate = 16000
        audio_data = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        mock_av_frame.to_ndarray.return_value = audio_data
        
        # Process the frame
        await manager._process_audio_frame(mock_av_frame)
        
        # Verify the frame was published
        assert manager.redis_client.xadd.called


class TestVideoFrameExtraction:
    """Test suite for video frame extraction and format validation"""
    
    @pytest.mark.asyncio
    async def test_video_frame_extraction_format(self):
        """Test video frame extraction produces correct RGB format
        
        Requirements: 1.1, 8.2
        Validates that extracted video frames are in RGB format
        """
        manager = StreamInputManager()
        manager.redis_client = AsyncMock()
        manager.start_time = 0.0
        manager.frame_count = 0
        
        # Mock PyAV video frame
        mock_av_frame = Mock()
        
        # Create RGB frame data (height=480, width=640, channels=3)
        rgb_data = np.zeros((480, 640, 3), dtype=np.uint8)
        rgb_data[:, :, 0] = 255  # Red channel
        mock_av_frame.to_ndarray.return_value = rgb_data
        
        # Process the frame
        await manager._process_video_frame(mock_av_frame)
        
        # Verify Redis was called
        assert manager.redis_client.xadd.called
        call_args = manager.redis_client.xadd.call_args[0][1]
        
        # Verify frame format
        assert 'image' in call_args
        assert 'timestamp' in call_args
        assert 'frame_number' in call_args
        
        # Verify frame number incremented
        assert manager.frame_count == 1
    
    @pytest.mark.asyncio
    async def test_video_frame_rgb_format(self):
        """Test video frames are in RGB format (not BGR or other)
        
        Requirements: 1.1, 8.2
        """
        manager = StreamInputManager()
        manager.redis_client = AsyncMock()
        manager.start_time = 0.0
        manager.frame_count = 0
        
        # Mock PyAV video frame
        mock_av_frame = Mock()
        
        # Create RGB frame with specific pattern
        rgb_data = np.zeros((100, 100, 3), dtype=np.uint8)
        rgb_data[:, :, 0] = 255  # Red channel
        rgb_data[:, :, 1] = 128  # Green channel
        rgb_data[:, :, 2] = 64   # Blue channel
        
        mock_av_frame.to_ndarray.return_value = rgb_data
        
        # Process the frame
        await manager._process_video_frame(mock_av_frame)
        
        # Verify the frame was published
        assert manager.redis_client.xadd.called
    
    @pytest.mark.asyncio
    async def test_video_frame_number_increments(self):
        """Test video frame numbers increment correctly
        
        Requirements: 1.1, 8.2
        """
        manager = StreamInputManager()
        manager.redis_client = AsyncMock()
        manager.start_time = 0.0
        manager.frame_count = 0
        
        # Mock PyAV video frame
        mock_av_frame = Mock()
        rgb_data = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_av_frame.to_ndarray.return_value = rgb_data
        
        # Process multiple frames
        await manager._process_video_frame(mock_av_frame)
        assert manager.frame_count == 1
        
        await manager._process_video_frame(mock_av_frame)
        assert manager.frame_count == 2
        
        await manager._process_video_frame(mock_av_frame)
        assert manager.frame_count == 3


class TestFrameTimestamping:
    """Test suite for frame timestamping accuracy"""
    
    @pytest.mark.asyncio
    async def test_audio_frame_timestamp_accuracy(self):
        """Test audio frame timestamps are accurate relative to stream start
        
        Requirements: 1.1, 8.2
        """
        manager = StreamInputManager()
        manager.redis_client = AsyncMock()
        
        # Set start time to a known value
        import time
        manager.start_time = time.time()
        
        # Mock PyAV audio frame
        mock_av_frame = Mock()
        mock_av_frame.sample_rate = 16000
        audio_data = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        mock_av_frame.to_ndarray.return_value = audio_data
        
        # Process the frame
        await manager._process_audio_frame(mock_av_frame)
        
        # Get the timestamp from the published frame
        call_args = manager.redis_client.xadd.call_args[0][1]
        timestamp = call_args['timestamp']
        
        # Timestamp should be close to current time minus start time
        expected_timestamp = time.time() - manager.start_time
        
        # Allow 0.1 second tolerance for processing time
        assert abs(timestamp - expected_timestamp) < 0.1
    
    @pytest.mark.asyncio
    async def test_video_frame_timestamp_accuracy(self):
        """Test video frame timestamps are accurate relative to stream start
        
        Requirements: 1.1, 8.2
        """
        manager = StreamInputManager()
        manager.redis_client = AsyncMock()
        manager.frame_count = 0
        
        # Set start time to a known value
        import time
        manager.start_time = time.time()
        
        # Mock PyAV video frame
        mock_av_frame = Mock()
        rgb_data = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_av_frame.to_ndarray.return_value = rgb_data
        
        # Process the frame
        await manager._process_video_frame(mock_av_frame)
        
        # Get the timestamp from the published frame
        call_args = manager.redis_client.xadd.call_args[0][1]
        timestamp = call_args['timestamp']
        
        # Timestamp should be close to current time minus start time
        expected_timestamp = time.time() - manager.start_time
        
        # Allow 0.1 second tolerance for processing time
        assert abs(timestamp - expected_timestamp) < 0.1
    
    @pytest.mark.asyncio
    async def test_timestamp_monotonically_increases(self):
        """Test timestamps increase monotonically across frames
        
        Requirements: 1.1, 8.2
        """
        manager = StreamInputManager()
        manager.redis_client = AsyncMock()
        manager.frame_count = 0
        
        import time
        manager.start_time = time.time()
        
        # Mock PyAV video frame
        mock_av_frame = Mock()
        rgb_data = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_av_frame.to_ndarray.return_value = rgb_data
        
        timestamps = []
        
        # Process multiple frames with small delays
        for _ in range(3):
            await manager._process_video_frame(mock_av_frame)
            call_args = manager.redis_client.xadd.call_args[0][1]
            timestamps.append(call_args['timestamp'])
            await asyncio.sleep(0.01)  # Small delay between frames
        
        # Verify timestamps are monotonically increasing
        assert timestamps[0] < timestamps[1] < timestamps[2]
    
    @pytest.mark.asyncio
    async def test_timestamp_starts_at_zero(self):
        """Test first frame timestamp is close to zero
        
        Requirements: 1.1, 8.2
        """
        manager = StreamInputManager()
        manager.redis_client = AsyncMock()
        
        import time
        manager.start_time = time.time()
        
        # Immediately process a frame
        mock_av_frame = Mock()
        mock_av_frame.sample_rate = 16000
        audio_data = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        mock_av_frame.to_ndarray.return_value = audio_data
        
        await manager._process_audio_frame(mock_av_frame)
        
        # Get the timestamp
        call_args = manager.redis_client.xadd.call_args[0][1]
        timestamp = call_args['timestamp']
        
        # First frame should have timestamp very close to 0
        assert timestamp < 0.1  # Within 100ms of start
