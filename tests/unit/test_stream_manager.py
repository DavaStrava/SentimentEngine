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
