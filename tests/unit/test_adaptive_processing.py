"""Unit tests for adaptive processing and codec support in Stream Input Manager"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from src.input.stream_manager import StreamInputManager
from src.models.enums import StreamProtocol
from src.models.frames import AudioFrame, VideoFrame


class TestCodecValidation:
    """Test codec validation functionality"""
    
    def test_validate_supported_audio_codec(self):
        """Test that supported audio codecs are validated correctly"""
        manager = StreamInputManager()
        manager.supported_audio_codecs = ['aac', 'mp3', 'opus']
        
        assert manager._validate_codec('aac', 'audio') is True
        assert manager._validate_codec('mp3', 'audio') is True
        assert manager._validate_codec('opus', 'audio') is True
    
    def test_validate_unsupported_audio_codec(self):
        """Test that unsupported audio codecs are detected"""
        manager = StreamInputManager()
        manager.supported_audio_codecs = ['aac', 'mp3']
        
        assert manager._validate_codec('flac', 'audio') is False
        assert manager._validate_codec('unknown', 'audio') is False
    
    def test_validate_supported_video_codec(self):
        """Test that supported video codecs are validated correctly"""
        manager = StreamInputManager()
        manager.supported_video_codecs = ['h264', 'vp9', 'hevc']
        
        assert manager._validate_codec('h264', 'video') is True
        assert manager._validate_codec('vp9', 'video') is True
        assert manager._validate_codec('hevc', 'video') is True
    
    def test_validate_empty_codec_list_accepts_all(self):
        """Test that empty codec list accepts all codecs"""
        manager = StreamInputManager()
        manager.supported_audio_codecs = []
        manager.supported_video_codecs = []
        
        assert manager._validate_codec('any_codec', 'audio') is True
        assert manager._validate_codec('any_codec', 'video') is True


class TestQualityAssessment:
    """Test quality assessment functionality"""
    
    def test_assess_audio_quality_high_bitrate(self):
        """Test audio quality assessment with high bitrate"""
        manager = StreamInputManager()
        manager.target_audio_bitrate = 128000
        manager.min_audio_bitrate = 32000
        
        # Mock audio stream with high bitrate
        mock_stream = Mock()
        mock_stream.bit_rate = 256000
        mock_stream.codec_context.bit_rate = 256000
        
        quality = manager._assess_audio_quality(mock_stream)
        assert quality == 1.0
    
    def test_assess_audio_quality_medium_bitrate(self):
        """Test audio quality assessment with medium bitrate"""
        manager = StreamInputManager()
        manager.target_audio_bitrate = 128000
        manager.min_audio_bitrate = 32000
        
        # Mock audio stream with medium bitrate (80 kbps)
        mock_stream = Mock()
        mock_stream.bit_rate = 80000
        mock_stream.codec_context.bit_rate = 80000
        
        quality = manager._assess_audio_quality(mock_stream)
        assert 0.5 < quality < 1.0
    
    def test_assess_audio_quality_low_bitrate(self):
        """Test audio quality assessment with low bitrate"""
        manager = StreamInputManager()
        manager.target_audio_bitrate = 128000
        manager.min_audio_bitrate = 32000
        
        # Mock audio stream with low bitrate (16 kbps)
        mock_stream = Mock()
        mock_stream.bit_rate = 16000
        mock_stream.codec_context.bit_rate = 16000
        
        quality = manager._assess_audio_quality(mock_stream)
        assert quality < 0.5
    
    def test_assess_video_quality_full_hd(self):
        """Test video quality assessment with Full HD resolution"""
        manager = StreamInputManager()
        manager.min_resolution = {'width': 320, 'height': 240}
        
        # Mock video stream with Full HD
        mock_stream = Mock()
        mock_stream.width = 1920
        mock_stream.height = 1080
        mock_stream.bit_rate = 3000000
        mock_stream.codec_context.bit_rate = 3000000
        
        quality = manager._assess_video_quality(mock_stream)
        assert quality >= 0.9
    
    def test_assess_video_quality_hd(self):
        """Test video quality assessment with HD resolution"""
        manager = StreamInputManager()
        manager.min_resolution = {'width': 320, 'height': 240}
        
        # Mock video stream with HD
        mock_stream = Mock()
        mock_stream.width = 1280
        mock_stream.height = 720
        mock_stream.bit_rate = 2000000
        mock_stream.codec_context.bit_rate = 2000000
        
        quality = manager._assess_video_quality(mock_stream)
        assert 0.8 <= quality < 1.0
    
    def test_assess_video_quality_low_resolution(self):
        """Test video quality assessment with low resolution"""
        manager = StreamInputManager()
        manager.min_resolution = {'width': 320, 'height': 240}
        
        # Mock video stream with low resolution
        mock_stream = Mock()
        mock_stream.width = 320
        mock_stream.height = 240
        mock_stream.bit_rate = 500000
        mock_stream.codec_context.bit_rate = 500000
        
        quality = manager._assess_video_quality(mock_stream)
        assert quality <= 0.7


class TestAdaptiveParameters:
    """Test adaptive parameter adjustment"""
    
    def test_adaptive_parameters_high_quality(self):
        """Test adaptive parameters with high quality stream"""
        manager = StreamInputManager()
        manager.adaptive_enabled = True
        manager.quality_thresholds = {'high': 0.8, 'medium': 0.5, 'low': 0.3}
        manager.frame_skip_by_quality = {'high': 1, 'medium': 2, 'low': 3}
        manager.current_video_quality = 0.9
        
        manager._update_adaptive_parameters()
        
        assert manager.adaptive_frame_skip == 1
    
    def test_adaptive_parameters_medium_quality(self):
        """Test adaptive parameters with medium quality stream"""
        manager = StreamInputManager()
        manager.adaptive_enabled = True
        manager.quality_thresholds = {'high': 0.8, 'medium': 0.5, 'low': 0.3}
        manager.frame_skip_by_quality = {'high': 1, 'medium': 2, 'low': 3}
        manager.current_video_quality = 0.6
        
        manager._update_adaptive_parameters()
        
        assert manager.adaptive_frame_skip == 2
    
    def test_adaptive_parameters_low_quality(self):
        """Test adaptive parameters with low quality stream"""
        manager = StreamInputManager()
        manager.adaptive_enabled = True
        manager.quality_thresholds = {'high': 0.8, 'medium': 0.5, 'low': 0.3}
        manager.frame_skip_by_quality = {'high': 1, 'medium': 2, 'low': 3}
        manager.current_video_quality = 0.4
        
        manager._update_adaptive_parameters()
        
        assert manager.adaptive_frame_skip == 3
    
    def test_adaptive_parameters_disabled(self):
        """Test that adaptive parameters are not applied when disabled"""
        manager = StreamInputManager()
        manager.adaptive_enabled = False
        manager.current_video_quality = 0.3
        
        manager._update_adaptive_parameters()
        
        assert manager.adaptive_frame_skip == 1


class TestQualityIndicatorsInFrames:
    """Test that quality indicators are properly included in frames"""
    
    def test_audio_frame_includes_quality_indicators(self):
        """Test that AudioFrame includes quality score and codec"""
        samples = np.random.randn(1600).astype(np.float32)
        frame = AudioFrame(
            samples=samples,
            sample_rate=16000,
            timestamp=1.0,
            duration=0.1,
            quality_score=0.85,
            codec='aac'
        )
        
        assert frame.quality_score == 0.85
        assert frame.codec == 'aac'
    
    def test_video_frame_includes_quality_indicators(self):
        """Test that VideoFrame includes quality score, codec, and resolution"""
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        frame = VideoFrame(
            image=image,
            timestamp=1.0,
            frame_number=30,
            quality_score=0.75,
            codec='h264',
            resolution=(640, 480)
        )
        
        assert frame.quality_score == 0.75
        assert frame.codec == 'h264'
        assert frame.resolution == (640, 480)
    
    def test_audio_frame_default_quality_indicators(self):
        """Test that AudioFrame has default quality indicators"""
        samples = np.random.randn(1600).astype(np.float32)
        frame = AudioFrame(
            samples=samples,
            sample_rate=16000,
            timestamp=1.0,
            duration=0.1
        )
        
        assert frame.quality_score == 1.0
        assert frame.codec == 'unknown'
    
    def test_video_frame_default_quality_indicators(self):
        """Test that VideoFrame has default quality indicators"""
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        frame = VideoFrame(
            image=image,
            timestamp=1.0,
            frame_number=30
        )
        
        assert frame.quality_score == 1.0
        assert frame.codec == 'unknown'
        assert frame.resolution == (0, 0)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
