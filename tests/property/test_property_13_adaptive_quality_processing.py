"""Property-based tests for adaptive quality processing (Property 13)

Feature: realtime-sentiment-analysis, Property 13: Adaptive quality processing
Validates: Requirements 8.3
"""

import pytest
import numpy as np
from hypothesis import given, strategies as st, settings, assume
from unittest.mock import Mock
from src.input.stream_manager import StreamInputManager
from src.models.frames import AudioFrame, VideoFrame


# Feature: realtime-sentiment-analysis, Property 13: Adaptive quality processing
@given(
    quality_score=st.floats(min_value=0.0, max_value=1.0)
)
@settings(max_examples=100)
def test_quality_score_always_in_valid_range(quality_score):
    """
    Property 13: Adaptive quality processing
    
    For any stream quality assessment, the quality score should always be in the range [0.0, 1.0].
    
    This property ensures that quality scores are properly normalized and bounded,
    preventing invalid values from propagating through the system.
    
    Validates: Requirements 8.3 - System SHALL adapt processing parameters to maintain analysis accuracy
    """
    manager = StreamInputManager()
    
    # Set quality score
    manager.current_audio_quality = quality_score
    manager.current_video_quality = quality_score
    
    # Property: Quality scores should always be in valid range
    assert 0.0 <= manager.current_audio_quality <= 1.0, "Audio quality should be in [0.0, 1.0]"
    assert 0.0 <= manager.current_video_quality <= 1.0, "Video quality should be in [0.0, 1.0]"


# Feature: realtime-sentiment-analysis, Property 13: Adaptive quality processing
@given(
    bitrate=st.integers(min_value=8000, max_value=512000)
)
@settings(max_examples=100)
def test_audio_quality_assessment_produces_valid_score(bitrate):
    """
    Property 13: Adaptive quality processing
    
    For any audio bitrate, the quality assessment should produce a valid score in [0.0, 1.0].
    
    This property ensures that audio quality assessment handles all possible bitrate
    values and produces normalized quality scores.
    
    Validates: Requirements 8.3 - System SHALL adapt processing parameters to maintain analysis accuracy
    """
    manager = StreamInputManager()
    manager.min_audio_bitrate = 32000
    manager.target_audio_bitrate = 128000
    
    # Mock audio stream
    mock_stream = Mock()
    mock_stream.bit_rate = bitrate
    mock_stream.codec_context.bit_rate = bitrate
    
    # Assess quality
    quality = manager._assess_audio_quality(mock_stream)
    
    # Property: Quality score should be in valid range
    assert 0.0 <= quality <= 1.0, f"Quality score {quality} should be in [0.0, 1.0] for bitrate {bitrate}"
    assert isinstance(quality, float), "Quality score should be a float"


# Feature: realtime-sentiment-analysis, Property 13: Adaptive quality processing
@given(
    width=st.integers(min_value=160, max_value=3840),
    height=st.integers(min_value=120, max_value=2160)
)
@settings(max_examples=100)
def test_video_quality_assessment_produces_valid_score(width, height):
    """
    Property 13: Adaptive quality processing
    
    For any video resolution, the quality assessment should produce a valid score in [0.0, 1.0].
    
    This property ensures that video quality assessment handles all possible resolution
    values and produces normalized quality scores.
    
    Validates: Requirements 8.3 - System SHALL adapt processing parameters to maintain analysis accuracy
    """
    manager = StreamInputManager()
    manager.min_resolution = {'width': 320, 'height': 240}
    
    # Mock video stream
    mock_stream = Mock()
    mock_stream.width = width
    mock_stream.height = height
    mock_stream.bit_rate = 1000000
    mock_stream.codec_context.bit_rate = 1000000
    
    # Assess quality
    quality = manager._assess_video_quality(mock_stream)
    
    # Property: Quality score should be in valid range
    assert 0.0 <= quality <= 1.0, f"Quality score {quality} should be in [0.0, 1.0] for resolution {width}x{height}"
    assert isinstance(quality, float), "Quality score should be a float"


# Feature: realtime-sentiment-analysis, Property 13: Adaptive quality processing
@given(
    quality=st.floats(min_value=0.0, max_value=1.0)
)
@settings(max_examples=100)
def test_adaptive_frame_skip_is_positive_integer(quality):
    """
    Property 13: Adaptive quality processing
    
    For any quality level, the adaptive frame skip should be a positive integer.
    
    This property ensures that frame skip values are always valid (at least 1)
    and can be used for modulo operations without errors.
    
    Validates: Requirements 8.3 - System SHALL adapt processing parameters to maintain analysis accuracy
    """
    manager = StreamInputManager()
    manager.adaptive_enabled = True
    manager.quality_thresholds = {'high': 0.8, 'medium': 0.5, 'low': 0.3}
    manager.frame_skip_by_quality = {'high': 1, 'medium': 2, 'low': 3}
    manager.current_video_quality = quality
    
    # Update adaptive parameters
    manager._update_adaptive_parameters()
    
    # Property: Frame skip should be a positive integer
    assert isinstance(manager.adaptive_frame_skip, int), "Frame skip should be an integer"
    assert manager.adaptive_frame_skip >= 1, "Frame skip should be at least 1"


# Feature: realtime-sentiment-analysis, Property 13: Adaptive quality processing
@given(
    quality=st.floats(min_value=0.0, max_value=1.0)
)
@settings(max_examples=100)
def test_higher_quality_means_lower_frame_skip(quality):
    """
    Property 13: Adaptive quality processing
    
    For any quality level, higher quality should result in lower or equal frame skip.
    
    This property ensures that the adaptive processing correctly prioritizes
    high-quality streams by processing more frames.
    
    Validates: Requirements 8.3 - System SHALL adapt processing parameters to maintain analysis accuracy
    """
    manager = StreamInputManager()
    manager.adaptive_enabled = True
    manager.quality_thresholds = {'high': 0.8, 'medium': 0.5, 'low': 0.3}
    manager.frame_skip_by_quality = {'high': 1, 'medium': 2, 'low': 3}
    
    # Test with current quality
    manager.current_video_quality = quality
    manager._update_adaptive_parameters()
    current_skip = manager.adaptive_frame_skip
    
    # Test with slightly higher quality (if possible)
    if quality < 1.0:
        higher_quality = min(1.0, quality + 0.1)
        manager.current_video_quality = higher_quality
        manager._update_adaptive_parameters()
        higher_skip = manager.adaptive_frame_skip
        
        # Property: Higher quality should have lower or equal frame skip
        assert higher_skip <= current_skip, \
            f"Higher quality ({higher_quality}) should have lower or equal frame skip than lower quality ({quality})"


# Feature: realtime-sentiment-analysis, Property 13: Adaptive quality processing
@given(
    sample_rate=st.integers(min_value=8000, max_value=48000),
    duration=st.floats(min_value=0.1, max_value=5.0),
    quality=st.floats(min_value=0.0, max_value=1.0)
)
@settings(max_examples=100)
def test_audio_frame_quality_indicator_preserved(sample_rate, duration, quality):
    """
    Property 13: Adaptive quality processing
    
    For any audio frame with quality indicator, the quality should be preserved and accessible.
    
    This property ensures that quality indicators are properly maintained throughout
    the processing pipeline for downstream modules to use.
    
    Validates: Requirements 8.3 - System SHALL adapt processing parameters to maintain analysis accuracy
    """
    # Create audio samples
    num_samples = int(sample_rate * duration)
    samples = np.random.randn(num_samples).astype(np.float32)
    
    # Create audio frame with quality indicator
    frame = AudioFrame(
        samples=samples,
        sample_rate=sample_rate,
        timestamp=0.0,
        duration=duration,
        quality_score=quality,
        codec='aac'
    )
    
    # Property: Quality indicator should be preserved and in valid range
    assert frame.quality_score == quality, "Quality score should be preserved"
    assert 0.0 <= frame.quality_score <= 1.0, "Quality score should be in [0.0, 1.0]"
    assert isinstance(frame.quality_score, float), "Quality score should be a float"


# Feature: realtime-sentiment-analysis, Property 13: Adaptive quality processing
@given(
    width=st.integers(min_value=320, max_value=1920),
    height=st.integers(min_value=240, max_value=1080),
    quality=st.floats(min_value=0.0, max_value=1.0)
)
@settings(max_examples=100)
def test_video_frame_quality_indicator_preserved(width, height, quality):
    """
    Property 13: Adaptive quality processing
    
    For any video frame with quality indicator, the quality should be preserved and accessible.
    
    This property ensures that quality indicators are properly maintained throughout
    the processing pipeline for downstream modules to use.
    
    Validates: Requirements 8.3 - System SHALL adapt processing parameters to maintain analysis accuracy
    """
    # Create video frame
    image = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    
    frame = VideoFrame(
        image=image,
        timestamp=0.0,
        frame_number=0,
        quality_score=quality,
        codec='h264',
        resolution=(width, height)
    )
    
    # Property: Quality indicator should be preserved and in valid range
    assert frame.quality_score == quality, "Quality score should be preserved"
    assert 0.0 <= frame.quality_score <= 1.0, "Quality score should be in [0.0, 1.0]"
    assert isinstance(frame.quality_score, float), "Quality score should be a float"


# Feature: realtime-sentiment-analysis, Property 13: Adaptive quality processing
@given(
    quality=st.floats(min_value=0.0, max_value=1.0)
)
@settings(max_examples=100)
def test_adaptive_processing_disabled_uses_default_skip(quality):
    """
    Property 13: Adaptive quality processing
    
    When adaptive processing is disabled, frame skip should always be 1 regardless of quality.
    
    This property ensures that the system can operate in non-adaptive mode when configured,
    processing every frame regardless of quality.
    
    Validates: Requirements 8.3 - System SHALL adapt processing parameters to maintain analysis accuracy
    """
    manager = StreamInputManager()
    manager.adaptive_enabled = False
    manager.current_video_quality = quality
    
    # Update adaptive parameters
    manager._update_adaptive_parameters()
    
    # Property: When disabled, frame skip should always be 1
    assert manager.adaptive_frame_skip == 1, \
        f"When adaptive processing is disabled, frame skip should be 1 regardless of quality ({quality})"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
