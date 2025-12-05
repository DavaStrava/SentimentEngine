"""Property-based tests for stream format decoding (Property 12)

Feature: realtime-sentiment-analysis, Property 12: Stream format decoding
Validates: Requirements 8.2
"""

import pytest
import numpy as np
from hypothesis import given, strategies as st, settings, assume
from src.input.stream_manager import StreamInputManager
from src.models.frames import AudioFrame, VideoFrame
from src.models.enums import StreamProtocol


# Feature: realtime-sentiment-analysis, Property 12: Stream format decoding
@given(
    codec_name=st.sampled_from(['aac', 'mp3', 'opus', 'pcm_s16le', 'vorbis', 'h264', 'h265', 'vp9'])
)
@settings(max_examples=100)
def test_codec_validation_accepts_supported_codecs(codec_name):
    """
    Property 12: Stream format decoding
    
    For any supported codec, the Stream Input Manager should successfully validate it.
    
    This property ensures that all codecs listed in the supported codec configuration
    are properly recognized and validated by the system.
    
    Validates: Requirements 8.2 - System SHALL decode audio and video streams using appropriate codecs
    """
    manager = StreamInputManager()
    
    # Configure supported codecs
    manager.supported_audio_codecs = ['aac', 'mp3', 'opus', 'pcm_s16le', 'vorbis']
    manager.supported_video_codecs = ['h264', 'h265', 'hevc', 'vp8', 'vp9', 'av1']
    
    # Determine codec type
    if codec_name in manager.supported_audio_codecs:
        codec_type = 'audio'
    else:
        codec_type = 'video'
    
    # Validate codec
    is_valid = manager._validate_codec(codec_name, codec_type)
    
    # Property: All supported codecs should be validated successfully
    assert is_valid is True, f"Supported codec '{codec_name}' should be validated as True"


# Feature: realtime-sentiment-analysis, Property 12: Stream format decoding
@given(
    codec_name=st.text(min_size=1, max_size=20, alphabet=st.characters(blacklist_categories=('Cs',)))
)
@settings(max_examples=100)
def test_codec_validation_rejects_unsupported_codecs(codec_name):
    """
    Property 12: Stream format decoding
    
    For any unsupported codec, the Stream Input Manager should reject it during validation.
    
    This property ensures that codecs not in the supported list are properly detected
    and handled (with warnings but graceful continuation).
    
    Validates: Requirements 8.2 - System SHALL decode audio and video streams using appropriate codecs
    """
    manager = StreamInputManager()
    
    # Configure a limited set of supported codecs
    manager.supported_audio_codecs = ['aac', 'mp3']
    manager.supported_video_codecs = ['h264', 'vp9']
    
    # Assume codec is not in supported lists
    assume(codec_name not in manager.supported_audio_codecs)
    assume(codec_name not in manager.supported_video_codecs)
    
    # Validate codec for audio
    is_valid_audio = manager._validate_codec(codec_name, 'audio')
    
    # Validate codec for video
    is_valid_video = manager._validate_codec(codec_name, 'video')
    
    # Property: Unsupported codecs should be rejected
    assert is_valid_audio is False, f"Unsupported audio codec '{codec_name}' should be rejected"
    assert is_valid_video is False, f"Unsupported video codec '{codec_name}' should be rejected"


# Feature: realtime-sentiment-analysis, Property 12: Stream format decoding
@given(
    sample_rate=st.integers(min_value=8000, max_value=48000),
    duration=st.floats(min_value=0.1, max_value=5.0),
    codec=st.sampled_from(['aac', 'mp3', 'opus', 'pcm_s16le', 'vorbis'])
)
@settings(max_examples=100)
def test_audio_frame_preserves_codec_information(sample_rate, duration, codec):
    """
    Property 12: Stream format decoding
    
    For any audio frame with codec information, the frame should preserve the codec name.
    
    This property ensures that codec information is properly maintained throughout
    the frame processing pipeline.
    
    Validates: Requirements 8.2 - System SHALL decode audio and video streams using appropriate codecs
    """
    # Create audio samples
    num_samples = int(sample_rate * duration)
    samples = np.random.randn(num_samples).astype(np.float32)
    
    # Create audio frame with codec information
    frame = AudioFrame(
        samples=samples,
        sample_rate=sample_rate,
        timestamp=0.0,
        duration=duration,
        quality_score=0.8,
        codec=codec
    )
    
    # Property: Codec information should be preserved
    assert frame.codec == codec, f"Codec should be preserved as '{codec}'"
    assert isinstance(frame.codec, str), "Codec should be a string"
    assert len(frame.codec) > 0, "Codec should not be empty"


# Feature: realtime-sentiment-analysis, Property 12: Stream format decoding
@given(
    width=st.integers(min_value=320, max_value=1920),
    height=st.integers(min_value=240, max_value=1080),
    codec=st.sampled_from(['h264', 'h265', 'hevc', 'vp8', 'vp9', 'av1'])
)
@settings(max_examples=100)
def test_video_frame_preserves_codec_and_resolution(width, height, codec):
    """
    Property 12: Stream format decoding
    
    For any video frame with codec and resolution information, the frame should preserve both.
    
    This property ensures that codec and resolution information are properly maintained
    throughout the frame processing pipeline.
    
    Validates: Requirements 8.2 - System SHALL decode audio and video streams using appropriate codecs
    """
    # Create video frame
    image = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    
    frame = VideoFrame(
        image=image,
        timestamp=0.0,
        frame_number=0,
        quality_score=0.8,
        codec=codec,
        resolution=(width, height)
    )
    
    # Property: Codec and resolution information should be preserved
    assert frame.codec == codec, f"Codec should be preserved as '{codec}'"
    assert frame.resolution == (width, height), f"Resolution should be preserved as ({width}, {height})"
    assert isinstance(frame.codec, str), "Codec should be a string"
    assert isinstance(frame.resolution, tuple), "Resolution should be a tuple"
    assert len(frame.resolution) == 2, "Resolution should have 2 elements (width, height)"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
