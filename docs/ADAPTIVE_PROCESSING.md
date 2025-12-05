# Adaptive Processing and Codec Support

## Overview

Task 12 implements adaptive processing and multi-codec support for the Stream Input Manager, enabling the system to handle varying stream qualities and multiple codec formats while maintaining analysis accuracy.

## Features Implemented

### 1. Multi-Codec Support (Requirement 8.2)

The Stream Input Manager now validates and supports multiple audio and video codecs:

**Supported Audio Codecs:**
- AAC
- MP3
- Opus
- PCM (various formats)
- Vorbis

**Supported Video Codecs:**
- H.264
- H.265/HEVC
- VP8
- VP9
- AV1

The system automatically detects the codec used by the stream and validates it against the configured list. If a codec is not in the supported list, a warning is logged but processing continues (graceful degradation).

### 2. Quality Assessment (Requirement 8.3)

The system now assesses stream quality based on multiple factors:

**Audio Quality Assessment:**
- Bitrate analysis (comparing against minimum and target thresholds)
- Quality score: 0.0 (poor) to 1.0 (excellent)
- Default thresholds:
  - Minimum: 32 kbps
  - Target: 128 kbps

**Video Quality Assessment:**
- Resolution analysis (comparing against minimum thresholds)
- Bitrate analysis (when available)
- Quality score: 0.0 (poor) to 1.0 (excellent)
- Quality levels:
  - Full HD (1920x1080+): 1.0
  - HD (1280x720): 0.9
  - SD (640x480): 0.7
  - Minimum (320x240): 0.5
  - Below minimum: 0.3

### 3. Adaptive Frame Processing (Requirement 8.3)

The system dynamically adjusts processing parameters based on stream quality:

**Frame Skip Adaptation:**
- High quality (≥0.8): Process every frame (skip=1)
- Medium quality (≥0.5): Process every 2nd frame (skip=2)
- Low quality (<0.5): Process every 3rd frame (skip=3)

This adaptive approach:
- Maintains analysis accuracy for high-quality streams
- Reduces computational load for lower-quality streams
- Prevents system overload when processing degraded streams

### 4. Quality Indicators Throughout Pipeline

Quality indicators are now propagated through the entire analysis pipeline:

**AudioFrame:**
- `quality_score`: Float (0.0 to 1.0) indicating stream quality
- `codec`: String identifying the audio codec

**VideoFrame:**
- `quality_score`: Float (0.0 to 1.0) indicating stream quality
- `codec`: String identifying the video codec
- `resolution`: Tuple (width, height) of frame resolution

**Analysis Modules:**
All analysis modules (acoustic, visual, linguistic) now consider stream quality when computing confidence scores:
- Acoustic: Combines stream quality with analysis quality (noise, clipping)
- Visual: Combines stream quality with analysis quality (occlusion, lighting)
- Linguistic: Averages stream quality across buffered frames

## Configuration

All adaptive processing parameters are configurable in `config/config.yaml`:

```yaml
stream:
  # Codec support configuration
  supported_audio_codecs:
    - aac
    - mp3
    - opus
    - pcm_s16le
    - vorbis
  supported_video_codecs:
    - h264
    - h265
    - hevc
    - vp8
    - vp9
    - av1
  
  # Adaptive processing configuration
  adaptive_processing:
    enabled: true
    quality_thresholds:
      high: 0.8
      medium: 0.5
      low: 0.3
    
    frame_skip_by_quality:
      high: 1
      medium: 2
      low: 3
    
    min_resolution:
      width: 320
      height: 240
    
    min_audio_bitrate: 32000
    target_audio_bitrate: 128000
```

## Usage

The adaptive processing features are automatically enabled when a stream is connected:

```python
from src.input.stream_manager import StreamInputManager
from src.models.enums import StreamProtocol

manager = StreamInputManager()

# Connect to stream - quality assessment happens automatically
connection = manager.connect("video.mp4", StreamProtocol.FILE)

# Quality scores are logged
# INFO: Audio quality score: 0.85
# INFO: Video quality score: 0.92
# INFO: Adaptive processing: quality=0.92 (high), frame_skip=1

# Start streaming - frames include quality indicators
await manager.start_streaming()
```

## Benefits

1. **Robustness**: System handles varying stream qualities without crashing
2. **Efficiency**: Adaptive frame skipping reduces computational load for low-quality streams
3. **Transparency**: Quality indicators allow downstream modules to adjust confidence appropriately
4. **Flexibility**: Supports multiple codec formats for broader compatibility
5. **Accuracy**: Quality-aware confidence scoring provides more reliable sentiment analysis

## Testing

Comprehensive unit tests verify all adaptive processing features:

- `tests/unit/test_adaptive_processing.py`: Tests codec validation, quality assessment, and adaptive parameters
- All existing tests pass with the new quality indicators

Run tests:
```bash
python -m pytest tests/unit/test_adaptive_processing.py -v
```

## Requirements Validation

This implementation satisfies:

- **Requirement 8.2**: "WHEN the stream format is identified THEN the system SHALL decode audio and video streams using appropriate codecs"
  - ✅ Multi-codec support with validation
  - ✅ Automatic codec detection
  - ✅ Graceful handling of unsupported codecs

- **Requirement 8.3**: "WHEN stream quality varies THEN the system SHALL adapt processing parameters to maintain analysis accuracy"
  - ✅ Quality assessment for audio and video
  - ✅ Adaptive frame skipping based on quality
  - ✅ Quality indicators propagated through pipeline
  - ✅ Quality-aware confidence scoring in all analysis modules
