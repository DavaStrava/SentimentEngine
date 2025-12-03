# Analysis Modules

This directory contains the analysis modules for the SentimentEngine.

## Acoustic Analysis Module

**File:** `acoustic.py`

**Class:** `AcousticAnalyzer`

### Features

1. **Audio Feature Extraction**
   - Pitch (F0) using librosa's pyin algorithm
   - Energy (RMS)
   - Speaking rate (based on onset detection)
   - Spectral centroid
   - Zero crossing rate

2. **Noise Filtering**
   - Spectral subtraction for noise reduction
   - Configurable via `audio.noise_reduction` in config

3. **Emotion Classification**
   - Pre-trained wav2vec2 model for emotion recognition
   - Supports emotions: angry, happy, sad, neutral, fearful, disgust, surprised
   - GPU acceleration when available

4. **Quality Assessment**
   - Detects low energy audio
   - Detects clipping
   - Detects high noise levels (via zero crossing rate)
   - Adjusts confidence based on quality indicators

5. **Asynchronous Processing**
   - Consumes audio frames from Redis Streams
   - Non-blocking operation
   - Result caching for Fusion Engine access

### Usage

```python
from src.analysis.acoustic import AcousticAnalyzer

# Initialize analyzer
analyzer = AcousticAnalyzer()

# Start consuming from Redis (runs as asyncio task)
await analyzer.start()

# Or analyze a single frame
result = await analyzer.analyze_audio(audio_frame)

# Get latest cached result (for Fusion Engine)
latest = analyzer.get_latest_result()
```

### Configuration

Configure in `config/config.yaml`:

```yaml
acoustic:
  model_path: "models/acoustic/wav2vec2_emotion"
  confidence_threshold: 0.05
  speaker_diarization: true

audio:
  sample_rate: 16000
  frame_duration: 0.5
  noise_reduction: true
```

### Error Handling

The module implements graceful error handling:
- Returns low-confidence neutral results on processing errors
- Logs warnings for degraded audio quality
- Never crashes the pipeline due to single-frame errors

### Testing

Unit tests are available in `tests/unit/test_acoustic_analyzer.py`:
- Feature extraction tests
- Noise reduction tests
- Quality assessment tests
- Emotion classification tests (with mocked models)
- Error handling tests

Run tests:
```bash
pytest tests/unit/test_acoustic_analyzer.py -v
```
