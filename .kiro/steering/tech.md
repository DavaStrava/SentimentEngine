# Technology Stack

## Language & Runtime

- **Python 3.10+** - Primary language for ML/AI ecosystem support
- **asyncio** - Asynchronous task management and event-driven architecture

## Core Infrastructure

- **Redis Streams** - Asynchronous frame distribution and message queue
- **OpenCV** - Video processing and face detection
- **PyAV** - Audio/video stream decoding

## ML/AI Libraries

- **librosa** - Audio feature extraction (pitch, energy, spectral features)
- **Whisper** - Speech-to-text transcription (base or small model)
- **transformers** - Sentiment analysis (DistilBERT fine-tuned on emotion)
- **MediaPipe** - Face detection and facial landmark extraction
- **NumPy** - Numerical processing for fusion algorithms

## Pre-trained Models

- **Acoustic**: wav2vec2 fine-tuned on emotion datasets (RAVDESS, IEMOCAP)
- **Visual**: CNN trained on FER2013 or EfficientNet for facial expressions
- **Linguistic**: DistilBERT fine-tuned on emotion datasets

## UI & Testing

- **Streamlit** - Rapid prototyping of visualization interface
- **pytest** - Unit testing framework
- **Hypothesis** - Property-based testing (minimum 100 iterations per property)

## Architecture Pattern

Asynchronous, event-driven pipeline with parallel processing:
- Three independent analysis modules (acoustic, visual, linguistic) run concurrently
- Result caching with timestamps for non-blocking fusion
- Time-windowed fusion on fixed 1-second intervals
- Graceful degradation when modalities fail or produce low-quality results

## Common Commands

```bash
# Setup virtual environment
python -m venv venv
source venv/bin/activate  # On macOS/Linux

# Install dependencies
pip install redis opencv-python av librosa openai-whisper transformers mediapipe numpy streamlit pytest hypothesis

# Run tests
pytest tests/

# Run property-based tests
pytest tests/ -k property

# Start the application
python src/main.py

# Start Redis (required for async processing)
redis-server
```

## Performance Considerations

- GPU acceleration recommended for model inference (Whisper, visual models)
- Frame skipping in visual analysis (process every 2-3 frames)
- Lower-frequency linguistic processing (every 2-3 seconds)
- Adaptive processing based on stream quality
- Target latency: 1 second end-to-end (max: 3 seconds)
