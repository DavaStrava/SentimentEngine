# SentimentEngine

Real-Time Multimedia Sentiment Analysis Engine - A proof-of-concept system that analyzes live multimedia streams to provide continuous emotional intelligence.

## Overview

SentimentEngine processes acoustic, visual, and linguistic signals from live video feeds to generate a unified, real-time sentiment score. The system detects emotional shifts instantly—such as drops in confidence, spikes in excitement, or undertones of skepticism—transforming complex human emotion into actionable, objective data.

## Features

- **Multi-modal Analysis**: Combines acoustic (vocal tone), visual (facial expressions), and linguistic (semantic content) signals
- **Real-time Processing**: Sub-3-second latency (target: 1 second)
- **Quality-aware Fusion**: Adapts to varying signal quality across modalities
- **Continuous Scoring**: Normalized sentiment scores in [-1, 1] range
- **Emotional Intelligence**: Detects shifts and provides confidence indicators

## Project Structure

```
SentimentEngine/
├── src/                    # Source code
│   ├── config/            # Configuration management
│   ├── models/            # Data models and interfaces
│   ├── input/             # Stream ingestion
│   ├── analysis/          # Analysis modules (acoustic, visual, linguistic)
│   ├── fusion/            # Fusion engine
│   └── ui/                # User interface (Streamlit)
├── tests/                 # Test suite
│   ├── unit/             # Unit tests
│   ├── property/         # Property-based tests
│   ├── integration/      # Integration tests
│   ├── performance/      # Performance tests
│   └── fixtures/         # Test data
├── models/               # Pre-trained ML models
├── config/               # Configuration files
├── scripts/              # Utility scripts
└── logs/                 # Application logs
```

## Setup

### 1. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On macOS/Linux
# or
venv\Scripts\activate  # On Windows
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Download Models

```bash
python scripts/download_models.py
```

See `models/README.md` for detailed model acquisition instructions.

### 4. Start Redis

Redis is required for asynchronous frame distribution:

```bash
redis-server
```

### 5. Configure

Edit `config/config.yaml` to customize model paths, fusion parameters, and other settings.

## Testing the System

You have several options to test SentimentEngine:

### Option 1: Quick Demo (Recommended for First Test)

Run the standalone demo with synthetic data - no models or video files required:

```bash
python demo_simple.py
```

This demonstrates:
- All three analysis modules working
- Fusion engine combining modalities
- Graceful degradation when models aren't loaded
- Processing 10 synthetic frames with real-time output

### Option 2: Run Automated Tests

```bash
# All tests (unit + property-based)
pytest tests/

# Unit tests only
pytest tests/unit/

# Property-based tests only (100+ iterations each)
pytest tests/property/

# Specific test file
pytest tests/property/test_property_01_acoustic_completeness.py -v

# With coverage report
pytest tests/ --cov=src --cov-report=html
```

### Option 3: Test with Real Video

Once you've downloaded the models (see Setup step 3):

```bash
# Process a video file
python src/main.py --input path/to/video.mp4
```

### Option 4: Run with Streamlit UI

Launch the interactive web interface:

```bash
# Using the helper script
./run_streamlit.sh

# Or using Python launcher
python run_app.py
```

The UI will be available at http://localhost:8501

**Note**: The Streamlit UI currently has some threading issues with asyncio. Use the demo script or automated tests for reliable testing.

### Option 5: Verify Setup

Check that all dependencies and components are properly configured:

```bash
python scripts/verify_setup.py
```

## What to Expect

### Without Downloaded Models
- System runs with graceful degradation
- Acoustic analysis returns low-confidence neutral results
- Visual and linguistic analysis work normally
- Fusion engine combines all modalities successfully

### With Downloaded Models
- Full acoustic emotion recognition (happy, sad, angry, etc.)
- Enhanced confidence scores
- More accurate sentiment detection
- Better emotion category breakdown

## Architecture

The system follows an asynchronous, event-driven architecture:

1. **Stream Input Manager**: Ingests and decodes multimedia streams
2. **Analysis Modules**: Three independent modules process frames in parallel
   - Acoustic: Extracts vocal tone features and classifies emotions
   - Visual: Detects faces and classifies expressions
   - Linguistic: Transcribes speech and analyzes sentiment
3. **Fusion Engine**: Combines multi-modal signals with quality-aware weighting
4. **Display Interface**: Visualizes real-time sentiment scores

## Requirements

- Python 3.10+
- Redis 6.0+
- GPU recommended for optimal performance

## Documentation

- **Testing Guide**: `TESTING.md` - Comprehensive guide to all testing options
- **Quick Start**: `QUICKSTART.md` - Get started in 5 minutes
- Requirements: `.kiro/specs/realtime-sentiment-analysis/requirements.md`
- Design: `.kiro/specs/realtime-sentiment-analysis/design.md`
- Tasks: `.kiro/specs/realtime-sentiment-analysis/tasks.md`
- Steering guides: `.kiro/steering/`

## License

This is a proof-of-concept project for demonstration purposes.
