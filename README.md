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

## Usage

### Run the Application

```bash
python src/main.py
```

The Streamlit UI will be available at http://localhost:8501

### Run Tests

```bash
# All tests
pytest tests/

# Unit tests only
pytest tests/unit/

# Property-based tests only
pytest tests/property/

# With coverage
pytest tests/ --cov=src --cov-report=html
```

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

- Requirements: `.kiro/specs/realtime-sentiment-analysis/requirements.md`
- Design: `.kiro/specs/realtime-sentiment-analysis/design.md`
- Tasks: `.kiro/specs/realtime-sentiment-analysis/tasks.md`
- Steering guides: `.kiro/steering/`

## License

This is a proof-of-concept project for demonstration purposes.
