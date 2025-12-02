# Project Structure

## Directory Organization

```
SentimentEngine/
├── .kiro/
│   ├── specs/
│   │   └── realtime-sentiment-analysis/
│   │       ├── requirements.md    # User stories and acceptance criteria
│   │       ├── design.md          # Architecture and correctness properties
│   │       └── tasks.md           # Implementation plan
│   └── steering/                  # AI assistant guidance documents
├── src/                           # Source code
│   ├── main.py                    # Application entry point and orchestration
│   ├── config/                    # Configuration management
│   │   └── config_loader.py      # Configuration loader class
│   ├── models/                    # Data models and interfaces
│   │   ├── frames.py              # AudioFrame, VideoFrame dataclasses
│   │   ├── results.py             # Analysis result dataclasses
│   │   └── interfaces.py          # Base interfaces for modules
│   ├── input/                     # Stream ingestion
│   │   └── stream_manager.py     # StreamInputManager class
│   ├── analysis/                  # Analysis modules
│   │   ├── acoustic.py            # AcousticAnalyzer
│   │   ├── visual.py              # VisualAnalyzer
│   │   ├── linguistic.py          # LinguisticAnalyzer
│   │   └── av_sync.py             # AudioVisualSync component
│   ├── fusion/                    # Fusion engine
│   │   └── fusion_engine.py      # FusionEngine class
│   └── ui/                        # User interface
│       └── display.py             # SentimentDisplay (Streamlit)
├── tests/                         # Test suite
│   ├── unit/                      # Unit tests for individual components
│   ├── property/                  # Property-based tests (Hypothesis)
│   ├── integration/               # End-to-end pipeline tests
│   ├── performance/               # Latency and performance tests
│   └── fixtures/                  # Test data and fixtures
├── scripts/                       # Deployment and utility scripts
│   └── deploy.sh                  # Deployment script for Streamlit app
├── models/                        # Pre-trained model files
├── config/                        # Configuration files
│   └── config.yaml                # Model paths, parameters, thresholds
└── requirements.txt               # Python dependencies
```

## Module Responsibilities

### Input Layer (`src/input/`)
Handles stream ingestion, decoding, and frame distribution via Redis Streams.

### Analysis Layer (`src/analysis/`)
Three independent modules that consume frames asynchronously and produce timestamped emotion results:
- **acoustic.py**: Extracts vocal tone features and classifies emotions
- **visual.py**: Detects faces, extracts landmarks, classifies expressions
- **linguistic.py**: Transcribes speech and analyzes semantic sentiment
- **av_sync.py**: Identifies primary speaker in multi-face scenarios

### Fusion Layer (`src/fusion/`)
Combines multi-modal signals with quality-aware weighting, conflict resolution, and temporal smoothing.

### UI Layer (`src/ui/`)
Streamlit-based visualization showing real-time scores, modality contributions, and history.

## Key Design Patterns

- **Asynchronous Processing**: All analysis modules run as independent asyncio tasks
- **Result Caching**: Each module maintains timestamped results for non-blocking fusion
- **Time-Windowed Fusion**: Fusion operates on fixed intervals, using latest available data
- **Graceful Degradation**: Missing or low-quality modalities don't block the pipeline

## Testing Organization

- **Unit tests**: Verify individual component behaviors with concrete examples
- **Property tests**: Verify universal properties across randomly generated inputs (tagged with property numbers from design.md)
- **Integration tests**: Verify end-to-end pipeline with real video samples
- **Performance tests**: Validate latency requirements

## Configuration

All configurable parameters (model paths, fusion weights, thresholds, buffer sizes) should be externalized to `config/config.yaml` rather than hardcoded.
