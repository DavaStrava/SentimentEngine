# Dependency Setup

## Initial Project Setup

When initializing the project, always create a Python 3.10+ virtual environment and set up dependencies immediately:

```bash
python -m venv venv
source venv/bin/activate
```

## Requirements File

Create `requirements.txt` with all core dependencies explicitly listed to prevent dependency conflicts:

```
# Async & Infrastructure
redis>=4.5.0

# Stream Processing
opencv-python>=4.8.0
av>=10.0.0

# ML/AI Libraries
librosa>=0.10.0
openai-whisper>=20230314
transformers>=4.30.0
mediapipe>=0.10.0
numpy>=1.24.0
torch>=2.0.0

# UI & Testing
streamlit>=1.25.0
pytest>=7.4.0
hypothesis>=6.82.0

# Additional utilities
pyyaml>=6.0
```

**Note**: `asyncio` is built into Python 3.10+ and should not be listed as a dependency. Use `redis>=4.5.0` with built-in asyncio support via `redis.asyncio`.

## Installation Command

Always install all dependencies at once to resolve version conflicts early:

```bash
pip install -r requirements.txt
```

## Rationale

This automates Task 1 and ensures the complex ML/Async stack is immediately available, preventing dependency hell during development. All required libraries for acoustic, visual, and linguistic analysis are present from the start.
