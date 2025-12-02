# Data Model Standards

## Type Safety Requirements

All data models MUST be implemented as Python dataclasses with complete type hints:

```python
from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np

@dataclass
class AudioFrame:
    samples: np.ndarray  # PCM audio samples
    sample_rate: int     # e.g., 16000 Hz
    timestamp: float     # seconds since stream start
    duration: float      # frame duration in seconds

@dataclass
class AcousticResult:
    emotion_scores: Dict[str, float]  # e.g., {"happy": 0.7, "sad": 0.1}
    confidence: float
    features: AcousticFeatures
    timestamp: float
```

## NumPy Array Annotations

When fields contain NumPy arrays, always use `np.ndarray` type hints:

- `samples: np.ndarray` for audio data
- `image: np.ndarray` for video frames
- `points: np.ndarray` for facial landmarks

## Required Fields

Every result dataclass must include:
- `timestamp: float` - When the result was generated
- `confidence: float` - Quality indicator for fusion weighting
- Emotion scores as `Dict[str, float]` for consistency

## Validation

Consider adding `__post_init__` validation for critical constraints:

```python
@dataclass
class SentimentScore:
    score: float  # -1.0 to 1.0
    confidence: float
    timestamp: float
    
    def __post_init__(self):
        assert -1.0 <= self.score <= 1.0, "Score must be in [-1, 1]"
        assert 0.0 <= self.confidence <= 1.0, "Confidence must be in [0, 1]"
```

## Rationale

This automates Task 2 boilerplate and ensures high type safety, aligning with the quality needed for ML data exchange. Proper type hints enable IDE autocomplete, catch errors early, and make the codebase more maintainable.
