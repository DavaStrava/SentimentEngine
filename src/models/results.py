"""Data models for analysis results"""

from dataclasses import dataclass
from typing import Dict, Optional
from src.models.features import AcousticFeatures, FaceLandmarks


@dataclass
class AcousticResult:
    """Result from acoustic analysis
    
    Attributes:
        emotion_scores: Dictionary mapping emotion names to scores [0, 1]
                       e.g., {"happy": 0.7, "sad": 0.1, "angry": 0.2}
        confidence: Overall confidence in this result [0, 1]
        features: Extracted acoustic features
        timestamp: When this result was generated (seconds)
    """
    emotion_scores: Dict[str, float]
    confidence: float
    features: Optional[AcousticFeatures]
    timestamp: float
    
    def __post_init__(self):
        """Validate result data"""
        assert 0.0 <= self.confidence <= 1.0, "Confidence must be in [0, 1]"
        assert self.timestamp >= 0, "Timestamp must be non-negative"
        for emotion, score in self.emotion_scores.items():
            assert 0.0 <= score <= 1.0, f"Emotion score for {emotion} must be in [0, 1]"


@dataclass
class VisualResult:
    """Result from visual analysis
    
    Attributes:
        emotion_scores: Dictionary mapping emotion names to scores [0, 1]
        confidence: Overall confidence in this result [0, 1]
        face_detected: Whether a face was detected
        face_landmarks: Facial landmarks if face detected
        timestamp: When this result was generated (seconds)
    """
    emotion_scores: Dict[str, float]
    confidence: float
    face_detected: bool
    face_landmarks: Optional[FaceLandmarks]
    timestamp: float
    
    def __post_init__(self):
        """Validate result data"""
        assert 0.0 <= self.confidence <= 1.0, "Confidence must be in [0, 1]"
        assert self.timestamp >= 0, "Timestamp must be non-negative"
        for emotion, score in self.emotion_scores.items():
            assert 0.0 <= score <= 1.0, f"Emotion score for {emotion} must be in [0, 1]"


@dataclass
class LinguisticResult:
    """Result from linguistic analysis
    
    Attributes:
        transcription: Transcribed text
        emotion_scores: Dictionary mapping emotion names to scores [0, 1]
        confidence: Overall confidence in this result [0, 1]
        transcription_confidence: Confidence in transcription quality [0, 1]
        timestamp: When this result was generated (seconds)
    """
    transcription: str
    emotion_scores: Dict[str, float]
    confidence: float
    transcription_confidence: float
    timestamp: float
    
    def __post_init__(self):
        """Validate result data"""
        assert 0.0 <= self.confidence <= 1.0, "Confidence must be in [0, 1]"
        assert 0.0 <= self.transcription_confidence <= 1.0, "Transcription confidence must be in [0, 1]"
        assert self.timestamp >= 0, "Timestamp must be non-negative"
        for emotion, score in self.emotion_scores.items():
            assert 0.0 <= score <= 1.0, f"Emotion score for {emotion} must be in [0, 1]"


@dataclass
class SentimentScore:
    """Unified sentiment score from fusion
    
    Attributes:
        score: Sentiment score in range [-1.0, 1.0]
               -1.0 = very negative, 0.0 = neutral, 1.0 = very positive
        confidence: Overall confidence in this score [0, 1]
        modality_contributions: Weight contribution from each modality
        emotion_breakdown: Breakdown by emotion categories
        timestamp: When this score was generated (seconds)
    """
    score: float  # -1.0 to 1.0
    confidence: float
    modality_contributions: Dict[str, float]
    emotion_breakdown: Dict[str, float]
    timestamp: float
    
    def __post_init__(self):
        """Validate sentiment score"""
        assert -1.0 <= self.score <= 1.0, "Score must be in [-1, 1]"
        assert 0.0 <= self.confidence <= 1.0, "Confidence must be in [0, 1]"
        assert self.timestamp >= 0, "Timestamp must be non-negative"
