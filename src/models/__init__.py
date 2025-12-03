"""Data models and interfaces"""

from src.models.frames import AudioFrame, VideoFrame
from src.models.features import AcousticFeatures, FaceLandmarks, FaceRegion
from src.models.results import (
    AcousticResult,
    VisualResult,
    LinguisticResult,
    SentimentScore
)
from src.models.enums import StreamProtocol, StreamConnection
from src.models.interfaces import (
    AnalysisModule,
    AcousticAnalyzerInterface,
    VisualAnalyzerInterface,
    LinguisticAnalyzerInterface
)

__all__ = [
    # Frames
    "AudioFrame",
    "VideoFrame",
    # Features
    "AcousticFeatures",
    "FaceLandmarks",
    "FaceRegion",
    # Results
    "AcousticResult",
    "VisualResult",
    "LinguisticResult",
    "SentimentScore",
    # Enums
    "StreamProtocol",
    "StreamConnection",
    # Interfaces
    "AnalysisModule",
    "AcousticAnalyzerInterface",
    "VisualAnalyzerInterface",
    "LinguisticAnalyzerInterface",
]
