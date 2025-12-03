"""Data models for extracted features"""

from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class AcousticFeatures:
    """Acoustic features extracted from audio
    
    Attributes:
        pitch_mean: Mean fundamental frequency (F0) in Hz
        pitch_std: Standard deviation of pitch
        energy_mean: Mean RMS energy
        speaking_rate: Syllables per second
        spectral_centroid: Center of mass of spectrum
        zero_crossing_rate: Rate of sign changes in signal
    """
    pitch_mean: float
    pitch_std: float
    energy_mean: float
    speaking_rate: float  # syllables per second
    spectral_centroid: float
    zero_crossing_rate: float


@dataclass
class FaceLandmarks:
    """Facial landmark points
    
    Attributes:
        points: (N, 2) array of (x, y) coordinates
        confidence: Detection confidence score [0, 1]
    """
    points: np.ndarray   # (N, 2) array of (x, y) coordinates
    confidence: float
    
    def __post_init__(self):
        """Validate landmark data"""
        assert 0.0 <= self.confidence <= 1.0, "Confidence must be in [0, 1]"
        assert isinstance(self.points, np.ndarray), "Points must be numpy array"
        if len(self.points) > 0:
            assert len(self.points.shape) == 2, "Points must be 2D array"
            assert self.points.shape[1] == 2, "Points must have 2 coordinates (x, y)"


@dataclass
class FaceRegion:
    """Region containing a detected face
    
    Attributes:
        bounding_box: (x, y, width, height) tuple
        landmarks: Facial landmarks
        face_id: Unique identifier for this face
    """
    bounding_box: tuple[int, int, int, int]  # (x, y, width, height)
    landmarks: FaceLandmarks
    face_id: int
