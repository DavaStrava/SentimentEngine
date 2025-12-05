"""Data models for audio and video frames"""

from dataclasses import dataclass
import numpy as np


@dataclass
class AudioFrame:
    """Represents a single audio frame from the stream
    
    Attributes:
        samples: PCM audio samples as numpy array
        sample_rate: Sample rate in Hz (e.g., 16000)
        timestamp: Seconds since stream start
        duration: Frame duration in seconds
        quality_score: Quality indicator (0.0 to 1.0), based on bitrate and codec
        codec: Audio codec name (e.g., "aac", "mp3", "pcm")
    """
    samples: np.ndarray  # PCM audio samples
    sample_rate: int     # e.g., 16000 Hz
    timestamp: float     # seconds since stream start
    duration: float      # frame duration in seconds
    quality_score: float = 1.0  # Quality indicator (0.0 to 1.0)
    codec: str = "unknown"  # Audio codec name
    
    def __post_init__(self):
        """Validate audio frame data integrity.
        
        Ensures that all audio frame fields meet the requirements for downstream
        acoustic analysis processing. This validation is critical for maintaining
        data quality throughout the analysis pipeline.
        
        Validates:
            - Sample rate is positive (required for feature extraction)
            - Timestamp is non-negative (required for temporal ordering)
            - Duration is positive (required for windowing)
            - Samples is a numpy array (required for signal processing)
        
        Raises:
            AssertionError: If any validation check fails
        
        Requirements:
            - Req 1.2: Acoustic Analysis Module extracts vocal tone features continuously
            - Req 3.1: System extracts pitch, energy, speaking rate, and voice quality features
        
        Properties:
            - Prop 1: Acoustic feature extraction completeness
        """
        assert self.sample_rate > 0, "Sample rate must be positive"
        assert self.timestamp >= 0, "Timestamp must be non-negative"
        assert self.duration > 0, "Duration must be positive"
        assert isinstance(self.samples, np.ndarray), "Samples must be numpy array"


@dataclass
class VideoFrame:
    """Represents a single video frame from the stream
    
    Attributes:
        image: RGB image as numpy array (H, W, 3)
        timestamp: Seconds since stream start
        frame_number: Sequential frame number
        quality_score: Quality indicator (0.0 to 1.0), based on resolution and bitrate
        codec: Video codec name (e.g., "h264", "vp9", "hevc")
        resolution: Tuple of (width, height)
    """
    image: np.ndarray    # RGB image (H, W, 3)
    timestamp: float     # seconds since stream start
    frame_number: int
    quality_score: float = 1.0  # Quality indicator (0.0 to 1.0)
    codec: str = "unknown"  # Video codec name
    resolution: tuple = (0, 0)  # (width, height)
    
    def __post_init__(self):
        """Validate video frame data integrity.
        
        Ensures that all video frame fields meet the requirements for downstream
        visual analysis processing. This validation is critical for face detection,
        landmark extraction, and expression classification.
        
        Validates:
            - Timestamp is non-negative (required for temporal ordering and fusion)
            - Frame number is non-negative (required for frame tracking)
            - Image is a numpy array (required for computer vision processing)
            - Image is 3D array with shape (H, W, C) (required for RGB processing)
            - Image has exactly 3 channels (RGB format required by face detection)
        
        Raises:
            AssertionError: If any validation check fails
        
        Requirements:
            - Req 1.3: Visual Analysis Module extracts facial expression features continuously
            - Req 4.1: System detects faces present in the frame
            - Req 4.2: System extracts facial landmarks and expression features
        
        Properties:
            - Prop 2: Visual feature extraction completeness
        """
        assert self.timestamp >= 0, "Timestamp must be non-negative"
        assert self.frame_number >= 0, "Frame number must be non-negative"
        assert isinstance(self.image, np.ndarray), "Image must be numpy array"
        assert len(self.image.shape) == 3, "Image must be 3D array (H, W, C)"
        assert self.image.shape[2] == 3, "Image must have 3 channels (RGB)"
