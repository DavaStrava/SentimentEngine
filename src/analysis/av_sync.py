"""Audio-Visual Synchronization Component

This module identifies the primary speaker in multi-face scenarios by correlating
lip movement with audio signals. It enables the visual analysis module to focus
on the speaking person when multiple faces are present in the frame.

Requirements:
    - Req 4.4: System analyzes primary speaker based on audio-visual synchronization
"""

import logging
from typing import Optional, List
import numpy as np
import cv2

from src.models.frames import AudioFrame
from src.models.features import FaceRegion, FaceLandmarks


logger = logging.getLogger(__name__)


class AudioVisualSync:
    """Identifies the primary speaker by correlating lip movement with audio signals.
    
    This class implements audio-visual synchronization to determine which face in
    a multi-face video frame corresponds to the active speaker. It analyzes the
    correlation between lip movements (extracted from facial landmarks) and audio
    energy/phoneme features to identify the speaking person.
    
    The synchronization algorithm:
    1. Extracts lip movement features from facial landmarks (mouth region)
    2. Computes audio energy and phoneme-related features from audio frame
    3. Calculates correlation between lip movement and audio signals
    4. Returns the face_id with the highest audio-visual correlation
    
    This enables the visual analysis module to focus emotion classification on
    the primary speaker, improving accuracy in multi-person scenarios.
    
    Attributes:
        mouth_landmark_indices: Indices of mouth landmarks in MediaPipe 468-point model
    """
    
    def __init__(self):
        """Initialize the audio-visual synchronization component."""
        # MediaPipe 468-point face mesh mouth landmark indices
        # Outer lip: 61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291
        # Inner lip: 78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308
        self.mouth_outer_indices = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291]
        self.mouth_inner_indices = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308]
        
        logger.info("AudioVisualSync initialized")
    
    def _extract_lip_movement(self, landmarks: FaceLandmarks) -> float:
        """Extract lip movement features from facial landmarks.
        
        Computes a scalar measure of lip movement based on mouth opening (vertical
        distance between upper and lower lips). This provides a simple but effective
        indicator of speech activity. The measurement is normalized by face height
        to ensure scale invariance across different video resolutions and face sizes.
        
        The algorithm:
        1. Extracts upper lip center (landmark 13) and lower lip center (landmark 14)
        2. Computes vertical distance between lips (mouth opening)
        3. Normalizes by face height (forehead to chin distance)
        4. Returns normalized opening as lip movement score
        
        Args:
            landmarks: FaceLandmarks object containing MediaPipe 468-point face mesh
                      with (x, y) coordinates for each landmark point
            
        Returns:
            float: Lip movement score in [0, 1] range where higher values indicate
                  greater mouth opening (more speech activity). Returns 0.0 if:
                  - Landmarks are insufficient (< 468 points)
                  - Face height is too small (< 1e-6)
                  - Extraction fails due to invalid landmark data
        
        Validates:
            - Req 4.4: System analyzes primary speaker based on audio-visual synchronization
            - Prop 5: Multi-face video handling
        """
        try:
            points = landmarks.points
            
            # Check if we have enough landmarks (MediaPipe 468-point model)
            if len(points) < 468:
                logger.debug(f"Insufficient landmarks for lip movement: {len(points)}")
                return 0.0
            
            # Extract mouth landmarks
            # Upper lip center: index 13
            # Lower lip center: index 14
            upper_lip = points[13]
            lower_lip = points[14]
            
            # Compute vertical distance (mouth opening)
            mouth_opening = np.linalg.norm(upper_lip - lower_lip)
            
            # Normalize by face height for scale invariance
            # Use distance between forehead (10) and chin (152)
            forehead = points[10]
            chin = points[152]
            face_height = np.linalg.norm(forehead - chin)
            
            if face_height < 1e-6:
                return 0.0
            
            normalized_opening = mouth_opening / face_height
            
            return float(normalized_opening)
            
        except Exception as e:
            logger.warning(f"Failed to extract lip movement: {e}")
            return 0.0
    
    def _compute_audio_energy(self, audio_frame: AudioFrame) -> float:
        """Compute audio energy from audio frame.
        
        Calculates the root mean square (RMS) energy of the audio signal, which
        correlates with speech activity and volume. Higher energy typically
        indicates active speech. The energy is normalized to [0, 1] range assuming
        16-bit PCM audio format.
        
        The algorithm:
        1. Extracts PCM samples from audio frame
        2. Computes RMS energy: sqrt(mean(samples^2))
        3. Normalizes by maximum 16-bit value (32768)
        4. Clamps to [0, 1] range
        
        Args:
            audio_frame: AudioFrame containing PCM audio samples (float32 array),
                        sample rate, timestamp, and duration
            
        Returns:
            float: Normalized RMS energy in [0, 1] range where higher values
                  indicate louder audio (more speech activity). Returns 0.0 if
                  computation fails due to invalid audio data.
        
        Validates:
            - Req 4.4: System analyzes primary speaker based on audio-visual synchronization
            - Prop 5: Multi-face video handling
        """
        try:
            samples = audio_frame.samples
            
            # Compute RMS energy
            rms_energy = np.sqrt(np.mean(samples ** 2))
            
            # Normalize to [0, 1] range (assuming 16-bit audio)
            # Max value for 16-bit audio is 32768
            normalized_energy = min(rms_energy / 32768.0, 1.0)
            
            return float(normalized_energy)
            
        except Exception as e:
            logger.warning(f"Failed to compute audio energy: {e}")
            return 0.0
    
    def _compute_correlation(self, lip_movement: float, audio_energy: float) -> float:
        """Compute correlation score between lip movement and audio energy.
        
        This is a simplified correlation metric that combines lip movement and
        audio energy. In a production system, this would use temporal correlation
        over a sliding window of frames to account for audio-visual delay.
        
        For the MVP, we use a simple heuristic: both lip movement and audio energy
        should be high for an active speaker. The correlation is computed as the
        product of the two normalized values, with a threshold applied to reduce
        noise from small movements or quiet audio.
        
        The algorithm:
        1. Multiplies lip_movement and audio_energy (both in [0, 1])
        2. Applies threshold (< 0.01) to filter out noise
        3. Returns correlation score
        
        Args:
            lip_movement: float in [0, 1] range representing normalized mouth opening
                         (higher = more lip movement)
            audio_energy: float in [0, 1] range representing normalized RMS energy
                         (higher = louder audio)
            
        Returns:
            float: Correlation score in [0, 1] range where higher values indicate
                  better audio-visual synchronization (likely active speaker).
                  Returns 0.0 if correlation is below noise threshold (< 0.01).
        
        Validates:
            - Req 4.4: System analyzes primary speaker based on audio-visual synchronization
            - Prop 5: Multi-face video handling
        """
        # Simple multiplicative correlation
        # Both should be high for active speaker
        correlation = lip_movement * audio_energy
        
        # Apply threshold to reduce noise
        if correlation < 0.01:
            correlation = 0.0
        
        return float(correlation)
    
    def identify_primary_speaker(
        self,
        faces: List[FaceRegion],
        audio_frame: AudioFrame
    ) -> Optional[int]:
        """Identify the primary speaker from multiple faces using audio-visual sync.
        
        Analyzes each detected face to determine which one has the highest correlation
        between lip movement and audio energy. This enables the visual analysis module
        to focus on the speaking person for more accurate emotion classification.
        
        The algorithm:
        1. Extracts lip movement features from each face's landmarks
        2. Computes audio energy from the audio frame
        3. Calculates correlation score for each face
        4. Returns the face_id with the highest correlation
        5. Returns None if no face has sufficient correlation (no active speaker)
        
        Args:
            faces: List of detected face regions with landmarks and face_ids
            audio_frame: Current audio frame for energy computation
            
        Returns:
            face_id of the primary speaker (highest audio-visual correlation)
            Returns None if:
            - No faces provided (empty list)
            - No face has sufficient correlation with audio (threshold < 0.05)
            - Audio energy is too low (no speech activity)
            
        Validates:
            - Req 4.4: System analyzes primary speaker based on audio-visual synchronization
            - Prop 5: Multi-face video handling
        """
        if not faces:
            logger.debug("No faces provided for speaker identification")
            return None
        
        if len(faces) == 1:
            # Only one face, return it directly
            return faces[0].face_id
        
        # Compute audio energy once
        audio_energy = self._compute_audio_energy(audio_frame)
        
        # Check if there's sufficient audio activity
        if audio_energy < 0.05:
            logger.debug(f"Low audio energy ({audio_energy:.3f}), no active speaker")
            return None
        
        # Compute correlation for each face
        correlations = []
        for face in faces:
            lip_movement = self._extract_lip_movement(face.landmarks)
            correlation = self._compute_correlation(lip_movement, audio_energy)
            correlations.append((face.face_id, correlation))
            logger.debug(f"Face {face.face_id}: lip_movement={lip_movement:.3f}, "
                        f"correlation={correlation:.3f}")
        
        # Find face with highest correlation
        best_face_id, best_correlation = max(correlations, key=lambda x: x[1])
        
        # Apply threshold to ensure sufficient correlation
        if best_correlation < 0.05:
            logger.debug(f"Best correlation ({best_correlation:.3f}) below threshold, "
                        f"no clear speaker")
            return None
        
        logger.debug(f"Primary speaker identified: face_id={best_face_id}, "
                    f"correlation={best_correlation:.3f}")
        
        return best_face_id
