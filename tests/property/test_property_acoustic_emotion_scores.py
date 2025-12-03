"""Property-based tests for acoustic emotion classification structure

Feature: realtime-sentiment-analysis, Property: Acoustic emotion scores structure
Validates: Requirements 3.2
"""

import pytest
import numpy as np
from hypothesis import given, strategies as st, settings
from unittest.mock import patch, MagicMock
import torch

from src.analysis.acoustic import AcousticAnalyzer
from src.models.frames import AudioFrame


# Custom strategies for generating test data
@st.composite
def valid_audio_frame_strategy(draw):
    """Generate random valid AudioFrame instances with realistic audio content.
    
    This strategy generates audio frames that represent valid audio content:
    - Sample rates between 8kHz and 48kHz (common audio sample rates)
    - Durations between 0.3s and 1.0s (reasonable for real-time processing)
    - Sample counts that match duration and sample rate (minimum 2048 samples for librosa)
    - Audio samples with realistic amplitude ranges
    """
    sample_rate = draw(st.sampled_from([8000, 16000, 22050, 44100, 48000]))
    
    # Ensure minimum 2048 samples for librosa's pyin algorithm
    min_duration = max(0.3, 2048.0 / sample_rate + 0.01)
    duration = draw(st.floats(min_value=min_duration, max_value=1.0))
    timestamp = draw(st.floats(min_value=0.0, max_value=1000.0))
    
    # Calculate sample count based on duration and sample rate
    sample_count = int(sample_rate * duration)
    
    # Generate realistic audio samples
    seed = draw(st.integers(min_value=0, max_value=2**31-1))
    rng = np.random.RandomState(seed)
    samples_array = rng.uniform(-1.0, 1.0, sample_count).astype(np.float32)
    
    return AudioFrame(
        samples=samples_array,
        sample_rate=sample_rate,
        timestamp=timestamp,
        duration=duration
    )


# Feature: realtime-sentiment-analysis, Property: Acoustic emotion scores structure
@settings(max_examples=100, deadline=None)
@given(valid_audio_frame_strategy())
@pytest.mark.asyncio
async def test_acoustic_emotion_scores_structure(audio_frame):
    """
    Property: Acoustic emotion scores structure
    
    For any audio frame with valid audio content, when the Acoustic Analysis Module
    classifies the tone into emotional categories, the emotion scores must have a
    valid structure with:
    1. A dictionary mapping emotion names (strings) to scores (floats)
    2. All emotion scores in the range [0, 1]
    3. At least one emotion category present
    4. Emotion scores that sum to approximately 1.0 (representing probabilities)
    5. Valid confidence score in range [0, 1]
    
    This property verifies the structural correctness of emotion classification output,
    ensuring that the acoustic analyzer produces well-formed emotion scores that can
    be reliably consumed by the fusion engine.
    
    Validates:
    - Req 3.2: WHEN vocal features are extracted THEN the Acoustic Analysis Module 
               SHALL classify the tone into emotional categories with confidence scores
    """
    # Mock the model and processor using patch context managers
    with patch('src.analysis.acoustic.Wav2Vec2ForSequenceClassification') as mock_model_class, \
         patch('src.analysis.acoustic.Wav2Vec2Processor') as mock_processor_class:
        
        mock_processor = MagicMock()
        mock_model = MagicMock()
        
        mock_processor_class.from_pretrained.return_value = mock_processor
        mock_model_class.from_pretrained.return_value = mock_model
        
        # Mock model output with realistic emotion scores that sum to ~1.0
        # Simulating softmax output from emotion classification model
        mock_logits = torch.tensor([[0.5, 2.0, 0.3, 0.1, -0.5, -0.8, -0.3]])
        mock_outputs = MagicMock()
        mock_outputs.logits = mock_logits
        mock_model.return_value = mock_outputs
        mock_model.to.return_value = mock_model
        
        # Mock processor output
        mock_processor.return_value = {
            'input_values': torch.randn(1, len(audio_frame.samples))
        }
        
        # Create analyzer and analyze audio
        analyzer = AcousticAnalyzer()
        result = await analyzer.analyze_audio(audio_frame)
        
        # Property assertions: Emotion scores must have valid structure
        
        # 1. Result must not be None
        assert result is not None, "AcousticResult should not be None for valid audio"
        
        # 2. Result must contain emotion_scores attribute
        assert hasattr(result, 'emotion_scores'), "Result must have emotion_scores attribute"
        
        # 3. emotion_scores must be a dictionary
        assert isinstance(result.emotion_scores, dict), \
            f"emotion_scores must be a dictionary, got {type(result.emotion_scores)}"
        
        # 4. emotion_scores must contain at least one emotion category
        assert len(result.emotion_scores) > 0, \
            "emotion_scores must contain at least one emotion category"
        
        # 5. All emotion keys must be strings
        for emotion_key in result.emotion_scores.keys():
            assert isinstance(emotion_key, str), \
                f"Emotion key must be string, got {type(emotion_key)}: {emotion_key}"
        
        # 6. All emotion scores must be numeric (int or float)
        for emotion, score in result.emotion_scores.items():
            assert isinstance(score, (int, float)), \
                f"Emotion score for '{emotion}' must be numeric, got {type(score)}"
        
        # 7. All emotion scores must be in valid range [0, 1]
        for emotion, score in result.emotion_scores.items():
            assert 0.0 <= score <= 1.0, \
                f"Emotion score for '{emotion}' must be in [0, 1], got {score}"
        
        # 8. Emotion scores should sum to approximately 1.0 (allowing for floating point error)
        # This validates that scores represent probabilities from softmax
        total_score = sum(result.emotion_scores.values())
        assert 0.95 <= total_score <= 1.05, \
            f"Emotion scores should sum to approximately 1.0 (got {total_score}), " \
            f"indicating they represent probability distribution"
        
        # 9. Result must contain confidence score
        assert hasattr(result, 'confidence'), "Result must have confidence attribute"
        
        # 10. Confidence must be numeric and in valid range [0, 1]
        assert isinstance(result.confidence, (int, float)), \
            f"Confidence must be numeric, got {type(result.confidence)}"
        assert 0.0 <= result.confidence <= 1.0, \
            f"Confidence must be in [0, 1], got {result.confidence}"


# Feature: realtime-sentiment-analysis, Property: Acoustic emotion scores structure
@settings(max_examples=100, deadline=None)
@given(valid_audio_frame_strategy())
@pytest.mark.asyncio
async def test_acoustic_emotion_classification_consistency(audio_frame):
    """
    Property: Acoustic emotion classification consistency
    
    For any audio frame, when classified multiple times with the same model state,
    the emotion scores structure should remain consistent (same emotion categories,
    valid ranges, proper probability distribution).
    
    This property verifies that the emotion classification is deterministic and
    produces consistent output structure regardless of input variations.
    
    Validates:
    - Req 3.2: Acoustic Analysis Module SHALL classify tone into emotional categories
               with confidence scores
    """
    # Mock the model and processor
    with patch('src.analysis.acoustic.Wav2Vec2ForSequenceClassification') as mock_model_class, \
         patch('src.analysis.acoustic.Wav2Vec2Processor') as mock_processor_class:
        
        mock_processor = MagicMock()
        mock_model = MagicMock()
        
        mock_processor_class.from_pretrained.return_value = mock_processor
        mock_model_class.from_pretrained.return_value = mock_model
        
        # Mock model output - use deterministic output for consistency test
        mock_logits = torch.tensor([[1.0, 0.5, 0.3, 0.2, 0.1, 0.05, 0.02]])
        mock_outputs = MagicMock()
        mock_outputs.logits = mock_logits
        mock_model.return_value = mock_outputs
        mock_model.to.return_value = mock_model
        
        # Mock processor output
        mock_processor.return_value = {
            'input_values': torch.randn(1, len(audio_frame.samples))
        }
        
        # Create analyzer
        analyzer = AcousticAnalyzer()
        
        # Analyze the same audio frame twice
        result1 = await analyzer.analyze_audio(audio_frame)
        result2 = await analyzer.analyze_audio(audio_frame)
        
        # Property assertions: Both results should have consistent structure
        
        # 1. Both results should not be None
        assert result1 is not None and result2 is not None
        
        # 2. Both should have emotion_scores dictionaries
        assert isinstance(result1.emotion_scores, dict)
        assert isinstance(result2.emotion_scores, dict)
        
        # 3. Both should have the same emotion categories (keys)
        assert set(result1.emotion_scores.keys()) == set(result2.emotion_scores.keys()), \
            "Emotion categories should be consistent across classifications"
        
        # 4. All scores in both results should be in valid range
        for emotion, score in result1.emotion_scores.items():
            assert 0.0 <= score <= 1.0
        for emotion, score in result2.emotion_scores.items():
            assert 0.0 <= score <= 1.0
        
        # 5. Both should sum to approximately 1.0
        assert 0.95 <= sum(result1.emotion_scores.values()) <= 1.05
        assert 0.95 <= sum(result2.emotion_scores.values()) <= 1.05
        
        # 6. Both should have valid confidence scores
        assert 0.0 <= result1.confidence <= 1.0
        assert 0.0 <= result2.confidence <= 1.0


# Feature: realtime-sentiment-analysis, Property: Acoustic emotion scores structure
@settings(max_examples=100, deadline=None)
@given(valid_audio_frame_strategy())
@pytest.mark.asyncio
async def test_acoustic_emotion_scores_with_quality_adjustment(audio_frame):
    """
    Property: Acoustic emotion scores structure with quality adjustment
    
    For any audio frame (including low-quality audio), the emotion scores structure
    must remain valid even when confidence is adjusted based on audio quality.
    The system should gracefully handle poor quality audio by:
    1. Still producing valid emotion score structure
    2. Adjusting confidence appropriately (may be lower)
    3. Not crashing or returning invalid data
    
    This property verifies that quality assessment doesn't break the emotion
    classification output structure.
    
    Validates:
    - Req 3.2: Acoustic Analysis Module SHALL classify tone into emotional categories
               with confidence scores
    - Req 3.5: System SHALL report quality indicator and reduce confidence when
               audio quality is insufficient
    """
    # Create low-quality audio by adding significant noise
    noise = np.random.randn(len(audio_frame.samples)).astype(np.float32) * 0.3
    noisy_samples = np.clip(audio_frame.samples + noise, -1.0, 1.0)
    noisy_frame = AudioFrame(
        samples=noisy_samples,
        sample_rate=audio_frame.sample_rate,
        timestamp=audio_frame.timestamp,
        duration=audio_frame.duration
    )
    
    # Mock the model and processor
    with patch('src.analysis.acoustic.Wav2Vec2ForSequenceClassification') as mock_model_class, \
         patch('src.analysis.acoustic.Wav2Vec2Processor') as mock_processor_class:
        
        mock_processor = MagicMock()
        mock_model = MagicMock()
        
        mock_processor_class.from_pretrained.return_value = mock_processor
        mock_model_class.from_pretrained.return_value = mock_model
        
        # Mock model output
        mock_logits = torch.tensor([[0.3, 0.3, 0.2, 0.15, 0.03, 0.01, 0.01]])
        mock_outputs = MagicMock()
        mock_outputs.logits = mock_logits
        mock_model.return_value = mock_outputs
        mock_model.to.return_value = mock_model
        
        # Mock processor output
        mock_processor.return_value = {
            'input_values': torch.randn(1, len(noisy_frame.samples))
        }
        
        # Create analyzer and analyze noisy audio
        analyzer = AcousticAnalyzer()
        result = await analyzer.analyze_audio(noisy_frame)
        
        # Property assertions: Even with quality adjustment, structure must be valid
        
        # 1. Result must not be None (graceful degradation)
        assert result is not None, \
            "AcousticResult should not be None even for low-quality audio"
        
        # 2. emotion_scores must still be a valid dictionary
        assert isinstance(result.emotion_scores, dict)
        assert len(result.emotion_scores) > 0
        
        # 3. All emotion scores must still be in valid range
        for emotion, score in result.emotion_scores.items():
            assert isinstance(emotion, str)
            assert isinstance(score, (int, float))
            assert 0.0 <= score <= 1.0, \
                f"Even with quality adjustment, emotion score for '{emotion}' " \
                f"must be in [0, 1], got {score}"
        
        # 4. Scores should still sum to approximately 1.0
        total_score = sum(result.emotion_scores.values())
        assert 0.95 <= total_score <= 1.05, \
            f"Even with quality adjustment, emotion scores should sum to ~1.0, got {total_score}"
        
        # 5. Confidence must still be in valid range (may be reduced due to quality)
        assert isinstance(result.confidence, (int, float))
        assert 0.0 <= result.confidence <= 1.0, \
            f"Confidence must be in [0, 1] even with quality adjustment, got {result.confidence}"
        
        # 6. Confidence may be lower for noisy audio, but structure remains valid
        # (We don't assert confidence is lower because that depends on quality assessment,
        #  but we verify the structure is still correct)
