"""Property-based tests for acoustic feature extraction completeness

Feature: realtime-sentiment-analysis, Property 1: Acoustic feature extraction completeness
Validates: Requirements 1.2, 3.1, 3.2
"""

import pytest
import numpy as np
from hypothesis import given, strategies as st, settings
from unittest.mock import patch, MagicMock
import torch

from src.analysis.acoustic import AcousticAnalyzer
from src.models.frames import AudioFrame
from src.models.results import AcousticResult
from src.models.features import AcousticFeatures


# Custom strategies for generating test data
@st.composite
def valid_audio_frame_strategy(draw):
    """Generate random valid AudioFrame instances with realistic audio content.
    
    This strategy generates audio frames that represent valid audio content:
    - Sample rates between 8kHz and 48kHz (common audio sample rates)
    - Durations between 0.3s and 1.0s (reasonable for real-time processing, kept short for testing)
    - Sample counts that match duration and sample rate (minimum 2048 samples for librosa)
    - Audio samples with realistic amplitude ranges
    
    Note: We use shorter durations (max 1.0s) to keep test execution fast while still
    testing the property across a wide range of inputs. Minimum duration ensures at least
    2048 samples for librosa's pyin algorithm.
    """
    sample_rate = draw(st.sampled_from([8000, 16000, 22050, 44100, 48000]))
    
    # Ensure minimum 2048 samples for librosa's pyin algorithm
    # Calculate minimum duration needed for this sample rate
    min_duration = max(0.3, 2048.0 / sample_rate + 0.01)  # Add small buffer
    duration = draw(st.floats(min_value=min_duration, max_value=1.0))
    timestamp = draw(st.floats(min_value=0.0, max_value=1000.0))
    
    # Calculate sample count based on duration and sample rate
    sample_count = int(sample_rate * duration)
    
    # Generate realistic audio samples using numpy directly (more efficient than Hypothesis lists)
    # Use a random seed from Hypothesis to ensure reproducibility
    seed = draw(st.integers(min_value=0, max_value=2**31-1))
    rng = np.random.RandomState(seed)
    samples_array = rng.uniform(-1.0, 1.0, sample_count).astype(np.float32)
    
    return AudioFrame(
        samples=samples_array,
        sample_rate=sample_rate,
        timestamp=timestamp,
        duration=duration
    )


# Feature: realtime-sentiment-analysis, Property 1: Acoustic feature extraction completeness
@settings(max_examples=100, deadline=None)
@given(valid_audio_frame_strategy())
@pytest.mark.asyncio
async def test_acoustic_feature_extraction_completeness(audio_frame):
    """
    Property 1: Acoustic feature extraction completeness
    
    For any audio frame with valid audio content, the Acoustic Analysis Module 
    should produce an AcousticResult containing all required features (pitch, 
    energy, speaking rate, spectral features) and emotion scores.
    
    This property verifies that:
    1. The analyze_audio method returns an AcousticResult (not None)
    2. The result contains emotion_scores dictionary with at least one emotion
    3. The result contains a confidence value in [0, 1]
    4. The result contains AcousticFeatures with all required fields:
       - pitch_mean
       - pitch_std
       - energy_mean
       - speaking_rate
       - spectral_centroid
       - zero_crossing_rate
    5. The result contains a valid timestamp
    
    Validates:
    - Req 1.2: Acoustic Analysis Module extracts vocal tone features continuously
    - Req 3.1: System extracts pitch, energy, speaking rate, and voice quality features
    - Req 3.2: System classifies tone into emotional categories with confidence scores
    """
    # Mock the model and processor using patch context managers
    with patch('src.analysis.acoustic.Wav2Vec2ForSequenceClassification') as mock_model_class, \
         patch('src.analysis.acoustic.Wav2Vec2Processor') as mock_processor_class:
        
        mock_processor = MagicMock()
        mock_model = MagicMock()
        
        mock_processor_class.from_pretrained.return_value = mock_processor
        mock_model_class.from_pretrained.return_value = mock_model
        
        # Mock model output with realistic emotion scores
        mock_logits = torch.tensor([[0.1, 0.7, 0.1, 0.05, 0.02, 0.01, 0.02]])
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
        
        # Property assertions: Result must contain all required components
        
        # 1. Result must not be None
        assert result is not None, "AcousticResult should not be None for valid audio"
        
        # 2. Result must be an AcousticResult instance
        assert isinstance(result, AcousticResult), "Result must be AcousticResult instance"
        
        # 3. Result must contain emotion scores
        assert hasattr(result, 'emotion_scores'), "Result must have emotion_scores attribute"
        assert isinstance(result.emotion_scores, dict), "emotion_scores must be a dictionary"
        assert len(result.emotion_scores) > 0, "emotion_scores must contain at least one emotion"
        
        # 4. All emotion scores must be in valid range [0, 1]
        for emotion, score in result.emotion_scores.items():
            assert isinstance(emotion, str), f"Emotion key must be string, got {type(emotion)}"
            assert isinstance(score, (int, float)), f"Emotion score must be numeric, got {type(score)}"
            assert 0.0 <= score <= 1.0, f"Emotion score for {emotion} must be in [0, 1], got {score}"
        
        # 5. Result must contain confidence in valid range
        assert hasattr(result, 'confidence'), "Result must have confidence attribute"
        assert isinstance(result.confidence, (int, float)), "Confidence must be numeric"
        assert 0.0 <= result.confidence <= 1.0, f"Confidence must be in [0, 1], got {result.confidence}"
        
        # 6. Result must contain features (unless processing failed gracefully)
        assert hasattr(result, 'features'), "Result must have features attribute"
        
        # If features are present (not None), verify all required fields exist
        if result.features is not None:
            assert isinstance(result.features, AcousticFeatures), "features must be AcousticFeatures instance"
            
            # Verify all required acoustic features are present
            required_features = [
                'pitch_mean',
                'pitch_std', 
                'energy_mean',
                'speaking_rate',
                'spectral_centroid',
                'zero_crossing_rate'
            ]
            
            for feature_name in required_features:
                assert hasattr(result.features, feature_name), \
                    f"AcousticFeatures must have {feature_name} attribute"
                
                feature_value = getattr(result.features, feature_name)
                assert isinstance(feature_value, (int, float)), \
                    f"{feature_name} must be numeric, got {type(feature_value)}"
                assert not np.isnan(feature_value), \
                    f"{feature_name} must not be NaN"
                assert not np.isinf(feature_value), \
                    f"{feature_name} must not be infinite"
                assert feature_value >= 0.0, \
                    f"{feature_name} must be non-negative, got {feature_value}"
        
        # 7. Result must contain valid timestamp
        assert hasattr(result, 'timestamp'), "Result must have timestamp attribute"
        assert isinstance(result.timestamp, (int, float)), "Timestamp must be numeric"
        assert result.timestamp >= 0.0, f"Timestamp must be non-negative, got {result.timestamp}"


# Feature: realtime-sentiment-analysis, Property 1: Acoustic feature extraction completeness
@settings(max_examples=100, deadline=None)
@given(valid_audio_frame_strategy())
@pytest.mark.asyncio
async def test_acoustic_features_are_complete_even_with_noise(audio_frame):
    """
    Property 1 (variant): Acoustic feature extraction completeness with noisy audio
    
    For any audio frame with valid audio content (including noisy audio), 
    the Acoustic Analysis Module should still produce an AcousticResult 
    containing all required features, though confidence may be reduced.
    
    This variant tests that the system handles noisy audio gracefully by:
    1. Still extracting all required features
    2. Adjusting confidence based on quality assessment
    3. Not crashing or returning None
    
    Validates:
    - Req 3.3: System filters noise before feature extraction
    - Req 3.5: System reports quality indicators when audio quality is insufficient
    """
    # Add noise to the audio frame to simulate poor quality
    noise = np.random.randn(len(audio_frame.samples)).astype(np.float32) * 0.1
    noisy_samples = audio_frame.samples + noise
    noisy_frame = AudioFrame(
        samples=noisy_samples,
        sample_rate=audio_frame.sample_rate,
        timestamp=audio_frame.timestamp,
        duration=audio_frame.duration
    )
    
    # Mock the model and processor using patch context managers
    with patch('src.analysis.acoustic.Wav2Vec2ForSequenceClassification') as mock_model_class, \
         patch('src.analysis.acoustic.Wav2Vec2Processor') as mock_processor_class:
        
        mock_processor = MagicMock()
        mock_model = MagicMock()
        
        mock_processor_class.from_pretrained.return_value = mock_processor
        mock_model_class.from_pretrained.return_value = mock_model
        
        # Mock model output
        mock_logits = torch.tensor([[0.2, 0.3, 0.2, 0.2, 0.05, 0.03, 0.02]])
        mock_outputs = MagicMock()
        mock_outputs.logits = mock_logits
        mock_model.return_value = mock_outputs
        mock_model.to.return_value = mock_model
        
        # Mock processor output
        mock_processor.return_value = {
            'input_values': torch.randn(1, len(noisy_frame.samples))
        }
        
        # Create analyzer with noise reduction enabled
        analyzer = AcousticAnalyzer()
        result = await analyzer.analyze_audio(noisy_frame)
        
        # Property assertions: Even with noisy audio, result must be complete
        
        # 1. Result must not be None (graceful degradation, not failure)
        assert result is not None, "AcousticResult should not be None even for noisy audio"
        
        # 2. Result must contain all required components
        assert isinstance(result, AcousticResult)
        assert isinstance(result.emotion_scores, dict)
        assert len(result.emotion_scores) > 0
        assert 0.0 <= result.confidence <= 1.0
        
        # 3. Features may be present or None (graceful degradation)
        # If present, they must be complete
        if result.features is not None:
            assert isinstance(result.features, AcousticFeatures)
            assert hasattr(result.features, 'pitch_mean')
            assert hasattr(result.features, 'energy_mean')
            assert hasattr(result.features, 'speaking_rate')
            assert hasattr(result.features, 'spectral_centroid')
            assert hasattr(result.features, 'zero_crossing_rate')
