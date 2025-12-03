"""Unit tests for Acoustic Analysis Module"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import torch

from src.analysis.acoustic import AcousticAnalyzer, AudioProcessingError
from src.models.frames import AudioFrame
from src.models.features import AcousticFeatures
from src.models.results import AcousticResult


@pytest.fixture
def sample_audio_frame():
    """Create a sample audio frame for testing"""
    # Generate a simple sine wave
    duration = 0.5
    sample_rate = 16000
    t = np.linspace(0, duration, int(sample_rate * duration))
    samples = np.sin(2 * np.pi * 440 * t).astype(np.float32)  # 440 Hz sine wave
    
    return AudioFrame(
        samples=samples,
        sample_rate=sample_rate,
        timestamp=0.0,
        duration=duration
    )


@pytest.fixture
def acoustic_analyzer():
    """Create an AcousticAnalyzer instance for testing"""
    analyzer = AcousticAnalyzer()
    return analyzer


def test_acoustic_analyzer_initialization(acoustic_analyzer):
    """Test that AcousticAnalyzer initializes correctly"""
    assert acoustic_analyzer.model is None  # Model not loaded yet
    assert acoustic_analyzer.processor is None
    assert acoustic_analyzer.latest_result is None
    assert acoustic_analyzer.confidence_threshold == 0.05
    assert acoustic_analyzer.sample_rate == 16000


def test_extract_features(acoustic_analyzer, sample_audio_frame):
    """Test acoustic feature extraction"""
    features = acoustic_analyzer._extract_features(sample_audio_frame)
    
    assert isinstance(features, AcousticFeatures)
    assert features.pitch_mean >= 0
    assert features.energy_mean >= 0
    assert features.spectral_centroid >= 0
    assert features.zero_crossing_rate >= 0
    assert features.speaking_rate >= 0


def test_apply_noise_reduction(acoustic_analyzer, sample_audio_frame):
    """Test noise reduction functionality"""
    original_samples = sample_audio_frame.samples.copy()
    
    # Apply noise reduction
    clean_samples = acoustic_analyzer._apply_noise_reduction(
        sample_audio_frame.samples,
        sample_audio_frame.sample_rate
    )
    
    # Check that output has same length
    assert len(clean_samples) == len(original_samples)
    
    # Check that output is a numpy array
    assert isinstance(clean_samples, np.ndarray)


def test_assess_audio_quality(acoustic_analyzer, sample_audio_frame):
    """Test audio quality assessment"""
    features = acoustic_analyzer._extract_features(sample_audio_frame)
    quality_score = acoustic_analyzer._assess_audio_quality(sample_audio_frame, features)
    
    # Quality score should be in [0, 1]
    assert 0.0 <= quality_score <= 1.0


def test_assess_audio_quality_low_energy(acoustic_analyzer):
    """Test quality assessment with low energy audio"""
    # Create very quiet audio
    samples = np.random.randn(8000).astype(np.float32) * 0.001
    audio_frame = AudioFrame(
        samples=samples,
        sample_rate=16000,
        timestamp=0.0,
        duration=0.5
    )
    
    features = acoustic_analyzer._extract_features(audio_frame)
    quality_score = acoustic_analyzer._assess_audio_quality(audio_frame, features)
    
    # Quality should be reduced for low energy
    assert quality_score < 1.0


def test_assess_audio_quality_clipping(acoustic_analyzer):
    """Test quality assessment with clipped audio"""
    # Create clipped audio
    samples = np.ones(8000, dtype=np.float32) * 0.99
    audio_frame = AudioFrame(
        samples=samples,
        sample_rate=16000,
        timestamp=0.0,
        duration=0.5
    )
    
    features = acoustic_analyzer._extract_features(audio_frame)
    quality_score = acoustic_analyzer._assess_audio_quality(audio_frame, features)
    
    # Quality should be reduced for clipping
    assert quality_score < 1.0


@patch('src.analysis.acoustic.Wav2Vec2ForSequenceClassification')
@patch('src.analysis.acoustic.Wav2Vec2Processor')
def test_classify_emotion(mock_processor_class, mock_model_class, acoustic_analyzer, sample_audio_frame):
    """Test emotion classification"""
    # Mock the model and processor
    mock_processor = MagicMock()
    mock_model = MagicMock()
    
    mock_processor_class.from_pretrained.return_value = mock_processor
    mock_model_class.from_pretrained.return_value = mock_model
    
    # Mock model output
    mock_logits = torch.tensor([[0.1, 0.7, 0.1, 0.05, 0.02, 0.01, 0.02]])
    mock_outputs = MagicMock()
    mock_outputs.logits = mock_logits
    mock_model.return_value = mock_outputs
    mock_model.to.return_value = mock_model
    
    # Mock processor output
    mock_processor.return_value = {
        'input_values': torch.randn(1, 8000)
    }
    
    # Classify emotion
    emotion_scores = acoustic_analyzer._classify_emotion(sample_audio_frame)
    
    # Check that we get emotion scores
    assert isinstance(emotion_scores, dict)
    assert len(emotion_scores) > 0
    
    # Check that scores are in [0, 1]
    for emotion, score in emotion_scores.items():
        assert 0.0 <= score <= 1.0


@pytest.mark.asyncio
@patch('src.analysis.acoustic.Wav2Vec2ForSequenceClassification')
@patch('src.analysis.acoustic.Wav2Vec2Processor')
async def test_analyze_audio(mock_processor_class, mock_model_class, acoustic_analyzer, sample_audio_frame):
    """Test full audio analysis pipeline"""
    # Mock the model and processor
    mock_processor = MagicMock()
    mock_model = MagicMock()
    
    mock_processor_class.from_pretrained.return_value = mock_processor
    mock_model_class.from_pretrained.return_value = mock_model
    
    # Mock model output
    mock_logits = torch.tensor([[0.1, 0.7, 0.1, 0.05, 0.02, 0.01, 0.02]])
    mock_outputs = MagicMock()
    mock_outputs.logits = mock_logits
    mock_model.return_value = mock_outputs
    mock_model.to.return_value = mock_model
    
    # Mock processor output
    mock_processor.return_value = {
        'input_values': torch.randn(1, 8000)
    }
    
    # Analyze audio
    result = await acoustic_analyzer.analyze_audio(sample_audio_frame)
    
    # Check result
    assert isinstance(result, AcousticResult)
    assert result.confidence >= 0.0
    assert result.confidence <= 1.0
    assert len(result.emotion_scores) > 0
    assert result.timestamp > 0
    
    # Check that result is cached
    assert acoustic_analyzer.get_latest_result() == result


def test_get_latest_result_none(acoustic_analyzer):
    """Test getting latest result when none exists"""
    result = acoustic_analyzer.get_latest_result()
    assert result is None


@pytest.mark.asyncio
@patch('src.analysis.acoustic.Wav2Vec2ForSequenceClassification')
@patch('src.analysis.acoustic.Wav2Vec2Processor')
async def test_analyze_audio_error_handling(mock_processor_class, mock_model_class, acoustic_analyzer):
    """Test error handling in audio analysis"""
    # Create problematic audio frame (very short, noisy)
    problematic_frame = AudioFrame(
        samples=np.random.randn(10).astype(np.float32),  # Very short samples
        sample_rate=16000,
        timestamp=0.0,
        duration=0.001  # Very short duration
    )
    
    # Mock the model and processor to raise an error
    mock_processor = MagicMock()
    mock_model = MagicMock()
    
    mock_processor_class.from_pretrained.return_value = mock_processor
    mock_model_class.from_pretrained.return_value = mock_model
    mock_model.to.return_value = mock_model
    
    # Make the model raise an error
    mock_model.side_effect = Exception("Model inference failed")
    
    # Analyze should handle error gracefully
    result = await acoustic_analyzer.analyze_audio(problematic_frame)
    
    # Should return low-confidence neutral result or None
    if result is not None:
        assert result.confidence < 0.2
        assert "neutral" in result.emotion_scores
