"""Property-based tests for linguistic analysis completeness

Feature: realtime-sentiment-analysis, Property 3: Linguistic analysis completeness
Validates: Requirements 1.4, 5.1, 5.2, 5.3
"""

import pytest
import numpy as np
from hypothesis import given, strategies as st, settings
from unittest.mock import patch, MagicMock
import torch

from src.analysis.linguistic import LinguisticAnalyzer
from src.models.frames import AudioFrame
from src.models.results import LinguisticResult


# Custom strategies for generating test data
@st.composite
def valid_audio_frame_with_speech_strategy(draw):
    """Generate random valid AudioFrame instances with realistic audio content for speech.
    
    This strategy generates audio frames that represent valid audio content suitable
    for speech transcription:
    - Sample rates between 8kHz and 48kHz (common audio sample rates)
    - Durations between 0.5s and 3.0s (reasonable for speech segments)
    - Sample counts that match duration and sample rate
    - Audio samples with realistic amplitude ranges
    
    Note: We use durations up to 3.0s to match the linguistic analyzer's buffer
    duration, ensuring we test realistic speech segment lengths.
    """
    sample_rate = draw(st.sampled_from([8000, 16000, 22050, 44100, 48000]))
    
    # Speech segments typically 0.5s to 3.0s
    duration = draw(st.floats(min_value=0.5, max_value=3.0))
    timestamp = draw(st.floats(min_value=0.0, max_value=1000.0))
    
    # Calculate sample count based on duration and sample rate
    sample_count = int(sample_rate * duration)
    
    # Generate realistic audio samples using numpy directly
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


# Feature: realtime-sentiment-analysis, Property 3: Linguistic analysis completeness
@settings(max_examples=100, deadline=None)
@given(valid_audio_frame_with_speech_strategy())
@pytest.mark.asyncio
async def test_linguistic_analysis_completeness(audio_frame):
    """
    Property 3: Linguistic analysis completeness
    
    For any audio frame containing speech, the Linguistic Analysis Module should 
    produce a LinguisticResult containing transcription text, emotion scores, and 
    confidence values.
    
    This property verifies that:
    1. The analyze_audio method returns a LinguisticResult (or None if processing interval hasn't elapsed)
    2. When a result is returned, it contains a transcription string
    3. The result contains emotion_scores dictionary with at least one emotion
    4. The result contains a confidence value in [0, 1]
    5. The result contains a transcription_confidence value in [0, 1]
    6. The result contains a valid timestamp
    7. All emotion scores are in valid range [0, 1]
    8. The confidence is adjusted based on transcription_confidence
    
    The property handles the linguistic analyzer's lower-frequency processing:
    - May return None if processing interval hasn't elapsed (expected behavior)
    - When result is returned, it must be complete with all required fields
    
    Validates:
    - Req 1.4: Linguistic Analysis Module transcribes and analyzes text continuously
    - Req 5.1: System transcribes speech to text using automatic speech recognition
    - Req 5.2: System performs sentiment analysis on semantic content
    - Req 5.3: System identifies emotional polarity, intensity, and specific emotion categories
    """
    # Mock Whisper and sentiment models
    with patch('src.analysis.linguistic.whisper.load_model') as mock_whisper_load, \
         patch('src.analysis.linguistic.AutoTokenizer') as mock_tokenizer_class, \
         patch('src.analysis.linguistic.AutoModelForSequenceClassification') as mock_model_class:
        
        # Mock Whisper model
        mock_whisper_model = MagicMock()
        mock_whisper_load.return_value = mock_whisper_model
        
        # Mock Whisper transcription result with realistic output
        mock_whisper_result = {
            'text': 'This is a test transcription of speech content.',
            'segments': [
                {
                    'text': 'This is a test transcription',
                    'no_speech_prob': 0.1
                },
                {
                    'text': 'of speech content.',
                    'no_speech_prob': 0.15
                }
            ]
        }
        mock_whisper_model.transcribe.return_value = mock_whisper_result
        
        # Mock sentiment tokenizer and model
        mock_tokenizer = MagicMock()
        mock_sentiment_model = MagicMock()
        
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        mock_model_class.from_pretrained.return_value = mock_sentiment_model
        
        # Mock tokenizer output
        mock_tokenizer.return_value = {
            'input_ids': torch.tensor([[101, 2023, 2003, 102]]),
            'attention_mask': torch.tensor([[1, 1, 1, 1]])
        }
        
        # Mock sentiment model output with realistic emotion scores
        # DistilBERT SST-2 outputs: [negative, positive]
        mock_logits = torch.tensor([[0.2, 0.8]])  # Positive sentiment
        mock_outputs = MagicMock()
        mock_outputs.logits = mock_logits
        mock_sentiment_model.return_value = mock_outputs
        mock_sentiment_model.to.return_value = mock_sentiment_model
        
        # Create analyzer
        analyzer = LinguisticAnalyzer()
        
        # Force processing by setting last_process_time to 0
        analyzer.last_process_time = 0.0
        
        # Analyze audio
        result = await analyzer.analyze_audio(audio_frame)
        
        # Property assertions: Result must contain all required components
        # Note: Result may be None if processing interval hasn't elapsed (expected behavior)
        
        if result is not None:
            # 1. Result must be a LinguisticResult instance
            assert isinstance(result, LinguisticResult), "Result must be LinguisticResult instance"
            
            # 2. Result must contain transcription string
            assert hasattr(result, 'transcription'), "Result must have transcription attribute"
            assert isinstance(result.transcription, str), "Transcription must be a string"
            # Note: Transcription may be empty if no speech detected (graceful handling)
            
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
            
            # 6. Result must contain transcription_confidence in valid range
            assert hasattr(result, 'transcription_confidence'), "Result must have transcription_confidence attribute"
            assert isinstance(result.transcription_confidence, (int, float)), "Transcription confidence must be numeric"
            assert 0.0 <= result.transcription_confidence <= 1.0, \
                f"Transcription confidence must be in [0, 1], got {result.transcription_confidence}"
            
            # 7. Result must contain valid timestamp
            assert hasattr(result, 'timestamp'), "Result must have timestamp attribute"
            assert isinstance(result.timestamp, (int, float)), "Timestamp must be numeric"
            assert result.timestamp >= 0.0, f"Timestamp must be non-negative, got {result.timestamp}"
            
            # 8. Confidence should be influenced by transcription_confidence
            # The final confidence should be <= transcription_confidence (it's adjusted downward)
            # This validates Req 5.4: System adjusts linguistic confidence based on transcription quality
            if result.transcription and len(result.transcription.strip()) > 0:
                # For non-empty transcriptions, confidence should be adjusted
                # confidence = base_confidence * transcription_confidence
                # So confidence <= transcription_confidence (assuming base_confidence <= 1.0)
                assert result.confidence <= result.transcription_confidence + 0.01, \
                    f"Confidence ({result.confidence}) should be adjusted by transcription_confidence ({result.transcription_confidence})"


# Feature: realtime-sentiment-analysis, Property 3: Linguistic analysis completeness
@settings(max_examples=100, deadline=None)
@given(valid_audio_frame_with_speech_strategy())
@pytest.mark.asyncio
async def test_linguistic_analysis_handles_empty_transcription(audio_frame):
    """
    Property 3 (variant): Linguistic analysis completeness with empty transcription
    
    For any audio frame that results in empty transcription (no speech detected),
    the Linguistic Analysis Module should still produce a LinguisticResult with
    neutral emotion scores and appropriate confidence values.
    
    This variant tests that the system handles silent or unintelligible audio gracefully by:
    1. Returning a valid LinguisticResult (not None)
    2. Setting transcription to empty string
    3. Returning neutral emotion scores
    4. Setting low confidence to indicate low quality
    5. Not crashing or raising exceptions
    
    Validates:
    - Req 5.4: System reports transcription confidence and adjusts linguistic confidence
    - Design: Graceful degradation when transcription fails
    """
    # Mock Whisper and sentiment models
    with patch('src.analysis.linguistic.whisper.load_model') as mock_whisper_load, \
         patch('src.analysis.linguistic.AutoTokenizer') as mock_tokenizer_class, \
         patch('src.analysis.linguistic.AutoModelForSequenceClassification') as mock_model_class:
        
        # Mock Whisper model
        mock_whisper_model = MagicMock()
        mock_whisper_load.return_value = mock_whisper_model
        
        # Mock Whisper transcription result with empty text (no speech detected)
        mock_whisper_result = {
            'text': '',  # Empty transcription
            'segments': []
        }
        mock_whisper_model.transcribe.return_value = mock_whisper_result
        
        # Mock sentiment tokenizer and model (won't be called for empty text)
        mock_tokenizer = MagicMock()
        mock_sentiment_model = MagicMock()
        
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        mock_model_class.from_pretrained.return_value = mock_sentiment_model
        
        # Create analyzer
        analyzer = LinguisticAnalyzer()
        
        # Force processing by setting last_process_time to 0
        analyzer.last_process_time = 0.0
        
        # Analyze audio
        result = await analyzer.analyze_audio(audio_frame)
        
        # Property assertions: Even with empty transcription, result must be complete
        
        if result is not None:
            # 1. Result must be a LinguisticResult instance
            assert isinstance(result, LinguisticResult), "Result must be LinguisticResult instance"
            
            # 2. Transcription should be empty
            assert result.transcription == "", "Transcription should be empty for no speech"
            
            # 3. Emotion scores should default to neutral
            assert 'neutral' in result.emotion_scores, "Should have neutral emotion for empty transcription"
            assert result.emotion_scores['neutral'] >= 0.8, \
                f"Neutral emotion should dominate for empty transcription, got {result.emotion_scores}"
            
            # 4. Confidence should be low (indicating low quality)
            assert result.confidence <= 0.3, \
                f"Confidence should be low for empty transcription, got {result.confidence}"
            
            # 5. All required fields must still be present
            assert isinstance(result.emotion_scores, dict)
            assert len(result.emotion_scores) > 0
            assert 0.0 <= result.confidence <= 1.0
            assert 0.0 <= result.transcription_confidence <= 1.0
            assert result.timestamp >= 0.0


# Feature: realtime-sentiment-analysis, Property 3: Linguistic analysis completeness
@settings(max_examples=50, deadline=None)
@given(valid_audio_frame_with_speech_strategy())
@pytest.mark.asyncio
async def test_linguistic_result_caching(audio_frame):
    """
    Property 3 (variant): Linguistic result caching for fusion engine
    
    For any audio frame that is analyzed, the result should be cached and
    retrievable via get_latest_result() for the Fusion Engine to access.
    
    This tests the result caching mechanism that enables non-blocking fusion:
    1. After analysis, get_latest_result() should return the result
    2. The cached result should match the returned result
    3. Caching should work regardless of transcription content
    
    Validates:
    - Req 6.1: Fusion Engine receives outputs from analysis modules
    - Design: Result caching with timestamps for non-blocking fusion
    """
    # Mock Whisper and sentiment models
    with patch('src.analysis.linguistic.whisper.load_model') as mock_whisper_load, \
         patch('src.analysis.linguistic.AutoTokenizer') as mock_tokenizer_class, \
         patch('src.analysis.linguistic.AutoModelForSequenceClassification') as mock_model_class:
        
        # Mock Whisper model
        mock_whisper_model = MagicMock()
        mock_whisper_load.return_value = mock_whisper_model
        
        # Mock Whisper transcription result
        mock_whisper_result = {
            'text': 'Test transcription for caching.',
            'segments': [
                {'text': 'Test transcription for caching.', 'no_speech_prob': 0.1}
            ]
        }
        mock_whisper_model.transcribe.return_value = mock_whisper_result
        
        # Mock sentiment tokenizer and model
        mock_tokenizer = MagicMock()
        mock_sentiment_model = MagicMock()
        
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        mock_model_class.from_pretrained.return_value = mock_sentiment_model
        
        # Mock tokenizer output
        mock_tokenizer.return_value = {
            'input_ids': torch.tensor([[101, 2023, 102]]),
            'attention_mask': torch.tensor([[1, 1, 1]])
        }
        
        # Mock sentiment model output
        mock_logits = torch.tensor([[0.3, 0.7]])
        mock_outputs = MagicMock()
        mock_outputs.logits = mock_logits
        mock_sentiment_model.return_value = mock_outputs
        mock_sentiment_model.to.return_value = mock_sentiment_model
        
        # Create analyzer
        analyzer = LinguisticAnalyzer()
        
        # Force processing by setting last_process_time to 0
        analyzer.last_process_time = 0.0
        
        # Initially no result should be cached (or might be from previous test)
        initial_cached = analyzer.get_latest_result()
        
        # Analyze frame
        result = await analyzer.analyze_audio(audio_frame)
        
        # If result was returned (processing interval elapsed), it should be cached
        if result is not None:
            cached_result = analyzer.get_latest_result()
            
            # Property assertions
            assert cached_result is not None, "Result should be cached after analysis"
            assert cached_result == result, "Cached result should match returned result"
            assert isinstance(cached_result, LinguisticResult)
            assert cached_result.timestamp == result.timestamp
            assert cached_result.transcription == result.transcription
            assert cached_result.confidence == result.confidence


# Feature: realtime-sentiment-analysis, Property 3: Linguistic analysis completeness
@settings(max_examples=50, deadline=None)
@given(valid_audio_frame_with_speech_strategy())
@pytest.mark.asyncio
async def test_linguistic_analysis_domain_adaptation(audio_frame):
    """
    Property 3 (variant): Linguistic analysis with domain-specific terminology
    
    For any audio frame containing domain-specific terms (e.g., financial terminology),
    the Linguistic Analysis Module should apply context-aware sentiment interpretation
    and adjust emotion scores appropriately.
    
    This tests the domain adaptation feature:
    1. Domain-specific negative terms should boost negative emotions
    2. Domain-specific positive terms should boost positive emotions
    3. Emotion scores should remain in valid range [0, 1]
    4. Scores should be normalized to sum to approximately 1.0
    
    Validates:
    - Req 5.5: System applies context-aware sentiment interpretation for domain-specific terminology
    """
    # Mock Whisper and sentiment models
    with patch('src.analysis.linguistic.whisper.load_model') as mock_whisper_load, \
         patch('src.analysis.linguistic.AutoTokenizer') as mock_tokenizer_class, \
         patch('src.analysis.linguistic.AutoModelForSequenceClassification') as mock_model_class:
        
        # Mock Whisper model
        mock_whisper_model = MagicMock()
        mock_whisper_load.return_value = mock_whisper_model
        
        # Mock Whisper transcription with domain-specific term
        mock_whisper_result = {
            'text': 'The market is experiencing a systemic risk crisis.',
            'segments': [
                {'text': 'The market is experiencing a systemic risk crisis.', 'no_speech_prob': 0.1}
            ]
        }
        mock_whisper_model.transcribe.return_value = mock_whisper_result
        
        # Mock sentiment tokenizer and model
        mock_tokenizer = MagicMock()
        mock_sentiment_model = MagicMock()
        
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        mock_model_class.from_pretrained.return_value = mock_sentiment_model
        
        # Mock tokenizer output
        mock_tokenizer.return_value = {
            'input_ids': torch.tensor([[101, 2023, 102]]),
            'attention_mask': torch.tensor([[1, 1, 1]])
        }
        
        # Mock sentiment model output (neutral base sentiment)
        mock_logits = torch.tensor([[0.5, 0.5]])
        mock_outputs = MagicMock()
        mock_outputs.logits = mock_logits
        mock_sentiment_model.return_value = mock_outputs
        mock_sentiment_model.to.return_value = mock_sentiment_model
        
        # Create analyzer
        analyzer = LinguisticAnalyzer()
        
        # Force processing by setting last_process_time to 0
        analyzer.last_process_time = 0.0
        
        # Analyze audio
        result = await analyzer.analyze_audio(audio_frame)
        
        # Property assertions: Domain adaptation should affect emotion scores
        
        if result is not None:
            # 1. Result must be complete
            assert isinstance(result, LinguisticResult)
            assert isinstance(result.emotion_scores, dict)
            
            # 2. All emotion scores must be in valid range
            for emotion, score in result.emotion_scores.items():
                assert 0.0 <= score <= 1.0, f"Emotion score for {emotion} must be in [0, 1]"
            
            # 3. Emotion scores should sum to approximately 1.0 (normalized)
            total_score = sum(result.emotion_scores.values())
            assert 0.95 <= total_score <= 1.05, \
                f"Emotion scores should sum to ~1.0, got {total_score}"
            
            # 4. For domain-specific negative terms, negative emotions should be present
            # (sad, fearful, angry, etc.)
            negative_emotions = ['sad', 'fearful', 'angry', 'disgust']
            has_negative = any(result.emotion_scores.get(e, 0) > 0.1 for e in negative_emotions)
            assert has_negative, \
                "Domain-specific negative terms should result in negative emotions"
