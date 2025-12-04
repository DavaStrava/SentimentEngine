"""Property-based tests for domain-specific sentiment interpretation

Feature: realtime-sentiment-analysis, Property 15: Domain-specific sentiment interpretation
Validates: Requirements 5.5
"""

import pytest
import numpy as np
from hypothesis import given, strategies as st, settings
from unittest.mock import patch, MagicMock
import torch

from src.analysis.linguistic import LinguisticAnalyzer
from src.models.frames import AudioFrame
from src.models.results import LinguisticResult


# Domain-specific terms for testing
NEGATIVE_FINANCIAL_TERMS = [
    'systemic risk', 'crisis', 'crash', 'bear market', 'recession',
    'default', 'bankruptcy', 'collapse', 'downturn'
]

POSITIVE_FINANCIAL_TERMS = [
    'bull market', 'growth', 'rally', 'surge', 'boom',
    'profit', 'gains', 'recovery', 'expansion'
]


@st.composite
def domain_specific_transcription_strategy(draw):
    """Generate transcriptions with domain-specific terms in neutral context.
    
    This strategy creates transcriptions that contain domain-specific financial
    terminology embedded in otherwise neutral surrounding text. This tests whether
    the linguistic analyzer correctly interprets the domain-specific sentiment
    regardless of the neutral context.
    
    Returns:
        tuple: (transcription text, expected sentiment direction)
               where sentiment direction is 'negative' or 'positive'
    """
    # Choose whether to test negative or positive domain term
    is_negative = draw(st.booleans())
    
    if is_negative:
        domain_term = draw(st.sampled_from(NEGATIVE_FINANCIAL_TERMS))
        sentiment_direction = 'negative'
    else:
        domain_term = draw(st.sampled_from(POSITIVE_FINANCIAL_TERMS))
        sentiment_direction = 'positive'
    
    # Generate neutral context phrases
    neutral_prefixes = [
        "The report mentions",
        "According to analysts",
        "The data shows",
        "Experts are discussing",
        "The presentation covered",
        "The speaker talked about",
        "The analysis indicates",
        "Recent studies examine"
    ]
    
    neutral_suffixes = [
        "in the current market.",
        "for the upcoming quarter.",
        "according to recent data.",
        "in today's economic climate.",
        "based on historical trends.",
        "in the financial sector.",
        "across various industries.",
        "in the global economy."
    ]
    
    prefix = draw(st.sampled_from(neutral_prefixes))
    suffix = draw(st.sampled_from(neutral_suffixes))
    
    # Construct transcription with domain term in neutral context
    transcription = f"{prefix} {domain_term} {suffix}"
    
    return transcription, sentiment_direction


@st.composite
def audio_frame_for_domain_test_strategy(draw):
    """Generate audio frame for domain-specific sentiment testing.
    
    Creates a simple audio frame with standard parameters suitable for
    testing domain-specific sentiment interpretation.
    """
    sample_rate = 16000  # Standard sample rate
    duration = 2.0  # 2 seconds
    timestamp = draw(st.floats(min_value=0.0, max_value=100.0))
    
    sample_count = int(sample_rate * duration)
    
    # Generate simple audio samples
    seed = draw(st.integers(min_value=0, max_value=2**31-1))
    rng = np.random.RandomState(seed)
    samples_array = rng.uniform(-0.5, 0.5, sample_count).astype(np.float32)
    
    return AudioFrame(
        samples=samples_array,
        sample_rate=sample_rate,
        timestamp=timestamp,
        duration=duration
    )


# Feature: realtime-sentiment-analysis, Property 15: Domain-specific sentiment interpretation
@settings(max_examples=100, deadline=None)
@given(
    audio_frame=audio_frame_for_domain_test_strategy(),
    transcription_data=domain_specific_transcription_strategy()
)
@pytest.mark.asyncio
async def test_domain_specific_sentiment_interpretation(audio_frame, transcription_data):
    """
    Property 15: Domain-specific sentiment interpretation
    
    For any known domain-specific term with strong sentiment (e.g., "systemic risk" 
    in finance, "bear market"), the linguistic analysis should produce a sentiment 
    score that reflects the domain-specific meaning regardless of surrounding neutral context.
    
    This property verifies that:
    1. Domain-specific negative terms result in elevated negative emotion scores
    2. Domain-specific positive terms result in elevated positive emotion scores
    3. The domain-specific sentiment is detected even in neutral surrounding context
    4. Emotion scores remain in valid range [0, 1] after domain adaptation
    5. Scores are properly normalized after domain adaptation
    
    The test uses transcriptions with domain-specific financial terms embedded in
    neutral context to verify that the domain-specific meaning is correctly interpreted.
    
    Validates:
    - Req 5.5: System applies context-aware sentiment interpretation for domain-specific terminology
    """
    transcription, expected_sentiment = transcription_data
    
    # Mock Whisper and sentiment models
    with patch('src.analysis.linguistic.whisper.load_model') as mock_whisper_load, \
         patch('src.analysis.linguistic.AutoTokenizer') as mock_tokenizer_class, \
         patch('src.analysis.linguistic.AutoModelForSequenceClassification') as mock_model_class:
        
        # Mock Whisper model
        mock_whisper_model = MagicMock()
        mock_whisper_load.return_value = mock_whisper_model
        
        # Mock Whisper transcription result with domain-specific term
        mock_whisper_result = {
            'text': transcription,
            'segments': [
                {'text': transcription, 'no_speech_prob': 0.1}
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
        
        # Mock sentiment model output with NEUTRAL base sentiment
        # This is key: the base model returns neutral, but domain adaptation should adjust it
        mock_logits = torch.tensor([[0.5, 0.5]])  # Neutral: 50% negative, 50% positive
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
        
        # Property assertions: Domain-specific sentiment should be reflected
        
        if result is not None:
            # 1. Result must be complete
            assert isinstance(result, LinguisticResult), "Result must be LinguisticResult instance"
            assert isinstance(result.emotion_scores, dict), "emotion_scores must be a dictionary"
            assert len(result.emotion_scores) > 0, "emotion_scores must contain at least one emotion"
            
            # 2. All emotion scores must be in valid range [0, 1]
            for emotion, score in result.emotion_scores.items():
                assert 0.0 <= score <= 1.0, \
                    f"Emotion score for {emotion} must be in [0, 1], got {score}"
            
            # 3. Emotion scores should sum to approximately 1.0 (normalized)
            total_score = sum(result.emotion_scores.values())
            assert 0.95 <= total_score <= 1.05, \
                f"Emotion scores should sum to ~1.0 after domain adaptation, got {total_score}"
            
            # 4. Domain-specific sentiment should be reflected in emotion scores
            if expected_sentiment == 'negative':
                # For negative domain terms, negative emotions should be elevated
                negative_emotions = ['sad', 'fearful', 'angry', 'disgust']
                total_negative = sum(result.emotion_scores.get(e, 0) for e in negative_emotions)
                
                # Domain adaptation should boost negative emotions
                # Even with neutral base sentiment, domain terms should push scores
                assert total_negative > 0.3, \
                    f"Domain-specific negative term '{transcription}' should result in elevated " \
                    f"negative emotions (sad, fearful, angry, disgust), got total_negative={total_negative:.3f}, " \
                    f"scores={result.emotion_scores}"
                
            else:  # expected_sentiment == 'positive'
                # For positive domain terms, positive emotions should be elevated
                positive_emotions = ['happy', 'surprised']
                total_positive = sum(result.emotion_scores.get(e, 0) for e in positive_emotions)
                
                # Domain adaptation should boost positive emotions
                assert total_positive > 0.3, \
                    f"Domain-specific positive term '{transcription}' should result in elevated " \
                    f"positive emotions (happy, surprised), got total_positive={total_positive:.3f}, " \
                    f"scores={result.emotion_scores}"
            
            # 5. Transcription should contain the domain-specific term
            assert any(term.lower() in result.transcription.lower() 
                      for term in (NEGATIVE_FINANCIAL_TERMS + POSITIVE_FINANCIAL_TERMS)), \
                f"Transcription should contain domain-specific term, got: '{result.transcription}'"


# Feature: realtime-sentiment-analysis, Property 15: Domain-specific sentiment interpretation
@settings(max_examples=50, deadline=None)
@given(audio_frame=audio_frame_for_domain_test_strategy())
@pytest.mark.asyncio
async def test_domain_specific_negative_terms(audio_frame):
    """
    Property 15 (variant): Domain-specific negative term interpretation
    
    For any known domain-specific negative term (e.g., "systemic risk", "crisis", 
    "bear market"), the linguistic analysis should produce elevated negative emotion 
    scores regardless of surrounding neutral context.
    
    This focused test verifies that specific negative financial terms are correctly
    interpreted with appropriate negative sentiment.
    
    Validates:
    - Req 5.5: System applies context-aware sentiment interpretation for domain-specific terminology
    """
    # Test with a specific negative term
    transcription = "The analysis discusses systemic risk in the financial sector."
    
    # Mock Whisper and sentiment models
    with patch('src.analysis.linguistic.whisper.load_model') as mock_whisper_load, \
         patch('src.analysis.linguistic.AutoTokenizer') as mock_tokenizer_class, \
         patch('src.analysis.linguistic.AutoModelForSequenceClassification') as mock_model_class:
        
        # Mock Whisper model
        mock_whisper_model = MagicMock()
        mock_whisper_load.return_value = mock_whisper_model
        
        # Mock Whisper transcription result
        mock_whisper_result = {
            'text': transcription,
            'segments': [
                {'text': transcription, 'no_speech_prob': 0.1}
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
        
        # Mock sentiment model output with neutral base sentiment
        mock_logits = torch.tensor([[0.5, 0.5]])
        mock_outputs = MagicMock()
        mock_outputs.logits = mock_logits
        mock_sentiment_model.return_value = mock_outputs
        mock_sentiment_model.to.return_value = mock_sentiment_model
        
        # Create analyzer
        analyzer = LinguisticAnalyzer()
        
        # Force processing
        analyzer.last_process_time = 0.0
        
        # Analyze audio
        result = await analyzer.analyze_audio(audio_frame)
        
        # Property assertions
        if result is not None:
            # Negative emotions should be elevated
            negative_emotions = ['sad', 'fearful']
            
            # Check that sad and fearful emotions are boosted (domain adaptation adds 0.2 to sad, 0.1 to fearful)
            assert result.emotion_scores.get('sad', 0) > 0.15, \
                f"'sad' emotion should be boosted for 'systemic risk', got {result.emotion_scores.get('sad', 0):.3f}"
            
            assert result.emotion_scores.get('fearful', 0) > 0.05, \
                f"'fearful' emotion should be boosted for 'systemic risk', got {result.emotion_scores.get('fearful', 0):.3f}"


# Feature: realtime-sentiment-analysis, Property 15: Domain-specific sentiment interpretation
@settings(max_examples=50, deadline=None)
@given(audio_frame=audio_frame_for_domain_test_strategy())
@pytest.mark.asyncio
async def test_domain_specific_positive_terms(audio_frame):
    """
    Property 15 (variant): Domain-specific positive term interpretation
    
    For any known domain-specific positive term (e.g., "bull market", "growth", 
    "rally"), the linguistic analysis should produce elevated positive emotion 
    scores regardless of surrounding neutral context.
    
    This focused test verifies that specific positive financial terms are correctly
    interpreted with appropriate positive sentiment.
    
    Validates:
    - Req 5.5: System applies context-aware sentiment interpretation for domain-specific terminology
    """
    # Test with a specific positive term
    transcription = "The report mentions bull market conditions in the economy."
    
    # Mock Whisper and sentiment models
    with patch('src.analysis.linguistic.whisper.load_model') as mock_whisper_load, \
         patch('src.analysis.linguistic.AutoTokenizer') as mock_tokenizer_class, \
         patch('src.analysis.linguistic.AutoModelForSequenceClassification') as mock_model_class:
        
        # Mock Whisper model
        mock_whisper_model = MagicMock()
        mock_whisper_load.return_value = mock_whisper_model
        
        # Mock Whisper transcription result
        mock_whisper_result = {
            'text': transcription,
            'segments': [
                {'text': transcription, 'no_speech_prob': 0.1}
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
        
        # Mock sentiment model output with neutral base sentiment
        mock_logits = torch.tensor([[0.5, 0.5]])
        mock_outputs = MagicMock()
        mock_outputs.logits = mock_logits
        mock_sentiment_model.return_value = mock_outputs
        mock_sentiment_model.to.return_value = mock_sentiment_model
        
        # Create analyzer
        analyzer = LinguisticAnalyzer()
        
        # Force processing
        analyzer.last_process_time = 0.0
        
        # Analyze audio
        result = await analyzer.analyze_audio(audio_frame)
        
        # Property assertions
        if result is not None:
            # Positive emotions should be elevated
            # Check that happy emotion is boosted (domain adaptation adds 0.2 to happy)
            assert result.emotion_scores.get('happy', 0) > 0.15, \
                f"'happy' emotion should be boosted for 'bull market', got {result.emotion_scores.get('happy', 0):.3f}"


# Feature: realtime-sentiment-analysis, Property 15: Domain-specific sentiment interpretation
@settings(max_examples=50, deadline=None)
@given(
    audio_frame=audio_frame_for_domain_test_strategy(),
    negative_term=st.sampled_from(NEGATIVE_FINANCIAL_TERMS),
    positive_term=st.sampled_from(POSITIVE_FINANCIAL_TERMS)
)
@pytest.mark.asyncio
async def test_domain_adaptation_normalization(audio_frame, negative_term, positive_term):
    """
    Property 15 (variant): Domain adaptation maintains score normalization
    
    For any domain-specific term, after applying domain adaptation, the emotion
    scores should remain properly normalized (sum to approximately 1.0) and all
    scores should remain in valid range [0, 1].
    
    This tests that the domain adaptation logic correctly re-normalizes scores
    after boosting specific emotions.
    
    Validates:
    - Req 5.5: System applies context-aware sentiment interpretation
    - Design: Emotion scores are normalized probability distributions
    """
    # Test with negative term
    transcription_negative = f"The market faces {negative_term} according to experts."
    
    # Mock Whisper and sentiment models
    with patch('src.analysis.linguistic.whisper.load_model') as mock_whisper_load, \
         patch('src.analysis.linguistic.AutoTokenizer') as mock_tokenizer_class, \
         patch('src.analysis.linguistic.AutoModelForSequenceClassification') as mock_model_class:
        
        # Mock Whisper model
        mock_whisper_model = MagicMock()
        mock_whisper_load.return_value = mock_whisper_model
        
        # Mock Whisper transcription result
        mock_whisper_result = {
            'text': transcription_negative,
            'segments': [
                {'text': transcription_negative, 'no_speech_prob': 0.1}
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
        mock_logits = torch.tensor([[0.5, 0.5]])
        mock_outputs = MagicMock()
        mock_outputs.logits = mock_logits
        mock_sentiment_model.return_value = mock_outputs
        mock_sentiment_model.to.return_value = mock_sentiment_model
        
        # Create analyzer
        analyzer = LinguisticAnalyzer()
        
        # Force processing
        analyzer.last_process_time = 0.0
        
        # Analyze audio
        result = await analyzer.analyze_audio(audio_frame)
        
        # Property assertions: Normalization after domain adaptation
        if result is not None:
            # 1. All scores must be in valid range
            for emotion, score in result.emotion_scores.items():
                assert 0.0 <= score <= 1.0, \
                    f"After domain adaptation, {emotion} score must be in [0, 1], got {score}"
            
            # 2. Scores must sum to approximately 1.0
            total = sum(result.emotion_scores.values())
            assert 0.95 <= total <= 1.05, \
                f"After domain adaptation, scores must sum to ~1.0, got {total}"
            
            # 3. No score should exceed 1.0 (clamping should work)
            max_score = max(result.emotion_scores.values())
            assert max_score <= 1.0, \
                f"After domain adaptation, max score should not exceed 1.0, got {max_score}"
