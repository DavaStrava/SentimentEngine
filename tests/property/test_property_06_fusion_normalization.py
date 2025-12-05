"""Property-based tests for fusion score normalization

Feature: realtime-sentiment-analysis, Property 6: Fusion score normalization
Validates: Requirements 6.4
"""

import pytest
import numpy as np
from hypothesis import given, strategies as st, settings

from src.fusion.fusion_engine import FusionEngine
from src.models.results import AcousticResult, VisualResult, LinguisticResult, SentimentScore
from src.models.features import AcousticFeatures


# Custom strategies for generating test data
@st.composite
def emotion_scores_strategy(draw):
    """Generate random emotion score dictionaries.
    
    Emotion scores are dictionaries mapping emotion names to scores in [0, 1].
    The scores should sum to approximately 1.0 (normalized probability distribution).
    """
    emotions = ['happy', 'sad', 'angry', 'neutral', 'fearful', 'surprised', 'disgust']
    
    # Generate random scores
    num_emotions = draw(st.integers(min_value=1, max_value=len(emotions)))
    selected_emotions = draw(st.lists(
        st.sampled_from(emotions),
        min_size=num_emotions,
        max_size=num_emotions,
        unique=True
    ))
    
    # Generate random scores and normalize to sum to 1.0
    raw_scores = [draw(st.floats(min_value=0.0, max_value=1.0)) for _ in selected_emotions]
    total = sum(raw_scores)
    
    if total > 0:
        normalized_scores = {emotion: score / total for emotion, score in zip(selected_emotions, raw_scores)}
    else:
        # Fallback to equal distribution
        normalized_scores = {emotion: 1.0 / len(selected_emotions) for emotion in selected_emotions}
    
    return normalized_scores


@st.composite
def acoustic_result_strategy(draw):
    """Generate random AcousticResult instances."""
    emotion_scores = draw(emotion_scores_strategy())
    confidence = draw(st.floats(min_value=0.0, max_value=1.0))
    timestamp = draw(st.floats(min_value=0.0, max_value=1000.0))
    
    # Features can be None or valid AcousticFeatures
    has_features = draw(st.booleans())
    if has_features:
        features = AcousticFeatures(
            pitch_mean=draw(st.floats(min_value=50.0, max_value=500.0)),
            pitch_std=draw(st.floats(min_value=0.0, max_value=100.0)),
            energy_mean=draw(st.floats(min_value=0.0, max_value=1.0)),
            speaking_rate=draw(st.floats(min_value=0.0, max_value=10.0)),
            spectral_centroid=draw(st.floats(min_value=0.0, max_value=8000.0)),
            zero_crossing_rate=draw(st.floats(min_value=0.0, max_value=1.0))
        )
    else:
        features = None
    
    return AcousticResult(
        emotion_scores=emotion_scores,
        confidence=confidence,
        features=features,
        timestamp=timestamp
    )


@st.composite
def visual_result_strategy(draw):
    """Generate random VisualResult instances."""
    emotion_scores = draw(emotion_scores_strategy())
    confidence = draw(st.floats(min_value=0.0, max_value=1.0))
    timestamp = draw(st.floats(min_value=0.0, max_value=1000.0))
    face_detected = draw(st.booleans())
    
    return VisualResult(
        emotion_scores=emotion_scores,
        confidence=confidence,
        face_detected=face_detected,
        face_landmarks=None,  # Not needed for fusion testing
        timestamp=timestamp
    )


@st.composite
def linguistic_result_strategy(draw):
    """Generate random LinguisticResult instances."""
    emotion_scores = draw(emotion_scores_strategy())
    confidence = draw(st.floats(min_value=0.0, max_value=1.0))
    transcription_confidence = draw(st.floats(min_value=0.0, max_value=1.0))
    timestamp = draw(st.floats(min_value=0.0, max_value=1000.0))
    
    # Generate random transcription text
    transcription = draw(st.text(min_size=0, max_size=100))
    
    return LinguisticResult(
        transcription=transcription,
        emotion_scores=emotion_scores,
        confidence=confidence,
        transcription_confidence=transcription_confidence,
        timestamp=timestamp
    )


# Feature: realtime-sentiment-analysis, Property 6: Fusion score normalization
@settings(max_examples=100, deadline=None)
@given(
    acoustic=st.one_of(st.none(), acoustic_result_strategy()),
    visual=st.one_of(st.none(), visual_result_strategy()),
    linguistic=st.one_of(st.none(), linguistic_result_strategy())
)
def test_fusion_score_normalization(acoustic, visual, linguistic):
    """
    Property 6: Fusion score normalization
    
    For any combination of acoustic, visual, and linguistic results, 
    the Fusion Engine should produce a sentiment score in the range [-1.0, 1.0].
    
    This property verifies that:
    1. The fuse method returns a SentimentScore (not None)
    2. The score field is in the range [-1.0, 1.0]
    3. The score is a valid number (not NaN or infinite)
    4. This holds regardless of:
       - Which modalities are present (all, some, or none)
       - The confidence levels of each modality
       - The emotion scores in each modality
       - The combination of positive and negative emotions
    
    This is a critical correctness property that ensures the fusion output
    is always in the expected range, preventing downstream errors and
    ensuring consistent interpretation of sentiment scores.
    
    Validates:
    - Req 6.4: System normalizes score to consistent range [-1, 1]
    """
    # Create fusion engine
    engine = FusionEngine()
    
    # Fuse the modality results
    result = engine.fuse(acoustic, visual, linguistic)
    
    # Property assertions: Score must always be in [-1.0, 1.0]
    
    # 1. Result must not be None
    assert result is not None, "FusionEngine.fuse() should never return None"
    
    # 2. Result must be a SentimentScore instance
    assert isinstance(result, SentimentScore), \
        f"Result must be SentimentScore instance, got {type(result)}"
    
    # 3. Score must be in valid range [-1.0, 1.0]
    assert -1.0 <= result.score <= 1.0, \
        f"Sentiment score must be in [-1.0, 1.0], got {result.score}"
    
    # 4. Score must be a valid number (not NaN or infinite)
    assert not np.isnan(result.score), \
        f"Sentiment score must not be NaN, got {result.score}"
    assert not np.isinf(result.score), \
        f"Sentiment score must not be infinite, got {result.score}"
    
    # 5. Score must be a numeric type
    assert isinstance(result.score, (int, float)), \
        f"Sentiment score must be numeric, got {type(result.score)}"


# Feature: realtime-sentiment-analysis, Property 6: Fusion score normalization
@settings(max_examples=100, deadline=None)
@given(
    acoustic=acoustic_result_strategy(),
    visual=visual_result_strategy(),
    linguistic=linguistic_result_strategy()
)
def test_fusion_score_normalization_with_all_modalities(acoustic, visual, linguistic):
    """
    Property 6 (variant): Fusion score normalization with all modalities present
    
    When all three modalities are present, the fusion score must still be 
    in the range [-1.0, 1.0], regardless of the specific emotion scores 
    and confidence levels.
    
    This variant specifically tests the case where all modalities contribute,
    which exercises the full fusion algorithm including:
    - Quality-aware weighting
    - Conflict resolution
    - Temporal smoothing
    
    Validates:
    - Req 6.4: System normalizes score to consistent range [-1, 1]
    """
    # Create fusion engine
    engine = FusionEngine()
    
    # Fuse all three modalities
    result = engine.fuse(acoustic, visual, linguistic)
    
    # Property assertions
    assert result is not None
    assert isinstance(result, SentimentScore)
    assert -1.0 <= result.score <= 1.0, \
        f"Sentiment score must be in [-1.0, 1.0] with all modalities, got {result.score}"
    assert not np.isnan(result.score)
    assert not np.isinf(result.score)


# Feature: realtime-sentiment-analysis, Property 6: Fusion score normalization
@settings(max_examples=100, deadline=None)
@given(
    modality_choice=st.sampled_from(['acoustic', 'visual', 'linguistic']),
    result_data=st.one_of(
        acoustic_result_strategy(),
        visual_result_strategy(),
        linguistic_result_strategy()
    )
)
def test_fusion_score_normalization_with_single_modality(modality_choice, result_data):
    """
    Property 6 (variant): Fusion score normalization with single modality
    
    When only one modality is present, the fusion score must still be 
    in the range [-1.0, 1.0].
    
    This variant tests the edge case where only one modality contributes,
    ensuring that the fusion algorithm handles sparse input gracefully.
    
    Validates:
    - Req 6.4: System normalizes score to consistent range [-1, 1]
    """
    # Create fusion engine
    engine = FusionEngine()
    
    # Create inputs with only one modality
    acoustic = None
    visual = None
    linguistic = None
    
    if modality_choice == 'acoustic' and isinstance(result_data, AcousticResult):
        acoustic = result_data
    elif modality_choice == 'visual' and isinstance(result_data, VisualResult):
        visual = result_data
    elif modality_choice == 'linguistic' and isinstance(result_data, LinguisticResult):
        linguistic = result_data
    else:
        # Type mismatch, skip this test case
        return
    
    # Fuse with single modality
    result = engine.fuse(acoustic, visual, linguistic)
    
    # Property assertions
    assert result is not None
    assert isinstance(result, SentimentScore)
    assert -1.0 <= result.score <= 1.0, \
        f"Sentiment score must be in [-1.0, 1.0] with single modality, got {result.score}"
    assert not np.isnan(result.score)
    assert not np.isinf(result.score)


# Feature: realtime-sentiment-analysis, Property 6: Fusion score normalization
@settings(max_examples=100, deadline=None)
@given(st.integers(min_value=1, max_value=10))
def test_fusion_score_normalization_with_extreme_emotions(num_iterations):
    """
    Property 6 (variant): Fusion score normalization with extreme emotion scores
    
    Even with extreme emotion scores (all happy or all sad), the fusion score 
    must remain in the range [-1.0, 1.0].
    
    This variant tests boundary conditions where emotions are maximally positive
    or maximally negative, ensuring the normalization works correctly at extremes.
    
    Validates:
    - Req 6.4: System normalizes score to consistent range [-1, 1]
    """
    # Create fusion engine
    engine = FusionEngine()
    
    # Test with maximally positive emotions (all happy)
    acoustic_positive = AcousticResult(
        emotion_scores={'happy': 1.0},
        confidence=1.0,
        features=None,
        timestamp=0.0
    )
    visual_positive = VisualResult(
        emotion_scores={'happy': 1.0},
        confidence=1.0,
        face_detected=True,
        face_landmarks=None,
        timestamp=0.0
    )
    linguistic_positive = LinguisticResult(
        transcription="This is great!",
        emotion_scores={'happy': 1.0},
        confidence=1.0,
        transcription_confidence=1.0,
        timestamp=0.0
    )
    
    result_positive = engine.fuse(acoustic_positive, visual_positive, linguistic_positive)
    assert -1.0 <= result_positive.score <= 1.0, \
        f"Score with all positive emotions must be in [-1.0, 1.0], got {result_positive.score}"
    
    # Test with maximally negative emotions (all sad)
    acoustic_negative = AcousticResult(
        emotion_scores={'sad': 1.0},
        confidence=1.0,
        features=None,
        timestamp=0.0
    )
    visual_negative = VisualResult(
        emotion_scores={'sad': 1.0},
        confidence=1.0,
        face_detected=True,
        face_landmarks=None,
        timestamp=0.0
    )
    linguistic_negative = LinguisticResult(
        transcription="This is terrible.",
        emotion_scores={'sad': 1.0},
        confidence=1.0,
        transcription_confidence=1.0,
        timestamp=0.0
    )
    
    result_negative = engine.fuse(acoustic_negative, visual_negative, linguistic_negative)
    assert -1.0 <= result_negative.score <= 1.0, \
        f"Score with all negative emotions must be in [-1.0, 1.0], got {result_negative.score}"
    
    # Test with mixed extreme emotions
    acoustic_mixed = AcousticResult(
        emotion_scores={'happy': 1.0},
        confidence=1.0,
        features=None,
        timestamp=0.0
    )
    visual_mixed = VisualResult(
        emotion_scores={'sad': 1.0},
        confidence=1.0,
        face_detected=True,
        face_landmarks=None,
        timestamp=0.0
    )
    linguistic_mixed = LinguisticResult(
        transcription="Mixed feelings",
        emotion_scores={'angry': 1.0},
        confidence=1.0,
        transcription_confidence=1.0,
        timestamp=0.0
    )
    
    result_mixed = engine.fuse(acoustic_mixed, visual_mixed, linguistic_mixed)
    assert -1.0 <= result_mixed.score <= 1.0, \
        f"Score with mixed extreme emotions must be in [-1.0, 1.0], got {result_mixed.score}"


# Feature: realtime-sentiment-analysis, Property 6: Fusion score normalization
@settings(max_examples=100, deadline=None)
@given(
    confidence_a=st.floats(min_value=0.0, max_value=1.0),
    confidence_v=st.floats(min_value=0.0, max_value=1.0),
    confidence_l=st.floats(min_value=0.0, max_value=1.0)
)
def test_fusion_score_normalization_with_varying_confidence(confidence_a, confidence_v, confidence_l):
    """
    Property 6 (variant): Fusion score normalization with varying confidence levels
    
    Regardless of the confidence levels of each modality (including very low 
    or very high confidence), the fusion score must remain in [-1.0, 1.0].
    
    This variant tests that confidence weighting doesn't cause the score to 
    exceed the valid range.
    
    Validates:
    - Req 6.4: System normalizes score to consistent range [-1, 1]
    - Req 6.1: System computes weighted combination based on signal quality
    """
    # Create fusion engine
    engine = FusionEngine()
    
    # Create results with varying confidence levels
    acoustic = AcousticResult(
        emotion_scores={'happy': 0.6, 'neutral': 0.4},
        confidence=confidence_a,
        features=None,
        timestamp=0.0
    )
    visual = VisualResult(
        emotion_scores={'sad': 0.7, 'neutral': 0.3},
        confidence=confidence_v,
        face_detected=True,
        face_landmarks=None,
        timestamp=0.0
    )
    linguistic = LinguisticResult(
        transcription="Test",
        emotion_scores={'angry': 0.5, 'neutral': 0.5},
        confidence=confidence_l,
        transcription_confidence=0.8,
        timestamp=0.0
    )
    
    # Fuse with varying confidence
    result = engine.fuse(acoustic, visual, linguistic)
    
    # Property assertions
    assert result is not None
    assert isinstance(result, SentimentScore)
    assert -1.0 <= result.score <= 1.0, \
        f"Score with confidence ({confidence_a:.2f}, {confidence_v:.2f}, {confidence_l:.2f}) " \
        f"must be in [-1.0, 1.0], got {result.score}"
    assert not np.isnan(result.score)
    assert not np.isinf(result.score)
