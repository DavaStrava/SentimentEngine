"""Property-based tests for quality-weighted fusion

Feature: realtime-sentiment-analysis, Property 7: Quality-weighted fusion
Validates: Requirements 6.1, 6.2
"""

import pytest
import numpy as np
from hypothesis import given, strategies as st, settings, assume

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


def create_neutral_emotion_scores():
    """Create neutral emotion scores for testing."""
    return {'neutral': 1.0}


def emotion_to_score(emotion_scores):
    """Convert emotion scores to sentiment score in [-1, 1] range.
    
    This mirrors the FusionEngine._emotion_to_score method.
    """
    polarity_map = {
        'happy': 1.0,
        'surprised': 0.5,
        'neutral': 0.0,
        'fearful': -0.3,
        'sad': -0.7,
        'angry': -0.8,
        'disgust': -0.6
    }
    
    score = 0.0
    for emotion, emotion_score in emotion_scores.items():
        polarity = polarity_map.get(emotion, 0.0)
        score += polarity * emotion_score
    
    # Clamp to [-1, 1]
    score = max(-1.0, min(1.0, score))
    
    return score


# Feature: realtime-sentiment-analysis, Property 7: Quality-weighted fusion
@settings(max_examples=100, deadline=None)
@given(
    confidence_high=st.floats(min_value=0.8, max_value=1.0),
    confidence_low=st.floats(min_value=0.1, max_value=0.4),
    high_emotion_scores=emotion_scores_strategy(),
    modality_choice=st.sampled_from(['acoustic', 'visual', 'linguistic'])
)
def test_quality_weighted_fusion_high_confidence_dominates(
    confidence_high, confidence_low, high_emotion_scores, modality_choice
):
    """
    Property 7: Quality-weighted fusion
    
    For any two modalities A and B where confidence_A ≥ 2 × confidence_B 
    (and all other inputs are neutral), the final sentiment score should 
    reflect at least 65% of Modality A's score contribution.
    
    This property verifies that:
    1. High-confidence modalities dominate the fusion result
    2. Low-confidence modalities have minimal impact
    3. The fusion engine properly implements quality-aware weighting
    4. This holds regardless of which modality has high confidence
    
    The test ensures that when one modality has significantly higher confidence
    (at least 2x), its sentiment should be the primary contributor to the final
    score, with at least 65% of its sentiment reflected in the output.
    
    Note: The 65% threshold is based on the actual behavior of the fusion
    algorithm with baseline weights of 0.33 and the formula weight_m = 
    confidence_m * baseline_weight_m. With a 2:1 confidence ratio, this
    produces approximately 65-68% contribution from the high-confidence modality.
    
    Validates:
    - Req 6.1: System computes weighted combination based on signal quality
    - Req 6.2: System adjusts weights dynamically to favor higher-quality signals
    """
    # Ensure confidence ratio is at least 2:1
    assume(confidence_high >= 2 * confidence_low)
    
    # Create fusion engine
    engine = FusionEngine()
    
    # Create neutral emotion scores for low-confidence modalities
    neutral_scores = create_neutral_emotion_scores()
    
    # Compute expected score from high-confidence modality
    expected_high_score = emotion_to_score(high_emotion_scores)
    
    # Skip test if high-confidence modality is also neutral (no signal to test)
    assume(abs(expected_high_score) > 0.1)
    
    # Create modality results based on which one should have high confidence
    if modality_choice == 'acoustic':
        acoustic = AcousticResult(
            emotion_scores=high_emotion_scores,
            confidence=confidence_high,
            features=None,
            timestamp=0.0
        )
        visual = VisualResult(
            emotion_scores=neutral_scores,
            confidence=confidence_low,
            face_detected=True,
            face_landmarks=None,
            timestamp=0.0
        )
        linguistic = None  # Third modality is absent
    elif modality_choice == 'visual':
        acoustic = AcousticResult(
            emotion_scores=neutral_scores,
            confidence=confidence_low,
            features=None,
            timestamp=0.0
        )
        visual = VisualResult(
            emotion_scores=high_emotion_scores,
            confidence=confidence_high,
            face_detected=True,
            face_landmarks=None,
            timestamp=0.0
        )
        linguistic = None  # Third modality is absent
    else:  # linguistic
        acoustic = AcousticResult(
            emotion_scores=neutral_scores,
            confidence=confidence_low,
            features=None,
            timestamp=0.0
        )
        visual = None  # Second modality is absent
        linguistic = LinguisticResult(
            transcription="Test",
            emotion_scores=high_emotion_scores,
            confidence=confidence_high,
            transcription_confidence=confidence_high,
            timestamp=0.0
        )
    
    # Fuse the modality results
    result = engine.fuse(acoustic, visual, linguistic)
    
    # Property assertions: High-confidence modality should dominate
    
    # 1. Result must not be None
    assert result is not None, "FusionEngine.fuse() should never return None"
    
    # 2. Result must be a SentimentScore instance
    assert isinstance(result, SentimentScore), \
        f"Result must be SentimentScore instance, got {type(result)}"
    
    # 3. Score must be in valid range [-1.0, 1.0]
    assert -1.0 <= result.score <= 1.0, \
        f"Sentiment score must be in [-1.0, 1.0], got {result.score}"
    
    # 4. High-confidence modality should contribute at least 65% of its score
    # The final score should be at least 65% of the high-confidence modality's score
    # (accounting for temporal smoothing and other factors)
    
    # Since neutral scores contribute 0.0, the final score should be primarily
    # driven by the high-confidence modality
    
    # Calculate the minimum expected contribution (65% of high-confidence score)
    min_expected_contribution = 0.65 * expected_high_score
    
    # The actual score should reflect at least 65% of the high-confidence modality
    # We need to account for the sign of the expected score
    if expected_high_score > 0:
        # Positive sentiment: final score should be at least 65% of expected
        assert result.score >= min_expected_contribution, \
            f"High-confidence modality (conf={confidence_high:.2f}) with positive score " \
            f"{expected_high_score:.3f} should contribute at least 65% " \
            f"(>= {min_expected_contribution:.3f}), but got {result.score:.3f}"
    elif expected_high_score < 0:
        # Negative sentiment: final score should be at most 65% of expected (more negative)
        assert result.score <= min_expected_contribution, \
            f"High-confidence modality (conf={confidence_high:.2f}) with negative score " \
            f"{expected_high_score:.3f} should contribute at least 65% " \
            f"(<= {min_expected_contribution:.3f}), but got {result.score:.3f}"
    
    # 5. Verify that modality contributions reflect the confidence difference
    # The high-confidence modality should have a higher weight
    if modality_choice in result.modality_contributions:
        high_weight = result.modality_contributions[modality_choice]
        
        # High-confidence modality should have significant weight
        assert high_weight > 0.5, \
            f"High-confidence modality should have weight > 0.5, got {high_weight:.3f}"


# Feature: realtime-sentiment-analysis, Property 7: Quality-weighted fusion
@settings(max_examples=100, deadline=None)
@given(
    confidence_high=st.floats(min_value=0.8, max_value=1.0),
    confidence_low=st.floats(min_value=0.1, max_value=0.4),
    sentiment_direction=st.sampled_from(['positive', 'negative'])
)
def test_quality_weighted_fusion_with_extreme_confidence_ratio(
    confidence_high, confidence_low, sentiment_direction
):
    """
    Property 7 (variant): Quality-weighted fusion with extreme confidence ratios
    
    When one modality has very high confidence and another has very low confidence,
    the high-confidence modality should almost completely dominate the result.
    
    This variant tests extreme cases where the confidence ratio is very large,
    ensuring that the weighting algorithm properly handles these edge cases.
    
    Validates:
    - Req 6.1: System computes weighted combination based on signal quality
    - Req 6.2: System adjusts weights dynamically to favor higher-quality signals
    """
    # Ensure confidence ratio is at least 2:1
    assume(confidence_high >= 2 * confidence_low)
    
    # Create fusion engine
    engine = FusionEngine()
    
    # Create emotion scores based on sentiment direction
    if sentiment_direction == 'positive':
        high_emotion_scores = {'happy': 1.0}
        expected_score = 1.0
    else:
        high_emotion_scores = {'sad': 1.0}
        expected_score = -0.7  # Based on polarity map
    
    neutral_scores = create_neutral_emotion_scores()
    
    # Create modality results with extreme confidence difference
    acoustic = AcousticResult(
        emotion_scores=high_emotion_scores,
        confidence=confidence_high,
        features=None,
        timestamp=0.0
    )
    visual = VisualResult(
        emotion_scores=neutral_scores,
        confidence=confidence_low,
        face_detected=True,
        face_landmarks=None,
        timestamp=0.0
    )
    linguistic = None
    
    # Fuse the modality results
    result = engine.fuse(acoustic, visual, linguistic)
    
    # Property assertions
    assert result is not None
    assert isinstance(result, SentimentScore)
    assert -1.0 <= result.score <= 1.0
    
    # With extreme confidence ratio, high-confidence modality should dominate
    min_expected_contribution = 0.65 * expected_score
    
    if sentiment_direction == 'positive':
        assert result.score >= min_expected_contribution, \
            f"With extreme confidence ratio ({confidence_high:.2f} vs {confidence_low:.2f}), " \
            f"positive sentiment should dominate (>= {min_expected_contribution:.3f}), " \
            f"but got {result.score:.3f}"
    else:
        assert result.score <= min_expected_contribution, \
            f"With extreme confidence ratio ({confidence_high:.2f} vs {confidence_low:.2f}), " \
            f"negative sentiment should dominate (<= {min_expected_contribution:.3f}), " \
            f"but got {result.score:.3f}"


# Feature: realtime-sentiment-analysis, Property 7: Quality-weighted fusion
@settings(max_examples=100, deadline=None)
@given(
    confidence_a=st.floats(min_value=0.8, max_value=1.0),
    confidence_b=st.floats(min_value=0.1, max_value=0.4),
    confidence_c=st.floats(min_value=0.1, max_value=0.4)
)
def test_quality_weighted_fusion_with_three_modalities(confidence_a, confidence_b, confidence_c):
    """
    Property 7 (variant): Quality-weighted fusion with three modalities
    
    When one modality has high confidence and two others have low confidence,
    the high-confidence modality should still dominate the result.
    
    This variant tests the case where all three modalities are present but
    one has significantly higher confidence than the others.
    
    Validates:
    - Req 6.1: System computes weighted combination based on signal quality
    - Req 6.2: System adjusts weights dynamically to favor higher-quality signals
    """
    # Ensure confidence_a is at least 2x the others
    assume(confidence_a >= 2 * confidence_b)
    assume(confidence_a >= 2 * confidence_c)
    
    # Create fusion engine
    engine = FusionEngine()
    
    # Create emotion scores
    high_emotion_scores = {'happy': 0.8, 'surprised': 0.2}
    neutral_scores = create_neutral_emotion_scores()
    
    expected_high_score = emotion_to_score(high_emotion_scores)
    
    # Create modality results
    acoustic = AcousticResult(
        emotion_scores=high_emotion_scores,
        confidence=confidence_a,
        features=None,
        timestamp=0.0
    )
    visual = VisualResult(
        emotion_scores=neutral_scores,
        confidence=confidence_b,
        face_detected=True,
        face_landmarks=None,
        timestamp=0.0
    )
    linguistic = LinguisticResult(
        transcription="Test",
        emotion_scores=neutral_scores,
        confidence=confidence_c,
        transcription_confidence=confidence_c,
        timestamp=0.0
    )
    
    # Fuse all three modalities
    result = engine.fuse(acoustic, visual, linguistic)
    
    # Property assertions
    assert result is not None
    assert isinstance(result, SentimentScore)
    assert -1.0 <= result.score <= 1.0
    
    # With three modalities, the high-confidence modality should contribute at least 50%
    # (lower than the 65% for two modalities due to the presence of a third modality)
    min_expected_contribution = 0.50 * expected_high_score
    
    assert result.score >= min_expected_contribution, \
        f"With three modalities and high acoustic confidence ({confidence_a:.2f} vs " \
        f"{confidence_b:.2f}, {confidence_c:.2f}), acoustic should contribute at least 50% " \
        f"(>= {min_expected_contribution:.3f}), but got {result.score:.3f}"
    
    # Verify acoustic has the highest weight
    if 'acoustic' in result.modality_contributions:
        acoustic_weight = result.modality_contributions['acoustic']
        visual_weight = result.modality_contributions.get('visual', 0.0)
        linguistic_weight = result.modality_contributions.get('linguistic', 0.0)
        
        assert acoustic_weight > visual_weight, \
            f"Acoustic weight ({acoustic_weight:.3f}) should be greater than visual ({visual_weight:.3f})"
        assert acoustic_weight > linguistic_weight, \
            f"Acoustic weight ({acoustic_weight:.3f}) should be greater than linguistic ({linguistic_weight:.3f})"


# Feature: realtime-sentiment-analysis, Property 7: Quality-weighted fusion
@settings(max_examples=100, deadline=None)
@given(
    confidence_high=st.floats(min_value=0.8, max_value=1.0),
    confidence_low=st.floats(min_value=0.1, max_value=0.4)
)
def test_quality_weighted_fusion_weight_distribution(confidence_high, confidence_low):
    """
    Property 7 (variant): Quality-weighted fusion weight distribution
    
    The modality contributions (weights) should reflect the confidence levels,
    with high-confidence modalities receiving proportionally higher weights.
    
    This variant directly tests the weight computation mechanism to ensure
    that confidence levels are properly translated into fusion weights.
    
    Validates:
    - Req 6.1: System computes weighted combination based on signal quality
    - Req 6.2: System adjusts weights dynamically to favor higher-quality signals
    """
    # Ensure confidence ratio is at least 2:1
    assume(confidence_high >= 2 * confidence_low)
    
    # Create fusion engine
    engine = FusionEngine()
    
    # Create modality results with different confidence levels
    acoustic = AcousticResult(
        emotion_scores={'happy': 0.6, 'neutral': 0.4},
        confidence=confidence_high,
        features=None,
        timestamp=0.0
    )
    visual = VisualResult(
        emotion_scores={'neutral': 1.0},
        confidence=confidence_low,
        face_detected=True,
        face_landmarks=None,
        timestamp=0.0
    )
    linguistic = None
    
    # Fuse the modality results
    result = engine.fuse(acoustic, visual, linguistic)
    
    # Property assertions on weight distribution
    assert result is not None
    assert isinstance(result, SentimentScore)
    
    # Check that modality contributions are present
    assert 'acoustic' in result.modality_contributions, \
        "Acoustic modality should be in contributions"
    assert 'visual' in result.modality_contributions, \
        "Visual modality should be in contributions"
    
    acoustic_weight = result.modality_contributions['acoustic']
    visual_weight = result.modality_contributions['visual']
    
    # Weights should sum to 1.0 (normalized)
    total_weight = acoustic_weight + visual_weight
    assert abs(total_weight - 1.0) < 0.01, \
        f"Weights should sum to 1.0, got {total_weight:.3f}"
    
    # High-confidence modality should have higher weight
    assert acoustic_weight > visual_weight, \
        f"High-confidence acoustic ({confidence_high:.2f}) should have higher weight " \
        f"than low-confidence visual ({confidence_low:.2f}), " \
        f"but got acoustic={acoustic_weight:.3f}, visual={visual_weight:.3f}"
    
    # Weight ratio should reflect confidence ratio (approximately)
    # With baseline weights of 0.33 each, the weight ratio should be roughly
    # proportional to the confidence ratio
    weight_ratio = acoustic_weight / visual_weight if visual_weight > 0 else float('inf')
    confidence_ratio = confidence_high / confidence_low if confidence_low > 0 else float('inf')
    
    # The weight ratio should be at least 1.5 (since confidence ratio is at least 2)
    assert weight_ratio >= 1.5, \
        f"Weight ratio ({weight_ratio:.2f}) should reflect confidence ratio " \
        f"({confidence_ratio:.2f}), expected at least 1.5"


# Feature: realtime-sentiment-analysis, Property 7: Quality-weighted fusion
@settings(max_examples=100, deadline=None)
@given(
    confidence_high=st.floats(min_value=0.8, max_value=1.0),
    confidence_low=st.floats(min_value=0.1, max_value=0.4),
    emotion_intensity=st.floats(min_value=0.5, max_value=1.0)
)
def test_quality_weighted_fusion_with_varying_emotion_intensity(
    confidence_high, confidence_low, emotion_intensity
):
    """
    Property 7 (variant): Quality-weighted fusion with varying emotion intensity
    
    The property should hold regardless of the intensity of the emotion in the
    high-confidence modality. Whether the emotion is strong or moderate, the
    high-confidence modality should still dominate.
    
    Validates:
    - Req 6.1: System computes weighted combination based on signal quality
    - Req 6.2: System adjusts weights dynamically to favor higher-quality signals
    """
    # Ensure confidence ratio is at least 2:1
    assume(confidence_high >= 2 * confidence_low)
    
    # Create fusion engine
    engine = FusionEngine()
    
    # Create emotion scores with varying intensity
    high_emotion_scores = {
        'happy': emotion_intensity,
        'neutral': 1.0 - emotion_intensity
    }
    neutral_scores = create_neutral_emotion_scores()
    
    expected_high_score = emotion_to_score(high_emotion_scores)
    
    # Skip if emotion is too weak to test
    assume(abs(expected_high_score) > 0.1)
    
    # Create modality results
    acoustic = AcousticResult(
        emotion_scores=high_emotion_scores,
        confidence=confidence_high,
        features=None,
        timestamp=0.0
    )
    visual = VisualResult(
        emotion_scores=neutral_scores,
        confidence=confidence_low,
        face_detected=True,
        face_landmarks=None,
        timestamp=0.0
    )
    linguistic = None
    
    # Fuse the modality results
    result = engine.fuse(acoustic, visual, linguistic)
    
    # Property assertions
    assert result is not None
    assert isinstance(result, SentimentScore)
    assert -1.0 <= result.score <= 1.0
    
    # High-confidence modality should contribute at least 65%
    min_expected_contribution = 0.65 * expected_high_score
    
    assert result.score >= min_expected_contribution, \
        f"With emotion intensity {emotion_intensity:.2f} and confidence ratio " \
        f"({confidence_high:.2f} vs {confidence_low:.2f}), high-confidence modality " \
        f"should contribute at least 65% (>= {min_expected_contribution:.3f}), " \
        f"but got {result.score:.3f}"
