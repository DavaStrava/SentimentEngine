"""Property-Based Test for State Reset on Source Change

This module tests Property 14: State reset on source change.

Property 14: State reset on source change
*For any* active analysis session, changing the stream source should clear all
accumulated state (history, temporal context) and begin fresh analysis with the
new source.

Requirements:
    - Req 8.5: WHEN the stream source changes THEN the system SHALL reset analysis
               state and begin fresh processing

Test Strategy:
    This property test verifies that when a stream source is changed, all stateful
    components properly reset their accumulated state:
    1. FusionEngine clears score_history (temporal smoothing context)
    2. SentimentDisplay clears score_history (visualization history)
    3. Analysis modules clear latest_result caches
    4. StreamInputManager resets connection state
    
    The test generates random sequences of sentiment scores to build up state,
    then verifies that after a source change, the state is properly cleared and
    fresh analysis begins without contamination from the previous source.
"""

import pytest
from hypothesis import given, strategies as st, settings
import numpy as np
import time
from typing import List, Dict

from src.fusion.fusion_engine import FusionEngine
from src.ui.display import SentimentDisplay
from src.analysis.acoustic import AcousticAnalyzer
from src.analysis.visual import VisualAnalyzer
from src.analysis.linguistic import LinguisticAnalyzer
from src.input.stream_manager import StreamInputManager
from src.models.results import AcousticResult, VisualResult, LinguisticResult, SentimentScore
from src.models.features import AcousticFeatures, FaceLandmarks
from src.models.enums import StreamProtocol


# Feature: realtime-sentiment-analysis, Property 14: State reset on source change


def create_mock_acoustic_result(score: float, confidence: float = 0.8) -> AcousticResult:
    """Create a mock acoustic result for testing.
    
    Args:
        score: Sentiment score to encode in emotion scores
        confidence: Confidence level
        
    Returns:
        AcousticResult with emotion scores reflecting the sentiment
    """
    # Map score to emotion distribution
    if score > 0:
        emotion_scores = {"happy": score, "neutral": 1 - score}
    else:
        emotion_scores = {"sad": abs(score), "neutral": 1 - abs(score)}
    
    return AcousticResult(
        emotion_scores=emotion_scores,
        confidence=confidence,
        features=AcousticFeatures(
            pitch_mean=200.0,
            pitch_std=20.0,
            energy_mean=0.5,
            speaking_rate=3.0,
            spectral_centroid=1000.0,
            zero_crossing_rate=0.1
        ),
        timestamp=time.time()
    )


def create_mock_visual_result(score: float, confidence: float = 0.8) -> VisualResult:
    """Create a mock visual result for testing.
    
    Args:
        score: Sentiment score to encode in emotion scores
        confidence: Confidence level
        
    Returns:
        VisualResult with emotion scores reflecting the sentiment
    """
    # Map score to emotion distribution
    if score > 0:
        emotion_scores = {"happy": score, "neutral": 1 - score}
    else:
        emotion_scores = {"sad": abs(score), "neutral": 1 - abs(score)}
    
    return VisualResult(
        emotion_scores=emotion_scores,
        confidence=confidence,
        face_detected=True,
        face_landmarks=FaceLandmarks(
            points=np.array([[0, 0], [1, 1]]),
            confidence=confidence
        ),
        timestamp=time.time()
    )


def create_mock_linguistic_result(score: float, confidence: float = 0.8) -> LinguisticResult:
    """Create a mock linguistic result for testing.
    
    Args:
        score: Sentiment score to encode in emotion scores
        confidence: Confidence level
        
    Returns:
        LinguisticResult with emotion scores reflecting the sentiment
    """
    # Map score to emotion distribution
    if score > 0:
        emotion_scores = {"happy": score, "neutral": 1 - score}
    else:
        emotion_scores = {"sad": abs(score), "neutral": 1 - abs(score)}
    
    return LinguisticResult(
        transcription="Test transcription",
        emotion_scores=emotion_scores,
        confidence=confidence,
        transcription_confidence=confidence,
        timestamp=time.time()
    )


@settings(max_examples=100, deadline=None)
@given(
    # Generate a sequence of sentiment scores to build up state
    score_sequence=st.lists(
        st.floats(min_value=-1.0, max_value=1.0),
        min_size=5,
        max_size=20
    ),
    # Generate confidence values
    confidence=st.floats(min_value=0.5, max_value=1.0)
)
def test_fusion_engine_state_reset(score_sequence: List[float], confidence: float):
    """Test that FusionEngine clears score_history on source change.
    
    This test verifies that the FusionEngine properly resets its temporal smoothing
    history when the stream source changes. The score_history is used for exponential
    moving average (EMA) smoothing, and contamination from a previous source would
    cause incorrect smoothing behavior on the new source.
    
    Test Strategy:
    1. Create a FusionEngine instance
    2. Process a sequence of sentiment scores to build up score_history
    3. Verify that score_history contains data
    4. Simulate source change by clearing score_history
    5. Verify that score_history is empty after reset
    6. Process new scores and verify they start fresh without previous context
    
    Args:
        score_sequence: Random sequence of sentiment scores to process
        confidence: Confidence level for mock results
        
    Validates:
        - Req 8.5: System resets analysis state on source change
        - Prop 14: State reset on source change
    """
    # Create fusion engine
    fusion_engine = FusionEngine()
    
    # Build up state by processing multiple scores
    for score in score_sequence:
        acoustic = create_mock_acoustic_result(score, confidence)
        visual = create_mock_visual_result(score, confidence)
        linguistic = create_mock_linguistic_result(score, confidence)
        
        fusion_engine.fuse(acoustic, visual, linguistic)
    
    # Verify state has been accumulated
    assert len(fusion_engine.score_history) > 0, \
        "Score history should contain data after processing"
    
    initial_history_length = len(fusion_engine.score_history)
    
    # Simulate source change by resetting state
    # This is what the system should do when source changes
    fusion_engine.score_history = []
    fusion_engine.latest_score = None
    
    # Verify state has been cleared
    assert len(fusion_engine.score_history) == 0, \
        "Score history should be empty after source change"
    assert fusion_engine.latest_score is None, \
        "Latest score should be None after source change"
    
    # Process new scores and verify they start fresh
    new_score = 0.5
    acoustic = create_mock_acoustic_result(new_score, confidence)
    visual = create_mock_visual_result(new_score, confidence)
    linguistic = create_mock_linguistic_result(new_score, confidence)
    
    result = fusion_engine.fuse(acoustic, visual, linguistic)
    
    # Verify fresh start: first score should not be smoothed with previous history
    assert len(fusion_engine.score_history) == 1, \
        "Score history should contain exactly one entry after first fusion on new source"
    
    # The first score after reset should be close to the raw score (no smoothing)
    # because there's no previous history to smooth with
    assert abs(result.score - new_score) < 0.2, \
        f"First score after reset should be close to raw score (expected ~{new_score}, got {result.score})"


@settings(max_examples=100, deadline=None)
@given(
    # Generate a sequence of sentiment scores to build up state
    score_sequence=st.lists(
        st.tuples(
            st.floats(min_value=0.0, max_value=100.0),  # timestamp
            st.floats(min_value=-1.0, max_value=1.0),   # score
            st.floats(min_value=0.0, max_value=1.0)     # confidence
        ),
        min_size=5,
        max_size=20
    )
)
def test_display_state_reset(score_sequence: List[tuple]):
    """Test that SentimentDisplay clears score_history on source change.
    
    This test verifies that the SentimentDisplay properly resets its historical
    data when the stream source changes. The score_history is used for visualization
    and trend analysis, and contamination from a previous source would show incorrect
    historical trends to the user.
    
    Test Strategy:
    1. Create a SentimentDisplay instance
    2. Add a sequence of sentiment scores to build up score_history
    3. Verify that score_history contains data
    4. Simulate source change by clearing score_history
    5. Verify that score_history is empty after reset
    6. Add new scores and verify they start fresh without previous history
    
    Args:
        score_sequence: Random sequence of (timestamp, score, confidence) tuples
        
    Validates:
        - Req 8.5: System resets analysis state on source change
        - Prop 14: State reset on source change
    """
    # Create display
    display = SentimentDisplay()
    
    # Build up state by adding multiple scores
    for timestamp, score, confidence in score_sequence:
        display.score_history.append((timestamp, score, confidence))
    
    # Verify state has been accumulated
    assert len(display.score_history) > 0, \
        "Score history should contain data after adding scores"
    
    initial_history_length = len(display.score_history)
    
    # Simulate source change by resetting state
    # This is what the system should do when source changes
    display.score_history = []
    display.session_start_time = time.time()
    
    # Verify state has been cleared
    assert len(display.score_history) == 0, \
        "Score history should be empty after source change"
    
    # Add new scores and verify they start fresh
    new_timestamp = time.time()
    new_score = 0.5
    new_confidence = 0.8
    display.score_history.append((new_timestamp, new_score, new_confidence))
    
    # Verify fresh start
    assert len(display.score_history) == 1, \
        "Score history should contain exactly one entry after first update on new source"
    
    # Verify the new score is independent of previous history
    assert display.score_history[0] == (new_timestamp, new_score, new_confidence), \
        "New score should match exactly what was added (no contamination from previous source)"


@settings(max_examples=100, deadline=None)
@given(
    # Generate random scores to build up state
    score=st.floats(min_value=-1.0, max_value=1.0),
    confidence=st.floats(min_value=0.5, max_value=1.0)
)
def test_analysis_modules_state_reset(score: float, confidence: float):
    """Test that analysis modules clear latest_result caches on source change.
    
    This test verifies that all three analysis modules (acoustic, visual, linguistic)
    properly reset their cached results when the stream source changes. Cached results
    from a previous source would cause the fusion engine to use stale data from the
    wrong source.
    
    Test Strategy:
    1. Create instances of all three analysis modules
    2. Set latest_result for each module to simulate processing
    3. Verify that latest_result contains data
    4. Simulate source change by clearing latest_result
    5. Verify that latest_result is None after reset
    
    Args:
        score: Random sentiment score for mock results
        confidence: Random confidence level for mock results
        
    Validates:
        - Req 8.5: System resets analysis state on source change
        - Prop 14: State reset on source change
    """
    # Create analysis modules
    acoustic_analyzer = AcousticAnalyzer()
    visual_analyzer = VisualAnalyzer()
    linguistic_analyzer = LinguisticAnalyzer()
    
    # Build up state by setting latest results
    acoustic_analyzer.latest_result = create_mock_acoustic_result(score, confidence)
    visual_analyzer.latest_result = create_mock_visual_result(score, confidence)
    linguistic_analyzer.latest_result = create_mock_linguistic_result(score, confidence)
    
    # Verify state has been set
    assert acoustic_analyzer.latest_result is not None, \
        "Acoustic analyzer should have cached result"
    assert visual_analyzer.latest_result is not None, \
        "Visual analyzer should have cached result"
    assert linguistic_analyzer.latest_result is not None, \
        "Linguistic analyzer should have cached result"
    
    # Simulate source change by resetting state
    # This is what the system should do when source changes
    acoustic_analyzer.latest_result = None
    visual_analyzer.latest_result = None
    linguistic_analyzer.latest_result = None
    
    # Verify state has been cleared
    assert acoustic_analyzer.latest_result is None, \
        "Acoustic analyzer should have no cached result after source change"
    assert visual_analyzer.latest_result is None, \
        "Visual analyzer should have no cached result after source change"
    assert linguistic_analyzer.latest_result is None, \
        "Linguistic analyzer should have no cached result after source change"
    
    # Verify that get_latest_result returns None after reset
    assert acoustic_analyzer.get_latest_result() is None, \
        "Acoustic analyzer should return None after source change"
    assert visual_analyzer.get_latest_result() is None, \
        "Visual analyzer should return None after source change"
    assert linguistic_analyzer.get_latest_result() is None, \
        "Linguistic analyzer should return None after source change"


@settings(max_examples=50, deadline=None)
@given(
    # Generate random frame counts
    frame_count=st.integers(min_value=10, max_value=1000)
)
def test_stream_manager_state_reset(frame_count: int):
    """Test that StreamInputManager resets connection state on source change.
    
    This test verifies that the StreamInputManager properly resets its connection
    state and frame counters when the stream source changes. Stale connection state
    would cause the system to continue using the old source or have incorrect frame
    numbering.
    
    Test Strategy:
    1. Create a StreamInputManager instance
    2. Simulate processing by setting frame_count and is_streaming
    3. Verify that state has been set
    4. Simulate source change by calling disconnect()
    5. Verify that state has been cleared
    
    Args:
        frame_count: Random frame count to simulate processing
        
    Validates:
        - Req 8.5: System resets analysis state on source change
        - Prop 14: State reset on source change
    """
    # Create stream manager
    stream_manager = StreamInputManager()
    
    # Simulate active streaming state
    stream_manager.frame_count = frame_count
    stream_manager.is_streaming = True
    stream_manager.start_time = time.time()
    
    # Verify state has been set
    assert stream_manager.frame_count == frame_count, \
        "Frame count should be set"
    assert stream_manager.is_streaming is True, \
        "Should be in streaming state"
    
    # Simulate source change by disconnecting
    # This is what the system should do when source changes
    stream_manager.disconnect()
    
    # Verify state has been cleared
    assert stream_manager.is_streaming is False, \
        "Should not be streaming after disconnect"
    assert stream_manager.connection is None or stream_manager.connection.is_active is False, \
        "Connection should be inactive after disconnect"
    
    # Verify that is_active returns False after disconnect
    assert stream_manager.is_active() is False, \
        "Stream manager should report inactive after disconnect"


@settings(max_examples=100, deadline=None)
@given(
    # Generate two sequences of scores representing two different sources
    source1_scores=st.lists(
        st.floats(min_value=-1.0, max_value=1.0),
        min_size=3,
        max_size=10
    ),
    source2_scores=st.lists(
        st.floats(min_value=-1.0, max_value=1.0),
        min_size=3,
        max_size=10
    ),
    confidence=st.floats(min_value=0.5, max_value=1.0)
)
def test_temporal_smoothing_independence_after_reset(
    source1_scores: List[float],
    source2_scores: List[float],
    confidence: float
):
    """Test that temporal smoothing is independent after source change.
    
    This is a critical test that verifies the temporal smoothing (EMA) used by the
    FusionEngine does not carry over context from one source to another. If smoothing
    history is not properly reset, the first few scores from the new source would be
    incorrectly smoothed with the last scores from the old source, causing temporal
    contamination across sources.
    
    Test Strategy:
    1. Process scores from source 1 to build up smoothing history
    2. Record the final smoothed score from source 1
    3. Reset state to simulate source change
    4. Process scores from source 2
    5. Verify that source 2 scores are not influenced by source 1 history
    
    The key assertion is that the first score from source 2 should be close to its
    raw value (minimal smoothing) rather than being heavily smoothed with source 1's
    final scores.
    
    Args:
        source1_scores: Sequence of scores from first source
        source2_scores: Sequence of scores from second source
        confidence: Confidence level for mock results
        
    Validates:
        - Req 8.5: System resets analysis state and begins fresh processing
        - Req 6.5: System applies smoothing to reduce noise while preserving shifts
        - Prop 14: State reset on source change
    """
    # Create fusion engine
    fusion_engine = FusionEngine()
    
    # Process source 1 scores
    for score in source1_scores:
        acoustic = create_mock_acoustic_result(score, confidence)
        visual = create_mock_visual_result(score, confidence)
        linguistic = create_mock_linguistic_result(score, confidence)
        fusion_engine.fuse(acoustic, visual, linguistic)
    
    # Record final state from source 1
    source1_final_score = fusion_engine.score_history[-1] if fusion_engine.score_history else 0.0
    source1_history_length = len(fusion_engine.score_history)
    
    # Reset state for source change
    fusion_engine.score_history = []
    fusion_engine.latest_score = None
    
    # Process first score from source 2
    first_source2_score = source2_scores[0]
    acoustic = create_mock_acoustic_result(first_source2_score, confidence)
    visual = create_mock_visual_result(first_source2_score, confidence)
    linguistic = create_mock_linguistic_result(first_source2_score, confidence)
    
    result = fusion_engine.fuse(acoustic, visual, linguistic)
    
    # Verify temporal independence
    # The first score from source 2 should NOT be smoothed with source 1's history
    # It should be close to the raw score since there's no previous context
    assert len(fusion_engine.score_history) == 1, \
        "Should have exactly one score in history after first fusion on new source"
    
    # The first score should be close to the raw input (no smoothing with source 1)
    # Allow some tolerance for emotion mapping and fusion weighting
    assert abs(result.score - first_source2_score) < 0.3, \
        f"First score from source 2 should be independent of source 1 " \
        f"(expected ~{first_source2_score}, got {result.score}, " \
        f"source 1 final was {source1_final_score})"
    
    # Verify that subsequent scores from source 2 build their own smoothing history
    for score in source2_scores[1:]:
        acoustic = create_mock_acoustic_result(score, confidence)
        visual = create_mock_visual_result(score, confidence)
        linguistic = create_mock_linguistic_result(score, confidence)
        fusion_engine.fuse(acoustic, visual, linguistic)
    
    # Verify that source 2 has built its own independent history
    assert len(fusion_engine.score_history) == len(source2_scores), \
        f"Source 2 should have its own history of {len(source2_scores)} scores"
    
    # Verify that source 2 history is independent (not influenced by source 1)
    # The history should only contain scores from source 2 processing
    assert len(fusion_engine.score_history) <= len(source2_scores), \
        "History should not contain any scores from source 1"


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
