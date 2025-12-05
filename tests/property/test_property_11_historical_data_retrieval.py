"""Property-based tests for historical data retrieval

Feature: realtime-sentiment-analysis, Property 11: Historical data retrieval
Validates: Requirements 7.3
"""

import pytest
import time
from hypothesis import given, strategies as st, settings, assume

from src.ui.display import SentimentDisplay
from src.models.results import SentimentScore


# Custom strategies for generating test data
@st.composite
def sentiment_score_strategy(draw, timestamp=None):
    """Generate random SentimentScore instances with optional fixed timestamp."""
    score = draw(st.floats(min_value=-1.0, max_value=1.0))
    confidence = draw(st.floats(min_value=0.0, max_value=1.0))
    
    # Generate modality contributions
    num_modalities = draw(st.integers(min_value=1, max_value=3))
    modalities = ['acoustic', 'visual', 'linguistic']
    selected_modalities = draw(st.lists(
        st.sampled_from(modalities),
        min_size=num_modalities,
        max_size=num_modalities,
        unique=True
    ))
    
    # Generate random weights and normalize
    raw_weights = [draw(st.floats(min_value=0.1, max_value=1.0)) for _ in selected_modalities]
    total = sum(raw_weights)
    modality_contributions = {
        modality: weight / total 
        for modality, weight in zip(selected_modalities, raw_weights)
    }
    
    # Generate emotion breakdown
    emotions = ['happy', 'sad', 'angry', 'neutral']
    num_emotions = draw(st.integers(min_value=1, max_value=len(emotions)))
    selected_emotions = draw(st.lists(
        st.sampled_from(emotions),
        min_size=num_emotions,
        max_size=num_emotions,
        unique=True
    ))
    raw_scores = [draw(st.floats(min_value=0.1, max_value=1.0)) for _ in selected_emotions]
    total_emotion = sum(raw_scores)
    emotion_breakdown = {
        emotion: score / total_emotion 
        for emotion, score in zip(selected_emotions, raw_scores)
    }
    
    # Use provided timestamp or generate one
    if timestamp is None:
        timestamp = draw(st.floats(min_value=0.0, max_value=10000.0))
    
    return SentimentScore(
        score=score,
        confidence=confidence,
        modality_contributions=modality_contributions,
        emotion_breakdown=emotion_breakdown,
        timestamp=timestamp
    )


@st.composite
def sentiment_sequence_strategy(draw):
    """Generate a sequence of sentiment scores with increasing timestamps.
    
    Timestamps are generated relative to current time to avoid being trimmed
    by the history duration limit in SentimentDisplay.update_score().
    """
    # Generate number of scores
    num_scores = draw(st.integers(min_value=1, max_value=50))
    
    # Use current time as base, minus a small offset to ensure all scores are recent
    current_time = time.time()
    # Start from 50 seconds ago (well within default 60 second history duration)
    base_timestamp = current_time - 50.0
    
    # Generate scores with increasing timestamps
    scores = []
    current_timestamp = base_timestamp
    
    for i in range(num_scores):
        # Increment timestamp by a small random amount (0.1 to 1.0 seconds)
        # Keep total span under 50 seconds to stay within history window
        time_delta = draw(st.floats(min_value=0.1, max_value=1.0))
        current_timestamp += time_delta
        
        # Generate sentiment score with this timestamp
        sentiment = draw(sentiment_score_strategy(timestamp=current_timestamp))
        scores.append(sentiment)
    
    return scores


# Feature: realtime-sentiment-analysis, Property 11: Historical data retrieval
@settings(max_examples=100, deadline=None)
@given(sentiments=sentiment_sequence_strategy())
def test_historical_data_returns_all_scores_in_order(sentiments):
    """
    Property 11: Historical data retrieval
    
    For any sequence of sentiment scores generated during a session, requesting 
    historical data should return all scores in chronological order with no 
    missing entries.
    
    This property verifies that:
    1. All sentiment scores added to the display are stored in history
    2. Scores are returned in chronological order (by timestamp)
    3. No scores are missing from the history
    4. The order is preserved regardless of:
       - The number of scores (1 to many)
       - The time intervals between scores
       - The sentiment values
       - The confidence levels
       - Which modalities contributed
    
    This is a critical correctness property that ensures users can reliably
    access the complete history of sentiment analysis for the current session,
    enabling trend analysis and review of past emotional states.
    
    Validates:
    - Req 7.3: WHEN the user requests historical data THEN the system SHALL 
               provide sentiment score history for the current session
    """
    # Create display instance
    display = SentimentDisplay()
    
    # Add all sentiment scores to display
    for sentiment in sentiments:
        display.update_score(sentiment)
    
    # Property assertions: Historical data must contain all scores in order
    
    # 1. History must contain exactly the same number of entries as added
    assert len(display.score_history) == len(sentiments), \
        f"History must contain all {len(sentiments)} scores, got {len(display.score_history)}"
    
    # 2. Each entry in history must be a tuple of (timestamp, score, confidence)
    for i, entry in enumerate(display.score_history):
        assert isinstance(entry, tuple), \
            f"History entry {i} must be a tuple, got {type(entry)}"
        assert len(entry) == 3, \
            f"History entry {i} must have 3 elements (timestamp, score, confidence), got {len(entry)}"
        
        timestamp, score, confidence = entry
        assert isinstance(timestamp, (int, float)), \
            f"History entry {i} timestamp must be numeric, got {type(timestamp)}"
        assert isinstance(score, (int, float)), \
            f"History entry {i} score must be numeric, got {type(score)}"
        assert isinstance(confidence, (int, float)), \
            f"History entry {i} confidence must be numeric, got {type(confidence)}"
    
    # 3. History must be in chronological order (timestamps increasing)
    timestamps = [entry[0] for entry in display.score_history]
    for i in range(len(timestamps) - 1):
        assert timestamps[i] <= timestamps[i + 1], \
            f"Timestamps must be in chronological order: " \
            f"timestamp[{i}]={timestamps[i]} > timestamp[{i+1}]={timestamps[i+1]}"
    
    # 4. Each sentiment score must be present in history with correct values
    for i, sentiment in enumerate(sentiments):
        history_entry = display.score_history[i]
        history_timestamp, history_score, history_confidence = history_entry
        
        # Timestamp must match
        assert abs(history_timestamp - sentiment.timestamp) < 1e-6, \
            f"Score {i}: timestamp mismatch. Expected {sentiment.timestamp}, got {history_timestamp}"
        
        # Score must match
        assert abs(history_score - sentiment.score) < 1e-6, \
            f"Score {i}: score mismatch. Expected {sentiment.score}, got {history_score}"
        
        # Confidence must match
        assert abs(history_confidence - sentiment.confidence) < 1e-6, \
            f"Score {i}: confidence mismatch. Expected {sentiment.confidence}, got {history_confidence}"
    
    # 5. No duplicate entries (each timestamp should be unique or properly ordered)
    # This ensures no scores are duplicated in history
    for i in range(len(display.score_history)):
        for j in range(i + 1, len(display.score_history)):
            entry_i = display.score_history[i]
            entry_j = display.score_history[j]
            
            # If timestamps are the same, entries should be identical
            # (though this is unlikely with generated data)
            if abs(entry_i[0] - entry_j[0]) < 1e-6:
                assert entry_i == entry_j, \
                    f"Entries {i} and {j} have same timestamp but different values"


# Feature: realtime-sentiment-analysis, Property 11: Historical data retrieval
@settings(max_examples=100, deadline=None)
@given(sentiments=sentiment_sequence_strategy())
def test_historical_data_respects_duration_limit(sentiments):
    """
    Property 11 (variant): Historical data retrieval with duration limit
    
    When a history duration limit is set, the system should only return scores
    within that time window, but all scores within the window should be present
    in chronological order.
    
    This variant tests that the history trimming functionality works correctly
    while still maintaining chronological order and completeness within the
    specified time window. Uses the default 60-second history duration.
    
    Validates:
    - Req 7.3: System provides sentiment score history for the current session
    """
    # Assume we have at least 2 scores to make the test meaningful
    assume(len(sentiments) >= 2)
    
    # Create display instance with default history duration (60 seconds)
    display = SentimentDisplay()
    
    # Add all sentiment scores to display
    for sentiment in sentiments:
        display.update_score(sentiment)
    
    # Property assertions: History respects duration limit
    
    # 1. All entries in history must be within the duration window (relative to current time)
    current_time = time.time()
    for entry in display.score_history:
        timestamp = entry[0]
        age = current_time - timestamp
        assert age <= display.history_duration + 1.0, \
            f"History entry with timestamp {timestamp} is too old " \
            f"(age={age:.1f}s, limit={display.history_duration}s)"
    
    # 2. History must still be in chronological order
    timestamps = [entry[0] for entry in display.score_history]
    for i in range(len(timestamps) - 1):
        assert timestamps[i] <= timestamps[i + 1], \
            f"Timestamps must be in chronological order even with duration limit"
    
    # 3. All scores within the window should be present
    # Since our test data is generated to be within the last 50 seconds,
    # and default history is 60 seconds, all scores should be present
    assert len(display.score_history) == len(sentiments), \
        f"History should contain all {len(sentiments)} scores, " \
        f"got {len(display.score_history)}"


# Feature: realtime-sentiment-analysis, Property 11: Historical data retrieval
@settings(max_examples=100, deadline=None)
@given(sentiments=sentiment_sequence_strategy())
def test_historical_data_preserves_score_values(sentiments):
    """
    Property 11 (variant): Historical data preserves exact score values
    
    The historical data must preserve the exact score and confidence values
    from the original sentiment scores, not just the timestamps.
    
    This variant ensures data integrity - that the values retrieved from
    history match exactly what was stored.
    
    Validates:
    - Req 7.3: System provides sentiment score history for the current session
    """
    # Create display instance
    display = SentimentDisplay()
    
    # Add all sentiment scores to display
    for sentiment in sentiments:
        display.update_score(sentiment)
    
    # Property assertions: Values are preserved exactly
    
    for i, sentiment in enumerate(sentiments):
        history_entry = display.score_history[i]
        history_timestamp, history_score, history_confidence = history_entry
        
        # 1. Score value must be preserved exactly (within floating point precision)
        assert abs(history_score - sentiment.score) < 1e-9, \
            f"Score {i}: score value not preserved. " \
            f"Expected {sentiment.score}, got {history_score}, " \
            f"difference={abs(history_score - sentiment.score)}"
        
        # 2. Confidence value must be preserved exactly
        assert abs(history_confidence - sentiment.confidence) < 1e-9, \
            f"Score {i}: confidence value not preserved. " \
            f"Expected {sentiment.confidence}, got {history_confidence}, " \
            f"difference={abs(history_confidence - sentiment.confidence)}"
        
        # 3. Timestamp must be preserved exactly
        assert abs(history_timestamp - sentiment.timestamp) < 1e-9, \
            f"Score {i}: timestamp not preserved. " \
            f"Expected {sentiment.timestamp}, got {history_timestamp}, " \
            f"difference={abs(history_timestamp - sentiment.timestamp)}"
        
        # 4. Score must be in valid range
        assert -1.0 <= history_score <= 1.0, \
            f"Score {i}: history score must be in [-1, 1], got {history_score}"
        
        # 5. Confidence must be in valid range
        assert 0.0 <= history_confidence <= 1.0, \
            f"Score {i}: history confidence must be in [0, 1], got {history_confidence}"


# Feature: realtime-sentiment-analysis, Property 11: Historical data retrieval
@settings(max_examples=100, deadline=None)
@given(
    first_batch=sentiment_sequence_strategy(),
    second_batch=sentiment_sequence_strategy()
)
def test_historical_data_accumulates_across_updates(first_batch, second_batch):
    """
    Property 11 (variant): Historical data accumulates across multiple updates
    
    When sentiment scores are added in multiple batches, the history should
    accumulate all scores in chronological order, maintaining completeness
    across the entire session.
    
    This variant tests that history accumulation works correctly when scores
    are added incrementally over time, which is the normal operating mode.
    
    Validates:
    - Req 7.3: System provides sentiment score history for the current session
    """
    # Assume we have at least one score in each batch
    assume(len(first_batch) >= 1 and len(second_batch) >= 1)
    
    # Adjust second batch timestamps to come after first batch
    if first_batch and second_batch:
        last_first_timestamp = first_batch[-1].timestamp
        first_second_timestamp = second_batch[0].timestamp
        
        # If second batch starts before first batch ends, adjust timestamps
        if first_second_timestamp <= last_first_timestamp:
            time_offset = last_first_timestamp - first_second_timestamp + 1.0
            
            # Create new second batch with adjusted timestamps
            adjusted_second_batch = []
            for sentiment in second_batch:
                adjusted_sentiment = SentimentScore(
                    score=sentiment.score,
                    confidence=sentiment.confidence,
                    modality_contributions=sentiment.modality_contributions,
                    emotion_breakdown=sentiment.emotion_breakdown,
                    timestamp=sentiment.timestamp + time_offset
                )
                adjusted_second_batch.append(adjusted_sentiment)
            second_batch = adjusted_second_batch
    
    # Create display instance
    display = SentimentDisplay()
    
    # Add first batch
    for sentiment in first_batch:
        display.update_score(sentiment)
    
    # Verify first batch is in history
    assert len(display.score_history) == len(first_batch), \
        f"After first batch, history should have {len(first_batch)} entries"
    
    # Add second batch
    for sentiment in second_batch:
        display.update_score(sentiment)
    
    # Property assertions: History accumulates correctly
    
    # 1. Total history size should be sum of both batches
    total_expected = len(first_batch) + len(second_batch)
    assert len(display.score_history) == total_expected, \
        f"History should contain {total_expected} total scores, " \
        f"got {len(display.score_history)}"
    
    # 2. History must still be in chronological order
    timestamps = [entry[0] for entry in display.score_history]
    for i in range(len(timestamps) - 1):
        assert timestamps[i] <= timestamps[i + 1], \
            f"Timestamps must remain in chronological order after multiple batches"
    
    # 3. All scores from both batches must be present
    all_sentiments = first_batch + second_batch
    for i, sentiment in enumerate(all_sentiments):
        history_entry = display.score_history[i]
        history_timestamp, history_score, history_confidence = history_entry
        
        assert abs(history_timestamp - sentiment.timestamp) < 1e-6, \
            f"Score {i}: timestamp mismatch in accumulated history"
        assert abs(history_score - sentiment.score) < 1e-6, \
            f"Score {i}: score mismatch in accumulated history"
        assert abs(history_confidence - sentiment.confidence) < 1e-6, \
            f"Score {i}: confidence mismatch in accumulated history"


# Feature: realtime-sentiment-analysis, Property 11: Historical data retrieval
@settings(max_examples=100, deadline=None)
@given(sentiment=sentiment_score_strategy())
def test_historical_data_single_score(sentiment):
    """
    Property 11 (variant): Historical data retrieval with single score
    
    Even with a single sentiment score, the history should contain that score
    with all its values preserved correctly.
    
    This variant tests the edge case of minimal history (one score), ensuring
    the system works correctly even at the start of a session.
    
    Validates:
    - Req 7.3: System provides sentiment score history for the current session
    """
    # Create display instance
    display = SentimentDisplay()
    
    # Adjust timestamp to be recent (within history window)
    recent_sentiment = SentimentScore(
        score=sentiment.score,
        confidence=sentiment.confidence,
        modality_contributions=sentiment.modality_contributions,
        emotion_breakdown=sentiment.emotion_breakdown,
        timestamp=time.time() - 10.0  # 10 seconds ago, well within 60 second window
    )
    
    # Add single sentiment score
    display.update_score(recent_sentiment)
    
    # Property assertions: Single score is stored correctly
    
    # 1. History must contain exactly one entry
    assert len(display.score_history) == 1, \
        f"History should contain exactly 1 score, got {len(display.score_history)}"
    
    # 2. The entry must match the input sentiment
    history_entry = display.score_history[0]
    history_timestamp, history_score, history_confidence = history_entry
    
    assert abs(history_timestamp - recent_sentiment.timestamp) < 1e-9, \
        f"Timestamp mismatch: expected {recent_sentiment.timestamp}, got {history_timestamp}"
    assert abs(history_score - recent_sentiment.score) < 1e-9, \
        f"Score mismatch: expected {recent_sentiment.score}, got {history_score}"
    assert abs(history_confidence - recent_sentiment.confidence) < 1e-9, \
        f"Confidence mismatch: expected {recent_sentiment.confidence}, got {history_confidence}"
    
    # 3. Values must be in valid ranges
    assert -1.0 <= history_score <= 1.0, \
        f"Score must be in [-1, 1], got {history_score}"
    assert 0.0 <= history_confidence <= 1.0, \
        f"Confidence must be in [0, 1], got {history_confidence}"
    assert history_timestamp >= 0, \
        f"Timestamp must be non-negative, got {history_timestamp}"


# Feature: realtime-sentiment-analysis, Property 11: Historical data retrieval
@settings(max_examples=100, deadline=None)
@given(sentiments=sentiment_sequence_strategy())
def test_historical_data_no_corruption(sentiments):
    """
    Property 11 (variant): Historical data is not corrupted by retrieval
    
    Accessing the historical data should not modify or corrupt the stored
    values. Multiple accesses should return the same data.
    
    This variant ensures that reading history is a pure operation that doesn't
    have side effects on the stored data.
    
    Validates:
    - Req 7.3: System provides sentiment score history for the current session
    """
    # Create display instance
    display = SentimentDisplay()
    
    # Add all sentiment scores
    for sentiment in sentiments:
        display.update_score(sentiment)
    
    # Get history multiple times
    history_1 = list(display.score_history)
    history_2 = list(display.score_history)
    history_3 = list(display.score_history)
    
    # Property assertions: History is not corrupted by access
    
    # 1. All accesses should return the same data
    assert len(history_1) == len(history_2) == len(history_3), \
        "Multiple accesses should return same number of entries"
    
    # 2. Each entry should be identical across accesses
    for i in range(len(history_1)):
        entry_1 = history_1[i]
        entry_2 = history_2[i]
        entry_3 = history_3[i]
        
        assert entry_1 == entry_2 == entry_3, \
            f"Entry {i} differs across accesses: {entry_1} vs {entry_2} vs {entry_3}"
    
    # 3. Original data should still match input
    for i, sentiment in enumerate(sentiments):
        history_entry = display.score_history[i]
        history_timestamp, history_score, history_confidence = history_entry
        
        assert abs(history_timestamp - sentiment.timestamp) < 1e-9, \
            f"Score {i}: timestamp corrupted after multiple accesses"
        assert abs(history_score - sentiment.score) < 1e-9, \
            f"Score {i}: score corrupted after multiple accesses"
        assert abs(history_confidence - sentiment.confidence) < 1e-9, \
            f"Score {i}: confidence corrupted after multiple accesses"
