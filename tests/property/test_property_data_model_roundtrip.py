"""Property-based tests for data model serialization round-trip consistency

Feature: realtime-sentiment-analysis, Property: Data model round-trip consistency
Validates: Requirements 1.2, 1.3, 1.4
"""

import json
import pickle
import numpy as np
from hypothesis import given, strategies as st, settings
from src.models import (
    AudioFrame,
    VideoFrame,
    AcousticFeatures,
    AcousticResult,
    VisualResult,
    LinguisticResult,
    SentimentScore
)


# Custom strategies for generating test data
@st.composite
def audio_frame_strategy(draw):
    """Generate random AudioFrame instances"""
    sample_count = draw(st.integers(min_value=100, max_value=48000))
    samples = np.random.randn(sample_count).astype(np.float32)
    
    return AudioFrame(
        samples=samples,
        sample_rate=draw(st.integers(min_value=8000, max_value=48000)),
        timestamp=draw(st.floats(min_value=0.0, max_value=1000.0)),
        duration=draw(st.floats(min_value=0.01, max_value=10.0))
    )


@st.composite
def video_frame_strategy(draw):
    """Generate random VideoFrame instances"""
    height = draw(st.integers(min_value=100, max_value=1080))
    width = draw(st.integers(min_value=100, max_value=1920))
    image = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    
    return VideoFrame(
        image=image,
        timestamp=draw(st.floats(min_value=0.0, max_value=1000.0)),
        frame_number=draw(st.integers(min_value=0, max_value=100000))
    )


@st.composite
def emotion_scores_strategy(draw):
    """Generate random emotion score dictionaries"""
    emotions = ["happy", "sad", "angry", "neutral", "surprised", "fearful"]
    num_emotions = draw(st.integers(min_value=1, max_value=len(emotions)))
    selected_emotions = draw(st.lists(
        st.sampled_from(emotions),
        min_size=num_emotions,
        max_size=num_emotions,
        unique=True
    ))
    
    scores = {}
    for emotion in selected_emotions:
        scores[emotion] = draw(st.floats(min_value=0.0, max_value=1.0))
    
    return scores


@st.composite
def acoustic_result_strategy(draw):
    """Generate random AcousticResult instances"""
    features = AcousticFeatures(
        pitch_mean=draw(st.floats(min_value=50.0, max_value=500.0)),
        pitch_std=draw(st.floats(min_value=0.0, max_value=100.0)),
        energy_mean=draw(st.floats(min_value=0.0, max_value=1.0)),
        speaking_rate=draw(st.floats(min_value=0.5, max_value=10.0)),
        spectral_centroid=draw(st.floats(min_value=100.0, max_value=8000.0)),
        zero_crossing_rate=draw(st.floats(min_value=0.0, max_value=1.0))
    )
    
    return AcousticResult(
        emotion_scores=draw(emotion_scores_strategy()),
        confidence=draw(st.floats(min_value=0.0, max_value=1.0)),
        features=features,
        timestamp=draw(st.floats(min_value=0.0, max_value=1000.0))
    )


@st.composite
def visual_result_strategy(draw):
    """Generate random VisualResult instances"""
    return VisualResult(
        emotion_scores=draw(emotion_scores_strategy()),
        confidence=draw(st.floats(min_value=0.0, max_value=1.0)),
        face_detected=draw(st.booleans()),
        face_landmarks=None,  # Simplified for now
        timestamp=draw(st.floats(min_value=0.0, max_value=1000.0))
    )


@st.composite
def linguistic_result_strategy(draw):
    """Generate random LinguisticResult instances"""
    return LinguisticResult(
        transcription=draw(st.text(min_size=0, max_size=500)),
        emotion_scores=draw(emotion_scores_strategy()),
        confidence=draw(st.floats(min_value=0.0, max_value=1.0)),
        transcription_confidence=draw(st.floats(min_value=0.0, max_value=1.0)),
        timestamp=draw(st.floats(min_value=0.0, max_value=1000.0))
    )


@st.composite
def sentiment_score_strategy(draw):
    """Generate random SentimentScore instances"""
    return SentimentScore(
        score=draw(st.floats(min_value=-1.0, max_value=1.0)),
        confidence=draw(st.floats(min_value=0.0, max_value=1.0)),
        modality_contributions={
            "acoustic": draw(st.floats(min_value=0.0, max_value=1.0)),
            "visual": draw(st.floats(min_value=0.0, max_value=1.0)),
            "linguistic": draw(st.floats(min_value=0.0, max_value=1.0))
        },
        emotion_breakdown=draw(emotion_scores_strategy()),
        timestamp=draw(st.floats(min_value=0.0, max_value=1000.0))
    )


# Helper functions for serialization
def serialize_audio_frame(frame: AudioFrame) -> dict:
    """Serialize AudioFrame to dictionary"""
    return {
        "samples": frame.samples.tolist(),
        "sample_rate": frame.sample_rate,
        "timestamp": frame.timestamp,
        "duration": frame.duration
    }


def deserialize_audio_frame(data: dict) -> AudioFrame:
    """Deserialize dictionary to AudioFrame"""
    return AudioFrame(
        samples=np.array(data["samples"], dtype=np.float32),
        sample_rate=data["sample_rate"],
        timestamp=data["timestamp"],
        duration=data["duration"]
    )


def serialize_video_frame(frame: VideoFrame) -> dict:
    """Serialize VideoFrame to dictionary"""
    return {
        "image": frame.image.tolist(),
        "timestamp": frame.timestamp,
        "frame_number": frame.frame_number
    }


def deserialize_video_frame(data: dict) -> VideoFrame:
    """Deserialize dictionary to VideoFrame"""
    return VideoFrame(
        image=np.array(data["image"], dtype=np.uint8),
        timestamp=data["timestamp"],
        frame_number=data["frame_number"]
    )


def serialize_acoustic_result(result: AcousticResult) -> dict:
    """Serialize AcousticResult to dictionary"""
    return {
        "emotion_scores": result.emotion_scores,
        "confidence": result.confidence,
        "features": {
            "pitch_mean": result.features.pitch_mean,
            "pitch_std": result.features.pitch_std,
            "energy_mean": result.features.energy_mean,
            "speaking_rate": result.features.speaking_rate,
            "spectral_centroid": result.features.spectral_centroid,
            "zero_crossing_rate": result.features.zero_crossing_rate
        } if result.features else None,
        "timestamp": result.timestamp
    }


def deserialize_acoustic_result(data: dict) -> AcousticResult:
    """Deserialize dictionary to AcousticResult"""
    features = None
    if data["features"]:
        features = AcousticFeatures(**data["features"])
    
    return AcousticResult(
        emotion_scores=data["emotion_scores"],
        confidence=data["confidence"],
        features=features,
        timestamp=data["timestamp"]
    )


# Property-based tests
@given(audio_frame_strategy())
def test_audio_frame_pickle_roundtrip(frame):
    """
    Property: For any AudioFrame, pickling then unpickling should produce equivalent data
    
    This tests that AudioFrame instances can be serialized and deserialized
    without data loss using pickle.
    """
    # Serialize
    pickled = pickle.dumps(frame)
    
    # Deserialize
    restored = pickle.loads(pickled)
    
    # Verify equivalence
    assert restored.sample_rate == frame.sample_rate
    assert restored.timestamp == frame.timestamp
    assert restored.duration == frame.duration
    assert np.allclose(restored.samples, frame.samples, rtol=1e-5)


@given(audio_frame_strategy())
@settings(deadline=None)  # JSON serialization of large audio arrays can be slow, disable deadline
def test_audio_frame_json_roundtrip(frame):
    """
    Property: For any AudioFrame, JSON serialization then deserialization should preserve data
    
    This tests that AudioFrame instances can be converted to/from JSON-compatible
    dictionaries without data loss.
    """
    # Serialize to dict then JSON
    serialized = serialize_audio_frame(frame)
    json_str = json.dumps(serialized)
    
    # Deserialize from JSON then dict
    restored_dict = json.loads(json_str)
    restored = deserialize_audio_frame(restored_dict)
    
    # Verify equivalence
    assert restored.sample_rate == frame.sample_rate
    assert abs(restored.timestamp - frame.timestamp) < 1e-6
    assert abs(restored.duration - frame.duration) < 1e-6
    assert np.allclose(restored.samples, frame.samples, rtol=1e-5)


@given(video_frame_strategy())
def test_video_frame_pickle_roundtrip(frame):
    """
    Property: For any VideoFrame, pickling then unpickling should produce equivalent data
    """
    pickled = pickle.dumps(frame)
    restored = pickle.loads(pickled)
    
    assert restored.timestamp == frame.timestamp
    assert restored.frame_number == frame.frame_number
    assert np.array_equal(restored.image, frame.image)


@given(video_frame_strategy())
@settings(deadline=None)  # JSON serialization of large images can be slow, disable deadline
def test_video_frame_json_roundtrip(frame):
    """
    Property: For any VideoFrame, JSON serialization then deserialization should preserve data
    """
    serialized = serialize_video_frame(frame)
    json_str = json.dumps(serialized)
    restored_dict = json.loads(json_str)
    restored = deserialize_video_frame(restored_dict)
    
    assert abs(restored.timestamp - frame.timestamp) < 1e-6
    assert restored.frame_number == frame.frame_number
    assert np.array_equal(restored.image, frame.image)


@given(acoustic_result_strategy())
def test_acoustic_result_json_roundtrip(result):
    """
    Property: For any AcousticResult, JSON serialization then deserialization should preserve data
    
    Validates: Requirements 1.2 (acoustic analysis results must be serializable)
    """
    serialized = serialize_acoustic_result(result)
    json_str = json.dumps(serialized)
    restored_dict = json.loads(json_str)
    restored = deserialize_acoustic_result(restored_dict)
    
    assert restored.emotion_scores == result.emotion_scores
    assert abs(restored.confidence - result.confidence) < 1e-6
    assert abs(restored.timestamp - result.timestamp) < 1e-6
    
    if result.features:
        assert abs(restored.features.pitch_mean - result.features.pitch_mean) < 1e-6
        assert abs(restored.features.energy_mean - result.features.energy_mean) < 1e-6


@given(visual_result_strategy())
def test_visual_result_pickle_roundtrip(result):
    """
    Property: For any VisualResult, pickling then unpickling should produce equivalent data
    
    Validates: Requirements 1.3 (visual analysis results must be serializable)
    """
    pickled = pickle.dumps(result)
    restored = pickle.loads(pickled)
    
    assert restored.emotion_scores == result.emotion_scores
    assert restored.confidence == result.confidence
    assert restored.face_detected == result.face_detected
    assert restored.timestamp == result.timestamp


@given(linguistic_result_strategy())
def test_linguistic_result_pickle_roundtrip(result):
    """
    Property: For any LinguisticResult, pickling then unpickling should produce equivalent data
    
    Validates: Requirements 1.4 (linguistic analysis results must be serializable)
    """
    pickled = pickle.dumps(result)
    restored = pickle.loads(pickled)
    
    assert restored.transcription == result.transcription
    assert restored.emotion_scores == result.emotion_scores
    assert restored.confidence == result.confidence
    assert restored.transcription_confidence == result.transcription_confidence
    assert restored.timestamp == result.timestamp


@given(sentiment_score_strategy())
def test_sentiment_score_pickle_roundtrip(score):
    """
    Property: For any SentimentScore, pickling then unpickling should produce equivalent data
    
    This ensures fusion results can be serialized for storage or transmission.
    """
    pickled = pickle.dumps(score)
    restored = pickle.loads(pickled)
    
    assert abs(restored.score - score.score) < 1e-6
    assert abs(restored.confidence - score.confidence) < 1e-6
    assert restored.modality_contributions == score.modality_contributions
    assert restored.emotion_breakdown == score.emotion_breakdown
    assert abs(restored.timestamp - score.timestamp) < 1e-6
