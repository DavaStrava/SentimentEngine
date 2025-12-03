"""Unit tests for Visual Analysis Module

Tests the VisualAnalyzer class to ensure proper face detection, landmark extraction,
and emotion classification functionality.
"""

import pytest
import numpy as np
import time
from src.analysis.visual import VisualAnalyzer
from src.models.frames import VideoFrame
from src.models.results import VisualResult


@pytest.fixture
def visual_analyzer():
    """Create a VisualAnalyzer instance for testing"""
    analyzer = VisualAnalyzer()
    analyzer._load_models()
    return analyzer


@pytest.fixture
def sample_video_frame():
    """Create a sample video frame with a simple synthetic image"""
    # Create a simple RGB image (480x640x3)
    image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    return VideoFrame(
        image=image,
        timestamp=1.0,
        frame_number=1
    )


@pytest.fixture
def black_video_frame():
    """Create a black video frame (no face)"""
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    
    return VideoFrame(
        image=image,
        timestamp=2.0,
        frame_number=2
    )


@pytest.mark.asyncio
async def test_analyze_frame_returns_result(visual_analyzer, sample_video_frame):
    """Test that analyze_frame returns a VisualResult"""
    result = await visual_analyzer.analyze_frame(sample_video_frame)
    
    assert result is not None
    assert isinstance(result, VisualResult)
    assert isinstance(result.emotion_scores, dict)
    assert isinstance(result.confidence, float)
    assert isinstance(result.face_detected, bool)
    assert isinstance(result.timestamp, float)


@pytest.mark.asyncio
async def test_analyze_frame_no_face_detected(visual_analyzer, black_video_frame):
    """Test that analyze_frame handles frames with no face"""
    result = await visual_analyzer.analyze_frame(black_video_frame)
    
    assert result is not None
    assert result.face_detected is False
    assert result.confidence <= 0.2  # Low confidence when no face
    assert result.face_landmarks is None
    assert "neutral" in result.emotion_scores


@pytest.mark.asyncio
async def test_analyze_frame_emotion_scores_valid(visual_analyzer, sample_video_frame):
    """Test that emotion scores are in valid range [0, 1]"""
    result = await visual_analyzer.analyze_frame(sample_video_frame)
    
    assert result is not None
    for emotion, score in result.emotion_scores.items():
        assert 0.0 <= score <= 1.0, f"Emotion score for {emotion} out of range: {score}"


@pytest.mark.asyncio
async def test_analyze_frame_confidence_valid(visual_analyzer, sample_video_frame):
    """Test that confidence is in valid range [0, 1]"""
    result = await visual_analyzer.analyze_frame(sample_video_frame)
    
    assert result is not None
    assert 0.0 <= result.confidence <= 1.0


@pytest.mark.asyncio
async def test_get_latest_result_caching(visual_analyzer, sample_video_frame):
    """Test that results are properly cached"""
    # Initially no result
    assert visual_analyzer.get_latest_result() is None
    
    # Analyze a frame
    result = await visual_analyzer.analyze_frame(sample_video_frame)
    
    # Should be cached
    cached_result = visual_analyzer.get_latest_result()
    assert cached_result is not None
    assert cached_result == result


def test_assess_lighting_returns_valid_score(visual_analyzer):
    """Test that lighting assessment returns valid score"""
    # Create test images with different lighting conditions
    
    # Bright image
    bright_image = np.ones((480, 640, 3), dtype=np.uint8) * 200
    bright_score = visual_analyzer._assess_lighting(bright_image)
    assert 0.0 <= bright_score <= 1.0
    
    # Dark image
    dark_image = np.ones((480, 640, 3), dtype=np.uint8) * 50
    dark_score = visual_analyzer._assess_lighting(dark_image)
    assert 0.0 <= dark_score <= 1.0
    
    # Normal image
    normal_image = np.ones((480, 640, 3), dtype=np.uint8) * 128
    normal_score = visual_analyzer._assess_lighting(normal_image)
    assert 0.0 <= normal_score <= 1.0


def test_frame_skip_configuration(visual_analyzer):
    """Test that frame skip is properly configured"""
    assert visual_analyzer.frame_skip >= 1
    assert isinstance(visual_analyzer.frame_skip, int)


@pytest.mark.asyncio
async def test_analyze_frame_timestamp_set(visual_analyzer, sample_video_frame):
    """Test that result timestamp is set correctly"""
    before = time.time()
    result = await visual_analyzer.analyze_frame(sample_video_frame)
    after = time.time()
    
    assert result is not None
    assert before <= result.timestamp <= after


def test_video_frame_validation_rejects_invalid_image():
    """Test that VideoFrame validation rejects invalid image shapes"""
    analyzer = VisualAnalyzer()
    
    # VideoFrame should reject 2D images (missing channel dimension)
    with pytest.raises(AssertionError, match="Image must be 3D array"):
        invalid_frame = VideoFrame(
            image=np.zeros((10, 10), dtype=np.uint8),  # 2D instead of 3D
            timestamp=1.0,
            frame_number=1
        )
    
    # VideoFrame should reject images with wrong number of channels
    with pytest.raises(AssertionError, match="Image must have 3 channels"):
        invalid_frame = VideoFrame(
            image=np.zeros((10, 10, 4), dtype=np.uint8),  # 4 channels instead of 3
            timestamp=1.0,
            frame_number=1
        )
