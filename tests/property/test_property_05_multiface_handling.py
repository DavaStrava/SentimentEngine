"""Property-based tests for multi-face video handling

Feature: realtime-sentiment-analysis, Property 5: Multi-face video handling
Validates: Requirements 4.4
"""

import pytest
import numpy as np
from hypothesis import given, strategies as st, settings, assume
from unittest.mock import patch, MagicMock

from src.analysis.visual import VisualAnalyzer
from src.models.frames import VideoFrame, AudioFrame
from src.models.results import VisualResult
from src.models.features import FaceLandmarks, FaceRegion


# Shared analyzer instance to avoid MediaPipe reinitialization issues
_shared_analyzer = None


@pytest.fixture(scope="module")
def shared_analyzer():
    """Create a shared analyzer instance for all tests to avoid MediaPipe crashes."""
    global _shared_analyzer
    if _shared_analyzer is None:
        _shared_analyzer = VisualAnalyzer()
        _shared_analyzer._load_models()
    return _shared_analyzer


# Custom strategies for generating test data
@st.composite
def multi_face_video_frame_strategy(draw):
    """Generate video frames that simulate multiple faces.
    
    This strategy creates synthetic video frames with metadata suggesting
    multiple faces are present. The actual image content is random, but
    we'll mock the face detection to return multiple faces.
    
    Returns:
        VideoFrame with random image content
    """
    # Common video resolutions (width, height)
    resolutions = [
        (640, 480),   # VGA
        (1280, 720),  # HD
        (1920, 1080)  # Full HD
    ]
    
    width, height = draw(st.sampled_from(resolutions))
    timestamp = draw(st.floats(min_value=0.0, max_value=1000.0))
    frame_number = draw(st.integers(min_value=0, max_value=10000))
    
    # Generate random RGB image
    seed = draw(st.integers(min_value=0, max_value=2**31-1))
    rng = np.random.RandomState(seed)
    image = rng.randint(0, 256, (height, width, 3), dtype=np.uint8)
    
    return VideoFrame(
        image=image,
        timestamp=timestamp,
        frame_number=frame_number
    )


@st.composite
def audio_frame_strategy(draw):
    """Generate random audio frames for audio-visual synchronization.
    
    Returns:
        AudioFrame with random audio samples
    """
    sample_rate = draw(st.sampled_from([8000, 16000, 44100, 48000]))
    duration = draw(st.floats(min_value=0.01, max_value=1.0))
    timestamp = draw(st.floats(min_value=0.0, max_value=1000.0))
    
    # Generate random audio samples
    num_samples = int(sample_rate * duration)
    seed = draw(st.integers(min_value=0, max_value=2**31-1))
    rng = np.random.RandomState(seed)
    samples = rng.uniform(-1.0, 1.0, num_samples).astype(np.float32)
    
    return AudioFrame(
        samples=samples,
        sample_rate=sample_rate,
        timestamp=timestamp,
        duration=duration
    )


@st.composite
def face_landmarks_strategy(draw):
    """Generate random facial landmarks for testing.
    
    Creates a FaceLandmarks object with 468 points (MediaPipe format)
    and a confidence score.
    
    Returns:
        FaceLandmarks with random but valid landmark points
    """
    # MediaPipe uses 468 landmark points
    num_points = 468
    
    # Generate random landmark coordinates
    # Assume face is roughly in center of 640x480 frame
    seed = draw(st.integers(min_value=0, max_value=2**31-1))
    rng = np.random.RandomState(seed)
    
    # Face region roughly 200x200 pixels centered at (320, 240)
    center_x, center_y = 320, 240
    face_size = 200
    
    points = []
    for _ in range(num_points):
        x = rng.uniform(center_x - face_size/2, center_x + face_size/2)
        y = rng.uniform(center_y - face_size/2, center_y + face_size/2)
        points.append([x, y])
    
    points_array = np.array(points, dtype=np.float32)
    confidence = draw(st.floats(min_value=0.5, max_value=1.0))
    
    return FaceLandmarks(
        points=points_array,
        confidence=confidence
    )


def create_mock_face_regions(num_faces: int, landmarks_list: list) -> list:
    """Create mock FaceRegion objects for testing.
    
    Args:
        num_faces: Number of faces to create
        landmarks_list: List of FaceLandmarks objects
        
    Returns:
        List of FaceRegion objects
    """
    face_regions = []
    for i in range(num_faces):
        # Create bounding box
        bbox = (100 + i * 150, 100, 120, 150)  # (x, y, width, height)
        
        face_region = FaceRegion(
            bounding_box=bbox,
            landmarks=landmarks_list[i] if i < len(landmarks_list) else landmarks_list[0],
            face_id=i
        )
        face_regions.append(face_region)
    
    return face_regions


# Feature: realtime-sentiment-analysis, Property 5: Multi-face video handling
@settings(max_examples=100, deadline=None)
@given(
    video_frame=multi_face_video_frame_strategy(),
    audio_frame=audio_frame_strategy(),
    num_faces=st.integers(min_value=2, max_value=5),
    landmarks_list=st.lists(face_landmarks_strategy(), min_size=2, max_size=5)
)
@pytest.mark.asyncio
async def test_multiface_handling_produces_valid_result(shared_analyzer, video_frame, audio_frame, num_faces, landmarks_list):
    """
    Property 5: Multi-face video handling
    
    For any video frame containing multiple faces, the Visual Analysis Module 
    should identify and analyze the primary speaker without failing or producing 
    invalid results.
    
    This property verifies that:
    1. The analyzer handles multiple faces without crashing
    2. A valid VisualResult is returned (not None)
    3. The result contains all required components (emotion_scores, confidence, etc.)
    4. The result has valid values (confidence in [0,1], emotion scores in [0,1])
    5. The analyzer successfully identifies a primary speaker (or gracefully handles no clear speaker)
    6. The result is cached for fusion engine access
    
    The test mocks face detection to return multiple faces and verifies that
    the audio-visual synchronization component is used to identify the primary
    speaker, and that the final result is valid regardless of the outcome.
    
    Validates:
    - Req 4.4: System analyzes primary speaker based on audio-visual synchronization
    """
    # Ensure we have enough landmarks for the number of faces
    assume(len(landmarks_list) >= num_faces)
    
    # Use shared analyzer
    analyzer = shared_analyzer
    
    # Set the latest audio frame for AV sync
    analyzer.latest_audio_frame = audio_frame
    
    # Mock face detection to return multiple faces
    mock_bboxes = [(100 + i * 150, 100, 120, 150) for i in range(num_faces)]
    
    with patch.object(analyzer, '_detect_faces', return_value=mock_bboxes):
        with patch.object(analyzer, '_extract_landmarks_all', return_value=landmarks_list[:num_faces]):
            # Mock expression classification to return valid emotions
            mock_emotions = {
                "happy": 0.3,
                "neutral": 0.4,
                "sad": 0.2,
                "angry": 0.1
            }
            with patch.object(analyzer, '_classify_expression', return_value=mock_emotions):
                # Analyze frame with multiple faces
                result = await analyzer.analyze_frame(video_frame)
    
    # Property assertions: Multi-face handling must produce valid results
    
    # 1. Result must not be None (no crashes)
    assert result is not None, \
        f"VisualResult should not be None for video frame with {num_faces} faces"
    
    # 2. Result must be a VisualResult instance
    assert isinstance(result, VisualResult), \
        "Result must be VisualResult instance"
    
    # 3. Result must contain emotion scores
    assert hasattr(result, 'emotion_scores'), \
        "Result must have emotion_scores attribute"
    assert isinstance(result.emotion_scores, dict), \
        "emotion_scores must be a dictionary"
    assert len(result.emotion_scores) > 0, \
        "emotion_scores must contain at least one emotion"
    
    # 4. All emotion scores must be in valid range [0, 1]
    for emotion, score in result.emotion_scores.items():
        assert isinstance(emotion, str), \
            f"Emotion key must be string, got {type(emotion)}"
        assert isinstance(score, (int, float)), \
            f"Emotion score must be numeric, got {type(score)}"
        assert 0.0 <= score <= 1.0, \
            f"Emotion score for {emotion} must be in [0, 1], got {score}"
    
    # 5. Result must contain confidence in valid range
    assert hasattr(result, 'confidence'), \
        "Result must have confidence attribute"
    assert isinstance(result.confidence, (int, float)), \
        "Confidence must be numeric"
    assert 0.0 <= result.confidence <= 1.0, \
        f"Confidence must be in [0, 1], got {result.confidence}"
    
    # 6. Result must indicate face was detected (since we mocked multiple faces)
    assert hasattr(result, 'face_detected'), \
        "Result must have face_detected attribute"
    assert result.face_detected is True, \
        "face_detected should be True when multiple faces are present"
    
    # 7. Result must contain face_landmarks (should be one of the detected faces)
    assert hasattr(result, 'face_landmarks'), \
        "Result must have face_landmarks attribute"
    # Landmarks may be None if extraction failed, but should be present for mocked data
    assert result.face_landmarks is not None, \
        "face_landmarks should be present when faces are detected"
    assert isinstance(result.face_landmarks, FaceLandmarks), \
        "face_landmarks must be FaceLandmarks instance"
    
    # 8. Result must contain valid timestamp
    assert hasattr(result, 'timestamp'), \
        "Result must have timestamp attribute"
    assert isinstance(result.timestamp, (int, float)), \
        "Timestamp must be numeric"
    assert result.timestamp >= 0.0, \
        f"Timestamp must be non-negative, got {result.timestamp}"
    
    # 9. Result should be cached
    cached_result = analyzer.get_latest_result()
    assert cached_result is not None, \
        "Result should be cached after analysis"
    assert cached_result == result, \
        "Cached result should match returned result"


# Feature: realtime-sentiment-analysis, Property 5: Multi-face video handling
@settings(max_examples=100, deadline=None)
@given(
    video_frame=multi_face_video_frame_strategy(),
    audio_frame=audio_frame_strategy(),
    num_faces=st.integers(min_value=2, max_value=5),
    landmarks_list=st.lists(face_landmarks_strategy(), min_size=2, max_size=5)
)
@pytest.mark.asyncio
async def test_multiface_handling_uses_av_sync(shared_analyzer, video_frame, audio_frame, num_faces, landmarks_list):
    """
    Property 5 (variant): Multi-face handling uses audio-visual synchronization
    
    For any video frame containing multiple faces, the Visual Analysis Module
    should use the AudioVisualSync component to identify the primary speaker
    before performing emotion classification.
    
    This property verifies that:
    1. When multiple faces are detected, the AV sync component is invoked
    2. The primary speaker identification is attempted
    3. The analyzer handles both cases: clear primary speaker and no clear speaker
    4. The final result is based on the identified primary speaker (or fallback)
    
    Validates:
    - Req 4.4: System analyzes primary speaker based on audio-visual synchronization
    """
    # Ensure we have enough landmarks for the number of faces
    assume(len(landmarks_list) >= num_faces)
    
    # Use shared analyzer
    analyzer = shared_analyzer
    
    # Set the latest audio frame for AV sync
    analyzer.latest_audio_frame = audio_frame
    
    # Mock face detection to return multiple faces
    mock_bboxes = [(100 + i * 150, 100, 120, 150) for i in range(num_faces)]
    
    # Track whether AV sync was called
    av_sync_called = False
    original_identify = analyzer.av_sync.identify_primary_speaker
    
    def mock_identify_primary_speaker(faces, audio):
        nonlocal av_sync_called
        av_sync_called = True
        # Call the original method
        return original_identify(faces, audio)
    
    with patch.object(analyzer, '_detect_faces', return_value=mock_bboxes):
        with patch.object(analyzer, '_extract_landmarks_all', return_value=landmarks_list[:num_faces]):
            with patch.object(analyzer.av_sync, 'identify_primary_speaker', side_effect=mock_identify_primary_speaker):
                # Mock expression classification
                mock_emotions = {
                    "happy": 0.3,
                    "neutral": 0.4,
                    "sad": 0.2,
                    "angry": 0.1
                }
                with patch.object(analyzer, '_classify_expression', return_value=mock_emotions):
                    # Analyze frame with multiple faces
                    result = await analyzer.analyze_frame(video_frame)
    
    # Property assertions
    
    # 1. AV sync should have been called for multiple faces
    assert av_sync_called, \
        f"AudioVisualSync.identify_primary_speaker should be called when {num_faces} faces are detected"
    
    # 2. Result must be valid
    assert result is not None, \
        "Result should not be None even if AV sync doesn't identify clear speaker"
    assert isinstance(result, VisualResult)
    assert result.face_detected is True
    assert 0.0 <= result.confidence <= 1.0


# Feature: realtime-sentiment-analysis, Property 5: Multi-face video handling
@settings(max_examples=50, deadline=None)
@given(
    video_frame=multi_face_video_frame_strategy(),
    num_faces=st.integers(min_value=2, max_value=5),
    landmarks_list=st.lists(face_landmarks_strategy(), min_size=2, max_size=5)
)
@pytest.mark.asyncio
async def test_multiface_handling_without_audio(shared_analyzer, video_frame, num_faces, landmarks_list):
    """
    Property 5 (variant): Multi-face handling without audio frame
    
    For any video frame containing multiple faces but no audio frame available,
    the Visual Analysis Module should gracefully fall back to analyzing the
    first detected face without crashing.
    
    This property verifies that:
    1. The analyzer handles missing audio frame gracefully
    2. Falls back to first face when AV sync cannot be performed
    3. Still produces a valid result
    4. Does not crash or return None
    
    Validates:
    - Req 4.4: System analyzes primary speaker based on audio-visual synchronization
    - Design: Graceful degradation when modalities are missing
    """
    # Ensure we have enough landmarks for the number of faces
    assume(len(landmarks_list) >= num_faces)
    
    # Use shared analyzer
    analyzer = shared_analyzer
    
    # No audio frame available
    analyzer.latest_audio_frame = None
    
    # Mock face detection to return multiple faces
    mock_bboxes = [(100 + i * 150, 100, 120, 150) for i in range(num_faces)]
    
    with patch.object(analyzer, '_detect_faces', return_value=mock_bboxes):
        with patch.object(analyzer, '_extract_landmarks_all', return_value=landmarks_list[:num_faces]):
            # Mock expression classification
            mock_emotions = {
                "happy": 0.3,
                "neutral": 0.4,
                "sad": 0.2,
                "angry": 0.1
            }
            with patch.object(analyzer, '_classify_expression', return_value=mock_emotions):
                # Analyze frame with multiple faces but no audio
                result = await analyzer.analyze_frame(video_frame)
    
    # Property assertions: Should gracefully handle missing audio
    
    # 1. Result must not be None (graceful fallback)
    assert result is not None, \
        "Result should not be None even without audio frame for AV sync"
    
    # 2. Result must be valid
    assert isinstance(result, VisualResult)
    assert result.face_detected is True
    assert isinstance(result.emotion_scores, dict)
    assert len(result.emotion_scores) > 0
    assert 0.0 <= result.confidence <= 1.0
    
    # 3. Should have used first face as fallback
    assert result.face_landmarks is not None, \
        "Should have landmarks from fallback face"
    assert isinstance(result.face_landmarks, FaceLandmarks)


# Feature: realtime-sentiment-analysis, Property 5: Multi-face video handling
@settings(max_examples=50, deadline=None)
@given(
    video_frame=multi_face_video_frame_strategy(),
    audio_frame=audio_frame_strategy(),
    num_faces=st.integers(min_value=2, max_value=5),
    landmarks_list=st.lists(face_landmarks_strategy(), min_size=2, max_size=5),
    primary_speaker_id=st.integers(min_value=0, max_value=4)
)
@pytest.mark.asyncio
async def test_multiface_selects_correct_primary_speaker(
    shared_analyzer, video_frame, audio_frame, num_faces, landmarks_list, primary_speaker_id
):
    """
    Property 5 (variant): Multi-face handling selects identified primary speaker
    
    For any video frame containing multiple faces, when the AV sync component
    identifies a specific face as the primary speaker, the Visual Analysis Module
    should use that face's landmarks for emotion classification.
    
    This property verifies that:
    1. When AV sync identifies a primary speaker, that face is used
    2. The selected face's landmarks are used for emotion classification
    3. The result reflects the primary speaker's emotion, not other faces
    
    Validates:
    - Req 4.4: System analyzes primary speaker based on audio-visual synchronization
    """
    # Ensure we have enough landmarks and valid primary speaker ID
    assume(len(landmarks_list) >= num_faces)
    assume(primary_speaker_id < num_faces)
    
    # Use shared analyzer
    analyzer = shared_analyzer
    
    # Set the latest audio frame for AV sync
    analyzer.latest_audio_frame = audio_frame
    
    # Mock face detection to return multiple faces
    mock_bboxes = [(100 + i * 150, 100, 120, 150) for i in range(num_faces)]
    
    # Track which face was used for classification
    classified_landmarks = None
    
    def mock_classify_expression(image, landmarks):
        nonlocal classified_landmarks
        classified_landmarks = landmarks
        return {
            "happy": 0.3,
            "neutral": 0.4,
            "sad": 0.2,
            "angry": 0.1
        }
    
    with patch.object(analyzer, '_detect_faces', return_value=mock_bboxes):
        with patch.object(analyzer, '_extract_landmarks_all', return_value=landmarks_list[:num_faces]):
            # Mock AV sync to return specific primary speaker
            with patch.object(analyzer.av_sync, 'identify_primary_speaker', return_value=primary_speaker_id):
                with patch.object(analyzer, '_classify_expression', side_effect=mock_classify_expression):
                    # Analyze frame
                    result = await analyzer.analyze_frame(video_frame)
    
    # Property assertions
    
    # 1. Result must be valid
    assert result is not None
    assert isinstance(result, VisualResult)
    
    # 2. The landmarks used for classification should be from the primary speaker
    assert classified_landmarks is not None, \
        "Expression classification should have been called with landmarks"
    
    # 3. The classified landmarks should match the primary speaker's landmarks
    # We can verify this by checking if it's one of the landmarks in our list
    # Use numpy array comparison since 'in' doesn't work with numpy arrays
    found_match = False
    for lm in landmarks_list[:num_faces]:
        if np.array_equal(classified_landmarks.points, lm.points):
            found_match = True
            break
    assert found_match, \
        "Classified landmarks should be from one of the detected faces"
    
    # 4. The result's face_landmarks should be from the primary speaker
    assert result.face_landmarks is not None
    # The landmarks should be the same object (or equal) to the primary speaker's landmarks
    expected_landmarks = landmarks_list[primary_speaker_id]
    assert np.array_equal(result.face_landmarks.points, expected_landmarks.points), \
        f"Result should contain landmarks from primary speaker (face_id={primary_speaker_id})"
