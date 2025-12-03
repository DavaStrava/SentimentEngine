"""Unit tests for data models"""

import numpy as np
import pytest
from src.models import (
    AudioFrame,
    VideoFrame,
    AcousticFeatures,
    FaceLandmarks,
    AcousticResult,
    VisualResult,
    LinguisticResult,
    SentimentScore,
    StreamProtocol,
    StreamConnection
)


class TestAudioFrame:
    """Tests for AudioFrame model"""
    
    def test_create_valid_audio_frame(self):
        """Test creating a valid audio frame"""
        samples = np.random.randn(16000)  # 1 second at 16kHz
        frame = AudioFrame(
            samples=samples,
            sample_rate=16000,
            timestamp=1.5,
            duration=1.0
        )
        
        assert frame.sample_rate == 16000
        assert frame.timestamp == 1.5
        assert frame.duration == 1.0
        assert len(frame.samples) == 16000
    
    def test_audio_frame_validation(self):
        """Test audio frame validation"""
        samples = np.random.randn(16000)
        
        # Invalid sample rate
        with pytest.raises(AssertionError):
            AudioFrame(samples=samples, sample_rate=-1, timestamp=0.0, duration=1.0)
        
        # Invalid timestamp
        with pytest.raises(AssertionError):
            AudioFrame(samples=samples, sample_rate=16000, timestamp=-1.0, duration=1.0)
        
        # Invalid duration
        with pytest.raises(AssertionError):
            AudioFrame(samples=samples, sample_rate=16000, timestamp=0.0, duration=-1.0)


class TestVideoFrame:
    """Tests for VideoFrame model"""
    
    def test_create_valid_video_frame(self):
        """Test creating a valid video frame"""
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        frame = VideoFrame(
            image=image,
            timestamp=2.0,
            frame_number=60
        )
        
        assert frame.timestamp == 2.0
        assert frame.frame_number == 60
        assert frame.image.shape == (480, 640, 3)
    
    def test_video_frame_validation(self):
        """Test video frame validation"""
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Invalid timestamp
        with pytest.raises(AssertionError):
            VideoFrame(image=image, timestamp=-1.0, frame_number=0)
        
        # Invalid frame number
        with pytest.raises(AssertionError):
            VideoFrame(image=image, timestamp=0.0, frame_number=-1)
        
        # Invalid image shape (not 3D)
        with pytest.raises(AssertionError):
            VideoFrame(image=np.zeros((480, 640)), timestamp=0.0, frame_number=0)
        
        # Invalid image channels (not RGB)
        with pytest.raises(AssertionError):
            VideoFrame(image=np.zeros((480, 640, 4)), timestamp=0.0, frame_number=0)


class TestAcousticResult:
    """Tests for AcousticResult model"""
    
    def test_create_valid_acoustic_result(self):
        """Test creating a valid acoustic result"""
        features = AcousticFeatures(
            pitch_mean=150.0,
            pitch_std=20.0,
            energy_mean=0.5,
            speaking_rate=3.5,
            spectral_centroid=2000.0,
            zero_crossing_rate=0.1
        )
        
        result = AcousticResult(
            emotion_scores={"happy": 0.7, "sad": 0.2, "neutral": 0.1},
            confidence=0.85,
            features=features,
            timestamp=1.0
        )
        
        assert result.confidence == 0.85
        assert result.emotion_scores["happy"] == 0.7
        assert result.features.pitch_mean == 150.0
    
    def test_acoustic_result_validation(self):
        """Test acoustic result validation"""
        # Invalid confidence
        with pytest.raises(AssertionError):
            AcousticResult(
                emotion_scores={"happy": 0.5},
                confidence=1.5,
                features=None,
                timestamp=0.0
            )
        
        # Invalid emotion score
        with pytest.raises(AssertionError):
            AcousticResult(
                emotion_scores={"happy": 1.5},
                confidence=0.5,
                features=None,
                timestamp=0.0
            )


class TestVisualResult:
    """Tests for VisualResult model"""
    
    def test_create_valid_visual_result(self):
        """Test creating a valid visual result"""
        landmarks = FaceLandmarks(
            points=np.array([[100, 200], [150, 250]]),
            confidence=0.9
        )
        
        result = VisualResult(
            emotion_scores={"happy": 0.8, "neutral": 0.2},
            confidence=0.9,
            face_detected=True,
            face_landmarks=landmarks,
            timestamp=2.0
        )
        
        assert result.face_detected is True
        assert result.confidence == 0.9
        assert result.face_landmarks.confidence == 0.9


class TestLinguisticResult:
    """Tests for LinguisticResult model"""
    
    def test_create_valid_linguistic_result(self):
        """Test creating a valid linguistic result"""
        result = LinguisticResult(
            transcription="This is a test",
            emotion_scores={"positive": 0.6, "neutral": 0.4},
            confidence=0.75,
            transcription_confidence=0.95,
            timestamp=3.0
        )
        
        assert result.transcription == "This is a test"
        assert result.confidence == 0.75
        assert result.transcription_confidence == 0.95


class TestSentimentScore:
    """Tests for SentimentScore model"""
    
    def test_create_valid_sentiment_score(self):
        """Test creating a valid sentiment score"""
        score = SentimentScore(
            score=0.5,
            confidence=0.8,
            modality_contributions={"acoustic": 0.3, "visual": 0.4, "linguistic": 0.3},
            emotion_breakdown={"happy": 0.6, "neutral": 0.4},
            timestamp=4.0
        )
        
        assert score.score == 0.5
        assert score.confidence == 0.8
        assert -1.0 <= score.score <= 1.0
    
    def test_sentiment_score_validation(self):
        """Test sentiment score validation"""
        # Invalid score (out of range)
        with pytest.raises(AssertionError):
            SentimentScore(
                score=1.5,
                confidence=0.8,
                modality_contributions={},
                emotion_breakdown={},
                timestamp=0.0
            )
        
        # Invalid confidence
        with pytest.raises(AssertionError):
            SentimentScore(
                score=0.5,
                confidence=1.5,
                modality_contributions={},
                emotion_breakdown={},
                timestamp=0.0
            )


class TestStreamConnection:
    """Tests for StreamConnection model"""
    
    def test_create_stream_connection(self):
        """Test creating a stream connection"""
        conn = StreamConnection(
            url="rtmp://example.com/stream",
            protocol=StreamProtocol.RTMP,
            is_active=True,
            audio_codec="aac",
            video_codec="h264"
        )
        
        assert conn.url == "rtmp://example.com/stream"
        assert conn.protocol == StreamProtocol.RTMP
        assert conn.is_active is True
        assert conn.audio_codec == "aac"
        assert conn.video_codec == "h264"
    
    def test_stream_protocol_enum(self):
        """Test StreamProtocol enum values"""
        assert StreamProtocol.RTMP.value == "rtmp"
        assert StreamProtocol.HLS.value == "hls"
        assert StreamProtocol.WEBRTC.value == "webrtc"
        assert StreamProtocol.FILE.value == "file"
