"""
Integration tests for end-to-end sentiment analysis pipeline.

Tests Requirements 9.1 and 9.4:
- End-to-end latency validation (target: 1s, max: 3s)
- Pipeline functionality with real video files
- All modalities contributing to final score
- Error recovery scenarios
"""

import pytest
import asyncio
import time
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.frames import AudioFrame, VideoFrame
from src.models.results import AcousticResult, VisualResult, LinguisticResult, SentimentScore
from src.analysis.acoustic import AcousticAnalyzer
from src.analysis.visual import VisualAnalyzer
from src.analysis.linguistic import LinguisticAnalyzer
from src.fusion.fusion_engine import FusionEngine
from src.input.stream_manager import StreamInputManager


class MockAnalyzer:
    """Mock analyzer for testing."""
    def __init__(self, result):
        self.result = result
    def get_latest_result(self):
        return self.result


@pytest.mark.asyncio
async def test_latency_requirements():
    """
    Test that end-to-end latency meets requirements.
    
    Validates Requirement 9.1:
    - Target latency: 1 second
    - Maximum latency: 3 seconds
    """
    # Create test frames
    audio_frame = AudioFrame(
        samples=np.random.randn(16000).astype(np.float32),
        sample_rate=16000,
        timestamp=0.0,
        duration=1.0
    )
    
    video_frame = VideoFrame(
        image=np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
        timestamp=0.0,
        frame_number=0
    )
    
    # Initialize analyzers
    acoustic_analyzer = AcousticAnalyzer()
    visual_analyzer = VisualAnalyzer()
    linguistic_analyzer = LinguisticAnalyzer()
    
    # Measure latency for multiple iterations
    latencies = []
    num_iterations = 10
    
    for i in range(num_iterations):
        start = time.time()
        
        # Process through all modalities
        acoustic_result = await acoustic_analyzer.analyze_audio(audio_frame)
        visual_result = await visual_analyzer.analyze_frame(video_frame)
        linguistic_result = await linguistic_analyzer.analyze_audio(audio_frame)
        
        # Fuse results
        fusion_engine = FusionEngine(
            acoustic_analyzer=MockAnalyzer(acoustic_result),
            visual_analyzer=MockAnalyzer(visual_result),
            linguistic_analyzer=MockAnalyzer(linguistic_result)
        )
        
        sentiment = fusion_engine.fuse(acoustic_result, visual_result, linguistic_result)
        
        # Record latency
        latency = time.time() - start
        latencies.append(latency)
        
        # Validate individual iteration
        assert latency < 3.0, f"Latency {latency:.3f}s exceeds maximum 3s"
    
    # Get statistics
    mean_latency = np.mean(latencies)
    max_latency = np.max(latencies)
    p95_latency = np.percentile(latencies, 95)
    
    print(f"\n✓ Latency test passed:")
    print(f"  Mean latency: {mean_latency:.3f}s")
    print(f"  Max latency: {max_latency:.3f}s")
    print(f"  P95 latency: {p95_latency:.3f}s")
    
    # Validate requirements
    assert max_latency < 3.0, f"Maximum latency {max_latency:.3f}s exceeds 3s requirement"
    
    # Warn if target not met
    if mean_latency > 1.0:
        print(f"  ⚠ Mean latency {mean_latency:.3f}s exceeds 1s target (but within 3s max)")


@pytest.mark.asyncio
async def test_all_modalities_contribute():
    """
    Test that all modalities contribute to the final sentiment score.
    
    Validates that acoustic, visual, and linguistic analysis all
    influence the fusion result.
    """
    # Create test data with distinct sentiment signals
    acoustic_result = AcousticResult(
        emotion_scores={"happy": 0.9, "neutral": 0.1},
        confidence=0.9,
        features=None,
        timestamp=0.0
    )
    
    visual_result = VisualResult(
        emotion_scores={"sad": 0.8, "neutral": 0.2},
        confidence=0.8,
        face_detected=True,
        face_landmarks=None,
        timestamp=0.0
    )
    
    linguistic_result = LinguisticResult(
        transcription="This is terrible news",
        emotion_scores={"sad": 0.9, "neutral": 0.1},
        confidence=0.9,
        transcription_confidence=0.9,
        timestamp=0.0
    )
    
    # Test with all modalities
    fusion_engine = FusionEngine(
        acoustic_analyzer=MockAnalyzer(acoustic_result),
        visual_analyzer=MockAnalyzer(visual_result),
        linguistic_analyzer=MockAnalyzer(linguistic_result)
    )
    
    score_all = fusion_engine.fuse(acoustic_result, visual_result, linguistic_result)
    
    # Test with only acoustic
    score_acoustic_only = fusion_engine.fuse(acoustic_result, None, None)
    
    # Test with only visual
    score_visual_only = fusion_engine.fuse(None, visual_result, None)
    
    # Test with only linguistic
    score_linguistic_only = fusion_engine.fuse(None, None, linguistic_result)
    
    # Validate all scores are different
    assert score_all.score != score_acoustic_only.score, "Acoustic doesn't influence fusion"
    assert score_all.score != score_visual_only.score, "Visual doesn't influence fusion"
    assert score_all.score != score_linguistic_only.score, "Linguistic doesn't influence fusion"
    
    # Validate modality contributions are tracked
    assert 'acoustic' in score_all.modality_contributions
    assert 'visual' in score_all.modality_contributions
    assert 'linguistic' in score_all.modality_contributions
    
    print(f"\n✓ Modality contribution test passed:")
    print(f"  All modalities score: {score_all.score:.3f}")
    print(f"  Acoustic only: {score_acoustic_only.score:.3f}")
    print(f"  Visual only: {score_visual_only.score:.3f}")
    print(f"  Linguistic only: {score_linguistic_only.score:.3f}")


@pytest.mark.asyncio
async def test_error_recovery_missing_modality():
    """
    Test error recovery when one modality fails.
    
    Validates Requirement 9.4: System continues with remaining modalities.
    """
    # Create results with one modality missing (None)
    acoustic_result = AcousticResult(
        emotion_scores={"happy": 0.8, "neutral": 0.2},
        confidence=0.8,
        features=None,
        timestamp=0.0
    )
    
    visual_result = None  # Simulate visual analysis failure
    
    linguistic_result = LinguisticResult(
        transcription="Good news",
        emotion_scores={"happy": 0.7, "neutral": 0.3},
        confidence=0.7,
        transcription_confidence=0.8,
        timestamp=0.0
    )
    
    fusion_engine = FusionEngine(
        acoustic_analyzer=MockAnalyzer(acoustic_result),
        visual_analyzer=MockAnalyzer(visual_result),
        linguistic_analyzer=MockAnalyzer(linguistic_result)
    )
    
    # Should not crash with missing modality
    sentiment = fusion_engine.fuse(acoustic_result, visual_result, linguistic_result)
    
    assert sentiment is not None, "Fusion failed with missing modality"
    assert -1.0 <= sentiment.score <= 1.0
    assert 0.0 <= sentiment.confidence <= 1.0
    
    print(f"\n✓ Error recovery test passed:")
    print(f"  Score with missing visual: {sentiment.score:.3f}")
    print(f"  Confidence: {sentiment.confidence:.3f}")


@pytest.mark.asyncio
async def test_error_recovery_low_confidence():
    """
    Test error recovery when modalities report low confidence.
    
    Validates that low-confidence results are handled gracefully.
    """
    # Create results with very low confidence
    acoustic_result = AcousticResult(
        emotion_scores={"neutral": 1.0},
        confidence=0.05,  # Very low confidence
        features=None,
        timestamp=0.0
    )
    
    visual_result = VisualResult(
        emotion_scores={"neutral": 1.0},
        confidence=0.03,  # Very low confidence
        face_detected=False,
        face_landmarks=None,
        timestamp=0.0
    )
    
    linguistic_result = LinguisticResult(
        transcription="",
        emotion_scores={"neutral": 1.0},
        confidence=0.02,  # Very low confidence
        transcription_confidence=0.1,
        timestamp=0.0
    )
    
    fusion_engine = FusionEngine(
        acoustic_analyzer=MockAnalyzer(acoustic_result),
        visual_analyzer=MockAnalyzer(visual_result),
        linguistic_analyzer=MockAnalyzer(linguistic_result)
    )
    
    # Should handle low confidence gracefully
    sentiment = fusion_engine.fuse(acoustic_result, visual_result, linguistic_result)
    
    assert sentiment is not None, "Fusion failed with low confidence inputs"
    assert -1.0 <= sentiment.score <= 1.0
    assert 0.0 <= sentiment.confidence <= 1.0
    
    # Overall confidence should be low
    assert sentiment.confidence < 0.5, "Confidence should reflect low-quality inputs"
    
    print(f"\n✓ Low confidence recovery test passed:")
    print(f"  Score: {sentiment.score:.3f}")
    print(f"  Confidence: {sentiment.confidence:.3f}")


@pytest.mark.asyncio
async def test_stream_reconnection():
    """
    Test stream reconnection after interruption.
    
    Validates Requirement 8.4: System handles stream interruption.
    """
    from src.models.enums import StreamProtocol
    
    stream_manager = StreamInputManager()
    
    # Connect to a file
    video_path = Path("temp_video.mp4")
    if not video_path.exists():
        pytest.skip("Sample video file not found")
    
    try:
        # Initial connection
        connection = stream_manager.connect(str(video_path), StreamProtocol.FILE)
        assert connection is not None
        assert connection.is_active
        
        # Disconnect
        stream_manager.disconnect()
        assert not stream_manager.is_active()
        
        # Reconnect
        connection = stream_manager.connect(str(video_path), StreamProtocol.FILE)
        assert connection is not None
        assert connection.is_active
        
        print(f"\n✓ Reconnection test passed")
        
    finally:
        stream_manager.disconnect()


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "-s"])
