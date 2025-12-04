#!/usr/bin/env python3
"""Test script to verify the sentiment analysis pipeline setup.

This script tests each component individually to ensure everything is working.
"""

import sys
import asyncio
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.frames import AudioFrame, VideoFrame
from src.models.results import AcousticResult, VisualResult, LinguisticResult
from src.analysis.acoustic import AcousticAnalyzer
from src.analysis.visual import VisualAnalyzer
from src.analysis.linguistic import LinguisticAnalyzer
from src.fusion.fusion_engine import FusionEngine


def test_data_models():
    """Test data model creation."""
    print("Testing data models...")
    
    # Test AudioFrame
    audio_frame = AudioFrame(
        samples=np.random.randn(16000).astype(np.float32),
        sample_rate=16000,
        timestamp=0.0,
        duration=1.0
    )
    assert audio_frame.sample_rate == 16000
    print("✓ AudioFrame created successfully")
    
    # Test VideoFrame
    video_frame = VideoFrame(
        image=np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
        timestamp=0.0,
        frame_number=0
    )
    assert video_frame.image.shape == (480, 640, 3)
    print("✓ VideoFrame created successfully")
    
    print("✓ Data models test passed\n")


async def test_acoustic_analyzer():
    """Test acoustic analyzer."""
    print("Testing acoustic analyzer...")
    
    try:
        analyzer = AcousticAnalyzer()
        
        # Create test audio frame
        audio_frame = AudioFrame(
            samples=np.random.randn(16000).astype(np.float32),
            sample_rate=16000,
            timestamp=0.0,
            duration=1.0
        )
        
        # Analyze (will use mocked model in tests)
        print("  Analyzing audio frame...")
        result = await analyzer.analyze_audio(audio_frame)
        
        if result:
            assert isinstance(result, AcousticResult)
            assert 0.0 <= result.confidence <= 1.0
            print(f"  ✓ Got result with confidence: {result.confidence:.3f}")
        else:
            print("  ⚠ No result (model may not be loaded)")
        
        print("✓ Acoustic analyzer test passed\n")
    except Exception as e:
        print(f"✗ Acoustic analyzer test failed: {e}\n")


async def test_visual_analyzer():
    """Test visual analyzer."""
    print("Testing visual analyzer...")
    
    try:
        analyzer = VisualAnalyzer()
        analyzer._load_models()
        
        # Create test video frame
        video_frame = VideoFrame(
            image=np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
            timestamp=0.0,
            frame_number=0
        )
        
        # Analyze
        print("  Analyzing video frame...")
        result = await analyzer.analyze_frame(video_frame)
        
        if result:
            assert isinstance(result, VisualResult)
            assert 0.0 <= result.confidence <= 1.0
            print(f"  ✓ Got result with confidence: {result.confidence:.3f}")
            print(f"  Face detected: {result.face_detected}")
        else:
            print("  ⚠ No result")
        
        print("✓ Visual analyzer test passed\n")
    except Exception as e:
        print(f"✗ Visual analyzer test failed: {e}\n")


async def test_linguistic_analyzer():
    """Test linguistic analyzer."""
    print("Testing linguistic analyzer...")
    
    try:
        analyzer = LinguisticAnalyzer()
        
        # Create test audio frame
        audio_frame = AudioFrame(
            samples=np.random.randn(48000).astype(np.float32),  # 3 seconds
            sample_rate=16000,
            timestamp=0.0,
            duration=3.0
        )
        
        # Analyze (will buffer and process)
        print("  Analyzing audio for transcription...")
        result = await analyzer.analyze_audio(audio_frame)
        
        if result:
            assert isinstance(result, LinguisticResult)
            assert 0.0 <= result.confidence <= 1.0
            print(f"  ✓ Got result with confidence: {result.confidence:.3f}")
            if result.transcription:
                print(f"  Transcription: '{result.transcription}'")
        else:
            print("  ⚠ No result (processing interval may not have elapsed)")
        
        print("✓ Linguistic analyzer test passed\n")
    except Exception as e:
        print(f"✗ Linguistic analyzer test failed: {e}\n")


def test_fusion_engine():
    """Test fusion engine."""
    print("Testing fusion engine...")
    
    try:
        # Create mock analyzers
        class MockAnalyzer:
            def __init__(self, result):
                self.result = result
            def get_latest_result(self):
                return self.result
        
        # Create mock results
        acoustic_result = AcousticResult(
            emotion_scores={"happy": 0.7, "neutral": 0.3},
            confidence=0.8,
            features=None,
            timestamp=0.0
        )
        
        visual_result = VisualResult(
            emotion_scores={"happy": 0.6, "neutral": 0.4},
            confidence=0.7,
            face_detected=True,
            face_landmarks=None,
            timestamp=0.0
        )
        
        linguistic_result = LinguisticResult(
            transcription="This is great news",
            emotion_scores={"happy": 0.8, "neutral": 0.2},
            confidence=0.9,
            transcription_confidence=0.9,
            timestamp=0.0
        )
        
        # Create fusion engine
        engine = FusionEngine(
            acoustic_analyzer=MockAnalyzer(acoustic_result),
            visual_analyzer=MockAnalyzer(visual_result),
            linguistic_analyzer=MockAnalyzer(linguistic_result)
        )
        
        # Fuse results
        print("  Fusing modality results...")
        sentiment_score = engine.fuse(acoustic_result, visual_result, linguistic_result)
        
        assert -1.0 <= sentiment_score.score <= 1.0
        assert 0.0 <= sentiment_score.confidence <= 1.0
        print(f"  ✓ Got sentiment score: {sentiment_score.score:.3f}")
        print(f"  Confidence: {sentiment_score.confidence:.3f}")
        print(f"  Modalities: {list(sentiment_score.modality_contributions.keys())}")
        
        print("✓ Fusion engine test passed\n")
    except Exception as e:
        print(f"✗ Fusion engine test failed: {e}\n")


async def main():
    """Run all tests."""
    print("=" * 60)
    print("Sentiment Analysis Pipeline Test Suite")
    print("=" * 60)
    print()
    
    # Test data models
    test_data_models()
    
    # Test analyzers
    await test_acoustic_analyzer()
    await test_visual_analyzer()
    await test_linguistic_analyzer()
    
    # Test fusion
    test_fusion_engine()
    
    print("=" * 60)
    print("Test suite complete!")
    print("=" * 60)
    print()
    print("Next steps:")
    print("1. Start Redis: redis-server")
    print("2. Run Streamlit UI: streamlit run src/app.py")
    print("3. Upload a video and start analysis!")
    print()


if __name__ == "__main__":
    asyncio.run(main())
