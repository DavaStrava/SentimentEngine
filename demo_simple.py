#!/usr/bin/env python3
"""Simple demo of the sentiment analysis pipeline without Streamlit.

This demonstrates the core functionality without the complexity of
running asyncio in Streamlit threads.
"""

import asyncio
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.models.frames import AudioFrame, VideoFrame
from src.models.results import AcousticResult, VisualResult, LinguisticResult
from src.analysis.acoustic import AcousticAnalyzer
from src.analysis.visual import VisualAnalyzer
from src.analysis.linguistic import LinguisticAnalyzer
from src.fusion.fusion_engine import FusionEngine
import numpy as np


async def demo_pipeline():
    """Demonstrate the sentiment analysis pipeline with synthetic data."""
    
    print("=" * 60)
    print("Sentiment Analysis Pipeline Demo")
    print("=" * 60)
    print()
    
    # Initialize analyzers
    print("Initializing analyzers...")
    acoustic_analyzer = AcousticAnalyzer()
    visual_analyzer = VisualAnalyzer()
    visual_analyzer._load_models()
    linguistic_analyzer = LinguisticAnalyzer()
    
    # Initialize fusion engine
    fusion_engine = FusionEngine(
        acoustic_analyzer=acoustic_analyzer,
        visual_analyzer=visual_analyzer,
        linguistic_analyzer=linguistic_analyzer
    )
    
    print("✓ All analyzers initialized")
    print()
    
    # Simulate processing 10 frames
    print("Processing 10 synthetic frames...")
    print("-" * 60)
    
    for i in range(10):
        print(f"\nFrame {i+1}/10:")
        
        # Create synthetic audio frame
        audio_frame = AudioFrame(
            samples=np.random.randn(16000).astype(np.float32),
            sample_rate=16000,
            timestamp=float(i),
            duration=1.0
        )
        
        # Create synthetic video frame
        video_frame = VideoFrame(
            image=np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
            timestamp=float(i),
            frame_number=i
        )
        
        # Analyze audio
        print("  Analyzing audio...", end=" ")
        acoustic_result = await acoustic_analyzer.analyze_audio(audio_frame)
        if acoustic_result:
            print(f"✓ (confidence: {acoustic_result.confidence:.3f})")
        else:
            print("✗")
        
        # Analyze video
        print("  Analyzing video...", end=" ")
        visual_result = await visual_analyzer.analyze_frame(video_frame)
        if visual_result:
            print(f"✓ (confidence: {visual_result.confidence:.3f}, face: {visual_result.face_detected})")
        else:
            print("✗")
        
        # Analyze linguistic (every 2 seconds)
        if i % 2 == 0:
            print("  Analyzing speech...", end=" ")
            linguistic_result = await linguistic_analyzer.analyze_audio(audio_frame)
            if linguistic_result:
                print(f"✓ (confidence: {linguistic_result.confidence:.3f})")
            else:
                print("  (buffering)")
        
        # Fuse results
        print("  Fusing modalities...", end=" ")
        sentiment_score = fusion_engine.fuse(
            acoustic_result,
            visual_result,
            linguistic_analyzer.get_latest_result()
        )
        
        print(f"✓")
        print(f"  → Sentiment Score: {sentiment_score.score:+.3f}")
        print(f"  → Confidence: {sentiment_score.confidence:.3f}")
        print(f"  → Modalities: {list(sentiment_score.modality_contributions.keys())}")
        
        # Small delay
        await asyncio.sleep(0.5)
    
    print()
    print("-" * 60)
    print("Demo complete!")
    print()
    
    # Show final summary
    print("Summary:")
    print(f"  Total frames processed: 10")
    print(f"  Final sentiment: {sentiment_score.score:+.3f}")
    print(f"  Final confidence: {sentiment_score.confidence:.3f}")
    print(f"  Emotion breakdown:")
    for emotion, score in sentiment_score.emotion_breakdown.items():
        if score > 0.05:
            print(f"    - {emotion}: {score:.1%}")
    
    print()
    print("=" * 60)
    print("This demonstrates the core pipeline working!")
    print("All three modalities are being analyzed and fused.")
    print("=" * 60)


if __name__ == "__main__":
    try:
        asyncio.run(demo_pipeline())
    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
