"""
Performance tests for latency validation.

Tests Requirement 9.1:
- Target latency: 1 second
- Maximum latency: 3 seconds
"""

import pytest
import asyncio
import time
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.frames import AudioFrame, VideoFrame
from src.models.results import AcousticResult, VisualResult, LinguisticResult
from src.analysis.acoustic import AcousticAnalyzer
from src.analysis.visual import VisualAnalyzer
from src.analysis.linguistic import LinguisticAnalyzer
from src.fusion.fusion_engine import FusionEngine


class MockAnalyzer:
    """Mock analyzer for testing."""
    def __init__(self, result):
        self.result = result
    def get_latest_result(self):
        return self.result


@pytest.mark.asyncio
async def test_component_latency_breakdown():
    """
    Measure latency of individual components.
    
    Helps identify bottlenecks in the pipeline.
    """
    # Create test data
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
    
    # Measure each component
    acoustic_times = []
    visual_times = []
    linguistic_times = []
    fusion_times = []
    
    num_iterations = 20
    
    for i in range(num_iterations):
        # Measure acoustic analysis
        start = time.time()
        acoustic_result = await acoustic_analyzer.analyze_audio(audio_frame)
        acoustic_times.append(time.time() - start)
        
        # Measure visual analysis
        start = time.time()
        visual_result = await visual_analyzer.analyze_frame(video_frame)
        visual_times.append(time.time() - start)
        
        # Measure linguistic analysis (less frequently)
        if i % 5 == 0:
            start = time.time()
            linguistic_result = await linguistic_analyzer.analyze_audio(audio_frame)
            linguistic_times.append(time.time() - start)
        else:
            linguistic_result = None
        
        # Measure fusion
        if acoustic_result or visual_result or linguistic_result:
            fusion_engine = FusionEngine(
                acoustic_analyzer=MockAnalyzer(acoustic_result),
                visual_analyzer=MockAnalyzer(visual_result),
                linguistic_analyzer=MockAnalyzer(linguistic_result)
            )
            
            start = time.time()
            sentiment = fusion_engine.fuse(acoustic_result, visual_result, linguistic_result)
            fusion_times.append(time.time() - start)
    
    # Print summary
    print("\n" + "=" * 70)
    print("PERFORMANCE SUMMARY")
    print("=" * 70)
    
    if acoustic_times:
        print(f"\nACOUSTIC:")
        print(f"  Mean:   {np.mean(acoustic_times)*1000:.2f}ms")
        print(f"  Max:    {np.max(acoustic_times)*1000:.2f}ms")
        assert np.max(acoustic_times) < 0.5, f"Acoustic analysis too slow"
    
    if visual_times:
        print(f"\nVISUAL:")
        print(f"  Mean:   {np.mean(visual_times)*1000:.2f}ms")
        print(f"  Max:    {np.max(visual_times)*1000:.2f}ms")
        assert np.max(visual_times) < 0.5, f"Visual analysis too slow"
    
    if linguistic_times:
        print(f"\nLINGUISTIC:")
        print(f"  Mean:   {np.mean(linguistic_times)*1000:.2f}ms")
        print(f"  Max:    {np.max(linguistic_times)*1000:.2f}ms")
    
    if fusion_times:
        print(f"\nFUSION:")
        print(f"  Mean:   {np.mean(fusion_times)*1000:.2f}ms")
        print(f"  Max:    {np.max(fusion_times)*1000:.2f}ms")
        assert np.max(fusion_times) < 0.1, f"Fusion too slow"
    
    print("\n" + "=" * 70)
    print("\n✓ Component latency test passed")


@pytest.mark.asyncio
async def test_end_to_end_latency_under_load():
    """
    Test end-to-end latency under sustained load.
    
    Validates Requirement 9.1 under realistic conditions.
    """
    # Create test data
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
    
    # Simulate sustained load
    latencies = []
    num_iterations = 30
    
    for i in range(num_iterations):
        start = time.time()
        
        # Process all modalities
        acoustic_result = await acoustic_analyzer.analyze_audio(audio_frame)
        visual_result = await visual_analyzer.analyze_frame(video_frame)
        
        # Linguistic less frequently
        if i % 3 == 0:
            linguistic_result = await linguistic_analyzer.analyze_audio(audio_frame)
        else:
            linguistic_result = None
        
        # Fuse results
        fusion_engine = FusionEngine(
            acoustic_analyzer=MockAnalyzer(acoustic_result),
            visual_analyzer=MockAnalyzer(visual_result),
            linguistic_analyzer=MockAnalyzer(linguistic_result)
        )
        
        sentiment = fusion_engine.fuse(acoustic_result, visual_result, linguistic_result)
        
        # Record end-to-end time
        elapsed = time.time() - start
        latencies.append(elapsed)
        
        # Validate individual iteration
        assert elapsed < 3.0, f"Iteration {i} exceeded 3s: {elapsed:.3f}s"
    
    # Print summary
    mean_latency = np.mean(latencies)
    max_latency = np.max(latencies)
    p95_latency = np.percentile(latencies, 95)
    p99_latency = np.percentile(latencies, 99)
    
    print(f"\n✓ End-to-end latency test:")
    print(f"  Mean:   {mean_latency:.3f}s")
    print(f"  Max:    {max_latency:.3f}s")
    print(f"  P95:    {p95_latency:.3f}s")
    print(f"  P99:    {p99_latency:.3f}s")
    
    # Validate requirements
    assert max_latency < 3.0, f"Maximum latency {max_latency:.3f}s exceeds 3s requirement"
    assert p99_latency < 3.0, f"P99 latency {p99_latency:.3f}s exceeds 3s requirement"
    
    # Warn if target not met
    if mean_latency > 1.0:
        print(f"\n⚠ Mean latency {mean_latency:.3f}s exceeds 1s target")
        print("  (but within 3s maximum requirement)")
    else:
        print(f"\n✓ Mean latency {mean_latency:.3f}s meets 1s target")


@pytest.mark.asyncio
async def test_parallel_processing_performance():
    """
    Test that parallel processing improves performance.
    
    Validates that async architecture provides benefits.
    """
    # Create test data
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
    
    # Sequential processing
    start = time.time()
    acoustic_result = await acoustic_analyzer.analyze_audio(audio_frame)
    visual_result = await visual_analyzer.analyze_frame(video_frame)
    linguistic_result = await linguistic_analyzer.analyze_audio(audio_frame)
    sequential_time = time.time() - start
    
    # Parallel processing
    start = time.time()
    results = await asyncio.gather(
        acoustic_analyzer.analyze_audio(audio_frame),
        visual_analyzer.analyze_frame(video_frame),
        linguistic_analyzer.analyze_audio(audio_frame)
    )
    parallel_time = time.time() - start
    
    print(f"\n✓ Parallel processing test:")
    print(f"  Sequential: {sequential_time*1000:.2f}ms")
    print(f"  Parallel:   {parallel_time*1000:.2f}ms")
    print(f"  Speedup:    {sequential_time/parallel_time:.2f}x")
    
    # Parallel should be faster or similar
    assert parallel_time <= sequential_time * 1.2, "Parallel processing slower than sequential"


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "-s"])
