#!/usr/bin/env python3
"""Simplified Web UI Runner - No Redis Required

This script runs a simplified version of the sentiment analysis system
with a Streamlit web interface that doesn't require Redis.
"""

import streamlit as st
import numpy as np
import time
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.models.frames import AudioFrame, VideoFrame
from src.models.results import AcousticResult, VisualResult, LinguisticResult
from src.analysis.acoustic import AcousticAnalyzer
from src.analysis.visual import VisualAnalyzer
from src.analysis.linguistic import LinguisticAnalyzer
from src.fusion.fusion_engine import FusionEngine


def create_demo_audio_frame(timestamp: float) -> AudioFrame:
    """Create a demo audio frame with synthetic data."""
    # Generate 0.5 seconds of audio at 16kHz
    duration = 0.5
    sample_rate = 16000
    samples = np.random.randn(int(sample_rate * duration)).astype(np.float32) * 0.1
    
    return AudioFrame(
        samples=samples,
        sample_rate=sample_rate,
        timestamp=timestamp,
        duration=duration
    )


def create_demo_video_frame(timestamp: float, frame_number: int) -> VideoFrame:
    """Create a demo video frame with synthetic data."""
    # Generate a 480x640 RGB image
    image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    return VideoFrame(
        image=image,
        timestamp=timestamp,
        frame_number=frame_number
    )


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="Real-Time Sentiment Analysis",
        page_icon="🎭",
        layout="wide"
    )
    
    st.title("🎭 Real-Time Sentiment Analysis Engine")
    st.markdown("**Demo Mode** - Processing synthetic data (no video file required)")
    st.markdown("---")
    
    # Initialize components in session state
    if 'initialized' not in st.session_state:
        st.session_state.initialized = False
        st.session_state.acoustic_analyzer = None
        st.session_state.visual_analyzer = None
        st.session_state.linguistic_analyzer = None
        st.session_state.fusion_engine = None
        st.session_state.frame_count = 0
        st.session_state.running = False
        st.session_state.sentiment_history = []
    
    # Sidebar controls
    with st.sidebar:
        st.header("Controls")
        
        if st.button("🚀 Initialize System", disabled=st.session_state.initialized):
            with st.spinner("Loading models..."):
                try:
                    st.session_state.acoustic_analyzer = AcousticAnalyzer()
                    st.session_state.visual_analyzer = VisualAnalyzer()
                    st.session_state.linguistic_analyzer = LinguisticAnalyzer()
                    st.session_state.fusion_engine = FusionEngine()
                    st.session_state.initialized = True
                    st.success("✅ System initialized!")
                except Exception as e:
                    st.error(f"Initialization error: {e}")
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("▶️ Start", disabled=not st.session_state.initialized or st.session_state.running):
                st.session_state.running = True
                st.session_state.frame_count = 0
                st.session_state.sentiment_history = []
        
        with col2:
            if st.button("⏹️ Stop", disabled=not st.session_state.running):
                st.session_state.running = False
        
        st.markdown("---")
        st.markdown("### Status")
        if st.session_state.running:
            st.success("🟢 Running")
        elif st.session_state.initialized:
            st.info("🟡 Ready")
        else:
            st.warning("⚪ Not Initialized")
        
        st.markdown("---")
        st.markdown(f"**Frames Processed:** {st.session_state.frame_count}")
    
    # Main display area
    if not st.session_state.initialized:
        st.info("👈 Click 'Initialize System' to start")
        
        st.markdown("### How it works")
        st.markdown("""
        1. **Initialize** the system (loads analysis models)
        2. **Click Start** to begin processing demo frames
        3. **Watch** real-time sentiment scores update
        4. **View** acoustic, visual, and linguistic contributions
        5. **Track** emotion breakdowns and historical trends
        """)
        
        st.markdown("### Features")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### 🎤 Acoustic Analysis")
            st.markdown("- Vocal tone detection")
            st.markdown("- Pitch & energy analysis")
            st.markdown("- Speaking rate tracking")
        
        with col2:
            st.markdown("#### 👁️ Visual Analysis")
            st.markdown("- Facial expression recognition")
            st.markdown("- Emotion classification")
            st.markdown("- Quality indicators")
        
        with col3:
            st.markdown("#### 💬 Linguistic Analysis")
            st.markdown("- Speech-to-text transcription")
            st.markdown("- Sentiment analysis")
            st.markdown("- Domain adaptation")
    
    elif st.session_state.running:
        # Process a frame
        timestamp = st.session_state.frame_count * 0.5
        
        # Create demo frames
        audio_frame = create_demo_audio_frame(timestamp)
        video_frame = create_demo_video_frame(timestamp, st.session_state.frame_count)
        
        # Analyze frames
        with st.spinner("Analyzing..."):
            acoustic_result = st.session_state.acoustic_analyzer.analyze_audio(audio_frame)
            visual_result = st.session_state.visual_analyzer.analyze_frame(video_frame)
            linguistic_result = st.session_state.linguistic_analyzer.analyze_audio(audio_frame)
        
        # Fuse results
        sentiment_score = st.session_state.fusion_engine.fuse(
            acoustic_result,
            visual_result,
            linguistic_result
        )
        
        # Store in history
        st.session_state.sentiment_history.append({
            'timestamp': timestamp,
            'score': sentiment_score.score,
            'confidence': sentiment_score.confidence,
            'acoustic': sentiment_score.modality_contributions.get('acoustic', 0),
            'visual': sentiment_score.modality_contributions.get('visual', 0),
            'linguistic': sentiment_score.modality_contributions.get('linguistic', 0)
        })
        
        # Keep last 60 seconds
        if len(st.session_state.sentiment_history) > 120:
            st.session_state.sentiment_history = st.session_state.sentiment_history[-120:]
        
        st.session_state.frame_count += 1
        
        # Display results
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Sentiment Score",
                f"{sentiment_score.score:.3f}",
                delta=None
            )
        
        with col2:
            st.metric(
                "Confidence",
                f"{sentiment_score.confidence:.1%}",
                delta=None
            )
        
        with col3:
            # Determine dominant emotion
            if sentiment_score.emotion_breakdown:
                dominant = max(sentiment_score.emotion_breakdown.items(), key=lambda x: x[1])
                st.metric(
                    "Dominant Emotion",
                    dominant[0].title(),
                    delta=None
                )
        
        st.markdown("---")
        
        # Modality contributions
        st.subheader("Modality Contributions")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("🎤 Acoustic", f"{sentiment_score.modality_contributions.get('acoustic', 0):.1%}")
        
        with col2:
            st.metric("👁️ Visual", f"{sentiment_score.modality_contributions.get('visual', 0):.1%}")
        
        with col3:
            st.metric("💬 Linguistic", f"{sentiment_score.modality_contributions.get('linguistic', 0):.1%}")
        
        # History chart
        if len(st.session_state.sentiment_history) > 1:
            st.markdown("---")
            st.subheader("Sentiment History")
            
            import pandas as pd
            df = pd.DataFrame(st.session_state.sentiment_history)
            st.line_chart(df.set_index('timestamp')['score'])
        
        # Emotion breakdown
        if sentiment_score.emotion_breakdown:
            st.markdown("---")
            st.subheader("Emotion Breakdown")
            
            import pandas as pd
            emotion_df = pd.DataFrame([
                {'Emotion': k.title(), 'Score': v}
                for k, v in sentiment_score.emotion_breakdown.items()
            ])
            st.bar_chart(emotion_df.set_index('Emotion'))
        
        # Auto-refresh
        time.sleep(0.5)
        st.rerun()
    
    else:
        st.info("👈 Click 'Start' to begin processing")


if __name__ == "__main__":
    main()
