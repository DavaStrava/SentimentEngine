"""Streamlit Application Runner

This module provides a Streamlit-based interface that integrates with the
sentiment analysis pipeline for real-time visualization.
"""

import streamlit as st
import asyncio
import threading
import time
from pathlib import Path

from src.main import SentimentEngine
from src.ui.display import SentimentDisplay


class StreamlitApp:
    """Streamlit application wrapper for sentiment analysis.
    
    This class manages the integration between the async sentiment engine
    and the Streamlit UI, running the engine in a background thread.
    """
    
    def __init__(self):
        """Initialize the Streamlit app."""
        # Initialize session state
        if 'engine' not in st.session_state:
            st.session_state.engine = None
        if 'engine_thread' not in st.session_state:
            st.session_state.engine_thread = None
        if 'running' not in st.session_state:
            st.session_state.running = False
    
    def run_engine_async(self, video_path: str):
        """Run the sentiment engine in async context.
        
        Args:
            video_path: Path to video file
        """
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            engine = SentimentEngine()
            st.session_state.engine = engine
            loop.run_until_complete(engine.run(video_path))
        except Exception as e:
            st.error(f"Engine error: {e}")
        finally:
            loop.close()
    
    def start_engine(self, video_path: str):
        """Start the sentiment engine in background thread.
        
        Args:
            video_path: Path to video file
        """
        if not st.session_state.running:
            # Start engine in background thread
            thread = threading.Thread(
                target=self.run_engine_async,
                args=(video_path,),
                daemon=True
            )
            thread.start()
            st.session_state.engine_thread = thread
            st.session_state.running = True
    
    def stop_engine(self):
        """Stop the sentiment engine."""
        if st.session_state.running and st.session_state.engine:
            asyncio.run(st.session_state.engine.shutdown())
            st.session_state.running = False
    
    def render(self):
        """Render the Streamlit interface."""
        st.set_page_config(
            page_title="Real-Time Sentiment Analysis",
            page_icon="🎭",
            layout="wide"
        )
        
        st.title("🎭 Real-Time Sentiment Analysis Engine")
        st.markdown("---")
        
        # Sidebar controls
        with st.sidebar:
            st.header("Video Input")
            
            # Video file upload
            video_file = st.file_uploader(
                "Upload Video File",
                type=['mp4', 'avi', 'mov', 'mkv']
            )
            
            # Or provide path
            video_path = st.text_input(
                "Or enter video path",
                placeholder="/path/to/video.mp4"
            )
            
            # Start/Stop buttons
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("▶️ Start", disabled=st.session_state.running):
                    if video_file:
                        # Save uploaded file
                        temp_path = Path("temp_video.mp4")
                        temp_path.write_bytes(video_file.read())
                        self.start_engine(str(temp_path))
                        st.success("Engine started!")
                    elif video_path and Path(video_path).exists():
                        self.start_engine(video_path)
                        st.success("Engine started!")
                    else:
                        st.error("Please provide a valid video file")
            
            with col2:
                if st.button("⏹️ Stop", disabled=not st.session_state.running):
                    self.stop_engine()
                    st.info("Engine stopped")
            
            # Status
            st.markdown("---")
            st.markdown("### Status")
            if st.session_state.running:
                st.success("🟢 Running")
            else:
                st.info("⚪ Stopped")
        
        # Main display area
        if st.session_state.running and st.session_state.engine:
            # Get display from engine
            display = st.session_state.engine.display
            
            # Render display
            display.render()
            
            # Auto-refresh
            time.sleep(0.1)
            st.rerun()
        else:
            st.info("👈 Upload a video file or provide a path to start analysis")
            
            # Show example
            st.markdown("### How it works")
            st.markdown("""
            1. **Upload** a video file or provide a path
            2. **Click Start** to begin analysis
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


def main():
    """Main entry point for Streamlit app."""
    app = StreamlitApp()
    app.render()


if __name__ == "__main__":
    main()
