"""Sentiment Display Interface

This module provides a Streamlit-based visualization interface for real-time sentiment
analysis. It displays current sentiment scores, modality contributions, emotion breakdowns,
and historical trends.

Requirements:
    - Req 7.1: System displays sentiment score with timestamp
    - Req 7.2: System shows individual contributions from acoustic, visual, and linguistic analysis
    - Req 7.3: System provides sentiment score history for current session
    - Req 7.4: System highlights emotional shifts visually
    - Req 7.5: System provides summary report of sentiment trends and significant events
"""

import streamlit as st
import pandas as pd
import numpy as np
import time
from datetime import datetime
from typing import Optional, List, Dict
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.models.results import SentimentScore


class SentimentDisplay:
    """Streamlit-based visualization interface for real-time sentiment analysis.
    
    This class implements the display interface that:
    1. Shows current sentiment score as gauge and line chart
    2. Displays individual modality contributions (acoustic, visual, linguistic)
    3. Plots sentiment history over time
    4. Shows emotion category breakdown
    5. Highlights significant emotional shifts
    6. Provides session summary on stream end
    
    Attributes:
        history_duration: Duration of history to display (seconds)
        update_interval: UI update interval (seconds)
        score_history: List of (timestamp, score) tuples
        session_start_time: Timestamp when session started
    """
    
    def __init__(self, fusion_engine=None):
        """Initialize the sentiment display interface.
        
        Args:
            fusion_engine: Reference to FusionEngine for score access
        """
        self.fusion_engine = fusion_engine
        self.history_duration = 60  # Show last 60 seconds
        self.update_interval = 0.1  # Update every 100ms
        self.score_history: List[tuple] = []  # (timestamp, score, confidence)
        self.session_start_time = time.time()
        
        # Configure Streamlit page
        st.set_page_config(
            page_title="Real-Time Sentiment Analysis",
            page_icon="🎭",
            layout="wide",
            initial_sidebar_state="expanded"
        )
    
    def _get_sentiment_color(self, score: float) -> str:
        """Get color for sentiment score based on polarity thresholds.
        
        Maps sentiment scores to visual color indicators for intuitive display.
        Uses thresholds at -0.3 and +0.3 to distinguish negative, neutral, and
        positive sentiment ranges.
        
        Args:
            score: Sentiment score in [-1, 1] range where -1 is very negative,
                  0 is neutral, and +1 is very positive
            
        Returns:
            Color string: "red" for negative (< -0.3), "orange" for neutral
            (-0.3 to +0.3), or "green" for positive (> +0.3)
            
        Validates:
            - Req 7.1: System displays sentiment score with timestamp
            - Req 7.4: System highlights emotional shifts visually
        """
        if score < -0.3:
            return "red"
        elif score > 0.3:
            return "green"
        else:
            return "orange"
    
    def _get_sentiment_label(self, score: float) -> str:
        """Get human-readable label for sentiment score.
        
        Converts numerical sentiment scores to categorical labels for display.
        Uses the same thresholds as _get_sentiment_color for consistency.
        
        Args:
            score: Sentiment score in [-1, 1] range where -1 is very negative,
                  0 is neutral, and +1 is very positive
            
        Returns:
            Label string: "Negative" for scores < -0.3, "Neutral" for scores
            between -0.3 and +0.3, or "Positive" for scores > +0.3
            
        Validates:
            - Req 7.1: System displays sentiment score with timestamp
        """
        if score < -0.3:
            return "Negative"
        elif score > 0.3:
            return "Positive"
        else:
            return "Neutral"
    
    def _create_gauge_chart(self, score: float, confidence: float) -> go.Figure:
        """Create interactive gauge chart visualization for current sentiment score.
        
        Generates a Plotly gauge indicator that displays the sentiment score on a
        0-100 scale (converted from [-1, 1] range) with color-coded regions for
        negative, neutral, and positive sentiment. The gauge includes the confidence
        level as a subtitle and shows the delta from neutral (50).
        
        The gauge uses three color regions:
        - Red zone (0-35): Negative sentiment
        - Yellow zone (35-65): Neutral sentiment  
        - Green zone (65-100): Positive sentiment
        
        Args:
            score: Sentiment score in [-1, 1] range where -1 is very negative,
                  0 is neutral, and +1 is very positive
            confidence: Confidence score in [0, 1] range indicating the reliability
                       of the sentiment score
            
        Returns:
            Plotly Figure object containing the gauge chart with interactive hover
            and visual indicators
            
        Validates:
            - Req 7.1: System displays sentiment score with timestamp
            - Req 7.2: System shows individual contributions from analysis modules
        """
        # Convert score from [-1, 1] to [0, 100] for gauge
        gauge_value = (score + 1) * 50
        
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=gauge_value,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': f"Sentiment Score<br><span style='font-size:0.8em;color:gray'>Confidence: {confidence:.2%}</span>"},
            delta={'reference': 50},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkgray"},
                'bar': {'color': self._get_sentiment_color(score)},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 35], 'color': '#ffcccc'},
                    {'range': [35, 65], 'color': '#ffffcc'},
                    {'range': [65, 100], 'color': '#ccffcc'}
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 4},
                    'thickness': 0.75,
                    'value': gauge_value
                }
            }
        ))
        
        fig.update_layout(
            height=300,
            margin=dict(l=20, r=20, t=60, b=20)
        )
        
        return fig
    
    def _create_history_chart(self) -> go.Figure:
        """Create time-series line chart for sentiment score history.
        
        Generates an interactive Plotly line chart showing sentiment scores over time
        with confidence intervals displayed as a shaded area. The chart includes:
        - Main sentiment line with markers
        - Confidence band (±20% of confidence value)
        - Zero reference line (neutral sentiment)
        - Color-coded background regions for positive/negative zones
        - Relative time axis (seconds since session start)
        
        If no history data is available, displays a placeholder message.
        
        Returns:
            Plotly Figure object containing the time-series chart with sentiment
            history, confidence bands, and reference lines. Returns empty chart
            with "No data yet" message if score_history is empty.
            
        Validates:
            - Req 7.3: System provides sentiment score history for current session
            - Req 7.4: System highlights emotional shifts visually
        """
        if not self.score_history:
            # Empty chart
            fig = go.Figure()
            fig.add_annotation(
                text="No data yet",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=20, color="gray")
            )
            fig.update_layout(
                height=300,
                xaxis_title="Time",
                yaxis_title="Sentiment Score",
                yaxis_range=[-1.1, 1.1],
                margin=dict(l=20, r=20, t=40, b=20)
            )
            return fig
        
        # Extract data
        timestamps = [t for t, s, c in self.score_history]
        scores = [s for t, s, c in self.score_history]
        confidences = [c for t, s, c in self.score_history]
        
        # Convert timestamps to relative seconds
        relative_times = [t - self.session_start_time for t in timestamps]
        
        # Create figure
        fig = go.Figure()
        
        # Add sentiment line
        fig.add_trace(go.Scatter(
            x=relative_times,
            y=scores,
            mode='lines+markers',
            name='Sentiment',
            line=dict(color='blue', width=2),
            marker=dict(size=4),
            hovertemplate='Time: %{x:.1f}s<br>Score: %{y:.3f}<extra></extra>'
        ))
        
        # Add confidence as shaded area
        fig.add_trace(go.Scatter(
            x=relative_times + relative_times[::-1],
            y=[s + c * 0.2 for s, c in zip(scores, confidences)] + 
              [s - c * 0.2 for s, c in zip(scores[::-1], confidences[::-1])],
            fill='toself',
            fillcolor='rgba(0, 100, 255, 0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        # Add zero line
        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
        
        # Add positive/negative regions
        fig.add_hrect(y0=0.3, y1=1.0, fillcolor="green", opacity=0.1, line_width=0)
        fig.add_hrect(y0=-1.0, y1=-0.3, fillcolor="red", opacity=0.1, line_width=0)
        
        fig.update_layout(
            title="Sentiment History",
            xaxis_title="Time (seconds)",
            yaxis_title="Sentiment Score",
            yaxis_range=[-1.1, 1.1],
            height=300,
            margin=dict(l=20, r=20, t=40, b=20),
            hovermode='x unified'
        )
        
        return fig
    
    def _create_modality_chart(self, modality_contributions: Dict[str, float]) -> go.Figure:
        """Create bar chart showing relative contributions from each analysis modality.
        
        Visualizes how much each modality (acoustic, visual, linguistic) contributed
        to the final sentiment score through quality-aware weighted fusion. Each bar
        is color-coded by modality type and displays the percentage contribution.
        
        Color scheme:
        - Acoustic: Red (#FF6B6B)
        - Visual: Teal (#4ECDC4)
        - Linguistic: Mint (#95E1D3)
        
        Args:
            modality_contributions: Dictionary mapping modality names (str) to their
                                   normalized weight contributions (float in [0, 1]).
                                   Weights should sum to approximately 1.0.
                                   Example: {"acoustic": 0.4, "visual": 0.3, "linguistic": 0.3}
            
        Returns:
            Plotly Figure object containing the bar chart. Returns empty chart with
            "No modality data" message if contributions dictionary is empty.
            
        Validates:
            - Req 7.2: System shows individual contributions from acoustic, visual,
                      and linguistic analysis
        """
        if not modality_contributions:
            fig = go.Figure()
            fig.add_annotation(
                text="No modality data",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16, color="gray")
            )
            fig.update_layout(height=200, margin=dict(l=20, r=20, t=20, b=20))
            return fig
        
        modalities = list(modality_contributions.keys())
        weights = list(modality_contributions.values())
        
        colors = {
            'acoustic': '#FF6B6B',
            'visual': '#4ECDC4',
            'linguistic': '#95E1D3'
        }
        
        bar_colors = [colors.get(m, 'gray') for m in modalities]
        
        fig = go.Figure(data=[
            go.Bar(
                x=modalities,
                y=weights,
                marker_color=bar_colors,
                text=[f"{w:.1%}" for w in weights],
                textposition='auto',
            )
        ])
        
        fig.update_layout(
            title="Modality Contributions",
            xaxis_title="Modality",
            yaxis_title="Weight",
            yaxis_range=[0, 1],
            height=200,
            margin=dict(l=20, r=20, t=40, b=20),
            showlegend=False
        )
        
        return fig
    
    def _create_emotion_chart(self, emotion_breakdown: Dict[str, float]) -> go.Figure:
        """Create pie chart showing distribution of detected emotion categories.
        
        Visualizes the breakdown of emotion categories from the fused multi-modal
        analysis. Filters out emotions with very low scores (< 5%) for clarity.
        Each emotion is color-coded for intuitive recognition.
        
        Emotion color scheme:
        - Happy: Yellow (#FFD93D)
        - Sad: Purple (#6C5CE7)
        - Angry: Red (#FF6B6B)
        - Fearful: Light Purple (#A29BFE)
        - Disgust: Blue (#74B9FF)
        - Surprised: Orange (#FFA502)
        - Neutral: Gray (#DFE6E9)
        
        Args:
            emotion_breakdown: Dictionary mapping emotion category names (str) to
                              their probability scores (float in [0, 1]). Scores
                              should sum to approximately 1.0.
                              Example: {"happy": 0.6, "neutral": 0.3, "sad": 0.1}
            
        Returns:
            Plotly Figure object containing the pie chart with emotion distribution.
            Returns empty chart with "No emotion data" message if breakdown is empty.
            
        Validates:
            - Req 7.1: System displays sentiment score with timestamp
            - Req 7.2: System shows individual contributions from analysis modules
        """
        if not emotion_breakdown:
            fig = go.Figure()
            fig.add_annotation(
                text="No emotion data",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16, color="gray")
            )
            fig.update_layout(height=300, margin=dict(l=20, r=20, t=20, b=20))
            return fig
        
        # Filter out very small values
        filtered_emotions = {k: v for k, v in emotion_breakdown.items() if v > 0.05}
        
        if not filtered_emotions:
            filtered_emotions = emotion_breakdown
        
        emotions = list(filtered_emotions.keys())
        scores = list(filtered_emotions.values())
        
        colors = {
            'happy': '#FFD93D',
            'sad': '#6C5CE7',
            'angry': '#FF6B6B',
            'fearful': '#A29BFE',
            'disgust': '#74B9FF',
            'surprised': '#FFA502',
            'neutral': '#DFE6E9'
        }
        
        pie_colors = [colors.get(e, 'gray') for e in emotions]
        
        fig = go.Figure(data=[go.Pie(
            labels=emotions,
            values=scores,
            marker=dict(colors=pie_colors),
            textinfo='label+percent',
            hovertemplate='%{label}<br>%{percent}<extra></extra>'
        )])
        
        fig.update_layout(
            title="Emotion Breakdown",
            height=300,
            margin=dict(l=20, r=20, t=40, b=20),
            showlegend=True
        )
        
        return fig
    
    def _detect_shifts(self) -> List[tuple]:
        """Detect significant emotional shifts in sentiment score history.
        
        Analyzes the score history to identify moments where sentiment changed
        dramatically (absolute change >= 0.4). These shifts represent important
        emotional transitions that may warrant user attention.
        
        The detection algorithm:
        1. Compares each score with the previous score
        2. Calculates absolute change
        3. Records shifts that exceed the threshold (0.4)
        
        Returns:
            List of tuples, each containing (timestamp, old_score, new_score) for
            detected shifts. Empty list if fewer than 2 data points exist.
            Example: [(10.5, 0.2, 0.7), (25.3, 0.6, -0.1)]
            
        Validates:
            - Req 7.4: System highlights emotional shifts visually
            - Req 7.5: System provides summary report of sentiment trends and
                      significant events
        """
        if len(self.score_history) < 2:
            return []
        
        shifts = []
        threshold = 0.4  # Minimum change to be considered significant
        
        for i in range(1, len(self.score_history)):
            prev_time, prev_score, _ = self.score_history[i-1]
            curr_time, curr_score, _ = self.score_history[i]
            
            change = abs(curr_score - prev_score)
            if change >= threshold:
                shifts.append((curr_time, prev_score, curr_score))
        
        return shifts
    
    def update_score(self, sentiment: SentimentScore) -> None:
        """Update display with new sentiment score.
        
        Args:
            sentiment: New sentiment score to display
            
        Validates:
            - Req 7.1: System displays sentiment score with timestamp
            - Req 7.2: System shows individual contributions from analysis modules
        """
        # Add to history
        self.score_history.append((sentiment.timestamp, sentiment.score, sentiment.confidence))
        
        # Trim history to duration
        current_time = time.time()
        self.score_history = [
            (t, s, c) for t, s, c in self.score_history
            if current_time - t <= self.history_duration
        ]
    
    def show_history(self, duration_seconds: int) -> None:
        """Update history duration.
        
        Args:
            duration_seconds: Duration of history to display
            
        Validates:
            - Req 7.3: System provides sentiment score history for current session
        """
        self.history_duration = duration_seconds
    
    def show_modality_breakdown(self) -> None:
        """Display modality breakdown (handled in render method)."""
        pass
    
    def generate_summary(self) -> Dict:
        """Generate session summary report.
        
        Returns:
            Dictionary containing summary statistics
            
        Validates:
            - Req 7.5: System provides summary report of sentiment trends
        """
        if not self.score_history:
            return {
                'duration': 0,
                'avg_score': 0,
                'min_score': 0,
                'max_score': 0,
                'shifts': []
            }
        
        scores = [s for _, s, _ in self.score_history]
        
        summary = {
            'duration': time.time() - self.session_start_time,
            'avg_score': np.mean(scores),
            'min_score': np.min(scores),
            'max_score': np.max(scores),
            'std_score': np.std(scores),
            'shifts': self._detect_shifts()
        }
        
        return summary
    
    def render(self):
        """Render the Streamlit interface.
        
        This is the main rendering method that creates the complete UI layout
        with all visualizations and controls.
        """
        # Title
        st.title("🎭 Real-Time Sentiment Analysis")
        st.markdown("---")
        
        # Sidebar controls
        with st.sidebar:
            st.header("Controls")
            
            # History duration slider
            history_duration = st.slider(
                "History Duration (seconds)",
                min_value=10,
                max_value=300,
                value=60,
                step=10
            )
            self.show_history(history_duration)
            
            # Session info
            st.markdown("### Session Info")
            session_duration = time.time() - self.session_start_time
            st.metric("Session Duration", f"{session_duration:.0f}s")
            st.metric("Data Points", len(self.score_history))
            
            # Summary button
            if st.button("Generate Summary"):
                summary = self.generate_summary()
                st.markdown("### Summary")
                st.metric("Average Score", f"{summary['avg_score']:.3f}")
                st.metric("Score Range", f"{summary['min_score']:.3f} to {summary['max_score']:.3f}")
                st.metric("Volatility (Std)", f"{summary['std_score']:.3f}")
                st.metric("Significant Shifts", len(summary['shifts']))
        
        # Get latest score
        if self.fusion_engine:
            latest_score = self.fusion_engine.get_latest_score()
        else:
            latest_score = None
        
        if latest_score:
            # Update history
            self.update_score(latest_score)
            
            # Main content - 2 columns
            col1, col2 = st.columns([1, 1])
            
            with col1:
                # Gauge chart
                gauge_fig = self._create_gauge_chart(latest_score.score, latest_score.confidence)
                st.plotly_chart(gauge_fig, use_container_width=True)
                
                # Current values
                st.markdown("### Current Sentiment")
                sentiment_label = self._get_sentiment_label(latest_score.score)
                sentiment_color = self._get_sentiment_color(latest_score.score)
                st.markdown(f"**Status:** :{sentiment_color}[{sentiment_label}]")
                st.metric("Score", f"{latest_score.score:.3f}")
                st.metric("Confidence", f"{latest_score.confidence:.2%}")
                
                # Timestamp
                timestamp_str = datetime.fromtimestamp(latest_score.timestamp).strftime("%H:%M:%S")
                st.caption(f"Last updated: {timestamp_str}")
            
            with col2:
                # Emotion breakdown
                emotion_fig = self._create_emotion_chart(latest_score.emotion_breakdown)
                st.plotly_chart(emotion_fig, use_container_width=True)
            
            # Full width charts
            st.markdown("---")
            
            # History chart
            history_fig = self._create_history_chart()
            st.plotly_chart(history_fig, use_container_width=True)
            
            # Modality contributions
            modality_fig = self._create_modality_chart(latest_score.modality_contributions)
            st.plotly_chart(modality_fig, use_container_width=True)
            
            # Detect and highlight shifts
            shifts = self._detect_shifts()
            if shifts:
                st.markdown("### 🔔 Significant Emotional Shifts")
                for shift_time, old_score, new_score in shifts[-5:]:  # Show last 5
                    relative_time = shift_time - self.session_start_time
                    change = new_score - old_score
                    direction = "↑" if change > 0 else "↓"
                    st.info(f"**{relative_time:.1f}s**: {direction} Shift from {old_score:.3f} to {new_score:.3f} (Δ{abs(change):.3f})")
        
        else:
            # No data yet
            st.info("⏳ Waiting for sentiment data...")
            st.markdown("The system is initializing. Sentiment scores will appear shortly.")


def main():
    """Main entry point for Streamlit app."""
    display = SentimentDisplay()
    
    # Placeholder for fusion engine integration
    st.warning("⚠️ This is a standalone display. Connect to FusionEngine for live data.")
    
    # Render interface
    display.render()


if __name__ == "__main__":
    main()
