#!/bin/bash
# Start the Sentiment Analysis Web UI

echo "🎭 Starting Real-Time Sentiment Analysis Web UI..."
echo ""
echo "The browser will open automatically at http://localhost:8501"
echo ""
echo "Instructions:"
echo "  1. Click '🚀 Initialize System' (takes ~30 seconds)"
echo "  2. Click '▶️ Start' to begin processing"
echo "  3. Watch real-time sentiment analysis!"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Activate venv and run streamlit
source venv/bin/activate
streamlit run run_web_ui.py
