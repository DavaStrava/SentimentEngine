# Web UI Quick Start Guide

## 🚀 Launch the Web Interface

The easiest way to see the sentiment analysis system in action!

### Method 1: Using the Launch Script (Easiest)

```bash
./start_web_ui.sh
```

### Method 2: Direct Command

```bash
source venv/bin/activate
streamlit run run_web_ui.py
```

## 📖 How to Use

1. **Browser Opens Automatically**
   - The UI will open at http://localhost:8501
   - If it doesn't, manually navigate to that URL

2. **Initialize the System**
   - Click the "🚀 Initialize System" button in the sidebar
   - Wait ~30 seconds while models load
   - You'll see "✅ System initialized!" when ready

3. **Start Processing**
   - Click "▶️ Start" to begin analysis
   - The system will process synthetic demo frames
   - Watch the sentiment scores update in real-time!

4. **What You'll See**
   - **Sentiment Score**: Current emotional valence (-1 to +1)
   - **Confidence**: How confident the system is
   - **Dominant Emotion**: The strongest detected emotion
   - **Modality Contributions**: How much each analysis module contributes
   - **Sentiment History**: Chart showing scores over time
   - **Emotion Breakdown**: Bar chart of all detected emotions

5. **Stop Processing**
   - Click "⏹️ Stop" when you're done
   - Click "▶️ Start" again to restart

## 🎯 What's Happening Behind the Scenes

The web UI demonstrates the complete sentiment analysis pipeline:

1. **Acoustic Analysis** 🎤
   - Extracts vocal tone features (pitch, energy, speaking rate)
   - Classifies emotions from audio signals
   - Reports confidence based on audio quality

2. **Visual Analysis** 👁️
   - Detects faces in video frames
   - Analyzes facial expressions
   - Classifies emotions from visual cues

3. **Linguistic Analysis** 💬
   - Transcribes speech to text
   - Analyzes semantic sentiment
   - Applies domain-specific interpretation

4. **Fusion Engine** 🔄
   - Combines all three modalities
   - Applies quality-aware weighting
   - Resolves conflicts between modalities
   - Applies temporal smoothing

## 💡 Tips

- **First Time**: Initialization takes ~30 seconds to load ML models
- **Demo Mode**: This version uses synthetic data to show the system working
- **Real Video**: To process actual video files, see the main README
- **Performance**: Processing updates every 0.5 seconds
- **History**: The chart shows the last 60 seconds of data

## 🛑 Stopping the Server

Press `Ctrl+C` in the terminal to stop the Streamlit server.

## ❓ Troubleshooting

### "Command not found: streamlit"
Make sure you've activated the virtual environment:
```bash
source venv/bin/activate
```

### "Module not found" errors
Install dependencies:
```bash
pip install -r requirements.txt
```

### Models taking too long to load
This is normal on first run. Subsequent runs will be faster as models are cached.

### Browser doesn't open automatically
Manually navigate to: http://localhost:8501

## 🎓 Understanding the Output

### Sentiment Score
- **+1.0**: Very positive emotion (happy, excited)
- **0.0**: Neutral emotion
- **-1.0**: Very negative emotion (sad, angry)

### Confidence
- **>80%**: High confidence, reliable result
- **50-80%**: Moderate confidence
- **<50%**: Low confidence, use with caution

### Modality Contributions
Shows how much each analysis module contributed to the final score:
- Should roughly sum to 100%
- Higher contribution = more influence on final score
- Varies based on signal quality

### Emotions
Common emotions detected:
- **happy**: Positive, joyful
- **sad**: Negative, sorrowful
- **angry**: Negative, frustrated
- **neutral**: No strong emotion
- **surprised**: Sudden reaction
- **fearful**: Anxious, worried

## 🔗 Next Steps

Once you've seen the demo:
1. Try processing real video files (see main README)
2. Adjust fusion weights in `config/config.yaml`
3. Run the full test suite: `pytest tests/`
4. Check out the design document: `.kiro/specs/realtime-sentiment-analysis/design.md`

---

**Enjoy exploring the Real-Time Sentiment Analysis Engine!** 🎭
