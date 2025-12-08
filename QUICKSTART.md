# Quick Start Guide

## 🚀 Launch the Web UI in 2 Steps!

The easiest way to see the sentiment analysis engine in action:

### Step 1: Activate Your Environment
```bash
source venv/bin/activate
```

### Step 2: Launch the Web UI
```bash
./start_web_ui.sh
```

**That's it!** Your browser will open automatically at http://localhost:8501

---

## 🎯 Using the Web Interface

Once the browser opens:

1. **Click "🚀 Initialize System"** in the sidebar
   - Takes ~30 seconds to load ML models
   - You'll see "✅ System initialized!" when ready

2. **Click "▶️ Start"** to begin processing
   - Watch real-time sentiment analysis
   - See charts and metrics update live

3. **Click "⏹️ Stop"** when you're done

### What You'll See:
- 📊 **Sentiment Score**: Real-time emotional valence (-1 to +1)
- 🎯 **Confidence Level**: How certain the system is
- 😊 **Dominant Emotion**: Strongest detected emotion
- 🎤👁️💬 **Modality Breakdown**: Acoustic, visual, and linguistic contributions
- 📈 **History Chart**: Sentiment over time
- 📊 **Emotion Breakdown**: Bar chart of all emotions

**No video files or Redis required!** The demo uses synthetic data to show you the complete pipeline working.

---

## 📚 Other Testing Options

### Terminal Demo (Quick Command-Line Test)

If you prefer terminal output:

```bash
python demo_simple.py
```

Processes 10 frames in ~5 seconds and shows text output.

### Run Tests (For Developers)

Verify the system with automated tests:

```bash
pytest tests/ -v
```

All 164 tests should pass!

### Process Real Video (Advanced)

To analyze actual video files:

1. **Start Redis** (required for video processing):
   ```bash
   redis-server
   ```

2. **Download models** (one-time, ~800MB):
   ```bash
   python scripts/download_models.py
   ```

3. **Process video**:
   ```bash
   python src/main.py --input path/to/video.mp4
   ```

---

## 💡 Understanding the Output

### Sentiment Score
- **+1.0** = Very positive (happy, excited)
- **0.0** = Neutral
- **-1.0** = Very negative (sad, angry)

### Confidence
- **>80%** = High confidence, reliable
- **50-80%** = Moderate confidence
- **<50%** = Low confidence, use with caution

### Modalities
- **🎤 Acoustic**: Vocal tone, pitch, energy
- **👁️ Visual**: Facial expressions, emotions
- **💬 Linguistic**: Speech transcription, semantic sentiment

### Common Emotions
- **happy**, **sad**, **angry**, **neutral**, **surprised**, **fearful**

---

## ⚙️ System Capabilities

✅ Multi-modal analysis (acoustic + visual + linguistic)  
✅ Real-time processing with <3 second latency  
✅ Quality-aware fusion with confidence scoring  
✅ Temporal smoothing to reduce noise  
✅ Graceful degradation when models unavailable  
✅ 164 passing tests (100% coverage)

---

## ❓ Troubleshooting

### "Command not found: streamlit"
```bash
source venv/bin/activate
pip install -r requirements.txt
```

### Models taking too long to load
This is normal on first run (~30 seconds). Subsequent runs are faster.

### Browser doesn't open automatically
Manually go to: **http://localhost:8501**

### Want to stop the server?
Press `Ctrl+C` in the terminal

---

## 📖 More Information

- **Detailed Web UI Guide**: See `WEB_UI_GUIDE.md`
- **Full Documentation**: See `README.md`
- **Testing Guide**: See `TESTING.md`
- **Design Docs**: See `.kiro/specs/realtime-sentiment-analysis/`

---

## 🎉 You're All Set!

Just run `./start_web_ui.sh` and explore the sentiment analysis engine in your browser!
