# Quick Start Guide

## 🎯 You're Ready to Test the POC!

The core sentiment analysis pipeline is now complete. Here's how to test it:

## Prerequisites Check

1. **Redis Server** (required for async frame distribution)
   ```bash
   # Check if Redis is running:
   ps aux | grep redis-server
   
   # If not running, start it:
   redis-server
   ```

2. **Python Dependencies** (should already be installed)
   ```bash
   source venv/bin/activate
   pip install -r requirements.txt
   ```

## Testing Options (Easiest to Most Advanced)

### Option 1: Quick Demo (⚡ Start Here!)

**Best for**: First-time testing, no setup required

Run the standalone demo with synthetic data:

```bash
python demo_simple.py
```

**What you'll see**:
- 10 frames processed in ~5 seconds
- All three modalities working (acoustic, visual, linguistic)
- Fusion engine combining results
- Real-time confidence scores
- Works even without downloaded models!

**Expected output**:
```
Frame 1/10:
  Analyzing audio... ✓ (confidence: 0.100)
  Analyzing video... ✓ (confidence: 0.200, face: True)
  Analyzing speech... ✓ (confidence: 0.200)
  Fusing modalities... ✓
  → Sentiment Score: +0.000
  → Confidence: 0.979
```

### Option 2: Run Automated Tests

**Best for**: Verifying correctness and coverage

```bash
# All tests
pytest tests/ -v

# Just unit tests (fast)
pytest tests/unit/ -v

# Property-based tests (thorough, 100+ iterations each)
pytest tests/property/ -v

# Specific test
pytest tests/property/test_property_01_acoustic_completeness.py -v
```

### Option 3: Verify Setup

**Best for**: Checking dependencies and configuration

```bash
python scripts/verify_setup.py
```

This checks:
- Python version
- All dependencies installed
- Redis connection
- Model files (if downloaded)
- Configuration validity

### Option 4: Process Real Video (Requires Models)

**Best for**: Testing with actual content

First, download models (one-time setup, ~800MB):
```bash
python scripts/download_models.py
```

Then process a video:
```bash
python src/main.py --input path/to/video.mp4
```

### Option 5: Streamlit UI (Currently Has Issues)

**Note**: The Streamlit UI has threading issues with asyncio. Use other options for now.

```bash
# If you want to try it anyway:
./run_streamlit.sh
# or
python run_app.py
```

## What You'll See

### In Streamlit UI:
- **Gauge Chart**: Current sentiment score (-1 to +1)
- **History Chart**: Sentiment over time with confidence bands
- **Modality Contributions**: How much each module contributes
- **Emotion Breakdown**: Pie chart of detected emotions
- **Significant Shifts**: Alerts for major emotional changes

### In Logs:
```
2024-01-15 10:30:15 - Acoustic analysis complete: confidence=0.85
2024-01-15 10:30:15 - Visual analysis complete: confidence=0.72
2024-01-15 10:30:16 - Linguistic analysis complete: 'market rally continues'
2024-01-15 10:30:16 - Fusion complete: score=0.65, confidence=0.78
```

## Test Video Suggestions

For best results, use videos with:
- Clear audio (speech)
- Visible faces
- Emotional content (news, presentations, interviews)
- Duration: 30 seconds to 5 minutes

## Current Capabilities

✅ **Acoustic Analysis**: Extracts vocal tone, pitch, energy, speaking rate
✅ **Visual Analysis**: Detects faces, analyzes facial expressions
✅ **Linguistic Analysis**: Transcribes speech, analyzes sentiment
✅ **Fusion Engine**: Combines all three with quality-aware weighting
✅ **Real-Time Display**: Updates every second with live visualizations

## Known Limitations (MVP)

- Audio-visual sync for multi-face scenarios not yet implemented
- Model loading takes 30-60 seconds on first run
- GPU acceleration recommended for Whisper (linguistic analysis)
- Processing speed depends on video resolution and model size

## Troubleshooting

### Redis Connection Error
```
Error: Connection refused to Redis
```
**Solution**: Start Redis server with `redis-server`

### Model Not Found
```
Error: Model file not found
```
**Solution**: Run `python scripts/download_models.py`

### Slow Processing
**Solution**: 
- Use smaller Whisper model (edit `config/config.yaml`: `whisper_model: "tiny"`)
- Enable GPU if available
- Reduce video resolution

### No Sentiment Scores
**Solution**:
- Check logs: `tail -f logs/sentiment_engine.log`
- Ensure video has audio and visible faces
- Wait 5-10 seconds for models to initialize

## Next Steps

Once you've tested the basic POC, you can:
1. Adjust fusion weights in `config/config.yaml`
2. Try different videos to see how it handles various content
3. Monitor latency and performance
4. Provide feedback on accuracy and usability

## Configuration

Edit `config/config.yaml` to customize:
- Model paths and sizes
- Fusion weights and smoothing
- Processing intervals
- Quality thresholds

## Support

Check logs for detailed error messages:
```bash
tail -f logs/sentiment_engine.log
```

---

**You're all set!** 🚀 Start with the Streamlit UI for the best experience.
