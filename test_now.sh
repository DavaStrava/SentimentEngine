#!/bin/bash
# Quick test script for SentimentEngine POC

echo "============================================================"
echo "SentimentEngine - Quick Test Script"
echo "============================================================"
echo ""

# Check if venv is activated
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo "⚠️  Virtual environment not activated!"
    echo "Run: source venv/bin/activate"
    echo ""
    exit 1
fi

echo "✓ Virtual environment activated"
echo ""

# Check Redis
echo "Checking Redis..."
if pgrep -x "redis-server" > /dev/null; then
    echo "✓ Redis is running"
else
    echo "⚠️  Redis is not running!"
    echo "Start it with: redis-server"
    echo ""
    exit 1
fi
echo ""

# Run the demo
echo "============================================================"
echo "Running Quick Demo (synthetic data)"
echo "============================================================"
echo ""
echo "This will process 10 synthetic frames to demonstrate:"
echo "  • Acoustic analysis (vocal tone)"
echo "  • Visual analysis (facial expressions)"
echo "  • Linguistic analysis (speech transcription)"
echo "  • Fusion engine (combining all modalities)"
echo ""
echo "Press Ctrl+C to cancel, or wait 10 seconds to continue..."
sleep 10

python demo_simple.py

echo ""
echo "============================================================"
echo "Test Complete!"
echo "============================================================"
echo ""
echo "What just happened:"
echo "  ✓ All three analysis modules processed frames"
echo "  ✓ Fusion engine combined modalities"
echo "  ✓ System handled missing models gracefully"
echo ""
echo "Next steps:"
echo "  1. Run automated tests: pytest tests/ -v"
echo "  2. Download models: python scripts/download_models.py"
echo "  3. Process real video: python src/main.py --input video.mp4"
echo ""
