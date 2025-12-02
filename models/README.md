# Pre-trained Models

This directory contains pre-trained models for the SentimentEngine.

## Directory Structure

```
models/
├── acoustic/
│   └── wav2vec2_emotion/          # Acoustic emotion recognition
├── visual/
│   └── fer2013_cnn.pt             # Facial expression recognition
└── linguistic/
    ├── whisper/                    # Whisper models (auto-downloaded)
    └── distilbert/                 # Sentiment analysis (auto-downloaded)
```

## Model Acquisition

See `.kiro/steering/model-acquisition.md` for detailed instructions on downloading and configuring models.

### Quick Start

Run the model download script:

```bash
python scripts/download_models.py
```

This will automatically download:
- Acoustic emotion recognition models (wav2vec2)
- Linguistic analysis models (Whisper, DistilBERT)
- Visual models (MediaPipe auto-downloads on first use)

### Manual Download

For the FER2013 CNN model, download manually from:
https://github.com/WuJie1010/Facial-Expression-Recognition.Pytorch

Place in `models/visual/fer2013_cnn.pt`

## Model Sizes

- Acoustic (wav2vec2): ~300MB
- Visual (FER2013 CNN): ~100MB
- Linguistic (Whisper base): ~140MB
- Linguistic (DistilBERT): ~250MB
- **Total**: ~800MB

## GPU Support

For GPU acceleration, ensure PyTorch is installed with CUDA support:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
