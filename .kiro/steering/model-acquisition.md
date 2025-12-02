# Model Acquisition Guide

## Overview

The SentimentEngine requires several pre-trained models for acoustic, visual, and linguistic analysis. This guide explains how to obtain and configure these models.

## Model Directory Structure

```
models/
├── acoustic/
│   └── wav2vec2_emotion.pt          # Acoustic emotion recognition
├── visual/
│   └── fer2013_cnn.pt                # Facial expression recognition
├── linguistic/
│   ├── whisper/                      # Whisper models (auto-downloaded)
│   └── distilbert/                   # Sentiment analysis (auto-downloaded)
└── README.md                         # Model documentation
```

## Acoustic Models

### Option 1: Pre-trained wav2vec2 (Recommended)

Use Hugging Face transformers to download:

```python
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2Processor

# Download and save model
model_name = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
model = Wav2Vec2ForSequenceClassification.from_pretrained(model_name)
processor = Wav2Vec2Processor.from_pretrained(model_name)

# Save locally
model.save_pretrained("models/acoustic/wav2vec2_emotion")
processor.save_pretrained("models/acoustic/wav2vec2_emotion")
```

### Option 2: Alternative Models

- **RAVDESS-trained models**: Available on Hugging Face
- **IEMOCAP-trained models**: Requires dataset access
- **Custom models**: Train on your own emotion dataset

## Visual Models

### Option 1: FER2013 CNN (Recommended)

Download pre-trained model:

```python
import torch
from torchvision import models

# Option A: Use pre-trained model from repository
# Download from: https://github.com/WuJie1010/Facial-Expression-Recognition.Pytorch
# Place in models/visual/fer2013_cnn.pt

# Option B: Use EfficientNet with emotion classification head
model = models.efficientnet_b0(pretrained=True)
# Fine-tune on FER2013 or use pre-trained emotion classifier
```

### Option 2: MediaPipe Face Mesh

MediaPipe models are automatically downloaded on first use:

```python
import mediapipe as mp

# Models auto-download to ~/.mediapipe/
mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh
```

### Option 3: Alternative Models

- **AffectNet-trained models**: Higher accuracy, larger size
- **EmotiW models**: Competition-winning models
- **Custom CNN**: Train on domain-specific faces

## Linguistic Models

### Whisper (Speech-to-Text)

Whisper models are automatically downloaded by OpenAI's library:

```python
import whisper

# Models auto-download on first use
model = whisper.load_model("base")  # Options: tiny, base, small, medium, large
```

Model sizes:
- `tiny`: 39M params, ~1GB RAM, fastest
- `base`: 74M params, ~1GB RAM, good balance (recommended)
- `small`: 244M params, ~2GB RAM, better accuracy
- `medium`: 769M params, ~5GB RAM, high accuracy
- `large`: 1550M params, ~10GB RAM, best accuracy

### DistilBERT (Sentiment Analysis)

DistilBERT models are automatically downloaded via transformers:

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Auto-download on first use
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
```

### Alternative Sentiment Models

- **RoBERTa**: `cardiffnlp/twitter-roberta-base-sentiment`
- **BERT**: `nlptown/bert-base-multilingual-uncased-sentiment`
- **Emotion-specific**: `j-hartmann/emotion-english-distilroberta-base`

## Model Download Script

Create `scripts/download_models.py`:

```python
#!/usr/bin/env python3
"""Download all required models for SentimentEngine"""

import os
from pathlib import Path
from transformers import (
    Wav2Vec2ForSequenceClassification,
    Wav2Vec2Processor,
    AutoTokenizer,
    AutoModelForSequenceClassification
)
import whisper

def download_acoustic_models():
    """Download acoustic emotion recognition models"""
    print("Downloading acoustic models...")
    model_name = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
    model = Wav2Vec2ForSequenceClassification.from_pretrained(model_name)
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    
    output_dir = Path("models/acoustic/wav2vec2_emotion")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    print(f"✓ Acoustic models saved to {output_dir}")

def download_linguistic_models():
    """Download linguistic analysis models"""
    print("Downloading linguistic models...")
    
    # Whisper
    print("  - Whisper (base)...")
    whisper.load_model("base")
    
    # DistilBERT
    print("  - DistilBERT sentiment...")
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    
    output_dir = Path("models/linguistic/distilbert")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    tokenizer.save_pretrained(output_dir)
    model.save_pretrained(output_dir)
    print(f"✓ Linguistic models saved to {output_dir}")

def download_visual_models():
    """Download visual analysis models"""
    print("Downloading visual models...")
    print("  - MediaPipe models will auto-download on first use")
    print("  - For FER2013 CNN, please download manually from:")
    print("    https://github.com/WuJie1010/Facial-Expression-Recognition.Pytorch")
    print("    and place in models/visual/fer2013_cnn.pt")

if __name__ == "__main__":
    print("SentimentEngine Model Download")
    print("=" * 50)
    
    download_acoustic_models()
    download_linguistic_models()
    download_visual_models()
    
    print("\n" + "=" * 50)
    print("Model download complete!")
    print("\nNote: Some models will auto-download on first use.")
```

Run with:
```bash
python scripts/download_models.py
```

## Configuration

Update `config/config.yaml` with model paths:

```yaml
acoustic:
  model_path: "models/acoustic/wav2vec2_emotion"
  
visual:
  model_path: "models/visual/fer2013_cnn.pt"
  use_mediapipe: true
  
linguistic:
  whisper_model: "base"  # Will auto-load from cache
  sentiment_model: "models/linguistic/distilbert"
```

## Model Caching

Models are cached in:
- **Hugging Face models**: `~/.cache/huggingface/`
- **Whisper models**: `~/.cache/whisper/`
- **MediaPipe models**: `~/.mediapipe/`

To use cached models in Docker, mount these directories or copy to the container.

## GPU Support

For GPU acceleration:

```bash
# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Verify GPU availability
python -c "import torch; print(torch.cuda.is_available())"
```

Update config:
```yaml
performance:
  use_gpu: true
  device: "cuda:0"  # or "cpu" for CPU-only
```

## Model Size Considerations

Total disk space required:
- Acoustic (wav2vec2): ~300MB
- Visual (FER2013 CNN): ~100MB
- Linguistic (Whisper base): ~140MB
- Linguistic (DistilBERT): ~250MB
- **Total**: ~800MB

For production deployments, consider:
- Using smaller model variants (Whisper tiny, smaller CNNs)
- Model quantization to reduce size
- Lazy loading models only when needed

## Troubleshooting

**Issue**: Models fail to download
- Check internet connection
- Verify Hugging Face Hub access
- Try manual download and place in models/ directory

**Issue**: Out of memory errors
- Use smaller model variants
- Reduce batch sizes in config
- Enable CPU offloading for large models

**Issue**: Slow inference
- Enable GPU acceleration
- Use quantized models
- Reduce input resolution/sample rate
