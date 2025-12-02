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
    
    try:
        model = Wav2Vec2ForSequenceClassification.from_pretrained(model_name)
        processor = Wav2Vec2Processor.from_pretrained(model_name)
        
        output_dir = Path("models/acoustic/wav2vec2_emotion")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        model.save_pretrained(output_dir)
        processor.save_pretrained(output_dir)
        print(f"✓ Acoustic models saved to {output_dir}")
    except Exception as e:
        print(f"✗ Failed to download acoustic models: {e}")


def download_linguistic_models():
    """Download linguistic analysis models"""
    print("Downloading linguistic models...")
    
    # Whisper
    try:
        print("  - Whisper (base)...")
        whisper.load_model("base")
        print("  ✓ Whisper model downloaded")
    except Exception as e:
        print(f"  ✗ Failed to download Whisper: {e}")
    
    # DistilBERT
    try:
        print("  - DistilBERT sentiment...")
        model_name = "distilbert-base-uncased-finetuned-sst-2-english"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        
        output_dir = Path("models/linguistic/distilbert")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        tokenizer.save_pretrained(output_dir)
        model.save_pretrained(output_dir)
        print(f"  ✓ Linguistic models saved to {output_dir}")
    except Exception as e:
        print(f"  ✗ Failed to download DistilBERT: {e}")


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
    print()
    download_linguistic_models()
    print()
    download_visual_models()
    
    print("\n" + "=" * 50)
    print("Model download complete!")
    print("\nNote: Some models will auto-download on first use.")
