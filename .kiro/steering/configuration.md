# Configuration Management

## Configuration File Location

All configurable parameters must be externalized to `config/config.yaml`:

```yaml
# config/config.yaml

# Stream Input Configuration
stream:
  reconnect_attempts: 3
  reconnect_backoff_base: 2  # Exponential backoff base (seconds)
  buffer_size: 100  # Frame buffer size

# Audio Processing Configuration
audio:
  sample_rate: 16000  # Hz
  frame_duration: 0.5  # seconds
  noise_reduction: true
  
# Acoustic Analysis Configuration
acoustic:
  model_path: "models/wav2vec2_emotion.pt"
  confidence_threshold: 0.05
  speaker_diarization: true

# Visual Analysis Configuration
visual:
  model_path: "models/fer2013_cnn.pt"
  face_detection_confidence: 0.5
  frame_skip: 2  # Process every Nth frame
  max_faces: 5  # Maximum faces to track

# Linguistic Analysis Configuration
linguistic:
  whisper_model: "base"  # Options: tiny, base, small, medium, large
  sentiment_model: "distilbert-base-uncased-finetuned-sst-2-english"
  buffer_duration: 3.0  # seconds
  processing_interval: 2.0  # seconds

# Fusion Configuration
fusion:
  timer_interval: 1.0  # seconds
  baseline_weights:
    acoustic: 0.33
    visual: 0.33
    linguistic: 0.34
  smoothing_alpha: 0.3  # EMA smoothing factor
  conflict_threshold: 0.5  # Threshold for detecting conflicting modalities
  outlier_weight_reduction: 0.5  # Reduce outlier weight by this factor

# Redis Configuration
redis:
  url: "redis://localhost:6379"
  audio_stream: "audio_frames"
  video_stream: "video_frames"
  result_stream: "sentiment_scores"

# UI Configuration
ui:
  update_interval: 0.1  # seconds
  history_duration: 60  # seconds to display
  port: 8501

# Logging Configuration
logging:
  level: "INFO"  # DEBUG, INFO, WARNING, ERROR
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  file: "logs/sentiment_engine.log"

# Performance Configuration
performance:
  use_gpu: true
  max_latency: 3.0  # seconds (warning threshold)
  target_latency: 1.0  # seconds (target)
```

## Loading Configuration

Create a configuration loader module:

```python
# src/config/config_loader.py

import yaml
from pathlib import Path
from typing import Any, Dict

class Config:
    """Configuration manager for SentimentEngine"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config_path = Path(config_path)
        self._config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value using dot notation"""
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default
        
        return value if value is not None else default
    
    def __getitem__(self, key: str) -> Any:
        """Allow dictionary-style access"""
        return self.get(key)

# Global config instance
config = Config()
```

## Using Configuration

Access configuration values throughout the codebase:

```python
from src.config.config_loader import config

class AcousticAnalyzer:
    def __init__(self):
        self.model_path = config.get('acoustic.model_path')
        self.confidence_threshold = config.get('acoustic.confidence_threshold', 0.05)
        self.sample_rate = config.get('audio.sample_rate', 16000)
        
        # Load model
        self.model = self._load_model(self.model_path)
```

## Environment-Specific Configuration

Support different configurations for development, testing, and production:

```python
import os

class Config:
    def __init__(self, config_path: str = None):
        if config_path is None:
            env = os.getenv('SENTIMENT_ENV', 'development')
            config_path = f"config/config.{env}.yaml"
        
        self.config_path = Path(config_path)
        self._config = self._load_config()
```

Configuration files:
- `config/config.development.yaml` - Development settings
- `config/config.testing.yaml` - Test settings (smaller models, faster processing)
- `config/config.production.yaml` - Production settings

## Configuration Validation

Validate configuration at startup:

```python
def validate_config(config: Config) -> None:
    """Validate configuration values"""
    
    # Check required paths exist
    model_paths = [
        config.get('acoustic.model_path'),
        config.get('visual.model_path')
    ]
    
    for path in model_paths:
        if path and not Path(path).exists():
            raise ValueError(f"Model file not found: {path}")
    
    # Check value ranges
    alpha = config.get('fusion.smoothing_alpha')
    if not 0 <= alpha <= 1:
        raise ValueError(f"Invalid smoothing_alpha: {alpha}, must be in [0, 1]")
    
    # Check Redis connection
    redis_url = config.get('redis.url')
    if not redis_url:
        raise ValueError("Redis URL not configured")
```

## Configuration Best Practices

1. **Never hardcode values** - All tunable parameters go in config.yaml
2. **Provide defaults** - Use `config.get(key, default)` for optional values
3. **Validate early** - Check configuration at startup, not during processing
4. **Document parameters** - Add comments in config.yaml explaining each setting
5. **Version control** - Commit config templates, not environment-specific values
6. **Sensitive data** - Use environment variables for secrets (Redis passwords, API keys)

## Overriding Configuration

Allow command-line overrides for testing:

```python
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--config', default='config/config.yaml')
parser.add_argument('--fusion-interval', type=float, help='Override fusion timer interval')
parser.add_argument('--use-gpu', action='store_true', help='Enable GPU acceleration')

args = parser.parse_args()

config = Config(args.config)

# Override specific values
if args.fusion_interval:
    config._config['fusion']['timer_interval'] = args.fusion_interval
if args.use_gpu:
    config._config['performance']['use_gpu'] = True
```
