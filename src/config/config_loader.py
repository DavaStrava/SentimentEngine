"""Configuration loader for SentimentEngine"""

import yaml
from pathlib import Path
from typing import Any, Dict
import os


class Config:
    """Configuration manager for SentimentEngine"""
    
    def __init__(self, config_path: str = None):
        if config_path is None:
            env = os.getenv('SENTIMENT_ENV', 'development')
            # Try environment-specific config first, fall back to default
            env_config = Path(f"config/config.{env}.yaml")
            if env_config.exists():
                config_path = str(env_config)
            else:
                config_path = "config/config.yaml"
        
        self.config_path = Path(config_path)
        self._config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value using dot notation
        
        Args:
            key: Configuration key in dot notation (e.g., 'fusion.timer_interval')
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
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
    
    def validate(self) -> None:
        """Validate configuration values"""
        # Check value ranges
        alpha = self.get('fusion.smoothing_alpha')
        if alpha is not None and not 0 <= alpha <= 1:
            raise ValueError(f"Invalid smoothing_alpha: {alpha}, must be in [0, 1]")
        
        # Check Redis connection
        redis_url = self.get('redis.url')
        if not redis_url:
            raise ValueError("Redis URL not configured")


# Global config instance
config = Config()
