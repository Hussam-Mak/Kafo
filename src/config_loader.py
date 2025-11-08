"""
Configuration management for the document classifier.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Config:
    """Application configuration manager."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize configuration from YAML file and environment variables."""
        self.config_path = Path(config_path)
        self.config: Dict[str, Any] = {}
        self.load_config()
    
    def load_config(self) -> None:
        """Load configuration from YAML file."""
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                self.config = yaml.safe_load(f) or {}
        else:
            self.config = {}
        
        # Override with environment variables
        self.config['gemini'] = self.config.get('gemini', {})
        self.config['gemini']['api_key'] = os.getenv('GEMINI_API_KEY', '')
        
        # Override log level from env
        if os.getenv('LOG_LEVEL'):
            self.config.setdefault('app', {})['log_level'] = os.getenv('LOG_LEVEL')
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value using dot notation (e.g., 'app.log_level')."""
        keys = key.split('.')
        value = self.config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default
            if value is None:
                return default
        return value
    
    def get_gemini_api_key(self) -> str:
        """Get Gemini API key from environment."""
        return os.getenv('GEMINI_API_KEY', '')
    
    def get_path(self, key: str) -> Path:
        """Get a path configuration value as Path object."""
        path_str = self.get(f'paths.{key}', f'./data/{key}')
        return Path(path_str)


# Global configuration instance
config = Config()

