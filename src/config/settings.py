"""
Configuration Management for SpeakSmart AI
Author: Vedant Singh (Backend & Data Management)

Centralized configuration management with environment support
"""

import os
from dataclasses import dataclass
from typing import Dict, Any, Tuple
from pathlib import Path

@dataclass
class AudioConfig:
    """Audio processing configuration"""
    DEFAULT_SAMPLE_RATE: int = 44100
    FRAME_DURATION: float = 0.05  # 50ms frames
    CONFIDENCE_WINDOW: int = 10   # seconds
    PITCH_RANGE: Tuple[int, int] = (80, 400)
    MIN_SPEAKING_ENERGY: float = 0.01
    
    # Quality settings
    QUALITY_SETTINGS = {
        "Standard": 22050,
        "High": 44100, 
        "Professional": 48000
    }

@dataclass
class VideoConfig:
    """Video processing configuration"""
    MIN_DETECTION_CONFIDENCE: float = 0.7
    MIN_TRACKING_CONFIDENCE: float = 0.5
    MAX_NUM_FACES: int = 1
    MAX_NUM_HANDS: int = 2
    MODEL_COMPLEXITY: int = 1
    
    # Frame processing settings
    FRAME_SKIP: int = 10  # Process every nth frame
    MAX_FRAMES_BUFFER: int = 1000

@dataclass
class ModelConfig:
    """Pre-trained model configuration"""
    # Emotion detection models
    EMOTION_MODEL_NAME: str = "j-hartmann/emotion-english-distilroberta-base"
    FACIAL_EMOTION_MODEL: str = "microsoft/DialoGPT-medium"
    
    # Speech analysis models  
    SPEECH_EMOTION_MODEL: str = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
    SPEAKER_VERIFICATION_MODEL: str = "microsoft/wavlm-base-plus-sv"
    
    # Text analysis models
    SENTIMENT_MODEL: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    CONFIDENCE_CLASSIFIER: str = "unitary/toxic-bert"
    
    # Model paths
    MODEL_CACHE_DIR: str = str(Path.home() / ".cache" / "speaksmart_models")
    USE_GPU: bool = True

class AppConfig:
    """Main application configuration manager"""
    
    def __init__(self, config_path: str = None):
        self.audio = AudioConfig()
        self.video = VideoConfig()
        self.models = ModelConfig()
        
        # Application settings
        self.APP_NAME = "SpeakSmart AI"
        self.VERSION = "2.0.0"
        self.DEBUG = os.getenv("SPEAKSMART_DEBUG", "False").lower() == "true"
        
        # Performance settings
        self.TARGET_CONFIDENCE = 80
        self.MAX_NERVOUSNESS = 20
        self.SESSION_TIMEOUT = 3600  # 1 hour
        
        # File handling
        self.MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB
        self.SUPPORTED_AUDIO_FORMATS = ['wav', 'mp3', 'm4a', 'flac', 'aac']
        self.SUPPORTED_VIDEO_FORMATS = ['mp4', 'avi', 'mov', 'mkv']
        
        # UI Configuration
        self.ui_config = {
            'theme': 'default',
            'sidebar_expanded': True,
            'show_advanced_options': False,
            'auto_refresh': True,
            'chart_colors': {
                'primary': '#667eea',
                'secondary': '#764ba2', 
                'success': '#28a745',
                'warning': '#ffc107',
                'danger': '#dc3545'
            }
        }
        
        # Load custom config if provided
        if config_path:
            self._load_config(config_path)
    
    def _load_config(self, config_path: str):
        """Load configuration from file"""
        try:
            import yaml
            with open(config_path, 'r') as f:
                custom_config = yaml.safe_load(f)
            self._update_config(custom_config)
        except Exception as e:
            print(f"Warning: Could not load config from {config_path}: {e}")
    
    def _update_config(self, custom_config: Dict[str, Any]):
        """Update configuration with custom values"""
        for section, values in custom_config.items():
            if hasattr(self, section):
                section_obj = getattr(self, section)
                if hasattr(section_obj, '__dict__'):
                    for key, value in values.items():
                        if hasattr(section_obj, key):
                            setattr(section_obj, key, value)
    
    def get_model_config(self, model_type: str) -> Dict[str, Any]:
        """Get configuration for specific model type"""
        model_configs = {
            'emotion': {
                'model_name': self.models.EMOTION_MODEL_NAME,
                'cache_dir': self.models.MODEL_CACHE_DIR,
                'use_gpu': self.models.USE_GPU
            },
            'speech_emotion': {
                'model_name': self.models.SPEECH_EMOTION_MODEL,
                'cache_dir': self.models.MODEL_CACHE_DIR,
                'use_gpu': self.models.USE_GPU
            },
            'sentiment': {
                'model_name': self.models.SENTIMENT_MODEL,
                'cache_dir': self.models.MODEL_CACHE_DIR,
                'use_gpu': self.models.USE_GPU
            }
        }
        
        return model_configs.get(model_type, {})
    
    def validate_file(self, file_type: str, file_size: int) -> Tuple[bool, str]:
        """Validate uploaded file"""
        if file_size > self.MAX_FILE_SIZE:
            return False, f"File size exceeds maximum limit of {self.MAX_FILE_SIZE / (1024*1024):.0f}MB"
        
        if file_type == 'audio':
            supported = self.SUPPORTED_AUDIO_FORMATS
        elif file_type == 'video':
            supported = self.SUPPORTED_VIDEO_FORMATS
        else:
            return False, "Unsupported file type"
        
        return True, "File validation passed"

# Global configuration instance
config = AppConfig()