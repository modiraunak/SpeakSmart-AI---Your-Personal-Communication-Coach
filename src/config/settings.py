"""
Configuration Manager for SpeakSmart AI
Handles all application settings and configurations

Authors: Team SpeakSmart (Group 256)
- Unnati Lohana (ML Models & AI Integration)
- Vedant Singh (Backend & Data Management)
"""

import os
import json
from dataclasses import dataclass
from typing import Dict, Any, Optional
from pathlib import Path

@dataclass
class AudioConfig:
    """Audio analysis configuration"""
    sample_rate: int = 44100
    chunk_duration: float = 0.05  # 50ms chunks
    min_pitch: int = 75
    max_pitch: int = 400
    silence_threshold: float = 0.003
    analysis_window: int = 8  # seconds
    pitch_variation_threshold: float = 0.25
    confidence_threshold: int = 50
    noise_reduction: bool = True
    
    # Advanced settings
    hop_length: int = 512
    n_mfcc: int = 13
    n_fft: int = 2048
    window_type: str = "hann"

@dataclass
class VideoConfig:
    """Video analysis configuration"""
    min_detection_confidence: float = 0.7
    min_tracking_confidence: float = 0.5
    model_complexity: int = 1
    max_num_faces: int = 1
    max_num_hands: int = 2
    refine_landmarks: bool = True
    
    # Processing settings
    frame_skip: int = 10  # Analyze every 10th frame
    resize_width: int = 640
    resize_height: int = 480
    
    # Feature extraction
    enable_emotion_detection: bool = True
    enable_gesture_analysis: bool = True
    enable_pose_estimation: bool = True
    enable_eye_contact_analysis: bool = True

@dataclass
class ModelConfig:
    """Machine learning model configuration"""
    # Emotion detection
    emotion_model_name: str = "j-hartmann/emotion-english-distilroberta-base"
    emotion_confidence_threshold: float = 0.6
    
    # Speech recognition (if needed)
    speech_model_name: str = "openai/whisper-base"
    
    # Device settings
    use_gpu: bool = True
    device: str = "auto"  # auto, cpu, cuda
    
    # Model paths
    model_cache_dir: str = "./models"
    
    # Performance settings
    batch_size: int = 16
    max_sequence_length: int = 512

@dataclass
class UIConfig:
    """UI configuration settings"""
    theme: str = "light"
    sidebar_state: str = "expanded"
    layout: str = "wide"
    
    # Chart settings
    chart_height: int = 400
    chart_width: int = 800
    color_scheme: Dict[str, str] = None
    
    # Animation settings
    enable_animations: bool = True
    progress_update_interval: float = 0.1
    
    def __post_init__(self):
        if self.color_scheme is None:
            self.color_scheme = {
                'primary': '#667eea',
                'secondary': '#764ba2',
                'success': '#28a745',
                'warning': '#ffc107',
                'danger': '#dc3545',
                'info': '#17a2b8'
            }

class AppConfig:
    """Main application configuration manager"""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize configuration manager"""
        
        self.config_path = config_path or self._get_default_config_path()
        self.config_data = {}
        
        # Initialize default configurations
        self.audio_config = AudioConfig()
        self.video_config = VideoConfig()
        self.model_config = ModelConfig()
        self.ui_config = UIConfig()
        
        # Load custom configuration if exists
        self.load_config()
        
        # Setup directories
        self._setup_directories()
    
    def _get_default_config_path(self) -> str:
        """Get default configuration file path"""
        project_root = Path(__file__).parent.parent.parent
        return str(project_root / "config" / "app_config.json")
    
    def _setup_directories(self):
        """Setup necessary directories"""
        directories = [
            self.model_config.model_cache_dir,
            "logs",
            "exports",
            "temp"
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def load_config(self):
        """Load configuration from file"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    self.config_data = json.load(f)
                
                # Update configurations with loaded data
                self._update_configs_from_data()
                
        except Exception as e:
            print(f"Warning: Could not load config file: {e}")
            print("Using default configuration")
    
    def save_config(self):
        """Save current configuration to file"""
        try:
            # Create config directory if it doesn't exist
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            
            # Prepare config data
            config_data = {
                'audio': self._dataclass_to_dict(self.audio_config),
                'video': self._dataclass_to_dict(self.video_config),
                'model': self._dataclass_to_dict(self.model_config),
                'ui': self._dataclass_to_dict(self.ui_config)
            }
            
            with open(self.config_path, 'w') as f:
                json.dump(config_data, f, indent=2)
                
        except Exception as e:
            print(f"Error saving config: {e}")
    
    def _update_configs_from_data(self):
        """Update configuration objects from loaded data"""
        if 'audio' in self.config_data:
            self._update_dataclass_from_dict(self.audio_config, self.config_data['audio'])
        
        if 'video' in self.config_data:
            self._update_dataclass_from_dict(self.video_config, self.config_data['video'])
        
        if 'model' in self.config_data:
            self._update_dataclass_from_dict(self.model_config, self.config_data['model'])
        
        if 'ui' in self.config_data:
            self._update_dataclass_from_dict(self.ui_config, self.config_data['ui'])
    
    def _dataclass_to_dict(self, obj) -> dict:
        """Convert dataclass to dictionary"""
        result = {}
        for field, value in obj.__dict__.items():
            if isinstance(value, dict):
                result[field] = value
            else:
                result[field] = value
        return result
    
    def _update_dataclass_from_dict(self, obj, data: dict):
        """Update dataclass from dictionary"""
        for key, value in data.items():
            if hasattr(obj, key):
                setattr(obj, key, value)
    
    def get_audio_config(self) -> AudioConfig:
        """Get audio configuration"""
        return self.audio_config
    
    def get_video_config(self) -> VideoConfig:
        """Get video configuration"""
        return self.video_config
    
    def get_model_config(self) -> ModelConfig:
        """Get model configuration"""
        return self.model_config
    
    def get_ui_config(self) -> UIConfig:
        """Get UI configuration"""
        return self.ui_config
    
    def update_audio_config(self, **kwargs):
        """Update audio configuration"""
        for key, value in kwargs.items():
            if hasattr(self.audio_config, key):
                setattr(self.audio_config, key, value)
    
    def update_video_config(self, **kwargs):
        """Update video configuration"""
        for key, value in kwargs.items():
            if hasattr(self.video_config, key):
                setattr(self.video_config, key, value)
    
    def update_model_config(self, **kwargs):
        """Update model configuration"""
        for key, value in kwargs.items():
            if hasattr(self.model_config, key):
                setattr(self.model_config, key, value)
    
    def update_ui_config(self, **kwargs):
        """Update UI configuration"""
        for key, value in kwargs.items():
            if hasattr(self.ui_config, key):
                setattr(self.ui_config, key, value)
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get system information for debugging"""
        import platform
        import sys
        
        try:
            import torch
            cuda_available = torch.cuda.is_available()
            cuda_version = torch.version.cuda if cuda_available else None
        except ImportError:
            cuda_available = False
            cuda_version = None
        
        return {
            'platform': platform.platform(),
            'python_version': sys.version,
            'cuda_available': cuda_available,
            'cuda_version': cuda_version,
            'config_path': self.config_path
        }
    
    def validate_config(self) -> Dict[str, Any]:
        """Validate configuration settings"""
        issues = []
        warnings = []
        
        # Validate audio config
        if self.audio_config.min_pitch >= self.audio_config.max_pitch:
            issues.append("Audio: min_pitch must be less than max_pitch")
        
        if self.audio_config.silence_threshold < 0 or self.audio_config.silence_threshold > 1:
            warnings.append("Audio: silence_threshold should be between 0 and 1")
        
        # Validate video config
        if self.video_config.min_detection_confidence < 0 or self.video_config.min_detection_confidence > 1:
            issues.append("Video: min_detection_confidence must be between 0 and 1")
        
        # Validate model config
        if not os.path.exists(self.model_config.model_cache_dir):
            warnings.append(f"Model cache directory does not exist: {self.model_config.model_cache_dir}")
        
        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'warnings': warnings
        }
    
    def reset_to_defaults(self):
        """Reset all configurations to default values"""
        self.audio_config = AudioConfig()
        self.video_config = VideoConfig()
        self.model_config = ModelConfig()
        self.ui_config = UIConfig()
    
    def export_config(self, filepath: str):
        """Export configuration to a specific file"""
        config_data = {
            'audio': self._dataclass_to_dict(self.audio_config),
            'video': self._dataclass_to_dict(self.video_config),
            'model': self._dataclass_to_dict(self.model_config),
            'ui': self._dataclass_to_dict(self.ui_config),
            'system_info': self.get_system_info(),
            'export_timestamp': str(datetime.now())
        }
        
        with open(filepath, 'w') as f:
            json.dump(config_data, f, indent=2)
    
    def import_config(self, filepath: str):
        """Import configuration from a file"""
        try:
            with open(filepath, 'r') as f:
                imported_data = json.load(f)
            
            self.config_data = imported_data
            self._update_configs_from_data()
            
            return True
        except Exception as e:
            print(f"Error importing config: {e}")
            return False

# Global configuration instance
_config_instance = None

def get_config() -> AppConfig:
    """Get global configuration instance"""
    global _config_instance
    if _config_instance is None:
        _config_instance = AppConfig()
    return _config_instance

def reload_config():
    """Reload configuration from file"""
    global _config_instance
    if _config_instance is not None:
        _config_instance.load_config()

def initialize_config(config_path: Optional[str] = None) -> AppConfig:
    """Initialize configuration with custom path"""
    global _config_instance
    _config_instance = AppConfig(config_path)
    return _config_instance

if __name__ == "__main__":
    # Test configuration
    config = AppConfig()
    
    print("Configuration loaded successfully!")
    print(f"Audio sample rate: {config.audio_config.sample_rate}")
    print(f"Video model complexity: {config.video_config.model_complexity}")
    print(f"Model cache directory: {config.model_config.model_cache_dir}")
    
    # Validate configuration
    validation = config.validate_config()
    print(f"\nConfiguration valid: {validation['valid']}")
    if validation['issues']:
        print("Issues:", validation['issues'])
    if validation['warnings']:
        print("Warnings:", validation['warnings'])