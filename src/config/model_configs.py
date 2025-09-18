"""
ML Model Configurations for SpeakSmart AI
Author: Unnati Lohana (ML Models & AI Integration)

Centralized configuration for all machine learning models
Includes model paths, parameters, and fallback options
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import torch

@dataclass
class ModelConfig:
    """Base configuration class for ML models"""
    name: str
    model_path: str
    use_cache: bool = True
    device: str = "auto"  # auto, cpu, cuda
    precision: str = "float32"  # float32, float16, int8
    max_memory_gb: float = 4.0
    
    def get_device(self) -> str:
        """Get appropriate device for model"""
        if self.device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.device

@dataclass
class AudioModelConfig(ModelConfig):
    """Configuration for audio processing models"""
    sample_rate: int = 16000
    chunk_length: float = 30.0  # seconds
    overlap: float = 0.0
    normalize: bool = True
    
@dataclass
class VisionModelConfig(ModelConfig):
    """Configuration for computer vision models"""
    input_size: Tuple[int, int] = (224, 224)
    batch_size: int = 1
    confidence_threshold: float = 0.5
    
@dataclass
class TextModelConfig(ModelConfig):
    """Configuration for text processing models"""
    max_length: int = 512
    truncation: bool = True
    padding: str = "max_length"

class ModelRegistry:
    """Registry of all available models with configurations"""
    
    def __init__(self):
        self.audio_models = self._setup_audio_models()
        self.vision_models = self._setup_vision_models()
        self.text_models = self._setup_text_models()
        self.multimodal_models = self._setup_multimodal_models()
    
    def _setup_audio_models(self) -> Dict[str, AudioModelConfig]:
        """Configure audio processing models"""
        return {
            # Speech Emotion Recognition
            "speech_emotion_primary": AudioModelConfig(
                name="wav2vec2-emotion",
                model_path="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition",
                sample_rate=16000,
                chunk_length=10.0,
                device="auto",
                max_memory_gb=2.0
            ),
            
            "speech_emotion_backup": AudioModelConfig(
                name="hubert-emotion",
                model_path="superb/hubert-large-superb-er",
                sample_rate=16000,
                chunk_length=15.0,
                device="cpu",  # Fallback to CPU
                max_memory_gb=1.0
            ),
            
            # Speaker Verification
            "speaker_verification": AudioModelConfig(
                name="wavlm-speaker",
                model_path="microsoft/wavlm-base-plus-sv",
                sample_rate=16000,
                chunk_length=5.0,
                device="auto",
                max_memory_gb=1.5
            ),
            
            # Voice Activity Detection
            "voice_activity": AudioModelConfig(
                name="silero-vad",
                model_path="silero/silero-vad",
                sample_rate=16000,
                chunk_length=1.0,
                device="cpu",  # VAD is fast on CPU
                max_memory_gb=0.5
            ),
            
            # Speech Enhancement
            "speech_enhancement": AudioModelConfig(
                name="speechbrain-enhance",
                model_path="speechbrain/metricgan-plus-voicebank",
                sample_rate=16000,
                chunk_length=8.0,
                device="auto",
                max_memory_gb=1.0
            )
        }
    
    def _setup_vision_models(self) -> Dict[str, VisionModelConfig]:
        """Configure computer vision models"""
        return {
            # Facial Emotion Recognition
            "facial_emotion_primary": VisionModelConfig(
                name="emotion-ferplus",
                model_path="microsoft/FER-2013",  # Hypothetical - replace with actual
                input_size=(48, 48),
                confidence_threshold=0.6,
                device="auto",
                max_memory_gb=1.0
            ),
            
            "facial_emotion_backup": VisionModelConfig(
                name="resnet-emotion",
                model_path="microsoft/resnet-50",
                input_size=(224, 224),
                confidence_threshold=0.5,
                device="cpu",
                max_memory_gb=0.8
            ),
            
            # Face Detection
            "face_detection": VisionModelConfig(
                name="mtcnn",
                model_path="mtcnn",  # Uses facenet-pytorch
                input_size=(160, 160),
                confidence_threshold=0.9,
                device="auto",
                max_memory_gb=0.5
            ),
            
            # Gaze Estimation
            "gaze_estimation": VisionModelConfig(
                name="gaze360",
                model_path="eth-ait/gaze360",
                input_size=(224, 224),
                confidence_threshold=0.7,
                device="auto",
                max_memory_gb=1.2
            ),
            
            # Age/Gender Detection
            "age_gender": VisionModelConfig(
                name="fairface",
                model_path="dlib/fairface",
                input_size=(224, 224),
                confidence_threshold=0.8,
                device="auto",
                max_memory_gb=0.8
            )
        }
    
    def _setup_text_models(self) -> Dict[str, TextModelConfig]:
        """Configure text processing models"""
        return {
            # Sentiment Analysis
            "sentiment_primary": TextModelConfig(
                name="roberta-sentiment",
                model_path="cardiffnlp/twitter-roberta-base-sentiment-latest",
                max_length=128,
                device="auto",
                max_memory_gb=1.0
            ),
            
            # Confidence Classification
            "confidence_classifier": TextModelConfig(
                name="bert-confidence",
                model_path="nlptown/bert-base-multilingual-uncased-sentiment",
                max_length=256,
                device="auto",
                max_memory_gb=1.2
            ),
            
            # Text Summarization
            "summarization": TextModelConfig(
                name="bart-summarize",
                model_path="facebook/bart-large-cnn",
                max_length=1024,
                device="auto",
                max_memory_gb=2.0
            ),
            
            # Language Detection
            "language_detection": TextModelConfig(
                name="xlm-lang-detect",
                model_path="papluca/xlm-roberta-base-language-detection",
                max_length=512,
                device="cpu",  # Fast on CPU
                max_memory_gb=0.5
            )
        }
    
    def _setup_multimodal_models(self) -> Dict[str, ModelConfig]:
        """Configure multimodal models"""
        return {
            # Audio-Visual Emotion
            "multimodal_emotion": ModelConfig(
                name="audio-visual-emotion",
                model_path="custom/audio-visual-emotion",  # Custom trained
                device="auto",
                max_memory_gb=3.0
            ),
            
            # Presentation Analysis
            "presentation_analyzer": ModelConfig(
                name="presentation-scorer",
                model_path="custom/presentation-analyzer",
                device="auto",
                max_memory_gb=2.5
            )
        }

# Model Performance Configurations
MODEL_PERFORMANCE_CONFIGS = {
    "high_performance": {
        "batch_size_multiplier": 2.0,
        "use_mixed_precision": True,
        "enable_optimization": True,
        "memory_efficient": False
    },
    
    "balanced": {
        "batch_size_multiplier": 1.0,
        "use_mixed_precision": True,
        "enable_optimization": True,
        "memory_efficient": True
    },
    
    "memory_efficient": {
        "batch_size_multiplier": 0.5,
        "use_mixed_precision": True,
        "enable_optimization": True,
        "memory_efficient": True
    },
    
    "cpu_optimized": {
        "batch_size_multiplier": 0.3,
        "use_mixed_precision": False,
        "enable_optimization": False,
        "memory_efficient": True
    }
}

# Fallback Model Hierarchy
FALLBACK_HIERARCHY = {
    "speech_emotion": [
        "speech_emotion_primary",
        "speech_emotion_backup",
        "heuristic_audio_analysis"
    ],
    
    "facial_emotion": [
        "facial_emotion_primary", 
        "facial_emotion_backup",
        "mediapipe_basic_emotion"
    ],
    
    "sentiment": [
        "sentiment_primary",
        "confidence_classifier",
        "rule_based_sentiment"
    ]
}

# Model Download URLs and Checksums
MODEL_SOURCES = {
    "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition": {
        "url": "https://huggingface.co/ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition",
        "size_mb": 1200,
        "checksum": "sha256:abc123...",  # Would be actual checksum
        "license": "MIT"
    },
    
    "microsoft/wavlm-base-plus-sv": {
        "url": "https://huggingface.co/microsoft/wavlm-base-plus-sv",
        "size_mb": 400,
        "checksum": "sha256:def456...",
        "license": "MIT"
    },
    
    "cardiffnlp/twitter-roberta-base-sentiment-latest": {
        "url": "https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest",
        "size_mb": 500,
        "checksum": "sha256:ghi789...",
        "license": "MIT"
    }
}

# Model Preprocessing Configurations
PREPROCESSING_CONFIGS = {
    "audio": {
        "normalize_volume": True,
        "remove_silence": True,
        "noise_reduction": True,
        "resample_quality": "high",
        "window_function": "hann",
        "overlap_ratio": 0.25
    },
    
    "video": {
        "face_detection_model": "mediapipe",
        "face_alignment": True,
        "illumination_normalization": True,
        "resize_method": "bilinear",
        "color_space": "RGB",
        "frame_skip": 3
    },
    
    "text": {
        "lowercase": True,
        "remove_punctuation": False,
        "remove_stopwords": False,
        "lemmatization": False,
        "encoding": "utf-8"
    }
}

# Model Evaluation Metrics
EVALUATION_METRICS = {
    "classification": ["accuracy", "precision", "recall", "f1_score", "auc_roc"],
    "regression": ["mse", "mae", "r2_score", "rmse"],
    "multiclass": ["weighted_f1", "macro_f1", "confusion_matrix"],
    "multilabel": ["hamming_loss", "jaccard_score", "label_ranking_loss"]
}

class ModelConfigManager:
    """Manager for model configurations and selection"""
    
    def __init__(self):
        self.registry = ModelRegistry()
        self.current_performance_mode = "balanced"
        self.cache_dir = Path("./data/models")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def get_model_config(self, model_type: str, model_name: str) -> Optional[ModelConfig]:
        """Get configuration for specific model"""
        model_dict = getattr(self.registry, f"{model_type}_models", {})
        return model_dict.get(model_name)
    
    def get_best_available_model(self, model_category: str) -> Optional[ModelConfig]:
        """Get best available model for category based on system capabilities"""
        fallback_list = FALLBACK_HIERARCHY.get(model_category, [])
        
        for model_name in fallback_list:
            # Try each model type
            for model_type in ["audio", "vision", "text", "multimodal"]:
                config = self.get_model_config(model_type, model_name)
                if config and self._is_model_available(config):
                    return config
        
        return None
    
    def _is_model_available(self, config: ModelConfig) -> bool:
        """Check if model is available and can run on current system"""
        # Check memory requirements
        if hasattr(torch.cuda, 'get_device_properties'):
            if config.device == "cuda" and torch.cuda.is_available():
                gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                if gpu_memory_gb < config.max_memory_gb:
                    return False
        
        # Check if model files exist in cache
        model_cache_path = self.cache_dir / config.name
        if config.use_cache and model_cache_path.exists():
            return True
        
        # For HuggingFace models, they'll be downloaded automatically
        if config.model_path.count("/") == 1:  # HF format: org/model
            return True
        
        return False
    
    def set_performance_mode(self, mode: str):
        """Set performance optimization mode"""
        if mode in MODEL_PERFORMANCE_CONFIGS:
            self.current_performance_mode = mode
        else:
            raise ValueError(f"Unknown performance mode: {mode}")
    
    def get_performance_config(self) -> Dict[str, Any]:
        """Get current performance configuration"""
        return MODEL_PERFORMANCE_CONFIGS[self.current_performance_mode]
    
    def estimate_memory_usage(self, models: List[str]) -> float:
        """Estimate total memory usage for list of models"""
        total_memory = 0
        
        for model_identifier in models:
            # Parse model identifier (type:name format)
            if ":" in model_identifier:
                model_type, model_name = model_identifier.split(":", 1)
                config = self.get_model_config(model_type, model_name)
                if config:
                    total_memory += config.max_memory_gb
        
        return total_memory
    
    def get_recommended_models(self, use_case: str) -> List[str]:
        """Get recommended models for specific use case"""
        recommendations = {
            "presentation_analysis": [
                "audio:speech_emotion_primary",
                "vision:facial_emotion_primary",
                "text:sentiment_primary"
            ],
            
            "interview_practice": [
                "audio:speech_emotion_primary",
                "vision:gaze_estimation",
                "text:confidence_classifier"
            ],
            
            "public_speaking": [
                "audio:voice_activity",
                "vision:facial_emotion_primary",
                "multimodal:presentation_analyzer"
            ],
            
            "basic_analysis": [
                "audio:speech_emotion_backup",
                "vision:facial_emotion_backup",
                "text:sentiment_primary"
            ]
        }
        
        return recommendations.get(use_case, recommendations["basic_analysis"])
    
    def validate_model_compatibility(self, models: List[str]) -> Dict[str, bool]:
        """Validate if models are compatible with current system"""
        results = {}
        
        for model_identifier in models:
            if ":" in model_identifier:
                model_type, model_name = model_identifier.split(":", 1)
                config = self.get_model_config(model_type, model_name)
                results[model_identifier] = config is not None and self._is_model_available(config)
            else:
                results[model_identifier] = False
        
        return results

# Global instance for easy access
model_config_manager = ModelConfigManager()

# Convenience functions
def get_audio_model_config(model_name: str) -> Optional[AudioModelConfig]:
    """Get audio model configuration"""
    return model_config_manager.get_model_config("audio", model_name)

def get_vision_model_config(model_name: str) -> Optional[VisionModelConfig]:
    """Get vision model configuration"""
    return model_config_manager.get_model_config("vision", model_name)

def get_text_model_config(model_name: str) -> Optional[TextModelConfig]:
    """Get text model configuration"""
    return model_config_manager.get_model_config("text", model_name)

def get_recommended_setup(performance_level: str = "balanced") -> Dict[str, str]:
    """Get recommended model setup for performance level"""
    setups = {
        "high_performance": {
            "speech_emotion": "audio:speech_emotion_primary",
            "facial_emotion": "vision:facial_emotion_primary", 
            "sentiment": "text:sentiment_primary",
            "gaze": "vision:gaze_estimation"
        },
        
        "balanced": {
            "speech_emotion": "audio:speech_emotion_primary",
            "facial_emotion": "vision:facial_emotion_backup",
            "sentiment": "text:sentiment_primary"
        },
        
        "memory_efficient": {
            "speech_emotion": "audio:speech_emotion_backup",
            "facial_emotion": "vision:facial_emotion_backup",
            "sentiment": "text:confidence_classifier"
        }
    }
    
    return setups.get(performance_level, setups["balanced"])

if __name__ == "__main__":
    # Test model configurations
    manager = ModelConfigManager()
    
    print("Available Audio Models:")
    for name in manager.registry.audio_models.keys():
        print(f"  - {name}")
    
    print("\nRecommended setup for balanced performance:")
    setup = get_recommended_setup("balanced")
    for category, model in setup.items():
        print(f"  {category}: {model}")
    
    print(f"\nEstimated memory usage: {manager.estimate_memory_usage(list(setup.values())):.1f} GB")