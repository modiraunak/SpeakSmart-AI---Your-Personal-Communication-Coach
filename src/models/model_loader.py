"""
Model Loader - Manages loading and initialization of ML models
"""
from src.models.emotion_models import EmotionModel
from src.models.speech_models import SpeechModel

class ModelLoader:
    def init(self):
        self.models = {}

    def load_all(self):
        """Load all required models"""
        self.models["emotion"] = EmotionModel()
        self.models["emotion"].load()

        self.models["speech"] = SpeechModel()

        print("✅ All models initialized")
        return self.models

    def get(self, model_name):
        """Retrieve a model by name"""
        return self.models.get(model_name, None)