"""
Emotion Models - Handles loading and inference of emotion recognition models
"""

try:
    from transformers import pipeline
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


class EmotionModel:
    def init(self, model_name="j-hartmann/emotion-english-distilroberta-base"):
        self.model_name = model_name
        self.model = None
        self.available = TRANSFORMERS_AVAILABLE

    def load(self):
        """Load the emotion recognition model if available"""
        if not self.available:
            print("Transformers not installed, using fallback emotion model")
            return None

        try:
            self.model = pipeline(
                "image-classification",
                model=self.model_name,
                device=0 if torch.cuda.is_available() else -1
            )
            print(f" Emotion model '{self.model_name}' loaded successfully")
        except Exception as e:
            print(f" Error loading emotion model: {e}")
            self.model = None

        return self.model

    def predict(self, image):
        """Run prediction on an image"""
        if self.model is None:
            return {"neutral": 1.0}
        results = self.model(image)
        return {r["label"]: r["score"] for r in results}
