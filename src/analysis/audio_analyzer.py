"""
Advanced Audio Analysis Module for SpeakSmart AI
Author: Raunak Kumar Modi (Team Lead)

Enhanced audio processing with machine learning integration
Using pre-trained models for emotion detection and speech analysis
"""

import numpy as np
import librosa
import soundfile as sf
from typing import Dict, List, Tuple, Optional, Any
from collections import deque
import warnings
warnings.filterwarnings('ignore')

try:
    from transformers import pipeline, Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    import webrtcvad
    VAD_AVAILABLE = True
except ImportError:
    VAD_AVAILABLE = False

from ..models.emotion_models import SpeechEmotionDetector
from ..utils.audio_utils import AudioPreprocessor
from ..utils.logger import setup_logger

logger = setup_logger("audio_analyzer")

class AdvancedAudioAnalyzer:
    """
    Advanced audio analysis system with ML integration
    
    Features:
    - Real-time pitch and energy analysis
    - Speech emotion recognition using pre-trained models
    - Voice activity detection
    - Speaking confidence assessment
    - Nervousness pattern detection
    """
    
    def __init__(self, sample_rate: int = 44100, config: Dict = None):
        self.sr = sample_rate
        self.frame_duration = 0.025  # 25ms frames for better precision
        self.hop_length = int(self.sr * self.frame_duration)
        
        # Initialize components
        self.preprocessor = AudioPreprocessor(sample_rate=self.sr)
        self.emotion_detector = SpeechEmotionDetector()
        
        # Analysis buffers
        self.feature_buffer = deque(maxlen=2000)  # ~50 seconds at 25ms frames
        self.confidence_history = deque(maxlen=200)
        self.emotion_history = deque(maxlen=100)
        
        # Configuration
        self.config = config or self._default_config()
        
        # Initialize pre-trained models
        self._initialize_models()
        
        logger.info("Audio analyzer initialized successfully")
    
    def _default_config(self) -> Dict:
        """Default configuration for audio analysis"""
        return {
            'pitch_range': (75, 400),
            'energy_threshold': 0.01,
            'confidence_window': 10,
            'nervousness_threshold': 0.6,
            'use_pretrained_models': True,
            'model_cache_dir': '.cache/audio_models'
        }
    
    def _initialize_models(self):
        """Initialize pre-trained models for enhanced analysis"""
        self.models = {}
        
        if not TRANSFORMERS_AVAILABLE or not self.config['use_pretrained_models']:
            logger.warning("Pre-trained models not available, using fallback methods")
            return
        
        try:
            # Speech emotion recognition model
            logger.info("Loading speech emotion recognition model...")
            self.models['speech_emotion'] = pipeline(
                "audio-classification",
                model="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition",
                device=0 if torch.cuda.is_available() else -1
            )
            
            # Voice activity detection
            if VAD_AVAILABLE:
                self.vad = webrtcvad.Vad()
                self.vad.set_mode(2)  # Moderate aggressiveness
                logger.info("Voice Activity Detection initialized")
            
            logger.info("Pre-trained models loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize models: {str(e)}")
            self.models = {}
    
    def analyze_audio_stream(self, audio_data: np.ndarray) -> Dict[str, Any]:
        """
        Analyze audio stream and return comprehensive metrics
        
        Args:
            audio_data: Raw audio data as numpy array
            
        Returns:
            Dictionary containing analysis results
        """
        try:
            # Preprocess audio
            processed_audio = self.preprocessor.preprocess(audio_data)
            
            # Extract basic features
            basic_features = self._extract_basic_features(processed_audio)
            
            # Advanced ML-based analysis
            ml_features = self._extract_ml_features(processed_audio)
            
            # Combine features
            combined_features = {**basic_features, **ml_features}
            
            # Update buffers
            self.feature_buffer.append(combined_features)
            
            # Calculate derived metrics
            analysis_results = self._calculate_analysis_metrics()
            
            return analysis_results
            
        except Exception as e:
            logger.error(f"Audio analysis failed: {str(e)}")
            return self._empty_results()
    
    def _extract_basic_features(self, audio_data: np.ndarray) -> Dict[str, float]:
        """Extract traditional audio features"""
        features = {}
        
        # Energy/Volume analysis
        rms_energy = np.sqrt(np.mean(audio_data**2))
        features['energy'] = float(rms_energy)
        features['db_level'] = float(20 * np.log10(rms_energy + 1e-10))
        
        # Pitch analysis using multiple methods for robustness
        try:
            # YIN algorithm for pitch detection
            f0_yin = librosa.yin(audio_data, 
                               fmin=self.config['pitch_range'][0], 
                               fmax=self.config['pitch_range'][1],
                               sr=self.sr)
            
            # Filter valid pitch values
            valid_f0 = f0_yin[f0_yin > 0]
            
            if len(valid_f0) > 0:
                features['pitch'] = float(np.median(valid_f0))
                features['pitch_std'] = float(np.std(valid_f0))
                features['pitch_range'] = float(np.ptp(valid_f0))  # peak-to-peak
                features['pitch_stability'] = float(1 - np.std(valid_f0) / np.mean(valid_f0))
            else:
                features.update({
                    'pitch': 0.0, 'pitch_std': 0.0, 
                    'pitch_range': 0.0, 'pitch_stability': 0.0
                })
                
        except Exception as e:
            logger.warning(f"Pitch extraction failed: {e}")
            features.update({'pitch': 0.0, 'pitch_std': 0.0, 
                           'pitch_range': 0.0, 'pitch_stability': 0.0})
        
        # Spectral features
        try:
            spectral_centroids = librosa.feature.spectral_centroid(y=audio_data, sr=self.sr)
            features['spectral_centroid'] = float(np.mean(spectral_centroids))
            
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=self.sr)
            features['spectral_rolloff'] = float(np.mean(spectral_rolloff))
            
            zero_crossing_rate = librosa.feature.zero_crossing_rate(audio_data)
            features['zero_crossing_rate'] = float(np.mean(zero_crossing_rate))
            
            # MFCC features (first 13 coefficients)
            mfccs = librosa.feature.mfcc(y=audio_data, sr=self.sr, n_mfcc=13)
            for i, mfcc_mean in enumerate(np.mean(mfccs, axis=1)):
                features[f'mfcc_{i}'] = float(mfcc_mean)
                
        except Exception as e:
            logger.warning(f"Spectral feature extraction failed: {e}")
        
        # Voice quality indicators
        features['is_voiced'] = float(features.get('pitch', 0) > 0)
        features['speech_rate'] = self._estimate_speech_rate(audio_data)
        
        return features
    
    def _extract_ml_features(self, audio_data: np.ndarray) -> Dict[str, Any]:
        """Extract features using pre-trained ML models"""
        ml_features = {}
        
        if not self.models:
            return ml_features
        
        try:
            # Speech emotion recognition
            if 'speech_emotion' in self.models:
                emotion_results = self._analyze_speech_emotion(audio_data)
                ml_features['emotions'] = emotion_results
                
                # Extract dominant emotion
                if emotion_results:
                    dominant_emotion = max(emotion_results.items(), key=lambda x: x[1])
                    ml_features['dominant_emotion'] = dominant_emotion[0]
                    ml_features['emotion_confidence'] = dominant_emotion[1]
            
            # Voice activity detection
            if hasattr(self, 'vad'):
                vad_result = self._detect_voice_activity(audio_data)
                ml_features['voice_activity'] = vad_result
                
        except Exception as e:
            logger.warning(f"ML feature extraction failed: {e}")
        
        return ml_features
    
    def _analyze_speech_emotion(self, audio_data: np.ndarray) -> Dict[str, float]:
        """Analyze emotional content of speech using pre-trained model"""
        try:
            # Ensure audio is in correct format for the model
            if len(audio_data) < self.sr * 0.1:  # Minimum 100ms
                return {}
            
            # Resample if needed (most models expect 16kHz)
            if self.sr != 16000:
                audio_16k = librosa.resample(audio_data, orig_sr=self.sr, target_sr=16000)
            else:
                audio_16k = audio_data
            
            # Run emotion classification
            results = self.models['speech_emotion'](audio_16k)
            
            # Convert to standardized format
            emotion_scores = {}
            for result in results:
                emotion_scores[result['label'].lower()] = float(result['score'])
            
            return emotion_scores
            
        except Exception as e:
            logger.warning(f"Speech emotion analysis failed: {e}")
            return {}
    
    def _detect_voice_activity(self, audio_data: np.ndarray) -> float:
        """Detect voice activity using WebRTC VAD"""
        try:
            # Convert to 16-bit PCM at 16kHz (required by WebRTC VAD)
            if self.sr != 16000:
                audio_16k = librosa.resample(audio_data, orig_sr=self.sr, target_sr=16000)
            else:
                audio_16k = audio_data
            
            # Convert to 16-bit integer
            audio_int16 = (audio_16k * 32767).astype(np.int16)
            
            # Split into 30ms frames (required by VAD)
            frame_duration = 0.03  # 30ms
            frame_size = int(16000 * frame_duration)
            
            voice_frames = 0
            total_frames = 0
            
            for i in range(0, len(audio_int16) - frame_size, frame_size):
                frame = audio_int16[i:i + frame_size]
                if len(frame) == frame_size:
                    if self.vad.is_speech(frame.tobytes(), 16000):
                        voice_frames += 1
                    total_frames += 1
            
            return voice_frames / max(total_frames, 1)
            
        except Exception as e:
            logger.warning(f"Voice activity detection failed: {e}")
            return 0.0
    
    def _estimate_speech_rate(self, audio_data: np.ndarray) -> float:
        """Estimate speaking rate (words per minute estimate)"""
        try:
            # Simple syllable-based estimation
            # Count energy peaks as proxy for syllables
            energy = librosa.feature.rms(y=audio_data, hop_length=self.hop_length)[0]
            
            # Find peaks in energy
            from scipy.signal import find_peaks
            peaks, _ = find_peaks(energy, height=np.mean(energy), distance=10)
            
            # Estimate syllables per second
            duration = len(audio_data) / self.sr
            syllables_per_sec = len(peaks) / max(duration, 0.001)
            
            # Convert to approximate words per minute (assuming ~1.5 syllables per word)
            words_per_minute = syllables_per_sec * 60 / 1.5
            
            return float(words_per_minute)
            
        except Exception as e:
            logger.warning(f"Speech rate estimation failed: {e}")
            return 0.0
    
    def _calculate_analysis_metrics(self) -> Dict[str, Any]:
        """Calculate high-level analysis metrics from feature history"""
        if len(self.feature_buffer) < 10:
            return self._empty_results()
        
        recent_features = list(self.feature_buffer)[-200:]  # Last ~5 seconds
        
        # Calculate confidence score
        confidence_score = self._calculate_confidence(recent_features)
        self.confidence_history.append(confidence_score)
        
        # Detect nervousness patterns
        nervousness_score, nervousness_indicators = self._detect_nervousness(recent_features)
        
        # Calculate stability metrics
        stability_metrics = self._calculate_stability_metrics(recent_features)
        
        # Emotional state analysis
        emotion_summary = self._analyze_emotion_trends(recent_features)
        
        return {
            'confidence': confidence_score,
            'nervousness': nervousness_score,
            'nervousness_indicators': nervousness_indicators,
            'stability': stability_metrics,
            'emotions': emotion_summary,
            'current_features': recent_features[-1] if recent_features else {},
            'analysis_quality': self._assess_analysis_quality(recent_features)
        }
    
    def _calculate_confidence(self, features: List[Dict]) -> int:
        """Calculate speaking confidence based on multiple factors"""
        if not features:
            return 0
        
        scores = []
        weights = {
            'pitch_stability': 0.25,
            'energy_consistency': 0.20,
            'speaking_continuity': 0.20,
            'speech_clarity': 0.15,
            'emotional_positivity': 0.20
        }
        
        # Pitch stability score
        pitch_values = [f.get('pitch', 0) for f in features if f.get('pitch', 0) > 0]
        if pitch_values:
            pitch_cv = np.std(pitch_values) / (np.mean(pitch_values) + 1e-6)
            pitch_stability = max(0, 1 - pitch_cv)
            scores.append(('pitch_stability', pitch_stability))
        
        # Energy consistency
        energy_values = [f.get('energy', 0) for f in features]
        if energy_values:
            energy_cv = np.std(energy_values) / (np.mean(energy_values) + 1e-6)
            energy_consistency = max(0, 1 - energy_cv * 2)
            scores.append(('energy_consistency', energy_consistency))
        
        # Speaking continuity
        voiced_frames = sum(1 for f in features if f.get('is_voiced', 0) > 0.5)
        continuity = voiced_frames / len(features)
        scores.append(('speaking_continuity', continuity))
        
        # Speech clarity (based on spectral features)
        clarity_scores = []
        for f in features:
            if f.get('spectral_centroid', 0) > 0:
                # Higher spectral centroid often indicates clearer speech
                clarity = min(1.0, f['spectral_centroid'] / 3000)
                clarity_scores.append(clarity)
        
        if clarity_scores:
            speech_clarity = np.mean(clarity_scores)
            scores.append(('speech_clarity', speech_clarity))
        
        # Emotional positivity (if available)
        emotion_scores = [f.get('emotions', {}) for f in features if f.get('emotions')]
        if emotion_scores:
            positive_emotions = ['happy', 'confident', 'calm', 'neutral']
            positive_score = 0
            total_score = 0
            
            for emotions in emotion_scores:
                for emotion, score in emotions.items():
                    total_score += score
                    if any(pos in emotion.lower() for pos in positive_emotions):
                        positive_score += score
            
            if total_score > 0:
                emotional_positivity = positive_score / total_score
                scores.append(('emotional_positivity', emotional_positivity))
        
        # Calculate weighted confidence score
        total_weight = 0
        weighted_score = 0
        
        for metric, score in scores:
            weight = weights.get(metric, 0.1)
            weighted_score += score * weight
            total_weight += weight
        
        confidence = (weighted_score / max(total_weight, 0.001)) * 100
        return int(max(0, min(100, confidence)))
    
    def _detect_nervousness(self, features: List[Dict]) -> Tuple[int, List[str]]:
        """Detect nervousness patterns in speech"""
        if not features or len(features) < 20:
            return 0, []
        
        indicators = []
        nervousness_score = 0
        
        # Voice tremor detection
        pitch_values = [f.get('pitch', 0) for f in features if f.get('pitch', 0) > 0]
        if len(pitch_values) > 10:
            pitch_variations = np.diff(pitch_values)
            tremor_intensity = np.std(pitch_variations)
            
            if tremor_intensity > 15:
                nervousness_score += 20
                indicators.append("Voice tremor detected")
        
        # Speaking pace irregularities
        speech_rates = [f.get('speech_rate', 0) for f in features if f.get('speech_rate', 0) > 0]
        if speech_rates:
            rate_cv = np.std(speech_rates) / (np.mean(speech_rates) + 1e-6)
            if rate_cv > 0.3:
                nervousness_score += 15
                indicators.append("Irregular speaking pace")
        
        # Energy fluctuations
        energy_values = [f.get('energy', 0) for f in features]
        if energy_values:
            energy_cv = np.std(energy_values) / (np.mean(energy_values) + 1e-6)
            if energy_cv > 0.6:
                nervousness_score += 15
                indicators.append("Inconsistent volume levels")
        
        # Frequent pauses
        silence_count = sum(1 for f in features if f.get('energy', 0) < 0.005)
        silence_ratio = silence_count / len(features)
        
        if silence_ratio > 0.4:
            nervousness_score += 20
            indicators.append("Frequent hesitations or pauses")
        
        # Emotional stress indicators
        for f in features:
            emotions = f.get('emotions', {})
            stress_emotions = emotions.get('angry', 0) + emotions.get('fear', 0) + emotions.get('sad', 0)
            if stress_emotions > 0.3:
                nervousness_score += 10
                indicators.append("Emotional stress detected")
                break
        
        return min(100, nervousness_score), indicators
    
    def _calculate_stability_metrics(self, features: List[Dict]) -> Dict[str, float]:
        """Calculate various stability metrics"""
        if not features:
            return {}
        
        metrics = {}
        
        # Pitch stability
        pitch_values = [f.get('pitch', 0) for f in features if f.get('pitch', 0) > 0]
        if pitch_values:
            metrics['pitch_stability'] = 1 - (np.std(pitch_values) / np.mean(pitch_values))
        
        # Volume stability  
        energy_values = [f.get('energy', 0) for f in features]
        if energy_values:
            metrics['volume_stability'] = 1 - (np.std(energy_values) / (np.mean(energy_values) + 1e-6))
        
        # Speech rate consistency
        rates = [f.get('speech_rate', 0) for f in features if f.get('speech_rate', 0) > 0]
        if rates:
            metrics['rate_consistency'] = 1 - (np.std(rates) / (np.mean(rates) + 1e-6))
        
        return {k: max(0, min(1, v)) for k, v in metrics.items()}
    
    def _analyze_emotion_trends(self, features: List[Dict]) -> Dict[str, Any]:
        """Analyze emotional trends over recent features"""
        emotion_data = [f.get('emotions', {}) for f in features if f.get('emotions')]
        
        if not emotion_data:
            return {}
        
        # Aggregate emotions
        emotion_totals = {}
        for emotions in emotion_data:
            for emotion, score in emotions.items():
                emotion_totals[emotion] = emotion_totals.get(emotion, 0) + score
        
        # Normalize
        total = sum(emotion_totals.values())
        if total > 0:
            emotion_averages = {k: v/total for k, v in emotion_totals.items()}
        else:
            emotion_averages = {}
        
        return {
            'current_emotions': emotion_averages,
            'dominant_emotion': max(emotion_averages.items(), key=lambda x: x[1])[0] if emotion_averages else 'neutral',
            'emotion_stability': self._calculate_emotion_stability(emotion_data),
            'trend': self._detect_emotion_trend(emotion_data)
        }
    
    def _calculate_emotion_stability(self, emotion_data: List[Dict]) -> float:
        """Calculate emotional stability over time"""
        if len(emotion_data) < 5:
            return 0.5
        
        # Track variance in dominant emotions
        dominant_emotions = []
        for emotions in emotion_data:
            if emotions:
                dominant = max(emotions.items(), key=lambda x: x[1])[0]
                dominant_emotions.append(dominant)
        
        # Calculate consistency
        if not dominant_emotions:
            return 0.5
        
        from collections import Counter
        emotion_counts = Counter(dominant_emotions)
        most_common_ratio = emotion_counts.most_common(1)[0][1] / len(dominant_emotions)
        
        return float(most_common_ratio)
    
    def _detect_emotion_trend(self, emotion_data: List[Dict]) -> str:
        """Detect trending emotional direction"""
        if len(emotion_data) < 10:
            return "stable"
        
        # Compare first and second half
        mid = len(emotion_data) // 2
        first_half = emotion_data[:mid]
        second_half = emotion_data[mid:]
        
        # Calculate positive emotion scores for each half
        def positive_score(emotions_list):
            positive_emotions = ['happy', 'confident', 'calm', 'neutral']
            total_positive = 0
            total_count = 0
            for emotions in emotions_list:
                for emotion, score in emotions.items():
                    total_count += score
                    if any(pos in emotion.lower() for pos in positive_emotions):
                        total_positive += score
            return total_positive / max(total_count, 0.001)
        
        first_positive = positive_score(first_half)
        second_positive = positive_score(second_half)
        
        diff = second_positive - first_positive
        
        if diff > 0.1:
            return "improving"
        elif diff < -0.1:
            return "declining"
        else:
            return "stable"
    
    def _assess_analysis_quality(self, features: List[Dict]) -> Dict[str, float]:
        """Assess the quality of analysis data"""
        if not features:
            return {'overall': 0.0}
        
        quality_metrics = {}
        
        # Data completeness
        total_fields = ['pitch', 'energy', 'spectral_centroid', 'emotions']
        complete_count = 0
        
        for feature in features:
            fields_present = sum(1 for field in total_fields if feature.get(field, 0) > 0)
            complete_count += fields_present / len(total_fields)
        
        quality_metrics['completeness'] = complete_count / len(features)
        
        # Signal quality (based on energy levels)
        energy_values = [f.get('energy', 0) for f in features]
        if energy_values:
            avg_energy = np.mean(energy_values)
            quality_metrics['signal_strength'] = min(1.0, avg_energy / 0.1)  # Normalize to 0.1 as good level
        
        # Analysis consistency
        pitch_values = [f.get('pitch', 0) for f in features if f.get('pitch', 0) > 0]
        if len(pitch_values) > 10:
            pitch_consistency = 1 - (np.std(pitch_values) / np.mean(pitch_values))
            quality_metrics['consistency'] = max(0, pitch_consistency)
        
        # Overall quality score
        quality_metrics['overall'] = np.mean(list(quality_metrics.values()))
        
        return quality_metrics
    
    def _empty_results(self) -> Dict[str, Any]:
        """Return empty results structure"""
        return {
            'confidence': 0,
            'nervousness': 0,
            'nervousness_indicators': [],
            'stability': {},
            'emotions': {},
            'current_features': {},
            'analysis_quality': {'overall': 0.0}
        }
    
    def get_session_summary(self) -> Dict[str, Any]:
        """Get comprehensive session summary"""
        if not self.confidence_history:
            return {}
        
        confidence_scores = list(self.confidence_history)
        
        return {
            'session_duration': len(self.feature_buffer) * self.frame_duration,
            'average_confidence': np.mean(confidence_scores),
            'confidence_trend': 'improving' if confidence_scores[-1] > confidence_scores[0] else 'stable',
            'peak_confidence': max(confidence_scores),
            'total_features_analyzed': len(self.feature_buffer),
            'analysis_quality': self._assess_analysis_quality(list(self.feature_buffer))
        }
    
    def reset_session(self):
        """Reset analyzer for new session"""
        self.feature_buffer.clear()
        self.confidence_history.clear()
        self.emotion_history.clear()
        logger.info("Audio analyzer session reset")