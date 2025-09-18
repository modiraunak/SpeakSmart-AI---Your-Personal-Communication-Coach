"""
Confidence Scoring Module
Author: Raunak Kumar Modi(Team Lead)

This module provides comprehensive confidence scoring algorithms that combine
audio, video, and behavioral analysis to generate accurate speaking confidence metrics.
Uses multiple weighted factors and machine learning approaches for robust scoring.
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any
from collections import deque
from dataclasses import dataclass
from enum import Enum
import json
from datetime import datetime, timedelta


# Configure logging
logger = logging.getLogger(__name__)


class ConfidenceLevel(Enum):
    """Enumeration for confidence levels"""
    POOR = "poor"
    FAIR = "fair"
    GOOD = "good"
    EXCELLENT = "excellent"


@dataclass
class ConfidenceMetrics:
    """Data class for confidence scoring metrics"""
    overall_score: float
    vocal_confidence: float
    visual_confidence: float
    behavioral_confidence: float
    stability_score: float
    improvement_trend: float
    level: ConfidenceLevel
    timestamp: datetime
    factors: Dict[str, float]


class VocalConfidenceAnalyzer:
    """Analyzes vocal patterns for confidence indicators"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.pitch_history = deque(maxlen=500)
        self.volume_history = deque(maxlen=500)
        self.pace_history = deque(maxlen=500)
        
        # Thresholds for confidence assessment
        self.optimal_pitch_range = self.config.get('optimal_pitch_range', (120, 180))
        self.optimal_volume_range = self.config.get('optimal_volume_range', (0.3, 0.8))
        self.optimal_pace_range = self.config.get('optimal_pace_range', (2.0, 4.0))  # words per second
    
    def analyze_vocal_confidence(self, audio_features: List[Dict[str, float]]) -> Dict[str, float]:
        """Analyze vocal patterns and return confidence indicators"""
        
        if not audio_features:
            return self._get_default_vocal_scores()
        
        try:
            # Extract key vocal metrics
            pitch_values = [f.get('pitch', 0) for f in audio_features if f.get('pitch', 0) > 0]
            volume_values = [f.get('energy', 0) for f in audio_features]
            spectral_features = [f.get('spectral_centroid', 1000) for f in audio_features]
            
            scores = {}
            
            # Pitch stability and range
            scores['pitch_stability'] = self._analyze_pitch_stability(pitch_values)
            scores['pitch_range'] = self._analyze_pitch_range(pitch_values)
            
            # Volume consistency and appropriateness
            scores['volume_consistency'] = self._analyze_volume_consistency(volume_values)
            scores['volume_level'] = self._analyze_volume_level(volume_values)
            
            # Speaking pace and fluency
            scores['speaking_pace'] = self._analyze_speaking_pace(volume_values)
            scores['speech_clarity'] = self._analyze_speech_clarity(spectral_features)
            
            # Vocal stress indicators
            scores['stress_indicators'] = self._detect_vocal_stress(audio_features)
            
            # Overall vocal confidence
            scores['overall_vocal'] = self._calculate_vocal_confidence(scores)
            
            return scores
            
        except Exception as e:
            logger.warning(f"Vocal confidence analysis failed: {e}")
            return self._get_default_vocal_scores()
    
    def _analyze_pitch_stability(self, pitch_values: List[float]) -> float:
        """Analyze pitch stability for confidence assessment"""
        
        if len(pitch_values) < 10:
            return 0.5
        
        # Calculate coefficient of variation
        mean_pitch = np.mean(pitch_values)
        std_pitch = np.std(pitch_values)
        
        if mean_pitch > 0:
            cv = std_pitch / mean_pitch
            # Lower coefficient of variation indicates better stability
            stability = max(0, 1 - cv * 2)  # Scale CV to 0-1
            return min(1.0, stability)
        
        return 0.5
    
    def _analyze_pitch_range(self, pitch_values: List[float]) -> float:
        """Analyze if pitch range is appropriate for confident speaking"""
        
        if not pitch_values:
            return 0.5
        
        mean_pitch = np.mean(pitch_values)
        min_optimal, max_optimal = self.optimal_pitch_range
        
        # Score based on proximity to optimal range
        if min_optimal <= mean_pitch <= max_optimal:
            return 1.0
        elif mean_pitch < min_optimal:
            # Too low
            distance = min_optimal - mean_pitch
            return max(0, 1 - distance / 50)  # Penalize based on distance
        else:
            # Too high
            distance = mean_pitch - max_optimal
            return max(0, 1 - distance / 100)  # High pitch less penalized
    
    def _analyze_volume_consistency(self, volume_values: List[float]) -> float:
        """Analyze volume consistency"""
        
        if len(volume_values) < 10:
            return 0.5
        
        # Calculate consistency as inverse of coefficient of variation
        mean_vol = np.mean(volume_values)
        std_vol = np.std(volume_values)
        
        if mean_vol > 0:
            cv = std_vol / mean_vol
            consistency = max(0, 1 - cv)  # Lower CV = higher consistency
            return min(1.0, consistency)
        
        return 0.5
    
    def _analyze_volume_level(self, volume_values: List[float]) -> float:
        """Analyze if volume level is appropriate"""
        
        if not volume_values:
            return 0.5
        
        mean_volume = np.mean(volume_values)
        min_optimal, max_optimal = self.optimal_volume_range
        
        if min_optimal <= mean_volume <= max_optimal:
            return 1.0
        elif mean_volume < min_optimal:
            # Too quiet
            return max(0, mean_volume / min_optimal)
        else:
            # Too loud
            excess = mean_volume - max_optimal
            return max(0, 1 - excess)
    
    def _analyze_speaking_pace(self, volume_values: List[float]) -> float:
        """Analyze speaking pace from volume patterns"""
        
        if len(volume_values) < 20:
            return 0.5
        
        # Estimate speaking segments (simplified)
        threshold = np.mean(volume_values) * 0.3
        speaking_frames = sum(1 for v in volume_values if v > threshold)
        
        # Estimate words per second (very rough approximation)
        frame_duration = 0.05  # 50ms frames
        total_speaking_time = speaking_frames * frame_duration
        
        if total_speaking_time > 0:
            # Assume average of 3 words per second for estimation
            estimated_pace = (speaking_frames / len(volume_values)) * 3
            
            min_optimal, max_optimal = self.optimal_pace_range
            
            if min_optimal <= estimated_pace <= max_optimal:
                return 1.0
            else:
                # Penalize based on distance from optimal
                if estimated_pace < min_optimal:
                    return max(0, estimated_pace / min_optimal)
                else:
                    excess = estimated_pace - max_optimal
                    return max(0, 1 - excess / max_optimal)
        
        return 0.5
    
    def _analyze_speech_clarity(self, spectral_features: List[float]) -> float:
        """Analyze speech clarity from spectral features"""
        
        if not spectral_features:
            return 0.5
        
        # Higher spectral centroid often indicates clearer speech
        mean_centroid = np.mean(spectral_features)
        
        # Optimal range for speech clarity (rough estimate)
        if 1500 <= mean_centroid <= 3000:
            return 1.0
        elif mean_centroid < 1500:
            return max(0, mean_centroid / 1500)
        else:
            # Very high centroid might indicate tension
            excess = mean_centroid - 3000
            return max(0, 1 - excess / 2000)
    
    def _detect_vocal_stress(self, audio_features: List[Dict]) -> float:
        """Detect vocal stress indicators"""
        
        if not audio_features:
            return 0.0
        
        stress_score = 0.0
        
        # Analyze for tremor in pitch
        pitch_values = [f.get('pitch', 0) for f in audio_features if f.get('pitch', 0) > 0]
        if len(pitch_values) > 10:
            pitch_diff = np.diff(pitch_values)
            tremor_intensity = np.std(pitch_diff)
            
            if tremor_intensity > 20:  # High pitch variation
                stress_score += 0.3
        
        # Analyze for irregular breathing patterns (approximated from energy)
        energy_values = [f.get('energy', 0) for f in audio_features]
        if len(energy_values) > 20:
            # Look for irregular patterns in energy
            energy_diff = np.diff(energy_values)
            irregularity = np.std(energy_diff) / (np.mean(energy_values) + 1e-6)
            
            if irregularity > 1.0:
                stress_score += 0.2
        
        # Higher stress score means lower confidence
        return min(1.0, stress_score)
    
    def _calculate_vocal_confidence(self, scores: Dict[str, float]) -> float:
        """Calculate overall vocal confidence from component scores"""
        
        # Weighted combination of vocal factors
        weights = {
            'pitch_stability': 0.25,
            'pitch_range': 0.15,
            'volume_consistency': 0.20,
            'volume_level': 0.15,
            'speaking_pace': 0.15,
            'speech_clarity': 0.10
        }
        
        weighted_score = sum(scores.get(factor, 0.5) * weight 
                           for factor, weight in weights.items())
        
        # Penalize for stress indicators
        stress_penalty = scores.get('stress_indicators', 0) * 0.3
        final_score = max(0, weighted_score - stress_penalty)
        
        return min(1.0, final_score)
    
    def _get_default_vocal_scores(self) -> Dict[str, float]:
        """Default vocal confidence scores"""
        
        return {
            'pitch_stability': 0.5,
            'pitch_range': 0.5,
            'volume_consistency': 0.5,
            'volume_level': 0.5,
            'speaking_pace': 0.5,
            'speech_clarity': 0.5,
            'stress_indicators': 0.3,
            'overall_vocal': 0.5
        }


class VisualConfidenceAnalyzer:
    """Analyzes visual cues for confidence assessment"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.eye_contact_history = deque(maxlen=300)
        self.posture_history = deque(maxlen=300)
        self.emotion_history = deque(maxlen=300)
    
    def analyze_visual_confidence(self, video_analysis: Dict[str, Any]) -> Dict[str, float]:
        """Analyze visual cues for confidence indicators"""
        
        try:
            scores = {}
            
            # Eye contact analysis
            eye_contact = video_analysis.get('eye_contact', 0.5)
            scores['eye_contact'] = self._score_eye_contact(eye_contact)
            
            # Posture analysis
            posture_data = video_analysis.get('pose_analysis', {})
            scores['posture'] = self._analyze_posture_confidence(posture_data)
            
            # Facial expression analysis
            emotion_data = video_analysis.get('emotion_analysis', {})
            scores['facial_expressions'] = self._analyze_facial_confidence(emotion_data)
            
            # Gesture analysis
            gesture_data = video_analysis.get('gesture_analysis', {})
            scores['gestures'] = self._analyze_gesture_confidence(gesture_data)
            
            # Head pose and orientation
            face_data = video_analysis.get('face_analysis', {})
            scores['head_pose'] = self._analyze_head_pose_confidence(face_data)
            
            # Overall visual confidence
            scores['overall_visual'] = self._calculate_visual_confidence(scores)
            
            return scores
            
        except Exception as e:
            logger.warning(f"Visual confidence analysis failed: {e}")
            return self._get_default_visual_scores()
    
    def _score_eye_contact(self, eye_contact_score: float) -> float:
        """Score eye contact quality for confidence"""
        
        # Good eye contact is crucial for confidence
        if eye_contact_score >= 0.7:
            return 1.0
        elif eye_contact_score >= 0.5:
            return 0.8
        elif eye_contact_score >= 0.3:
            return 0.6
        else:
            return 0.4
    
    def _analyze_posture_confidence(self, posture_data: Dict[str, Any]) -> float:
        """Analyze posture for confidence indicators"""
        
        if not posture_data.get('posture_detected', False):
            return 0.5
        
        scores = []
        
        # Shoulder alignment
        shoulder_alignment = posture_data.get('shoulder_alignment', 0.5)
        scores.append(shoulder_alignment)
        
        # Spine straightness
        spine_straightness = posture_data.get('spine_straightness', 0.5)
        scores.append(spine_straightness)
        
        # Body openness
        body_openness = posture_data.get('body_openness', 0.5)
        scores.append(body_openness)
        
        # Overall posture score
        overall_posture = posture_data.get('overall_posture_score', 0.5)
        scores.append(overall_posture * 1.2)  # Give extra weight to overall score
        
        return min(1.0, np.mean(scores))
    
    def _analyze_facial_confidence(self, emotion_data: Dict[str, Any]) -> float:
        """Analyze facial expressions for confidence indicators"""
        
        emotions = emotion_data.get('emotions', {})
        if not emotions:
            return 0.5
        
        # Positive emotions that indicate confidence
        confident_emotions = ['confident', 'happy', 'focused', 'calm']
        nervous_emotions = ['nervous', 'fearful', 'surprised', 'concerned']
        
        confident_score = sum(emotions.get(emotion, 0) for emotion in confident_emotions)
        nervous_score = sum(emotions.get(emotion, 0) for emotion in nervous_emotions)
        
        # Calculate confidence based on emotion distribution
        if confident_score + nervous_score > 0:
            confidence_ratio = confident_score / (confident_score + nervous_score)
        else:
            confidence_ratio = 0.5
        
        # Also consider dominant emotion
        dominant_emotion = emotion_data.get('dominant_emotion', 'neutral')
        if dominant_emotion in confident_emotions:
            confidence_ratio += 0.1
        elif dominant_emotion in nervous_emotions:
            confidence_ratio -= 0.1
        
        return max(0, min(1.0, confidence_ratio))
    
    def _analyze_gesture_confidence(self, gesture_data: Dict[str, Any]) -> float:
        """Analyze hand gestures for confidence indicators"""
        
        hands_detected = gesture_data.get('hands_detected', 0)
        gesture_types = gesture_data.get('gesture_types', [])
        
        if hands_detected == 0:
            # No gestures might indicate nervousness or lack of engagement
            return 0.4
        
        # Confident gestures
        confident_gestures = ['open_palm', 'pointing', 'descriptive']
        nervous_gestures = ['fist', 'fidgeting', 'self_touch']
        
        confident_count = sum(1 for gesture in gesture_types if gesture in confident_gestures)
        nervous_count = sum(1 for gesture in gesture_types if gesture in nervous_gestures)
        
        if confident_count + nervous_count > 0:
            gesture_confidence = confident_count / (confident_count + nervous_count)
        else:
            gesture_confidence = 0.6  # Neutral gestures
        
        # Moderate gesture activity is good
        if hands_detected == 2 and len(gesture_types) > 0:
            gesture_confidence += 0.1
        
        return max(0, min(1.0, gesture_confidence))
    
    def _analyze_head_pose_confidence(self, face_data: Dict[str, Any]) -> float:
        """Analyze head pose for confidence indicators"""
        
        if not face_data.get('face_detected', False):
            return 0.5
        
        orientation = face_data.get('face_orientation', {})
        if not orientation:
            return 0.5
        
        yaw = abs(orientation.get('yaw', 0))
        pitch = abs(orientation.get('pitch', 0))
        roll = abs(orientation.get('roll', 0))
        
        # Good head pose: looking straight ahead
        head_pose_score = 1.0
        
        # Penalize excessive head movements
        if yaw > 20:
            head_pose_score -= (yaw - 20) / 60
        if pitch > 15:
            head_pose_score -= (pitch - 15) / 45
        if roll > 10:
            head_pose_score -= (roll - 10) / 30
        
        return max(0, min(1.0, head_pose_score))
    
    def _calculate_visual_confidence(self, scores: Dict[str, float]) -> float:
        """Calculate overall visual confidence"""
        
        weights = {
            'eye_contact': 0.30,
            'posture': 0.25,
            'facial_expressions': 0.20,
            'head_pose': 0.15,
            'gestures': 0.10
        }
        
        weighted_score = sum(scores.get(factor, 0.5) * weight 
                           for factor, weight in weights.items())
        
        return min(1.0, weighted_score)
    
    def _get_default_visual_scores(self) -> Dict[str, float]:
        """Default visual confidence scores"""
        
        return {
            'eye_contact': 0.5,
            'posture': 0.5,
            'facial_expressions': 0.5,
            'gestures': 0.5,
            'head_pose': 0.5,
            'overall_visual': 0.5
        }


class BehavioralConfidenceAnalyzer:
    """Analyzes behavioral patterns for confidence assessment"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.temporal_window = self.config.get('temporal_window', 30)  # seconds
        self.confidence_history = deque(maxlen=1000)
    
    def analyze_behavioral_confidence(self, session_data: Dict[str, Any]) -> Dict[str, float]:
        """Analyze behavioral patterns over time"""
        
        try:
            scores = {}
            
            # Consistency over time
            scores['temporal_consistency'] = self._analyze_temporal_consistency(session_data)
            
            # Improvement or decline trends
            scores['trend_analysis'] = self._analyze_confidence_trends(session_data)
            
            # Stress indicators
            scores['stress_patterns'] = self._analyze_stress_patterns(session_data)
            
            # Speaking fluency
            scores['fluency_patterns'] = self._analyze_fluency_patterns(session_data)
            
            # Overall behavioral confidence
            scores['overall_behavioral'] = self._calculate_behavioral_confidence(scores)
            
            return scores
            
        except Exception as e:
            logger.warning(f"Behavioral confidence analysis failed: {e}")
            return self._get_default_behavioral_scores()
    
    def _analyze_temporal_consistency(self, session_data: Dict[str, Any]) -> float:
        """Analyze consistency of confidence over time"""
        
        confidence_timeline = session_data.get('confidence_timeline', [])
        if len(confidence_timeline) < 5:
            return 0.5
        
        # Calculate coefficient of variation for consistency
        mean_confidence = np.mean(confidence_timeline)
        std_confidence = np.std(confidence_timeline)
        
        if mean_confidence > 0:
            cv = std_confidence / mean_confidence
            consistency = max(0, 1 - cv)  # Lower CV = higher consistency
            return min(1.0, consistency)
        
        return 0.5
    
    def _analyze_confidence_trends(self, session_data: Dict[str, Any]) -> float:
        """Analyze if confidence is improving or declining"""
        
        confidence_timeline = session_data.get('confidence_timeline', [])
        if len(confidence_timeline) < 10:
            return 0.5
        
        # Simple linear trend analysis
        x = np.arange(len(confidence_timeline))
        y = np.array(confidence_timeline)
        
        # Calculate slope
        slope = np.polyfit(x, y, 1)[0]
        
        # Positive slope indicates improvement
        if slope > 0.1:
            return min(1.0, 0.7 + slope)
        elif slope < -0.1:
            return max(0, 0.7 + slope)
        else:
            return 0.7  # Stable
    
    def _analyze_stress_patterns(self, session_data: Dict[str, Any]) -> float:
        """Analyze stress patterns in behavior"""
        
        stress_indicators = session_data.get('stress_indicators', [])
        session_duration = session_data.get('duration', 60)
        
        if not stress_indicators:
            return 1.0  # No stress indicators
        
        # Calculate stress density (indicators per minute)
        stress_density = len(stress_indicators) / (session_duration / 60)
        
        # Lower stress density indicates higher confidence
        if stress_density <= 1:
            return 1.0
        elif stress_density <= 3:
            return 0.8
        elif stress_density <= 5:
            return 0.6
        else:
            return 0.4
    
    def _analyze_fluency_patterns(self, session_data: Dict[str, Any]) -> float:
        """Analyze speaking fluency patterns"""
        
        # This would analyze pauses, hesitations, repetitions, etc.
        # For now, use a simplified approach
        
        speaking_segments = session_data.get('speaking_segments', [])
        if not speaking_segments:
            return 0.5
        
        # Analyze segment lengths and gaps
        segment_lengths = [seg.get('duration', 1) for seg in speaking_segments]
        
        if segment_lengths:
            avg_segment_length = np.mean(segment_lengths)
            
            # Longer average segments often indicate better fluency
            if avg_segment_length >= 5:  # 5+ seconds
                return 1.0
            elif avg_segment_length >= 3:
                return 0.8
            elif avg_segment_length >= 2:
                return 0.6
            else:
                return 0.4
        
        return 0.5
    
    def _calculate_behavioral_confidence(self, scores: Dict[str, float]) -> float:
        """Calculate overall behavioral confidence"""
        
        weights = {
            'temporal_consistency': 0.30,
            'trend_analysis': 0.25,
            'stress_patterns': 0.25,
            'fluency_patterns': 0.20
        }
        
        weighted_score = sum(scores.get(factor, 0.5) * weight 
                           for factor, weight in weights.items())
        
        return min(1.0, weighted_score)
    
    def _get_default_behavioral_scores(self) -> Dict[str, float]:
        """Default behavioral confidence scores"""
        
        return {
            'temporal_consistency': 0.5,
            'trend_analysis': 0.5,
            'stress_patterns': 0.6,
            'fluency_patterns': 0.5,
            'overall_behavioral': 0.5
        }


class ConfidenceScorer:
    """Main confidence scoring engine that combines all analysis methods"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        
        # Initialize component analyzers
        self.vocal_analyzer = VocalConfidenceAnalyzer(config.get('vocal', {}))
        self.visual_analyzer = VisualConfidenceAnalyzer(config.get('visual', {}))
        self.behavioral_analyzer = BehavioralConfidenceAnalyzer(config.get('behavioral', {}))
        
        # Scoring weights
        self.weights = self.config.get('weights', {
            'vocal': 0.40,
            'visual': 0.35,
            'behavioral': 0.25
        })
        
        # History for temporal analysis
        self.score_history = deque(maxlen=1000)
    
    def calculate_confidence_score(self, 
                                 audio_features: Optional[List[Dict]] = None,
                                 video_analysis: Optional[Dict] = None,
                                 session_data: Optional[Dict] = None) -> ConfidenceMetrics:
        """Calculate comprehensive confidence score"""
        
        try:
            # Initialize component scores
            vocal_scores = {}
            visual_scores = {}
            behavioral_scores = {}
            
            # Analyze vocal confidence
            if audio_features:
                vocal_scores = self.vocal_analyzer.analyze_vocal_confidence(audio_features)
            
            # Analyze visual confidence
            if video_analysis:
                visual_scores = self.visual_analyzer.analyze_visual_confidence(video_analysis)
            
            # Analyze behavioral confidence
            if session_data:
                behavioral_scores = self.behavioral_analyzer.analyze_behavioral_confidence(session_data)
            
            # Calculate overall confidence
            overall_score = self._calculate_overall_confidence(
                vocal_scores, visual_scores, behavioral_scores
            )
            
            # Calculate stability score
            stability_score = self._calculate_stability_score()
            
            # Calculate improvement trend
            improvement_trend = self._calculate_improvement_trend()
            
            # Determine confidence level
            confidence_level = self._determine_confidence_level(overall_score)
            
            # Create metrics object
            metrics = ConfidenceMetrics(
                overall_score=overall_score,
                vocal_confidence=vocal_scores.get('overall_vocal', 0.5),
                visual_confidence=visual_scores.get('overall_visual', 0.5),
                behavioral_confidence=behavioral_scores.get('overall_behavioral', 0.5),
                stability_score=stability_score,
                improvement_trend=improvement_trend,
                level=confidence_level,
                timestamp=datetime.now(),
                factors={
                    **vocal_scores,
                    **visual_scores,
                    **behavioral_scores
                }
            )
            
            # Store in history
            self.score_history.append(overall_score)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Confidence scoring failed: {e}")
            return self._get_default_metrics()
    
    def _calculate_overall_confidence(self, vocal_scores: Dict, visual_scores: Dict, 
                                    behavioral_scores: Dict) -> float:
        """Calculate weighted overall confidence score"""
        
        component_scores = {
            'vocal': vocal_scores.get('overall_vocal', 0.5),
            'visual': visual_scores.get('overall_visual', 0.5),
            'behavioral': behavioral_scores.get('overall_behavioral', 0.5)
        }
        
        # Calculate weighted average
        weighted_score = sum(score * self.weights.get(component, 0.33) 
                           for component, score in component_scores.items())
        
        return min(1.0, max(0.0, weighted_score))
    
    def _calculate_stability_score(self) -> float:
        """Calculate confidence stability over recent history"""
        
        if len(self.score_history) < 5:
            return 0.5
        
        recent_scores = list(self.score_history)[-20:]  # Last 20 scores
        
        # Calculate coefficient of variation
        mean_score = np.mean(recent_scores)
        std_score = np.std(recent_scores)
        
        if mean_score > 0:
            cv = std_score / mean_score
            stability = max(0, 1 - cv)
            return min(1.0, stability)
        
        return 0.5
    
    def _calculate_improvement_trend(self) -> float:
        """Calculate improvement trend over recent history"""
        
        if len(self.score_history) < 10:
            return 0.0
        
        recent_scores = list(self.score_history)[-50:]  # Last 50 scores
        
        # Simple linear trend
        x = np.arange(len(recent_scores))
        slope = np.polyfit(x, recent_scores, 1)[0]
        
        # Normalize slope to meaningful range
        normalized_trend = np.clip(slope * 10, -1.0, 1.0)
        
        return normalized_trend
    
    def _determine_confidence_level(self, score: float) -> ConfidenceLevel:
        """Determine confidence level category"""
        
        if score >= 0.8:
            return ConfidenceLevel.EXCELLENT
        elif score >= 0.65:
            return ConfidenceLevel.GOOD
        elif score >= 0.45:
            return ConfidenceLevel.FAIR
        else:
            return ConfidenceLevel.POOR
    
    def _get_default_metrics(self) -> ConfidenceMetrics:
        """Return default confidence metrics when calculation fails"""
        
        return ConfidenceMetrics(
            overall_score=0.5,
            vocal_confidence=0.5,
            visual_confidence=0.5,
            behavioral_confidence=0.5,
            stability_score=0.5,
            improvement_trend=0.0,
            level=ConfidenceLevel.FAIR,
            timestamp=datetime.now(),
            factors={}
        )
    
    def get_confidence_insights(self, metrics: ConfidenceMetrics) -> Dict[str, str]:
        """Generate human-readable insights from confidence metrics"""
        
        insights = {}
        
        # Overall assessment
        if metrics.level == ConfidenceLevel.EXCELLENT:
            insights['overall'] = "Excellent speaking confidence! You demonstrate strong vocal control and confident body language."
        elif metrics.level == ConfidenceLevel.GOOD:
            insights['overall'] = "Good confidence level with room for minor improvements."
        elif metrics.level == ConfidenceLevel.FAIR:
            insights['overall'] = "Fair confidence level. Focus on key areas for improvement."
        else:
            insights['overall'] = "Areas for improvement identified. Consider targeted practice."
        
        # Vocal insights
        if metrics.vocal_confidence >= 0.7:
            insights['vocal'] = "Strong vocal delivery with good pitch control and volume."
        elif metrics.vocal_confidence >= 0.5:
            insights['vocal'] = "Adequate vocal delivery with room for improvement in consistency."
        else:
            insights['vocal'] = "Focus on vocal control, pitch stability, and speaking pace."
        
        # Visual insights
        if metrics.visual_confidence >= 0.7:
            insights['visual'] = "Confident body language and good eye contact."
        elif metrics.visual_confidence >= 0.5:
            insights['visual'] = "Reasonable visual presence with some areas to develop."
        else:
            insights['visual'] = "Work on posture, eye contact, and facial expressions."
        
        # Stability insights
        if metrics.stability_score >= 0.7:
            insights['stability'] = "Consistent performance throughout the session."
        elif metrics.stability_score >= 0.5:
            insights['stability'] = "Some variation in confidence levels during speaking."
        else:
            insights['stability'] = "Focus on maintaining consistent confidence throughout."
        
        # Trend insights
        if metrics.improvement_trend > 0.3:
            insights['trend'] = "Showing clear improvement during the session!"
        elif metrics.improvement_trend > 0.1:
            insights['trend'] = "Slight improvement trend observed."
        elif metrics.improvement_trend < -0.3:
            insights['trend'] = "Consider taking breaks to maintain energy levels."
        else:
            insights['trend'] = "Stable performance maintained."
        
        return insights
    
    def export_confidence_report(self, metrics: ConfidenceMetrics) -> Dict[str, Any]:
        """Export comprehensive confidence report"""
        
        return {
            'timestamp': metrics.timestamp.isoformat(),
            'overall_score': round(metrics.overall_score * 100, 1),
            'confidence_level': metrics.level.value,
            'component_scores': {
                'vocal': round(metrics.vocal_confidence * 100, 1),
                'visual': round(metrics.visual_confidence * 100, 1),
                'behavioral': round(metrics.behavioral_confidence * 100, 1)
            },
            'performance_metrics': {
                'stability': round(metrics.stability_score * 100, 1),
                'improvement_trend': round(metrics.improvement_trend * 100, 1)
            },
            'detailed_factors': {
                factor: round(score * 100, 1) 
                for factor, score in metrics.factors.items()
            },
            'insights': self.get_confidence_insights(metrics),
            'recommendations': self._generate_recommendations(metrics)
        }
    
    def _generate_recommendations(self, metrics: ConfidenceMetrics) -> List[str]:
        """Generate personalized recommendations based on confidence analysis"""
        
        recommendations = []
        
        # Vocal recommendations
        if metrics.vocal_confidence < 0.6:
            vocal_factors = {k: v for k, v in metrics.factors.items() if 'pitch' in k or 'volume' in k or 'pace' in k}
            
            if vocal_factors.get('pitch_stability', 1.0) < 0.5:
                recommendations.append("Practice breath control exercises to improve pitch stability")
            
            if vocal_factors.get('volume_consistency', 1.0) < 0.5:
                recommendations.append("Work on maintaining consistent volume levels throughout your speech")
            
            if vocal_factors.get('speaking_pace', 1.0) < 0.5:
                recommendations.append("Practice speaking at a moderate, controlled pace")
        
        # Visual recommendations
        if metrics.visual_confidence < 0.6:
            visual_factors = {k: v for k, v in metrics.factors.items() if any(x in k for x in ['eye', 'posture', 'gesture', 'head'])}
            
            if visual_factors.get('eye_contact', 1.0) < 0.5:
                recommendations.append("Practice maintaining eye contact by focusing on specific points")
            
            if visual_factors.get('posture', 1.0) < 0.5:
                recommendations.append("Work on upright posture and open body language")
            
            if visual_factors.get('facial_expressions', 1.0) < 0.5:
                recommendations.append("Practice natural facial expressions that match your content")
        
        # Stability recommendations
        if metrics.stability_score < 0.5:
            recommendations.append("Focus on maintaining consistent energy and confidence throughout")
            recommendations.append("Consider preparation techniques to reduce performance anxiety")
        
        # Trend-based recommendations
        if metrics.improvement_trend < -0.2:
            recommendations.append("Take regular breaks during long speaking sessions")
            recommendations.append("Consider warm-up exercises before important presentations")
        
        # General recommendations based on overall score
        if metrics.overall_score < 0.4:
            recommendations.extend([
                "Start with shorter practice sessions to build confidence gradually",
                "Record yourself speaking to identify specific areas for improvement",
                "Consider joining a speaking group for regular practice and feedback"
            ])
        elif metrics.overall_score < 0.7:
            recommendations.extend([
                "Focus on your strongest areas while gradually improving weaker ones",
                "Practice with different topics to build versatility",
                "Seek feedback from others to validate your progress"
            ])
        
        return recommendations[:8]  # Limit to top 8 recommendations


class ConfidenceCalibrator:
    """Calibrates confidence scoring based on individual baselines and context"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.user_baselines = {}
        self.context_adjustments = {
            'presentation': 1.1,      # Slightly higher standards
            'interview': 1.15,        # Higher standards for interviews
            'casual': 0.9,           # More lenient for casual speaking
            'practice': 0.85         # Most lenient for practice sessions
        }
    
    def calibrate_score(self, raw_score: float, user_id: str, 
                       context: str = 'general') -> float:
        """Calibrate confidence score based on user baseline and context"""
        
        try:
            # Get user baseline
            baseline = self.user_baselines.get(user_id, 0.5)
            
            # Apply context adjustment
            context_multiplier = self.context_adjustments.get(context, 1.0)
            
            # Calibrate score
            # If user typically scores low, be more encouraging
            # If user typically scores high, maintain high standards
            if baseline < 0.4:
                # Boost scores for users who typically struggle
                calibrated = raw_score * 1.1 + 0.05
            elif baseline > 0.7:
                # Maintain high standards for confident speakers
                calibrated = raw_score * 0.95
            else:
                calibrated = raw_score
            
            # Apply context adjustment
            final_score = calibrated / context_multiplier
            
            return max(0.0, min(1.0, final_score))
            
        except Exception as e:
            logger.warning(f"Score calibration failed: {e}")
            return raw_score
    
    def update_user_baseline(self, user_id: str, new_score: float):
        """Update user's baseline confidence score"""
        
        if user_id not in self.user_baselines:
            self.user_baselines[user_id] = new_score
        else:
            # Exponential moving average
            alpha = 0.1
            self.user_baselines[user_id] = (
                alpha * new_score + (1 - alpha) * self.user_baselines[user_id]
            )


class ConfidenceTrendAnalyzer:
    """Analyzes confidence trends over multiple sessions"""
    
    def __init__(self):
        self.session_history = deque(maxlen=100)
    
    def add_session(self, session_metrics: ConfidenceMetrics):
        """Add session metrics to trend analysis"""
        
        session_data = {
            'timestamp': session_metrics.timestamp,
            'overall_score': session_metrics.overall_score,
            'vocal_score': session_metrics.vocal_confidence,
            'visual_score': session_metrics.visual_confidence,
            'behavioral_score': session_metrics.behavioral_confidence
        }
        
        self.session_history.append(session_data)
    
    def analyze_trends(self, days: int = 30) -> Dict[str, Any]:
        """Analyze confidence trends over specified time period"""
        
        if not self.session_history:
            return {'error': 'No session data available'}
        
        # Filter sessions within time period
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_sessions = [
            s for s in self.session_history 
            if s['timestamp'] >= cutoff_date
        ]
        
        if len(recent_sessions) < 2:
            return {'error': 'Insufficient data for trend analysis'}
        
        # Calculate trends for each component
        trends = {}
        
        for component in ['overall_score', 'vocal_score', 'visual_score', 'behavioral_score']:
            scores = [s[component] for s in recent_sessions]
            
            if len(scores) >= 3:
                # Linear regression for trend
                x = np.arange(len(scores))
                slope, intercept = np.polyfit(x, scores, 1)
                
                trends[component] = {
                    'slope': slope,
                    'current': scores[-1],
                    'average': np.mean(scores),
                    'improvement': slope > 0.01,
                    'change_rate': slope * len(scores)  # Total change over period
                }
        
        return {
            'period_days': days,
            'session_count': len(recent_sessions),
            'trends': trends,
            'overall_improvement': trends.get('overall_score', {}).get('improvement', False)
        }


# Factory function for easy initialization
def create_confidence_scorer(config: Optional[Dict] = None) -> ConfidenceScorer:
    """Factory function to create configured confidence scorer"""
    
    default_config = {
        'weights': {
            'vocal': 0.40,
            'visual': 0.35, 
            'behavioral': 0.25
        },
        'vocal': {
            'optimal_pitch_range': (120, 180),
            'optimal_volume_range': (0.3, 0.8),
            'optimal_pace_range': (2.0, 4.0)
        },
        'visual': {
            'eye_contact_weight': 0.30,
            'posture_weight': 0.25
        },
        'behavioral': {
            'temporal_window': 30,
            'consistency_weight': 0.30
        }
    }
    
    if config:
        # Merge user config with defaults
        merged_config = {**default_config, **config}
        for section in ['vocal', 'visual', 'behavioral']:
            if section in config:
                merged_config[section] = {**default_config[section], **config[section]}
    else:
        merged_config = default_config
    
    return ConfidenceScorer(merged_config)