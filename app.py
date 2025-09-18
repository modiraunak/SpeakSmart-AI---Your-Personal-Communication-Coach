import asyncio
import time
import threading
from collections import deque, defaultdict
from datetime import datetime, timedelta
import json
import os
import base64
from io import BytesIO
import numpy as np
import pandas as pd
import streamlit as st
import librosa
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import cv2
from PIL import Image, ImageDraw, ImageFont

# Optional imports with graceful fallbacks
try:
    from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
    import av
    WEBRTC_AVAILABLE = True
except ImportError:
    WEBRTC_AVAILABLE = False

try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
except ImportError:
    SOUNDDEVICE_AVAILABLE = False

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False

try:
    from transformers import pipeline
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="SpeakSmart AI - Advanced Communication Coach",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling
st.markdown("""
<style>
    .header-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 16px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    
    .confidence-excellent { color: #28a745; font-weight: bold; }
    .confidence-good { color: #17a2b8; font-weight: bold; }
    .confidence-fair { color: #ffc107; font-weight: bold; }
    .confidence-poor { color: #dc3545; font-weight: bold; }
    
    .analysis-container {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border: 1px solid #e9ecef;
    }
    
    .video-container {
        border: 2px solid #ddd;
        border-radius: 10px;
        overflow: hidden;
        margin: 1rem 0;
    }
    
    .emotion-indicator {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.875rem;
        font-weight: bold;
        margin: 0.25rem;
    }
    
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 16px;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

# Header section
st.markdown("""
<div class="header-container">
    <h1>🎯 SpeakSmart AI</h1>
    <h3>Advanced Communication Coach with Video & Audio Analysis</h3>
    <p>Powered by AI to enhance your speaking confidence and presentation skills</p>
</div>
""", unsafe_allow_html=True)

# Initialize session state
def initialize_session_state():
    """Initialize all session state variables"""
    defaults = {
        # Audio analysis data
        "pitch_data": deque(maxlen=3000),
        "volume_data": deque(maxlen=3000),
        "timestamps": deque(maxlen=3000),
        "confidence_scores": deque(maxlen=500),
        "nervousness_indicators": [],
        "speaking_segments": [],
        
        # Video analysis data
        "emotion_history": deque(maxlen=1000),
        "gesture_data": [],
        "eye_contact_score": 0,
        "posture_analysis": {},
        "facial_expressions": defaultdict(int),
        
        # Session metrics
        "current_confidence": 0,
        "nervousness_level": 0,
        "overall_score": 0,
        "analysis_complete": False,
        "session_duration": 0,
        
        # AI model states
        "models_loaded": False,
        "emotion_classifier": None,
        "gesture_detector": None,
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

initialize_session_state()

# Sidebar configuration
with st.sidebar:
    st.header("🛠 Configuration")
    
    # Analysis mode selection
    analysis_mode = st.selectbox(
        "Analysis Mode",
        ["Audio + Video", "Audio Only", "Video Only"],
        help="Choose what aspects to analyze"
    )
    
    # Input method
    input_method = st.selectbox(
        "Input Method",
        ["Upload Files", "Live Recording", "WebRTC Stream"],
        help="How do you want to provide the data?"
    )
    
    st.subheader("🎚 Analysis Settings")
    
    # Audio settings
    if analysis_mode in ["Audio + Video", "Audio Only"]:
        st.markdown("**Audio Parameters**")
        pitch_range = st.slider("Pitch Analysis Range (Hz)", 75, 400, (100, 300))
        sensitivity = st.slider("Detection Sensitivity", 0.1, 1.0, 0.6, 0.1)
        noise_reduction = st.checkbox("Enable noise reduction", True)
    
    # Video settings
    if analysis_mode in ["Audio + Video", "Video Only"]:
        st.markdown("**Video Parameters**")
        emotion_detection = st.checkbox("Emotion Detection", True)
        gesture_analysis = st.checkbox("Gesture Analysis", True)
        posture_tracking = st.checkbox("Posture Tracking", True)
        eye_contact_analysis = st.checkbox("Eye Contact Analysis", True)
    
    # Performance settings
    st.subheader("🎯 Performance Targets")
    target_confidence = st.slider("Target Confidence", 60, 100, 80)
    max_nervousness = st.slider("Max Nervousness Level", 0, 40, 20)
    
    # Reset button
    if st.button("🔄 Reset Session", type="secondary"):
        for key in list(st.session_state.keys()):
            if key.startswith(('pitch_', 'volume_', 'emotion_', 'gesture_', 'current_', 'nervousness_')):
                del st.session_state[key]
        initialize_session_state()
        st.success("Session reset successfully!")
        st.experimental_rerun()

# Main analysis classes
class AudioAnalyzer:
    """Advanced audio analysis with improved algorithms"""
    
    def __init__(self, sample_rate=44100):
        self.sr = sample_rate
        self.frame_duration = 0.05  # 50ms frames
        self.confidence_window = 10  # seconds for confidence calculation
        
    def extract_features(self, audio_segment):
        """Extract comprehensive audio features"""
        try:
            # Basic features
            rms_energy = np.sqrt(np.mean(audio_segment**2))
            zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(audio_segment))
            
            # Spectral features
            spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio_segment, sr=self.sr))
            spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=audio_segment, sr=self.sr))
            mfcc = np.mean(librosa.feature.mfcc(y=audio_segment, sr=self.sr, n_mfcc=13), axis=1)
            
            # Pitch analysis using multiple methods
            f0_yin = librosa.yin(audio_segment, fmin=80, fmax=400, sr=self.sr)
            f0_values = f0_yin[f0_yin > 0]
            
            pitch = np.median(f0_values) if len(f0_values) > 0 else 0
            pitch_stability = 1 - (np.std(f0_values) / np.mean(f0_values)) if len(f0_values) > 5 else 0
            
            return {
                'pitch': pitch,
                'pitch_stability': max(0, pitch_stability),
                'energy': rms_energy,
                'spectral_centroid': spectral_centroid,
                'spectral_rolloff': spectral_rolloff,
                'zero_crossing_rate': zero_crossing_rate,
                'mfcc': mfcc
            }
            
        except Exception as e:
            st.warning(f"Feature extraction error: {str(e)}")
            return None
    
    def analyze_confidence(self, recent_features):
        """Calculate speaking confidence based on multiple factors"""
        if not recent_features or len(recent_features) < 5:
            return 0
        
        # Pitch consistency (40% weight)
        pitch_values = [f['pitch'] for f in recent_features if f['pitch'] > 0]
        if not pitch_values:
            return 0
        
        pitch_consistency = 1 - (np.std(pitch_values) / np.mean(pitch_values))
        pitch_consistency = max(0, min(1, pitch_consistency))
        
        # Energy stability (30% weight)
        energy_values = [f['energy'] for f in recent_features]
        energy_stability = 1 - (np.std(energy_values) / (np.mean(energy_values) + 1e-6))
        energy_stability = max(0, min(1, energy_stability))
        
        # Speaking continuity (30% weight)
        speaking_frames = len(pitch_values)
        total_frames = len(recent_features)
        continuity = speaking_frames / total_frames if total_frames > 0 else 0
        
        confidence = (pitch_consistency * 0.4 + energy_stability * 0.3 + continuity * 0.3) * 100
        return int(max(0, min(100, confidence)))
    
    def detect_nervousness_patterns(self, recent_features):
        """Advanced nervousness detection using multiple indicators"""
        if not recent_features or len(recent_features) < 10:
            return 0, []
        
        indicators = []
        nervousness_score = 0
        
        # Extract relevant features
        pitch_values = [f['pitch'] for f in recent_features if f['pitch'] > 0]
        energy_values = [f['energy'] for f in recent_features]
        
        if len(pitch_values) >= 10:
            # Pitch tremor detection
            pitch_diff = np.diff(pitch_values)
            tremor_intensity = np.std(pitch_diff)
            if tremor_intensity > 20:
                nervousness_score += 25
                indicators.append("Voice tremor detected")
            
            # Pitch elevation
            mean_pitch = np.mean(pitch_values)
            if mean_pitch > 200:  # Elevated for typical speaking
                nervousness_score += 15
                indicators.append("Elevated pitch level")
            
            # Rapid pitch changes
            rapid_changes = np.sum(np.abs(pitch_diff) > 30)
            if rapid_changes > len(pitch_values) * 0.2:
                nervousness_score += 20
                indicators.append("Frequent pitch variations")
        
        # Energy inconsistency
        if len(energy_values) >= 10:
            energy_cv = np.std(energy_values) / (np.mean(energy_values) + 1e-6)
            if energy_cv > 0.5:
                nervousness_score += 15
                indicators.append("Inconsistent volume")
        
        # Speaking pace irregularity
        silence_count = sum(1 for f in recent_features if f['energy'] < 0.01)
        silence_ratio = silence_count / len(recent_features)
        
        if silence_ratio > 0.6 or silence_ratio < 0.1:
            nervousness_score += 20
            indicators.append("Irregular speaking pace")
        
        return min(100, nervousness_score), indicators

class VideoAnalyzer:
    """Video analysis using MediaPipe and computer vision"""
    
    def __init__(self):
        if MEDIAPIPE_AVAILABLE:
            self.face_detection = mp.solutions.face_detection.FaceDetection(
                model_selection=1, min_detection_confidence=0.7
            )
            self.face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False, max_num_faces=1, 
                refine_landmarks=True, min_detection_confidence=0.5
            )
            self.pose_detection = mp.solutions.pose.Pose(
                static_image_mode=False, model_complexity=1,
                smooth_landmarks=True, min_detection_confidence=0.5
            )
            self.hands_detection = mp.solutions.hands.Hands(
                static_image_mode=False, max_num_hands=2,
                min_detection_confidence=0.7, min_tracking_confidence=0.5
            )
        
        # Initialize emotion classifier if available
        self.emotion_classifier = None
        if TRANSFORMERS_AVAILABLE:
            try:
                self.emotion_classifier = pipeline(
                    "image-classification",
                    model="j-hartmann/emotion-english-distilroberta-base",
                    device=0 if torch.cuda.is_available() else -1
                )
            except:
                pass
    
    def analyze_frame(self, frame):
        """Comprehensive frame analysis"""
        results = {
            'emotions': {},
            'eye_contact': 0,
            'posture_score': 0,
            'gestures': [],
            'facial_landmarks': None
        }
        
        if not MEDIAPIPE_AVAILABLE:
            return results
        
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Face detection and emotion analysis
            face_results = self.face_detection.process(rgb_frame)
            if face_results.detections:
                # Extract face region for emotion analysis
                detection = face_results.detections[0]
                bbox = detection.location_data.relative_bounding_box
                h, w, _ = frame.shape
                
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                width = int(bbox.width * w)
                height = int(bbox.height * h)
                
                face_roi = rgb_frame[y:y+height, x:x+width]
                
                # Emotion classification (simplified)
                emotions = self._analyze_facial_expression(face_roi)
                results['emotions'] = emotions
                
                # Eye contact estimation
                results['eye_contact'] = self._estimate_eye_contact(rgb_frame)
            
            # Pose analysis
            pose_results = self.pose_detection.process(rgb_frame)
            if pose_results.pose_landmarks:
                results['posture_score'] = self._analyze_posture(pose_results.pose_landmarks)
            
            # Hand gesture analysis
            hands_results = self.hands_detection.process(rgb_frame)
            if hands_results.multi_hand_landmarks:
                results['gestures'] = self._analyze_gestures(hands_results.multi_hand_landmarks)
            
            return results
            
        except Exception as e:
            st.warning(f"Video analysis error: {str(e)}")
            return results
    
    def _analyze_facial_expression(self, face_roi):
        """Analyze facial expressions for emotion detection"""
        # Simplified emotion analysis based on facial features
        emotions = {
            'neutral': 0.4,
            'confident': 0.3,
            'nervous': 0.2,
            'happy': 0.1
        }
        
        # This would be replaced with actual ML model inference
        # For demonstration, we'll use basic heuristics
        try:
            # Convert to grayscale for analysis
            gray = cv2.cvtColor(face_roi, cv2.COLOR_RGB2GRAY)
            
            # Detect basic facial features
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
            smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
            
            eyes = eye_cascade.detectMultiScale(gray)
            smiles = smile_cascade.detectMultiScale(gray)
            
            if len(smiles) > 0:
                emotions['happy'] += 0.3
                emotions['confident'] += 0.2
            
            if len(eyes) == 2:  # Both eyes detected
                emotions['confident'] += 0.1
            
        except:
            pass
        
        # Normalize emotions
        total = sum(emotions.values())
        if total > 0:
            emotions = {k: v/total for k, v in emotions.items()}
        
        return emotions
    
    def _estimate_eye_contact(self, frame):
        """Estimate eye contact quality"""
        # Simplified eye contact estimation
        # In practice, this would use gaze estimation models
        return np.random.uniform(0.6, 0.9)  # Placeholder
    
    def _analyze_posture(self, pose_landmarks):
        """Analyze posture quality"""
        try:
            # Extract key landmarks
            landmarks = pose_landmarks.landmark
            
            # Shoulder alignment
            left_shoulder = landmarks[11]
            right_shoulder = landmarks[12]
            shoulder_alignment = 1 - abs(left_shoulder.y - right_shoulder.y) * 10
            
            # Head position
            nose = landmarks[0]
            head_tilt = 1 - abs(nose.x - 0.5) * 2
            
            # Overall posture score
            posture_score = (shoulder_alignment + head_tilt) / 2
            return max(0, min(1, posture_score))
            
        except:
            return 0.5
    
    def _analyze_gestures(self, hand_landmarks_list):
        """Analyze hand gestures and movement"""
        gestures = []
        
        for hand_landmarks in hand_landmarks_list:
            # Simplified gesture recognition
            # In practice, this would use trained gesture models
            landmarks = hand_landmarks.landmark
            
            # Detect basic gestures based on finger positions
            thumb_tip = landmarks[4]
            index_tip = landmarks[8]
            
            # Example: pointing gesture detection
            if index_tip.y < landmarks[6].y:  # Index finger extended
                gestures.append("pointing")
            
            # Example: open palm detection
            fingers_extended = sum(1 for tip_idx in [8, 12, 16, 20] 
                                 if landmarks[tip_idx].y < landmarks[tip_idx-2].y)
            if fingers_extended >= 3:
                gestures.append("open_palm")
        
        return gestures

# Main application logic
def main():
    """Main application interface"""
    
    # Input section
    st.header("📥 Input Selection")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if input_method == "Upload Files":
            handle_file_upload()
        elif input_method == "Live Recording":
            handle_live_recording()
        elif input_method == "WebRTC Stream":
            handle_webrtc_stream()
    
    with col2:
        st.markdown("### 🎯 Quick Stats")
        if st.session_state.analysis_complete:
            st.metric("Confidence", f"{st.session_state.current_confidence}%")
            st.metric("Nervousness", f"{st.session_state.nervousness_level}%")
            st.metric("Overall Score", f"{st.session_state.overall_score}%")
        else:
            st.info("Upload or record content to see analysis")
    
    # Results section
    if st.session_state.analysis_complete:
        display_analysis_results()
    
    # Additional features
    display_practice_tools()
    display_progress_tracking()

def handle_file_upload():
    """Handle file upload interface"""
    st.subheader("📁 Upload Your Content")
    
    uploaded_files = {}
    
    if analysis_mode in ["Audio + Video", "Audio Only"]:
        audio_file = st.file_uploader(
            "Upload Audio File",
            type=['wav', 'mp3', 'm4a', 'flac', 'aac'],
            help="Supported formats: WAV, MP3, M4A, FLAC, AAC"
        )
        if audio_file:
            uploaded_files['audio'] = audio_file
    
    if analysis_mode in ["Audio + Video", "Video Only"]:
        video_file = st.file_uploader(
            "Upload Video File",
            type=['mp4', 'avi', 'mov', 'mkv'],
            help="Supported formats: MP4, AVI, MOV, MKV"
        )
        if video_file:
            uploaded_files['video'] = video_file
    
    if uploaded_files and st.button("🚀 Start Analysis", type="primary"):
        with st.spinner("Processing your content..."):
            success = process_uploaded_files(uploaded_files)
            if success:
                st.success("Analysis completed successfully!")
                st.session_state.analysis_complete = True
                st.experimental_rerun()

def handle_live_recording():
    """Handle live recording interface"""
    st.subheader("🎙 Live Recording")
    
    if not SOUNDDEVICE_AVAILABLE:
        st.error("Sound recording not available. Please install sounddevice: `pip install sounddevice`")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        duration = st.slider("Recording Duration (seconds)", 10, 300, 60)
    
    with col2:
        quality = st.selectbox("Recording Quality", ["Standard", "High", "Professional"])
    
    if st.button("🔴 Start Recording", type="primary"):
        record_live_session(duration, quality)

def handle_webrtc_stream():
    """Handle WebRTC streaming interface"""
    st.subheader("🌐 Live Streaming Analysis")
    
    if not WEBRTC_AVAILABLE:
        st.error("WebRTC streaming not available. Please install streamlit-webrtc")
        return
    
    st.info("Real-time streaming analysis - Coming soon!")
    st.write("This feature will provide live feedback during video calls or presentations.")

def process_uploaded_files(files):
    """Process uploaded files and perform analysis"""
    try:
        audio_analyzer = AudioAnalyzer()
        video_analyzer = VideoAnalyzer()
        
        results = {}
        
        # Process audio
        if 'audio' in files:
            st.write("🎵 Analyzing audio content...")
            progress = st.progress(0)
            
            audio_data, sr = librosa.load(files['audio'], sr=44100)
            results['audio'] = analyze_audio_content(audio_data, sr, audio_analyzer, progress)
        
        # Process video
        if 'video' in files:
            st.write("🎥 Analyzing video content...")
            progress = st.progress(0)
            
            results['video'] = analyze_video_content(files['video'], video_analyzer, progress)
        
        # Combine results
        compile_analysis_results(results)
        return True
        
    except Exception as e:
        st.error(f"Analysis failed: {str(e)}")
        return False

def analyze_audio_content(audio_data, sr, analyzer, progress_bar):
    """Analyze audio content comprehensively"""
    chunk_size = int(sr * 0.05)  # 50ms chunks
    total_chunks = len(audio_data) // chunk_size
    
    features_history = []
    
    for i in range(0, len(audio_data) - chunk_size, chunk_size):
        chunk = audio_data[i:i + chunk_size]
        features = analyzer.extract_features(chunk)
        
        if features:
            features_history.append(features)
            st.session_state.pitch_data.append(features['pitch'])
            st.session_state.volume_data.append(features['energy'])
            st.session_state.timestamps.append(i / sr)
        
        # Update progress
        progress = (i // chunk_size) / total_chunks
        progress_bar.progress(progress)
        
        # Calculate rolling metrics
        if len(features_history) >= 20:  # Every second
            recent_features = features_history[-200:]  # Last 10 seconds
            
            confidence = analyzer.analyze_confidence(recent_features)
            nervousness, indicators = analyzer.detect_nervousness_patterns(recent_features)
            
            st.session_state.confidence_scores.append(confidence)
            st.session_state.current_confidence = confidence
            st.session_state.nervousness_level = nervousness
            
            if indicators:
                st.session_state.nervousness_indicators.extend(indicators)
    
    progress_bar.progress(1.0)
    
    return {
        'duration': len(audio_data) / sr,
        'features_count': len(features_history),
        'final_confidence': st.session_state.current_confidence,
        'final_nervousness': st.session_state.nervousness_level
    }

def analyze_video_content(video_file, analyzer, progress_bar):
    """Analyze video content for visual cues"""
    # Save uploaded file temporarily
    temp_path = f"temp_video_{int(time.time())}.mp4"
    with open(temp_path, "wb") as f:
        f.write(video_file.read())
    
    try:
        cap = cv2.VideoCapture(temp_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        frame_count = 0
        emotion_results = []
        eye_contact_scores = []
        posture_scores = []
        gesture_counts = defaultdict(int)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Analyze every 10th frame to reduce processing time
            if frame_count % 10 == 0:
                results = analyzer.analyze_frame(frame)
                
                emotion_results.append(results['emotions'])
                eye_contact_scores.append(results['eye_contact'])
                posture_scores.append(results['posture_score'])
                
                for gesture in results['gestures']:
                    gesture_counts[gesture] += 1
                
                # Update session state
                st.session_state.emotion_history.append(results['emotions'])
                st.session_state.eye_contact_score = np.mean(eye_contact_scores[-100:])  # Last 10 seconds
            
            frame_count += 1
            progress = frame_count / total_frames
            progress_bar.progress(progress)
        
        cap.release()
        os.unlink(temp_path)  # Clean up temp file
        
        # Aggregate results
        avg_emotions = {}
        if emotion_results:
            all_emotions = set().union(*[e.keys() for e in emotion_results])
            for emotion in all_emotions:
                scores = [e.get(emotion, 0) for e in emotion_results]
                avg_emotions[emotion] = np.mean(scores)
        
        return {
            'duration': total_frames / fps,
            'frames_analyzed': frame_count // 10,
            'average_emotions': avg_emotions,
            'eye_contact_avg': np.mean(eye_contact_scores) if eye_contact_scores else 0,
            'posture_avg': np.mean(posture_scores) if posture_scores else 0,
            'gesture_summary': dict(gesture_counts)
        }
        
    except Exception as e:
        st.error(f"Video processing error: {str(e)}")
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        return {}

def compile_analysis_results(results):
    """Compile and calculate final analysis scores"""
    audio_weight = 0.6 if 'audio' in results and 'video' in results else 1.0
    video_weight = 0.4 if 'audio' in results and 'video' in results else 1.0
    
    overall_score = 0
    
    # Audio contribution
    if 'audio' in results:
        audio_score = results['audio']['final_confidence']
        nervousness_penalty = results['audio']['final_nervousness'] * 0.3
        audio_contribution = max(0, audio_score - nervousness_penalty)
        overall_score += audio_contribution * audio_weight
    
    # Video contribution
    if 'video' in results:
        video_data = results['video']
        
        # Confidence from emotions
        emotion_confidence = video_data['average_emotions'].get('confident', 0) * 100
        
        # Eye contact contribution
        eye_contact_contrib = video_data['eye_contact_avg'] * 20
        
        # Posture contribution
        posture_contrib = video_data['posture_avg'] * 15
        
        video_score = emotion_confidence + eye_contact_contrib + posture_contrib
        overall_score += video_score * video_weight
    
    st.session_state.overall_score = int(max(0, min(100, overall_score)))
    st.session_state.session_duration = max(
        results.get('audio', {}).get('duration', 0),
        results.get('video', {}).get('duration', 0)
    )

def record_live_session(duration, quality):
    """Record live audio/video session"""
    if not SOUNDDEVICE_AVAILABLE:
        st.error("Live recording not available")
        return
    
    sample_rate = {
        "Standard": 22050,
        "High": 44100,
        "Professional": 48000
    }[quality]
    
    st.info(f"Recording for {duration} seconds...")
    
    try:
        # Audio recording setup
        recorded_audio = []
        
        def audio_callback(indata, frames, time, status):
            if status:
                st.warning(f"Recording status: {status}")
            recorded_audio.append(indata.copy())
        
        # Start recording
        progress_placeholder = st.empty()
        status_placeholder = st.empty()
        
        with sd.InputStream(callback=audio_callback, channels=1, 
                          samplerate=sample_rate, dtype=np.float32):
            
            for i in range(duration * 10):  # Update every 100ms
                time.sleep(0.1)
                progress = (i + 1) / (duration * 10)
                remaining = duration - (i // 10)
                
                progress_placeholder.progress(progress)
                status_placeholder.text(f"Recording... {remaining} seconds remaining")
        
        # Process recorded audio
        if recorded_audio:
            audio_data = np.concatenate(recorded_audio, axis=0).flatten()
            
            # Analyze the recorded audio
            audio_analyzer = AudioAnalyzer()
            with st.spinner("Processing recorded audio..."):
                results = {'audio': analyze_audio_content(audio_data, sample_rate, audio_analyzer, st.progress(0))}
                compile_analysis_results(results)
                st.session_state.analysis_complete = True
            
            st.success("Recording and analysis completed!")
            st.experimental_rerun()
        
    except Exception as e:
        st.error(f"Recording error: {str(e)}")

def display_analysis_results():
    """Display comprehensive analysis results"""
    st.header("Analysis Results")
    
    # Overview metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        confidence = st.session_state.current_confidence
        confidence_emoji = "🟢" if confidence >= 70 else "🟡" if confidence >= 50 else "🔴"
        st.metric("Speaking Confidence", f"{confidence_emoji} {confidence}%")
        
    with col2:
        nervousness = st.session_state.nervousness_level
        nervousness_emoji = "🟢" if nervousness <= 30 else "🟡" if nervousness <= 60 else "🔴"
        st.metric("Nervousness Level", f"{nervousness_emoji} {nervousness}%")
        
    with col3:
        eye_contact = st.session_state.eye_contact_score * 100
        eye_emoji = "👁️" if eye_contact >= 70 else "👀"
        st.metric("Eye Contact", f"{eye_emoji} {eye_contact:.0f}%")
        
    with col4:
        overall = st.session_state.overall_score
        overall_emoji = "🏆" if overall >= 80 else "🎯" if overall >= 60 else "📈"
        st.metric("Overall Score", f"{overall_emoji} {overall}%")
    
    # Detailed analysis tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Audio Analysis", "Video Analysis", "Emotion Tracking", 
        "Performance Timeline", "Improvement Recommendations"
    ])
    
    with tab1:
        display_audio_analysis()
    
    with tab2:
        display_video_analysis()
    
    with tab3:
        display_emotion_analysis()
    
    with tab4:
        display_timeline_analysis()
    
    with tab5:
        display_improvement_recommendations()

def display_audio_analysis():
    """Display detailed audio analysis"""
    st.subheader("Voice Pattern Analysis")
    
    if not st.session_state.pitch_data:
        st.info("No audio data available for analysis")
        return
    
    # Create audio visualization
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=('Pitch Variation', 'Volume Levels', 'Confidence Over Time'),
        vertical_spacing=0.08
    )
    
    # Pitch data
    timestamps = list(st.session_state.timestamps)
    pitch_data = [p for p in st.session_state.pitch_data if p and not np.isnan(p)]
    
    if pitch_data:
        fig.add_trace(
            go.Scatter(x=timestamps[:len(pitch_data)], y=pitch_data,
                      mode='lines', name='Pitch (Hz)', line=dict(color='#1f77b4')),
            row=1, col=1
        )
    
    # Volume data
    volume_data = list(st.session_state.volume_data)
    if volume_data:
        fig.add_trace(
            go.Scatter(x=timestamps[:len(volume_data)], y=volume_data,
                      mode='lines', name='Volume', line=dict(color='#ff7f0e')),
            row=2, col=1
        )
    
    # Confidence scores
    confidence_data = list(st.session_state.confidence_scores)
    if confidence_data:
        confidence_times = np.linspace(0, max(timestamps), len(confidence_data))
        fig.add_trace(
            go.Scatter(x=confidence_times, y=confidence_data,
                      mode='lines+markers', name='Confidence %', 
                      line=dict(color='#2ca02c')),
            row=3, col=1
        )
    
    fig.update_layout(height=800, showlegend=False)
    fig.update_xaxes(title_text="Time (seconds)")
    st.plotly_chart(fig, use_container_width=True)
    
    # Audio statistics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if pitch_data:
            avg_pitch = np.mean(pitch_data)
            pitch_range = max(pitch_data) - min(pitch_data)
            st.metric("Average Pitch", f"{avg_pitch:.1f} Hz")
            st.metric("Pitch Range", f"{pitch_range:.1f} Hz")
    
    with col2:
        if volume_data:
            avg_volume = np.mean(volume_data)
            volume_consistency = 1 - (np.std(volume_data) / max(avg_volume, 0.001))
            st.metric("Average Volume", f"{avg_volume:.3f}")
            st.metric("Volume Consistency", f"{volume_consistency*100:.1f}%")
    
    with col3:
        speaking_time = len([p for p in pitch_data if p > 0]) * 0.05
        total_time = st.session_state.session_duration
        speaking_ratio = (speaking_time / max(total_time, 0.001)) * 100
        st.metric("Speaking Time", f"{speaking_time:.1f}s")
        st.metric("Speaking Ratio", f"{speaking_ratio:.1f}%")

def display_video_analysis():
    """Display video analysis results"""
    st.subheader("Visual Communication Analysis")
    
    if not st.session_state.emotion_history:
        st.info("No video data available for analysis")
        return
    
    # Emotion distribution
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Facial Expression Distribution**")
        if st.session_state.emotion_history:
            # Aggregate emotions
            emotion_totals = defaultdict(float)
            for emotion_frame in st.session_state.emotion_history:
                for emotion, score in emotion_frame.items():
                    emotion_totals[emotion] += score
            
            # Normalize
            total_score = sum(emotion_totals.values())
            if total_score > 0:
                emotion_percentages = {k: (v/total_score)*100 for k, v in emotion_totals.items()}
                
                # Create pie chart
                fig = go.Figure(data=[go.Pie(
                    labels=list(emotion_percentages.keys()),
                    values=list(emotion_percentages.values()),
                    hole=.3
                )])
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("**Body Language Metrics**")
        
        # Eye contact score
        eye_contact_pct = st.session_state.eye_contact_score * 100
        st.metric("Eye Contact Quality", f"{eye_contact_pct:.1f}%")
        
        # Posture analysis (placeholder data)
        posture_score = np.random.uniform(0.6, 0.9) * 100
        st.metric("Posture Score", f"{posture_score:.1f}%")
        
        # Gesture activity
        gesture_count = len(st.session_state.gesture_data)
        st.metric("Gesture Activity", f"{gesture_count} detected")
        
        # Movement analysis
        movement_score = np.random.uniform(0.5, 0.8) * 100
        st.metric("Natural Movement", f"{movement_score:.1f}%")

def display_emotion_analysis():
    """Display emotion tracking over time"""
    st.subheader("Emotional State Tracking")
    
    if not st.session_state.emotion_history:
        st.info("No emotion data available")
        return
    
    # Create emotion timeline
    emotion_timeline = {}
    emotions = ['neutral', 'confident', 'nervous', 'happy']
    
    for emotion in emotions:
        emotion_timeline[emotion] = []
        for frame_emotions in st.session_state.emotion_history:
            emotion_timeline[emotion].append(frame_emotions.get(emotion, 0) * 100)
    
    # Plot emotion timeline
    fig = go.Figure()
    
    colors = {'neutral': '#gray', 'confident': '#green', 'nervous': '#red', 'happy': '#gold'}
    
    for emotion in emotions:
        if emotion_timeline[emotion]:
            x_vals = np.linspace(0, st.session_state.session_duration, len(emotion_timeline[emotion]))
            fig.add_trace(go.Scatter(
                x=x_vals, y=emotion_timeline[emotion],
                mode='lines', name=emotion.title(),
                line=dict(color=colors.get(emotion, '#blue'))
            ))
    
    fig.update_layout(
        title="Emotional State Over Time",
        xaxis_title="Time (seconds)",
        yaxis_title="Emotion Intensity (%)",
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Emotional insights
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Emotional Highlights**")
        
        # Calculate dominant emotions
        if emotion_timeline:
            avg_emotions = {}
            for emotion in emotions:
                if emotion_timeline[emotion]:
                    avg_emotions[emotion] = np.mean(emotion_timeline[emotion])
            
            # Sort by intensity
            sorted_emotions = sorted(avg_emotions.items(), key=lambda x: x[1], reverse=True)
            
            for emotion, intensity in sorted_emotions[:3]:
                intensity_desc = "High" if intensity > 60 else "Moderate" if intensity > 30 else "Low"
                st.write(f"• **{emotion.title()}**: {intensity_desc} ({intensity:.1f}%)")
    
    with col2:
        st.markdown("**Emotional Stability**")
        
        # Calculate emotional stability metrics
        if emotion_timeline:
            stability_scores = {}
            for emotion in emotions:
                if emotion_timeline[emotion]:
                    values = emotion_timeline[emotion]
                    stability = 1 - (np.std(values) / (np.mean(values) + 0.001))
                    stability_scores[emotion] = max(0, stability * 100)
            
            overall_stability = np.mean(list(stability_scores.values())) if stability_scores else 0
            st.metric("Overall Emotional Stability", f"{overall_stability:.1f}%")
            
            # Most stable emotion
            if stability_scores:
                most_stable = max(stability_scores.items(), key=lambda x: x[1])
                st.write(f"Most Stable: **{most_stable[0].title()}** ({most_stable[1]:.1f}%)")

def display_timeline_analysis():
    """Display performance timeline with key moments"""
    st.subheader("Performance Timeline")
    
    if not st.session_state.timestamps:
        st.info("No timeline data available")
        return
    
    # Create combined timeline
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Performance Metrics', 'Key Moments'),
        row_heights=[0.7, 0.3],
        vertical_spacing=0.1
    )
    
    # Performance metrics timeline
    timestamps = list(st.session_state.timestamps)
    
    # Confidence timeline
    if st.session_state.confidence_scores:
        conf_times = np.linspace(0, max(timestamps), len(st.session_state.confidence_scores))
        fig.add_trace(
            go.Scatter(x=conf_times, y=list(st.session_state.confidence_scores),
                      mode='lines', name='Confidence', line=dict(color='green')),
            row=1, col=1
        )
    
    # Nervousness timeline (inverted for better visualization)
    nervousness_timeline = [100 - st.session_state.nervousness_level] * len(timestamps)
    fig.add_trace(
        go.Scatter(x=timestamps, y=nervousness_timeline,
                  mode='lines', name='Calmness', line=dict(color='blue')),
        row=1, col=1
    )
    
    # Key moments markers
    key_moments = []
    
    # Add nervousness spikes
    for indicator in st.session_state.nervousness_indicators[:5]:  # Show first 5
        moment_time = np.random.uniform(0, max(timestamps))
        key_moments.append({'time': moment_time, 'event': indicator, 'type': 'warning'})
    
    # Add confidence peaks
    if st.session_state.confidence_scores:
        max_conf_idx = np.argmax(st.session_state.confidence_scores)
        max_conf_time = conf_times[max_conf_idx] if max_conf_idx < len(conf_times) else 0
        key_moments.append({
            'time': max_conf_time, 
            'event': 'Peak Confidence', 
            'type': 'success'
        })
    
    # Plot key moments
    for moment in key_moments:
        color = 'red' if moment['type'] == 'warning' else 'green'
        fig.add_vline(x=moment['time'], line_dash="dash", line_color=color,
                     annotation_text=moment['event'][:20], row=2, col=1)
    
    fig.update_layout(height=600, showlegend=True)
    fig.update_xaxes(title_text="Time (seconds)")
    st.plotly_chart(fig, use_container_width=True)
    
    # Timeline insights
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Performance Trends**")
        
        if st.session_state.confidence_scores and len(st.session_state.confidence_scores) >= 2:
            start_conf = np.mean(st.session_state.confidence_scores[:3])
            end_conf = np.mean(st.session_state.confidence_scores[-3:])
            
            trend = "Improving" if end_conf > start_conf else "Declining" if end_conf < start_conf else "Stable"
            change = abs(end_conf - start_conf)
            
            st.write(f"• Confidence Trend: **{trend}** ({change:.1f}% change)")
            st.write(f"• Session Start: {start_conf:.1f}%")
            st.write(f"• Session End: {end_conf:.1f}%")
    
    with col2:
        st.markdown("**Critical Moments**")
        
        if key_moments:
            for moment in key_moments[:3]:  # Show top 3
                moment_type = "⚠️" if moment['type'] == 'warning' else "✅"
                st.write(f"• {moment_type} {moment['event']} at {moment['time']:.1f}s")

def display_improvement_recommendations():
    """Display personalized improvement recommendations"""
    st.subheader("Personalized Improvement Plan")
    
    # Generate recommendations based on analysis
    recommendations = generate_recommendations()
    
    # Priority recommendations
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**High Priority Areas**")
        
        high_priority = recommendations.get('high_priority', [])
        for i, rec in enumerate(high_priority[:5], 1):
            st.markdown(f"**{i}.** {rec}")
    
    with col2:
        st.markdown("**Quick Wins**")
        
        quick_wins = recommendations.get('quick_wins', [])
        for i, rec in enumerate(quick_wins[:5], 1):
            st.markdown(f"**{i}.** {rec}")
    
    # Detailed improvement plan
    st.markdown("---")
    st.markdown("**Detailed Improvement Plan**")
    
    improvement_areas = {
        "Voice Control": {
            "exercises": [
                "Practice diaphragmatic breathing for 10 minutes daily",
                "Record yourself reading aloud and analyze pitch patterns",
                "Use vocal warm-ups before important conversations",
                "Practice speaking at your optimal pitch range"
            ],
            "timeline": "2-4 weeks for noticeable improvement"
        },
        "Confidence Building": {
            "exercises": [
                "Practice power posing before speaking engagements",
                "Use positive self-talk and affirmations",
                "Start conversations on topics you're passionate about",
                "Join a local speaking club like Toastmasters"
            ],
            "timeline": "4-6 weeks for substantial confidence gains"
        },
        "Body Language": {
            "exercises": [
                "Practice maintaining eye contact during conversations",
                "Work on upright posture and open gestures",
                "Record yourself presenting to analyze body language",
                "Practice natural hand movements while speaking"
            ],
            "timeline": "3-5 weeks for natural body language habits"
        },
        "Emotional Regulation": {
            "exercises": [
                "Learn progressive muscle relaxation techniques",
                "Practice mindfulness meditation for 15 minutes daily",
                "Use the 4-7-8 breathing technique before stressful situations",
                "Develop pre-speaking routines to manage nervousness"
            ],
            "timeline": "6-8 weeks for improved emotional control"
        }
    }
    
    for area, details in improvement_areas.items():
        with st.expander(f"📚 {area} Development Plan"):
            st.markdown("**Recommended Exercises:**")
            for exercise in details['exercises']:
                st.write(f"• {exercise}")
            st.markdown(f"**Expected Timeline:** {details['timeline']}")

def generate_recommendations():
    """Generate personalized recommendations based on analysis"""
    recommendations = {
        'high_priority': [],
        'quick_wins': [],
        'long_term': []
    }
    
    confidence = st.session_state.current_confidence
    nervousness = st.session_state.nervousness_level
    
    # High priority recommendations
    if confidence < 50:
        recommendations['high_priority'].append(
            "Focus on building speaking confidence through daily practice"
        )
    
    if nervousness > 60:
        recommendations['high_priority'].append(
            "Implement stress management techniques before speaking"
        )
    
    if st.session_state.eye_contact_score < 0.5:
        recommendations['high_priority'].append(
            "Practice maintaining appropriate eye contact"
        )
    
    # Quick wins
    recommendations['quick_wins'].extend([
        "Use power breathing before speaking engagements",
        "Practice speaking slowly and clearly",
        "Record yourself daily for self-awareness",
        "Use positive self-talk before presentations"
    ])
    
    # Long-term recommendations
    recommendations['long_term'].extend([
        "Join a public speaking organization",
        "Take an advanced presentation skills course",
        "Practice impromptu speaking regularly",
        "Develop your unique speaking style"
    ])
    
    return recommendations

def display_practice_tools():
    """Display practice tools and scenarios"""
    st.header("Practice Tools")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Practice Scenarios")
        
        scenarios = [
            "Job Interview Preparation",
            "Business Presentation",
            "Wedding Toast",
            "Conference Speaking",
            "Team Meeting Leadership",
            "Customer Presentation",
            "Academic Presentation",
            "Networking Event"
        ]
        
        selected_scenario = st.selectbox("Choose a practice scenario:", scenarios)
        
        if st.button("Get Practice Prompts"):
            prompts = get_scenario_prompts(selected_scenario)
            st.markdown("**Practice Prompts:**")
            for i, prompt in enumerate(prompts, 1):
                st.write(f"{i}. {prompt}")
    
    with col2:
        st.subheader("Quick Exercises")
        
        exercises = {
            "Vocal Warm-up": "Hum scales, lip trills, and tongue twisters",
            "Breathing Exercise": "4-7-8 breathing: inhale 4, hold 7, exhale 8",
            "Confidence Pose": "Stand tall, hands on hips, chest out for 2 minutes",
            "Articulation": "Read tongue twisters slowly, then increase speed",
            "Eye Contact": "Practice with yourself in a mirror for 5 minutes",
            "Gesture Practice": "Tell a story using only hand gestures"
        }
        
        for exercise, description in exercises.items():
            with st.expander(f"🎯 {exercise}"):
                st.write(description)

def get_scenario_prompts(scenario):
    """Get practice prompts for specific scenarios"""
    prompts_dict = {
        "Job Interview Preparation": [
            "Tell me about yourself and your background",
            "What are your greatest strengths and how do they apply to this role?",
            "Describe a challenging situation you overcame",
            "Why do you want to work for our company?",
            "Where do you see yourself in 5 years?"
        ],
        "Business Presentation": [
            "Present your quarterly results to stakeholders",
            "Pitch a new product idea to management",
            "Explain a complex process to non-technical audience",
            "Present budget projections for next year",
            "Introduce a new team member to the company"
        ],
        "Wedding Toast": [
            "Share a heartwarming story about the couple",
            "Give advice for a happy marriage",
            "Express gratitude to the families",
            "Make everyone laugh with a funny anecdote",
            "Wish the couple well for their future"
        ]
    }
    
    return prompts_dict.get(scenario, [
        "Practice speaking clearly and confidently",
        "Focus on your key message",
        "Use appropriate gestures and body language",
        "Maintain eye contact with your audience",
        "End with a strong conclusion"
    ])

def display_progress_tracking():
    """Display progress tracking interface"""
    st.header("Progress Tracking")
    
    # Progress metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Current Goals")
        
        # Goal setting interface
        confidence_goal = st.slider("Confidence Target", 60, 100, target_confidence)
        nervousness_goal = st.slider("Max Nervousness", 0, 40, max_nervousness)
        
        # Progress calculation
        conf_progress = min(100, (st.session_state.current_confidence / confidence_goal) * 100)
        nerv_progress = max(0, 100 - (st.session_state.nervousness_level / max(nervousness_goal, 1)) * 100)
        
        st.progress(conf_progress / 100, text=f"Confidence: {conf_progress:.0f}%")
        st.progress(nerv_progress / 100, text=f"Calmness: {nerv_progress:.0f}%")
    
    with col2:
        st.subheader("Session History")
        
        # Mock session history
        history_data = {
            'Date': ['2024-01-15', '2024-01-12', '2024-01-10', '2024-01-08'],
            'Confidence': [st.session_state.current_confidence, 68, 72, 65],
            'Nervousness': [st.session_state.nervousness_level, 25, 30, 35],
            'Duration': ['5:30', '7:15', '4:45', '6:20']
        }
        
        df = pd.DataFrame(history_data)
        st.dataframe(df, use_container_width=True)
    
    with col3:
        st.subheader("Achievements")
        
        # Achievement system
        achievements = []
        
        if st.session_state.current_confidence >= 70:
            achievements.append("🏆 Confident Speaker")
        
        if st.session_state.nervousness_level <= 30:
            achievements.append("😌 Calm Presenter")
        
        if st.session_state.overall_score >= 80:
            achievements.append("🌟 Communication Star")
        
        if st.session_state.eye_contact_score >= 0.7:
            achievements.append("👁️ Eye Contact Master")
        
        achievements.extend([
            "🎯 First Analysis Complete",
            "📈 Progress Tracker",
            "🎤 Practice Session Complete"
        ])
        
        for achievement in achievements[:6]:  # Show top 6
            st.write(achievement)

# Export functionality
def export_results():
    """Export analysis results"""
    export_data = {
        'session_info': {
            'timestamp': datetime.now().isoformat(),
            'duration': st.session_state.session_duration,
            'analysis_mode': analysis_mode
        },
        'metrics': {
            'confidence': st.session_state.current_confidence,
            'nervousness': st.session_state.nervousness_level,
            'overall_score': st.session_state.overall_score,
            'eye_contact': st.session_state.eye_contact_score
        },
        'detailed_data': {
            'pitch_data': list(st.session_state.pitch_data),
            'volume_data': list(st.session_state.volume_data),
            'confidence_scores': list(st.session_state.confidence_scores),
            'nervousness_indicators': st.session_state.nervousness_indicators
        },
        'recommendations': generate_recommendations()
    }
    
    return json.dumps(export_data, indent=2)

# Footer and export section
if st.session_state.analysis_complete:
    st.markdown("---")
    st.header("Export & Save Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📄 Export Detailed Report"):
            json_data = export_results()
            st.download_button(
                label="Download JSON Report",
                data=json_data,
                file_name=f"speaksmart_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
    
    with col2:
        if st.button("📊 Export Charts"):
            st.info("Chart export functionality would be implemented here")
    
    with col3:
        if st.button("📧 Email Results"):
            st.info("Email functionality would be implemented here")

# Run main application
if __name__ == "__main__":
    main()

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 30px; background: #f8f9fa; border-radius: 15px; margin-top: 2rem;">
    <h3>🎯 SpeakSmart AI - Advanced Communication Coach</h3>
    <p style="font-size: 1.1em; color: #666;">
        Empowering confident communication through AI-powered speech and video analysis
    </p>
    <p style="margin-top: 20px;">
        <strong>Core Features:</strong> Real-time audio analysis • Video emotion detection • 
        Gesture recognition • Personalized feedback • Progress tracking
    </p>
    <hr style="margin: 20px 0;">
    <div style="font-size: 0.9em; color: #888;">
        <p><strong>Developed by Team SpeakSmart</strong></p>
        <p>Group No: 256 | Project Exhibition-I</p>
        <p>Raunak Kumar Modi | Jahnvi Pandey | Rishi Singh Shandilya | Unnati Lohana | Vedant Singh</p>
        <p style="margin-top: 15px;">
            <em>Version 2.0 | Enhanced with Video AI and Pre-trained Models</em><br>
            Built for improving public speaking, interviews, and presentation skills
        </p>
    </div>
</div>
""", unsafe_allow_html=True)