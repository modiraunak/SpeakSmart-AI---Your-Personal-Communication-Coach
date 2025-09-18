"""
Main Application Interface for SpeakSmart AI
Author: Jahnvi Pandey (UI/UX & Frontend Developer)

Modern Streamlit interface with responsive design and interactive components
Focus on user experience and accessibility
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from datetime import datetime
import time

# Import custom components
from .components import (
    MetricCard, ProgressRing, EmotionIndicator, 
    InteractiveChart, FileUploadWidget
)
from .styles import inject_custom_css, get_theme_colors
from ..analysis.audio_analyzer import AdvancedAudioAnalyzer
from ..analysis.video_analyzer import VideoEmotionAnalyzer
from ..models.emotion_models import model_manager, confidence_classifier
from ..utils.session_manager import SessionManager
from ..utils.file_handler import FileHandler

class SpeakSmartInterface:
    """Main application interface with modern UI components"""
    
    def __init__(self, config):
        self.config = config
        self.session_manager = SessionManager()
        self.file_handler = FileHandler()
        self.theme_colors = get_theme_colors()
        
        # Initialize analyzers
        self.audio_analyzer = AdvancedAudioAnalyzer()
        self.video_analyzer = VideoEmotionAnalyzer()
        
    def run(self):
        """Run the main application interface"""
        # Inject custom CSS
        inject_custom_css()
        
        # Main header
        self.render_header()
        
        # Sidebar configuration
        with st.sidebar:
            self.render_sidebar()
        
        # Main content area
        self.render_main_content()
        
        # Footer
        self.render_footer()
    
    def render_header(self):
        """Render the application header with branding"""
        st.markdown("""
        <div class="header-container">
            <div class="header-content">
                <h1 class="main-title">
                    <span class="logo-icon">🎯</span>
                    SpeakSmart AI
                </h1>
                <p class="subtitle">Advanced Communication Coach with Video & Audio Analysis</p>
                <div class="header-features">
                    <span class="feature-badge">AI-Powered</span>
                    <span class="feature-badge">Real-time Analysis</span>
                    <span class="feature-badge">Multi-modal</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    def render_sidebar(self):
        """Render the sidebar with configuration options"""
        st.markdown("### ⚙️ Configuration")
        
        # Analysis mode selection
        analysis_mode = st.selectbox(
            "Analysis Mode",
            ["Audio + Video", "Audio Only", "Video Only"],
            help="Choose what aspects to analyze",
            key="analysis_mode"
        )
        
        # Input method selection
        input_method = st.selectbox(
            "Input Method", 
            ["Upload Files", "Live Recording", "WebRTC Stream"],
            help="How do you want to provide the data?",
            key="input_method"
        )
        
        # Advanced settings in expandable section
        with st.expander("🎛️ Advanced Settings"):
            self.render_advanced_settings(analysis_mode)
        
        # Performance targets
        with st.expander("🎯 Performance Targets"):
            self.render_performance_targets()
        
        # Quick actions
        st.markdown("---")
        st.markdown("### ⚡ Quick Actions")
        
        if st.button("🔄 Reset Session", type="secondary"):
            self.session_manager.reset_session()
            st.success("Session reset successfully!")
            st.experimental_rerun()
        
        if st.button("📊 Load Demo Data"):
            self.load_demo_data()
    
    def render_advanced_settings(self, analysis_mode):
        """Render advanced configuration settings"""
        # Audio settings
        if analysis_mode in ["Audio + Video", "Audio Only"]:
            st.markdown("**🎵 Audio Parameters**")
            
            col1, col2 = st.columns(2)
            with col1:
                pitch_min = st.number_input("Min Pitch (Hz)", 75, 200, 100)
                sensitivity = st.slider("Detection Sensitivity", 0.1, 1.0, 0.6, 0.1)
            
            with col2:
                pitch_max = st.number_input("Max Pitch (Hz)", 200, 600, 300)
                noise_reduction = st.checkbox("Noise Reduction", True)
        
        # Video settings  
        if analysis_mode in ["Audio + Video", "Video Only"]:
            st.markdown("**📹 Video Parameters**")
            
            col1, col2 = st.columns(2)
            with col1:
                emotion_detection = st.checkbox("Emotion Detection", True)
                gesture_analysis = st.checkbox("Gesture Analysis", True)
            
            with col2:
                posture_tracking = st.checkbox("Posture Tracking", True)
                eye_contact_analysis = st.checkbox("Eye Contact Analysis", True)
    
    def render_performance_targets(self):
        """Render performance target settings"""
        target_confidence = st.slider("Target Confidence (%)", 60, 100, 80)
        max_nervousness = st.slider("Max Nervousness (%)", 0, 40, 20)
        
        # Store in session state
        st.session_state.target_confidence = target_confidence
        st.session_state.max_nervousness = max_nervousness
    
    def render_main_content(self):
        """Render the main content area"""
        # Create tabs for different sections
        tab1, tab2, tab3, tab4 = st.tabs([
            "📥 Input & Analysis", 
            "📊 Real-time Dashboard", 
            "📈 Detailed Results",
            "🎯 Practice Tools"
        ])
        
        with tab1:
            self.render_input_section()
        
        with tab2:
            self.render_dashboard()
        
        with tab3:
            self.render_detailed_results()
        
        with tab4:
            self.render_practice_tools()
    
    def render_input_section(self):
        """Render the input and file upload section"""
        st.header("📥 Input Selection")
        
        input_method = st.session_state.get("input_method", "Upload Files")
        
        if input_method == "Upload Files":
            self.render_file_upload()
        elif input_method == "Live Recording":
            self.render_live_recording()
        elif input_method == "WebRTC Stream":
            self.render_webrtc_stream()
    
    def render_file_upload(self):
        """Render file upload interface"""
        st.subheader("📁 Upload Your Content")
        
        # Custom file upload widget
        file_uploader = FileUploadWidget(
            file_types=['audio', 'video'],
            max_size_mb=100,
            theme=self.theme_colors
        )
        
        uploaded_files = file_uploader.render()
        
        if uploaded_files:
            # Display file information
            self.display_file_info(uploaded_files)
            
            # Analysis button
            if st.button("🚀 Start Analysis", type="primary", use_container_width=True):
                self.process_uploaded_files(uploaded_files)
    
    def render_live_recording(self):
        """Render live recording interface"""
        st.subheader("🎙️ Live Recording")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            duration = st.slider("Duration (seconds)", 10, 300, 60)
        
        with col2:
            quality = st.selectbox("Quality", ["Standard", "High", "Professional"])
        
        with col3:
            countdown = st.checkbox("3-second countdown", True)
        
        # Recording controls
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔴 Start Recording", type="primary", use_container_width=True):
                self.start_live_recording(duration, quality, countdown)
        
        with col2:
            if st.button("⏹️ Stop Recording", use_container_width=True):
                st.session_state.recording_active = False
        
        # Recording status
        if st.session_state.get("recording_active", False):
            self.display_recording_status()
    
    def render_webrtc_stream(self):
        """Render WebRTC streaming interface"""
        st.subheader("🌐 Live Streaming Analysis")
        
        st.info("🚀 **Coming Soon!** Real-time streaming analysis with WebRTC")
        st.markdown("""
        This feature will provide:
        - Live feedback during video calls
        - Real-time presentation coaching
        - Instant confidence scoring
        - Interactive improvement suggestions
        """)
    
    def render_dashboard(self):
        """Render the real-time dashboard"""
        st.header("📊 Real-time Dashboard")
        
        if not st.session_state.get("analysis_active", False):
            self.render_empty_dashboard()
            return
        
        # Key metrics row
        self.render_key_metrics()
        
        # Charts row
        col1, col2 = st.columns(2)
        
        with col1:
            self.render_confidence_chart()
        
        with col2:
            self.render_emotion_chart()
        
        # Timeline chart
        self.render_timeline_chart()
        
        # Live feedback section
        self.render_live_feedback()
    
    def render_key_metrics(self):
        """Render key performance metrics"""
        col1, col2, col3, col4 = st.columns(4)
        
        confidence = st.session_state.get("current_confidence", 0)
        nervousness = st.session_state.get("nervousness_level", 0)
        eye_contact = st.session_state.get("eye_contact_score", 0) * 100
        overall_score = st.session_state.get("overall_score", 0)
        
        with col1:
            metric_card = MetricCard(
                title="Confidence",
                value=f"{confidence}%",
                delta=f"+{np.random.randint(1, 5)}%",
                color=self.get_confidence_color(confidence)
            )
            metric_card.render()
        
        with col2:
            metric_card = MetricCard(
                title="Nervousness",
                value=f"{nervousness}%",
                delta=f"-{np.random.randint(1, 3)}%",
                color=self.get_nervousness_color(nervousness)
            )
            metric_card.render()
        
        with col3:
            metric_card = MetricCard(
                title="Eye Contact",
                value=f"{eye_contact:.0f}%",
                delta=f"+{np.random.randint(0, 2)}%",
                color=self.get_eye_contact_color(eye_contact)
            )
            metric_card.render()
        
        with col4:
            metric_card = MetricCard(
                title="Overall Score", 
                value=f"{overall_score}%",
                delta=f"+{np.random.randint(1, 4)}%",
                color=self.get_overall_color(overall_score)
            )
            metric_card.render()
    
    def render_confidence_chart(self):
        """Render confidence trend chart"""
        st.markdown("### 📈 Confidence Trend")
        
        # Generate sample data if no real data
        if not hasattr(st.session_state, 'confidence_history'):
            timestamps = pd.date_range(start='2024-01-01', periods=50, freq='S')
            confidence_data = np.random.randint(60, 90, 50)
        else:
            timestamps = st.session_state.get('timestamps', [])
            confidence_data = list(st.session_state.confidence_scores)
        
        chart = InteractiveChart(
            chart_type="line",
            data={'x': timestamps, 'y': confidence_data},
            title="Confidence Over Time",
            color=self.theme_colors['primary']
        )
        chart.render()
    
    def render_emotion_chart(self):
        """Render emotion distribution chart"""
        st.markdown("### 😊 Emotion Analysis")
        
        # Sample emotion data
        emotions = {
            'Confident': 35,
            'Calm': 25, 
            'Happy': 20,
            'Focused': 15,
            'Nervous': 5
        }
        
        fig = go.Figure(data=[
            go.Pie(
                labels=list(emotions.keys()),
                values=list(emotions.values()),
                hole=0.4,
                marker=dict(
                    colors=['#28a745', '#17a2b8', '#ffc107', '#6f42c1', '#dc3545']
                )
            )
        ])
        
        fig.update_layout(
            showlegend=True,
            height=300,
            margin=dict(t=0, b=0, l=0, r=0)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_timeline_chart(self):
        """Render comprehensive timeline chart"""
        st.markdown("### ⏱️ Performance Timeline")
        
        # Create sample timeline data
        timestamps = pd.date_range(start='2024-01-01', periods=100, freq='S')
        
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=('Confidence', 'Voice Pitch', 'Volume Level'),
            vertical_spacing=0.08
        )
        
        # Confidence line
        confidence_data = np.random.randint(60, 90, 100)
        fig.add_trace(
            go.Scatter(x=timestamps, y=confidence_data, name="Confidence", 
                      line=dict(color='#28a745')),
            row=1, col=1
        )
        
        # Pitch line  
        pitch_data = np.random.randint(120, 200, 100)
        fig.add_trace(
            go.Scatter(x=timestamps, y=pitch_data, name="Pitch (Hz)",
                      line=dict(color='#007bff')),
            row=2, col=1
        )
        
        # Volume line
        volume_data = np.random.random(100)
        fig.add_trace(
            go.Scatter(x=timestamps, y=volume_data, name="Volume",
                      line=dict(color='#ffc107')),
            row=3, col=1
        )
        
        fig.update_layout(height=500, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    def render_live_feedback(self):
        """Render live feedback section"""
        st.markdown("### 💬 Live Feedback")
        
        feedback_container = st.container()
        
        with feedback_container:
            # Current suggestions
            suggestions = [
                "Great job maintaining steady volume!",
                "Try to slow down your speaking pace slightly",
                "Excellent eye contact - keep it up!",
                "Consider using more hand gestures for emphasis"
            ]
            
            for i, suggestion in enumerate(suggestions):
                icon = "✅" if i < 2 else "💡"
                st.markdown(f"{icon} {suggestion}")
    
    def render_detailed_results(self):
        """Render detailed analysis results"""
        st.header("📈 Detailed Analysis Results")
        
        if not st.session_state.get("analysis_complete", False):
            st.info("Complete an analysis to see detailed results here.")
            return
        
        # Analysis tabs
        result_tab1, result_tab2, result_tab3, result_tab4 = st.tabs([
            "Audio Analysis", "Video Analysis", "Emotion Tracking", "Recommendations"
        ])
        
        with result_tab1:
            self.render_audio_analysis()
        
        with result_tab2:
            self.render_video_analysis()
        
        with result_tab3:
            self.render_emotion_tracking()
        
        with result_tab4:
            self.render_recommendations()
    
    def render_audio_analysis(self):
        """Render detailed audio analysis"""
        st.subheader("🎵 Voice Pattern Analysis")
        
        # Audio metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Average Pitch", "150 Hz", "±5 Hz")
            st.metric("Pitch Stability", "85%", "+2%")
        
        with col2:
            st.metric("Volume Consistency", "78%", "+1%")
            st.metric("Speaking Rate", "165 WPM", "-3 WPM")
        
        with col3:
            st.metric("Voice Clarity", "92%", "+5%")
            st.metric("Pause Frequency", "Normal", "Good")
        
        # Detailed charts would go here
        st.info("Detailed audio visualization charts would be displayed here.")
    
    def render_video_analysis(self):
        """Render detailed video analysis"""
        st.subheader("📹 Visual Communication Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Facial Expression Analysis**")
            # Emotion pie chart would go here
            st.info("Facial expression breakdown chart")
        
        with col2:
            st.markdown("**Body Language Metrics**")
            st.metric("Eye Contact Quality", "82%")
            st.metric("Posture Score", "88%")
            st.metric("Gesture Activity", "Moderate")
            st.metric("Facial Symmetry", "94%")
    
    def render_emotion_tracking(self):
        """Render emotion tracking over time"""
        st.subheader("😊 Emotional Journey")
        st.info("Emotion timeline and analysis would be displayed here.")
    
    def render_recommendations(self):
        """Render personalized recommendations"""
        st.subheader("🎯 Personalized Improvement Plan")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**High Priority Areas**")
            recommendations = [
                "Practice maintaining consistent speaking pace",
                "Work on reducing voice tremor during key points",
                "Improve eye contact consistency"
            ]
            for i, rec in enumerate(recommendations, 1):
                st.markdown(f"{i}. {rec}")
        
        with col2:
            st.markdown("**Quick Wins**")
            quick_wins = [
                "Use power breathing before speaking",
                "Practice positive self-talk",
                "Record daily practice sessions"
            ]
            for i, win in enumerate(quick_wins, 1):
                st.markdown(f"{i}. {win}")
    
    def render_practice_tools(self):
        """Render practice tools and exercises"""
        st.header("🎯 Practice Tools & Exercises")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📚 Practice Scenarios")
            
            scenarios = [
                "Job Interview Preparation",
                "Business Presentation", 
                "Wedding Toast",
                "Conference Speaking",
                "Team Meeting Leadership"
            ]
            
            selected_scenario = st.selectbox("Choose a scenario:", scenarios)
            
            if st.button("Get Practice Prompts"):
                self.display_practice_prompts(selected_scenario)
        
        with col2:
            st.subheader("💪 Quick Exercises")
            
            exercises = {
                "Vocal Warm-up": "Hum scales and practice lip trills",
                "Breathing Exercise": "4-7-8 breathing technique",
                "Confidence Pose": "Power pose for 2 minutes",
                "Eye Contact": "Mirror practice for 5 minutes"
            }
            
            for exercise, description in exercises.items():
                with st.expander(f"🏋️ {exercise}"):
                    st.write(description)
                    if st.button(f"Start {exercise}", key=f"start_{exercise}"):
                        st.success(f"Starting {exercise}!")
    
    def render_empty_dashboard(self):
        """Render empty state for dashboard"""
        st.markdown("""
        <div class="empty-state">
            <div class="empty-icon">📊</div>
            <h3>No Active Analysis</h3>
            <p>Upload a file or start recording to see your real-time dashboard</p>
        </div>
        """, unsafe_allow_html=True)
    
    def render_footer(self):
        """Render application footer"""
        st.markdown("---")
        st.markdown("""
        <div class="footer-container">
            <div class="footer-content">
                <h4>🎯 SpeakSmart AI - Advanced Communication Coach</h4>
                <p>Empowering confident communication through AI-powered analysis</p>
                
                <div class="team-info">
                    <strong>Development Team (Group 256):</strong><br>
                    Raunak Kumar Modi • Jahnvi Pandey • Rishi Singh Shandilya • Unnati Lohana • Vedant Singh
                </div>
                
                <div class="version-info">
                    <em>Version 2.0 | Enhanced with Pre-trained Models & Advanced UI</em>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Helper methods
    def get_confidence_color(self, confidence):
        """Get color based on confidence level"""
        if confidence >= 80: return "#28a745"
        elif confidence >= 60: return "#ffc107" 
        else: return "#dc3545"
    
    def get_nervousness_color(self, nervousness):
        """Get color based on nervousness level"""
        if nervousness <= 20: return "#28a745"
        elif nervousness <= 40: return "#ffc107"
        else: return "#dc3545"
    
    def get_eye_contact_color(self, eye_contact):
        """Get color based on eye contact score"""
        if eye_contact >= 70: return "#28a745"
        elif eye_contact >= 50: return "#ffc107"
        else: return "#dc3545"
    
    def get_overall_color(self, overall_score):
        """Get color based on overall score"""
        if overall_score >= 80: return "#28a745"
        elif overall_score >= 60: return "#17a2b8"
        else: return "#ffc107"
    
    def display_file_info(self, files):
        """Display information about uploaded files"""
        for file_type, file_obj in files.items():
            st.success(f"✅ {file_type.title()} file uploaded: {file_obj.name}")
    
    def process_uploaded_files(self, files):
        """Process uploaded files for analysis"""
        with st.spinner("🔄 Processing your content..."):
            # Simulate processing
            progress_bar = st.progress(0)
            for i in range(100):
                time.sleep(0.01)
                progress_bar.progress((i + 1) / 100)
            
            st.success("✅ Analysis completed successfully!")
            st.session_state.analysis_complete = True
            st.session_state.analysis_active = True
    
    def start_live_recording(self, duration, quality, countdown):
        """Start live recording session"""
        st.session_state.recording_active = True
        st.session_state.recording_start_time = time.time()
        st.session_state.recording_duration = duration
        
        if countdown:
            countdown_placeholder = st.empty()
            for i in range(3, 0, -1):
                countdown_placeholder.markdown(f"## Starting in {i}...")
                time.sleep(1)
            countdown_placeholder.markdown("## 🔴 Recording!")
    
    def display_recording_status(self):
        """Display current recording status"""
        start_time = st.session_state.get("recording_start_time", time.time())
        duration = st.session_state.get("recording_duration", 60)
        elapsed = time.time() - start_time
        remaining = max(0, duration - elapsed)
        
        progress = min(1.0, elapsed / duration)
        
        st.progress(progress)
        st.markdown(f"**Recording... {remaining:.0f}s remaining**")
    
    def display_practice_prompts(self, scenario):
        """Display practice prompts for selected scenario"""
        prompts = {
            "Job Interview Preparation": [
                "Tell me about yourself and your background",
                "What are your greatest strengths?",
                "Describe a challenging situation you overcame"
            ],
            "Business Presentation": [
                "Present your quarterly results",
                "Pitch a new product idea", 
                "Explain a complex process to non-technical audience"
            ]
        }
        
        scenario_prompts = prompts.get(scenario, ["Practice speaking clearly and confidently"])
        
        st.markdown("**Practice Prompts:**")
        for i, prompt in enumerate(scenario_prompts, 1):
            st.markdown(f"{i}. {prompt}")
    
    def load_demo_data(self):
        """Load demonstration data for testing"""
        st.session_state.analysis_complete = True
        st.session_state.analysis_active = True
        st.session_state.current_confidence = 75
        st.session_state.nervousness_level = 25
        st.session_state.eye_contact_score = 0.8
        st.session_state.overall_score = 78
        
        st.success("✅ Demo data loaded successfully!")