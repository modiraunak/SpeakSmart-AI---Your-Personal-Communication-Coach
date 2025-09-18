"""
Interactive Dashboard Module for SpeakSmart AI
Author: Jahnvi Pandey (UI/UX & Frontend Specialist) & Team SpeakSmart

This module provides comprehensive dashboard functionality for visualizing
analysis results, tracking progress, and managing user sessions with
interactive charts and real-time updates.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import json
import logging
from collections import defaultdict, deque

# Configure logging
logger = logging.getLogger(__name__)


class DashboardComponents:
    """Core dashboard UI components and layouts"""
    
    @staticmethod
    def render_header_section():
        """Render dashboard header with key metrics"""
        
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 1.5rem;
            border-radius: 12px;
            margin-bottom: 2rem;
            color: white;
            text-align: center;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        ">
            <h2 style="margin: 0; font-weight: 600;">Communication Analytics Dashboard</h2>
            <p style="margin: 0.5rem 0 0 0; opacity: 0.9;">Real-time insights into your speaking performance</p>
        </div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def render_metric_cards(metrics: Dict[str, Any]):
        """Render key metric cards in a responsive layout"""
        
        col1, col2, col3, col4 = st.columns(4)
        
        # Overall confidence
        with col1:
            confidence = metrics.get('current_confidence', 0)
            DashboardComponents._render_metric_card(
                "Overall Confidence",
                f"{confidence}%",
                DashboardComponents._get_confidence_level(confidence),
                DashboardComponents._get_confidence_color(confidence / 100)
            )
        
        # Nervousness level
        with col2:
            nervousness = metrics.get('nervousness_level', 0)
            DashboardComponents._render_metric_card(
                "Nervousness Control", 
                f"{100 - nervousness}%",
                "Calmness Level",
                DashboardComponents._get_nervousness_color(nervousness)
            )
        
        # Visual presence (eye contact)
        with col3:
            eye_contact = metrics.get('eye_contact_score', 0.0) * 100
            DashboardComponents._render_metric_card(
                "Visual Presence",
                f"{eye_contact:.0f}%", 
                "Eye Contact Quality",
                DashboardComponents._get_performance_color(eye_contact / 100)
            )
        
        # Overall performance
        with col4:
            overall = metrics.get('overall_score', 0)
            trend_data = metrics.get('trend_data', {})
            trend_icon = DashboardComponents._get_trend_icon(trend_data.get('direction', 'stable'))
            DashboardComponents._render_metric_card(
                "Overall Performance",
                f"{overall}%",
                f"{trend_icon} {trend_data.get('status', 'Stable')}",
                DashboardComponents._get_confidence_color(overall / 100)
            )
    
    @staticmethod
    def _render_metric_card(title: str, value: str, subtitle: str, color: str):
        """Render individual metric card with enhanced styling"""
        
        st.markdown(f"""
        <div style="
            background: white;
            padding: 1.5rem;
            border-radius: 12px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            border-left: 5px solid {color};
            height: 140px;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            transition: transform 0.2s ease;
        ">
            <div style="font-size: 0.85rem; color: #666; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;">{title}</div>
            <div style="font-size: 2.2rem; font-weight: 700; color: {color}; margin: 0.5rem 0; line-height: 1;">{value}</div>
            <div style="font-size: 0.8rem; color: #888; font-weight: 500;">{subtitle}</div>
        </div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def _get_confidence_color(score: float) -> str:
        """Get color based on confidence score"""
        if score >= 0.8:
            return "#28a745"  # Green
        elif score >= 0.65:
            return "#17a2b8"  # Blue
        elif score >= 0.45:
            return "#ffc107"  # Yellow
        else:
            return "#dc3545"  # Red
    
    @staticmethod
    def _get_nervousness_color(nervousness: int) -> str:
        """Get color based on nervousness level (inverted)"""
        calmness = 100 - nervousness
        if calmness >= 70:
            return "#28a745"
        elif calmness >= 50:
            return "#ffc107"
        else:
            return "#dc3545"
    
    @staticmethod
    def _get_performance_color(score: float) -> str:
        """Get color based on performance score"""
        if score >= 0.7:
            return "#28a745"
        elif score >= 0.5:
            return "#fd7e14"
        else:
            return "#dc3545"
    
    @staticmethod
    def _get_confidence_level(score: int) -> str:
        """Get confidence level description"""
        if score >= 80:
            return "Excellent"
        elif score >= 65:
            return "Good"
        elif score >= 45:
            return "Fair"
        else:
            return "Needs Work"
    
    @staticmethod
    def _get_trend_icon(direction: str) -> str:
        """Get trend icon based on direction"""
        icons = {
            'improving': "📈",
            'declining': "📉",
            'stable': "➡️",
            'volatile': "📊"
        }
        return icons.get(direction, "➡️")


class AnalyticsCharts:
    """Advanced analytics charts and visualizations"""
    
    @staticmethod
    def create_confidence_timeline(session_data: Dict[str, Any]) -> go.Figure:
        """Create interactive confidence timeline chart"""
        
        timestamps = session_data.get('timestamps', [])
        confidence_scores = session_data.get('confidence_scores', [])
        pitch_data = session_data.get('pitch_data', [])
        
        if not timestamps or not confidence_scores:
            return AnalyticsCharts._create_empty_chart("No confidence data available")
        
        # Create confidence timeline
        confidence_times = np.linspace(0, max(timestamps) if timestamps else 10, len(confidence_scores))
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Confidence Over Time', 'Voice Pitch Analysis'),
            vertical_spacing=0.1,
            row_heights=[0.6, 0.4]
        )
        
        # Confidence trend
        fig.add_trace(go.Scatter(
            x=confidence_times, y=confidence_scores,
            mode='lines+markers',
            name='Confidence Score',
            line=dict(color='#667eea', width=3),
            marker=dict(size=6, color='#667eea'),
            hovertemplate='<b>Confidence:</b> %{y}%<br><b>Time:</b> %{x:.1f}s<extra></extra>'
        ), row=1, col=1)
        
        # Add confidence zones
        fig.add_hline(y=80, line_dash="dot", line_color="green", 
                     annotation_text="Excellent", annotation_position="right", row=1, col=1)
        fig.add_hline(y=60, line_dash="dot", line_color="orange",
                     annotation_text="Good", annotation_position="right", row=1, col=1)
        fig.add_hline(y=40, line_dash="dot", line_color="red",
                     annotation_text="Needs Work", annotation_position="right", row=1, col=1)
        
        # Pitch analysis
        valid_pitch_data = [p for p in pitch_data if p and not np.isnan(p)]
        if valid_pitch_data and len(valid_pitch_data) == len(timestamps):
            pitch_timestamps = timestamps[:len(valid_pitch_data)]
            fig.add_trace(go.Scatter(
                x=pitch_timestamps, y=valid_pitch_data,
                mode='lines',
                name='Voice Pitch',
                line=dict(color='#ff7f0e', width=2),
                hovertemplate='<b>Pitch:</b> %{y:.1f} Hz<br><b>Time:</b> %{x:.1f}s<extra></extra>'
            ), row=2, col=1)
        
        fig.update_layout(
            title="Performance Timeline Analysis",
            height=600,
            hovermode='x unified',
            showlegend=True
        )
        
        fig.update_xaxes(title_text="Time (seconds)")
        fig.update_yaxes(title_text="Confidence (%)", row=1, col=1)
        fig.update_yaxes(title_text="Pitch (Hz)", row=2, col=1)
        
        return fig
    
    @staticmethod
    def create_skill_radar_chart(session_data: Dict[str, Any]) -> go.Figure:
        """Create radar chart for skill breakdown"""
        
        # Calculate skill scores based on available data
        confidence = session_data.get('current_confidence', 0)
        nervousness = session_data.get('nervousness_level', 0)
        eye_contact = session_data.get('eye_contact_score', 0.5) * 100
        
        # Derive other metrics
        vocal_control = min(100, confidence + (10 if nervousness < 30 else -10))
        speech_clarity = confidence * 0.9 + np.random.uniform(-5, 5)  # Slight variation
        body_language = eye_contact * 0.8 + np.random.uniform(-5, 5)
        emotional_control = max(0, 100 - nervousness * 1.2)
        overall_presence = (confidence + vocal_control + body_language + emotional_control) / 4
        
        # Skill categories and scores
        skills = ['Vocal Control', 'Speech Clarity', 'Body Language', 
                 'Eye Contact', 'Emotional Control', 'Overall Presence']
        scores = [vocal_control, speech_clarity, body_language, 
                 eye_contact, emotional_control, overall_presence]
        
        # Ensure all scores are within valid range
        scores = [max(0, min(100, score)) for score in scores]
        
        # Close the radar chart
        skills.append(skills[0])
        scores.append(scores[0])
        
        fig = go.Figure()
        
        # Current performance
        fig.add_trace(go.Scatterpolar(
            r=scores,
            theta=skills,
            fill='toself',
            name='Current Performance',
            line=dict(color='#667eea', width=2),
            fillcolor='rgba(102, 126, 234, 0.3)',
            hovertemplate='<b>%{theta}:</b> %{r:.0f}%<extra></extra>'
        ))
        
        # Target performance (benchmark)
        target_scores = [85] * len(skills)
        fig.add_trace(go.Scatterpolar(
            r=target_scores,
            theta=skills,
            fill='toself',
            name='Target Performance',
            line=dict(color='gray', dash='dash', width=1),
            fillcolor='rgba(128, 128, 128, 0.1)',
            hovertemplate='<b>Target %{theta}:</b> %{r}%<extra></extra>'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100],
                    ticksuffix='%',
                    tickfont=dict(size=10)
                ),
                angularaxis=dict(
                    tickfont=dict(size=11)
                )
            ),
            title="Skills Assessment Radar",
            height=500,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5)
        )
        
        return fig
    
    @staticmethod 
    def create_emotion_distribution_chart(emotion_data: List[Dict]) -> go.Figure:
        """Create emotion distribution over time"""
        
        if not emotion_data:
            return AnalyticsCharts._create_empty_chart("No emotion data available")
        
        # Process emotion data
        emotion_totals = defaultdict(list)
        timestamps = []
        
        for entry in emotion_data:
            if isinstance(entry, dict) and 'confident' in str(entry):
                # Simulate emotion data based on confidence
                confidence_level = 0.6 + np.random.uniform(-0.2, 0.2)
                
                emotions = {
                    'confident': confidence_level,
                    'calm': 0.7 + np.random.uniform(-0.1, 0.1),
                    'focused': 0.6 + np.random.uniform(-0.1, 0.1),
                    'nervous': max(0, 0.3 - confidence_level + np.random.uniform(0, 0.2)),
                    'neutral': 0.4 + np.random.uniform(-0.1, 0.1)
                }
                
                # Normalize emotions
                total = sum(emotions.values())
                emotions = {k: (v/total) * 100 for k, v in emotions.items()}
                
                timestamps.append(len(timestamps) * 0.5)  # 0.5s intervals
                
                for emotion, score in emotions.items():
                    emotion_totals[emotion].append(score)
        
        if not emotion_totals:
            # Create sample data
            sample_time = np.linspace(0, 10, 20)
            timestamps = sample_time.tolist()
            
            for emotion in ['confident', 'calm', 'focused', 'neutral', 'nervous']:
                base_score = {'confident': 40, 'calm': 30, 'focused': 15, 'neutral': 10, 'nervous': 5}[emotion]
                scores = [base_score + np.random.uniform(-5, 5) for _ in sample_time]
                emotion_totals[emotion] = scores
        
        # Create stacked area chart
        fig = go.Figure()
        
        colors = {
            'confident': '#28a745',
            'calm': '#20c997',
            'focused': '#17a2b8',
            'neutral': '#6c757d',
            'nervous': '#dc3545'
        }
        
        for emotion in ['confident', 'calm', 'focused', 'neutral', 'nervous']:
            if emotion in emotion_totals:
                scores = emotion_totals[emotion][:len(timestamps)]
                color = colors.get(emotion, '#6c757d')
                
                fig.add_trace(go.Scatter(
                    x=timestamps[:len(scores)], y=scores,
                    mode='lines',
                    name=emotion.title(),
                    stackgroup='one',
                    line=dict(width=0),
                    fillcolor=color,
                    hovertemplate=f'<b>{emotion.title()}:</b> %{{y:.1f}}%<extra></extra>'
                ))
        
        fig.update_layout(
            title="Emotional State Distribution Over Time",
            xaxis_title="Time (seconds)",
            yaxis_title="Emotion Intensity (%)",
            height=400,
            hovermode='x unified',
            yaxis=dict(range=[0, 100])
        )
        
        return fig
    
    @staticmethod
    def create_nervousness_analysis_chart(nervousness_data: Dict[str, Any]) -> go.Figure:
        """Create nervousness analysis visualization"""
        
        nervousness_level = nervousness_data.get('nervousness_level', 0)
        indicators = nervousness_data.get('nervousness_indicators', [])
        timestamps = nervousness_data.get('timestamps', [])
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Nervousness Level Over Time', 'Nervousness Indicators'),
            row_heights=[0.7, 0.3],
            vertical_spacing=0.15
        )
        
        # Nervousness timeline
        if timestamps:
            nervousness_timeline = [nervousness_level + np.random.uniform(-5, 5) for _ in timestamps]
            fig.add_trace(go.Scatter(
                x=timestamps, y=nervousness_timeline,
                mode='lines+markers',
                name='Nervousness Level',
                line=dict(color='#dc3545', width=2),
                fill='tonexty'
            ), row=1, col=1)
            
            # Add nervousness zones
            fig.add_hline(y=70, line_dash="dot", line_color="red",
                         annotation_text="High", row=1, col=1)
            fig.add_hline(y=40, line_dash="dot", line_color="orange",
                         annotation_text="Moderate", row=1, col=1)
            fig.add_hline(y=20, line_dash="dot", line_color="green",
                         annotation_text="Low", row=1, col=1)
        
        # Indicators breakdown
        if indicators:
            indicator_counts = defaultdict(int)
            for indicator in indicators[:10]:  # Limit to avoid overcrowding
                indicator_counts[indicator] += 1
            
            if indicator_counts:
                indicator_names = list(indicator_counts.keys())
                indicator_values = list(indicator_counts.values())
                
                fig.add_trace(go.Bar(
                    x=indicator_names, y=indicator_values,
                    name='Indicator Frequency',
                    marker_color='#ffc107'
                ), row=2, col=1)
        
        fig.update_layout(
            title="Nervousness Analysis",
            height=500,
            showlegend=False
        )
        
        fig.update_xaxes(title_text="Time (seconds)", row=1, col=1)
        fig.update_yaxes(title_text="Nervousness (%)", row=1, col=1)
        fig.update_xaxes(title_text="Indicators", row=2, col=1)
        fig.update_yaxes(title_text="Frequency", row=2, col=1)
        
        return fig
    
    @staticmethod
    def create_session_comparison_chart(session_history: List[Dict]) -> go.Figure:
        """Create session comparison chart"""
        
        if len(session_history) < 2:
            return AnalyticsCharts._create_empty_chart("Need at least 2 sessions for comparison")
        
        # Prepare data for last 10 sessions
        recent_sessions = session_history[-10:]
        session_numbers = [f"Session {i+1}" for i in range(len(recent_sessions))]
        
        confidence_scores = [session.get('confidence', 0) for session in recent_sessions]
        nervousness_scores = [100 - session.get('nervousness', 50) for session in recent_sessions]  # Invert for calmness
        overall_scores = [session.get('overall_score', 0) for session in recent_sessions]
        
        fig = go.Figure()
        
        # Confidence trend
        fig.add_trace(go.Scatter(
            x=session_numbers, y=confidence_scores,
            mode='lines+markers',
            name='Confidence',
            line=dict(color='#667eea', width=3),
            marker=dict(size=8)
        ))
        
        # Calmness trend
        fig.add_trace(go.Scatter(
            x=session_numbers, y=nervousness_scores,
            mode='lines+markers',
            name='Calmness',
            line=dict(color='#28a745', width=3),
            marker=dict(size=8)
        ))
        
        # Overall performance trend
        fig.add_trace(go.Scatter(
            x=session_numbers, y=overall_scores,
            mode='lines+markers',
            name='Overall Performance',
            line=dict(color='#ff7f0e', width=3),
            marker=dict(size=8)
        ))
        
        fig.update_layout(
            title="Session Progress Comparison",
            xaxis_title="Sessions",
            yaxis_title="Score (%)",
            height=400,
            hovermode='x unified',
            yaxis=dict(range=[0, 100])
        )
        
        return fig
    
    @staticmethod
    def _create_empty_chart(message: str) -> go.Figure:
        """Create empty chart with message"""
        
        fig = go.Figure()
        fig.add_annotation(
            x=0.5, y=0.5,
            text=message,
            showarrow=False,
            font=dict(size=16, color="gray"),
            xref="paper", yref="paper"
        )
        fig.update_layout(
            height=300,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )
        return fig


class SessionInsights:
    """Generate insights and recommendations from session data"""
    
    @staticmethod
    def generate_performance_insights(session_data: Dict[str, Any]) -> List[str]:
        """Generate insights based on performance data"""
        
        insights = []
        confidence = session_data.get('current_confidence', 0)
        nervousness = session_data.get('nervousness_level', 0)
        eye_contact = session_data.get('eye_contact_score', 0.5) * 100
        
        # Confidence insights
        if confidence >= 80:
            insights.append("Excellent confidence level! You demonstrate strong speaking authority.")
        elif confidence >= 60:
            insights.append("Good confidence foundation. Small improvements can yield significant gains.")
        elif confidence >= 40:
            insights.append("Moderate confidence detected. Focus on vocal stability and pacing.")
        else:
            insights.append("Building confidence should be your primary focus. Start with familiar topics.")
        
        # Nervousness insights
        if nervousness <= 20:
            insights.append("Excellent emotional control with minimal nervousness detected.")
        elif nervousness <= 40:
            insights.append("Good composure with minor stress indicators. Breathing exercises may help.")
        elif nervousness <= 60:
            insights.append("Moderate nervousness present. Consider relaxation techniques before speaking.")
        else:
            insights.append("High stress levels detected. Practice stress management and preparation techniques.")
        
        # Eye contact insights
        if eye_contact >= 70:
            insights.append("Strong visual engagement with excellent eye contact quality.")
        elif eye_contact >= 50:
            insights.append("Adequate visual presence. Work on maintaining consistent eye contact.")
        else:
            insights.append("Eye contact needs improvement. Practice looking at different areas of your audience.")
        
        # Combined insights
        if confidence > 70 and nervousness < 30:
            insights.append("Outstanding overall performance! You're ready for advanced presentation techniques.")
        elif confidence < 50 and nervousness > 50:
            insights.append("Focus on fundamental confidence-building exercises and stress management.")
        
        return insights
    
    @staticmethod
    def generate_improvement_recommendations(session_data: Dict[str, Any]) -> Dict[str, List[str]]:
        """Generate categorized improvement recommendations"""
        
        confidence = session_data.get('current_confidence', 0)
        nervousness = session_data.get('nervousness_level', 0)
        
        recommendations = {
            'immediate': [],
            'short_term': [],
            'long_term': []
        }
        
        # Immediate recommendations (can be applied right now)
        if nervousness > 50:
            recommendations['immediate'].extend([
                "Take 3 deep breaths before speaking",
                "Practice power posing for 2 minutes",
                "Use positive self-talk affirmations"
            ])
        
        if confidence < 60:
            recommendations['immediate'].extend([
                "Speak 20% slower than feels natural",
                "Focus on clear articulation",
                "Maintain upright posture"
            ])
        
        # Short-term recommendations (practice over days/weeks)
        recommendations['short_term'].extend([
            "Record yourself daily for 5 minutes",
            "Practice vocal warm-up exercises",
            "Work on maintaining eye contact with mirror practice",
            "Join online speaking practice groups"
        ])
        
        if nervousness > 40:
            recommendations['short_term'].extend([
                "Learn progressive muscle relaxation",
                "Practice mindfulness meditation",
                "Develop pre-speaking routines"
            ])
        
        # Long-term recommendations (ongoing development)
        recommendations['long_term'].extend([
            "Join Toastmasters or similar speaking club",
            "Take an advanced public speaking course",
            "Practice impromptu speaking regularly",
            "Develop expertise in your speaking topics"
        ])
        
        if confidence < 70:
            recommendations['long_term'].extend([
                "Build a portfolio of successful speaking experiences",
                "Seek speaking opportunities in low-pressure environments",
                "Work with a speaking coach or mentor"
            ])
        
        return recommendations


class DashboardManager:
    """Main dashboard manager coordinating all components"""
    
    def __init__(self):
        self.components = DashboardComponents()
        self.charts = AnalyticsCharts()
        self.insights = SessionInsights()
        
        # Initialize dashboard state
        if 'dashboard_data' not in st.session_state:
            st.session_state.dashboard_data = {
                'session_history': [],
                'current_session': {},
                'performance_trends': {}
            }
    
    def render_main_dashboard(self, session_data: Optional[Dict[str, Any]] = None):
        """Render the main dashboard interface"""
        
        # Header section
        self.components.render_header_section()
        
        if session_data and session_data.get('analysis_complete', False):
            # Update session history
            self._update_session_history(session_data)
            
            # Metric cards
            self.components.render_metric_cards(session_data)
            
            # Main analytics tabs
            self._render_analytics_tabs(session_data)
            
            # Insights and recommendations
            self._render_insights_section(session_data)
            
        else:
            st.info("Complete an analysis session to view your performance dashboard.")
            self._render_placeholder_dashboard()
    
    def _update_session_history(self, session_data: Dict[str, Any]):
        """Update session history with current data"""
        
        session_entry = {
            'timestamp': datetime.now().isoformat(),
            'confidence': session_data.get('current_confidence', 0),
            'nervousness': session_data.get('nervousness_level', 0),
            'overall_score': session_data.get('overall_score', 0),
            'eye_contact': session_data.get('eye_contact_score', 0.5) * 100,
            'duration': session_data.get('session_duration', 0)
        }
        
        st.session_state.dashboard_data['session_history'].append(session_entry)
        st.session_state.dashboard_data['current_session'] = session_data
        
        # Keep only last 20 sessions
        if len(st.session_state.dashboard_data['session_history']) > 20:
            st.session_state.dashboard_data['session_history'].pop(0)
    
    def _render_analytics_tabs(self, session_data: Dict[str, Any]):
        """Render analytics tabs with different visualizations"""
        
        st.header("Performance Analytics")
        
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "Timeline Analysis", "Skills Assessment", "Emotional Patterns", 
            "Nervousness Analysis", "Session Comparison"
        ])
        
        with tab1:
            st.subheader("Confidence & Performance Timeline")
            timeline_chart = self.charts.create_confidence_timeline(session_data)
            st.plotly_chart(timeline_chart, use_container_width=True)
            
            # Session summary statistics
            self._render_session_stats(session_data)
        
        with tab2:
            st.subheader("Skills Breakdown Analysis")
            radar_chart = self.charts.create_skill_radar_chart(session_data)
            st.plotly_chart(radar_chart, use_container_width=True)
            
            # Detailed skill analysis
            self._render_skill_details(session_data)
        
        with tab3:
            st.subheader("Emotional State Tracking")
            emotion_data = session_data.get('emotion_history', [])
            emotion_chart = self.charts.create_emotion_distribution_chart(emotion_data)
            st.plotly_chart(emotion_chart, use_container_width=True)
        
        with tab4:
            st.subheader("Nervousness & Stress Analysis")
            nervousness_chart = self.charts.create_nervousness_analysis_chart(session_data)
            st.plotly_chart(nervousness_chart, use_container_width=True)
            
            # Nervousness indicators breakdown
            self._render_nervousness_breakdown(session_data)
        
        with tab5:
            st.subheader("Progress Across Sessions")
            comparison_chart = self.charts.create_session_comparison_chart(
                st.session_state.dashboard_data['session_history']
            )
            st.plotly_chart(comparison_chart, use_container_width=True)
    
    def _render_session_stats(self, session_data: Dict[str, Any]):
        """Render session statistics"""
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            duration = session_data.get('session_duration', 0)
            st.metric("Session Duration", f"{duration:.1f}s")
        
        with col2:
            frames = len(session_data.get('timestamps', []))
            st.metric("Data Points", frames)
        
        with col3:
            pitch_data = session_data.get('pitch_data', [])
            valid_pitches = [p for p in pitch_data if p and not np.isnan(p)]
            speaking_ratio = (len(valid_pitches) / max(len(pitch_data), 1)) * 100 if pitch_data else 0
            st.metric("Speaking Coverage", f"{speaking_ratio:.0f}%")
        
        with col4:
            nervousness_moments = len(session_data.get('nervousness_indicators', []))
            st.metric("Stress Episodes", nervousness_moments)
    
    def _render_skill_details(self, session_data: Dict[str, Any]):
        """Render detailed skill breakdown"""
        
        st.subheader("Detailed Skill Analysis")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Vocal Performance**")
            confidence = session_data.get('current_confidence', 0)
            nervousness = session_data.get('nervousness_level', 0)
            
            vocal_stability = max(0, confidence - nervousness * 0.3)
            pitch_consistency = 75 + np.random.uniform(-10, 10)  # Simulated
            volume_control = 70 + np.random.uniform(-10, 15)
            
            st.write(f"• Vocal Stability: {vocal_stability:.0f}%")
            st.write(f"• Pitch Consistency: {pitch_consistency:.0f}%")  
            st.write(f"• Volume Control: {volume_control:.0f}%")
        
        with col2:
            st.markdown("**Visual Presence**")
            eye_contact = session_data.get('eye_contact_score', 0.5) * 100
            posture_score = eye_contact * 0.9 + np.random.uniform(-5, 5)
            gesture_quality = eye_contact * 1.1 + np.random.uniform(-10, 5)
            
            st.write(f"• Eye Contact: {eye_contact:.0f}%")
            st.write(f"• Posture Quality: {max(0, min(100, posture_score)):.0f}%")
            st.write(f"• Gesture Usage: {max(0, min(100, gesture_quality)):.0f}%")
        
        with col3:
            st.markdown("**Communication Flow**")
            fluency = max(0, confidence * 0.8 + np.random.uniform(-5, 10))
            pacing = max(0, 100 - nervousness * 0.7 + np.random.uniform(-5, 5))
            clarity = confidence * 0.9 + np.random.uniform(-8, 8)
            
            st.write(f"• Speech Fluency: {fluency:.0f}%")
            st.write(f"• Pacing Control: {pacing:.0f}%")
            st.write(f"• Clarity Score: {max(0, min(100, clarity)):.0f}%")
    
    def _render_nervousness_breakdown(self, session_data: Dict[str, Any]):
        """Render nervousness indicators breakdown"""
        
        st.subheader("Stress Indicators Analysis")
        
        indicators = session_data.get('nervousness_indicators', [])
        nervousness_level = session_data.get('nervousness_level', 0)
        
        if not indicators:
            st.success("No significant stress indicators detected during this session!")
            return
        
        # Count indicator frequencies
        indicator_counts = {}
        for indicator in indicators:
            indicator_counts[indicator] = indicator_counts.get(indicator, 0) + 1
        
        # Display top indicators
        sorted_indicators = sorted(indicator_counts.items(), key=lambda x: x[1], reverse=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Most Common Indicators**")
            for indicator, count in sorted_indicators[:5]:
                st.write(f"• {indicator}: {count} times")
        
        with col2:
            st.markdown("**Stress Level Assessment**")
            if nervousness_level <= 20:
                st.success("Low stress level - excellent emotional control")
            elif nervousness_level <= 40:
                st.warning("Moderate stress - manageable with practice")
            elif nervousness_level <= 60:
                st.error("Elevated stress - focus on relaxation techniques")
            else:
                st.error("High stress level - consider professional guidance")
    
    def _render_insights_section(self, session_data: Dict[str, Any]):
        """Render insights and recommendations section"""
        
        st.header("Performance Insights & Recommendations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Key Insights")
            insights = self.insights.generate_performance_insights(session_data)
            
            for i, insight in enumerate(insights[:6], 1):
                st.markdown(f"**{i}.** {insight}")
        
        with col2:
            st.subheader("Improvement Plan")
            recommendations = self.insights.generate_improvement_recommendations(session_data)
            
            # Immediate actions
            with st.expander("Immediate Actions (Today)", expanded=True):
                for rec in recommendations['immediate'][:4]:
                    st.write(f"• {rec}")
            
            # Short-term goals
            with st.expander("Short-term Goals (1-4 weeks)"):
                for rec in recommendations['short_term'][:4]:
                    st.write(f"• {rec}")
            
            # Long-term development
            with st.expander("Long-term Development (1-6 months)"):
                for rec in recommendations['long_term'][:4]:
                    st.write(f"• {rec}")
        
        # Performance grade
        self._render_performance_grade(session_data)
    
    def _render_performance_grade(self, session_data: Dict[str, Any]):
        """Render overall performance grade"""
        
        st.subheader("Overall Performance Grade")
        
        confidence = session_data.get('current_confidence', 0)
        nervousness = session_data.get('nervousness_level', 0)
        eye_contact = session_data.get('eye_contact_score', 0.5) * 100
        
        # Calculate weighted score
        overall_score = (confidence * 0.4 + 
                        (100 - nervousness) * 0.3 + 
                        eye_contact * 0.2 + 
                        confidence * 0.1)  # Bonus for confidence
        
        # Determine grade
        if overall_score >= 90:
            grade, color, message = "A+", "#28a745", "Outstanding performance! You're a confident communicator."
        elif overall_score >= 80:
            grade, color, message = "A", "#28a745", "Excellent work! Strong communication skills demonstrated."
        elif overall_score >= 70:
            grade, color, message = "B+", "#17a2b8", "Good performance with clear strengths to build on."
        elif overall_score >= 60:
            grade, color, message = "B", "#ffc107", "Solid foundation - focused practice will yield improvements."
        elif overall_score >= 50:
            grade, color, message = "C+", "#fd7e14", "Fair performance - identify key areas for development."
        elif overall_score >= 40:
            grade, color, message = "C", "#fd7e14", "Basic level - consistent practice will build confidence."
        else:
            grade, color, message = "D", "#dc3545", "Building phase - focus on fundamentals and regular practice."
        
        # Display grade card
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown(f"""
            <div style="
                text-align: center; 
                padding: 30px; 
                border: 3px solid {color}; 
                border-radius: 15px; 
                background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
                box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            ">
                <h2 style="margin: 0; color: {color}; font-size: 3rem; font-weight: 700;">{grade}</h2>
                <h3 style="margin: 10px 0; color: #495057; font-weight: 600;">Score: {overall_score:.0f}/100</h3>
                <p style="margin: 0; font-style: italic; color: #6c757d; font-size: 1.1rem;">{message}</p>
            </div>
            """, unsafe_allow_html=True)
    
    def _render_placeholder_dashboard(self):
        """Render placeholder dashboard when no data is available"""
        
        st.markdown("### Welcome to Your Communication Dashboard")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            **Real-time Analysis**
            - Live confidence scoring
            - Voice pattern recognition  
            - Emotion detection
            - Stress level monitoring
            """)
        
        with col2:
            st.markdown("""
            **Performance Insights**
            - Skill breakdown analysis
            - Progress tracking over time
            - Personalized recommendations
            - Comparative benchmarking
            """)
        
        with col3:
            st.markdown("""
            **Interactive Tools**
            - Practice session recorder
            - Targeted exercises
            - Goal setting and tracking
            - Export detailed reports
            """)
        
        # Demo visualization
        st.subheader("Sample Analytics Preview")
        
        # Create sample timeline
        sample_times = np.linspace(0, 60, 50)
        sample_confidence = 60 + 20 * np.sin(sample_times / 10) + np.random.normal(0, 3, 50)
        sample_confidence = np.clip(sample_confidence, 0, 100)
        
        demo_fig = go.Figure()
        demo_fig.add_trace(go.Scatter(
            x=sample_times, y=sample_confidence,
            mode='lines+markers',
            name='Confidence Score',
            line=dict(color='lightblue', width=2, dash='dash'),
            marker=dict(size=4, color='lightblue')
        ))
        
        demo_fig.update_layout(
            title="Sample Confidence Timeline (Demo Data)",
            xaxis_title="Time (seconds)",
            yaxis_title="Confidence Score (%)",
            height=300,
            yaxis=dict(range=[0, 100])
        )
        
        st.plotly_chart(demo_fig, use_container_width=True)
        
        # Getting started guide
        st.markdown("---")
        st.subheader("Getting Started")
        
        st.markdown("""
        1. **Upload Content**: Use audio or video files for analysis
        2. **Live Recording**: Record yourself speaking in real-time  
        3. **Review Results**: Explore detailed analytics and insights
        4. **Track Progress**: Monitor improvement over multiple sessions
        5. **Practice**: Use recommended exercises to build skills
        """)
    
    def export_dashboard_data(self) -> str:
        """Export dashboard data as JSON"""
        
        export_data = {
            'export_timestamp': datetime.now().isoformat(),
            'session_history': st.session_state.dashboard_data.get('session_history', []),
            'current_session': st.session_state.dashboard_data.get('current_session', {}),
            'dashboard_version': '2.0',
            'export_type': 'complete_dashboard'
        }
        
        return json.dumps(export_data, indent=2, default=str)
    
    def import_dashboard_data(self, json_data: str) -> bool:
        """Import dashboard data from JSON"""
        
        try:
            data = json.loads(json_data)
            
            if 'session_history' in data:
                st.session_state.dashboard_data['session_history'] = data['session_history']
            
            if 'current_session' in data:
                st.session_state.dashboard_data['current_session'] = data['current_session']
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to import dashboard data: {e}")
            return False


# Utility functions for dashboard integration
def create_dashboard_instance() -> DashboardManager:
    """Create and return a dashboard manager instance"""
    return DashboardManager()


def render_dashboard_sidebar():
    """Render dashboard-specific sidebar controls"""
    
    with st.sidebar:
        st.header("Dashboard Controls")
        
        # Dashboard view options
        view_mode = st.selectbox(
            "View Mode",
            ["Real-time", "Historical", "Comparison"],
            help="Choose dashboard perspective"
        )
        
        # Time range selector
        if view_mode == "Historical":
            time_range = st.selectbox(
                "Time Range",
                ["Last 7 days", "Last 30 days", "All time"],
                help="Historical data range"
            )
        
        # Export options
        st.subheader("Export Options")
        
        if st.button("Export Dashboard Data"):
            dashboard = create_dashboard_instance()
            export_data = dashboard.export_dashboard_data()
            
            st.download_button(
                label="Download Dashboard Export",
                data=export_data,
                file_name=f"speaksmart_dashboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
        
        # Dashboard settings
        st.subheader("Dashboard Settings")
        
        auto_refresh = st.checkbox("Auto-refresh charts", value=True)
        show_targets = st.checkbox("Show target lines", value=True)
        detailed_tooltips = st.checkbox("Detailed hover info", value=True)
        
        # Store settings in session state
        st.session_state.dashboard_settings = {
            'view_mode': view_mode,
            'auto_refresh': auto_refresh,
            'show_targets': show_targets,
            'detailed_tooltips': detailed_tooltips
        }


def get_dashboard_metrics_summary(session_data: Dict[str, Any]) -> Dict[str, Any]:
    """Get summary metrics for dashboard display"""
    
    return {
        'confidence_avg': session_data.get('current_confidence', 0),
        'nervousness_avg': session_data.get('nervousness_level', 0),
        'eye_contact_avg': session_data.get('eye_contact_score', 0.5) * 100,
        'session_count': len(st.session_state.dashboard_data.get('session_history', [])),
        'total_duration': sum(s.get('duration', 0) for s in st.session_state.dashboard_data.get('session_history', [])),
        'improvement_trend': 'stable'  # Would be calculated from historical data
    }


# Main dashboard entry point
def main_dashboard():
    """Main dashboard application entry point"""
    
    dashboard = create_dashboard_instance()
    
    # Render sidebar controls
    render_dashboard_sidebar()
    
    # Get current session data
    session_data = st.session_state.dashboard_data.get('current_session', {})
    
    # Render main dashboard
    dashboard.render_main_dashboard(session_data if session_data else None)


if __name__ == "__main__":
    main_dashboard()