"""
Session State Management for SpeakSmart AI
Author: Vedant Singh (Backend & Data Management)

Centralized session state management with persistence and cleanup
"""

import streamlit as st
from collections import deque, defaultdict
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
import json
import pickle
import os
from pathlib import Path

class SessionManager:
    """Manages application session state and data persistence"""
    
    def __init__(self, session_timeout: int = 3600):
        self.session_timeout = session_timeout
        self.session_dir = Path(".sessions")
        self.session_dir.mkdir(exist_ok=True)
        
    def initialize_session(self):
        """Initialize session state with default values"""
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
            "eye_contact_score": 0.0,
            "posture_analysis": {},
            "facial_expressions": defaultdict(int),
            
            # Session metrics
            "current_confidence": 0,
            "nervousness_level": 0,
            "overall_score": 0,
            "analysis_complete": False,
            "analysis_active": False,
            "session_duration": 0.0,
            "session_start_time": datetime.now(),
            
            # Configuration
            "analysis_mode": "Audio + Video",
            "input_method": "Upload Files",
            "target_confidence": 80,
            "max_nervousness": 20,
            
            # Recording state
            "recording_active": False,
            "recording_start_time": None,
            "recording_duration": 60,
            
            # UI state
            "show_advanced_settings": False,
            "current_tab": "input",
            "theme": "default"
        }
        
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value
        
        # Set session ID if not exists
        if "session_id" not in st.session_state:
            st.session_state.session_id = self._generate_session_id()
    
    def _generate_session_id(self) -> str:
        """Generate unique session ID"""
        import uuid
        return str(uuid.uuid4())[:8]
    
    def reset_session(self):
        """Reset session state to defaults"""
        # Clear analysis data
        analysis_keys = [
            "pitch_data", "volume_data", "timestamps", "confidence_scores",
            "nervousness_indicators", "speaking_segments", "emotion_history",
            "gesture_data", "eye_contact_score", "posture_analysis",
            "facial_expressions", "current_confidence", "nervousness_level",
            "overall_score", "session_duration"
        ]
        
        for key in analysis_keys:
            if key in st.session_state:
                if key.endswith("_data") or key.endswith("_history") or key.endswith("_scores"):
                    st.session_state[key].clear()
                elif key.endswith("_indicators") or key.endswith("_segments"):
                    st.session_state[key] = []
                elif key.endswith("_analysis") or key.endswith("_expressions"):
                    st.session_state[key] = {}
                else:
                    st.session_state[key] = 0
        
        # Reset state flags
        st.session_state.analysis_complete = False
        st.session_state.analysis_active = False
        st.session_state.recording_active = False
        st.session_state.session_start_time = datetime.now()
    
    def save_session(self, session_data: Optional[Dict] = None) -> bool:
        """Save current session to disk"""
        try:
            session_id = st.session_state.get("session_id", self._generate_session_id())
            session_file = self.session_dir / f"session_{session_id}.json"
            
            # Prepare session data
            if session_data is None:
                session_data = self._serialize_session_state()
            
            with open(session_file, 'w') as f:
                json.dump(session_data, f, indent=2, default=str)
            
            return True
            
        except Exception as e:
            print(f"Failed to save session: {e}")
            return False
    
    def load_session(self, session_id: str) -> bool:
        """Load session from disk"""
        try:
            session_file = self.session_dir / f"session_{session_id}.json"
            
            if not session_file.exists():
                return False
            
            with open(session_file, 'r') as f:
                session_data = json.load(f)
            
            self._deserialize_session_state(session_data)
            return True
            
        except Exception as e:
            print(f"Failed to load session: {e}")
            return False
    
    def _serialize_session_state(self) -> Dict[str, Any]:
        """Serialize session state for storage"""
        serializable_data = {}
        
        for key, value in st.session_state.items():
            try:
                if isinstance(value, (deque, defaultdict)):
                    serializable_data[key] = list(value)
                elif isinstance(value, datetime):
                    serializable_data[key] = value.isoformat()
                else:
                    # Test if JSON serializable
                    json.dumps(value)
                    serializable_data[key] = value
            except (TypeError, ValueError):
                # Skip non-serializable values
                continue
        
        return serializable_data
    
    def _deserialize_session_state(self, data: Dict[str, Any]):
        """Deserialize session state from storage"""
        for key, value in data.items():
            if key.endswith("_data") or key.endswith("_history") or key.endswith("_scores"):
                st.session_state[key] = deque(value, maxlen=3000)
            elif key.endswith("_expressions"):
                st.session_state[key] = defaultdict(int, value)
            elif key.endswith("_time") and isinstance(value, str):
                try:
                    st.session_state[key] = datetime.fromisoformat(value)
                except ValueError:
                    st.session_state[key] = datetime.now()
            else:
                st.session_state[key] = value
    
    def cleanup_old_sessions(self, days_old: int = 7):
        """Clean up old session files"""
        cutoff_date = datetime.now() - timedelta(days=days_old)
        
        for session_file in self.session_dir.glob("session_*.json"):
            try:
                if session_file.stat().st_mtime < cutoff_date.timestamp():
                    session_file.unlink()
            except Exception as e:
                print(f"Failed to delete old session {session_file}: {e}")
    
    def get_session_stats(self) -> Dict[str, Any]:
        """Get current session statistics"""
        start_time = st.session_state.get("session_start_time", datetime.now())
        current_time = datetime.now()
        session_duration = (current_time - start_time).total_seconds()
        
        stats = {
            "session_id": st.session_state.get("session_id", "unknown"),
            "session_duration": session_duration,
            "analysis_complete": st.session_state.get("analysis_complete", False),
            "analysis_active": st.session_state.get("analysis_active", False),
            "current_confidence": st.session_state.get("current_confidence", 0),
            "overall_score": st.session_state.get("overall_score", 0),
            "data_points": {
                "pitch_data_count": len(st.session_state.get("pitch_data", [])),
                "confidence_scores_count": len(st.session_state.get("confidence_scores", [])),
                "emotion_history_count": len(st.session_state.get("emotion_history", []))
            }
        }
        
        return stats
    
    def update_metric(self, metric_name: str, value: Any):
        """Update a specific metric in session state"""
        st.session_state[metric_name] = value
    
    def add_to_history(self, history_name: str, value: Any, max_length: int = 1000):
        """Add value to a history deque"""
        if history_name not in st.session_state:
            st.session_state[history_name] = deque(maxlen=max_length)
        
        st.session_state[history_name].append(value)
    
    def get_metric(self, metric_name: str, default: Any = None) -> Any:
        """Get a metric value from session state"""
        return st.session_state.get(metric_name, default)
    
    def is_session_expired(self) -> bool:
        """Check if current session has expired"""
        start_time = st.session_state.get("session_start_time", datetime.now())
        current_time = datetime.now()
        session_duration = (current_time - start_time).total_seconds()
        
        return session_duration > self.session_timeout
    
    def export_session_data(self) -> Dict[str, Any]:
        """Export session data for analysis or backup"""
        return {
            "session_info": self.get_session_stats(),
            "analysis_data": {
                "pitch_data": list(st.session_state.get("pitch_data", [])),
                "volume_data": list(st.session_state.get("volume_data", [])),
                "confidence_scores": list(st.session_state.get("confidence_scores", [])),
                "emotion_history": list(st.session_state.get("emotion_history", [])),
                "nervousness_indicators": st.session_state.get("nervousness_indicators", [])
            },
            "metrics": {
                "current_confidence": st.session_state.get("current_confidence", 0),
                "nervousness_level": st.session_state.get("nervousness_level", 0),
                "overall_score": st.session_state.get("overall_score", 0),
                "eye_contact_score": st.session_state.get("eye_contact_score", 0)
            },
            "export_timestamp": datetime.now().isoformat()
        }