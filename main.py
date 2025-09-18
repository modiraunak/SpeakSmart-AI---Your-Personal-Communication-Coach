"""
SpeakSmart AI - Advanced Communication Coach
Main Application Entry Point

Authors: Team SpeakSmart (Group 256)
- Raunak Kumar Modi (Team Lead & Audio Analysis)
- Jahnvi Pandey (UI/UX & Frontend)
- Rishi Singh Shandilya (Video Processing)
- Unnati Lohana (ML Models & AI Integration)
- Vedant Singh (Backend & Data Management)

Project Exhibition-I
Version 2.0 - Enhanced with Pre-trained Models
"""

import streamlit as st
import os
import sys
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent
sys.path.append(str(PROJECT_ROOT))

# Import custom modules
from src.ui.app_interface import SpeakSmartInterface
from src.config.settings import AppConfig
from src.utils.session_manager import SessionManager
from src.utils.logger import setup_logger

# Configure logging
logger = setup_logger("main_app")

def main():
    """Main application entry point"""
    try:
        # Initialize configuration
        config = AppConfig()
        
        # Setup page configuration
        st.set_page_config(
            page_title="SpeakSmart AI - Advanced Communication Coach",
            page_icon="🎯",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Initialize session manager
        session_manager = SessionManager()
        session_manager.initialize_session()
        
        # Create main interface
        app_interface = SpeakSmartInterface(config)
        
        # Run the application
        app_interface.run()
        
        logger.info("Application started successfully")
        
    except Exception as e:
        logger.error(f"Application startup failed: {str(e)}")
        st.error("Failed to start SpeakSmart AI. Please check your installation.")
        st.exception(e)

if __name__ == "__main__":
    main()