"""
Custom CSS Styling System for SpeakSmart AI
Author: Jahnvi Pandey (UI/UX & Frontend Developer)

Modern CSS styling with responsive design and accessibility features
Provides consistent theming across all components
"""

import streamlit as st
from typing import Dict, Optional

# Color palette and theme definitions
THEME_COLORS = {
    'primary': '#667eea',
    'secondary': '#764ba2', 
    'success': '#28a745',
    'warning': '#ffc107',
    'danger': '#dc3545',
    'info': '#17a2b8',
    'light': '#f8f9fa',
    'dark': '#343a40',
    'white': '#ffffff',
    'gray_100': '#f8f9fa',
    'gray_200': '#e9ecef',
    'gray_300': '#dee2e6',
    'gray_400': '#ced4da',
    'gray_500': '#adb5bd',
    'gray_600': '#6c757d',
    'gray_700': '#495057',
    'gray_800': '#343a40',
    'gray_900': '#212529'
}

# Gradient definitions
GRADIENTS = {
    'primary': 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
    'success': 'linear-gradient(135deg, #28a745 0%, #20c997 100%)',
    'warning': 'linear-gradient(135deg, #ffc107 0%, #fd7e14 100%)',
    'danger': 'linear-gradient(135deg, #dc3545 0%, #e83e8c 100%)',
    'info': 'linear-gradient(135deg, #17a2b8 0%, #6f42c1 100%)',
    'light': 'linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%)',
    'dark': 'linear-gradient(135deg, #343a40 0%, #495057 100%)'
}

# Typography scale
TYPOGRAPHY = {
    'font_family': '"Inter", "Segoe UI", "Roboto", sans-serif',
    'font_sizes': {
        'xs': '0.75rem',    # 12px
        'sm': '0.875rem',   # 14px
        'base': '1rem',     # 16px
        'lg': '1.125rem',   # 18px
        'xl': '1.25rem',    # 20px
        '2xl': '1.5rem',    # 24px
        '3xl': '1.875rem',  # 30px
        '4xl': '2.25rem',   # 36px
        '5xl': '3rem',      # 48px
    },
    'font_weights': {
        'light': 300,
        'normal': 400,
        'medium': 500,
        'semibold': 600,
        'bold': 700,
        'extrabold': 800
    },
    'line_heights': {
        'tight': 1.25,
        'normal': 1.5,
        'relaxed': 1.625,
        'loose': 2
    }
}

# Spacing system (in rem)
SPACING = {
    '0': '0',
    '1': '0.25rem',   # 4px
    '2': '0.5rem',    # 8px
    '3': '0.75rem',   # 12px
    '4': '1rem',      # 16px
    '5': '1.25rem',   # 20px
    '6': '1.5rem',    # 24px
    '8': '2rem',      # 32px
    '10': '2.5rem',   # 40px
    '12': '3rem',     # 48px
    '16': '4rem',     # 64px
    '20': '5rem',     # 80px
    '24': '6rem'      # 96px
}

# Border radius values
BORDER_RADIUS = {
    'none': '0',
    'sm': '0.125rem',   # 2px
    'base': '0.25rem',  # 4px
    'md': '0.375rem',   # 6px
    'lg': '0.5rem',     # 8px
    'xl': '0.75rem',    # 12px
    '2xl': '1rem',      # 16px
    'full': '9999px'
}

# Shadow definitions
SHADOWS = {
    'sm': '0 1px 2px 0 rgba(0, 0, 0, 0.05)',
    'base': '0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px 0 rgba(0, 0, 0, 0.06)',
    'md': '0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)',
    'lg': '0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05)',
    'xl': '0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04)',
    '2xl': '0 25px 50px -12px rgba(0, 0, 0, 0.25)',
    'inner': 'inset 0 2px 4px 0 rgba(0, 0, 0, 0.06)',
    'none': 'none'
}

def get_theme_colors() -> Dict[str, str]:
    """Get the current theme color palette"""
    return THEME_COLORS.copy()

def inject_custom_css():
    """Inject comprehensive custom CSS into the Streamlit app"""
    
    css = f"""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    /* Root CSS Variables */
    :root {{
        --primary-color: {THEME_COLORS['primary']};
        --secondary-color: {THEME_COLORS['secondary']};
        --success-color: {THEME_COLORS['success']};
        --warning-color: {THEME_COLORS['warning']};
        --danger-color: {THEME_COLORS['danger']};
        --info-color: {THEME_COLORS['info']};
        --light-color: {THEME_COLORS['light']};
        --dark-color: {THEME_COLORS['dark']};
        
        --font-family: {TYPOGRAPHY['font_family']};
        --border-radius: {BORDER_RADIUS['md']};
        --shadow-base: {SHADOWS['base']};
        
        --gradient-primary: {GRADIENTS['primary']};
        --gradient-success: {GRADIENTS['success']};
        --gradient-warning: {GRADIENTS['warning']};
        --gradient-danger: {GRADIENTS['danger']};
    }}
    
    /* Global Styles */
    html, body {{
        font-family: var(--font-family);
        background-color: #fafbfc;
        color: {THEME_COLORS['gray_800']};
        line-height: {TYPOGRAPHY['line_heights']['normal']};
    }}
    
    /* Streamlit Container Modifications */
    .main .block-container {{
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 100%;
    }}
    
    /* Header Styles */
    .header-container {{
        background: var(--gradient-primary);
        padding: {SPACING['8']} {SPACING['6']};
        border-radius: {BORDER_RADIUS['2xl']};
        margin-bottom: {SPACING['8']};
        text-align: center;
        color: white;
        box-shadow: {SHADOWS['xl']};
        position: relative;
        overflow: hidden;
    }}
    
    .header-container::before {{
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
        animation: float 6s ease-in-out infinite;
    }}
    
    .header-content {{
        position: relative;
        z-index: 1;
    }}
    
    .main-title {{
        font-size: {TYPOGRAPHY['font_sizes']['5xl']};
        font-weight: {TYPOGRAPHY['font_weights']['bold']};
        margin-bottom: {SPACING['4']};
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }}
    
    .subtitle {{
        font-size: {TYPOGRAPHY['font_sizes']['xl']};
        font-weight: {TYPOGRAPHY['font_weights']['medium']};
        margin-bottom: {SPACING['6']};
        opacity: 0.95;
    }}
    
    .header-features {{
        display: flex;
        justify-content: center;
        gap: {SPACING['4']};
        flex-wrap: wrap;
    }}
    
    .feature-badge {{
        background: rgba(255, 255, 255, 0.2);
        padding: {SPACING['2']} {SPACING['4']};
        border-radius: {BORDER_RADIUS['full']};
        font-size: {TYPOGRAPHY['font_sizes']['sm']};
        font-weight: {TYPOGRAPHY['font_weights']['medium']};
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.3);
        transition: all 0.3s ease;
    }}
    
    .feature-badge:hover {{
        background: rgba(255, 255, 255, 0.3);
        transform: translateY(-2px);
    }}
    
    /* Metric Card Styles */
    .metric-card {{
        background: linear-gradient(145deg, #ffffff, #f0f2f6);
        padding: {SPACING['6']};
        border-radius: {BORDER_RADIUS['xl']};
        box-shadow: {SHADOWS['lg']};
        margin: {SPACING['4']} 0;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        border: 1px solid rgba(255, 255, 255, 0.8);
        position: relative;
        overflow: hidden;
    }}
    
    .metric-card::before {{
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: var(--gradient-primary);
        transform: scaleX(0);
        transform-origin: left;
        transition: transform 0.3s ease;
    }}
    
    .metric-card:hover {{
        transform: translateY(-4px) scale(1.02);
        box-shadow: {SHADOWS['2xl']};
    }}
    
    .metric-card:hover::before {{
        transform: scaleX(1);
    }}
    
    /* Analysis Container */
    .analysis-container {{
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        padding: {SPACING['8']};
        border-radius: {BORDER_RADIUS['xl']};
        margin: {SPACING['6']} 0;
        border: 1px solid {THEME_COLORS['gray_200']};
        box-shadow: {SHADOWS['md']};
        position: relative;
    }}
    
    .analysis-container::before {{
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: var(--gradient-primary);
        border-radius: {BORDER_RADIUS['base']} {BORDER_RADIUS['base']} 0 0;
    }}
    
    /* Video Container */
    .video-container {{
        border: 2px solid {THEME_COLORS['gray_300']};
        border-radius: {BORDER_RADIUS['xl']};
        overflow: hidden;
        margin: {SPACING['6']} 0;
        box-shadow: {SHADOWS['lg']};
        transition: all 0.3s ease;
    }}
    
    .video-container:hover {{
        border-color: var(--primary-color);
        box-shadow: {SHADOWS['xl']};
    }}
    
    /* Emotion Indicators */
    .emotion-indicator {{
        display: inline-block;
        padding: {SPACING['2']} {SPACING['4']};
        border-radius: {BORDER_RADIUS['full']};
        font-size: {TYPOGRAPHY['font_sizes']['sm']};
        font-weight: {TYPOGRAPHY['font_weights']['semibold']};
        margin: {SPACING['1']};
        transition: all 0.2s ease;
        cursor: default;
    }}
    
    .emotion-happy {{
        background: linear-gradient(135deg, #FFD700, #FFA500);
        color: white;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.3);
    }}
    
    .emotion-confident {{
        background: var(--gradient-success);
        color: white;
    }}
    
    .emotion-calm {{
        background: var(--gradient-info);
        color: white;
    }}
    
    .emotion-nervous {{
        background: var(--gradient-warning);
        color: {THEME_COLORS['gray_800']};
    }}
    
    .emotion-indicator:hover {{
        transform: scale(1.1);
        z-index: 10;
        position: relative;
    }}
    
    /* Button Styles */
    .stButton > button {{
        background: var(--gradient-primary);
        color: white;
        border: none;
        border-radius: {BORDER_RADIUS['lg']};
        padding: {SPACING['3']} {SPACING['6']};
        font-weight: {TYPOGRAPHY['font_weights']['semibold']};
        font-family: var(--font-family);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: {SHADOWS['md']};
        position: relative;
        overflow: hidden;
    }}
    
    .stButton > button::before {{
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
        transition: left 0.5s;
    }}
    
    .stButton > button:hover {{
        transform: translateY(-2px);
        box-shadow: {SHADOWS['lg']};
    }}
    
    .stButton > button:hover::before {{
        left: 100%;
    }}
    
    .stButton > button:active {{
        transform: translateY(0);
        box-shadow: {SHADOWS['base']};
    }}
    
    /* Sidebar Styles */
    .css-1d391kg {{
        background: linear-gradient(180deg, #f8f9fa 0%, #ffffff 100%);
        border-right: 1px solid {THEME_COLORS['gray_200']};
    }}
    
    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {{
        gap: {SPACING['2']};
        background: {THEME_COLORS['gray_100']};
        border-radius: {BORDER_RADIUS['lg']};
        padding: {SPACING['1']};
    }}
    
    .stTabs [data-baseweb="tab-list"] button {{
        background: transparent;
        border-radius: {BORDER_RADIUS['md']};
        padding: {SPACING['3']} {SPACING['6']};
        font-weight: {TYPOGRAPHY['font_weights']['medium']};
        transition: all 0.2s ease;
    }}
    
    .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {{
        background: var(--gradient-primary);
        color: white;
        box-shadow: {SHADOWS['sm']};
    }}
    
    .stTabs [data-baseweb="tab-list"] button:hover {{
        background: rgba(102, 126, 234, 0.1);
        transform: translateY(-1px);
    }}
    
    /* File Uploader Styling */
    .stFileUploader {{
        border: 2px dashed var(--primary-color);
        border-radius: {BORDER_RADIUS['xl']};
        padding: {SPACING['8']};
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.03), rgba(118, 75, 162, 0.03));
        transition: all 0.3s ease;
    }}
    
    .stFileUploader:hover {{
        border-color: var(--secondary-color);
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.08), rgba(118, 75, 162, 0.08));
        transform: translateY(-2px);
    }}
    
    /* Progress Bar Styling */
    .stProgress > div > div > div > div {{
        background: var(--gradient-primary);
        border-radius: {BORDER_RADIUS['full']};
        height: 12px;
    }}
    
    .stProgress > div > div {{
        background: {THEME_COLORS['gray_200']};
        border-radius: {BORDER_RADIUS['full']};
        height: 12px;
    }}
    
    /* Metric Styling */
    .css-1xarl3l {{
        background: linear-gradient(135deg, #ffffff, #f8f9fa);
        padding: {SPACING['4']};
        border-radius: {BORDER_RADIUS['lg']};
        border-left: 4px solid var(--primary-color);
        box-shadow: {SHADOWS['sm']};
    }}
    
    /* Selectbox Styling */
    .stSelectbox > div > div {{
        border-radius: {BORDER_RADIUS['md']};
        border: 1px solid {THEME_COLORS['gray_300']};
        transition: all 0.2s ease;
    }}
    
    .stSelectbox > div > div:focus-within {{
        border-color: var(--primary-color);
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }}
    
    /* Slider Styling */
    .stSlider > div > div > div > div {{
        background: var(--gradient-primary);
    }}
    
    /* Empty State */
    .empty-state {{
        text-align: center;
        padding: {SPACING['20']};
        background: linear-gradient(135deg, #f8f9fa, #ffffff);
        border-radius: {BORDER_RADIUS['xl']};
        border: 2px dashed {THEME_COLORS['gray_300']};
        margin: {SPACING['8']} 0;
    }}
    
    .empty-icon {{
        font-size: {TYPOGRAPHY['font_sizes']['5xl']};
        opacity: 0.5;
        margin-bottom: {SPACING['4']};
    }}
    
    .empty-state h3 {{
        color: {THEME_COLORS['gray_600']};
        margin-bottom: {SPACING['2']};
        font-weight: {TYPOGRAPHY['font_weights']['semibold']};
    }}
    
    .empty-state p {{
        color: {THEME_COLORS['gray_500']};
        font-size: {TYPOGRAPHY['font_sizes']['sm']};
    }}
    
    /* Footer Styles */
    .footer-container {{
        background: var(--gradient-primary);
        padding: {SPACING['8']} {SPACING['6']};
        border-radius: {BORDER_RADIUS['xl']};
        margin-top: {SPACING['12']};
        text-align: center;
        color: white;
        position: relative;
        overflow: hidden;
    }}
    
    .footer-content h4 {{
        margin-bottom: {SPACING['4']};
        font-size: {TYPOGRAPHY['font_sizes']['2xl']};
        font-weight: {TYPOGRAPHY['font_weights']['bold']};
    }}
    
    .team-info {{
        background: rgba(255, 255, 255, 0.1);
        padding: {SPACING['4']};
        border-radius: {BORDER_RADIUS['lg']};
        margin: {SPACING['4']} 0;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }}
    
    .version-info {{
        font-size: {TYPOGRAPHY['font_sizes']['sm']};
        opacity: 0.8;
        margin-top: {SPACING['4']};
    }}
    
    /* Animations */
    @keyframes float {{
        0%, 100% {{ transform: translateY(0px) rotate(0deg); }}
        50% {{ transform: translateY(-20px) rotate(180deg); }}
    }}
    
    @keyframes fadeInUp {{
        from {{
            opacity: 0;
            transform: translateY(30px);
        }}
        to {{
            opacity: 1;
            transform: translateY(0);
        }}
    }}
    
    @keyframes pulse {{
        0%, 100% {{ opacity: 1; }}
        50% {{ opacity: 0.5; }}
    }}
    
    @keyframes slideIn {{
        from {{
            transform: translateX(-100%);
            opacity: 0;
        }}
        to {{
            transform: translateX(0);
            opacity: 1;
        }}
    }}
    
    /* Utility Classes */
    .fade-in-up {{
        animation: fadeInUp 0.6s ease-out;
    }}
    
    .pulse {{
        animation: pulse 2s infinite;
    }}
    
    .slide-in {{
        animation: slideIn 0.5s ease-out;
    }}
    
    /* Responsive Design */
    @media (max-width: 768px) {{
        .main-title {{
            font-size: {TYPOGRAPHY['font_sizes']['3xl']};
        }}
        
        .header-container {{
            padding: {SPACING['6']} {SPACING['4']};
        }}
        
        .metric-card {{
            padding: {SPACING['4']};
        }}
        
        .analysis-container {{
            padding: {SPACING['6']};
        }}
        
        .feature-badge {{
            font-size: {TYPOGRAPHY['font_sizes']['xs']};
            padding: {SPACING['1']} {SPACING['3']};
        }}
    }}
    
    /* Dark Mode Support */
    @media (prefers-color-scheme: dark) {{
        :root {{
            --text-color: #ffffff;
            --bg-color: #1a1a1a;
            --card-bg: #2d2d2d;
        }}
        
        .metric-card {{
            background: linear-gradient(145deg, #2d2d2d, #1a1a1a);
            color: white;
        }}
        
        .analysis-container {{
            background: linear-gradient(135deg, #2d2d2d 0%, #1a1a1a 100%);
            color: white;
        }}
    }}
    
    /* Print Styles */
    @media print {{
        .header-container {{
            background: white !important;
            color: black !important;
            box-shadow: none !important;
        }}
        
        .metric-card {{
            box-shadow: none !important;
            border: 1px solid #ccc !important;
        }}
        
        .footer-container {{
            background: white !important;
            color: black !important;
        }}
    }}
    </style>
    """
    
    st.markdown(css, unsafe_allow_html=True)

def get_component_style(component_type: str, variant: str = "default") -> str:
    """Get CSS class for specific component type and variant"""
    
    styles = {
        'card': {
            'default': 'metric-card',
            'success': 'metric-card metric-card-success',
            'warning': 'metric-card metric-card-warning',
            'danger': 'metric-card metric-card-danger'
        },
        'button': {
            'primary': 'btn-primary',
            'secondary': 'btn-secondary',
            'success': 'btn-success',
            'danger': 'btn-danger'
        },
        'alert': {
            'success': 'alert-success',
            'warning': 'alert-warning',
            'danger': 'alert-danger',
            'info': 'alert-info'
        }
    }
    
    return styles.get(component_type, {}).get(variant, '')

def create_color_palette(primary_color: str) -> Dict[str, str]:
    """Generate a color palette based on primary color"""
    # This would implement color theory to generate complementary colors
    # For now, returning default palette
    return THEME_COLORS

def apply_theme(theme_name: str = "default"):
    """Apply a specific theme to the application"""
    
    themes = {
        'default': THEME_COLORS,
        'dark': {
            **THEME_COLORS,
            'primary': '#8B5CF6',
            'secondary': '#A855F7',
            'light': '#374151',
            'dark': '#111827'
        },
        'ocean': {
            **THEME_COLORS,
            'primary': '#0EA5E9',
            'secondary': '#0284C7',
            'success': '#059669',
            'info': '#0891B2'
        },
        'sunset': {
            **THEME_COLORS,
            'primary': '#F59E0B',
            'secondary': '#D97706',
            'danger': '#DC2626',
            'warning': '#F59E0B'
        }
    }
    
    selected_theme = themes.get(theme_name, THEME_COLORS)
    return selected_theme

# Utility functions for dynamic styling
def hex_to_rgba(hex_color: str, alpha: float = 1.0) -> str:
    """Convert hex color to RGBA"""
    hex_color = hex_color.lstrip('#')
    if len(hex_color) != 6:
        return f"rgba(0, 0, 0, {alpha})"
    
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    
    return f"rgba({r}, {g}, {b}, {alpha})"

def lighten_color(hex_color: str, amount: float = 0.2) -> str:
    """Lighten a hex color by a percentage"""
    hex_color = hex_color.lstrip('#')
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16) 
    b = int(hex_color[4:6], 16)
    
    r = min(255, int(r + (255 - r) * amount))
    g = min(255, int(g + (255 - g) * amount))
    b = min(255, int(b + (255 - b) * amount))
    
    return f"#{r:02x}{g:02x}{b:02x}"

def darken_color(hex_color: str, amount: float = 0.2) -> str:
    """Darken a hex color by a percentage"""
    hex_color = hex_color.lstrip('#')
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    
    r = max(0, int(r * (1 - amount)))
    g = max(0, int(g * (1 - amount)))
    b = max(0, int(b * (1 - amount)))
    
    return f"#{r:02x}{g:02x}{b:02x}"

# Export main functions
__all__ = [
    'get_theme_colors',
    'inject_custom_css', 
    'get_component_style',
    'create_color_palette',
    'apply_theme',
    'hex_to_rgba',
    'lighten_color',
    'darken_color',
    'THEME_COLORS',
    'GRADIENTS',
    'TYPOGRAPHY',
    'SPACING',
    'BORDER_RADIUS',
    'SHADOWS'
]