"""
StealthMesh Dashboard - Streamlit UI
Adaptive Stealth Communication and Decentralized Defense for MSMEs
"""

import streamlit as st
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Page config - MUST be first Streamlit call
st.set_page_config(
    page_title="StealthMesh Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ──────────────────────────────────────────────
st.markdown("""
<style>
    /* Global */
    .main .block-container { padding-top: 1rem; }
    
    /* Metric cards */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #1a1f2e 0%, #2d3548 100%);
        border: 1px solid #3d4f6f;
        border-radius: 12px;
        padding: 15px 20px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }
    div[data-testid="stMetric"] label { color: #8899aa !important; font-size: 0.9rem; }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] { color: #ffffff !important; }
    
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0e1117 0%, #1a1f2e 100%);
    }
    
    /* Headers */
    .hero-title {
        font-size: 2.8rem; font-weight: 800;
        background: linear-gradient(135deg, #e74c3c 0%, #f39c12 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .hero-subtitle { color: #8899aa; font-size: 1.1rem; margin-top: 0; }
    
    /* Module cards */
    .module-card {
        background: linear-gradient(135deg, #1a1f2e 0%, #2d3548 100%);
        border: 1px solid #3d4f6f; border-radius: 12px;
        padding: 20px; margin: 8px 0;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .module-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(231,76,60,0.2);
    }
    .module-card h4 { color: #e74c3c; margin: 0 0 8px 0; }
    .module-card p { color: #aabbcc; margin: 0; font-size: 0.9rem; }
    
    /* Status indicators */
    .status-active { color: #2ecc71; font-weight: bold; }
    .status-warning { color: #f39c12; font-weight: bold; }
    .status-danger { color: #e74c3c; font-weight: bold; }
    
    /* Divider */
    .section-divider {
        border: 0; height: 1px;
        background: linear-gradient(90deg, transparent, #3d4f6f, transparent);
        margin: 30px 0;
    }
    
    /* Log viewer */
    .log-line { font-family: 'Courier New', monospace; font-size: 0.85rem; }
    .log-info { color: #3498db; }
    .log-warn { color: #f39c12; }
    .log-error { color: #e74c3c; }
    .log-success { color: #2ecc71; }
</style>
""", unsafe_allow_html=True)


# ── Sidebar Navigation ─────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛡️ StealthMesh")
    st.markdown("*Cyber Defense Dashboard*")
    st.markdown("---")
    
    page = st.radio(
        "Navigation",
        [
            "🏠 Home",
            "📊 ML Model Results",
            "🔬 StealthMesh Modules",
            "🎮 Live Simulation",
            "🏗️ Architecture",
            "📄 About"
        ],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    st.markdown("##### System Status")
    st.markdown("🟢 All modules operational")
    st.markdown(f"📁 Models: `{len([f for f in os.listdir('models') if f.endswith('.pkl')])}` loaded")
    st.markdown(f"📊 Datasets: `4` processed")


# ── Page Router ─────────────────────────────────────────────
if page == "🏠 Home":
    from ui.page_home import render
    render()
elif page == "📊 ML Model Results":
    from ui.page_results import render
    render()
elif page == "🔬 StealthMesh Modules":
    from ui.page_modules import render
    render()
elif page == "🎮 Live Simulation":
    from ui.page_simulation import render
    render()
elif page == "🏗️ Architecture":
    from ui.page_architecture import render
    render()
elif page == "📄 About":
    from ui.page_about import render
    render()
