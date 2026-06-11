"""
Mall Surveillance AI — Streamlit Dashboard

Main application entry point for the multi-page Streamlit dashboard.
"""

import streamlit as st

# ---------------------------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Mall Surveillance AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS for premium dark theme
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Global styles */
    .stApp {
        font-family: 'Inter', sans-serif;
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0f23 0%, #1a1a3e 100%);
    }
    [data-testid="stSidebar"] .stMarkdown {
        color: #e0e0ff;
    }

    /* Header card */
    .header-card {
        background: linear-gradient(135deg, #1e1e4a 0%, #2d1b69 50%, #1a1a3e 100%);
        border-radius: 16px;
        padding: 24px 32px;
        margin-bottom: 24px;
        border: 1px solid rgba(139, 92, 246, 0.3);
        box-shadow: 0 4px 20px rgba(139, 92, 246, 0.15);
    }
    .header-card h1 {
        color: #ffffff;
        font-size: 28px;
        font-weight: 700;
        margin: 0;
    }
    .header-card p {
        color: #a5b4fc;
        font-size: 14px;
        margin: 4px 0 0;
    }

    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #1e1e4a 0%, #252560 100%);
        border-radius: 12px;
        padding: 20px;
        border: 1px solid rgba(139, 92, 246, 0.2);
        text-align: center;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(139, 92, 246, 0.2);
    }
    .metric-value {
        font-size: 36px;
        font-weight: 700;
        color: #8b5cf6;
        display: block;
    }
    .metric-label {
        font-size: 13px;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 4px;
    }

    /* Alert badge */
    .alert-p1 {
        background: linear-gradient(135deg, #dc2626, #b91c1c);
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 12px;
    }
    .alert-p2 {
        background: linear-gradient(135deg, #f59e0b, #d97706);
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 12px;
    }
    .alert-p3 {
        background: linear-gradient(135deg, #3b82f6, #2563eb);
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 12px;
    }

    /* Status indicators */
    .status-online { color: #22c55e; }
    .status-offline { color: #ef4444; }

    /* Risk badges */
    .risk-critical { background: #dc2626; color: white; padding: 2px 8px; border-radius: 4px; font-size: 11px; }
    .risk-high { background: #f97316; color: white; padding: 2px 8px; border-radius: 4px; font-size: 11px; }
    .risk-medium { background: #eab308; color: #1a1a2e; padding: 2px 8px; border-radius: 4px; font-size: 11px; }
    .risk-low { background: #22c55e; color: white; padding: 2px 8px; border-radius: 4px; font-size: 11px; }

    /* Table styling */
    .stDataFrame {
        border-radius: 8px;
        overflow: hidden;
    }

    /* Button styling */
    .stButton > button {
        border-radius: 8px;
        font-weight: 500;
        transition: all 0.2s;
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(139, 92, 246, 0.3);
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 16px 0;">
        <h2 style="color: #8b5cf6; margin: 0;">🛡️ Mall AI</h2>
        <p style="color: #94a3b8; font-size: 12px; margin: 4px 0;">Intelligent Surveillance System</p>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # Navigation
    page = st.radio(
        "Navigation",
        [
            "🎥 Live Cameras",
            "🚨 Active Alerts",
            "🔍 Event Investigation",
            "🔎 Event Search",
            "📊 Analytics",
            "📹 Camera Management",
            "💻 System Health",
        ],
        label_visibility="collapsed",
    )

    st.divider()

    # API Status
    api_url = "http://localhost:8000"
    st.markdown(f"""
    <div style="padding: 8px; background: rgba(139,92,246,0.1); border-radius: 8px; text-align: center;">
        <span style="color: #94a3b8; font-size: 12px;">API: </span>
        <span style="color: #22c55e; font-size: 12px;">● {api_url}</span>
    </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------
import httpx

API_BASE = "http://localhost:8000/api"


def api_get(endpoint: str, params: dict | None = None) -> dict | list | None:
    """Make a GET request to the API."""
    try:
        resp = httpx.get(f"{API_BASE}{endpoint}", params=params, timeout=5.0)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return None


def api_post(endpoint: str, json_data: dict | None = None) -> dict | None:
    """Make a POST request to the API."""
    try:
        resp = httpx.post(f"{API_BASE}{endpoint}", json=json_data, timeout=5.0)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Page Routing
# ---------------------------------------------------------------------------
if page == "🎥 Live Cameras":
    from frontend.pages.page_live_cameras import render
    render(api_get)

elif page == "🚨 Active Alerts":
    from frontend.pages.page_active_alerts import render
    render(api_get, api_post)

elif page == "🔍 Event Investigation":
    from frontend.pages.page_event_investigation import render
    render(api_get)

elif page == "🔎 Event Search":
    from frontend.pages.page_event_search import render
    render(api_get)

elif page == "📊 Analytics":
    from frontend.pages.page_analytics import render
    render(api_get)

elif page == "📹 Camera Management":
    from frontend.pages.page_camera_management import render
    render(api_get, api_post)

elif page == "💻 System Health":
    from frontend.pages.page_system_health import render
    render(api_get)
