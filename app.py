"""
Telco Churn Prediction System — Main Entry
==========================================
Streamlit multi-page app with model loading, session state, and sidebar navigation.
"""

import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os

# ─── Page Configuration ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="TelcoGuard · Churn Intelligence",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1117 0%, #161b22 100%);
    border-right: 1px solid #30363d;
}
[data-testid="stSidebar"] * {
    color: #e6edf3 !important;
}
[data-testid="stSidebar"] .stRadio label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.85rem;
    letter-spacing: 0.04em;
}

/* Main background */
.main .block-container {
    background: #0d1117;
    padding-top: 2rem;
}

/* Metric cards */
.metric-card {
    background: linear-gradient(135deg, #161b22 0%, #1c2128 100%);
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 1.4rem 1.6rem;
    transition: border-color 0.2s;
}
.metric-card:hover { border-color: #58a6ff; }
.metric-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #8b949e;
    margin-bottom: 0.4rem;
}
.metric-value {
    font-size: 2rem;
    font-weight: 700;
    color: #e6edf3;
    line-height: 1.1;
}
.metric-delta {
    font-size: 0.78rem;
    color: #3fb950;
    margin-top: 0.3rem;
}

/* Risk badges */
.badge-low    { background:#1a3a2a; color:#3fb950; border:1px solid #3fb950; padding:3px 10px; border-radius:20px; font-size:0.78rem; font-weight:600; }
.badge-medium { background:#3a2e0a; color:#d29922; border:1px solid #d29922; padding:3px 10px; border-radius:20px; font-size:0.78rem; font-weight:600; }
.badge-high   { background:#3a1a1a; color:#f85149; border:1px solid #f85149; padding:3px 10px; border-radius:20px; font-size:0.78rem; font-weight:600; }

/* Section headers */
.section-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #58a6ff;
    border-bottom: 1px solid #21262d;
    padding-bottom: 0.5rem;
    margin-bottom: 1.2rem;
}

/* Recommendation cards */
.rec-card {
    background: #161b22;
    border-left: 3px solid #58a6ff;
    border-radius: 0 8px 8px 0;
    padding: 0.9rem 1.1rem;
    margin-bottom: 0.7rem;
    color: #c9d1d9;
    font-size: 0.9rem;
}
.rec-card .rec-icon { font-size: 1.1rem; margin-right: 0.5rem; }

/* Probability gauge */
.prob-container {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 1.5rem;
    text-align: center;
}

/* Stmetric overrides */
[data-testid="metric-container"] {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 1rem;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #238636, #2ea043);
    color: white;
    border: none;
    border-radius: 8px;
    font-family: 'IBM Plex Sans', sans-serif;
    font-weight: 600;
    letter-spacing: 0.03em;
    transition: all 0.2s;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #2ea043, #3fb950);
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(63,185,80,0.25);
}

/* Selectbox / input styling */
.stSelectbox > div > div, .stNumberInput > div > div {
    background: #161b22 !important;
    border-color: #30363d !important;
    color: #e6edf3 !important;
}

/* Plotly chart dark background fix */
.js-plotly-plot .plotly { border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

# ─── Model Loading (cached) ────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading prediction model…")
def load_model():
    """Load the trained pipeline and optimized threshold."""
    base = os.path.dirname(__file__)
    model     = joblib.load(os.path.join(base, "telco_best_model.pkl"))
    threshold = float(np.load(os.path.join(base, "telco_best_threshold.npy"))[0])
    return model, threshold


@st.cache_data(show_spinner="Loading dataset…")
def load_data():
    """Load and lightly clean the Telco dataset."""
    base = os.path.dirname(__file__)
    df = pd.read_csv(os.path.join(base, "Telco-Customer-Churn.csv"))
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df.dropna(subset=["TotalCharges"], inplace=True)
    df["Churn_Binary"] = (df["Churn"] == "Yes").astype(int)
    return df


# ─── Session State Bootstrap ──────────────────────────────────────────────────
if "model" not in st.session_state:
    try:
        st.session_state.model, st.session_state.threshold = load_model()
        st.session_state.model_loaded = True
    except Exception as e:
        st.session_state.model_loaded = False
        st.session_state.model_error  = str(e)

if "df" not in st.session_state:
    try:
        st.session_state.df = load_data()
        st.session_state.data_loaded = True
    except Exception as e:
        st.session_state.data_loaded = False
        st.session_state.data_error  = str(e)

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:1.2rem 0 1.5rem; border-bottom:1px solid #30363d; margin-bottom:1.2rem;'>
        <div style='font-size:1.6rem; font-weight:700; color:#58a6ff; letter-spacing:-0.02em;'>📡 TelcoGuard</div>
        <div style='font-size:0.72rem; color:#8b949e; font-family:"IBM Plex Mono",monospace; letter-spacing:0.1em; text-transform:uppercase; margin-top:0.3rem;'>Churn Intelligence Platform</div>
    </div>
    """, unsafe_allow_html=True)

    # Model status
    if st.session_state.get("model_loaded"):
        threshold = st.session_state.threshold
        st.markdown(f"""
        <div style='background:#1a3a2a;border:1px solid #3fb950;border-radius:8px;padding:0.7rem 1rem;margin-bottom:1rem;'>
            <div style='font-size:0.68rem;color:#3fb950;font-family:"IBM Plex Mono",monospace;letter-spacing:0.1em;text-transform:uppercase;'>Model Status</div>
            <div style='color:#e6edf3;font-size:0.85rem;margin-top:0.2rem;'>✓ Loaded · Threshold <strong style='color:#58a6ff;'>{threshold:.3f}</strong></div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.error("⚠️ Model not found. Place `telco_best_model.pkl` and `telco_best_threshold.npy` in the app root.")

    st.markdown("""
    <div style='font-size:0.68rem;color:#8b949e;font-family:"IBM Plex Mono",monospace;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:0.5rem;'>Navigation</div>
    """, unsafe_allow_html=True)

    pages = {
        "🏠  Home":              "Home",
        "📊  Dashboard":         "pages/1_Dashboard.py",
        "👤  Single Prediction": "pages/2_Single_Customer.py",
        "📂  Batch Upload":      "pages/3_Batch_Upload.py",
    }
    for label in pages:
        st.markdown(f"<div style='padding:0.25rem 0;color:#c9d1d9;font-size:0.88rem;'>{label}</div>", unsafe_allow_html=True)

    st.markdown("---")
    if st.session_state.get("data_loaded"):
        df = st.session_state.df
        churn_rate = df["Churn_Binary"].mean() * 100
        st.markdown(f"""
        <div style='font-size:0.75rem;color:#8b949e;'>
            <div>📋 Records: <strong style='color:#e6edf3;'>{len(df):,}</strong></div>
            <div style='margin-top:0.3rem;'>📉 Churn Rate: <strong style='color:#f85149;'>{churn_rate:.1f}%</strong></div>
        </div>
        """, unsafe_allow_html=True)

# ─── Home Page ────────────────────────────────────────────────────────────────
st.markdown("""
<div style='margin-bottom:2rem;'>
    <h1 style='font-size:2.4rem;font-weight:700;color:#e6edf3;margin-bottom:0.4rem;letter-spacing:-0.03em;'>
        📡 TelcoGuard
        <span style='font-size:1rem;font-weight:400;color:#58a6ff;font-family:"IBM Plex Mono",monospace;margin-left:0.8rem;'>v1.0</span>
    </h1>
    <p style='color:#8b949e;font-size:1.05rem;max-width:620px;line-height:1.6;'>
        AI-powered customer churn prediction and retention intelligence. 
        Use the sidebar or pages below to explore the platform.
    </p>
</div>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class='metric-card'>
        <div style='font-size:2rem;margin-bottom:0.5rem;'>📊</div>
        <div style='font-size:1.1rem;font-weight:600;color:#e6edf3;margin-bottom:0.4rem;'>Dashboard</div>
        <div style='color:#8b949e;font-size:0.88rem;line-height:1.5;'>
            Fleet-level KPIs, churn distributions, feature importance, and interactive visualizations.
        </div>
        <div style='margin-top:1rem;font-size:0.75rem;color:#58a6ff;font-family:"IBM Plex Mono",monospace;'>→ Page 1</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class='metric-card'>
        <div style='font-size:2rem;margin-bottom:0.5rem;'>👤</div>
        <div style='font-size:1.1rem;font-weight:600;color:#e6edf3;margin-bottom:0.4rem;'>Single Prediction</div>
        <div style='color:#8b949e;font-size:0.88rem;line-height:1.5;'>
            Real-time churn scoring for individual customers with risk level, top reasons, and tailored recommendations.
        </div>
        <div style='margin-top:1rem;font-size:0.75rem;color:#58a6ff;font-family:"IBM Plex Mono",monospace;'>→ Page 2</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class='metric-card'>
        <div style='font-size:2rem;margin-bottom:0.5rem;'>📂</div>
        <div style='font-size:1.1rem;font-weight:600;color:#e6edf3;margin-bottom:0.4rem;'>Batch Upload</div>
        <div style='color:#8b949e;font-size:0.88rem;line-height:1.5;'>
            Upload a CSV to score thousands of customers at once. Download enriched results with probabilities and risk tiers.
        </div>
        <div style='margin-top:1rem;font-size:0.75rem;color:#58a6ff;font-family:"IBM Plex Mono",monospace;'>→ Page 3</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("""
<div style='background:#161b22;border:1px solid #30363d;border-radius:10px;padding:1.2rem 1.5rem;color:#8b949e;font-size:0.85rem;line-height:1.6;'>
    <strong style='color:#58a6ff;font-family:"IBM Plex Mono",monospace;font-size:0.72rem;letter-spacing:0.1em;text-transform:uppercase;'>Getting Started</strong><br><br>
    1. Navigate to <strong style='color:#e6edf3;'>Dashboard</strong> for an overview of your customer base.<br>
    2. Use <strong style='color:#e6edf3;'>Single Prediction</strong> to assess an individual customer's churn risk in real time.<br>
    3. Upload a CSV to <strong style='color:#e6edf3;'>Batch Upload</strong> for bulk scoring and export.<br>
    <br>
    Ensure <code style='background:#0d1117;padding:1px 5px;border-radius:3px;color:#79c0ff;'>telco_best_model.pkl</code>, 
    <code style='background:#0d1117;padding:1px 5px;border-radius:3px;color:#79c0ff;'>telco_best_threshold.npy</code>, and 
    <code style='background:#0d1117;padding:1px 5px;border-radius:3px;color:#79c0ff;'>Telco-Customer-Churn.csv</code> are in the app root directory.
</div>
""", unsafe_allow_html=True)