"""
3_Batch_Upload.py — Bulk Customer Churn Scoring
================================================
Upload CSV → score all customers → display results → download enriched CSV.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import io
from utils import predict_batch, get_model, risk_color, PLOTLY_LAYOUT, PLOTLY_COLORS

st.set_page_config(page_title="Batch Upload · TelcoGuard", page_icon="📂", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');
html, body, [class*="css"] { font-family:'IBM Plex Sans',sans-serif; }
.main .block-container { background:#0d1117; padding-top:1.5rem; }
[data-testid="stSidebar"] { background:linear-gradient(180deg,#0d1117,#161b22); border-right:1px solid #30363d; }
[data-testid="stSidebar"] * { color:#e6edf3 !important; }
.section-title { font-family:'IBM Plex Mono',monospace; font-size:.68rem; letter-spacing:.12em; text-transform:uppercase; color:#58a6ff; border-bottom:1px solid #21262d; padding-bottom:.4rem; margin:1.5rem 0 .8rem; }
.kpi-card { background:linear-gradient(135deg,#161b22,#1c2128); border:1px solid #30363d; border-radius:12px; padding:1.1rem 1.3rem; }
.kpi-label { font-family:'IBM Plex Mono',monospace; font-size:.68rem; letter-spacing:.1em; text-transform:uppercase; color:#8b949e; }
.kpi-value { font-size:1.9rem; font-weight:700; color:#e6edf3; line-height:1.1; margin-top:.2rem; }
.upload-zone { background:#161b22; border:2px dashed #30363d; border-radius:14px; padding:2.5rem; text-align:center; transition:border-color .2s; }
.stButton>button { background:linear-gradient(135deg,#1f6feb,#388bfd); color:white; border:none; border-radius:8px; font-weight:600; letter-spacing:.03em; transition:all .2s; }
.stButton>button:hover { background:linear-gradient(135deg,#388bfd,#58a6ff); transform:translateY(-1px); }
.dl-button>button { background:linear-gradient(135deg,#238636,#2ea043) !important; }
.dl-button>button:hover { background:linear-gradient(135deg,#2ea043,#3fb950) !important; }
</style>
""", unsafe_allow_html=True)

# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown("<h2 style='color:#e6edf3;font-size:1.8rem;font-weight:700;margin-bottom:.2rem;'>📂 Batch Customer Scoring</h2>", unsafe_allow_html=True)
st.markdown("<p style='color:#8b949e;font-size:.9rem;margin-bottom:1.5rem;'>Upload a CSV of customers to score churn probability for the entire file at once.</p>", unsafe_allow_html=True)

model, threshold = get_model()
if model is None:
    st.error("Model not loaded. Ensure `telco_best_model.pkl` is in the app root.")
    st.stop()

# ─── Upload Zone ──────────────────────────────────────────────────────────────
st.markdown("<div class='section-title'>Upload Data</div>", unsafe_allow_html=True)

col_upload, col_info = st.columns([2, 1], gap="large")

with col_upload:
    uploaded_file = st.file_uploader(
        "Drop a CSV file here or click to browse",
        type=["csv"],
        help="CSV must contain the same columns used during model training.",
    )

with col_info:
    st.markdown("""
    <div style='background:#161b22;border:1px solid #30363d;border-radius:10px;padding:1.1rem 1.3rem;'>
        <div style='font-size:.68rem;font-family:"IBM Plex Mono",monospace;letter-spacing:.1em;text-transform:uppercase;color:#58a6ff;margin-bottom:.7rem;'>Expected Columns</div>
        <div style='color:#c9d1d9;font-size:.82rem;line-height:1.7;'>
            gender, SeniorCitizen, Partner, Dependents,<br>
            tenure, PhoneService, MultipleLines,<br>
            InternetService, OnlineSecurity, OnlineBackup,<br>
            DeviceProtection, TechSupport, StreamingTV,<br>
            StreamingMovies, Contract, PaperlessBilling,<br>
            PaymentMethod, MonthlyCharges, TotalCharges
        </div>
        <div style='margin-top:.8rem;font-size:.78rem;color:#8b949e;'>
            <em>customerID and Churn columns are optional and will be preserved.</em>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ─── Processing ───────────────────────────────────────────────────────────────
if uploaded_file is not None:
    try:
        raw_df = pd.read_csv(uploaded_file)
        st.success(f"✅ File loaded: **{uploaded_file.name}** — {len(raw_df):,} rows, {len(raw_df.columns)} columns")
    except Exception as e:
        st.error(f"Failed to parse CSV: {e}")
        st.stop()

    # Preview raw
    st.markdown("<div class='section-title'>Raw Data Preview</div>", unsafe_allow_html=True)
    st.dataframe(raw_df.head(10), use_container_width=True, height=260)

    # Run predictions
    if st.button("⚡  Run Batch Predictions", use_container_width=False):
        with st.spinner("Scoring customers…"):
            try:
                result_df = predict_batch(raw_df)
                st.session_state.batch_results = result_df
                st.session_state.batch_filename = uploaded_file.name
            except Exception as e:
                st.error(f"Prediction error: {e}")
                st.stop()
        st.success(f"✅ Scored {len(result_df):,} customers successfully.")

# ─── Results ──────────────────────────────────────────────────────────────────
if st.session_state.get("batch_results") is not None:
    result_df = st.session_state.batch_results

    # ── KPI strip ─────────────────────────────────────────────────────────────
    st.markdown("<div class='section-title'>Batch Summary</div>", unsafe_allow_html=True)

    total_scored   = len(result_df)
    predicted_churn = int(result_df["churn_prediction"].sum())
    churn_rate_pred = predicted_churn / total_scored * 100 if total_scored else 0

    risk_counts = result_df["risk_level"].value_counts()
    high_risk   = int(risk_counts.get("High",   0))
    medium_risk = int(risk_counts.get("Medium", 0))
    low_risk    = int(risk_counts.get("Low",    0))
    avg_prob    = result_df["churn_probability"].mean() * 100

    c1, c2, c3, c4, c5 = st.columns(5)
    kpi_data = [
        (c1, "Scored",        f"{total_scored:,}",        "#58a6ff"),
        (c2, "Predicted Churn",f"{predicted_churn:,}",   "#f85149"),
        (c3, "High Risk",     f"{high_risk:,}",           "#f85149"),
        (c4, "Medium Risk",   f"{medium_risk:,}",         "#d29922"),
        (c5, "Avg Probability",f"{avg_prob:.1f}%",        "#bc8cff"),
    ]
    for col, label, val, color in kpi_data:
        with col:
            st.markdown(f"""
            <div class='kpi-card' style='border-top:3px solid {color};'>
                <div class='kpi-label'>{label}</div>
                <div class='kpi-value' style='color:{color};'>{val}</div>
            </div>
            """, unsafe_allow_html=True)

    # ── Charts ────────────────────────────────────────────────────────────────
    st.markdown("<div class='section-title'>Prediction Analytics</div>", unsafe_allow_html=True)
    ch1, ch2 = st.columns(2)

    with ch1:
        # Risk level donut
        risk_vals = [high_risk, medium_risk, low_risk]
        risk_labels = ["High", "Medium", "Low"]
        risk_colors_list = [PLOTLY_COLORS["red"], PLOTLY_COLORS["yellow"], PLOTLY_COLORS["green"]]
        fig_donut = go.Figure(go.Pie(
            labels=risk_labels, values=risk_vals,
            hole=0.55,
            marker=dict(colors=risk_colors_list, line=dict(color="#0d1117", width=2)),
            textfont=dict(color="#e6edf3"),
            hovertemplate="<b>%{label}</b><br>Count: %{value:,}<br>Share: %{percent}<extra></extra>",
        ))
        fig_donut.update_layout(**PLOTLY_LAYOUT, title="Risk Level Distribution", height=300,
                                annotations=[dict(text=f"{churn_rate_pred:.1f}%<br><span style='font-size:10px'>Churn Rate</span>",
                                                  x=0.5, y=0.5, font_size=18, showarrow=False,
                                                  font=dict(color="#e6edf3"))])
        st.plotly_chart(fig_donut, use_container_width=True)

    with ch2:
        # Probability histogram
        fig_hist = go.Figure(go.Histogram(
            x=result_df["churn_probability"],
            nbinsx=30,
            marker_color=PLOTLY_COLORS["blue"],
            opacity=0.8,
            hovertemplate="Prob: %{x:.2f}<br>Count: %{y}<extra></extra>",
        ))
        fig_hist.add_vline(x=threshold, line_dash="dash", line_color="#f85149",
                           annotation_text=f"Threshold {threshold:.3f}",
                           annotation_font_color="#f85149", annotation_font_size=11)
        fig_hist.update_layout(**PLOTLY_LAYOUT, title="Churn Probability Distribution",
                               height=300, xaxis_title="Churn Probability", yaxis_title="Count")
        st.plotly_chart(fig_hist, use_container_width=True)

    # ── Results table ─────────────────────────────────────────────────────────
    st.markdown("<div class='section-title'>Enriched Results</div>", unsafe_allow_html=True)

    # Color styling for risk_level column
    def style_risk(val):
        colors = {"High": "color: #f85149; font-weight:600;",
                  "Medium": "color: #d29922; font-weight:600;",
                  "Low": "color: #3fb950; font-weight:600;"}
        return colors.get(val, "")

    def style_prob(val):
        try:
            v = float(val)
            if v >= 0.65:   return "color:#f85149;"
            elif v >= 0.35: return "color:#d29922;"
            return "color:#3fb950;"
        except Exception:
            return ""

    display_cols = [c for c in result_df.columns if c in
                    (["customerID", "gender", "tenure", "Contract", "MonthlyCharges",
                      "InternetService", "churn_probability", "churn_prediction", "risk_level"])]
    display_df = result_df[display_cols] if display_cols else result_df

    styled = display_df.style.applymap(style_risk, subset=["risk_level"]) \
                             .applymap(style_prob, subset=["churn_probability"]) \
                             .format({"churn_probability": "{:.4f}"})

    st.dataframe(styled, use_container_width=True, height=380)

    # ── Filters ───────────────────────────────────────────────────────────────
    with st.expander("🎛️  Filter & Explore Results"):
        risk_filter = st.multiselect("Filter by Risk Level",
                                     options=["High", "Medium", "Low"],
                                     default=["High", "Medium", "Low"])
        filtered = result_df[result_df["risk_level"].isin(risk_filter)]
        st.dataframe(filtered, use_container_width=True, height=300)
        st.markdown(f"<span style='color:#8b949e;font-size:.82rem;'>{len(filtered):,} customers match the filter.</span>", unsafe_allow_html=True)

    # ── Download ──────────────────────────────────────────────────────────────
    st.markdown("<div class='section-title'>Export Results</div>", unsafe_allow_html=True)

    csv_bytes = result_df.to_csv(index=False).encode("utf-8")
    orig_name = st.session_state.get("batch_filename", "results")
    out_name  = orig_name.replace(".csv", "") + "_churn_predictions.csv"

    st.markdown("<div class='dl-button'>", unsafe_allow_html=True)
    st.download_button(
        label="⬇️  Download Predictions CSV",
        data=csv_bytes,
        file_name=out_name,
        mime="text/csv",
        use_container_width=False,
    )
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown(f"<span style='color:#8b949e;font-size:.82rem;'>File will contain all original columns plus: <code>churn_probability</code>, <code>churn_prediction</code>, <code>risk_level</code>.</span>", unsafe_allow_html=True)

else:
    if uploaded_file is None:
        st.markdown("""
        <div style='background:#161b22;border:2px dashed #30363d;border-radius:14px;padding:3rem;text-align:center;margin-top:1rem;'>
            <div style='font-size:3rem;margin-bottom:.8rem;'>📂</div>
            <div style='color:#8b949e;font-size:1rem;'>Upload a CSV file above to begin batch scoring.</div>
            <div style='color:#58a6ff;font-size:.82rem;margin-top:.5rem;font-family:"IBM Plex Mono",monospace;'>Supports files with 1 to 100,000+ rows</div>
        </div>
        """, unsafe_allow_html=True)