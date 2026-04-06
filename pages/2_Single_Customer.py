"""
2_Single_Customer.py — Individual Churn Risk Assessment
========================================================
Input form → preprocessing → model scoring → risk display + recommendations.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from utils import (
    predict_single, get_recommendations, get_feature_importance,
    get_model, risk_color, PLOTLY_LAYOUT, PLOTLY_COLORS
)

st.set_page_config(page_title="Single Prediction · TelcoGuard", page_icon="👤", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');
html, body, [class*="css"] { font-family:'IBM Plex Sans',sans-serif; }
.main .block-container { background:#0d1117; padding-top:1.5rem; }
[data-testid="stSidebar"] { background:linear-gradient(180deg,#0d1117,#161b22); border-right:1px solid #30363d; }
[data-testid="stSidebar"] * { color:#e6edf3 !important; }
.section-title { font-family:'IBM Plex Mono',monospace; font-size:.68rem; letter-spacing:.12em; text-transform:uppercase; color:#58a6ff; border-bottom:1px solid #21262d; padding-bottom:.4rem; margin:1.5rem 0 .8rem; }
.rec-card { background:#161b22; border-left:3px solid #58a6ff; border-radius:0 8px 8px 0; padding:.85rem 1.1rem; margin-bottom:.6rem; color:#c9d1d9; font-size:.88rem; line-height:1.5; }
.result-panel { background:linear-gradient(135deg,#161b22,#1c2128); border:1px solid #30363d; border-radius:14px; padding:1.6rem; }
.prob-big { font-size:3.5rem; font-weight:700; line-height:1; letter-spacing:-.02em; }
.risk-label { font-size:1.1rem; font-weight:600; margin-top:.4rem; }
.reason-item { background:#161b22; border:1px solid #30363d; border-radius:8px; padding:.75rem 1rem; margin-bottom:.5rem; display:flex; align-items:center; gap:.75rem; }
.reason-rank { background:#21262d; border-radius:50%; width:24px; height:24px; display:flex; align-items:center; justify-content:center; font-size:.75rem; font-weight:700; color:#58a6ff; flex-shrink:0; }
.reason-text { color:#c9d1d9; font-size:.87rem; }
.stButton>button { background:linear-gradient(135deg,#238636,#2ea043); color:white; border:none; border-radius:8px; font-weight:600; letter-spacing:.03em; padding:.55rem 1.5rem; transition:all .2s; }
.stButton>button:hover { background:linear-gradient(135deg,#2ea043,#3fb950); transform:translateY(-1px); }
.input-card { background:#161b22; border:1px solid #30363d; border-radius:10px; padding:1.1rem 1.2rem; margin-bottom:.8rem; }
</style>
""", unsafe_allow_html=True)

# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown("<h2 style='color:#e6edf3;font-size:1.8rem;font-weight:700;margin-bottom:.2rem;'>👤 Single Customer Prediction</h2>", unsafe_allow_html=True)
st.markdown("<p style='color:#8b949e;font-size:.9rem;margin-bottom:1.5rem;'>Score an individual customer's churn risk in real time.</p>", unsafe_allow_html=True)

model, threshold = get_model()
if model is None:
    st.error("Model not loaded. Ensure `telco_best_model.pkl` is in the app root.")
    st.stop()

# ─── Layout: Form left, Results right ─────────────────────────────────────────
form_col, result_col = st.columns([1, 1], gap="large")

with form_col:
    st.markdown("<div class='section-title'>Customer Profile</div>", unsafe_allow_html=True)

    with st.container():
        st.markdown("<div class='input-card'>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            tenure          = st.number_input("Tenure (months)", min_value=0, max_value=72, value=12, step=1)
            monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=200.0, value=65.0, step=0.5)
        with c2:
            total_charges   = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0,
                                               value=round(tenure * monthly_charges, 2), step=1.0)
            senior_citizen  = st.selectbox("Senior Citizen", ["No", "Yes"])
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='section-title'>Account Details</div>", unsafe_allow_html=True)
    with st.container():
        st.markdown("<div class='input-card'>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            gender          = st.selectbox("Gender", ["Male", "Female"])
            partner         = st.selectbox("Partner", ["Yes", "No"])
        with c2:
            dependents      = st.selectbox("Dependents", ["No", "Yes"])
            contract        = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        with c3:
            payment_method  = st.selectbox("Payment Method",
                                           ["Electronic check", "Mailed check",
                                            "Bank transfer (automatic)", "Credit card (automatic)"])
            paperless       = st.selectbox("Paperless Billing", ["Yes", "No"])
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='section-title'>Services</div>", unsafe_allow_html=True)
    with st.container():
        st.markdown("<div class='input-card'>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            phone_service   = st.selectbox("Phone Service", ["Yes", "No"])
            multiple_lines  = st.selectbox("Multiple Lines",  ["No", "Yes", "No phone service"])
            internet_svc    = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
        with c2:
            online_sec      = st.selectbox("Online Security",  ["No", "Yes", "No internet service"])
            online_backup   = st.selectbox("Online Backup",    ["No", "Yes", "No internet service"])
            device_prot     = st.selectbox("Device Protection",["No", "Yes", "No internet service"])
        with c3:
            tech_support    = st.selectbox("Tech Support",     ["No", "Yes", "No internet service"])
            streaming_tv    = st.selectbox("Streaming TV",     ["No", "Yes", "No internet service"])
            streaming_movies= st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
        st.markdown("</div>", unsafe_allow_html=True)

    predict_btn = st.button("🔍  Predict Churn Risk", use_container_width=True)

# ─── Results panel ────────────────────────────────────────────────────────────
with result_col:
    if predict_btn or st.session_state.get("last_prediction"):

        # Build customer dict
        customer = {
            "gender":             gender,
            "SeniorCitizen":      1 if senior_citizen == "Yes" else 0,
            "Partner":            partner,
            "Dependents":         dependents,
            "tenure":             tenure,
            "PhoneService":       phone_service,
            "MultipleLines":      multiple_lines,
            "InternetService":    internet_svc,
            "OnlineSecurity":     online_sec,
            "OnlineBackup":       online_backup,
            "DeviceProtection":   device_prot,
            "TechSupport":        tech_support,
            "StreamingTV":        streaming_tv,
            "StreamingMovies":    streaming_movies,
            "Contract":           contract,
            "PaperlessBilling":   paperless,
            "PaymentMethod":      payment_method,
            "MonthlyCharges":     monthly_charges,
            "TotalCharges":       total_charges,
        }

        if predict_btn:
            prob, pred, level = predict_single(customer)
            st.session_state.last_prediction = (prob, pred, level, customer)
        else:
            prob, pred, level, customer = st.session_state.last_prediction

        if prob is None:
            st.error("Prediction failed. Check model compatibility.")
            st.stop()

        color = risk_color(level)
        port_out = "YES" if pred == 1 else "NO"
        port_color = "#f85149" if pred == 1 else "#3fb950"

        # ── Probability gauge ─────────────────────────────────────────────────
        st.markdown("<div class='section-title'>Risk Assessment</div>", unsafe_allow_html=True)

        gauge_fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=round(prob * 100, 1),
            number=dict(suffix="%", font=dict(size=40, color=color, family="IBM Plex Sans")),
            gauge=dict(
                axis=dict(range=[0, 100], tickwidth=1, tickcolor="#30363d",
                          tickfont=dict(color="#8b949e", size=11)),
                bar=dict(color=color, thickness=0.25),
                bgcolor="#161b22",
                borderwidth=0,
                steps=[
                    dict(range=[0,  35], color="#1a3a2a"),
                    dict(range=[35, 65], color="#3a2e0a"),
                    dict(range=[65, 100], color="#3a1a1a"),
                ],
                threshold=dict(
                    line=dict(color="#e6edf3", width=2),
                    thickness=0.75,
                    value=threshold * 100,
                ),
            ),
            title=dict(text=f"Churn Probability", font=dict(color="#8b949e", size=13)),
        ))
        gauge_fig.update_layout(
            paper_bgcolor="#0d1117",
            font=dict(family="IBM Plex Sans"),
            height=260,
            margin=dict(l=30, r=30, t=30, b=10),
        )
        st.plotly_chart(gauge_fig, use_container_width=True)

        # ── Summary strip ─────────────────────────────────────────────────────
        col_lv, col_po = st.columns(2)
        with col_lv:
            st.markdown(f"""
            <div style='background:#161b22;border:1px solid {color};border-radius:10px;padding:1rem;text-align:center;'>
                <div style='font-size:.68rem;font-family:"IBM Plex Mono",monospace;letter-spacing:.1em;text-transform:uppercase;color:#8b949e;'>Risk Level</div>
                <div style='font-size:1.5rem;font-weight:700;color:{color};margin-top:.3rem;'>{level}</div>
            </div>
            """, unsafe_allow_html=True)
        with col_po:
            st.markdown(f"""
            <div style='background:#161b22;border:1px solid {port_color};border-radius:10px;padding:1rem;text-align:center;'>
                <div style='font-size:.68rem;font-family:"IBM Plex Mono",monospace;letter-spacing:.1em;text-transform:uppercase;color:#8b949e;'>Port-Out Risk</div>
                <div style='font-size:1.5rem;font-weight:700;color:{port_color};margin-top:.3rem;'>{port_out}</div>
            </div>
            """, unsafe_allow_html=True)

        # ── Top 3 Reasons ──────────────────────────────────────────────────────
        st.markdown("<div class='section-title'>Top Churn Drivers</div>", unsafe_allow_html=True)

        feat_df = get_feature_importance(model, top_n=20)
        if not feat_df.empty:
            # Map feature names to readable labels (best effort)
            clean_feats = []
            for fn in feat_df["feature"][:3]:
                # strip sklearn transformer prefixes like "remainder__", "cat__", "num__"
                parts = fn.split("__")
                readable = parts[-1].replace("_", " ").title()
                clean_feats.append(readable)

            for rank, feat_name in enumerate(clean_feats, 1):
                st.markdown(f"""
                <div class='reason-item'>
                    <div class='reason-rank'>{rank}</div>
                    <div class='reason-text'><strong style='color:#e6edf3;'>{feat_name}</strong> — key driver influencing this customer's churn likelihood.</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            for i, txt in enumerate([
                "Contract type is a strong predictor — month-to-month has highest risk.",
                "Tenure is inversely correlated — shorter tenure increases churn risk.",
                "Monthly charges level affects retention — higher bills drive churn.",
            ], 1):
                st.markdown(f"""
                <div class='reason-item'>
                    <div class='reason-rank'>{i}</div>
                    <div class='reason-text'>{txt}</div>
                </div>
                """, unsafe_allow_html=True)

        # ── Recommendations ────────────────────────────────────────────────────
        st.markdown("<div class='section-title'>Retention Recommendations</div>", unsafe_allow_html=True)
        recs = get_recommendations(customer, max_recs=3)
        for rec in recs:
            st.markdown(f"""
            <div class='rec-card'>
                <span style='font-size:1.1rem;'>{rec['icon']}</span>&nbsp;&nbsp;{rec['text']}
            </div>
            """, unsafe_allow_html=True)

    else:
        # Placeholder
        st.markdown("""
        <div style='background:#161b22;border:1px dashed #30363d;border-radius:14px;padding:3rem 2rem;text-align:center;margin-top:3rem;'>
            <div style='font-size:3rem;margin-bottom:1rem;'>🔍</div>
            <div style='color:#8b949e;font-size:1rem;line-height:1.6;'>
                Fill in the customer profile on the left<br>and click <strong style='color:#58a6ff;'>Predict Churn Risk</strong>.
            </div>
        </div>
        """, unsafe_allow_html=True)