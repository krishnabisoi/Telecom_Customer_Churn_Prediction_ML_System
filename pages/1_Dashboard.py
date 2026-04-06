"""
1_Dashboard.py — Fleet-level Analytics Dashboard
=================================================
KPI cards, churn distributions, contract analysis,
tenure scatter, and feature importance chart.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utils import get_data, get_model, get_feature_importance, PLOTLY_LAYOUT, PLOTLY_COLORS, risk_level

st.set_page_config(page_title="Dashboard · TelcoGuard", page_icon="📊", layout="wide")

# ─── Inject shared CSS ────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
.main .block-container { background: #0d1117; padding-top: 1.5rem; }
[data-testid="stSidebar"] { background: linear-gradient(180deg,#0d1117,#161b22); border-right:1px solid #30363d; }
[data-testid="stSidebar"] * { color:#e6edf3 !important; }
.kpi-card { background:linear-gradient(135deg,#161b22,#1c2128); border:1px solid #30363d; border-radius:12px; padding:1.3rem 1.5rem; }
.kpi-label { font-family:'IBM Plex Mono',monospace; font-size:0.68rem; letter-spacing:.1em; text-transform:uppercase; color:#8b949e; }
.kpi-value { font-size:2.1rem; font-weight:700; color:#e6edf3; line-height:1.1; margin-top:.25rem; }
.kpi-sub { font-size:.78rem; color:#8b949e; margin-top:.25rem; }
.section-title { font-family:'IBM Plex Mono',monospace; font-size:.68rem; letter-spacing:.12em; text-transform:uppercase; color:#58a6ff; border-bottom:1px solid #21262d; padding-bottom:.4rem; margin:1.5rem 0 1rem; }
</style>
""", unsafe_allow_html=True)

# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown("<h2 style='color:#e6edf3;font-size:1.8rem;font-weight:700;margin-bottom:.2rem;'>📊 Analytics Dashboard</h2>", unsafe_allow_html=True)
st.markdown("<p style='color:#8b949e;font-size:.9rem;margin-bottom:1.5rem;'>Fleet-level churn intelligence · Real-time KPIs and trend analysis</p>", unsafe_allow_html=True)

# ─── Load Data ────────────────────────────────────────────────────────────────
df = get_data()
if df.empty:
    st.error("Dataset not loaded. Ensure `Telco-Customer-Churn.csv` is in the app root and restart.")
    st.stop()

model, threshold = get_model()

# ─── Sidebar filters ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("<div style='font-size:.72rem;color:#8b949e;font-family:\"IBM Plex Mono\",monospace;letter-spacing:.1em;text-transform:uppercase;margin-bottom:.5rem;'>Filters</div>", unsafe_allow_html=True)
    contract_filter = st.multiselect(
        "Contract Type",
        options=df["Contract"].unique().tolist(),
        default=df["Contract"].unique().tolist(),
    )
    internet_filter = st.multiselect(
        "Internet Service",
        options=df["InternetService"].unique().tolist(),
        default=df["InternetService"].unique().tolist(),
    )

dff = df[df["Contract"].isin(contract_filter) & df["InternetService"].isin(internet_filter)]

# ─── KPI Cards ────────────────────────────────────────────────────────────────
total     = len(dff)
churn_cnt = dff["Churn_Binary"].sum()
churn_pct = churn_cnt / total * 100 if total else 0
avg_charge = dff["MonthlyCharges"].mean()

# High-risk: predict_proba above threshold
if model is not None:
    try:
        from utils import preprocess_batch
        X_kpi = preprocess_batch(dff)
        probs_kpi = model.predict_proba(X_kpi)[:, 1]
        high_risk_cnt = int((probs_kpi >= threshold).sum())
        high_risk_pct = high_risk_cnt / total * 100
    except Exception:
        high_risk_cnt = churn_cnt
        high_risk_pct = churn_pct
else:
    high_risk_cnt = churn_cnt
    high_risk_pct = churn_pct

avg_tenure = dff["tenure"].mean()

c1, c2, c3, c4 = st.columns(4)
kpis = [
    (c1, "Total Customers",        f"{total:,}",              f"After applied filters"),
    (c2, "Churn Rate",             f"{churn_pct:.1f}%",        f"{churn_cnt:,} churned customers"),
    (c3, "High-Risk Customers",    f"{high_risk_cnt:,}",       f"{high_risk_pct:.1f}% above threshold"),
    (c4, "Avg Monthly Charges",    f"${avg_charge:.2f}",       f"Avg tenure {avg_tenure:.0f} mo"),
]
colors_kpi = ["#58a6ff", "#f85149", "#d29922", "#3fb950"]
for col, label, val, sub in kpis:
    with col:
        idx = kpis.index((col, label, val, sub))
        st.markdown(f"""
        <div class='kpi-card' style='border-top:3px solid {colors_kpi[idx]};'>
            <div class='kpi-label'>{label}</div>
            <div class='kpi-value' style='color:{colors_kpi[idx]};'>{val}</div>
            <div class='kpi-sub'>{sub}</div>
        </div>
        """, unsafe_allow_html=True)

# ─── Row 1: Churn Distribution + Contract vs Churn ───────────────────────────
st.markdown("<div class='section-title'>Churn & Contract Analysis</div>", unsafe_allow_html=True)
col_a, col_b = st.columns(2)

with col_a:
    # Churn distribution bar
    churn_counts = dff["Churn"].value_counts().reset_index()
    churn_counts.columns = ["Churn", "Count"]
    churn_counts["Color"] = churn_counts["Churn"].map({"Yes": PLOTLY_COLORS["red"], "No": PLOTLY_COLORS["green"]})

    fig_churn = go.Figure(go.Bar(
        x=churn_counts["Churn"],
        y=churn_counts["Count"],
        marker_color=churn_counts["Color"],
        text=churn_counts["Count"].apply(lambda x: f"{x:,}"),
        textposition="outside",
        textfont=dict(color="#e6edf3", size=13),
        hovertemplate="<b>%{x}</b><br>Count: %{y:,}<extra></extra>",
    ))
    fig_churn.update_layout(**PLOTLY_LAYOUT, title="Churn Distribution", height=320,
                            showlegend=False, xaxis_title="", yaxis_title="Customers")
    st.plotly_chart(fig_churn, use_container_width=True)

with col_b:
    # Contract vs Churn grouped bar
    ct = dff.groupby(["Contract", "Churn"]).size().reset_index(name="Count")
    fig_contract = px.bar(
        ct, x="Contract", y="Count", color="Churn",
        color_discrete_map={"Yes": PLOTLY_COLORS["red"], "No": PLOTLY_COLORS["green"]},
        barmode="group",
        text="Count",
    )
    fig_contract.update_traces(textposition="outside", textfont=dict(color="#e6edf3", size=11))
    fig_contract.update_layout(**PLOTLY_LAYOUT, title="Contract Type vs Churn", height=320,
                               xaxis_title="", yaxis_title="Customers")
    st.plotly_chart(fig_contract, use_container_width=True)

# ─── Row 2: Tenure vs Monthly Charges Scatter + Internet vs Churn ────────────
st.markdown("<div class='section-title'>Usage Patterns</div>", unsafe_allow_html=True)
col_c, col_d = st.columns(2)

with col_c:
    # Scatter: tenure vs MonthlyCharges, colored by churn
    sample = dff.sample(min(1500, len(dff)), random_state=42)
    fig_scatter = px.scatter(
        sample, x="tenure", y="MonthlyCharges", color="Churn",
        color_discrete_map={"Yes": PLOTLY_COLORS["red"], "No": PLOTLY_COLORS["blue"]},
        opacity=0.65,
        hover_data=["Contract", "InternetService"],
        labels={"tenure": "Tenure (months)", "MonthlyCharges": "Monthly Charges ($)"},
    )
    fig_scatter.update_traces(marker=dict(size=5))
    fig_scatter.update_layout(**PLOTLY_LAYOUT, title="Tenure vs Monthly Charges", height=320)
    st.plotly_chart(fig_scatter, use_container_width=True)

with col_d:
    # Internet service churn rate heatmap (horizontal bar)
    internet_churn = (
        dff.groupby("InternetService")["Churn_Binary"]
        .agg(["mean", "sum", "count"])
        .reset_index()
        .rename(columns={"mean": "ChurnRate", "sum": "Churned", "count": "Total"})
    )
    internet_churn["ChurnRate_Pct"] = (internet_churn["ChurnRate"] * 100).round(1)
    colors_bar = [PLOTLY_COLORS["red"] if r > 30 else PLOTLY_COLORS["yellow"] if r > 15 else PLOTLY_COLORS["green"]
                  for r in internet_churn["ChurnRate_Pct"]]

    fig_internet = go.Figure(go.Bar(
        x=internet_churn["ChurnRate_Pct"],
        y=internet_churn["InternetService"],
        orientation="h",
        marker_color=colors_bar,
        text=internet_churn["ChurnRate_Pct"].apply(lambda x: f"{x:.1f}%"),
        textposition="outside",
        textfont=dict(color="#e6edf3"),
        hovertemplate="<b>%{y}</b><br>Churn Rate: %{x:.1f}%<extra></extra>",
    ))
    fig_internet.update_layout(**PLOTLY_LAYOUT, title="Churn Rate by Internet Service",
                               height=320, xaxis_title="Churn Rate (%)", yaxis_title="",
                               showlegend=False)
    st.plotly_chart(fig_internet, use_container_width=True)

# ─── Row 3: Feature Importance + Monthly Charges Dist ─────────────────────────
st.markdown("<div class='section-title'>Model Intelligence</div>", unsafe_allow_html=True)
col_e, col_f = st.columns(2)

with col_e:
    if model is not None:
        feat_df = get_feature_importance(model, top_n=12)
        if not feat_df.empty:
            fig_feat = go.Figure(go.Bar(
                x=feat_df["importance"][::-1],
                y=feat_df["feature"][::-1],
                orientation="h",
                marker=dict(
                    color=feat_df["importance"][::-1],
                    colorscale=[[0, "#21262d"], [0.5, "#58a6ff"], [1, "#bc8cff"]],
                    showscale=False,
                ),
                hovertemplate="<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>",
            ))
            fig_feat.update_layout(**PLOTLY_LAYOUT, title="Top Feature Importances",
                                   height=360, xaxis_title="Importance", yaxis_title="")
            st.plotly_chart(fig_feat, use_container_width=True)
        else:
            st.info("Feature importance not available for this model type.")
    else:
        st.warning("Model not loaded — feature importance unavailable.")

with col_f:
    # Monthly charges distribution by churn
    fig_hist = go.Figure()
    for churn_val, color in [("No", PLOTLY_COLORS["blue"]), ("Yes", PLOTLY_COLORS["red"])]:
        subset = dff[dff["Churn"] == churn_val]["MonthlyCharges"]
        fig_hist.add_trace(go.Histogram(
            x=subset, name=f"Churn: {churn_val}",
            marker_color=color, opacity=0.72,
            nbinsx=30,
            hovertemplate="Range: %{x}<br>Count: %{y}<extra></extra>",
        ))
    fig_hist.update_layout(**PLOTLY_LAYOUT, title="Monthly Charges Distribution by Churn",
                           barmode="overlay", height=360,
                           xaxis_title="Monthly Charges ($)", yaxis_title="Count")
    st.plotly_chart(fig_hist, use_container_width=True)

# ─── Row 4: Tenure buckets ────────────────────────────────────────────────────
st.markdown("<div class='section-title'>Tenure Cohort Analysis</div>", unsafe_allow_html=True)

dff2 = dff.copy()
dff2["Tenure Bucket"] = pd.cut(
    dff2["tenure"],
    bins=[0, 12, 24, 36, 48, 60, 72],
    labels=["0–12 mo", "13–24 mo", "25–36 mo", "37–48 mo", "49–60 mo", "61–72 mo"],
)
tenure_churn = (
    dff2.groupby("Tenure Bucket", observed=True)["Churn_Binary"]
    .agg(["mean", "count"])
    .reset_index()
    .rename(columns={"mean": "ChurnRate", "count": "Total"})
)
tenure_churn["ChurnRate_Pct"] = (tenure_churn["ChurnRate"] * 100).round(1)

fig_tenure = go.Figure()
fig_tenure.add_trace(go.Bar(
    x=tenure_churn["Tenure Bucket"].astype(str),
    y=tenure_churn["Total"],
    name="Total Customers",
    marker_color=PLOTLY_COLORS["blue"],
    opacity=0.5,
    yaxis="y",
    hovertemplate="<b>%{x}</b><br>Total: %{y:,}<extra></extra>",
))
fig_tenure.add_trace(go.Scatter(
    x=tenure_churn["Tenure Bucket"].astype(str),
    y=tenure_churn["ChurnRate_Pct"],
    name="Churn Rate %",
    mode="lines+markers",
    marker=dict(size=8, color=PLOTLY_COLORS["red"]),
    line=dict(color=PLOTLY_COLORS["red"], width=2.5),
    yaxis="y2",
    hovertemplate="<b>%{x}</b><br>Churn Rate: %{y:.1f}%<extra></extra>",
))
fig_tenure.update_layout(
    **PLOTLY_LAYOUT,
    title="Customer Count & Churn Rate by Tenure Cohort",
    height=320,
    yaxis=dict(title="Total Customers", gridcolor="#21262d"),
    yaxis2=dict(title="Churn Rate (%)", overlaying="y", side="right", gridcolor="#21262d"),
    legend=dict(bgcolor="#161b22", bordercolor="#30363d", borderwidth=1),
)
st.plotly_chart(fig_tenure, use_container_width=True)