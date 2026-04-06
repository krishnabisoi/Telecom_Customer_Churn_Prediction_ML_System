"""
utils.py — Shared helpers for TelcoGuard
=========================================
Preprocessing, prediction, risk classification, and recommendations.
"""

import pandas as pd
import numpy as np
import streamlit as st
import joblib
import os


# ─── Constants ────────────────────────────────────────────────────────────────

RISK_THRESHOLDS = {"low": 0.35, "medium": 0.65}  # < low → Low, < medium → Medium, else → High

# Columns expected by the model (mirrors training preprocessing)
BINARY_YES_NO_COLS = [
    "Partner", "Dependents", "PhoneService", "MultipleLines",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "PaperlessBilling",
]

# Rule-based recommendation triggers
RECOMMENDATION_RULES = [
    {
        "condition": lambda r: r.get("Contract") == "Month-to-month",
        "icon": "📋",
        "text": "Customer is on a month-to-month plan — offer a discounted annual or two-year contract to lock in loyalty.",
    },
    {
        "condition": lambda r: float(r.get("MonthlyCharges", 0)) > 70,
        "icon": "💰",
        "text": "Monthly charges are elevated — consider a targeted retention discount or bundle upgrade offer.",
    },
    {
        "condition": lambda r: r.get("TechSupport") in ("No", "No internet service"),
        "icon": "🛠️",
        "text": "No TechSupport enrolled — proactively pitch the TechSupport add-on to improve stickiness.",
    },
    {
        "condition": lambda r: r.get("OnlineSecurity") in ("No", "No internet service"),
        "icon": "🔒",
        "text": "Online Security not active — highlight security risks and offer a free trial of the service.",
    },
    {
        "condition": lambda r: r.get("InternetService") == "Fiber optic",
        "icon": "📶",
        "text": "Fiber customer with potential churn — reinforce network quality messaging and offer a speed upgrade.",
    },
    {
        "condition": lambda r: int(r.get("tenure", 0)) < 12,
        "icon": "🎁",
        "text": "Customer tenure < 12 months — send a 'first-year anniversary' loyalty reward to build long-term relationship.",
    },
    {
        "condition": lambda r: r.get("PaymentMethod") == "Electronic check",
        "icon": "💳",
        "text": "Paying by electronic check — encourage switching to auto-pay (credit card/bank transfer) with a small bill credit.",
    },
]


# ─── Model / Data Access ──────────────────────────────────────────────────────

def get_model():
    """Retrieve model and threshold from session state."""
    return st.session_state.get("model"), st.session_state.get("threshold", 0.5)


def get_data() -> pd.DataFrame:
    """Retrieve the loaded dataset from session state."""
    return st.session_state.get("df", pd.DataFrame())


# ─── Preprocessing ────────────────────────────────────────────────────────────

def preprocess_input(raw: dict) -> pd.DataFrame:
    """
    Convert a raw dict of customer inputs into a single-row DataFrame
    ready for model.predict_proba().
    Mirrors training-time preprocessing.
    """
    df = pd.DataFrame([raw])

    # Numeric coercion
    for col in ["tenure", "MonthlyCharges", "TotalCharges"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # Drop customerID / Churn if present
    df.drop(columns=[c for c in ["customerID", "Churn"] if c in df.columns], inplace=True)

    return df


def preprocess_batch(df: pd.DataFrame) -> pd.DataFrame:
    """Clean a batch DataFrame before prediction."""
    df = df.copy()
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df.dropna(subset=["TotalCharges"], inplace=True)
    df.drop(columns=[c for c in ["customerID", "Churn", "Churn_Binary"] if c in df.columns], inplace=True)
    return df


# ─── Prediction ───────────────────────────────────────────────────────────────

def predict_single(raw: dict):
    """
    Returns (probability, prediction, risk_level) for one customer dict.
    """
    model, threshold = get_model()
    if model is None:
        return None, None, None

    X = preprocess_input(raw)
    prob  = model.predict_proba(X)[0][1]
    pred  = 1 if prob >= threshold else 0
    level = risk_level(prob)
    return float(prob), int(pred), level


def predict_batch(df: pd.DataFrame):
    """
    Returns a copy of df with churn_probability, churn_prediction, risk_level columns.
    """
    model, threshold = get_model()
    if model is None:
        return df

    X    = preprocess_batch(df)
    probs = model.predict_proba(X)[:, 1]
    preds = (probs >= threshold).astype(int)

    out = df.copy()
    # Re-index to avoid mismatches after dropna in preprocess_batch
    out = out.loc[X.index].copy()
    out["churn_probability"] = np.round(probs, 4)
    out["churn_prediction"]  = preds
    out["risk_level"]        = [risk_level(p) for p in probs]
    return out


# ─── Risk Classification ──────────────────────────────────────────────────────

def risk_level(prob: float) -> str:
    if prob < RISK_THRESHOLDS["low"]:
        return "Low"
    elif prob < RISK_THRESHOLDS["medium"]:
        return "Medium"
    return "High"


def risk_color(level: str) -> str:
    return {"Low": "#3fb950", "Medium": "#d29922", "High": "#f85149"}.get(level, "#8b949e")


def risk_badge_html(level: str) -> str:
    cls = {"Low": "badge-low", "Medium": "badge-medium", "High": "badge-high"}.get(level, "")
    return f"<span class='{cls}'>{level} Risk</span>"


# ─── Feature Importance ───────────────────────────────────────────────────────

def get_feature_importance(model, top_n: int = 10) -> pd.DataFrame:
    """
    Extract feature importances from the pipeline's final estimator.
    Falls back to coefficient magnitudes for linear models.
    Returns DataFrame(feature, importance).
    """
    try:
        # Navigate pipeline steps to find the estimator
        estimator = model
        feature_names = None

        # Try to get feature names from pipeline preprocessor
        if hasattr(model, "named_steps"):
            steps = list(model.named_steps.values())
            estimator = steps[-1]  # last step = classifier

            # Try to reconstruct feature names from preprocessor
            if hasattr(steps[0], "get_feature_names_out"):
                try:
                    feature_names = list(steps[0].get_feature_names_out())
                except Exception:
                    pass

        # Get importances
        if hasattr(estimator, "feature_importances_"):
            importances = estimator.feature_importances_
        elif hasattr(estimator, "coef_"):
            importances = np.abs(estimator.coef_[0])
        else:
            return pd.DataFrame()

        if feature_names is None or len(feature_names) != len(importances):
            feature_names = [f"Feature {i}" for i in range(len(importances))]

        df = pd.DataFrame({"feature": feature_names, "importance": importances})
        df = df.sort_values("importance", ascending=False).head(top_n).reset_index(drop=True)
        return df

    except Exception:
        return pd.DataFrame()


# ─── Recommendations ─────────────────────────────────────────────────────────

def get_recommendations(customer_dict: dict, max_recs: int = 3) -> list:
    """
    Apply rule-based recommendation engine.
    Returns list of dicts with icon + text.
    """
    recs = []
    for rule in RECOMMENDATION_RULES:
        try:
            if rule["condition"](customer_dict):
                recs.append({"icon": rule["icon"], "text": rule["text"]})
        except Exception:
            pass
        if len(recs) >= max_recs:
            break
    if not recs:
        recs.append({"icon": "✅", "text": "No immediate retention actions required. Monitor usage quarterly."})
    return recs


# ─── Plotly theme helper ──────────────────────────────────────────────────────

PLOTLY_LAYOUT = dict(
    paper_bgcolor="#0d1117",
    plot_bgcolor="#0d1117",
    font=dict(family="IBM Plex Sans, sans-serif", color="#c9d1d9", size=12),
    xaxis=dict(gridcolor="#21262d", linecolor="#30363d"),
    yaxis=dict(gridcolor="#21262d", linecolor="#30363d"),
    margin=dict(l=20, r=20, t=40, b=20),
    legend=dict(bgcolor="#161b22", bordercolor="#30363d", borderwidth=1),
)

PLOTLY_COLORS = {
    "blue":   "#58a6ff",
    "green":  "#3fb950",
    "yellow": "#d29922",
    "red":    "#f85149",
    "purple": "#bc8cff",
    "teal":   "#39d353",
    "grey":   "#8b949e",
}