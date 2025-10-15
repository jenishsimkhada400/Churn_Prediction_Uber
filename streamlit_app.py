# app/streamlit_app.py
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from joblib import load as joblib_load

# =========================  
# Paths & basic constants
# =========================
APP_DIR = Path(__file__).resolve().parent
ROOT_DIR = APP_DIR.parent
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
FIGS_DIR = ROOT_DIR / "notebooks" / "figs"  # where your notebooks saved figures (optional)

LOGIT_PATH = MODELS_DIR / "logit_pipeline.pkl"
XGB_PATH = MODELS_DIR / "xgb_pipeline.pkl"

SAMPLE_PARQUET = DATA_DIR / "churn_dataset.parquet"
SAMPLE_CSV = DATA_DIR / "churn_dataset.csv"
TARGET_COL = "churn_30d"

# =========================================================
# Feature alignment helpers (legacy -> current, auto-complete)
# =========================================================
LEGACY_RENAME = {
    # Older "lookback" naming used in early notebook versions:
    "num__fare_sum_lb": "num__fare_sum_90d",
    "num__fare_mean_lb": "num__fare_mean_90d",
    "num__dist_sum_lb": "num__dist_sum_90d",
    "num__dist_mean_lb": "num__dist_mean_90d",
    "num__rides_lookback": "num__rides_90d",
    # Add more mappings here if you discover other legacy names.
}

def get_required_raw_columns(pipeline) -> list[str] | None:
    """
    Returns the raw input column names expected by the ColumnTransformer
    inside the pipeline's 'prep' step (before one-hot encoding).
    """
    prep = getattr(pipeline, "named_steps", {}).get("prep")
    if prep is None:
        return None
    required = []
    for _, _trans, cols in getattr(prep, "transformers_", []):
        if isinstance(cols, (list, tuple)):
            required.extend(list(cols))
    # De-duplicate while keeping order
    return list(dict.fromkeys(required))

def align_features_to_pipeline(df: pd.DataFrame, pipeline) -> pd.DataFrame:
    """
    - Renames legacy columns to the new *_90d schema
    - Adds any missing raw input columns (as NaN; imputer will handle)
    - Returns the re-ordered dataframe (required columns first)
    """
    if df is None or df.empty:
        return df
    df = df.copy()

    # 1) rename old -> new
    df.rename(columns=LEGACY_RENAME, inplace=True)

    # 2) ensure required raw columns exist
    required = get_required_raw_columns(pipeline)
    if not required:
        return df

    for col in required:
        if col not in df.columns:
            df[col] = np.nan

    # 3) keep required columns first (tidy; not strictly required)
    extras = [c for c in df.columns if c not in required]
    df = df[required + extras]
    return df

# =========================
# Loading utilities
# =========================
@st.cache_resource(show_spinner=False)
def load_pipelines():
    logit_pipe = joblib_load(LOGIT_PATH)
    xgb_pipe = joblib_load(XGB_PATH)
    return logit_pipe, xgb_pipe

def load_sample_features(n_rows: int | None = 1000) -> pd.DataFrame:
    """Read the sample features built by the 02 notebook (drops TARGET_COL if present)."""
    if SAMPLE_PARQUET.exists():
        df = pd.read_parquet(SAMPLE_PARQUET)
    elif SAMPLE_CSV.exists():
        df = pd.read_csv(SAMPLE_CSV)
    else:
        raise FileNotFoundError(
            "Could not find sample features. Run the 02_feature_building notebook first."
        )
    if TARGET_COL in df.columns:
        df = df.drop(columns=[TARGET_COL])
    if n_rows:
        df = df.head(n_rows)
    return df

def detect_id_column(df: pd.DataFrame) -> str | None:
    """Best-effort to find a user/customer id column for display/export."""
    candidates = [
        "Customer ID",
        "customer_id",
        "user_id",
        "User ID",
        "cid",
        "uid",
    ]
    for c in candidates:
        if c in df.columns:
            return c
    # fallback: if any column name contains 'customer' or 'user'
    for c in df.columns:
        lc = c.lower()
        if "customer" in lc or "user" in lc:
            return c
    return None

# =========================
# Scoring helpers
# =========================
def score_with_pipeline(pipe, X_raw: pd.DataFrame) -> np.ndarray:
    """Align features to the pipeline's expected inputs and return positive-class probabilities."""
    X = align_features_to_pipeline(X_raw, pipe)
    probs = pipe.predict_proba(X)[:, 1]
    return probs

def make_retention_list(df: pd.DataFrame, probs: np.ndarray, frac: float, id_col: str | None):
    out = df.copy()
    out["prob_churn"] = probs
    out = out.sort_values("prob_churn", ascending=False)
    k = max(1, int(len(out) * frac))
    top = out.head(k)
    export_cols = [id_col, "prob_churn"] if id_col and id_col in top.columns else ["prob_churn"]
    return top[export_cols].reset_index(drop=True)

# =========================
# Streamlit App
# =========================
st.set_page_config(page_title="Churn Prediction — Scoring", layout="wide")
st.title("🧭 Churn Prediction — Scoring & Exploration")

logit_pipe, xgb_pipe = load_pipelines()

# ---- Sidebar: configuration ----
st.sidebar.header("Configuration")
model_choice = st.sidebar.selectbox("Model", ["XGBoost", "Logistic Regression"], index=0)
retention_frac = st.sidebar.slider("Top fraction for retention list", 0.01, 0.50, 0.15, 0.01)

st.sidebar.markdown("## Data source")
use_sample = st.sidebar.radio(
    "Choose input data",
    ["Use sample features (from data/churn_dataset.*)", "Upload CSV"],
    index=0,
)

# ---- Load dataframe to score ----
df_input = None
upload = None
if use_sample.startswith("Use sample"):
    df_input = load_sample_features(n_rows=None)
else:
    upload = st.sidebar.file_uploader("Upload a CSV with the same schema used to train", type=["csv"])
    if upload is not None:
        df_input = pd.read_csv(upload)

if df_input is None or df_input.empty:
    st.warning("No input data available. Provide a CSV or use the sample features.")
    st.stop()

id_col = detect_id_column(df_input)

# Preview
st.success(f"Loaded {len(df_input):,} rows with {df_input.shape[1]} columns.")
st.dataframe(df_input.head(10), use_container_width=True)

# ---- Choose pipeline & score ----
pipe = xgb_pipe if model_choice == "XGBoost" else logit_pipe

try:
    probs = score_with_pipeline(pipe, df_input)
except Exception as e:
    st.error(
        "Scoring failed. Most common cause is a column mismatch.\n\n"
        f"Details: {type(e).__name__}: {e}"
    )
    st.stop()

# ---- Show ranked table ----
scored = df_input.copy()
scored["prob_churn"] = probs
scored = scored.sort_values("prob_churn", ascending=False)

st.subheader("Predictions (top 20 by churn probability)")
display_cols = [id_col, "prob_churn"] if id_col and id_col in scored.columns else ["prob_churn"]
st.dataframe(scored[display_cols].head(20), use_container_width=True)

# ---- Retention list creation / download ----
st.subheader("Retention list export")
top_df = make_retention_list(df_input, probs, retention_frac, id_col)

col1, col2 = st.columns(2)
with col1:
    st.write(f"Top fraction: **{retention_frac:.0%}** → **{len(top_df):,}** customers")
with col2:
    st.write(f"Baseline positive rate is not shown here (computed in Notebook 03).")

csv_bytes = top_df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="⬇️ Download retention list (CSV)",
    data=csv_bytes,
    file_name="retention_list_top_fraction.csv",
    mime="text/csv",
)

# Also save alongside the app run (optional convenience)
out_path = ROOT_DIR / "retention_list_top_decile.csv"
try:
    top_df.to_csv(out_path, index=False)
    st.caption(f"Saved retention list to: `{out_path}`")
except Exception:
    pass

# ---- (Optional) show lift & calibration images if present ----
with st.expander("Optional visuals (if generated in notebooks)"):
    import base64

    def show_png(path: Path, label: str):
        if path.exists():
            b = path.read_bytes()
            st.markdown(f"**{label}**")
            st.image(b)
        else:
            st.caption(f"{label}: not found at `{path}`")

    show_png(FIGS_DIR / "logreg_lift_curve.png", "LogReg Lift Curve")
    show_png(FIGS_DIR / "xgb_lift_curve.png", "XGB Lift Curve")
    show_png(FIGS_DIR / "logreg_calibration.png", "LogReg Calibration")
    show_png(FIGS_DIR / "xgb_calibration.png", "XGB Calibration")