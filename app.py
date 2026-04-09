import os
import io
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="MOLM-1 Chromatin Dashboard",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# CUSTOM STYLING
# =========================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: linear-gradient(180deg, #f8f9ff 0%, #fff8fc 35%, #f7fbff 100%);
}

.block-container {
    max-width: 1450px;
    padding-top: 1rem;
    padding-bottom: 2rem;
}

.hero {
    background: linear-gradient(135deg, rgba(124,58,237,0.14), rgba(37,99,235,0.10), rgba(236,72,153,0.10));
    border: 1px solid rgba(124,58,237,0.12);
    border-radius: 24px;
    padding: 1.3rem 1.4rem;
    margin-bottom: 1rem;
    box-shadow: 0 10px 30px rgba(15,23,42,0.06);
}

.hero-title {
    font-size: 2.2rem;
    font-weight: 800;
    color: #1e293b;
}

.hero-sub {
    font-size: 1rem;
    color: #64748b;
    margin-top: 0.35rem;
}

.note {
    background: #f5f3ff;
    border-left: 5px solid #7c3aed;
    padding: 0.95rem 1rem;
    border-radius: 14px;
    margin: 0.8rem 0 1rem 0;
    color: #334155;
}

.info-soft {
    background: linear-gradient(135deg, #eff6ff, #fdf2f8);
    border-left: 5px solid #2563eb;
    padding: 0.95rem 1rem;
    border-radius: 14px;
    margin: 0.8rem 0 1rem 0;
    color: #334155;
}

.success-soft {
    background: linear-gradient(135deg, #ecfdf5, #f0fdf4);
    border-left: 5px solid #16a34a;
    padding: 0.95rem 1rem;
    border-radius: 14px;
    margin: 0.8rem 0 1rem 0;
    color: #14532d;
}

.warn-soft {
    background: linear-gradient(135deg, #fff7ed, #fef2f2);
    border-left: 5px solid #ea580c;
    padding: 0.95rem 1rem;
    border-radius: 14px;
    margin: 0.8rem 0 1rem 0;
    color: #7c2d12;
}

.card {
    background: #ffffff;
    border: 1px solid rgba(148,163,184,0.18);
    border-radius: 18px;
    padding: 1rem;
    box-shadow: 0 6px 20px rgba(15,23,42,0.05);
}

.section-title {
    font-size: 1.25rem;
    font-weight: 800;
    color: #1e293b;
    margin-top: 0.4rem;
    margin-bottom: 0.65rem;
}

.algorithm-step {
    background: linear-gradient(135deg, #ffffff, #faf5ff);
    border: 1px solid rgba(124,58,237,0.15);
    border-left: 5px solid #7c3aed;
    border-radius: 16px;
    padding: 0.9rem 1rem;
    margin-bottom: 0.75rem;
    color: #334155;
}

.algorithm-step b {
    color: #4c1d95;
}

.metric-shell {
    background: white;
    border-radius: 18px;
    border: 1px solid rgba(148,163,184,0.16);
    padding: 0.5rem;
    box-shadow: 0 4px 18px rgba(15,23,42,0.05);
}

[data-testid="stMetric"] {
    background: #fff;
    border: 1px solid rgba(148,163,184,0.12);
    border-radius: 16px;
    padding: 0.8rem;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #f8fafc 0%, #eef2ff 100%);
}

.small-text {
    font-size: 0.95rem;
    color: #475569;
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# DATA LOADING
# =========================================================
@st.cache_data(show_spinner=False)
def load_interaction_file(file_bytes, name):
    ext = os.path.splitext(name)[1].lower()
    bio = io.BytesIO(file_bytes)

    if ext == ".csv":
        df = pd.read_csv(bio)
    elif ext in [".xlsx", ".xls"]:
        df = pd.read_excel(bio)
    else:
        raise ValueError("Unsupported format. Please upload CSV or Excel file.")

    df = df.loc[:, ~df.columns.astype(str).str.contains("^Unnamed")]
    if "" in df.columns:
        df = df.drop(columns=[""])

    return df


def clean_chr(x):
    if pd.isna(x):
        return np.nan
    x = str(x).strip().lower().replace("chromosome", "").replace("chr", "").strip()
    return "chr" + x if x else np.nan


def clean_strand(x):
    if pd.isna(x):
        return "unknown"
    x = str(x).strip().lower()
    mapping = {
        "1": "+",
        "+1": "+",
        "+": "+",
        "plus": "+",
        "-1": "-",
        "-": "-",
        "minus": "-"
    }
    return mapping.get(x, x)


def get_condition(row):
    labels = []
    if row.get("Normal", 0) == 1:
        labels.append("Normal")
    if row.get("CarboplatinTreated", 0) == 1:
        labels.append("Carboplatin")
    if row.get("GemcitabineTreated", 0) == 1:
        labels.append("Gemcitabine")

    if len(labels) == 0:
        return "Unlabeled"
    if len(labels) == 1:
        return labels[0]
    return "+".join(labels)


def distance_class(x):
    if pd.isna(x):
        return "trans_or_unknown"
    if x <= 100000:
        return "short_range"
    if x <= 1000000:
        return "medium_range"
    return "long_range"


def shape_class(row):
    d = row.get("genomic_distance_final", np.nan)
    w = row.get("interactor_width", np.nan)
    s = row.get("anchor_span", np.nan)

    if pd.isna(d):
        return "trans_shape"
    if d <= 100000 and pd.notna(w) and w <= 2000:
        return "compact_loop"
    if d <= 1000000 and 
