# =========================================================
# IMPORT STATEMENTS
# =========================================================
import os
import io
import zipfile
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# =========================================================
# ALGORITHM CODE
# =========================================================

@st.cache_data(show_spinner=False)
def load_interaction_file(file_bytes, name):
    ext = os.path.splitext(name)[1].lower()
    bio = io.BytesIO(file_bytes)
    
    if ext == ".csv":
        df = pd.read_csv(bio)
    elif ext in [".xlsx", ".xls"]:
        df = pd.read_excel(bio)
    elif ext == ".zip":
        with zipfile.ZipFile(bio) as z:
            # Find the first valid data file in the zip
            valid_files = [f for f in z.namelist() if f.endswith(('.csv', '.xlsx', '.xls'))]
            if not valid_files:
                raise ValueError("No supported files (CSV/XLSX) found inside the ZIP.")
            with z.open(valid_files[0]) as f:
                if valid_files[0].endswith('.csv'):
                    df = pd.read_csv(f)
                else:
                    df = pd.read_excel(f)
    else:
        raise ValueError("Unsupported format. Please upload CSV, XLSX, or ZIP.")

    df = df.loc[:, ~df.columns.astype(str).str.contains("^Unnamed")]
    if "" in df.columns:
        df = df.drop(columns=[""])
    return df

def clean_chr(x):
    if pd.isna(x): return np.nan
    x = str(x).strip().lower()
    x = x.replace("chromosome", "").replace("chr", "").strip()
    return "chr" + x if x else np.nan

def clean_strand(x):
    if pd.isna(x): return "unknown"
    x = str(x).strip().lower()
    mapping = {"1": "+", "+1": "+", "+": "+", "plus": "+", "-1": "-", "-": "-", "minus": "-"}
    return mapping.get(x, x)

def get_condition(row):
    labels = []
    if row.get("Normal", 0) == 1: labels.append("Normal")
    if row.get("CarboplatinTreated", 0) == 1: labels.append("Carboplatin")
    if row.get("GemcitabineTreated", 0) == 1: labels.append("Gemcitabine")
    return "+".join(labels) if labels else "Unlabeled"

def distance_class(x):
    if pd.isna(x): return "trans_or_unknown"
    elif x <= 100000: return "short_range"
    elif x <= 1000000: return "medium_range"
    else: return "long_range"

def shape_class(row):
    d, w, s = row.get("genomic_distance_final"), row.get("interactor_width"), row.get("anchor_span")
    if pd.isna(d): return "trans_shape"
    elif d <= 100000 and pd.notna(w) and w <= 2000: return "compact_loop"
    elif d <= 1000000 and pd.notna(s) and s <= 1000000: return "local_arc"
    elif d > 1000000 and pd.notna(s) and s > 1000000: return "extended_loop"
    else: return "broad_contact"

def dna_structure_class(row):
    d, ext, comp = row.get("genomic_distance_final"), row.get("shape_extension_ratio"), row.get("shape_compactness")
    if pd.isna(d): return "unknown_structure"
    elif d <= 100000 and pd.notna(comp) and comp < 0.08: return "tight_fold"
    elif d <= 1000000 and pd.notna(ext) and ext <= 30: return "arched_domain"
    elif d > 1000000: return "open_domain"
    else: return "mixed_structure"

def interaction_summary_text(sub_df):
    if len(sub_df) == 0: return "No data available."
    parts = [f"Rows: {len(sub_df):,}"]
    if "genomic_distance_final" in sub_df.columns:
        parts.append(f"Mean distance: {sub_df['genomic_distance_final'].mean():,.0f} bp")
    return " | ".join(parts)

def safe_sample(df, n):
    return df.sample(n, random_state=42).copy() if len(df) > n else df.copy()

@st.cache_data(show_spinner=False)
def process_data(df):
    df = df.copy()
    num_cols = ["Feature_Start", "Interactor_Start", "Interactor_End", "abs_distance", "NofInts", 
                "MG1_SuppPairs", "MG2_SuppPairs", "MC1_SuppPairs", "MC2_SuppPairs", "MN1_SuppPairs", "MN2_SuppPairs",
                "Normal", "CarboplatinTreated", "GemcitabineTreated"]
    for col in num_cols:
        if col in df.columns: df[col] = pd.to_numeric(df[col], errors="coerce")
    
    if "Feature_Chr" in df.columns: df["Feature_Chr"] = df["Feature_Chr"].apply(clean_chr)
    if "Interactor_Chr" in df.columns: df["Interactor_Chr"] = df["Interactor_Chr"].apply(clean_chr)
    df["Strand"] = df["Strand"].apply(clean_strand) if "Strand" in df.columns else "unknown"
    
    if {"Interactor_Start", "Interactor_End"}.issubset(df.columns):
        df["Interactor_Mid"] = ((df["Interactor_Start"] + df["Interactor_End"]) / 2).round()
        df["interactor_width"] = (df["Interactor_End"] - df["Interactor_Start"]).abs()
    
    df["interaction_type"] = np.where(df["Feature_Chr"] == df["Interactor_Chr"], "cis", "trans")
    df["computed_distance"] = np.where(df["interaction_type"] == "cis", (df["Feature_Start"] - df["Interactor_Mid"]).abs(), np.nan)
    df["genomic_distance_final"] = df["abs_distance"].fillna(df["computed_distance"]) if "abs_distance" in df.columns else df["computed_distance"]
    df.loc[df["interaction_type"] == "trans", "genomic_distance_final"] = np.nan
    
    df["Condition"] = df.apply(get_condition, axis=1)
    df["anchor_span"] = (df["Feature_Start"] - df["Interactor_Mid"]).abs()
    
    bins = [-1, 100000, 500000, 1000000, 5000000, 10000000, np.inf]
    labels = ["0-100kb", "100kb-500kb", "500kb-1Mb", "1Mb-5Mb", "5Mb-10Mb", ">10Mb"]
    df["distance_bin"] = pd.cut(df["genomic_distance_final"], bins=bins, labels=labels)
    
    df["interaction_strength_proxy"] = np.select(
        [df["Condition"] == "Normal", df["Condition"] == "Carboplatin", df["Condition"] == "Gemcitabine"],
        [df.get("MN1_SuppPairs", 0) + df.get("MN2_SuppPairs", 0), 
         df.get("MC1_SuppPairs", 0) + df.get("MC2_SuppPairs", 0), 
         df.get("MG1_SuppPairs", 0) + df.get("MG2_SuppPairs", 0)],
        default=df["NofInts"] if "NofInts" in df.columns else 0
    )
    
    df["shape_bucket"] = df.apply(shape_class, axis=1)
    df["shape_compactness"] = df["interactor_width"] / df["anchor_span"].replace(0, np.nan)
    df["shape_extension_ratio"] = df["anchor_span"] / df["interactor_width"].replace(0, np.nan)
    df["shape_contact_decay"] = df["interaction_strength_proxy"] / df["genomic_distance_final"].replace(0, np.nan)
    df["log_distance"] = np.log10(df["genomic_distance_final"].replace(0, np.nan))
    df["range_group"] = np.where(df["genomic_distance_final"] < 1_000_000, "Short-Range", "Long-Range")
    df["dna_structure_class"] = df.apply(dna_structure_class, axis=1)
    
    return df

# =========================================================
# MAIN CODE
# =========================================================

st.set_page_config(page_title="MOLM-1 Intelligence Dashboard", page_icon="🧬", layout="wide")

if "theme_mode" not in st.session_state:
    st.session_state.theme_mode = "Dark"

with st.sidebar:
    st.header("Settings")
    # Compact Sun/Moon Toggle
    t1, t2 = st.columns([1, 4])
    with t1:
        if st.session_state.theme_mode == "Dark":
            if st.button("☀️"):
                st.session_state.theme_mode = "Light"
                st.rerun()
        else:
            if st.button("🌙"):
                st.session_state.theme_mode = "Dark"
                st.rerun()
    with t2:
        st.write(f"**{st.session_state.theme_mode} Mode**")
    
    uploaded = st.file_uploader("Upload Data", type=["csv", "xlsx", "xls", "zip"])
    max_points = st.slider("Scatter detail", 500, 12000, 3500, 500)
    show_raw = st.toggle("Show raw preview", False)

# Dynamic CSS based on Theme Mode
if st.session_state.theme_mode == "Dark":
    bg_css = """
    <style>
    header, [data-testid="stToolbar"] {display: none;}
    .stApp {background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%); color: #e2e8f0;}
    .main-card {background: rgba(255, 255, 255, 0.08); border-radius: 20px; padding: 1.5rem; margin-bottom: 1rem; border: 1px solid rgba(255,255,255,0.1);}
    .hero {background: rgba(255,255,255,0.05); border-radius: 24px; padding: 2rem; margin-bottom: 1.5rem; border: 1px solid rgba(255,255,255,0.1);}
    .hero-title {font-size: 2.5rem; font-weight: 800; color: #ffffff;}
    .hero-sub {font-size: 1.1rem; color: #cbd5e1; margin-top: 0.5rem;}
    .section-title {font-size: 1.3rem; font-weight: 700; color: #ffffff; margin-bottom: 1rem;}
    .summary-glow {background: rgba(99, 102, 241, 0.15); border-radius: 12px; padding: 1rem; margin-top: 1rem;}
    </style>
    """
else:
    bg_css = """
    <style>
    header, [data-testid="stToolbar"] {display: none;}
    .stApp {background: #f8fafc; color: #1e293b;}
    .main-card {background: white; border-radius: 20px; padding: 1.5rem; margin-bottom: 1rem; border: 1px solid #e2e8f0; box-shadow: 0 4px 12px rgba(0,0,0,0.05);}
    .hero {background: white; border-radius: 24px; padding: 2rem; margin-bottom: 1.5rem; border: 1px solid #e2e8f0;}
    .hero-title {font-size: 2.5rem; font-weight: 800; color: #0f172a;}
    .hero-sub {font-size: 1.1rem; color: #475569; margin-top: 0.5rem;}
    .section-title {font-size: 1.3rem; font-weight: 700; color: #0f172a; margin-bottom: 1rem;}
    .summary-glow {background: #f1f5f9; border-radius: 12px; padding: 1rem; margin-top: 1rem;}
    </style>
    """

st.markdown(bg_css, unsafe_allow_html=True)

# Cleaned Header
st.markdown("""
    <div class="hero">
        <div class="hero-title">🧬 MOLM-1 Chromatin Intelligence Dashboard</div>
        <div class="hero-sub">
            An advanced analytical platform designed to visualize and interpret distance-dependent 
            chromatin interactions. Compare drug treatments (Carboplatin vs. Gemcitabine), 
            analyze structural variations, and explore the geometric landscape of the MOLM-1 genome.
        </div>
    </div>
""", unsafe_allow_html=True)

if uploaded is None:
    st.info("Please upload a CSV, XLSX, or ZIP file to begin the analysis.")
    st.stop()

try:
    raw_df = load_interaction_file(uploaded.getvalue(), uploaded.name)
    df = process_data(raw_df)
except Exception as e:
    st.error(f"Error processing file: {e}")
    st.stop()

if show_raw:
    st.markdown('<div class="main-card"><div class="section-title">Raw Data Preview</div></div>', unsafe_allow_html=True)
    st.dataframe(raw_df.head(50), use_container_width=True)

# Filtering
conditions = sorted(df["Condition"].unique().tolist())
selected = st.multiselect("Filter Conditions", conditions, default=conditions)
fdf = df[df["Condition"].isin(selected)].copy()

if fdf.empty:
    st.warning("No data matches selected filters.")
    st.stop()

# Layout for Charts
plot_template = "plotly_dark" if st.session_state.theme_mode == "Dark" else "plotly_white"

st.markdown('<div class="main-card"><div class="section-title">Distance-Dependent Interaction Analysis</div>', unsafe_allow_html=True)
scatter_df = safe_sample(fdf.dropna(subset=["log_distance"]), max_points)
fig1 = px.scatter(scatter_df, x="log_distance", y="interaction_strength_proxy", color="Condition", 
                 title="Genomic Distance vs Interaction Strength", template=plot_template)
st.plotly_chart(fig1, use_container_width=True)
st.markdown(f'<div class="summary-glow"><b>Insight:</b> {interaction_summary_text(fdf)}</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# Data Table
st.markdown('<div class="main-card"><div class="section-title">Processed Data View</div>', unsafe_allow_html=True)
st.dataframe(fdf.head(100), use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)
