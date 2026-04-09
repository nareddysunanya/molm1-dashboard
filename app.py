# =========================================================
# IMPORT STATEMENTS
# =========================================================
import os
import io
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# =========================================================
# ALGORITHM KA CODE
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
    if d <= 1000000 and pd.notna(s) and s <= 1000000:
        return "local_arc"
    if d > 1000000 and pd.notna(s) and s > 1000000:
        return "extended_loop"
    return "broad_contact"


def dna_structure_class(row):
    d = row.get("genomic_distance_final", np.nan)
    ext_ratio = row.get("shape_extension_ratio", np.nan)
    compactness = row.get("shape_compactness", np.nan)

    if pd.isna(d):
        return "unknown_structure"
    if d <= 100000 and pd.notna(compactness) and compactness < 0.08:
        return "tight_fold"
    if d <= 1000000 and pd.notna(ext_ratio) and ext_ratio <= 30:
        return "arched_domain"
    if d > 1000000:
        return "open_domain"
    return "mixed_structure"


def interaction_summary_text(sub_df):
    if len(sub_df) == 0:
        return "No data available for this section."

    mean_distance = sub_df["genomic_distance_final"].mean() if "genomic_distance_final" in sub_df.columns else np.nan
    mean_strength = sub_df["interaction_strength_proxy"].mean() if "interaction_strength_proxy" in sub_df.columns else np.nan
    cis_share = (sub_df["interaction_type"].eq("cis").mean() * 100) if "interaction_type" in sub_df.columns else np.nan
    short_share = (sub_df["range_group"].eq("Short-Range").mean() * 100) if "range_group" in sub_df.columns else np.nan

    parts = []
    parts.append(f"Rows: {len(sub_df):,}")
    if pd.notna(mean_distance):
        parts.append(f"Mean distance: {mean_distance:,.0f} bp")
    if pd.notna(mean_strength):
        parts.append(f"Mean strength: {mean_strength:.2f}")
    if pd.notna(cis_share):
        parts.append(f"Cis share: {cis_share:.1f}%")
    if pd.notna(short_share):
        parts.append(f"Short-range share: {short_share:.1f}%")

    return " | ".join(parts)


def safe_sample(df, n):
    if len(df) <= n:
        return df.copy()
    return df.sample(n, random_state=42).copy()


@st.cache_data(show_spinner=False)
def process_data(df):
    df = df.copy()

    numeric_cols = [
        "Feature_Start", "Interactor_Start", "Interactor_End", "abs_distance", "NofInts",
        "MG1_SuppPairs", "MG2_SuppPairs", "MC1_SuppPairs", "MC2_SuppPairs",
        "MN1_SuppPairs", "MN2_SuppPairs", "Normal", "CarboplatinTreated", "GemcitabineTreated"
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "Feature_Chr" in df.columns:
        df["Feature_Chr"] = df["Feature_Chr"].apply(clean_chr)
    if "Interactor_Chr" in df.columns:
        df["Interactor_Chr"] = df["Interactor_Chr"].apply(clean_chr)

    if "Strand" in df.columns:
        df["Strand"] = df["Strand"].apply(clean_strand)
    else:
        df["Strand"] = "unknown"

    if {"Interactor_Start", "Interactor_End"}.issubset(df.columns):
        df["Interactor_Mid"] = ((df["Interactor_Start"] + df["Interactor_End"]) / 2).round()
        df["interactor_width"] = (df["Interactor_End"] - df["Interactor_Start"]).abs()
    else:
        df["Interactor_Mid"] = np.nan
        df["interactor_width"] = np.nan

    if {"Feature_Chr", "Interactor_Chr"}.issubset(df.columns):
        df["interaction_type"] = np.where(df["Feature_Chr"] == df["Interactor_Chr"], "cis", "trans")
    else:
        df["interaction_type"] = "unknown"

    if {"Feature_Start", "Interactor_Mid"}.issubset(df.columns):
        df["computed_distance"] = np.where(
            df["interaction_type"] == "cis",
            (df["Feature_Start"] - df["Interactor_Mid"]).abs(),
            np.nan
        )
    else:
        df["computed_distance"] = np.nan

    if "abs_distance" in df.columns:
        df["genomic_distance_final"] = pd.to_numeric(df["abs_distance"], errors="coerce").fillna(df["computed_distance"])
    else:
        df["genomic_distance_final"] = df["computed_distance"]

    df.loc[df["interaction_type"] == "trans", "genomic_distance_final"] = np.nan
    df["Condition"] = df.apply(get_condition, axis=1)
    df["distance_class"] = df["genomic_distance_final"].apply(distance_class)

    if {"Feature_Start", "Interactor_Mid"}.issubset(df.columns):
        df["anchor_span"] = (df["Feature_Start"] - df["Interactor_Mid"]).abs()
    else:
        df["anchor_span"] = np.nan

    bins = [-1, 100000, 500000, 1000000, 5000000, 10000000, np.inf]
    labels = ["0-100kb", "100kb-500kb", "500kb-1Mb", "1Mb-5Mb", "5Mb-10Mb", ">10Mb"]
    df["distance_bin"] = pd.cut(df["genomic_distance_final"], bins=bins, labels=labels)

    for c in ["MG1_SuppPairs", "MG2_SuppPairs", "MC1_SuppPairs", "MC2_SuppPairs", "MN1_SuppPairs", "MN2_SuppPairs"]:
        if c not in df.columns:
            df[c] = 0

    df["interaction_strength_proxy"] = np.select(
        [
            df["Condition"].eq("Normal"),
            df["Condition"].eq("Carboplatin"),
            df["Condition"].eq("Gemcitabine")
        ],
        [
            df["MN1_SuppPairs"] + df["MN2_SuppPairs"],
            df["MC1_SuppPairs"] + df["MC2_SuppPairs"],
            df["MG1_SuppPairs"] + df["MG2_SuppPairs"]
        ],
        default=df["NofInts"] if "NofInts" in df.columns else np.nan
    )

    if "NofInts" in df.columns:
        df["interaction_strength_proxy"] = pd.Series(df["interaction_strength_proxy"]).fillna(df["NofInts"])

    df["strand_group"] = df["Strand"].fillna("unknown")
    df["shape_bucket"] = df.apply(shape_class, axis=1)
    df["shape_compactness"] = df["interactor_width"] / df["anchor_span"].replace(0, np.nan)
    df["shape_extension_ratio"] = df["anchor_span"] / df["interactor_width"].replace(0, np.nan)
    df["shape_contact_decay"] = df["interaction_strength_proxy"] / df["genomic_distance_final"].replace(0, np.nan)
    df["strand_distance_score"] = df["genomic_distance_final"] / df["interaction_strength_proxy"].replace(0, np.nan)
    df["log_distance"] = np.log10(df["genomic_distance_final"].replace(0, np.nan))
    df["distance_mb"] = df["genomic_distance_final"] / 1_000_000
    df["range_group"] = np.where(df["genomic_distance_final"] < 1_000_000, "Short-Range", "Long-Range")
    df["dna_structure_class"] = df.apply(dna_structure_class, axis=1)

    q1 = df["interaction_strength_proxy"].quantile(0.33)
    q2 = df["interaction_strength_proxy"].quantile(0.66)
    df["strength_level"] = pd.cut(
        df["interaction_strength_proxy"],
        bins=[-np.inf, q1, q2, np.inf],
        labels=["Weak", "Moderate", "Strong"]
    )

    return df


# =========================================================
# MAIN CODE
# =========================================================
st.set_page_config(
    page_title="MOLM-1 Neon Genome Dashboard",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at 10% 20%, rgba(99,102,241,0.22), transparent 28%),
        radial-gradient(circle at 85% 15%, rgba(236,72,153,0.18), transparent 24%),
        radial-gradient(circle at 80% 80%, rgba(34,211,238,0.14), transparent 26%),
        linear-gradient(135deg, #0f172a 0%, #111827 25%, #131c3a 55%, #1e1b4b 100%);
    color: #e2e8f0;
}

.block-container {
    max-width: 1500px;
    padding-top: 1rem;
    padding-bottom: 2rem;
}

.main-card {
    background: rgba(255, 255, 255, 0.08);
    backdrop-filter: blur(18px);
    -webkit-backdrop-filter: blur(18px);
    border: 1px solid rgba(255,255,255,0.12);
    box-shadow: 0 8px 32px rgba(0,0,0,0.25);
    border-radius: 24px;
    padding: 1.2rem 1.2rem;
    margin-bottom: 1rem;
}

.hero {
    background:
        linear-gradient(135deg, rgba(124,58,237,0.45), rgba(59,130,246,0.28), rgba(236,72,153,0.30)),
        rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.14);
    box-shadow: 0 0 35px rgba(139,92,246,0.18);
    border-radius: 28px;
    padding: 1.6rem 1.6rem;
    margin-bottom: 1rem;
}

.hero-badge {
    display: inline-block;
    padding: 0.32rem 0.75rem;
    border-radius: 999px;
    background: linear-gradient(90deg, #22d3ee, #8b5cf6, #f472b6);
    color: white;
    font-size: 0.78rem;
    font-weight: 800;
    margin-bottom: 0.8rem;
    letter-spacing: 0.02em;
}

.hero-title {
    font-size: 2.6rem;
    font-weight: 900;
    line-height: 1.05;
    color: #ffffff;
    margin-bottom: 0.35rem;
}

.hero-sub {
    font-size: 1.02rem;
    color: #dbeafe;
    max-width: 960px;
}

.section-title {
    font-size: 1.25rem;
    font-weight: 900;
    color: #ffffff;
    margin-top: 0.35rem;
    margin-bottom: 0.8rem;
    letter-spacing: 0.01em;
}

.glow-note {
    background: linear-gradient(135deg, rgba(59,130,246,0.22), rgba(236,72,153,0.18));
    border: 1px solid rgba(255,255,255,0.10);
    border-left: 5px solid #22d3ee;
    border-radius: 18px;
    padding: 1rem 1rem;
    color: #e2e8f0;
    margin-bottom: 1rem;
    box-shadow: 0 0 22px rgba(34,211,238,0.10);
}

.summary-glow {
    background: linear-gradient(135deg, rgba(255,255,255,0.10), rgba(255,255,255,0.06));
    border: 1px solid rgba(255,255,255,0.09);
    border-radius: 16px;
    padding: 0.9rem 1rem;
    color: #dbeafe;
    margin-top: 0.45rem;
    margin-bottom: 1rem;
    box-shadow: 0 0 18px rgba(168,85,247,0.08);
}

.metric-card {
    background: linear-gradient(135deg, rgba(99,102,241,0.22), rgba(236,72,153,0.14));
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: 22px;
    padding: 1rem;
    box-shadow: 0 0 24px rgba(99,102,241,0.15);
    text-align: center;
    margin-bottom: 0.8rem;
}

.metric-emoji {
    font-size: 1.5rem;
    margin-bottom: 0.3rem;
}

.metric-label {
    font-size: 0.92rem;
    color: #cbd5e1;
    margin-bottom: 0.15rem;
}

.metric-value {
    font-size: 1.55rem;
    font-weight: 900;
    color: #ffffff;
}

.algorithm-step {
    background: linear-gradient(135deg, rgba(255,255,255,0.10), rgba(255,255,255,0.05));
    border: 1px solid rgba(255,255,255,0.09);
    border-left: 4px solid #f472b6;
    border-radius: 16px;
    padding: 0.9rem 1rem;
    margin-bottom: 0.7rem;
    color: #e2e8f0;
}

.algorithm-step b {
    color: #ffffff;
}

.chart-wrap {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 22px;
    padding: 0.9rem;
    box-shadow: 0 6px 22px rgba(0,0,0,0.18);
    margin-bottom: 1rem;
}

.mini-tag {
    display: inline-block;
    padding: 0.28rem 0.7rem;
    border-radius: 999px;
    background: rgba(255,255,255,0.08);
    color: #c4b5fd;
    border: 1px solid rgba(255,255,255,0.08);
    font-size: 0.78rem;
    font-weight: 700;
    margin-bottom: 0.65rem;
}

.dataframe-shell {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 22px;
    padding: 0.9rem;
    margin-top: 0.6rem;
}

[data-testid="stSidebar"] {
    background:
        linear-gradient(180deg, rgba(15,23,42,0.96) 0%, rgba(30,41,59,0.96) 100%);
    border-right: 1px solid rgba(255,255,255,0.06);
}

[data-testid="stSidebar"] * {
    color: #e5e7eb !important;
}

[data-testid="stMetric"] {
    background: transparent;
    border: none;
    box-shadow: none;
}

h1, h2, h3, h4, h5, h6, p, label, div {
    color: inherit;
}

.stSelectbox label, .stMultiSelect label, .stSlider label, .stFileUploader label {
    color: #e5e7eb !important;
    font-weight: 700 !important;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
}

.stTabs [data-baseweb="tab"] {
    background: rgba(255,255,255,0.06);
    border-radius: 14px;
    color: #e2e8f0;
    padding: 10px 16px;
    border: 1px solid rgba(255,255,255,0.08);
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(90deg, rgba(99,102,241,0.35), rgba(236,72,153,0.28));
    color: white !important;
}

div[data-testid="stDataFrame"] {
    border-radius: 16px;
    overflow: hidden;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
    <div class="hero-badge">NEXT-GENOME UI · VIBRANT MODE</div>
    <div class="hero-title">🧬 MOLM-1 Chromatin Intelligence Dashboard</div>
    <div class="hero-sub">
        A distance-driven, visually rich genome interaction dashboard for MOLM-1 cells that compares
        Normal, Carboplatin, and Gemcitabine conditions with strand analysis, shape analysis, DNA structure proxy views,
        and chart-wise summaries in a beginner-friendly flow.
    </div>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("Dashboard Controls")
    uploaded = st.file_uploader("Upload CSV / XLSX / XLS", type=["csv", "xlsx", "xls"])
    max_points = st.slider("Scatter detail", 500, 12000, 3500, 500)
    show_raw = st.toggle("Show raw preview", False)

if uploaded is None:
    st.markdown("""
    <div class="main-card">
        <div class="section-title">Start Here</div>
        <div class="glow-note">
        Upload your project dataset to activate the full dashboard. This version is intentionally designed
        to look more premium, colorful, and user-friendly while still following the abstract:
        genomic distance computation, distance-dependent interaction strength analysis, treatment comparison,
        short-range vs long-range interactions, strand analysis, shape visualizations, and DNA structure analysis.
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

try:
    raw_df = load_interaction_file(uploaded.getvalue(), uploaded.name)
    df = process_data(raw_df)
except Exception as e:
    st.error(f"Loading failed: {e}")
    st.stop()

if show_raw:
    st.markdown('<div class="main-card"><div class="section-title">Raw Data Preview</div></div>', unsafe_allow_html=True)
    st.dataframe(raw_df.head(50), use_container_width=True)

conditions = sorted(df["Condition"].dropna().unique().tolist())
selected_conditions = st.multiselect("Choose conditions to display", conditions, default=conditions)
fdf = df[df["Condition"].isin(selected_conditions)].copy()

if len(fdf) == 0:
    st.warning("No data available after filtering.")
    st.stop()

# METRIC CARDS
mean_distance = fdf["genomic_distance_final"].mean()
mean_strength = fdf["interaction_strength_proxy"].mean()
cis_count = (fdf["interaction_type"] == "cis").sum()

mc1, mc2, mc3, mc4 = st.columns(4)

with mc1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-emoji">🧩</div>
        <div class="metric-label">Total Interactions</div>
        <div class="metric-value">{len(fdf):,}</div>
    </div>
    """, unsafe_allow_html=True)

with mc2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-emoji">🧬</div>
        <div class="metric-label">Cis Interactions</div>
        <div class="metric-value">{cis_count:,}</div>
    </div>
    """, unsafe_allow_html=True)

with mc3:
    md_text = f"{mean_distance:,.0f} bp" if pd.notna(mean_distance) else "NA"
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-emoji">📏</div>
        <div class="metric-label">Mean Distance</div>
        <div class="metric-value">{md_text}</div>
    </div>
    """, unsafe_allow_html=True)

with mc4:
    ms_text = f"{mean_strength:.2f}" if pd.notna(mean_strength) else "NA"
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-emoji">⚡</div>
        <div class="metric-label">Mean Strength</div>
        <div class="metric-value">{ms_text}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<div class="main-card">
    <div class="section-title">Why this dashboard matters</div>
    <div class="glow-note">
    This tool turns raw chromatin interaction rows into an easy visual story. Every major analysis here is rooted in
    genomic distance, because your abstract specifically focuses on the relationship between distance and interaction strength
    in MOLM-1 cells under Normal, Carboplatin, and Gemcitabine treatment conditions.
    </div>
</div>
""", unsafe_allow_html=True)

# ALGORITHM
st.markdown('<div class="main-card"><div class="section-title">Algorithm Flow</div>', unsafe_allow_html=True)
algo_steps = [
    "Load the uploaded chromatin interaction dataset.",
    "Clean chromosome names, strand values, and numeric fields.",
    "Detect cis and trans interactions using chromosome matching.",
    "Compute genomic distance for cis interactions from feature and interactor coordinates.",
    "Estimate interaction strength from treatment-linked support pair columns.",
    "Build distance bins and classify short-range, medium-range, and long-range interactions.",
    "Generate strand analysis by drug and distance.",
    "Generate shape and DNA structure proxy features from anchor span, interactor width, compactness, and decay.",
    "Display all results in a one-page, beginner-friendly, visually rich dashboard."
]
for i, step in enumerate(algo_steps, start=1):
    st.markdown(
        f'<div class="algorithm-step"><b>Step {i}:</b> {step}</div>',
        unsafe_allow_html=True
    )
st.markdown('</div>', unsafe_allow_html=True)

# SECTION 1
st.markdown('<div class="main-card"><div class="section-title">Distance-Dependent Interaction Analysis</div><div class="mini-tag">Abstract core analysis</div>', unsafe_allow_html=True)

scatter_df = safe_sample(
    fdf.dropna(subset=["log_distance", "interaction_strength_proxy"]),
    max_points
)

fig1 = px.scatter(
    scatter_df,
    x="log_distance",
    y="interaction_strength_proxy",
    color="Condition",
    title="Log Genomic Distance vs Interaction Strength",
    opacity=0.78,
    color_discrete_sequence=["#38bdf8", "#a855f7", "#fb7185"],
    template="plotly_dark"
)
fig1.update_layout(
    height=470,
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(255,255,255,0.02)",
    font=dict(color="white"),
    legend_title_text="Condition"
)
st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
st.plotly_chart(fig1, use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)
st.markdown(
    f'<div class="summary-glow"><b>Summary:</b> {interaction_summary_text(scatter_df)}. This view directly shows how interaction strength changes with genomic distance under each treatment condition.</div>',
    unsafe_allow_html=True
)

c1, c2 = st.columns(2)

with c1:
    fig2 = px.histogram(
        fdf.dropna(subset=["genomic_distance_final"]),
        x="genomic_distance_final",
        color="Condition",
        nbins=45,
        title="Distance Distribution Across Conditions",
        color_discrete_sequence=["#60a5fa", "#c084fc", "#fb7185"],
        template="plotly_dark"
    )
    fig2.update_layout(
        height=420,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.02)",
        font=dict(color="white")
    )
    st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
    st.plotly_chart(fig2, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="summary-glow"><b>Summary:</b> This distribution highlights whether certain treatment conditions are associated with more local contacts or more distant chromatin interactions.</div>',
        unsafe_allow_html=True
    )

with c2:
    fig3 = px.box(
        fdf.dropna(subset=["interaction_strength_proxy"]),
        x="Condition",
        y="interaction_strength_proxy",
        color="Condition",
        title="Strength Distribution by Treatment",
        color_discrete_sequence=["#22d3ee", "#8b5cf6", "#f472b6"],
        template="plotly_dark"
    )
    fig3.update_layout(
        height=420,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.02)",
        font=dict(color="white"),
        showlegend=False
    )
    st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
    st.plotly_chart(fig3, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="summary-glow"><b>Summary:</b> This chart compares the spread of interaction strengths under Normal, Carboplatin, and Gemcitabine conditions.</div>',
        unsafe_allow_html=True
    )

heat_df = fdf.dropna(subset=["distance_bin", "interaction_strength_proxy"]).groupby(
    ["Condition", "distance_bin"], dropna=False
)["interaction_strength_proxy"].mean().reset_index()

if len(heat_df) > 0:
    heat_pivot = heat_df.pivot(index="distance_bin", columns="Condition", values="interaction_strength_proxy")
    fig_heat = px.imshow(
        heat_pivot,
        text_auto=True,
        color_continuous_scale="Plasma",
        aspect="auto",
        title="Mean Strength Across Distance Bins",
        template="plotly_dark"
    )
    fig_heat.update_layout(
        height=430,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.02)",
        font=dict(color="white")
    )
    st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
    st.plotly_chart(fig_heat, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="summary-glow"><b>Summary:</b> The heatmap makes distance-dependent strength differences easier to understand by grouping interactions into clear genomic distance ranges.</div>',
        unsafe_allow_html=True
    )

st.markdown('</div>', unsafe_allow_html=True)

# SECTION 2
st.markdown('<div class="main-card"><div class="section-title">Short-Range vs Long-Range Patterns</div><div class="mini-tag">Distance grouping by treatment</div>', unsafe_allow_html=True)

range_stats = fdf.groupby(["Condition", "range_group"], dropna=False).agg(
    interaction_count=("Condition", "size"),
    mean_strength=("interaction_strength_proxy", "mean")
).reset_index()

c3, c4 = st.columns(2)

with c3:
    fig4 = px.bar(
        range_stats,
        x="Condition",
        y="interaction_count",
        color="range_group",
        barmode="group",
        title="Short-Range vs Long-Range Counts",
        color_discrete_sequence=["#38bdf8", "#fb7185"],
        template="plotly_dark"
    )
    fig4.update_layout(
        height=420,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.02)",
        font=dict(color="white")
    )
    st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
    st.plotly_chart(fig4, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="summary-glow"><b>Summary:</b> This chart reveals whether treatment shifts the interaction population toward shorter or longer genomic distances.</div>',
        unsafe_allow_html=True
    )

with c4:
    fig5 = px.bar(
        range_stats,
        x="Condition",
        y="mean_strength",
        color="range_group",
        barmode="group",
        title="Strength in Short vs Long Range",
        color_discrete_sequence=["#22d3ee", "#f59e0b"],
        template="plotly_dark"
    )
    fig5.update_layout(
        height=420,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.02)",
        font=dict(color="white")
    )
    st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
    st.plotly_chart(fig5, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="summary-glow"><b>Summary:</b> This comparison helps show whether drug treatment changes not only the number of interactions, but also how strong short and long contacts are.</div>',
        unsafe_allow_html=True
    )

st.markdown('</div>', unsafe_allow_html=True)

# SECTION 3
st.markdown('<div class="main-card"><div class="section-title">Strand Analysis Based on Drug and Distance</div><div class="mini-tag">Requested strand analysis</div>', unsafe_allow_html=True)

strand_view = fdf.groupby(["Condition", "strand_group"], dropna=False).agg(
    interaction_count=("Condition", "size"),
    mean_distance=("genomic_distance_final", "mean"),
    mean_strength=("interaction_strength_proxy", "mean")
).reset_index()

c5, c6 = st.columns(2)

with c5:
    fig6 = px.bar(
        strand_view,
        x="Condition",
        y="interaction_count",
        color="strand_group",
        barmode="group",
        title="Strand Distribution by Drug",
        color_discrete_sequence=px.colors.qualitative.Prism,
        template="plotly_dark"
    )
    fig6.update_layout(
        height=420,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.02)",
        font=dict(color="white")
    )
    st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
    st.plotly_chart(fig6, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="summary-glow"><b>Summary:</b> This shows how strand groups are distributed across Normal, Carboplatin, and Gemcitabine conditions.</div>',
        unsafe_allow_html=True
    )

with c6:
    strand_scatter = safe_sample(
        fdf.dropna(subset=["genomic_distance_final", "interaction_strength_proxy"]),
        max_points
    )
    fig7 = px.scatter(
        strand_scatter,
        x="genomic_distance_final",
        y="interaction_strength_proxy",
        color="strand_group",
        symbol="Condition",
        title="Distance vs Strength by Strand and Drug",
        opacity=0.74,
        color_discrete_sequence=px.colors.qualitative.Bold,
        template="plotly_dark"
    )
    fig7.update_layout(
        height=420,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.02)",
        font=dict(color="white")
    )
    st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
    st.plotly_chart(fig7, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="summary-glow"><b>Summary:</b> {interaction_summary_text(strand_scatter)}. This chart combines drug, strand, distance, and interaction strength in a single visual.</div>',
        unsafe_allow_html=True
    )

st.markdown('</div>', unsafe_allow_html=True)

# SECTION 4
st.markdown('<div class="main-card"><div class="section-title">DNA Shape Visualizations</div><div class="mini-tag">Shape features from distance and geometry</div>', unsafe_allow_html=True)

shape_view = fdf.groupby(["Condition", "shape_bucket"], dropna=False).agg(
    interaction_count=("Condition", "size"),
    mean_extension_ratio=("shape_extension_ratio", "mean"),
    mean_decay=("shape_contact_decay", "mean")
).reset_index()

c7, c8 = st.columns(2)

with c7:
    fig8 = px.bar(
        shape_view,
        x="Condition",
        y="interaction_count",
        color="shape_bucket",
        barmode="group",
        title="Shape Bucket by Condition",
        color_discrete_sequence=px.colors.qualitative.G10,
        template="plotly_dark"
    )
    fig8.update_layout(
        height=420,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.02)",
        font=dict(color="white")
    )
    st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
    st.plotly_chart(fig8, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="summary-glow"><b>Summary:</b> Shape buckets classify contacts into compact loops, local arcs, extended loops, and broad contacts using genomic distance and width-based rules.</div>',
        unsafe_allow_html=True
    )

with c8:
    shape_scatter = safe_sample(
        fdf.dropna(subset=["anchor_span", "interaction_strength_proxy"]),
        max_points
    )
    fig9 = px.scatter(
        shape_scatter,
        x="anchor_span",
        y="interaction_strength_proxy",
        color="Condition",
        size="interactor_width",
        title="Anchor Span vs Interaction Strength",
        opacity=0.76,
        color_discrete_sequence=["#22d3ee", "#8b5cf6", "#fb7185"],
        template="plotly_dark"
    )
    fig9.update_layout(
        height=420,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.02)",
        font=dict(color="white")
    )
    st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
    st.plotly_chart(fig9, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="summary-glow"><b>Summary:</b> {interaction_summary_text(shape_scatter)}. Bubble size reflects interactor width, helping visualize geometry-related contact patterns.</div>',
        unsafe_allow_html=True
    )

st.markdown('</div>', unsafe_allow_html=True)

# SECTION 5
st.markdown('<div class="main-card"><div class="section-title">DNA Structure Proxy Analysis</div><div class="mini-tag">Tight folds, arches, and open domains</div>', unsafe_allow_html=True)

dna_view = fdf.groupby(["Condition", "dna_structure_class"], dropna=False).agg(
    interaction_count=("Condition", "size"),
    mean_distance=("genomic_distance_final", "mean"),
    mean_strength=("interaction_strength_proxy", "mean")
).reset_index()

c9, c10 = st.columns(2)

with c9:
    fig10 = px.bar(
        dna_view,
        x="Condition",
        y="interaction_count",
        color="dna_structure_class",
        barmode="group",
        title="DNA Structure Class by Condition",
        color_discrete_sequence=px.colors.qualitative.Safe,
        template="plotly_dark"
    )
    fig10.update_layout(
        height=420,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.02)",
        font=dict(color="white")
    )
    st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
    st.plotly_chart(fig10, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="summary-glow"><b>Summary:</b> DNA structure proxy classes translate geometric and distance-derived interaction features into easier structural categories.</div>',
        unsafe_allow_html=True
    )

with c10:
    dna_scatter = safe_sample(
        fdf.dropna(subset=["shape_extension_ratio", "shape_contact_decay"]),
        max_points
    )
    fig11 = px.scatter(
        dna_scatter,
        x="shape_extension_ratio",
        y="shape_contact_decay",
        color="Condition",
        title="DNA Structure Proxy Space",
        opacity=0.76,
        color_discrete_sequence=["#38bdf8", "#a855f7", "#fb7185"],
        template="plotly_dark"
    )
    fig11.update_layout(
        height=420,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.02)",
        font=dict(color="white")
    )
    st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
    st.plotly_chart(fig11, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="summary-glow"><b>Summary:</b> This structural map shows how interaction organization may shift under treatment when extension ratio and contact decay are viewed together.</div>',
        unsafe_allow_html=True
    )

st.markdown('</div>', unsafe_allow_html=True)

# TABLE
st.markdown('<div class="main-card"><div class="section-title">Processed Interaction Table</div><div class="mini-tag">Clean output for review and download</div>', unsafe_allow_html=True)

useful_cols = [
    "Feature_Chr", "Feature_Start", "Interactor_Chr", "Interactor_Start", "Interactor_End",
    "interaction_type", "Condition", "Strand", "strand_group", "genomic_distance_final",
    "distance_mb", "distance_class", "distance_bin", "range_group",
    "interaction_strength_proxy", "strength_level", "anchor_span", "interactor_width",
    "shape_bucket", "dna_structure_class", "shape_compactness", "shape_extension_ratio",
    "shape_contact_decay"
]
useful_cols = [c for c in useful_cols if c in fdf.columns]

st.markdown('<div class="dataframe-shell">', unsafe_allow_html=True)
st.dataframe(fdf[useful_cols].head(300), use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

csv = fdf[useful_cols].to_csv(index=False).encode("utf-8")
st.download_button(
    "Download processed data as CSV",
    data=csv,
    file_name="molm1_processed_dashboard_data.csv",
    mime="text/csv"
)

st.markdown('</div>', unsafe_allow_html=True)

st.markdown("""
<div style="text-align:center; color:#cbd5e1; font-size:0.9rem; padding-top:0.6rem; padding-bottom:0.2rem;">
MOLM-1 Neon Dashboard · Distance-based chromatin interaction analysis · Visual, interactive, beginner-friendly
</div>
""", unsafe_allow_html=True)
