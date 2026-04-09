import os
import io
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(
    page_title="MOLM-1 Chromatin Dashboard",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# PAGE STYLE
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
# HELPERS
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
    x = str(x).strip().lower()
    x = x.replace("chromosome", "").replace("chr", "").strip()
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
    elif x <= 1000000:
        return "medium_range"
    else:
        return "long_range"


def shape_class(row):
    d = row.get("genomic_distance_final", np.nan)
    w = row.get("interactor_width", np.nan)
    s = row.get("anchor_span", np.nan)

    if pd.isna(d):
        return "trans_shape"
    elif d <= 100000 and pd.notna(w) and w <= 2000:
        return "compact_loop"
    elif d <= 1000000 and pd.notna(s) and s <= 1000000:
        return "local_arc"
    elif d > 1000000 and pd.notna(s) and s > 1000000:
        return "extended_loop"
    else:
        return "broad_contact"


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

    for col in ["MG1_SuppPairs", "MG2_SuppPairs", "MC1_SuppPairs", "MC2_SuppPairs", "MN1_SuppPairs", "MN2_SuppPairs"]:
        if col not in df.columns:
            df[col] = 0

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

    strength_q1 = df["interaction_strength_proxy"].quantile(0.33)
    strength_q2 = df["interaction_strength_proxy"].quantile(0.66)
    df["strength_level"] = pd.cut(
        df["interaction_strength_proxy"],
        bins=[-np.inf, strength_q1, strength_q2, np.inf],
        labels=["Weak", "Moderate", "Strong"]
    )

    return df


@st.cache_data(show_spinner=False)
def summarize(df):
    condition_summary = df.groupby("Condition", dropna=False).agg(
        total_interactions=("Condition", "size"),
        cis_interactions=("interaction_type", lambda x: (x == "cis").sum()),
        mean_distance=("genomic_distance_final", "mean"),
        mean_strength=("interaction_strength_proxy", "mean")
    ).reset_index()

    distance_summary = df.groupby(["Condition", "distance_bin"], dropna=False).agg(
        interaction_count=("distance_bin", "size"),
        mean_strength=("interaction_strength_proxy", "mean")
    ).reset_index()

    strand_summary = df.groupby(["Condition", "strand_group"], dropna=False).agg(
        interaction_count=("strand_group", "size"),
        mean_distance=("genomic_distance_final", "mean")
    ).reset_index()

    shape_summary = df.groupby(["Condition", "shape_bucket"], dropna=False).agg(
        interaction_count=("shape_bucket", "size"),
        mean_extension_ratio=("shape_extension_ratio", "mean"),
        mean_decay=("shape_contact_decay", "mean")
    ).reset_index()

    return condition_summary, distance_summary, strand_summary, shape_summary


# =========================================================
# HEADER
# =========================================================
st.markdown("""
<div class="hero">
    <div class="hero-title">🧬 MOLM-1 Chromatin Dashboard</div>
    <div class="hero-sub">
        Computational Analysis of Distance-Dependent Chromatin Interactions in MOLM-1 Cells
    </div>
</div>
""", unsafe_allow_html=True)

# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.header("Upload & Controls")
    uploaded = st.file_uploader("Upload CSV / XLSX / XLS", type=["csv", "xlsx", "xls"])
    page = st.radio("Select page", ["Overview", "Detailed analysis"])
    max_points = st.slider("Max scatter points", 500, 15000, 4000, 500)
    show_raw = st.toggle("Show raw preview", False)

# =========================================================
# NO FILE
# =========================================================
if uploaded is None:
    st.markdown("""
    <div class="info-soft">
    Upload your project dataset to begin. This dashboard is designed for your MOLM-1 chromatin interaction
    dataset and will compute genomic distance, interaction strength trends, treatment comparison,
    strand analysis, and DNA shape-style proxy analysis.
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# =========================================================
# LOAD FILE
# =========================================================
try:
    file_bytes = uploaded.getvalue()
    raw_df = load_interaction_file(file_bytes, uploaded.name)
    df = process_data(raw_df)
    condition_summary, distance_summary, strand_summary, shape_summary = summarize(df)
except Exception as e:
    st.error(f"Loading failed: {e}")
    st.stop()

if show_raw:
    st.markdown('<div class="section-title">Raw Data Preview</div>', unsafe_allow_html=True)
    st.dataframe(raw_df.head(50), use_container_width=True)

conditions = sorted(df["Condition"].dropna().unique().tolist())
selected_conditions = st.multiselect("Choose conditions", conditions, default=conditions)

fdf = df[df["Condition"].isin(selected_conditions)].copy()
condition_summary = condition_summary[condition_summary["Condition"].isin(selected_conditions)]
distance_summary = distance_summary[distance_summary["Condition"].isin(selected_conditions)]
strand_summary = strand_summary[strand_summary["Condition"].isin(selected_conditions)]
shape_summary = shape_summary[shape_summary["Condition"].isin(selected_conditions)]

if len(fdf) == 0:
    st.warning("No data available after filtering.")
    st.stop()

# =========================================================
# METRICS
# =========================================================
m1, m2, m3, m4 = st.columns(4)

with m1:
    st.metric("Total interactions", f"{len(fdf):,}")

with m2:
    st.metric("Cis interactions", f"{(fdf['interaction_type'] == 'cis').sum():,}")

with m3:
    mean_distance = fdf["genomic_distance_final"].mean()
    st.metric("Mean distance", f"{mean_distance:,.0f} bp" if pd.notna(mean_distance) else "NA")

with m4:
    mean_strength = fdf["interaction_strength_proxy"].mean()
    st.metric("Mean strength", f"{mean_strength:.2f}" if pd.notna(mean_strength) else "NA")

st.markdown(
    '<div class="note"><b>Dashboard note:</b> This version is lighter, faster, colorful, beginner-friendly, and structured as a 2-page dashboard with analysis based on distance computation, treatment comparison, strand analysis, and DNA shape-style proxy features.</div>',
    unsafe_allow_html=True
)

# =========================================================
# PAGE 1
# =========================================================
if page == "Overview":
    st.markdown('<div class="section-title">Project Overview</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="card small-text">
    This dashboard follows the project abstract by studying how chromatin interaction strength changes
    with genomic distance in MOLM-1 leukemia cells under Normal, Carboplatin, and Gemcitabine conditions.
    It is built to be colorful, user-friendly, and easy for a beginner to understand.
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">What this tool shows</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="success-soft">
    1. Genomic distance computation between interacting regions.<br>
    2. Distance-dependent interaction strength analysis.<br>
    3. Drug-condition comparison across Normal, Carboplatin, and Gemcitabine.<br>
    4. Short-range versus long-range interaction changes.<br>
    5. Strand analysis based on condition and distance.<br>
    6. DNA shape-style proxy analysis using anchor and interactor geometry.
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Algorithm</div>', unsafe_allow_html=True)

    algo_steps = [
        "Load the uploaded chromatin interaction dataset.",
        "Clean chromosome labels, strand values, and numeric fields.",
        "Identify cis and trans interactions using chromosome matching.",
        "Compute genomic distance for cis interactions from genomic coordinates.",
        "Use abs_distance when available, otherwise use computed distance.",
        "Create distance bins and classify interactions as short, medium, or long range.",
        "Estimate interaction strength based on treatment-linked support-pair columns.",
        "Analyze strand distribution and DNA shape-style proxy features.",
        "Visualize all results in a clear and informative dashboard."
    ]

    for i, step in enumerate(algo_steps, start=1):
        st.markdown(
            f'<div class="algorithm-step"><b>Step {i}:</b> {step}</div>',
            unsafe_allow_html=True
        )

    st.markdown('<div class="section-title">Quick Visual Overview</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        fig_cond = px.bar(
            condition_summary,
            x="Condition",
            y="total_interactions",
            color="Condition",
            title="Total Interactions by Condition",
            color_discrete_sequence=px.colors.qualitative.Bold,
            template="plotly_white"
        )
        fig_cond.update_layout(height=420)
        st.plotly_chart(fig_cond, use_container_width=True)

    with col2:
        range_df = fdf.groupby(["Condition", "range_group"], dropna=False).size().reset_index(name="count")
        fig_range = px.bar(
            range_df,
            x="Condition",
            y="count",
            color="range_group",
            barmode="group",
            title="Short-Range vs Long-Range Interactions",
            color_discrete_sequence=["#7c3aed", "#ec4899"],
            template="plotly_white"
        )
        fig_range.update_layout(height=420)
        st.plotly_chart(fig_range, use_container_width=True)

    st.markdown('<div class="section-title">Beginner Explanation</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-soft">
    <b>Genomic distance</b> means how far two DNA regions are from each other on the genome.<br><br>
    <b>Interaction strength</b> means how strongly those regions are interacting.<br><br>
    <b>Short-range interactions</b> are more local contacts, while <b>long-range interactions</b> reflect
    larger structural contacts in the genome. Drug treatment can shift these patterns.
    </div>
    """, unsafe_allow_html=True)

# =========================================================
# PAGE 2
# =========================================================
else:
    st.markdown('<div class="section-title">Detailed Analysis Dashboard</div>', unsafe_allow_html=True)

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Distance & Strength",
        "Drug Comparison",
        "Short vs Long",
        "Strand Analysis",
        "DNA Shape & Table"
    ])

    with tab1:
        st.markdown("""
        <div class="info-soft">
        This section examines how interaction strength changes with genomic distance, which is the central
        theme of the abstract.
        </div>
        """, unsafe_allow_html=True)

        scatter_df = safe_sample(
            fdf.dropna(subset=["log_distance", "interaction_strength_proxy"]),
            max_points
        )

        fig1 = px.scatter(
            scatter_df,
            x="log_distance",
            y="interaction_strength_proxy",
            color="Condition",
            title="Log Distance vs Interaction Strength",
            opacity=0.70,
            color_discrete_sequence=px.colors.qualitative.Bold,
            template="plotly_white"
        )
        fig1.update_layout(height=500)
        st.plotly_chart(fig1, use_container_width=True)

        c1, c2 = st.columns(2)

        with c1:
            fig2 = px.histogram(
                fdf.dropna(subset=["genomic_distance_final"]),
                x="genomic_distance_final",
                color="Condition",
                nbins=40,
                title="Distribution of Genomic Distance",
                color_discrete_sequence=px.colors.qualitative.Pastel,
                template="plotly_white"
            )
            fig2.update_layout(height=430)
            st.plotly_chart(fig2, use_container_width=True)

        with c2:
            fig3 = px.box(
                fdf.dropna(subset=["interaction_strength_proxy"]),
                x="Condition",
                y="interaction_strength_proxy",
                color="Condition",
                title="Interaction Strength Distribution",
                color_discrete_sequence=px.colors.qualitative.Set2,
                template="plotly_white"
            )
            fig3.update_layout(height=430)
            st.plotly_chart(fig3, use_container_width=True)

        heat_df = fdf.dropna(subset=["distance_bin", "interaction_strength_proxy"]).groupby(
            ["Condition", "distance_bin"], dropna=False
        )["interaction_strength_proxy"].mean().reset_index()

        if len(heat_df) > 0:
            heat_pivot = heat_df.pivot(index="distance_bin", columns="Condition", values="interaction_strength_proxy")
            fig_heat = px.imshow(
                heat_pivot,
                text_auto=True,
                color_continuous_scale="RdPu",
                aspect="auto",
                title="Mean Interaction Strength Across Distance Bins",
                template="plotly_white"
            )
            fig_heat.update_layout(height=460)
            st.plotly_chart(fig_heat, use_container_width=True)

    with tab2:
        st.markdown("""
        <div class="warn-soft">
        This comparison highlights how Normal, Carboplatin, and Gemcitabine conditions differ in chromatin
        interaction behavior.
        </div>
        """, unsafe_allow_html=True)

        drug_stats = fdf.groupby("Condition", dropna=False).agg(
            total_interactions=("Condition", "size"),
            mean_strength=("interaction_strength_proxy", "mean"),
            mean_distance_bp=("genomic_distance_final", "mean"),
            cis_interactions=("interaction_type", lambda x: (x == "cis").sum())
        ).reset_index()

        c1, c2 = st.columns(2)

        with c1:
            fig4 = px.bar(
                drug_stats,
                x="Condition",
                y="mean_strength",
                color="Condition",
                title="Mean Interaction Strength by Condition",
                color_discrete_sequence=px.colors.qualitative.Vivid,
                template="plotly_white"
            )
            fig4.update_layout(height=430)
            st.plotly_chart(fig4, use_container_width=True)

        with c2:
            fig5 = px.bar(
                drug_stats,
                x="Condition",
                y="mean_distance_bp",
                color="Condition",
                title="Mean Genomic Distance by Condition",
                color_discrete_sequence=px.colors.qualitative.Safe,
                template="plotly_white"
            )
            fig5.update_layout(height=430)
            st.plotly_chart(fig5, use_container_width=True)

        st.dataframe(drug_stats.round(3), use_container_width=True)

    with tab3:
        range_stats = fdf.groupby(["Condition", "range_group"], dropna=False).agg(
            interaction_count=("Condition", "size"),
            mean_strength=("interaction_strength_proxy", "mean")
        ).reset_index()

        c1, c2 = st.columns(2)

        with c1:
            fig6 = px.bar(
                range_stats,
                x="Condition",
                y="interaction_count",
                color="range_group",
                barmode="group",
                title="Short-Range vs Long-Range Counts",
                color_discrete_sequence=["#8b5cf6", "#f43f5e"],
                template="plotly_white"
            )
            fig6.update_layout(height=430)
            st.plotly_chart(fig6, use_container_width=True)

        with c2:
            fig7 = px.bar(
                range_stats,
                x="Condition",
                y="mean_strength",
                color="range_group",
                barmode="group",
                title="Mean Strength in Short vs Long Range",
                color_discrete_sequence=["#06b6d4", "#f59e0b"],
                template="plotly_white"
            )
            fig7.update_layout(height=430)
            st.plotly_chart(fig7, use_container_width=True)

        strength_stats = fdf.groupby(["Condition", "strength_level"], dropna=False).size().reset_index(name="count")
        fig8 = px.bar(
            strength_stats,
            x="Condition",
            y="count",
            color="strength_level",
            barmode="group",
            title="Weak / Moderate / Strong Interactions",
            color_discrete_sequence=["#fb7185", "#fbbf24", "#22c55e"],
            template="plotly_white"
        )
        fig8.update_layout(height=430)
        st.plotly_chart(fig8, use_container_width=True)

    with tab4:
        st.markdown("""
        <div class="info-soft">
        This strand analysis uses the strand field in your dataset together with condition and genomic distance.
        </div>
        """, unsafe_allow_html=True)

        strand_view = fdf.groupby(["Condition", "strand_group"], dropna=False).agg(
            interaction_count=("Condition", "size"),
            mean_distance=("genomic_distance_final", "mean"),
            mean_strength=("interaction_strength_proxy", "mean")
        ).reset_index()

        c1, c2 = st.columns(2)

        with c1:
            fig9 = px.bar(
                strand_view,
                x="Condition",
                y="interaction_count",
                color="strand_group",
                barmode="group",
                title="Strand Distribution by Condition",
                color_discrete_sequence=px.colors.qualitative.Prism,
                template="plotly_white"
            )
            fig9.update_layout(height=430)
            st.plotly_chart(fig9, use_container_width=True)

        with c2:
            strand_scatter = safe_sample(
                fdf.dropna(subset=["genomic_distance_final", "interaction_strength_proxy"]),
                max_points
            )
            fig10 = px.scatter(
                strand_scatter,
                x="genomic_distance_final",
                y="interaction_strength_proxy",
                color="strand_group",
                symbol="Condition",
                title="Distance vs Strength by Strand and Condition",
                opacity=0.70,
                color_discrete_sequence=px.colors.qualitative.Dark24,
                template="plotly_white"
            )
            fig10.update_layout(height=430)
            st.plotly_chart(fig10, use_container_width=True)

        st.dataframe(strand_view.round(3), use_container_width=True)

    with tab5:
        st.markdown("""
        <div class="success-soft">
        DNA shape-style analysis here is shown through proxy features derived from anchor span,
        interactor width, genomic distance, extension ratio, and contact decay.
        </div>
        """, unsafe_allow_html=True)

        shape_view = fdf.groupby(["Condition", "shape_bucket"], dropna=False).agg(
            interaction_count=("Condition", "size"),
            mean_extension_ratio=("shape_extension_ratio", "mean"),
            mean_decay=("shape_contact_decay", "mean")
        ).reset_index()

        c1, c2 = st.columns(2)

        with c1:
            fig11 = px.bar(
                shape_view,
                x="Condition",
                y="interaction_count",
                color="shape_bucket",
                barmode="group",
                title="DNA Shape Bucket by Condition",
                color_discrete_sequence=px.colors.qualitative.G10,
                template="plotly_white"
            )
            fig11.update_layout(height=430)
            st.plotly_chart(fig11, use_container_width=True)

        with c2:
            shape_scatter = safe_sample(
                fdf.dropna(subset=["anchor_span", "interaction_strength_proxy"]),
                max_points
            )
            fig12 = px.scatter(
                shape_scatter,
                x="anchor_span",
                y="interaction_strength_proxy",
                color="Condition",
                size="interactor_width",
                title="Anchor Span vs Interaction Strength",
                opacity=0.70,
                color_discrete_sequence=px.colors.qualitative.Bold,
                template="plotly_white"
            )
            fig12.update_layout(height=430)
            st.plotly_chart(fig12, use_container_width=True)

        st.dataframe(shape_view.round(3), use_container_width=True)

        st.markdown('<div class="section-title">Processed Data Table</div>', unsafe_allow_html=True)

        useful_cols = [
            "Feature_Chr", "Feature_Start", "Interactor_Chr", "Interactor_Start", "Interactor_End",
            "interaction_type", "Condition", "Strand", "strand_group", "genomic_distance_final",
            "distance_mb", "distance_class", "distance_bin", "range_group",
            "interaction_strength_proxy", "strength_level", "anchor_span", "interactor_width",
            "shape_bucket", "shape_compactness", "shape_extension_ratio", "shape_contact_decay"
        ]

        useful_cols = [c for c in useful_cols if c in fdf.columns]

        st.dataframe(fdf[useful_cols].head(300), use_container_width=True)

        csv = fdf[useful_cols].to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download processed data as CSV",
            data=csv,
            file_name="molm1_processed_dashboard_data.csv",
            mime="text/csv"
        )

# =========================================================
# FOOTER
# =========================================================
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#64748b; font-size:0.9rem; padding-bottom:0.6rem;">
MOLM-1 Dashboard · Computational Analysis of Distance-Dependent Chromatin Interactions
</div>
""", unsafe_allow_html=True)
