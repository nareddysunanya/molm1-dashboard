import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO

st.set_page_config(
    page_title="MOLM-1 Chromatin Explorer",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at top left, rgba(37,99,235,0.12), transparent 22%),
        radial-gradient(circle at top right, rgba(16,185,129,0.10), transparent 24%),
        linear-gradient(180deg, #f6faff 0%, #eef4ff 45%, #f9fbff 100%);
    color: #13263c;
}

.block-container {
    padding-top: 1rem;
    padding-bottom: 2rem;
    max-width: 1450px;
}

.hero-box {
    background: linear-gradient(135deg, #1d4ed8, #7c3aed);
    color: white;
    border-radius: 24px;
    padding: 28px 30px;
    box-shadow: 0 18px 45px rgba(15, 23, 42, 0.18);
    margin-bottom: 18px;
}

.hero-box h1 {
    margin: 0;
    font-size: 2.3rem;
    font-weight: 800;
}

.hero-box p {
    margin-top: 10px;
    margin-bottom: 0;
    line-height: 1.65;
    color: rgba(255,255,255,0.94);
}

.info-card {
    background: rgba(255,255,255,0.88);
    border: 1px solid rgba(15,23,42,0.07);
    border-radius: 20px;
    padding: 18px;
    box-shadow: 0 8px 24px rgba(15,23,42,0.06);
    margin-bottom: 14px;
}

.metric-card {
    background: rgba(255,255,255,0.95);
    border: 1px solid rgba(15,23,42,0.07);
    border-radius: 20px;
    padding: 18px;
    box-shadow: 0 8px 22px rgba(15,23,42,0.06);
    min-height: 120px;
}

.metric-label {
    color: #58708a;
    font-size: 0.92rem;
    font-weight: 600;
}

.metric-value {
    color: #16283d;
    font-size: 2rem;
    font-weight: 800;
    margin-top: 10px;
}

.soft-box {
    background: linear-gradient(135deg, rgba(37,99,235,0.10), rgba(16,185,129,0.08));
    border-left: 5px solid #2563eb;
    border-radius: 14px;
    padding: 14px 16px;
    margin-top: 10px;
    margin-bottom: 10px;
    color: #16324a;
}

.warn-box {
    background: linear-gradient(135deg, rgba(245,158,11,0.14), rgba(251,191,36,0.10));
    border-left: 5px solid #d97706;
    border-radius: 14px;
    padding: 14px 16px;
    margin-top: 10px;
    margin-bottom: 10px;
    color: #5c3a00;
}

.treat-box {
    background: linear-gradient(135deg, rgba(139,92,246,0.12), rgba(236,72,153,0.08));
    border-left: 5px solid #7c3aed;
    border-radius: 14px;
    padding: 15px 16px;
    margin-top: 14px;
    margin-bottom: 10px;
    color: #3b1d5a;
}

.condition-chip-row {
    display: flex;
    gap: 10px;
    flex-wrap: wrap;
    margin-bottom: 12px;
}

.condition-chip {
    border-radius: 999px;
    padding: 7px 12px;
    color: white;
    font-size: 0.85rem;
    font-weight: 700;
}

.normal-chip { background: #2563eb; }
.carb-chip { background: #f59e0b; }
.gem-chip { background: #10b981; }

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #eef4ff 0%, #f8fbff 100%);
}

hr {
    border: none;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(37,99,235,0.28), transparent);
    margin: 15px 0;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero-box">
    <h1>🧬 MOLM-1 Chromatin Explorer</h1>
    <p>
        An interactive dashboard for analyzing chromatin interactions in MOLM-1 leukemia cells.
        This tool supports genomic distance analysis, chromosome-level interaction patterns,
        and explicit comparison of <b>Normal</b>, <b>Carboplatin</b>, and <b>Gemcitabine</b> conditions.
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="info-card">
    <b>What this tool does:</b><br>
    It reads chromatin interaction tables, identifies same-chromosome and different-chromosome contacts,
    calculates genomic distance when start-position columns are available, and creates interactive plots
    to explain how treatment may change short-range and long-range interaction behavior.
</div>
""", unsafe_allow_html=True)

with st.expander("Simple meaning of the biology words"):
    st.write("""
- Chromatin interactions = physical contacts between DNA regions.
- Within same chromosome = local DNA contacts.
- Between different chromosomes = cross-chromosome contacts.
- Short-range = nearby DNA contacts.
- Long-range = farther-apart DNA contacts.
- Treatment comparison = how interaction patterns change in Normal, Carboplatin, and Gemcitabine conditions.
""")

uploaded_file = st.file_uploader(
    "Upload a data file",
    type=["csv", "tsv", "txt", "xlsx", "xls"]
)

def auto_pick(columns, keywords):
    for key in keywords:
        for col in columns:
            if key in str(col).lower().strip():
                return col
    return None

def safe_numeric(series):
    return pd.to_numeric(series, errors="coerce")

def classify_distance(distance_mb):
    if pd.isna(distance_mb):
        return np.nan
    return "Short-range" if distance_mb < 1 else "Long-range"

def compare_direction(current, baseline):
    if pd.isna(current) or pd.isna(baseline) or baseline == 0:
        return "could not be compared clearly"
    pct = ((current - baseline) / baseline) * 100
    if pct > 5:
        return f"increased by {pct:.2f}%"
    elif pct < -5:
        return f"decreased by {abs(pct):.2f}%"
    return "stayed relatively stable"

@st.cache_data(show_spinner=False)
def get_excel_sheet_names(file_bytes):
    xls = pd.ExcelFile(BytesIO(file_bytes))
    return xls.sheet_names

@st.cache_data(show_spinner=False)
def load_data(file_name, file_bytes, selected_sheet=None):
    lower = file_name.lower()

    if lower.endswith(".csv"):
        return pd.read_csv(BytesIO(file_bytes), sep=None, engine="python", low_memory=False)

    if lower.endswith(".tsv") or lower.endswith(".txt"):
        try:
            return pd.read_csv(BytesIO(file_bytes), sep="\t", low_memory=False)
        except:
            return pd.read_csv(BytesIO(file_bytes), sep=None, engine="python", low_memory=False)

    if lower.endswith(".xlsx") or lower.endswith(".xls"):
        return pd.read_excel(BytesIO(file_bytes), sheet_name=selected_sheet if selected_sheet is not None else 0)

    raise ValueError("Unsupported file format.")

if uploaded_file is not None:
    try:
        file_bytes = uploaded_file.getvalue()
        file_name = uploaded_file.name

        selected_sheet = None
        if file_name.lower().endswith((".xlsx", ".xls")):
            sheet_names = get_excel_sheet_names(file_bytes)
            selected_sheet = st.selectbox("Choose Excel sheet", sheet_names, index=0)

        with st.spinner("Loading and analyzing your file..."):
            df = load_data(file_name, file_bytes, selected_sheet)

        if df is None or df.empty:
            st.error("The uploaded table is empty.")
        else:
            st.success(f"Loaded: {file_name}")
            st.write(f"Rows: {len(df):,} | Columns: {len(df.columns)}")
            st.dataframe(df.head(12), use_container_width=True)

            all_cols = list(df.columns)
            numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

            st.sidebar.header("Column mapping")

            feature_chr_default = auto_pick(all_cols, ["feature_chr", "bait_chr", "chr1", "feature chr", "chr"])
            interactor_chr_default = auto_pick(all_cols, ["interactor_chr", "target_chr", "chr2", "interactor chr", "other_chr"])
            feature_start_default = auto_pick(all_cols, ["feature_start", "bait_start", "start1", "feature start"])
            interactor_start_default = auto_pick(all_cols, ["interactor_start", "target_start", "start2", "interactor start"])

            feature_chr = st.sidebar.selectbox(
                "First region chromosome column",
                all_cols,
                index=all_cols.index(feature_chr_default) if feature_chr_default in all_cols else 0
            )

            interactor_chr = st.sidebar.selectbox(
                "Second region chromosome column",
                all_cols,
                index=all_cols.index(interactor_chr_default) if interactor_chr_default in all_cols else min(1, len(all_cols)-1)
            )

            feature_start = st.sidebar.selectbox(
                "First region start column",
                ["None"] + all_cols,
                index=(["None"] + all_cols).index(feature_start_default) if feature_start_default in (["None"] + all_cols) else 0
            )

            interactor_start = st.sidebar.selectbox(
                "Second region start column",
                ["None"] + all_cols,
                index=(["None"] + all_cols).index(interactor_start_default) if interactor_start_default in (["None"] + all_cols) else 0
            )

            st.sidebar.markdown("### Treatment condition mapping")
            normal_col = st.sidebar.selectbox("Normal column", ["None"] + all_cols, index=0)
            carboplatin_col = st.sidebar.selectbox("Carboplatin column", ["None"] + all_cols, index=0)
            gemcitabine_col = st.sidebar.selectbox("Gemcitabine column", ["None"] + all_cols, index=0)

            extra_signal_cols = st.sidebar.multiselect(
                "Optional extra numeric columns",
                numeric_cols,
                default=[]
            )

            max_rows = st.sidebar.slider(
                "Rows to analyze",
                min_value=100,
                max_value=max(100, len(df)),
                value=min(len(df), 10000)
            )

            work_df = df.head(max_rows).copy()
            work_df["Feature_chr_clean"] = work_df[feature_chr].astype(str).str.strip()
            work_df["Interactor_chr_clean"] = work_df[interactor_chr].astype(str).str.strip()

            work_df["Interaction_Type"] = np.where(
                work_df["Feature_chr_clean"] == work_df["Interactor_chr_clean"],
                "Within same chromosome",
                "Between different chromosomes"
            )

            if feature_start != "None" and interactor_start != "None":
                work_df["Feature_start_num"] = safe_numeric(work_df[feature_start])
                work_df["Interactor_start_num"] = safe_numeric(work_df[interactor_start])
                work_df["Genomic_Distance"] = np.where(
                    work_df["Interaction_Type"] == "Within same chromosome",
                    (work_df["Interactor_start_num"] - work_df["Feature_start_num"]).abs(),
                    np.nan
                )
                work_df["Distance_Mb"] = work_df["Genomic_Distance"] / 1_000_000
                work_df["Distance_Class"] = work_df["Distance_Mb"].apply(classify_distance)
            else:
                work_df["Genomic_Distance"] = np.nan
                work_df["Distance_Mb"] = np.nan
                work_df["Distance_Class"] = np.nan

            total_interactions = len(work_df)
            same_chr = int((work_df["Interaction_Type"] == "Within same chromosome").sum())
            diff_chr = int((work_df["Interaction_Type"] == "Between different chromosomes").sum())
            unique_feature_chr = work_df["Feature_chr_clean"].nunique()
            unique_interactor_chr = work_df["Interactor_chr_clean"].nunique()

            st.subheader("Dashboard summary")
            c1, c2, c3, c4, c5 = st.columns(5)
            with c1:
                st.markdown(f'<div class="metric-card"><div class="metric-label">Total interactions</div><div class="metric-value">{total_interactions:,}</div></div>', unsafe_allow_html=True)
            with c2:
                st.markdown(f'<div class="metric-card"><div class="metric-label">Within same chromosome</div><div class="metric-value">{same_chr:,}</div></div>', unsafe_allow_html=True)
            with c3:
                st.markdown(f'<div class="metric-card"><div class="metric-label">Between chromosomes</div><div class="metric-value">{diff_chr:,}</div></div>', unsafe_allow_html=True)
            with c4:
                st.markdown(f'<div class="metric-card"><div class="metric-label">Unique first-region chr</div><div class="metric-value">{unique_feature_chr}</div></div>', unsafe_allow_html=True)
            with c5:
                st.markdown(f'<div class="metric-card"><div class="metric-label">Unique second-region chr</div><div class="metric-value">{unique_interactor_chr}</div></div>', unsafe_allow_html=True)

            st.markdown(f"""
<div class="soft-box">
<b>Quick summary:</b> The current dataset contains <b>{total_interactions:,}</b> interaction records.
This overview helps show whether the chromatin map is mostly local within chromosomes or more spread across chromosomes.
</div>
""", unsafe_allow_html=True)

            tabs = st.tabs([
                "Overview",
                "Distance analysis",
                "Interaction strength",
                "Chromosome hubs",
                "Treatment comparison",
                "Processed table"
            ])

            with tabs[0]:
                col1, col2 = st.columns(2)

                interaction_counts = work_df["Interaction_Type"].value_counts().reset_index()
                interaction_counts.columns = ["Interaction_Type", "Count"]

                fig1 = px.bar(
                    interaction_counts,
                    x="Interaction_Type",
                    y="Count",
                    color="Interaction_Type",
                    title="Distribution of interaction types",
                    color_discrete_sequence=["#2563eb", "#ef4444"]
                )
                fig1.update_layout(template="plotly_white", height=420)
                col1.plotly_chart(fig1, use_container_width=True)

                chr_counts = pd.concat([
                    work_df["Feature_chr_clean"],
                    work_df["Interactor_chr_clean"]
                ]).value_counts().head(12).reset_index()
                chr_counts.columns = ["Chromosome", "Count"]

                fig2 = px.bar(
                    chr_counts,
                    x="Chromosome",
                    y="Count",
                    color="Count",
                    title="Top chromosomes in the interaction map",
                    color_continuous_scale="Blues"
                )
                fig2.update_layout(template="plotly_white", height=420)
                col2.plotly_chart(fig2, use_container_width=True)

                st.markdown("""
<div class="soft-box">
This section shows the main chromatin interaction story first: whether contacts are mainly local or cross-chromosomal,
and which chromosomes appear most frequently in the network.
</div>
""", unsafe_allow_html=True)

            with tabs[1]:
                if work_df["Distance_Mb"].notna().sum() > 0:
                    dist_df = work_df[work_df["Distance_Mb"].notna()].copy()
                    col3, col4 = st.columns(2)

                    fig3 = px.histogram(
                        dist_df,
                        x="Distance_Mb",
                        nbins=50,
                        title="Genomic distance distribution for same-chromosome contacts",
                        color_discrete_sequence=["#14b8a6"]
                    )
                    fig3.update_layout(template="plotly_white", height=420)
                    col3.plotly_chart(fig3, use_container_width=True)

                    class_counts = dist_df["Distance_Class"].value_counts().reset_index()
                    class_counts.columns = ["Distance_Class", "Count"]
                    fig4 = px.pie(
                        class_counts,
                        names="Distance_Class",
                        values="Count",
                        title="Short-range vs long-range interactions",
                        color="Distance_Class",
                        color_discrete_map={"Short-range": "#2563eb", "Long-range": "#8b5cf6"}
                    )
                    fig4.update_layout(height=420)
                    col4.plotly_chart(fig4, use_container_width=True)

                    short_count = int((dist_df["Distance_Class"] == "Short-range").sum())
                    long_count = int((dist_df["Distance_Class"] == "Long-range").sum())

                    st.markdown(f"""
<div class="soft-box">
<b>Distance summary:</b> The current selection contains <b>{short_count:,}</b> short-range interactions and
<b>{long_count:,}</b> long-range interactions. This directly supports the abstract’s focus on short-range and long-range chromatin behavior.
</div>
""", unsafe_allow_html=True)
                else:
                    st.markdown("""
<div class="warn-box">
Distance analysis is not available yet because valid genomic start columns were not selected.
Choose the correct start-position columns in the sidebar.
</div>
""", unsafe_allow_html=True)

            with tabs[2]:
                usable_numeric = []
                for col in extra_signal_cols:
                    series = safe_numeric(work_df[col])
                    if series.notna().sum() > 0:
                        work_df[col] = series
                        usable_numeric.append(col)

                if work_df["Distance_Mb"].notna().sum() > 0 and len(usable_numeric) > 0:
                    signal_col = st.selectbox("Choose signal column for distance-vs-strength plot", usable_numeric)
                    plot_df = work_df[["Distance_Mb", signal_col, "Interaction_Type"]].copy().dropna()

                    if not plot_df.empty:
                        fig5 = px.scatter(
                            plot_df,
                            x="Distance_Mb",
                            y=signal_col,
                            color="Interaction_Type",
                            title=f"Interaction strength vs genomic distance: {signal_col}",
                            opacity=0.7,
                            color_discrete_map={
                                "Within same chromosome": "#2563eb",
                                "Between different chromosomes": "#ef4444"
                            }
                        )
                        fig5.update_layout(template="plotly_white", height=500)
                        st.plotly_chart(fig5, use_container_width=True)

                        st.markdown("""
<div class="soft-box">
This plot helps connect interaction strength with genomic distance, which is one of the main scientific ideas in chromatin interaction analysis.
</div>
""", unsafe_allow_html=True)
                    else:
                        st.info("No valid rows available for the selected signal column.")
                else:
                    st.markdown("""
<div class="warn-box">
To use this section, choose at least one valid numeric signal column and valid genomic start-position columns.
</div>
""", unsafe_allow_html=True)

            with tabs[3]:
                inter_df = work_df[work_df["Interaction_Type"] == "Between different chromosomes"].copy()

                if not inter_df.empty:
                    hub_df = pd.concat([
                        inter_df["Feature_chr_clean"],
                        inter_df["Interactor_chr_clean"]
                    ]).value_counts().head(15).reset_index()
                    hub_df.columns = ["Chromosome", "Connections"]

                    col5, col6 = st.columns([1.35, 1])
                    fig6 = px.bar(
                        hub_df,
                        x="Chromosome",
                        y="Connections",
                        color="Connections",
                        title="Top chromosome hubs",
                        color_continuous_scale="Sunsetdark"
                    )
                    fig6.update_layout(template="plotly_white", height=450)
                    col5.plotly_chart(fig6, use_container_width=True)
                    col6.dataframe(hub_df, use_container_width=True)

                    top_chr = hub_df.iloc[0]["Chromosome"]
                    top_conn = int(hub_df.iloc[0]["Connections"])

                    st.markdown(f"""
<div class="soft-box">
<b>Hub summary:</b> Chromosome <b>{top_chr}</b> appears as the strongest cross-chromosome hub with
<b>{top_conn}</b> detected connections in the current selection.
</div>
""", unsafe_allow_html=True)
                else:
                    st.info("No between-chromosome interactions were detected in the selected rows.")

            with tabs[4]:
                st.markdown("""
<div class="condition-chip-row">
    <div class="condition-chip normal-chip">Normal</div>
    <div class="condition-chip carb-chip">Carboplatin</div>
    <div class="condition-chip gem-chip">Gemcitabine</div>
</div>
""", unsafe_allow_html=True)

                selected_map = {
                    "Normal": normal_col,
                    "Carboplatin": carboplatin_col,
                    "Gemcitabine": gemcitabine_col
                }

                usable_conditions = {}
                for label, colname in selected_map.items():
                    if colname != "None" and colname in work_df.columns:
                        numeric_series = safe_numeric(work_df[colname])
                        if numeric_series.notna().sum() > 0:
                            work_df[colname] = numeric_series
                            usable_conditions[label] = colname

                if len(usable_conditions) >= 2:
                    total_rows = []
                    mean_rows = []

                    for label, colname in usable_conditions.items():
                        total_rows.append({"Condition": label, "Total_Signal": work_df[colname].sum(skipna=True)})
                        mean_rows.append({"Condition": label, "Mean_Signal": work_df[colname].mean(skipna=True)})

                    total_df = pd.DataFrame(total_rows)
                    mean_df = pd.DataFrame(mean_rows)

                    col7, col8 = st.columns(2)

                    fig7 = px.bar(
                        total_df,
                        x="Condition",
                        y="Total_Signal",
                        color="Condition",
                        title="Normal vs Carboplatin vs Gemcitabine: total signal",
                        color_discrete_map={
                            "Normal": "#2563eb",
                            "Carboplatin": "#f59e0b",
                            "Gemcitabine": "#10b981"
                        }
                    )
                    fig7.update_layout(template="plotly_white", height=420)
                    col7.plotly_chart(fig7, use_container_width=True)

                    fig8 = px.line(
                        mean_df,
                        x="Condition",
                        y="Mean_Signal",
                        color="Condition",
                        markers=True,
                        title="Average interaction signal across biological conditions",
                        color_discrete_map={
                            "Normal": "#2563eb",
                            "Carboplatin": "#f59e0b",
                            "Gemcitabine": "#10b981"
                        }
                    )
                    fig8.update_layout(template="plotly_white", height=420, showlegend=False)
                    col8.plotly_chart(fig8, use_container_width=True)

                    normal_total = total_df.loc[total_df["Condition"] == "Normal", "Total_Signal"]
                    carb_total = total_df.loc[total_df["Condition"] == "Carboplatin", "Total_Signal"]
                    gem_total = total_df.loc[total_df["Condition"] == "Gemcitabine", "Total_Signal"]

                    normal_total = normal_total.iloc[0] if len(normal_total) else np.nan
                    carb_total = carb_total.iloc[0] if len(carb_total) else np.nan
                    gem_total = gem_total.iloc[0] if len(gem_total) else np.nan

                    if work_df["Distance_Mb"].notna().sum() > 0:
                        dist_df = work_df[work_df["Distance_Mb"].notna()].copy()
                        short_long_rows = []

                        for label, colname in usable_conditions.items():
                            temp = dist_df[["Distance_Class", colname]].copy().dropna()
                            grouped = temp.groupby("Distance_Class")[colname].sum().reset_index() if not temp.empty else pd.DataFrame()
                            short_val = grouped.loc[grouped["Distance_Class"] == "Short-range", colname] if not grouped.empty else pd.Series(dtype=float)
                            long_val = grouped.loc[grouped["Distance_Class"] == "Long-range", colname] if not grouped.empty else pd.Series(dtype=float)

                            short_long_rows.append({
                                "Condition": label,
                                "Short-range": short_val.iloc[0] if len(short_val) else 0,
                                "Long-range": long_val.iloc[0] if len(long_val) else 0
                            })

                        sl_df = pd.DataFrame(short_long_rows)

                        if not sl_df.empty:
                            sl_melt = sl_df.melt(
                                id_vars="Condition",
                                value_vars=["Short-range", "Long-range"],
                                var_name="Range_Type",
                                value_name="Signal"
                            )

                            fig9 = px.bar(
                                sl_melt,
                                x="Condition",
                                y="Signal",
                                color="Range_Type",
                                barmode="group",
                                title="Short-range vs long-range changes across conditions",
                                color_discrete_map={"Short-range": "#2563eb", "Long-range": "#8b5cf6"}
                            )
                            fig9.update_layout(template="plotly_white", height=440)
                            st.plotly_chart(fig9, use_container_width=True)

                            normal_short = sl_df.loc[sl_df["Condition"] == "Normal", "Short-range"]
                            carb_short = sl_df.loc[sl_df["Condition"] == "Carboplatin", "Short-range"]
                            gem_short = sl_df.loc[sl_df["Condition"] == "Gemcitabine", "Short-range"]
                            normal_long = sl_df.loc[sl_df["Condition"] == "Normal", "Long-range"]
                            carb_long = sl_df.loc[sl_df["Condition"] == "Carboplatin", "Long-range"]
                            gem_long = sl_df.loc[sl_df["Condition"] == "Gemcitabine", "Long-range"]

                            normal_short = normal_short.iloc[0] if len(normal_short) else np.nan
                            carb_short = carb_short.iloc[0] if len(carb_short) else np.nan
                            gem_short = gem_short.iloc[0] if len(gem_short) else np.nan
                            normal_long = normal_long.iloc[0] if len(normal_long) else np.nan
                            carb_long = carb_long.iloc[0] if len(carb_long) else np.nan
                            gem_long = gem_long.iloc[0] if len(gem_long) else np.nan

                            st.markdown(f"""
<div class="treat-box">
<b>Plain-language treatment summary:</b><br><br>
Compared with <b>Normal</b>, total signal in <b>Carboplatin</b> {compare_direction(carb_total, normal_total)}.<br>
Compared with <b>Normal</b>, total signal in <b>Gemcitabine</b> {compare_direction(gem_total, normal_total)}.<br>
Short-range signal in <b>Carboplatin</b> {compare_direction(carb_short, normal_short)} relative to Normal.<br>
Long-range signal in <b>Carboplatin</b> {compare_direction(carb_long, normal_long)} relative to Normal.<br>
Short-range signal in <b>Gemcitabine</b> {compare_direction(gem_short, normal_short)} relative to Normal.<br>
Long-range signal in <b>Gemcitabine</b> {compare_direction(gem_long, normal_long)} relative to Normal.
</div>
""", unsafe_allow_html=True)
                        else:
                            st.markdown("""
<div class="warn-box">
Condition columns were found, but short-range and long-range treatment summary could not be computed from the available rows.
</div>
""", unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
<div class="treat-box">
<b>Plain-language treatment summary:</b><br><br>
Compared with <b>Normal</b>, total signal in <b>Carboplatin</b> {compare_direction(carb_total, normal_total)}.<br>
Compared with <b>Normal</b>, total signal in <b>Gemcitabine</b> {compare_direction(gem_total, normal_total)}.<br><br>
To compare short-range and long-range treatment effects, select valid genomic start-position columns.
</div>
""", unsafe_allow_html=True)
                else:
                    st.markdown("""
<div class="warn-box">
Please choose at least two valid biological condition columns from the sidebar.
For this project, the clearest setup is to map columns for Normal, Carboplatin, and Gemcitabine.
</div>
""", unsafe_allow_html=True)

            with tabs[5]:
                st.dataframe(work_df, use_container_width=True)

                csv_data = work_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download processed analysis table",
                    data=csv_data,
                    file_name="molm1_chromatin_analysis.csv",
                    mime="text/csv"
                )

                with st.expander("See all detected column names"):
                    st.write(all_cols)

    except Exception as e:
        st.error(f"Error while processing the file: {e}")
else:
    st.info("Upload a CSV, TSV, TXT, XLSX, or XLS file to start the analysis.")