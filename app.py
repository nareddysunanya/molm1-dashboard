import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
import os

# ========================================
# 1. LOAD ANY FILE
# ========================================
def load_interaction_file(uploaded_file):
    file_name = uploaded_file.name if hasattr(uploaded_file, "name") else str(uploaded_file)
    ext = os.path.splitext(file_name)[1].lower()

    if ext == ".csv":
        df = pd.read_csv(uploaded_file)
    elif ext in [".xlsx", ".xls"]:
        df = pd.read_excel(uploaded_file)
    else:
        raise ValueError("Unsupported file format. Please upload CSV or Excel file.")
    
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
    if "" in df.columns:
        df = df.drop(columns=[""])
    return df

# ========================================
# 2. CLEAN HELPER FUNCTIONS
# ========================================
def clean_chr(x):
    if pd.isna(x):
        return np.nan
    x = str(x).strip().lower().replace("chromosome", "").replace("chr", "").strip()
    return "chr" + x if x else np.nan

def clean_strand(x):
    if pd.isna(x):
        return "unknown"
    x = str(x).strip().lower()
    mapping = {"1": "+", "+1": "+", "plus": "+", "+": "+", "-1": "-", "minus": "-", "-": "-"}
    return mapping.get(x, x)

def get_condition(row):
    labels = []
    if row.get("Normal", 0) == 1: labels.append("Normal")
    if row.get("CarboplatinTreated", 0) == 1: labels.append("Carboplatin")
    if row.get("GemcitabineTreated", 0) == 1: labels.append("Gemcitabine")
    return labels[0] if len(labels) == 1 else ("+".join(labels) if len(labels) > 1 else "Unlabeled")

def classify_distance(x):
    if pd.isna(x): return "trans_or_unknown"
    if x <= 100000: return "short_range"
    if x <= 1000000: return "medium_range"
    return "long_range"

def classify_shape(row):
    d = row.get('genomic_distance_final', np.nan)
    region_width = row.get('interactor_width', np.nan)
    anchor_span = row.get('anchor_span', np.nan)
    if pd.isna(d): return 'trans_shape'
    if d <= 100000 and pd.notna(region_width) and region_width <= 2000: return 'compact_loop'
    if d <= 1000000 and pd.notna(anchor_span) and anchor_span <= 1000000: return 'local_arc'
    if d > 1000000 and pd.notna(anchor_span) and anchor_span > 1000000: return 'extended_loop'
    return 'broad_contact'

# ========================================
# 3. MAIN DISTANCE-FIRST ALGORITHM
# ========================================
def process_molm1_data(df):
    df = df.copy()
    
    # Clean text columns
    text_cols = ['RefSeqName', 'TranscriptName', 'Feature_Chr', 'Interactor_Chr', 'InteractorName', 
                'InteractorID', 'Strand', 'Annotation', 'InteractorAnnotation', 'IntGroup']
    for col in text_cols:
        if col in df.columns: df[col] = df[col].astype(str).str.strip()

    # Convert numeric columns
    num_cols = ['Feature_Start', 'Interactor_Start', 'Interactor_End', 'distance', 'abs_distance',
               'MG1_SuppPairs', 'MG2_SuppPairs', 'MC1_SuppPairs', 'MC2_SuppPairs', 
               'MN1_SuppPairs', 'MN2_SuppPairs', 'MG1_p_value', 'MG2_p_value', 
               'MC1_p_value', 'MC2_p_value', 'MN1_p_value', 'MN2_p_value',
               'Normal', 'CarboplatinTreated', 'GemcitabineTreated', 'NofInts']
    for col in num_cols:
        if col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce')

    # Standardize chromosomes and strand
    if 'Feature_Chr' in df.columns: df['Feature_Chr'] = df['Feature_Chr'].apply(clean_chr)
    if 'Interactor_Chr' in df.columns: df['Interactor_Chr'] = df['Interactor_Chr'].apply(clean_chr)
    if 'Strand' in df.columns: df['Strand'] = df['Strand'].apply(clean_strand)
    else: df['Strand'] = 'unknown'

    # Interactor geometry
    if {'Interactor_Start', 'Interactor_End'}.issubset(df.columns):
        df['Interactor_Mid'] = ((df['Interactor_Start'] + df['Interactor_End']) / 2).round()
        df['interactor_width'] = (df['Interactor_End'] - df['Interactor_Start']).abs()
    else:
        df['Interactor_Mid'] = df['interactor_width'] = np.nan

    # Cis/trans
    if {'Feature_Chr', 'Interactor_Chr'}.issubset(df.columns):
        df['interaction_type'] = np.where(df['Feature_Chr'] == df['Interactor_Chr'], 'cis', 'trans')
    else: df['interaction_type'] = 'unknown'

    # Distance calculation
    if {'Feature_Start', 'Interactor_Mid'}.issubset(df.columns):
        df['computed_distance'] = np.where(df['interaction_type'] == 'cis', 
                                         (df['Feature_Start'] - df['Interactor_Mid']).abs(), np.nan)
    else: df['computed_distance'] = np.nan

    df['genomic_distance_final'] = df['abs_distance'].fillna(df['computed_distance']) if 'abs_distance' in df.columns else df['computed_distance']
    df.loc[df['interaction_type'] == 'trans', 'genomic_distance_final'] = np.nan

    # Distance bins
    distance_bins = [-1, 100000, 500000, 1000000, 5000000, 10000000, np.inf]
    distance_labels = ['0-100kb', '100kb-500kb', '500kb-1Mb', '1Mb-5Mb', '5Mb-10Mb', '>10Mb']
    df['distance_bin'] = pd.cut(df['genomic_distance_final'], bins=distance_bins, labels=distance_labels)
    df['distance_class'] = df['genomic_distance_final'].apply(classify_distance)

    # Conditions and support
    df['Condition'] = df.apply(get_condition, axis=1)
    for c in ['MG1_SuppPairs', 'MG2_SuppPairs', 'MC1_SuppPairs', 'MC2_SuppPairs', 'MN1_SuppPairs', 'MN2_SuppPairs']:
        if c not in df: df[c] = 0
    df['Gem_Support'] = df['MG1_SuppPairs'] + df['MG2_SuppPairs']
    df['Carbo_Support'] = df['MC1_SuppPairs'] + df['MC2_SuppPairs']
    df['Normal_Support'] = df['MN1_SuppPairs'] + df['MN2_SuppPairs']

    def active_support(row):
        if row['Condition'] == 'Normal': return row['Normal_Support']
        elif row['Condition'] == 'Carboplatin': return row['Carbo_Support']
        elif row['Condition'] == 'Gemcitabine': return row['Gem_Support']
        return df.get('NofInts', np.nan)[df.index.get_loc(row.name)] if 'NofInts' in df else np.nan

    df['Active_Support'] = df.apply(active_support, axis=1)
    df['interaction_strength_proxy'] = df['Active_Support'].fillna(df.get('NofInts', np.nan))

    # Distance-based features
    df['strand_group'] = df['Strand'].fillna('unknown')
    df['strand_distance_score'] = np.where(df['genomic_distance_final'].notna(),
                                         df['genomic_distance_final'] / df['interaction_strength_proxy'].replace(0, np.nan), np.nan)
    df['log_distance'] = np.log10(df['genomic_distance_final'].replace(0, np.nan))
    df['distance_weighted_signal'] = df['interaction_strength_proxy'] / df['genomic_distance_final'].replace(0, np.nan)
    df['anchor_span'] = (df['Feature_Start'] - df['Interactor_Mid']).abs() if 'Feature_Start' in df and 'Interactor_Mid' in df else np.nan
    df['shape_compactness'] = df['interactor_width'] / df['anchor_span'].replace(0, np.nan)
    df['shape_bucket'] = df.apply(classify_shape, axis=1)

    return df

# ========================================
# 4. SUMMARY TABLES
# ========================================
def create_analysis_tables(df):
    distance_summary = df.groupby('Condition', dropna=False).agg(
        total_interactions=('Condition', 'size'),
        cis_interactions=('interaction_type', lambda x: (x == 'cis').sum()),
        mean_distance=('genomic_distance_final', 'mean'),
        mean_strength=('interaction_strength_proxy', 'mean')
    ).reset_index()

    strand_summary = df.groupby(['Condition', 'strand_group'], dropna=False).agg(
        interaction_count=('strand_group', 'size'),
        mean_distance=('genomic_distance_final', 'mean'),
        mean_strength=('interaction_strength_proxy', 'mean')
    ).reset_index()

    distance_bin_summary = df.groupby(['Condition', 'distance_bin'], dropna=False).agg(
        interaction_count=('distance_bin', 'size'),
        mean_strength=('interaction_strength_proxy', 'mean')
    ).reset_index()

    shape_summary = df.groupby(['Condition', 'shape_bucket'], dropna=False).agg(
        interaction_count=('shape_bucket', 'size'),
        mean_distance=('genomic_distance_final', 'mean'),
        mean_strength=('interaction_strength_proxy', 'mean')
    ).reset_index()

    return distance_summary, distance_bin_summary, strand_summary, shape_summary

# ========================================
# 5. STREAMLIT UI (MUST BE LAST)
# ========================================
st.set_page_config(page_title='MOLM-1 Distance Analysis Tool', layout='wide')

st.markdown('''
<style>
.block-container {padding-top: 1.2rem;}
[data-testid="stMetricValue"] {font-size: 1.6rem;}
.section-title {font-weight: 700; font-size: 1.2rem; margin: 1rem 0 0.5rem 0;}
</style>
''', unsafe_allow_html=True)

st.title('🔬 MOLM-1 Distance-Dependent Chromatin Analysis')
st.caption('Upload CSV/Excel → Distance-first analysis → Strand + Shape insights')

# Sidebar
with st.sidebar:
    st.header('📁 Upload')
    uploaded_file = st.file_uploader('Choose file', type=['csv', 'xlsx', 'xls'])
    show_raw = st.toggle('Show raw data', value=False)

if uploaded_file is None:
    st.info('👆 Upload your MOLM-1 dataset to start analysis')
    st.stop()

try:
    raw_df = load_interaction_file(uploaded_file)
    processed_df = process_molm1_data(raw_df)
    distance_summary, distance_bin_summary, strand_summary, shape_summary = create_analysis_tables(processed_df)
except Exception as e:
    st.error(f'Error: {e}')
    st.stop()

# KPIs
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Interactions", f"{len(processed_df):,}")
col2.metric("Cis Interactions", f"{(processed_df['interaction_type'] == 'cis').sum():,}")
col3.metric("Mean Distance", f"{processed_df['genomic_distance_final'].mean():,.0f} bp")
col4.metric("Mean Strength", f"{processed_df['interaction_strength_proxy'].mean():.1f}")

conditions = sorted(processed_df['Condition'].dropna().unique())
selected_conditions = st.multiselect('Filter conditions', conditions, default=conditions)
filtered_df = processed_df[processed_df['Condition'].isin(selected_conditions)]

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(['📊 Overview', '📏 Distance', '🧬 Strand', '🔺 Shape'])

with tab1:
    st.markdown('<div class="section-title">Condition Summary</div>', unsafe_allow_html=True)
    st.dataframe(distance_summary)
    
    col_a, col_b = st.columns(2)
    with col_a: st.plotly_chart(px.bar(distance_summary, x='Condition', y='mean_distance', color='Condition', title='Mean Distance'))
    with col_b: st.plotly_chart(px.bar(distance_summary, x='Condition', y='mean_strength', color='Condition', title='Mean Strength'))

with tab2:
    st.markdown('<div class="section-title">Distance Analysis</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(filtered_df.dropna(subset=['genomic_distance_final']), 
                          x='genomic_distance_final', color='Condition', nbins=50, 
                          title='Distance Distribution')
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.plotly_chart(px.bar(distance_bin_summary, x='distance_bin', y='interaction_count', 
                              color='Condition', barmode='group', title='Interactions by Distance Bin'))
    
    st.dataframe(distance_bin_summary)

with tab3:
    st.markdown('<div class="section-title">Strand Analysis</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1: st.plotly_chart(px.bar(strand_summary, x='strand_group', y='interaction_count', color='Condition', barmode='group'))
    with col2: st.plotly_chart(px.scatter(filtered_df, x='strand_group', y='strand_distance_score', color='Condition'))
    st.dataframe(strand_summary)

with tab4:
    st.markdown('<div class="section-title">DNA Shape Analysis</div>', unsafe_allow_html=True)
    st.info('Shape categories derived from genomic geometry (compactness, extension ratio)')
    col1, col2 = st.columns(2)
    with col1: st.plotly_chart(px.bar(shape_summary, x='shape_bucket', y='interaction_count', color='Condition', barmode='group'))
    with col2: st.plotly_chart(px.scatter(filtered_df.dropna(subset=['shape_compactness']), 
                                        x='shape_compactness', y='interaction_strength_proxy', 
                                        color='shape_bucket', title='Shape vs Strength'))
    st.dataframe(shape_summary)

if show_raw:
    st.markdown('<div class="section-title">Raw Data</div>', unsafe_allow_html=True)
    st.dataframe(raw_df.head(100))

# Download
csv = filtered_df.to_csv(index=False).encode('utf-8')
st.download_button('💾 Download Processed Data', csv, 'molm1_processed.csv', 'text/csv')
