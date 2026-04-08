import os
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title='MOLM-1 Chromatin Dashboard', page_icon='🧬', layout='wide', initial_sidebar_state='expanded')

st.markdown('''
<style>
.stApp {
    background: linear-gradient(180deg, #f8f9ff 0%, #fff8fc 100%);
}
.block-container {max-width: 1420px; padding-top: 1rem; padding-bottom: 2rem;}
.hero {
    background: linear-gradient(135deg, rgba(124,58,237,0.10), rgba(37,99,235,0.08), rgba(236,72,153,0.08));
    border: 1px solid rgba(124,58,237,0.10);
    border-radius: 22px;
    padding: 1.2rem 1.3rem;
    margin-bottom: 1rem;
}
.hero-title {font-size: 2rem; font-weight: 800; color:#1e293b;}
.hero-sub {font-size: 1rem; color:#64748b; margin-top:0.35rem;}
.note {background:#f5f3ff; border-left:5px solid #7c3aed; padding:0.9rem 1rem; border-radius:14px; margin:0.75rem 0 1rem 0; color:#334155;}
.card {background:#ffffff; border:1px solid rgba(148,163,184,0.18); border-radius:18px; padding:1rem; box-shadow:0 6px 20px rgba(15,23,42,0.05);}
[data-testid="stMetric"] {background:#fff; border:1px solid rgba(148,163,184,0.18); border-radius:16px; padding:0.8rem;}
</style>
''', unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def load_interaction_file(file_bytes, name):
    ext = os.path.splitext(name)[1].lower()
    from io import BytesIO
    bio = BytesIO(file_bytes)
    if ext == '.csv':
        df = pd.read_csv(bio)
    elif ext in ['.xlsx', '.xls']:
        df = pd.read_excel(bio)
    else:
        raise ValueError('Unsupported format. Use CSV or Excel.')
    df = df.loc[:, ~df.columns.astype(str).str.contains('^Unnamed')]
    if '' in df.columns:
        df = df.drop(columns=[''])
    return df


def clean_chr(x):
    if pd.isna(x):
        return np.nan
    x = str(x).strip().lower().replace('chromosome', '').replace('chr', '').strip()
    return 'chr' + x if x else np.nan


def clean_strand(x):
    if pd.isna(x):
        return 'unknown'
    x = str(x).strip().lower()
    return {'1': '+', '+1': '+', '+': '+', 'plus': '+', '-1': '-', '-': '-', 'minus': '-'}.get(x, x)


def get_condition(row):
    labels = []
    if row.get('Normal', 0) == 1:
        labels.append('Normal')
    if row.get('CarboplatinTreated', 0) == 1:
        labels.append('Carboplatin')
    if row.get('GemcitabineTreated', 0) == 1:
        labels.append('Gemcitabine')
    if len(labels) == 0:
        return 'Unlabeled'
    if len(labels) == 1:
        return labels[0]
    return '+'.join(labels)


def distance_class(x):
    if pd.isna(x):
        return 'trans_or_unknown'
    if x <= 100000:
        return 'short_range'
    if x <= 1000000:
        return 'medium_range'
    return 'long_range'


def shape_class(row):
    d = row.get('genomic_distance_final', np.nan)
    w = row.get('interactor_width', np.nan)
    s = row.get('anchor_span', np.nan)
    if pd.isna(d):
        return 'trans_shape'
    if d <= 100000 and pd.notna(w) and w <= 2000:
        return 'compact_loop'
    if d <= 1000000 and pd.notna(s) and s <= 1000000:
        return 'local_arc'
    if d > 1000000 and pd.notna(s) and s > 1000000:
        return 'extended_loop'
    return 'broad_contact'

@st.cache_data(show_spinner=False)
def process_data(df):
    df = df.copy()
    for col in ['Feature_Start', 'Interactor_Start', 'Interactor_End', 'abs_distance', 'NofInts', 'MG1_SuppPairs', 'MG2_SuppPairs', 'MC1_SuppPairs', 'MC2_SuppPairs', 'MN1_SuppPairs', 'MN2_SuppPairs', 'Normal', 'CarboplatinTreated', 'GemcitabineTreated']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    if 'Feature_Chr' in df.columns:
        df['Feature_Chr'] = df['Feature_Chr'].apply(clean_chr)
    if 'Interactor_Chr' in df.columns:
        df['Interactor_Chr'] = df['Interactor_Chr'].apply(clean_chr)
    if 'Strand' in df.columns:
        df['Strand'] = df['Strand'].apply(clean_strand)
    else:
        df['Strand'] = 'unknown'

    if {'Interactor_Start', 'Interactor_End'}.issubset(df.columns):
        df['Interactor_Mid'] = ((df['Interactor_Start'] + df['Interactor_End']) / 2).round()
        df['interactor_width'] = (df['Interactor_End'] - df['Interactor_Start']).abs()
    else:
        df['Interactor_Mid'] = np.nan
        df['interactor_width'] = np.nan

    if {'Feature_Chr', 'Interactor_Chr'}.issubset(df.columns):
        df['interaction_type'] = np.where(df['Feature_Chr'] == df['Interactor_Chr'], 'cis', 'trans')
    else:
        df['interaction_type'] = 'unknown'

    if {'Feature_Start', 'Interactor_Mid'}.issubset(df.columns):
        df['computed_distance'] = np.where(df['interaction_type'] == 'cis', (df['Feature_Start'] - df['Interactor_Mid']).abs(), np.nan)
    else:
        df['computed_distance'] = np.nan

    if 'abs_distance' in df.columns:
        df['genomic_distance_final'] = pd.to_numeric(df['abs_distance'], errors='coerce').fillna(df['computed_distance'])
    else:
        df['genomic_distance_final'] = df['computed_distance']

    df.loc[df['interaction_type'] == 'trans', 'genomic_distance_final'] = np.nan
    df['Condition'] = df.apply(get_condition, axis=1)
    df['distance_class'] = df['genomic_distance_final'].apply(distance_class)
    df['anchor_span'] = (df['Feature_Start'] - df['Interactor_Mid']).abs() if {'Feature_Start', 'Interactor_Mid'}.issubset(df.columns) else np.nan

    bins = [-1, 100000, 500000, 1000000, 5000000, 10000000, np.inf]
    labels = ['0-100kb', '100kb-500kb', '500kb-1Mb', '1Mb-5Mb', '5Mb-10Mb', '>10Mb']
    df['distance_bin'] = pd.cut(df['genomic_distance_final'], bins=bins, labels=labels)

    for c in ['MG1_SuppPairs', 'MG2_SuppPairs', 'MC1_SuppPairs', 'MC2_SuppPairs', 'MN1_SuppPairs', 'MN2_SuppPairs']:
        if c not in df.columns:
            df[c] = 0

    df['interaction_strength_proxy'] = np.select(
        [df['Condition'].eq('Normal'), df['Condition'].eq('Carboplatin'), df['Condition'].eq('Gemcitabine')],
        [df['MN1_SuppPairs'] + df['MN2_SuppPairs'], df['MC1_SuppPairs'] + df['MC2_SuppPairs'], df['MG1_SuppPairs'] + df['MG2_SuppPairs']],
        default=df['NofInts'] if 'NofInts' in df.columns else np.nan
    )
    if 'NofInts' in df.columns:
        df['interaction_strength_proxy'] = pd.Series(df['interaction_strength_proxy']).fillna(df['NofInts'])

    df['strand_group'] = df['Strand'].fillna('unknown')
    df['shape_bucket'] = df.apply(shape_class, axis=1)
    df['shape_compactness'] = df['interactor_width'] / df['anchor_span'].replace(0, np.nan)
    df['shape_extension_ratio'] = df['anchor_span'] / df['interactor_width'].replace(0, np.nan)
    df['shape_contact_decay'] = df['interaction_strength_proxy'] / df['genomic_distance_final'].replace(0, np.nan)
    df['strand_distance_score'] = df['genomic_distance_final'] / df['interaction_strength_proxy'].replace(0, np.nan)
    df['log_distance'] = np.log10(df['genomic_distance_final'].replace(0, np.nan))
    return df

@st.cache_data(show_spinner=False)
def summarize(df):
    condition_summary = df.groupby('Condition', dropna=False).agg(
        total_interactions=('Condition', 'size'),
        cis_interactions=('interaction_type', lambda x: (x == 'cis').sum()),
        mean_distance=('genomic_distance_final', 'mean'),
        mean_strength=('interaction_strength_proxy', 'mean')
    ).reset_index()
    distance_summary = df.groupby(['Condition', 'distance_bin'], dropna=False).agg(
        interaction_count=('distance_bin', 'size'),
        mean_strength=('interaction_strength_proxy', 'mean')
    ).reset_index()
    strand_summary = df.groupby(['Condition', 'strand_group'], dropna=False).agg(
        interaction_count=('strand_group', 'size'),
        mean_distance=('genomic_distance_final', 'mean')
    ).reset_index()
    shape_summary = df.groupby(['Condition', 'shape_bucket'], dropna=False).agg(
        interaction_count=('shape_bucket', 'size'),
        mean_extension_ratio=('shape_extension_ratio', 'mean'),
        mean_decay=('shape_contact_decay', 'mean')
    ).reset_index()
    return condition_summary, distance_summary, strand_summary, shape_summary

st.markdown('<div class="hero"><div class="hero-title">🧬 MOLM-1 Chromatin Dashboard</div><div class="hero-sub">Fast, simpler, and more stable version with algorithm-based distance analysis, drug comparison, strand analysis, and shape analysis.</div></div>', unsafe_allow_html=True)

with st.sidebar:
    st.header('Upload & Controls')
    uploaded = st.file_uploader('Upload CSV / XLSX / XLS', type=['csv', 'xlsx', 'xls'])
    page = st.radio('Select page', ['Overview', 'Detailed analysis'])
    max_points = st.slider('Max scatter points', 1000, 20000, 5000, 1000)
    show_raw = st.toggle('Show raw preview', False)

if uploaded is None:
    st.info('Upload your file to begin.')
    st.stop()

try:
    file_bytes = uploaded.getvalue()
    raw_df = load_interaction_file(file_bytes, uploaded.name)
    df = process_data(raw_df)
    condition_summary, distance_summary, strand_summary, shape_summary = summarize(df)
except Exception as e:
    st.error(f'Loading failed: {e}')
    st.stop()

conditions = sorted(df['Condition'].dropna().unique().tolist())
selected_conditions = st.multiselect('Choose conditions', conditions, default=conditions)
fdf = df[df['Condition'].isin(selected_conditions)].copy()
condition_summary = condition_summary[condition_summary['Condition'].isin(selected_conditions)]
distance_summary = distance_summary[distance_summary['Condition'].isin(selected_conditions)]
strand_summary = strand_summary[strand_summary['Condition'].isin(selected_conditions)]
shape_summary = shape_summary[shape_summary['Condition'].isin(selected_conditions)]

m1, m2, m3, m4 = st.columns(4)
m1.metric('Total interactions', f"{len(fdf):,}")
m2.metric('Cis interactions', f"{(fdf['interaction_type'] == 'cis').sum():,}")
md = fdf['genomic_distance_final'].mean()
m3.metric('Mean distance', f"{md:,.0f} bp" if pd.notna(md) else 'NA')
ms = fdf['interaction_strength_proxy'].mean()
m4.metric('Mean strength', f"{ms:.2f}" if pd.notna(ms) else 'NA')

st.markdown('<div class="note"><b>Fix note:</b> This version is lighter and faster. It uses caching, fewer heavy visuals on first load, and a scatter-point limit to avoid blank screens and slow rendering with large files.</div>', unsafe_allow_html=True)
