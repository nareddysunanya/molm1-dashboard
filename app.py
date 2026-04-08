import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title='MOLM-1 Chromatin Intelligence Dashboard',
    page_icon='🧬',
    layout='wide',
    initial_sidebar_state='expanded'
)

st.markdown('''
<style>
:root {
    --bg1:#f6f8ff;
    --bg2:#fff6fb;
    --txt:#1e293b;
    --muted:#64748b;
    --violet:#7c3aed;
    --blue:#2563eb;
    --pink:#ec4899;
    --teal:#0891b2;
    --green:#059669;
    --orange:#ea580c;
    --card:rgba(255,255,255,0.86);
}
.stApp {
    background:
        radial-gradient(circle at 0% 0%, rgba(124,58,237,0.12), transparent 28%),
        radial-gradient(circle at 100% 0%, rgba(37,99,235,0.11), transparent 28%),
        radial-gradient(circle at 100% 100%, rgba(236,72,153,0.10), transparent 26%),
        linear-gradient(180deg, var(--bg1), var(--bg2));
}
.block-container {max-width: 1480px; padding-top: 1rem; padding-bottom: 2rem;}
.hero {
    background: linear-gradient(135deg, rgba(124,58,237,0.15), rgba(37,99,235,0.11), rgba(236,72,153,0.12));
    border: 1px solid rgba(124,58,237,0.15);
    border-radius: 28px;
    padding: 1.35rem 1.45rem;
    box-shadow: 0 16px 34px rgba(15,23,42,0.09);
    margin-bottom: 1rem;
}
.hero h1 {margin: 0; color: var(--txt); font-size: 2.35rem; font-weight: 850;}
.hero p {margin: 0.45rem 0 0 0; color: var(--muted); font-size: 1rem;}
.info-card {
    background: var(--card);
    backdrop-filter: blur(10px);
    border-radius: 22px;
    border: 1px solid rgba(255,255,255,0.8);
    box-shadow: 0 12px 28px rgba(15,23,42,0.07);
    padding: 1rem;
}
.soft-note {
    background: linear-gradient(135deg, #edf6ff, #f8efff);
    border-left: 5px solid var(--violet);
    border-radius: 16px;
    padding: 0.95rem 1rem;
    color: #334155;
    margin: 0.7rem 0 1rem 0;
}
.minihead {font-size: 1.16rem; font-weight: 800; color: var(--txt); margin-bottom: 0.45rem;}
.smalltext {color: var(--muted); font-size: 0.94rem;}
[data-testid="stMetric"] {
    background: linear-gradient(135deg, rgba(255,255,255,0.97), rgba(247,249,255,0.96));
    border: 1px solid rgba(99,102,241,0.11);
    border-radius: 18px;
    padding: 0.85rem 0.95rem;
    box-shadow: 0 8px 22px rgba(15,23,42,0.06);
}
[data-testid="stMetricLabel"] {font-weight: 700;}
[data-testid="stTabs"] button[role="tab"] {
    border-radius: 999px;
    padding: 0.42rem 1rem;
}
</style>
''', unsafe_allow_html=True)


def load_interaction_file(uploaded_file):
    ext = os.path.splitext(uploaded_file.name)[1].lower()
    if ext == '.csv':
        df = pd.read_csv(uploaded_file)
    elif ext in ['.xlsx', '.xls']:
        df = pd.read_excel(uploaded_file)
    else:
        raise ValueError('Upload CSV, XLSX, or XLS only.')
    df = df.loc[:, ~df.columns.astype(str).str.contains('^Unnamed')]
    if '' in df.columns:
        df = df.drop(columns=[''])
    return df


def clean_chr(val):
    if pd.isna(val):
        return np.nan
    val = str(val).strip().lower().replace('chromosome', '').replace('chr', '').strip()
    return f'chr{val}' if val else np.nan


def clean_strand(val):
    if pd.isna(val):
        return 'unknown'
    val = str(val).strip().lower()
    mapping = {'+': '+', 'plus': '+', '+1': '+', '1': '+', '-': '-', '-1': '-', 'minus': '-'}
    return mapping.get(val, val)


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


def distance_bucket(distance):
    if pd.isna(distance):
        return 'trans_or_unknown'
    if distance <= 100000:
        return 'short_range'
    if distance <= 1000000:
        return 'medium_range'
    return 'long_range'


def shape_bucket(row):
    d = row.get('genomic_distance_final', np.nan)
    width = row.get('interactor_width', np.nan)
    span = row.get('anchor_span', np.nan)
    if pd.isna(d):
        return 'trans_shape'
    if d <= 100000 and pd.notna(width) and width <= 2000:
        return 'compact_loop'
    if d <= 1000000 and pd.notna(span) and span <= 1000000:
        return 'local_arc'
    if d > 1000000 and pd.notna(span) and span > 1000000:
        return 'extended_loop'
    return 'broad_contact'


def process_data(df):
    df = df.copy()
    text_cols = ['RefSeqName', 'TranscriptName', 'Feature_Chr', 'Interactor_Chr', 'InteractorName', 'InteractorID', 'Strand', 'IntGroup']
    num_cols = [
        'Feature_Start', 'Interactor_Start', 'Interactor_End', 'abs_distance', 'distance', 'NofInts',
        'MG1_SuppPairs', 'MG2_SuppPairs', 'MC1_SuppPairs', 'MC2_SuppPairs', 'MN1_SuppPairs', 'MN2_SuppPairs',
        'MG1_p_value', 'MG2_p_value', 'MC1_p_value', 'MC2_p_value', 'MN1_p_value', 'MN2_p_value',
        'Normal', 'CarboplatinTreated', 'GemcitabineTreated'
    ]

    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    if 'Feature_Chr' in df.columns:
        df['Feature_Chr'] = df['Feature_Chr'].apply(clean_chr)
    if 'Interactor_Chr' in df.columns:
        df['Interactor_Chr'] = df['Interactor_Chr'].apply(clean_chr)
    df['Strand'] = df['Strand'].apply(clean_strand) if 'Strand' in df.columns else 'unknown'

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
    df['distance_class'] = df['genomic_distance_final'].apply(distance_bucket)

    bins = [-1, 100000, 500000, 1000000, 5000000, 10000000, np.inf]
    labels = ['0-100kb', '100kb-500kb', '500kb-1Mb', '1Mb-5Mb', '5Mb-10Mb', '>10Mb']
    df['distance_bin'] = pd.cut(df['genomic_distance_final'], bins=bins, labels=labels)

    for col in ['MG1_SuppPairs', 'MG2_SuppPairs', 'MC1_SuppPairs', 'MC2_SuppPairs', 'MN1_SuppPairs', 'MN2_SuppPairs']:
        if col not in df.columns:
            df[col] = 0

    df['Gem_Support'] = df['MG1_SuppPairs'] + df['MG2_SuppPairs']
    df['Carbo_Support'] = df['MC1_SuppPairs'] + df['MC2_SuppPairs']
    df['Normal_Support'] = df['MN1_SuppPairs'] + df['MN2_SuppPairs']

    def select_support(row):
        if row['Condition'] == 'Normal':
            return row['Normal_Support']
        if row['Condition'] == 'Carboplatin':
            return row['Carbo_Support']
        if row['Condition'] == 'Gemcitabine':
            return row['Gem_Support']
        return row.get('NofInts', np.nan)

    df['interaction_strength_proxy'] = df.apply(select_support, axis=1)
    if 'NofInts' in df.columns:
        df['interaction_strength_proxy'] = df['interaction_strength_proxy'].fillna(df['NofInts'])

    def select_pvalue(row):
        vals = []
        if row['Condition'] == 'Normal':
            vals = [row.get('MN1_p_value', np.nan), row.get('MN2_p_value', np.nan)]
        elif row['Condition'] == 'Carboplatin':
            vals = [row.get('MC1_p_value', np.nan), row.get('MC2_p_value', np.nan)]
        elif row['Condition'] == 'Gemcitabine':
            vals = [row.get('MG1_p_value', np.nan), row.get('MG2_p_value', np.nan)]
        vals = [v for v in vals if pd.notna(v)]
        return np.mean(vals) if vals else np.nan

    df['active_p_value'] = df.apply(select_pvalue, axis=1)
    df['strand_group'] = df['Strand'].fillna('unknown')
    df['anchor_span'] = (df['Feature_Start'] - df['Interactor_Mid']).abs() if {'Feature_Start', 'Interactor_Mid'}.issubset(df.columns) else np.nan
    df['support_density'] = df['interaction_strength_proxy'] / df['interactor_width'].replace(0, np.nan)
    df['distance_weighted_signal'] = df['interaction_strength_proxy'] / df['genomic_distance_final'].replace(0, np.nan)
    df['strand_distance_score'] = df['genomic_distance_final'] / df['interaction_strength_proxy'].replace(0, np.nan)
    df['shape_compactness'] = df['interactor_width'] / df['anchor_span'].replace(0, np.nan)
    df['shape_extension_ratio'] = df['anchor_span'] / df['interactor_width'].replace(0, np.nan)
    df['shape_contact_decay'] = df['interaction_strength_proxy'] / df['genomic_distance_final'].replace(0, np.nan)
    df['shape_bucket'] = df.apply(shape_bucket, axis=1)
    df['log_distance'] = np.log10(df['genomic_distance_final'].replace(0, np.nan))
    df['log_strength'] = np.log10(df['interaction_strength_proxy'].replace(0, np.nan))
    return df


def build_tables(df):
    condition_summary = df.groupby('Condition', dropna=False).agg(
        total_interactions=('Condition', 'size'),
        cis_interactions=('interaction_type', lambda x: (x == 'cis').sum()),
        trans_interactions=('interaction_type', lambda x: (x == 'trans').sum()),
        mean_distance=('genomic_distance_final', 'mean'),
        median_distance=('genomic_distance_final', 'median'),
        mean_strength=('interaction_strength_proxy', 'mean'),
        mean_p_value=('active_p_value', 'mean')
    ).reset_index()

    distance_summary = df

