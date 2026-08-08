import sys
from pathlib import Path

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

APP_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = APP_ROOT.parent
sys.path.append(str(APP_ROOT))

from constants import ACCENT, ACCENT_SOFT, PLOTLY_TEMPLATE # noqa: E402

FIGURES_DIR =PROJECT_ROOT/"results"/"figures"/"streamlit_images"

st.markdown(
    """
<style>
/* -- typography -- */
@import url("https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:weight@400;600&family=IBM+Plex_Sans:ital,wght@0,400;0,600;1,400&display=swap');
html, body, [class*="css"] {font-family: 'IBM Plex Sans', sans-serif;}

/* --sectionlabel (eyebrow text) -- */
.section-label{
font-family: 'IBM Plex Mono', monospace;
font-size: 0.72rem;
font-weight: 600;
letter-spacing: 0.12em;
text-transform: uppercase;
color #7C3AED;
margin-bottom: 0.25rem;
}

/* --section heading --*/
.section-heading{
font-size: 1.25rem;
font-weight: 600;
color: #1E1B4B;
margin-top: 0;
margin-bottom: 0.6rem;
border-left: 3px solid #7C3AED;
padding-left: 0.75rem;
}

/* --body text inside sections --*/
.section-body{
font-size: 0.95rem;
line-height: 1.75;
color: #374151
}

/* --inline monospace code terms --*/
.mono {font-family: 'IBM Plex Mono', monospace; font-size: 0.88em; color: #5B21B6; }

/* -- key insight callout -- */
.insight-box{
background: linear-gradient(135deg, #F53FF 0%, #EDEFE 100%);
border: 1.5px solid #7C3AED;
border-radius: 10px;
padding: 1.2rem 1.5rem;
margin: 1rem 0;
}
.insight-box .insight-title{
font-family: 'IBM Plex Mono', monospace;
font-size: 0.8rem;
font-weight: 600;
color: #5B21B6;
letter-spacing: 0.08em;
font-wieght: 600;
color: #5B21B6;
letter-spacing: 0.08em;
text-transform: uppercase;
margin-bottom: 0.4rem;
}
.insight-box .insight-text{
font-size: 0.95rem;
color: #1E1B4B;
line-height: 1.65;
}

/* -- observation highlight -- */
.obs-box{
background: #FEFCE8;
border-left:4px solid #EAB308;
border-radius: 0 8px 8px 0;
padding: 0.9rem 1.2rem;
margin: 0.8rem 0;
font-size 0.93rem;
color: #713F12;
line-height: 1.6;
}

/* --figure placeholder -- */
.fig-placeholder{
background: #F8F7FF;
border: 1.5px dashed #A78BFA;
border-radius: 8px;
padding: 2.5rem 1rem;
text-align: centre;
color: #7C3AED;
font-family: 'IBM Plex Mono', monospace;
font-size: 0.82rem;
margin: 0.5rem 0 1rem 0;
}

/* --thin rule between sections --*/
.section-rule{
border: none;
border-top: 1px solid #EDE9FE;
margin: 2rem 0;
}

/* --flow node --*/
.flow-node{
backgorund #7C3aed;
color: white;
border-radius: 8px;
padding: 0.5rem 1rem;
text-align: center;
font-family: 'IBM Plex Mono', monospace;
font-size: 0.82rem;
font-weight: 600;
margin: 0.15rem auto;
width: fit-content;
min-width: 220px;
}
.flow-node.light{
background: #EDE9FE;
color #5B21B6;
}
.flow-node.terminal{
background: #1E1B4B;
color: white;
}
.flow-arrow{
text-align: center;
color: #A78BFA;
font-size: 1.1rem;
line-height: 1.2;
}

</style>
""", unsafe_allow_html= True
)

# Title
st.markdown(
    '<p class="section-label">Core Research Contribution</p>', unsafe_allow_html=True,
)
st.title("🔬 Evidence-Based Feature Selection Pipeline")
st.markdown(
    '<p class="section-body">The primary contribution of this research is not the Normalizing Flow model itself, but the development of an'
    '<strong>Evidence-Based Multi-Method Feature Selection Pipeline</strong>'
    'Instead of relying on a single statistical ranking algorithm, multiple complementray analyses from signal processing, statistics, visualization'
    'and machine learning were combined to construct  a compact, interpretable and discriminative handcrafted feature representation.</p>', unsafe_allow_html=True,
)

# Headline metrics
m1, m2, m3, m4 = st.columns(4)
m1.metric("Initial Features", "50")
m2.metric("Final Features", "9")
m3.metric("Dimensionality Reduction", "82%")
m4.metric("Selection Strategy", "Multi-Method Voting")

st.markdown('<hr class="section-rule">', unsafe_allow_html=True)

# S1 - Initial Feature Set
st.markdown('<p class="section-label">Section 1</p>', unsafe_allow_html=True)
st.markdown('<h3 class="section-heading">Initial Handcrafted Feature Set</h3>', unsafe_allow_html=True)
st.markdown(
    '<p class="section-body">50 handcrafted acoustic descriptors were extracted from each audio recording, spanning three complementary signal domians.'
    'unlike high-dimensional Log-Mel Spectrograms (320 features), these descriptors are physically interptable and directly reflect measurable machine behaviour.</p>',
    unsafe_allow_html=True,
)
c1, c2, c3 = st.columns(3)
with c1:
    with st.container(border=True):
        st.markdown("**⏱️ Time Domian**")
        st.markdown("- RMS Energy\n- Zero Crossing Rate")
    with c2:
        with st.container(border=True):
            st.markdown("**Spectral**")
            st.markdown("- Spectral Centroid\n- Mean\n- Std\n- Min\n- Max\n- Delta\n- IQR")
    with c3:
        with st.container(border=True):
            st.markdown("**Cepstral (MFCCs)**")
            st.markdown("- MFCC Mean (x20)\n- MFCC Standard Deviation (x20)")
st.markdown('<hr class="section-rule">', unsafe_allow_html=True)

# S2 - Pipeline Flow Diagram

st.markdown('<p class="section-body">No single analysis determined the final feature set.'
            'Seven independent analytical lenses were applied sequentially, each casting a vote.'
            'Only features consistently supported across multiple methods were retained.</p>', unsafe_allow_html=True)

flow_steps = [
    ("terminal", "🎙️ Raw Audio"),
    ("light", "⚙️ 50 Initial Features"),
    ("light", "📐 Signal Procesing & Redundancy"),
    ("light", "📊 PCA"),
    ("light", "🔍 t-SNE Visualization"),
    ("light", "📈 ANOVA"),
    ("light", "🤝 Mutual Information"),
    ("light", "🌲Random Forest"),
    ("light", "🗳️ Evidence-Based Voting"),
    ("terminal", "✅ Final Features"),
]
flow_html = '<div style="display:flex;flex-direction:column;align-items:center;gap:0;">'
for i, (style, label) in enumerate(flow_steps):
    flow_html += f'<div class="flow-node {style}">{label}</div>'
    if i < len(flow_steps) - 1:
        flow_html += f'<div class="flow-arrow"> ↓</div>'
    flow_html += "</div>"

_, flow_col, _ = st.columns([1, 2, 1]) 
with flow_col:
    st.markdown(flow_html, unsafe_allow_html=True)

st.markdown('<hr class = "section-rule">', unsafe_allow_html=True)

# Section 3 -8 in tabs

st.markdown('<p class="section-label">Sections 3 - 8</p>', unsafe_allow_html=True)
st.markdown('<h3 class="section-heading">Analytical Methods</h3>', unsafe_allow_html=True)
tab_sp, tab_pca, tab_tsne, tab_anova, tab_mi, tab_rf = st.tabs(
    [
        "📐 Signal Procesing & Redundancy",
        "📊 PCA",
        "🔍 t-SNE Visualization",
        "📈 ANOVA",
        "🤝 Mutual Information",
        "🌲Random Forest",
    ]
)

# S3
with tab_sp:
    st.markdown("#### Signal Processing & Redundancy Analysis")
    st.markdown('<p class="section-body">The first stage prioritised engineering intuition over statistical machinery.'
                'The goals were to remove highly correlated features, eliminate redundant information, preserve physically interpretable'
                'descriptors, and reduce unnecessary dimensionality before any learning-based analysis.<br><br>'
                'Highly correlated features carry duplicated information - retaining both increases noise without adding discriminative signal.</p>',
                unsafe_allow_html=True)
    corr_path = FIGURES_DIR/"feature_correlation_matrix.png"
    if corr_path.is_file():
        st.image(str(corr_path), caption="Correlation Heatmap")
    else:
        st.markdown(
            '<div class="fig-placeholder">📊Figure - Correlation Heatmap<br>'
            '<span style="opacity:0.6">feature_correlation_matrix.png</span></div>', unsafe_allow_html=True,
        )

# S4
with tab_pca:
    st.markdown("#### Prinicipal Component Analysis (PCA)")
    st.markdown('<p class="section-body">PCA identified features contributing most strongly to the overall variance of the dataset.'
                'Features with consistently high loading across the leading principal components received stronger evidence'
                'during the voting process.</p>', unsafe_allow_html=True)
    pca_path = FIGURES_DIR/"pca_variance.png"
    if pca_path.is_file():
        st.image(str(pca_path), caption="PCA Explained Variance")
    else: 
        st.markdown('<div class="fig-placeholder">📊 Figure - PCA Explained Variance Plot<br>'
                    '<span style = "opacity:0.6">pca_explained_variance.png</span></div>', unsafe_allow_html=True)

# S5
with tab_tsne:
    st.markdown("#### t-SNE Visualization")
    st.markdown('<p class="section-body">t-SNE projected the full 50-feature space into two dimensions.'
                'The image below shows the t-SNE projection comparison of "All raw feature and 9-selected features".', unsafe_allow_html=True)
    tsne_path = FIGURES_DIR / "tsne_comparison.png"
    if tsne_path.is_file():
        st.image(str(tsne_path), caption = 't-SNE Projection')
    else:
        st.markdown(
            '<div class="fig-placeholder">📊 Figure - t-SNE Projection<br>'
            '<span style=opacity:0.6">tsne_comparison.png</span></div>',
            unsafe_allow_html=True,
        )

# S6
with tab_anova:
    st.markdown("#### ANOVA Statistical Analysis")
    st.markdown(
        '<p class="section-body">One-way ANOVA evaluated whether feature distributions'
        'differed significantly between normal and abnormal recordings. Features with'
        'lower p-values provided stronger statistical evidence of class discrimination.</p>',
        unsafe_allow_html=True,
    )
    anova_path = FIGURES_DIR/"anova_fScore.png"
    if anova_path.is_file():
        st.image(str(anova_path), caption="ANOVA Ranking")
    else:
        st.markdown(
            'div class="fig-placeholder">📊 Figure - ANOVA Ranking Plot<br>'
            '<span style="opacity:0.6">anova_ranking.png</span></div>',
            unsafe_allow_html=True,
        )

# S7
with tab_mi:
    st.markdown("#### Mutual Information")
    st.markdown('<p class="section-body">Mutual Information measures how informative each'
                'feature is with respect to machine condition. Higher MI indicates a stronger'
                'statistical dependency between the feature and the normal/abnormal label,'
                'reflecting genuine discriminative capability independent of the distributional'
                'assumptions.</p>', unsafe_allow_html=True)
    mi_path = FIGURES_DIR/"mutual_information.png"
    if mi_path.is_file():
        st.image(str(mi_path), caption = "Mutual Information")
    else:
        st.markdown(
            '<div class="fig-placefolder">📊 Figure - Mutual Information <br>'
            '<span style="opacity:0.6">mutual_information.png</span></div>',
            unsafe_allow_html=True,
        )

# S8
with tab_rf:
    st.markdown("#### Random Forest Feature Importance")
    st.markdown(
        '<p class="section-body">A Random Forest classifier was trained solely for'
        'feature ranking - it was <em>not</em>used as the anomaly detection model.'
        'Its importance scores served as an independent, non-parametric source of'
        'evidence during voting, capturing non-linear feature interactions that'
        'statistical tests may miss.</p>', unsafe_allow_html=True,
    )
    rf_path = FIGURES_DIR/"rf_importance.png"
    if rf_path.is_file():
        st.image(str(rf_path), caption="Random Forest Feature Importance")
    else:
        st.markdown(
            '<div class="fig-placeholder">📊 Figure - Random Forest Feature Importance<br>'
            '<span style="opacity:0.6">rf_importance.png</span></div>', unsafe_allow_html=True,
        )
st.markdown('<hr class="section-rule">', unsafe_allow_html=True)

# S9 - Evidence-Based Voting
st.markdown('<p class="section-label">Section 9</p>', unsafe_allow_html=True)
st.markdown(
    '<h3 class="section-heading">Evidence-Based Voting</h3>', unsafe_allow_html=True,
)
st.markdown(
    '<p class="section-body">No individual analysis determined the final feature set. '
    'Every feature accumulated evidence from multiple independent analyses. Only '
    'features consistently supported across methods were retained.</p>',
    unsafe_allow_html=True,
)
 
# Interactive voting diagram using Plotly Sankey-style bar
methods = [
    "Signal Processing",
    "Correlation Analysis",
    "PCA",
    "Scatter Plot",
    "t-SNE",
    "ANOVA",
    "Mutual Information",
    "Random Forest",
]
 
#  vote counts per method for the 9 selected features 
selected_features = ["RMS","MFCC1 Mean","MFCC1 Std","MFCC2 Mean","MFCC2 Std","MFCC5 Std","MFCC8 Mean","Centroid Std","Centroid IQR"]
 
np.random.seed(42)
vote_matrix = np.array([
    [1, 1, 1, 1, 1, 0, 0, 1, 1],  # Signal Processing
    [0, 1, 0, 1, 0, 0, 1, 1, 1],  # Feature Space
    [1, 1, 0, 0, 1, 1, 1, 0, 0],  # ANOVA
    [1, 1, 1, 0, 1, 1, 1, 1, 1],  # MI
    [1, 1, 1, 0, 1, 1, 1, 0, 0],  # RF
])
 
vote_totals = vote_matrix.sum(axis=0)
 
fig_votes = go.Figure()
fig_votes.add_trace(go.Bar(
    x=selected_features,
    y=vote_totals,
    marker_color=[ACCENT if v >= 3 else ACCENT_SOFT for v in vote_totals],
    text=vote_totals,
    textposition="outside",
))
fig_votes.update_layout(
    template=PLOTLY_TEMPLATE,
    title="Evidence Votes Accumulated per Selected Feature",
    yaxis_title="Methods Supporting Feature",
    yaxis=dict(range=[0, len(methods) + 1], dtick=1),
    height=380,
    margin=dict(l=10, r=10, t=50, b=10),
)
st.plotly_chart(fig_votes, use_container_width=True)
st.caption(
    "Bar heights show how many independent methods supported each feature. "
    "Darker bars received support from 3+ methods."
)
 
st.markdown('<hr class="section-rule">', unsafe_allow_html=True)
 

# S10 — Final Feature Table

st.markdown('<p class="section-label">Section 10</p>', unsafe_allow_html=True)
st.markdown(
    '<h3 class="section-heading">Final Selected Feature Representation</h3>',
    unsafe_allow_html=True,
)
 
final_features = pd.DataFrame(
    {
        "ID": ["F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "F9"],
        "Feature": [
            "RMS Energy",
            "MFCC-1 Mean",
            "MFCC-1 Std",
            "MFCC-2 Mean",
            "MFCC-2 Std",
            "MFCC-5 Std",
            "MFCC-8 Mean",
            "Spectral Centroid Std",
            "Spectral Centroid IQR",
        ],
        "Primary Supporting Evidence": [
            "Signal Processing + Redundancy",
            "ANOVA + MI + Random Forest",
            "ANOVA + MI",
            "Scatter Plot + ANOVA",
            "ANOVA + MI",
            "MI + Random Forest",
            "PCA + MI",
            "Signal Processing + PCA",
            "Redundancy + Random Forest",
        ],
        "Physical Interpretation": [
            "Overall acoustic energy level",
            "Dominant spectral envelope shape",
            "Temporal variability of spectral envelope",
            "Strong visual class separation",
            "Temporal variation of lower-order cepstral info",
            "Variability in higher-order spectral information",
            "Fine spectral texture characteristics",
            "Frequency instability over time",
            "Robust dispersion of dominant frequency",
        ],
    }
)
 
st.dataframe(final_features, use_container_width=True, hide_index=True)
 
st.markdown('<hr class="section-rule">', unsafe_allow_html=True)
 

# S12 — Experimental Validation

st.markdown('<p class="section-label">Section 11</p>', unsafe_allow_html=True)
st.markdown(
    '<h3 class="section-heading">Experimental Validation</h3>',
    unsafe_allow_html=True,
)
 
from constants import METRICS  # noqa: E402
 
validation_df = pd.DataFrame(
    {
        "Feature Representation": [
            "Log-Mel Spectrogram (Baseline)",
            "Evidence-Based Handcrafted Features",
        ],
        "# Features": ["320 (high-dimensional)", "9 (compact)"],
        "Autoencoder ROC-AUC": [
            f"{METRICS['Baseline Autoencoder']}",
            f"{METRICS['Handcrafted Autoencoder']}",
        ],
        "Normalizing Flow ROC-AUC": [
            "—",
            f"{METRICS['Normalizing Flow']}",
        ],
    }
)
st.dataframe(validation_df, use_container_width=True, hide_index=True)
 
# Visual comparison bar chart
fig_compare = go.Figure()
fig_compare.add_trace(go.Bar(
    name="Baseline AE (Log-Mel)",
    x=["Autoencoder"],
    y=[METRICS["Baseline Autoencoder"]],
    marker_color="#CBD5E1",
))
fig_compare.add_trace(go.Bar(
    name="Handcrafted AE",
    x=["Autoencoder"],
    y=[METRICS["Handcrafted Autoencoder"]],
    marker_color=ACCENT_SOFT,
))
fig_compare.add_trace(go.Bar(
    name="Normalizing Flow",
    x=["Normalizing Flow"],
    y=[METRICS["Normalizing Flow"]],
    marker_color=ACCENT,
))
fig_compare.update_layout(
    template=PLOTLY_TEMPLATE,
    title="ROC-AUC: Feature Representation Impact",
    yaxis_title="ROC-AUC",
    yaxis_range=[0, 1.08],
    barmode="group",
    height=380,
    margin=dict(l=10, r=10, t=50, b=10),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)
st.plotly_chart(fig_compare, use_container_width=True)
 
st.markdown(
    '<p class="section-body">The compact handcrafted representation substantially '
    'improved detection performance while reducing training time from '
    '<span class="mono">20–25 min → 0.5–1 min</span> and computational complexity '
    'by <span class="mono">97%</span> (320 → 9 input features).</p>',
    unsafe_allow_html=True,
)
 
st.markdown('<hr class="section-rule">', unsafe_allow_html=True)
 
# ─────────────────────────────────────────────
# Final Takeaway
# ─────────────────────────────────────────────
st.markdown(
    """
    <div class="insight-box">
        <div class="insight-title">⭐ Key Research Insight</div>
        <div class="insight-text">
            The largest performance gain in EchoGuard does not originate from replacing
            the Autoencoder with a Normalizing Flow. Instead, the most significant
            improvement comes from constructing a <strong>compact, evidence-based
            handcrafted feature representation</strong> — reducing 50 candidate
            descriptors to 9 through multi-method voting.<br><br>
            The Normalizing Flow then further improves performance by modelling the
            probability distribution of this carefully selected feature space,
            rather than relying on reconstruction error alone.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)