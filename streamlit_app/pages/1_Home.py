import sys
from pathlib import Path

import streamlit as st

# Make constants.py importable regardless of which page is running
APP_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(APP_ROOT))

from constants import METRICS, DATASET_INFO, PIPELINE_STEPS  # noqa: E402

def metric_card(label, value):
    st.markdown(
        f"""
        <div style="
            background-color: rgba(124, 58, 237, 0.06);
            border: 1px solid rgba(124, 58, 237, 0.15);
            border-radius: 12px;
            padding: 0.9rem 1rem;
        ">
            <div style="font-size: 0.72rem; font-weight: 600; color: #7C3AED; line-height:1.4;">
                {label}
            </div>
            <div style="font-size: 0.95rem; font-weight: 700; margin-top: 0.3rem; line-height: 1.5;">
                {value}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# -------------------------------------------------------------------------
# Hero section
# -------------------------------------------------------------------------
st.title("🎧 EchoGuard")
st.subheader(
    "Evidence-Based Feature Engineering for Unsupervised Machine Audio Anomaly Detection"
)

st.markdown(
    """
Welcome to **EchoGuard**, an interactive research dashboard for an
undergraduate research project on **unsupervised machine audio anomaly
detection**. The core contribution of this project is an
**Evidence-Based Multi-Method Feature Selection Pipeline** that reduces
50 handcrafted acoustic descriptors to 9 highly discriminative features —
achieving a dramatic improvement in detection performance while remaining
physically interpretable.
"""
)

metric_cols = st.columns(3)
with metric_cols[0]:
    metric_card("Baseline Autoencoder ROC-AUC and pAUC (%)", METRICS['Baseline Autoencoder'])
with metric_cols[1]:
    metric_card("Handcrafted Autoencoder ROC-AUC and pAUC (%)", METRICS['Handcrafted Autoencoder'])
with metric_cols[2]:
    metric_card("Normalizing Flow ROC-AUC and pAUC (%)", METRICS['Normalizing Flow'])

st.markdown("---")

# -------------------------------------------------------------------------
# Tabbed content — keeps a long single-scroll page organized and interactive
# -------------------------------------------------------------------------
tab_overview, tab_dataset, tab_pipeline, tab_findings = st.tabs(
    ["🎯 Overview", "📂 Dataset", "🔬 Pipeline", "🏆 Key Findings"]
)

with tab_overview:
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Problem Statement")
        st.write(
            """
            Industrial machines often develop faults gradually. Detecting
            these faults early can reduce maintenance costs and prevent
            unexpected failures. Unlike supervised learning, anomaly
            detection assumes only **normal operating sounds** are
            available during training — the model has to learn what
            "normal" looks like and flag anything that deviates from it.
            """
        )

    with col2:
        st.markdown("#### Objective")
        st.write(
            """
                Develop an **Evidence-Based Multi-Method Feature Selection Pipeline**
                that constructs a compact, interpretable, and discriminative handcrafted
                feature representation for unsupervised machine audio anomaly detection —
                and evaluate whether this representation improves upon the original
                Log-Mel Spectrogram baseline.
            """
        )

    st.markdown("#### Goals")
    goal_col1, goal_col2 = st.columns(2)
    with goal_col1:
        st.markdown(
            """
            - ✔ Reproduce the original MIMII Autoencoder baseline
            - ✔ Understand machine audio through signal processing
            - ✔ Extract 50 handcrafted acoustic descriptors
            - ✔ Develop an evidence-based feature selection pipeline
            """
        )
    with goal_col2:
        st.markdown(
            """
            - ✔ Reduce 50 features to 9 through multi-method voting
            - ✔ Validate the pipeline through experimental comparison
            - ✔ Implement a Normalizing Flow as a secondary contribution
            - ✔ Build an interpretable, deployable research dashboard
            """
        )

with tab_dataset:
    st.markdown("#### MIMII Pump Dataset")
    d1, d2, d3 = st.columns(3)
    with d1:
        metric_card("Dataset", DATASET_INFO["Dataset"])
        st.markdown("<div style='margin-top:0.5rem'></div>", unsafe_allow_html=True)
        metric_card("Machine Type", DATASET_INFO["Machine Type"])
    with d2:
        metric_card("Machine ID", DATASET_INFO["Machine ID"])
        st.markdown("<div style='margin-top:0.5rem'></div>", unsafe_allow_html=True)
        metric_card("Channel", DATASET_INFO["Channel"])
    with d3:
        metric_card("Training Recordings", DATASET_INFO["Training Recordings"])
        st.markdown("<div style='margin-top:0.5rem'></div>", unsafe_allow_html=True)
        metric_card("Evaluation Recordings", DATASET_INFO["Evaluation Recordings"])
    st.caption(
        "Head to **Dataset Explorer** in the sidebar to listen to individual "
        "recordings and inspect their waveform, spectrogram, and acoustic features."
    )

with tab_pipeline:
    st.markdown("#### Research Workflow")
    st.caption("Each stage feeds into the next — from raw audio to an interactive prediction.")

    step_cols = st.columns(len(PIPELINE_STEPS) * 2 - 1)
    for i, (icon, label) in enumerate(PIPELINE_STEPS):
        col_index = i * 2
        with step_cols[col_index]:
            st.markdown(
                f"""
                <div style="text-align:center;">
                    <div style="font-size:1.8rem;">{icon}</div>
                    <div style="font-size:0.78rem; font-weight:600;">{label}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        if col_index + 1 < len(step_cols):
            with step_cols[col_index + 1]:
                st.markdown(
                    "<div style='text-align:center; font-size:1.4rem; color:#94A3B8;'>&rarr;</div>",
                    unsafe_allow_html=True,
                )

with tab_findings:
    st.success(f"**Baseline Autoencoder (Log-Mel):** ROC-AUC and pAUC = **{METRICS['Baseline Autoencoder']}**")
    st.success(f"**Handcrafted Feature Autoencoder:** ROC-AUC and pAUC= **{METRICS['Handcrafted Autoencoder']}**")
    st.success(f"**Normalizing Flow:** ROC-AUC and pAUC = **{METRICS['Normalizing Flow']}**")
    st.info(
        """
            The experiments confirm that **the Evidence-Based Feature Selection
            Pipeline is the primary driver of performance improvement**. Reducing
            50 features to 9 through multi-method voting improved ROC-AUC from
            the baseline while cutting training time by ~96%. The Normalizing
            Flow provides a further incremental gain on top of this foundation.
        """
    )

st.markdown("---")

# Quick navigation 
st.markdown("### 🚀 Explore the Dashboard")
nav_col1, nav_col2, nav_col3, nav_col4 = st.columns(4)
with nav_col1:
    st.page_link("pages/2_FeatureSelection_Pipeline.py", label="Feature Selection Pipeline", icon="🔬")
with nav_col2:
    st.page_link("pages/3_Dataset_Explorer.py", label="Dataset Explorer", icon="📂")
with nav_col3:
    st.page_link("pages/4_Model_Comparison.py", label="Model Comparison", icon="📊")
with nav_col4:
    st.page_link("pages/5_Interactive_Prediction.py", label="Interactive Prediction", icon="🎯")
