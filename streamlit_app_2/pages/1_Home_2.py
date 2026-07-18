import sys
from pathlib import Path

import streamlit as st

# Make constants.py importable regardless of which page is running
APP_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(APP_ROOT))

from constants import METRICS, DATASET_INFO, PIPELINE_STEPS  # noqa: E402

# -------------------------------------------------------------------------
# Hero section
# -------------------------------------------------------------------------
st.title("🎧 EchoGuard")
st.subheader(
    "Exploring Unsupervised Machine Audio Anomaly Detection — "
    "From Signal Analysis to Normalizing Flow"
)

st.markdown(
    """
Welcome to **EchoGuard**, an interactive research dashboard built for an
undergraduate research project on **unsupervised machine audio anomaly
detection**. Rather than only exposing a prediction tool, this dashboard
walks through the full research workflow — signal analysis, feature
engineering, baseline reproduction, Normalizing Flow modeling, and
comparative evaluation.
"""
)

metric_cols = st.columns(3)
metric_cols[0].metric("Baseline Autoencoder ROC-AUC", f"{METRICS['Baseline Autoencoder']:.3f}")
metric_cols[1].metric(
    "Handcrafted Autoencoder ROC-AUC",
    f"{METRICS['Handcrafted Autoencoder']:.3f}",
    delta=f"+{METRICS['Handcrafted Autoencoder'] - METRICS['Baseline Autoencoder']:.3f} vs baseline",
)
metric_cols[2].metric(
    "Normalizing Flow ROC-AUC",
    f"{METRICS['Normalizing Flow']:.4f}",
    delta=f"+{METRICS['Normalizing Flow'] - METRICS['Baseline Autoencoder']:.3f} vs baseline",
)

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
            Investigate whether **Normalizing Flows** can improve
            unsupervised machine audio anomaly detection compared to the
            original Autoencoder baseline proposed for the MIMII dataset.
            """
        )

    st.markdown("#### Goals")
    goal_col1, goal_col2 = st.columns(2)
    with goal_col1:
        st.markdown(
            """
            - ✔ Reproduce the original MIMII Autoencoder baseline
            - ✔ Understand machine audio through signal analysis
            - ✔ Identify informative acoustic features
            - ✔ Build a handcrafted-feature Autoencoder
            """
        )
    with goal_col2:
        st.markdown(
            """
            - ✔ Implement a Normalizing Flow model
            - ✔ Compare reconstruction vs. density estimation
            - ✔ Develop an interactive Streamlit dashboard
            - ✔ Improve interpretability of anomaly detection
            """
        )

with tab_dataset:
    st.markdown("#### MIMII Pump Dataset")
    d1, d2, d3 = st.columns(3)
    d1.metric("Dataset", DATASET_INFO["Dataset"])
    d1.metric("Machine Type", DATASET_INFO["Machine Type"])
    d2.metric("Machine ID", DATASET_INFO["Machine ID"])
    d2.metric("Channel", DATASET_INFO["Channel"])
    d3.metric("Training Recordings", DATASET_INFO["Training Recordings"])
    d3.metric("Evaluation Recordings", DATASET_INFO["Evaluation Recordings"])
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
    st.success(f"**Baseline Autoencoder (Log-Mel):** ROC-AUC = **{METRICS['Baseline Autoencoder']:.3f}**")
    st.success(f"**Handcrafted Feature Autoencoder:** ROC-AUC = **{METRICS['Handcrafted Autoencoder']:.3f}**")
    st.success(f"**Normalizing Flow:** ROC-AUC = **{METRICS['Normalizing Flow']:.4f}**")
    st.info(
        """
        The experiments indicate that **feature representation had a much
        larger impact on anomaly detection performance than changing the
        model architecture** for the evaluated MIMII Pump id_00, Channel 0
        dataset.
        """
    )

st.markdown("---")

# -------------------------------------------------------------------------
# Quick navigation — uses st.page_link (works with st.navigation) instead of
# the previous st.switch_page buttons, so pages can also be Ctrl/Cmd-clicked
# to open in a new tab.
# -------------------------------------------------------------------------
st.markdown("### 🚀 Explore the Dashboard")
nav_col1, nav_col2, nav_col3 = st.columns(3)
with nav_col1:
    st.page_link("pages/2_Dataset_Explorer_2.py", label="Dataset Explorer", icon="📂")
with nav_col2:
    st.page_link("pages/3_Model_Comparison_2.py", label="Model Comparison", icon="📊")
with nav_col3:
    st.page_link("pages/4_Interactive_Prediction_2.py", label="Interactive Prediction", icon="🎯")
