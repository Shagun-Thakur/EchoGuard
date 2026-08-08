import sys
from pathlib import Path

import pandas as pd
import streamlit as st
import plotly.graph_objects as go

APP_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = APP_ROOT.parent
sys.path.append(str(APP_ROOT))

from constants import METRICS, COMPARISON_TABLE, ACCENT, ACCENT_SOFT, PLOTLY_TEMPLATE  # noqa: E402

st.title("🤖 Model Comparison")
st.write(
    "This section compares the reproduced Autoencoder baseline with the "
    "proposed handcrafted-feature Autoencoder and Normalizing Flow model."
)
st.markdown("---")

tab_table, tab_figures, tab_findings = st.tabs(
    ["📊 Table", "🖼️ Figures", "🔍 Key Findings"]
)

# -------------------------------------------------------------------------
# Table
# -------------------------------------------------------------------------
with tab_table:
    st.header("Performance Comparison")
    comparison = pd.DataFrame(COMPARISON_TABLE)
    st.dataframe(comparison, use_container_width=True, hide_index=True)

    def metric_card(label, value):
        st.markdown(
            f"""
            <div style="
                background-color: rgba(124, 58, 237, 0.06);
                border: 1px solid rgba(124, 58, 237, 0.15);
                border-radius: 12px;
                padding: 0.9rem 1rem;
            ">
                <div style="font-size: 0.75rem; font-weight: 600; color: #7C3AED;">
                    {label}
                </div>
                <div style="font-size: 0.95rem; font-weight: 700; margin-top: 0.3rem; line-height: 1.5;">
                    {value}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown("#### Best Results")
col1, col2, col3 = st.columns(3)
with col1:
    metric_card("Baseline AE", METRICS['Baseline Autoencoder'])
with col2:
    metric_card("Handcrafted AE", METRICS['Handcrafted Autoencoder'])
with col3:
    metric_card("Normalizing Flow", METRICS['Normalizing Flow'])
# -------------------------------------------------------------------------
# Figures (guarded — the original file referenced images that may not exist)
# -------------------------------------------------------------------------
with tab_figures:
    st.header("Experimental Results")
    st.write(
        "Drop the figures generated in your notebooks into a **figures/** "
        "folder at the project root (next to `streamlit_app/`) with the "
        "filenames below to have them appear here automatically."
    )

    figure_specs = [
        ("Baseline Autoencoder", "baseline_histogram.png"),
        ("Handcrafted Autoencoder", "handcrafted_autoencoder_histogram.png"),
        ("Normalizing Flow", "normalizing_flow_histogram.png"),
    ]
    FIGURES_DIR = PROJECT_ROOT/"results"/"figures"/"streamlit_images"
    cols = st.columns(len(figure_specs))
    for col, (label, filename) in zip(cols, figure_specs):
        with col:
            st.subheader(label)
            
            image_path = FIGURES_DIR / filename
            if image_path.is_file():
                st.image(str(image_path))
            else:
                st.info(f"`{filename}` not found yet.")

    st.subheader("ROC-AUC Boxplot Comparison")
    roc_path = PROJECT_ROOT/"results"/"figures"/"streamlit_images"/"auc_boxplot.png"
    if roc_path.is_file():
        st.image(str(roc_path))
    else:
        st.info("`figures/streamlit_images/auc_boxplot.png` not found yet.")

    st.subheader("pAUC Boxplot Comparison")
    pauc_path = PROJECT_ROOT/"results"/"figures"/"streamlit_images"/"pauc_boxplot.png"
    if pauc_path.is_file():
        st.image(str(pauc_path))
    else:
        st.info("`figures/streamlit_images/auc_boxplot.png` not found yet.")

# -------------------------------------------------------------------------
# Key findings
# -------------------------------------------------------------------------
with tab_findings:
    st.header("Key Findings")
    st.success(
        f"""
        Feature engineering dramatically improved anomaly detection
        performance. Using only **9 handcrafted features**, the
        Autoencoder improved from **{METRICS['Baseline Autoencoder']}**
        to **{METRICS['Handcrafted Autoencoder']} ROC-AUC and pAUC** while
        reducing training time from roughly **~25 minutes** to **~1 minute**.
        """
    )
    st.info(
        f"""
        Normalizing Flow achieved the highest ROC-AUC and pAUC
        (**{METRICS['Normalizing Flow']}**) while maintaining similar
        computational efficiency.

        For the evaluated **MIMII Pump id_00 (Channel 0)** dataset,
        **feature representation proved to be a more influential factor
        than the choice of anomaly detection model itself.**
        """
    )
