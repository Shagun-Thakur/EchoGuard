import streamlit as st

# -------------------------------------------------------------------------
# Page configuration (must be the first Streamlit call)
# -------------------------------------------------------------------------
st.set_page_config(
    page_title="EchoGuard",
    page_icon="🎧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------------------------------------------------------------------------
# Global styling
# -------------------------------------------------------------------------
# A small shared stylesheet so every page (accessed through st.navigation)
# gets the same look without repeating CSS in each file.
st.markdown(
    """
    <style>
    .block-container {padding-top: 2rem; padding-bottom: 3rem;}
    [data-testid="stMetric"] {
        background-color: rgba(124, 58, 237, 0.06);
        border: 1px solid rgba(124, 58, 237, 0.15);
        border-radius: 12px;
        padding: 0.9rem 1rem 0.6rem 1rem;
    }
    [data-testid="stMetricLabel"] { font-weight: 600; font-size: 0.50rem !important;}
    section[data-testid="stSidebar"] {
        border-right: 1px solid rgba(124, 58, 237, 0.15);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------------------------------------------------------
# Sidebar branding
# -------------------------------------------------------------------------

with st.sidebar:
    st.markdown("## 🎧 EchoGuard")
    st.caption("Unsupervised Machine Audio Anomaly Detection")
    st.markdown("---")

home = st.Page("pages/1_Home.py", title="Home", icon="🏠", default=True)
feature_selection_pipeline = st.Page(
    "pages/2_FeatureSelection_Pipeline.py", title="Feature Selection Pipeline", icon="🔬"
)
dataset_explorer = st.Page(
    "pages/3_Dataset_Explorer.py", title="Dataset Explorer", icon="📂"
)
model_comparison = st.Page(
    "pages/4_Model_Comparison.py", title="Model Comparison", icon="📊"
)
interactive_prediction = st.Page(
    "pages/5_Interactive_Prediction.py", title="Interactive Prediction", icon="🎯"
)

pg = st.navigation(
    [home, feature_selection_pipeline, dataset_explorer, model_comparison, interactive_prediction]
)

with st.sidebar:
    st.markdown("---")
    st.caption("Developed for the EchoGuard research project.")

pg.run()
