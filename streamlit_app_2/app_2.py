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
    [data-testid="stMetricLabel"] { font-weight: 600; }
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
# NOTE (structural fix): the previous version defined the Home page in TWO
# places — inline in this file AND in pages/1_Home.py — which made Streamlit
# show two different "home" entries in the sidebar and made the two copies
# drift out of sync (different ROC-AUC numbers, different wording). Using
# st.navigation() below gives this app a single, explicit list of pages, and
# all shared numbers now live in constants.py instead of being duplicated.
with st.sidebar:
    st.markdown("## 🎧 EchoGuard")
    st.caption("Unsupervised Machine Audio Anomaly Detection")
    st.markdown("---")

home = st.Page("pages/1_Home_2.py", title="Home", icon="🏠", default=True)
dataset_explorer = st.Page(
    "pages/2_Dataset_Explorer_2.py", title="Dataset Explorer", icon="📂"
)
model_comparison = st.Page(
    "pages/3_Model_Comparison_2.py", title="Model Comparison", icon="📊"
)
interactive_prediction = st.Page(
    "pages/4_Interactive_Prediction_2.py", title="Interactive Prediction", icon="🎯"
)

pg = st.navigation(
    [home, dataset_explorer, model_comparison, interactive_prediction]
)

with st.sidebar:
    st.markdown("---")
    st.caption("Developed for the EchoGuard research project.")

pg.run()
