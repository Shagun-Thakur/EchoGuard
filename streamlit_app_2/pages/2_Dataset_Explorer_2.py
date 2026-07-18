import os
import sys
from pathlib import Path

import streamlit as st
import plotly.graph_objects as go

# -------------------------------------------------------------------------
# Path setup
# -------------------------------------------------------------------------
APP_ROOT = Path(__file__).resolve().parents[1]          # streamlit_app/
PROJECT_ROOT = APP_ROOT.parent                           # project root
sys.path.append(str(APP_ROOT))
sys.path.append(str(PROJECT_ROOT))

from constants import ACCENT, PLOTLY_TEMPLATE  # noqa: E402

# Project modules (only imported once the paths above are set)
try:
    from src.config import normal_path, abnormal_path
    from src.dataset import get_audio_files
    from src.io import load_audio
    from src.feature import compute_log_mel, rms, spectral_centroid, mfcc
except Exception as e:  # pragma: no cover - defensive import guard
    st.error(
        "Could not import the project's `src` modules. Make sure this "
        "dashboard is launched from the project root so that `src/` is "
        f"importable.\n\nDetails: {e}"
    )
    st.stop()

# -------------------------------------------------------------------------
# Page header
# -------------------------------------------------------------------------
st.title("📂 Dataset Explorer")
st.write(
    "Listen to recordings from the **MIMII Pump dataset** and explore "
    "their acoustic characteristics interactively."
)
st.markdown("---")


# -------------------------------------------------------------------------
# Cached helpers — avoid recomputing audio/feature loads on every rerun
# -------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def cached_audio_files(kind: str):
    path = normal_path if kind == "Normal" else abnormal_path
    return get_audio_files(path)


@st.cache_data(show_spinner=False)
def cached_load_audio(file_path: str):
    return load_audio(file_path, mono=True)


@st.cache_data(show_spinner=False)
def cached_features(signal, sr):
    return {
        "log_mel": compute_log_mel(signal),
        "rms": rms(signal),
        "centroid": spectral_centroid(signal, sr),
        "mfcc": mfcc(signal, sr),
    }


def line_chart(y, x_label, y_label, title):
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=y, mode="lines", line=dict(color=ACCENT, width=1.4)))
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        height=320,
        margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig


def heatmap(matrix, title, y_label):
    fig = go.Figure(data=go.Heatmap(z=matrix, colorscale="Viridis"))
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title=title,
        xaxis_title="Frame",
        yaxis_title=y_label,
        height=380,
        margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig


# -------------------------------------------------------------------------
# Sidebar controls
# -------------------------------------------------------------------------
recording_type = st.sidebar.selectbox("Recording Type", ["Normal", "Abnormal"])

audio_files = cached_audio_files(recording_type)

if not audio_files:
    st.warning(
        f"No **{recording_type.lower()}** recordings were found. Check that "
        "the dataset paths in `src/config.py` point to an existing folder."
    )
    st.stop()

selected_file = st.selectbox(
    "Choose Recording",
    audio_files,
    format_func=lambda x: os.path.basename(x),
)

# -------------------------------------------------------------------------
# Load audio + show quick facts
# -------------------------------------------------------------------------
try:
    signal, sr = cached_load_audio(str(selected_file))
except Exception as e:
    st.error(f"Could not load `{os.path.basename(selected_file)}`: {e}")
    st.stop()

st.audio(str(selected_file))

info_col1, info_col2, info_col3 = st.columns(3)
info_col1.metric("Sampling Rate", f"{sr} Hz")
info_col2.metric("Duration", f"{len(signal) / sr:.2f} s")
info_col3.metric("Samples", f"{len(signal):,}")

st.markdown("---")

# -------------------------------------------------------------------------
# Interactive tabs for each analysis view
# -------------------------------------------------------------------------
with st.spinner("Computing acoustic features..."):
    feats = cached_features(signal, sr)

tab_wave, tab_mel, tab_rms, tab_centroid, tab_mfcc = st.tabs(
    ["Waveform", "Log-Mel Spectrogram", "RMS Energy", "Spectral Centroid", "MFCC"]
)

with tab_wave:
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=signal, mode="lines", line=dict(color=ACCENT, width=0.8)))
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title="Waveform",
        xaxis_title="Samples",
        yaxis_title="Amplitude",
        height=350,
        margin=dict(l=10, r=10, t=40, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)

with tab_mel:
    st.plotly_chart(
        heatmap(feats["log_mel"], "Log-Mel Spectrogram", "Mel Bin"),
        use_container_width=True,
    )

with tab_rms:
    st.plotly_chart(
        line_chart(feats["rms"], "Frame", "RMS", "RMS Energy"),
        use_container_width=True,
    )

with tab_centroid:
    st.plotly_chart(
        line_chart(feats["centroid"], "Frame", "Hz", "Spectral Centroid"),
        use_container_width=True,
    )

with tab_mfcc:
    st.plotly_chart(
        heatmap(feats["mfcc"], "MFCC", "Coefficient"),
        use_container_width=True,
    )

st.caption(
    "Tip: hover over any chart to inspect exact values, and drag to zoom — "
    "double-click to reset the view."
)
