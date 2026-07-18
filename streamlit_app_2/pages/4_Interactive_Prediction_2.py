import os
import sys
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import joblib
import torch

# -------------------------------------------------------------------------
# Path setup
# -------------------------------------------------------------------------
APP_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = APP_ROOT.parent
sys.path.append(str(APP_ROOT))
sys.path.append(str(PROJECT_ROOT))

from constants import ACCENT, DANGER, SUCCESS, PLOTLY_TEMPLATE  # noqa: E402

try:
    from src.io import load_audio
    from src.feature import extract_handcrafted_features
except Exception as e:  # pragma: no cover
    st.error(
        "Could not import the project's `src` modules. Make sure this "
        f"dashboard is launched from the project root.\n\nDetails: {e}"
    )
    st.stop()

# -------------------------------------------------------------------------
# Page configuration
# -------------------------------------------------------------------------

st.title("🎯 Interactive Prediction")
st.write(
    "Upload a machine audio recording and compare the predictions of the "
    "Handcrafted Feature Autoencoder and the proposed Normalizing Flow model."
)

# -------------------------------------------------------------------------
# Paths & thresholds
# -------------------------------------------------------------------------
AE_MODEL_PATH = PROJECT_ROOT / "results/models/handcrafted_autoencoder_full.pth"
NF_MODEL_PATH = PROJECT_ROOT / "results/models/normalizing_flow_full.pth"
SCALER_PATH = PROJECT_ROOT / "results/models/feature_scaler.pkl"

AE_THRESHOLD = 30.971419
NF_THRESHOLD = 1000.11544

FEATURE_NAMES = [
    "RMS",
    "MFCC1 Mean",
    "MFCC1 Std",
    "MFCC2 Mean",
    "MFCC2 Std",
    "MFCC5 Std",
    "MFCC8 Mean",
    "Centroid Std",
    "Centroid IQR",
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@st.cache_resource(show_spinner=False)
def load_models():
    missing = [p for p in (AE_MODEL_PATH, NF_MODEL_PATH, SCALER_PATH) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing model artifact(s): "
            + ", ".join(str(p.relative_to(PROJECT_ROOT)) for p in missing)
        )
    ae = torch.load(AE_MODEL_PATH, map_location=device, weights_only=False)
    nf = torch.load(NF_MODEL_PATH, map_location=device, weights_only=False)
    ae.eval()
    nf.eval()
    scaler = joblib.load(SCALER_PATH)
    return ae, nf, scaler


def gauge(value: float, threshold: float, title: str, is_anomalous: bool) -> go.Figure:
    """Interactive gauge showing a score against its decision threshold."""
    span = max(value, threshold) * 1.5 or 1.0
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            number={"valueformat": ".4f"},
            title={"text": title},
            gauge={
                "axis": {"range": [0, span]},
                "bar": {"color": DANGER if is_anomalous else SUCCESS},
                "steps": [
                    {"range": [0, threshold], "color": "rgba(34, 197, 94, 0.15)"},
                    {"range": [threshold, span], "color": "rgba(239, 68, 68, 0.12)"},
                ],
                "threshold": {
                    "line": {"color": ACCENT, "width": 3},
                    "thickness": 0.85,
                    "value": threshold,
                },
            },
        )
    )
    fig.update_layout(height=260, margin=dict(l=20, r=20, t=50, b=10))
    return fig


# -------------------------------------------------------------------------
# Load models once (cached) — surfaced as a friendly error instead of a crash
# -------------------------------------------------------------------------
try:
    with st.spinner("Loading models..."):
        ae_model, nf_model, scaler = load_models()
except Exception as e:
    st.error(f"Could not load the trained models: {e}")
    st.stop()

st.markdown("---")

# -------------------------------------------------------------------------
# Upload
# -------------------------------------------------------------------------
uploaded_file = st.file_uploader("Upload a WAV file", type=["wav"])

if uploaded_file is None:
    st.info(
        "📁 Upload a WAV file to compare predictions from the Handcrafted "
        "Autoencoder and Normalizing Flow models."
    )
    with st.expander("About the anomaly scores"):
        st.markdown(
            """
            **Handcrafted Autoencoder**
            - Learns to reconstruct only normal machine behaviour.
            - Higher reconstruction error indicates greater deviation from normality.

            **Normalizing Flow**
            - Learns the probability distribution of normal feature vectors.
            - Lower likelihood (higher negative log-likelihood) indicates a more anomalous recording.
            """
        )
    st.stop()

# From this point on, uploaded_file is guaranteed to exist, so the whole
# pipeline is written as one straight-line block (the original file split
# this into two separate `if uploaded_file is not None:` sections lower
# down, which made the control flow harder to follow for no benefit).
temp_audio_path = None
try:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_file.read())
        temp_audio_path = tmp.name

    st.audio(temp_audio_path)

    # ---------------------------------------------------------------
    # Load audio
    # ---------------------------------------------------------------
    try:
        signal, sr = load_audio(temp_audio_path)
    except Exception as e:
        st.error(f"Could not read this audio file: {e}")
        st.stop()

    st.markdown("---")
    a_col1, a_col2 = st.columns(2)
    a_col1.metric("Sampling Rate", f"{sr} Hz")
    a_col2.metric("Audio Length", f"10.00 sec")

    # ---------------------------------------------------------------
    # Feature extraction
    # ---------------------------------------------------------------
    with st.spinner("Extracting handcrafted features..."):
        feature_vector = extract_handcrafted_features(temp_audio_path)

    st.subheader("Extracted Features")
    st.dataframe(
        pd.DataFrame({"Feature": FEATURE_NAMES, "Value": feature_vector}),
        use_container_width=True,
        hide_index=True,
    )

    scaled_features = scaler.transform(feature_vector.reshape(1, -1))
    feature_tensor = torch.tensor(scaled_features, dtype=torch.float32, device=device)

    # ---------------------------------------------------------------
    # Inference
    # ---------------------------------------------------------------
    with st.spinner("Running inference..."):
        with torch.no_grad():
            reconstruction = ae_model(feature_tensor)
            frame_errors = torch.mean((feature_tensor - reconstruction) ** 2, dim=1)
            ae_score = frame_errors.mean().item()
            ae_is_anomalous = ae_score >= AE_THRESHOLD
            ae_prediction = "🔴 Abnormal" if ae_is_anomalous else "🟢 Normal"

            log_prob = nf_model.log_prob(feature_tensor)
            nf_score = (-log_prob).mean().item()
            nf_is_anomalous = nf_score >= NF_THRESHOLD
            nf_prediction = "🔴 Abnormal" if nf_is_anomalous else "🟢 Normal"

    # ---------------------------------------------------------------
    # Results
    # ---------------------------------------------------------------
    st.markdown("---")
    st.header("Prediction Results")

    res_col1, res_col2 = st.columns(2)
    with res_col1:
        st.subheader("Handcrafted Autoencoder")
        st.plotly_chart(
            gauge(ae_score, AE_THRESHOLD, "Reconstruction Error", ae_is_anomalous),
            use_container_width=True,
        )
        if not ae_is_anomalous:
            st.success(ae_prediction)
        else:
            st.error(ae_prediction)

    with res_col2:
        st.subheader("Normalizing Flow")
        st.plotly_chart(
            gauge(nf_score, NF_THRESHOLD, "Negative Log-Likelihood", nf_is_anomalous),
            use_container_width=True,
        )
        if not nf_is_anomalous:
            st.success(nf_prediction)
        else:
            st.error(nf_prediction)

    # ---------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------
    st.markdown("---")
    st.header("Model Summary")
    st.dataframe(
        {
            "Model": ["Handcrafted Autoencoder", "Normalizing Flow"],
            "Score": [round(ae_score, 4), round(nf_score, 4)],
            "Threshold": [round(AE_THRESHOLD, 4), round(NF_THRESHOLD, 4)],
            "Prediction": [ae_prediction, nf_prediction],
        },
        use_container_width=True,
        hide_index=True,
    )

    # ---------------------------------------------------------------
    # Interpretation
    # ---------------------------------------------------------------
    st.markdown("---")
    st.header("Interpretation")

    if not ae_is_anomalous and not nf_is_anomalous:
        st.success(
            "Both models classify this recording as **Normal** — it lies "
            "within the learned distribution of normal machine behaviour."
        )
    elif ae_is_anomalous and nf_is_anomalous:
        st.error(
            "Both models classify this recording as **Abnormal** — it "
            "deviates significantly from normal operating behaviour."
        )
    else:
        st.warning(
            "The two models disagree. This recording lies close to the "
            "decision boundary — further inspection is recommended."
        )

    with st.expander("About the anomaly scores"):
        st.markdown(
            """
            **Handcrafted Autoencoder**
            - Learns to reconstruct only normal machine behaviour.
            - Higher reconstruction error indicates greater deviation from normality.

            **Normalizing Flow**
            - Learns the probability distribution of normal feature vectors.
            - Lower likelihood (higher negative log-likelihood) indicates a more anomalous recording.
            """
        )

finally:
    # Structural fix: the original file tried to remove `temp_audio_path`
    # unconditionally at module level, even on the branch where no file was
    # ever uploaded (NameError, silently swallowed). Cleanup now only runs
    # when a temp file was actually created.
    if temp_audio_path is not None:
        try:
            os.remove(temp_audio_path)
        except OSError:
            pass
