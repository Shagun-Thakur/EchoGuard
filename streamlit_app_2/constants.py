"""
Central place for numbers, labels, and copy that are reused across
multiple pages of the EchoGuard dashboard.

"""

# ---------------------------------------------------------------------------
# Headline metrics (ROC-AUC) — single source of truth
# ---------------------------------------------------------------------------
METRICS = {
    "Baseline Autoencoder": "72.23 ± 2.05, 79.27 ± 3.07",
    "Handcrafted Autoencoder": "96.93 ± 0.74, 89.92 ± 1.03",
    "Normalizing Flow": "98.34 ± 0.76, 94.64 ± 1.45",
}

# Extra columns for the detailed comparison table on the Model Comparison page
COMPARISON_TABLE = {
    "Metric": [
        "ROC-AUC and pAUC",
        "Input Features",
        "Training Time",
        "Anomaly Score",
        "Training Data",
    ],
    "Baseline Autoencoder": [
        METRICS["Baseline Autoencoder"],
        "320 Log-Mel",
        "~25 min",
        "Reconstruction Error",
        "Normal recordings",
    ],
    "Handcrafted Autoencoder": [
        METRICS["Handcrafted Autoencoder"],
        "9 Selected Features",
        "~1 min",
        "Reconstruction Error",
        "Normal recordings",
    ],
    "Normalizing Flow": [
        METRICS["Normalizing Flow"],
        "9 Selected Features",
        "~1 min",
        "Negative Log-Likelihood",
        "Normal recordings",
    ],
}

# ---------------------------------------------------------------------------
# Dataset facts
# ---------------------------------------------------------------------------
DATASET_INFO = {
    "Dataset": "MIMII",
    "Machine Type": "Pump",
    "Machine ID": "id_00",
    "Channel": "0",
    "Training Recordings": 1006,
    "Evaluation Recordings": 1149,
}

# ---------------------------------------------------------------------------
# Research pipeline steps (used to draw the flow diagram on the Home page)
# ---------------------------------------------------------------------------
PIPELINE_STEPS = [
    ("🎙️", "Machine Audio"),
    ("📈", "Signal Analysis"),
    ("🧮", "Feature Engineering"),
    ("🧠", "Baseline Autoencoder"),
    ("🌊", "Normalizing Flow"),
    ("📊", "Model Comparison"),
    ("🎯", "Interactive Inference"),
]

# ---------------------------------------------------------------------------
# Shared color / theme tokens
# ---------------------------------------------------------------------------
ACCENT = "#7C3AED"       # violet accent used across charts
ACCENT_SOFT = "#A78BFA"
SUCCESS = "#22C55E"
DANGER = "#EF4444"
NEUTRAL = "#94A3B8"

PLOTLY_TEMPLATE = "plotly_white"
