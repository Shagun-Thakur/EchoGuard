"""
Central place for numbers, labels, and copy that are reused across
multiple pages of the EchoGuard dashboard.

NOTE: The original app had different ROC-AUC values for the same model
on different pages (e.g. Handcrafted Autoencoder was reported as 0.980,
0.984 AND 0.968 in three different spots; Normalizing Flow was reported
as 0.9871 AND 0.9895). Whoever maintains this project should replace the
placeholders below with the exact values from the final notebook run --
right now they default to the value that appeared most often.
"""

# ---------------------------------------------------------------------------
# Headline metrics (ROC-AUC) — single source of truth
# ---------------------------------------------------------------------------
METRICS = {
    "Baseline Autoencoder": 0.6830,
    "Handcrafted Autoencoder": 0.9840,
    "Normalizing Flow": 0.9871,
}

# Extra columns for the detailed comparison table on the Model Comparison page
COMPARISON_TABLE = {
    "Metric": [
        "ROC-AUC",
        "Input Features",
        "Training Time",
        "Anomaly Score",
        "Training Data",
    ],
    "Baseline Autoencoder": [
        f"{METRICS['Baseline Autoencoder']:.3f}",
        "320 Log-Mel",
        "10-15 min",
        "Reconstruction Error",
        "Normal recordings",
    ],
    "Handcrafted Autoencoder": [
        f"{METRICS['Handcrafted Autoencoder']:.3f}",
        "9 Selected Features",
        "1-2 min",
        "Reconstruction Error",
        "Normal recordings",
    ],
    "Normalizing Flow": [
        f"{METRICS['Normalizing Flow']:.4f}",
        "9 Selected Features",
        "1-2 min",
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
