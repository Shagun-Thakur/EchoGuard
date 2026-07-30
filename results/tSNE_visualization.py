"""
t-SNE Visualization for Paper Figure
Two subplots:
    Left - t-SNE of all raw features (high-dimensional, messy)
    Right - t-SNE of 9 handcrafted features (compact, clean separation)

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from src.config import *

RANDOM_SEED = 42

# Load data
print("Loading data...")

df_9   = pd.read_csv(results_path/"selected_features.csv")   # 9 features + label
df_all = pd.read_csv(results_path/"features.csv")             # all features + label

#Separate features and labels 


# -- 9-feature set
X9     = df_9.drop(columns= ["file_name", "Research_Group", "label"]).values
y9     = df_9["label"].values

# -- Full feature set
X_all  = df_all.drop(columns= ["file_name", "Research_Group", "label", "Group", "recording_id"]).values
y_all  = df_all["label"].values

#Standardise 
print("Standardising features...")
X9_scaled    = StandardScaler().fit_transform(X9)
X_all_scaled = StandardScaler().fit_transform(X_all)

#Run t-SNE
print("Running t-SNE on full feature set... ")
tsne_all = TSNE(
    n_components  = 2,
    perplexity    = 30,
    n_iter        = 1000,
    random_state  = RANDOM_SEED,
    init          = "pca",       # more stable than random
    learning_rate = "auto"
)
Z_all = tsne_all.fit_transform(X_all_scaled)

print("Running t-SNE on 9-feature set...")
tsne_9 = TSNE(
    n_components  = 2,
    perplexity    = 30,
    n_iter        = 1000,
    random_state  = RANDOM_SEED,
    init          = "pca",
    learning_rate = "auto"
)
Z9 = tsne_9.fit_transform(X9_scaled)

#Colour map

def get_colors(labels):
    unique = np.unique(labels)
    # Try to identify which label is normal vs anomalous
    color_map = {}
    for u in unique:
        if u.lower() in ["normal", "0"]:
            color_map[u] = "#2196F3"   # blue  → normal
        else:
            color_map[u] = "#F44336"   # red   → anomalous
    return np.array([color_map[l] for l in labels]), color_map

colors_all, cmap_all = get_colors(y_all)
colors_9,   cmap_9   = get_colors(y9)

# ── 6. Plot ───────────────────────────────────────────────────────────────────
print("Plotting...")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.patch.set_facecolor("white")

ALPHA      = 0.65
DOT_SIZE   = 18
TITLE_SIZE = 13
LABEL_SIZE = 11

# ── Left: full feature space ──────────────────────────────────────────────────
ax = axes[0]
ax.scatter(
    Z_all[:, 0], Z_all[:, 1],
    c=colors_all, s=DOT_SIZE, alpha=ALPHA,
    linewidths=0.2, edgecolors="white"
)
ax.set_title(
    f"(a) All Raw Features\n(320-dim Log-Mel Spectrogram Space)",
    fontsize=TITLE_SIZE, fontweight="bold", pad=10
)
ax.set_xlabel("t-SNE Dimension 1", fontsize=LABEL_SIZE)
ax.set_ylabel("t-SNE Dimension 2", fontsize=LABEL_SIZE)
ax.tick_params(labelsize=9)
ax.spines[["top", "right"]].set_visible(False)
ax.set_facecolor("#FAFAFA")

# ── Right: 9-feature space ────────────────────────────────────────────────────
ax = axes[1]
ax.scatter(
    Z9[:, 0], Z9[:, 1],
    c=colors_9, s=DOT_SIZE, alpha=ALPHA,
    linewidths=0.2, edgecolors="white"
)
ax.set_title(
    f"(b) Handcrafted 9-Feature Vector\n(Evidence-Based Pipeline Output)",
    fontsize=TITLE_SIZE, fontweight="bold", pad=10
)
ax.set_xlabel("t-SNE Dimension 1", fontsize=LABEL_SIZE)
ax.set_ylabel("t-SNE Dimension 2", fontsize=LABEL_SIZE)
ax.tick_params(labelsize=9)
ax.spines[["top", "right"]].set_visible(False)
ax.set_facecolor("#FAFAFA")

# ── Shared legend ─────────────────────────────────────────────────────────────
legend_handles = [
    mpatches.Patch(color="#2196F3", label="Normal"),
    mpatches.Patch(color="#F44336", label="Anomalous"),
]
fig.legend(
    handles=legend_handles,
    loc="lower center",
    ncol=2,
    fontsize=LABEL_SIZE,
    frameon=True,
    framealpha=0.9,
    edgecolor="#CCCCCC",
    bbox_to_anchor=(0.5, -0.04)
)

# ── Caption note ──────────────────────────────────────────────────────────────
fig.text(
    0.5, -0.10,
    "Fig. X. t-SNE projections of (a) the 320-dimensional log-mel spectrogram space and\n"
    "(b) the nine-dimensional handcrafted feature space (MIMII Pump id_00, SNR 0 dB, Channel 0).\n"
    "Blue: Normal recordings. Red: Anomalous recordings. Perplexity=30, n_iter=1000, seed=42.",
    ha="center", fontsize=9, color="#444444", style="italic"
)

plt.tight_layout(rect=[0, 0.05, 1, 1])

# ── 7. Save ───────────────────────────────────────────────────────────────────
OUTPUT_PATH = results_path/"figures"/"statistical_evaluation"/"tsne_comparison.png"
plt.savefig(
    OUTPUT_PATH,
    dpi=300,
    bbox_inches="tight",
    facecolor="white"
)
print(f"\nSaved -> {OUTPUT_PATH}")
print("Done. Add this figure to your paper as Fig. X in Section IV or VII-A.")
plt.show()