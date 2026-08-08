# 🎧 EchoGuard

**Evidence-Based Handcrafted Feature Representation for Industrial Audio Anomaly Detection**

---

## Overview

EchoGuard is a research project investigating how acoustic **feature representation** influences unsupervised industrial machine audio anomaly detection.

The central idea is simple:

> Before choosing a more sophisticated anomaly detection model, understand whether the *representation* of the audio itself is appropriate for the task.

Rather than treating anomaly detection as a purely deep-learning problem, EchoGuard combines signal processing, statistical analysis, machine learning-based feature importance, dimensionality analysis, and visualization to construct a compact, interpretable handcrafted representation of machine audio.

The final representation reduces **50 handcrafted acoustic descriptors to 9 selected features**, which are then evaluated using an Autoencoder and a Normalizing Flow.

**LIVE DEMO:** [EchoGuard](https://echoguard-etbpz3hmxxfke7mxfpgdt7.streamlit.app/)


---

## Project Status

✅ **Completed** — The complete experimental pipeline has been implemented and evaluated.

---

## Research Question

> Can a carefully selected, low-dimensional handcrafted acoustic representation provide **better and more efficient** anomaly detection than a conventional high-dimensional Log-Mel Spectrogram representation?

This is evaluated through a controlled comparison between three configurations:

| Configuration | Feature Representation | Model |
|---|---|---|
| Baseline | Log-Mel Spectrogram | Autoencoder |
| Proposed (AE) | 9 Handcrafted Features | Autoencoder |
| Proposed (NF) | 9 Handcrafted Features | Normalizing Flow |

---

## Core Contribution — Evidence-Based Feature Selection Pipeline

The primary contribution of EchoGuard is the **Evidence-Based Multi-Method Feature Selection Pipeline**.

Instead of selecting features using a single ranking algorithm, multiple complementary sources of evidence were combined:

```
50 Handcrafted Features
        ↓
Signal Processing & Redundancy Analysis
        ↓
PCA
        ↓
t-SNE Visualization
        ↓
ANOVA
        ↓
Mutual Information
        ↓
Random Forest Feature Importance
        ↓
Evidence-Based Voting
        ↓
9 Final Features
```

Only features demonstrating consistent support across multiple independent analyses were retained.

### Final Feature Set

| ID | Feature | Physical Interpretation |
|---|---|---|
| F1 | RMS Energy | Overall acoustic energy level |
| F2 | MFCC-1 Mean | Dominant spectral envelope shape |
| F3 | MFCC-1 Std | Temporal variability of spectral envelope |
| F4 | MFCC-2 Mean | Strong visual class separation |
| F5 | MFCC-2 Std | Temporal variation of lower-order cepstral info |
| F6 | MFCC-5 Std | Variability in higher-order spectral information |
| F7 | MFCC-8 Mean | Fine spectral texture characteristics |
| F8 | Spectral Centroid Std | Frequency instability over time |
| F9 | Spectral Centroid IQR | Robust dispersion of dominant frequency |

> **Why MFCC-2 Mean?** Although it was not consistently the highest-ranked feature statistically, its visual feature-space separation between normal and abnormal recordings provided strong supporting evidence. A feature earns its place through multiple complementary forms of evidence, not through a single score.

---

## Dataset

EchoGuard uses the **MIMII** (Malfunctioning Industrial Machine Investigation and Inspection) dataset — a benchmark for audio-based industrial machine condition monitoring.

### Experimental Scope

This study intentionally focuses on a controlled subset:

| Parameter | Value |
|---|---|
| Machine Type | Pump |
| Machine ID | id_00 |
| Channel | 0 |
| Sampling Rate | 16 kHz |
| Recording Length | ~10 seconds |
| Microphone Array | TAMAGO-03 (8 channels) |

> ⚠️ **Important:** Conclusions should not be interpreted as universal across all MIMII machine types or IDs. This restricted scope allows detailed investigation of the relationship between feature representation and detection performance.

### Why Channel 0?

An exploratory channel-selection analysis was performed before the main experiments. Channel 0 was selected based on cross-channel correlation analysis, RMS energy characteristics, spectral characteristics, and consistency with the reference MIMII configuration.

---

## Models

### Handcrafted Feature Autoencoder
Learns to reconstruct normal machine behaviour. During inference, higher reconstruction error indicates greater deviation from normality.

### Normalizing Flow
Models the probability distribution of normal handcrafted feature vectors. During inference, lower likelihood (higher negative log-likelihood) indicates a more anomalous recording.

---

## Evaluation Methodology

Each model was trained and evaluated over **10 independent trials** to account for stochastic variation in neural network training. Results are reported as:

- **Mean ± Standard Deviation**
- **ROC-AUC** — overall discrimination performance
- **pAUC** — partial AUC at low false-positive rates, relevant for industrial monitoring systems where excessive false alarms reduce trust

Pairwise statistical significance testing was performed to confirm that observed differences reflect genuine performance differences rather than random variation.

---

## Key Findings

1. **Feature representation matters** — The handcrafted representation substantially outperformed the Log-Mel Spectrogram baseline.
2. **Compact representations can be effective** — 9 features achieved stronger performance than 320 Log-Mel features, with significantly lower training time.
3. **Normalizing Flow provides additional improvement** — The Normalizing Flow achieved the highest average performance among evaluated models when trained on the selected handcrafted features.

---

## Experimental Results

Mean performance across 10 trials:

| Model | Feature Representation | ROC-AUC | pAUC |
|---|---|---|---|
| Baseline Autoencoder | Log-Mel Spectrogram | 72.23±2.05 | 79.27±3.07 |
| Handcrafted Autoencoder | 9 Handcrafted Features | 96.93±0.74 |89.92±1.03|
| Normalizing Flow | 9 Handcrafted Features | 0.9834±0.76 | 0.9464±1.45 |


Performance differences between models were statistically significant for both ROC-AUC and pAUC.

---

## Repository Structure

```
EchoGuard/
│   .gitignore
│   LICENSE
│   pyproject.toml
│   README.md
│   requirements.txt
│   setup.py
│
├───data
│   │   .gitkeep
│   │   README.md
│   │
│   └───Processed
│           eval_data.npy
│           eval_files.npy
│           eval_labels.npy
│           train_data.npy
│           train_files.npy
│
├───notebooks
│   │   01_dataset_exploration.ipynb
│   │   02_channel_selection.ipynb
│   │   03_waveform_analysis.ipynb
│   │   04_rms_analysis.ipynb
│   │   05_spectrogram_analysis.ipynb
│   │   06_spectral_centroid_analysis.ipynb
│   │   07_mfcc_analysis.ipynb
│   │   08_feature_space_analysis.ipynb
│   │   09_feature_selection.ipynb
│   │   10_baseline_reproduction.ipynb
│   │   11_handcrafted_features_Autoencoder.ipynb
│   │   12_normalizing_flow_handcrafted_features.ipynb
│   └───13_statistical_evaluation.ipynb
│   
├───results
│   │   anova_feature_scores.csv
│   │   automatic_discussion.txt
│   │   benchmark_split.csv
│   │   experiment_split.csv
│   │   features.csv
│   │   features_statistics.csv
│   │   feature_evidence_matrix.csv
│   │   feature_ranking.csv
│   │   group_a.csv
│   │   group_b.csv
│   │   group_c.csv
│   │   mfcc_group_summary.csv
│   │   mfcc_summary.csv
│   │   mutual_information_scores.csv
│   │   obvious_mfcc_data.csv
│   │   overlap_mfcc_data.csv
│   │   paper_table_performance.csv
│   │   pca_loadings.csv
│   │   performance_improvement.csv
│   │   random_forest_importance.csv
│   │   research_evidence.csv
│   │   rms_overlap_files.csv
│   │   rms_summary.csv
│   │   selected_features.csv
│   │   spectral_centroid_summary.csv
│   │   statistical_significance.csv
│   │   strong_features_correlation.csv
│   │   summary_statistics.csv
│   │   test_run_results.csv
│   │   tSNE_visualization.py
│   │
│   ├───figures
│   │   ├───baseline_autoencoder
│   │   │       anomaly_score_distributaion_baseline.png
│   │   │       roc-curve(baseline).png
│   │   │
│   │   ├───feature_selection
│   │   │       anova_vs_mi.png
│   │   │       evidence_distribution.png
│   │   │
│   │   ├───feature_space
│   │   │   │   feature_correlation_matrix.png
│   │   │   │   pca.png
│   │   │   │   pca_biplot.png
│   │   │   │   pca_projection.png
│   │   │   │   pca_variance.png
│   │   │   │   top_pca_laodings.png
│   │   │   │   tsne.png
│   │   │   └───tsne_projection.png
│   │   │
│   │   ├───handcrafted_features_autoencoder
│   │   │   │   error_distribution.png
│   │   │   │   error_research_group.png
│   │   │   │   performance_comparison.png
│   │   │   └───roc-curve.png
│   │   ├───mfcc
│   │   │   │   mfcc1_mean_boxplot.png
│   │   │   │   mfcc1_std_boxplot.png
│   │   │   │   mfcc2_mean_boxplot.png
│   │   │   │   mfcc2_std_boxplot.png
│   │   │   │   mfcc3_mean_boxplot.png
│   │   │   │   mfcc3_std_boxplot.png
│   │   │   │   mfcc4_mean_boxplot.png
│   │   │   │   mfcc4_std_boxplot.png
│   │   │   │   mfcc5_mean_boxplot.png
│   │   │   │   mfcc5_std_boxplot.png
│   │   │   │   mfcc_correlation_matrix.png
│   │   │   │   mfcc_mean_profile.png
│   │   │   └───mfcc_std_profile.png
│   │   ├───normalizing_flow_handcrafted_features
│   │   │       anomaly_score_distribution.png
│   │   │       anomaly_score_distribution_boxplot.png
│   │   │       model_performance_comparison.png
│   │   │
│   │   ├───statistical_evaluation
│   │   │       auc_boxplot.png
│   │   │       auc_errorbar.png
│   │   │       auc_histograms.png
│   │   │       pauc_boxplot.png
│   │   │       pauc_errorbar.png
│   │   │       pauc_histograms.png
│   │   │       tsne_comparison.png
│   │   │
│   │   └───streamlit_images
│   │           anova_fScore.png
│   │           auc_boxplot.png
│   │           baseline_histogram.png
│   │           feature_correlation_matrix.png
│   │           handcrafted_autoencoder_histogram.png
│   │           mutual_information.png
│   │           normalizing_flow_histogram.png
│   │           pauc_boxplot.png
│   │           pca_variance.png
│   │           rf_importance.png
│   │           tsne_comparison.png
│   │
│   └───models
│           baseline_autoencoder.pth
│           feature_scaler.pkl
│           handcrafted_autoencoder.pth
│           handcrafted_autoencoder_full.pth
│           normalizing_flow.pth
│           normalizing_flow_full.pth
│
├───sample_dataset_for_app
│   ├───abnormal
│   └───normal
│           
│
├───src
│   │   config.py
│   │   dataset.py
│   │   feature.py
│   │   io.py
│   │   local_config.example.py
│   │   models.py
│   │   utils.py
│   │   visualization.py
│   └─── __init__.py
│
└───streamlit_app
    │   app.py
    │   constants.py
    │
    └───pages
        │   1_Home.py
        │   2_FeatureSelection_Pipeline.py
        │   3_Dataset_Explorer.py
        │   4_Model_Comparison.py
        └─── 5_Interactive_Prediction.py
   

```
---

## Running the Project

### Research Experiments

The notebooks contain the complete experimental workflow — dataset analysis, feature engineering, feature selection, model training, evaluation, and statistical analysis.

The **full MIMII dataset** is required for reproducing experiments. It is not included in this repository due to size and distribution constraints.

### Streamlit Dashboard

The interactive dashboard can be run using the included sample dataset and trained model artifacts:

```bash
# Install dependencies
pip install -r requirements.txt

# Launch the dashboard from the project root
streamlit run streamlit_app/app.py
```

---

## Streamlit Dashboard

The interactive dashboard demonstrates the complete research workflow:

| Page | Content |
|---|---|
| 🏠 Home | Project overview and key findings |
| 🔬 Feature Selection Pipeline | Interactive walkthrough of the core contribution |
| 📂 Dataset Explorer | Listen to recordings, inspect waveforms and features |
| 📊 Model Comparison | ROC-AUC results and figure comparisons |
| 🎯 Interactive Prediction | Upload a WAV file and get model predictions |

---

## Limitations

- Evaluation is restricted to **Pump id_00, Channel 0**
- Results should not be generalised across all MIMII machine types or IDs
- Abnormal recordings were used during the offline feature-selection process — the pipeline is therefore **not fully label-free**
- Statistical conclusions are based on 10 trials on a single machine configuration

---

## Future Work

- Evaluation across additional Pump machine IDs
- Evaluation across other MIMII machine categories (Fans, Valves, Slide Rails)
- Investigation of fully label-free feature selection
- Validation on additional industrial audio datasets
- Cross-machine generalisation experiments
- Evaluation under different noise and SNR conditions

---

## Project Philosophy

> Better anomaly detection may begin with better **representation**, not a more complicated model.

EchoGuard treats feature engineering and representation analysis as first-class research problems, rather than simply increasing model complexity.

---

## Reference

**MIMII Dataset**

Purohit et al., *"MIMII Dataset: Sound Dataset for Malfunctioning Industrial Machine Investigation and Inspection."*

---

## Project Status Checklist

- ✅ Dataset exploration
- ✅ Microphone channel selection
- ✅ Signal processing analysis (RMS, Spectral Centroid, MFCC, Log-Mel)
- ✅ Feature correlation and redundancy analysis
- ✅ PCA analysis
- ✅ Scatter plot feature investigation
- ✅ t-SNE visualization
- ✅ ANOVA analysis
- ✅ Mutual Information analysis
- ✅ Random Forest feature importance
- ✅ Evidence-based multi-method feature selection
- ✅ Handcrafted Feature Autoencoder
- ✅ Normalizing Flow anomaly detector
- ✅ Repeated-trial evaluation (10 trials)
- ✅ ROC-AUC and pAUC evaluation
- ✅ Statistical significance analysis
- ✅ Interactive Streamlit dashboard
