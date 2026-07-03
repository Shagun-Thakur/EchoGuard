# EchoGuard
### Unsupervised Audio Anomaly Detection using Deep Learning
EchoGuard is a research-driven project focused on <b>machine audio anomaly detection</b> using unsupervised learning and signal processing techniques. Rather than treating anomaly detection as a black-box modeling problem, the project begins with a systematic investigation of machine acoustics to understand how faults manifest in audio recordings.

Current focus:

* Dataset exploration
* Signal processing analysis
* Acoustic feature investigation
* Unsupervised anomaly detection

---

## Project Status
🚧 In Progress

Current stage:

* Dataset exploration completed
* Channel selection completed
* RMS Energy analysis completed
* Spectrogram analysis completed
* Spectral Centroid analysis completed
* MFCC analysis in progress
* Model development pending

---

## Dataset
This project uses the <b>MIMII (Malfunctioning Industrial Machine Investigation and Inspection)</b> dataset.

MIMII is a benchmark dataset designed for machine condition monitoring and audio-based anomaly detection.

#### Machine Types
The dataset contains recordings from four industrial machine categories:
* Pumps
* Fans
* Valves
* Slide Rails
Each machine category contains multiple machine IDs representing different physical machines.

---

## Current Scope
The current analysis focuses exclusively on:
```
Machine Type: Pump
Machine ID  : id_00
Channel     : 0
```
Limiting the scope to a single machine allows detailed investigation of acoustic behaviour expanding to other machine types.

---

## Recording Setup
The MIMII dataset was recorded using:
* TAMAGO-03 circular microphone array
* 8 microphone channels
* Industrial operating environment
* Normal and abnormal operating conditions
Each recording contains approximately 10 seconds of machine audio

---

## Why Channel 0?
The original recordings contains 8 microphone channels.

An exploratory analysis was performed to identify the most representative channel.

Channel 0 was selected because:

* It exhibited the highest minimum cross-correlation with the remaining microphones.
* It showed slightly higher RMS energy, indicating improved signal quality.
* Its spectral centroid occupied a central frequency range without strong directional bias.
* It is the reference microphone used in the original MIMII publication.

---

## Research Questions
This project is driven by the following questions:

<b>Q1:</b>  Can classical signal processing features explain why some anomalies are difficult to distinguish from normal machine sounds?

<b>Q2:</b>  Why do certain abnormal recordings overlap with normal recordings in RMS Energy?

<b>Q3:</b>  Do subtle anomalies exhibit different spectral characteristics despite having similar energy distributions?

<b>Q4:</b>  Can unsupervised deep learning models learn these subtle differences automatically?

---

## Key Observations So Far

##### RMS Energy
* A subset of abnormal recordings overlaps significantly with the normal RMS distribution.
* Simple energy thresholding is therefore insufficient.

##### Spectrogram Analysis
Two anomaly categories emerged:
* Obvious anomalies
* Subtle anomalies
Some faults produce clear broadband disturbances, while others remain visually similar to normal recordings.

##### Spectral Centroid
* Mean spectral centroid provides limited separation.
* Temporal variability measures (Std and IQR) contains more useful information.

##### MFCC Analysis
Current hypothesis:

<i>Temporal stability of cepstral features may be more informative than their avergae values when detecting subtle machine faults.</i>
 
---

## Repository Structure

```
EchoGuard/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   └── id_00
|         |── abnormal
|         └── normal
│
├── notebooks/
│   ├── 01_dataset_exploration.ipynb
│   ├── 02_waveform_analysis.ipynb
│   ├── 03_rms_analysis.ipynb
│   ├── 04_spectrogram_analysis.ipynb
│   ├── 05_spectral_centroid_analysis.ipynb
│   ├── 06_mfcc_analysis.ipynb
│   ├── 07_feature_summary.ipynb
│   └── 08_model.ipynb
│
├── src/
│   ├── __init__.py
│   ├── utils.py
│   ├── features.py
│   ├── visualization.py
│   └── dataset.py
│
├── figures/
│
├── results/
│
└── research_notes/
    ├── Research_Note_01.md
    ├── Research_Note_02.md
```
---

## Planned Work
* Complete MFCC investigation
* Feature correlation analysis
* Log-Mel spectrogram exploration
* Autoencoder baseline
* Normalizing Flow-based anomaly detection
* Comparative evaluation
* real-world machine audio validation

---

## References:

MIMII Dataset:
* Purohit et al., "MIMII Dataset: Sound Dataset for Malfunctioning Industrial Machine Investigation and Inspection"

---
