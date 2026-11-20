# cardiac-acoustic-classification
Binary classification of normal vs. abnormal heartbeat audio recordings using MFCC feature extraction, KNN, and Linear SVM in Python (librosa &amp; scikit-learn).

# Cardiac Acoustic Anomaly Classification

End-to-end supervised machine learning pipeline to classify **normal** versus **abnormal** heart sounds from clinical phonocardiogram (PCG) recordings using acoustic feature engineering (`librosa`) and classical classifiers (`scikit-learn`).

## Overview & Motivation
Heart murmurs and structural cardiac abnormalities produce distinct acoustic signatures in time and frequency domains. This project transforms raw, variable-gain heartbeat audio recordings into structured tabular features and benchmarks distance-based (**K-Nearest Neighbors**) against margin-based (**Linear Support Vector Machine**) classification, prioritizing **Recall (Sensitivity)** and **F1-score** to minimize false-negative diagnostic risk.

## Dataset & Preprocessing
* **Source:** University of Michigan Heart Sound & Murmur Library (23 continuous clinical recordings, ~1 minute each).
* **Segmentation:** Raw recordings were segmented into **1,783 individual labeled cardiac cycles** (`normal` vs. `abnormal`).
* **Amplitude Normalization:** Applied peak amplitude normalization (`librosa.util.normalize`) across clips so recording-gain differences do not bias distance or margin calculations.
* **Feature Extraction:** Computed **14 Mel-Frequency Cepstral Coefficients (MFCCs)** averaged across time frames per cardiac cycle, yielding a structured `1783 × 14` feature matrix.
* **Train/Test Split:** Stratified `80/20` split (`random_state=42`, `358` test samples: `100` normal, `258` abnormal).

## Model Benchmarking & Results

| Model | Hyperparameters | Accuracy | Precision | Recall (Sensitivity) | Specificity | F1-Score | Confusion Matrix (`TN, FP, FN, TP`) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **K-Nearest Neighbors (KNN)** | `n_neighbors=5` | **99.72%** | **100.00%** | **99.61%** | **100.00%** | **99.81%** | `100, 0, 1, 257` |
| **Linear SVM** | `kernel='linear', C=1.0` | **98.04%** | **97.72%** | **99.61%** | **94.00%** | **98.66%** | `94, 6, 1, 257` |

### Key Takeaways
1. **High Sensitivity on Abnormal Cycles:** Both models achieved **99.61% recall** (`257/258` abnormal test beats detected), limiting false negatives to a single sample.
2. **MFCC Cluster Separability:** 2D projection of `MFCC 1` vs. `MFCC 2` confirmed distinct clustering between normal and pathological cardiac cycles, explaining the strong linear and neighborhood separation.
3. **Methodological Note on Group Generalization:** Because the 1,783 cycles were segmented from 23 source recordings, clip-level random splitting places adjacent beats from the same recording across train and test sets. For clinical deployment on unseen patients, evaluation should use `GroupKFold` grouped by source recording ID alongside `StandardScaler` pipelines.

## Tech Stack
* **Language:** Python
* **Audio & Signal Processing:** `librosa`, `NumPy`
* **Machine Learning & Metrics:** `scikit-learn` (`KNeighborsClassifier`, `SVC`, `train_test_split`, `confusion_matrix`)
* **Visualization:** `Matplotlib`

## Author
**Juan Pablo Salazar Pinto** — B.S. in Sound Engineering, Universidad de San Buenaventura
