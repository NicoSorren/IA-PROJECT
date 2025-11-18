# Multimodal Vegetable Recognition System 🥕🍆🥔

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Librosa](https://img.shields.io/badge/Audio-Librosa-orange)
![OpenCV](https://img.shields.io/badge/Vision-OpenCV-green)
![Scikit-learn](https://img.shields.io/badge/ML-Scikit--Learn-red)

## 📋 Overview

This project implements an intelligent agent capable of identifying vegetables (**Eggplant, Carrot, Potato, Sweet Potato**) using a **multimodal approach**: Computer Vision and Voice Recognition. 

The system integrates **Supervised Learning (KNN)** for audio classification and **Unsupervised Learning (K-Means)** for image segmentation and classification. It was developed as a final project for the Artificial Intelligence I course at *Universidad Nacional de Cuyo*.

## 🧠 Key Features

### 🗣️ Audio Recognition Pipeline
A robust pipeline designed to classify spoken words with **89.26% accuracy**.
* **Preprocessing:** Noise reduction (`noisereduce`), silence removal, and amplitude normalization.
* **Feature Extraction (149 features):** * MFCCs (13 coeffs) + Delta MFCCs.
    * Spectral Contrast.
    * Zero Crossing Rate (ZCR).
    * Formant estimation (first 3 peaks).
    * Signal Energy.
* **Dimensionality Reduction:** **PCA (Principal Component Analysis)** reduced the feature space from 149 to **30 components**, retaining ~77% of variance.
* **Classification:** **K-Nearest Neighbors (KNN)** with $K=7$, validated via 5-Fold Cross-Validation.

### 📷 Computer Vision Pipeline
An image processing system focused on color-based classification with **85% accuracy**.
* **Preprocessing:** Background removal, HSV Saturation boost (+120), **CLAHE** (Contrast Limited Adaptive Histogram Equalization) on LAB channel, and sharpening kernels.
* **Segmentation:** **K-Means Clustering** ($K=4$) to isolate dominant color regions.
* **Classification:** Unsupervised clustering mapped to ground-truth labels based on color centroids.

## 🛠️ Tech Stack

* **Language:** Python
* **Audio Processing:** `librosa`, `noisereduce`, `soundfile`
* **Computer Vision:** `OpenCV` (cv2), `NumPy`
* **Machine Learning:** `scikit-learn` (PCA, StandardScaler, KMeans, KNN)
* **Visualization:** `Matplotlib` (3D Scatter plots for PCA/Clusters)

## 📊 Methodology & Results

### Dataset
* **Audio:** 84 recordings from 21 different speakers to ensure variability in tone and accent.
* **Images:** Controlled dataset with white background, 10 samples per class.

### Audio Performance (5-Fold CV)
The KNN model achieved high robustness across different validation folds:

| Fold | Accuracy | Precision (Avg) | Recall (Avg) |
| :--- | :---: | :---: | :---: |
| 1 | 94.12% | 0.95 | 0.94 |
| 2 | 94.12% | 0.96 | 0.94 |
| 3 | 82.35% | 0.82 | 0.84 |
| 4 | 88.24% | 0.83 | 0.90 |
| 5 | 87.50% | 0.92 | 0.90 |
| **Avg**| **89.26%** | - | - |

### PCA Analysis
The Scree Plot analysis determined that **30 Principal Components** were optimal, balancing computational efficiency with information retention (77% explained variance).

## 🚀 How to Run

1.  **Clone the repository**
    ```bash
    git clone [https://github.com/your-username/repo-name.git](https://github.com/your-username/repo-name.git)
    ```
2.  **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Run the Main Script**
    ```bash
    python main.py
    ```
    *(Ensure you have images in `TempImagenes` or audio in `TempAudios` folders for prediction)*.

## 👤 Author

**Nicolás Sorrentino** Mechatronics Engineering Student | Embodied AI Enthusiast  
[LinkedIn](https://www.linkedin.com/in/nicolas-sorrentino)

---
*Based on the Final Report "Reconocimiento de Imágenes y Voz" (Feb 2025).*
