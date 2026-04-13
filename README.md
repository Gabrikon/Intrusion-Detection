# 🛡️ Acoustic Intrusion Detection System (A-IDS)

A hierarchical deep learning framework designed to identify security-relevant audio events such as glass breaking, gunshots, drilling, and jackhammers using Convolutional Neural Networks (CNNs).

---

## 📌 Project Overview
Traditional security systems often rely on visual cues, which can be limited by lighting conditions or physical obstacles. This project implements an **Acoustic Intrusion Detection System** that monitors environmental audio to trigger alerts. 

The core innovation is a **two-stage hierarchical classification pipeline**:
1.  **Stage 1 (Binary Detector):** Filters ambient background noise from potential intrusion sounds to minimize false positives.
2.  **Stage 2 (Multiclass Classifier):** Specifically identifies the type of intrusion once a threat is detected.



---

## 🏗️ System Architecture
The system processes audio through a structured pipeline:
* **Preprocessing:** Audio is resampled to **16 kHz mono** and segmented into **2-second clips**.
* **Feature Extraction:** Raw waveforms are converted into **Log-Mel Spectrograms** (128 mel bands), transforming the sound into a 2D representation suitable for Computer Vision techniques.
* **Model:** A Sequential CNN architecture optimized for spatial pattern recognition in spectrograms.



---

## 📊 Datasets
The models were trained using a combination of high-quality environmental audio datasets:
* **ESC-50:** 2,000 environmental recordings categorized into 50 classes.
* **UrbanSound8K:** 8,732 urban sound excerpts from 10 social-ecological classes.
* **Custom Augmentation:** Standardized time-padding and resampling to ensure consistency across all input tensors.

---

## 🚀 Key Features
* **Hierarchical Logic:** Stage 2 (Multiclass) only activates if Stage 1 (Binary) triggers a positive detection, saving computational resources.
* **Persistence-First Design:** The training pipeline utilizes a manifest system to log and save artifacts (models, plots, and metrics) to Google Drive sequentially, ensuring progress is never lost.
* **Streamlit Integration:** Designed for deployment as a web application for real-time inference via microphone or file upload.

---

## 📈 Performance & Evaluation
The system is evaluated based on:
* **Precision/Recall:** Tuned specifically to minimize "False Negatives" (missed intrusions).
* **Confusion Matrix:** Analyzed to ensure the model distinguishes between high-energy mechanical sounds (e.g., Drilling vs. Jackhammers).
* **Binary Thresholding:** Adjustable sensitivity in Stage 1 to adapt to different background noise environments.
