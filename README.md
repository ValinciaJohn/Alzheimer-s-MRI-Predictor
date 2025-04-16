# 🧠 Alzheimer’s MRI Predictor

A Streamlit-based application for classifying Alzheimer's Disease (AD), Mild Cognitive Impairment (MCI), and Cognitively Normal (CN) individuals using brain MRI scans and clinical features (age, sex). This multimodal deep learning project combines powerful models and educational tools to support early AD diagnosis and awareness.

---

## 📌 Problem Statement

Alzheimer’s disease and its precursor, Mild Cognitive Impairment (MCI), are difficult to diagnose early due to subtle MRI changes and clinical similarities. Misclassification, especially of MCI, delays treatment. This project addresses these challenges by building a robust classification model and user-friendly interface.

---

## 📄 Abstract

**Alzheimer’s Assistant** is a Streamlit app that integrates an ensemble of ResNet50, DenseNet121, and EfficientNet-B0 to classify MRI scans and clinical features into AD, MCI, or CN. A SMOTE-balanced dataset (~1197 samples) ensures equal class distribution. The ensemble achieves:
- **Macro F1**: 0.8008
- **AD Recall**: 0.8875
- **MCI Recall**: 0.7037 (Improved from EfficientNet’s 0.6049)

The app includes an interactive chatbot for Alzheimer’s-related Q&A to enhance user engagement and awareness.

---

## 📊 Dataset

Sourced from the **ADNI (Alzheimer’s Disease Neuroimaging Initiative)** project. It includes MRI scans and clinical data for:
- **AD** (Alzheimer’s Disease)
- **MCI** (Mild Cognitive Impairment)
- **CN** (Cognitively Normal)

📎 [ADNI 1 Standardized 1.5T List – Complete 1 Year](#)

---

## 🔍 Literature Reference

**Paper**: *Automated Alzheimer’s disease classification using deep learning models with Soft-NMS and improved ResNet50 integration*  
Summary:  
The study proposes an AD classification model combining improved ResNet50, Soft-NMS in Faster R–CNN, and Bi-GRU for sequential MRI feature extraction. The model achieved **98.91% accuracy** on AD vs CN classification using the ADNI dataset.

---

## ⚙️ Methods & Implementation

### Data Preprocessing
- `.nii` MRI files converted to 224×224 grayscale PNGs (middle slice)
- Normalization:
  - Training: mean = 0.5, std = 0.5
  - Inference: mean = 0, std = 1
- Clinical Features:
  - Age normalized to [0, 1]
  - Sex one-hot encoded
- **SMOTE** used to balance training data (~239 samples per class)

### Models
- **ResNet50**: 2048 features + 64 clinical, CE loss
- **DenseNet121**: 1024 features, CE loss
- **EfficientNet-B0**: 1280 features, Focal loss (γ = 2.0)
- **Ensemble (Soft Voting)**: Equal weight average of outputs

### Training
- Optimizer: Adam
- Learning Rate: 3e-4
- Batch Size: 16
- Weighted sampling (MCI weight = 1.5)

---

## ✅ Results

### Achievements
- MCI recall improved to **0.7037** using ensemble
- AD recall reached **0.8875** (clinically relevant)
- Streamlit app built for accessible inference and Q&A

### Areas for Improvement
- MCI recall below target (0.75–0.80)
- Macro F1 (0.8008) lower than ResNet50 alone (0.8502)
- CN precision (0.7416) shows some confusion
- Chatbot not yet integrated with prediction context

---

## 🚀 App Features

- Upload `.nii` MRI + input age/sex → get AD/MCI/CN prediction
- View middle-slice MRI image
- Chatbot with 22 Alzheimer’s-related Q&A entries
- Clean two-column layout for results & interaction

---

## 🧠 Learnings

- MCI classification remains challenging due to clinical overlap and subtle features
- Weighted ensemble (e.g., 0.5 ResNet50, 0.3 DenseNet121, 0.2 EfficientNet) could improve results
- Multi-slice input and normalization alignment needed for robust deployment

---


