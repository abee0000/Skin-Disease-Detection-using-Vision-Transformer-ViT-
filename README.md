<h1 align="center">Skin Disease Detection Based on Vision Transformer (ViT)</h1>

<p align="center">
  <strong>AI-Powered Dermatology: Leveraging Vision Transformers for Accurate Skin Disease Diagnosis</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10-blue?logo=python" />
  <img src="https://img.shields.io/badge/TensorFlow-2.17-orange?logo=tensorflow" />
  <img src="https://img.shields.io/badge/Transformers-HuggingFace-yellow?logo=huggingface" />
  <img src="https://img.shields.io/badge/License-MIT-green" />
</p>

---

## Overview
Skin diseases affect millions globally, and early detection plays a crucial role in preventing severe complications.  
This project introduces a **deep learning-based diagnostic system** using **Vision Transformers (ViT)** — a state-of-the-art model architecture renowned for its ability to understand global image dependencies better than traditional Convolutional Neural Networks (CNNs).

Through extensive experimentation, we developed, trained, and deployed a Vision Transformer model capable of classifying **four major skin conditions** with **91.19% test accuracy**.

---

##  Key Highlights
- **90%+ classification accuracy** using ViT-B/16 architecture  
- **Comparison between CNN and ViT models**  
- **Desktop application with Tkinter** for real-time disease classification  
- **Confidence score visualization** for user interpretability  
- **Dataset of 8,000 labeled dermoscopic images**  

---

## Project Architecture

### Workflow

### Vision Transformer Pipeline
1. **Image Tokenization:** Split input image into 16×16 patches  
2. **Linear Embedding:** Flatten and embed patches into 1D tokens  
3. **Positional Encoding:** Retain spatial structure  
4. **Self-Attention:** Each patch attends to all others for global context  
5. **Classification Head:** Final layer predicts disease category  

---

## Repository Structure

---

## Tech Stack

| Category | Tools Used |
|-----------|-------------|
| Programming Language | Python 3.10+ |
| Frameworks | TensorFlow, HuggingFace Transformers |
| ML Libraries | Scikit-learn, NumPy, Pandas |
| Visualization | Matplotlib, Seaborn |
| GUI | Tkinter, tkinterDnD2 |
| Environment | Jupyter Notebook / VSCode |
| OS | Windows / Linux / macOS |

---

## Model Summary

| Model | Accuracy | F1-Score | AUC |
|--------|-----------|----------|------|
| CNN (Benchmark) | 78.3% | 77% | 0.88 |
| ViT-B16 | **91.19%** | **90%+** | **0.99** |

**ROC-AUC by Class:**
- **Basal Cell Carcinoma (BCC):** 0.99  
- **Benign Keratosis (BKL):** 0.96  
- **Melanoma (MEL):** 1.00  
- **Melanocytic Nevi (NV):** 0.99  

---

## Visual Results

### Accuracy vs Loss
![Accuracy and Loss Curve](https://github.com/user-attachments/assets/6dea8004-a356-4dfb-9221-b1e9940a98d6)


### Confusion Matrix
![Confusion Matrix](https://github.com/user-attachments/assets/8dde5f17-58e0-4d9a-af6b-3b8b66147846)


### ROC Curves
![ROC Curve](https://github.com/user-attachments/assets/87069875-74c4-409d-9041-164d2ca323c0)


### Accuracy predicted
![op0](https://github.com/user-attachments/assets/781b68cc-37d6-4aa7-b112-f7127c4539d2)


###Real Time Output Result
![op1](https://github.com/user-attachments/assets/fa601f5e-1363-410c-a578-e3d608563878)
![op2](https://github.com/user-attachments/assets/d40b572e-2d41-4f73-aa64-746dc1b69e2f)





)

---
