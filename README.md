# Skin Cancer ISIC Image Classification using Fine-Tuned DenseNet201 🩺✨

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-yellow)](https://huggingface.co/spaces/Suweeraya/SkinCancer_Classification)
[![Medium](https://img.shields.io/badge/Medium-Blog%20Post-black?logo=medium)](https://medium.com/@suweeraya/skin-cancer-classification-using-densenet-201-13ec8e8e9b6e)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

![Project Banner](https://github.com/nadiasuweer4ya/SkinCancer-ISIC-ClassificationUsing-DenseNet201/assets/135404371/9b99ddbd-e30a-4688-ae69-a056de416683)

## 📌 Project Overview
This repository contains an end-to-end medical deep learning pipeline designed to classify skin lesions into **9 distinct diagnostic categories** using a fine-tuned **DenseNet201** architecture. Trained on the benchmark **ISIC (International Skin Imaging Collaboration)** dataset, this system demonstrates high-precision computational oncology aimed at democratizing early cancer screening and supporting clinical decision-making.

### 🔗 Key Artifacts
* **Interactive Live Web App:** Deploy on [Hugging Face Spaces 🌻](https://huggingface.co/spaces/Suweeraya/SkinCancer_Classification) (Built via Gradio)
* **In-depth Technical Blog:** Read the full engineering walkthrough on [Medium 🥗](https://medium.com/@suweeraya/skin-cancer-classification-using-densenet-201-13ec8e8e9b6e)

---

## 📊 Dataset & Multi-Class Scope
The model is trained on the ISIC dataset, which addresses major class-imbalance issues inherent in medical imaging, to detect the following 9 dermatological classifications:

1. **Actinic keratosis** (Pre-cancerous)
2. **Basal cell carcinoma** (Malignant)
3. **Benign keratosis** (Pigmented/Benign)
4. **Dermatofibroma** (Benign)
5. **Melanoma** (Highly Malignant)
6. **Nevus** (Benign)
7. **Seborrheic keratosis** (Benign)
8. **Squamous cell carcinoma** (Malignant)
9. **Vascular lesion** (Benign)

---

## 🧠 Model Architecture & Technical Pipeline

The core architecture leverages transfer learning via **DenseNet201**, chosen specifically for its dense connectivity pattern that maximizes feature reuse and alleviates the vanishing-gradient problem—critical for subtle texture differentiation in medical scans.
```text
[Raw Skin Image] ──> [Resize to 75x100] ──> [Min-Max Normalization]
 |                                                   
[Softmax Output] <── [Dense Layers + Dropout] <── [Fine-Tuned DenseNet201]

```

* **Framework:** TensorFlow 2.x & Keras
* **Input Resolution:** $75 \times 100$ pixels (Optimized for computational efficiency and aspect ratio consistency)
* **Optimization:** Fine-tuned top layers with categorical cross-entropy loss to combat minority class misclassification.

---

## 🛠️ Installation & Local Deployment

Follow these instructions to clone the repository and run the interface locally:

### Prerequisites
- Python 3.9 or higher
- Pip package manager

1. Clone the Repository
```bash
git clone [https://github.com/nadiasuweer4ya/SkinCancer-ISIC-ClassificationUsing-DenseNet201.git](https://github.com/nadiasuweer4ya/SkinCancer-ISIC-ClassificationUsing-DenseNet201.git)
cd SkinCancer-ISIC-ClassificationUsing-DenseNet201
```

2. Install Dependencies
```bash
pip install -r requirements.txt
```

3. Run the Gradio Web Application
Launch the interactive interface on your localhost:
```bash
python app.py
```
After executing, navigate to http://127.0.0.1:7860 in your web browser to test your own images.

---

## 🔬 Notebooks & Experiments
This repository includes complete experimental code for transparency and reproducibility:

* **[Skin_Cancer_Classification_Using_DenseNet201.ipynb](Skin_Cancer_Classification_Using_DenseNet201.ipynb)**: Core notebook covering data exploration, preprocessing pipelines, model training, and evaluation metrics.
* **[Skin_Cancer_Comparation.ipynb](Skin_Cancer_Comparation.ipynb)**: Benchmark notebook comparing DenseNet201 performance against baseline convolutional architectures.

---

## 🌟 Inspiration & Vision

> "The prize is the pleasure of finding the thing out, the kick in the discovery, the observation that other people use it." 
> — **Richard Feynman**

This project is fundamentally driven by the philosophy of **Richard Feynman**. His passion for breaking down complex systems into their absolute truths through imagination and curiosity serves as the primary inspiration.

By applying computer science and robust deep learning architectures to healthcare infrastructure, this project aims to turn complex computational matrix operations into meaningful, life-saving clinical tools[cite: 2]. We seek to push boundaries, ask vital questions, and develop technology that makes high-tier medical screening accessible worldwide. 🌍🩺

---

## 📝 License
Distributed under the MIT License. See `LICENSE` for more information.
