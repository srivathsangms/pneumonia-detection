
<div align="center">

# 🫁 Pneumonia Detection using Deep Learning

**Classifying chest X-rays as Normal or Pneumonia using Transfer Learning (MobileNetV2), with Grad-CAM explainability.**

![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-FF6F00?logo=tensorflow&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/License-Educational-lightgrey)

</div>

---

## 📖 Overview

Pneumonia is a serious lung infection that can be diagnosed from chest X-ray images. This project uses a **transfer learning** approach — built on **MobileNetV2** pretrained on ImageNet — to classify chest X-rays as:

- 🟢 **Normal**
- 🔴 **Pneumonia**

To make the model's decisions interpretable, **Grad-CAM** is used to visualize the exact lung regions that influenced each prediction, highlighting *why* the model made its call rather than treating it as a black box.

---

## ✨ Features

- 🧠 Transfer learning on **MobileNetV2** for fast, accurate binary classification
- 🔥 **Grad-CAM** heatmaps for visual model explainability
- 🖼️ Image augmentation (rotation, zoom, shifts, flips) for better generalization
- 📊 Trained and evaluated on the Kaggle Chest X-ray Pneumonia dataset
- 🌐 Streamlit-ready for an interactive web demo

---

## 🏗️ Model Architecture

| Component | Details |
|---|---|
| **Base model** | MobileNetV2 (pretrained on ImageNet) |
| **Custom head** | Global Average Pooling → Dense (ReLU) → Dropout → Dense (Sigmoid) |
| **Loss function** | Binary Crossentropy |
| **Optimizer** | Adam |
| **Input size** | 224 × 224 |
| **Batch size** | 32 |
| **Epochs** | 10 |

**Data augmentation:** rotation, zoom, width shift, height shift, horizontal flip.

---

## 🗂️ Dataset

Trained on the **[Chest X-ray Pneumonia dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)** (Kaggle).

Expected folder structure:

```
chest_xray/
│
├── train/
├── val/
└── test/
```

---

## 🚀 Getting Started

### 1. Install dependencies

```bash
pip install tensorflow opencv-python matplotlib seaborn scikit-learn streamlit
```

### 2. Download the dataset

Download the [Chest X-ray Pneumonia dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) from Kaggle and place the `chest_xray/` folder in the project directory.

### 3. Run the project

```bash
python sri_med_project.py
```

The trained model is saved as `pneumonia_model.h5`.

---

## 🔍 Grad-CAM Visualization

Grad-CAM overlays a heatmap on each X-ray showing which regions most influenced the model's prediction — useful for:

- Verifying the model is focusing on lung regions, not artifacts
- Building trust in the model's predictions
- Making the classification explainable to non-technical users

---

## 🛠️ Tech Stack

`Python` · `TensorFlow / Keras` · `OpenCV` · `Matplotlib` · `NumPy` · `Scikit-learn` · `Streamlit`

---

## 🔮 Future Improvements

- [ ] Train for more epochs to improve accuracy
- [ ] Deploy a full interactive web app using Streamlit
- [ ] Improve UI and add richer user input/upload features

---

## 👤 Author

**Srivathsan GMS**
B.Tech Artificial Intelligence & Data Science Student

---

## 📄 License

This project is for **educational purposes**.
