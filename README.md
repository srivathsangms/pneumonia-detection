# Pneumonia Detection using Deep Learning

This project detects **Pneumonia from Chest X-ray images** using a **MobileNetV2 Deep Learning model**.
It also visualizes model predictions using **Grad-CAM** to highlight important regions in the X-ray.

---

## Project Overview

Pneumonia is a lung infection that can be identified using chest X-rays.
In this project, a **Transfer Learning** approach is used to classify X-ray images into:

* Normal
* Pneumonia

The model is trained on the **Chest X-ray Pneumonia dataset from Kaggle**.

---

## Technologies Used

* Python
* TensorFlow / Keras
* OpenCV
* Matplotlib
* NumPy
* Scikit-learn
* Streamlit

---

## Dataset

Dataset used:
Chest X-ray Pneumonia Dataset (Kaggle)

Folder structure:
chest_xray/
│
├── train/
├── val/
└── test/

---

## Model Architecture

Base Model: MobileNetV2 (Pretrained on ImageNet)

Custom Layers:

* Global Average Pooling
* Dense Layer (ReLU)
* Dropout
* Output Layer (Sigmoid)

Loss Function: Binary Crossentropy
Optimizer: Adam

---

## Training Details

* Image size: 224x224
* Batch size: 32
* Epochs: 10

Data augmentation used:

* Rotation
* Zoom
* Width shift
* Height shift
* Horizontal flip

---

## Grad-CAM Visualization

Grad-CAM is used to visualize:

* Which part of the lung image influenced prediction
* Helps in model explainability

---

## Installation

Install dependencies:

pip install tensorflow opencv-python matplotlib seaborn scikit-learn streamlit

---

## How to Run

1. Download dataset from Kaggle
2. Place dataset in project folder
3. Run the Python file:

python sri_med_project.py

---

## Model File

Saved model:
pneumonia_model.h5

---

## Future Improvements

* Increase epochs for better accuracy
* Deploy full web app using Streamlit
* Improve UI and user input features

---

## Author

B.Tech AI & DS Student

---

## License

This project is for educational purposes.
