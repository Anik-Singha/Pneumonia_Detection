# 🫁 Chest X-ray Pneumonia Detection using ResNet50

A Deep Learning web application that detects **Pneumonia** from Chest X-ray images using **Transfer Learning with ResNet50**. The project includes data preprocessing, model training, evaluation, and a user-friendly Streamlit interface for real-time prediction.

---

## 📌 Features

- Deep Learning based Pneumonia Detection
- Transfer Learning using ResNet50
- Data Augmentation
- Model Evaluation using Classification Report & Confusion Matrix
- Interactive Streamlit Web Application
- Upload Chest X-ray and receive instant prediction
- Displays prediction confidence and class probabilities

---

## 🖼 Demo

### Streamlit Interface


![App Screenshot](../../OneDrive/Pictures/Screenshots/Screenshot%202026-07-07%20123125.png)

---

## 📂 Dataset

Dataset: **Chest X-ray Images (Pneumonia)**

Classes:

- NORMAL
- PNEUMONIA

Dataset Structure

```
train/
    NORMAL/
    PNEUMONIA/

test/
    NORMAL/
    PNEUMONIA/
```

---

## 🧠 Model Architecture

Transfer Learning with **ResNet50**

Architecture:

```
Input Image (224×224×3)
        │
Data Augmentation
        │
ResNet50 (ImageNet Weights)
        │
GlobalAveragePooling2D
        │
Dropout (0.3)
        │
Dense (128, ReLU)
        │
Dropout (0.3)
        │
Dense (2, Softmax)
```

---

## ⚙ Technologies Used

- Python
- TensorFlow
- Keras
- Streamlit
- NumPy
- Matplotlib
- Scikit-learn
- Pillow

---

## 📊 Model Performance

| Metric | Value |
|---------|------:|
| Test Accuracy | **95.38%** |
| Test Loss | **0.1298** |

Classification Report

| Class | Precision | Recall | F1-score |
|------|----------:|-------:|---------:|
| NORMAL | 0.90 | 0.91 | 0.91 |
| PNEUMONIA | 0.97 | 0.97 | 0.97 |

---

## 📈 Training Curves

Add your accuracy/loss graph.

```
outputs/training_curves.png
```

---

## 📉 Confusion Matrix

Add your confusion matrix.

```
outputs/confusion_matrix.png
```

---

## 🚀 Installation

Clone the repository

```bash
git clone https://github.com/YourUsername/Pneumonia_Detection.git

cd Pneumonia_Detection
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶ Running the Application

```bash
streamlit run app.py
```

---

## 📁 Project Structure

```
Pneumonia_Detection/
│
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│
├── models/
│   └── resnet50_pneumonia.keras
│
├── notebooks/
│   ├── 01_dataset_exploration.ipynb
│   ├── 02_model_training.ipynb
│
└── outputs/
```

---

## 📌 Future Improvements

- Grad-CAM Explainability
- Fine-tune ResNet50
- Deploy on Streamlit Community Cloud
- Support DICOM images
- Multi-class Chest Disease Detection

---

## 👨‍💻 Author

**Anik Singha**

GitHub: https://github.com/Anik-Singha

LinkedIn: www.linkedin.com/in/anik-singha2001

---

## ⭐ If you found this project useful, consider giving it a star!