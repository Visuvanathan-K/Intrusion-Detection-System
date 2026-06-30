# Intrusion Detection System using LSTM

## Overview

# OCR Text Extraction App

## Live Demo

🔗 https://intrusion-detection-system-cv9rr8m3nhghmbov7j6vbc.streamlit.app/

This project is an AI-powered Intrusion Detection System developed using a Long Short-Term Memory (LSTM) deep learning model.

The system analyzes network traffic data and predicts whether the traffic belongs to a normal connection or an intrusion attack.

A Streamlit web interface allows users to upload network traffic files and receive predictions instantly.

---

## Features

- Data preprocessing
- Feature scaling
- LSTM Deep Learning Model
- Model Training
- Model Evaluation
- Confusion Matrix
- Accuracy Graph
- Loss Graph
- Predict on New Files
- Streamlit Web Application

---

## Dataset

Dataset Used:

Darknet Network Traffic Dataset

CSV Format

---

## Technologies

- Python
- TensorFlow / Keras
- Streamlit
- Pandas
- NumPy
- Matplotlib
- Scikit-Learn

---

## Folder Structure

```
src/
    preprocess.py
    model.py
    train.py
    evaluate.py
    predict.py

models/
outputs/

app.py
main.py
```

---

## Installation

Clone the repository

```bash
git clone https://github.com/Visuvanathan-K/Intrusion-Detection-System.git
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run Streamlit

```bash
streamlit run app.py
```

Run Console Version

```bash
python main.py
```

---

## Workflow

1. Load Dataset
2. Preprocess Data
3. Train LSTM Model
4. Save Model
5. Evaluate Accuracy
6. Upload New Network Traffic File
7. Predict Attack Type

---

## Output

- Training Accuracy Graph
- Training Loss Graph
- Confusion Matrix
- Predicted Attack Labels

---

## Future Improvements

- Real-time Network Monitoring
- Support Multiple File Formats
- Explainable AI (XAI)
- Cloud Deployment
- Live Dashboard

---

## Author

Visuvanathan K
