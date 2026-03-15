# Real-Time Sign Language Recognition and Speech Conversion

## Overview

This project is a Real-Time Sign Language Recognition system that converts hand gestures into text and speech using Deep Learning and Computer Vision. The system helps bridge the communication gap between deaf or mute individuals and people who do not understand sign language.

The application captures hand gestures through a webcam, detects hand landmarks using MediaPipe, and classifies the gestures using a trained deep learning model built with TensorFlow/Keras. Once a gesture is recognized, the predicted word is converted into text and then spoken aloud using a Text-to-Speech engine.

## Features

* Real-time hand gesture detection
* Sign language recognition using deep learning
* Hand landmark extraction using MediaPipe
* Text output for recognized gestures
* Speech output using Text-to-Speech
* Simple and efficient real-time communication system

## Technologies Used

* Python
* OpenCV
* MediaPipe
* TensorFlow / Keras
* NumPy
* Scikit-learn
* pyttsx3 (Text-to-Speech)

## Dataset

The model is trained using the WLASL processed dataset from Kaggle, which contains labeled sign language videos used for gesture recognition.

Dataset Link:
https://www.kaggle.com/datasets/risangbaskoro/wlasl-processed

This guide provides a comprehensive `README.md` for your **WLASL Sign-to-Speech** project. It outlines the pipeline from data extraction to real-time inference using the files you provided.


## 🚀 System Architecture

1. **Frame Extraction**: Sampling videos into discrete images.
2. **Landmark Extraction**: Using MediaPipe to extract and normalize 63 hand coordinates ($21 \text{ landmarks} \times 3 \text{ dimensions}$).
3. **LSTM Training**: Training a Many-to-One sequence model to classify sign movements.
4. **Inference**: Real-time webcam translation with a stability filter and non-blocking audio.

---

## 🛠️ Installation

Ensure you have Python 3.8+ installed.

```bash
pip install opencv-python mediapipe tensorflow pandas numpy scikit-learn tqdm pyttsx3 flask-socketio

```

---

## 📂 Project Pipeline

### 1. Data Preparation

First, extract frames from your WLASL video dataset. This script skips frames to capture essential movement without redundancy.

* **Script:** `extract_frames.py`
* **Action:** Reads `WLASL_v0.3.json`, processes up to 2000 words, and saves every 5th frame.

### 2. Feature Engineering (Landmarks)

Convert images into numerical data. We use **Advanced Normalization** to ensure the model is invariant to hand size or distance from the camera.

* **Script:** `land.py`
* **Logic:** * **Translation:** Wrist (Landmark 0) becomes $(0,0,0)$.
* **Scaling:** Coordinates are divided by the distance between the wrist and the middle finger MCP.



### 3. Model Training

Train a stacked LSTM model with Data Augmentation (Gaussian noise and scaling) to improve robustness.

* **Script:** `train.py`
* **Architecture:** * LSTM (128 units, sequences) $\rightarrow$ BatchNormalization $\rightarrow$ Dropout
* LSTM (64 units) $\rightarrow$ BatchNormalization $\rightarrow$ Dropout
* Dense (64, ReLU) $\rightarrow$ Softmax Output.



### 4. Real-Time Translation

Run the live interface that captures webcam feed and speaks the recognized word.

* **Option A (Desktop):** `signtospeech.py`
* Includes a stability buffer (7/10 frames must match) to prevent "flickering" predictions.
* Uses a threaded TTS worker to prevent camera lag during speech.


* **Option B (Web/Socket):** `app.py`
* A Flask-SocketIO server for streaming landmarks from a web-based frontend.



---

## 📊 Configuration

You can tune the following parameters in the scripts:
| Parameter | Default Value | Description |
| :--- | :--- | :--- |
| `SEQUENCE_LENGTH` | 20 | Number of consecutive frames used for one prediction. |
| `CONFIDENCE_THRESHOLD` | 0.85 | Minimum model certainty required to trigger speech. |
| `FRAME_SKIP` | 5 | Interval of frames extracted from raw videos. |
| `STABILITY_WINDOW` | 10 | The look-back window to ensure a sign is consistent. |

---

## 📝 Usage Note

To run the real-time predictor, ensure your trained model and labels are in the `models/` folder:

```text
models/
├── sign_lstm_model.h5
└── labels.pkl



This project is licensed under the MIT License.
