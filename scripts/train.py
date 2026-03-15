import os
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# -----------------------------
# Configuration
# -----------------------------
DATASET_PATH = "Dataset/landmarks.csv"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "sign_lstm_model.h5")
LABEL_PATH = os.path.join(MODEL_DIR, "labels.pkl")
SEQUENCE_LENGTH = 20

os.makedirs(MODEL_DIR, exist_ok=True)

# -----------------------------
# Data Augmentation Logic
# -----------------------------
def augment_data(X_data, y_data):
    """
    Adds Gaussian noise and scaling to landmarks to simulate 
    camera jitter and different hand sizes.
    """
    augmented_X = []
    augmented_y = []
    
    for i in range(len(X_data)):
        # Original data
        augmented_X.append(X_data[i])
        augmented_y.append(y_data[i])
        
        # 1. Add slight noise
        noise = np.random.normal(0, 0.005, X_data[i].shape)
        augmented_X.append(X_data[i] + noise)
        augmented_y.append(y_data[i])
        
        # 2. Slight scaling (simulating distance from camera)
        scale = np.random.uniform(0.95, 1.05)
        augmented_X.append(X_data[i] * scale)
        augmented_y.append(y_data[i])
        
    return np.array(augmented_X), np.array(augmented_y)

# -----------------------------
# Load and Prepare Data
# -----------------------------
data = pd.read_csv(DATASET_PATH, header=None)
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Encoding labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Group into sequences
X_seq, y_seq = [], []
for i in range(len(X) - SEQUENCE_LENGTH):
    # Only group if the labels are the same across the sequence window
    if len(set(y[i:i+SEQUENCE_LENGTH])) == 1:
        X_seq.append(X[i:i+SEQUENCE_LENGTH])
        y_seq.append(y_encoded[i])

X_seq = np.array(X_seq)
y_seq = np.array(y_seq)

# Apply Augmentation
print(f"Original sequence count: {len(X_seq)}")
X_seq, y_seq = augment_data(X_seq, y_seq)
print(f"Augmented sequence count: {len(X_seq)}")

y_cat = to_categorical(y_seq)
X_train, X_test, y_train, y_test = train_test_split(X_seq, y_cat, test_size=0.2, random_state=42)

# -----------------------------
# Build/Refine Model
# -----------------------------
# If you want to continue training the existing model, use: model = load_model(MODEL_PATH)
model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(SEQUENCE_LENGTH, X_train.shape[2])),
    BatchNormalization(),
    Dropout(0.3),
    LSTM(64, return_sequences=False),
    BatchNormalization(),
    Dropout(0.3),
    Dense(64, activation="relu"),
    Dense(y_cat.shape[1], activation="softmax")
])

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# -----------------------------
# Training
# -----------------------------
callbacks = [
    EarlyStopping(monitor="val_loss", patience=12, restore_best_weights=True),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6)
]

model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=100,
    batch_size=32,
    callbacks=callbacks
)

# Save
model.save(MODEL_PATH)
with open(LABEL_PATH, "wb") as f:
    pickle.dump(label_encoder, f)

print("Training complete with augmentation.")