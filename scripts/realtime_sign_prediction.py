
import cv2
import numpy as np
import pickle
from collections import deque

import mediapipe as mp
from tensorflow.keras.models import load_model


# ----------------------------
# Load model and labels
# ----------------------------

model = load_model("models/sign_lstm_model.h5")

with open("models/labels.pkl", "rb") as f:
    label_encoder = pickle.load(f)


# ----------------------------
# MediaPipe Setup
# ----------------------------

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ----------------------------
# Sequence buffer
# ----------------------------

SEQUENCE_LENGTH = 20
sequence = deque(maxlen=SEQUENCE_LENGTH)

# ----------------------------
# Webcam
# ----------------------------

cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.flip(frame, 1)

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(image_rgb)

    if results.multi_hand_landmarks:

        for hand_landmarks in results.multi_hand_landmarks:

            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            landmarks = []

            for lm in hand_landmarks.landmark:
                landmarks.append(lm.x)
                landmarks.append(lm.y)
                landmarks.append(lm.z)

            sequence.append(landmarks)

            if len(sequence) == SEQUENCE_LENGTH:

                input_data = np.array(sequence)
                input_data = np.expand_dims(input_data, axis=0)

                prediction = model.predict(input_data, verbose=0)

                class_id = np.argmax(prediction)
                word = label_encoder.inverse_transform([class_id])[0]

                confidence = np.max(prediction)

                text = f"{word} ({confidence:.2f})"

                cv2.putText(
                    frame,
                    text,
                    (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0,255,0),
                    2
                )

    cv2.imshow("Sign Recognition", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break


cap.release()
cv2.destroyAllWindows()
