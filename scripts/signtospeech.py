import cv2
import numpy as np
import pickle
import time
import pyttsx3
import threading
from collections import deque
import mediapipe as mp
from tensorflow.keras.models import load_model

# ----------------------------
# 1. Non-Blocking Text-to-Speech
# ----------------------------
class TTSWorker:
    def __init__(self):
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 160)
        self.is_speaking = False

    def _speak(self, text):
        self.is_speaking = True
        self.engine.say(text)
        self.engine.runAndWait()
        self.is_speaking = False

    def say(self, text):
        if not self.is_speaking:
            # Threading prevents the camera from freezing while talking
            threading.Thread(target=self._speak, args=(text,), daemon=True).start()

tts = TTSWorker()

# ----------------------------
# 2. Setup and Model Loading
# ----------------------------
MODEL_PATH = "models/sign_lstm_model.h5"
LABEL_PATH = "models/labels.pkl"

try:
    model = load_model(MODEL_PATH)
    with open(LABEL_PATH, "rb") as f:
        label_encoder = pickle.load(f)
    print(f"Success: Model and all 100 labels loaded.")
except Exception as e:
    print(f"Error loading model files: {e}")
    exit()

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.75,
    min_tracking_confidence=0.75
)

# ----------------------------
# 3. Parameters & Logic
# ----------------------------
SEQUENCE_LENGTH = 20 # Matches your current sign_lstm_model.h5
sequence = deque(maxlen=SEQUENCE_LENGTH)
stability_buffer = deque(maxlen=10) # Used to smooth out flickering

last_spoken_word = ""
last_speech_time = 0
CONFIDENCE_THRESHOLD = 0.85 
SPEECH_COOLDOWN = 2.0 

def get_distance(p1, p2):
    return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2)

def extract_normalized_landmarks(hand_landmarks):
    """
    Standardizes the hand size and position.
    Wrist (0) becomes the center.
    Distance to Middle-Finger MCP (9) scales the hand.
    """
    wrist = hand_landmarks.landmark[0]
    ref_point = hand_landmarks.landmark[9] 
    scale_factor = get_distance(wrist, ref_point)
    if scale_factor == 0: scale_factor = 1
    
    normalized = []
    for lm in hand_landmarks.landmark:
        normalized.extend([
            (lm.x - wrist.x) / scale_factor,
            (lm.y - wrist.y) / scale_factor,
            (lm.z - wrist.z) / scale_factor
        ])
    return normalized

# ----------------------------
# 4. Processing Loop
# ----------------------------
cap = cv2.VideoCapture(0)

print("Starting Real-time Translation. Press 'q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    current_display_word = "..."
    current_display_conf = 0.0

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Normalize landmarks (21 points * 3 dims = 63 features)
            landmarks_63 = extract_normalized_landmarks(hand_landmarks)
            sequence.append(landmarks_63)

            if len(sequence) == SEQUENCE_LENGTH:
                # Prepare input (Batch, Sequence, Features)
                input_data = np.expand_dims(list(sequence), axis=0)
                predictions = model.predict(input_data, verbose=0)[0]
                
                class_id = np.argmax(predictions)
                conf = predictions[class_id]
                current_display_conf = conf
                
                # Stability Filter
                stability_buffer.append(class_id)
                if len(stability_buffer) == 10:
                    # Find most frequent prediction in the buffer
                    stable_id = max(set(stability_buffer), key=list(stability_buffer).count)
                    
                    # If stable for 7/10 frames and high confidence
                    if stability_buffer.count(stable_id) >= 7 and conf > CONFIDENCE_THRESHOLD:
                        word = label_encoder.inverse_transform([stable_id])[0]
                        current_display_word = word
                        
                        # Trigger Speech
                        now = time.time()
                        if (word != last_spoken_word) or (now - last_speech_time > SPEECH_COOLDOWN):
                            tts.say(word)
                            last_spoken_word = word
                            last_speech_time = now
                            sequence.clear() # Force fresh start for next word

    # Visual Interface
    cv2.rectangle(frame, (0, h-90), (450, h), (40, 40, 40), -1)
    txt_color = (0, 255, 0) if current_display_word != "..." else (0, 165, 255)
    
    cv2.putText(frame, f"SIGN: {current_display_word.upper()}", (20, h-55), 
                cv2.FONT_HERSHEY_DUPLEX, 1.1, txt_color, 2)
    cv2.putText(frame, f"CONF: {current_display_conf:.2f}", (20, h-20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    cv2.imshow('Sign to Speech Pro', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()