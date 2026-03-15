import os
import cv2
import csv
import mediapipe as mp
import numpy as np
from tqdm import tqdm

FRAME_DIR = "Dataset/frames"
OUTPUT_FILE = "Dataset/landmarks.csv"

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True,
                       max_num_hands=1, 
                       min_detection_confidence=0.5)

data = []

words = os.listdir(FRAME_DIR)

print(f"Starting Pro-Level landmark extraction for {len(words)} words...")

def get_distance(p1, p2):
    return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2)

for word in tqdm(words):
    word_path = os.path.join(FRAME_DIR, word)
    if not os.path.isdir(word_path):
        continue

    for img_name in os.listdir(word_path):
        img_path = os.path.join(word_path, img_name)
        image = cv2.imread(img_path)
        if image is None: continue

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            
            # --- ADVANCED NORMALIZATION ---
            # 1. Translation: Wrist (0) becomes (0,0,0)
            wrist = hand_landmarks.landmark[0]
            
            # 2. Scaling: Normalize by the distance between wrist (0) and middle finger MCP (9)
            # This makes the hand size constant regardless of distance to camera
            scale_factor = get_distance(hand_landmarks.landmark[0], hand_landmarks.landmark[9])
            if scale_factor == 0: scale_factor = 1
            
            row = []
            for lm in hand_landmarks.landmark:
                # Subtract wrist and divide by scale factor
                row.append((lm.x - wrist.x) / scale_factor)
                row.append((lm.y - wrist.y) / scale_factor)
                row.append((lm.z - wrist.z) / scale_factor)

            row.append(word)
            data.append(row)

# Save to CSV
with open(OUTPUT_FILE, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(data)

print(f"Extraction complete! Saved {len(data)} pro-normalized samples to {OUTPUT_FILE}")