import os
import json
import cv2
from tqdm import tqdm

JSON_PATH = "Dataset/WLASL_v0.3.json"
VIDEO_DIR = "Dataset/videos"
OUTPUT_DIR = "Dataset/frames"

os.makedirs(OUTPUT_DIR, exist_ok=True)

with open(JSON_PATH, "r") as f:
    data = json.load(f)

print("Total words in dataset:", len(data))

LIMIT = 2000
FRAME_SKIP = 5

for item in tqdm(data[:LIMIT]):

    word = item["gloss"]
    instances = item["instances"]

    for inst in instances:

        video_id = inst["video_id"]
        video_file = f"{int(video_id):05d}.mp4"

        video_path = os.path.join(VIDEO_DIR, video_file)

        if not os.path.exists(video_path):
            continue

        cap = cv2.VideoCapture(video_path)

        save_dir = os.path.join(OUTPUT_DIR, word)
        os.makedirs(save_dir, exist_ok=True)

        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % FRAME_SKIP == 0:
                frame_name = f"{video_id}_{frame_count}.jpg"
                frame_path = os.path.join(save_dir, frame_name)
                cv2.imwrite(frame_path, frame)

            frame_count += 1

        cap.release()

print("Frame extraction finished.")