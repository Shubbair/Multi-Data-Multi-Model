import os
import sys
import cv2
from tqdm import tqdm
import numpy as np
import mediapipe as mp

SRC_ROOT = "_rgb/"
OUT_ROOT = "_keypoints/"

EXTS = (".png", ".jpg", ".jpeg")
NUM_LANDMARKS = 21

os.makedirs(OUT_ROOT, exist_ok=True)


def create_keypoint_images(src_root: str, dst_root: str, max_per_folder: int = None):
    # gather all image paths first (so tqdm shows total)
    image_entries = []
    for subdir, _, files in os.walk(src_root):
        rel = os.path.relpath(subdir, src_root)
        dst_sub = os.path.join(dst_root, rel) if rel != "." else dst_root
        os.makedirs(dst_sub, exist_ok=True)
        # filter image files
        image_files = [f for f in files if f.lower().endswith(EXTS)]
        if max_per_folder:
            image_files = image_files[:max_per_folder]
        for fname in image_files:
            src_path = os.path.join(subdir, fname)
            dst_path = os.path.join(dst_sub, os.path.splitext(fname)[0] + ".png")
            image_entries.append((src_path, dst_path))

    if len(image_entries) == 0:
        print("No images found in", src_root, file=sys.stderr)
        return

    mp_drawing = mp.solutions.drawing_utils

    with mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5) as hands:
        for src_path, dst_path in tqdm(image_entries, desc="keypoints"):
            img = cv2.imread(src_path)
            if img is None:
                print("Skipping unreadable:", src_path, file=sys.stderr)
                continue
            h, w = img.shape[:2]
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            try:
                results = hands.process(img_rgb)
            except Exception as e:
                print(f"MediaPipe failed for {src_path}: {e}", file=sys.stderr)
                continue

            if not results or not results.multi_hand_landmarks:
                # no hand detected
                continue

            # draw landmarks on a blank RGB canvas for clarity
            viz = np.zeros((h, w, 3), dtype=np.uint8)
            mp_drawing.draw_landmarks(viz, results.multi_hand_landmarks[0], mp.solutions.hands.HAND_CONNECTIONS)

            # convert RGB->BGR for cv2.imwrite
            viz_bgr = cv2.cvtColor(viz, cv2.COLOR_RGB2BGR)
            success = cv2.imwrite(dst_path, viz_bgr)
            if not success:
                print("Failed to write:", dst_path, file=sys.stderr)

    print("Keypoint images saved to:", dst_root)


if __name__ == "__main__":
    create_keypoint_images(SRC_ROOT, OUT_ROOT, max_per_folder=50)