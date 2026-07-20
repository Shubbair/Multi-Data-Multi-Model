#!/usr/bin/env python3.11

import os 
import cv2
import shutil
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils import *

def create_edge_images(src_root: str, dst_root: str, low_thresh: int = 100, high_thresh: int = 200):
    exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp')
    for subdir, _, files in tqdm(os.walk(src_root)):
        rel = os.path.relpath(subdir, src_root)
        dst_sub = os.path.join(dst_root, rel) if rel != "." else dst_root
        os.makedirs(dst_sub, exist_ok=True)
        for fname in files:
            if not fname.lower().endswith(exts):
                continue
            src_path = os.path.join(subdir, fname)
            dst_path = os.path.join(dst_sub, fname)
            img = cv2.imread(src_path)

            edges = extract_edges(img)
            base, _ = os.path.splitext(dst_path)
            dst_path = base + ".png"
            plt.imsave(dst_path, edges.squeeze(), cmap='gray')

    print("Edge images saved to:", dst_root)


create_edge_images("_rgb", "_edges")
