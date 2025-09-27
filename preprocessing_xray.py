import cv2
import numpy as np
import os
from tqdm import tqdm

def preprocess_xray(img_path, save_path=None):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Warning: Could not read", img_path)
        return None
    
    # Adaptive masking
    min_val, max_val = np.min(img), np.max(img)
    threshold = min_val + 0.9 * (max_val - min_val)
    _, mask = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5,5), np.uint8)
    mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    img_masked = cv2.bitwise_and(img, img, mask=255-mask_closed)
    
    # Histogram Equalization + Gaussian Blur
    img_eq = cv2.equalizeHist(img_masked)
    img_final = cv2.GaussianBlur(img_eq, (5,5), 0)
    
    # Save
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, img_final)
    
    return img_final

# Directories
input_dir = r"D:\important\XMEDFUSION\IU-DATASET"
output_dir = r"D:\important\XMEDFUSION\PREPROCESSED-IU-DATASET"

# Make sure top-level folder exists
os.makedirs(output_dir, exist_ok=True)

# Process all images preserving folder structure
for study_folder in tqdm(os.listdir(input_dir)):
    study_path = os.path.join(input_dir, study_folder)
    
    # If it's a file (like a PNG directly in root), save it at top level
    if os.path.isfile(study_path):
        out_path = os.path.join(output_dir, study_folder)
        preprocess_xray(study_path, out_path)
        continue

    # If it's a folder, process all images inside
    for img_file in os.listdir(study_path):
        in_path = os.path.join(study_path, img_file)
        out_path = os.path.join(output_dir, study_folder, img_file)
        preprocess_xray(in_path, out_path)

print("All images preprocessed and saved in:", output_dir)
