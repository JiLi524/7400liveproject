import cv2
import pandas as pd
import numpy as np
import os

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--raw_dir", type=str, default="data/raw")
args = parser.parse_args()

raw_dir = args.raw_dir
metrics_dir = "data/metrics"

subfolders = [
    f for f in os.listdir(raw_dir)
    if os.path.isdir(os.path.join(raw_dir, f))
]
print(subfolders)

# Creating the directory path specified labels_dir
os.makedirs(metrics_dir, exist_ok=True)

# Stating the thresholds
# All thresholds computed from distribution of each metric
blue_excess_min, blue_excess_max = -0.25, 0.025
exposure_min, exposure_max = 0.4, 0.8
blur_min = 30
saturation_min = 0.20
occupancy_score_min = 0.20
rows = []

for folder_id in subfolders:
    folder_path = os.path.join(raw_dir, folder_id)

    for filename in os.listdir(folder_path):
        if not filename.lower().endswith((".jpg", ".jpeg", ".png", ".heic")):
            continue

        img_path = os.path.join(folder_path, filename)
        img = cv2.imread(img_path)

        # Check if the file is corrupt
        if img is None:
            print(f"{filename} Failed, corrupt image detected")
            continue
        image_id = os.path.splitext(filename)[0]
        # Resizing to 90x90
        # Normalizing the image pixel data to be 0-1
        img_resized = cv2.resize(img, (90, 90), interpolation=cv2.INTER_AREA)
        img_norm = img_resized / 255.0

        # Blue channel dominance
        # relative blue excess
        b = img_norm[:, :, 0]
        g = img_norm[:, :, 1]
        r = img_norm[:, :, 2]
        # blue higher than average of R and G
        blue_excess = (b - 0.5*(r + g)).mean()

        # Saturation
        hsv = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)
        h, w = hsv.shape[:2]
        cy0, cy1 = int(0.20*h), int(0.80*h)
        cx0, cx1 = int(0.20*w), int(0.80*w)
        sat_center = (hsv[cy0:cy1, cx0:cx1, 1] / 255.0).mean()

        # Exposure
        v = hsv[:, :, 2] / 255.0
        exposure_score = v.mean() # 0-1

        # Occupancy
        h, s, v = cv2.split(hsv)
        s_norm = s / 255.0
        v_norm = v / 255.0
        color_mask = s_norm > 0.15
        brightness_mask = v_norm > 0.05
        fruit_mask = color_mask & brightness_mask
        fruit_mask = fruit_mask.astype("uint8") * 255

        kernel = np.ones((7,7), np.uint8)
        fruit_mask = cv2.morphologyEx(fruit_mask, cv2.MORPH_CLOSE, kernel)
        fruit_mask = cv2.morphologyEx(fruit_mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(fruit_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        clean_mask = np.zeros_like(fruit_mask)

        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            cv2.drawContours(clean_mask, [largest_contour], -1, 255, thickness=-1)

        occupancy_score = np.sum(clean_mask > 0) / clean_mask.size


        # Blur
        gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        blur_valid = blur_score > blur_min

        qc_checks = {
            "blue_excess_issue": blue_excess_min <= blue_excess <= blue_excess_max,
            "exposure_issue": exposure_min <= exposure_score <= exposure_max,
            "blur_issue": blur_valid,
            "saturation_issue": sat_center >= saturation_min,
            "occupancy_issue": occupancy_score >= occupancy_score_min
            }

        reasons = []
        if not qc_checks["blue_excess_issue"]: reasons.append("blue_channel_issue")
        if not qc_checks["exposure_issue"]: reasons.append("exposure_issue")
        if not qc_checks["blur_issue"]: reasons.append("blur_issue")
        if not qc_checks["saturation_issue"]: reasons.append("saturation_issue")
        if not qc_checks["occupancy_issue"]: reasons.append("occupancy_issue")

        qc_status = "pass" if not reasons else "flagged"

        rows.append({
            "image_id": image_id,
            "student_id": str(folder_id),
            "qc_status": qc_status,
            "blue_score": blue_excess,
            "exposure_score": exposure_score,
            "blur_score": blur_score,
            "saturation_score": sat_center,
            "occupancy_score" : occupancy_score,
            "fail_reasons": ", ".join(reasons) if reasons else "none"
        })

qc = pd.DataFrame(rows)
qc.to_csv("data/metrics/qc_metrics.csv", index=False)
print(qc["qc_status"].value_counts())
