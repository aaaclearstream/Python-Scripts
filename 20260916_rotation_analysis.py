# ================================================================
# rotation_analysis_20251203.py
# Updated: 2025-12
# ================================================================
# Description:
#   Detect a red marker and body center,
#   compute CW/CCW rotations (including last 20 min),
#   compute locomotion distance/speed in millimeters,
#   save XY trajectory plot,
#   batch process videos.
#
# Usage example:
# python3 "/Users/li/Desktop/20260916_rotation_analysis.py" --folder "/Users/li/Desktop/20260917_PDS008_Rotation_3w" --start 11 --end 51 --mm 2.7
# the red dot goes around the green dot to the right (CCW), it goes to the left (CW)
# # ================================================================

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse
import glob
from math import atan2, degrees, sqrt
import sys
import mimetypes

# ---------------------------
# HSV-based color mask
# ---------------------------
def get_color_mask(hsv, color='red'):
    if color == 'red':
        lower1 = np.array([0, 45, 40])
        upper1 = np.array([15, 255, 255])
        lower2 = np.array([165, 45, 40])
        upper2 = np.array([180, 255, 255])
        return cv2.inRange(hsv, lower1, upper1) | cv2.inRange(hsv, lower2, upper2)
    
    elif color == 'green':
        lower = np.array([35, 35, 25])
        upper = np.array([105, 255, 255])
        return cv2.inRange(hsv, lower, upper)
    
    raise ValueError("Color must be 'red' or 'green'")

# ---------------------------
# Detect red dot centroid
# ---------------------------
def detect_dot_center(frame, color='red'):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = get_color_mask(hsv, color)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        c = max(contours, key=cv2.contourArea)
        M = cv2.moments(c)
        if M["m00"] > 0:
            return int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
    return None

# ---------------------------
# Detect mouse body center (green dot)
# ---------------------------
def detect_center_of_mass(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = get_color_mask(hsv, color='green')
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        c = max(contours, key=cv2.contourArea)
        M = cv2.moments(c)
        if M["m00"] > 0:
            return int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
    return None

# ---------------------------
# Compute full rotations (360°)
# ---------------------------
def compute_full_rotations(dot_coords, body_coords):
    cw = 0
    ccw = 0
    angle_accum = 0
    prev_angle = None

    for i in range(len(dot_coords)):
        if dot_coords[i] is None or body_coords[i] is None:
            continue

        dx = dot_coords[i][0] - body_coords[i][0]
        dy = dot_coords[i][1] - body_coords[i][1]
        angle = atan2(dy, dx)

        if prev_angle is not None:
            delta = degrees(angle - prev_angle)

            # unwrap
            if delta > 180:
                delta -= 360
            elif delta < -180:
                delta += 360

            angle_accum += delta

            if angle_accum >= 360:
                cw += 1
                angle_accum = 0
            elif angle_accum <= -360:
                ccw += 1
                angle_accum = 0

        prev_angle = angle

    return cw, ccw

# ---------------------------
# Compute CCW/CW in last N minutes
# ---------------------------
def compute_partial_rotations(dot_coords, body_coords, fps, last_N_minutes=20):
    total_frames = len(dot_coords)
    start_idx = max(0, total_frames - int(last_N_minutes * 60 * fps))
    return compute_full_rotations(dot_coords[start_idx:], body_coords[start_idx:])

# ---------------------------
# Distance + speed calculation (px → mm)
# ---------------------------
def compute_distance_speed(coords, fps, start_min, end_min):
    dist_px = 0
    per_frame_speed_px = []

    for i in range(1, len(coords)):
        if coords[i] is None or coords[i - 1] is None:
            per_frame_speed_px.append(0)
            continue

        d = sqrt((coords[i][0] - coords[i - 1][0])**2 +
                 (coords[i][1] - coords[i - 1][1])**2)

        dist_px += d
        per_frame_speed_px.append(d * fps)

    duration_min = end_min - start_min
    mean_speed_px = dist_px / (duration_min * 60 * fps) if duration_min > 0 else 0

    return dist_px, mean_speed_px, duration_min, per_frame_speed_px

# ---------------------------
# Trajectory plot
# ---------------------------
def plot_locomotion_trajectory(dot_coords, output_path):
    xs = [p[0] for p in dot_coords if p is not None]
    ys = [p[1] for p in dot_coords if p is not None]

    plt.figure(figsize=(6, 6))
    plt.plot(xs, ys, lw=0.7, color='blue')
    plt.xlabel("X (px)")
    plt.ylabel("Y (px)")
    plt.title("Locomotion Trajectory")
    plt.gca().invert_yaxis()
    plt.grid(True, linewidth=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

# ---------------------------
# Single video analysis
# ---------------------------
def analyze_rotation(video_path, start_min, end_min, resize_factor, px_per_mm=1.0):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    if not cap.isOpened() or fps == 0:
        print(f"❌ ERROR: Cannot read video: {video_path}")
        return None

    start_frame = int(start_min * 60 * fps)
    end_frame   = int(end_min   * 60 * fps)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    dot_pts = []
    body_pts = []
    lost = 0

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    for f in range(start_frame, min(end_frame, total_frames)):
        ret, frame = cap.read()
        if not ret:
            break

        if resize_factor != 1.0:
            frame = cv2.resize(frame, (0, 0), fx=resize_factor, fy=resize_factor)

        dot = detect_dot_center(frame)
        body = detect_center_of_mass(frame)

        if dot is None or body is None:
            lost += 1

        dot_pts.append(dot)
        body_pts.append(body)

    cap.release()

    # Rotations
    cw, ccw = compute_full_rotations(dot_pts, body_pts)
    cw20, ccw20 = compute_partial_rotations(dot_pts, body_pts, fps, last_N_minutes=20)

    # Distance & speed
    dist_px, speed_px, duration_min, per_frame_speed_px = compute_distance_speed(body_pts, fps, start_min, end_min)

    # px → mm conversion
    dist_mm = dist_px / px_per_mm
    speed_mmps = speed_px / px_per_mm

    # Rotation rate
    rotation_per_min = (ccw - cw) / duration_min if duration_min > 0 else 0
    rotation_last20  = (ccw20 - cw20) / 20 if duration_min >= 20 else 0

    lost_time_min = lost / fps / 60

    # Save plots
    base = os.path.splitext(video_path)[0]
    plot_locomotion_trajectory(dot_pts, base + "_locomotion_trajectory.tiff")

    return {
        "Video": os.path.basename(video_path),
        "Start_min": start_min,
        "End_min": end_min,
        "Duration_min": round(duration_min, 2),

        "CCW": ccw,
        "CW": cw,
        "NetRotation_bias": ccw - cw,

        "CCW_last20": ccw20,
        "CW_last20": cw20,
        "NetRotation_last20": ccw20 - cw20,

        "Distance_mm": round(dist_mm, 2),
        "MeanSpeed_mm_per_sec": round(speed_mmps, 2),

        "Rotation_per_min": round(rotation_per_min, 2),
        "Rotation_last20min": round(rotation_last20, 2),

        "LostTime_min": round(lost_time_min, 2)
    }

# ---------------------------
# Batch processing
# ---------------------------
def run_batch(video_paths, start_min, end_min, resize_factor, px_per_mm):
    results = []

    for path in video_paths:
        print(f"\n🎥 Processing: {path}")
        res = analyze_rotation(path, start_min, end_min, resize_factor, px_per_mm)
        if res:
            results.append(res)

    if results:
        df = pd.DataFrame(results)
        outpath = os.path.join(os.path.dirname(video_paths[0]), "rotation_analysis_summary_all.csv")
        df.to_csv(outpath, index=False)
        print(f"\n✅ Summary saved to: {outpath}")

# ---------------------------
# CLI entry point
# ---------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--folder", help="Folder with .mov videos")
    parser.add_argument("--video", help="Single video path")
    parser.add_argument("--start", type=float, default=0, help="Start time (min)")
    parser.add_argument("--end", type=float, default=20, help="End time (min)")
    parser.add_argument("--resize", type=float, default=1.0, help="Resize factor")
    parser.add_argument("--mm", type=float, default=1.0, help="Pixels per mm")

    args = parser.parse_args()

    start_min = args.start
    end_min = args.end
    resize_factor = args.resize
    px_per_mm = args.mm

    if args.folder:
        video_paths = sorted(glob.glob(os.path.join(args.folder, "*.[mM][oO][vV]")))
    elif args.video:
        video_paths = [args.video]
    else:
        print("❌ ERROR: Provide --folder or --video")
        sys.exit(1)

    run_batch(video_paths, start_min, end_min, resize_factor, px_per_mm)