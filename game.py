#!/usr/bin/env python3

# game.py (filter kuis kucing)

import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path
import math
import random
import time
import sys

ASSET_DIR = Path("asset")
# pattern for cat images (adjust if your files use different names)
IMG_PATTERN = "kucing*.png"

# thresholds / timings
TILT_THRESHOLD_DEG = 12.0   # degrees to consider a tilt
HOLD_TIME = 0.60            # seconds user must hold tilt to confirm selection
COOLDOWN_TIME = 1.0         # seconds after a confirmed choice before next can be accepted

CAT_SCALE = 0.45  # base scale multiplier (will be scaled by face width)

def overlay_rgba(background: np.ndarray, overlay_rgba: np.ndarray, x: int, y: int):
    bh, bw = background.shape[:2]
    oh, ow = overlay_rgba.shape[:2]

    if x >= bw or y >= bh or x + ow <= 0 or y + oh <= 0:
        return background

    x1 = max(x, 0)
    y1 = max(y, 0)
    x2 = min(x + ow, bw)
    y2 = min(y + oh, bh)

    ox1 = x1 - x
    oy1 = y1 - y
    ox2 = ox1 + (x2 - x1)
    oy2 = oy1 + (y2 - y1)

    overlay_crop = overlay_rgba[oy1:oy2, ox1:ox2]
    if overlay_crop.shape[2] < 4:
        alpha = np.ones((overlay_crop.shape[0], overlay_crop.shape[1], 1), dtype=np.uint8) * 255
        overlay_crop = np.concatenate([overlay_crop, alpha], axis=2)

    overlay_bgr = overlay_crop[..., :3].astype(float)
    alpha_mask = (overlay_crop[..., 3] / 255.0).astype(float)

    bg_region = background[y1:y2, x1:x2].astype(float)
    alpha_3 = np.stack([alpha_mask, alpha_mask, alpha_mask], axis=2)
    blended = alpha_3 * overlay_bgr + (1 - alpha_3) * bg_region
    background[y1:y2, x1:x2] = blended.astype(np.uint8)
    return background

def load_image_rgba(path: Path):
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return img

def get_point(landmark, frame_w, frame_h):
    return int(landmark.x * frame_w), int(landmark.y * frame_h)

def compute_roll_deg(landmarks, w, h):
    right_eye_idx = 33
    left_eye_idx = 263
    r_x, r_y = get_point(landmarks[right_eye_idx], w, h)
    l_x, l_y = get_point(landmarks[left_eye_idx], w, h)
    dx = l_x - r_x
    dy = l_y - r_y
    angle_rad = math.atan2(dy, dx)
    angle_deg = math.degrees(angle_rad)
    return angle_deg

def pick_two_images(img_paths, prev_pair=None):
    if len(img_paths) < 2:
        raise ValueError("Need at least 2 images in asset folder.")
    # choose two distinct at random; try to avoid repeating previous pair
    attempts = 0
    while True:
        pair = random.sample(img_paths, 2)
        if prev_pair is None:
            return pair
        # allow reorder; if identical set, retry a few times
        if set(pair) != set(prev_pair) or attempts > 6:
            return pair
        attempts += 1

def main():
    img_files = sorted(list(ASSET_DIR.glob(IMG_PATTERN)))
    if len(img_files) < 2:
        print(f"[ERROR] Found {len(img_files)} images matching {IMG_PATTERN} in {ASSET_DIR}. Need at least 2.")
        sys.exit(1)

    # current pair selection (Paths)
    left_path, right_path = pick_two_images(img_files, prev_pair=None)
    left_img = load_image_rgba(left_path)
    right_img = load_image_rgba(right_path)

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False,
                                      max_num_faces=1,
                                      refine_landmarks=True,
                                      min_detection_confidence=0.5,
                                      min_tracking_confidence=0.5)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        return

    tilt_state = "CENTER"  # 'LEFT', 'RIGHT', 'CENTER'
    tilt_start_time = None
    last_confirm_time = 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            selection_text = "No face"
            selection_color = (200, 200, 200)
            confirmed_text = ""

            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark

                # estimate face width
                try:
                    cheek_left = get_point(landmarks[127], w, h)
                    cheek_right = get_point(landmarks[356], w, h)
                    face_width = max(1, abs(cheek_right[0] - cheek_left[0]))
                except Exception:
                    r_eye = get_point(landmarks[33], w, h)
                    l_eye = get_point(landmarks[263], w, h)
                    face_width = max(1, abs(l_eye[0] - r_eye[0]))

                scale_factor = (face_width / 200.0) * CAT_SCALE
                cat_left_resized = cv2.resize(left_img, (0, 0), fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)
                cat_right_resized = cv2.resize(right_img, (0, 0), fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)

                forehead_idx = 10
                fx, fy = get_point(landmarks[forehead_idx], w, h)
                offset_y = int(face_width * 0.55)
                offset_x = int(face_width * 0.7)

                left_x = int(fx - offset_x - cat_left_resized.shape[1] // 2)
                left_y = int(fy - offset_y - cat_left_resized.shape[0] // 2)
                right_x = int(fx + offset_x - cat_right_resized.shape[1] // 2)
                right_y = int(fy - offset_y - cat_right_resized.shape[0] // 2)

                # overlay cats
                frame = overlay_rgba(frame, cat_left_resized, left_x, left_y)
                frame = overlay_rgba(frame, cat_right_resized, right_x, right_y)

                # compute tilt angle
                roll_deg = compute_roll_deg(landmarks, w, h)
                ang_text = f"tilt {roll_deg:.1f}°"
                cv2.putText(frame, ang_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2, cv2.LINE_AA)

                # determine tilt state
                prev_state = tilt_state
                if roll_deg > TILT_THRESHOLD_DEG:
                    tilt_state = "RIGHT"
                    selection_text = "SELECT: RIGHT"
                    selection_color = (0, 200, 0)
                elif roll_deg < -TILT_THRESHOLD_DEG:
                    tilt_state = "LEFT"
                    selection_text = "SELECT: LEFT"
                    selection_color = (0, 200, 0)
                else:
                    tilt_state = "CENTER"
                    selection_text = "CENTER"
                    selection_color = (220, 220, 0)

                # handle hold/confirm logic
                now = time.time()
                if tilt_state in ("LEFT", "RIGHT"):
                    if prev_state != tilt_state:
                        tilt_start_time = now  # started new tilt
                    else:
                        # continued tilt: check duration and cooldown
                        if tilt_start_time is not None and (now - tilt_start_time >= HOLD_TIME) and (now - last_confirm_time >= COOLDOWN_TIME):
                            # confirmed selection
                            confirmed_text = f"CONFIRMED: {tilt_state}"
                            last_confirm_time = now
                            # pick new random pair (avoid identical pair)
                            prev_pair = (left_path, right_path)
                            try:
                                left_path, right_path = pick_two_images(img_files, prev_pair=prev_pair)
                            except ValueError:
                                pass
                            left_img = load_image_rgba(left_path)
                            right_img = load_image_rgba(right_path)
                            # reset tilt_start_time to avoid immediate re-trigger
                            tilt_start_time = None
                else:
                    tilt_start_time = None

            # draw selection panel
            cv2.rectangle(frame, (0, h - 60), (320, h), (50, 50, 50), -1)
            cv2.putText(frame, selection_text, (10, h - 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, selection_color, 2, cv2.LINE_AA)
            if confirmed_text:
                cv2.putText(frame, confirmed_text, (10, h - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow("Cat Quiz Filter (ESC to exit)", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        face_mesh.close()

if __name__ == "__main__":
    main()
