#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Angle-Based Face Tracker
=========================
Deteksi kemiringan kepala berdasarkan SUDUT, bukan pixel movement.
Lebih stabil dan tidak tergantung jarak kamera.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
import math
from typing import Optional
import sys

import cv2
import numpy as np

# Fix encoding
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')


# Thresholds berdasarkan SUDUT (derajat)
TILT_ANGLE_THRESHOLD = 5.0  # Sensitif - 5°
HOLD_TIME = 0.0              # INSTANT - TANPA HOLD!
COOLDOWN_TIME = 0.5          # Cooldown untuk prevent false positive - 0.5s


@dataclass
class FaceTrackerState:
    face_detected: bool
    roll_deg: float
    tilt_state: str  # LEFT, RIGHT, CENTER, NO_FACE
    tilt_confirmed: bool
    timestamp: float


class AngleFaceTracker:
    """
    Face tracker berbasis sudut kemiringan.
    Menggunakan deteksi mata untuk hitung slope -> angle.
    """

    def __init__(
        self,
        tilt_threshold: float = TILT_ANGLE_THRESHOLD,
        hold_time: float = HOLD_TIME,
        cooldown_time: float = COOLDOWN_TIME,
    ) -> None:
        self.tilt_threshold = tilt_threshold
        self.hold_time = hold_time
        self.cooldown_time = cooldown_time

        # Load cascades
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye.xml'
        )

        self._cap: Optional[cv2.VideoCapture] = None
        self._tilt_state = "CENTER"
        self._tilt_start: Optional[float] = None
        self._last_confirm = 0.0

        # Smoothing buffer untuk angle (5 frames untuk stabilitas)
        self._angle_buffer = []
        self._buffer_size = 5  # 5 frames - balance antara responsive & stable

        # Face tracking
        self._face_lost_frames = 0
        self._last_valid_face = None
        self._last_valid_angle = 0.0

    def _reset_tilt(self) -> None:
        self._tilt_state = "CENTER"
        self._tilt_start = None
        self._last_confirm = 0.0
        self._angle_buffer.clear()

    def start(self) -> None:
        self._cap = cv2.VideoCapture(0)
        if not self._cap.isOpened():
            raise RuntimeError("Cannot open camera")

        # Camera settings
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self._cap.set(cv2.CAP_PROP_FPS, 30)

        self._reset_tilt()

    def stop(self) -> None:
        if self._cap:
            self._cap.release()
            self._cap = None
        cv2.destroyAllWindows()

    def _detect_eyes_and_angle(self, frame, face_rect):
        """
        Deteksi mata dan hitung angle kemiringan.
        Return: angle dalam derajat (negatif = kiri, positif = kanan)
        """
        x, y, w, h = face_rect

        # ROI untuk deteksi mata (bagian atas wajah)
        roi_gray = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
        roi_color = frame[y:y+h, x:x+w]

        # Deteksi mata
        eyes = self.eye_cascade.detectMultiScale(
            roi_gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(20, 20)
        )

        if len(eyes) < 2:
            # Tidak bisa deteksi 2 mata, gunakan last valid
            return self._last_valid_angle

        # Sort eyes by x position (kiri ke kanan)
        eyes = sorted(eyes, key=lambda e: e[0])

        # Ambil 2 mata (leftmost dan rightmost)
        left_eye = eyes[0]
        right_eye = eyes[-1]

        # Center point masing-masing mata
        left_center = (left_eye[0] + left_eye[2]//2, left_eye[1] + left_eye[3]//2)
        right_center = (right_eye[0] + right_eye[2]//2, right_eye[1] + right_eye[3]//2)

        # Hitung angle dari slope
        dx = right_center[0] - left_center[0]
        dy = right_center[1] - left_center[1]

        # Angle dalam radian, konversi ke derajat
        angle_rad = math.atan2(dy, dx)
        angle_deg = math.degrees(angle_rad)

        # Normalisasi: 0° = horizontal, negatif = kiri, positif = kanan
        # atan2 memberikan angle dari horizontal line
        # Jika mata kanan lebih tinggi (dy positif) -> kemiringan kanan -> angle positif
        # Jika mata kiri lebih tinggi (dy negatif) -> kemiringan kiri -> angle negatif

        # Store untuk drawing
        self._current_eyes = (left_center, right_center, left_eye, right_eye, x, y)

        return angle_deg

    def _evaluate_state(self, frame):
        now = time.time()

        # Detect face
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(100, 100)
        )

        if len(faces) == 0:
            self._face_lost_frames += 1

            # Tolerance: gunakan last valid face beberapa frame
            if self._face_lost_frames < 15 and self._last_valid_face is not None:
                face = self._last_valid_face
            else:
                self._reset_tilt()
                self._last_valid_face = None
                return FaceTrackerState(
                    face_detected=False,
                    roll_deg=0.0,
                    tilt_state="NO_FACE",
                    tilt_confirmed=False,
                    timestamp=now,
                )
        else:
            # Face detected
            self._face_lost_frames = 0
            face = max(faces, key=lambda f: f[2] * f[3])  # Largest face
            self._last_valid_face = face

        # Detect angle
        angle_deg = self._detect_eyes_and_angle(frame, face)

        # Smoothing dengan buffer
        self._angle_buffer.append(angle_deg)
        if len(self._angle_buffer) > self._buffer_size:
            self._angle_buffer.pop(0)

        # Average angle (smoothed)
        avg_angle = np.mean(self._angle_buffer) if self._angle_buffer else 0.0

        # Store for drawing
        self._current_face = face
        self._last_valid_angle = avg_angle

        # Debug output (optional - comment out untuk performance)
        # print(f"[ANGLE] {avg_angle:.1f}° | threshold: ±{self.tilt_threshold}°")

        # Determine tilt state berdasarkan ANGLE
        prev_state = self._tilt_state

        if avg_angle < -self.tilt_threshold:
            self._tilt_state = "LEFT"
        elif avg_angle > self.tilt_threshold:
            self._tilt_state = "RIGHT"
        else:
            self._tilt_state = "CENTER"

        # Confirmation: langsung confirm saat berubah dari CENTER ke LEFT/RIGHT
        confirmed = False
        if self._tilt_state in ("LEFT", "RIGHT"):
            # Jika baru berubah dari CENTER
            if prev_state == "CENTER":
                # Check cooldown (prevent double trigger)
                if now - self._last_confirm >= self.cooldown_time:
                    confirmed = True
                    self._last_confirm = now
                    print(f"[CONFIRMED] {self._tilt_state}!")

        # Reset tilt start jika kembali ke center
        if self._tilt_state == "CENTER":
            self._tilt_start = None

        return FaceTrackerState(
            face_detected=True,
            roll_deg=avg_angle,
            tilt_state=self._tilt_state,
            tilt_confirmed=confirmed,
            timestamp=now,
        )

    def run(self, callback=None, preview=True) -> None:
        """Main loop"""
        self.start()
        try:
            while True:
                assert self._cap is not None
                ret, frame = self._cap.read()
                if not ret:
                    break

                frame = cv2.flip(frame, 1)
                state = self._evaluate_state(frame)

                # Draw debug info
                if state.face_detected and hasattr(self, '_current_face'):
                    x, y, w, h = self._current_face
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

                    # Draw eyes if available
                    if hasattr(self, '_current_eyes'):
                        left_c, right_c, left_e, right_e, face_x, face_y = self._current_eyes
                        # Adjust coordinates (relative to face)
                        lx, ly = left_c[0] + face_x, left_c[1] + face_y
                        rx, ry = right_c[0] + face_x, right_c[1] + face_y

                        # Draw eye centers
                        cv2.circle(frame, (lx, ly), 5, (255, 0, 0), -1)
                        cv2.circle(frame, (rx, ry), 5, (255, 0, 0), -1)

                        # Draw line between eyes
                        cv2.line(frame, (lx, ly), (rx, ry), (0, 255, 255), 2)

                if callback:
                    callback(state, frame)

                if preview:
                    self._draw_debug(frame, state)
                    key = cv2.waitKey(1) & 0xFF
                    if key == 27:  # ESC
                        break
                else:
                    if cv2.waitKey(1) & 0xFF == 27:
                        break
        finally:
            self.stop()

    @staticmethod
    def _draw_debug(frame, state: FaceTrackerState) -> None:
        h, w = frame.shape[:2]

        # Status bar
        cv2.rectangle(frame, (0, h - 80), (w, h), (50, 50, 50), -1)

        if not state.face_detected:
            text = "No face detected"
            color = (200, 200, 200)
        else:
            text = f"{state.tilt_state} | Angle: {state.roll_deg:.1f}°"
            if state.tilt_state == "LEFT":
                color = (0, 200, 255)  # Orange
            elif state.tilt_state == "RIGHT":
                color = (255, 200, 0)  # Cyan
            else:
                color = (200, 200, 0)  # Yellow

        cv2.putText(frame, text, (10, h - 45), cv2.FONT_HERSHEY_SIMPLEX,
                   0.8, color, 2, cv2.LINE_AA)

        if state.tilt_confirmed:
            cv2.putText(frame, ">>> CONFIRMED! <<<", (10, h - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

        # Angle indicator bar
        bar_width = 400
        bar_x = w//2 - bar_width//2
        bar_y = h - 25

        # Background bar
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + 15), (100, 100, 100), -1)

        # Center line
        center_x = bar_x + bar_width // 2
        cv2.line(frame, (center_x, bar_y), (center_x, bar_y + 15), (255, 255, 255), 2)

        # Angle indicator
        if state.face_detected:
            # Map angle to bar position (-30° to +30°)
            normalized = max(-30, min(30, state.roll_deg))
            indicator_offset = int((normalized / 30) * (bar_width // 2))
            indicator_x = center_x + indicator_offset

            cv2.circle(frame, (indicator_x, bar_y + 7), 8, (0, 255, 0), -1)

        cv2.imshow("Angle Face Tracker (ESC to exit)", frame)


if __name__ == "__main__":
    tracker = AngleFaceTracker()
    tracker.run()
