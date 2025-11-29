#!/usr/bin/env python3
"""
Face tracker module.

Menyediakan utilitas untuk mendeteksi wajah dengan MediaPipe Face Mesh
dan melaporkan status kemiringan kepala secara realtime. Modul ini tidak
mengurusi pemilihan meme ataupun skor, sehingga bisa dipakai ulang oleh
GUI/game logic.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Callable, Optional, Tuple

import cv2
import mediapipe as mp


TILT_THRESHOLD_DEG = 12.0
HOLD_TIME = 0.60
COOLDOWN_TIME = 1.0


@dataclass
class FaceTrackerState:
    face_detected: bool
    roll_deg: float
    tilt_state: str  # LEFT, RIGHT, CENTER, NO_FACE
    tilt_confirmed: bool
    timestamp: float
    # Center of detected face in frame coordinates (x, y). None if no face.
    face_center: Optional[Tuple[int, int]] = None
    # Approximate face bounding box size (width, height) in pixels (None if no face)
    face_size: Optional[Tuple[int, int]] = None


class FaceTracker:
    """Kamera + status wajah/kemiringan."""

    def __init__(
        self,
        tilt_threshold: float = TILT_THRESHOLD_DEG,
        hold_time: float = HOLD_TIME,
        cooldown_time: float = COOLDOWN_TIME,
    ) -> None:
        self.tilt_threshold = tilt_threshold
        self.hold_time = hold_time
        self.cooldown_time = cooldown_time
        self._face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._cap: Optional[cv2.VideoCapture] = None
        self._tilt_state = "CENTER"
        self._tilt_start: Optional[float] = None
        self._last_confirm = 0.0

    def _reset_tilt(self) -> None:
        self._tilt_state = "CENTER"
        self._tilt_start = None
        self._last_confirm = 0.0

    def start(self) -> None:
        self._cap = cv2.VideoCapture(1)
        if not self._cap.isOpened():
            raise RuntimeError("Cannot open camera")
        self._reset_tilt()

    def stop(self) -> None:
        if self._cap:
            self._cap.release()
            self._cap = None
        self._face_mesh.close()
        cv2.destroyAllWindows()

    def _compute_roll_deg(self, landmarks, width: int, height: int) -> float:
        right_eye_idx = 33
        left_eye_idx = 263
        r_x = int(landmarks[right_eye_idx].x * width)
        r_y = int(landmarks[right_eye_idx].y * height)
        l_x = int(landmarks[left_eye_idx].x * width)
        l_y = int(landmarks[left_eye_idx].y * height)
        dx = l_x - r_x
        dy = l_y - r_y
        angle_rad = math.atan2(dy, dx)
        return math.degrees(angle_rad)

    def _evaluate_state(self, frame):
        height, width = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._face_mesh.process(rgb)
        now = time.time()

        if not results.multi_face_landmarks:
            self._reset_tilt()
            return FaceTrackerState(
                face_detected=False,
                roll_deg=0.0,
                tilt_state="NO_FACE",
                tilt_confirmed=False,
                timestamp=now,
                face_center=None,
            )

        landmarks = results.multi_face_landmarks[0].landmark
        roll_deg = self._compute_roll_deg(landmarks, width, height)

        prev_state = self._tilt_state
        if roll_deg > self.tilt_threshold:
            self._tilt_state = "RIGHT"
        elif roll_deg < -self.tilt_threshold:
            self._tilt_state = "LEFT"
        else:
            self._tilt_state = "CENTER"

        confirmed = False
        if self._tilt_state in ("LEFT", "RIGHT"):
            if prev_state != self._tilt_state:
                self._tilt_start = now
            elif (
                self._tilt_start is not None
                and now - self._tilt_start >= self.hold_time
                and now - self._last_confirm >= self.cooldown_time
            ):
                confirmed = True
                self._last_confirm = now
                self._tilt_start = None
        else:
            self._tilt_start = None

        # Compute approximate face center from landmarks (min/max of landmark coords)
        xs = [int(p.x * width) for p in landmarks]
        ys = [int(p.y * height) for p in landmarks]
        if xs and ys:
            minx, maxx = min(xs), max(xs)
            miny, maxy = min(ys), max(ys)
            face_center = ((minx + maxx) // 2, (miny + maxy) // 2)
            face_size = (maxx - minx, maxy - miny)
        else:
            face_center = None
            face_size = None

        return FaceTrackerState(
            face_detected=True,
            roll_deg=roll_deg,
            tilt_state=self._tilt_state,
            tilt_confirmed=confirmed,
            timestamp=now,
            face_center=face_center,
            face_size=face_size,
        )

    def run(
        self,
        callback: Optional[Callable[[FaceTrackerState, "cv2.Mat"], None]] = None,
        preview: bool = True,
    ) -> None:
        """
        Jalankan loop kamera. Callback dipanggil dengan state terbaru
        dan frame kamera (sudah di-flip horizontal).
        """
        self.start()
        try:
            while True:
                assert self._cap is not None
                ret, frame = self._cap.read()
                if not ret:
                    break
                frame = cv2.flip(frame, 1)
                state = self._evaluate_state(frame)
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
        h, _ = frame.shape[:2]
        if not state.face_detected:
            text = "No face detected"
            color = (200, 200, 200)
        else:
            text = f"{state.tilt_state} ({state.roll_deg:.1f} deg)"
            color = (0, 200, 0) if state.tilt_state in ("LEFT", "RIGHT") else (200, 200, 0)
        cv2.rectangle(frame, (0, h - 60), (320, h), (50, 50, 50), -1)
        cv2.putText(frame, text, (10, h - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
        if state.tilt_confirmed:
            cv2.putText(frame, "CONFIRMED", (10, h - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow("Face Tracker Preview (ESC to exit)", frame)


if __name__ == "__main__":
    tracker = FaceTracker()
    tracker.run()
