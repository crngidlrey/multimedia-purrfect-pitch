#!/usr/bin/env python3
"""
Waveform View Module.

Komponen GUI untuk visualisasi waveform audio.
Menerima data waveform dari audio_manager dan menggambar ke OpenCV frame.
Support animasi playback progress.
"""

from __future__ import annotations

from typing import List, Tuple, Optional
import numpy as np
import cv2


class WaveformView:
    """
    Komponen GUI untuk menampilkan waveform audio.
    Render waveform sebagai bar chart atau line plot dengan progress indicator.
    """

    def __init__(
        self,
        bg_color: Tuple[int, int, int] = (40, 40, 40),
        waveform_color: Tuple[int, int, int] = (100, 200, 255),
        progress_color: Tuple[int, int, int] = (0, 255, 255),
        border_color: Tuple[int, int, int] = (80, 80, 80),
        style: str = "bars"  # "bars" atau "line"
    ):
        """
        Inisialisasi waveform view.

        Args:
            bg_color (tuple): Warna background (BGR)
            waveform_color (tuple): Warna waveform (BGR)
            progress_color (tuple): Warna progress indicator (BGR)
            border_color (tuple): Warna border (BGR)
            style (str): Style visualisasi ("bars" atau "line")
        """
        self.bg_color = bg_color
        self.waveform_color = waveform_color
        self.progress_color = progress_color
        self.border_color = border_color
        self.style = style

        # Waveform data
        self._waveform_data: List[float] = []
        self._playback_progress: float = 0.0  # 0.0 - 1.0
        self._is_playing: bool = False

    def set_waveform_data(self, data: List[float]) -> None:
        """
        Set data waveform untuk visualisasi.

        Args:
            data (list): List amplitudo ter-normalisasi (0.0 - 1.0)
        """
        self._waveform_data = data.copy() if data else []
        self._playback_progress = 0.0

    def set_playback_progress(self, progress: float) -> None:
        """
        Set progress playback untuk animasi.

        Args:
            progress (float): Progress 0.0 - 1.0 (0% - 100%)
        """
        self._playback_progress = max(0.0, min(1.0, progress))

    def set_playing(self, is_playing: bool) -> None:
        """
        Set status playing untuk visual feedback.

        Args:
            is_playing (bool): True jika audio sedang playing
        """
        self._is_playing = is_playing

    def clear(self) -> None:
        """
        Clear waveform data dan reset state.
        """
        self._waveform_data.clear()
        self._playback_progress = 0.0
        self._is_playing = False

    def draw(
        self,
        frame: np.ndarray,
        x: int,
        y: int,
        width: int,
        height: int,
        show_border: bool = True
    ) -> None:
        """
        Gambar waveform ke frame pada posisi dan ukuran tertentu.

        Args:
            frame (np.ndarray): Frame target (BGR)
            x (int): Posisi X (top-left)
            y (int): Posisi Y (top-left)
            width (int): Lebar area waveform
            height (int): Tinggi area waveform
            show_border (bool): Tampilkan border atau tidak
        """
        # Validasi bounds
        frame_h, frame_w = frame.shape[:2]
        if x < 0 or y < 0 or x + width > frame_w or y + height > frame_h:
            # Clipping jika keluar frame
            x = max(0, x)
            y = max(0, y)
            width = min(width, frame_w - x)
            height = min(height, frame_h - y)

        if width <= 0 or height <= 0:
            return

        # Draw background
        cv2.rectangle(frame, (x, y), (x + width, y + height), self.bg_color, -1)

        # Draw border
        if show_border:
            cv2.rectangle(frame, (x, y), (x + width, y + height), self.border_color, 2)

        # Draw waveform jika ada data
        if self._waveform_data and len(self._waveform_data) > 0:
            if self.style == "bars":
                self._draw_bars(frame, x, y, width, height)
            elif self.style == "line":
                self._draw_line(frame, x, y, width, height)
        else:
            # Tampilkan placeholder text
            self._draw_placeholder(frame, x, y, width, height)

    def _draw_bars(
        self,
        frame: np.ndarray,
        x: int,
        y: int,
        width: int,
        height: int
    ) -> None:
        """
        Gambar waveform sebagai vertical bars.
        """
        num_samples = len(self._waveform_data)
        if num_samples == 0:
            return

        # Hitung lebar setiap bar dengan spacing
        bar_spacing = 2
        bar_width = max(1, (width - (num_samples - 1) * bar_spacing) // num_samples)

        # Tinggi maksimum bar (beri margin atas/bawah)
        margin = 4
        max_bar_height = height - 2 * margin

        # Center line
        center_y = y + height // 2

        # Progress threshold (sample index)
        progress_idx = int(self._playback_progress * num_samples)

        # Draw bars
        for i, amplitude in enumerate(self._waveform_data):
            # Posisi X bar
            bar_x = x + margin + i * (bar_width + bar_spacing)

            # Tinggi bar berdasarkan amplitude (symmetrical dari center)
            bar_h = int(amplitude * max_bar_height / 2)

            # Warna: progress color jika sudah diplay, waveform color jika belum
            color = self.progress_color if i < progress_idx else self.waveform_color

            # Draw bar (dari center ke atas dan bawah)
            if bar_h > 0:
                # Bar atas
                cv2.rectangle(
                    frame,
                    (bar_x, center_y - bar_h),
                    (bar_x + bar_width, center_y),
                    color,
                    -1
                )
                # Bar bawah
                cv2.rectangle(
                    frame,
                    (bar_x, center_y),
                    (bar_x + bar_width, center_y + bar_h),
                    color,
                    -1
                )

    def _draw_line(
        self,
        frame: np.ndarray,
        x: int,
        y: int,
        width: int,
        height: int
    ) -> None:
        """
        Gambar waveform sebagai continuous line.
        """
        num_samples = len(self._waveform_data)
        if num_samples < 2:
            return

        # Margin
        margin = 4
        max_amplitude_height = (height - 2 * margin) // 2
        center_y = y + height // 2

        # Progress threshold
        progress_idx = int(self._playback_progress * num_samples)

        # Generate points
        points = []
        for i, amplitude in enumerate(self._waveform_data):
            px = x + margin + int((i / (num_samples - 1)) * (width - 2 * margin))
            py = center_y - int(amplitude * max_amplitude_height)
            points.append((px, py))

        # Draw line segments dengan warna berbeda untuk progress
        for i in range(len(points) - 1):
            color = self.progress_color if i < progress_idx else self.waveform_color
            cv2.line(frame, points[i], points[i + 1], color, 2, cv2.LINE_AA)

        # Draw center line (subtle)
        cv2.line(
            frame,
            (x + margin, center_y),
            (x + width - margin, center_y),
            (60, 60, 60),
            1
        )

    def _draw_placeholder(
        self,
        frame: np.ndarray,
        x: int,
        y: int,
        width: int,
        height: int
    ) -> None:
        """
        Gambar placeholder text ketika tidak ada waveform data.
        """
        text = "No Waveform Data"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1

        # Hitung ukuran text
        (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)

        # Center text
        text_x = x + (width - text_w) // 2
        text_y = y + (height + text_h) // 2

        cv2.putText(
            frame,
            text,
            (text_x, text_y),
            font,
            font_scale,
            (100, 100, 100),
            thickness,
            cv2.LINE_AA
        )


if __name__ == "__main__":
    # Test waveform view dengan dummy data
    print("=== Test Waveform View ===")

    # Buat dummy waveform data (sine wave)
    import math
    num_samples = 100
    dummy_waveform = [abs(math.sin(i / 10)) for i in range(num_samples)]

    # Inisialisasi waveform view
    waveform_bars = WaveformView(style="bars")
    waveform_line = WaveformView(style="line", waveform_color=(150, 255, 150))

    # Set data
    waveform_bars.set_waveform_data(dummy_waveform)
    waveform_line.set_waveform_data(dummy_waveform)

    # Test visualisasi
    print("\nTest 1: Static waveform (bars & line)...")
    frame_width, frame_height = 800, 600

    for frame_idx in range(150):  # 5 detik @ 30fps
        # Buat frame
        frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
        frame[:] = (20, 20, 20)

        # Simulasi playback progress
        progress = (frame_idx % 100) / 100.0
        waveform_bars.set_playback_progress(progress)
        waveform_line.set_playback_progress(progress)

        # Draw waveform bars di atas
        waveform_bars.draw(frame, x=50, y=100, width=700, height=150)

        # Draw waveform line di bawah
        waveform_line.draw(frame, x=50, y=300, width=700, height=150)

        # Info text
        cv2.putText(
            frame,
            f"Bars Style - Progress: {progress*100:.0f}%",
            (50, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

        cv2.putText(
            frame,
            f"Line Style - Progress: {progress*100:.0f}%",
            (50, 280),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

        cv2.imshow("Waveform View Test", frame)

        key = cv2.waitKey(33)  # ~30fps
        if key == 27:  # ESC
            break

    # Test 2: Empty waveform (placeholder)
    print("\nTest 2: Empty waveform (placeholder)...")
    waveform_empty = WaveformView()

    for _ in range(60):
        frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
        frame[:] = (20, 20, 20)

        waveform_empty.draw(frame, x=100, y=200, width=600, height=200)

        cv2.putText(
            frame,
            "Empty Waveform Test",
            (100, 180),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

        cv2.imshow("Waveform View Test", frame)

        if cv2.waitKey(33) == 27:
            break

    cv2.destroyAllWindows()
    print("\nTest selesai!")
