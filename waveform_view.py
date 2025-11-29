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
        SEMUA elemen dijamin berada DALAM box [x, y, width, height].

        Args:
            frame (np.ndarray): Frame target (BGR)
            x (int): Posisi X (top-left) dari bounding box
            y (int): Posisi Y (top-left) dari bounding box
            width (int): Lebar TOTAL area waveform (termasuk border/margin)
            height (int): Tinggi TOTAL area waveform (termasuk border/margin)
            show_border (bool): Tampilkan border atau tidak
        """
        # Validasi bounds - pastikan tidak keluar frame
        frame_h, frame_w = frame.shape[:2]
        if x < 0 or y < 0 or x + width > frame_w or y + height > frame_h:
            # Clipping jika keluar frame
            x = max(0, x)
            y = max(0, y)
            width = min(width, frame_w - x)
            height = min(height, frame_h - y)

        if width <= 0 or height <= 0:
            return

        # TIDAK menggambar background - sudah ada di main.py
        # Background panel transparan sudah di-handle di main.py

        # Draw border DALAM area box (thickness mengurangi area dalam)
        # Border tidak boleh menambah ukuran box!
        if show_border:
            # Border 1px thickness agar tidak memakan banyak space
            cv2.rectangle(frame, (x, y), (x + width - 1, y + height - 1), self.border_color, 1)

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
        DIJAMIN semua bars berada DALAM box [x, y, width, height].
        """
        num_samples = len(self._waveform_data)
        if num_samples == 0:
            return

        # === STEP 1: Hitung area untuk menggambar (center-aligned) ===
        margin_horizontal = 12
        margin_vertical = 12
        usable_width = width - 2 * margin_horizontal
        usable_height = height - 2 * margin_vertical

        if usable_width <= 0 or usable_height <= 0:
            return

        # === STEP 2: Hitung ukuran bar ===
        bar_spacing = 2
        available_for_bars = usable_width - (num_samples - 1) * bar_spacing
        bar_width = max(1, available_for_bars // num_samples)

        # Tinggi maksimum bar (setengah dari usable_height karena symmetrical)
        max_bar_height = usable_height // 2

        # === STEP 3: Hitung center line ===
        center_y = y + height // 2

        # === STEP 4: Bounding box untuk clipping ===
        box_left = x + margin_horizontal
        box_right = x + width - margin_horizontal
        box_top = y + margin_vertical
        box_bottom = y + height - margin_vertical

        # Progress threshold (sample index)
        progress_idx = int(self._playback_progress * num_samples)

        # === STEP 5: Draw bars dengan clipping ketat ===
        for i, amplitude in enumerate(self._waveform_data):
            # Posisi X bar
            total_bar_width = num_samples * bar_width + (num_samples - 1) * bar_spacing
            start_x = x + (width - total_bar_width) // 2
            bar_x = start_x + i * (bar_width + bar_spacing)

            # CLAMP X agar tidak keluar box
            bar_x = max(box_left, min(bar_x, box_right - bar_width))

            # Tinggi bar berdasarkan amplitude (0.0 - 1.0)
            # SCALE dan CLAMP amplitude
            amplitude_clamped = max(0.0, min(1.0, amplitude))
            bar_h = int(amplitude_clamped * max_bar_height)

            # Warna: progress color jika sudah diplay, waveform color jika belum
            color = self.progress_color if i < progress_idx else self.waveform_color

            # Draw bar (dari center ke atas dan bawah)
            if bar_h > 0:
                # Koordinat bar atas
                top_y = center_y - bar_h
                # Koordinat bar bawah
                bottom_y = center_y + bar_h

                # CLAMP Y agar tidak keluar box
                top_y = max(box_top, min(top_y, center_y))
                bottom_y = max(center_y, min(bottom_y, box_bottom))

                # Bar atas (jika ada ruang)
                if top_y < center_y:
                    cv2.rectangle(
                        frame,
                        (bar_x, top_y),
                        (bar_x + bar_width, center_y),
                        color,
                        -1
                    )

                # Bar bawah (jika ada ruang)
                if bottom_y > center_y:
                    cv2.rectangle(
                        frame,
                        (bar_x, center_y),
                        (bar_x + bar_width, bottom_y),
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
        DIJAMIN semua line berada DALAM box [x, y, width, height].
        """
        num_samples = len(self._waveform_data)
        if num_samples < 2:
            return

        # === STEP 1: Hitung area yang aman untuk menggambar ===
        margin_horizontal = 12  # Safety margin kiri-kanan
        margin_vertical = 12    # Safety margin atas-bawah

        usable_width = width - 2 * margin_horizontal
        usable_height = height - 2 * margin_vertical

        if usable_width <= 0 or usable_height <= 0:
            return

        # Tinggi maksimum amplitude (setengah dari usable_height)
        max_amplitude_height = usable_height // 2

        # === STEP 2: Hitung center line ===
        center_y = y + height // 2

        # === STEP 3: Bounding box untuk clipping ===
        box_left = x + margin_horizontal
        box_right = x + width - margin_horizontal
        box_top = y + margin_vertical
        box_bottom = y + height - margin_vertical

        # Progress threshold
        progress_idx = int(self._playback_progress * num_samples)

        # === STEP 4: Generate points dengan clipping ===
        points = []
        for i, amplitude in enumerate(self._waveform_data):
            # Mapping index sample -> X coordinate
            px = box_left + int((i / (num_samples - 1)) * usable_width)

            # CLAMP amplitude ke [0.0, 1.0]
            amplitude_clamped = max(0.0, min(1.0, amplitude))

            # Mapping amplitude -> Y coordinate (dari center)
            py = center_y - int(amplitude_clamped * max_amplitude_height)

            # CLAMP koordinat X dan Y agar tidak keluar box
            px = max(box_left, min(px, box_right))
            py = max(box_top, min(py, box_bottom))

            points.append((px, py))

        # === STEP 5: Draw line segments ===
        for i in range(len(points) - 1):
            color = self.progress_color if i < progress_idx else self.waveform_color
            # cv2.line sudah melakukan clipping otomatis, tapi kita sudah clamp manual
            cv2.line(frame, points[i], points[i + 1], color, 2, cv2.LINE_AA)

        # Draw center line (subtle, di dalam box)
        cv2.line(
            frame,
            (box_left, center_y),
            (box_right, center_y),
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
