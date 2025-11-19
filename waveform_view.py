#!/usr/bin/env python3
"""
Waveform View Module.

Komponen GUI khusus untuk menampilkan visualisasi waveform audio.
Menggunakan matplotlib untuk rendering yang dapat di-embed ke window.
Menerima data waveform dari AudioManager dan menggambar ulang sesuai status game.
"""

from __future__ import annotations

from typing import List, Optional
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import cv2

# Set backend non-interactive untuk performa lebih baik
matplotlib.use('Agg')


class WaveformView:
    """
    Komponen GUI untuk visualisasi waveform audio.
    Render waveform sebagai image yang bisa ditampilkan di OpenCV window.
    """

    def __init__(
        self,
        width: int = 800,
        height: int = 150,
        color: tuple = (0, 200, 255),
        bg_color: tuple = (20, 20, 30)
    ) -> None:
        """
        Inisialisasi waveform viewer dengan konfigurasi visual.
        
        Args:
            width (int): Lebar canvas dalam pixel (default: 800)
            height (int): Tinggi canvas dalam pixel (default: 150)
            color (tuple): Warna waveform dalam BGR (default: cyan)
            bg_color (tuple): Warna background dalam BGR (default: dark blue)
        """
        self.width = width
        self.height = height
        self.color = color
        self.bg_color = bg_color
        
        # State untuk data waveform
        self._waveform_data: List[float] = []
        self._is_playing = False
        self._playback_position = 0.0  # 0.0 - 1.0 (persentase)
        
        # Setup matplotlib figure
        self._setup_figure()
        
        # Cache untuk image hasil render
        self._cached_image: Optional[np.ndarray] = None
        self._needs_redraw = True

    def _setup_figure(self) -> None:
        """
        Setup matplotlib figure dan axis dengan konfigurasi optimal.
        """
        # Hitung DPI berdasarkan ukuran yang diinginkan
        dpi = 100
        figsize = (self.width / dpi, self.height / dpi)
        
        # Buat figure dengan background transparan
        self._fig = Figure(figsize=figsize, dpi=dpi, facecolor='none')
        self._canvas = FigureCanvasAgg(self._fig)
        
        # Buat axis tanpa margin
        self._ax = self._fig.add_subplot(111)
        self._ax.set_facecolor((
            self.bg_color[2] / 255,  # R
            self.bg_color[1] / 255,  # G
            self.bg_color[0] / 255   # B
        ))
        
        # Hilangkan semua border dan ticks
        self._ax.set_xticks([])
        self._ax.set_yticks([])
        self._ax.spines['top'].set_visible(False)
        self._ax.spines['right'].set_visible(False)
        self._ax.spines['bottom'].set_visible(False)
        self._ax.spines['left'].set_visible(False)
        
        # Set tight layout untuk minimize whitespace
        self._fig.tight_layout(pad=0)

    def set_waveform_data(self, data: List[float]) -> None:
        """
        Set data waveform baru untuk ditampilkan.
        Data harus sudah ter-normalisasi (0.0 - 1.0).
        
        Args:
            data (List[float]): Data amplitudo waveform ter-normalisasi
        """
        self._waveform_data = data.copy() if data else []
        self._needs_redraw = True
        self._playback_position = 0.0

    def set_playback_state(self, is_playing: bool, position: float = 0.0) -> None:
        """
        Update status playback untuk visualisasi.
        
        Args:
            is_playing (bool): Apakah audio sedang diputar
            position (float): Posisi playback (0.0 - 1.0)
        """
        self._is_playing = is_playing
        self._playback_position = max(0.0, min(1.0, position))
        self._needs_redraw = True

    def clear(self) -> None:
        """
        Kosongkan waveform dan reset state.
        """
        self._waveform_data.clear()
        self._is_playing = False
        self._playback_position = 0.0
        self._needs_redraw = True
        self._cached_image = None

    def _render_waveform(self) -> np.ndarray:
        """
        Render waveform ke image menggunakan matplotlib.
        
        Returns:
            np.ndarray: Image BGR format (height, width, 3)
        """
        # Clear axis
        self._ax.clear()
        self._ax.set_facecolor((
            self.bg_color[2] / 255,
            self.bg_color[1] / 255,
            self.bg_color[0] / 255
        ))
        
        if not self._waveform_data:
            # Jika tidak ada data, tampilkan garis datar
            self._ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.3)
            self._ax.text(
                0.5, 0.5, 'No Audio Loaded',
                ha='center', va='center',
                color='gray', fontsize=12,
                transform=self._ax.transAxes
            )
        else:
            # Konversi data ke numpy array
            waveform = np.array(self._waveform_data)
            total = len(waveform)

            # --- tambahan: sliding window agar waveform bergerak mengikuti playback ---
            window_size = 300  # ukuran window tampilan
            center = int(self._playback_position * total)
            start = max(0, center - window_size // 2)
            end = min(total, start + window_size)
            waveform = waveform[start:end]
            # ---------------------------------------------------------------------------

            x = np.linspace(0, 1, len(waveform))
            
            # Warna waveform dalam RGB normalized
            wave_color = (
                self.color[2] / 255,  # R
                self.color[1] / 255,  # G
                self.color[0] / 255   # B
            )
            
            # Gambar waveform sebagai filled area
            self._ax.fill_between(
                x, 0, waveform,
                color=wave_color,
                alpha=0.7,
                linewidth=0
            )
            
            # Gambar outline waveform
            self._ax.plot(
                x, waveform,
                color=wave_color,
                linewidth=1.5,
                alpha=0.9
            )
            
            # Jika sedang playing, tampilkan playback position
            if self._is_playing and self._playback_position > 0:
                # posisi relatif dalam window (dibuat proporsional)
                relative_x = (center - start) / max(1, (end - start))
                self._ax.axvline(
                    x=relative_x,
                    color='red',
                    linewidth=2,
                    linestyle='-',
                    alpha=0.8
                )
        
        # Set limits
        self._ax.set_xlim(0, 1)
        self._ax.set_ylim(0, 1)
        
        # Hilangkan ticks
        self._ax.set_xticks([])
        self._ax.set_yticks([])
        
        # Render canvas ke buffer
        self._canvas.draw()
        
        # Convert canvas ke numpy array
        buf = np.frombuffer(self._canvas.buffer_rgba(), dtype=np.uint8)
        buf = buf.reshape(self._canvas.get_width_height()[::-1] + (4,))
        
        # Convert RGBA ke BGR untuk OpenCV
        img_bgr = cv2.cvtColor(buf, cv2.COLOR_RGBA2BGR)
        
        return img_bgr

    def get_image(self, force_redraw: bool = False) -> np.ndarray:
        """
        Dapatkan image waveform yang sudah di-render.
        Menggunakan cache jika tidak ada perubahan.
        
        Args:
            force_redraw (bool): Paksa render ulang meskipun tidak ada perubahan
            
        Returns:
            np.ndarray: Image BGR format (height, width, 3)
        """
        if self._needs_redraw or force_redraw or self._cached_image is None:
            self._cached_image = self._render_waveform()
            self._needs_redraw = False
        
        return self._cached_image.copy()

    def update_playback_position(self, elapsed_time: float, total_duration: float) -> None:
        """
        Update posisi playback berdasarkan waktu.
        Helper method untuk auto-update position dari timer.
        
        Args:
            elapsed_time (float): Waktu yang sudah berlalu (detik)
            total_duration (float): Total durasi audio (detik)
        """
        if total_duration > 0:
            position = elapsed_time / total_duration
            self.set_playback_state(self._is_playing, position)

    def draw_on_frame(self, frame: np.ndarray, x: int, y: int) -> None:
        """
        Gambar waveform langsung ke frame OpenCV.
        
        Args:
            frame (np.ndarray): Frame target untuk menggambar
            x (int): Posisi X (top-left corner)
            y (int): Posisi Y (top-left corner)
        """
        waveform_img = self.get_image()
        h, w = waveform_img.shape[:2]
        
        # Pastikan tidak keluar dari frame boundary
        frame_h, frame_w = frame.shape[:2]
        if y + h > frame_h:
            h = frame_h - y
            waveform_img = waveform_img[:h, :]
        if x + w > frame_w:
            w = frame_w - x
            waveform_img = waveform_img[:, :w]
        
        # Copy waveform ke frame
        if h > 0 and w > 0:
            frame[y:y+h, x:x+w] = waveform_img


if __name__ == "__main__":
    # Test waveform view dengan data dummy
    import time
    
    print("=== Test Waveform View ===")
    
    # Buat waveform viewer
    viewer = WaveformView(width=800, height=150)
    
    # Test 1: Tampilkan waveform kosong
    print("\nTest 1: Empty waveform")
    empty_img = viewer.get_image()
    cv2.imshow("Waveform Test", empty_img)
    cv2.waitKey(1000)
    
    # Test 2: Tampilkan waveform dengan data dummy (sine wave)
    print("\nTest 2: Sine wave")
    t = np.linspace(0, 4 * np.pi, 512)
    sine_wave = (np.sin(t) + 1) / 2  # Normalize ke 0-1
    viewer.set_waveform_data(sine_wave.tolist())
    
    sine_img = viewer.get_image()
    cv2.imshow("Waveform Test", sine_img)
    cv2.waitKey(1000)
    
    # Test 3: Simulasi playback
    print("\nTest 3: Simulated playback (5 seconds)")
    viewer.set_playback_state(is_playing=True, position=0.0)
    
    steps = 50
    for i in range(steps):
        position = i / (steps - 1)
        viewer.set_playback_state(is_playing=True, position=position)
        
        img = viewer.get_image()
        cv2.imshow("Waveform Test", img)
        
        key = cv2.waitKey(100)
        if key == 27:  # ESC
            break
    
    # Test 4: Gambar waveform di frame kamera
    print("\nTest 4: Overlay on camera frame")
    test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    test_frame[:] = (50, 50, 50)  # Gray background
    
    # Generate waveform baru (random)
    random_wave = np.random.rand(512) * 0.8 + 0.1
    viewer.set_waveform_data(random_wave.tolist())
    viewer.set_playback_state(is_playing=False)
    
    # Draw waveform di tengah frame
    viewer.draw_on_frame(test_frame, x=20, y=165)
    
    # Tambahkan text
    cv2.putText(
        test_frame, "Waveform Overlay Test",
        (20, 140), cv2.FONT_HERSHEY_SIMPLEX,
        0.7, (255, 255, 255), 2
    )
    
    cv2.imshow("Waveform Test", test_frame)
    cv2.waitKey(2000)
    
    cv2.destroyAllWindows()
    print("\nTest selesai!")