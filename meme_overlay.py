#!/usr/bin/env python3
"""
Meme Overlay Module.

Komponen GUI yang menggambar dua sprite meme di kiri/kanan kepala pemain.
Menangani animasi masuk/keluar serta highlight pilihan berdasarkan orientasi kepala.
Saat dipilih (highlight) gambarnya membesar sedikit.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple
import cv2
import numpy as np
import time


class MemeSprite:
    """
    Class untuk satu sprite meme dengan animasi dan state.
    """
    
    def __init__(
        self,
        image_path: Path,
        side: str,
        target_size: Tuple[int, int] = (150, 150),
        highlight_zoom: float = 1.15,  # sedikit lebih besar saat highlight
    ):
        """
        Inisialisasi meme sprite.
        
        Args:
            image_path (Path): Path ke file gambar meme
            side (str): Posisi sprite ("LEFT" atau "RIGHT")
            target_size (tuple): Ukuran target sprite (width, height)
            highlight_zoom (float): Faktor pembesaran sementara saat highlight (1.0 = no zoom)
        """
        self.image_path = image_path
        self.side = side.upper()
        self.target_size = target_size
        self.highlight_zoom = float(highlight_zoom)
        
        # Load image (bisa BGRA jika ada alpha)
        self.original_image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if self.original_image is None:
            raise FileNotFoundError(f"Tidak dapat load image: {image_path}")
        
        # Konversi ke BGRA jika belum (agar mudah menangani alpha)
        if self.original_image.ndim == 2:
            # grayscale -> convert to BGRA
            self.original_image = cv2.cvtColor(self.original_image, cv2.COLOR_GRAY2BGRA)
        elif self.original_image.shape[2] == 3:
            # BGR -> BGRA (set alpha penuh)
            bgr = self.original_image
            alpha = np.ones((bgr.shape[0], bgr.shape[1], 1), dtype=np.uint8) * 255
            self.original_image = np.concatenate([bgr, alpha], axis=2)
        # jika sudah 4 channel tetap dipakai
        
        # Resize image ke target_size **dengan menjaga aspect ratio**
        # Kita simpan base canvas ukuran target_size; saat render nanti kita skalakan canvas ini
        self.image = self._resize_keep_aspect(self.original_image, target_size)
        
        # State animasi
        self.scale = 0.0  # 0.0 - 1.0 (untuk animasi masuk/keluar)
        self.is_highlighted = False
        self.animation_start_time: Optional[float] = None
        self.target_scale = 0.0
        # Opacity (0.0 - 1.0) untuk efek fade in/out
        # Record start scale for animation interpolation
        self._start_scale: float = self.scale
        
    def _resize_keep_aspect(self, img: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """
        Resize gambar agar masuk ke target_size tanpa merubah aspect ratio.
        Hasil diletakkan di canvas transparan (BGRA) berukuran target_size.
        """
        tw, th = target_size
        h, w = img.shape[:2]
        
        # Hitung scale untuk fit ke dalam target (fit inside)
        scale = min(tw / w, th / h)
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Buat canvas transparan BGRA
        canvas = np.zeros((th, tw, 4), dtype=np.uint8)
        # Tempatkan gambar di tengah canvas
        x_off = (tw - new_w) // 2
        y_off = (th - new_h) // 2
        canvas[y_off:y_off+new_h, x_off:x_off+new_w] = resized
        return canvas
    
    def start_animation(self, target_scale: float = 1.0) -> None:
        """
        Mulai animasi scale (masuk/keluar).
        
        Args:
            target_scale (float): Target scale (1.0 = full size, 0.0 = hidden)
        """
        self.target_scale = max(0.0, min(1.0, target_scale))
        # record start value for smooth interpolation
        self._start_scale = float(self.scale)
        self.animation_start_time = time.time()
    
    def update_animation(self, duration: float = 0.1) -> bool:
        """
        Update state animasi.
        
        Args:
            duration (float): Durasi animasi dalam detik
            
        Returns:
            bool: True jika animasi masih berjalan, False jika sudah selesai
        """
        if self.animation_start_time is None:
            return False

        elapsed = time.time() - self.animation_start_time
        progress = min(1.0, elapsed / duration)

        # Ease-out animation (smooth deceleration)
        ease = 1 - (1 - progress) ** 3

        # Interpolate scale from recorded start value
        self.scale = float(self._start_scale + (self.target_scale - self._start_scale) * ease)

        # Jika sudah mencapai target, stop animasi
        if progress >= 1.0:
            self.scale = float(self.target_scale)
            self.animation_start_time = None
            return False

        return True
    
    def set_highlight(self, highlighted: bool) -> None:
        """
        Set status highlight sprite.
        
        Args:
            highlighted (bool): True untuk highlight, False untuk normal
        """
        self.is_highlighted = highlighted
    
    def get_rendered_image(self) -> Optional[np.ndarray]:
        """
        Dapatkan image sprite yang sudah di-render dengan efek.
        
        Returns:
            np.ndarray: Image BGRA (dengan alpha channel) atau None jika scale = 0
        """
        if self.scale <= 0.01:
            return None
        
        # Hitung ukuran berdasarkan scale (scales canvas)
        base_w = int(self.target_size[0] * self.scale)
        base_h = int(self.target_size[1] * self.scale)
        
        if base_w <= 0 or base_h <= 0:
            return None
        
        # Jika highlighted, tambahkan zoom factor (sementara, tanpa animasi terpisah)
        zoom = self.highlight_zoom if self.is_highlighted else 1.0
        scaled_w = max(1, int(base_w * zoom))
        scaled_h = max(1, int(base_h * zoom))
        
        # Resize sesuai final scaled size (skalakan canvas original)
        scaled_img = cv2.resize(self.image, (scaled_w, scaled_h), interpolation=cv2.INTER_AREA)
        
        # Pastikan BGRA
        if scaled_img.shape[2] == 3:
            scaled_img = cv2.cvtColor(scaled_img, cv2.COLOR_BGR2BGRA)

        # Ensure BGRA
        if scaled_img.shape[2] == 3:
            scaled_img = cv2.cvtColor(scaled_img, cv2.COLOR_BGR2BGRA)

        return scaled_img


class MemeOverlay:
    """
    Manager untuk overlay dua meme sprite di kiri dan kanan kepala.
    """
    
    def __init__(
        self,
        offset_x: int = 180,
        offset_y: int = -50,
        sprite_size: Tuple[int, int] = (240, 240)
    ):
        """
        Inisialisasi meme overlay.
        
        Args:
            offset_x (int): Jarak horizontal sprite dari kepala (pixel)
            offset_y (int): Jarak vertikal sprite dari kepala (pixel, negatif = atas)
            sprite_size (tuple): Ukuran sprite (width, height)
        """
        self.offset_x = offset_x
        self.offset_y = offset_y
        self.sprite_size = sprite_size
        
        self.left_sprite: Optional[MemeSprite] = None
        self.right_sprite: Optional[MemeSprite] = None
        
        # State untuk head position (center of face)
        self.head_position: Optional[Tuple[int, int]] = None
        # Optional face size in pixels (width, height) in the same coordinate space as head_position.
        # When provided, the overlay will position memes at the temple (pelipis) using this size.
        self.face_size: Optional[Tuple[int, int]] = None

    def load_memes(self, left_image_path: Path, right_image_path: Path) -> bool:
        """
        Load dua meme image untuk kiri dan kanan.
        
        Args:
            left_image_path (Path): Path image meme kiri
            right_image_path (Path): Path image meme kanan
            
        Returns:
            bool: True jika berhasil load kedua image
        """
        try:
            # Anda bisa atur highlight_zoom per sprite jika mau (contoh: 1.15)
            self.left_sprite = MemeSprite(left_image_path, "LEFT", self.sprite_size, highlight_zoom=1.12)
            self.right_sprite = MemeSprite(right_image_path, "RIGHT", self.sprite_size, highlight_zoom=1.12)
            return True
        except Exception as e:
            print(f"[ERROR] Gagal load meme images: {e}")
            self.left_sprite = None
            self.right_sprite = None
            return False

    def show_memes(self, animate: bool = True) -> None:
        """
        Tampilkan kedua meme sprite dengan animasi masuk.
        
        Args:
            animate (bool): True untuk animasi masuk, False untuk langsung tampil
        """
        if self.left_sprite:
            if animate:
                self.left_sprite.start_animation(target_scale=1.0)
            else:
                self.left_sprite.scale = 1.0
        
        if self.right_sprite:
            if animate:
                self.right_sprite.start_animation(target_scale=1.0)
            else:
                self.right_sprite.scale = 1.0

    def hide_memes(self, animate: bool = True) -> None:
        """
        Sembunyikan kedua meme sprite dengan animasi keluar.
        
        Args:
            animate (bool): True untuk animasi keluar, False untuk langsung hilang
        """
        if self.left_sprite:
            if animate:
                self.left_sprite.start_animation(target_scale=0.0)
            else:
                self.left_sprite.scale = 0.0
        
        if self.right_sprite:
            if animate:
                self.right_sprite.start_animation(target_scale=0.0)
            else:
                self.right_sprite.scale = 0.0

    def set_head_position(self, x: int, y: int, face_size: Optional[Tuple[int, int]] = None) -> None:
        """
        Set posisi kepala untuk anchor point sprite.
        
        Args:
            x (int): Koordinat X center kepala
            y (int): Koordinat Y center kepala
        """
        self.head_position = (x, y)
        # Store face size (width, height) when available so draw() can compute temple anchors
        if face_size is not None:
            self.face_size = face_size
        else:
            self.face_size = None

    def set_highlight(self, side: Optional[str]) -> None:
        """
        Set highlight pada salah satu sprite berdasarkan orientasi kepala.
        
        Args:
            side (str): "LEFT", "RIGHT", atau None untuk tidak highlight
        """
        if self.left_sprite:
            self.left_sprite.set_highlight(side == "LEFT")
        
        if self.right_sprite:
            self.right_sprite.set_highlight(side == "RIGHT")

    def update(self) -> None:
        """
        Update animasi kedua sprite.
        Panggil method ini di setiap frame.
        """
        if self.left_sprite:
            self.left_sprite.update_animation()
        
        if self.right_sprite:
            self.right_sprite.update_animation()

    def draw(self, frame: np.ndarray) -> None:
        """
        Gambar kedua sprite meme ke frame.
        
        Args:
            frame (np.ndarray): Frame target (BGR format)
        """
        if self.head_position is None:
            return
        
        head_x, head_y = self.head_position
        # If face size is known, position sprites at the temple (pelipis) relative to face size.
        # Otherwise fallback to using configured offsets.
        use_temple = self.face_size is not None

        # Gambar sprite kiri
        if self.left_sprite:
            left_img = self.left_sprite.get_rendered_image()
            if left_img is not None:
                if use_temple and self.face_size:
                    face_w, face_h = self.face_size
                    # Scale visual sprite size relative to face width so it shrinks when head is far
                    # Preferred sizing: a bit smaller than face width so the meme fits near the temple.
                    # Multiplier and bounds chosen for balanced appearance across webcams.
                    desired_w = max(80, min(300, int(face_w * 0.85)))
                    # Maintain aspect ratio of left_img
                    if left_img.shape[1] > 0 and left_img.shape[1] != desired_w:
                        scale_factor = desired_w / float(left_img.shape[1])
                        new_h = max(1, int(left_img.shape[0] * scale_factor))
                        left_img = cv2.resize(left_img, (desired_w, new_h), interpolation=cv2.INTER_AREA)

                    # temple offset ~ 0.48 * face width, vertical moved further up to forehead/hairline
                    temple_dx = max(18, int(face_w * 0.50))
                    temple_dy = -max(18, int(face_h * 0.70))
                    sprite_x = head_x - temple_dx - left_img.shape[1] // 2
                    sprite_y = head_y + temple_dy - left_img.shape[0] // 2
                else:
                    # Hitung posisi sprite kiri (di kiri kepala)
                    sprite_x = head_x - self.offset_x - left_img.shape[1] // 2
                    sprite_y = head_y + self.offset_y - left_img.shape[0] // 2
                self._draw_sprite(frame, left_img, sprite_x, sprite_y)
        
        # Gambar sprite kanan
        if self.right_sprite:
            right_img = self.right_sprite.get_rendered_image()
            if right_img is not None:
                if use_temple and self.face_size:
                    face_w, face_h = self.face_size
                    # Preferred sizing for right sprite as well
                    desired_w = max(80, min(300, int(face_w * 0.85)))
                    if right_img.shape[1] > 0 and right_img.shape[1] != desired_w:
                        scale_factor = desired_w / float(right_img.shape[1])
                        new_h = max(1, int(right_img.shape[0] * scale_factor))
                        right_img = cv2.resize(right_img, (desired_w, new_h), interpolation=cv2.INTER_AREA)

                    temple_dx = max(18, int(face_w * 0.50))
                    temple_dy = -max(18, int(face_h * 0.70))
                    sprite_x = head_x + temple_dx - right_img.shape[1] // 2
                    sprite_y = head_y + temple_dy - right_img.shape[0] // 2
                else:
                    # Hitung posisi sprite kanan (di kanan kepala)
                    sprite_x = head_x + self.offset_x - right_img.shape[1] // 2
                    sprite_y = head_y + self.offset_y - right_img.shape[0] // 2
                self._draw_sprite(frame, right_img, sprite_x, sprite_y)

    def _draw_sprite(self, frame: np.ndarray, sprite: np.ndarray, x: int, y: int) -> None:
        """
        Gambar sprite dengan alpha blending ke frame.
        
        Args:
            frame (np.ndarray): Frame target
            sprite (np.ndarray): Sprite image (BGRA)
            x (int): Posisi X (top-left)
            y (int): Posisi Y (top-left)
        """
        sprite_h, sprite_w = sprite.shape[:2]
        frame_h, frame_w = frame.shape[:2]
        
        # Clipping untuk memastikan sprite tidak keluar dari frame
        x0, y0 = x, y
        x1, y1 = x + sprite_w, y + sprite_h
        
        # Hitung overlap region
        ox0 = max(0, x0)
        oy0 = max(0, y0)
        ox1 = min(frame_w, x1)
        oy1 = min(frame_h, y1)
        
        if ox0 >= ox1 or oy0 >= oy1:
            return  # tidak ada overlap
        
        sx0 = ox0 - x0
        sy0 = oy0 - y0
        sx1 = sx0 + (ox1 - ox0)
        sy1 = sy0 + (oy1 - oy0)
        
        sprite_crop = sprite[sy0:sy1, sx0:sx1]
        sprite_hc, sprite_wc = sprite_crop.shape[:2]
        
        # Extract alpha channel
        if sprite_crop.shape[2] == 4:
            alpha = sprite_crop[:, :, 3] / 255.0
            sprite_bgr = sprite_crop[:, :, :3]
        else:
            alpha = np.ones((sprite_hc, sprite_wc))
            sprite_bgr = sprite_crop
        
        # Alpha blending
        roi = frame[oy0:oy1, ox0:ox1].astype(np.float32)
        for c in range(3):
            roi[:, :, c] = (
                alpha * sprite_bgr[:, :, c] +
                (1 - alpha) * roi[:, :, c]
            )
        frame[oy0:oy1, ox0:ox1] = roi.astype(np.uint8)

    def clear(self) -> None:
        """
        Bersihkan semua sprite dan reset state.
        """
        self.left_sprite = None
        self.right_sprite = None
        self.head_position = None


if __name__ == "__main__":
    # Test meme overlay dengan dummy data
    print("=== Test Meme Overlay ===")
    
    # Setup dummy meme images
    # CATATAN: Ganti dengan path image yang valid untuk test
    left_meme = Path("asset/kucing1.png")
    right_meme = Path("asset/kucing2.png")
    
    # Buat dummy meme jika tidak ada
    if not left_meme.exists() or not right_meme.exists():
        print("[INFO] Membuat dummy meme images untuk test...")
        dummy_folder = Path("asset")
        dummy_folder.mkdir(exist_ok=True)
        
        # Buat dummy image (circle dengan text) sebagai BGRA agar alpha tersedia
        for idx, path in enumerate([left_meme, right_meme], 1):
            if not path.exists():
                h, w = 200, 200
                # Buat background transparan
                img = np.zeros((h, w, 4), dtype=np.uint8)
                color = (255, 100, 100, 255) if idx == 1 else (100, 100, 255, 255)
                # gambar circle di tengah (B,G,R,A)
                cv2.circle(img, (w//2, h//2), 80, color, -1)
                # teks putih (gunakan BGR kemudian set alpha 255 untuk area teks)
                # buat layer BGR untuk teks, lalu gabungkan ke BGRA agar aman
                bgr_layer = img[:, :, :3].copy()
                cv2.putText(
                    bgr_layer, f"MEME {idx}",
                    (40, 110), cv2.FONT_HERSHEY_DUPLEX,
                    0.8, (255, 255, 255), 2, lineType=cv2.LINE_AA
                )
                img[:, :, :3] = bgr_layer
                # buat alpha dari luminance sederhana supaya teks dan circle punya alpha
                gray = cv2.cvtColor(bgr_layer, cv2.COLOR_BGR2GRAY)
                _, alpha = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
                img[:, :, 3] = alpha
                # simpan PNG
                cv2.imwrite(str(path), img)
        print("[INFO] Dummy meme images dibuat!")
    
    # Inisialisasi overlay
    overlay = MemeOverlay(offset_x=200, offset_y=-80, sprite_size=(150, 150))
    
    if overlay.load_memes(left_meme, right_meme):
        print("[OK] Meme images loaded!")
        
        # Buat dummy frame (640x480)
        frame_width, frame_height = 640, 480
        
        # Test 1: Animasi masuk
        print("\nTest 1: Animasi masuk...")
        overlay.set_head_position(frame_width // 2, frame_height // 2)
        overlay.show_memes(animate=True)
        
        for i in range(60):  # 2 detik @ 30fps
            frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
            frame[:] = (50, 50, 50)
            
            # Update dan gambar
            overlay.update()
            overlay.draw(frame)
            
            # Info text
            cv2.putText(
                frame, "Animasi Masuk",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (255, 255, 255), 2, lineType=cv2.LINE_AA
            )
            
            cv2.imshow("Meme Overlay Test", frame)
            if cv2.waitKey(33) == 27:  # ESC
                break
        
        # Test 2: Highlight kiri-kanan (gambar membesar saat highlight)
        print("\nTest 2: Highlight LEFT -> RIGHT -> CENTER...")
        states = [
            ("LEFT", 60),
            ("RIGHT", 60),
            (None, 60)
        ]
        
        for state, frames in states:
            overlay.set_highlight(state)
            
            for i in range(frames):
                frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
                frame[:] = (50, 50, 50)
                
                overlay.update()
                overlay.draw(frame)
                
                # Info text
                text = f"Highlight: {state if state else 'NONE'}"
                cv2.putText(
                    frame, text,
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 255), 2, lineType=cv2.LINE_AA
                )
                
                cv2.imshow("Meme Overlay Test", frame)
                if cv2.waitKey(33) == 27:
                    break
        
        # Test 3: Animasi keluar
        print("\nTest 3: Animasi keluar...")
        overlay.hide_memes(animate=True)
        
        for i in range(60):
            frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
            frame[:] = (50, 50, 50)
            
            overlay.update()
            overlay.draw(frame)
            
            cv2.putText(
                frame, "Animasi Keluar",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (255, 255, 255), 2, lineType=cv2.LINE_AA
            )
            
            cv2.imshow("Meme Overlay Test", frame)
            if cv2.waitKey(33) == 27:
                break
        
        cv2.destroyAllWindows()
        print("\nTest selesai!")
    else:
        print("[ERROR] Gagal load meme images!")
