# 🐱 Purrfect Pitch - Interactive Cat Sound Matching Game

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![Pygame](https://img.shields.io/badge/Pygame-2.5+-00A86B?style=for-the-badge&logo=python&logoColor=white)
![Status](https://img.shields.io/badge/Status-Production_Ready-success?style=for-the-badge)

**Tugas Besar Mata Kuliah Multimedia**
Institut Teknologi Sumatera


</div>

---

## 👥 Tim Pengembang

| Nama                 | NIM        | ID GitHub                                                |
|----------------------|------------|----------------------------------------------------------|
| Elma Nurul Fatika    | 122140069  | [122140069-ElmaNF](https://github.com/122140069-ElmaNF)  |
| Lois Novel E Gurning | 122140098  | [crngidlrey](https://github.com/crngidlrey)              |
| Dina Rahma Dita      | 122140184  | [dinarahmadita12](https://github.com/dinarahmadita12)    |

---

## 📋 Tentang Proyek

**Purrfect Pitch** adalah game interaktif multimedia yang menggabungkan **audio processing**, **computer vision**, dan **game logic**. Pemain mendengarkan suara kucing yang pitch-nya sudah dimodifikasi, lalu memilih gambar kucing yang sesuai dengan **memiringkan kepala** (face tracking) atau menggunakan keyboard.


### ✨ Fitur Utama

- 🎵 **Audio Processing** - Pitch shifting menggunakan librosa
- 👁️ **Face Tracking** - Deteksi kemiringan kepala (OpenCV Haar Cascade, threshold 8°)
- 🎮 **Game Logic** - Timer 30 detik, skor otomatis, soal diacak
- 🖥️ **GUI** - Interface interaktif dengan Pygame
- 📊 **Waveform Viz** - Visualisasi audio real-time

---

## 🛠️ Technology Stack

| Category | Technology |
|----------|-----------|
| **Language** | Python 3.11+ |
| **Audio** | Librosa, SoundFile |
| **Computer Vision** | OpenCV |
| **Game Engine** | Pygame |
| **Numerical** | NumPy |

---

## 📥 Instalasi

### Prasyarat
- Python 3.11+
- Speaker/Headphone

### Cara Instalasi

```bash
# Clone repository
git clone https://github.com/crngidlrey/multimedia-purrfect-pitch.git
cd multimedia-purrfect-pitch

# Install dependencies
pip install -r requirements.txt

# Generate audio files (opsional)
python audio_processing.py -i asset -o asset_output -s -5

# Generate metadata
python generate_metadata.py

# Jalankan game
python main.py
```

---

## 📁 Struktur Proyek

```
multimedia-purrfect-pitch/
├── asset/                      # Asset asli (10 audio + 10 gambar)
├── asset_output/               # Audio yang sudah diproses + metadata.json
├── main.py                     # Main game loop
├── angle_face_tracker.py       # Modul face tracking
├── audio_processing.py         # Utilitas audio
├── game_logic.py               # Manajemen state game
├── generate_metadata.py        # Generator metadata
├── gui.py                      # Manajemen GUI
└── requirements.txt            # Dependencies
```

---

## 🎮 Cara Bermain

### Kontrol

**Mode Face Tracking** (default):
- `SPACE` - Mulai/Restart game
- **Miringkan kepala KIRI** (> 12°) → Pilih kiri
- **Miringkan kepala KANAN** (> 12°) → Pilih kanan
- `ESC` - Keluar

**Mode Keyboard** (fallback):
- `LEFT ARROW` - Pilih kiri
- `RIGHT ARROW` - Pilih kanan

### Alur Permainan

1. Setting kamera untuk input video pada `face tracker.py`. Gunakan (0) untuk kamera bawaan device dan (1) untuk webcam/camera eksternal.
2. Tekan `SPACE` untuk mulai
3. Dengarkan audio kucing
4. Tunggu audio selesai (opsional)
5. Pilih gambar dengan head tilt atau keyboard
6. Dapat feedback (benar/salah)
7. Soal berikutnya muncul otomatis
8. Game over setelah 45 detik

---

## 📊 Logbook Pengembangan

<details>
<summary><b>Lihat Riwayat Perkembangan Proyek</b></summary>

| Minggu | Tanggal | Progress |
|--------|---------|----------|
| 1 | 27/10/2025 - 03/11/2025 | • Brainstorming ide proyek<br>• Pencarian referensi game serupa<br>• Pembuatan repository GitHub |
| 2 | 03/11/2025 - 10/11/2025 | • Breakdown ide dan fitur game<br>• Merancang struktur code (MVC pattern)<br>• Desain GUI mockup |
| 3 | 10/11/2025 - 17/11/2025 | • Pengumpulan asset (10 audio kucing + 10 gambar)<br>• Implementasi audio processing (librosa pitch shifting)<br>• Setup face tracking dengan OpenCV |
| 4 | 17/11/2025 - 24/11/2025 | • Implementasi game logic dan state management<br>• Integrasi GUI dengan Pygame<br>• Mulai menyusun laporan dokumentasi |
| 5 | 24/11/2025 - 01/12/2025 | • Revisi code (debugging audio looping bug)<br>• Optimasi head tilt detection<br>• Finalisasi code dan laporan |

</details>

---

## 📚 Referensi

- [Librosa Documentation](https://librosa.org/) – Audio processing
- [OpenCV Documentation](https://docs.opencv.org/) – Computer vision
- [Pygame Documentation](https://www.pygame.org/docs/) – Game development
- [MediaPipe Documentation](https://google.github.io/mediapipe/) – Face tracking
- Inspirasi ide awal:
  - [TikTok: anna_shimmy - Test Math](https://www.tiktok.com/@anna_shimmy/video/7121219892405112107)
  - [TikTok: Guessing Cat](https://vt.tiktok.com/ZSy1Gc2wo/)

---

## 🙏 Ucapan Terima Kasih

- Dosen pengampu: Martin Clinton Tosima Manullang, S.T., M.T., Ph.D.
- Rekan-rekan kelompok yang telah berkontribusi

---

<div align="center">

**Made with ❤️ for Multimedia Course**
Institut Teknologi Sumatera © 2025

[![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?logo=opencv&logoColor=white)](https://opencv.org/)
[![Pygame](https://img.shields.io/badge/Pygame-00A86B?logo=python&logoColor=white)](https://www.pygame.org/)

[⬆ Back to Top](#-purrfect-pitch---interactive-cat-sound-matching-game)

</div>
