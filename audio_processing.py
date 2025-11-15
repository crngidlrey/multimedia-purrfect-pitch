#!/usr/bin/env python3

"""
Audio Pitch Shifting Tool
=========================
Program untuk mengubah pitch (nada) audio secara batch processing.
Mendukung format: MP3, WAV, OGG, FLAC, M4A
"""

import argparse
from pathlib import Path
import json
import librosa
import soundfile as sf
import numpy as np
from tqdm import tqdm

def pitch_shift_file(input_path: Path, output_path: Path, n_semitones: float, sr_target=None):
    """
    Mengubah pitch audio tanpa mengubah kecepatan/tempo.
    
    Fungsi ini melakukan pitch shifting pada file audio dengan mempertahankan
    jumlah channel asli (mono/stereo). Hasil disimpan dalam format WAV.
    
    Args:
        input_path (Path): Path file audio input
        output_path (Path): Path file audio output (format WAV)
        n_semitones (float): Jumlah semitone untuk shift
                           (negatif = pitch down, positif = pitch up)
        sr_target (int, optional): Target sample rate. None = gunakan sample rate asli
    
    Returns:
        tuple: (sample_rate, duration) dari file output
            - sample_rate (int): Sample rate file output
            - duration (float): Durasi audio dalam detik
    """

    # Load audio dengan mempertahankan channel asli (mono/stereo)
    # mono=False: tidak konversi ke mono, sr=None: pakai sample rate asli
    y, sr = librosa.load(str(input_path), sr=sr_target, mono=False)

    # Proses audio berdasarkan jumlah dimensi array
    # ndim == 1: audio mono, ndim > 1: audio multi-channel (stereo/surround)
    if y.ndim == 1:
        # Audio mono: proses langsung dalam satu operasi
        y_shift = librosa.effects.pitch_shift(y, sr=sr, n_steps=n_semitones)
        sf.write(str(output_path), y_shift, sr)
    else:
        # Audio multi-channel: proses setiap channel secara terpisah
        # y shape: (channels, samples)
        channels = []
        for ch_idx in range(y.shape[0]):
            ch = y[ch_idx]
            # Terapkan pitch shift ke setiap channel dengan parameter yang sama
            ch_shift = librosa.effects.pitch_shift(ch, sr=sr, n_steps=n_semitones)
            channels.append(ch_shift)
        
        # Samakan panjang semua channel jika berbeda setelah pitch shift
        # Padding dengan nilai 0 di akhir array
        max_len = max(map(len, channels))
        channels = [np.pad(c, (0, max_len - len(c)), mode="constant") for c in channels]
        
        # Gabungkan semua channel menjadi satu array 2D
        y_shift = np.vstack(channels)  # shape: (channels, samples)
        
        # Transpose karena soundfile butuh format (samples, channels)
        sf.write(str(output_path), y_shift.T, sr)

    # Dapatkan durasi file output untuk metadata
    out_duration = librosa.get_duration(filename=str(output_path))
    return sr, out_duration

def generate_waveform_data(audio_path: Path, num_samples: int = 512):
    """
    Membuat data waveform ter-normalisasi dari file audio.

    Args:
        audio_path (Path): Path file audio sumber.
        num_samples (int): Jumlah sampel waveform yang diinginkan.

    Returns:
        list[float]: List amplitudo (0..1) sebanyak num_samples.
    """
    if num_samples <= 0:
        raise ValueError("num_samples harus lebih besar dari 0")

    y, _ = librosa.load(str(audio_path), sr=None, mono=True)
    if y.size == 0:
        return [0.0] * num_samples

    amplitudes = np.abs(y)
    positions = np.linspace(0, amplitudes.size - 1, num=num_samples)
    samples = np.interp(positions, np.arange(amplitudes.size), amplitudes)

    max_amp = samples.max()
    if max_amp > 0:
        samples = samples / max_amp

    return samples.tolist()


def find_audio_files(folder: Path, exts=(".mp3", ".wav", ".ogg", ".flac", ".m4a")):
    """
    Mencari semua file audio dengan ekstensi yang didukung dalam folder. Fungsi ini melakukan pencarian file berdasarkan ekstensi yang ditentukan dan mengembalikan list file yang sudah diurutkan.
    
    Args:
        folder (Path): Path folder yang akan dicari
        exts (tuple): Tuple ekstensi file yang didukung.
                     Default: (".mp3", ".wav", ".ogg", ".flac", ".m4a")
    
    Returns:
        list: List Path object dari file audio yang ditemukan (sorted alfabetis)
    """
    files = []
    # Loop setiap ekstensi dan cari file yang cocok
    for ext in exts:
        # Gunakan glob untuk pattern matching, lalu sort hasilnya
        files.extend(sorted(folder.glob(f"*{ext}")))
    return files

def main():
    """
    Fungsi utama untuk menjalankan batch processing pitch shift.
    
    Alur kerja:
    1. Parse command-line arguments untuk konfigurasi
    2. Validasi folder input
    3. Cari semua file audio dengan ekstensi yang didukung
    4. Proses setiap file dengan pitch shift
    5. Simpan metadata hasil processing ke file JSON
    
    Command-line Arguments:
        --in_folder, -i: Folder input (default: asset)
        --out_folder, -o: Folder output (default: asset_output)
        --semitones, -s: Jumlah semitone shift (default: -5.0)
        --sr: Target sample rate (default: None/original)
    
    Output:
        - File audio WAV yang sudah di-pitch shift
        - File metadata.json berisi detail processing
    """
    # Setup argument parser untuk command-line interface
    parser = argparse.ArgumentParser(description="Batch pitch-down audio files in a folder.")
    parser.add_argument("--in_folder", "-i", type=Path, default=Path("asset"), help="Input folder (default: asset)")
    parser.add_argument("--out_folder", "-o", type=Path, default=Path("asset_output"), help="Output folder (default: asset_output)")
    parser.add_argument("--semitones", "-s", type=float, default=-5.0, help="Semitones to shift (negative => pitch down). Default -5")
    parser.add_argument("--sr", type=int, default=None, help="Target sample rate (None = keep original)")
    args = parser.parse_args()

    # Ekstrak nilai dari arguments yang di-parse
    in_folder: Path = args.in_folder
    out_folder: Path = args.out_folder
    semitones: float = args.semitones
    sr_target = args.sr

    # Validasi: pastikan folder input ada dan merupakan direktori
    if not in_folder.exists() or not in_folder.is_dir():
        print(f"[ERROR] Input folder not found: {in_folder}")
        return

    # Buat folder output jika belum ada (parents=True: buat parent dirs juga)
    out_folder.mkdir(parents=True, exist_ok=True)

    # Cari semua file audio di folder input
    files = find_audio_files(in_folder)
    if not files:
        print(f"[INFO] No audio files found in {in_folder}. Supported exts: .mp3 .wav .ogg .flac .m4a")
        return

    # Inisialisasi list untuk menyimpan metadata setiap file
    metadata = []
    print(f"[INFO] Found {len(files)} audio files. Processing with semitones={semitones} ...")

    # Loop setiap file dan proses dengan progress bar (tqdm)
    for f in tqdm(files, desc="Processing audio files"):
        try:
            # Generate nama file output dengan format: nama_pitch{semitones}.wav
            name = f.stem
            out_name = f"{name}_pitch{int(semitones)}.wav" if float(semitones).is_integer() else f"{name}_pitch{semitones}.wav"
            out_path = out_folder / out_name

            # Proses pitch shift dan dapatkan sample rate + durasi output
            sr_out, out_duration = pitch_shift_file(f, out_path, n_semitones=semitones, sr_target=sr_target)

            # Dapatkan durasi file asli untuk perbandingan di metadata
            orig_duration = librosa.get_duration(filename=str(f))
            
            # Simpan informasi detail processing ke metadata
            metadata.append({
                "input": str(f.relative_to(Path.cwd())),
                "output": str(out_path.relative_to(Path.cwd())),
                "semitones": semitones,
                "orig_duration_s": round(orig_duration, 3),
                "out_duration_s": round(out_duration, 3),
                "sr_out": sr_out
            })
        except Exception as e:
            # Tangkap error tapi lanjutkan processing file lain
            print(f"[WARN] Failed processing {f.name}: {e}")

    # Tulis semua metadata ke file JSON
    meta_path = out_folder / "metadata.json"
    with open(meta_path, "w", encoding="utf-8") as mf:
        json.dump(metadata, mf, indent=2, ensure_ascii=False)

    print(f"[DONE] Processed {len(metadata)} files. Outputs in: {out_folder}")
    print(f"[INFO] Metadata written to {meta_path}")

if __name__ == "__main__":
    main()
    
