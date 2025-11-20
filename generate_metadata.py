#!/usr/bin/env python3
"""
Generate Metadata for Game.

Script untuk membuat metadata game yang berisi mapping antara:
- Audio file yang sudah di-pitch shift
- Gambar kucing yang sesuai
- Pasangan meme untuk pilihan kiri/kanan
- Jawaban yang benar (LEFT/RIGHT)
"""

from pathlib import Path
import json
import random
import librosa


def generate_game_metadata(
    asset_folder: Path = Path("asset"),
    output_folder: Path = Path("asset_output"),
    metadata_file: Path = Path("asset_output/game_metadata.json"),
    num_questions: int = 14
):
    """
    Generate metadata untuk game dari asset yang tersedia.

    Args:
        asset_folder (Path): Folder berisi asset asli (gambar kucing)
        output_folder (Path): Folder berisi audio yang sudah diproses
        metadata_file (Path): Path file metadata output
        num_questions (int): Jumlah soal yang akan dibuat
    """
    print(f"[INFO] Generating game metadata...")

    # Cari semua file audio yang sudah diproses (pitch-shifted)
    audio_files = sorted(output_folder.glob("audio-kucing*_pitch-*.wav"))

    # Cari semua gambar kucing
    image_files = sorted(asset_folder.glob("kucing*.png"))

    if not audio_files:
        print(f"[ERROR] Tidak ada audio file di {output_folder}")
        return

    if not image_files:
        print(f"[ERROR] Tidak ada image file di {asset_folder}")
        return

    print(f"[INFO] Ditemukan {len(audio_files)} audio files")
    print(f"[INFO] Ditemukan {len(image_files)} image files")

    # Generate questions
    questions = []

    # Untuk setiap audio, buat soal dengan 2 pilihan gambar
    for i, audio_path in enumerate(audio_files[:num_questions]):
        # Extract ID dari nama file (contoh: audio-kucing1_pitch-5.wav -> 1)
        audio_name = audio_path.stem
        # Parse ID dari nama file (ambil angka setelah "kucing")
        try:
            parts = audio_name.split("_")[0]  # audio-kucing1
            audio_id = int(parts.replace("audio-kucing", ""))
        except:
            print(f"[WARN] Gagal parse ID dari {audio_name}, skip")
            continue

        # Gambar yang sesuai (correct answer)
        correct_image_name = f"kucing{audio_id}.png"
        correct_image_path = asset_folder / correct_image_name

        if not correct_image_path.exists():
            print(f"[WARN] Gambar {correct_image_name} tidak ditemukan, skip")
            continue

        # Pilih gambar lain secara random untuk pilihan yang salah
        other_images = [img for img in image_files if img.name != correct_image_name]
        if not other_images:
            print(f"[WARN] Tidak ada gambar lain untuk soal {audio_id}, skip")
            continue

        wrong_image_path = random.choice(other_images)

        # Random posisi: correct di kiri atau kanan
        correct_side = random.choice(["LEFT", "RIGHT"])

        if correct_side == "LEFT":
            left_meme = str(correct_image_path).replace("\\", "/")
            right_meme = str(wrong_image_path).replace("\\", "/")
        else:
            left_meme = str(wrong_image_path).replace("\\", "/")
            right_meme = str(correct_image_path).replace("\\", "/")

        # Dapatkan durasi audio
        try:
            duration = librosa.get_duration(path=str(audio_path))
        except:
            # Fallback jika librosa gagal
            try:
                duration = librosa.get_duration(filename=str(audio_path))
            except:
                duration = 3.0  # default fallback

        # Buat question entry
        question = {
            "id": f"q{i+1}",
            "audio_id": audio_id,
            "audio_path": str(audio_path).replace("\\", "/"),
            "duration": round(duration, 2),
            "left_meme": left_meme,
            "right_meme": right_meme,
            "correct_side": correct_side,
            "correct_image": str(correct_image_path).replace("\\", "/")
        }

        questions.append(question)
        print(f"  [OK] Question {i+1}: audio-kucing{audio_id} -> {correct_side}")

    # Simpan metadata
    metadata = {
        "version": "1.0",
        "total_questions": len(questions),
        "game_duration_seconds": 30,
        "questions": questions
    }

    metadata_file.parent.mkdir(parents=True, exist_ok=True)
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"\n[DONE] Generated {len(questions)} questions")
    print(f"[INFO] Metadata saved to: {metadata_file}")

    return metadata


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate game metadata")
    parser.add_argument("--asset", "-a", type=Path, default=Path("asset"),
                       help="Asset folder (default: asset)")
    parser.add_argument("--output", "-o", type=Path, default=Path("asset_output"),
                       help="Output folder (default: asset_output)")
    parser.add_argument("--metadata", "-m", type=Path,
                       default=Path("asset_output/game_metadata.json"),
                       help="Metadata file path (default: asset_output/game_metadata.json)")
    parser.add_argument("--num", "-n", type=int, default=14,
                       help="Number of questions (default: 14)")

    args = parser.parse_args()

    generate_game_metadata(
        asset_folder=args.asset,
        output_folder=args.output,
        metadata_file=args.metadata,
        num_questions=args.num
    )
