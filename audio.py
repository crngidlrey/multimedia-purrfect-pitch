#!/usr/bin/env python3
"""
audio.py
Batch pitch-down audio files in ./asset and save results to ./asset_output.

Default: semitones = -5
Outputs are WAV files (safer for writing without ffmpeg).

Usage:
  python audio.py
  python audio.py --semitones -4
  python audio.py --in_folder asset --out_folder asset_output --semitones -6
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
    Load audio (keeps channels), pitch-shift each channel, and write output WAV.
    """
    # librosa.load with mono=False preserves channels; sr=None keeps original sample rate
    y, sr = librosa.load(str(input_path), sr=sr_target, mono=False)

    # If mono, librosa returns 1-D array
    if y.ndim == 1:
        y_shift = librosa.effects.pitch_shift(y, sr=sr, n_steps=n_semitones)
        sf.write(str(output_path), y_shift, sr)
    else:
        # y shape: (channels, samples)
        channels = []
        for ch_idx in range(y.shape[0]):
            ch = y[ch_idx]
            ch_shift = librosa.effects.pitch_shift(ch, sr=sr, n_steps=n_semitones)
            channels.append(ch_shift)
        # pad channels to same length
        max_len = max(map(len, channels))
        channels = [np.pad(c, (0, max_len - len(c)), mode="constant") for c in channels]
        y_shift = np.vstack(channels)  # shape (channels, samples)
        # soundfile expects (samples, channels)
        sf.write(str(output_path), y_shift.T, sr)

    # return sample rate and duration of OUTPUT file for metadata
    out_duration = librosa.get_duration(filename=str(output_path))
    return sr, out_duration

def find_audio_files(folder: Path, exts=(".mp3", ".wav", ".ogg", ".flac", ".m4a")):
    files = []
    for ext in exts:
        files.extend(sorted(folder.glob(f"*{ext}")))
    return files

def main():
    parser = argparse.ArgumentParser(description="Batch pitch-down audio files in a folder.")
    parser.add_argument("--in_folder", "-i", type=Path, default=Path("asset"), help="Input folder (default: asset)")
    parser.add_argument("--out_folder", "-o", type=Path, default=Path("asset_output"), help="Output folder (default: asset_output)")
    parser.add_argument("--semitones", "-s", type=float, default=-5.0, help="Semitones to shift (negative => pitch down). Default -5")
    parser.add_argument("--sr", type=int, default=None, help="Target sample rate (None = keep original)")
    args = parser.parse_args()

    in_folder: Path = args.in_folder
    out_folder: Path = args.out_folder
    semitones: float = args.semitones
    sr_target = args.sr

    if not in_folder.exists() or not in_folder.is_dir():
        print(f"[ERROR] Input folder not found: {in_folder}")
        return

    out_folder.mkdir(parents=True, exist_ok=True)

    files = find_audio_files(in_folder)
    if not files:
        print(f"[INFO] No audio files found in {in_folder}. Supported exts: .mp3 .wav .ogg .flac .m4a")
        return

    metadata = []
    print(f"[INFO] Found {len(files)} audio files. Processing with semitones={semitones} ...")

    for f in tqdm(files, desc="Processing audio files"):
        try:
            name = f.stem
            out_name = f"{name}_pitch{int(semitones)}.wav" if float(semitones).is_integer() else f"{name}_pitch{semitones}.wav"
            out_path = out_folder / out_name

            sr_out, out_duration = pitch_shift_file(f, out_path, n_semitones=semitones, sr_target=sr_target)

            # original duration (for metadata)
            orig_duration = librosa.get_duration(filename=str(f))
            metadata.append({
                "input": str(f.relative_to(Path.cwd())),
                "output": str(out_path.relative_to(Path.cwd())),
                "semitones": semitones,
                "orig_duration_s": round(orig_duration, 3),
                "out_duration_s": round(out_duration, 3),
                "sr_out": sr_out
            })
        except Exception as e:
            print(f"[WARN] Failed processing {f.name}: {e}")

    # write metadata
    meta_path = out_folder / "metadata.json"
    with open(meta_path, "w", encoding="utf-8") as mf:
        json.dump(metadata, mf, indent=2, ensure_ascii=False)

    print(f"[DONE] Processed {len(metadata)} files. Outputs in: {out_folder}")
    print(f"[INFO] Metadata written to {meta_path}")

if __name__ == "__main__":
    main()
