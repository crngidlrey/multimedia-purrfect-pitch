#!/usr/bin/env python3
"""
Main Entry Point - Purrfect Pitch Game

Game tebak suara kucing dengan pitch-shifted audio.
Pemain menebak meme mana yang cocok dengan suara kucing dengan memiringkan kepala.

Alur Game:
1. Load audio dan meme dari folder asset
2. Proses audio dengan pitch shift
3. Mulai game dengan timer 30 detik
4. Tampilkan soal: play audio + tampilkan 2 meme
5. Deteksi kemiringan kepala untuk jawaban
6. Hitung skor dan lanjut soal berikutnya
7. Game over setelah waktu habis atau soal selesai
"""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path
from typing import List, Optional
import cv2
import numpy as np

# Import semua modul game
from face_tracker import FaceTracker, FaceTrackerState
from game_logic import GameLogic, Question, GameState
from audio_manager import AudioManager
from waveform_view import WaveformView
from meme_overlay import MemeOverlay
import audio_processing


class GameApp:
    """
    Aplikasi utama game yang mengoordinasi semua modul.
    """
    
    def __init__(
        self,
        asset_folder: Path = Path("asset"),
        output_folder: Path = Path("asset_output"),
        game_duration: float = 30.0,
        pitch_semitones: float = -5.0
    ):
        """
        Inisialisasi aplikasi game.
        
        Args:
            asset_folder (Path): Folder berisi audio dan meme original
            output_folder (Path): Folder untuk menyimpan audio yang sudah diproses
            game_duration (float): Durasi game dalam detik
            pitch_semitones (float): Pitch shift dalam semitone
        """
        self.asset_folder = asset_folder
        self.output_folder = output_folder
        self.game_duration = game_duration
        self.pitch_semitones = pitch_semitones
        
        # State aplikasi
        self.is_running = False
        self.game_started = False
        
        # Inisialisasi modul-modul
        print("[INIT] Menginisialisasi modul...")
        self._init_modules()
        
        # Data game
        self.questions: List[Question] = []
        self.current_answer_locked = False
        self.audio_start_time: Optional[float] = None

    def _init_modules(self) -> None:
        """
        Inisialisasi semua modul game.
        """
        # 1. Face Tracker untuk deteksi orientasi kepala
        print("  - Face Tracker")
        self.face_tracker = FaceTracker(
            tilt_threshold=12.0,
            hold_time=0.6,
            cooldown_time=1.0
        )
        
        # 2. Audio Manager untuk playback dan transformasi audio
        print("  - Audio Manager")
        self.audio_manager = AudioManager(
            sample_rate=44100,
            buffer_size=512,
            output_folder=self.output_folder
        )
        
        # 3. Waveform View untuk visualisasi audio
        print("  - Waveform View")
        self.waveform_view = WaveformView(
            width=600,
            height=100,
            color=(0, 200, 255),  # Cyan
            bg_color=(20, 20, 30)
        )
        
        # 4. Meme Overlay untuk tampilan sprite meme
        print("  - Meme Overlay")
        self.meme_overlay = MemeOverlay(
            offset_x=200,
            offset_y=-80,
            sprite_size=(150, 150)
        )
        
        # 5. Game Logic (akan diinit setelah load questions)
        self.game_logic: Optional[GameLogic] = None
        
        print("[OK] Semua modul berhasil diinisialisasi")

    def load_assets(self) -> bool:
        """
        Load dan proses semua asset (audio + meme).
        
        Returns:
            bool: True jika berhasil load minimal 1 soal
        """
        print(f"\n[LOAD] Memuat asset dari {self.asset_folder}...")
        
        if not self.asset_folder.exists():
            print(f"[ERROR] Folder asset tidak ditemukan: {self.asset_folder}")
            return False
        
        # 1. Cari semua file audio
        audio_files = audio_processing.find_audio_files(self.asset_folder)
        if not audio_files:
            print(f"[ERROR] Tidak ada file audio di {self.asset_folder}")
            return False
        
        print(f"[INFO] Ditemukan {len(audio_files)} file audio")
        
        # 2. Cari semua file meme (gambar)
        meme_exts = (".png", ".jpg", ".jpeg")
        meme_files = []
        for ext in meme_exts:
            meme_files.extend(sorted(self.asset_folder.glob(f"*{ext}")))
        
        if len(meme_files) < 2:
            print(f"[ERROR] Minimal butuh 2 file meme, ditemukan {len(meme_files)}")
            return False
        
        print(f"[INFO] Ditemukan {len(meme_files)} file meme")
        
        # 3. Proses audio dengan pitch shifting
        print(f"[PROCESS] Memproses audio (pitch {self.pitch_semitones} semitones)...")
        audio_clips = self.audio_manager.prepare_audio_clips(
            audio_files=audio_files,
            semitones=self.pitch_semitones,
            waveform_samples=512
        )
        
        if not audio_clips:
            print("[ERROR] Gagal memproses audio")
            return False
        
        # 4. Buat questions dari kombinasi audio + meme
        print("[BUILD] Membuat soal...")
        self.questions = []
        
        for clip in audio_clips:
            # Pilih 2 meme random untuk soal ini
            if len(meme_files) < 2:
                continue
            
            meme_pair = random.sample(meme_files, 2)
            correct_side = random.choice(["LEFT", "RIGHT"])
            
            question = Question(
                id=clip.id,
                audio_path=clip.processed_path,
                waveform_data=clip.waveform_data,
                left_meme=meme_pair[0],
                right_meme=meme_pair[1],
                correct_side=correct_side
            )
            self.questions.append(question)
        
        if not self.questions:
            print("[ERROR] Gagal membuat soal")
            return False
        
        print(f"[OK] Berhasil membuat {len(self.questions)} soal")
        
        # 5. Set queue audio di AudioManager
        self.audio_manager.set_queue(audio_clips)
        
        # 6. Inisialisasi GameLogic dengan questions
        self.game_logic = GameLogic(
            questions=self.questions,
            duration_seconds=self.game_duration
        )
        
        return True

    def start_game(self) -> None:
        """
        Mulai game baru.
        """
        if self.game_logic is None:
            print("[ERROR] Game logic belum diinisialisasi")
            return
        
        print("\n[START] Memulai game...")
        self.game_logic.start_game(shuffle=True)
        self.game_started = True
        self.current_answer_locked = False
        
        # Load soal pertama
        self._load_current_question()

    def _load_current_question(self) -> None:
        """
        Load soal saat ini (audio + meme).
        """
        if self.game_logic is None:
            return
        
        question = self.game_logic.current_question()
        if question is None:
            return
        
        print(f"\n[QUESTION] Loading: {question.id}")
        
        # 1. Load audio clip
        if self.audio_manager.load_clip_by_id(question.id):
            # Set waveform data ke view
            self.waveform_view.set_waveform_data(question.waveform_data)
            self.waveform_view.set_playback_state(is_playing=False)
        
        # 2. Load meme sprites
        self.meme_overlay.load_memes(question.left_meme, question.right_meme)
        self.meme_overlay.show_memes(animate=True)
        
        # 3. Play audio
        self.audio_start_time = time.time()
        self.audio_manager.play(on_finish=self._on_audio_finished)
        self.waveform_view.set_playback_state(is_playing=True)
        
        # 4. Reset answer lock
        self.current_answer_locked = False
        
        print(f"[INFO] Jawaban benar: {question.correct_side}")

    def _on_audio_finished(self) -> None:
        """
        Callback ketika audio selesai diputar.
        """
        print("[AUDIO] Playback selesai")
        self.waveform_view.set_playback_state(is_playing=False, position=1.0)

    def _on_head_tilt_confirmed(self, side: str) -> None:
        """
        Callback ketika kepala miring dikonfirmasi (user menjawab).
        
        Args:
            side (str): "LEFT" atau "RIGHT"
        """
        if not self.game_started or self.current_answer_locked:
            return
        
        if self.game_logic is None:
            return
        
        # Lock answer untuk prevent double answer
        self.current_answer_locked = True
        
        print(f"\n[ANSWER] Pemain pilih: {side}")
        
        # Submit answer ke game logic
        is_correct = self.game_logic.submit_answer(side)
        
        if is_correct:
            print("[RESULT] ✓ BENAR!")
        else:
            print("[RESULT] ✗ SALAH!")
        
        # Hide meme dengan animasi
        self.meme_overlay.hide_memes(animate=True)
        
        # Stop audio jika masih playing
        self.audio_manager.stop()
        
        # Tunggu sebentar untuk animasi, lalu load soal berikutnya
        time.sleep(0.5)
        
        # Cek apakah masih ada soal
        state = self.game_logic.get_state()
        if state.is_running:
            self._load_current_question()
        else:
            self._end_game()

    def _end_game(self) -> None:
        """
        Akhiri game dan tampilkan hasil.
        """
        if self.game_logic is None:
            return
        
        state = self.game_logic.get_state()
        self.game_started = False
        
        print("\n" + "="*50)
        print("GAME OVER!")
        print("="*50)
        print(f"Skor: {state.score} / {state.total_questions}")
        print(f"Akurasi: {state.score / state.total_questions * 100:.1f}%")
        print("="*50)

    def _process_face_tracker_state(self, state: FaceTrackerState) -> None:
        """
        Proses state dari face tracker dan update GUI.
        
        Args:
            state (FaceTrackerState): State terbaru dari face tracker
        """
        if not state.face_detected:
            self.meme_overlay.set_highlight(None)
            return
        
        # Update posisi meme overlay berdasarkan posisi wajah
        # (akan di-set di callback dari face_tracker)
        
        # Update highlight berdasarkan tilt state
        if state.tilt_state in ("LEFT", "RIGHT"):
            self.meme_overlay.set_highlight(state.tilt_state)
        else:
            self.meme_overlay.set_highlight(None)
        
        # Jika tilt confirmed, submit answer
        if state.tilt_confirmed and self.game_started:
            self._on_head_tilt_confirmed(state.tilt_state)

    def _draw_ui(self, frame: np.ndarray, state: GameState) -> None:
        """
        Gambar UI game ke frame.
        
        Args:
            frame (np.ndarray): Frame kamera
            state (GameState): State game saat ini
        """
        h, w = frame.shape[:2]
        
        # 1. Draw timer dan skor di top bar
        bar_height = 80
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, bar_height), (40, 40, 40), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Timer
        timer_text = f"Time: {int(state.remaining_time)}s"
        cv2.putText(
            frame, timer_text,
            (20, 50), cv2.FONT_HERSHEY_DUPLEX,
            1.2, (0, 255, 255), 3
        )
        
        # Skor
        score_text = f"Score: {state.score}/{state.total_questions}"
        cv2.putText(
            frame, score_text,
            (w - 250, 50), cv2.FONT_HERSHEY_DUPLEX,
            1.2, (0, 255, 0), 3
        )
        
        # Progress
        if state.total_questions > 0:
            progress_text = f"Q: {state.current_index + 1}/{state.total_questions}"
            cv2.putText(
                frame, progress_text,
                (w // 2 - 50, 50), cv2.FONT_HERSHEY_DUPLEX,
                1.0, (255, 255, 255), 2
            )
        
        # 2. Draw waveform di bottom
        waveform_y = h - 120
        self.waveform_view.draw_on_frame(frame, x=20, y=waveform_y)
        
        # 3. Update waveform playback position
        if self.audio_start_time and self.audio_manager.is_playing():
            metadata = self.audio_manager.get_current_metadata()
            if metadata:
                elapsed = time.time() - self.audio_start_time
                self.waveform_view.update_playback_position(
                    elapsed_time=elapsed,
                    total_duration=metadata['duration']
                )
        
        # 4. Instruksi di bottom
        if not self.game_started:
            instruction = "Tekan SPASI untuk mulai | ESC untuk keluar"
            text_size = cv2.getTextSize(instruction, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            text_x = (w - text_size[0]) // 2
            cv2.putText(
                frame, instruction,
                (text_x, h - 20), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (255, 255, 255), 2
            )

    def _camera_callback(self, state: FaceTrackerState, frame: np.ndarray) -> None:
        """
        Callback dari face tracker untuk setiap frame kamera.
        
        Args:
            state (FaceTrackerState): State deteksi wajah
            frame (np.ndarray): Frame kamera (sudah di-flip)
        """
        # Proses state face tracker
        self._process_face_tracker_state(state)
        
        # Update posisi head untuk meme overlay
        if state.face_detected:
            h, w = frame.shape[:2]
            # Estimasi posisi kepala di tengah frame
            head_x, head_y = w // 2, h // 3
            self.meme_overlay.set_head_position(head_x, head_y)
        
        # Update animasi meme overlay
        self.meme_overlay.update()
        
        # Draw meme overlay ke frame
        self.meme_overlay.draw(frame)
        
        # Update audio manager (check if finished)
        self.audio_manager.check_finish()
        
        # Draw UI game
        if self.game_logic:
            game_state = self.game_logic.get_state()
            self._draw_ui(frame, game_state)
            
            # Cek jika waktu habis
            if game_state.is_running and game_state.remaining_time <= 0:
                self._end_game()
        
        # Show frame
        cv2.imshow("Purrfect Pitch Game", frame)

    def run(self) -> None:
        """
        Jalankan main loop game.
        """
        print("\n" + "="*60)
        print("PURRFECT PITCH - Cat Sound Quiz Game")
        print("="*60)
        print("Instruksi:")
        print("- Dengarkan suara kucing")
        print("- Pilih meme yang cocok dengan memiringkan kepala")
        print("- Miringkan kepala ke KIRI untuk pilih meme kiri")
        print("- Miringkan kepala ke KANAN untuk pilih meme kanan")
        print("- Tekan SPASI untuk mulai game")
        print("- Tekan ESC untuk keluar")
        print("="*60 + "\n")
        
        self.is_running = True
        
        try:
            # Setup face tracker dengan callback
            self.face_tracker.start()
            
            # Main loop
            while self.is_running:
                ret, frame = self.face_tracker._cap.read()
                if not ret:
                    break
                
                frame = cv2.flip(frame, 1)
                state = self.face_tracker._evaluate_state(frame)
                
                # Panggil callback
                self._camera_callback(state, frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                
                if key == 27:  # ESC
                    print("\n[EXIT] Keluar dari game...")
                    self.is_running = False
                
                elif key == ord(' ') and not self.game_started:  # SPASI
                    if self.game_logic:
                        self.start_game()
                
                elif key == ord('r') and not self.game_started:  # R untuk restart
                    if self.game_logic:
                        print("\n[RESTART] Restart game...")
                        self.start_game()
        
        finally:
            self.cleanup()

    def cleanup(self) -> None:
        """
        Bersihkan semua resource sebelum keluar.
        """
        print("\n[CLEANUP] Membersihkan resource...")
        self.face_tracker.stop()
        self.audio_manager.cleanup()
        cv2.destroyAllWindows()
        print("[OK] Cleanup selesai")


def main():
    """
    Entry point utama aplikasi.
    """
    parser = argparse.ArgumentParser(
        description="Purrfect Pitch - Cat Sound Quiz Game"
    )
    parser.add_argument(
        "--asset", "-a",
        type=Path,
        default=Path("asset"),
        help="Folder asset (default: asset)"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("asset_output"),
        help="Folder output (default: asset_output)"
    )
    parser.add_argument(
        "--duration", "-d",
        type=float,
        default=30.0,
        help="Durasi game dalam detik (default: 30.0)"
    )
    parser.add_argument(
        "--pitch", "-p",
        type=float,
        default=-5.0,
        help="Pitch shift dalam semitone (default: -5.0)"
    )
    
    args = parser.parse_args()
    
    # Buat aplikasi
    app = GameApp(
        asset_folder=args.asset,
        output_folder=args.output,
        game_duration=args.duration,
        pitch_semitones=args.pitch
    )
    
    # Load assets
    if not app.load_assets():
        print("[ERROR] Gagal load assets. Pastikan folder asset berisi audio dan meme!")
        return
    
    # Run game
    app.run()


if __name__ == "__main__":
    main()