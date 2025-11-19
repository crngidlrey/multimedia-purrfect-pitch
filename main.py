#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Purrfect Pitch - Main Game
===========================
Interactive cat sound matching game dengan face tracking.

Tugas Besar Mata Kuliah Multimedia
Institut Teknologi Sumatera
"""

from __future__ import annotations

import sys
import json
import time
from pathlib import Path
from typing import Optional, List
from enum import Enum

import cv2
import numpy as np

# Import modul game
from angle_face_tracker import AngleFaceTracker, FaceTrackerState
from game_logic import GameLogic, Question, GameState
from audio_manager import AudioManager, AudioClip
from meme_overlay import MemeOverlay
from waveform_view import WaveformView
import audio_processing


class GamePhase(Enum):
    """Phase/state game."""
    IDLE = "idle"
    PLAYING_AUDIO = "playing_audio"
    WAITING_ANSWER = "waiting_answer"
    SHOW_FEEDBACK = "show_feedback"
    GAME_OVER = "game_over"


class PurrfectPitchGame:
    """
    Main game orchestrator.
    Menghubungkan semua modul dan mengatur game flow.
    """

    def __init__(
        self,
        metadata_path: Path = Path("asset_output/game_metadata.json"),
        window_width: int = 1280,
        window_height: int = 720
    ):
        """
        Inisialisasi game.

        Args:
            metadata_path (Path): Path ke file metadata game
            window_width (int): Lebar window game
            window_height (int): Tinggi window game
        """
        self.window_width = window_width
        self.window_height = window_height
        self.metadata_path = metadata_path

        # Game phase
        self.phase = GamePhase.IDLE
        self.feedback_message = ""
        self.feedback_start_time = 0.0
        self.feedback_duration = 2.0

        # Answer tracking
        self.answer_submitted = False

        # Load metadata
        self.metadata = self._load_metadata()
        if not self.metadata:
            raise RuntimeError(f"Gagal load metadata dari {metadata_path}")

        # Inisialisasi modul
        print("[INIT] Inisialisasi Face Tracker...")
        self.face_tracker = AngleFaceTracker(
            tilt_threshold=5.0,
            hold_time=0.0,
            cooldown_time=0.5
        )

        print("[INIT] Inisialisasi Audio Manager...")
        self.audio_manager = AudioManager(output_folder=Path("asset_output"))

        print("[INIT] Prepare questions...")
        self.questions = self._prepare_questions()

        print("[INIT] Inisialisasi Game Logic...")
        self.game_logic = GameLogic(self.questions, duration_seconds=30.0)

        print("[INIT] Inisialisasi GUI components...")
        self.meme_overlay = MemeOverlay(
            offset_x=200,
            offset_y=-80,
            sprite_size=(180, 180)
        )

        self.waveform_view = WaveformView(style="bars")

        # State tracking
        self.current_audio_start_time: Optional[float] = None
        self.last_tilt_state = "CENTER"

        print("[INIT] Game ready!")

    def _load_metadata(self) -> dict:
        """Load metadata game dari JSON."""
        if not self.metadata_path.exists():
            print(f"[ERROR] Metadata file tidak ditemukan: {self.metadata_path}")
            return {}

        with open(self.metadata_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _prepare_questions(self) -> List[Question]:
        """
        Prepare questions dari metadata.
        Load audio clips dan generate waveform.
        """
        questions = []

        for q_data in self.metadata.get("questions", []):
            audio_path = Path(q_data["audio_path"])

            # Generate waveform
            waveform_data = audio_processing.generate_waveform_data(
                audio_path=audio_path,
                num_samples=512
            )

            # Buat Question object
            question = Question(
                id=q_data["id"],
                audio_path=audio_path,
                waveform_data=waveform_data,
                left_meme=Path(q_data["left_meme"]),
                right_meme=Path(q_data["right_meme"]),
                correct_side=q_data["correct_side"]
            )

            questions.append(question)

        print(f"[INFO] Loaded {len(questions)} questions")
        return questions

    def start_game(self) -> None:
        """Mulai game baru."""
        print("\n[GAME] Starting new game...")
        self.game_logic.start_game(shuffle=True)
        self.phase = GamePhase.IDLE
        self.feedback_message = ""
        self.answer_submitted = False
        self._load_next_question()

    def _load_next_question(self) -> None:
        """Load soal berikutnya."""
        # Reset answer flag untuk soal baru
        self.answer_submitted = False

        state = self.game_logic.get_state()

        if not state.is_running:
            self.phase = GamePhase.GAME_OVER
            return

        question = state.current_question
        if question is None:
            self.phase = GamePhase.GAME_OVER
            return

        print(f"\n[QUESTION] Loading {question.id}...")

        # Load audio ke audio manager
        clip = AudioClip(
            id=question.id,
            original_path=question.audio_path,
            processed_path=question.audio_path,
            waveform_data=question.waveform_data,
            duration=3.0,
            sample_rate=44100
        )

        # Prepare audio manager
        self.audio_manager.set_queue([clip])
        if self.audio_manager.load_clip(clip):
            self.waveform_view.set_waveform_data(question.waveform_data)
        else:
            print(f"[WARN] Gagal load audio {question.id}")

        # Load meme images
        if not self.meme_overlay.load_memes(question.left_meme, question.right_meme):
            print(f"[WARN] Gagal load meme images untuk {question.id}")

        # Play audio
        self._play_audio()

    def _play_audio(self) -> None:
        """Play audio soal."""
        self.phase = GamePhase.PLAYING_AUDIO
        self.current_audio_start_time = time.time()

        # Show memes dengan animasi
        self.meme_overlay.show_memes(animate=True)

        # Play audio dengan callback
        self.audio_manager.play(on_finish=self._on_audio_finished)

        print("[AUDIO] Playing...")

    def _on_audio_finished(self) -> None:
        """Callback ketika audio selesai."""
        print("[AUDIO] Finished, waiting for answer...")
        self.phase = GamePhase.WAITING_ANSWER

    def _submit_answer(self, side: str) -> None:
        """
        Submit jawaban pemain.

        Args:
            side (str): "LEFT" atau "RIGHT"
        """
        # Bisa jawab saat audio playing atau waiting answer
        if self.phase not in (GamePhase.PLAYING_AUDIO, GamePhase.WAITING_ANSWER):
            return

        # Cek apakah sudah pernah jawab untuk soal ini
        if self.answer_submitted:
            print(f"[ANSWER] Already answered, ignoring...")
            return

        # Mark as submitted
        self.answer_submitted = True

        print(f"[ANSWER] Submitted: {side}")

        # Stop audio jika masih playing
        if self.audio_manager.is_playing():
            self.audio_manager.stop()

        # Submit ke game logic
        is_correct = self.game_logic.submit_answer(side)

        # Show feedback
        self.feedback_message = "BENAR!" if is_correct else "SALAH!"
        self.feedback_start_time = time.time()
        self.phase = GamePhase.SHOW_FEEDBACK

        # Hide memes
        self.meme_overlay.hide_memes(animate=True)

        print(f"[FEEDBACK] {self.feedback_message}")

    def _on_face_tracking_update(self, state: FaceTrackerState, frame: np.ndarray) -> None:
        """
        Callback dari face tracker.

        Args:
            state (FaceTrackerState): State face tracking terbaru
            frame (np.ndarray): Frame kamera (sudah flipped)
        """
        # Update head position untuk meme overlay
        if state.face_detected:
            head_x = frame.shape[1] // 2
            head_y = frame.shape[0] // 2
            self.meme_overlay.set_head_position(head_x, head_y)

        # Update highlight berdasarkan tilt state
        if self.phase in (GamePhase.PLAYING_AUDIO, GamePhase.WAITING_ANSWER):
            if state.tilt_state in ("LEFT", "RIGHT"):
                self.meme_overlay.set_highlight(state.tilt_state)
            else:
                self.meme_overlay.set_highlight(None)

            # Confirm answer jika tilt confirmed
            if state.tilt_confirmed and state.tilt_state in ("LEFT", "RIGHT"):
                self._submit_answer(state.tilt_state)
        else:
            # Reset highlight jika tidak dalam phase playing/waiting
            self.meme_overlay.set_highlight(None)

    def _update(self) -> None:
        """Update game state."""
        state = self.game_logic.get_state()

        # Check game over (timer habis atau soal habis)
        if not state.is_running and self.phase != GamePhase.GAME_OVER:
            # Stop semua yang sedang berjalan
            self.audio_manager.stop()
            self.meme_overlay.hide_memes(animate=False)

            self.phase = GamePhase.GAME_OVER
            print("\n[GAME] GAME OVER!")
            print(f"[SCORE] Final score: {state.score}/{state.total_questions}")

        # Update audio manager (check finish callback)
        self.audio_manager.check_finish()

        # Update meme overlay animations
        self.meme_overlay.update()

        # Update waveform progress jika audio playing
        if self.phase == GamePhase.PLAYING_AUDIO and self.audio_manager.is_playing():
            if self.current_audio_start_time:
                elapsed = time.time() - self.current_audio_start_time
                progress = min(1.0, elapsed / 3.0)
                self.waveform_view.set_playback_progress(progress)
        else:
            self.waveform_view.set_playback_progress(0.0)

        # Check feedback timeout
        if self.phase == GamePhase.SHOW_FEEDBACK:
            if time.time() - self.feedback_start_time >= self.feedback_duration:
                self.feedback_message = ""
                # Cek apakah game masih running sebelum load next question
                if state.is_running:
                    self._load_next_question()
                else:
                    # Game sudah selesai, langsung ke game over
                    self.phase = GamePhase.GAME_OVER

    def _render(self, camera_frame: np.ndarray) -> np.ndarray:
        """
        Render game UI ke frame.

        Args:
            camera_frame (np.ndarray): Frame dari kamera

        Returns:
            np.ndarray: Frame dengan UI overlay
        """
        # Resize camera frame untuk fit di kiri
        cam_h, cam_w = camera_frame.shape[:2]
        target_cam_w = self.window_width // 2
        target_cam_h = int(cam_h * (target_cam_w / cam_w))

        camera_resized = cv2.resize(camera_frame, (target_cam_w, target_cam_h))

        # Buat main canvas
        canvas = np.zeros((self.window_height, self.window_width, 3), dtype=np.uint8)
        canvas[:] = (30, 30, 30)

        # Place camera feed di kiri
        cam_y_offset = (self.window_height - target_cam_h) // 2
        canvas[cam_y_offset:cam_y_offset + target_cam_h, 0:target_cam_w] = camera_resized

        # Draw meme overlay on camera
        self.meme_overlay.draw(canvas[cam_y_offset:cam_y_offset + target_cam_h, 0:target_cam_w])

        # Right panel: game info
        right_x = target_cam_w + 20
        right_y = 50

        state = self.game_logic.get_state()

        # Title
        cv2.putText(
            canvas, "PURRFECT PITCH",
            (right_x, right_y),
            cv2.FONT_HERSHEY_DUPLEX, 1.2, (100, 200, 255), 3, cv2.LINE_AA
        )

        right_y += 60

        # Timer
        timer_text = f"Time: {int(state.remaining_time)}s"
        timer_color = (0, 255, 255) if state.remaining_time > 10 else (0, 100, 255)
        cv2.putText(
            canvas, timer_text,
            (right_x, right_y),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, timer_color, 2, cv2.LINE_AA
        )

        right_y += 50

        # Score
        score_text = f"Score: {state.score} / {state.total_questions}"
        cv2.putText(
            canvas, score_text,
            (right_x, right_y),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA
        )

        right_y += 80

        # Waveform
        waveform_w = self.window_width - target_cam_w - 60
        waveform_h = 120
        self.waveform_view.draw(canvas, right_x, right_y, waveform_w, waveform_h)

        right_y += waveform_h + 40

        # Status text
        status_lines = []
        if self.phase == GamePhase.IDLE:
            status_lines = ["Press SPACE to start"]
        elif self.phase == GamePhase.PLAYING_AUDIO:
            status_lines = ["Listening to the cat sound..."]
        elif self.phase == GamePhase.WAITING_ANSWER:
            status_lines = [
                "Tilt your head LEFT or RIGHT",
                "to choose the cat!"
            ]
        elif self.phase == GamePhase.SHOW_FEEDBACK:
            status_lines = [self.feedback_message]
        elif self.phase == GamePhase.GAME_OVER:
            status_lines = [
                "GAME OVER!",
                f"Final Score: {state.score}/{state.total_questions}",
                "",
                "Press SPACE to restart",
                "Press ESC to exit"
            ]

        for i, line in enumerate(status_lines):
            color = (0, 255, 0) if "BENAR" in line else (0, 100, 255) if "SALAH" in line else (200, 200, 200)
            size = 0.9 if "BENAR" in line or "SALAH" in line else 0.7
            thickness = 2 if "BENAR" in line or "SALAH" in line else 1

            cv2.putText(
                canvas, line,
                (right_x, right_y + i * 35),
                cv2.FONT_HERSHEY_SIMPLEX, size, color, thickness, cv2.LINE_AA
            )

        # Controls info (bottom)
        controls = [
            "SPACE: Start/Restart",
            "ESC: Exit",
            "Arrow Keys: Manual Select (fallback)"
        ]

        ctrl_y = self.window_height - 80
        for i, ctrl in enumerate(controls):
            cv2.putText(
                canvas, ctrl,
                (20, ctrl_y + i * 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1, cv2.LINE_AA
            )

        return canvas

    def run(self) -> None:
        """Main game loop."""
        print("\n" + "="*60)
        print("  PURRFECT PITCH - Interactive Cat Sound Matching Game")
        print("  Tugas Besar Mata Kuliah Multimedia - ITERA")
        print("="*60)
        print("\nControls:")
        print("  SPACE      - Start/Restart game")
        print("  HEAD TILT  - Select cat (LEFT/RIGHT)")
        print("  ARROW KEYS - Manual select (fallback)")
        print("  ESC        - Exit game")
        print("\n" + "="*60 + "\n")

        # Start face tracker
        self.face_tracker.start()

        try:
            while True:
                # Read camera frame
                ret, frame = self.face_tracker._cap.read()
                if not ret:
                    print("[ERROR] Cannot read camera frame")
                    break

                # Flip frame (mirror)
                frame = cv2.flip(frame, 1)

                # Evaluate face tracking
                face_state = self.face_tracker._evaluate_state(frame)

                # Face tracking callback
                self._on_face_tracking_update(face_state, frame)

                # Update game logic
                self._update()

                # Render game UI
                game_frame = self._render(frame)

                # Show window
                cv2.imshow("Purrfect Pitch", game_frame)

                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF

                if key == 27:  # ESC
                    print("\n[EXIT] User quit")
                    break
                elif key == ord(' '):  # SPACE
                    if self.phase in (GamePhase.IDLE, GamePhase.GAME_OVER):
                        self.start_game()
                elif key == 81 or key == 2:  # LEFT ARROW
                    if self.phase == GamePhase.WAITING_ANSWER:
                        self._submit_answer("LEFT")
                elif key == 83 or key == 3:  # RIGHT ARROW
                    if self.phase == GamePhase.WAITING_ANSWER:
                        self._submit_answer("RIGHT")

        except KeyboardInterrupt:
            print("\n[EXIT] Interrupted by user")
        finally:
            # Cleanup
            print("\n[CLEANUP] Closing game...")
            self.face_tracker.stop()
            self.audio_manager.cleanup()
            cv2.destroyAllWindows()
            print("[EXIT] Goodbye!")


if __name__ == "__main__":
    try:
        game = PurrfectPitchGame()
        game.run()
    except Exception as e:
        print(f"\n[FATAL ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
