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
from face_tracker import FaceTracker, FaceTrackerState
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
        # Make tilt detection less sensitive: require larger angle to trigger
        self.face_tracker = FaceTracker(
            tilt_threshold=12.0,
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
            duration=2.0,
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

        # Print feedback briefly in log (visual feedback will be shown by existing UI
        # for the very short time before next question loads)
        self.feedback_message = "BENAR!" if is_correct else "SALAH!"
        print(f"[FEEDBACK] {self.feedback_message}")

        # Hide memes and immediately load next question (no delay)
        self.meme_overlay.hide_memes(animate=False)

        # Immediately proceed to next question (stop audio already done above)
        # This removes the waiting/show-feedback pause and makes transitions faster.
        state = self.game_logic.get_state()
        if state.is_running:
            self._load_next_question()
        else:
            self.phase = GamePhase.GAME_OVER

    def _on_face_tracking_update(self, state: FaceTrackerState, frame: np.ndarray) -> None:
        """
        Callback dari face tracker.

        Args:
            state (FaceTrackerState): State face tracking terbaru
            frame (np.ndarray): Frame kamera (sudah flipped)
        """
        # Update head position for meme overlay.
        # The face tracker returns coordinates in the camera frame size; we need to
        # scale them to the fullscreen canvas size used in _render.
        frame_h, frame_w = frame.shape[:2]
        if state.face_detected and getattr(state, 'face_center', None) is not None:
            fx, fy = state.face_center

            # Scale coordinates from capture frame -> window canvas
            sx = self.window_width / frame_w
            sy = self.window_height / frame_h
            head_x = int(fx * sx)
            head_y = int(fy * sy)

            # If face size is available, scale offset_x based on face width so memes sit near ears
            if getattr(state, 'face_size', None) is not None:
                face_w, face_h = state.face_size
                scaled_face_w = int(face_w * sx)
                # Set offset_x to ~0.6 * face width, clamp to reasonable range
                new_offset_x = max(80, min(int(scaled_face_w * 0.6), self.window_width // 3))
                self.meme_overlay.offset_x = new_offset_x

            self.meme_overlay.set_head_position(head_x, head_y)
        elif state.face_detected:
            # Fallback: center of window canvas
            head_x = self.window_width // 2
            head_y = self.window_height // 2
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
        Render game UI ke frame dengan layout modern dan center-aligned.

        Args:
            camera_frame (np.ndarray): Frame dari kamera

        Returns:
            np.ndarray: Frame dengan UI overlay
        """
        # Resize camera frame to fill the window (fullscreen camera)
        camera_resized = cv2.resize(camera_frame, (self.window_width, self.window_height))

        # Use the camera frame as the main canvas so UI overlays appear on the camera
        canvas = camera_resized

        # UI overlay - center elements on the camera
        state = self.game_logic.get_state()

        # Draw meme overlay only when game has started (not on initial menu)
        if self.phase not in (GamePhase.IDLE,):
            self.meme_overlay.draw(canvas)

        # Layout coordinates (centered)
        center_x = self.window_width // 2

        title_y = 40
        info_y = 110
        waveform_y = 160
        waveform_w = min(700, self.window_width - 100)
        waveform_h = 140
        status_y = waveform_y + waveform_h + 50
        button_y = self.window_height - 90

        # Title
        title = "PURRFECT PITCH"
        title_size = cv2.getTextSize(title, cv2.FONT_HERSHEY_DUPLEX, 1.2, 3)[0]
        title_x = center_x - title_size[0] // 2
        cv2.putText(canvas, title, (title_x, title_y), cv2.FONT_HERSHEY_DUPLEX, 1.2, (100, 200, 255), 3, cv2.LINE_AA)

        # If we're in the playing/waiting/feedback phases, draw timer, score, waveform and memes
        if self.phase in (GamePhase.PLAYING_AUDIO, GamePhase.WAITING_ANSWER, GamePhase.SHOW_FEEDBACK):
            # Timer and score side-by-side, centered under the title
            timer_text = f"Time: {int(state.remaining_time)}s"
            score_text = f"Score: {state.score}/{state.total_questions}"
            timer_color = (0, 255, 255) if state.remaining_time > 10 else (0, 100, 255)

            timer_size = cv2.getTextSize(timer_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            score_size = cv2.getTextSize(score_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            spacing = 40
            total_info_width = timer_size[0] + spacing + score_size[0]
            timer_x = center_x - total_info_width // 2

            cv2.putText(canvas, timer_text, (timer_x, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, timer_color, 2, cv2.LINE_AA)
            cv2.putText(canvas, score_text, (timer_x + timer_size[0] + spacing, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

            # Waveform background (semi-transparent panel)
            waveform_x = center_x - waveform_w // 2
            overlay = canvas.copy()
            panel_tl = (waveform_x - 10, waveform_y - 10)
            panel_br = (waveform_x + waveform_w + 10, waveform_y + waveform_h + 10)
            cv2.rectangle(overlay, panel_tl, panel_br, (0, 0, 0), -1)
            alpha = 0.5
            canvas[panel_tl[1]:panel_br[1], panel_tl[0]:panel_br[0]] = cv2.addWeighted(
                canvas[panel_tl[1]:panel_br[1], panel_tl[0]:panel_br[0]], 1 - alpha,
                overlay[panel_tl[1]:panel_br[1], panel_tl[0]:panel_br[0]], alpha, 0
            )

            # Draw waveform on overlay
            self.waveform_view.draw(canvas, waveform_x, waveform_y, waveform_w, waveform_h, show_border=False)

        # Status / feedback text (centered)
        status_lines = []
        if self.phase == GamePhase.IDLE:
            status_lines = ["Press SPACE to start"]
        elif self.phase == GamePhase.PLAYING_AUDIO:
            status_lines = ["Listening to the cat sound..."]
        elif self.phase == GamePhase.WAITING_ANSWER:
            status_lines = ["Tilt your head LEFT or RIGHT", "to choose the cat!"]
        elif self.phase == GamePhase.SHOW_FEEDBACK:
            status_lines = [self.feedback_message]
        elif self.phase == GamePhase.GAME_OVER:
            status_lines = ["GAME OVER!", f"Final Score: {state.score}/{state.total_questions}"]

        for i, line in enumerate(status_lines):
            if "BENAR" in line:
                color = (0, 255, 0); size = 1.2; thickness = 3
            elif "SALAH" in line:
                color = (0, 100, 255); size = 1.2; thickness = 3
            elif "GAME OVER" in line:
                color = (0, 100, 255); size = 1.0; thickness = 2
            else:
                color = (230, 230, 230); size = 0.8; thickness = 1

            text_size = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, size, thickness)[0]
            text_x = center_x - text_size[0] // 2
            cv2.putText(canvas, line, (text_x, status_y + i * 40), cv2.FONT_HERSHEY_SIMPLEX, size, color, thickness, cv2.LINE_AA)

        # Buttons when IDLE or GAME_OVER (centered near bottom)
        if self.phase in (GamePhase.GAME_OVER, GamePhase.IDLE):
            button_spacing = 30
            restart_text = "[  Restart  ]"
            exit_text = "[  Exit  ]"
            restart_size = cv2.getTextSize(restart_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
            exit_size = cv2.getTextSize(exit_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
            total_button_width = restart_size[0] + button_spacing + exit_size[0]
            restart_x = center_x - total_button_width // 2
            exit_x = restart_x + restart_size[0] + button_spacing
            cv2.putText(canvas, restart_text, (restart_x, button_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 200, 255), 1, cv2.LINE_AA)
            cv2.putText(canvas, exit_text, (exit_x, button_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 100), 1, cv2.LINE_AA)

        # Controls info at very bottom
        controls = "SPACE: Start/Restart   |   ESC: Exit   |   Arrow Keys: Manual Select"
        ctrl_size = cv2.getTextSize(controls, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0]
        ctrl_x = center_x - ctrl_size[0] // 2
        ctrl_y = self.window_height - 20
        cv2.putText(canvas, controls, (ctrl_x, ctrl_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)

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
