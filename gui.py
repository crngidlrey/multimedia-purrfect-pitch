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
import math

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
    START_POPUP = "start_popup"
    COUNTDOWN = "countdown"
    GAME_OVER = "game_over"
    PAUSED_NO_FACE = "paused_no_face"


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

        # Game phase: start screen by default
        self.phase = GamePhase.START_POPUP
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
        self._bgm_path = Path("asset/backsound.m4a")
        self._bgm_started = False
        self._bgm_delay = 1.5  # seconds
        self._bgm_timer = time.time()
        self._win_sound_played = False
        # Confetti animation
        self._confetti_path = Path("asset/confetti.mp4")
        self._confetti_cap: Optional[cv2.VideoCapture] = None
        self._confetti_active = False

        print("[INIT] Prepare questions...")
        self.questions = self._prepare_questions()

        print("[INIT] Inisialisasi Game Logic...")
        # Don't start game logic yet - wait for user to press SPACE
        self.game_logic = GameLogic(self.questions, duration_seconds=45.0)

        print("[INIT] Inisialisasi GUI components...")
        self.meme_overlay = MemeOverlay(
            offset_x=200,
            offset_y=-80,
            sprite_size=(240, 240)
        )

        self.waveform_view = WaveformView(style="bars")

        # State tracking
        self.current_audio_start_time: Optional[float] = None
        self.last_tilt_state = "CENTER"
        # Smoothed head position to reduce jitter (x, y) in window coordinates
        self._smoothed_head: Optional[tuple] = None
        # Scheduled next-question timing (to allow a brief smooth transition)
        self._scheduled_next_question_time: Optional[float] = None
        self._scheduled_end_game: bool = False
        # Countdown state
        self.countdown_start_time: Optional[float] = None
        self.countdown_end_time: Optional[float] = None
        self.countdown_duration: int = 3

        # Popup image assets for menus (start / game over / score board / instruction board)
        self.start_img = self._load_image(Path("asset/start.png"))
        self.gameover_img = self._load_image(Path("asset/gameover.png"))
        self.scoreboard_img = self._load_image(Path("asset/score_board.png"))
        self.instruction_board_img = self._load_image(Path("asset/instruction_board.png"))
        # Popup animation state (start image visible at launch)
        self.popup_start_time: Optional[float] = time.time()
        self.popup_anim_duration: float = 0.6
        # Camera transform cache (scale and crop offsets for fit-to-window)
        self._camera_transform: Optional[tuple[float, int, int]] = None
        # Face detection status & pause helpers
        self.face_present: bool = False
        self._phase_before_pause: Optional[GamePhase] = None
        self._paused_countdown_remaining: Optional[float] = None
        self._scheduled_transition_remaining: Optional[float] = None
        self._scheduled_transition_was_end: bool = False
        self._was_audio_playing: bool = False
        self._pending_audio_start: bool = False
        self._audio_elapsed_before_pause: float = 0.0

        # Verify start image loaded
        if self.start_img is not None:
            print(f"[INIT] Start image loaded successfully: {self.start_img.shape}")
        else:
            print("[WARN] Start image (start.png) failed to load!")

        print("[INIT] Game ready!")

    def _load_metadata(self) -> dict:
        """Load metadata game dari JSON."""
        if not self.metadata_path.exists():
            print(f"[ERROR] Metadata file tidak ditemukan: {self.metadata_path}")
            return {}

        with open(self.metadata_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _load_image(self, path: Path) -> Optional[np.ndarray]:
        """Load an image with alpha if available; return None if missing."""
        try:
            if path.exists():
                img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
                if img is None:
                    print(f"[WARN] Failed to load image: {path}")
                return img
            else:
                # Not fatal: allow fallback to text
                return None
        except Exception as e:
            print(f"[WARN] Error loading image {path}: {e}")
            return None

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
        print("\n[GAME] Preparing new game (showing start screen)...")
        # Show start popup first; actual countdown begins when player presses SPACE
        # (no reset call — GameLogic will be started when countdown begins)
        self.phase = GamePhase.START_POPUP
        self.popup_start_time = time.time()
        self._scheduled_countdown_time = None
        self.countdown_start_time = None
        self.countdown_end_time = None
        self.feedback_message = ""
        self.answer_submitted = False
        self._phase_before_pause = None
        self._paused_countdown_remaining = None
        self._scheduled_transition_remaining = None
        self._scheduled_transition_was_end = False
        self._pending_audio_start = False
        self._was_audio_playing = False
        self._audio_elapsed_before_pause = 0.0
        self._win_sound_played = False
        self._stop_confetti()

    def start_countdown_and_game(self) -> None:
        """Begin the countdown then start the first question (called when player presses SPACE)."""
        print("\n[GAME] Starting countdown and game...")
        # Start game logic now and begin countdown immediately
        self.game_logic.start_game(shuffle=True)
        self.phase = GamePhase.COUNTDOWN
        self.countdown_start_time = time.time()
        self.countdown_end_time = self.countdown_start_time + self.countdown_duration
        self.popup_start_time = None
        self._scheduled_countdown_time = None
        self._pending_audio_start = False
        self._phase_before_pause = None
        self._scheduled_transition_remaining = None
        self._scheduled_transition_was_end = False
        self._paused_countdown_remaining = None
        self._was_audio_playing = False
        self._audio_elapsed_before_pause = 0.0
        self._win_sound_played = False
        self._stop_confetti()
        if not self.face_present:
            self._pause_due_to_face_loss()

    def _load_next_question(self) -> None:
        """Load soal berikutnya."""
        # Reset answer flag untuk soal baru
        self.answer_submitted = False
        self._pending_audio_start = False

        state = self.game_logic.get_state()

        if not state.is_running:
            self.phase = GamePhase.GAME_OVER
            self.popup_start_time = time.time()
            return

        question = state.current_question
        if question is None:
            self.phase = GamePhase.GAME_OVER
            self.popup_start_time = time.time()
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
        self._pending_audio_start = False
        self._audio_elapsed_before_pause = 0.0
        if not self.face_present:
            self._pending_audio_start = True
            self._pause_due_to_face_loss()
            return

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
        if not self.face_present:
            return

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
        self._play_answer_sound(is_correct)

        # Hide memes instantly and schedule the next question to make
        # transitions feel smooth (no fade, but a short delay so the user sees feedback)
        transition_delay = 0.25  # seconds; tune for responsiveness
        self.meme_overlay.hide_memes(animate=False)

        state = self.game_logic.get_state()
        if state.is_running:
            # schedule next question after transition_delay
            self._scheduled_next_question_time = time.time() + transition_delay
            self._scheduled_end_game = False
            # show feedback briefly (phase) while transitioning
            self.phase = GamePhase.SHOW_FEEDBACK
            self.feedback_start_time = time.time()
        else:
            # schedule game over popup after short delay so hide animation can play
            self._scheduled_next_question_time = time.time() + transition_delay
            self._scheduled_end_game = True
            self.phase = GamePhase.SHOW_FEEDBACK
            self.feedback_start_time = time.time()

    def _compute_camera_transform(self, frame_w: int, frame_h: int):
        """
        Hitung skala dan offset crop agar kamera mengisi seluruh window tanpa distorsi.
        Menggunakan pendekatan cover (zoom + crop) sehingga tidak ada ruang kosong saat window dibesarkan.
        """
        scale = max(self.window_width / frame_w, self.window_height / frame_h)
        scaled_w = int(frame_w * scale)
        scaled_h = int(frame_h * scale)
        crop_x = max(0, (scaled_w - self.window_width) // 2)
        crop_y = max(0, (scaled_h - self.window_height) // 2)
        self._camera_transform = (scale, crop_x, crop_y)
        return self._camera_transform

    def _get_camera_transform(self, frame_w: int, frame_h: int):
        if self._camera_transform is None:
            return self._compute_camera_transform(frame_w, frame_h)
        return self._camera_transform

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
        # Position the meme overlay to follow the detected face.
        # Convert face center from capture frame coords -> window coords, and pass scaled face_size.
        # Apply a small EMA smoothing to reduce jitter.
        frame_h, frame_w = frame.shape[:2]
        was_present = self.face_present
        scale, crop_x, crop_y = self._get_camera_transform(frame_w, frame_h)
        if state.face_detected and getattr(state, 'face_center', None) is not None:
            fx, fy = state.face_center

            # Map dari frame capture -> canvas yang sudah dicrop
            head_x = fx * scale - crop_x
            head_y = fy * scale - crop_y

            # Scaled face size when available
            if getattr(state, 'face_size', None) is not None:
                face_w, face_h = state.face_size
                scaled_face_w = int(face_w * scale)
                scaled_face_h = int(face_h * scale)
            else:
                scaled_face_w = None
                scaled_face_h = None

            # EMA smoothing: alpha closer to 1 -> more responsive; 0.6 is a reasonable tradeoff
            alpha = 0.6
            if self._smoothed_head is None:
                sx_pos, sy_pos = int(head_x), int(head_y)
            else:
                prev_x, prev_y = self._smoothed_head
                sx_pos = int(prev_x * (1 - alpha) + head_x * alpha)
                sy_pos = int(prev_y * (1 - alpha) + head_y * alpha)

            self._smoothed_head = (sx_pos, sy_pos)

            # Provide face_size when available so overlay can compute temple anchors
            if scaled_face_w is not None and scaled_face_h is not None:
                self.meme_overlay.set_head_position(sx_pos, sy_pos, face_size=(scaled_face_w, scaled_face_h))
            else:
                self.meme_overlay.set_head_position(sx_pos, sy_pos)
            self.face_present = True
        else:
            # Hide overlay and mark face lost
            self.face_present = False
            self._smoothed_head = None
            self.meme_overlay.hide_memes(animate=False)

        if self.face_present and not was_present:
            self._resume_from_face_pause()
        elif not self.face_present:
            self._pause_due_to_face_loss()

        # Update highlight hanya ketika wajah ada
        if (
            self.phase in (GamePhase.PLAYING_AUDIO, GamePhase.WAITING_ANSWER)
            and self.face_present
        ):
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

    def _pause_due_to_face_loss(self) -> None:
        """Pause game state ketika wajah hilang."""
        if self.phase in (GamePhase.START_POPUP, GamePhase.IDLE, GamePhase.GAME_OVER):
            return
        if self.phase == GamePhase.PAUSED_NO_FACE:
            return

        print("[PAUSE] No face detected - pausing game flow.")
        self._phase_before_pause = self.phase

        # Pause countdown timer
        if self.phase == GamePhase.COUNTDOWN and self.countdown_end_time is not None:
            self._paused_countdown_remaining = max(0.0, self.countdown_end_time - time.time())
            self.countdown_end_time = None

        # Pause pending transitions between questions
        if self._scheduled_next_question_time is not None:
            self._scheduled_transition_remaining = max(0.0, self._scheduled_next_question_time - time.time())
            self._scheduled_next_question_time = None
            self._scheduled_transition_was_end = self._scheduled_end_game

        # Pause audio playback if needed
        if self.phase == GamePhase.PLAYING_AUDIO and self.current_audio_start_time is not None:
            self._audio_elapsed_before_pause = time.time() - self.current_audio_start_time
        else:
            self._audio_elapsed_before_pause = 0.0
        self._was_audio_playing = self.audio_manager.is_playing()
        if self._was_audio_playing:
            self.audio_manager.pause()

        self.game_logic.pause()
        self.phase = GamePhase.PAUSED_NO_FACE

    def _resume_from_face_pause(self) -> None:
        """Resume game setelah wajah kembali terdeteksi."""
        if self.phase != GamePhase.PAUSED_NO_FACE:
            return

        print("[PAUSE] Face detected - resuming game.")
        self.game_logic.resume()

        next_phase = self._phase_before_pause or GamePhase.IDLE
        self._phase_before_pause = None

        # Resume countdown if it was active
        if next_phase == GamePhase.COUNTDOWN:
            remaining = self._paused_countdown_remaining if self._paused_countdown_remaining is not None else self.countdown_duration
            remaining = max(0.0, remaining)
            elapsed = self.countdown_duration - remaining
            self.countdown_start_time = time.time() - elapsed
            self.countdown_end_time = time.time() + remaining
        self._paused_countdown_remaining = None
        if next_phase != GamePhase.COUNTDOWN:
            self.countdown_end_time = None

        # Resume pending transitions
        if self._scheduled_transition_remaining is not None:
            self._scheduled_next_question_time = time.time() + self._scheduled_transition_remaining
            self._scheduled_transition_remaining = None
            self._scheduled_end_game = self._scheduled_transition_was_end
            self._scheduled_transition_was_end = False

        self.phase = next_phase

        if self._pending_audio_start:
            self._pending_audio_start = False
            self._play_audio()
        elif self._was_audio_playing:
            self.audio_manager.resume()
            if self.current_audio_start_time is not None:
                self.current_audio_start_time = time.time() - self._audio_elapsed_before_pause
            self._audio_elapsed_before_pause = 0.0
        else:
            self._audio_elapsed_before_pause = 0.0

        if self.phase in (GamePhase.PLAYING_AUDIO, GamePhase.WAITING_ANSWER, GamePhase.SHOW_FEEDBACK):
            self.meme_overlay.show_memes(animate=False)

        self._was_audio_playing = False

    def _update(self) -> None:
        """Update game state."""
        state = self.game_logic.get_state()

        if self.phase == GamePhase.PAUSED_NO_FACE:
            self.waveform_view.set_playback_progress(0.0)
            return

        # Check scheduled next-question/game-over timing
        if self._scheduled_next_question_time is not None:
            if time.time() >= self._scheduled_next_question_time:
                # Clear scheduled time first to avoid re-entrancy
                sched_end = self._scheduled_end_game
                self._scheduled_next_question_time = None
                self._scheduled_end_game = False
                if sched_end:
                    # Transition to game over popup
                    self.phase = GamePhase.GAME_OVER
                    self.popup_start_time = time.time()
                    self._play_win_sound()
                    self._start_confetti()
                    print("\n[GAME] GAME OVER! (scheduled)")
                    print(f"[SCORE] Final score: {state.score}/{state.total_questions}")
                else:
                    # Load next question
                    self._load_next_question()
                    # return early since _load_next_question resets relevant state
                    return

        # START_POPUP: wait for player input (SPACE) to begin countdown

        # Handle countdown -> when finished, load first question
        if self.phase == GamePhase.COUNTDOWN:
            if self.countdown_start_time is None:
                self.countdown_start_time = time.time()
                self.countdown_end_time = self.countdown_start_time + self.countdown_duration
            if self.countdown_end_time is not None and time.time() >= self.countdown_end_time:
                # Start first question immediately
                self.countdown_end_time = None
                self._load_next_question()
                return

        # Check game over (timer habis atau soal habis)
        # BUT don't check if we're still on start screen!
        if not state.is_running and self.phase not in (GamePhase.GAME_OVER, GamePhase.START_POPUP):
            # Stop semua yang sedang berjalan
            self.audio_manager.stop()
            self.meme_overlay.hide_memes(animate=False)

            self.phase = GamePhase.GAME_OVER
            # Trigger popup animation for game over
            self.popup_start_time = time.time()
            self._play_win_sound()
            self._start_confetti()
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
                # (removed stray START_POPUP handling here)

    def _play_win_sound(self) -> None:
        """Play victory sound exactly once until the player restarts."""
        if self._win_sound_played:
            return
        win_sound = Path("asset/win_sound.wav")
        self.audio_manager.play_effect(win_sound, volume=0.8)
        self._win_sound_played = True

    def _play_answer_sound(self, is_correct: bool) -> None:
        """Mainkan SFX saat jawaban benar/salah."""
        sound_name = "correct.wav" if is_correct else "wrong.wav"
        effect_path = Path("asset") / sound_name
        self.audio_manager.play_effect(effect_path, volume=0.6)

    def _start_confetti(self) -> None:
        """Mulai animasi confetti menggunakan video."""
        if self._confetti_active:
            return
        if not self._confetti_path.exists():
            print(f"[WARN] Confetti video not found: {self._confetti_path}")
            return
        cap = cv2.VideoCapture(str(self._confetti_path))
        if not cap.isOpened():
            print(f"[WARN] Failed to open confetti video: {self._confetti_path}")
            return
        self._confetti_cap = cap
        self._confetti_active = True

    def _stop_confetti(self) -> None:
        """Hentikan animasi confetti dan reset video."""
        self._confetti_active = False
        if self._confetti_cap is not None:
            self._confetti_cap.release()
            self._confetti_cap = None

    def _render(self, camera_frame: np.ndarray) -> np.ndarray:
        """
        Render game UI ke frame dengan layout modern dan center-aligned.

        Args:
            camera_frame (np.ndarray): Frame dari kamera

        Returns:
            np.ndarray: Frame dengan UI overlay
        """
        frame_h, frame_w = camera_frame.shape[:2]
        scale, crop_x, crop_y = self._compute_camera_transform(frame_w, frame_h)

        # Resize camera frame dengan mempertahankan rasio, kemudian crop agar mengisi window
        scaled_w = int(frame_w * scale)
        scaled_h = int(frame_h * scale)
        resized = cv2.resize(camera_frame, (scaled_w, scaled_h))

        # Pastikan crop area valid
        crop_x = min(max(0, crop_x), max(0, scaled_w - self.window_width))
        crop_y = min(max(0, crop_y), max(0, scaled_h - self.window_height))
        canvas = resized[crop_y:crop_y + self.window_height, crop_x:crop_x + self.window_width].copy()

        # UI overlay - center elements on the camera
        state = self.game_logic.get_state()
        face_available = self.face_present

        # Draw meme overlay only when game has started and face detected
        if self.phase not in (GamePhase.IDLE, GamePhase.START_POPUP) and face_available:
            self.meme_overlay.draw(canvas)

        # Layout coordinates (centered)
        center_x = self.window_width // 2

        title_y = 40
        # Move timer/score to top-right
        info_y = 40
        # Raise waveform slightly
        waveform_y = 140
        # Make waveform smaller
        waveform_w = min(420, self.window_width - 200)
        waveform_h = 50
        status_y = waveform_y + waveform_h + 50
        button_y = self.window_height - 90

        title = "PURRFECT PITCH"
        title_size = cv2.getTextSize(title, cv2.FONT_HERSHEY_DUPLEX, 1.2, 3)[0]
        title_x = center_x - title_size[0] // 2
        cv2.putText(canvas, title, (title_x, title_y), cv2.FONT_HERSHEY_DUPLEX, 1.2, (100, 200, 255), 3, cv2.LINE_AA)
        # Title
        title = "PURRFECT PITCH"
        title_size = cv2.getTextSize(title, cv2.FONT_HERSHEY_DUPLEX, 1.2, 3)[0]
        title_x = center_x - title_size[0] // 2
        cv2.putText(canvas, title, (title_x, title_y), cv2.FONT_HERSHEY_DUPLEX, 1.2, (100, 200, 255), 3, cv2.LINE_AA)

        # Draw start / gameover popup images (centered) with simple "popup" animation
        def _draw_popup_image(img: np.ndarray, center_x: int, center_y: int, elapsed: float, duration: float, extra_scale: float = 1.0):
            if img is None:
                return 0
            t = min(1.0, max(0.0, elapsed / max(1e-6, duration)))
            # ease-out cubic
            ease = 1 - (1 - t) ** 3
            base_scale = 0.7
            scale = (base_scale + 0.3 * ease) * extra_scale

            h0, w0 = img.shape[:2]
            new_w = max(1, int(w0 * scale))
            new_h = max(1, int(h0 * scale))
            try:
                img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            except Exception:
                return 0

            x = int(center_x - new_w // 2)
            y = int(center_y - new_h // 2)

            # Clip to canvas
            x0 = max(0, x)
            y0 = max(0, y)
            x1 = min(self.window_width, x + new_w)
            y1 = min(self.window_height, y + new_h)

            roi_w = x1 - x0
            roi_h = y1 - y0
            if roi_w <= 0 or roi_h <= 0:
                return 0

            src_x0 = x0 - x
            src_y0 = y0 - y
            src_x1 = src_x0 + roi_w
            src_y1 = src_y0 + roi_h

            src = img_resized[src_y0:src_y1, src_x0:src_x1]

            # If image has alpha channel, composite
            if src.shape[2] == 4:
                alpha = (src[..., 3:4] / 255.0).astype(np.float32)
                src_bgr = src[..., :3].astype(np.float32)
                dst = canvas[y0:y1, x0:x1].astype(np.float32)
                out = alpha * src_bgr + (1 - alpha) * dst
                canvas[y0:y1, x0:x1] = out.astype(np.uint8)
            else:
                canvas[y0:y1, x0:x1] = src
            return new_h

        # START_POPUP: draw the START image instead of GAME OVER
        if self.phase == GamePhase.START_POPUP:
            if self.start_img is not None and self.popup_start_time is not None:
                elapsed = time.time() - self.popup_start_time
                bounce = 1.0 + 0.05 * math.sin(time.time() * 4.0)
                _draw_popup_image(self.start_img, center_x, self.window_height // 2, elapsed, self.popup_anim_duration, extra_scale=bounce)

        # GAME_OVER: draw game over image closer to title with aligned score
        elif self.phase == GamePhase.GAME_OVER:
            elapsed = 0.0 if self.popup_start_time is None else (time.time() - self.popup_start_time)
            if self.gameover_img is not None:
                # Position image a bit below the main title
                img_center_y = title_y + title_size[1] + 50
                _draw_popup_image(self.gameover_img, center_x, img_center_y, elapsed, self.popup_anim_duration)

            label_text = "Final Score:"
            label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
            value_text = f"{state.score}/{state.total_questions}"
            value_size = cv2.getTextSize(value_text, cv2.FONT_HERSHEY_SIMPLEX, 1.1, 3)[0]
            label_x = center_x - label_size[0] // 2
            label_y = max(80, self.window_height // 2 - 40)
            value_x = center_x - value_size[0] // 2
            value_y = label_y + value_size[1] + 28

            # Draw score board image (scaled down) and align score text with it
            if self.scoreboard_img is not None:
                sb_scale_target = 0.27
                sb_w0 = max(1, self.scoreboard_img.shape[1])
                available_scale = (self.window_width - 100) / sb_w0
                sb_scale = min(sb_scale_target, available_scale)
                try:
                    sb_resized = cv2.resize(
                        self.scoreboard_img,
                        (max(1, int(sb_w0 * sb_scale)), max(1, int(self.scoreboard_img.shape[0] * sb_scale))),
                        interpolation=cv2.INTER_AREA
                    )
                except Exception:
                    sb_resized = None

                if sb_resized is not None:
                    sb_h, sb_w = sb_resized.shape[:2]
                    min_x = 20
                    max_x = self.window_width - sb_w - 20
                    sb_x = min(max_x, max(min_x, self.window_width - sb_w - 30))

                    sb_y = max(title_y + title_size[1] + 20, self.window_height - sb_h - -15)
                    sb_center_y = sb_y + sb_h // 2

                    x0 = max(0, sb_x)
                    y0 = max(0, sb_y)
                    x1 = min(self.window_width, sb_x + sb_w)
                    y1 = min(self.window_height, sb_y + sb_h)

                    roi_w = x1 - x0
                    roi_h = y1 - y0
                    if roi_w > 0 and roi_h > 0:
                        src_x0 = x0 - sb_x
                        src_y0 = y0 - sb_y
                        src_x1 = src_x0 + roi_w
                        src_y1 = src_y0 + roi_h
                        src = sb_resized[src_y0:src_y1, src_x0:src_x1]
                        if src.shape[2] == 4:
                            alpha = (src[..., 3:4] / 255.0).astype(np.float32)
                            src_bgr = src[..., :3].astype(np.float32)
                            dst = canvas[y0:y1, x0:x1].astype(np.float32)
                            out = alpha * src_bgr + (1 - alpha) * dst
                            canvas[y0:y1, x0:x1] = out.astype(np.uint8)
                        else:
                            canvas[y0:y1, x0:x1] = src

                        label_x = sb_x + (sb_w - label_size[0] + 115) // 2
                        text_top = sb_y + max(20, int(sb_h * 0.3))
                        label_y = text_top + label_size[1]
                        value_x = sb_x + (sb_w - value_size[0] + 115) // 2
                        value_y = min(sb_y + sb_h - 12, label_y + value_size[1] + 18)

            cv2.putText(canvas, label_text, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(canvas, value_text, (value_x, value_y), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 3, cv2.LINE_AA)

        # If we're in the playing/waiting/feedback phases, draw timer, score, waveform and memes
        if face_available and self.phase in (GamePhase.PLAYING_AUDIO, GamePhase.WAITING_ANSWER, GamePhase.SHOW_FEEDBACK):
            # Timer and score at top-right corner
            timer_text = f"Time: {int(state.remaining_time)}s"
            score_text = f"Score: {state.score}/{state.total_questions}"
            timer_color = (0, 255, 255) if state.remaining_time > 10 else (0, 100, 255)

            timer_size = cv2.getTextSize(timer_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            score_size = cv2.getTextSize(score_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            spacing = 20
            # Right aligned positions
            right_margin = 20
            score_x = self.window_width - right_margin - score_size[0]
            timer_x = score_x - spacing - timer_size[0]

            # Draw timer then score at top-right
            cv2.putText(canvas, timer_text, (timer_x, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, timer_color, 2, cv2.LINE_AA)
            cv2.putText(canvas, score_text, (score_x, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

            # Waveform background (semi-transparent panel) centered below title
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
        if face_available:
            if self.phase == GamePhase.SHOW_FEEDBACK:
                status_lines = [self.feedback_message]
            elif self.phase == GamePhase.GAME_OVER:
                status_lines = []

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

        # If playing audio, draw the listening text centered above the waveform
        if face_available:
            if self.phase == GamePhase.PLAYING_AUDIO:
                msg = "Listening to the cat sound..."
                color = (200, 200, 200)
            elif self.phase == GamePhase.WAITING_ANSWER:
                msg = "Tilt your head LEFT or RIGHT to choose the cat!"
                color = (255, 220, 150)
            else:
                msg = None

            if msg:
                lt_size = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                lt_x = center_x - lt_size[0] // 2
                lt_y = waveform_y - 12  # slightly above the waveform panel
                cv2.putText(canvas, msg, (lt_x, lt_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)

        # If in countdown, draw big centered number (3..1)
        if face_available and self.phase == GamePhase.COUNTDOWN and self.countdown_start_time is not None:
            elapsed = time.time() - self.countdown_start_time
            remaining = max(0.0, self.countdown_duration - elapsed)
            count = int(math.ceil(remaining))
            # Clamp count between 1..countdown_duration
            count = max(1, min(self.countdown_duration, count))
            cnt_text = str(count)
            # Big size for countdown
            font_scale = 5.0
            thickness = 8
            cnt_size = cv2.getTextSize(cnt_text, cv2.FONT_HERSHEY_DUPLEX, font_scale, thickness)[0]
            cnt_x = center_x - cnt_size[0] // 2
            cnt_y = (self.window_height // 2) + cnt_size[1] // 2
            # Semi-transparent dark overlay to emphasize countdown
            overlay = canvas.copy()
            cv2.rectangle(overlay, (0, 0), (self.window_width, self.window_height), (0, 0, 0), -1)
            alpha = 0.45
            canvas[:] = cv2.addWeighted(canvas, 1 - alpha, overlay, alpha, 0)
            cv2.putText(canvas, cnt_text, (cnt_x, cnt_y), cv2.FONT_HERSHEY_DUPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

        # Buttons when IDLE or GAME_OVER (centered near bottom)
        if self.phase in (GamePhase.GAME_OVER, GamePhase.IDLE):
            # Buttons remain accessible via keyboard shortcuts; no on-screen labels needed.
            button_spacing = 30
            restart_text = "[  Restart  ]"
            exit_text = "[  Exit  ]"
            restart_size = cv2.getTextSize(restart_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
            exit_size = cv2.getTextSize(exit_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
            total_button_width = restart_size[0] + button_spacing + exit_size[0]
            restart_x = center_x - total_button_width // 2
            exit_x = restart_x + restart_size[0] + button_spacing

        # Instruction board (visual cue only; controls remain active without text)
        if self.instruction_board_img is not None:
            board_w0 = max(1, self.instruction_board_img.shape[1])
            max_scale = (self.window_width - 200) / board_w0
            board_scale = min(0.17, max_scale)
            try:
                board_resized = cv2.resize(
                    self.instruction_board_img,
                    (max(1, int(self.instruction_board_img.shape[1] * board_scale)),
                     max(1, int(self.instruction_board_img.shape[0] * board_scale))),
                    interpolation=cv2.INTER_AREA
                )
            except Exception:
                board_resized = None

            if board_resized is not None:
                board_h, board_w = board_resized.shape[:2]
                board_x = -55
                board_y = self.window_height - board_h - 0

                x0 = max(0, board_x)
                y0 = max(0, board_y)
                x1 = min(self.window_width, board_x + board_w)
                y1 = min(self.window_height, board_y + board_h)

                roi_w = x1 - x0
                roi_h = y1 - y0
                if roi_w > 0 and roi_h > 0:
                    src_x0 = x0 - board_x
                    src_y0 = y0 - board_y
                    src_x1 = src_x0 + roi_w
                    src_y1 = src_y0 + roi_h
                    src = board_resized[src_y0:src_y1, src_x0:src_x1]
                    if src.shape[2] == 4:
                        alpha = (src[..., 3:4] / 255.0).astype(np.float32)
                        src_bgr = src[..., :3].astype(np.float32)
                        dst = canvas[y0:y1, x0:x1].astype(np.float32)
                        out = alpha * src_bgr + (1 - alpha) * dst
                        canvas[y0:y1, x0:x1] = out.astype(np.uint8)
                    else:
                        canvas[y0:y1, x0:x1] = src

        if not face_available:
            warn = "NO FACE DETECTED - PLEASE LOOK AT THE CAMERA"
            warn_size = cv2.getTextSize(warn, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            warn_x = (self.window_width - warn_size[0]) // 2
            warn_y = self.window_height // 2
            cv2.putText(
                canvas, warn,
                (warn_x, warn_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (0, 0, 255), 2, cv2.LINE_AA
            )

        # Overlay confetti animation when active (typically during Game Over)
        if self._confetti_active and self._confetti_cap is not None:
            ret, conf_frame = self._confetti_cap.read()
            if not ret:
                self._confetti_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, conf_frame = self._confetti_cap.read()
            if ret:
                conf_frame = cv2.resize(conf_frame, (self.window_width, self.window_height))
                if conf_frame.shape[2] == 4:
                    alpha = (conf_frame[..., 3:] / 255.0).astype(np.float32)
                    conf_rgb = conf_frame[..., :3].astype(np.float32)
                    dst = canvas.astype(np.float32)
                    canvas = (alpha * conf_rgb + (1 - alpha) * dst).astype(np.uint8)
                else:
                    # Treat dark (near-black) pixels as transparent so confetti doesn't darken canvas
                    gray = cv2.cvtColor(conf_frame, cv2.COLOR_BGR2GRAY)
                    mask = (gray > 25).astype(np.float32)[..., None]
                    conf_rgb = conf_frame.astype(np.float32)
                    dst = canvas.astype(np.float32)
                    blended = mask * conf_rgb + (1 - mask) * dst
                    canvas = blended.astype(np.uint8)

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

        # Prepare window (allow dynamic resize)
        window_name = "Purrfect Pitch"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, self.window_width, self.window_height)

        # Start face tracker
        self.face_tracker.start()

        try:
            while True:
                # Read camera frame for all phases (including START_POPUP)
                ret, frame = self.face_tracker._cap.read()
                if not ret:
                    print("[ERROR] Cannot read camera frame")
                    break

                # Flip frame (mirror)
                frame = cv2.flip(frame, 1)

                # Evaluate face tracking for every phase
                face_state = self.face_tracker._evaluate_state(frame)
                self._on_face_tracking_update(face_state, frame)

                # Start background music after slight delay
                if not self._bgm_started and (time.time() - self._bgm_timer) >= self._bgm_delay:
                    self.audio_manager.start_background_music(self._bgm_path, volume=0.06)
                    self._bgm_started = True

                # Update game logic
                self._update()

                # Render game UI (frame may be black canvas for START_POPUP)
                game_frame = self._render(frame)

                # Show window
                cv2.imshow(window_name, game_frame)

                # Update window size to allow responsive layout
                rect = cv2.getWindowImageRect(window_name)
                if rect[2] > 0 and rect[3] > 0:
                    if rect[2] != self.window_width or rect[3] != self.window_height:
                        self.window_width = rect[2]
                        self.window_height = rect[3]
                        self._camera_transform = None  # recalc scale/crop

                # Check if window was closed using the X button
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    print("\n[EXIT] Window closed by user")
                    break

                # Handle keyboard input (use waitKeyEx for special keys)
                raw_key = cv2.waitKeyEx(1)
                key = raw_key & 0xFF if raw_key != -1 else -1

                if key == 27:  # ESC
                    print("\n[EXIT] User quit")
                    break
                elif key == ord(' '):  # SPACE
                    # If on the start screen or after game over, pressing SPACE starts the countdown and game
                    if self.phase in (GamePhase.START_POPUP, GamePhase.IDLE, GamePhase.GAME_OVER):
                        self.start_countdown_and_game()
                elif raw_key in (81, 65361, 2424832, 0x250000, 0xFF51):  # LEFT arrow variants
                    if self.phase in (GamePhase.PLAYING_AUDIO, GamePhase.WAITING_ANSWER):
                        self._submit_answer("LEFT")
                elif raw_key in (83, 65363, 2555904, 0x270000, 0xFF53):  # RIGHT arrow variants
                    if self.phase in (GamePhase.PLAYING_AUDIO, GamePhase.WAITING_ANSWER):
                        self._submit_answer("RIGHT")

        except KeyboardInterrupt:
            print("\n[EXIT] Interrupted by user")
        finally:
            # Cleanup
            print("\n[CLEANUP] Closing game...")
            self.face_tracker.stop()
            self.audio_manager.cleanup()
            self._stop_confetti()
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
