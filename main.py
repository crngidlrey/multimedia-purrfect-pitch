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
