#!/usr/bin/env python3
"""
Game logic module.

Mengatur state permainan: daftar soal, timer utama, dan skor.
Modul ini tidak tahu soal kamera/GUI sehingga mudah diuji secara terpisah.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
import time
from typing import List, Optional, Sequence


@dataclass
class Question:
    id: str
    audio_path: Path
    waveform_data: Sequence[float]
    left_meme: Path
    right_meme: Path
    correct_side: str  # "LEFT" atau "RIGHT"


@dataclass
class GameState:
    is_running: bool
    remaining_time: float
    score: int
    current_index: int
    total_questions: int
    current_question: Optional[Question]


class GameLogic:
    def __init__(self, questions: List[Question], duration_seconds: float = 30.0) -> None:
        if not questions:
            raise ValueError("Minimal butuh satu question untuk memulai game.")
        self._questions = questions
        self.duration_seconds = duration_seconds
        self._start_time: Optional[float] = None
        self._current_idx = 0
        self._score = 0
        self._is_running = False

    def start_game(self, shuffle: bool = True) -> None:
        self._score = 0
        self._is_running = True
        self._start_time = time.monotonic()
        self._current_idx = 0
        if shuffle:
            random.shuffle(self._questions)

    def stop_game(self) -> None:
        self._is_running = False
        self._start_time = None

    def _remaining_time(self) -> float:
        if not self._is_running or self._start_time is None:
            return self.duration_seconds
        elapsed = time.monotonic() - self._start_time
        remaining = max(0.0, self.duration_seconds - elapsed)
        if remaining <= 0.0:
            self.stop_game()
        return remaining

    def current_question(self) -> Optional[Question]:
        if 0 <= self._current_idx < len(self._questions):
            return self._questions[self._current_idx]
        return None

    def submit_answer(self, side: str) -> bool:
        if not self._is_running:
            return False
        question = self.current_question()
        if question is None:
            return False
        is_correct = side.upper() == question.correct_side.upper()
        if is_correct:
            self._score += 1
        self._current_idx += 1
        if self._current_idx >= len(self._questions):
            self.stop_game()
        return is_correct

    def get_state(self) -> GameState:
        return GameState(
            is_running=self._is_running,
            remaining_time=self._remaining_time(),
            score=self._score,
            current_index=self._current_idx,
            total_questions=len(self._questions),
            current_question=self.current_question(),
        )


if __name__ == "__main__":
    dummy_questions = [
        Question(
            id="q1",
            audio_path=Path("asset_output/cat1.wav"),
            waveform_data=[0.1, 0.5, 0.2],
            left_meme=Path("asset/meme1.png"),
            right_meme=Path("asset/meme2.png"),
            correct_side="LEFT",
        ),
        Question(
            id="q2",
            audio_path=Path("asset_output/cat2.wav"),
            waveform_data=[0.3, 0.9, 0.1],
            left_meme=Path("asset/meme3.png"),
            right_meme=Path("asset/meme4.png"),
            correct_side="RIGHT",
        ),
    ]
    logic = GameLogic(dummy_questions, duration_seconds=10)
    logic.start_game(shuffle=False)
    print("Game started:", logic.get_state())
    logic.submit_answer("LEFT")
    print("After answer:", logic.get_state())
