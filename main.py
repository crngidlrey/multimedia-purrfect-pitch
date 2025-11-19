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
