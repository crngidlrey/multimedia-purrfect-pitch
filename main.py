#!/usr/bin/env python3
"""
Entry point for the Purrfect Pitch game.

Fokus file ini hanya menjalankan permainan dengan memanggil modul GUI.
Semua logika tampilan/loop berada di gui.py agar main tetap sederhana.
"""

import sys

from gui import PurrfectPitchGame


def main() -> None:
    """Start the Purrfect Pitch game."""
    game = PurrfectPitchGame()
    game.run()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover
        print(f"\n[FATAL ERROR] {exc}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
