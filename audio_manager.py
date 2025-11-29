#!/usr/bin/env python3
"""
Audio Manager Module.

Orchestrator untuk audio playback dalam game. Modul ini bertanggung jawab untuk:
- Memanggil audio_processing untuk transformasi audio (pitch shift)
- Mengatur queue clip soal yang akan dimainkan
- Start/stop audio sinkron dengan game state
- Menyediakan metadata waveform ke GUI untuk visualisasi
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional, List, Dict
import pygame
from dataclasses import dataclass

# Import fungsi dari audio_processing untuk transformasi
import audio_processing


@dataclass
class AudioClip:
    """Data struktur untuk satu clip audio soal."""
    id: str
    original_path: Path
    processed_path: Path
    waveform_data: List[float]
    duration: float
    sample_rate: int


class AudioManager:
    """
    Orchestrator untuk audio playback game.
    Mengelola transformasi audio, queue playback, dan metadata waveform.
    """

    def __init__(
        self, 
        sample_rate: int = 44100, 
        buffer_size: int = 512,
        output_folder: Path = Path("asset_output")
    ) -> None:
        """
        Inisialisasi audio manager dengan pygame mixer.
        
        Args:
            sample_rate (int): Sample rate audio (default: 44100 Hz)
            buffer_size (int): Ukuran buffer audio (default: 512)
            output_folder (Path): Folder untuk menyimpan audio yang sudah diproses
        """
        pygame.mixer.init(frequency=sample_rate, size=-16, channels=2, buffer=buffer_size)
        self._output_folder = output_folder
        self._output_folder.mkdir(parents=True, exist_ok=True)
        
        # Queue management
        self._audio_queue: List[AudioClip] = []
        self._current_clip: Optional[AudioClip] = None
        self._current_sound: Optional[pygame.mixer.Sound] = None
        
        # Callback untuk notifikasi
        self._on_finish_callback: Optional[Callable[[], None]] = None
        self._is_playing = False
        self._is_paused = False

    def prepare_audio_clips(
        self, 
        audio_files: List[Path], 
        semitones: float = -5.0,
        waveform_samples: int = 512
    ) -> List[AudioClip]:
        """
        Proses batch audio files dengan pitch shifting dan generate waveform.
        Method ini memanggil audio_processing untuk transformasi.
        
        Args:
            audio_files (List[Path]): List path file audio original
            semitones (float): Jumlah semitone untuk pitch shift (default: -5.0)
            waveform_samples (int): Jumlah sample untuk data waveform (default: 512)
            
        Returns:
            List[AudioClip]: List clip audio yang sudah diproses
        """
        clips = []
        print(f"[AudioManager] Memproses {len(audio_files)} file audio...")
        
        for audio_file in audio_files:
            try:
                # Generate nama file output
                stem = audio_file.stem
                output_name = f"{stem}_pitch{int(semitones)}.wav"
                output_path = self._output_folder / output_name
                
                # Panggil audio_processing untuk pitch shift
                sr_out, duration = audio_processing.pitch_shift_file(
                    input_path=audio_file,
                    output_path=output_path,
                    n_semitones=semitones,
                    sr_target=None  # Gunakan sample rate original
                )
                
                # Generate waveform data untuk visualisasi
                waveform_data = audio_processing.generate_waveform_data(
                    audio_path=output_path,
                    num_samples=waveform_samples
                )
                
                # Buat AudioClip object
                clip = AudioClip(
                    id=stem,
                    original_path=audio_file,
                    processed_path=output_path,
                    waveform_data=waveform_data,
                    duration=duration,
                    sample_rate=sr_out
                )
                clips.append(clip)
                
                print(f"  ✓ {audio_file.name} -> {output_name}")
                
            except Exception as e:
                print(f"[ERROR] Gagal memproses {audio_file.name}: {e}")
        
        print(f"[AudioManager] Selesai memproses {len(clips)} clip")
        return clips

    def set_queue(self, clips: List[AudioClip]) -> None:
        """
        Set queue audio clips yang akan dimainkan.
        
        Args:
            clips (List[AudioClip]): List audio clips untuk di-queue
        """
        # Stop audio yang sedang playing
        self.stop()
        
        # Set queue baru
        self._audio_queue = clips.copy()
        self._current_clip = None
        print(f"[AudioManager] Queue di-set dengan {len(clips)} clip")

    def load_clip_by_id(self, clip_id: str) -> bool:
        """
        Load audio clip dari queue berdasarkan ID.
        
        Args:
            clip_id (str): ID clip yang akan di-load
            
        Returns:
            bool: True jika berhasil, False jika clip tidak ditemukan
        """
        # Cari clip di queue
        clip = next((c for c in self._audio_queue if c.id == clip_id), None)
        
        if clip is None:
            print(f"[WARN] Clip dengan ID '{clip_id}' tidak ditemukan di queue")
            return False
        
        return self.load_clip(clip)

    def load_clip(self, clip: AudioClip) -> bool:
        """
        Load audio clip ke memory untuk playback.
        
        Args:
            clip (AudioClip): Clip yang akan di-load
            
        Returns:
            bool: True jika berhasil load
        """
        try:
            # Stop audio sebelumnya
            self.stop()

            # Load audio sebagai Sound object
            self._current_sound = pygame.mixer.Sound(str(clip.processed_path))
            self._current_clip = clip

            print(f"[AudioManager] Loaded: {clip.id}")
            return True

        except Exception as e:
            print(f"[ERROR] Gagal load clip {clip.id}: {e}")
            self._current_sound = None
            self._current_clip = None
            return False

    def get_current_waveform(self) -> List[float]:
        """
        Dapatkan data waveform dari clip yang sedang di-load.
        Digunakan oleh GUI untuk visualisasi waveform.
        
        Returns:
            List[float]: Data waveform ter-normalisasi (0.0 - 1.0)
                        Empty list jika tidak ada clip yang di-load
        """
        if self._current_clip:
            return self._current_clip.waveform_data
        return []

    def get_current_metadata(self) -> Optional[Dict]:
        """
        Dapatkan metadata lengkap dari clip yang sedang di-load.
        
        Returns:
            Dict: Metadata clip (id, duration, sample_rate, dll)
                  None jika tidak ada clip yang di-load
        """
        if self._current_clip:
            return {
                "id": self._current_clip.id,
                "duration": self._current_clip.duration,
                "sample_rate": self._current_clip.sample_rate,
                "original_path": str(self._current_clip.original_path),
                "processed_path": str(self._current_clip.processed_path),
                "waveform_samples": len(self._current_clip.waveform_data)
            }
        return None

    def get_queue_status(self) -> Dict:
        """
        Dapatkan status queue audio.
        
        Returns:
            Dict: Info queue (total, current_id, queue_list)
        """
        current_id = self._current_clip.id if self._current_clip else None
        return {
            "total_clips": len(self._audio_queue),
            "current_clip_id": current_id,
            "queue_ids": [c.id for c in self._audio_queue]
        }
    def play(self, on_finish: Optional[Callable[[], None]] = None) -> bool:
        """
        Mainkan audio yang sudah di-load.
        Sinkron dengan game state - hanya play jika ada clip yang di-load.
        
        Args:
            on_finish (Callable, optional): Callback yang dipanggil ketika audio selesai
            
        Returns:
            bool: True jika berhasil play, False jika tidak ada audio atau error
        """
        if self._current_sound is None or self._current_clip is None:
            print("[WARN] Tidak ada audio clip yang di-load untuk dimainkan")
            return False

        try:
            # Set callback untuk notifikasi selesai
            self._on_finish_callback = on_finish

            # Mainkan audio (loops=0 artinya play sekali saja)
            self._current_sound.play(loops=0)
            self._is_playing = True
            self._is_paused = False

            print(f"[AudioManager] Playing: {self._current_clip.id}")
            return True

        except Exception as e:
            print(f"[ERROR] Gagal memainkan audio: {e}")
            return False

    def stop(self) -> None:
        """
        Stop pemutaran audio saat ini.
        Menghentikan semua channel audio yang sedang aktif.
        """
        if self._is_playing:
            pygame.mixer.stop()  # Stop semua channel
            self._is_playing = False
            self._is_paused = False
            self._on_finish_callback = None

    def is_playing(self) -> bool:
        """
        Cek apakah audio sedang diputar.
        
        Returns:
            bool: True jika ada audio yang sedang playing
        """
        # pygame.mixer.get_busy() return True jika ada channel yang aktif
        return pygame.mixer.get_busy()

    def pause(self) -> None:
        """
        Pause playback (digunakan saat wajah tidak terdeteksi).
        """
        if self._is_playing and not self._is_paused:
            pygame.mixer.pause()
            self._is_playing = False
            self._is_paused = True

    def resume(self) -> None:
        """
        Lanjutkan playback setelah pause.
        """
        if self._is_paused:
            pygame.mixer.unpause()
            self._is_paused = False
            self._is_playing = True

    def set_volume(self, volume: float) -> None:
        """
        Set volume audio playback.
        
        Args:
            volume (float): Level volume (0.0 = mute, 1.0 = max volume)
        """
        if self._current_sound:
            # Clamp volume ke range 0.0 - 1.0
            volume = max(0.0, min(1.0, volume))
            self._current_sound.set_volume(volume)

    def check_finish(self) -> None:
        """
        Cek apakah audio sudah selesai dan panggil callback jika ada.
        Method ini harus dipanggil secara periodik dari main loop.
        """
        # Jika sebelumnya playing tapi sekarang sudah tidak busy
        if self._is_paused:
            return

        if self._is_playing and not self.is_playing():
            self._is_playing = False

            # Panggil callback jika sudah di-set
            if self._on_finish_callback:
                callback = self._on_finish_callback
                self._on_finish_callback = None
                callback()

    def clear_queue(self) -> None:
        """
        Kosongkan queue dan reset semua state.
        """
        self.stop()
        self._audio_queue.clear()
        self._current_clip = None
        self._current_sound = None
        print("[AudioManager] Queue dikosongkan")

    def cleanup(self) -> None:
        """
        Bersihkan resource audio dan quit pygame mixer.
        Panggil method ini sebelum aplikasi ditutup.
        """
        self.stop()
        self.clear_queue()
        pygame.mixer.quit()


if __name__ == "__main__":
    # Test audio manager dengan workflow lengkap
    import time
    
    # Inisialisasi manager
    manager = AudioManager(output_folder=Path("asset_output"))
    
    # Test 1: Prepare audio clips (pitch shifting + waveform generation)
    print("\n=== Test 1: Prepare Audio Clips ===")
    audio_folder = Path("asset")
    if audio_folder.exists():
        audio_files = audio_processing.find_audio_files(audio_folder)
        if audio_files:
            # Ambil 2 file pertama untuk test
            test_files = audio_files[:2]
            clips = manager.prepare_audio_clips(test_files, semitones=-5.0)
            
            # Test 2: Set queue
            print("\n=== Test 2: Set Queue ===")
            manager.set_queue(clips)
            print(f"Queue status: {manager.get_queue_status()}")
            
            # Test 3: Load dan play clip pertama
            if clips:
                print("\n=== Test 3: Load & Play ===")
                first_clip = clips[0]
                
                def on_done():
                    print("[CALLBACK] Audio selesai!")
                
                if manager.load_clip(first_clip):
                    # Tampilkan metadata
                    print(f"Metadata: {manager.get_current_metadata()}")
                    
                    # Tampilkan sample waveform
                    waveform = manager.get_current_waveform()
                    print(f"Waveform samples: {len(waveform)}")
                    print(f"First 10 samples: {waveform[:10]}")
                    
                    # Play audio
                    manager.play(on_finish=on_done)
                    
                    while manager.is_playing():
                        manager.check_finish()
                        time.sleep(0.1)
                    
                    time.sleep(0.5)  # Wait for callback
        else:
            print("Tidak ada file audio di folder 'asset'")
    else:
        print("Folder 'asset' tidak ditemukan")
    
    # Cleanup
    print("\n=== Cleanup ===")
    manager.cleanup()
    print("Test selesai!")
