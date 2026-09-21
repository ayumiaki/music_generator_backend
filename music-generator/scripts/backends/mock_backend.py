"""
Mock backend for music generation.
Generates a simple sine wave audio file if numpy and scipy are available,
otherwise writes a placeholder text file.
"""

import time
from pathlib import Path
from .base_backend import BaseBackend
from config import MOCK_MIN_DELAY, MOCK_MAX_DELAY


class MockBackend(BaseBackend):
    def __init__(self):
        super().__init__()
        # Lazy-import numpy/scipy so module loads without them
        self._can_generate_audio = False
        try:
            import numpy  # noqa: F401
            from scipy.io import wavfile  # noqa: F401
            self._can_generate_audio = True
        except ImportError:
            pass

    def generate(self, job_id: str, prompt: str, mood: str, tempo: int, key: str, length: int, seed: int | None = None) -> dict:
        # Lazy-import numpy for random delay
        np = None
        if self._can_generate_audio:
            try:
                import numpy as _np
                np = _np
            except ImportError:
                self._can_generate_audio = False

        # Simulate processing delay
        if np is not None:
            delay = np.random.uniform(float(MOCK_MIN_DELAY), float(MOCK_MAX_DELAY))
        else:
            delay = 0.01
        time.sleep(delay)

        output_file = self.output_dir / f"{job_id}.wav"

        if self._can_generate_audio:
            try:
                from scipy.io import wavfile
                import numpy as np

                # Simple sine wave at 440 Hz (A4) for demonstration
                sample_rate = 44100
                t = np.linspace(0, length, int(sample_rate * length), False)
                # Generate a tone based on key (simplistic mapping)
                note_frequencies = {
                    'C': 261.63,
                    'C#': 277.18,
                    'D': 293.66,
                    'D#': 311.13,
                    'E': 329.63,
                    'F': 349.23,
                    'F#': 369.99,
                    'G': 392.00,
                    'G#': 415.30,
                    'A': 440.00,
                    'A#': 466.16,
                    'B': 493.88
                }
                # Default to A4 if key not found
                freq = note_frequencies.get(key.upper(), 440.0)
                # Adjust tempo? Not needed for tone length
                audio = np.sin(freq * 2 * np.pi * t)
                # Apply envelope to avoid clicks
                envelope = np.ones_like(audio)
                attack_len = int(0.01 * sample_rate)
                release_len = int(0.01 * sample_rate)
                if len(envelope) > attack_len + release_len:
                    envelope[:attack_len] = np.linspace(0, 1, attack_len)
                    envelope[-release_len:] = np.linspace(1, 0, release_len)
                audio = audio * envelope
                # Normalize to 16-bit range
                audio = np.int16(audio * 32767)
                wavfile.write(output_file, sample_rate, audio)

                return {
                    "status": "success",
                    "output_file": str(output_file),
                    "metadata": {
                        "sample_rate": sample_rate,
                        "duration": length,
                        "key": key,
                        "tempo": tempo,
                        "mood": mood,
                        "frequency_hz": freq,
                        "synthesis": "sine wave"
                    }
                }
            except Exception as e:
                # Fallback to text file on any error
                pass

        # Fallback: write a text file with job details
        output_file = self.output_dir / f"{job_id}.txt"
        output_file.write_text(
            f"Mock music generation\n"
            f"Job ID: {job_id}\n"
            f"Prompt: {prompt}\n"
            f"Mood: {mood}\n"
            f"Tempo: {tempo}\n"
            f"Key: {key}\n"
            f"Length: {length} seconds\n"
            f"Generated at: {time.ctime()}\n"
        )

        return {
            "status": "success",
            "output_file": str(output_file),
            "metadata": {
                "sample_rate": 0,
                "duration": length,
                "key": key,
                "tempo": tempo,
                "mood": mood,
                "note": "Generated placeholder text file; install numpy and scipy for audio output."
            }
        }