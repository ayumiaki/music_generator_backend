"""Synth backend — deterministic DSP producing actual PCM audio."""
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from backends.base_backend import BaseBackend
from synth.synth import SynthEngine
from synth.renderer import write_wav
from config import OUTPUT_DIR


class SynthBackend(BaseBackend):
    """Deterministic synthesizer backend.

    Produces actual audio from note events — oscillators, envelopes,
    filters, drums, arrangement. Byte-identical PCM for identical input+seed.
    """

    def __init__(self) -> None:
        super().__init__()
        self._engine: SynthEngine | None = None

    def generate(
        self,
        job_id: str,
        prompt: str,
        mood: str,
        tempo: int,
        key: str,
        length: int,
        seed: int | None = None,
    ) -> dict:
        """Generate audio using the deterministic synth engine."""
        # Parse prompt for notes (simple format: "C4 E4 G4" etc.)
        notes = self._parse_notes(prompt)

        config = SynthConfig(
            bpm=tempo,
            bars=max(1, length // 16),
            seed=seed,
            sample_rate=48000,
        )
        engine = SynthEngine(config)

        # Add notes from prompt
        beat_pos = 0.0
        for freq in notes:
            engine.add_note(
                time_beats=beat_pos,
                freq=freq,
                duration_beats=0.5,
                amplitude=0.8,
            )
            beat_pos += 0.5

        # Add drums
        engine.add_drums(bars=config.bars)

        # Render
        output_dir = Path(OUTPUT_DIR)
        output_dir.mkdir(exist_ok=True)
        wav_path = output_dir / f"{job_id}.wav"

        integrity = engine.render(output_path=wav_path)

        return {
            "status": "success",
            "output_file": str(wav_path),
            "sample_rate": integrity.sample_rate,
            "frame_count": integrity.frame_count,
            "duration_sec": integrity.duration_sec,
            "clipping": integrity.clipping,
            "dc_offset": integrity.dc_offset,
            "all_finite": integrity.all_finite,
            "max_amplitude": integrity.max_amplitude,
            "rms": integrity.rms,
            "seed": seed,
        }

    def _parse_notes(self, prompt: str) -> list[float]:
        """Parse a simple note prompt into frequencies.

        Supports note names like C4, E4, G4 with optional #/b.
        Falls back to a C major chord if parsing fails.
        """
        note_map = {
            "C": 261.63, "C#": 277.18, "Db": 277.18,
            "D": 293.66, "D#": 311.13, "Eb": 311.13,
            "E": 329.63, "F": 349.23, "F#": 369.99, "Gb": 369.99,
            "G": 392.00, "G#": 415.30, "Ab": 415.30,
            "A": 440.00, "A#": 466.16, "Bb": 466.16,
            "B": 493.88,
        }

        tokens = prompt.strip().split()
        freqs = []
        for token in tokens:
            # Try to extract note name + octave
            for i in range(1, len(token)):
                name = token[:i]
                octave_str = token[i:]
                if name in note_map:
                    try:
                        octave = int(octave_str)
                        freq = note_map[name] * (2 ** (octave - 4))
                        freqs.append(freq)
                        break
                    except ValueError:
                        continue
            else:
                # Fallback: use C4
                freqs.append(261.63)

        if not freqs:
            # Default C major chord
            freqs = [261.63, 329.63, 392.00]

        return freqs


# Re-export config for use by the engine
from synth.synth import SynthConfig