"""Synth backend — deterministic DSP producing actual PCM audio."""
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from backends.base_backend import BaseBackend
from synth.synth import SynthEngine, SynthConfig
from synth.renderer import write_wav, render_integrity
from config import OUTPUT_DIR


# Key to root frequency mapping
KEY_MAP = {
    "C": 261.63, "C#": 277.18, "Db": 277.18,
    "D": 293.66, "D#": 311.13, "Eb": 311.13,
    "E": 329.63, "F": 349.23, "F#": 369.99, "Gb": 369.99,
    "G": 392.00, "G#": 415.30, "Ab": 415.30,
    "A": 440.00, "A#": 466.16, "Bb": 466.16,
    "B": 493.88,
}

# Mood to synthesis parameters
MOOD_MAP = {
    "happy": {"cutoff": 12000, "attack": 100, "decay": 200, "sustain": 0.8, "release": 300, "waveform": "saw"},
    "sad": {"cutoff": 3000, "attack": 1000, "decay": 500, "sustain": 0.5, "release": 1000, "waveform": "sine"},
    "energetic": {"cutoff": 18000, "attack": 50, "decay": 100, "sustain": 0.9, "release": 200, "waveform": "square"},
    "calm": {"cutoff": 5000, "attack": 2000, "decay": 1000, "sustain": 0.6, "release": 2000, "waveform": "triangle"},
}


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
        output_path: str | None = None,
    ) -> dict:
        """Generate audio using the deterministic synth engine.

        If output_path is given, render directly there (used by worker for
        atomic temp→final rename). Otherwise use the default output dir.
        """
        # Validate seed — the API layer returns 400 for out-of-range seeds,
        # so reaching here with an invalid seed is a programming error
        if seed is not None and (seed < 0 or seed > 2**32 - 1):
            raise ValueError(
                f"Invalid seed: {seed}. Seed must be in range [0, {2**32 - 1}]."
            )

        # Parse prompt for notes
        notes = self._parse_notes(prompt, key)

        # Duration: length is in seconds — passed through directly.
        # The arrangement derives the exact frame count:
        # int(length * sample_rate), rendering and trimming to match.
        # Bars are only used for drum pattern scheduling.
        seconds_per_bar = 60.0 / tempo * 4.0
        bars = max(1, int(-(-length / seconds_per_bar)))  # ceil: cover the length

        # Get mood parameters
        mood_params = MOOD_MAP.get(mood, MOOD_MAP["happy"])

        config = SynthConfig(
            bpm=tempo,
            bars=bars,
            length=float(length),
            seed=seed,
            sample_rate=48000,
            waveform=mood_params["waveform"],
            attack_samples=mood_params["attack"],
            decay_samples=mood_params["decay"],
            sustain_level=mood_params["sustain"],
            release_samples=mood_params["release"],
            filter_cutoff=mood_params["cutoff"],
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
        engine.add_drums(bars=bars)

        # Render: use output_path if given (worker temp file), else default
        if output_path:
            wav_path = Path(output_path)
            wav_path.parent.mkdir(parents=True, exist_ok=True)
        else:
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

    def _parse_notes(self, prompt: str, key: str) -> list[float]:
        """Parse a simple note prompt into frequencies.

        Supports note names like C4, E4, G4 with optional #/b.
        Falls back to a chord based on the key if parsing fails.
        """
        # Get root frequency from key
        root_freq = KEY_MAP.get(key, 261.63)  # Default to C4

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
                # Fallback: use root frequency
                freqs.append(root_freq)

        if not freqs:
            # Default chord based on key (major triad)
            freqs = [root_freq, root_freq * 1.25, root_freq * 1.5]

        return freqs


# Re-export config for use by the engine
from synth.synth import SynthConfig