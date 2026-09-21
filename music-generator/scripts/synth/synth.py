"""SynthEngine — ties oscillators, envelopes, filters, drums, arrangement together."""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import numpy as np
import sys
import os

# Ensure scripts dir is on path
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from .oscillator import WavetableOscillator, make_wavetable
from .envelope import ADSREnvelope
from .filter import LadderFilter
from .voice import VoiceEngine, VoiceConfig
from .drums import DrumPattern
from .arrangement import Arrangement, ArrangementConfig
from .renderer import render, render_integrity, RenderIntegrity


@dataclass
class SynthConfig:
    """Configuration for the SynthEngine."""
    sample_rate: int = 48000
    polyphony: int = 8
    bpm: float = 120.0
    bars: int = 1
    wavetable_size: int = 4096
    waveform: str = "saw"
    attack_samples: int = 4800
    decay_samples: int = 2400
    sustain_level: float = 0.7
    release_samples: int = 4800
    filter_cutoff: float = 20000.0
    filter_resonance: float = 0.0
    retrigger: bool = True
    seed: int | None = None
    output_dir: str = "output"
    queue_dir: str = "queue"


class SynthEngine:
    """Deterministic synthesizer engine.

    Produces actual PCM audio from note events — no API-shaped cardboard scenery.
    Identical input + seed → byte-identical PCM output.
    """

    def __init__(self, config: Optional[SynthConfig] = None) -> None:
        self.config = config or SynthConfig()
        self._rng = np.random.RandomState(self.config.seed)
        self._arrangement = Arrangement(
            ArrangementConfig(
                bpm=self.config.bpm,
                bars=self.config.bars,
                polyphony=self.config.polyphony,
                sample_rate=self.config.sample_rate,
                wavetable_size=self.config.wavetable_size,
                waveform=self.config.waveform,
                attack_samples=self.config.attack_samples,
                decay_samples=self.config.decay_samples,
                sustain_level=self.config.sustain_level,
                release_samples=self.config.release_samples,
                filter_cutoff=self.config.filter_cutoff,
                filter_resonance=self.config.filter_resonance,
                retrigger=self.config.retrigger,
                seed=self.config.seed,
            )
        )
        self._drums = DrumPattern(
            bpm=self.config.bpm,
            sample_rate=self.config.sample_rate,
            seed=self.config.seed,
        )
        self._rendered: Optional[np.ndarray] = None
        self._integrity: Optional[RenderIntegrity] = None

    def add_note(
        self,
        time_beats: float,
        freq: float,
        duration_beats: float = 0.25,
        amplitude: float = 1.0,
        waveform: str = "saw",
    ) -> None:
        """Add a note to the arrangement."""
        self._arrangement.add_note(
            time_beats=time_beats,
            freq=freq,
            duration_beats=duration_beats,
            amplitude=amplitude,
            waveform=waveform,
        )

    def add_drums(
        self,
        bars: int = 1,
        kick_pattern: str = "four_on_floor",
        snare_pattern: str = "backbeat",
        hihat_pattern: str = "eighth",
    ) -> None:
        """Add drum pattern."""
        self._drum_bars = bars
        self._kick_pattern = kick_pattern
        self._snare_pattern = snare_pattern
        self._hihat_pattern = hihat_pattern

    def render(self, output_path: Optional[str | Path] = None) -> RenderIntegrity:
        """Render the arrangement + drums to PCM. Optionally write WAV."""
        # Render arrangement
        audio = self._arrangement.render()

        # Render drums and mix
        if hasattr(self, "_drum_bars"):
            drums = self._drums.render(
                bars=self._drum_bars,
                kick_pattern=self._kick_pattern,
                snare_pattern=self._snare_pattern,
                hihat_pattern=self._hihat_pattern,
            )
            # Mix drums at lower volume
            if len(drums) > len(audio):
                audio = np.pad(audio, (0, len(drums) - len(audio)))
            elif len(audio) > len(drums):
                drums = np.pad(drums, (0, len(audio) - len(drums)))
            audio = audio + drums * 0.5

        # Global limiter
        max_val = np.max(np.abs(audio)) if len(audio) > 0 else 1.0
        if max_val > 0.95:
            audio = audio / max_val * 0.95

        self._rendered = audio

        # Integrity check
        self._integrity = render_integrity(audio, self.config.sample_rate)

        # Write WAV if path provided
        if output_path is not None:
            from .renderer import write_wav
            write_wav(output_path, audio, self.config.sample_rate)

        return self._integrity

    def get_integrity(self) -> Optional[RenderIntegrity]:
        return self._integrity

    def get_audio(self) -> Optional[np.ndarray]:
        return self._rendered

    def reset(self) -> None:
        """Reset engine state."""
        self._arrangement.clear()
        self._rendered = None
        self._integrity = None