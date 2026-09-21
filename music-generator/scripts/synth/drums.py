"""Deterministic drum pattern — timing from sample positions, no floating-point sleeps."""
import numpy as np
from numpy.typing import NDArray


# Kick: low-frequency sine burst with zero DC and clean tail
def _kick(duration_samples: int, sample_rate: int) -> NDArray[np.float64]:
    t = np.arange(duration_samples, dtype=np.float64) / sample_rate
    freq = 60.0
    freq_env = freq * np.exp(-t * 40.0)
    phase = 2.0 * np.pi * np.cumsum(freq_env) / sample_rate
    signal = np.sin(phase) * np.exp(-t * 8.0)
    # Remove DC offset
    signal = signal - np.mean(signal)
    # Ensure zero termination (fade out last 10%)
    fade_len = min(duration_samples // 10, 480)
    if fade_len > 0:
        signal[-fade_len:] *= np.linspace(1.0, 0.0, fade_len)
    return signal


# Snare: noise burst + tone with zero DC and clean tail
def _snare(duration_samples: int, sample_rate: int, rng: np.random.RandomState) -> NDArray[np.float64]:
    t = np.arange(duration_samples, dtype=np.float64) / sample_rate
    tone = np.sin(2.0 * np.pi * 200 * t) * np.exp(-t * 15.0)
    noise = rng.uniform(-1, 1, duration_samples) * np.exp(-t * 6.0)
    signal = tone * 0.5 + noise * 0.5
    # Remove DC offset
    signal = signal - np.mean(signal)
    # Ensure zero termination
    fade_len = min(duration_samples // 10, 480)
    if fade_len > 0:
        signal[-fade_len:] *= np.linspace(1.0, 0.0, fade_len)
    return signal


# Hi-hat: short noise burst with zero DC and clean tail
def _hihat(duration_samples: int, sample_rate: int, rng: np.random.RandomState) -> NDArray[np.float64]:
    t = np.arange(duration_samples, dtype=np.float64) / sample_rate
    noise = rng.uniform(-1, 1, duration_samples) * np.exp(-t * 60.0)
    # High-pass via simple diff
    filtered = np.diff(noise, prepend=0.0)
    # Remove DC offset
    filtered = filtered - np.mean(filtered)
    # Ensure zero termination
    fade_len = min(duration_samples // 10, 480)
    if fade_len > 0:
        filtered[-fade_len:] *= np.linspace(1.0, 0.0, fade_len)
    return filtered


# Kick patterns: beats where kick falls (in beats, 0-indexed)
_KICK_PATTERNS = {
    "four_on_floor": [0, 1, 2, 3],
    "eighth": [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5],
    "half_time": [0, 2],
}

# Snare patterns
_SNARE_PATTERNS = {
    "backbeat": [1, 3],
    "every_beat": [0, 1, 2, 3],
    "offbeat": [0.5, 1.5, 2.5, 3.5],
}

# Hi-hat patterns
_HIHAT_PATTERNS = {
    "eighth": [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5],
    "quarter": [0, 1, 2, 3],
    "sixteenth": [i * 0.25 for i in range(16)],
}


class DrumPattern:
    """Deterministic drum pattern from sample positions.

    Timing is derived from sample positions, not cumulative sleeps.
    Multi-bar rendering with per-bar sample offset.
    """

    def __init__(
        self,
        bpm: float = 120.0,
        sample_rate: int = 48000,
        seed: int | None = None,
    ) -> None:
        self.bpm = bpm
        self.sample_rate = sample_rate
        self.beat_duration = 60.0 / bpm  # seconds per beat
        self._rng = np.random.RandomState(seed)

    def _beat_to_samples(self, beat: float) -> int:
        """Convert beat position to sample index."""
        return int(beat * self.beat_duration * self.sample_rate)

    def render(
        self,
        bars: int = 1,
        kick_pattern: str = "four_on_floor",
        snare_pattern: str = "backbeat",
        hihat_pattern: str = "eighth",
    ) -> NDArray[np.float64]:
        """Render drum pattern for given bars with multi-bar scheduling."""
        beats_per_bar = 4
        total_beats = bars * beats_per_bar
        total_samples = int(total_beats * self.beat_duration * self.sample_rate)
        output = np.zeros(total_samples, dtype=np.float64)

        # Render each bar with per-bar sample offset
        for bar in range(bars):
            bar_offset = bar * beats_per_bar  # beat offset for this bar
            bar_sample_offset = self._beat_to_samples(bar_offset)

            # Render kick
            kick_dur = int(0.1 * self.sample_rate)
            for beat in _KICK_PATTERNS.get(kick_pattern, _KICK_PATTERNS["four_on_floor"]):
                start = bar_sample_offset + self._beat_to_samples(beat)
                if start + kick_dur <= total_samples:
                    output[start : start + kick_dur] += _kick(kick_dur, self.sample_rate) * 0.8

            # Render snare
            snare_dur = int(0.15 * self.sample_rate)
            for beat in _SNARE_PATTERNS.get(snare_pattern, _SNARE_PATTERNS["backbeat"]):
                start = bar_sample_offset + self._beat_to_samples(beat)
                if start + snare_dur <= total_samples:
                    output[start : start + snare_dur] += _snare(snare_dur, self.sample_rate, self._rng) * 0.6

            # Render hihat
            hihat_dur = int(0.05 * self.sample_rate)
            for beat in _HIHAT_PATTERNS.get(hihat_pattern, _HIHAT_PATTERNS["eighth"]):
                start = bar_sample_offset + self._beat_to_samples(beat)
                if start + hihat_dur <= total_samples:
                    output[start : start + hihat_dur] += _hihat(hihat_dur, self.sample_rate, self._rng) * 0.4

        # Prevent clipping
        max_val = np.max(np.abs(output)) if len(output) > 0 else 1.0
        if max_val > 0.95:
            output = output / max_val * 0.95

        return output.astype(np.float64)