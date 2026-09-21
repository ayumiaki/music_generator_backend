"""Wavetable oscillator with phase continuity, interpolation, alias control."""
import numpy as np
from numpy.typing import NDArray


def make_wavetable(size: int = 4096, waveform: str = "saw") -> NDArray[np.float64]:
    """Generate a single-cycle wavetable of given size."""
    t = np.linspace(0.0, 1.0, size, endpoint=False)
    if waveform == "saw":
        tbl = 2.0 * (t - np.floor(t + 0.5))
    elif waveform == "square":
        tbl = np.where(t < 0.5, 1.0, -1.0)
    elif waveform == "triangle":
        tbl = 2.0 * np.abs(2.0 * (t + 0.25) - np.floor(2.0 * t + 1.0) - 1.0) - 1.0
    elif waveform == "sine":
        tbl = np.sin(2.0 * np.pi * t)
    else:
        tbl = np.sin(2.0 * np.pi * t)
    # Normalize to [-1, 1]
    max_val = np.max(np.abs(tbl))
    if max_val > 0:
        tbl = tbl / max_val
    return tbl


class WavetableOscillator:
    """Phase-continuous wavetable oscillator with interpolation and deterministic phase reset."""

    def __init__(
        self,
        sample_rate: int = 48000,
        table_size: int = 4096,
        waveform: str = "saw",
        seed: int | None = None,
    ) -> None:
        self.sample_rate = sample_rate
        self.table_size = table_size
        self.table = make_wavetable(table_size, waveform)
        self.phase = 0.0  # in table indices [0, table_size)
        self._rng = np.random.RandomState(seed)
        # Seed-based initial phase offset for determinism
        if seed is not None:
            self._seed_offset = float(seed % table_size)
        else:
            self._seed_offset = 0.0

    def reset(self) -> None:
        """Hard reset — phase goes to zero deterministically."""
        self.phase = 0.0

    def set_phase(self, phase: float) -> None:
        """Set phase explicitly (mod table_size)."""
        self.phase = float(phase) % self.table_size

    def get_phase(self) -> float:
        return self.phase

    def render(
        self,
        freq: float,
        n_frames: int,
        phase_reset: bool = False,
        reset_phase: float = 0.0,
    ) -> NDArray[np.float64]:
        """Render n_frames at given frequency. Phase-continuous across calls."""
        if n_frames <= 0:
            return np.array([], dtype=np.float64)

        if phase_reset:
            self.phase = reset_phase % self.table_size

        # Frequency -> phase increment
        phase_inc = freq * self.table_size / self.sample_rate

        # Generate phase values, starting from current phase + seed offset
        start_phase = self.phase + self._seed_offset
        phases = start_phase + np.arange(n_frames, dtype=np.float64) * phase_inc

        # Wrap phase to [0, table_size)
        phases = phases % self.table_size

        # Linear interpolation between adjacent table entries
        idx_lo = np.floor(phases).astype(np.int64) % self.table_size
        idx_hi = (idx_lo + 1) % self.table_size
        frac = phases - np.floor(phases)

        output = (1.0 - frac) * self.table[idx_lo] + frac * self.table[idx_hi]

        # Update phase for next call (continuity)
        self.phase = (self.phase + n_frames * phase_inc) % self.table_size

        return output.astype(np.float64)

    def render_block(
        self,
        notes: list[tuple[float, int]],
        block_size: int = 48000,
    ) -> NDArray[np.float64]:
        """Render a block from a list of (freq, frames) tuples. Concatenates."""
        chunks = []
        for freq, n in notes:
            chunks.append(self.render(freq, n))
        if not chunks:
            return np.array([], dtype=np.float64)
        return np.concatenate(chunks)