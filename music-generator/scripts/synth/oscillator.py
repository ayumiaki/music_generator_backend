"""Band-limited wavetable oscillator with PolyBLEP and integer-sample phase."""
import numpy as np
from numpy.typing import NDArray


def _polyblep(t: NDArray[np.float64], dt: float) -> NDArray[np.float64]:
    """PolyBLEP discontinuity correction for a saw/square wave.

    t : phase in cycles [0, 1)
    dt: phase increment per sample (freq / sample_rate)
    """
    if dt <= 0:
        return np.zeros_like(t)
    result = np.zeros_like(t, dtype=np.float64)
    # Region just before discontinuity (t near 1)
    mask = t > (1.0 - dt)
    x = (t[mask] - 1.0) / dt
    result[mask] = x * x + x + x + 1.0
    # Region just after discontinuity (t near 0)
    mask = t < dt
    x = t[mask] / dt
    result[mask] = x + x - x * x - 1.0
    return result


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
    max_val = np.max(np.abs(tbl))
    if max_val > 0:
        tbl = tbl / max_val
    return tbl


class WavetableOscillator:
    """Phase-continuous wavetable oscillator with PolyBLEP alias suppression.

    Phase authority is an integer sample counter — rendering is partition-invariant
    regardless of block boundaries. The seed applies a deterministic phase offset.
    """

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
        self._waveform = waveform
        self._sample_pos: int = 0
        self._rng = np.random.RandomState(seed)
        # Seed-based phase offset in cycles [0, 1)
        self._seed_phase_offset = (seed % table_size) / table_size if seed is not None else 0.0

    def reset(self) -> None:
        self._sample_pos = 0

    def set_phase(self, phase: float) -> None:
        self._sample_pos = int(phase * self.sample_rate)

    def get_phase(self) -> float:
        return (self._sample_pos % self.sample_rate) / self.sample_rate

    def render(
        self,
        freq: float,
        n_frames: int,
        phase_reset: bool = False,
        reset_phase: float = 0.0,
    ) -> NDArray[np.float64]:
        if n_frames <= 0:
            return np.array([], dtype=np.float64)

        if phase_reset:
            self._sample_pos = int(reset_phase * self.sample_rate)

        # Absolute sample indices for this block
        indices = np.arange(self._sample_pos, self._sample_pos + n_frames, dtype=np.float64)
        # Phase in cycles [0, 1) with seed offset
        t = (freq * indices / self.sample_rate + self._seed_phase_offset) % 1.0
        dt = freq / self.sample_rate

        if self._waveform == "sine":
            out = np.sin(2.0 * np.pi * t)
        elif self._waveform == "saw":
            # Ideal saw: 2*(t - 0.5), corrected with PolyBLEP
            out = 2.0 * (t - 0.5) - _polyblep(t, dt)
        elif self._waveform == "square":
            # Ideal square: +1 for t<0.5, -1 otherwise
            ideal = np.where(t < 0.5, 1.0, -1.0)
            # Corrections at t=0 (drop of 2) and t=0.5 (rise of 2)
            out = ideal - _polyblep(t, dt) + _polyblep((t + 0.5) % 1.0, dt)
        elif self._waveform == "triangle":
            # Triangle: 2*|2*(t+0.25) - floor(2t+1)| - 1, no PolyBLEP needed (continuous)
            out = 2.0 * np.abs(2.0 * (t + 0.25) - np.floor(2.0 * t + 1.0) - 1.0) - 1.0
        else:
            out = np.sin(2.0 * np.pi * t)

        self._sample_pos += n_frames
        return out.astype(np.float64)

    def render_block(
        self,
        notes: list[tuple[float, int]],
        block_size: int = 48000,
    ) -> NDArray[np.float64]:
        chunks = []
        for freq, n in notes:
            chunks.append(self.render(freq, n))
        if not chunks:
            return np.array([], dtype=np.float64)
        return np.concatenate(chunks)