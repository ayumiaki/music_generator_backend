"""Stable 4-pole lowpass filter — cascade of four one-pole lowpasses.

Each one-pole stage: y[n] = y[n-1] + alpha * (x[n] - y[n-1])
where alpha = 1 - exp(-2*pi*fc/fs)

Cascading four identical stages gives a 4-pole response (24 dB/octave)
with no resonance peak. This is numerically stable for all cutoff
frequencies — no TPT/ZDF coefficient explosions, no state resets.

The trade-off: no resonance control. If resonance is needed, it must
be added as a separate stage (e.g., a resonant one-pole with feedback).
"""
import numpy as np
from numpy.typing import NDArray


class LadderFilter:
    """4-pole lowpass filter via cascaded one-pole stages.

    Topology: four one-pole lowpass stages in series. Unconditionally stable.
    - cutoff: 20–22050 Hz (for 48kHz sample rate)
    - resonance: 0.0 (not implemented — this is a stability fallback)
    """

    def __init__(
        self,
        sample_rate: int = 48000,
        cutoff: float = 20000.0,
        resonance: float = 0.0,
    ) -> None:
        self.sample_rate = sample_rate
        self.cutoff = cutoff
        self.resonance = resonance
        # Four one-pole stage states
        self._s1 = 0.0
        self._s2 = 0.0
        self._s3 = 0.0
        self._s4 = 0.0

    def reset(self) -> None:
        self._s1 = 0.0
        self._s2 = 0.0
        self._s3 = 0.0
        self._s4 = 0.0

    def set_cutoff(self, cutoff: float) -> None:
        self.cutoff = max(1.0, min(cutoff, self.sample_rate / 2.0))

    def set_resonance(self, resonance: float) -> None:
        self.resonance = max(0.0, min(resonance, 1.0))

    def render(self, input_signal: NDArray[np.float64]) -> NDArray[np.float64]:
        if len(input_signal) == 0:
            return input_signal.copy()

        # One-pole coefficient: alpha = 1 - exp(-2*pi*fc/fs)
        # At fc=20kHz, fs=48kHz: alpha ≈ 0.917 (very open)
        # At fc=100Hz, fs=48kHz: alpha ≈ 0.013 (very closed)
        alpha = 1.0 - np.exp(-2.0 * np.pi * self.cutoff / self.sample_rate)
        alpha = max(0.0, min(alpha, 1.0))

        s1, s2, s3, s4 = self._s1, self._s2, self._s3, self._s4
        out = np.zeros(len(input_signal), dtype=np.float64)

        for i in range(len(input_signal)):
            x = float(input_signal[i])
            s1 = s1 + alpha * (x - s1)
            s2 = s2 + alpha * (s1 - s2)
            s3 = s3 + alpha * (s2 - s3)
            s4 = s4 + alpha * (s3 - s4)
            out[i] = s4

        self._s1 = s1
        self._s2 = s2
        self._s3 = s3
        self._s4 = s4

        return out.astype(np.float64)

    def get_state(self) -> NDArray[np.float64]:
        return np.array([self._s1, self._s2, self._s3, self._s4], dtype=np.float64)
