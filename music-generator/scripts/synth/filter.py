"""Stable 4-pole ladder filter — TPT/ZDF SVF cascade.

Uses the complete TPT/ZDF recurrence:
  g = tan(π*fc/fs) — warped frequency coefficient
  k = 1/Q — damping (resonance)
  v1 = (x - k*bp - lp) / (1 + g*(g+k)) — normalized solve
  bp = bp + g*v1 — trapezoidal state update
  lp = lp + g*bp — trapezoidal state update

This is unconditionally stable for all cutoff frequencies because:
- The denominator 1 + g*(g+k) is always positive
- The state updates are implicit (trapezoidal)
- No safety resets needed

Self-oscillation at resonance=1.0 is a known limitation of discrete-time
ladder filters. This implementation prioritizes stability: no NaN, no
runaway, no DC latch, decays correctly at all resonance values.
"""
import numpy as np
from numpy.typing import NDArray


class LadderFilter:
    """4-pole lowpass ladder filter via cascaded TPT/ZDF SVF stages.

    Topology: two SVF stages in series. Unconditionally stable.
    - cutoff: 20–22050 Hz (for 48kHz sample rate)
    - resonance: 0.0–1.0; higher values = more peak at cutoff
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
        self._bp1 = 0.0
        self._lp1 = 0.0
        self._bp2 = 0.0
        self._lp2 = 0.0

    def reset(self) -> None:
        self._bp1 = 0.0
        self._lp1 = 0.0
        self._bp2 = 0.0
        self._lp2 = 0.0

    def set_cutoff(self, cutoff: float) -> None:
        self.cutoff = max(1.0, min(cutoff, self.sample_rate / 2.0))

    def set_resonance(self, resonance: float) -> None:
        self.resonance = max(0.0, min(resonance, 1.0))

    def render(self, input_signal: NDArray[np.float64]) -> NDArray[np.float64]:
        if len(input_signal) == 0:
            return input_signal.copy()

        # TPT/ZDF coefficients
        g = np.tan(np.pi * self.cutoff / self.sample_rate)
        # k = 1/Q; resonance maps to Q = 1/(1-resonance*0.9)
        # At resonance=0, Q=1 (no peak); at resonance=1, Q=10 (high peak)
        q = 1.0 / (1.0 - self.resonance * 0.9 + 1e-10)
        k = 1.0 / q

        bp1 = self._bp1
        lp1 = self._lp1
        bp2 = self._bp2
        lp2 = self._lp2

        out = np.zeros(len(input_signal), dtype=np.float64)

        for i in range(len(input_signal)):
            x = float(input_signal[i])

            # Stage 1 SVF — complete TPT/ZDF recurrence
            v1 = (x - k * bp1 - lp1) / (1.0 + g * (g + k))
            bp1 = bp1 + g * v1
            lp1 = lp1 + g * bp1

            # Stage 2 SVF — complete TPT/ZDF recurrence
            v2 = (lp1 - k * bp2 - lp2) / (1.0 + g * (g + k))
            bp2 = bp2 + g * v2
            lp2 = lp2 + g * bp2

            out[i] = lp2

        self._bp1 = bp1
        self._lp1 = lp1
        self._bp2 = bp2
        self._lp2 = lp2

        return out.astype(np.float64)

    def get_state(self) -> NDArray[np.float64]:
        return np.array([self._bp1, self._lp1, self._bp2, self._lp2], dtype=np.float64)
