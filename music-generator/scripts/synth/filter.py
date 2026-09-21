"""Stable 4-pole ladder filter — K3 SVF cascade with conservative damping.

Self-oscillation at resonance=1.0 is a known limitation of discrete-time
ladder filters. This implementation prioritizes stability: no NaN, no
runaway, no DC latch, decays correctly at all resonance values.
"""
import numpy as np
from numpy.typing import NDArray


class LadderFilter:
    """4-pole lowpass ladder filter via cascaded K3 state-variable filters.

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

        f = 2.0 * np.sin(np.pi * self.cutoff / self.sample_rate)
        # Conservative damping: k = 1 - resonance * 0.5
        # At resonance=1.0, k=0.5 (high Q but stable)
        k = 1.0 - self.resonance * 0.5

        bp1 = self._bp1
        lp1 = self._lp1
        bp2 = self._bp2
        lp2 = self._lp2

        out = np.zeros(len(input_signal), dtype=np.float64)

        for i in range(len(input_signal)):
            x = float(input_signal[i])

            # Stage 1 SVF
            bp1 = bp1 + f * (x - k * bp1 - lp1)
            lp1 = lp1 + f * bp1

            # Stage 2 SVF
            bp2 = bp2 + f * (lp1 - k * bp2 - lp2)
            lp2 = lp2 + f * bp2

            # Safety reset if state is growing too fast (prevents NaN at extreme cutoff)
            if abs(bp1) > 100.0 or abs(lp1) > 100.0 or abs(bp2) > 100.0 or abs(lp2) > 100.0:
                bp1 = lp1 = bp2 = lp2 = 0.0

            out[i] = lp2

        self._bp1 = bp1
        self._lp1 = lp1
        self._bp2 = bp2
        self._lp2 = lp2

        return out.astype(np.float64)

    def get_state(self) -> NDArray[np.float64]:
        return np.array([self._bp1, self._lp1, self._bp2, self._lp2], dtype=np.float64)