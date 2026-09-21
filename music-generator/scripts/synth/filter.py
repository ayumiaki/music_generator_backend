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

    def _compute_alpha(self) -> float:
        """Compute per-stage alpha for calibrated composite cutoff.

        Four cascaded one-poles each at -3dB at the requested frequency
        give -12dB there. We need each stage at 2^(-1/8) magnitude at
        the composite cutoff so the cascade hits -3dB at the requested fc.

        For one-pole y[n] = alpha*x[n] + (1-alpha)*y[n-1]:
          |H(ω)|² = alpha² / (1 - 2r*cos(ω) + r²)  where r = 1-alpha

        Setting |H|² = 2^(-1/4) (i.e. |H| = 2^(-1/8)) and solving:
          (1-m²)r² + (-2 + 2m²cos(ω))r + (1-m²) = 0
          where m² = 2^(-1/4), ω = 2πfc/fs

        Pick root where 0 < r < 1, then alpha = 1 - r.
        """
        fc = max(1.0, min(self.cutoff, self.sample_rate / 2.0))
        omega = 2.0 * np.pi * fc / self.sample_rate
        m2 = 2.0 ** (-0.25)  # 2^(-1/4)

        # Quadratic coefficients: a*r² + b*r + c = 0
        a = 1.0 - m2
        b = -2.0 + 2.0 * m2 * np.cos(omega)
        c = 1.0 - m2

        discriminant = b * b - 4.0 * a * c
        if discriminant < 0:
            # Fallback: standard one-pole coefficient
            return 1.0 - np.exp(-2.0 * np.pi * fc / self.sample_rate)

        sqrt_disc = np.sqrt(discriminant)
        r1 = (-b + sqrt_disc) / (2.0 * a)
        r2 = (-b - sqrt_disc) / (2.0 * a)

        # Pick the root where 0 < r < 1
        r = r1 if 0.0 < r1 < 1.0 else r2
        if not (0.0 < r < 1.0):
            r = r2 if 0.0 < r2 < 1.0 else r1

        alpha = 1.0 - r
        return max(0.0, min(alpha, 1.0))

    def render(self, input_signal: NDArray[np.float64]) -> NDArray[np.float64]:
        if len(input_signal) == 0:
            return input_signal.copy()

        alpha = self._compute_alpha()

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
