"""Stable 4-pole ladder filter — no NaNs, no runaway gain."""
import numpy as np
from numpy.typing import NDArray


class LadderFilter:
    """4-pole ladder lowpass filter with stable resonance handling.

    Based on the Moog-style ladder topology with anti-aliasing clamp.
    cutoff: 20–22050 Hz (for 48kHz sample rate)
    resonance: 0.0–1.0 (1.0 = self-oscillation at cutoff)
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
        # 4 filter stages
        self._state = np.zeros(4, dtype=np.float64)
        self._prev_output = 0.0

    def reset(self) -> None:
        self._state[:] = 0.0
        self._prev_output = 0.0

    def set_cutoff(self, cutoff: float) -> None:
        self.cutoff = max(1.0, min(cutoff, self.sample_rate / 2.0))

    def set_resonance(self, resonance: float) -> None:
        self.resonance = max(0.0, min(resonance, 1.0))

    def _calc_coeff(self) -> float:
        """Calculate filter coefficient from cutoff."""
        # Approximate beta = 1.0 - exp(-2*pi*fc/fs)
        omega = 2.0 * np.pi * self.cutoff / self.sample_rate
        beta = 1.0 - np.exp(-omega)
        return beta

    def render(self, input_signal: NDArray[np.float64]) -> NDArray[np.float64]:
        """Process input_signal through the ladder filter."""
        if len(input_signal) == 0:
            return input_signal.copy()

        beta = self._calc_coeff()
        res_gain = self.resonance * 3.0  # Scale resonance for 4-pole (reduced from 4.0)

        output = np.zeros(len(input_signal), dtype=np.float64)
        s = self._state.copy()

        for i in range(len(input_signal)):
            x = float(input_signal[i])
            # 4 cascaded stages with feedback
            for stage in range(4):
                # Apply resonance feedback from last stage output
                feedback = res_gain * s[3] if stage == 3 else 0.0
                s[stage] += beta * (x + feedback - s[stage])
                x = s[stage]

            # Soft clamp to prevent runaway (tanh limiter)
            if abs(s[3]) > 1.0:
                s[3] = np.tanh(s[3])
                # Propagate clamp back through stages
                for stage in range(2, -1, -1):
                    s[stage] = s[stage + 1]

            # Final output with hard clip at ±1
            out = s[3]
            if out > 1.0:
                out = 1.0
            elif out < -1.0:
                out = -1.0

            output[i] = out

        self._state = s
        self._prev_output = output[-1] if len(output) > 0 else 0.0

        return output.astype(np.float64)

    def get_state(self) -> NDArray[np.float64]:
        return self._state.copy()