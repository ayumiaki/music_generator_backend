"""Sample-accurate ADSR envelope with retrigger and note-off semantics."""
import numpy as np
from numpy.typing import NDArray


class ADSREnvelope:
    """Sample-accurate ADSR envelope.

    - attack, decay, sustain, release in samples
    - sustain_level: 0.0–1.0
    - retrigger: if True, restarts from attack on new note while in release
    - note_off: triggers release phase immediately
    """

    def __init__(
        self,
        attack_samples: int = 4800,
        decay_samples: int = 2400,
        sustain_level: float = 0.7,
        release_samples: int = 4800,
        retrigger: bool = True,
    ) -> None:
        self.attack_samples = attack_samples
        self.decay_samples = decay_samples
        self.sustain_level = sustain_level
        self.release_samples = release_samples
        self.retrigger = retrigger
        self._phase = "idle"  # idle, attack, decay, sustain, release
        self._phase_pos = 0
        self._sustain_value = 0.0

    def reset(self) -> None:
        """Reset to idle."""
        self._phase = "idle"
        self._phase_pos = 0
        self._sustain_value = 0.0

    def note_on(self) -> None:
        """Trigger note on — start attack or retrigger."""
        if self._phase == "release" and not self.retrigger:
            # Continue release to completion
            return
        self._phase = "attack"
        self._phase_pos = 0

    def note_off(self) -> None:
        """Trigger note off — start release from current level."""
        if self._phase == "idle":
            return
        # Capture current sustain value as release start
        self._sustain_value = self._current_level()
        self._phase = "release"
        self._phase_pos = 0

    def _current_level(self) -> float:
        """Compute current envelope level based on phase position."""
        if self._phase == "idle":
            return 0.0
        if self._phase == "attack":
            if self.attack_samples <= 0:
                return 1.0
            return min(1.0, self._phase_pos / self.attack_samples)
        if self._phase == "decay":
            if self.decay_samples <= 0:
                return self.sustain_level
            t = self._phase_pos / self.decay_samples
            return 1.0 - (1.0 - self.sustain_level) * t
        if self._phase == "sustain":
            return self.sustain_level
        if self._phase == "release":
            if self.release_samples <= 0:
                return 0.0
            t = self._phase_pos / self.release_samples
            return self._sustain_value * (1.0 - t)
        return 0.0

    def render(self, n_frames: int) -> NDArray[np.float64]:
        """Render n_frames of envelope, advancing internal state."""
        if n_frames <= 0:
            return np.zeros(n_frames, dtype=np.float64)

        output = np.zeros(n_frames, dtype=np.float64)
        remaining = n_frames
        out_pos = 0

        while remaining > 0:
            if self._phase == "idle":
                # Stay at zero until note_on
                output[out_pos : out_pos + remaining] = 0.0
                break

            if self._phase == "attack":
                avail = self.attack_samples - self._phase_pos
                take = min(remaining, avail) if avail > 0 else remaining
                if avail <= 0:
                    # Attack done, move to decay
                    self._phase = "decay"
                    self._phase_pos = 0
                    continue
                t = np.arange(self._phase_pos, self._phase_pos + take) / self.attack_samples
                output[out_pos : out_pos + take] = np.clip(t, 0.0, 1.0)
                self._phase_pos += take
                out_pos += take
                remaining -= take
                if self._phase_pos >= self.attack_samples:
                    self._phase = "decay"
                    self._phase_pos = 0

            elif self._phase == "decay":
                avail = self.decay_samples - self._phase_pos
                take = min(remaining, avail) if avail > 0 else remaining
                if avail <= 0:
                    self._phase = "sustain"
                    self._phase_pos = 0
                    continue
                t = np.arange(self._phase_pos, self._phase_pos + take) / self.decay_samples
                level = 1.0 - (1.0 - self.sustain_level) * t
                output[out_pos : out_pos + take] = level
                self._phase_pos += take
                out_pos += take
                remaining -= take
                if self._phase_pos >= self.decay_samples:
                    self._phase = "sustain"
                    self._phase_pos = 0

            elif self._phase == "sustain":
                output[out_pos : out_pos + remaining] = self.sustain_level
                self._phase_pos += remaining
                out_pos += remaining
                remaining = 0

            elif self._phase == "release":
                avail = self.release_samples - self._phase_pos
                take = min(remaining, avail) if avail > 0 else remaining
                if avail <= 0:
                    output[out_pos : out_pos + remaining] = 0.0
                    self._phase = "idle"
                    self._phase_pos = 0
                    break
                t = np.arange(self._phase_pos, self._phase_pos + take) / self.release_samples
                level = self._sustain_value * (1.0 - t)
                output[out_pos : out_pos + take] = level
                self._phase_pos += take
                out_pos += take
                remaining -= take
                if self._phase_pos >= self.release_samples:
                    self._phase = "idle"
                    self._phase_pos = 0

        return output.astype(np.float64)

    def get_state(self) -> dict:
        return {
            "phase": self._phase,
            "phase_pos": self._phase_pos,
            "sustain_value": self._sustain_value,
        }