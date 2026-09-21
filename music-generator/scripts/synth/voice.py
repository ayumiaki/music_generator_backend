"""Voice engine: deterministic allocation, stealing, polyphony, modulation order."""
from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class VoiceConfig:
    """Configuration for a single voice."""
    wavetable_size: int = 4096
    waveform: str = "saw"
    attack_samples: int = 4800
    decay_samples: int = 2400
    sustain_level: float = 0.7
    release_samples: int = 4800
    filter_cutoff: float = 20000.0
    filter_resonance: float = 0.0
    retrigger: bool = True


@dataclass
class ActiveVoice:
    """An active voice instance."""
    voice_id: int
    freq: float
    amplitude: float = 1.0
    note_on: bool = True
    phase: float = 0.0
    env_phase_pos: int = 0
    filter_state: np.ndarray = field(default_factory=lambda: np.zeros(4, dtype=np.float64))
    osc_phase: float = 0.0


class VoiceEngine:
    """Deterministic voice allocator with stealing and polyphony.

    Allocation policy: lowest-ID free voice, then steal oldest if full.
    Modulation order: oscillator -> envelope -> filter.
    """

    def __init__(
        self,
        polyphony: int = 16,
        sample_rate: int = 48000,
        config: Optional[VoiceConfig] = None,
        seed: int | None = None,
    ) -> None:
        self.polyphony = polyphony
        self.sample_rate = sample_rate
        self.config = config or VoiceConfig()
        self._voices: list[ActiveVoice | None] = [None] * polyphony
        self._next_id = 0
        self._rng = np.random.RandomState(seed)
        self._alloc_order: list[int] = list(range(polyphony))

    def reset(self) -> None:
        self._voices = [None] * self.polyphony
        self._next_id = 0
        self._alloc_order = list(range(self.polyphony))

    def allocate(self, freq: float, amplitude: float = 1.0) -> int:
        """Allocate a voice for a note. Returns voice ID."""
        # Find free voice
        for i, v in enumerate(self._voices):
            if v is None:
                vid = i
                self._voices[vid] = ActiveVoice(
                    voice_id=vid,
                    freq=freq,
                    amplitude=amplitude,
                    note_on=True,
                    osc_phase=0.0,
                )
                return vid

        # Steal oldest (lowest ID in alloc order)
        steal_id = self._alloc_order.pop(0)
        self._alloc_order.append(steal_id)
        self._voices[steal_id] = ActiveVoice(
            voice_id=steal_id,
            freq=freq,
            amplitude=amplitude,
            note_on=True,
            osc_phase=0.0,
        )
        return steal_id

    def note_off(self, voice_id: int) -> None:
        """Trigger release on a voice."""
        if 0 <= voice_id < self.polyphony and self._voices[voice_id] is not None:
            self._voices[voice_id].note_on = False  # type: ignore[union-attr]

    def get_voice(self, voice_id: int) -> ActiveVoice | None:
        if 0 <= voice_id < self.polyphony:
            return self._voices[voice_id]
        return None

    def active_voices(self) -> list[ActiveVoice]:
        return [v for v in self._voices if v is not None]

    def render_block(
        self,
        notes: list[tuple[float, int, float]],
        block_size: int = 48000,
    ) -> np.ndarray:
        """Render a block of notes (freq, duration_frames, amplitude).

        Voices are shared across all notes in the block — polyphony and
        stealing are exercised. Note-off is called after each note's
        duration, and the release tail is rendered.
        """
        from .oscillator import WavetableOscillator
        from .envelope import ADSREnvelope
        from .filter import LadderFilter

        osc = WavetableOscillator(
            sample_rate=self.sample_rate,
            table_size=self.config.wavetable_size,
            waveform=self.config.waveform,
            seed=self._rng.randint(0, 2**31) if self._rng is not None else None,
        )
        env = ADSREnvelope(
            attack_samples=self.config.attack_samples,
            decay_samples=self.config.decay_samples,
            sustain_level=self.config.sustain_level,
            release_samples=self.config.release_samples,
            retrigger=self.config.retrigger,
        )
        flt = LadderFilter(
            sample_rate=self.sample_rate,
            cutoff=self.config.filter_cutoff,
            resonance=self.config.filter_resonance,
        )

        # Render each voice and mix
        mixed = np.zeros(block_size, dtype=np.float64)
        for freq, duration, amp in notes:
            if duration <= 0:
                continue
            voice_id = self.allocate(freq, amp)
            if voice_id is None:
                continue

            # Render oscillator
            osc_out = osc.render(freq, duration, phase_reset=True, reset_phase=0.0)

            # Apply envelope
            env.reset()
            env.note_on()
            env_out = env.render(duration)

            # Apply filter
            flt.reset()
            signal = osc_out * env_out * amp
            filtered = flt.render(signal[:duration])

            # Mix
            end = min(len(filtered), block_size)
            mixed[:end] += filtered[:end]

            # Note-off and release tail
            self.note_off(voice_id)
            release_samples = min(self.config.release_samples, block_size - duration)
            if release_samples > 0:
                env.reset()
                env.note_off()
                release_out = env.render(release_samples)
                # Render release tail through filter
                release_signal = np.zeros(release_samples, dtype=np.float64)
                release_filtered = flt.render(release_signal)
                release_end = min(release_samples, block_size - duration)
                mixed[duration:duration + release_end] += release_filtered[:release_end] * amp

        # Prevent clipping
        max_val = np.max(np.abs(mixed)) if len(mixed) > 0 else 1.0
        if max_val > 0.95:
            mixed = mixed / max_val * 0.95

        return mixed.astype(np.float64)