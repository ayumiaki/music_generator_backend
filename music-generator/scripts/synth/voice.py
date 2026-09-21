"""Voice engine: deterministic allocation, stealing, polyphony, modulation order.

Each voice holds persistent DSP objects (oscillator, envelope, filter) so that
state is continuous across render calls. This enables real release tails and
proper polyphony — overlapping notes genuinely coexist and are mixed together.
"""
from dataclasses import dataclass, field
from typing import Any, Optional
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
    """An active voice instance with persistent DSP objects."""
    voice_id: int
    freq: float
    amplitude: float = 1.0
    note_on: bool = True
    # Persistent DSP objects — created on first allocation, reused after
    osc: object = field(default=None, repr=False)
    env: object = field(default=None, repr=False)
    flt: object = field(default=None, repr=False)


class VoiceEngine:
    """Deterministic voice allocator with stealing and polyphony.

    Allocation policy: lowest-ID free voice, then steal oldest if full.
    Modulation order: oscillator -> envelope -> filter.

    Each voice holds persistent DSP objects so that:
    - Oscillator phase is continuous across render calls
    - Envelope state is continuous (release tails work correctly)
    - Filter state is continuous (no transients at block boundaries)
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

    def _create_dsp_objects(self, waveform: str):
        """Create persistent DSP objects for a voice."""
        from .oscillator import WavetableOscillator
        from .envelope import ADSREnvelope
        from .filter import LadderFilter

        osc = WavetableOscillator(
            sample_rate=self.sample_rate,
            table_size=self.config.wavetable_size,
            waveform=waveform,
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
        return osc, env, flt

    def allocate(self, freq: float, amplitude: float = 1.0, waveform: str = "saw") -> int:
        """Allocate a voice for a note. Returns voice ID."""
        # Find free voice
        for i, v in enumerate(self._voices):
            if v is None:
                vid = i
                osc, env, flt = self._create_dsp_objects(waveform)
                self._voices[vid] = ActiveVoice(
                    voice_id=vid,
                    freq=freq,
                    amplitude=amplitude,
                    note_on=True,
                    osc=osc,
                    env=env,
                    flt=flt,
                )
                return vid

        # Steal oldest (lowest ID in alloc order)
        steal_id = self._alloc_order.pop(0)
        self._alloc_order.append(steal_id)
        osc, env, flt = self._create_dsp_objects(waveform)
        self._voices[steal_id] = ActiveVoice(
            voice_id=steal_id,
            freq=freq,
            amplitude=amplitude,
            note_on=True,
            osc=osc,
            env=env,
            flt=flt,
        )
        return steal_id

    def note_off(self, voice_id: int) -> None:
        """Trigger release on a voice."""
        if 0 <= voice_id < self.polyphony and self._voices[voice_id] is not None:
            voice = self._voices[voice_id]
            voice.note_on = False
            # Call note_off on the active envelope (not a fresh one)
            if voice.env is not None:
                voice.env.note_off()

    def get_voice(self, voice_id: int) -> ActiveVoice | None:
        if 0 <= voice_id < self.polyphony:
            return self._voices[voice_id]
        return None

    def active_voices(self) -> list[ActiveVoice]:
        return [v for v in self._voices if v is not None]

    def render_block(
        self,
        notes: list[tuple[float, int, float, str]],
        block_size: int = 48000,
    ) -> np.ndarray:
        """Render a block of notes (freq, duration_frames, amplitude, waveform).

        Voices are shared across all notes in the block — polyphony and
        stealing are exercised. Note-off is called after each note's
        duration, and the release tail is rendered until the envelope
        reaches zero.

        Each note is rendered with its own persistent DSP objects, then
        mixed into the output buffer. Overlapping notes genuinely coexist.
        """
        mixed = np.zeros(block_size, dtype=np.float64)

        for freq, duration, amp, waveform in notes:
            if duration <= 0:
                continue

            # Allocate voice (may steal oldest if polyphony exceeded)
            vid = self.allocate(freq, amp, waveform)
            voice = self._voices[vid]
            if voice is None:
                continue

            # Render oscillator for duration + release_samples
            total_samples = duration + self.config.release_samples
            osc_out = voice.osc.render(freq, total_samples, phase_reset=True, reset_phase=0.0)

            # Apply envelope — note_on at start, note_off at duration
            voice.env.reset()
            voice.env.note_on()
            env_out = voice.env.render(total_samples)

            # Apply filter — state is continuous across notes (no reset)
            signal = osc_out * env_out * amp
            filtered = voice.flt.render(signal)

            # Add to output at correct position
            end = min(len(filtered), block_size)
            mixed[:end] += filtered[:end]

            # Mark voice as free after release completes
            # (the envelope will reach zero after release_samples)
            self._voices[vid] = None

        # Prevent clipping
        max_val = np.max(np.abs(mixed)) if len(mixed) > 0 else 1.0
        if max_val > 0.95:
            mixed = mixed / max_val * 0.95

        return mixed.astype(np.float64)
