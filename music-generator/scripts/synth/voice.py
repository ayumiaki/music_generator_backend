"""Voice engine: deterministic allocation, stealing, polyphony, modulation order.

Each voice holds persistent DSP objects (oscillator, envelope, filter).
Rendering is event-driven and chronological: note-on and note-off events
are processed at exact sample indices, and every block is rendered through
the currently allocated voices' persistent DSP chains (oscillator ->
envelope -> filter), mixed together. Overlapping notes genuinely coexist;
stolen voices genuinely disappear; released voices ring out through their
release tails.
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
    retrigger: bool = True


@dataclass
class ActiveVoice:
    """An active voice instance with persistent DSP objects.

    DSP objects (oscillator, envelope, filter) are created when the voice
    is allocated for a note and persist for that note's entire lifetime —
    held phase and release tail — so state is continuous across every
    render segment of the note.
    """
    voice_id: int
    freq: float
    amplitude: float = 1.0
    note_on: bool = True
    osc: object = field(default=None, repr=False)
    env: object = field(default=None, repr=False)
    flt: object = field(default=None, repr=False)


class VoiceEngine:
    """Deterministic voice allocator with stealing and polyphony.

    Allocation policy: lowest-ID free voice, then steal the oldest active
    voice (fewest samples rendered since allocation). Modulation order:
    oscillator -> envelope -> filter, per voice, mixed additively.

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
        # Allocation age: voice_id -> (samples rendered since allocation,
        # allocation sequence number). Stealing picks the voice with the
        # most rendered samples; ties break to the earliest allocation.
        self._age: dict[int, tuple[int, int]] = {}
        self._alloc_counter: int = 0

    def reset(self) -> None:
        self._voices = [None] * self.polyphony
        self._next_id = 0
        self._age = {}
        self._alloc_counter = 0

    def _create_dsp_objects(self, waveform: str):
        """Create persistent DSP objects for a voice."""
        from .oscillator import WavetableOscillator
        from .envelope import ADSREnvelope
        from .filter import CascadeFilter

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
        flt = CascadeFilter(
            sample_rate=self.sample_rate,
            cutoff=self.config.filter_cutoff,
        )
        return osc, env, flt

    def allocate(self, freq: float, amplitude: float = 1.0, waveform: str = "saw") -> int:
        """Allocate a voice for a note. Returns voice ID.

        Steals the oldest active voice (most samples rendered since its
        allocation) when all voices are occupied.
        """
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
                self._age[vid] = (0, self._alloc_counter)
                self._alloc_counter += 1
                return vid

        # Steal the oldest active voice: most rendered samples,
        # ties broken by earliest allocation sequence
        steal_id = max(self._age, key=lambda v: (self._age[v][0], -self._age[v][1]))
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
        self._age[steal_id] = (0, self._alloc_counter)
        self._alloc_counter += 1
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

    def _free_voice(self, voice_id: int) -> None:
        """Free a voice slot and forget its age."""
        if 0 <= voice_id < self.polyphony:
            self._voices[voice_id] = None
            self._age.pop(voice_id, None)

    def render_schedule(
        self,
        note_events: list[tuple[int, float, int, float, str]],
        total_samples: int,
    ) -> np.ndarray:
        """Render a chronological note schedule to a PCM buffer.

        note_events: list of (start_sample, freq, duration_samples,
        amplitude, waveform), processed at exact sample indices.

        The timeline is walked chronologically. Note-on events allocate
        voices (stealing the oldest when full); note-off events fire at
        exactly start + duration; every inter-event segment is rendered
        through the persistent DSP chains of all currently allocated
        voices and mixed. Voices are freed once their release tails
        complete. The output is exactly total_samples long.
        """
        # Sort events by start sample (stable, so equal starts keep order)
        events = sorted(note_events, key=lambda e: e[0])
        output = np.zeros(total_samples, dtype=np.float64)

        # Active note bookkeeping: voice_id -> note-off sample index
        # (None while the note is still held)
        held: dict[int, Optional[int]] = {}
        # Voice IDs whose note-off has fired but release is still ringing
        releasing: set[int] = set()

        pos = 0
        ei = 0
        n_events = len(events)

        while pos < total_samples:
            # 1. Fire note-offs due at or before pos
            for vid in list(held.keys()):
                off_at = held[vid]
                if off_at is not None and off_at <= pos:
                    self.note_off(vid)
                    del held[vid]
                    releasing.add(vid)

            # 2. Free voices whose release has completed
            for vid in list(releasing):
                voice = self._voices[vid]
                if voice is not None and voice.env is not None and voice.env._phase == "idle":
                    self._free_voice(vid)
                    releasing.discard(vid)

            # 3. Fire note-ons at pos
            while ei < n_events and events[ei][0] <= pos:
                start, freq, dur, amp, wf = events[ei]
                if start < total_samples and dur > 0:
                    vid = self.allocate(freq, amp, wf)
                    # If this slot was stolen mid-release, its old note is
                    # dead — drop it from the releasing set so it is not
                    # rendered twice
                    releasing.discard(vid)
                    voice = self._voices[vid]
                    if voice is not None and voice.env is not None:
                        voice.env.reset()
                        voice.env.note_on()
                    # If this voice was already active (stolen mid-flight),
                    # its old note bookkeeping is replaced
                    held[vid] = start + dur
                ei += 1

            # 4. Determine the next event position
            next_pos = total_samples
            if ei < n_events:
                next_pos = min(next_pos, max(events[ei][0], pos))
            for vid, off_at in held.items():
                if off_at is not None:
                    next_pos = min(next_pos, off_at)

            # 5. Render the segment [pos, next_pos) through active voices
            seg_len = min(next_pos - pos, total_samples - pos)
            if seg_len < 0:
                seg_len = 0
            if seg_len > 0:
                active_ids = [vid for vid in held.keys()] + [
                    vid for vid in releasing
                    if self._voices[vid] is not None
                ]
                if active_ids:
                    block = np.zeros(seg_len, dtype=np.float64)
                    for vid in active_ids:
                        voice = self._voices[vid]
                        if voice is None or voice.osc is None:
                            continue
                        osc_out = voice.osc.render(voice.freq, seg_len)
                        env_out = voice.env.render(seg_len)
                        sig = osc_out * env_out * voice.amplitude
                        block += voice.flt.render(sig)
                        age, seq = self._age.get(vid, (0, 0))
                        self._age[vid] = (age + seg_len, seq)
                    output[pos : pos + seg_len] += block

            if next_pos <= pos:
                # No further events: render remaining voices to the end
                break
            pos = next_pos

        # Free everything at the end of the schedule
        for vid in list(held.keys()) + list(releasing):
            self._free_voice(vid)

        return output

    def render_block(
        self,
        notes: list[tuple[float, int, float, str]],
        block_size: int = 48000,
    ) -> np.ndarray:
        """Legacy sequential rendering — notes start at sample 0, one after another.

        Kept for backward compatibility with simple monophonic use.
        New code should use render_schedule() for real event timing.
        """
        events = []
        pos = 0
        for freq, duration, amp, waveform in notes:
            if duration <= 0:
                continue
            events.append((pos, freq, duration, amp, waveform))
            pos += duration
        return self.render_schedule(events, max(block_size, pos))

    def render(
        self, n_frames: int = 48000) -> np.ndarray:
        """Render silence (no notes scheduled). Kept for API compatibility."""
        return np.zeros(n_frames, dtype=np.float64)
