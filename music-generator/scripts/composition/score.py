"""Score — the symbolic intermediate representation between composition and synthesis.

A Score is a fully determined musical idea: notes, chords, sections, and
voice roles expressed as data. It contains no audio logic and no DSP. The
synthesis layer consumes a Score and renders it; the composition layer
produces a Score from high-level intent.

Every note carries:
- time_beats / duration_beats: when it fires and how long it holds
- pitch_midi: MIDI note number (60 = middle C); None for percussion
- amplitude: 0..1 velocity-scaled level
- voice: "bass" | "harmony" | "melody" | "percussion"

Every chord carries:
- time_beats / duration_beats
- root_midi / quality: what it sounds like
- label: functional notation (I, ii7, V, viio, ...)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


VOICE_BASS = "bass"
VOICE_HARMONY = "harmony"
VOICE_MELODY = "melody"
VOICE_PERCUSSION = "percussion"

VOICE_ROLES = (VOICE_BASS, VOICE_HARMONY, VOICE_MELODY, VOICE_PERCUSSION)


@dataclass(frozen=True)
class ScoreNote:
    """A single note event in the symbolic score."""
    time_beats: float
    duration_beats: float
    pitch_midi: Optional[int]  # None for percussion
    amplitude: float = 1.0
    voice: str = VOICE_MELODY

    def __post_init__(self) -> None:
        if self.duration_beats <= 0:
            raise ValueError(f"duration_beats must be > 0, got {self.duration_beats}")
        if not 0.0 <= self.amplitude <= 1.0:
            raise ValueError(f"amplitude must be in [0,1], got {self.amplitude}")
        if self.voice not in VOICE_ROLES:
            raise ValueError(f"unknown voice {self.voice!r}")
        if self.pitch_midi is not None and not (0 <= self.pitch_midi <= 127):
            raise ValueError(f"pitch_midi must be in [0,127], got {self.pitch_midi}")


@dataclass(frozen=True)
class ScoreChord:
    """A chord label in the symbolic score — informational and verifiable."""
    time_beats: float
    duration_beats: float
    root_midi: int
    quality: str = "maj"
    label: str = ""
    extensions: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.duration_beats <= 0:
            raise ValueError(f"duration_beats must be > 0, got {self.duration_beats}")
        if not (0 <= self.root_midi <= 127):
            raise ValueError(f"root_midi must be in [0,127], got {self.root_midi}")


@dataclass
class Section:
    """A named span of the song (intro, A, B, outro, ...)."""
    name: str
    start_beat: float
    duration_beats: float
    notes: list[ScoreNote] = field(default_factory=list)
    chords: list[ScoreChord] = field(default_factory=list)

    @property
    def end_beat(self) -> float:
        return self.start_beat + self.duration_beats


@dataclass
class Score:
    """A fully determined, renderable musical idea."""
    key_root: str = "C"
    mode: str = "major"
    bpm: float = 120.0
    total_beats: float = 0.0
    sections: list[Section] = field(default_factory=list)
    seed: Optional[int] = None

    def all_notes(self) -> list[ScoreNote]:
        return [n for s in self.sections for n in s.notes]

    def all_chords(self) -> list[ScoreChord]:
        return [c for s in self.sections for c in s.chords]


def midi_to_freq(midi: int) -> float:
    """Convert MIDI note number to frequency in Hz (A4 = 69 = 440 Hz)."""
    return 440.0 * (2.0 ** ((midi - 69) / 12.0))
