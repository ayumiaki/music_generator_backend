"""Composition — turn high-level intent into a fully determined Score.

Pipeline:
    1. Resolve mood → MoodProfile
    2. Generate chord progression (functional grammar)
    3. Voice the progression (bass/harmony/melody)
    4. Generate melody rhythm & contour
    5. Build song structure (intro/A/B/A'/outro)
    6. Assemble the Score

Everything is deterministic given a seed. Two calls with the same
CompositionConfig + seed produce byte-identical Score JSON.
"""
from __future__ import annotations

import json
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from .harmony import (
    NOTE_NAME_TO_MIDI,
    VALID_MODES,
    diatonic_triad,
    generate_progression,
    note_to_midi,
)
from .mood import MoodProfile, get_mood
from .score import (
    VOICE_BASS,
    VOICE_HARMONY,
    VOICE_MELODY,
    Score,
    ScoreChord,
    ScoreNote,
    Section,
)
from .voice_leading import (
    VOICE_RANGES,
    VoiceRange,
    Voicing,
    voice_progression,
)


# Valid section structures (section name -> bars per section)
SECTION_TEMPLATES = {
    "simple": [("intro", 2), ("A", 8), ("outro", 2)],
    "standard": [("intro", 2), ("A", 8), ("B", 8), ("A'", 8), ("outro", 2)],
    "ABA": [("A", 8), ("B", 8), ("A'", 8)],
    "through": [("A", 16)],
    "long": [("intro", 4), ("A", 8), ("B", 8), ("A'", 8), ("B'", 8), ("outro", 4)],
}


@dataclass
class CompositionConfig:
    """High-level intent for a composition."""
    key: str = "C"
    mode: str = "major"
    mood: str = "happy"
    seed: int = 42
    structure: str = "standard"  # key into SECTION_TEMPLATES
    beats_per_bar: int = 4

    def __post_init__(self) -> None:
        if self.key not in NOTE_NAME_TO_MIDI and self.key not in (
            "Db", "Eb", "Gb", "Ab", "Bb", "E#", "B#", "Cb", "Fb",
        ):
            raise ValueError(f"unknown key {self.key!r}")
        if self.mode not in VALID_MODES:
            raise ValueError(f"unknown mode {self.mode!r}")
        if self.structure not in SECTION_TEMPLATES:
            raise ValueError(
                f"unknown structure {self.structure!r}; valid: {list(SECTION_TEMPLATES.keys())}"
            )


class Composition:
    """Generates a Score from a CompositionConfig."""

    def __init__(self, config: CompositionConfig) -> None:
        self.config = config
        self.mood: MoodProfile = get_mood(config.mood)
        self.rng = np.random.RandomState(config.seed)
        self.root_midi = note_to_midi(config.key, 4)  # middle octave

    def generate(self) -> Score:
        """Generate the full Score."""
        template = SECTION_TEMPLATES[self.config.structure]
        bpm = self.rng.randint(self.mood.tempo_low, self.mood.tempo_high + 1)

        score = Score(
            key_root=self.config.key,
            mode=self.mood.mode if self.mood.mode != "major" else self.config.mode,
            bpm=bpm,
            seed=self.config.seed,
        )

        beat_cursor = 0.0
        saved_a_progression = None  # cache A progression for A' mirror

        for sec_idx, (sec_name, sec_bars) in enumerate(template):
            sec_beats = sec_bars * self.config.beats_per_bar

            # Determine chord progression for this section
            if sec_name == "A'" and saved_a_progression is not None:
                # A' mirrors A exactly
                prog = saved_a_progression
            else:
                n_chords = sec_bars  # one chord per bar
                if sec_name.startswith("B"):
                    n_chords = max(4, sec_bars)

                # End degree: tonic for outro/A', V for B, tonic for intro
                if sec_name in ("outro", "A'"):
                    end_degree = 0
                elif sec_name == "intro":
                    end_degree = 0
                elif sec_name == "B":
                    end_degree = 4  # V
                else:
                    end_degree = 4  # A section ends on V to lead back

                prog = generate_progression(
                    self.root_midi,
                    self.mood.mode,
                    n_chords,
                    self.rng,
                    start_degree=0,
                    end_degree=end_degree,
                )

                # Save A progression for A'
                base_name = sec_name.rstrip("'")
                if base_name == "A":
                    saved_a_progression = prog

            # Voice the progression
            voicings = voice_progression(prog)

            # Build section notes and chords
            section = Section(
                name=sec_name,
                start_beat=beat_cursor,
                duration_beats=sec_beats,
            )

            beat_in_section = 0.0
            for i, ((root, quality, label), voicing) in enumerate(zip(prog, voicings)):
                chord_beats = sec_beats / max(len(prog), 1)
                chord_start = beat_cursor + beat_in_section

                # Add chord to score
                section.chords.append(ScoreChord(
                    time_beats=chord_start,
                    duration_beats=chord_beats,
                    root_midi=root,
                    quality=quality,
                    label=label,
                ))

                # Bass note: on the chord, whole-note feel
                section.notes.append(ScoreNote(
                    time_beats=chord_start,
                    duration_beats=chord_beats * self.mood.articulation,
                    pitch_midi=voicing.bass,
                    amplitude=0.8,
                    voice=VOICE_BASS,
                ))

                # Harmony: on the chord, sustained
                for j, hp in enumerate(voicings[i].harmony):
                    section.notes.append(ScoreNote(
                        time_beats=chord_start,
                        duration_beats=chord_beats * self.mood.articulation,
                        pitch_midi=hp,
                        amplitude=0.6,
                        voice=VOICE_HARMONY,
                    ))

                # Melody: rhythm derived from density + mood
                self._add_melody_notes(
                    section,
                    chord_start,
                    chord_beats,
                    voicing.melody,
                )

                beat_in_section += chord_beats

            score.sections.append(section)
            beat_cursor += sec_beats

        score.total_beats = beat_cursor
        return score

    def _add_melody_notes(
        self,
        section: Section,
        chord_start: float,
        chord_beats: float,
        melody_pitch: int,
    ) -> None:
        """Add melody notes for one chord's duration, based on mood density."""
        if chord_beats <= 0:
            return

        density = self.mood.density
        if density < 0.15:
            # Sparse: one long note per chord
            section.notes.append(ScoreNote(
                time_beats=chord_start,
                duration_beats=chord_beats * self.mood.articulation,
                pitch_midi=melody_pitch,
                amplitude=0.75,
                voice=VOICE_MELODY,
            ))
            return

        if density < 0.4:
            # Two notes per chord
            subdivisions = 2
        elif density < 0.7:
            # Four notes per chord
            subdivisions = 4
        else:
            # Eight notes per chord
            subdivisions = 8

        sub_beat = chord_beats / subdivisions
        for i in range(subdivisions):
            t = chord_start + i * sub_beat
            # Articulation: shorter notes for higher density
            dur = sub_beat * self.mood.articulation
            section.notes.append(ScoreNote(
                time_beats=t,
                duration_beats=dur,
                pitch_midi=melody_pitch,
                amplitude=0.7,
                voice=VOICE_MELODY,
            ))




def score_to_dict(score: Score) -> dict:
    """Serialise a Score to a JSON-compatible dict (for hashing/comparison)."""
    return {
        "key_root": score.key_root,
        "mode": score.mode,
        "bpm": score.bpm,
        "seed": score.seed,
        "total_beats": score.total_beats,
        "sections": [
            {
                "name": s.name,
                "start_beat": s.start_beat,
                "duration_beats": s.duration_beats,
                "notes": [
                    {
                        "time_beats": n.time_beats,
                        "duration_beats": n.duration_beats,
                        "pitch_midi": n.pitch_midi,
                        "amplitude": n.amplitude,
                        "voice": n.voice,
                    }
                    for n in s.notes
                ],
                "chords": [
                    {
                        "time_beats": c.time_beats,
                        "duration_beats": c.duration_beats,
                        "root_midi": c.root_midi,
                        "quality": c.quality,
                        "label": c.label,
                    }
                    for c in s.chords
                ],
            }
            for s in score.sections
        ],
    }


def score_to_json(score: Score) -> str:
    """Serialise a Score to a JSON string."""
    return json.dumps(score_to_dict(score), sort_keys=True, indent=2)
