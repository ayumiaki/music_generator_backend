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
    generate_progression,
    note_to_midi,
    scale_degrees,
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

        # Mode: mood decides unless it's major, then config takes priority.
        # This lets moods like "sad" force minor mode while "happy" defers to config.
        self.effective_mode = self.mood.mode if self.mood.mode != "major" else config.mode

    def generate(self) -> Score:
        """Generate the full Score."""
        template = SECTION_TEMPLATES[self.config.structure]
        bpm = self.rng.randint(self.mood.tempo_low, self.mood.tempo_high + 1)

        score = Score(
            key_root=self.config.key,
            mode=self.effective_mode,
            bpm=bpm,
            seed=self.config.seed,
        )

        beat_cursor = 0.0
        saved_a_state = None  # (progression, final_voicing) for A' mirror
        continuous_voicing: Optional[Voicing] = None

        for sec_idx, (sec_name, sec_bars) in enumerate(template):
            sec_beats = sec_bars * self.config.beats_per_bar

            # Determine chord progression for this section
            if sec_name == "A'" and saved_a_state is not None:
                prog, final_v = saved_a_state
                # For A', we reuse the progression but also start from
                # the final voicing of the original A section
                continuous_voicing_for_section = final_v
            else:
                continuous_voicing_for_section = continuous_voicing
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
                    self.effective_mode,
                    n_chords,
                    self.rng,
                    start_degree=0,
                    end_degree=end_degree,
                )

                # Save A progression + initial voicing for A' mirror
                base_name = sec_name.rstrip("'")
                if base_name == "A":
                    saved_a_state = (prog, None)  # will fill voicing after voicing

            # Voice the progression with continuous state
            voicings = voice_progression(
                prog,
                tension=self.mood.tension,
                register=self.mood.register,
                initial_voicing=continuous_voicing_for_section,
            )

            # Update continuous voicing to the last voicing of this section
            if voicings:
                continuous_voicing = voicings[-1]

            # If this is A, save the final voicing state for A'
            base_name = sec_name.rstrip("'")
            if base_name == "A":
                if saved_a_state is not None:
                    saved_a_state = (saved_a_state[0], continuous_voicing)

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

                # Bass note
                section.notes.append(ScoreNote(
                    time_beats=chord_start,
                    duration_beats=chord_beats * self.mood.articulation,
                    pitch_midi=voicing.bass,
                    amplitude=0.8,
                    voice=VOICE_BASS,
                ))

                # Harmony
                for j, hp in enumerate(voicings[i].harmony):
                    section.notes.append(ScoreNote(
                        time_beats=chord_start,
                        duration_beats=chord_beats * self.mood.articulation,
                        pitch_midi=hp,
                        amplitude=0.6,
                        voice=VOICE_HARMONY,
                    ))

                # Melody with contour
                phrase_start = (i == 0)
                phrase_end = (i == len(prog) - 1)
                self._add_melody_notes(
                    section,
                    chord_start,
                    chord_beats,
                    voicing.melody,
                    root,
                    quality,
                    phrase_start=phrase_start,
                    phrase_end=phrase_end,
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
        chord_root: int,
        chord_quality: str,
        phrase_start: bool = False,
        phrase_end: bool = False,
    ) -> None:
        """Add melody notes for one chord's duration with contour.

        Melody uses scale degrees near the chord tone, with stepwise motion
        and occasional leaps. Density controls note count.
        phrase_start/phrase_end bias toward stable tones for musical phrasing.
        """
        if chord_beats <= 0:
            return

        density = self.mood.density
        # Number of melody notes per chord based on density
        if density < 0.15:
            n_notes = 1
        elif density < 0.4:
            n_notes = 2
        elif density < 0.7:
            n_notes = 4
        else:
            n_notes = 8

        sub_beat = chord_beats / n_notes

        # Generate contour with phrase awareness
        contour = self._melody_contour(n_notes, chord_root, chord_quality,
                                       phrase_start=phrase_start, phrase_end=phrase_end)

        for i in range(n_notes):
            t = chord_start + i * sub_beat
            dur = sub_beat * self.mood.articulation
            pitch = contour[i]
            # Ensure melody pitch is in valid MIDI range
            pitch = max(0, min(127, pitch))
            section.notes.append(ScoreNote(
                time_beats=t,
                duration_beats=dur,
                pitch_midi=pitch,
                amplitude=0.7,
                voice=VOICE_MELODY,
            ))

    def _melody_contour(self, n_notes: int, chord_root: int, chord_quality: str,
                        phrase_start: bool = False, phrase_end: bool = False) -> list[int]:
        """Generate a melodic contour with diverse pitch content and musical shape.

        Uses a wide scale pool for maximum pitch diversity, with starting
        position offset by degree (transposition-consistent) to vary between
        chords. Ensures same seed + same degree → same contour shape.
        """
        from .harmony import CHORD_QUALITIES
        from .voice_leading import VOICE_RANGES
        melody_range = VOICE_RANGES["melody"]
        chord_intervals = CHORD_QUALITIES.get(chord_quality, [0, 4, 7])
        degree_offset = (chord_root - self.root_midi) % 12

        # Build a very wide scale pool for maximum pitch diversity
        scale = scale_degrees(self.root_midi, self.effective_mode)
        scale_pcs = set(s % 12 for s in scale)
        scale_pool = []
        for oct_shift in range(-5, 6):
            for s in scale:
                p = s + oct_shift * 12
                if melody_range.min_midi <= p <= melody_range.max_midi:
                    scale_pool.append(p)
        scale_pool.sort()

        if n_notes <= 0:
            return []

        # Starting position: offset by degree (transposition-consistent)
        # Different chords start from different positions for variety
        base_mid = (melody_range.preferred_low + melody_range.preferred_high) // 2
        range_span = melody_range.preferred_high - melody_range.preferred_low
        start_offset = (degree_offset % 5) * 4 - range_span // 2
        mid = base_mid + start_offset
        mid = max(melody_range.min_midi + 2, min(melody_range.max_midi - 2, mid))

        if phrase_end:
            start_candidates = [p for p in scale_pool if p % 12 == self.root_midi % 12]
        else:
            start_candidates = [p for p in scale_pool if p % 12 in set((chord_root + iv) % 12 for iv in chord_intervals)]
        if not start_candidates:
            start_candidates = scale_pool
        current = min(start_candidates, key=lambda p: abs(p - mid))

        contour = [current]

        for i in range(1, n_notes):
            r = self.rng.random()
            prev = contour[-1]

            # Phrase-end: bias toward tonic in last 2 notes
            if phrase_end and i >= n_notes - 2:
                tonic_pcs = self.root_midi % 12
                target = min(scale_pool, key=lambda p: (
                    0 if p % 12 == tonic_pcs else 1,
                    abs(p - prev)
                ))
                pitch = target
            elif r < 0.05:
                # Repeat note
                pitch = prev
            elif r < 0.55:
                # Stepwise: continue direction with momentum
                direction = 1 if prev < base_mid else -1
                step = direction * int(self.rng.choice([1, 1, 2]))
                pitch = prev + step
            elif r < 0.80:
                # Chord-tone leap: pick a chord tone 3-12 semitones away
                jitter = int(self.rng.randint(0, 3))
                ct_options = [p for p in scale_pool if p % 12 in set((chord_root + iv) % 12 for iv in chord_intervals) and 3 <= abs(p - prev) <= 12]
                if ct_options:
                    pitch = min(ct_options, key=lambda p, j=jitter: abs(p - prev) + j)
                else:
                    pitch = prev + int(self.rng.choice([-5, -3, 3, 5]))
            else:
                # Large leap: 4th, 5th, octave
                leap = int(self.rng.choice([-12, -8, -7, -5, 5, 7, 8, 12]))
                pitch = prev + leap

            # Quantize to scale and clamp
            pitch = self._quantize_to_scale(pitch, sorted(scale_pcs))
            pitch = max(melody_range.min_midi, min(melody_range.max_midi, pitch))
            contour.append(pitch)

        return contour

    def _quantize_to_scale(self, pitch: int, scale_pcs: list[int]) -> int:
        """Quantize a pitch to the nearest scale degree."""
        pc = pitch % 12
        if pc in scale_pcs:
            return pitch
        # Find nearest scale degree
        min_dist = 12
        best_pc = pc
        for spc in scale_pcs:
            dist = min((pc - spc) % 12, (spc - pc) % 12)
            if dist < min_dist:
                min_dist = dist
                best_pc = spc
        octave = pitch // 12
        return octave * 12 + best_pc


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
