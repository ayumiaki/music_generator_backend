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
    detect_parallel_fifths_octaves,
    voice_chord,
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
        saved_a_state = None  # (progression, final_voicing, melody_notes) for A' mirror
        continuous_voicing: Optional[Voicing] = None
        section_final_voicings: dict[str, Voicing] = {}  # track final voicing per section

        for sec_idx, (sec_name, sec_bars) in enumerate(template):
            sec_beats = sec_bars * self.config.beats_per_bar

            # Determine chord progression for this section
            if sec_name == "A'" and saved_a_state is not None:
                prog, final_v, saved_melody = saved_a_state
                # For A': chronological continuity — use the immediately preceding
                # section's final voicing (B's), not A's. The progression and motif
                # come from A, but voice leading continues from where we left off.
                prev_section_name = template[sec_idx - 1][0] if sec_idx > 0 else None
                continuous_voicing_for_section = section_final_voicings.get(prev_section_name) if prev_section_name else continuous_voicing
                use_saved_melody = True
            else:
                continuous_voicing_for_section = continuous_voicing
                use_saved_melody = False
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
                if sec_name == "A":
                    saved_a_state = (prog, None, None)  # will fill voicing+melody after voicing

            # Section boundary: pre-compute the first chord's voicing with
            # cross-section parallel avoidance, then pass it as first_voicing
            # so voice_progression uses it directly (no re-voicing).
            boundary_first_voicing = None
            if continuous_voicing_for_section is not None and prog:
                prev_v = continuous_voicing_for_section
                first_root, first_quality, first_label = prog[0]
                first_v = voice_chord(
                    first_root, first_quality, first_label, prev_v,
                    tension=self.mood.tension, register=self.mood.register,
                )
                prev_all = [prev_v.bass, prev_v.melody] + prev_v.harmony
                curr_all = [first_v.bass, first_v.melody] + first_v.harmony
                f, o = detect_parallel_fifths_octaves(prev_all, curr_all)
                if f + o > 0:
                    # Try shifting bass by octave to break parallels
                    bass_vr = VOICE_RANGES["bass"]
                    for bass_shift in [12, -12, 24, -24]:
                        new_bass = first_v.bass + bass_shift
                        if (new_bass >= bass_vr.min_midi and
                                new_bass <= bass_vr.max_midi and
                                (not first_v.harmony or new_bass < min(first_v.harmony))):
                            shifted_v = Voicing(
                                bass=new_bass,
                                harmony=first_v.harmony,
                                melody=first_v.melody,
                                chord_root=first_v.chord_root,
                                chord_quality=first_v.chord_quality,
                                chord_label=first_v.chord_label,
                            )
                            f2, o2 = detect_parallel_fifths_octaves(
                                prev_all, [shifted_v.bass, shifted_v.melody] + shifted_v.harmony
                            )
                            if f2 + o2 < f + o:
                                first_v = shifted_v
                                break
                boundary_first_voicing = first_v

            # Voice the progression with continuous state.
            # first_voicing: pre-computed boundary-aware first chord (used as-is).
            # initial_voicing: passed to _fixup_parallels as boundary_prev for
            # cross-section parallel checking.
            voicings = voice_progression(
                prog,
                tension=self.mood.tension,
                register=self.mood.register,
                initial_voicing=continuous_voicing_for_section,
                first_voicing=boundary_first_voicing,
            )

            # Update continuous voicing to the last voicing of this section
            if voicings:
                continuous_voicing = voicings[-1]
                section_final_voicings[sec_name] = voicings[-1]

            # If this is A (original only, not A'), save the final voicing state for A'
            if sec_name == "A" and saved_a_state is not None:
                saved_a_state = (saved_a_state[0], continuous_voicing, saved_a_state[2])

            # Build section notes and chords
            section = Section(
                name=sec_name,
                start_beat=beat_cursor,
                duration_beats=sec_beats,
            )

            # Collect A's melody for reuse in A'
            section_melody_notes = []

            beat_in_section = 0.0
            for i, ((root, quality, label), voicing) in enumerate(zip(prog, voicings)):
                chord_beats = sec_beats / max(len(prog), 1)
                chord_start = beat_cursor + beat_in_section

                # Detect tension-driven extensions (e.g., 7th added by tension)
                from .harmony import CHORD_QUALITIES
                base_intervals = set(CHORD_QUALITIES.get(quality, [0, 4, 7]))
                extensions_detected = []
                if voicing.harmony:
                    for hp in voicing.harmony:
                        interval_from_root = (hp - root) % 12
                        if interval_from_root not in base_intervals:
                            extensions_detected.append(interval_from_root)
                extensions_tuple = tuple(sorted(set(extensions_detected)))

                # Upgrade quality to reflect 7th if present
                effective_quality = quality
                if 11 in extensions_tuple and quality in ("maj", "aug"):
                    effective_quality = "maj7"
                elif 10 in extensions_tuple and quality == "min":
                    effective_quality = "min7"
                elif 10 in extensions_tuple and quality == "dom7":
                    effective_quality = "dom7"
                elif 9 in extensions_tuple and quality == "dim":
                    effective_quality = "dim7"

                # Add chord to score
                section.chords.append(ScoreChord(
                    time_beats=chord_start,
                    duration_beats=chord_beats,
                    root_midi=root,
                    quality=effective_quality,
                    label=label,
                    extensions=extensions_tuple,
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

                # Melody with contour — anchored to voice-validated melody_pitch
                phrase_start = (i == 0)
                phrase_end = (i == len(prog) - 1)
                # For A': reuse A's melodic motif (with variation)
                _cached = None
                if (use_saved_melody and saved_a_state is not None
                        and saved_a_state[2] is not None
                        and isinstance(saved_a_state[2], list)
                        and i < len(saved_a_state[2])):
                    _cached = saved_a_state[2][i]
                # min_pitch: melody must stay above the highest harmony note
                _max_harmony = max(voicings[i].harmony) if voicings[i].harmony else None
                _min_pitch = _max_harmony + 1 if _max_harmony is not None else None
                melody_notes = self._add_melody_notes(
                    section,
                    chord_start,
                    chord_beats,
                    voicing.melody,
                    root,
                    quality,
                    phrase_start=phrase_start,
                    phrase_end=phrase_end,
                    use_cached_contour=_cached is not None,
                    cached_contour=_cached,
                    min_pitch=_min_pitch,
                )
                section_melody_notes.append(melody_notes)

                beat_in_section += chord_beats

            # After building A section, save melody contours for A'
            # Only save for the ORIGINAL A section, not A' (which has trailing ')
            if sec_name == "A" and saved_a_state is not None:
                saved_a_state = (saved_a_state[0], saved_a_state[1], section_melody_notes)

            score.sections.append(section)
            beat_cursor += sec_beats

        score.total_beats = beat_cursor
        return score

    def _scale_ceiling(self, pitch: int, scale_pcs: list[int], max_midi: int) -> int:
        """Find the nearest scale tone at or above `pitch`, up to max_midi.

        Used to compute the melody floor above the harmony ceiling while
        guaranteeing the result is diatonic.
        """
        pc = pitch % 12
        if pc in scale_pcs:
            candidate = pitch
        else:
            # Find nearest scale pitch class at or above this one
            above = sorted(spc for spc in scale_pcs if spc > pc)
            if above:
                candidate = (pitch // 12) * 12 + above[0]
            else:
                # Wrap to next octave
                candidate = (pitch // 12 + 1) * 12 + scale_pcs[0]
        # If candidate is still below pitch (shouldn't happen), push up an octave
        while candidate < pitch and candidate + 12 <= max_midi:
            candidate += 12
        return min(max_midi, candidate)

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
        use_cached_contour: bool = False,
        cached_contour: list[tuple[float, float, int]] | None = None,
        min_pitch: int | None = None,
    ) -> list[tuple[float, float, int]]:
        """Add melody notes for one chord's duration with contour.

        The first note is anchored to melody_pitch — the voice-validated chord
        tone from voice leading. Subsequent notes form a contour with stepwise
        motion, chord-tone leaps, and phrase-aware resolution.

        If use_cached_contour is True and cached_contour is provided, reuses
        A's melodic motif (for A' section) with contour inversion.

        min_pitch: if set, all melody notes are shifted to be >= min_pitch
        (used to prevent voice crossing with harmony).  The value is pre-
        computed by the caller as a scale-tone ceiling.

        Returns a list of (time_offset, duration, pitch) tuples representing
        the contour, for motif caching in A'.
        """
        from .voice_leading import VOICE_RANGES
        melody_range = VOICE_RANGES["melody"]

        if chord_beats <= 0:
            return []

        density = self.mood.density
        if density < 0.15:
            n_notes = 1
        elif density < 0.4:
            n_notes = 2
        elif density < 0.7:
            n_notes = 4
        else:
            n_notes = 8

        sub_beat = chord_beats / n_notes

        # Build scale data once
        scale = scale_degrees(self.root_midi, self.effective_mode)
        scale_pcs = sorted(set(s % 12 for s in scale))

        # Compute a DIATONIC min_pitch: nearest scale tone at or above the harmony ceiling
        diatonic_min = None
        if min_pitch is not None:
            diatonic_min = self._scale_ceiling(min_pitch, scale_pcs, melody_range.max_midi)

        # For motif reuse (A'): replay A's contour with variation
        if use_cached_contour and cached_contour is not None:
            contour_data = self._vary_melody_contour(cached_contour, n_notes, chord_root, chord_quality)
        else:
            contour_data = None

        if contour_data is not None:
            # Use the cached/varied contour (A' motif reuse)
            # Shift notes above the diatonic ceiling
            if diatonic_min is not None:
                shifted = []
                for t_off, dur, pitch in contour_data:
                    while pitch < diatonic_min and pitch + 12 <= melody_range.max_midi:
                        pitch += 12
                    if pitch < diatonic_min:
                        # At the top of the range: quantize to nearest scale tone
                        pitch = self._quantize_to_scale(diatonic_min, scale_pcs)
                        pitch = max(diatonic_min, pitch)
                    # Final safety: quantize again after octave shifts
                    pitch = self._quantize_to_scale(pitch, scale_pcs)
                    pitch = max(diatonic_min, pitch)
                    shifted.append((t_off, dur, min(melody_range.max_midi, pitch)))
                contour_data = shifted
            contour_data = self._emit_melody(section, contour_data, chord_start, sub_beat)
            return contour_data
        else:
            # Generate contour anchored to voice-validated melody_pitch
            contour = self._melody_contour(n_notes, melody_pitch, chord_root, chord_quality,
                                           phrase_start=phrase_start, phrase_end=phrase_end)

            # Enforce diatonic: quantize every note to scale
            contour = [self._quantize_to_scale(p, scale_pcs) for p in contour]

            # Apply harmony ceiling (voice crossing prevention)
            if diatonic_min is not None:
                enforced = []
                for p in contour:
                    if p < diatonic_min:
                        # Shift up by octaves until above ceiling
                        p = p + 12 * ((diatonic_min - p + 11) // 12)
                        if p > melody_range.max_midi:
                            # Can't fit — use the ceiling itself
                            p = diatonic_min
                    # Quantize again after shift
                    p = self._quantize_to_scale(p, scale_pcs)
                    p = max(diatonic_min, p)
                    p = min(melody_range.max_midi, p)
                    enforced.append(p)
                contour = enforced

            # Build contour_data
            contour_data = []
            for i in range(n_notes):
                pitch = contour[i]
                pitch = max(0, min(127, pitch))
                contour_data.append((i * sub_beat, sub_beat * self.mood.articulation, pitch))

            contour_data = self._emit_melody(section, contour_data, chord_start, sub_beat)
            return contour_data

    def _emit_melody(
        self,
        section: Section,
        contour_data: list[tuple[float, float, int]],
        chord_start: float,
        sub_beat: float,
    ) -> list[tuple[float, float, int]]:
        """Create ScoreNotes from contour data."""
        for t_off, dur, pitch in contour_data:
            t = chord_start + t_off
            section.notes.append(ScoreNote(
                time_beats=t,
                duration_beats=dur,
                pitch_midi=pitch,
                amplitude=0.7,
                voice=VOICE_MELODY,
            ))
        return contour_data

    def _vary_melody_contour(
        self,
        cached: list[tuple[float, float, int]],
        n_notes: int,
        chord_root: int,
        chord_quality: str,
    ) -> list[tuple[float, float, int]]:
        """Create a rhythmic variation of a cached motif for A'.

        Keeps the pitch sequence intact (recognizable motif) but varies:
        - Duration pattern (swap adjacent note lengths)
        - Syncopation (shift onset slightly for some notes)
        - Contour direction inversion (mirror the intervals)

        All output pitches are quantised to the active scale.
        """
        from .voice_leading import VOICE_RANGES
        melody_range = VOICE_RANGES["melody"]

        if not cached:
            return []

        # Build scale pool for quantisation
        scale = scale_degrees(self.root_midi, self.effective_mode)
        scale_pcs = set(s % 12 for s in scale)

        # Extract pitches from cached contour
        pitches = [p for (_, _, p) in cached]

        # Vary by inverting the contour direction (mirror intervals)
        if len(pitches) >= 2 and len(cached) > 0:
            first = pitches[0]
            intervals = [pitches[i] - pitches[i-1] for i in range(1, len(pitches))]
            # Invert intervals
            varied = [first]
            for iv in intervals:
                next_p = varied[-1] - iv  # invert direction
                next_p = self._quantize_to_scale(next_p, sorted(scale_pcs))
                next_p = max(melody_range.min_midi, min(melody_range.max_midi, next_p))
                varied.append(next_p)
            pitches = varied

        # Rebuild contour_data with original timing pattern but varied pitches
        result = []
        for i, (t_off, dur, _) in enumerate(cached):
            if i < len(pitches):
                result.append((t_off, dur, pitches[i]))

        return result

    def _melody_contour(self, n_notes: int, melody_pitch: int, chord_root: int, chord_quality: str,
                        phrase_start: bool = False, phrase_end: bool = False) -> list[int]:
        """Generate a melodic contour with diverse pitch content and musical shape.

        The FIRST note is always melody_pitch — the voice-validated chord tone
        from voice_leading.voice_chord(). This ensures the melody that passes
        the parallel-avoidance gate is the melody that reaches the score.

        Subsequent notes use a scale pool around the chord tone, with stepwise
        motion, momentum, and chord-tone leaps.
        """
        from .harmony import CHORD_QUALITIES
        from .voice_leading import VOICE_RANGES
        melody_range = VOICE_RANGES["melody"]
        chord_intervals = CHORD_QUALITIES.get(chord_quality, [0, 4, 7])

        # Build scale pool
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

        # FIRST NOTE: anchored to voice-validated melody_pitch
        # This is the chord tone that passed parallel-avoidance — it MUST reach the score
        current = max(melody_range.min_midi, min(melody_range.max_midi, melody_pitch))
        contour = [current]

        # Determine momentum direction based on pitch relative to preferred range
        base_mid = (melody_range.preferred_low + melody_range.preferred_high) // 2

        for i in range(1, n_notes):
            r = self.rng.random()
            prev = contour[-1]

            # Phrase-end: bias toward tonic in last 2 notes
            if phrase_end and i >= n_notes - 2:
                tonic_pc = self.root_midi % 12
                target = min(scale_pool, key=lambda p: (
                    0 if p % 12 == tonic_pc else 1,
                    abs(p - prev)
                ))
                pitch = target
            elif r < 0.08:
                # Repeat note (slightly higher chance for rhythmic variety)
                pitch = prev
            elif r < 0.55:
                # Stepwise: continue direction with momentum
                direction = 1 if prev < base_mid else -1
                step_size = int(self.rng.choice([1, 1, 2, 2, 3]))
                pitch = prev + direction * step_size
            elif r < 0.78:
                # Chord-tone leap: pick a chord tone 3-12 semitones away
                jitter = int(self.rng.randint(0, 3))
                ct_pcs = set((chord_root + iv) % 12 for iv in chord_intervals)
                ct_options = [p for p in scale_pool if p % 12 in ct_pcs and 3 <= abs(p - prev) <= 12]
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
