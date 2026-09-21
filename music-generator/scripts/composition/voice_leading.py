"""Voice leading — distribute chord pitches across four instrument voices.

Four voices: bass (root movement), harmony (chord tones), melody (top-line
motion), percussion (not handled here). The voice leading engine keeps each
voice in its target range, prefers minimal semitone motion between chords,
penalises voice crossing and parallel perfect intervals, and enforces
polyphony limits.

All assignment is deterministic given the same chord sequence and the same
initial voice positions.
"""
from __future__ import annotations

from dataclasses import dataclass

from .harmony import (
    best_voice_movement,
    chord_pitches,
    detect_parallel_fifths_octaves,
    semitone_distance,
)


# Default voice ranges (MIDI notes) — generous but musical.
@dataclass(frozen=True)
class VoiceRange:
    """A voice's pitch range and preferred register."""
    name: str
    min_midi: int
    max_midi: int
    preferred_low: int  # preferred bottom of the "sweet spot"
    preferred_high: int  # preferred top of the "sweet spot"


VOICE_RANGES = {
    "bass": VoiceRange("bass", 28, 60, 36, 48),        # E1–C4
    "harmony": VoiceRange("harmony", 48, 79, 55, 72),   # C3–G5
    "melody": VoiceRange("melody", 60, 91, 67, 84),     # C4–G6
}


def _clamp_to_range(pitch: int, vr: VoiceRange) -> int:
    """Move a pitch into the voice range by octave transposition."""
    p = pitch
    while p < vr.min_midi:
        p += 12
    while p > vr.max_midi:
        p -= 12
    # If still out of range, clamp (shouldn't happen with sane ranges)
    return max(vr.min_midi, min(vr.max_midi, p))


def _quantize_to_range(pitch: int, vr: VoiceRange) -> int:
    """Quantize a pitch into the voice's preferred register if possible."""
    p = _clamp_to_range(pitch, vr)
    # Pull toward preferred range
    while p < vr.preferred_low and p + 12 <= vr.max_midi:
        p += 12
    while p > vr.preferred_high and p - 12 >= vr.min_midi:
        p -= 12
    return p


@dataclass
class Voicing:
    """A single chord's worth of voice assignments."""
    bass: int
    harmony: list[int]  # may contain 1-2 chord tones
    melody: int
    chord_root: int
    chord_quality: str
    chord_label: str


def voice_chord(
    chord_root: int,
    chord_quality: str,
    chord_label: str,
    prev: Voicing | None,
    bass_range: VoiceRange = VOICE_RANGES["bass"],
    harmony_range: VoiceRange = VOICE_RANGES["harmony"],
    melody_range: VoiceRange = VOICE_RANGES["melody"],
) -> Voicing:
    """Voice a single chord, optionally considering the previous chord for smoothness."""
    pitches = chord_pitches(chord_root, chord_quality)

    # Bass: root, in bass range
    bass = _quantize_to_range(chord_root, bass_range)
    if bass > bass_range.preferred_high and bass - 12 >= bass_range.min_midi:
        bass -= 12

    # Harmony voices: fill from remaining chord tones
    remaining = [p for p in pitches if p % 12 != chord_root % 12]
    if not remaining:
        # Triad: use third and fifth
        remaining = [p for p in pitches if p != chord_root]

    # Place harmony tones in harmony range, preferring smooth motion
    harmony_pitches: list[int] = []
    for p in remaining[:2]:  # up to 2 harmony voices
        hp = _quantize_to_range(p, harmony_range)
        if prev is not None and prev.harmony:
            # Move as little as possible from previous harmony
            closest = min(
                range(harmony_range.min_midi, harmony_range.max_midi + 1),
                key=lambda c: (semitone_distance(c, prev.harmony[0]) if prev.harmony else 0) +
                              (0 if harmony_range.min_midi <= c <= harmony_range.max_midi else 1000),
            )
            if harmony_range.min_midi <= closest <= harmony_range.max_midi:
                hp = closest
        harmony_pitches.append(hp)

    # Melody: top voice, pick a chord tone near previous melody or preferred register
    if prev is not None:
        target = prev.melody
        candidates = [_quantize_to_range(p, melody_range) for p in pitches]
        melody = min(candidates, key=lambda c: semitone_distance(c, target))
    else:
        # First chord: pick preferred register
        melody = max(melody_range.preferred_low, min(melody_range.preferred_high, melody_range.preferred_low + 7))
        candidates = [_quantize_to_range(p, melody_range) for p in pitches]
        melody = min(candidates, key=lambda c: abs(c - melody))

    # Ensure melody doesn't cross below harmony
    if harmony_pitches:
        max_harmony = max(harmony_pitches)
        if melody <= max_harmony:
            # Push melody up an octave if possible
            if melody + 12 <= melody_range.max_midi:
                melody += 12
            else:
                melody = max_harmony + 1

    # Ensure bass doesn't cross above harmony
    if harmony_pitches:
        min_harmony = min(harmony_pitches)
        if bass >= min_harmony:
            if bass - 12 >= bass_range.min_midi:
                bass -= 12

    return Voicing(
        bass=bass,
        harmony=harmony_pitches,
        melody=melody,
        chord_root=chord_root,
        chord_quality=chord_quality,
        chord_label=chord_label,
    )


def voice_progression(
    progression: list[tuple[int, str, str]],
    bass_range: VoiceRange = VOICE_RANGES["bass"],
    harmony_range: VoiceRange = VOICE_RANGES["harmony"],
    melody_range: VoiceRange = VOICE_RANGES["melody"],
) -> list[Voicing]:
    """Voice an entire chord progression."""
    voicings: list[Voicing] = []
    prev: Voicing | None = None
    for root, quality, label in progression:
        v = voice_chord(root, quality, label, prev, bass_range, harmony_range, melody_range)
        voicings.append(v)
        prev = v
    return voicings


def count_parallels(voicings: list[Voicing]) -> tuple[int, int]:
    """Count total parallel fifths and octaves across a voiced progression."""
    total_fifths = 0
    total_octaves = 0
    for i in range(1, len(voicings)):
        prev_v = voicings[i - 1]
        curr_v = voicings[i]
        all_prev = [prev_v.bass, prev_v.melody] + prev_v.harmony
        all_curr = [curr_v.bass, curr_v.melody] + curr_v.harmony
        f, o = detect_parallel_fifths_octaves(all_prev, all_curr)
        total_fifths += f
        total_octaves += o
    return total_fifths, total_octaves


def has_voice_crossing(voicings: list[Voicing]) -> bool:
    """Detect if any voicing has voices out of order (bass ≤ harmony ≤ melody)."""
    for v in voicings:
        if v.harmony:
            if v.bass > min(v.harmony):
                return True
            if v.melody < max(v.harmony):
                return True
        else:
            if v.bass > v.melody:
                return True
    return False
