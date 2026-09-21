"""Voice leading — distribute chord pitches across instrument voices.

Four voices: bass (root movement), harmony (chord tones), melody (top-line
motion), percussion (not handled here). The voice leading engine keeps each
voice in its target range, prefers minimal semitone motion between chords,
penalises voice crossing and parallel perfect intervals.

All assignment is deterministic given the same chord sequence and the same
initial voice positions.

NO chromatic fallback: harmony voices MUST be chord tones. If no chord tone
exists in the configured range, that's a configuration defect and fails loudly.
"""
from __future__ import annotations

from dataclasses import dataclass

from .harmony import (
    CHORD_QUALITIES,
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


def register_target(vr: VoiceRange, register: float) -> int:
    """Map register (0..1) to a target pitch within the voice range.

    0 = low end of preferred range, 1 = high end.
    This shifts where voices sit within their absolute range
    without moving the range itself.
    """
    return vr.preferred_low + int(register * (vr.preferred_high - vr.preferred_low))


def _clamp_to_range(pitch: int, vr: VoiceRange) -> int:
    """Move a pitch into the voice range by octave transposition."""
    p = pitch
    while p < vr.min_midi:
        p += 12
    while p > vr.max_midi:
        p -= 12
    return max(vr.min_midi, min(vr.max_midi, p))

def _quantize_to_range(pitch: int, vr: VoiceRange, register: float = 0.5) -> int:
    """Quantize a pitch into the voice's preferred register.

    Register (0..1) controls where within the range the pitch lands:
    0 = low end of preferred range, 1 = high end.
    This shifts the target without moving the absolute range.
    """
    p = _clamp_to_range(pitch, vr)
    target = register_target(vr, register)
    # Pull toward target
    while p < target and p + 12 <= vr.max_midi:
        p += 12
    while p > target and p - 12 >= vr.min_midi:
        p -= 12
    return p


def _chord_tone_candidates(
    root_midi: int, quality: str, vr: VoiceRange, tension: float = 0.0,
) -> list[int]:
    """All chord-tone MIDI pitches within a voice range, sorted low to high.

    At higher tension, 7th extensions are added as available chord tones.
    """
    intervals = list(CHORD_QUALITIES.get(quality, [0, 4, 7]))

    # At higher tension, add the diatonic 7th as an available chord tone
    if tension > 0.4:
        if quality in ("maj", "aug"):
            intervals.append(11)  # major 7th
        elif quality in ("min",):
            intervals.append(10)  # minor 7th
        elif quality == "dim":
            intervals.append(9)   # diminished 7th
        elif quality == "dom7":
            intervals.append(10)  # minor 7th (already in dom7)

    candidates: set[int] = set()
    for oct_shift in range(-3, 4):
        for iv in intervals:
            p = root_midi + iv + oct_shift * 12
            if vr.min_midi <= p <= vr.max_midi:
                candidates.add(p)

    return sorted(candidates)


@dataclass
class Voicing:
    """A single chord's worth of voice assignments."""
    bass: int
    harmony: list[int]  # may contain 1-3 chord tones
    melody: int
    chord_root: int
    chord_quality: str
    chord_label: str


def _pick_distinct_chord_tones(
    candidates: list[int],
    prev_harmony: list[int],
    n_voices: int = 2,
) -> list[int]:
    """Pick n_voices distinct chord tones, each close to its previous position.

    Voice identity is preserved: harmony[0] tracks prev[0], harmony[1] tracks prev[1].
    Each voice picks the closest available chord tone to its own previous pitch.
    """
    if not candidates:
        raise ValueError("No chord tones in range — voice range/configuration defect")

    if n_voices <= 0:
        return []

    if not prev_harmony:
        # No previous: pick well-separated chord tones
        if len(candidates) <= n_voices:
            return candidates[:n_voices]
        step = len(candidates) / n_voices
        return [candidates[int(i * step)] for i in range(n_voices)]

    assigned: list[int] = []
    used: set[int] = set()

    for voice_idx in range(n_voices):
        target = (
            prev_harmony[voice_idx]
            if voice_idx < len(prev_harmony)
            else (prev_harmony[-1] if prev_harmony else candidates[0])
        )

        # Rank candidates: unused first, then by distance to target
        ranked = sorted(
            candidates,
            key=lambda c: (0 if c not in used else 1, semitone_distance(c, target)),
        )

        chosen = None
        for c in ranked:
            if c not in used:
                chosen = c
                break

        if chosen is None:
            # All chord tones used; take the closest (allows doubling for triads)
            chosen = ranked[0]

        assigned.append(chosen)
        used.add(chosen)

    return assigned


def _check_parallels(prev_v: Voicing, curr_v: Voicing) -> tuple[int, int]:
    """Check parallel fifths/octaves between two voicings."""
    all_prev = [prev_v.bass, prev_v.melody] + prev_v.harmony
    all_curr = [curr_v.bass, curr_v.melody] + curr_v.harmony
    return detect_parallel_fifths_octaves(all_prev, all_curr)


def _try_avoid_parallels(
    v: Voicing,
    prev: Voicing,
    harmony_candidates: list[int],
    melody_candidates: list[int],
    bass_range: VoiceRange,
    harmony_range: VoiceRange,
    melody_range: VoiceRange,
) -> Voicing:
    """Try alternative voicings to eliminate parallel fifths/octaves.

    Exhaustively searches combinations of harmony[0], harmony[1], and melody
    alternatives to find a voicing with zero parallels. Falls back to the
    voicing with the fewest parallels if zero is unreachable.
    """
    best_v = v
    best_f, best_o = _check_parallels(prev, v)
    best_total = best_f + best_o

    if best_total == 0:
        return v

    # Generate alternatives for each voice
    h0_alts = [c for c in harmony_candidates if c != v.harmony[0]] or harmony_candidates
    h1_alts = [c for c in harmony_candidates if len(v.harmony) <= 1 or c != v.harmony[1]] or harmony_candidates
    m_alts = [c for c in melody_candidates if c != v.melody and c > max(v.harmony)] or melody_candidates

    # Try all combinations of alternatives (bounded search)
    for alt_h0 in h0_alts[:6]:  # limit search to top candidates
        for alt_h1 in h1_alts[:6]:
            # Ensure harmony voices are distinct
            if alt_h0 == alt_h1:
                continue
            for alt_m in m_alts[:4]:
                alt_v = Voicing(
                    bass=v.bass,
                    harmony=[alt_h0, alt_h1],
                    melody=alt_m,
                    chord_root=v.chord_root,
                    chord_quality=v.chord_quality,
                    chord_label=v.chord_label,
                )
                f, o = _check_parallels(prev, alt_v)
                if f + o < best_total:
                    best_v = alt_v
                    best_total = f + o
                    if best_total == 0:
                        return best_v

    return best_v


def voice_chord(
    chord_root: int,
    chord_quality: str,
    chord_label: str,
    prev: Voicing | None,
    bass_range: VoiceRange | None = None,
    harmony_range: VoiceRange | None = None,
    melody_range: VoiceRange | None = None,
    tension: float = 0.0,
    register: float = 0.5,
) -> Voicing:
    """Voice a single chord, optionally considering the previous chord for smoothness.

    Harmony voices are ALWAYS chord tones — no chromatic fallback.
    Each harmony voice tracks its own previous position independently.
    Parallel fifths/octaves are avoided by trying alternative voicings.
    Register (0..1) controls where within each voice range the pitches land.
    """
    if bass_range is None:
        bass_range = VOICE_RANGES["bass"]
    if harmony_range is None:
        harmony_range = VOICE_RANGES["harmony"]
    if melody_range is None:
        melody_range = VOICE_RANGES["melody"]

    # Bass: root in bass range
    bass = _quantize_to_range(chord_root, bass_range, register)
    if bass > bass_range.preferred_high and bass - 12 >= bass_range.min_midi:
        bass -= 12

    # Harmony: distinct chord tones, each voice tracking its own previous position
    harmony_candidates = _chord_tone_candidates(chord_root, chord_quality, harmony_range, tension)
    prev_harmony = prev.harmony if prev else []
    harmony_pitches = _pick_distinct_chord_tones(harmony_candidates, prev_harmony, n_voices=2)

    # Melody: chord tone closest to previous melody
    melody_candidates = _chord_tone_candidates(chord_root, chord_quality, melody_range, tension)
    if prev is not None:
        melody = min(melody_candidates, key=lambda c: semitone_distance(c, prev.melody))
    else:
        melody = _quantize_to_range(chord_root + 12, melody_range, register)

    # Ensure voice ordering (no crossing)
    if harmony_pitches:
        max_harmony = max(harmony_pitches)
        if melody <= max_harmony:
            if melody + 12 <= melody_range.max_midi:
                melody += 12
            else:
                melody = max_harmony + 1

        min_harmony = min(harmony_pitches)
        if bass >= min_harmony:
            if bass - 12 >= bass_range.min_midi:
                bass -= 12

    v = Voicing(
        bass=bass,
        harmony=harmony_pitches,
        melody=melody,
        chord_root=chord_root,
        chord_quality=chord_quality,
        chord_label=chord_label,
    )

    # Parallel avoidance: try alternative voicings if parallels detected
    if prev is not None:
        v = _try_avoid_parallels(
            v, prev, harmony_candidates, melody_candidates,
            bass_range, harmony_range, melody_range,
        )

    return v


def voice_progression(
    progression: list[tuple[int, str, str]],
    bass_range: VoiceRange | None = None,
    harmony_range: VoiceRange | None = None,
    melody_range: VoiceRange | None = None,
    tension: float = 0.0,
    register: float = 0.5,
    initial_voicing: Voicing | None = None,
) -> list[Voicing]:
    """Voice an entire chord progression with continuous voice state.

    Pass `initial_voicing` to continue from a previous section's final voicing
    (preserves voice leading across section boundaries).
    Register (0..1) controls where within each voice range pitches land.
    """
    voicings: list[Voicing] = []
    prev: Voicing | None = initial_voicing
    for root, quality, label in progression:
        v = voice_chord(
            root, quality, label, prev,
            bass_range, harmony_range, melody_range, tension, register,
        )
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
