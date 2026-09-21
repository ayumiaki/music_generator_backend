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
    Pitch-class distinctness is preferred: voices get different chord degrees
    where candidates permit, not octave doubles of the same degree.
    """
    if not candidates:
        raise ValueError("No chord tones in range — voice range/configuration defect")

    if n_voices <= 0:
        return []

    if not prev_harmony:
        # No previous: pick well-separated chord tones by pitch class
        if len(candidates) <= n_voices:
            return candidates[:n_voices]
        # Prefer candidates from different pitch classes
        by_pc: dict[int, list[int]] = {}
        for c in candidates:
            by_pc.setdefault(c % 12, []).append(c)
        result = []
        for pc in sorted(by_pc.keys()):
            result.append(by_pc[pc][len(by_pc[pc]) // 2])
            if len(result) >= n_voices:
                break
        if len(result) < n_voices:
            step = len(candidates) / n_voices
            return [candidates[int(i * step)] for i in range(n_voices)]
        return result

    assigned: list[int] = []
    used_pcs: set[int] = set()

    for voice_idx in range(n_voices):
        target = (
            prev_harmony[voice_idx]
            if voice_idx < len(prev_harmony)
            else (prev_harmony[-1] if prev_harmony else candidates[0])
        )

        # Rank: prefer unused pitch class, then by distance to target
        ranked = sorted(
            candidates,
            key=lambda c: (
                0 if c % 12 not in used_pcs else 1,
                semitone_distance(c, target),
            ),
        )

        chosen = None
        for c in ranked:
            if c % 12 not in used_pcs:
                chosen = c
                break

        if chosen is None:
            # All pitch classes used; take closest regardless (last resort)
            chosen = ranked[0]

        assigned.append(chosen)
        used_pcs.add(chosen % 12)

    return assigned


def _check_parallels(prev_v: Voicing, curr_v: Voicing) -> tuple[int, int]:
    """Check parallel fifths/octaves between two voicings."""
    all_prev = [prev_v.bass, prev_v.melody] + prev_v.harmony
    all_curr = [curr_v.bass, curr_v.melody] + curr_v.harmony
    return detect_parallel_fifths_octaves(all_prev, all_curr)


def _voices_unordered(v: Voicing) -> bool:
    """Check if voices are out of order (bass > harmony or harmony >= melody)."""
    if len(v.harmony) >= 2 and v.harmony[0] >= v.harmony[1]:
        return True
    if v.harmony and v.bass > v.harmony[0]:
        return True
    if v.harmony and v.melody < v.harmony[-1]:
        return True
    return False


def _bass_alternatives(v: Voicing, bass_range: VoiceRange, root: int) -> list[int]:
    """Generate alternative bass pitches for the same chord root.

    The bass is normally the root in the range's sweet spot, but for
    parallel avoidance we may need an octave displacement or inversion.
    """
    alts = [v.bass]
    # Octave displacements
    for oct_shift in [12, -12, 24, -24]:
        p = v.bass + oct_shift
        if bass_range.min_midi <= p <= bass_range.max_midi and p not in alts:
            alts.append(p)
    # Inversions: third or fifth in bass
    for iv in [4, 7, -5, -8, 3, -4, -9]:
        p = root + iv
        if bass_range.min_midi <= p <= bass_range.max_midi and p not in alts:
            alts.append(p)
    return alts


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

    Searches combinations of bass, harmony[0], harmony[1], and melody
    alternatives to minimise parallels while maintaining voice ordering.
    Falls back to the voicing with the fewest parallels if zero is unreachable.
    """
    best_v = v
    best_f, best_o = _check_parallels(prev, v)
    best_total = best_f + best_o

    if best_total == 0:
        return v

    # Generate alternatives for each voice (sorted by closeness to original)
    def by_dist(cands, target):
        return sorted(cands, key=lambda c: semitone_distance(c, target))

    bass_alts = _bass_alternatives(v, bass_range, v.chord_root)
    h0_alts = by_dist([c for c in harmony_candidates if c != v.harmony[0]], v.harmony[0]) or harmony_candidates
    h1_alts = by_dist([c for c in harmony_candidates if len(v.harmony) <= 1 or c != v.harmony[1]], v.harmony[1]) or harmony_candidates
    m_alts = by_dist([c for c in melody_candidates if c != v.melody and c > harmony_default(v)], v.melody) or melody_candidates

    # Exhaustive bounded search (bass × h0 × h1 × melody)
    for alt_b in bass_alts:
        for alt_h0 in h0_alts[:12]:
            for alt_h1 in h1_alts[:12]:
                if alt_h0 >= alt_h1:
                    continue
                if alt_b > alt_h0:
                    continue
                for alt_m in m_alts[:8]:
                    if alt_m <= alt_h1:
                        continue
                    alt_v = Voicing(
                        bass=alt_b,
                        harmony=[alt_h0, alt_h1],
                        melody=alt_m,
                        chord_root=v.chord_root,
                        chord_quality=v.chord_quality,
                        chord_label=v.chord_label,
                    )
                    f, o = _check_parallels(prev, alt_v)
                    score = f + o
                    if score < best_total:
                        best_v = alt_v
                        best_total = score
                        if best_total == 0:
                            return best_v

    return best_v


def harmony_default(v: Voicing) -> int:
    """Safe accessor for max harmony (returns 0 if empty)."""
    return max(v.harmony) if v.harmony else 0


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

    # Bass: root in bass range — choose octave closest to previous bass for
    # transposition invariance (same seed + different key → transposed output)
    if prev is not None:
        # Find the octave of chord_root closest to the previous bass
        prev_bass = prev.bass
        oct_shift = round((prev_bass - chord_root) / 12)
        bass = chord_root + oct_shift * 12
        # Clamp into range
        while bass < bass_range.min_midi:
            bass += 12
        while bass > bass_range.max_midi:
            bass -= 12
        # If still out of range, pick the closest boundary
        if bass < bass_range.min_midi:
            bass = bass_range.min_midi
        if bass > bass_range.max_midi:
            bass = bass_range.max_midi
    else:
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

    # Final ordering enforcement — always use the CURRENT values from v
    # (parallel avoidance may have changed bass/harmony/melody)
    bass = v.bass
    harmony_pitches = list(v.harmony)
    melody = v.melody
    if harmony_pitches:
        # Ensure bass is strictly below lowest harmony
        if bass >= harmony_pitches[0]:
            while bass >= harmony_pitches[0] and bass - 12 >= bass_range.min_midi:
                bass -= 12
            if bass >= harmony_pitches[0]:
                bass = harmony_pitches[0] - 12
            bass = max(bass_range.min_midi, bass)
        # Ensure melody is strictly above highest harmony
        if melody <= harmony_pitches[-1]:
            while melody <= harmony_pitches[-1] and melody + 12 <= melody_range.max_midi:
                melody += 12
            if melody <= harmony_pitches[-1]:
                melody = harmony_pitches[-1] + 12
            melody = min(melody_range.max_midi, melody)
        # Ensure harmony[0] < harmony[1]
        if len(harmony_pitches) >= 2 and harmony_pitches[0] >= harmony_pitches[1]:
            harmony_pitches = [harmony_pitches[1], harmony_pitches[0]]
        v = Voicing(
            bass=bass,
            harmony=harmony_pitches,
            melody=melody,
            chord_root=chord_root,
            chord_quality=chord_quality,
            chord_label=chord_label,
        )

    return v


def _all_voicings(
    root: int, quality: str, vr: VoiceRange, tension: float,
) -> list[int]:
    """All chord-tone MIDI pitches within a voice range."""
    return _chord_tone_candidates(root, quality, vr, tension)


def _build_voicing(
    root: int, quality: str, label: str,
    bass: int, harmony: list[int], melody: int,
) -> Voicing:
    """Build a Voicing from raw pitches."""
    return Voicing(
        bass=bass, harmony=harmony, melody=melody,
        chord_root=root, chord_quality=quality, chord_label=label,
    )


def _voices_valid(v: Voicing, bass_range: VoiceRange, melody_range: VoiceRange) -> bool:
    """Check voice ordering and range constraints."""
    if v.harmony:
        if len(v.harmony) >= 2 and v.harmony[0] >= v.harmony[1]:
            return False
        if v.bass >= v.harmony[0] and v.bass - 12 < bass_range.min_midi:
            return False
        if v.melody <= v.harmony[-1] and v.melody + 12 > melody_range.max_midi:
            return False
        if v.bass < bass_range.min_midi or v.bass > bass_range.max_midi:
            return False
        for h in v.harmony:
            if h < bass_range.min_midi or h > melody_range.max_midi:
                return False
        if v.melody < melody_range.min_midi or v.melody > melody_range.max_midi:
            return False
    return True


def _fixup_parallels(
    voicings: list[Voicing],
    bass_range: VoiceRange,
    harmony_range: VoiceRange,
    melody_range: VoiceRange,
    tension: float,
    max_passes: int = 3,
    boundary_prev: Voicing | None = None,
) -> list[Voicing]:
    """Post-processing pass to eliminate parallels via local search.

    For each consecutive pair with parallels, tries alternative voicings
    for BOTH chords (not just the current one). Multiple passes allow
    fixes to propagate through the progression.

    boundary_prev: if set, the first chord is also checked against this
    external voicing (for cross-section boundaries).
    """
    result = list(voicings)
    # Prepend boundary voicing for checking (not modified, just used as reference)
    if boundary_prev is not None:
        result = [boundary_prev] + result

    for _pass in range(max_passes):
        changed = False
        for i in range(1, len(result)):
            prev_v = result[i - 1]
            curr_v = result[i]
            f, o = _check_parallels(prev_v, curr_v)
            if f + o == 0:
                continue

            is_boundary = (boundary_prev is not None and i == 1)

            # Generate alternatives for both prev and curr
            curr_root, curr_qual, curr_label = curr_v.chord_root, curr_v.chord_quality, curr_v.chord_label
            prev_root, prev_qual, prev_label = prev_v.chord_root, prev_v.chord_quality, prev_v.chord_label

            curr_harmony_cands = _all_voicings(curr_root, curr_qual, harmony_range, tension)
            curr_melody_cands = _all_voicings(curr_root, curr_qual, melody_range, tension)

            best_total = f + o
            best_prev, best_curr = prev_v, curr_v

            # Try alternatives for current chord (keeping prev fixed)
            for h0 in curr_harmony_cands[:10]:
                curr_h1_cands = [c for c in curr_harmony_cands[:10] if c != h0]
                for h1 in curr_h1_cands:
                    for m in curr_melody_cands[:6]:
                        if m <= max(h0, h1):
                            continue
                        alt_v = _build_voicing(curr_root, curr_qual, curr_label,
                                               curr_v.bass, [h0, h1], m)
                        af, ao = _check_parallels(prev_v, alt_v)
                        if af + ao < best_total:
                            best_total = af + ao
                            best_curr = alt_v
                            if best_total == 0:
                                break
                    if best_total == 0:
                        break
                if best_total == 0:
                    break

            # If still parallels, try alternatives for previous chord (keeping curr fixed)
            # Skip if boundary — don't modify the previous section's voicing
            if best_total > 0 and not is_boundary:
                prev_harmony_cands = _all_voicings(prev_root, prev_qual, harmony_range, tension)
                prev_melody_cands = _all_voicings(prev_root, prev_qual, melody_range, tension)
                for h0 in prev_harmony_cands[:10]:
                    for h1 in [c for c in prev_harmony_cands[:10] if c != h0]:
                        for m in prev_melody_cands[:6]:
                            if m <= max(h0, h1):
                                continue
                            alt_prev = _build_voicing(prev_root, prev_qual, prev_label,
                                                       prev_v.bass, [h0, h1], m)
                            # Check parallels with i-2 (if exists)
                            if i >= 2:
                                f2, o2 = _check_parallels(result[i - 2], alt_prev)
                                if f2 + o2 > 0:
                                    continue
                            af, ao = _check_parallels(alt_prev, curr_v)
                            if af + ao < best_total:
                                best_total = af + ao
                                best_prev = alt_prev
                                if best_total == 0:
                                    break
                        if best_total == 0:
                            break
                    if best_total == 0:
                        break

            if best_curr is not curr_v:
                result[i] = best_curr
                changed = True
            if best_prev is not prev_v:
                result[i - 1] = best_prev
                changed = True

        if not changed:
            break

    # Strip boundary voicing if prepended
    if boundary_prev is not None:
        result = result[1:]

    return result


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

    Post-processing eliminates parallel fifths/octaves via local backtracking
    search across consecutive chord pairs.
    """
    if bass_range is None:
        bass_range = VOICE_RANGES["bass"]
    if harmony_range is None:
        harmony_range = VOICE_RANGES["harmony"]
    if melody_range is None:
        melody_range = VOICE_RANGES["melody"]

    voicings: list[Voicing] = []
    prev: Voicing | None = initial_voicing
    for root, quality, label in progression:
        v = voice_chord(
            root, quality, label, prev,
            bass_range, harmony_range, melody_range, tension, register,
        )
        voicings.append(v)
        prev = v

    # Post-processing: fix parallels via local search with backtracking
    # Pass initial_voicing as boundary to also check cross-section parallels
    voicings = _fixup_parallels(
        voicings, bass_range, harmony_range, melody_range, tension,
        boundary_prev=initial_voicing,
    )

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
