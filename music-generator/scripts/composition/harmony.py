"""Harmony — scales, chord vocabulary, and functional-progression grammar.

Everything is deterministic given a seed. No randomness creeps into the
composition logic itself: the RNG only seeds the choices, and the same
seed always walks the same path through the grammar.

Modes are interval patterns (semitone offsets from the root).
Chord qualities are interval patterns (semitone offsets from the chord root).
Functional progressions are a weighted grammar over chord functions per scale
degree, with controlled substitutions.
"""
from __future__ import annotations

import numpy as np
from typing import Optional


# --- Notes ---

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
NOTE_NAME_TO_MIDI = {name: i for i, name in enumerate(NOTE_NAMES)}
# Normalise common enharmonics
ENHARMONIC = {
    "Db": "C#", "Eb": "D#", "Gb": "F#", "Ab": "G#", "Bb": "A#",
    "E#": "F", "B#": "C", "Cb": "B", "Fb": "E",
}
NOTE_NAME_TO_MIDI.update({k: v for k, v in ((k, NOTE_NAME_TO_MIDI[v]) for k, v in ENHARMONIC.items())})


def note_to_midi(name: str, octave: int = 4) -> int:
    """Convert a named note to MIDI. e.g. note_to_midi('C', 4) == 60."""
    name = ENHARMONIC.get(name, name)
    return NOTE_NAME_TO_MIDI[name] + (octave + 1) * 12


def midi_to_note(midi: int) -> str:
    """MIDI note number -> name with octave. e.g. midi_to_note(60) == 'C4'."""
    return f"{NOTE_NAMES[midi % 12]}{(midi // 12) - 1}"


# --- Modes ---

MODES = {
    "major":       [0, 2, 4, 5, 7, 9, 11],
    "natural_minor": [0, 2, 3, 5, 7, 8, 10],
    "harmonic_minor": [0, 2, 3, 5, 7, 8, 11],
    "dorian":      [0, 2, 3, 5, 7, 9, 10],
    "mixolydian":  [0, 2, 4, 5, 7, 9, 10],
}

VALID_MODES = set(MODES.keys())


def scale_degrees(root_midi: int, mode: str) -> list[int]:
    """All scale degrees (one octave) as MIDI note numbers starting from root_midi."""
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}; valid: {sorted(VALID_MODES)}")
    return [root_midi + iv for iv in MODES[mode]]


def diatonic_pitch(root_midi: int, mode: str, scale_idx: int) -> int:
    """Pick a scale degree by index (0 = tonic), wrapping octaves."""
    degs = scale_degrees(root_midi, mode)
    octave, idx = divmod(scale_idx, len(degs))
    return degs[idx] + 12 * octave


# --- Chords ---

# Interval patterns (semitone offsets from root) for each quality
CHORD_QUALITIES = {
    "maj":       [0, 4, 7],
    "min":       [0, 3, 7],
    "dim":       [0, 3, 6],
    "aug":       [0, 4, 8],
    "dom7":      [0, 4, 7, 10],
    "maj7":      [0, 4, 7, 11],
    "min7":      [0, 3, 7, 10],
    "dim7":      [0, 3, 6, 9],
    "half_dim7": [0, 3, 6, 10],
    "sus2":      [0, 2, 7],
    "sus4":      [0, 5, 7],
}

VALID_QUALITIES = set(CHORD_QUALITIES.keys())


def chord_pitches(root_midi: int, quality: str) -> list[int]:
    """MIDI notes for a chord of the given quality."""
    if quality not in CHORD_QUALITIES:
        raise ValueError(f"unknown quality {quality!r}; valid: {sorted(VALID_QUALITIES)}")
    return [root_midi + iv for iv in CHORD_QUALITIES[quality]]


# --- Diatonic chord derivation ---

# For each mode, the triad quality built on each scale degree (0-indexed)
# I, ii, iii, IV, V, vi, vii° ...
MODE_TRIAD_QUALITIES = {
    "major":         ["maj", "min", "min", "maj", "maj", "min", "dim"],
    "natural_minor": ["min", "dim", "maj", "min", "min", "maj", "maj"],
    "harmonic_minor": ["min", "dim", "aug", "min", "maj", "maj", "dim"],
    "dorian":        ["min", "min", "maj", "maj", "min", "dim", "maj"],
    "mixolydian":     ["maj", "min", "dim", "maj", "min", "min", "maj"],
}


def diatonic_triad(root_midi: int, mode: str, degree: int) -> tuple[int, str]:
    """Build the triad on a scale degree. Returns (root_midi, quality)."""
    if mode not in MODE_TRIAD_QUALITIES:
        raise ValueError(f"unsupported mode {mode!r}")
    degs = scale_degrees(root_midi, mode)
    qualities = MODE_TRIAD_QUALITIES[mode]
    idx = degree % 7
    return (degs[idx], qualities[idx])


# --- Functional labels ---

ROMAN = ["I", "II", "III", "IV", "V", "VI", "VII"]

FUNCTION_TONIC = {0, 3, 5}     # I, IV, vi
FUNCTION_DOMINANT = {4, 6}     # V, vii°
FUNCTION_PREDOMINANT = {1, 3}  # ii, IV


def function_of(degree: int) -> str:
    """Classify a scale degree as tonic / predominant / dominant."""
    d = degree % 7
    if d in FUNCTION_TONIC:
        return "tonic"
    if d in FUNCTION_DOMINANT:
        return "dominant"
    return "predominant"


def functional_label(degree: int, quality: str) -> str:
    """Build a Roman-numeral label. Lowercase for minor/dim, ° for dim."""
    d = degree % 7
    if quality == "dim":
        return ROMAN[d] + "°"
    if quality == "aug":
        return ROMAN[d] + "+"
    if quality == "min" or quality == "min7":
        return ROMAN[d].lower()
    return ROMAN[d]


# --- Progression grammar ---

# Weighted continuation rules: for each current degree, what can follow?
# Weights are relative likelihoods (integer, higher = more common).
PROGRESSION_RULES: dict[int, list[tuple[int, int]]] = {
    # I → anything but strongly to IV, V, vi
    0: [(0, 1), (1, 2), (2, 1), (3, 4), (4, 4), (5, 3), (6, 1)],
    # ii → V strongly, sometimes vii°, I
    1: [(0, 1), (1, 0), (2, 0), (3, 1), (4, 6), (5, 1), (6, 3)],
    # iii → IV, vi
    2: [(0, 1), (1, 1), (2, 0), (3, 3), (4, 1), (5, 4), (6, 1)],
    # IV → I, V, ii
    3: [(0, 4), (1, 2), (2, 1), (3, 1), (4, 5), (5, 2), (6, 1)],
    # V → I (strong cadence), vi (deceptive), IV
    4: [(0, 8), (1, 1), (2, 0), (3, 2), (4, 0), (5, 3), (6, 1)],
    # vi → ii, IV, V, I
    5: [(0, 2), (1, 3), (2, 1), (3, 3), (4, 3), (5, 0), (6, 1)],
    # vii° → I (strong), V
    6: [(0, 6), (1, 0), (2, 0), (3, 1), (4, 2), (5, 0), (6, 0)],
}


def choose_successor(current_degree: int, rng: np.random.RandomState) -> int:
    """Pick the next scale degree in a progression, weighted by the grammar."""
    options = PROGRESSION_RULES[current_degree % 7]
    degrees, weights = zip(*options)
    total = sum(weights)
    if total == 0:
        return 0  # fallback to tonic
    probabilities = [w / total for w in weights]
    return int(rng.choice(degrees, p=probabilities))


def generate_progression(
    root_midi: int,
    mode: str,
    n_chords: int,
    rng: np.random.RandomState,
    start_degree: int = 0,
    end_degree: int = 0,
) -> list[tuple[int, str, str]]:
    """Generate a chord progression as a list of (root_midi, quality, label)."""
    if mode not in MODE_TRIAD_QUALITIES:
        raise ValueError(f"unsupported mode {mode!r}")
    if n_chords <= 0:
        return []

    progression = []
    current = start_degree % 7
    for i in range(n_chords):
        # Force the last chord to be the requested end degree
        if i == n_chords - 1 and end_degree is not None:
            current = end_degree % 7
        root, quality = diatonic_triad(root_midi, mode, current)
        label = functional_label(current, quality)
        progression.append((root, quality, label))
        if i < n_chords - 1:
            current = choose_successor(current, rng)
    return progression


# --- Voice leading primitives ---

def semitone_distance(a: int, b: int) -> int:
    """Absolute distance in semitones between two MIDI notes."""
    return abs(a - b)


def closest_pitch(target: int, candidates: list[int]) -> int:
    """Pick the candidate pitch closest to the target (minimise movement)."""
    return min(candidates, key=lambda c: semitone_distance(target, c))


def best_voice_movement(
    prev_pitches: list[int],
    chord_pitches_list: list[int],
) -> list[int]:
    """Assign chord voices to minimise total semitone movement from prev_pitches.

    This is the classic assignment problem, but with 3-4 voices a brute-force
    permutation search is fine (max 24 candidates). Each chord voice must map
    1:1 to a chord pitch, and we pick the ordering that minimises total movement.
    """
    from itertools import permutations

    n = len(prev_pitches)
    if n == 0:
        return chord_pitches_list[:4]

    best = None
    best_cost = float("inf")
    # We may have more chord pitches than voices; try all permutations of
    # the first n, or pad if fewer
    n_voices = len(chord_pitches_list)
    if n_voices < n:
        # Pad with the lowest chord pitch up an octave
        padded = list(chord_pitches_list) + [chord_pitches_list[-1] + 12] * (n - n_voices)
        chord_pitches_list = padded

    for perm in permutations(chord_pitches_list, n):
        cost = sum(semitone_distance(p, a) for p, a in zip(prev_pitches, perm))
        if cost < best_cost:
            best_cost = cost
            best = perm
    return list(best) if best else list(chord_pitches_list[:n])


def detect_parallel_fifths_octaves(
    prev: list[int],
    curr: list[int],
) -> tuple[int, int]:
    """Count parallel fifths (7 semitones) and octaves (12) between two voicings."""
    n = min(len(prev), len(curr))
    if n < 2:
        return 0, 0

    par_fifths = 0
    par_octaves = 0
    for i in range(n):
        for j in range(i + 1, n):
            prev_int = abs(prev[i] - prev[j])
            curr_int = abs(curr[i] - curr[j])
            # Check both voices moved (no stationary voice)
            if (prev[i] == curr[i] and prev[j] == curr[j]):
                continue
            # Parallel motion: both moved in the same direction
            dir_i = curr[i] - prev[i]
            dir_j = curr[j] - prev[j]
            if dir_i == 0 or dir_j == 0:
                continue
            same_dir = (dir_i > 0) == (dir_j > 0)
            if not same_dir:
                continue
            # Perfect intervals
            if prev_int % 12 == 0 and curr_int % 12 == 0:
                par_octaves += 1
            elif prev_int % 12 == 7 and curr_int % 12 == 7:
                par_fifths += 1
    return par_fifths, par_octaves
