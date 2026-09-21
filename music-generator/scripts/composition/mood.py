"""Mood — map high-level mood descriptors to concrete musical parameters.

Each mood profile controls:
- tempo (BPM range)
- mode preference (major, minor, dorian, mixolydian)
- harmonic tension (how often dissonant chords appear)
- rhythmic density (note events per bar)
- register (how high/low the melody sits)
- articulation (staccato vs legato — note duration as fraction of inter-onset interval)
- timbre (waveform preference)
- drum density

All values are deterministic given a mood string. The mood string is matched
against known profiles; unknown moods fall back to a neutral profile.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class MoodProfile:
    """Concrete musical parameters derived from a mood descriptor."""
    name: str
    tempo_low: int = 90
    tempo_high: int = 120
    mode: str = "major"
    tension: float = 0.3        # 0 = consonant, 1 = tense/dissonant
    density: float = 0.5        # 0 = sparse, 1 = busy
    register: float = 0.5       # 0 = low, 1 = high
    articulation: float = 0.7  # 0 = staccato, 1 = legato (fraction of IOI)
    waveform: str = "saw"
    drum_density: float = 0.5  # 0 = none, 1 = busy
    description: str = ""


# --- Built-in mood profiles ---

MOOD_PROFILES: dict[str, MoodProfile] = {
    "happy": MoodProfile(
        name="happy",
        tempo_low=110, tempo_high=130,
        mode="major",
        tension=0.2, density=0.6, register=0.7,
        articulation=0.6, waveform="square",
        drum_density=0.6,
        description="upbeat, bright, major key, moderate tempo",
    ),
    "sad": MoodProfile(
        name="sad",
        tempo_low=70, tempo_high=90,
        mode="natural_minor",
        tension=0.4, density=0.3, register=0.3,
        articulation=0.85, waveform="saw",
        drum_density=0.2,
        description="slow, minor key, legato, sparse",
    ),
    "energetic": MoodProfile(
        name="energetic",
        tempo_low=130, tempo_high=160,
        mode="major",
        tension=0.3, density=0.8, register=0.6,
        articulation=0.5, waveform="square",
        drum_density=0.8,
        description="fast, busy, bright, driving",
    ),
    "calm": MoodProfile(
        name="calm",
        tempo_low=60, tempo_high=80,
        mode="dorian",
        tension=0.15, density=0.2, register=0.4,
        articulation=0.9, waveform="sine",
        drum_density=0.1,
        description="slow, sparse, gentle, warm",
    ),
    "dark": MoodProfile(
        name="dark",
        tempo_low=80, tempo_high=110,
        mode="harmonic_minor",
        tension=0.6, density=0.5, register=0.25,
        articulation=0.7, waveform="saw",
        drum_density=0.5,
        description="minor, tense, low register, ominous",
    ),
    "triumphant": MoodProfile(
        name="triumphant",
        tempo_low=100, tempo_high=120,
        mode="mixolydian",
        tension=0.25, density=0.7, register=0.75,
        articulation=0.65, waveform="square",
        drum_density=0.7,
        description="anthemic, bright but grounded, strong rhythm",
    ),
    "melancholic": MoodProfile(
        name="melancholic",
        tempo_low=75, tempo_high=95,
        mode="natural_minor",
        tension=0.45, density=0.35, register=0.35,
        articulation=0.85, waveform="saw",
        drum_density=0.2,
        description="slow, minor, legato, reflective",
    ),
    "playful": MoodProfile(
        name="playful",
        tempo_low=115, tempo_high=135,
        mode="major",
        tension=0.2, density=0.7, register=0.65,
        articulation=0.45, waveform="square",
        drum_density=0.5,
        description="bouncy, staccato, bright, syncopated",
    ),
    "epic": MoodProfile(
        name="epic",
        tempo_low=85, tempo_high=105,
        mode="harmonic_minor",
        tension=0.5, density=0.6, register=0.6,
        articulation=0.75, waveform="saw",
        drum_density=0.7,
        description="cinematic, sweeping, tense, powerful",
    ),
    "dreamy": MoodProfile(
        name="dreamy",
        tempo_low=65, tempo_high=85,
        mode="dorian",
        tension=0.2, density=0.25, register=0.55,
        articulation=0.95, waveform="sine",
        drum_density=0.15,
        description="slow, legato, floating, soft",
    ),
}

DEFAULT_MOOD = MoodProfile(
    name="neutral",
    tempo_low=90, tempo_high=110,
    mode="major",
    tension=0.3, density=0.5, register=0.5,
    articulation=0.7, waveform="saw",
    drum_density=0.4,
    description="neutral fallback",
)


def get_mood(name: str) -> MoodProfile:
    """Look up a mood profile by name (case-insensitive). Falls back to neutral."""
    key = name.strip().lower()
    return MOOD_PROFILES.get(key, DEFAULT_MOOD)


def list_moods() -> list[str]:
    """List all available mood names."""
    return sorted(MOOD_PROFILES.keys())
