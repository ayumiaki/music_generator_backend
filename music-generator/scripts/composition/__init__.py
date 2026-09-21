"""Composition layer — turn musical intent into a symbolic Score.

Modules:
    score:        Score, ScoreNote, ScoreChord, Section, midi_to_freq
    harmony:      scales, chords, functional progressions, voice-leading helpers
    voice_leading: distribute chord pitches across bass/harmony/melody
    mood:         map mood strings to musical parameters
    composition:  generate a full Score from CompositionConfig
    conductor:    bridge a Score to a SynthEngine for rendering
"""
