"""Conductor — convert a SynthEngine's score into rendered audio.

The conductor is the thin bridge between the composition layer (symbolic Score)
and the synthesis layer (SynthEngine that renders PCM). It translates ScoreNote
events into SynthEngine.add_note() calls with frequencies derived from MIDI
pitch, and renders the result to WAV.

The conductor owns the tempo-to-duration conversion and the MIDI-to-frequency
translation so neither the score nor the synth has to know about each other.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Optional

import numpy as np

from .composition import Composition, CompositionConfig, score_to_dict
from .mood import MoodProfile, get_mood
from .score import Score, ScoreNote, midi_to_freq


def _midi_to_freq_safe(midi: Optional[int]) -> float:
    """Convert MIDI note to Hz. Percussion (None) returns 0 Hz (skipped)."""
    if midi is None:
        return 0.0
    return midi_to_freq(midi)


class Conductor:
    """Drive a SynthEngine from a Score."""

    def __init__(self, config: CompositionConfig) -> None:
        self.composition = Composition(config)
        self.score: Optional[Score] = None

    def compose(self) -> Score:
        """Generate the score (or return cached if already composed)."""
        if self.score is None:
            self.score = self.composition.generate()
        return self.score

    def render_to_engine(self, engine) -> None:
        """Feed the score into a SynthEngine, then render.

        The engine should be a fresh SynthEngine instance configured with
        the correct seed (matching the composition's seed).
        """
        score = self.compose()
        mood = self.composition.mood

        # Configure engine tempo
        engine.config.bpm = score.bpm

        for note in score.all_notes():
            freq = _midi_to_freq_safe(note.pitch_midi)
            if freq <= 0:
                continue  # skip percussion placeholders
            engine.add_note(
                time_beats=note.time_beats,
                freq=freq,
                duration_beats=note.duration_beats,
                amplitude=note.amplitude,
                waveform=mood.waveform,
            )

        # Add drums based on mood density
        if mood.drum_density > 0.1:
            n_bars = int(score.total_beats / 4) or 1
            engine.add_drums(
                bars=n_bars,
                kick_pattern="four_on_floor",
                snare_pattern="backbeat",
                hihat_pattern="eighth" if mood.drum_density > 0.4 else "quarter",
            )

    def render(
        self,
        engine_class,
        output_path: Optional[str | Path] = None,
    ) -> tuple[Score, object]:
        """Full pipeline: compose → render → optionally write WAV.

        Returns (score, render_integrity).
        """
        from synth.synth import SynthConfig

        score = self.compose()
        mood = self.composition.mood

        synth_config = SynthConfig(
            bpm=score.bpm,
            length=score.total_beats * 60.0 / score.bpm if score.bpm > 0 else 10.0,
            seed=score.seed,
            waveform=mood.waveform,
            output_dir=str(Path(output_path).parent) if output_path else "output",
        )
        engine = engine_class(synth_config)
        self.render_to_engine(engine)
        integrity = engine.render(output_path=output_path)
        return score, integrity


def score_fingerprint(score: Score) -> str:
    """SHA-256 fingerprint of a score's canonical JSON representation.

    Two scores with identical musical content produce the same fingerprint.
    """
    canonical = json.dumps(score_to_dict(score), sort_keys=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
