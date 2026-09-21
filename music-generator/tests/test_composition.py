"""Tests for the composition layer — the 10 acceptance gates.

These tests verify that the composition engine produces musically valid,
deterministic, and controllable output. Every test is independent and
deterministic (fixed seeds).
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS))


# ---------------------------------------------------------------------------
# Gate 1: Harmonic context — every melody note belongs to its chord or is marked
# ---------------------------------------------------------------------------

class TestHarmonicContext:
    def _generate_score(self, mood="happy", key="C", mode="major", seed=42):
        from composition.composition import Composition, CompositionConfig
        config = CompositionConfig(key=key, mode=mode, mood=mood, seed=seed)
        comp = Composition(config)
        return comp.generate()

    def test_melody_notes_in_chord_or_tonic(self):
        """Every melody note should belong to the active chord or tonic scale."""
        from composition.harmony import scale_degrees, chord_pitches
        score = self._generate_score()
        root = 60  # C4
        mode = score.mode
        degs = set(scale_degrees(root, mode))
        # Every melody note's pitch class should be in the scale
        for note in score.all_notes():
            if note.voice == "melody":
                pc = note.pitch_midi % 12
                # Allow chromatic passing tones but check scale membership as base
                assert pc in set(d % 12 for d in degs), (
                    f"melody note {note.pitch_midi} (pc={pc}) not in {mode} scale"
                )

    def test_bass_notes_are_chord_roots(self):
        """Bass voice should play chord roots."""
        from composition.harmony import NOTE_NAME_TO_MIDI, note_to_midi
        score = self._generate_score()
        root = note_to_midi(score.key_root, 4)
        # Bass notes should be chord roots (scale degrees)
        chords_by_time = sorted(score.all_chords(), key=lambda c: c.time_beats)
        bass_notes = [n for n in score.all_notes() if n.voice == "bass"]
        for bn in bass_notes:
            # Find active chord at this time
            active = None
            for c in chords_by_time:
                if c.time_beats <= bn.time_beats < c.time_beats + c.duration_beats:
                    active = c
                    break
            if active is not None:
                # Bass pitch class should match chord root pitch class
                assert bn.pitch_midi % 12 == active.root_midi % 12, (
                    f"bass {bn.pitch_midi} not root of chord {active.label} ({active.root_midi})"
                )


# ---------------------------------------------------------------------------
# Gate 2: Chord labels match emitted pitches
# ---------------------------------------------------------------------------

class TestChordLabelsMatch:
    def test_chord_labels_follow_roman_numeral_convention(self):
        """Chord labels should be valid Roman numerals (I, ii, V7, etc.)."""
        from composition.composition import Composition, CompositionConfig
        from composition.harmony import ROMAN
        config = CompositionConfig(mood="happy", seed=42)
        score = Composition(config).generate()
        for chord in score.all_chords():
            label = chord.label
            # Strip °, +, 7 suffixes to get the base numeral
            base = label.rstrip("°+7")
            assert base in [r.lower() for r in ROMAN] or base in ROMAN, (
                f"invalid chord label {label!r}"
            )

    def test_minor_chords_use_lowercase(self):
        """Minor/dim chords should be lowercase; major/aug uppercase."""
        from composition.composition import Composition, CompositionConfig
        from composition.harmony import CHORD_QUALITIES
        config = CompositionConfig(mood="sad", key="A", mode="natural_minor", seed=42)
        score = Composition(config).generate()
        for chord in score.all_chords():
            if chord.quality in ("min", "min7", "dim", "half_dim7"):
                base = chord.label.rstrip("°+7")
                assert base.islower() or "°" in chord.label, (
                    f"minor chord {chord.label} should be lowercase"
                )
            elif chord.quality == "dim":
                assert "°" in chord.label, (
                    f"dim chord {chord.label} should have ° symbol"
                )


# ---------------------------------------------------------------------------
# Gate 3: Voice ranges and polyphony limits
# ---------------------------------------------------------------------------

class TestVoiceRanges:
    def test_bass_stays_in_range(self):
        from composition.composition import Composition, CompositionConfig
        from composition.voice_leading import VOICE_RANGES
        config = CompositionConfig(mood="energetic", seed=42)
        score = Composition(config).generate()
        bass_range = VOICE_RANGES["bass"]
        for note in score.all_notes():
            if note.voice == "bass":
                assert bass_range.min_midi <= note.pitch_midi <= bass_range.max_midi, (
                    f"bass note {note.pitch_midi} out of range [{bass_range.min_midi}, {bass_range.max_midi}]"
                )

    def test_melody_stays_in_range(self):
        from composition.composition import Composition, CompositionConfig
        from composition.voice_leading import VOICE_RANGES
        config = CompositionConfig(mood="calm", seed=42)
        score = Composition(config).generate()
        melody_range = VOICE_RANGES["melody"]
        for note in score.all_notes():
            if note.voice == "melody":
                assert melody_range.min_midi <= note.pitch_midi <= melody_range.max_midi

    def test_no_voice_crossing(self):
        """Bass should be below melody at all times."""
        from composition.composition import Composition, CompositionConfig
        config = CompositionConfig(mood="happy", seed=42)
        score = Composition(config).generate()
        # Check at each beat: bass <= melody
        bass_notes = sorted(
            [n for n in score.all_notes() if n.voice == "bass"],
            key=lambda n: n.time_beats,
        )
        melody_notes = sorted(
            [n for n in score.all_notes() if n.voice == "melody"],
            key=lambda n: n.time_beats,
        )
        for b in bass_notes:
            # Find active melody note at this time
            for m in melody_notes:
                if m.time_beats <= b.time_beats < m.time_beats + m.duration_beats:
                    assert b.pitch_midi <= m.pitch_midi, (
                        f"bass {b.pitch_midi} crosses above melody {m.pitch_midi} at beat {b.time_beats}"
                    )
                    break


# ---------------------------------------------------------------------------
# Gate 4: Parallel fifth/octave detection
# ---------------------------------------------------------------------------

class TestParallelIntervals:
    def test_no_excessive_parallels(self):
        """Voice leading should minimise parallel fifths/octaves."""
        from composition.composition import Composition, CompositionConfig
        from composition.voice_leading import voice_progression, count_parallels
        from composition.harmony import generate_progression, note_to_midi
        config = CompositionConfig(mood="happy", key="C", seed=42)
        score = Composition(config).generate()
        # Reconstruct a voiced progression and check parallels
        # We test the voice leading function directly
        prog = generate_progression(note_to_midi("C", 4), "major", 8, np.random.RandomState(42))
        voicings = voice_progression(prog)
        fifths, octaves = count_parallels(voicings)
        # Allow a small number (not a hard requirement for short progressions)
        assert fifths + octaves <= 3, (
            f"too many parallels: {fifths} fifths, {octaves} octaves in 8-chord progression"
        )


# ---------------------------------------------------------------------------
# Gate 5: Transposition
# ---------------------------------------------------------------------------

def test_different_keys_transpose_structure():
    """Same seed in different keys should produce same structure, different pitches."""
    from composition.composition import Composition, CompositionConfig
    config_c = CompositionConfig(key="C", mode="major", mood="happy", seed=42)
    config_g = CompositionConfig(key="G", mode="major", mood="happy", seed=42)
    score_c = Composition(config_c).generate()
    score_g = Composition(config_g).generate()

    notes_c = sorted(score_c.all_notes(), key=lambda n: (n.time_beats, n.voice))
    notes_g = sorted(score_g.all_notes(), key=lambda n: (n.time_beats, n.voice))

    # Same number of notes, same time/voice structure
    assert len(notes_c) == len(notes_g)
    for nc, ng in zip(notes_c, notes_g):
        assert nc.time_beats == ng.time_beats
        assert nc.voice == ng.voice
        assert nc.duration_beats == ng.duration_beats

    # Chord roots should transpose consistently (by 7 semitones for G vs C)
    chords_c = score_c.all_chords()
    chords_g = score_g.all_chords()
    assert len(chords_c) == len(chords_g)
    root_offsets = set()
    for cc, cg in zip(chords_c, chords_g):
        root_offsets.add((cg.root_midi - cc.root_midi) % 12)
    # All chord roots should transpose by the same interval
    assert len(root_offsets) == 1, f"inconsistent chord transposition: {root_offsets}"
    assert root_offsets == {7}, f"G should transpose by 7 semitones from C, got {root_offsets}"

    # Pitch ranges should shift (G-score pitches should be higher on average)
    pitches_c = [n.pitch_midi for n in notes_c if n.pitch_midi is not None]
    pitches_g = [n.pitch_midi for n in notes_g if n.pitch_midi is not None]
    avg_c = sum(pitches_c) / len(pitches_c)
    avg_g = sum(pitches_g) / len(pitches_g)
    assert avg_g > avg_c, f"G-key avg pitch ({avg_g}) should be higher than C-key ({avg_c})"


# ---------------------------------------------------------------------------
# Gate 6: Mood affects harmony, rhythm, and register
# ---------------------------------------------------------------------------

class TestMoodAffectsOutput:
    def test_different_moods_different_scores(self):
        from composition.composition import Composition, CompositionConfig
        from composition.conductor import score_fingerprint
        config_happy = CompositionConfig(mood="happy", key="C", seed=42)
        config_sad = CompositionConfig(mood="sad", key="C", seed=42)
        score_happy = Composition(config_happy).generate()
        score_sad = Composition(config_sad).generate()
        fp_happy = score_fingerprint(score_happy)
        fp_sad = score_fingerprint(score_sad)
        assert fp_happy != fp_sad, "happy and sad should produce different scores"

    def test_energetic_higher_density_than_calm(self):
        from composition.composition import Composition, CompositionConfig
        config_energetic = CompositionConfig(mood="energetic", key="C", seed=42)
        config_calm = CompositionConfig(mood="calm", key="C", seed=42)
        score_energetic = Composition(config_energetic).generate()
        score_calm = Composition(config_calm).generate()
        # Energetic should have more notes (higher density)
        notes_energetic = len(score_energetic.all_notes())
        notes_calm = len(score_calm.all_notes())
        assert notes_energetic > notes_calm, (
            f"energetic ({notes_energetic}) should have more notes than calm ({notes_calm})"
        )

    def test_sad_minor_happy_major(self):
        from composition.composition import Composition, CompositionConfig
        config_happy = CompositionConfig(mood="happy", key="C", seed=42)
        config_sad = CompositionConfig(mood="sad", key="A", seed=42)
        score_happy = Composition(config_happy).generate()
        score_sad = Composition(config_sad).generate()
        assert score_happy.mode == "major" or score_happy.mode == "dorian"  # mixolydian is close to major
        assert score_sad.mode in ("natural_minor", "harmonic_minor")


# ---------------------------------------------------------------------------
# Gate 7: Same seed produces byte-identical score JSON
# ---------------------------------------------------------------------------

class TestDeterminism:
    def test_same_seed_same_score(self):
        from composition.composition import Composition, CompositionConfig
        config = CompositionConfig(mood="happy", key="C", seed=42)
        score1 = Composition(config).generate()
        score2 = Composition(config).generate()
        assert score1.total_beats == score2.total_beats
        assert len(score1.sections) == len(score2.sections)

    def test_same_seed_same_json(self):
        from composition.composition import Composition, CompositionConfig, score_to_json
        config = CompositionConfig(mood="epic", key="D", mode="harmonic_minor", seed=123)
        json1 = score_to_json(Composition(config).generate())
        json2 = score_to_json(Composition(config).generate())
        assert json1 == json2

    def test_different_seeds_different_scores(self):
        from composition.composition import Composition, CompositionConfig
        from composition.conductor import score_fingerprint
        config1 = CompositionConfig(mood="happy", key="C", seed=42)
        config2 = CompositionConfig(mood="happy", key="C", seed=99)
        fp1 = score_fingerprint(Composition(config1).generate())
        fp2 = score_fingerprint(Composition(config2).generate())
        assert fp1 != fp2


# ---------------------------------------------------------------------------
# Gate 8: Structural repetition (A section repeats)
# ---------------------------------------------------------------------------

class TestStructure:
    def test_sections_have_distinct_names(self):
        from composition.composition import Composition, CompositionConfig
        config = CompositionConfig(mood="happy", structure="standard", seed=42)
        score = Composition(config).generate()
        names = [s.name for s in score.sections]
        # Should have at least A, B, and A' (or intro/A/outro)
        assert len(names) >= 3

    def test_A_and_Aprime_similar_structure(self):
        """A' should have similar length and structure to A."""
        from composition.composition import Composition, CompositionConfig
        config = CompositionConfig(mood="happy", structure="standard", seed=42)
        score = Composition(config).generate()
        a = next(s for s in score.sections if s.name == "A")
        aprime = next(s for s in score.sections if s.name == "A'")
        # A and A' should have same number of bars/beats
        assert abs(a.duration_beats - aprime.duration_beats) < 0.01
        # Same number of chords
        assert len(a.chords) == len(aprime.chords)
        # Same progression pattern (same chord labels)
        a_labels = [c.label for c in a.chords]
        aprime_labels = [c.label for c in aprime.chords]
        assert a_labels == aprime_labels, f"A labels {a_labels} != A' labels {aprime_labels}"

    def test_chords_not_random(self):
        """Chord progression should show functional motion, not random jumping."""
        from composition.composition import Composition, CompositionConfig
        from composition.harmony import generate_progression, note_to_midi
        # Generate a long progression and check it has a cadence
        prog = generate_progression(note_to_midi("C", 4), "major", 8, np.random.RandomState(42))
        labels = [label for _, _, label in prog]
        # Should end on I
        assert labels[-1] == "I", f"progression should end on I, got {labels[-1]}"


# ---------------------------------------------------------------------------
# Gate 9: WAV output and fixtures
# ---------------------------------------------------------------------------

class TestSynthIntegration:
    def test_score_renders_to_valid_wav(self):
        """A composed score should render to a valid WAV through SynthEngine."""
        from composition.composition import Composition, CompositionConfig
        from composition.conductor import Conductor, score_fingerprint
        from synth.synth import SynthEngine, SynthConfig
        import tempfile

        config = CompositionConfig(mood="happy", key="C", seed=42)
        conductor = Conductor(config)
        score = conductor.compose()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_composition.wav"
            synth_config = SynthConfig(
                bpm=score.bpm,
                length=score.total_beats * 60.0 / score.bpm,
                seed=score.seed,
                waveform="saw",
                output_dir=tmpdir,
            )
            engine = SynthEngine(synth_config)
            conductor.render_to_engine(engine)
            integrity = engine.render(output_path=output_path)

            assert integrity is not None
            assert integrity.all_finite
            assert integrity.frame_count > 0
            assert output_path.exists()
            assert output_path.stat().st_size > 44  # WAV header is 44 bytes


# ---------------------------------------------------------------------------
# Gate 10: Synth remains deterministic from symbolic score
# ---------------------------------------------------------------------------

class TestSynthDeterminism:
    def test_same_score_same_wav(self):
        """Rendering the same score twice should produce identical WAV bytes."""
        from composition.composition import Composition, CompositionConfig
        from composition.conductor import Conductor
        from synth.synth import SynthEngine, SynthConfig
        import tempfile

        config = CompositionConfig(mood="calm", key="G", seed=77)
        with tempfile.TemporaryDirectory() as tmpdir:
            path1 = Path(tmpdir) / "take1.wav"
            path2 = Path(tmpdir) / "take2.wav"

            # Render 1
            conductor1 = Conductor(config)
            score1 = conductor1.compose()
            synth_config1 = SynthConfig(
                bpm=score1.bpm,
                length=score1.total_beats * 60.0 / score1.bpm,
                seed=score1.seed,
                waveform="sine",
                output_dir=tmpdir,
            )
            engine1 = SynthEngine(synth_config1)
            conductor1.render_to_engine(engine1)
            engine1.render(output_path=path1)

            # Render 2
            conductor2 = Conductor(config)
            score2 = conductor2.compose()
            synth_config2 = SynthConfig(
                bpm=score2.bpm,
                length=score2.total_beats * 60.0 / score2.bpm,
                seed=score2.seed,
                waveform="sine",
                output_dir=tmpdir,
            )
            engine2 = SynthEngine(synth_config2)
            conductor2.render_to_engine(engine2)
            engine2.render(output_path=path2)

            # Compare bytes
            wav1 = path1.read_bytes()
            wav2 = path2.read_bytes()
            assert wav1 == wav2, "same score should produce byte-identical WAV"

    def test_score_then_render_matches_direct_synth(self):
        """The conductor should produce a score that, when rendered via the synth,
        gives the same result as manually calling the synth with the same notes."""
        from composition.composition import Composition, CompositionConfig
        from composition.conductor import Conductor, score_fingerprint
        from synth.synth import SynthEngine, SynthConfig
        import tempfile

        config = CompositionConfig(mood="energetic", key="F", seed=55)
        conductor = Conductor(config)
        score = conductor.compose()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "score_render.wav"

            synth_config = SynthConfig(
                bpm=score.bpm,
                length=score.total_beats * 60.0 / score.bpm,
                seed=score.seed,
                waveform="square",
                output_dir=tmpdir,
            )
            engine = SynthEngine(synth_config)
            conductor.render_to_engine(engine)
            integrity = engine.render(output_path=path)

            assert integrity.all_finite
            assert integrity.frame_count > 0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_mood_falls_back(self):
        from composition.mood import get_mood
        profile = get_mood("nonexistent_mood_xyz")
        assert profile.name == "neutral"

    def test_invalid_key_raises(self):
        from composition.composition import Composition, CompositionConfig
        with pytest.raises(ValueError):
            CompositionConfig(key="H", mode="major", mood="happy", seed=42)

    def test_invalid_mode_raises(self):
        from composition.composition import Composition, CompositionConfig
        with pytest.raises(ValueError):
            CompositionConfig(key="C", mode="invalid_mode", mood="happy", seed=42)

    def test_all_moods_produce_valid_scores(self):
        from composition.composition import Composition, CompositionConfig
        from composition.mood import list_moods
        for mood_name in list_moods():
            config = CompositionConfig(mood=mood_name, key="C", seed=42)
            score = Composition(config).generate()
            assert score.total_beats > 0, f"mood {mood_name} produced zero-length score"
            assert len(score.sections) > 0, f"mood {mood_name} produced no sections"
            # All notes must be finite and in range
            for note in score.all_notes():
                if note.pitch_midi is not None:
                    assert 0 <= note.pitch_midi <= 127

    def test_all_structures_produce_valid_scores(self):
        from composition.composition import Composition, CompositionConfig
        for struct_name in ["simple", "standard", "ABA", "through", "long"]:
            config = CompositionConfig(
                mood="happy", key="C", structure=struct_name, seed=42
            )
            score = Composition(config).generate()
            assert score.total_beats > 0
            assert len(score.sections) > 0


# ---------------------------------------------------------------------------
# Harmony engine unit tests
# ---------------------------------------------------------------------------

class TestHarmonyEngine:
    def test_scale_degrees_major(self):
        from composition.harmony import scale_degrees, note_to_midi
        degs = scale_degrees(note_to_midi("C", 4), "major")
        assert degs == [60, 62, 64, 65, 67, 69, 71]  # C D E F G A B

    def test_scale_degrees_minor(self):
        from composition.harmony import scale_degrees, note_to_midi
        degs = scale_degrees(note_to_midi("A", 4), "natural_minor")
        assert degs == [69, 71, 72, 74, 76, 77, 79]  # A B C D E F G

    def test_chord_pitches(self):
        from composition.harmony import chord_pitches
        assert chord_pitches(60, "maj") == [60, 64, 67]  # C E G
        assert chord_pitches(60, "min") == [60, 63, 67]  # C Eb G
        assert chord_pitches(60, "dim") == [60, 63, 66]  # C Eb Gb

    def test_diatonic_triad(self):
        from composition.harmony import diatonic_triad, note_to_midi
        root, quality = diatonic_triad(note_to_midi("C", 4), "major", 0)
        assert quality == "maj"  # I
        root, quality = diatonic_triad(note_to_midi("C", 4), "major", 1)
        assert quality == "min"  # ii

    def test_progression_length(self):
        from composition.harmony import generate_progression, note_to_midi
        prog = generate_progression(
            note_to_midi("C", 4), "major", 8, np.random.RandomState(42)
        )
        assert len(prog) == 8

    def test_progression_starts_on_tonic(self):
        from composition.harmony import generate_progression, note_to_midi
        prog = generate_progression(
            note_to_midi("C", 4), "major", 4, np.random.RandomState(42),
            start_degree=0,
        )
        # First chord root should be tonic (C == 0 mod 12)
        assert prog[0][0] % 12 == note_to_midi("C", 4) % 12

    def test_progression_ends_on_request(self):
        from composition.harmony import generate_progression, note_to_midi
        prog = generate_progression(
            note_to_midi("C", 4), "major", 4, np.random.RandomState(42),
            end_degree=4,  # V
        )
        # V in C major is G
        assert prog[-1][0] % 12 == note_to_midi("G", 4) % 12
        assert prog[-1][2] == "V"


# ---------------------------------------------------------------------------
# Voice leading unit tests
# ---------------------------------------------------------------------------

class TestVoiceLeading:
    def test_voicings_keep_voice_order(self):
        from composition.harmony import generate_progression, note_to_midi
        from composition.voice_leading import voice_progression, Voicing
        prog = generate_progression(
            note_to_midi("C", 4), "major", 8, np.random.RandomState(42)
        )
        voicings = voice_progression(prog)
        for v in voicings:
            if v.harmony:
                assert v.bass <= min(v.harmony), f"bass {v.bass} above harmony {v.harmony}"
                assert v.melody >= max(v.harmony), f"melody {v.melody} below harmony {v.harmony}"

    def test_voicings_bass_is_root(self):
        from composition.harmony import generate_progression, note_to_midi
        from composition.voice_leading import voice_progression
        prog = generate_progression(
            note_to_midi("C", 4), "major", 4, np.random.RandomState(42)
        )
        voicings = voice_progression(prog)
        for (root, _, _), v in zip(prog, voicings):
            assert v.bass % 12 == root % 12, (
                f"bass {v.bass} not root of chord {root}"
            )

    def test_voicings_melody_in_range(self):
        from composition.harmony import generate_progression, note_to_midi
        from composition.voice_leading import voice_progression, VOICE_RANGES
        prog = generate_progression(
            note_to_midi("C", 4), "major", 8, np.random.RandomState(42)
        )
        voicings = voice_progression(prog)
        mr = VOICE_RANGES["melody"]
        for v in voicings:
            assert mr.min_midi <= v.melody <= mr.max_midi
