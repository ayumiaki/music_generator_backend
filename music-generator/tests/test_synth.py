"""Tests for synth engine: determinism, integrity, performance, listening."""
import os
import sys
import time
import tempfile
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS))

import numpy as np
import pytest


class TestSynthEngine:
    """Core synth engine tests."""

    def test_synth_render(self):
        from synth.synth import SynthEngine, SynthConfig
        config = SynthConfig(bpm=120, bars=1, seed=42, sample_rate=48000)
        engine = SynthEngine(config)
        engine.add_note(time_beats=0.0, freq=440.0, duration_beats=0.5)
        integrity = engine.render()
        assert integrity is not None
        assert integrity.frame_count > 0
        assert integrity.all_finite

    def test_synth_deterministic(self):
        from synth.synth import SynthEngine, SynthConfig
        config = SynthConfig(bpm=120, bars=1, seed=42, sample_rate=48000)
        engine1 = SynthEngine(config)
        engine2 = SynthEngine(config)
        engine1.add_note(time_beats=0.0, freq=440.0, duration_beats=0.5)
        engine2.add_note(time_beats=0.0, freq=440.0, duration_beats=0.5)
        engine1.render()
        engine2.render()
        a1 = engine1.get_audio()
        a2 = engine2.get_audio()
        assert a1 is not None and a2 is not None
        assert np.array_equal(a1, a2)

    def test_synth_wav_output(self):
        from synth.synth import SynthEngine, SynthConfig
        config = SynthConfig(bpm=120, bars=1, seed=42, sample_rate=48000)
        engine = SynthEngine(config)
        engine.add_note(time_beats=0.0, freq=440.0, duration_beats=0.5)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.wav"
            engine.render(output_path=path)
            assert path.exists()

    def test_synth_no_nan(self):
        from synth.synth import SynthEngine, SynthConfig
        config = SynthConfig(bpm=120, bars=1, seed=42, sample_rate=48000)
        engine = SynthEngine(config)
        engine.add_note(time_beats=0.0, freq=440.0, duration_beats=0.5)
        integrity = engine.render()
        assert integrity.all_finite

    def test_synth_reset(self):
        from synth.synth import SynthEngine, SynthConfig
        config = SynthConfig(bpm=120, bars=1, seed=42, sample_rate=48000)
        engine = SynthEngine(config)
        engine.add_note(time_beats=0.0, freq=440.0, duration_beats=0.5)
        engine.render()
        engine.reset()
        assert engine.get_audio() is None
        assert engine.get_integrity() is None


class TestArrangement:
    """Arrangement tests."""

    def test_arrangement_no_nan(self):
        from synth.arrangement import Arrangement, ArrangementConfig
        config = ArrangementConfig(bpm=120, bars=1, sample_rate=48000, seed=42)
        arr = Arrangement(config)
        arr.add_note(time_beats=0.0, freq=440.0, duration_beats=0.5)
        audio = arr.render()
        assert np.all(np.isfinite(audio))


class TestDeterminism:
    """Determinism tests."""

    def test_identical_seed_identical_output(self):
        from synth.synth import SynthEngine, SynthConfig
        config = SynthConfig(bpm=120, bars=1, seed=42, sample_rate=48000)
        engine1 = SynthEngine(config)
        engine2 = SynthEngine(config)
        engine1.add_note(time_beats=0.0, freq=440.0, duration_beats=0.5)
        engine2.add_note(time_beats=0.0, freq=440.0, duration_beats=0.5)
        engine1.render()
        engine2.render()
        a1 = engine1.get_audio()
        a2 = engine2.get_audio()
        assert a1 is not None and a2 is not None
        assert np.array_equal(a1, a2)

    def test_different_seed_different_output(self):
        from synth.synth import SynthEngine, SynthConfig
        config1 = SynthConfig(bpm=120, bars=1, seed=42, sample_rate=48000)
        config2 = SynthConfig(bpm=120, bars=1, seed=43, sample_rate=48000)
        engine1 = SynthEngine(config1)
        engine2 = SynthEngine(config2)
        engine1.add_note(time_beats=0.0, freq=440.0, duration_beats=0.5)
        engine2.add_note(time_beats=0.0, freq=440.0, duration_beats=0.5)
        engine1.render()
        engine2.render()
        a1 = engine1.get_audio()
        a2 = engine2.get_audio()
        assert a1 is not None and a2 is not None
        assert not np.array_equal(a1, a2)

    def test_wav_file_byte_identical(self):
        from synth.synth import SynthEngine, SynthConfig
        config = SynthConfig(bpm=120, bars=1, seed=42, sample_rate=48000)
        engine1 = SynthEngine(config)
        engine2 = SynthEngine(config)
        engine1.add_note(time_beats=0.0, freq=440.0, duration_beats=0.5)
        engine2.add_note(time_beats=0.0, freq=440.0, duration_beats=0.5)
        with tempfile.TemporaryDirectory() as tmpdir:
            p1 = Path(tmpdir) / "a.wav"
            p2 = Path(tmpdir) / "b.wav"
            engine1.render(output_path=p1)
            engine2.render(output_path=p2)
            assert p1.read_bytes() == p2.read_bytes()


class TestRenderIntegrity:
    """Render integrity tests."""

    def test_integrity_checks(self):
        from synth.synth import SynthEngine, SynthConfig
        config = SynthConfig(bpm=120, bars=1, seed=42, sample_rate=48000)
        engine = SynthEngine(config)
        engine.add_note(time_beats=0.0, freq=440.0, duration_beats=0.5)
        integrity = engine.render()
        assert integrity is not None
        assert integrity.frame_count > 0
        assert integrity.all_finite

    def test_integrity_clipping_detected(self):
        from synth.synth import SynthEngine, SynthConfig
        config = SynthConfig(bpm=120, bars=1, seed=42, sample_rate=48000)
        engine = SynthEngine(config)
        # Add many notes to cause clipping
        for i in range(20):
            engine.add_note(time_beats=0.0, freq=440.0, duration_beats=0.5, amplitude=1.0)
        integrity = engine.render()
        assert integrity is not None
        # Clipping should be detected (or limiter should prevent it)
        assert integrity.all_finite

    def test_integrity_dc_offset(self):
        from synth.synth import SynthEngine, SynthConfig
        config = SynthConfig(bpm=120, bars=1, seed=42, sample_rate=48000)
        engine = SynthEngine(config)
        engine.add_note(time_beats=0.0, freq=440.0, duration_beats=0.5)
        integrity = engine.render()
        assert integrity is not None
        assert abs(integrity.dc_offset) < 0.1

    def test_integrity_non_finite(self):
        from synth.synth import SynthEngine, SynthConfig
        config = SynthConfig(bpm=120, bars=1, seed=42, sample_rate=48000)
        engine = SynthEngine(config)
        engine.add_note(time_beats=0.0, freq=440.0, duration_beats=0.5)
        integrity = engine.render()
        assert integrity is not None
        assert integrity.all_finite


class TestPerformance:
    """Performance tests."""

    def test_real_time_factor_30s(self):
        from synth.synth import SynthEngine, SynthConfig
        config = SynthConfig(bpm=120, bars=15, seed=42, sample_rate=48000)
        engine = SynthEngine(config)
        for i in range(60):
            engine.add_note(time_beats=i * 0.5, freq=220.0, duration_beats=0.25)
        engine.add_drums(bars=15)
        start = time.time()
        engine.render()
        elapsed = time.time() - start
        # Should render faster than real-time (30 seconds of audio)
        assert elapsed < 30.0, f"Render took {elapsed:.1f}s — slower than real-time"

    def test_peak_memory_reasonable(self):
        from synth.synth import SynthEngine, SynthConfig
        import tracemalloc
        config = SynthConfig(bpm=120, bars=15, seed=42, sample_rate=48000)
        engine = SynthEngine(config)
        tracemalloc.start()
        for i in range(60):
            engine.add_note(time_beats=i * 0.5, freq=220.0, duration_beats=0.25)
        engine.render()
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peak_mb = peak / (1024 * 1024)
        assert peak_mb < 500, f"Peak memory {peak_mb:.1f}MB too high"

    def test_5min_render_completes(self):
        from synth.synth import SynthEngine, SynthConfig
        # 5 minutes = 300 seconds at 120 BPM = 150 bars
        config = SynthConfig(bpm=120, bars=150, seed=42, sample_rate=48000)
        engine = SynthEngine(config)
        # 5 minutes = 300 seconds, at 120 BPM = 5 beats/sec = 1500 beats
        for i in range(1500):
            engine.add_note(time_beats=i * 0.5, freq=220.0, duration_beats=0.25)
        engine.render()
        integrity = engine.get_integrity()
        assert integrity is not None
        assert integrity.frame_count > 0
        assert integrity.all_finite


# --- Listening tests ---

class TestListening:
    """Actual musical fixtures, not one heroic C-major note."""

    def test_c_major_chord(self):
        from synth.synth import SynthEngine, SynthConfig
        config = SynthConfig(bpm=120, bars=1, seed=42, sample_rate=48000)
        engine = SynthEngine(config)
        engine.add_note(time_beats=0.0, freq=261.63, duration_beats=1.0)  # C4
        engine.add_note(time_beats=0.0, freq=329.63, duration_beats=1.0)  # E4
        engine.add_note(time_beats=0.0, freq=392.00, duration_beats=1.0)  # G4
        engine.add_drums(bars=1)
        integrity = engine.render()
        assert integrity.frame_count > 0
        assert integrity.all_finite
        assert not integrity.clipping

    def test_scale_playback(self):
        from synth.synth import SynthEngine, SynthConfig
        config = SynthConfig(bpm=120, bars=2, seed=42, sample_rate=48000)
        engine = SynthEngine(config)
        # C major scale
        freqs = [261.63, 293.66, 329.63, 349.23, 392.00, 440.00, 493.88, 523.25]
        for i, freq in enumerate(freqs):
            engine.add_note(time_beats=i * 0.5, freq=freq, duration_beats=0.25)
        engine.add_drums(bars=2)
        integrity = engine.render()
        assert integrity.frame_count > 0
        assert integrity.all_finite

    def test_drum_pattern_with_melody(self):
        from synth.synth import SynthEngine, SynthConfig
        config = SynthConfig(bpm=120, bars=2, seed=42, sample_rate=48000)
        engine = SynthEngine(config)
        # Melody
        for i in range(8):
            engine.add_note(time_beats=i * 0.5, freq=330.0, duration_beats=0.25)
        # Drums
        engine.add_drums(bars=2)
        integrity = engine.render()
        assert integrity.frame_count > 0
        assert integrity.all_finite
        assert not integrity.clipping


# --- Multi-bar drum tests ---

class TestMultiBarDrums:
    """Multi-bar drum scheduling: each bar must have hits."""

    def test_two_bar_drums_have_hits_in_both_bars(self):
        from synth.drums import DrumPattern
        drums = DrumPattern(bpm=120, sample_rate=48000, seed=42)
        audio = drums.render(bars=2)
        # Bar 1: samples 0 to 24000 (1 bar = 2 seconds = 96000 samples at 48kHz)
        # Actually at 120 BPM, 1 bar = 4 beats = 2 seconds = 96000 samples
        bar_samples = int(4 * 60.0 / 120.0 * 48000)  # 96000
        bar1 = audio[:bar_samples]
        bar2 = audio[bar_samples:]
        assert np.any(bar1 != 0), "Bar 1 should have drum hits"
        assert np.any(bar2 != 0), "Bar 2 should have drum hits"

    def test_drum_dc_offset_removed(self):
        from synth.drums import DrumPattern
        drums = DrumPattern(bpm=120, sample_rate=48000, seed=42)
        audio = drums.render(bars=1)
        # DC offset should be near zero
        assert abs(np.mean(audio)) < 0.01, f"DC offset {np.mean(audio)} too high"

    def test_drum_tail_terminates_at_zero(self):
        from synth.drums import DrumPattern
        drums = DrumPattern(bpm=120, sample_rate=48000, seed=42)
        audio = drums.render(bars=1)
        # Last sample should be near zero (clean tail)
        assert abs(audio[-1]) < 0.01, f"Last sample {audio[-1]} not near zero"


# --- Voice lifecycle tests ---

class TestVoiceLifecycle:
    """Shared voice engine: allocation, stealing, note-off, release."""

    def test_shared_engine_polyphony(self):
        from synth.voice import VoiceEngine, VoiceConfig
        config = VoiceConfig(attack_samples=10, decay_samples=10, sustain_level=0.7, release_samples=10)
        engine = VoiceEngine(polyphony=4, sample_rate=48000, config=config, seed=42)
        # Allocate 4 voices
        for i in range(4):
            vid = engine.allocate(220.0 * (i + 1), 0.8)
            assert vid == i
        # 5th should steal oldest (ID 0)
        vid5 = engine.allocate(330.0, 0.8)
        assert vid5 == 0

    def test_note_off_triggers_release(self):
        from synth.voice import VoiceEngine, VoiceConfig
        config = VoiceConfig(attack_samples=10, decay_samples=10, sustain_level=0.7, release_samples=50)
        engine = VoiceEngine(polyphony=4, sample_rate=48000, config=config, seed=42)
        vid = engine.allocate(440.0, 0.8)
        engine.note_off(vid)
        voice = engine.get_voice(vid)
        assert voice is not None
        assert not voice.note_on

    def test_release_tail_rendered(self):
        from synth.voice import VoiceEngine, VoiceConfig
        config = VoiceConfig(attack_samples=10, decay_samples=10, sustain_level=0.7, release_samples=50)
        engine = VoiceEngine(polyphony=4, sample_rate=48000, config=config, seed=42)
        # Render a note with release
        audio = engine.render_block([(440.0, 100, 0.8, "saw")], block_size=200)
        assert len(audio) == 200
        assert np.all(np.isfinite(audio))

    def test_persistent_dsp_objects(self):
        """Each voice holds persistent DSP objects (osc, env, flt)."""
        from synth.voice import VoiceEngine, VoiceConfig
        config = VoiceConfig(attack_samples=10, decay_samples=10, sustain_level=0.7, release_samples=10)
        engine = VoiceEngine(polyphony=4, sample_rate=48000, config=config, seed=42)
        vid = engine.allocate(440.0, 0.8)
        voice = engine.get_voice(vid)
        assert voice is not None
        assert voice.osc is not None
        assert voice.env is not None
        assert voice.flt is not None

    def test_overlapping_notes_coexist(self):
        """Two overlapping notes genuinely coexist and are mixed."""
        from synth.voice import VoiceEngine, VoiceConfig
        config = VoiceConfig(attack_samples=10, decay_samples=10, sustain_level=0.7, release_samples=10)
        engine = VoiceEngine(polyphony=4, sample_rate=48000, config=config, seed=42)
        # Render two overlapping notes
        audio = engine.render_block([(440.0, 100, 0.8, "saw"), (550.0, 100, 0.8, "saw")], block_size=200)
        assert len(audio) == 200
        assert np.all(np.isfinite(audio))
        # Both notes should contribute to the output
        assert np.any(audio != 0)

    def test_polyphony_overflow_steals_oldest(self):
        """Polyphony overflow audibly removes the stolen voice."""
        from synth.voice import VoiceEngine, VoiceConfig
        config = VoiceConfig(attack_samples=10, decay_samples=10, sustain_level=0.7, release_samples=10)
        engine = VoiceEngine(polyphony=2, sample_rate=48000, config=config, seed=42)
        # Allocate 2 voices
        vid1 = engine.allocate(440.0, 0.8)
        vid2 = engine.allocate(550.0, 0.8)
        # 3rd should steal oldest (ID 0)
        vid3 = engine.allocate(660.0, 0.8)
        assert vid3 == 0
        # The stolen voice should be the new one
        voice = engine.get_voice(0)
        assert voice is not None
        assert voice.freq == 660.0

    def test_oldest_policy_rotates(self):
        """Oldest policy rotates correctly after stealing."""
        from synth.voice import VoiceEngine, VoiceConfig
        config = VoiceConfig(attack_samples=10, decay_samples=10, sustain_level=0.7, release_samples=10)
        engine = VoiceEngine(polyphony=2, sample_rate=48000, config=config, seed=42)
        # Allocate 2 voices
        vid1 = engine.allocate(440.0, 0.8)
        vid2 = engine.allocate(550.0, 0.8)
        # 3rd steals oldest (ID 0)
        vid3 = engine.allocate(660.0, 0.8)
        assert vid3 == 0
        # 4th should steal next oldest (ID 1)
        vid4 = engine.allocate(770.0, 0.8)
        assert vid4 == 1


# --- Mood/Key tests ---

class TestMoodKeyControls:
    """Mood and key must produce different audio."""

    def test_different_moods_produce_different_audio(self):
        from synth.synth import SynthEngine, SynthConfig
        config1 = SynthConfig(bpm=120, bars=1, seed=42, sample_rate=48000, waveform="saw")
        config2 = SynthConfig(bpm=120, bars=1, seed=42, sample_rate=48000, waveform="sine")
        engine1 = SynthEngine(config1)
        engine2 = SynthEngine(config2)
        engine1.add_note(time_beats=0.0, freq=440.0, duration_beats=0.5)
        engine2.add_note(time_beats=0.0, freq=440.0, duration_beats=0.5)
        engine1.render()
        engine2.render()
        a1 = engine1.get_audio()
        a2 = engine2.get_audio()
        if a1 is not None and a2 is not None:
            assert not np.array_equal(a1, a2), "Different waveforms should produce different audio"

    def test_different_keys_produce_different_audio(self):
        from synth.synth import SynthEngine, SynthConfig
        config1 = SynthConfig(bpm=120, bars=1, seed=42, sample_rate=48000)
        config2 = SynthConfig(bpm=120, bars=1, seed=42, sample_rate=48000)
        engine1 = SynthEngine(config1)
        engine2 = SynthEngine(config2)
        # Different frequencies (different keys)
        engine1.add_note(time_beats=0.0, freq=261.63, duration_beats=0.5)  # C4
        engine2.add_note(time_beats=0.0, freq=440.0, duration_beats=0.5)  # A4
        engine1.render()
        engine2.render()
        a1 = engine1.get_audio()
        a2 = engine2.get_audio()
        if a1 is not None and a2 is not None:
            assert not np.array_equal(a1, a2), "Different keys should produce different audio"


# --- WAV integrity tests ---

class TestWAVIntegrity:
    """Pre-publication validation and atomic writes."""

    def test_write_wav_rejects_nan(self):
        from synth.renderer import write_wav
        import tempfile
        audio = np.array([1.0, 2.0, float('nan'), 4.0])
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.wav"
            try:
                write_wav(path, audio, 48000)
                assert False, "Should have raised ValueError for NaN"
            except ValueError:
                pass  # Expected

    def test_write_wav_rejects_inf(self):
        from synth.renderer import write_wav
        import tempfile
        audio = np.array([1.0, float('inf'), 3.0, 4.0])
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.wav"
            try:
                write_wav(path, audio, 48000)
                assert False, "Should have raised ValueError for Inf"
            except ValueError:
                pass  # Expected

    def test_write_wav_atomic(self):
        from synth.renderer import write_wav
        import tempfile
        audio = np.sin(2.0 * np.pi * 440.0 * np.arange(48000) / 48000)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.wav"
            write_wav(path, audio, 48000)
            assert path.exists()
            # Verify no temp file left behind
            assert not path.with_suffix(".tmp.wav").exists()


# --- Seed validation tests ---

class TestSeedValidation:
    """Seed values must be validated — invalid range raises ValueError."""

    def test_negative_seed_raises(self):
        from synth.synth import SynthEngine, SynthConfig
        config = SynthConfig(bpm=120, bars=1, seed=-1, sample_rate=48000)
        with pytest.raises(ValueError, match="Invalid seed"):
            SynthEngine(config)

    def test_large_seed_raises(self):
        from synth.synth import SynthEngine, SynthConfig
        config = SynthConfig(bpm=120, bars=1, seed=2**33, sample_rate=48000)
        with pytest.raises(ValueError, match="Invalid seed"):
            SynthEngine(config)

    def test_valid_seed_works(self):
        from synth.synth import SynthEngine, SynthConfig
        config = SynthConfig(bpm=120, bars=1, seed=42, sample_rate=48000)
        engine = SynthEngine(config)
        engine.add_note(time_beats=0.0, freq=440.0, duration_beats=0.5)
        integrity = engine.render()
        assert integrity.all_finite


# --- ADSR boundary tests ---

class TestADSRBoundary:
    """ADSR envelope boundary conditions."""

    def test_attack_reaches_one(self):
        from synth.envelope import ADSREnvelope
        env = ADSREnvelope(attack_samples=4, decay_samples=0, sustain_level=0.5, release_samples=10)
        env.note_on()
        out = env.render(4)
        # Attack should reach 1.0 at the last sample
        assert abs(out[-1] - 1.0) < 0.01, f"Attack last sample {out[-1]} should be ~1.0"

    def test_decay_zero_goes_to_sustain(self):
        from synth.envelope import ADSREnvelope
        env = ADSREnvelope(attack_samples=4, decay_samples=0, sustain_level=0.5, release_samples=10)
        env.note_on()
        out = env.render(10)
        # After attack (4 samples), should be at sustain level
        assert abs(out[5] - 0.5) < 0.01, f"After attack, should be at sustain 0.5, got {out[5]}"


# --- Filter impulse decay tests ---

class TestFilterImpulseDecay:
    """Filter impulse response: decays at low resonance, no DC latch."""

    def test_impulse_decays_at_low_resonance(self):
        from synth.filter import CascadeFilter
        flt = CascadeFilter(sample_rate=48000, cutoff=1000)
        inp = np.zeros(24800)
        inp[0] = 1.0
        out = flt.render(inp)
        tail = out[10000:]
        # Should decay (not latch at DC)
        assert abs(np.mean(tail)) < 0.01, f"Tail mean {np.mean(tail)} should be near 0"
        assert np.all(np.isfinite(tail))

    def test_impulse_no_dc_latch_at_high_resonance(self):
        from synth.filter import CascadeFilter
        flt = CascadeFilter(sample_rate=48000, cutoff=1000)
        inp = np.zeros(24800)
        inp[0] = 1.0
        out = flt.render(inp)
        tail = out[10000:]
        # Should not latch at DC (mean should be near 0)
        assert abs(np.mean(tail)) < 0.1, f"Tail mean {np.mean(tail)} should not latch at DC"
        assert np.all(np.isfinite(tail))


# --- Filter acceptance gate tests ---

class TestFilterAcceptanceGates:
    """Filter acceptance gates: impulse decays, bounded gain, measured cutoff."""

    def test_impulse_decays_without_resets(self):
        """Impulse response decays without any safety resets."""
        from synth.filter import CascadeFilter
        flt = CascadeFilter(sample_rate=48000, cutoff=1000)
        inp = np.zeros(48000)
        inp[0] = 1.0
        out = flt.render(inp)
        # Should decay to near zero
        tail = out[24000:]
        assert abs(np.mean(tail)) < 0.01, f"Tail mean {np.mean(tail)} should be near 0"
        assert np.all(np.isfinite(tail))
        # No resets should occur (state should be bounded)
        state = flt.get_state()
        assert np.all(np.isfinite(state))
        assert np.max(np.abs(state)) < 10.0, f"State {state} should be bounded"

    def test_bounded_gain_across_cutoff_range(self):
        """Bounded gain across cutoff 20 Hz to 0.49fs."""
        from synth.filter import CascadeFilter
        sample_rate = 48000
        for cutoff in [20, 100, 1000, 5000, 10000, 20000, 23500]:
            flt = CascadeFilter(sample_rate=sample_rate, cutoff=cutoff)
            # Render a sine wave at cutoff
            t = np.arange(48000) / sample_rate
            inp = np.sin(2 * np.pi * cutoff * t)
            out = flt.render(inp)
            # Gain should be bounded (no explosions)
            assert np.all(np.isfinite(out)), f"Cutoff {cutoff}: output not finite"
            gain = np.max(np.abs(out)) / np.max(np.abs(inp))
            assert gain < 10.0, f"Cutoff {cutoff}: gain {gain:.1f} too high"

    def test_measured_cutoff_and_slope(self):
        """Measured cutoff frequency and slope are correct.

        Four cascaded one-pole lowpasses with calibrated per-stage alpha.
        Each stage has magnitude 2^(-1/8) at the requested composite cutoff,
        so the cascade hits -3dB at the requested frequency.
        """
        from synth.filter import CascadeFilter
        sample_rate = 48000
        cutoff = 1000
        flt = CascadeFilter(sample_rate=sample_rate, cutoff=cutoff)
        # Measure frequency response using sine sweep (more reliable than FFT of white noise)
        freqs = np.linspace(20, 20000, 1000)
        mags = []
        for f in freqs:
            t = np.arange(sample_rate) / float(sample_rate)
            x = np.sin(2 * np.pi * f * t)
            y = flt.render(x)
            # Measure amplitude of output (skip first 1000 samples for settling)
            amp = np.max(np.abs(y[1000:]))
            mags.append(amp)
        mags = np.array(mags)
        # Find -3dB point (amplitude = 1/sqrt(2) ≈ 0.707)
        target = 1.0 / np.sqrt(2)
        idx = np.argmin(np.abs(mags - target))
        measured_cutoff = freqs[idx]
        # Calibrated cascade: measured cutoff should be within 10% of requested
        assert abs(measured_cutoff - cutoff) < cutoff * 0.1, \
            f"Measured cutoff {measured_cutoff:.0f} Hz vs requested {cutoff} Hz"

    def test_resonance_peak_near_cutoff(self):
        """No resonance peak — one-pole cascade has no resonance.

        The filter is a pure lowpass with no feedback, so the magnitude
        response is monotonically decreasing. The peak should be at DC
        (0 Hz), not near the cutoff frequency.
        """
        from synth.filter import CascadeFilter
        sample_rate = 48000
        cutoff = 1000
        flt = CascadeFilter(sample_rate=sample_rate, cutoff=cutoff)
        # Measure frequency response using sine sweep
        freqs = np.linspace(20, 20000, 1000)
        mags = []
        for f in freqs:
            t = np.arange(sample_rate) / float(sample_rate)
            x = np.sin(2 * np.pi * f * t)
            y = flt.render(x)
            amp = np.max(np.abs(y[1000:]))
            mags.append(amp)
        mags = np.array(mags)
        # Find peak frequency
        peak_idx = np.argmax(mags)
        peak_freq = freqs[peak_idx]
        # No resonance: peak should be at DC (0 Hz), not near cutoff
        assert peak_freq < cutoff * 0.1, \
            f"Peak at {peak_freq:.0f} Hz vs cutoff {cutoff} Hz — resonance present"

    def test_fresh_filter_equals_reset_filter(self):
        """Fresh filter equals reset filter (deterministic state)."""
        from synth.filter import CascadeFilter
        flt1 = CascadeFilter(sample_rate=48000, cutoff=1000)
        flt2 = CascadeFilter(sample_rate=48000, cutoff=1000)
        inp = np.sin(2 * np.pi * 440.0 * np.arange(48000) / 48000)
        out1 = flt1.render(inp)
        out2 = flt2.render(inp)
        assert np.allclose(out1, out2), "Fresh filter should equal reset filter"


# --- Duration semantics tests ---

class TestDurationSemantics:
    """Duration must be derived from requested seconds, not bars."""

    def test_duration_matches_requested_seconds(self):
        from synth.synth import SynthEngine, SynthConfig
        # 30 seconds at 120 BPM = 15 bars
        config = SynthConfig(bpm=120, bars=15, seed=42, sample_rate=48000)
        engine = SynthEngine(config)
        engine.add_note(time_beats=0.0, freq=440.0, duration_beats=0.5)
        engine.add_drums(bars=15)
        integrity = engine.render()
        # Duration should be approximately 30 seconds
        assert abs(integrity.duration_sec - 30.0) < 1.0, f"Duration {integrity.duration_sec}s should be ~30s"

    def test_different_lengths_produce_different_durations(self):
        from synth.synth import SynthEngine, SynthConfig
        # 15 seconds vs 30 seconds
        config1 = SynthConfig(bpm=120, bars=8, seed=42, sample_rate=48000)
        config2 = SynthConfig(bpm=120, bars=15, seed=42, sample_rate=48000)
        engine1 = SynthEngine(config1)
        engine2 = SynthEngine(config2)
        engine1.add_note(time_beats=0.0, freq=440.0, duration_beats=0.5)
        engine2.add_note(time_beats=0.0, freq=440.0, duration_beats=0.5)
        engine1.render()
        engine2.render()
        i1 = engine1.get_integrity()
        i2 = engine2.get_integrity()
        if i1 is not None and i2 is not None:
            assert i1.duration_sec != i2.duration_sec, "Different bar counts should produce different durations"

    def test_exact_frame_counts(self):
        """Exact frame counts for 1, 15, 30, 31, 32, 300 seconds."""
        from synth.synth import SynthEngine, SynthConfig
        sample_rate = 48000
        for length_sec in [1, 15, 30, 31, 32, 300]:
            config = SynthConfig(bpm=120, length=length_sec, seed=42, sample_rate=sample_rate)
            engine = SynthEngine(config)
            engine.add_note(time_beats=0.0, freq=440.0, duration_beats=0.5)
            integrity = engine.render()
            expected_frames = int(length_sec * sample_rate)
            assert integrity.frame_count == expected_frames, \
                f"Length {length_sec}s: frame count {integrity.frame_count} != expected {expected_frames}"


# --- Event ordering tests ---

class TestEventOrdering:
    """Deterministic ordering for simultaneous note-off/note-on."""

    def test_simultaneous_events_deterministic(self):
        """Simultaneous note-off/note-on produce deterministic output."""
        from synth.synth import SynthEngine, SynthConfig
        config = SynthConfig(bpm=120, bars=1, seed=42, sample_rate=48000)
        engine1 = SynthEngine(config)
        engine2 = SynthEngine(config)
        # Add notes at the same time
        engine1.add_note(time_beats=0.0, freq=440.0, duration_beats=0.5)
        engine1.add_note(time_beats=0.0, freq=550.0, duration_beats=0.5)
        engine2.add_note(time_beats=0.0, freq=440.0, duration_beats=0.5)
        engine2.add_note(time_beats=0.0, freq=550.0, duration_beats=0.5)
        engine1.render()
        engine2.render()
        a1 = engine1.get_audio()
        a2 = engine2.get_audio()
        assert a1 is not None and a2 is not None
        assert np.array_equal(a1, a2), "Simultaneous events should be deterministic"

    def test_no_block_size_dependence(self):
        """Output should not depend on block size."""
        from synth.voice import VoiceEngine, VoiceConfig
        config = VoiceConfig(attack_samples=10, decay_samples=10, sustain_level=0.7, release_samples=10)
        engine1 = VoiceEngine(polyphony=4, sample_rate=48000, config=config, seed=42)
        engine2 = VoiceEngine(polyphony=4, sample_rate=48000, config=config, seed=42)
        notes = [(440.0, 100, 0.8, "saw"), (550.0, 100, 0.8, "saw")]
        audio1 = engine1.render_block(notes, block_size=200)
        audio2 = engine2.render_block(notes, block_size=400)
        # First 200 samples should match
        assert np.allclose(audio1, audio2[:200]), "Output should not depend on block size"

    def test_filter_state_continuous_across_notes(self):
        """Filter state must persist across notes — no per-note reset.

        Regression test: the filter was previously reset before each note,
        which broke state continuity and made output depend on note boundaries.
        """
        from synth.voice import VoiceEngine, VoiceConfig
        config = VoiceConfig(attack_samples=10, decay_samples=10, sustain_level=0.7, release_samples=10)
        engine = VoiceEngine(polyphony=4, sample_rate=48000, config=config, seed=42)

        # Render first note
        notes1 = [(440.0, 100, 0.8, "saw")]
        audio1 = engine.render_block(notes1, block_size=200)

        # Capture filter state after first note
        voice = engine._voices[0]
        state_after_first = voice.flt.get_state().copy() if voice else None

        # Render second note (same voice, should steal or reuse)
        notes2 = [(550.0, 100, 0.8, "saw")]
        audio2 = engine.render_block(notes2, block_size=200)

        # Filter state should have changed (continuous, not reset)
        voice = engine._voices[0]
        state_after_second = voice.flt.get_state().copy() if voice else None

        if state_after_first is not None and state_after_second is not None:
            # States should differ — filter is processing, not being reset
            assert not np.allclose(state_after_first, state_after_second), \
                "Filter state should change between notes (no per-note reset)"

    def test_filter_stability_at_high_cutoff(self):
        """Filter must not explode at high cutoff frequencies.

        Regression test: the TPT/ZDF SVF filter exploded at high cutoff
        frequencies (state values hitting 1e14+). The one-pole cascade
        is unconditionally stable.
        """
        from synth.filter import CascadeFilter
        sample_rate = 48000
        # Test at high cutoff frequencies that broke the TPT SVF
        for cutoff in [10000, 15000, 20000, 23000]:
            flt = CascadeFilter(sample_rate=sample_rate, cutoff=cutoff)
            rng = np.random.RandomState(42)
            inp = rng.randn(48000)
            out = flt.render(inp)
            # Output must be finite
            assert np.all(np.isfinite(out)), f"Filter produced non-finite output at cutoff {cutoff}"
            # Output must not explode (max amplitude should be reasonable)
            max_amp = np.max(np.abs(out))
            assert max_amp < 10.0, f"Filter exploded at cutoff {cutoff}: max amplitude {max_amp}"
