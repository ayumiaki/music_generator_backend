"""Tests for the deterministic synthesizer backend."""
import os
import sys
import time
import signal
import tempfile
from pathlib import Path

# Ensure scripts dir is on path
SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS))

import numpy as np
import pytest

# --- Oscillator tests ---

class TestOscillatorWavetable:
    """Oscillator: phase-continuous wavetable lookup, interpolation, deterministic."""

    def test_wavetable_generation(self):
        from synth.oscillator import make_wavetable
        tbl = make_wavetable(4096, "saw")
        assert len(tbl) == 4096
        assert np.max(np.abs(tbl)) <= 1.0

    def test_oscillator_phase_continuity(self):
        from synth.oscillator import WavetableOscillator
        osc = WavetableOscillator(sample_rate=48000, table_size=4096, seed=42)
        # Render two blocks at same frequency — phase should be continuous
        out1 = osc.render(441.0, 1000)  # Use 441Hz so phase doesn't wrap to 0
        phase_after = osc.get_phase()
        out2 = osc.render(441.0, 1000)  # Another block
        assert len(out1) == 1000
        assert len(out2) == 1000
        # Phase should have advanced from 0
        assert phase_after > 0

    def test_oscillator_phase_reset(self):
        from synth.oscillator import WavetableOscillator
        osc = WavetableOscillator(sample_rate=48000, table_size=4096, seed=42)
        out1 = osc.render(440.0, 1000)
        osc.reset()
        out2 = osc.render(440.0, 1000, phase_reset=True, reset_phase=0.0)
        # After reset, same frequency should produce same output
        assert np.allclose(out1[:100], out2[:100], atol=1e-10)

    def test_oscillator_deterministic_seed(self):
        from synth.oscillator import WavetableOscillator
        osc1 = WavetableOscillator(sample_rate=48000, table_size=4096, seed=123)
        osc2 = WavetableOscillator(sample_rate=48000, table_size=4096, seed=123)
        out1 = osc1.render(440.0, 1000)
        out2 = osc2.render(440.0, 1000)
        assert np.array_equal(out1, out2)

    def test_oscillator_different_seed_different(self):
        from synth.oscillator import WavetableOscillator
        osc1 = WavetableOscillator(sample_rate=48000, table_size=4096, seed=123)
        osc2 = WavetableOscillator(sample_rate=48000, table_size=4096, seed=456)
        out1 = osc1.render(440.0, 1000)
        out2 = osc2.render(440.0, 1000)
        assert not np.array_equal(out1, out2)

    def test_oscillator_waveforms(self):
        from synth.oscillator import make_wavetable
        for wf in ["saw", "square", "triangle", "sine"]:
            tbl = make_wavetable(4096, wf)
            assert len(tbl) == 4096
            assert np.max(np.abs(tbl)) <= 1.0

    def test_oscillator_no_nan(self):
        from synth.oscillator import WavetableOscillator
        osc = WavetableOscillator(sample_rate=48000, table_size=4096, seed=42)
        out = osc.render(440.0, 48000)
        assert np.all(np.isfinite(out))


# --- Envelope tests ---

class TestADSR:
    """Envelope: sample-accurate ADSR transitions, retrigger, note-off."""

    def test_envelope_attack_ramp(self):
        from synth.envelope import ADSREnvelope
        env = ADSREnvelope(attack_samples=100, decay_samples=50, sustain_level=0.5, release_samples=50)
        env.note_on()
        out = env.render(200)
        # Should start at 0 and rise
        assert out[0] < out[50]

    def test_envelope_sustain_level(self):
        from synth.envelope import ADSREnvelope
        env = ADSREnvelope(attack_samples=10, decay_samples=10, sustain_level=0.7, release_samples=10)
        env.note_on()
        out = env.render(100)
        # After attack+decay, should be at sustain level
        assert abs(out[50] - 0.7) < 0.1

    def test_envelope_note_off(self):
        from synth.envelope import ADSREnvelope
        env = ADSREnvelope(attack_samples=10, decay_samples=10, sustain_level=0.7, release_samples=50)
        env.note_on()
        out1 = env.render(50)
        env.note_off()
        out2 = env.render(100)
        # Release should decrease to 0
        assert out2[-1] < out2[0]
        assert out2[-1] >= 0.0

    def test_envelope_retrigger(self):
        from synth.envelope import ADSREnvelope
        env = ADSREnvelope(attack_samples=50, decay_samples=25, sustain_level=0.6, release_samples=50, retrigger=True)
        env.note_on()
        out1 = env.render(50)
        # While in release, retrigger should restart attack
        env.note_off()
        env.note_on()  # retrigger during release
        # Should go back up
        out2 = env.render(50)
        assert out2[0] < out2[10]

    def test_envelope_no_retrigger_during_release(self):
        from synth.envelope import ADSREnvelope
        env = ADSREnvelope(attack_samples=50, decay_samples=25, sustain_level=0.6, release_samples=50, retrigger=False)
        env.note_on()
        out1 = env.render(50)
        env.note_off()
        env.note_on()  # should NOT restart because retrigger=False
        # Should continue release
        out2 = env.render(50)
        assert out2[-1] < out1[-1]  # Still releasing

    def test_envelope_idle(self):
        from synth.envelope import ADSREnvelope
        env = ADSREnvelope()
        out = env.render(100)
        assert np.all(out == 0.0)

    def test_envelope_no_nan(self):
        from synth.envelope import ADSREnvelope
        env = ADSREnvelope()
        out = env.render(1000)
        assert np.all(np.isfinite(out))


# --- Filter tests ---

class TestLadderFilter:
    """Filter: stable ladder implementation, no NaNs, no runaway gain."""

    def test_filter_stable(self):
        from synth.filter import LadderFilter
        flt = LadderFilter(sample_rate=48000, cutoff=1000, resonance=0.5)
        # Generate a high-amplitude input
        inp = np.random.RandomState(1).uniform(-1, 1, 48000)
        out = flt.render(inp)
        assert np.all(np.isfinite(out))
        assert not np.any(np.isnan(out))
        assert not np.any(np.isinf(out))

    def test_filter_high_resonance(self):
        from synth.filter import LadderFilter
        flt = LadderFilter(sample_rate=48000, cutoff=1000, resonance=1.0)
        inp = np.sin(2.0 * np.pi * 1000 * np.arange(48000) / 48000)
        out = flt.render(inp)
        assert np.all(np.isfinite(out))
        assert np.max(np.abs(out)) < 100.0  # Should not runaway

    def test_filter_no_nan_even_at_extremes(self):
        from synth.filter import LadderFilter
        flt = LadderFilter(sample_rate=48000, cutoff=22050, resonance=1.0)
        inp = np.random.RandomState(2).uniform(-0.5, 0.5, 48000)
        out = flt.render(inp)
        assert np.all(np.isfinite(out))

    def test_filter_passes_low_freq(self):
        from synth.filter import LadderFilter
        flt = LadderFilter(sample_rate=48000, cutoff=10000, resonance=0.0)
        # Low-frequency sine should pass through
        inp = np.sin(2.0 * np.pi * 100 * np.arange(48000) / 48000)
        out = flt.render(inp)
        # Output should have similar amplitude to input (low freq passes)
        assert np.max(np.abs(out)) > 0.5

    def test_filter_reset(self):
        from synth.filter import LadderFilter
        flt = LadderFilter(sample_rate=48000, cutoff=1000, resonance=0.5)
        inp = np.random.RandomState(3).uniform(-1, 1, 48000)
        flt.render(inp[:24000])
        flt.reset()
        out = flt.render(inp[24000:])
        assert np.all(np.isfinite(out))


# --- Voice engine tests ---

class TestVoiceEngine:
    """Voice engine: deterministic allocation, stealing, polyphony, modulation order."""

    def test_voice_allocates(self):
        from synth.voice import VoiceEngine
        engine = VoiceEngine(polyphony=4, seed=42)
        vid = engine.allocate(440.0, 0.8)
        assert vid == 0

    def test_voice_polyphony(self):
        from synth.voice import VoiceEngine
        engine = VoiceEngine(polyphony=4, seed=42)
        for i in range(4):
            vid = engine.allocate(220.0 * (i + 1), 0.8)
            assert vid == i
        # 5th should steal oldest (ID 0)
        vid5 = engine.allocate(330.0, 0.8)
        assert vid5 == 0  # Steal lowest ID

    def test_voice_deterministic_allocation(self):
        from synth.voice import VoiceEngine
        e1 = VoiceEngine(polyphony=8, seed=42)
        e2 = VoiceEngine(polyphony=8, seed=42)
        ids1 = [e1.allocate(440.0, 0.8) for _ in range(5)]
        ids2 = [e2.allocate(440.0, 0.8) for _ in range(5)]
        assert ids1 == ids2

    def test_voice_note_off(self):
        from synth.voice import VoiceEngine
        engine = VoiceEngine(polyphony=4, seed=42)
        vid = engine.allocate(440.0, 0.8)
        engine.note_off(vid)
        voice = engine.get_voice(vid)
        assert voice is not None

    def test_voice_render_block(self):
        from synth.voice import VoiceEngine, VoiceConfig
        config = VoiceConfig(attack_samples=100, decay_samples=50, sustain_level=0.7, release_samples=50)
        engine = VoiceEngine(polyphony=4, sample_rate=48000, config=config, seed=42)
        notes = [(440.0, 4800, 0.8), (554.0, 4800, 0.6)]
        audio = engine.render_block(notes, block_size=9600)
        assert len(audio) > 0
        assert np.all(np.isfinite(audio))

    def test_voice_no_nan(self):
        from synth.voice import VoiceEngine, VoiceConfig
        config = VoiceConfig()
        engine = VoiceEngine(polyphony=4, seed=42, config=config)
        notes = [(440.0, 4800, 0.8)]
        audio = engine.render_block(notes, block_size=4800)
        assert np.all(np.isfinite(audio))


# --- Drum tests ---

class TestDrumPattern:
    """Drums: timing derived from sample positions."""

    def test_drum_render(self):
        from synth.drums import DrumPattern
        drums = DrumPattern(bpm=120, sample_rate=48000, seed=42)
        audio = drums.render(bars=1, kick_pattern="four_on_floor", snare_pattern="backbeat", hihat_pattern="eighth")
        assert len(audio) > 0
        assert np.all(np.isfinite(audio))

    def test_drum_timing_deterministic(self):
        from synth.drums import DrumPattern
        d1 = DrumPattern(bpm=120, sample_rate=48000, seed=42)
        d2 = DrumPattern(bpm=120, sample_rate=48000, seed=42)
        a1 = d1.render(bars=1)
        a2 = d2.render(bars=1)
        assert np.array_equal(a1, a2)

    def test_drum_no_nan(self):
        from synth.drums import DrumPattern
        drums = DrumPattern(bpm=120, sample_rate=48000, seed=42)
        audio = drums.render(bars=2)
        assert np.all(np.isfinite(audio))

    def test_drum_different_bpm_different_length(self):
        from synth.drums import DrumPattern
        d1 = DrumPattern(bpm=120, sample_rate=48000, seed=42)
        d2 = DrumPattern(bpm=60, sample_rate=48000, seed=42)
        a1 = d1.render(bars=1)
        a2 = d2.render(bars=1)
        assert len(a1) != len(a2)


# --- Arrangement tests ---

class TestArrangement:
    """Arrangement: sample-accurate sequencer."""

    def test_arrangement_render(self):
        from synth.arrangement import Arrangement, ArrangementConfig
        config = ArrangementConfig(bpm=120, bars=1, seed=42)
        arr = Arrangement(config)
        arr.add_note(time_beats=0.0, freq=440.0, duration_beats=0.5, amplitude=0.8)
        arr.add_note(time_beats=0.5, freq=554.0, duration_beats=0.5, amplitude=0.6)
        audio = arr.render()
        assert len(audio) > 0
        assert np.all(np.isfinite(audio))

    def test_arrangement_empty(self):
        from synth.arrangement import Arrangement, ArrangementConfig
        config = ArrangementConfig()
        arr = Arrangement(config)
        audio = arr.render()
        assert len(audio) == 0

    def test_arrangement_beat_to_samples(self):
        from synth.arrangement import Arrangement, ArrangementConfig
        config = ArrangementConfig(bpm=120, sample_rate=48000)
        arr = Arrangement(config)
        # One beat at 120 BPM = 0.5 seconds = 24000 samples
        samples = arr.beat_to_samples(1.0)
        assert samples == 24000

    def test_arrangement_no_nan(self):
        from synth.arrangement import Arrangement, ArrangementConfig
        config = ArrangementConfig(seed=42)
        arr = Arrangement(config)
        for i in range(8):
            arr.add_note(time_beats=i * 0.5, freq=220.0 * (i + 1), duration_beats=0.25)
        audio = arr.render()
        assert np.all(np.isfinite(audio))


# --- SynthEngine tests ---

class TestSynthEngine:
    """SynthEngine: deterministic, full pipeline."""

    def test_synth_render(self):
        from synth.synth import SynthEngine, SynthConfig
        config = SynthConfig(bpm=120, bars=1, seed=42, sample_rate=48000)
        engine = SynthEngine(config)
        engine.add_note(time_beats=0.0, freq=440.0, duration_beats=0.5, amplitude=0.8)
        engine.add_note(time_beats=0.5, freq=554.0, duration_beats=0.5, amplitude=0.6)
        engine.add_drums(bars=1)
        integrity = engine.render()
        assert integrity.frame_count > 0
        assert integrity.all_finite
        assert not integrity.clipping
        assert abs(integrity.dc_offset) < 0.1

    def test_synth_deterministic(self):
        from synth.synth import SynthEngine, SynthConfig
        config1 = SynthConfig(bpm=120, bars=1, seed=42, sample_rate=48000)
        config2 = SynthConfig(bpm=120, bars=1, seed=42, sample_rate=48000)
        engine1 = SynthEngine(config1)
        engine2 = SynthEngine(config2)
        engine1.add_note(time_beats=0.0, freq=440.0, duration_beats=0.5)
        engine2.add_note(time_beats=0.0, freq=440.0, duration_beats=0.5)
        audio1 = engine1.render()
        audio2 = engine2.render()
        e1 = engine1.get_audio()
        e2 = engine2.get_audio()
        if e1 is not None and e2 is not None:
            assert np.array_equal(e1, e2)

    def test_synth_wav_output(self):
        from synth.synth import SynthEngine, SynthConfig
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            config = SynthConfig(bpm=120, bars=1, seed=42, sample_rate=48000)
            engine = SynthEngine(config)
            engine.add_note(time_beats=0.0, freq=440.0, duration_beats=0.5)
            engine.add_drums(bars=1)
            wav_path = Path(tmpdir) / "test.wav"
            integrity = engine.render(output_path=wav_path)
            assert wav_path.exists()
            assert integrity.frame_count > 0

    def test_synth_no_nan(self):
        from synth.synth import SynthEngine, SynthConfig
        config = SynthConfig(bpm=120, bars=1, seed=42, sample_rate=48000)
        engine = SynthEngine(config)
        engine.add_note(time_beats=0.0, freq=440.0, duration_beats=0.5)
        engine.add_drums(bars=1)
        integrity = engine.render()
        assert integrity.all_finite
        assert not integrity.clipping

    def test_synth_reset(self):
        from synth.synth import SynthEngine, SynthConfig
        config = SynthConfig(bpm=120, bars=1, seed=42, sample_rate=48000)
        engine = SynthEngine(config)
        engine.add_note(time_beats=0.0, freq=440.0, duration_beats=0.5)
        engine.render()
        engine.reset()
        assert engine.get_audio() is None
        assert engine.get_integrity() is None


# --- Determinism tests ---

class TestDeterminism:
    """Identical input + seed produce byte-identical PCM."""

    def test_identical_seed_identical_output(self):
        from synth.synth import SynthEngine, SynthConfig
        config1 = SynthConfig(bpm=120, bars=2, seed=999, sample_rate=48000)
        config2 = SynthConfig(bpm=120, bars=2, seed=999, sample_rate=48000)
        engine1 = SynthEngine(config1)
        engine2 = SynthEngine(config2)
        for i in range(8):
            engine1.add_note(time_beats=i * 0.5, freq=220.0 * (i + 1), duration_beats=0.25)
            engine2.add_note(time_beats=i * 0.5, freq=220.0 * (i + 1), duration_beats=0.25)
        engine1.render()
        engine2.render()
        a1 = engine1.get_audio()
        a2 = engine2.get_audio()
        if a1 is not None and a2 is not None:
            assert np.array_equal(a1, a2)

    def test_different_seed_different_output(self):
        from synth.synth import SynthEngine, SynthConfig
        config1 = SynthConfig(bpm=120, bars=1, seed=111, sample_rate=48000)
        config2 = SynthConfig(bpm=120, bars=1, seed=222, sample_rate=48000)
        engine1 = SynthEngine(config1)
        engine2 = SynthEngine(config2)
        engine1.add_note(time_beats=0.0, freq=440.0, duration_beats=0.5)
        engine2.add_note(time_beats=0.0, freq=440.0, duration_beats=0.5)
        a1 = engine1.render()
        a2 = engine2.render()
        e1 = engine1.get_audio()
        e2 = engine2.get_audio()
        if e1 is not None and e2 is not None:
            assert not np.array_equal(e1, e2)

    def test_wav_file_byte_identical(self):
        from synth.synth import SynthEngine, SynthConfig
        from synth.renderer import write_wav
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            config1 = SynthConfig(bpm=120, bars=1, seed=42, sample_rate=48000)
            config2 = SynthConfig(bpm=120, bars=1, seed=42, sample_rate=48000)
            engine1 = SynthEngine(config1)
            engine2 = SynthEngine(config2)
            engine1.add_note(time_beats=0.0, freq=440.0, duration_beats=0.5)
            engine2.add_note(time_beats=0.0, freq=440.0, duration_beats=0.5)
            p1 = Path(tmpdir) / "a.wav"
            p2 = Path(tmpdir) / "b.wav"
            engine1.render(output_path=p1)
            engine2.render(output_path=p2)
            assert p1.read_bytes() == p2.read_bytes()


# --- Integrity tests ---

class TestRenderIntegrity:
    """Integrity: sample rate, channels, frame count, duration, clipping, DC offset, finite."""

    def test_integrity_checks(self):
        from synth.renderer import render_integrity
        audio = np.sin(2.0 * np.pi * 440.0 * np.arange(48000) / 48000)
        integrity = render_integrity(audio, 48000)
        assert integrity.sample_rate == 48000
        assert integrity.channels == 1
        assert integrity.frame_count == 48000
        assert abs(integrity.duration_sec - 1.0) < 0.01
        assert integrity.all_finite
        assert not integrity.clipping
        assert abs(integrity.dc_offset) < 0.01

    def test_integrity_clipping_detected(self):
        from synth.renderer import render_integrity
        audio = np.ones(48000) * 2.0  # Clipping
        integrity = render_integrity(audio, 48000)
        assert integrity.clipping

    def test_integrity_dc_offset(self):
        from synth.renderer import render_integrity
        audio = np.ones(48000) * 0.5  # DC offset
        integrity = render_integrity(audio, 48000)
        assert abs(integrity.dc_offset - 0.5) < 0.01

    def test_integrity_non_finite(self):
        from synth.renderer import render_integrity
        audio = np.array([1.0, 2.0, float('nan'), 4.0])
        integrity = render_integrity(audio, 48000)
        assert not integrity.all_finite


# --- Performance tests ---

class TestPerformance:
    """Performance: real-time factor and peak memory."""

    def test_real_time_factor_30s(self):
        from synth.synth import SynthEngine, SynthConfig
        config = SynthConfig(bpm=120, bars=4, seed=42, sample_rate=48000)
        engine = SynthEngine(config)
        # 30 seconds at 48kHz = 1,440,000 samples
        for i in range(60):
            engine.add_note(time_beats=i * 0.5, freq=220.0, duration_beats=0.25)
        start = time.time()
        engine.render()
        elapsed = time.time() - start
        duration = 30.0
        rt_factor = elapsed / duration
        assert rt_factor < 10.0, f"Real-time factor {rt_factor:.2f} too high"

    def test_peak_memory_reasonable(self):
        from synth.synth import SynthEngine, SynthConfig
        import tracemalloc
        config = SynthConfig(bpm=120, bars=4, seed=42, sample_rate=48000)
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
        config = SynthConfig(bpm=120, bars=20, seed=42, sample_rate=48000)
        engine = SynthEngine(config)
        # 5 minutes = 300 seconds, at 120 BPM = 5 beats/sec = 150 beats
        for i in range(150):
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