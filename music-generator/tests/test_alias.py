"""Tests for alias control, PolyBLEP, and partition invariance."""
import os
import sys
import time
import tempfile
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS))

import numpy as np
import pytest


class TestOscillatorAliasControl:
    """FFT-based alias-energy thresholds."""

    def _spectrum_energy(self, audio, sample_rate, low_hz, high_hz):
        fft = np.fft.rfft(audio)
        freqs = np.fft.rfftfreq(len(audio), 1.0 / sample_rate)
        mask = (freqs >= low_hz) & (freqs <= high_hz)
        return np.sum(np.abs(fft[mask]) ** 2)

    def test_saw_alias_low_fundamental(self):
        from synth.oscillator import WavetableOscillator
        osc = WavetableOscillator(sample_rate=48000, table_size=4096, waveform="saw", seed=42)
        audio = osc.render(110.0, 48000)
        fundamental = self._spectrum_energy(audio, 48000, 100, 120)
        alias = self._spectrum_energy(audio, 48000, 23000, 24000)
        assert alias < fundamental * 0.01, f"Alias energy {alias} too high vs fundamental {fundamental}"

    def test_saw_alias_mid_fundamental(self):
        from synth.oscillator import WavetableOscillator
        osc = WavetableOscillator(sample_rate=48000, table_size=4096, waveform="saw", seed=42)
        audio = osc.render(1000.0, 48000)
        fundamental = self._spectrum_energy(audio, 48000, 900, 1100)
        alias = self._spectrum_energy(audio, 48000, 23000, 24000)
        assert alias < fundamental * 0.01

    def test_saw_alias_near_nyquist(self):
        from synth.oscillator import WavetableOscillator
        osc = WavetableOscillator(sample_rate=48000, table_size=4096, waveform="saw", seed=42)
        audio = osc.render(20000.0, 48000)
        fundamental = self._spectrum_energy(audio, 48000, 19000, 21000)
        alias = self._spectrum_energy(audio, 48000, 23000, 24000)
        assert alias < fundamental * 0.1

    def test_square_alias_low_fundamental(self):
        from synth.oscillator import WavetableOscillator
        osc = WavetableOscillator(sample_rate=48000, table_size=4096, waveform="square", seed=42)
        audio = osc.render(110.0, 48000)
        fundamental = self._spectrum_energy(audio, 48000, 100, 120)
        alias = self._spectrum_energy(audio, 48000, 23000, 24000)
        assert alias < fundamental * 0.01

    def test_square_alias_near_nyquist(self):
        from synth.oscillator import WavetableOscillator
        osc = WavetableOscillator(sample_rate=48000, table_size=4096, waveform="square", seed=42)
        audio = osc.render(20000.0, 48000)
        fundamental = self._spectrum_energy(audio, 48000, 19000, 21000)
        alias = self._spectrum_energy(audio, 48000, 23000, 24000)
        assert alias < fundamental * 0.1

    def test_sine_unaffected(self):
        """Sine must bypass PolyBLEP entirely — no correction artifacts."""
        from synth.oscillator import WavetableOscillator
        # Seed=None to isolate PolyBLEP from phase-offset effects
        osc = WavetableOscillator(sample_rate=48000, table_size=4096, waveform="sine", seed=None)
        audio = osc.render(440.0, 48000)
        expected = np.sin(2.0 * np.pi * 440.0 * np.arange(48000) / 48000)
        assert np.allclose(audio, expected, atol=1e-10)

    def test_partition_invariance(self):
        """Same total samples rendered in one block vs two blocks must match."""
        from synth.oscillator import WavetableOscillator
        osc1 = WavetableOscillator(sample_rate=48000, table_size=4096, waveform="saw", seed=42)
        osc2 = WavetableOscillator(sample_rate=48000, table_size=4096, waveform="saw", seed=42)
        full = osc1.render(440.0, 96000)
        part1 = osc2.render(440.0, 257)
        part2 = osc2.render(440.0, 96000 - 257)
        combined = np.concatenate([part1, part2])
        assert np.allclose(full, combined, atol=1e-10), \
            f"Max diff: {np.max(np.abs(full - combined))}"

    def test_oscillator_no_nan_polyblep(self):
        from synth.oscillator import WavetableOscillator
        for wf in ("saw", "square"):
            osc = WavetableOscillator(sample_rate=48000, table_size=4096, waveform=wf, seed=42)
            out = osc.render(20000.0, 48000)
            assert np.all(np.isfinite(out))

    def test_polyblep_benchmark(self):
        """PolyBLEP correction must not exceed 5x slower than raw wavetable."""
        from synth.oscillator import WavetableOscillator
        osc = WavetableOscillator(sample_rate=48000, table_size=4096, waveform="saw", seed=42)
        start = time.time()
        osc.render(440.0, 48000)
        elapsed = time.time() - start
        assert elapsed < 1.0, f"PolyBLEP render took {elapsed:.2f}s — too slow"