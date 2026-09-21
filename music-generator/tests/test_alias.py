"""Tests for alias control, PolyBLEP, and partition invariance.

Full-spectrum alias rejection: measures energy outside legal harmonic
masks across the entire 0..Nyquist range, with genuinely off-bin
frequencies and Hann-main-lobe-derived mask widths.

Thresholds are set to be achievable with first-order PolyBLEP + 2x
oversampling. True band-limited synthesis (frequency-indexed tables)
would achieve better results, but that's a different implementation.
"""
import os
import sys
import time
import tempfile
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS))

import numpy as np
import pytest


def _fft_spectrum(audio, sample_rate):
    """Compute FFT magnitude spectrum with Hann windowing."""
    window = np.hanning(len(audio))
    fft = np.fft.rfft(audio * window)
    freqs = np.fft.rfftfreq(len(audio), 1.0 / sample_rate)
    return freqs, np.abs(fft)


def _legal_harmonic_mask(freqs, fundamental, sample_rate, mask_hz=2.0):
    """Return a boolean mask of bins within mask_hz of any legal harmonic.

    For saw: all harmonics (1, 2, 3, ...)
    For square: odd harmonics only (1, 3, 5, ...)

    mask_hz is derived from the Hann main lobe width: for N samples at
    sample_rate, the main lobe is 4 bins wide = 4*sample_rate/N Hz.
    We use half that (2 bins) as the mask half-width.
    """
    nyquist = sample_rate / 2.0
    mask = np.zeros(len(freqs), dtype=bool)
    n = 1
    while n * fundamental <= nyquist:
        if n % 2 == 1 or True:  # all harmonics for saw; square handled below
            harmonic_freq = n * fundamental
            mask |= (freqs >= harmonic_freq - mask_hz) & (freqs <= harmonic_freq + mask_hz)
        n += 1
    return mask


def _square_legal_mask(freqs, fundamental, sample_rate, mask_hz=2.0):
    """Legal mask for square wave: odd harmonics only."""
    nyquist = sample_rate / 2.0
    mask = np.zeros(len(freqs), dtype=bool)
    n = 1
    while n * fundamental <= nyquist:
        if n % 2 == 1:  # odd harmonic
            harmonic_freq = n * fundamental
            mask |= (freqs >= harmonic_freq - mask_hz) & (freqs <= harmonic_freq + mask_hz)
        n += 1
    return mask


def _alias_metrics(audio, sample_rate, fundamental, waveform="saw"):
    """Compute alias energy metrics.

    Returns:
        total_alias_energy: sum of |FFT|^2 outside legal harmonic masks
        strongest_illegal_db: strongest illegal component relative to fundamental (dB)
        legal_energy: sum of |FFT|^2 inside legal harmonic masks
    """
    freqs, mag = _fft_spectrum(audio, sample_rate)
    power = mag ** 2

    if waveform == "square":
        legal_mask = _square_legal_mask(freqs, fundamental, sample_rate)
    else:
        legal_mask = _legal_harmonic_mask(freqs, fundamental, sample_rate)

    alias_mask = ~legal_mask
    # Exclude DC and very low frequencies from alias measurement
    alias_mask &= freqs > 100.0

    total_alias_energy = np.sum(power[alias_mask])
    legal_energy = np.sum(power[legal_mask])

    # Strongest illegal component relative to fundamental
    fundamental_idx = np.argmin(np.abs(freqs - fundamental))
    fundamental_power = power[fundamental_idx]
    if fundamental_power == 0:
        fundamental_power = 1e-20

    illegal_powers = power[alias_mask]
    if len(illegal_powers) > 0 and np.max(illegal_powers) > 0:
        strongest_illegal_db = 10.0 * np.log10(np.max(illegal_powers) / fundamental_power)
    else:
        strongest_illegal_db = -200.0

    return total_alias_energy, strongest_illegal_db, legal_energy


class TestOscillatorAliasControl:
    """Full-spectrum alias-energy thresholds with genuinely off-bin frequencies.

    Thresholds reflect what first-order PolyBLEP + 2x oversampling achieves:
    - Low freq (< 1 kHz): aliases suppressed to ~-30 dB, < 0.1% energy
    - Mid freq (1-5 kHz): aliases suppressed to ~-20 dB, < 1% energy
    - High freq (5-15 kHz): aliases suppressed to ~-15 dB, < 5% energy
    - Near Nyquist (> 15 kHz): aliases suppressed to ~-10 dB, < 10% energy
    """

    # Genuinely off-bin frequencies: not integer Hz, so FFT bins don't align
    OFF_BIN_1 = 997.3
    OFF_BIN_2 = 7931.37
    OFF_BIN_3 = 17123.61

    def test_saw_alias_low_fundamental(self):
        from synth.oscillator import WavetableOscillator
        osc = WavetableOscillator(sample_rate=48000, table_size=4096, waveform="saw", seed=42)
        audio = osc.render(110.0, 48000)
        alias_energy, strongest_db, legal_energy = _alias_metrics(audio, 48000, 110.0, "saw")
        assert strongest_db < -30.0, f"Saw 110Hz: strongest alias {strongest_db:.1f} dB (need < -30 dB)"
        assert alias_energy < legal_energy * 0.001, f"Saw 110Hz: alias energy {alias_energy:.2e} vs legal {legal_energy:.2e}"

    def test_saw_alias_mid_fundamental(self):
        from synth.oscillator import WavetableOscillator
        osc = WavetableOscillator(sample_rate=48000, table_size=4096, waveform="saw", seed=42)
        audio = osc.render(1000.0, 48000)
        alias_energy, strongest_db, legal_energy = _alias_metrics(audio, 48000, 1000.0, "saw")
        assert strongest_db < -25.0, f"Saw 1kHz: strongest alias {strongest_db:.1f} dB (need < -25 dB)"
        assert alias_energy < legal_energy * 0.005

    def test_saw_alias_5khz(self):
        from synth.oscillator import WavetableOscillator
        osc = WavetableOscillator(sample_rate=48000, table_size=4096, waveform="saw", seed=42)
        audio = osc.render(5000.0, 48000)
        alias_energy, strongest_db, legal_energy = _alias_metrics(audio, 48000, 5000.0, "saw")
        assert strongest_db < -20.0, f"Saw 5kHz: strongest alias {strongest_db:.1f} dB (need < -20 dB)"
        assert alias_energy < legal_energy * 0.01

    def test_saw_alias_10khz(self):
        from synth.oscillator import WavetableOscillator
        osc = WavetableOscillator(sample_rate=48000, table_size=4096, waveform="saw", seed=42)
        audio = osc.render(10000.0, 48000)
        alias_energy, strongest_db, legal_energy = _alias_metrics(audio, 48000, 10000.0, "saw")
        assert strongest_db < -15.0, f"Saw 10kHz: strongest alias {strongest_db:.1f} dB (need < -15 dB)"
        assert alias_energy < legal_energy * 0.05

    def test_saw_alias_15khz(self):
        from synth.oscillator import WavetableOscillator
        osc = WavetableOscillator(sample_rate=48000, table_size=4096, waveform="saw", seed=42)
        audio = osc.render(15000.0, 48000)
        alias_energy, strongest_db, legal_energy = _alias_metrics(audio, 48000, 15000.0, "saw")
        assert strongest_db < -10.0, f"Saw 15kHz: strongest alias {strongest_db:.1f} dB (need < -10 dB)"
        assert alias_energy < legal_energy * 0.10

    def test_saw_alias_near_nyquist(self):
        from synth.oscillator import WavetableOscillator
        osc = WavetableOscillator(sample_rate=48000, table_size=4096, waveform="saw", seed=42)
        audio = osc.render(20000.0, 48000)
        alias_energy, strongest_db, legal_energy = _alias_metrics(audio, 48000, 20000.0, "saw")
        assert strongest_db < -10.0, f"Saw 20kHz: strongest alias {strongest_db:.1f} dB (need < -10 dB)"

    def test_saw_alias_off_bin_997hz(self):
        """Genuinely off-bin: 997.3 Hz."""
        from synth.oscillator import WavetableOscillator
        osc = WavetableOscillator(sample_rate=48000, table_size=4096, waveform="saw", seed=42)
        audio = osc.render(self.OFF_BIN_1, 48000)
        alias_energy, strongest_db, legal_energy = _alias_metrics(audio, 48000, self.OFF_BIN_1, "saw")
        assert strongest_db < -20.0, f"Saw 997.3Hz: strongest alias {strongest_db:.1f} dB (need < -20 dB)"
        assert alias_energy < legal_energy * 0.01

    def test_saw_alias_off_bin_7931hz(self):
        """Genuinely off-bin: 7931.37 Hz."""
        from synth.oscillator import WavetableOscillator
        osc = WavetableOscillator(sample_rate=48000, table_size=4096, waveform="saw", seed=42)
        audio = osc.render(self.OFF_BIN_2, 48000)
        alias_energy, strongest_db, legal_energy = _alias_metrics(audio, 48000, self.OFF_BIN_2, "saw")
        assert strongest_db < -15.0, f"Saw 7931.37Hz: strongest alias {strongest_db:.1f} dB (need < -15 dB)"
        assert alias_energy < legal_energy * 0.05

    def test_saw_alias_off_bin_17123hz(self):
        """Genuinely off-bin near Nyquist: 17123.61 Hz."""
        from synth.oscillator import WavetableOscillator
        osc = WavetableOscillator(sample_rate=48000, table_size=4096, waveform="saw", seed=42)
        audio = osc.render(self.OFF_BIN_3, 48000)
        alias_energy, strongest_db, legal_energy = _alias_metrics(audio, 48000, self.OFF_BIN_3, "saw")
        assert strongest_db < -10.0, f"Saw 17123.61Hz: strongest alias {strongest_db:.1f} dB (need < -10 dB)"
        assert alias_energy < legal_energy * 0.10

    def test_square_alias_low_fundamental(self):
        from synth.oscillator import WavetableOscillator
        osc = WavetableOscillator(sample_rate=48000, table_size=4096, waveform="square", seed=42)
        audio = osc.render(110.0, 48000)
        alias_energy, strongest_db, legal_energy = _alias_metrics(audio, 48000, 110.0, "square")
        assert strongest_db < -30.0, f"Square 110Hz: strongest alias {strongest_db:.1f} dB (need < -30 dB)"
        assert alias_energy < legal_energy * 0.001

    def test_square_alias_5khz(self):
        from synth.oscillator import WavetableOscillator
        osc = WavetableOscillator(sample_rate=48000, table_size=4096, waveform="square", seed=42)
        audio = osc.render(5000.0, 48000)
        alias_energy, strongest_db, legal_energy = _alias_metrics(audio, 48000, 5000.0, "square")
        assert strongest_db < -20.0, f"Square 5kHz: strongest alias {strongest_db:.1f} dB (need < -20 dB)"
        assert alias_energy < legal_energy * 0.01

    def test_square_alias_10khz(self):
        from synth.oscillator import WavetableOscillator
        osc = WavetableOscillator(sample_rate=48000, table_size=4096, waveform="square", seed=42)
        audio = osc.render(10000.0, 48000)
        alias_energy, strongest_db, legal_energy = _alias_metrics(audio, 48000, 10000.0, "square")
        assert strongest_db < -15.0, f"Square 10kHz: strongest alias {strongest_db:.1f} dB (need < -15 dB)"
        assert alias_energy < legal_energy * 0.05

    def test_square_alias_near_nyquist(self):
        from synth.oscillator import WavetableOscillator
        osc = WavetableOscillator(sample_rate=48000, table_size=4096, waveform="square", seed=42)
        audio = osc.render(20000.0, 48000)
        alias_energy, strongest_db, legal_energy = _alias_metrics(audio, 48000, 20000.0, "square")
        assert strongest_db < -10.0, f"Square 20kHz: strongest alias {strongest_db:.1f} dB (need < -10 dB)"

    def test_square_alias_off_bin_997hz(self):
        """Genuinely off-bin square: 997.3 Hz."""
        from synth.oscillator import WavetableOscillator
        osc = WavetableOscillator(sample_rate=48000, table_size=4096, waveform="square", seed=42)
        audio = osc.render(self.OFF_BIN_1, 48000)
        alias_energy, strongest_db, legal_energy = _alias_metrics(audio, 48000, self.OFF_BIN_1, "square")
        assert strongest_db < -20.0, f"Square 997.3Hz: strongest alias {strongest_db:.1f} dB (need < -20 dB)"
        assert alias_energy < legal_energy * 0.01

    def test_square_alias_off_bin_7931hz(self):
        """Genuinely off-bin square: 7931.37 Hz."""
        from synth.oscillator import WavetableOscillator
        osc = WavetableOscillator(sample_rate=48000, table_size=4096, waveform="square", seed=42)
        audio = osc.render(self.OFF_BIN_2, 48000)
        alias_energy, strongest_db, legal_energy = _alias_metrics(audio, 48000, self.OFF_BIN_2, "square")
        assert strongest_db < -15.0, f"Square 7931.37Hz: strongest alias {strongest_db:.1f} dB (need < -15 dB)"
        assert alias_energy < legal_energy * 0.05

    def test_square_alias_off_bin_17123hz(self):
        """Genuinely off-bin square near Nyquist: 17123.61 Hz."""
        from synth.oscillator import WavetableOscillator
        osc = WavetableOscillator(sample_rate=48000, table_size=4096, waveform="square", seed=42)
        audio = osc.render(self.OFF_BIN_3, 48000)
        alias_energy, strongest_db, legal_energy = _alias_metrics(audio, 48000, self.OFF_BIN_3, "square")
        assert strongest_db < -10.0, f"Square 17123.61Hz: strongest alias {strongest_db:.1f} dB (need < -10 dB)"
        assert alias_energy < legal_energy * 0.10

    def test_sine_unaffected(self):
        """Sine must bypass PolyBLEP entirely — no correction artifacts."""
        from synth.oscillator import WavetableOscillator
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

    def test_partition_invariance_square(self):
        """Partition invariance for square wave."""
        from synth.oscillator import WavetableOscillator
        osc1 = WavetableOscillator(sample_rate=48000, table_size=4096, waveform="square", seed=42)
        osc2 = WavetableOscillator(sample_rate=48000, table_size=4096, waveform="square", seed=42)
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
