"""Behavioral tests for the event-driven voice engine and production path.

These test the properties the synth gate demands:
- Notes start at their exact scheduled sample positions
- Overlapping notes genuinely coexist in the mix
- Note-off fires at the exact sample and produces an envelope-controlled
  release tail (not filter ringing)
- Voice stealing actually removes the oldest voice from the mix
- The production path renders exactly the requested number of frames
- Invalid seeds are rejected before any work happens
"""
import os
import sys
import tempfile
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS))

import numpy as np
import pytest


SR = 48000


class TestEventTiming:
    """Notes must start at their exact scheduled sample positions."""

    def _render_two_notes(self, gap_samples, note_len=4800):
        from synth.voice import VoiceEngine, VoiceConfig

        config = VoiceConfig(
            attack_samples=480, decay_samples=240, sustain_level=0.7,
            release_samples=480, filter_cutoff=20000.0,
        )
        engine = VoiceEngine(polyphony=8, sample_rate=SR, config=config, seed=42)
        events = [
            (0, 440.0, note_len, 0.8, "saw"),
            (gap_samples, 550.0, note_len, 0.8, "saw"),
        ]
        total = gap_samples + note_len + 4800
        return engine.render_schedule(events, total)

    def test_note_starts_at_scheduled_sample(self):
        """A note scheduled at sample N must have (near-)zero energy before N."""
        gap = 24000
        audio = self._render_two_notes(gap)
        # The first note occupies [0, 24000+release]; find where the second
        # note's onset changes the signal. The mix before the second note
        # must be exactly the first note alone: render the first note alone
        # and compare the pre-gap segment.
        from synth.voice import VoiceEngine, VoiceConfig
        config = VoiceConfig(
            attack_samples=480, decay_samples=240, sustain_level=0.7,
            release_samples=480, filter_cutoff=20000.0,
        )
        engine = VoiceEngine(polyphony=8, sample_rate=SR, config=config, seed=42)
        alone = engine.render_schedule([(0, 440.0, 4800, 0.8, "saw")], gap)
        # The mix up to the second note's start must equal the first note
        # rendered alone (byte-exact: same DSP, same seed sequence)
        assert np.array_equal(audio[:gap], alone[:gap]), (
            "Mix before the second note's scheduled start differs from the "
            "first note alone — the second note is leaking into earlier samples"
        )

    def test_second_note_audible_after_its_start(self):
        """After the second note starts, its frequency must appear in the mix."""
        gap = 24000
        audio = self._render_two_notes(gap)
        seg = audio[gap : gap + 4800]
        # The second note (550 Hz) must contribute measurable energy at 550 Hz
        window = np.hanning(len(seg))
        spectrum = np.abs(np.fft.rfft(seg * window))
        freqs = np.fft.rfftfreq(len(seg), 1.0 / SR)
        f550 = spectrum[np.argmin(np.abs(freqs - 550.0))]
        f440 = spectrum[np.argmin(np.abs(freqs - 440.0))]
        # Both notes' fundamentals should be present (first note is in
        # release, second in attack — both non-negligible)
        assert f550 > 0.1 * np.max(spectrum), (
            f"550 Hz component {f550:.1f} too weak in overlap segment"
        )

    def test_note_starts_at_exact_sample_zero(self):
        """A note at sample 0 must begin its attack immediately."""
        from synth.voice import VoiceEngine, VoiceConfig
        config = VoiceConfig(
            attack_samples=4800, decay_samples=2400, sustain_level=0.7,
            release_samples=4800, filter_cutoff=20000.0,
        )
        engine = VoiceEngine(polyphony=8, sample_rate=SR, config=config, seed=42)
        audio = engine.render_schedule([(0, 440.0, 4800, 0.8, "saw")], 9600)
        # Attack from zero: first samples must be tiny, growing
        assert np.abs(audio[0]) < 1e-6, "Note at sample 0 must start from silence"
        # By mid-attack the signal must be clearly nonzero
        assert np.max(np.abs(audio[2400:3600])) > 0.01, "Attack must be underway by mid-attack"


class TestVoiceCoexistence:
    """Overlapping notes must genuinely coexist."""

    def test_overlap_mix_exceeds_single_note(self):
        """Two simultaneous notes must be louder than either alone."""
        from synth.voice import VoiceEngine, VoiceConfig
        config = VoiceConfig(
            attack_samples=480, decay_samples=240, sustain_level=0.7,
            release_samples=480, filter_cutoff=20000.0,
        )
        total = 9600
        # Note A alone
        e1 = VoiceEngine(polyphony=8, sample_rate=SR, config=config, seed=42)
        a = e1.render_schedule([(0, 440.0, 4800, 0.8, "saw")], total)
        # Note B alone
        e2 = VoiceEngine(polyphony=8, sample_rate=SR, config=config, seed=42)
        b = e2.render_schedule([(0, 550.0, 4800, 0.8, "saw")], total)
        # Both together
        e3 = VoiceEngine(polyphony=8, sample_rate=SR, config=config, seed=42)
        ab = e3.render_schedule(
            [(0, 440.0, 4800, 0.8, "saw"), (0, 550.0, 4800, 0.8, "saw")], total
        )
        # During the held portion, the mix must contain both notes' energy
        seg = slice(2400, 4800)
        rms_a = np.sqrt(np.mean(a[seg] ** 2))
        rms_ab = np.sqrt(np.mean(ab[seg] ** 2))
        # Two incoherent notes at similar amplitude mix to more RMS than one
        # (sqrt(2) ideally; allow a generous margin)
        assert rms_ab > 1.2 * rms_a, (
            f"Mix RMS {rms_ab:.4f} vs single {rms_a:.4f} — notes are not coexisting"
        )

    def test_both_fundamentals_present_in_overlap(self):
        """Both notes' frequencies must be present in the overlapping segment."""
        from synth.voice import VoiceEngine, VoiceConfig
        config = VoiceConfig(
            attack_samples=480, decay_samples=240, sustain_level=0.7,
            release_samples=480, filter_cutoff=20000.0,
        )
        engine = VoiceEngine(polyphony=8, sample_rate=SR, config=config, seed=42)
        audio = engine.render_schedule(
            [(0, 440.0, 9600, 0.8, "saw"), (0, 660.0, 9600, 0.8, "saw")], 9600
        )
        seg = audio[2400:7200]
        window = np.hanning(len(seg))
        spectrum = np.abs(np.fft.rfft(seg * window))
        freqs = np.fft.rfftfreq(len(seg), 1.0 / SR)
        peak = np.max(spectrum)
        f440 = spectrum[np.argmin(np.abs(freqs - 440.0))]
        f660 = spectrum[np.argmin(np.abs(freqs - 660.0))]
        assert f440 > 0.2 * peak, "440 Hz missing from overlap"
        assert f660 > 0.2 * peak, "660 Hz missing from overlap"


class TestNoteOffRelease:
    """Note-off must fire at the exact sample and produce a real release tail."""

    def test_release_tail_is_envelope_controlled(self):
        """After note-off the voice must decay smoothly to zero.

        The release must follow the release envelope shape (linear decay
        from the level at note-off), reaching numerical zero within
        release_samples after note-off — not merely filter ringing.
        """
        from synth.voice import VoiceEngine, VoiceConfig
        release = 4800
        config = VoiceConfig(
            attack_samples=480, decay_samples=240, sustain_level=0.7,
            release_samples=release, filter_cutoff=20000.0,
        )
        engine = VoiceEngine(polyphony=8, sample_rate=SR, config=config, seed=42)
        note_len = 9600
        total = note_len + release + 4800
        audio = engine.render_schedule([(0, 440.0, note_len, 0.8, "saw")], total)

        # Level just before note-off (sustain region)
        pre = np.max(np.abs(audio[note_len - 480 : note_len]))
        assert pre > 0.05, f"Sustain level too low before note-off: {pre}"

        # The release region must decay: midpoint of release must be
        # clearly below the pre-off level
        mid = np.max(np.abs(audio[note_len + release // 2 - 240 : note_len + release // 2 + 240]))
        assert mid < 0.7 * pre, (
            f"Release midpoint level {mid:.4f} not decaying (pre-off {pre:.4f})"
        )

        # After release completes (+ filter ring-out margin), the signal
        # must be numerically zero — an envelope-driven release ends
        after = audio[note_len + release + 2400 :]
        assert np.max(np.abs(after)) < 1e-6, (
            f"Signal after release window not silent: {np.max(np.abs(after)):.2e} "
            "(release tail is not envelope-controlled)"
        )

    def test_release_length_matches_configuration(self):
        """The note must fall silent within release_samples of note-off."""
        from synth.voice import VoiceEngine, VoiceConfig
        release = 2400
        config = VoiceConfig(
            attack_samples=100, decay_samples=100, sustain_level=0.7,
            release_samples=release, filter_cutoff=20000.0,
        )
        engine = VoiceEngine(polyphony=8, sample_rate=SR, config=config, seed=42)
        note_len = 4800
        total = note_len + release + 9600
        audio = engine.render_schedule([(0, 440.0, note_len, 0.8, "saw")], total)
        # Well past note-off + release + filter decay: silent
        assert np.max(np.abs(audio[note_len + release + 4800 :])) < 1e-6

    def test_held_note_sustains_beyond_note_off_time(self):
        """A held note must still sound at its note-off sample (off fires there)."""
        from synth.voice import VoiceEngine, VoiceConfig
        config = VoiceConfig(
            attack_samples=480, decay_samples=240, sustain_level=0.7,
            release_samples=4800, filter_cutoff=20000.0,
        )
        engine = VoiceEngine(polyphony=8, sample_rate=SR, config=config, seed=42)
        note_len = 9600
        audio = engine.render_schedule([(0, 440.0, note_len, 0.8, "saw")], note_len + 100)
        # At the note-off sample the note is still at sustain level
        assert np.max(np.abs(audio[note_len - 100 : note_len])) > 0.05


class TestVoiceStealing:
    """Stealing must actually remove the oldest voice from the mix."""

    def test_stolen_voice_stops_contributing(self):
        """With polyphony=1, a new note must replace (not sum with) the old."""
        from synth.voice import VoiceEngine, VoiceConfig
        config = VoiceConfig(
            attack_samples=480, decay_samples=240, sustain_level=0.7,
            release_samples=480, filter_cutoff=20000.0,
        )
        engine = VoiceEngine(polyphony=1, sample_rate=SR, config=config, seed=42)
        # Note A from 0, note B from 4800 — B must steal A
        audio = engine.render_schedule(
            [(0, 440.0, 96000, 0.8, "saw"), (4800, 550.0, 96000, 0.8, "saw")],
            12000,
        )
        # After the steal, only note B sounds: 440 Hz must be gone
        seg = audio[7200:12000]
        window = np.hanning(len(seg))
        spectrum = np.abs(np.fft.rfft(seg * window))
        freqs = np.fft.rfftfreq(len(seg), 1.0 / SR)
        peak = np.max(spectrum)
        f440 = spectrum[np.argmin(np.abs(freqs - 440.0))]
        f550 = spectrum[np.argmin(np.abs(freqs - 550.0))]
        assert f550 > 0.3 * peak, "New note not audible after steal"
        assert f440 < 0.05 * peak, (
            f"Stolen note still audible: 440 Hz at {20*np.log10(f440/peak):.1f} dB below peak"
        )

    def test_polyphony_limit_respected(self):
        """With N voices and N+1 simultaneous notes, only N coexist."""
        from synth.voice import VoiceEngine, VoiceConfig
        config = VoiceConfig(
            attack_samples=480, decay_samples=240, sustain_level=0.7,
            release_samples=480, filter_cutoff=20000.0,
        )
        engine = VoiceEngine(polyphony=2, sample_rate=SR, config=config, seed=42)
        freqs = [440.0, 550.0, 660.0]
        events = [(0, f, 9600, 0.8, "saw") for f in freqs]
        audio = engine.render_schedule(events, 9600)
        seg = audio[2400:7200]
        window = np.hanning(len(seg))
        spectrum = np.abs(np.fft.rfft(seg * window))
        freq_bins = np.fft.rfftfreq(len(seg), 1.0 / SR)
        peak = np.max(spectrum)
        # 440 was allocated first and never rendered before the steal:
        # ages tie, allocation order breaks it — 440 (earliest) is stolen
        f440 = spectrum[np.argmin(np.abs(freq_bins - 440.0))]
        f550 = spectrum[np.argmin(np.abs(freq_bins - 550.0))]
        f660 = spectrum[np.argmin(np.abs(freq_bins - 660.0))]
        assert f550 > 0.2 * peak and f660 > 0.2 * peak, "Surviving notes must be audible"
        assert f440 < 0.1 * peak, "Oldest note must be stolen when polyphony is full"


class TestProductionDuration:
    """The production path must render exactly the requested frame count."""

    def test_backend_exact_durations(self):
        """SynthBackend.generate must produce exactly length * sample_rate frames."""
        import backends.synth_backend as sb
        from backends.synth_backend import SynthBackend

        backend = SynthBackend()
        for length in [1, 15, 31, 32, 7]:
            with tempfile.TemporaryDirectory() as td:
                sb.OUTPUT_DIR = td  # generate() writes here
                result = backend.generate(
                    job_id=f"job{length}",
                    prompt="C4 E4 G4",
                    mood="happy",
                    tempo=120,
                    key="C",
                    length=length,
                    seed=42,
                )
                frames = result["frame_count"]
                expected = length * 48000
                assert frames == expected, (
                    f"length={length}s: got {frames} frames, expected {expected} "
                    f"(duration_sec={result['duration_sec']})"
                )
                wav = Path(result["output_file"])
                assert wav.exists(), "artifact not written"
                # Verify the actual file: read header frame count
                import wave
                with wave.open(str(wav), "rb") as w:
                    assert w.getnframes() == expected, (
                        f"WAV file has {w.getnframes()} frames, expected {expected}"
                    )

    def test_engine_exact_frames_non_bar_aligned(self):
        """Engine with length=31s at 120bpm: 31*48000 frames exactly."""
        from synth.synth import SynthEngine, SynthConfig
        config = SynthConfig(bpm=120, bars=99, length=31.0, seed=42, sample_rate=48000)
        engine = SynthEngine(config)
        engine.add_note(time_beats=0.0, freq=440.0, duration_beats=0.5, amplitude=0.8)
        integrity = engine.render()
        assert integrity.frame_count == 31 * 48000
        assert integrity.duration_sec == pytest.approx(31.0, abs=1e-6)

    def test_engine_exact_frames_one_second(self):
        """Engine with length=1s at 120bpm: 1*48000 frames exactly (not 2s)."""
        from synth.synth import SynthEngine, SynthConfig
        config = SynthConfig(bpm=120, bars=99, length=1.0, seed=42, sample_rate=48000)
        engine = SynthEngine(config)
        engine.add_note(time_beats=0.0, freq=440.0, duration_beats=0.5, amplitude=0.8)
        integrity = engine.render()
        assert integrity.frame_count == 1 * 48000


class TestSeedValidation:
    """Invalid seeds must be rejected before any work happens."""

    def test_engine_rejects_negative_seed(self):
        from synth.synth import SynthEngine, SynthConfig
        with pytest.raises(ValueError, match="[Ss]eed"):
            SynthEngine(SynthConfig(seed=-1))

    def test_engine_rejects_oversized_seed(self):
        from synth.synth import SynthEngine, SynthConfig
        with pytest.raises(ValueError, match="[Ss]eed"):
            SynthEngine(SynthConfig(seed=2**32))

    def test_api_rejects_invalid_seed_with_400(self):
        """POST /generate with seed=-1 must return 400 immediately."""
        import importlib

        with tempfile.TemporaryDirectory() as td:
            os.environ["MG_QUEUE_DIR"] = str(Path(td) / "queue")
            os.environ["MG_OUTPUT_DIR"] = str(Path(td) / "output")
            # Fresh import so config picks up the temp dirs
            for mod in list(sys.modules):
                if mod in ("config", "api_server", "job_queue"):
                    del sys.modules[mod]
            sys.path.insert(0, str(SCRIPTS))
            api = importlib.import_module("api_server")
            client = api.app.test_client()

            for bad_seed in [-1, 2**32, 2**40]:
                resp = client.post(
                    "/generate",
                    json={"prompt": "C4 E4 G4", "length": 5, "seed": bad_seed},
                )
                assert resp.status_code == 400, (
                    f"seed={bad_seed}: expected 400, got {resp.status_code}"
                )

            # Valid seed must pass validation (job enqueues)
            resp = client.post(
                "/generate",
                json={"prompt": "C4 E4 G4", "length": 5, "seed": 12345},
            )
            assert resp.status_code == 202, (
                f"valid seed: expected 202, got {resp.status_code}"
            )

    def test_backend_rejects_invalid_seed(self):
        """SynthBackend.generate must raise, not silently substitute."""
        from backends.synth_backend import SynthBackend
        backend = SynthBackend()
        with pytest.raises(ValueError, match="[Ss]eed"):
            backend.generate("job", "C4 E4 G4", "happy", 120, "C", 5, seed=-1)


class TestArrangementTiming:
    """The arrangement must pass event start times to the engine."""

    def test_arrangement_note_starts_at_beat_position(self):
        """A note at beat 2 (of 120bpm) starts at sample 60000, not 0."""
        from synth.arrangement import Arrangement, ArrangementConfig
        config = ArrangementConfig(
            bpm=120.0, length=4.0, polyphony=8, sample_rate=SR,
            attack_samples=480, decay_samples=240, sustain_level=0.7,
            release_samples=480, filter_cutoff=20000.0, seed=42,
        )
        arr = Arrangement(config)
        # One note at beat 2.0 = 1.0s = sample 48000
        arr.add_note(time_beats=2.0, freq=440.0, duration_beats=0.5, amplitude=0.8)
        audio = arr.render()
        # Before sample 48000: silence
        assert np.max(np.abs(audio[:47900])) < 1e-9, (
            "Note scheduled at beat 2 is sounding before its start time"
        )
        # After: signal
        assert np.max(np.abs(audio[48100:96000])) > 0.01, "Note not sounding after its start"

    def test_arrangement_respects_each_event_time(self):
        """Two notes at different beats must not start together."""
        from synth.arrangement import Arrangement, ArrangementConfig
        config = ArrangementConfig(
            bpm=120.0, length=4.0, polyphony=8, sample_rate=SR,
            attack_samples=480, decay_samples=240, sustain_level=0.7,
            release_samples=480, filter_cutoff=20000.0, seed=42,
        )
        arr = Arrangement(config)
        arr.add_note(time_beats=1.0, freq=440.0, duration_beats=0.5, amplitude=0.8)
        arr.add_note(time_beats=3.0, freq=550.0, duration_beats=0.5, amplitude=0.8)
        audio = arr.render()
        # Beat 1 = 0.5s = 24000; beat 3 = 1.5s = 72000
        # Between the two notes (after first release, before second start):
        # first note ends at 24000+12000(hold)+480(release) ≈ 36480
        mid = audio[40000:70000]
        assert np.max(np.abs(mid)) < 0.05, (
            "Something is sounding between the two scheduled notes"
        )
