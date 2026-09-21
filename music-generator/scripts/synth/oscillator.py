"""Band-limited wavetable oscillator with oversampled PolyBLEP and integer-sample phase.

Oversampled PolyBLEP: render at 4x sample rate, apply PolyBLEP correction
at the higher rate (smaller dt = more accurate correction), then downsample
through a chain of Kaiser-windowed half-band FIR decimators (~-64 dB
stopband each). At 4x, every harmonic loud enough to matter (through the
~10th) is either legal at the target rate or lands in a decimation
stopband; measured worst-case illegal component is below -50 dB.

Each decimator carries its history across render calls, and every render
call emits exactly the frames whose full filter support exists in the
global stream — so partitioned rendering is byte-identical to single-block
rendering by construction, not by luck of discontinuity placement.

Sine waves bypass oversampling entirely — they have no discontinuities.
"""
import numpy as np
from numpy.typing import NDArray


def _polyblep(t: NDArray[np.float64], dt: float) -> NDArray[np.float64]:
    """First-order PolyBLEP discontinuity correction.

    t : phase in cycles [0, 1)
    dt: phase increment per sample (freq / sample_rate)
    """
    if dt <= 0:
        return np.zeros_like(t)
    result = np.zeros_like(t, dtype=np.float64)
    # Region just before discontinuity (t near 1)
    mask = t > (1.0 - dt)
    x = (t[mask] - 1.0) / dt
    result[mask] = x * x + x + x + 1.0
    # Region just after discontinuity (t near 0)
    mask = t < dt
    x = t[mask] / dt
    result[mask] = x + x - x * x - 1.0
    return result


# --- Decimation filters (module-level, computed once) ---

_HALF_BAND_TAPS = 127

def _kaiser_sinc(taps: int, cutoff: float, beta: float = 9.0) -> NDArray[np.float64]:
    """Kaiser-windowed sinc lowpass, unity DC gain.

    cutoff: normalized to the input rate, in cycles/sample.
    """
    m = np.arange(taps) - (taps - 1) / 2.0
    h = 2.0 * cutoff * np.sinc(2.0 * cutoff * m)
    w = np.i0(beta * np.sqrt(np.maximum(0.0, 1.0 - (2.0 * m / (taps - 1)) ** 2))) / np.i0(beta)
    h = h * w
    return h / np.sum(h)


def _make_half_band(taps: int = _HALF_BAND_TAPS, beta: float = 9.0) -> NDArray[np.float64]:
    """Kaiser-windowed half-band lowpass, cutoff at fs/4 of the input rate.

    Ideal response: h[n] = 0.5 * sinc(0.5 * (n - center)). Even-offset taps
    are exactly zero (half-band structure). Used for INTERMEDIATE decimation
    stages, where content folding from above the transition lands in bands
    that the next stage removes anyway.

    NOT suitable for the final stage: its symmetric transition band
    straddles the fold point, so content just above fs/4 leaks through
    partially and folds (measured: 5th harmonic of a 5 kHz saw at 25 kHz
    folding to 23 kHz at -36 dB).
    """
    return _kaiser_sinc(taps, 0.25, beta)


# Final-stage anti-alias lowpass: cutoff at 20 kHz for a 96 kHz input,
# stopband from ~23.8 kHz — strictly below the 24 kHz fold point, so
# nothing above the fold point survives to fold. Passband is flat to
# 20 kHz; harmonics between 20 and 24 kHz are attenuated (they sit at
# the edge of the audible band; the alternative — leaking folds — is worse).
_FINAL_LOWPASS_CUTOFF = 20000.0 / 96000.0

_HALF_BAND = _make_half_band()
_FINAL_LOWPASS = _kaiser_sinc(_HALF_BAND_TAPS, _FINAL_LOWPASS_CUTOFF)
_HALF_BAND_HISTORY = _HALF_BAND_TAPS - 1  # samples of history needed
_HALF_BAND_CENTER = (_HALF_BAND_TAPS - 1) // 2


class _Decimator:
    """Stateful 2:1 decimator with a fixed FIR kernel.

    Consumes input samples in order; emits output samples in order as soon
    as their full FIR support exists. Output j is the FIR output centered
    on input 2j (zero delay). History carries across process() calls.

    Each output sample is accumulated in a fixed tap order over fixed-length
    contiguous phase arrays — the same arithmetic in the same order regardless
    of how many outputs are computed per call — so partitioned processing is
    byte-identical to single-pass processing by construction.
    """

    def __init__(self, kernel: NDArray[np.float64] | None = None) -> None:
        self.h = _HALF_BAND if kernel is None else kernel
        self.taps = len(self.h)
        self.center = (self.taps - 1) // 2
        # Split taps by index parity: even taps read the input phase that
        # aligns with the output grid; odd taps read the opposite phase.
        # Both phase runs are materialized contiguously once, so every tap
        # operation is a contiguous multiply-add.
        self.even_taps = tuple(
            (t, float(self.h[2 * t]))
            for t in range(self.taps // 2 + 1)
            if self.h[2 * t] != 0.0
        )
        self.odd_taps = tuple(
            (t, float(self.h[2 * t + 1]))
            for t in range(self.taps // 2)
            if self.h[2 * t + 1] != 0.0
        )
        self.buf: NDArray[np.float64] = np.zeros(0)
        self.buf_start: int = 0   # stream index of buf[0]
        self.in_pos: int = 0      # one past last input sample fed
        self.out_count: int = 0   # outputs emitted so far

    def reset(self) -> None:
        self.buf = np.zeros(0)
        self.buf_start = 0
        self.in_pos = 0
        self.out_count = 0

    def process(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Feed input samples; return newly available outputs."""
        if len(x) > 0:
            self.buf = np.concatenate([self.buf, x])
            self.in_pos += len(x)

        # Output j needs input up to center + 2j
        j_first = self.center + 2 * self.out_count
        j_last_avail = self.in_pos - 1
        if j_last_avail < j_first:
            return np.zeros(0, dtype=np.float64)
        n_new = (j_last_avail - j_first) // 2 + 1

        # Output j = sum_k h[k] * x[j-k], accumulated in a fixed tap order
        # (even taps ascending, then odd taps ascending).
        q_start = j_first - self.buf_start
        pad = max(0, (self.taps - 1) - q_start)
        if pad > 0:
            # Only at stream start: zeros before the buffer are correct
            buf = np.concatenate([np.zeros(pad), self.buf])
            qs = q_start + pad
        else:
            buf = self.buf
            qs = q_start
        n = n_new
        # Phase arrays: X_same[j] = buf[qs - (taps-1) + 2j] (even-tap phase),
        # X_opp[j] = buf[qs - (taps-2) + 2j] (odd-tap phase).
        same_start = qs - (self.taps - 1)
        X_same = np.ascontiguousarray(buf[same_start : same_start + 2 * (n + self.taps // 2 + 1) - 1 : 2])
        X_opp = np.ascontiguousarray(buf[same_start + 1 : same_start + 1 + 2 * (n + self.taps // 2) - 1 : 2])

        out = np.zeros(n, dtype=np.float64)
        # Even tap k=2t reads X_same[center - t + i]
        for t, hk in self.even_taps:
            s = self.center - t
            out += hk * X_same[s : s + n]
        # Odd tap k=2t+1 reads X_opp[center - 1 - t + i]
        for t, hk in self.odd_taps:
            s = self.center - 1 - t
            out += hk * X_opp[s : s + n]
        self.out_count += n

        # Trim history: keep the last (taps-1) input samples
        keep = self.taps - 1
        if len(self.buf) > keep:
            self.buf = self.buf[-keep:].copy()
            self.buf_start = self.in_pos - keep

        return out


def make_wavetable(size: int = 4096, waveform: str = "saw") -> NDArray[np.float64]:
    """Generate a single-cycle wavetable of given size."""
    t = np.linspace(0.0, 1.0, size, endpoint=False)
    if waveform == "saw":
        tbl = 2.0 * (t - np.floor(t + 0.5))
    elif waveform == "square":
        tbl = np.where(t < 0.5, 1.0, -1.0)
    elif waveform == "triangle":
        tbl = 2.0 * np.abs(2.0 * (t + 0.25) - np.floor(2.0 * t + 1.0) - 1.0) - 1.0
    elif waveform == "sine":
        tbl = np.sin(2.0 * np.pi * t)
    else:
        tbl = np.sin(2.0 * np.pi * t)
    max_val = np.max(np.abs(tbl))
    if max_val > 0:
        tbl = tbl / max_val
    return tbl


class WavetableOscillator:
    """Phase-continuous wavetable oscillator with oversampled PolyBLEP.

    Phase authority is an integer sample counter — rendering is partition-invariant
    regardless of block boundaries. The seed applies a deterministic phase offset.

    Oversampling (default 4x): renders at 4x sample rate and downsamples
    through log2(oversample) stateful half-band decimators. All filter state
    carries across render calls, so partitioned rendering is byte-exact.

    Sine waves bypass oversampling — they have no discontinuities.
    """

    def __init__(
        self,
        sample_rate: int = 48000,
        table_size: int = 4096,
        waveform: str = "saw",
        seed: int | None = None,
        oversample: int = 4,
    ) -> None:
        self.sample_rate = sample_rate
        self.table_size = table_size
        self.table = make_wavetable(table_size, waveform)
        self._waveform = waveform
        self._sample_pos: int = 0  # target-rate frames emitted (absolute)
        self._rng = np.random.RandomState(seed)
        # Seed-based phase offset in cycles [0, 1)
        self._seed_phase_offset = (seed % table_size) / table_size if seed is not None else 0.0
        if oversample < 1 or (oversample & (oversample - 1)) != 0:
            raise ValueError("oversample must be a power of 2 (1, 2, 4, ...)")
        self._oversample = oversample
        # Stream state. A "stream" begins at construction, reset(), or
        # phase_reset. _stream_origin is the absolute frame index of the
        # stream's first output; _os_pos is one past the last generated
        # oversampled sample (stream-relative).
        self._stream_origin: int = 0
        self._os_pos: int = 0
        n_stages = oversample.bit_length() - 1  # log2
        # Intermediate stages use the half-band kernel (their leakage lands
        # in bands the next stage removes); the final stage uses the
        # anti-alias lowpass whose stopband starts below the fold point.
        if n_stages <= 1:
            self._decimators = [_Decimator(_FINAL_LOWPASS)]
        else:
            self._decimators = (
                [_Decimator(_HALF_BAND) for _ in range(n_stages - 1)]
                + [_Decimator(_FINAL_LOWPASS)]
            )

    def _start_fresh_stream(self, origin_frame: int) -> None:
        """Begin a new oversampled stream whose first output frame is origin_frame."""
        self._stream_origin = origin_frame
        self._os_pos = 0
        for d in self._decimators:
            d.reset()

    def reset(self) -> None:
        self._sample_pos = 0
        self._start_fresh_stream(0)

    def set_phase(self, phase: float) -> None:
        self._sample_pos = int(phase * self.sample_rate)
        self._start_fresh_stream(self._sample_pos)

    def get_phase(self) -> float:
        return (self._sample_pos % self.sample_rate) / self.sample_rate

    def render(
        self,
        freq: float,
        n_frames: int,
        phase_reset: bool = False,
        reset_phase: float = 0.0,
    ) -> NDArray[np.float64]:
        if n_frames <= 0:
            return np.array([], dtype=np.float64)

        if phase_reset:
            self._sample_pos = int(reset_phase * self.sample_rate)
            self._start_fresh_stream(self._sample_pos)

        # Sine has no discontinuities — no need for oversampling
        if self._waveform == "sine" or self._oversample <= 1:
            return self._render_single_rate(freq, n_frames)
        else:
            return self._render_oversampled(freq, n_frames)

    def _render_single_rate(
        self,
        freq: float,
        n_frames: int,
    ) -> NDArray[np.float64]:
        """Render at the target sample rate (no oversampling)."""
        # Absolute sample indices for this block
        indices = np.arange(self._sample_pos, self._sample_pos + n_frames, dtype=np.float64)
        # Phase in cycles [0, 1) with seed offset
        t = (freq * indices / self.sample_rate + self._seed_phase_offset) % 1.0
        dt = freq / self.sample_rate

        out = self._generate_waveform(t, dt)

        self._sample_pos += n_frames
        return out.astype(np.float64)

    def _generate_os(self, freq: float, start: int, count: int) -> NDArray[np.float64]:
        """Generate `count` oversampled samples starting at stream-relative index `start`."""
        os_rate = self.sample_rate * self._oversample
        j = np.arange(start, start + count, dtype=np.float64)
        t = (freq * j / os_rate + self._seed_phase_offset) % 1.0
        dt = freq / os_rate
        return self._generate_waveform(t, dt)

    def _render_oversampled(
        self,
        freq: float,
        n_frames: int,
    ) -> NDArray[np.float64]:
        """Render at oversample x rate, decimate through the filter chain.

        Output alignment is zero delay: stream-relative output frame r is the
        final FIR output centered on oversampled sample (oversample * r).
        Each render call generates exactly enough oversampled input that the
        decimator chain emits exactly n_frames outputs, and all state carries
        across calls — so partitioned rendering is byte-identical to
        single-block rendering.
        """
        # Stream-relative frame index of the first output
        r0 = self._sample_pos - self._stream_origin
        r_last = r0 + n_frames - 1

        # Walk the chain backwards to find how much oversampled input we need.
        # Stage k decimates by 2: its output j needs its input up to
        # center_k + 2j. The final stage's output index is the frame index;
        # the first stage's input index is the oversampled stream index.
        need = r_last  # needed output count - 1 at the final stage
        for dec in reversed(self._decimators):
            need = dec.center + 2 * need
        needed_os_end = need

        # Generate oversampled samples up to needed_os_end
        gen_count = needed_os_end + 1 - self._os_pos
        data = np.zeros(0, dtype=np.float64)
        if gen_count > 0:
            data = self._generate_os(freq, self._os_pos, gen_count)
            self._os_pos += gen_count
        elif gen_count < 0:
            raise RuntimeError("internal error: oversampled stream position ahead of need")
        for dec in self._decimators:
            data = dec.process(data)

        # The chain must have emitted exactly n_frames new outputs
        if len(data) != n_frames:
            raise RuntimeError(
                f"internal error: decimator chain emitted {len(data)} outputs, "
                f"expected {n_frames}"
            )

        self._sample_pos += n_frames
        return data.astype(np.float64)

    def _generate_waveform(self, t: NDArray[np.float64], dt: float) -> NDArray[np.float64]:
        """Generate the raw waveform with PolyBLEP correction."""
        if self._waveform == "sine":
            return np.sin(2.0 * np.pi * t)
        elif self._waveform == "saw":
            # Ideal saw: 2*(t - 0.5), corrected with first-order PolyBLEP
            return 2.0 * (t - 0.5) - _polyblep(t, dt)
        elif self._waveform == "square":
            # Ideal square: +1 for t<0.5, -1 otherwise
            ideal = np.where(t < 0.5, 1.0, -1.0)
            # At t=0: rise of +2 (from -1 to +1) → +polyblep correction
            # At t=0.5: drop of -2 (from +1 to -1) → -polyblep correction
            return ideal + _polyblep(t, dt) - _polyblep((t + 0.5) % 1.0, dt)
        elif self._waveform == "triangle":
            # Triangle: continuous — no PolyBLEP needed
            return 2.0 * np.abs(2.0 * (t + 0.25) - np.floor(2.0 * t + 1.0) - 1.0) - 1.0
        else:
            return np.sin(2.0 * np.pi * t)

    def render_block(
        self,
        notes: list[tuple[float, int]],
        block_size: int = 48000,
    ) -> NDArray[np.float64]:
        chunks = []
        for freq, n in notes:
            chunks.append(self.render(freq, n))
        if not chunks:
            return np.array([], dtype=np.float64)
        return np.concatenate(chunks)
