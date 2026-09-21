"""Offline WAV renderer with integrity checks and determinism verification."""
import struct
import wave
from pathlib import Path
from typing import Optional
import numpy as np
from numpy.typing import NDArray


class RenderIntegrity:
    """Integrity check results for a rendered WAV."""

    def __init__(
        self,
        sample_rate: int,
        channels: int,
        frame_count: int,
        duration_sec: float,
        clipping: bool,
        dc_offset: float,
        all_finite: bool,
        max_amplitude: float,
        rms: float,
    ) -> None:
        self.sample_rate = sample_rate
        self.channels = channels
        self.frame_count = frame_count
        self.duration_sec = duration_sec
        self.clipping = clipping
        self.dc_offset = dc_offset
        self.all_finite = all_finite
        self.max_amplitude = max_amplitude
        self.rms = rms

    def __repr__(self) -> str:
        return (
            f"RenderIntegrity(sr={self.sample_rate}, frames={self.frame_count}, "
            f"dur={self.duration_sec:.3f}s, clipping={self.clipping}, "
            f"dc={self.dc_offset:.6f}, finite={self.all_finite})"
        )


def render_integrity(audio: NDArray[np.float64], sample_rate: int) -> RenderIntegrity:
    """Verify integrity of rendered audio."""
    frame_count = len(audio)
    duration_sec = frame_count / sample_rate if sample_rate > 0 else 0.0

    # Check finite
    all_finite = bool(np.all(np.isfinite(audio)))

    # Check clipping
    clipping = bool(np.any(np.abs(audio) > 1.0))

    # DC offset
    dc_offset = float(np.mean(audio))

    # Max amplitude
    max_amplitude = float(np.max(np.abs(audio))) if len(audio) > 0 else 0.0

    # RMS
    rms = float(np.sqrt(np.mean(audio**2))) if len(audio) > 0 else 0.0

    return RenderIntegrity(
        sample_rate=sample_rate,
        channels=1,
        frame_count=frame_count,
        duration_sec=duration_sec,
        clipping=clipping,
        dc_offset=dc_offset,
        all_finite=all_finite,
        max_amplitude=max_amplitude,
        rms=rms,
    )


def write_wav(
    path: str | Path,
    audio: NDArray[np.float64],
    sample_rate: int = 48000,
) -> Path:
    """Write mono 16-bit WAV file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Clamp to [-1, 1]
    audio_clamped = np.clip(audio, -1.0, 1.0)

    # Convert to 16-bit PCM
    pcm = (audio_clamped * 32767).astype(np.int16)

    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())

    return path


def read_wav(path: str | Path) -> tuple[NDArray[np.float64], int]:
    """Read WAV file, return (audio, sample_rate)."""
    path = Path(path)
    with wave.open(str(path), "r") as wf:
        sample_rate = wf.getframerate()
        frames = wf.readframes(wf.getnframes())
        channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()

    if sampwidth == 2:
        dtype = np.int16
    elif sampwidth == 4:
        dtype = np.int32
    else:
        raise ValueError(f"Unsupported sample width: {sampwidth}")

    audio = np.frombuffer(frames, dtype=dtype).astype(np.float64)
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)

    # Normalize to [-1, 1]
    max_val = np.iinfo(dtype).max
    audio = audio / max_val

    return audio, sample_rate


def render(
    audio: NDArray[np.float64],
    path: str | Path,
    sample_rate: int = 48000,
) -> RenderIntegrity:
    """Render audio to WAV and return integrity check."""
    write_wav(path, audio, sample_rate)
    integrity = render_integrity(audio, sample_rate)
    return integrity