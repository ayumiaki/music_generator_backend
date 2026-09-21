"""Render I/O helpers: temp paths, atomic rename, and cleanup of orphaned partials."""

import os
from pathlib import Path


def temp_path(output_dir: str | Path, worker_id: str, job_id: str) -> Path:
    """Worker-specific temp render path: .tmp.<worker_id>.<job_id>.wav"""
    out = Path(output_dir)
    return out / f".tmp.{worker_id}.{job_id}.wav"


def final_path(output_dir: str | Path, job_id: str) -> Path:
    """Final artifact path after successful render."""
    return Path(output_dir) / f"{job_id}.wav"


def rename_to_final(temp: Path, final: Path) -> None:
    """Atomically move temp to final. os.rename is atomic on same filesystem."""
    os.replace(temp, final)  # os.replace is atomic and overwrites if exists


def cleanup_worker_temp(output_dir: str | Path, worker_id: str) -> list[Path]:
    """Remove all .tmp.<worker_id>.* files. Returns list of removed paths."""
    out = Path(output_dir)
    removed = []
    for f in out.glob(f".tmp.{worker_id}.*"):
        try:
            f.unlink()
            removed.append(f)
        except OSError:
            pass
    return removed


def cleanup_orphan_temp(output_dir: str | Path, known_worker_ids: set[str] | None = None) -> list[Path]:
    """Remove .tmp.* files from workers not in known_worker_ids.

    If known_worker_ids is None, removes ALL .tmp.* files (nuclear cleanup on startup).
    """
    out = Path(output_dir)
    removed = []
    for f in out.glob(".tmp.*.*.wav"):
        parts = f.name.split(".")
        # .tmp.<worker_id>.<job_id>.wav => worker_id is parts[2]
        if len(parts) >= 4:
            wid = parts[2]
            if known_worker_ids is None or wid not in known_worker_ids:
                try:
                    f.unlink()
                    removed.append(f)
                except OSError:
                    pass
    return removed
