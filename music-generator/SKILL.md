---
name: music-generator
description: "Music generation system with job queue and pluggable backends (mock and real)."
---
# Music Generator

A modular music generation system with job queue, mock backend, and extensible architecture for integrating real music generation APIs (e.g., Suno, MusicGen).

## Overview

This skill provides a foundation for building a music generation service that:
- Accepts generation requests via a job queue (file-based or Redis)
- Processes jobs with pluggable backends (mock or real)
- Includes a mock backend for testing and development
- Provides a worker daemon to process the queue
- Designed for extension to real backends via simple subclassing

## Components

- **Config**: Central configuration (`scripts/config.py`)
- **Queue**: File-based or Redis-backed job queue (`scripts/job_queue.py`)
- **Backend**: Abstract base class (`scripts/backends/base_backend.py`) with mock implementation (`scripts/backends/mock_backend.py`)
- **Worker**: Daemon that polls the queue and processes jobs (`scripts/worker.py`)
- **Job Item**: Data class representing a generation request (`scripts/job_queue.py`)

## Installation

1. Ensure Python 3.8+ is installed.
2. Install dependencies (if any): `pip install redis` (optional, for Redis queue).
3. The skill is self-contained under `~/workspace/skills/music-generator/`.

## Usage

### Configuration

Edit `scripts/config.py` or set environment variables:

- `MG_QUEUE_TYPE`: `file` (default) or `redis`
- `MG_BACKEND_TYPE`: `mock` (default) or `real` (future)
- `MG_HOST`, `MG_PORT`: For future HTTP API
- `MG_JOB_TIMEOUT`: Seconds before job timeout (default 30)
- `MG_MOCK_MIN_DELAY`, `MG_MOCK_MAX_DELAY`: Mock processing delay range (seconds)

### Enqueue a Job

```python
from scripts.job_queue import get_queue, QueueItem

queue = get_queue()
item = QueueItem(
    job_id="gen_001",  # optional; if omitted, a UUID is generated
    prompt="A warm lofi beat about coding",
    mood="chill",
    tempo=90,
    key="C",
    length=30
)
queue.enqueue(item)
```

### Start the Worker

```bash
cd ~/workspace/skills/music-generator/scripts
python worker.py
```

The worker will process pending jobs, update their status, and store results.

### Mock Backend Output

The mock backend generates:
- A WAV file with a simple sine wave (if `scipy` and `numpy` are installed)
- Otherwise, a text file with job metadata

Outputs are placed in `~/workspace/skills/music-generator/output/`.

## Extending to a Real Backend

To integrate a real music generation API:

1. Create a new backend class in `scripts/backends/` that inherits from `BaseBackend`.
2. Implement the `generate` method to call your API, store the output, and return a success dictionary.
3. Set `MG_BACKEND_TYPE=real` (or specify via config) and ensure the worker picks up your backend (you may need to modify `get_backend()` in `worker.py` to import your new class).

## Notes

- The mock backend is intended for development and testing only.
- For production, replace the mock with a real backend that interfaces with a music generation service.
- The queue system ensures durable job storage and retry capability.
- Future enhancements: HTTP API endpoint, web UI, priority queue, retry logic.

## References

- See `references/queue-design.md` for details on queue implementation choices.
- See `references/backend-interface.md` for the backend contract.
- See `templates/backend_template.py` for a starter backend template.