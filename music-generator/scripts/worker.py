"""
Worker daemon for processing music generation jobs.
Polls the queue, processes pending jobs with the selected backend,
and updates job status.
"""

import signal
import time
import sys
import os
from pathlib import Path
from typing import Optional

# Add scripts directory to path
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from config import QUEUE_TYPE, POLL_INTERVAL, JOB_TIMEOUT, BACKEND_TYPE
from job_queue import get_queue, QueueItem
from backends.base_backend import BaseBackend


class JobTimeoutError(Exception):
    pass


def _timeout_handler(signum, frame):
    raise JobTimeoutError(f"Job exceeded timeout of {JOB_TIMEOUT}s")


def get_backend() -> BaseBackend:
    """Instantiate the configured backend."""
    if BACKEND_TYPE == "mock":
        from backends.mock_backend import MockBackend
        return MockBackend()
    elif BACKEND_TYPE == "synth":
        from backends.synth_backend import SynthBackend
        return SynthBackend()
    else:
        raise NotImplementedError(f"Backend type '{BACKEND_TYPE}' not implemented")


def process_job(queue, backend, item: QueueItem, worker_id: Optional[str] = None) -> None:
    """Process a single job."""
    print(f"Processing job {item.job_id}: {item.prompt[:50]}...")
    # Note: item.status is already "processing" from dequeue()

    # Install alarm-based timeout before calling backend
    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    old_alarm = signal.alarm(JOB_TIMEOUT)
    render_start = time.perf_counter()
    try:
        result = backend.generate(
            job_id=item.job_id,
            prompt=item.prompt,
            mood=item.mood,
            tempo=item.tempo,
            key=item.key,
            length=item.length,
            seed=item.seed,
        )
        signal.alarm(0)  # cancel alarm on success
        render_duration_ms = (time.perf_counter() - render_start) * 1000
        result["render_duration_ms"] = render_duration_ms
        if result.get("status") == "success":
            queue.complete(item, result, worker_id=worker_id)
            print(f"Job {item.job_id} completed successfully")
        else:
            queue.fail(item, result.get("error", "Backend returned failure"), worker_id=worker_id)
            print(f"Job {item.job_id} failed: {result.get('error')}")
    except JobTimeoutError:
        signal.alarm(0)
        render_duration_ms = (time.perf_counter() - render_start) * 1000
        item.result["render_duration_ms"] = render_duration_ms
        queue.fail(item, f"Job exceeded timeout of {JOB_TIMEOUT}s", worker_id=worker_id)
        print(f"Job {item.job_id} timed out after {JOB_TIMEOUT}s")
    except Exception as e:
        signal.alarm(0)
        render_duration_ms = (time.perf_counter() - render_start) * 1000
        item.result["render_duration_ms"] = render_duration_ms
        queue.fail(item, f"Worker exception: {str(e)}", worker_id=worker_id)
        print(f"Job {item.job_id} failed with exception: {e}")
    finally:
        # Restore previous alarm state
        if old_alarm:
            signal.alarm(old_alarm)
        else:
            signal.signal(signal.SIGALRM, old_handler)


def main():
    print("Starting music generator worker...")
    print(f"Queue type: {QUEUE_TYPE}")
    print(f"Backend type: {BACKEND_TYPE}")
    print(f"Poll interval: {POLL_INTERVAL}s")
    print(f"Job timeout: {JOB_TIMEOUT}s")

    queue = get_queue()
    backend = get_backend()
    worker_id = f"worker-{os.getpid()}"

    # Recover any jobs stuck in processing state after a crash
    recovered = queue.recover()
    if recovered:
        print(f"Recovered {len(recovered)} job(s) from previous crash")
        for item in recovered:
            process_job(queue, backend, item, worker_id=worker_id)

    def signal_handler(sig, frame):
        print("\nShutting down worker...")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    while True:
        try:
            item = queue.dequeue(worker_id=worker_id)
            if item is None:
                time.sleep(POLL_INTERVAL)
                continue
            process_job(queue, backend, item, worker_id=worker_id)
        except Exception as e:
            print(f"Worker error: {e}")
            time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()