"""
Worker daemon for processing music generation jobs.
Polls the queue, processes pending jobs with the selected backend,
and updates job status.
"""

import time
import signal
import sys
from pathlib import Path

# Add scripts directory to path
SCRIPTS_DIR = Path(__file__).parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from config import QUEUE_TYPE, POLL_INTERVAL, JOB_TIMEOUT, BACKEND_TYPE
from job_queue import get_queue, QueueItem
from backends.base_backend import BaseBackend


def get_backend() -> BaseBackend:
    """Instantiate the configured backend."""
    if BACKEND_TYPE == "mock":
        from backends.mock_backend import MockBackend
        return MockBackend()
    else:
        # Future: import real backend based on config
        raise NotImplementedError(f"Backend type '{BACKEND_TYPE}' not implemented")


def process_job(queue, backend, item: QueueItem) -> None:
    """Process a single job."""
    print(f"Processing job {item.job_id}: {item.prompt[:50]}...")
    item.status = "processing"
    queue.update_item(item)

    try:
        result = backend.generate(
            job_id=item.job_id,
            prompt=item.prompt,
            mood=item.mood,
            tempo=item.tempo,
            key=item.key,
            length=item.length
        )
        if result.get("status") == "success":
            item.status = "completed"
            item.result = result
        else:
            item.status = "failed"
            item.result = result
    except Exception as e:
        item.status = "failed"
        item.result = {
            "status": "error",
            "error": f"Worker exception: {str(e)}"
        }
    finally:
        queue.update_item(item)
        print(f"Job {item.job_id} finished with status: {item.status}")


def main():
    print("Starting music generator worker...")
    print(f"Queue type: {QUEUE_TYPE}")
    print(f"Backend type: {BACKEND_TYPE}")
    print(f"Poll interval: {POLL_INTERVAL}s")
    print(f"Job timeout: {JOB_TIMEOUT}s")

    queue = get_queue()
    backend = get_backend()

    # Recover any jobs stuck in processing state after a crash
    recovered = queue.recover()
    if recovered:
        print(f"Recovered {len(recovered)} job(s) from previous crash")
        for item in recovered:
            process_job(queue, backend, item)

    def signal_handler(sig, frame):
        print("\nShutting down worker...")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    while True:
        try:
            # Dequeue a pending job (FIFO)
            item = queue.dequeue()
            if item is None:
                # No pending jobs, wait
                time.sleep(POLL_INTERVAL)
                continue

            # Process the job
            process_job(queue, backend, item)

        except Exception as e:
            print(f"Worker error: {e}")
            time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()