#!/usr/bin/env python3
"""
Load test harness for music generator API.
Measures true per-job TTFB: each concurrent task does submit→poll→download
independently, so no job's timeline is contaminated by waiting for others.

Usage:
    python load_test.py --url http://localhost:8000 --jobs 50 --concurrency 10
"""

import argparse
import hashlib
import json
import os
import sys
import time
import statistics
from collections import defaultdict
from concurrent import futures as concurrent_futures
from pathlib import Path
from threading import Lock, Thread

try:
    import requests
except ImportError:
    print("ERROR: requests library required. pip install requests")
    sys.exit(1)


def make_job(seed: int, length: int = 5) -> dict:
    """Create a deterministic job payload."""
    moods = ["calm", "energetic", "dark", "happy", "dreamy"]
    keys = ["C", "D", "E", "F", "G", "A", "B"]
    return {
        "prompt": f"test seed {seed}",
        "mood": moods[seed % len(moods)],
        "key": keys[seed % len(keys)],
        "tempo": 120,
        "length": length,
        "seed": seed,
    }


def pct(data, p):
    """Compute percentile."""
    if not data:
        return 0
    sorted_data = sorted(data)
    idx = int(len(sorted_data) * p / 100)
    idx = min(idx, len(sorted_data) - 1)
    return sorted_data[idx]


def run_single_job_lifecycle(url: str, job: dict, poll_interval: float = 0.5,
                             timeout: float = 300.0) -> dict:
    """One complete lifecycle: submit → poll → download, with OWN session.

    Returns per-job timing so no barrier contamination.
    """
    result = {
        "job_id": None,
        "submit_start": None,
        "submit_202": None,
        "completed_at": None,
        "artifact_request_start": None,
        "first_byte_at": None,
        "download_completed": None,
        "artifact_size": 0,
        "artifact_hash": None,
        "status": "unknown",
        "error": None,
        "queued_at": None,
        "processing_at": None,
        "worker_id": None,
        "attempt": None,
    }
    session = requests.Session()
    try:
        # 1. Submit
        result["submit_start"] = time.time()
        resp = session.post(f"{url}/generate", json=job, timeout=30)
        result["submit_202"] = time.time()
        if resp.status_code != 202:
            result["status"] = "rejected"
            result["error"] = f"HTTP {resp.status_code}: {resp.text[:200]}"
            return result
        data = resp.json()
        job_id = data.get("job_id")
        result["job_id"] = job_id
        if not job_id:
            result["status"] = "no_id"
            result["error"] = "No job_id in response"
            return result

        # 2. Poll until complete (independent of other jobs)
        t0 = time.time()
        while time.time() - t0 < timeout:
            resp = session.get(f"{url}/status/{job_id}", timeout=10)
            if resp.status_code == 200:
                st = resp.json()
                status = st.get("status")
                if status in ("completed", "failed", "dead"):
                    result["status"] = status
                    result["queued_at"] = st.get("queued_at")
                    result["processing_at"] = st.get("processing_at")
                    result["completed_at"] = time.time()
                    result["worker_id"] = st.get("worker_id")
                    result["attempt"] = st.get("attempt")
                    break
            time.sleep(poll_interval)
        else:
            result["status"] = "timeout"
            result["error"] = f"Poll timeout after {timeout}s"
            return result

        if result["status"] != "completed":
            return result

        # 3. Download artifact (independent — not waiting for other jobs)
        result["artifact_request_start"] = time.time()
        resp = session.get(f"{url}/artifact/{job_id}", stream=True, timeout=60)
        content = b""
        first_byte_time = None
        for chunk in resp.iter_content(chunk_size=8192):
            if chunk:
                if first_byte_time is None:
                    first_byte_time = time.time()
                content += chunk
        result["download_completed"] = time.time()
        result["first_byte_at"] = first_byte_time
        result["artifact_size"] = len(content)
        result["artifact_hash"] = hashlib.sha256(content).hexdigest() if content else None

    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
    finally:
        session.close()
    return result


class QueueSampler:
    """Periodically sample queue depth in the background."""

    def __init__(self, url: str, interval: float = 2.0):
        self.url = url
        self.interval = interval
        self.samples = []
        self._stop = False
        self._lock = Lock()

    def start(self):
        self._thread = Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop = True
        if hasattr(self, '_thread'):
            self._thread.join(timeout=5)

    def _run(self):
        while not self._stop:
            try:
                resp = requests.get(f"{self.url}/queue/status", timeout=5)
                if resp.status_code == 200:
                    data = resp.json()
                    with self._lock:
                        self.samples.append({
                            "time": time.time(),
                            **data,
                        })
            except Exception:
                pass
            time.sleep(self.interval)

    def get_samples(self):
        with self._lock:
            return list(self.samples)


def run_load_test(url: str, num_jobs: int, concurrency: int, job_length: int = 5):
    """Run the full load test with per-job independent lifecycle."""
    print(f"Load Test: {num_jobs} jobs, concurrency={concurrency}, length={job_length}s")
    print(f"Target: {url}")
    print()

    # Health check
    try:
        resp = requests.get(f"{url}/health", timeout=5)
        resp.raise_for_status()
    except Exception as e:
        print(f"ERROR: Health check failed: {e}")
        sys.exit(1)
    health = resp.json()
    print(f"Server: queue={health.get('queue')}, backend={health.get('backend')}, depth={health.get('queue_depth')}")
    print()

    # Start queue depth sampler
    sampler = QueueSampler(url=url, interval=2.0)
    sampler.start()

    # Submit + lifecycle all jobs with independent per-job timelines
    print(f"Running {num_jobs} jobs with independent lifecycle...")
    t_start = time.time()

    results = []
    with concurrent_futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = []
        for i in range(num_jobs):
            job = make_job(seed=i + 1, length=job_length)
            future = executor.submit(run_single_job_lifecycle, url, job)
            futures.append(future)

        for future in concurrent_futures.as_completed(futures):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                results.append({"status": "error", "error": str(e)})

    t_done = time.time()
    sampler.stop()

    print(f"All jobs finished in {t_done - t_start:.2f}s")

    # Compute metrics
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    # Status breakdown
    status_counts = defaultdict(int)
    for r in results:
        status_counts[r.get("status", "unknown")] += 1
    print("\nCompletion Status:")
    for status, count in sorted(status_counts.items()):
        print(f"  {status}: {count}")

    # Submission latency
    submit_lates = []
    for r in results:
        if r.get("submit_start") and r.get("submit_202"):
            submit_lates.append(r["submit_202"] - r["submit_start"])
    if submit_lates:
        print(f"\nSubmission Latency (POST → 202):")
        print(f"  p50: {pct(submit_lates, 50)*1000:.1f} ms")
        print(f"  p95: {pct(submit_lates, 95)*1000:.1f} ms")
        print(f"  p99: {pct(submit_lates, 99)*1000:.1f} ms")
        print(f"  max: {max(submit_lates)*1000:.1f} ms")

    # Queue wait + render time (from job's own timestamps)
    queue_waits = []
    render_times = []
    for r in results:
        qa = r.get("queued_at")
        pa = r.get("processing_at")
        ca = r.get("completed_at")
        if qa and pa and ca:
            queue_waits.append(pa - qa)
            # completed_at is from our poll, not server's completed_at
            # But we stored completed_at from status response... actually we used time.time()
            # Let's use the stored completed_at which is when poll saw it complete
            # This is approximate but consistent
            render_times.append(ca - pa)

    if queue_waits:
        print(f"\nQueue Wait (queued → processing):")
        print(f"  p50: {pct(queue_waits, 50)*1000:.1f} ms")
        print(f"  p95: {pct(queue_waits, 95)*1000:.1f} ms")
        print(f"  p99: {pct(queue_waits, 99)*1000:.1f} ms")
        print(f"  max: {max(queue_waits)*1000:.1f} ms")

    if render_times:
        print(f"\nRender+Poll Time (processing → complete seen by client):")
        print(f"  p50: {pct(render_times, 50)*1000:.1f} ms")
        print(f"  p95: {pct(render_times, 95)*1000:.1f} ms")
        print(f"  p99: {pct(render_times, 99)*1000:.1f} ms")

    # Artifact TTFB (from own request start to first byte)
    artifact_ttfb = []
    for r in results:
        if r.get("artifact_request_start") and r.get("first_byte_at"):
            artifact_ttfb.append(r["first_byte_at"] - r["artifact_request_start"])
    if artifact_ttfb:
        print(f"\nArtifact TTFB (GET → first byte, per job):")
        print(f"  p50: {pct(artifact_ttfb, 50)*1000:.1f} ms")
        print(f"  p95: {pct(artifact_ttfb, 95)*1000:.1f} ms")
        print(f"  p99: {pct(artifact_ttfb, 99)*1000:.1f} ms")
        print(f"  max: {max(artifact_ttfb)*1000:.1f} ms")

    # TRUE end-to-end: submit → first byte (independent per job)
    true_e2e = []
    for r in results:
        if r.get("submit_start") and r.get("first_byte_at"):
            true_e2e.append(r["first_byte_at"] - r["submit_start"])
    if true_e2e:
        print(f"\nTRUE End-to-End (submit → first WAV byte, NO barrier):")
        print(f"  p50: {pct(true_e2e, 50)*1000:.1f} ms")
        print(f"  p95: {pct(true_e2e, 95)*1000:.1f} ms")
        print(f"  p99: {pct(true_e2e, 99)*1000:.1f} ms")
        print(f"  max: {max(true_e2e)*1000:.1f} ms")

    # Artifact sizes
    sizes = [r["artifact_size"] for r in results if r.get("artifact_size", 0) > 0]
    if sizes:
        print(f"\nArtifact Sizes:")
        print(f"  min: {min(sizes):,} bytes")
        print(f"  max: {max(sizes):,} bytes")
        print(f"  mean: {statistics.mean(sizes):,.0f} bytes")

    # EXACT-ONCE processing check
    # Group by job status and check for duplicate worker assignments
    print(f"\nExact-Once Processing Check:")
    submitted_count = sum(1 for r in results if r.get("job_id"))
    completed_count = sum(1 for r in results if r.get("status") == "completed")
    print(f"  Submitted: {submitted_count}")
    print(f"  Completed: {completed_count}")

    # Check for duplicate worker processing (same job_id, different workers or multiple attempts)
    job_worker_map = defaultdict(list)
    for r in results:
        if r.get("job_id") and r.get("worker_id"):
            job_worker_map[r["job_id"]].append(r.get("worker_id"))

    multi_worker_jobs = {k: v for k, v in job_worker_map.items() if len(v) > 1}
    if multi_worker_jobs:
        print(f"  WARN: {len(multi_worker_jobs)} jobs processed by multiple workers")
        for jid, workers in list(multi_worker_jobs.items())[:3]:
            print(f"    {jid}: {workers}")
    else:
        print(f"  No multi-worker jobs detected")

    # Check attempt counts
    attempts = [r["attempt"] for r in results if r.get("attempt") is not None]
    if attempts:
        print(f"  Attempt counts: min={min(attempts)}, max={max(attempts)}, mean={statistics.mean(attempts):.1f}")
        multi_attempt = sum(1 for a in attempts if a > 1)
        if multi_attempt:
            print(f"  WARN: {multi_attempt} jobs required multiple attempts")

    # Artifact hash uniqueness (detect double-renders of same job)
    hash_to_jobs = defaultdict(list)
    for r in results:
        if r.get("artifact_hash"):
            hash_to_jobs[r["artifact_hash"]].append(r.get("job_id"))
    dup_hashes = {k: v for k, v in hash_to_jobs.items() if len(v) > 1}
    if dup_hashes:
        print(f"  WARN: {len(dup_hashes)} artifact hashes shared by multiple jobs")
    else:
        print(f"  All artifact hashes unique")

    # Final queue state
    try:
        resp = requests.get(f"{url}/queue/status", timeout=5)
        if resp.status_code == 200:
            qs = resp.json()
            print(f"\nFinal Queue State:")
            for k, v in sorted(qs.items()):
                print(f"  {k}: {v}")
    except Exception as e:
        print(f"\nFinal queue state unavailable: {e}")

    # Queue depth over time
    samples = sampler.get_samples()
    if samples:
        depths = [s.get("total_active", s.get("stream_length", 0)) for s in samples]
        if depths:
            print(f"\nQueue Depth Over Time (sampled every 2s):")
            print(f"  min: {min(depths)}")
            print(f"  max: {max(depths)}")
            print(f"  mean: {statistics.mean(depths):.1f}")
            print(f"  samples: {len(depths)}")

    # Throughput
    total_wall = t_done - t_start
    print(f"\nThroughput:")
    print(f"  Total wall time: {total_wall:.2f}s")
    print(f"  Jobs/s: {completed_count / total_wall:.2f}")


def main():
    parser = argparse.ArgumentParser(description="Music Generator Load Test")
    parser.add_argument("--url", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--jobs", type=int, default=10, help="Number of jobs to submit")
    parser.add_argument("--concurrency", type=int, default=5, help="Concurrent submissions")
    parser.add_argument("--length", type=int, default=5, help="WAV length in seconds")
    args = parser.parse_args()

    run_load_test(args.url, args.jobs, args.concurrency, args.length)


if __name__ == "__main__":
    main()
