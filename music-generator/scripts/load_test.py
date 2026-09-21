#!/usr/bin/env python3
"""
Load test harness for music generator API.
Measures real latency under concurrency with full lifecycle timing.

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
from concurrent import futures as concurrent_futures
from pathlib import Path

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


def submit_job(session: requests.Session, url: str, job: dict) -> dict:
    """Submit a job and return timing info."""
    t0 = time.time()
    resp = session.post(f"{url}/generate", json=job)
    t1 = time.time()
    data = resp.json()
    data["_submit_start"] = t0
    data["_202_received"] = t1
    return data


def wait_for_completion(session: requests.Session, url: str, job_id: str,
                        timeout: float = 300.0, poll_interval: float = 0.5) -> dict:
    """Poll status until job completes or fails."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        resp = session.get(f"{url}/status/{job_id}")
        if resp.status_code == 200:
            data = resp.json()
            if data.get("status") in ("completed", "failed"):
                data["_completed_at"] = time.time()
                return data
        time.sleep(poll_interval)
    return {"status": "timeout", "job_id": job_id, "_completed_at": time.time()}


def download_artifact(session: requests.Session, url: str, job_id: str) -> dict:
    """Download the artifact and measure TTFB."""
    t0 = time.time()
    resp = session.get(f"{url}/artifact/{job_id}", stream=True)
    # Read first byte
    content = b""
    for chunk in resp.iter_content(chunk_size=1024):
        if chunk:
            t1 = time.time()
            content += chunk
            first_byte_time = t1
            break
    # Drain the rest
    for chunk in resp.iter_content(chunk_size=8192):
        content += chunk
    t2 = time.time()
    return {
        "_artifact_request_start": t0,
        "_first_byte_at": first_byte_time if content else None,
        "_download_completed": t2,
        "_artifact_size": len(content),
        "_artifact_hash": hashlib.sha256(content).hexdigest() if content else None,
    }


def run_load_test(url: str, num_jobs: int, concurrency: int, job_length: int = 5):
    """Run the full load test."""
    print(f"Load Test: {num_jobs} jobs, concurrency={concurrency}, length={job_length}s")
    print(f"Target: {url}")
    print()

    # Health check
    resp = requests.get(f"{url}/health")
    if resp.status_code != 200:
        print(f"ERROR: Health check failed: {resp.status_code}")
        sys.exit(1)
    health = resp.json()
    print(f"Server: queue={health.get('queue')}, backend={health.get('backend')}, depth={health.get('queue_depth')}")
    print()

    # Submit jobs
    print(f"Submitting {num_jobs} jobs...")
    submit_results = []
    t_start = time.time()

    with requests.Session() as session:
        with concurrent_futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = []
            for i in range(num_jobs):
                job = make_job(seed=i + 1, length=job_length)
                future = executor.submit(submit_job, session, url, job)
                futures.append((i, future))

            for i, future in futures:
                try:
                    result = future.result(timeout=30)
                    submit_results.append(result)
                except Exception as e:
                    submit_results.append({"error": str(e), "job_id": None})

    t_submit_done = time.time()
    print(f"Submission complete in {t_submit_done - t_start:.2f}s")

    # Filter successful submissions
    successful = [r for r in submit_results if "job_id" in r and r["job_id"]]
    failed_submits = [r for r in submit_results if "error" in r]
    print(f"Successful submissions: {len(successful)}/{num_jobs}")
    if failed_submits:
        print(f"Failed submissions: {len(failed_submits)}")
        for fs in failed_submits[:3]:
            print(f"  {fs}")

    # Wait for all to complete
    print(f"\nWaiting for {len(successful)} jobs to complete...")
    completion_results = []
    t_wait_start = time.time()

    with requests.Session() as session:
        with concurrent_futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = {}
            for r in successful:
                future = executor.submit(wait_for_completion, session, url, r["job_id"])
                futures[r["job_id"]] = future

            for job_id, future in futures.items():
                try:
                    result = future.result(timeout=600)
                    completion_results.append(result)
                except Exception as e:
                    completion_results.append({"status": "error", "error": str(e), "job_id": job_id})

    t_wait_done = time.time()
    print(f"All jobs finished in {t_wait_done - t_wait_start:.2f}s")

    # Download artifacts for completed jobs
    completed = [r for r in completion_results if r.get("status") == "completed"]
    print(f"\nDownloading {len(completed)} artifacts...")
    artifact_results = {}

    with requests.Session() as session:
        with concurrent_futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = {}
            for r in completed:
                future = executor.submit(download_artifact, session, url, r["job_id"])
                futures[r["job_id"]] = future

            for job_id, future in futures.items():
                try:
                    result = future.result(timeout=60)
                    artifact_results[job_id] = result
                except Exception as e:
                    artifact_results[job_id] = {"error": str(e)}

    # Compute metrics
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    # Submission latency
    submit_latencies = [r["_202_received"] - r["_submit_start"] for r in successful]
    if submit_latencies:
        print(f"\nSubmission Latency (POST → 202):")
        print(f"  min:    {min(submit_latencies)*1000:.1f} ms")
        print(f"  max:    {max(submit_latencies)*1000:.1f} ms")
        print(f"  mean:   {statistics.mean(submit_latencies)*1000:.1f} ms")
        print(f"  median: {statistics.median(submit_latencies)*1000:.1f} ms")
        if len(submit_latencies) > 1:
            print(f"  stddev: {statistics.stdev(submit_latencies)*1000:.1f} ms")

    # Completion status
    status_counts = {}
    for r in completion_results:
        s = r.get("status", "unknown")
        status_counts[s] = status_counts.get(s, 0) + 1
    print(f"\nCompletion Status:")
    for status, count in sorted(status_counts.items()):
        print(f"  {status}: {count}")

    # Queue wait + render time (from job timestamps)
    queue_waits = []
    render_times = []
    total_times = []
    for r in completion_results:
        if r.get("status") == "completed":
            queued_at = r.get("queued_at")
            processing_at = r.get("processing_at")
            completed_at = r.get("completed_at")
            if queued_at and processing_at and completed_at:
                queue_waits.append(processing_at - queued_at)
                render_times.append(completed_at - processing_at)
                total_times.append(completed_at - queued_at)

    if queue_waits:
        print(f"\nQueue Wait (queued → processing):")
        print(f"  min:    {min(queue_waits)*1000:.1f} ms")
        print(f"  max:    {max(queue_waits)*1000:.1f} ms")
        print(f"  mean:   {statistics.mean(queue_waits)*1000:.1f} ms")
        print(f"  median: {statistics.median(queue_waits)*1000:.1f} ms")

    if render_times:
        print(f"\nRender Time (processing → completed):")
        print(f"  min:    {min(render_times)*1000:.1f} ms")
        print(f"  max:    {max(render_times)*1000:.1f} ms")
        print(f"  mean:   {statistics.mean(render_times)*1000:.1f} ms")
        print(f"  median: {statistics.median(render_times)*1000:.1f} ms")

    if total_times:
        print(f"\nTotal Job Time (queued → completed):")
        print(f"  min:    {min(total_times)*1000:.1f} ms")
        print(f"  max:    {max(total_times)*1000:.1f} ms")
        print(f"  mean:   {statistics.mean(total_times)*1000:.1f} ms")
        print(f"  median: {statistics.median(total_times)*1000:.1f} ms")

    # Artifact download
    ttfb_times = []
    download_times = []
    artifact_sizes = []
    for job_id, ar in artifact_results.items():
        if ar.get("_first_byte_at"):
            ttfb_times.append(ar["_first_byte_at"] - ar["_artifact_request_start"])
            download_times.append(ar["_download_completed"] - ar["_artifact_request_start"])
            artifact_sizes.append(ar.get("_artifact_size", 0))

    if ttfb_times:
        print(f"\nArtifact TTFB (GET → first byte):")
        print(f"  min:    {min(ttfb_times)*1000:.1f} ms")
        print(f"  max:    {max(ttfb_times)*1000:.1f} ms")
        print(f"  mean:   {statistics.mean(ttfb_times)*1000:.1f} ms")
        print(f"  median: {statistics.median(ttfb_times)*1000:.1f} ms")

    if download_times:
        print(f"\nArtifact Download Time:")
        print(f"  min:    {min(download_times)*1000:.1f} ms")
        print(f"  max:    {max(download_times)*1000:.1f} ms")
        print(f"  mean:   {statistics.mean(download_times)*1000:.1f} ms")

    if artifact_sizes:
        print(f"\nArtifact Sizes:")
        print(f"  min:    {min(artifact_sizes):,} bytes")
        print(f"  max:    {max(artifact_sizes):,} bytes")
        print(f"  mean:   {statistics.mean(artifact_sizes):,.0f} bytes")

    # True end-to-end (submit → first byte)
    true_ttfb = []
    for r in successful:
        job_id = r["job_id"]
        ar = artifact_results.get(job_id, {})
        if ar.get("_first_byte_at"):
            true_ttfb.append(ar["_first_byte_at"] - r["_submit_start"])
    if true_ttfb:
        print(f"\nTrue End-to-End (submit → first WAV byte):")
        print(f"  min:    {min(true_ttfb)*1000:.1f} ms")
        print(f"  max:    {max(true_ttfb)*1000:.1f} ms")
        print(f"  mean:   {statistics.mean(true_ttfb)*1000:.1f} ms")
        print(f"  median: {statistics.median(true_ttfb)*1000:.1f} ms")

    # Throughput
    total_wall = time.time() - t_start
    completed_count = status_counts.get("completed", 0)
    print(f"\nThroughput:")
    print(f"  Total wall time: {total_wall:.2f}s")
    print(f"  Jobs/s: {completed_count / total_wall:.2f}")

    # Exact-once check
    print(f"\nExact-Once Check:")
    print(f"  Submitted: {len(successful)}")
    print(f"  Completed: {completed_count}")
    if completed_count == len(successful):
        print(f"  PASS: All submitted jobs completed exactly once")
    else:
        print(f"  FAIL: {len(successful) - completed_count} jobs lost or failed")

    # Final queue depth
    resp = requests.get(f"{url}/queue/status")
    if resp.status_code == 200:
        qs = resp.json()
        print(f"\nFinal Queue State:")
        print(f"  depth: {qs.get('depth')}")
        print(f"  pending: {qs.get('pending')}")
        print(f"  processing: {qs.get('processing')}")
        print(f"  completed: {qs.get('completed')}")
        print(f"  failed: {qs.get('failed')}")


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
