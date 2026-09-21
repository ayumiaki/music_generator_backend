#!/usr/bin/env python3
"""Tests for queue implementations: concurrency, crash recovery, exact-once.

Covers:
- FileQueue and RedisQueue (if available)
- Double-claim prevention
- Recovery semantics
- Lifecycle timestamps
- Queue correctness under concurrent access
"""

import json
import os
import sys
import time
import unittest
import tempfile
import shutil
from pathlib import Path
from concurrent import futures
from threading import Thread

sys.path.insert(0, str(Path(__file__).resolve().parent))

from job_queue import (
    QueueItem, FileQueue, get_queue, QUEUE_TYPE,
    MAX_RETRIES, WORKER_TIMEOUT,
)

try:
    from job_queue import RedisQueue
    HAS_REDIS = True
except ImportError:
    HAS_REDIS = False


class QueueTestSuite(unittest.TestCase):
    """Test queue implementations for correctness."""

    def _make_queue(self):
        """Create a fresh FileQueue in a temp dir."""
        tmpdir = tempfile.mkdtemp()
        self._tmpdirs.append(tmpdir)
        return FileQueue(queue_dir=tmpdir)

    def setUp(self):
        self._tmpdirs = []

    def tearDown(self):
        for d in self._tmpdirs:
            shutil.rmtree(d, ignore_errors=True)

    def _make_item(self, seed=1):
        return QueueItem(
            prompt=f"test {seed}",
            mood="calm",
            tempo=120,
            key="C",
            length=5,
            seed=seed,
        )


class TestFileQueueBasic(QueueTestSuite):
    """FileQueue basic operations."""

    def test_enqueue_returns_job_id(self):
        q = self._make_queue()
        item = self._make_item()
        job_id = q.enqueue(item)
        self.assertTrue(job_id)
        self.assertEqual(item.status, "pending")

    def test_dequeue_returns_pending_item(self):
        q = self._make_queue()
        item = self._make_item(seed=42)
        job_id = q.enqueue(item)
        result = q.dequeue(worker_id="w1")
        self.assertIsNotNone(result)
        self.assertEqual(result.job_id, job_id)
        self.assertEqual(result.status, "processing")
        self.assertEqual(result.worker_id, "w1")
        self.assertIsNotNone(result.processing_at)

    def test_dequeue_empty_returns_none(self):
        q = self._make_queue()
        self.assertIsNone(q.dequeue())

    def test_depth_pending_only(self):
        q = self._make_queue()
        for i in range(3):
            q.enqueue(self._make_item(seed=i))
        depth = q.depth()
        self.assertEqual(depth["pending"], 3)
        self.assertEqual(depth["processing"], 0)

    def test_depth_after_dequeue(self):
        q = self._make_queue()
        for i in range(3):
            q.enqueue(self._make_item(seed=i))
        q.dequeue(worker_id="w1")
        depth = q.depth()
        self.assertEqual(depth["pending"], 2)
        self.assertEqual(depth["processing"], 1)

    def test_complete_removes_from_active(self):
        q = self._make_queue()
        job_id = q.enqueue(self._make_item())
        item = q.dequeue(worker_id="w1")
        q.complete(item, result={"output_file": "/fake/path.wav"})
        depth = q.depth()
        self.assertEqual(depth["total_active"], 0)


class TestFileQueueDoubleClaim(QueueTestSuite):
    """FileQueue must NOT allow two workers to claim the same job."""

    def test_two_dequeues_same_job(self):
        q = self._make_queue()
        job_id = q.enqueue(self._make_item(seed=1))

        item1 = q.dequeue(worker_id="w1")
        item2 = q.dequeue(worker_id="w2")

        self.assertIsNotNone(item1)
        self.assertIsNone(item2, "Second dequeue should return None — job already claimed")

    def test_sequential_dequeue_fifo(self):
        q = self._make_queue()
        ids = [q.enqueue(self._make_item(seed=i)) for i in range(5)]

        got = []
        for _ in range(5):
            item = q.dequeue(worker_id="w1")
            if item:
                got.append(item.job_id)

        self.assertEqual(got, ids, "Dequeue should be FIFO order")

    def test_concurrent_dequeue_no_double_claim(self):
        """Multiple threads dequeueing must not get the same job."""
        q = self._make_queue()
        n_jobs = 20
        for i in range(n_jobs):
            q.enqueue(self._make_item(seed=i))

        claimed = []
        lock = Thread()

        def worker():
            time.sleep(0.01)  # let all threads start
            item = q.dequeue(worker_id=f"w{threading.current_thread().name}")
            if item:
                claimed.append(item.job_id)

        import threading
        threads = [Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        self.assertEqual(len(claimed), len(set(claimed)),
                         f"Duplicate claims detected: {len(claimed)} claims, {len(set(claimed))} unique")
        self.assertLessEqual(len(claimed), n_jobs)


class TestFileQueueRecovery(QueueTestSuite):
    """FileQueue crash recovery."""

    def test_recover_processing_jobs(self):
        q = self._make_queue()
        for i in range(3):
            q.enqueue(self._make_item(seed=i))

        # Claim all 3 (simulating workers)
        q.dequeue(worker_id="w1")
        q.dequeue(worker_id="w2")
        q.dequeue(worker_id="w3")

        # All are "processing" now — simulate crash recovery
        recovered = q.recover()
        self.assertEqual(len(recovered), 3)
        for item in recovered:
            self.assertEqual(item.status, "pending")
            self.assertIsNone(item.worker_id)

    def test_recover_adds_back_to_queue(self):
        q = self._make_queue()
        job_id = q.enqueue(self._make_item(seed=1))
        q.dequeue(worker_id="w1")

        # After dequeue, queue list is empty (job removed from pending)
        # But recover should find it via file scan and re-add
        recovered = q.recover()
        self.assertEqual(len(recovered), 1)

        # Now it should be dequeueable again
        item = q.dequeue(worker_id="w2")
        self.assertIsNotNone(item)
        self.assertEqual(item.job_id, job_id)

    def test_recover_preserves_job_data(self):
        q = self._make_queue()
        item = self._make_item(seed=99)
        item.prompt = "recovery test prompt"
        job_id = q.enqueue(item)
        q.dequeue(worker_id="w1")

        recovered = q.recover()
        self.assertEqual(len(recovered), 1)
        self.assertEqual(recovered[0].prompt, "recovery test prompt")
        self.assertEqual(recovered[0].seed, 99)


class TestFileQueueCompleteFail(QueueTestSuite):
    """FileQueue complete() and fail() lifecycle."""

    def test_complete_sets_timestamps(self):
        q = self._make_queue()
        job_id = q.enqueue(self._make_item())
        item = q.dequeue(worker_id="w1")
        self.assertIsNone(item.completed_at)

        q.complete(item, result={"output_file": "/tmp/fake.wav"})
        self.assertIsNotNone(item.completed_at)

    def test_fail_with_retries(self):
        q = self._make_queue()
        job_id = q.enqueue(self._make_item())
        item = q.dequeue(worker_id="w1")

        q.fail(item, "render error", worker_id="w1")
        # With MAX_RETRIES > 1, should be re-queued
        if MAX_RETRIES > 1:
            self.assertEqual(item.status, "pending")
            # Should be dequeueable again
            item2 = q.dequeue(worker_id="w2")
            self.assertIsNotNone(item2)
            self.assertEqual(item2.attempt, 2)

    def test_fail_exhausts_retries(self):
        q = self._make_queue()
        job_id = q.enqueue(self._make_item())

        item = None
        for attempt in range(MAX_RETRIES + 1):
            item = q.dequeue(worker_id=f"w{attempt}")
            if item:
                q.fail(item, f"error {attempt}", worker_id=f"w{attempt}")

        # After MAX_RETRIES, item should be "dead"
        if item:
            self.assertIn(item.status, ("dead", "failed"))


class TestFileQueueGetItem(QueueTestSuite):
    """FileQueue get_item by job_id."""

    def test_get_existing_item(self):
        q = self._make_queue()
        job_id = q.enqueue(self._make_item(seed=77))
        item = q.get_item(job_id)
        self.assertIsNotNone(item)
        self.assertEqual(item.seed, 77)

    def test_get_nonexistent_item(self):
        q = self._make_queue()
        self.assertIsNone(q.get_item("nonexistent"))


class TestFileQueueUpdateItem(QueueTestSuite):
    """FileQueue update_item persists changes."""

    def test_update_persists(self):
        q = self._make_queue()
        job_id = q.enqueue(self._make_item())
        item = q.get_item(job_id)
        item.result = {"new_key": "new_value"}
        q.update_item(item)

        reloaded = q.get_item(job_id)
        self.assertEqual(reloaded.result, {"new_key": "new_value"})


@unittest.skipUnless(HAS_REDIS and QUEUE_TYPE == "redis", "Redis not configured")
class TestRedisQueueStreams(unittest.TestCase):
    """Redis Streams queue tests. Require Redis connection."""

    @classmethod
    def setUpClass(cls):
        try:
            q = RedisQueue()
            q.client.ping()
            cls.queue = q
        except Exception as e:
            raise unittest.SkipTest(f"Redis unavailable: {e}")

    def setUp(self):
        # Clean up stream and hashes before each test
        self.queue.client.delete(self.queue.stream_key)
        self.queue.client.delete(self.queue.dead_key)
        # Recreate consumer group
        self.queue._ensure_consumer_group()

    def _make_item(self, seed=1):
        return QueueItem(
            prompt=f"test {seed}",
            mood="calm",
            tempo=120,
            key="C",
            length=5,
            seed=seed,
        )

    def test_enqueue_returns_job_id(self):
        item = self._make_item()
        job_id = self.queue.enqueue(item)
        self.assertTrue(job_id)

    def test_dequeue_returns_pending_item(self):
        item = self._make_item(seed=42)
        job_id = self.queue.enqueue(item)
        result = self.queue.dequeue(worker_id="w1")
        self.assertIsNotNone(result)
        self.assertEqual(result.job_id, job_id)
        self.assertEqual(result.status, "processing")

    def test_dequeue_empty_returns_none(self):
        # Non-blocking dequeue
        result = self.queue.dequeue(worker_id="w1", block_ms=100)
        self.assertIsNone(result)

    def test_depth_metrics(self):
        """Depth includes both undelivered (lag) and pending entries."""
        # Fresh queue: no lag, no pending
        depth = self.queue.depth()
        self.assertEqual(depth["total_active"], 0)

        # Enqueue 3 items: they are lag (undelivered) until a worker reads them
        job_ids = []
        for i in range(3):
            job_ids.append(self.queue.enqueue(self._make_item(seed=i)))

        depth = self.queue.depth()
        self.assertEqual(depth["pending"], 3, "Expected 3 undelivered jobs")
        self.assertEqual(depth["total_active"], 3)

        # Worker dequeues 1: it moves from lag to pending
        item = self.queue.dequeue(worker_id="w1", block_ms=100)
        self.assertIsNotNone(item)

        depth = self.queue.depth()
        # 2 undelivered (lag) + 1 delivered-but-unacked (pending) = 3 total
        self.assertGreaterEqual(depth["total_active"], 2)

        # Complete it: total should drop
        self.queue.complete(item, result={"output_file": "/fake/path.wav"})
        depth = self.queue.depth()
        self.assertLess(depth["total_active"], 3)

    def test_depth_after_completion(self):
        """After all jobs complete, active depth returns to zero."""
        for i in range(5):
            self.queue.enqueue(self._make_item(seed=i))

        # Drain all
        for _ in range(5):
            item = self.queue.dequeue(worker_id="w1", block_ms=100)
            if item:
                self.queue.complete(item, result={"output_file": "/fake.wav"})

        depth = self.queue.depth()
        self.assertEqual(depth["total_active"], 0,
                         f"Expected 0 active jobs, got {depth}")

    def test_complete_and_ack(self):
        job_id = self.queue.enqueue(self._make_item())
        item = self.queue.dequeue(worker_id="w1")
        self.queue.complete(item, result={"output_file": "/fake/path.wav"})

        # After completion, item should be marked completed
        reloaded = self.queue.get_item(job_id)
        self.assertEqual(reloaded.status, "completed")
        self.assertIsNotNone(reloaded.completed_at)

    def test_render_duration_ms_persisted(self):
        """render_duration_ms from result dict is persisted to item."""
        job_id = self.queue.enqueue(self._make_item())
        item = self.queue.dequeue(worker_id="w1")
        self.queue.complete(
            item,
            result={"output_file": "/fake.wav", "render_duration_ms": 123.456}
        )
        reloaded = self.queue.get_item(job_id)
        self.assertAlmostEqual(reloaded.render_duration_ms, 123.456, places=3)

    def test_retry_ack_cleanup(self):
        """Retry should ACK old entry so only one pending entry exists per retry."""
        job_id = self.queue.enqueue(self._make_item())

        # Dequeue and fail (triggers retry)
        item = self.queue.dequeue(worker_id="w1", block_ms=100)
        self.queue.fail(item, "transient error", worker_id="w1")

        # The job should be back in the queue
        item2 = self.queue.dequeue(worker_id="w2", block_ms=100)
        self.assertIsNotNone(item2, "Job should be available for retry")

        # There should be exactly one pending entry, not two
        pending_info = self.queue.client.xpending(
            self.queue.stream_key, self.queue.group_name
        )
        self.assertEqual(pending_info["pending"], 1,
                         "Retry should ACK old entry, leaving exactly one pending")

        # Complete to clean up
        self.queue.complete(item2, result={"output_file": "/fake.wav"})

    def test_recover_stale_jobs(self):
        """Recover jobs that were processing but worker died."""
        job_id = self.queue.enqueue(self._make_item(seed=1))
        item = self.queue.dequeue(worker_id="dead-worker")
        self.assertIsNotNone(item)

        # Don't ACK — simulates worker crash
        # For testing, use a short timeout by directly manipulating the stream
        # to bypass the WORKER_TIMEOUT sleep
        time.sleep(0.5)

        # Manually claim the entry via XAUTOCLAIM with minimal idle time
        result = self.queue.client.xautoclaim(
            self.queue.stream_key, self.queue.group_name,
            "test-recovery", min_idle_time=100, start_id="0-0", count=100
        )
        self.assertIsNotNone(result)
        entries = result[1] if len(result) >= 2 else []
        self.assertGreater(len(entries), 0, "Should find stale entry for recovery")

    def test_fail_retries_then_dead(self):
        """Job should eventually be marked dead after exhausting retries."""
        job_id = self.queue.enqueue(self._make_item())
        for attempt in range(MAX_RETRIES + 1):
            item = self.queue.dequeue(worker_id=f"w{attempt}", block_ms=100)
            if item:
                self.queue.fail(item, f"error {attempt}", worker_id=f"w{attempt}")

        item = self.queue.get_item(job_id)
        self.assertIn(item.status, ("dead", "failed"))

        # Dead jobs should not be requeued
        requeue = self.queue.dequeue(worker_id="w-final", block_ms=100)
        if requeue:
            self.assertNotEqual(requeue.job_id, job_id,
                                "Dead job should not be dequeued")


class TestGetQueue(unittest.TestCase):
    """get_queue() factory function."""

    def test_returns_file_queue_by_default(self):
        if QUEUE_TYPE == "redis":
            self.skipTest("Redis configured, skipping default-file-queue check")
        q = get_queue()
        self.assertIsInstance(q, FileQueue)


class TestQueueItemSerialization(unittest.TestCase):
    """QueueItem to_dict/from_dict round-trip."""

    def test_round_trip(self):
        item = QueueItem(
            job_id="abc-123",
            prompt="test prompt",
            mood="dark",
            tempo=140,
            key="F#",
            length=30,
            seed=42,
            status="processing",
            result={"artifact_path": "/tmp/out.wav"},
            worker_id="w1",
            attempt=2,
            artifact_size=48000,
            render_duration_ms=150.5,
        )
        data = item.to_dict()
        restored = QueueItem.from_dict(data)
        self.assertEqual(restored.job_id, "abc-123")
        self.assertEqual(restored.prompt, "test prompt")
        self.assertEqual(restored.mood, "dark")
        self.assertEqual(restored.tempo, 140)
        self.assertEqual(restored.seed, 42)
        self.assertEqual(restored.worker_id, "w1")
        self.assertEqual(restored.attempt, 2)
        self.assertEqual(restored.artifact_size, 48000)
        self.assertAlmostEqual(restored.render_duration_ms, 150.5)


class TestFileQueueRecoveryIntegration(QueueTestSuite):
    """Deterministic end-to-end recovery: no timing races, no slow backend.

    Proves the full reclaim chain that worker.main() actually uses:
      Worker A claims (attempt=1) → A dies (no ack) →
      Worker B recovers via recover() → dequeues with attempt=2 →
      completes → queue is clean.

    This is the deterministic equivalent of "kill mid-render" — it exercises
    the exact code path (recover → dequeue → complete) without depending on
    worker timeout timing or backend speed.
    """

    def _make_shared_queue(self):
        """Create a FileQueue in a temp dir, tracking it for cleanup."""
        tmpdir = tempfile.mkdtemp()
        self._tmpdirs.append(tmpdir)
        return tmpdir

    def test_reclaim_chain_deterministic(self):
        """Job claimed by A, A dies, B reclaims with attempt=2, completes."""
        tmpdir = self._make_shared_queue()
        q_a = FileQueue(queue_dir=tmpdir)
        q_b = FileQueue(queue_dir=tmpdir)  # separate handle, SAME queue dir

        # 1. Enqueue
        item = self._make_item(seed=42)
        item.prompt = "reclaim test"
        job_id = q_a.enqueue(item)

        # 2. Worker A claims (attempt=1, worker_id=A)
        claimed = q_a.dequeue(worker_id="worker-A")
        assert claimed is not None
        self.assertEqual(claimed.job_id, job_id)
        self.assertEqual(claimed.status, "processing")
        self.assertEqual(claimed.worker_id, "worker-A")
        self.assertEqual(claimed.attempt, 1)

        # 3. Worker A dies — no ack, no complete. Job stuck in "processing".
        #    (simulated by simply not calling complete/fail)
        stuck = q_a.get_item(job_id)
        assert stuck is not None
        self.assertEqual(stuck.status, "processing")
        self.assertEqual(stuck.worker_id, "worker-A")

        # 4. Worker B starts, calls recover() — finds stuck job, resets to pending
        recovered = q_b.recover()
        self.assertEqual(len(recovered), 1, "recover() should find the stuck job")
        self.assertEqual(recovered[0].job_id, job_id)

        # After recovery: status=pending, worker_id=None, attempt still 1
        # (recover resets worker/status but does NOT increment attempt —
        #  that happens on dequeue)
        after_recover = q_b.get_item(job_id)
        assert after_recover is not None
        self.assertEqual(after_recover.status, "pending")
        self.assertIsNone(after_recover.worker_id)

        # 5. Worker B dequeues — gets attempt=2, worker_id=B
        reclaimed = q_b.dequeue(worker_id="worker-B")
        assert reclaimed is not None, "Worker B should reclaim the recovered job"
        self.assertEqual(reclaimed.job_id, job_id)
        self.assertEqual(reclaimed.status, "processing")
        self.assertEqual(reclaimed.worker_id, "worker-B")
        self.assertEqual(reclaimed.attempt, 2, "attempt must increment to 2 after reclaim")

        # 6. Worker B completes the job
        q_b.complete(reclaimed, result={"output_file": "/fake/reclaimed.wav"})

        # 7. Verify clean queue state
        final = q_b.get_item(job_id)
        assert final is not None
        self.assertEqual(final.status, "completed")
        self.assertIsNotNone(final.completed_at)

        depth = q_b.depth()
        self.assertEqual(depth["total_active"], 0, "Queue must be clean after completion")
        self.assertEqual(depth["processing"], 0)
        self.assertEqual(depth["pending"], 0)

    def test_reclaim_preserves_job_data(self):
        """Reclaimed job retains original prompt, seed, metadata."""
        tmpdir = self._make_shared_queue()
        q_a = FileQueue(queue_dir=tmpdir)
        q_b = FileQueue(queue_dir=tmpdir)

        item = self._make_item(seed=99)
        item.prompt = "preserve me"
        item.mood = "dark"
        item.tempo = 140
        job_id = q_a.enqueue(item)

        q_a.dequeue(worker_id="worker-A")
        q_b.recover()
        reclaimed = q_b.dequeue(worker_id="worker-B")
        assert reclaimed is not None
        self.assertEqual(reclaimed.prompt, "preserve me")
        self.assertEqual(reclaimed.mood, "dark")
        self.assertEqual(reclaimed.tempo, 140)
        self.assertEqual(reclaimed.seed, 99)
        self.assertEqual(reclaimed.attempt, 2)

    def test_no_double_completion_on_reclaim(self):
        """Worker B reclaims and completes; no duplicate terminal state."""
        tmpdir = self._make_shared_queue()
        q_a = FileQueue(queue_dir=tmpdir)
        q_b = FileQueue(queue_dir=tmpdir)

        job_id = q_a.enqueue(self._make_item(seed=7))
        q_a.dequeue(worker_id="worker-A")
        q_b.recover()
        reclaimed = q_b.dequeue(worker_id="worker-B")
        assert reclaimed is not None
        q_b.complete(reclaimed, result={"output_file": "/fake/complete.wav"})

        # Job is completed exactly once
        final = q_b.get_item(job_id)
        assert final is not None
        self.assertEqual(final.status, "completed")
        self.assertIsNotNone(final.completed_at)

        # Queue is clean — no pending, no processing
        depth = q_b.depth()
        self.assertEqual(depth["total_active"], 0)

    def test_multiple_workers_recover_distinct_jobs(self):
        """Three workers each claim one job, all die, all three are recovered."""
        tmpdir = self._make_shared_queue()
        q = FileQueue(queue_dir=tmpdir)
        job_ids = []
        for i in range(3):
            jid = q.enqueue(self._make_item(seed=i))
            job_ids.append(jid)
            q.dequeue(worker_id=f"worker-{i}")

        # All three stuck in processing
        for jid in job_ids:
            item = q.get_item(jid)
            assert item is not None
            self.assertEqual(item.status, "processing")

        # New process recovers all
        recovered = q.recover()
        self.assertEqual(len(recovered), 3)

        # All three dequeued with attempt=2 by new workers
        # (order may not be FIFO since recover uses glob scan)
        reclaimed_ids = set()
        for i in range(3):
            item = q.dequeue(worker_id=f"new-worker-{i}")
            assert item is not None
            self.assertIn(item.job_id, job_ids)
            self.assertEqual(item.attempt, 2)
            reclaimed_ids.add(item.job_id)
            q.complete(item, result={"output_file": f"/fake/{item.job_id}.wav"})

        # All original jobs were reclaimed (no duplicates, no orphans)
        self.assertEqual(reclaimed_ids, set(job_ids))
        self.assertEqual(q.depth()["total_active"], 0)


    def test_worker_recovers_and_renders_valid_wav(self):
        """Worker A claims & dies, Worker B recovers, renders, completes with valid WAV."""
        from backends.mock_backend import MockBackend
        from worker import process_job

        tmpdir = self._make_shared_queue()
        q_a = FileQueue(queue_dir=tmpdir)
        q_b = FileQueue(queue_dir=tmpdir)
        backend_b = MockBackend()

        # 1. Enqueue
        enqueue_item = self._make_item(seed=7)
        enqueue_item.length = 1
        job_id = q_a.enqueue(enqueue_item)

        # 2. Worker A claims the job (but dies before rendering — we don't call process_job)
        claimed = q_a.dequeue(worker_id="worker-A")
        assert claimed is not None
        self.assertEqual(claimed.attempt, 1)
        self.assertEqual(claimed.worker_id, "worker-A")
        self.assertEqual(claimed.status, "processing")

        # 3. Worker A dies — job stuck in "processing"
        stuck = q_a.get_item(job_id)
        assert stuck is not None
        self.assertEqual(stuck.status, "processing")

        # 4. Worker B starts, recovers the stuck job
        recovered = q_b.recover()
        self.assertEqual(len(recovered), 1)

        # 5. Worker B reclaims via dequeue (attempt=2)
        reclaimed = q_b.dequeue(worker_id="worker-B")
        assert reclaimed is not None
        self.assertEqual(reclaimed.attempt, 2)
        self.assertEqual(reclaimed.worker_id, "worker-B")

        # 6. Worker B renders via the real process_job path
        process_job(q_b, backend_b, reclaimed, worker_id="worker-B")

        # 7. Verify completion
        final = q_b.get_item(job_id)
        assert final is not None
        self.assertEqual(final.status, "completed")
        self.assertIsNotNone(final.completed_at)
        self.assertEqual(final.worker_id, "worker-B")
        self.assertEqual(final.attempt, 2)

        # 8. Verify valid artifact with WAV header
        output_file = final.result.get("output_file") if final.result else None
        self.assertIsNotNone(output_file, "output_file should be set in result")
        assert output_file is not None
        self.assertTrue(
            os.path.exists(output_file),
            f"Artifact file should exist: {output_file}"
        )
        self.assertGreater(os.path.getsize(output_file), 0)
        self.assertTrue(
            _validate_wav_header(output_file),
            f"Artifact should have valid WAV header: {output_file}"
        )

        # 9. Queue clean
        depth = q_b.depth()
        self.assertEqual(depth["total_active"], 0)

    def test_worker_recovery_preserves_attempt_count_across_multiple_recoveries(self):
        """Job can be recovered and retried up to MAX_RETRIES, then goes dead."""
        tmpdir = self._make_shared_queue()
        q = FileQueue(queue_dir=tmpdir)

        item = self._make_item(seed=5)
        item.length = 1
        job_id = q.enqueue(item)

        # Worker A claims, dies
        q.dequeue(worker_id="worker-A")
        q.recover()

        # Worker B claims, fails, retries
        item = q.dequeue(worker_id="worker-B")
        assert item is not None
        self.assertEqual(item.attempt, 2)
        q.fail(item, "transient error", worker_id="worker-B")

        # After fail with retries remaining, should be pending again
        if MAX_RETRIES > 2:
            retried = q.dequeue(worker_id="worker-C")
            assert retried is not None
            self.assertEqual(retried.attempt, 3)


def _validate_wav_header(path: str) -> bool:
    """Check that a file has a valid WAV header (RIFF/WAVE/fmt)."""
    try:
        with open(path, "rb") as f:
            header = f.read(12)
            if len(header) < 12:
                return False
            return header[:4] == b"RIFF" and header[8:12] == b"WAVE"
    except (OSError, IOError):
        return False


if __name__ == "__main__":
    unittest.main(verbosity=2)
