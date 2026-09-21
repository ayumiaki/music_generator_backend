"""
Job queue implementation for music generator.
Supports file-based and Redis backends with proper concurrency control.
"""

import json
import os
import time
import uuid
import fcntl
from pathlib import Path
from typing import Dict, List, Optional, Any
from abc import ABC, abstractmethod

# Import config
try:
    from config import (
        QUEUE_DIR, QUEUE_TYPE, OUTPUT_DIR,
        REDIS_HOST, REDIS_PORT, REDIS_DB,
        REDIS_QUEUE_KEY, REDIS_CONSUMER_GROUP, REDIS_DEAD_LETTER_KEY,
        MAX_RETRIES, WORKER_TIMEOUT, CLAIM_BATCH_SIZE,
    )
except ImportError:
    # Fallback defaults (should not happen if config.py exists)
    BASE_DIR = Path(__file__).resolve().parent.parent
    QUEUE_DIR = BASE_DIR / "queue"
    OUTPUT_DIR = BASE_DIR / "output"
    QUEUE_TYPE = "file"
    REDIS_HOST = "localhost"
    REDIS_PORT = 6379
    REDIS_DB = 0
    REDIS_QUEUE_KEY = "music_gen:jobs"
    REDIS_CONSUMER_GROUP = "renderers"
    REDIS_DEAD_LETTER_KEY = "music_gen:jobs:dead"
    MAX_RETRIES = 3
    WORKER_TIMEOUT = 300
    CLAIM_BATCH_SIZE = 1


class QueueItem:
    """Represents a job in the queue."""

    def __init__(
        self,
        job_id: str = "",
        prompt: str = "",
        mood: str = "calm",
        tempo: int = 120,
        key: str = "C",
        length: int = 30,
        seed: Optional[int] = None,
        status: str = "pending",
        result: Optional[Dict] = None,
        created_at: Optional[float] = None,
        queued_at: Optional[float] = None,
        processing_at: Optional[float] = None,
        completed_at: Optional[float] = None,
        failed_at: Optional[float] = None,
        worker_id: Optional[str] = None,
        attempt: int = 0,
        artifact_size: Optional[int] = None,
        render_duration_ms: Optional[float] = None,
        error: Optional[str] = None,
    ):
        self.job_id = job_id
        self.prompt = prompt
        self.mood = mood
        self.tempo = tempo
        self.key = key
        self.length = length
        self.seed = seed
        self.status = status  # pending, processing, completed, failed, dead
        self.result = result or {}
        self.created_at = created_at or time.time()
        self.updated_at = time.time()
        # Lifecycle timestamps for latency measurement
        self.queued_at = queued_at
        self.processing_at = processing_at
        self.completed_at = completed_at
        self.failed_at = failed_at
        self.worker_id = worker_id
        self.attempt = attempt
        self.artifact_size = artifact_size
        self.render_duration_ms = render_duration_ms
        self.error = error

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "prompt": self.prompt,
            "mood": self.mood,
            "tempo": self.tempo,
            "key": self.key,
            "length": self.length,
            "seed": self.seed,
            "status": self.status,
            "result": self.result,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "queued_at": self.queued_at,
            "processing_at": self.processing_at,
            "completed_at": self.completed_at,
            "failed_at": self.failed_at,
            "worker_id": self.worker_id,
            "attempt": self.attempt,
            "artifact_size": self.artifact_size,
            "render_duration_ms": self.render_duration_ms,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QueueItem":
        item = cls(
            job_id=data.get("job_id", ""),
            prompt=data.get("prompt", ""),
            mood=data.get("mood", "calm"),
            tempo=data.get("tempo", 120),
            key=data.get("key", "C"),
            length=data.get("length", 30),
            seed=data.get("seed"),
            status=data.get("status", "pending"),
            result=data.get("result", {}),
            created_at=data.get("created_at"),
            queued_at=data.get("queued_at"),
            processing_at=data.get("processing_at"),
            completed_at=data.get("completed_at"),
            failed_at=data.get("failed_at"),
            worker_id=data.get("worker_id"),
            attempt=data.get("attempt", 0),
            artifact_size=data.get("artifact_size"),
            render_duration_ms=data.get("render_duration_ms"),
            error=data.get("error"),
        )
        item.updated_at = data.get("updated_at", time.time())
        return item


class BaseQueue(ABC):
    """Abstract base queue."""

    @abstractmethod
    def enqueue(self, item: QueueItem) -> str:
        pass

    @abstractmethod
    def dequeue(self, worker_id: str = None) -> Optional[QueueItem]:
        pass

    @abstractmethod
    def get_item(self, job_id: str) -> Optional[QueueItem]:
        pass

    @abstractmethod
    def update_item(self, item: QueueItem) -> None:
        pass

    @abstractmethod
    def list_items(self, limit: int = 100) -> List[QueueItem]:
        pass

    @abstractmethod
    def recover(self) -> List[QueueItem]:
        """Pick up jobs stuck in processing state after a crash."""
        pass

    @abstractmethod
    def remove_item(self, job_id: str) -> bool:
        pass

    @abstractmethod
    def depth(self) -> Dict[str, int]:
        """Return queue depth: pending + processing counts."""
        pass

    def complete(self, item: QueueItem, result: dict = None, worker_id: str = None) -> None:
        """Mark job as completed with timestamps."""
        item.status = "completed"
        item.completed_at = time.time()
        item.processing_at = item.processing_at  # preserve
        if worker_id:
            item.worker_id = worker_id
        if result is not None:
            item.result = result
        self._persist_completion(item)

    def fail(self, item: QueueItem, error: str, worker_id: str = None) -> None:
        """Mark job as failed with timestamps."""
        item.status = "failed"
        item.completed_at = time.time()
        item.result = {"error": error}
        if worker_id:
            item.worker_id = worker_id
        self._persist_completion(item)

    def _persist_completion(self, item: QueueItem) -> None:
        """Persist completed/failed state and remove from active queue.
        Override this if your complete()/fail() methods need custom persistence."""
        pass


class FileQueue(BaseQueue):
    """File-based queue with proper inter-process locking."""

    def __init__(self, queue_dir=None):
        self.queue_dir = Path(queue_dir) if queue_dir else QUEUE_DIR
        self.queue_dir.mkdir(parents=True, exist_ok=True)
        self.queue_file = self.queue_dir / "queue.json"
        self.lock_file = self.queue_dir / ".queue.lock"
        self._ensure_queue_file()

    def _atomic_write(self, path: Path, content: str):
        """Atomic write: write to temp file then rename."""
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(content)
        tmp.replace(path)

    def _ensure_queue_file(self):
        if not self.queue_file.exists():
            self._atomic_write(self.queue_file, "[]")

    def _locked(self):
        """Return a context-manager lock file handle."""
        self.lock_file.touch(exist_ok=True)
        return open(self.lock_file, "r+")

    def _read_queue_ids(self) -> List[str]:
        try:
            return json.loads(self.queue_file.read_text())
        except (json.JSONDecodeError, FileNotFoundError):
            return []

    def _write_queue_ids(self, ids: List[str]):
        self._atomic_write(self.queue_file, json.dumps(ids, indent=2))

    def _job_file_path(self, job_id: str) -> Path:
        return self.queue_dir / f"{job_id}.json"

    def enqueue(self, item: QueueItem) -> str:
        if not item.job_id:
            item.job_id = str(uuid.uuid4())
        now = time.time()
        item.created_at = now
        item.queued_at = now
        job_file = self._job_file_path(item.job_id)
        self._atomic_write(job_file, json.dumps(item.to_dict(), indent=2))
        with self._locked() as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                ids = self._read_queue_ids()
                if item.job_id not in ids:
                    ids.append(item.job_id)
                    self._write_queue_ids(ids)
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)
        return item.job_id

    def dequeue(self, worker_id: str = None) -> Optional[QueueItem]:
        """Claim the first pending job atomically: remove from pending list + mark processing."""
        with self._locked() as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                ids = self._read_queue_ids()
                while ids:
                    job_id = ids.pop(0)
                    job_file = self._job_file_path(job_id)
                    if job_file.exists():
                        data = json.loads(job_file.read_text())
                        item = QueueItem.from_dict(data)
                        if item.status == "pending":
                            item.status = "processing"
                            item.processing_at = time.time()
                            item.worker_id = worker_id
                            item.attempt += 1
                            self._atomic_write(job_file, json.dumps(item.to_dict(), indent=2))
                            self._write_queue_ids(ids)
                            return item
                        # else: stale entry (completed/dead), skip
                    # else: orphaned ID, skip
                self._write_queue_ids(ids)
                return None
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    def _persist_completion(self, item: QueueItem) -> None:
        job_file = self._job_file_path(item.job_id)
        self._atomic_write(job_file, json.dumps(item.to_dict(), indent=2))

    def complete(self, item: QueueItem, result: dict = None, worker_id: str = None) -> None:
        """Mark job as completed."""
        item.status = "completed"
        item.completed_at = time.time()
        if worker_id:
            item.worker_id = worker_id
        if result is not None:
            item.result = result
        self._persist_completion(item)

    def fail(self, item: QueueItem, error: str, worker_id: str = None) -> None:
        """Mark job as failed. Retry or move to dead status."""
        item.error = error
        item.failed_at = time.time()
        if worker_id:
            item.worker_id = worker_id

        if item.attempt < MAX_RETRIES:
            item.status = "pending"
            item.processing_at = None
            self._atomic_write(self._job_file_path(item.job_id), json.dumps(item.to_dict(), indent=2))
            with self._locked() as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                try:
                    ids = self._read_queue_ids()
                    if item.job_id not in ids:
                        ids.append(item.job_id)
                    self._write_queue_ids(ids)
                finally:
                    fcntl.flock(lock, fcntl.LOCK_UN)
        else:
            item.status = "dead"
            self._persist_completion(item)

    def _remove_from_queue_list(self, job_id: str) -> None:
        ids = self._read_queue_ids()
        if job_id in ids:
            ids.remove(job_id)
            self._write_queue_ids(ids)

    def recover(self) -> List[QueueItem]:
        """Recover jobs stuck in 'processing' state after a crash.

        Scans all job files (not just queue list) since claimed jobs
        are removed from the pending list.
        """
        recovered = []
        with self._locked() as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                ids = self._read_queue_ids()
                for job_file in self.queue_dir.glob("*.json"):
                    if job_file.name in ("queue.json",):
                        continue
                    job_id = job_file.stem
                    try:
                        data = json.loads(job_file.read_text())
                        item = QueueItem.from_dict(data)
                        if item.status == "processing":
                            item.status = "pending"
                            item.processing_at = None
                            item.worker_id = None
                            self._atomic_write(job_file, json.dumps(item.to_dict(), indent=2))
                            if job_id not in ids:
                                ids.append(job_id)
                            recovered.append(item)
                    except (json.JSONDecodeError, KeyError):
                        pass
                self._write_queue_ids(ids)
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)
        return recovered

    def get_item(self, job_id: str) -> Optional[QueueItem]:
        job_file = self._job_file_path(job_id)
        if job_file.exists():
            try:
                data = json.loads(job_file.read_text())
                return QueueItem.from_dict(data)
            except (json.JSONDecodeError, KeyError):
                return None
        return None

    def update_item(self, item: QueueItem) -> None:
        job_file = self._job_file_path(item.job_id)
        item.updated_at = time.time()
        self._atomic_write(job_file, json.dumps(item.to_dict(), indent=2))

    def list_items(self, limit: int = 100) -> List[QueueItem]:
        """List all jobs by scanning job files (not just queue list)."""
        items = []
        for job_file in self.queue_dir.glob("*.json"):
            if job_file.name == "queue.json":
                continue
            try:
                data = json.loads(job_file.read_text())
                item = QueueItem.from_dict(data)
                items.append(item)
            except (json.JSONDecodeError, KeyError):
                pass
            if len(items) >= limit:
                break
        return items

    def remove_item(self, job_id: str) -> bool:
        job_file = self._job_file_path(job_id)
        if job_file.exists():
            job_file.unlink()
            with self._locked() as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                try:
                    ids = self._read_queue_ids()
                    if job_id in ids:
                        ids.remove(job_id)
                        self._write_queue_ids(ids)
                finally:
                    fcntl.flock(lock, fcntl.LOCK_UN)
            return True
        return False

    def depth(self) -> Dict[str, int]:
        """Return queue depth: pending + processing counts."""
        pending = 0
        processing = 0
        for job_file in self.queue_dir.glob("*.json"):
            if job_file.name == "queue.json":
                continue
            try:
                data = json.loads(job_file.read_text())
                status = data.get("status", "")
                if status == "pending":
                    pending += 1
                elif status == "processing":
                    processing += 1
            except (json.JSONDecodeError, KeyError):
                pass
        return {
            "pending": pending,
            "processing": processing,
            "total_active": pending + processing,
        }


# Try to import redis
import sys
try:
    import redis as redis_lib
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    # Create a dummy module for type checking
    class _DummyRedis:
        class exceptions:
            class ResponseError(Exception): pass
            class ConnectionError(Exception): pass
    redis_lib = _DummyRedis()  # type: ignore


class RedisQueue(BaseQueue):
    """Redis Streams-backed queue with consumer groups, XAUTOCLAIM recovery, and dead-letter.

    Uses Redis Streams for the queue itself, a hash per job for metadata,
    and a dead-letter stream for exhausted retries.
    """

    def __init__(self):
        self.client = redis_lib.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB)
        self.stream_key = REDIS_QUEUE_KEY
        self.group_name = REDIS_CONSUMER_GROUP
        self.dead_key = REDIS_DEAD_LETTER_KEY
        self._pending_acks: Dict[str, str] = {}  # job_id -> stream entry_id
        self._ensure_consumer_group()

    def _ensure_consumer_group(self):
        """Create the consumer group if it doesn't exist, starting from the beginning."""
        try:
            self.client.xgroup_create(
                self.stream_key, self.group_name, id="0", mkstream=True
            )
        except redis_lib.exceptions.ResponseError as e:
            if "BUSYGROUP" in str(e):
                pass  # group already exists
            else:
                raise

    def _ensure_group_exists(self):
        """Verify consumer group exists, recreate if missing."""
        try:
            groups = self.client.xinfo_groups(self.stream_key)
            group_names = [g.get("name", b"").decode() if isinstance(g.get("name"), bytes) else g.get("name", "") for g in groups]
            if self.group_name not in group_names:
                self._ensure_consumer_group()
        except redis_lib.exceptions.ResponseError:
            # Stream doesn't exist, create it with the group
            self._ensure_consumer_group()

    def enqueue(self, item: QueueItem) -> str:
        if not item.job_id:
            item.job_id = str(uuid.uuid4())
        now = time.time()
        item.created_at = now
        item.queued_at = now
        item.updated_at = now
        item.status = "pending"
        item.attempt = 0

        # Store job metadata in hash, then add to stream
        self.client.hset(self._hash_key(item.job_id), mapping=self._item_to_hash(item))
        self.client.xadd(self.stream_key, {"job_id": item.job_id})
        return item.job_id

    def dequeue(self, worker_id: str = None, block_ms: int = 5000) -> Optional[QueueItem]:
        """Claim a job from the stream via XREADGROUP.

        First tries to claim any pending (previously delivered but unacked) entries
        via XAUTOCLAIM. If none, reads new entries with '>'.
        """
        worker_id = worker_id or f"worker-{os.getpid()}"

        # Ensure consumer group exists (may have been deleted)
        self._ensure_group_exists()

        # 1. Try to auto-claim stale pending entries (worker death recovery)
        claimed = self._autoclaim(worker_id)
        if claimed:
            return claimed

        # 2. Read new entries
        result = self.client.xreadgroup(
            self.group_name, worker_id,
            {self.stream_key: ">"},
            count=CLAIM_BATCH_SIZE,
            block=block_ms,
        )
        if not result:
            return None

        # result: [(stream_name, [(entry_id, {field: value}), ...])]
        for stream_name, entries in result:
            for entry_id, fields in entries:
                job_id = fields.get(b"job_id") or fields.get("job_id")
                if isinstance(job_id, bytes):
                    job_id = job_id.decode()
                item = self._claim_entry(job_id, entry_id, worker_id)
                if item:
                    return item
        return None

    def _autoclaim(self, worker_id: str) -> Optional[QueueItem]:
        """Claim pending entries that have been idle longer than WORKER_TIMEOUT."""
        try:
            # XAUTOCLAIM: claim entries idle for WORKER_TIMEOUT * 1000 ms
            idle_ms = int(WORKER_TIMEOUT * 1000)
            result = self.client.xautoclaim(
                self.stream_key, self.group_name, worker_id,
                min_idle_time=idle_ms,
                start_id="0-0",
                count=CLAIM_BATCH_SIZE,
            )
            # result: (next_start_id, [(entry_id, fields), ...], [deleted_ids])
            if not result or len(result) < 2:
                return None
            entries = result[1]
            for entry_id, fields in entries:
                if not fields:
                    continue
                job_id = fields.get(b"job_id") or fields.get("job_id")
                if isinstance(job_id, bytes):
                    job_id = job_id.decode()
                item = self._claim_entry(job_id, entry_id, worker_id)
                if item:
                    return item
        except (redis_lib.exceptions.ResponseError, AttributeError):
            # XAUTOCLAIM requires Redis >= 6.2; fall back to manual recovery
            pass
        return None

    def _claim_entry(self, job_id: str, entry_id, worker_id: str) -> Optional[QueueItem]:
        """Mark a stream entry as processing and return the QueueItem."""
        item = self.get_item(job_id)
        if not item:
            # Orphaned entry — ack and skip
            self.client.xack(self.stream_key, self.group_name, entry_id)
            return None

        if item.status not in ("pending", "processing"):
            # Already completed/failed/dead — ack and skip
            self.client.xack(self.stream_key, self.group_name, entry_id)
            return None

        now = time.time()
        item.status = "processing"
        item.processing_at = now
        item.worker_id = worker_id
        item.attempt += 1
        item.updated_at = now
        self.client.hset(self._hash_key(item.job_id), mapping=self._item_to_hash(item))
        # Track entry_id for ACK after completion
        self._pending_acks[job_id] = entry_id if isinstance(entry_id, str) else entry_id.decode()
        return item

    def get_item(self, job_id: str) -> Optional[QueueItem]:
        data = self.client.hgetall(self._hash_key(job_id))
        if not data:
            return None
        return self._hash_to_item(data)

    def update_item(self, item: QueueItem) -> None:
        item.updated_at = time.time()
        self.client.hset(self._hash_key(item.job_id), mapping=self._item_to_hash(item))

    def complete(self, item: QueueItem, result: dict = None, worker_id: str = None) -> None:
        """Mark job as completed, persist, and ACK the stream entry."""
        item.status = "completed"
        item.completed_at = time.time()
        if worker_id:
            item.worker_id = worker_id
        if result is not None:
            item.result = result
        if "artifact_path" in item.result:
            try:
                item.artifact_size = os.path.getsize(item.result["artifact_path"])
            except OSError:
                pass
        self._persist_and_ack(item)

    def fail(self, item: QueueItem, error: str, worker_id: str = None) -> None:
        """Mark job as failed. Retry or move to dead-letter."""
        item.error = error
        item.failed_at = time.time()
        if worker_id:
            item.worker_id = worker_id

        if item.attempt < MAX_RETRIES:
            # Retry: reset to pending and re-add to stream
            item.status = "pending"
            item.processing_at = None
            item.updated_at = time.time()
            self.client.hset(self._hash_key(item.job_id), mapping=self._item_to_hash(item))
            self.client.xadd(self.stream_key, {"job_id": item.job_id})
        else:
            # Exhausted retries: move to dead-letter
            item.status = "dead"
            item.updated_at = time.time()
            self._move_to_dead(item)

    def _persist_and_ack(self, item: QueueItem) -> None:
        """Persist final state and ACK the stream entry."""
        pipe = self.client.pipeline()
        pipe.hset(self._hash_key(item.job_id), mapping=self._item_to_hash(item))
        entry_id = self._pending_acks.pop(item.job_id, None)
        if entry_id:
            pipe.xack(self.stream_key, self.group_name, entry_id)
        pipe.execute()

    def _move_to_dead(self, item: QueueItem) -> None:
        """Move job to dead-letter stream, ACK the original entry, and clean up."""
        entry_id = self._pending_acks.pop(item.job_id, None)
        pipe = self.client.pipeline()
        pipe.hset(self._hash_key(item.job_id), mapping=self._item_to_hash(item))
        pipe.xadd(self.dead_key, {
            "job_id": item.job_id,
            "error": item.error or "",
            "attempts": str(item.attempt),
            "failed_at": str(time.time()),
        })
        if entry_id:
            pipe.xack(self.stream_key, self.group_name, entry_id)
        pipe.execute()

    def recover(self) -> List[QueueItem]:
        """Recover pending entries from dead workers via XAUTOCLAIM for all consumers."""
        recovered = []
        try:
            # Get pending entries info
            pending = self.client.xpending_range(
                self.stream_key, self.group_name,
                min="-", max="+", count=1000,
            )
            if not pending:
                return recovered

            # Group by consumer and auto-claim stale ones
            consumers = set()
            for entry in pending:
                consumer = entry.get("consumer")
                if isinstance(consumer, bytes):
                    consumer = consumer.decode()
                consumers.add(consumer)

            for consumer in consumers:
                try:
                    idle_ms = int(WORKER_TIMEOUT * 1000)
                    result = self.client.xautoclaim(
                        self.stream_key, self.group_name, "recovery-worker",
                        min_idle_time=idle_ms,
                        start_id="0-0",
                        count=100,
                    )
                    if result and len(result) >= 2:
                        for entry_id, fields in result[1]:
                            if not fields:
                                continue
                            job_id = fields.get(b"job_id") or fields.get("job_id")
                            if isinstance(job_id, bytes):
                                job_id = job_id.decode()
                            item = self.get_item(job_id)
                            if item and item.status == "processing":
                                item.status = "pending"
                                item.processing_at = None
                                item.worker_id = None
                                item.updated_at = time.time()
                                self.client.hset(
                                    self._hash_key(job_id),
                                    mapping=self._item_to_hash(item),
                                )
                                recovered.append(item)
                except (redis_lib.exceptions.ResponseError, AttributeError):
                    continue
        except redis_lib.exceptions.ResponseError:
            pass
        return recovered

    def list_items(self, limit: int = 100) -> List[QueueItem]:
        """List recent jobs by scanning hash keys."""
        items = []
        cursor = 0
        pattern = "music_gen:job:*"
        while len(items) < limit:
            cursor, keys = self.client.scan(cursor, match=pattern, count=100)
            for key in keys:
                item = self.get_item(key.decode().split(":")[-1] if isinstance(key, bytes) else key.split(":")[-1])
                if item:
                    items.append(item)
                    if len(items) >= limit:
                        break
            if cursor == 0:
                break
        return items

    def remove_item(self, job_id: str) -> bool:
        """Remove job metadata. Note: does not remove from stream (already ACKed)."""
        deleted = self.client.delete(self._hash_key(job_id))
        return deleted > 0

    def depth(self) -> Dict[str, int]:
        """Return queue depth metrics."""
        try:
            info = self.client.xinfo_stream(self.stream_key)
            length = info.get("length", 0)
        except redis_lib.exceptions.ResponseError:
            length = 0

        try:
            pending_info = self.client.xpending(self.stream_key, self.group_name)
            pending = pending_info.get("pending", 0)
        except redis_lib.exceptions.ResponseError:
            pending = 0

        return {
            "stream_length": length,
            "pending": pending,
            "total_active": length,
        }

    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive queue metrics."""
        depth = self.depth()

        # Count by status via scan
        counts = {"pending": 0, "processing": 0, "completed": 0, "failed": 0, "dead": 0}
        cursor = 0
        pattern = "music_gen:job:*"
        while True:
            cursor, keys = self.client.scan(cursor, match=pattern, count=200)
            for key in keys:
                data = self.client.hget(key, "status")
                if data:
                    status = data.decode() if isinstance(data, bytes) else data
                    if status in counts:
                        counts[status] += 1
            if cursor == 0:
                break

        # Dead letter count
        try:
            dead_info = self.client.xinfo_stream(self.dead_key)
            dead_count = dead_info.get("length", 0)
        except redis_lib.exceptions.ResponseError:
            dead_count = 0

        return {
            **depth,
            **counts,
            "dead": dead_count,
        }

    # --- Key helpers ---

    def _hash_key(self, job_id: str) -> str:
        return f"music_gen:job:{job_id}"

    def _item_to_hash(self, item: QueueItem) -> Dict[str, str]:
        """Convert QueueItem to a flat hash mapping."""
        return {
            "job_id": item.job_id,
            "prompt": item.prompt,
            "mood": item.mood,
            "tempo": str(item.tempo),
            "key": item.key,
            "length": str(item.length),
            "seed": str(item.seed) if item.seed is not None else "",
            "status": item.status,
            "result": json.dumps(item.result),
            "created_at": str(item.created_at),
            "updated_at": str(item.updated_at),
            "queued_at": str(item.queued_at) if item.queued_at is not None else "",
            "processing_at": str(item.processing_at) if item.processing_at is not None else "",
            "completed_at": str(item.completed_at) if item.completed_at is not None else "",
            "failed_at": str(item.failed_at) if item.failed_at is not None else "",
            "worker_id": item.worker_id or "",
            "attempt": str(item.attempt),
            "artifact_size": str(item.artifact_size) if item.artifact_size is not None else "",
            "render_duration_ms": str(item.render_duration_ms) if item.render_duration_ms is not None else "",
            "error": item.error or "",
        }

    def _hash_to_item(self, data: Dict) -> QueueItem:
        """Convert a Redis hash dict back to a QueueItem."""
        def get_float(key):
            v = data.get(key.encode()) or data.get(key)
            if v is None:
                return None
            if isinstance(v, bytes):
                v = v.decode()
            return float(v) if v else None

        def get_int(key):
            v = data.get(key.encode()) or data.get(key)
            if v is None:
                return None
            if isinstance(v, bytes):
                v = v.decode()
            return int(v) if v else None

        def get_str(key):
            v = data.get(key.encode()) or data.get(key)
            if v is None:
                return ""
            if isinstance(v, bytes):
                v = v.decode()
            return v

        result_raw = get_str("result")
        try:
            result = json.loads(result_raw) if result_raw else {}
        except json.JSONDecodeError:
            result = {}

        return QueueItem(
            job_id=get_str("job_id"),
            prompt=get_str("prompt"),
            mood=get_str("mood"),
            tempo=get_int("tempo") or 120,
            key=get_str("key") or "C",
            length=get_int("length") or 30,
            seed=get_int("seed"),
            status=get_str("status") or "pending",
            result=result,
            created_at=get_float("created_at"),
            queued_at=get_float("queued_at"),
            processing_at=get_float("processing_at"),
            completed_at=get_float("completed_at"),
            failed_at=get_float("failed_at"),
            worker_id=get_str("worker_id") or None,
            attempt=get_int("attempt") or 0,
            artifact_size=get_int("artifact_size"),
            render_duration_ms=get_float("render_duration_ms"),
            error=get_str("error") or None,
        )


def get_queue() -> BaseQueue:
    """Factory function to get the configured queue instance.

    If Redis is configured but unavailable, raises RuntimeError instead of
    silently falling back to FileQueue.
    """
    if QUEUE_TYPE == "redis":
        if not REDIS_AVAILABLE:
            raise RuntimeError(
                "MG_QUEUE_TYPE=redis but redis package is not installed. "
                "Install with: pip install redis"
            )
        # Verify Redis is reachable
        try:
            q = RedisQueue()
            q.client.ping()
            return q
        except redis_lib.ConnectionError as e:
            raise RuntimeError(
                f"MG_QUEUE_TYPE=redis but cannot connect to Redis at "
                f"{REDIS_HOST}:{REDIS_PORT}: {e}"
            )
    else:
        return FileQueue()
