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
    from config import QUEUE_DIR, QUEUE_TYPE, REDIS_HOST, REDIS_PORT, REDIS_DB, REDIS_QUEUE_KEY
except ImportError:
    # Fallback defaults (should not happen if config.py exists)
    BASE_DIR = Path(__file__).resolve().parent.parent
    QUEUE_DIR = BASE_DIR / "queue"
    QUEUE_TYPE = "file"
    REDIS_HOST = "localhost"
    REDIS_PORT = 6379
    REDIS_DB = 0
    REDIS_QUEUE_KEY = "music_gen:queue"


class QueueItem:
    """Represents a job in the queue."""

    def __init__(self, job_id: str, prompt: str, mood: str, tempo: int, key: str, length: int,
                 seed: Optional[int] = None, status: str = "pending", result: Optional[Dict] = None,
                 created_at: Optional[float] = None, queued_at: Optional[float] = None,
                 processing_at: Optional[float] = None, completed_at: Optional[float] = None,
                 worker_id: Optional[str] = None):
        self.job_id = job_id
        self.prompt = prompt
        self.mood = mood
        self.tempo = tempo
        self.key = key
        self.length = length
        self.seed = seed  # persisted with job, restored on recovery
        self.status = status  # pending, processing, completed, failed
        self.result = result or {}
        self.created_at = created_at or time.time()
        self.updated_at = time.time()
        # Lifecycle timestamps for latency measurement
        self.queued_at = queued_at
        self.processing_at = processing_at
        self.completed_at = completed_at
        self.worker_id = worker_id

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
            "worker_id": self.worker_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QueueItem":
        item = cls(
            job_id=data["job_id"],
            prompt=data["prompt"],
            mood=data["mood"],
            tempo=data["tempo"],
            key=data["key"],
            length=data["length"],
            seed=data.get("seed"),
            status=data.get("status", "pending"),
            result=data.get("result", {}),
            created_at=data.get("created_at"),
            queued_at=data.get("queued_at"),
            processing_at=data.get("processing_at"),
            completed_at=data.get("completed_at"),
            worker_id=data.get("worker_id"),
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
    def depth(self) -> int:
        """Return the number of pending+processing jobs in the queue."""
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

    @abstractmethod
    def _persist_completion(self, item: QueueItem) -> None:
        """Persist completed/failed state and remove from active queue."""
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
        with self._locked() as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                ids = self._read_queue_ids()
                if not ids:
                    return None
                job_id = ids[0]
                job_file = self._job_file_path(job_id)
                if job_file.exists():
                    data = json.loads(job_file.read_text())
                    item = QueueItem.from_dict(data)
                    item.status = "processing"
                    item.processing_at = time.time()
                    item.worker_id = worker_id
                    self._atomic_write(job_file, json.dumps(item.to_dict(), indent=2))
                    return item
                else:
                    # Orphaned ID — remove and skip
                    ids.pop(0)
                    self._write_queue_ids(ids)
                    return None
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    def _persist_completion(self, item: QueueItem) -> None:
        job_file = self._job_file_path(item.job_id)
        self._atomic_write(job_file, json.dumps(item.to_dict(), indent=2))
        with self._locked() as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                self._remove_from_queue_list(item.job_id)
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    def _remove_from_queue_list(self, job_id: str) -> None:
        ids = self._read_queue_ids()
        if job_id in ids:
            ids.remove(job_id)
            self._write_queue_ids(ids)

    def recover(self) -> List[QueueItem]:
        """Pick up any jobs stuck in 'processing' state after a crash."""
        recovered = []
        with self._locked() as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                ids = self._read_queue_ids()
                for job_id in list(ids):
                    job_file = self._job_file_path(job_id)
                    if job_file.exists():
                        try:
                            data = json.loads(job_file.read_text())
                            item = QueueItem.from_dict(data)
                            if item.status == "processing":
                                item.status = "pending"
                                item.processing_at = None
                                item.worker_id = None
                                self._atomic_write(job_file, json.dumps(item.to_dict(), indent=2))
                                recovered.append(item)
                        except (json.JSONDecodeError, KeyError):
                            # Corrupt file — skip
                            pass
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
        ids = self._read_queue_ids()
        items = []
        for job_id in ids[:limit]:  # FIFO order
            item = self.get_item(job_id)
            if item:
                items.append(item)
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

    def depth(self) -> int:
        return len(self._read_queue_ids())


# Try to import redis
try:
    import redis as redis_lib
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False


class RedisQueue(BaseQueue):
    """Redis-backed queue with atomic dequeue via Lua script."""

    # Atomically pop from queue and mark as processing
    DEQUEUE_SCRIPT = """
    local job_id = redis.call('RPOP', KEYS[1])
    if job_id then
        local data = redis.call('HGET', KEYS[2], job_id)
        if data then
            local item = cjson.decode(data)
            item.status = 'processing'
            item.processing_at = tonumber(ARGV[1])
            item.worker_id = ARGV[2]
            redis.call('HSET', KEYS[2], job_id, cjson.encode(item))
            return {job_id, data}
        end
    end
    return nil
    """

    def __init__(self):
        self.client = redis_lib.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB)
        self.queue_key = REDIS_QUEUE_KEY
        self.hash_key = "music_gen:jobs"
        self.processing_key = "music_gen:processing"
        self._dequeue_script = self.client.register_script(self.DEQUEUE_SCRIPT)

    def enqueue(self, item: QueueItem) -> str:
        if not item.job_id:
            item.job_id = str(uuid.uuid4())
        now = time.time()
        item.created_at = now
        item.queued_at = now
        item.updated_at = now
        pipe = self.client.pipeline()
        pipe.hset(self.hash_key, item.job_id, json.dumps(item.to_dict()))
        pipe.rpush(self.queue_key, item.job_id)
        pipe.execute()
        return item.job_id

    def dequeue(self, worker_id: str = None) -> Optional[QueueItem]:
        worker_id = worker_id or f"worker-{os.getpid()}"
        now = time.time()
        result = self._dequeue_script(
            keys=[self.queue_key, self.hash_key],
            args=[str(now), worker_id]
        )
        if result:
            # result is [job_id, original_data]
            data = json.loads(result[1].decode() if isinstance(result[1], bytes) else result[1])
            item = QueueItem.from_dict(data)
            item.status = "processing"
            item.processing_at = now
            item.worker_id = worker_id
            # Track in processing set for crash recovery
            self.client.sadd(self.processing_key, item.job_id)
            return item
        return None

    def get_item(self, job_id: str) -> Optional[QueueItem]:
        data = self.client.hget(self.hash_key, job_id)
        if data:
            data = json.loads(data.decode() if isinstance(data, bytes) else data)
            return QueueItem.from_dict(data)
        return None

    def update_item(self, item: QueueItem) -> None:
        item.updated_at = time.time()
        self.client.hset(self.hash_key, item.job_id, json.dumps(item.to_dict()))

    def _persist_completion(self, item: QueueItem) -> None:
        pipe = self.client.pipeline()
        pipe.hset(self.hash_key, item.job_id, json.dumps(item.to_dict()))
        pipe.srem(self.processing_key, item.job_id)
        pipe.execute()

    def recover(self) -> List[QueueItem]:
        """Pick up jobs stuck in processing state after a crash."""
        recovered = []
        processing_ids = self.client.smembers(self.processing_key)
        for job_id_bytes in processing_ids:
            job_id = job_id_bytes.decode() if isinstance(job_id_bytes, bytes) else job_id_bytes
            data = self.client.hget(self.hash_key, job_id)
            if data:
                data = json.loads(data.decode() if isinstance(data, bytes) else data)
                item = QueueItem.from_dict(data)
                if item.status == "processing":
                    item.status = "pending"
                    item.processing_at = None
                    item.worker_id = None
                    # Re-enqueue at the front
                    pipe = self.client.pipeline()
                    pipe.hset(self.hash_key, job_id, json.dumps(item.to_dict()))
                    pipe.srem(self.processing_key, job_id)
                    pipe.lpush(self.queue_key, job_id)
                    pipe.execute()
                    recovered.append(item)
        return recovered

    def list_items(self, limit: int = 100) -> List[QueueItem]:
        all_ids = self.client.hkeys(self.hash_key)
        items = []
        for job_id_bytes in all_ids[:limit]:
            job_id = job_id_bytes.decode() if isinstance(job_id_bytes, bytes) else job_id_bytes
            item = self.get_item(job_id)
            if item:
                items.append(item)
        return items

    def remove_item(self, job_id: str) -> bool:
        pipe = self.client.pipeline()
        pipe.hdel(self.hash_key, job_id)
        pipe.srem(self.processing_key, job_id)
        pipe.lrem(self.queue_key, 0, job_id)
        pipe.execute()
        return True

    def depth(self) -> int:
        return self.client.llen(self.queue_key) + self.client.scard(self.processing_key)


def get_queue() -> BaseQueue:
    """Factory function to get the configured queue instance."""
    if QUEUE_TYPE == "redis" and REDIS_AVAILABLE:
        return RedisQueue()
    else:
        return FileQueue()
