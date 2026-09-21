"""
Job queue implementation for music generator.
Supports file-based and Redis backends.
"""

import json
import os
import time
import uuid
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
                 seed: Optional[int] = None, status: str = "pending", result: Optional[Dict] = None, created_at: float = None):
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
            "updated_at": self.updated_at
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
            created_at=data.get("created_at", time.time())
        )
        item.updated_at = data.get("updated_at", time.time())
        return item


class BaseQueue(ABC):
    """Abstract base queue."""

    @abstractmethod
    def enqueue(self, item: QueueItem) -> str:
        pass

    @abstractmethod
    def dequeue(self) -> Optional[QueueItem]:
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


class FileQueue(BaseQueue):
    """File-based queue storing each job as a JSON file in QUEUE_DIR."""

    def __init__(self, queue_dir=None):
        self.queue_dir = Path(queue_dir) if queue_dir else QUEUE_DIR
        self.queue_dir.mkdir(exist_ok=True)
        self.queue_file = self.queue_dir / "queue.json"  # simple list of job IDs
        self._ensure_queue_file()

    def _atomic_write(self, path: Path, content: str):
        """Atomic write: write to temp file then rename."""
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(content)
        tmp.replace(path)

    def _ensure_queue_file(self):
        if not self.queue_file.exists():
            self._atomic_write(self.queue_file, "[]")

    def _read_queue_ids(self) -> List[str]:
        try:
            return json.loads(self.queue_file.read_text())
        except (json.JSONDecodeError, FileNotFoundError):
            return []

    def _write_queue_ids(self, ids: List[str]):
        self._atomic_write(self.queue_file, json.dumps(ids, indent=2))

    def _job_file_path(self, job_id: str) -> Path:
        return QUEUE_DIR / f"{job_id}.json"

    def enqueue(self, item: QueueItem) -> str:
        if not item.job_id:
            item.job_id = str(uuid.uuid4())
        job_file = self._job_file_path(item.job_id)
        self._atomic_write(job_file, json.dumps(item.to_dict(), indent=2))
        ids = self._read_queue_ids()
        if item.job_id not in ids:
            ids.append(item.job_id)
            self._write_queue_ids(ids)
        return item.job_id

    def dequeue(self) -> Optional[QueueItem]:
        ids = self._read_queue_ids()
        if not ids:
            return None
        # Take first ID (FIFO) but keep it in the list for crash recovery
        job_id = ids[0]
        job_file = self._job_file_path(job_id)
        if job_file.exists():
            data = json.loads(job_file.read_text())
            item = QueueItem.from_dict(data)
            # Mark as processing in place - do NOT remove from queue list
            item.status = "processing"
            self._atomic_write(job_file, json.dumps(item.to_dict(), indent=2))
            return item
        return None

    def recover(self) -> List[QueueItem]:
        """Pick up any jobs stuck in 'processing' state after a crash."""
        ids = self._read_queue_ids()
        recovered = []
        for job_id in ids:
            job_file = self._job_file_path(job_id)
            if job_file.exists():
                data = json.loads(job_file.read_text())
                item = QueueItem.from_dict(data)
                if item.status == "processing":
                    item.status = "pending"
                    self._atomic_write(job_file, json.dumps(item.to_dict(), indent=2))
                    recovered.append(item)
        return recovered

    def get_item(self, job_id: str) -> Optional[QueueItem]:
        job_file = self._job_file_path(job_id)
        if job_file.exists():
            data = json.loads(job_file.read_text())
            return QueueItem.from_dict(data)
        return None

    def update_item(self, item: QueueItem) -> None:
        job_file = self._job_file_path(item.job_id)
        job_file.write_text(json.dumps(item.to_dict(), indent=2))
        # Update timestamp
        item.updated_at = time.time()

    def list_items(self, limit: int = 100) -> List[QueueItem]:
        ids = self._read_queue_ids()
        items = []
        for job_id in ids[-limit:]:  # most recent first? we'll just take last 'limit'
            item = self.get_item(job_id)
            if item:
                items.append(item)
        return items

    def remove_item(self, job_id: str) -> bool:
        job_file = self._job_file_path(job_id)
        if job_file.exists():
            job_file.unlink()
            ids = self._read_queue_ids()
            if job_id in ids:
                ids.remove(job_id)
                self._write_queue_ids(ids)
            return True
        return False


# Try to import redis
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False


if QUEUE_TYPE == "redis" and REDIS_AVAILABLE:
    class RedisQueue(BaseQueue):
        """Redis-backed queue using Redis lists and hashes."""

        def __init__(self):
            self.client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB)
            self.queue_key = REDIS_QUEUE_KEY
            self.hash_key = "music_gen:jobs"

        def enqueue(self, item: QueueItem) -> str:
            if not item.job_id:
                item.job_id = str(uuid.uuid4())
            # Store job hash
            self.client.hset(self.hash_key, item.job_id, json.dumps(item.to_dict()))
            # Push to queue list
            self.client.lpush(self.queue_key, item.job_id)
            return item.job_id

        def dequeue(self) -> Optional[QueueItem]:
            job_id = self.client.rpop(self.queue_key)  # FIFO: left push, right pop
            if job_id:
                job_id = job_id.decode() if isinstance(job_id, bytes) else job_id
                data = self.client.hget(self.hash_key, job_id)
                if data:
                    data = json.loads(data.decode() if isinstance(data, bytes) else data)
                    return QueueItem.from_dict(data)
            return None

        def get_item(self, job_id: str) -> Optional[QueueItem]:
            data = self.client.hget(self.hash_key, job_id)
            if data:
                data = json.loads(data.decode() if isinstance(data, bytes) else data)
                return QueueItem.from_dict(data)
            return None

        def update_item(self, item: QueueItem) -> None:
            self.client.hset(self.hash_key, item.job_id, json.dumps(item.to_dict()))

        def recover(self) -> List[QueueItem]:
            """Pick up jobs stuck in processing state after a crash."""
            all_ids = self.client.hkeys(self.hash_key)
            recovered = []
            for job_id in all_ids:
                job_id = job_id.decode() if isinstance(job_id, bytes) else job_id
                data = self.client.hget(self.hash_key, job_id)
                if data:
                    data = json.loads(data.decode() if isinstance(data, bytes) else data)
                    item = QueueItem.from_dict(data)
                    if item.status == "processing":
                        item.status = "pending"
                        self.client.hset(self.hash_key, job_id, json.dumps(item.to_dict()))
                        recovered.append(item)
            return recovered

        def list_items(self, limit: int = 100) -> List[QueueItem]:
            # Get all job IDs from hash
            all_ids = self.client.hkeys(self.hash_key)
            items = []
            for job_id in all_ids[-limit:]:
                job_id = job_id.decode() if isinstance(job_id, bytes) else job_id
                item = self.get_item(job_id)
                if item:
                    items.append(item)
            return items

        def remove_item(self, job_id: str) -> bool:
            # Remove from hash and queue list
            self.client.hdel(self.hash_key, job_id)
            # Remove all occurrences from queue list (not efficient but okay for small scale)
            self.client.lrem(self.queue_key, 0, job_id)
            return True

else:
    # Fallback to file queue if Redis not available or not selected
    RedisQueue = FileQueue  # type: ignore


def get_queue() -> BaseQueue:
    """Factory function to get the configured queue instance."""
    if QUEUE_TYPE == "redis" and REDIS_AVAILABLE:
        return RedisQueue()
    else:
        return FileQueue()