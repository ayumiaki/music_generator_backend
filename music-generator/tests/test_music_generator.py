"""Tests for the music generator backend."""
import os
import sys
import tempfile
import shutil
from pathlib import Path

# Ensure scripts dir is on path
SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS))

os.chdir(str(SCRIPTS))

import pytest
from job_queue import QueueItem, FileQueue
from backends.mock_backend import MockBackend


@pytest.fixture
def queue_dir(tmp_path):
    """Isolated queue directory for each test."""
    orig = os.environ.get("MG_QUEUE_DIR")
    os.environ["MG_QUEUE_DIR"] = str(tmp_path / "queue")
    yield tmp_path / "queue"
    if orig:
        os.environ["MG_QUEUE_DIR"] = orig
    else:
        os.environ.pop("MG_QUEUE_DIR", None)


@pytest.fixture
def q(queue_dir):
    return FileQueue(queue_dir=queue_dir)


class TestQueueItemSeed:
    def test_seed_persists_via_to_dict(self):
        item = QueueItem(job_id="j1", prompt="p", mood="c", tempo=90, key="C", length=30, seed=42)
        d = item.to_dict()
        assert d["seed"] == 42

    def test_seed_restored_via_from_dict(self):
        item = QueueItem(job_id="j1", prompt="p", mood="c", tempo=90, key="C", length=30, seed=99)
        restored = QueueItem.from_dict(item.to_dict())
        assert restored.seed == 99

    def test_seed_none_by_default(self):
        item = QueueItem(job_id="j1", prompt="p", mood="c", tempo=90, key="C", length=30)
        assert item.seed is None


class TestFileQueueAtomicity:
    def test_enqueue_dequeue_preserves_seed(self, q):
        item = QueueItem(job_id="a1", prompt="x", mood="c", tempo=100, key="G", length=20, seed=7)
        q.enqueue(item)
        got = q.dequeue()
        assert got.seed == 7
        assert got.status == "processing"

    def test_recover_resets_processing_to_pending(self, q):
        item = QueueItem(job_id="b1", prompt="y", mood="c", tempo=110, key="D", length=15, seed=13)
        q.enqueue(item)
        q.dequeue()  # marks processing
        recovered = q.recover()
        assert len(recovered) == 1
        assert recovered[0].seed == 13
        assert recovered[0].status == "pending"

    def test_complete_removes_from_queue_list(self, q):
        item = QueueItem(job_id="c1", prompt="z", mood="c", tempo=100, key="C", length=10, seed=1)
        q.enqueue(item)
        got = q.dequeue()  # processing
        q.complete(got, {"status": "success", "output_file": "/tmp/x.wav"})
        # Should not appear in dequeue anymore
        next_item = q.dequeue()
        assert next_item is None
        # But still queryable by ID
        query = q.get_item("c1")
        assert query is not None
        assert query.status == "completed"

    def test_fail_removes_from_queue_list(self, q):
        item = QueueItem(job_id="d1", prompt="z", mood="c", tempo=100, key="C", length=10, seed=2)
        q.enqueue(item)
        got = q.dequeue()
        q.fail(got, "some error")
        next_item = q.dequeue()
        assert next_item is None
        query = q.get_item("d1")
        assert query.status == "failed"

    def test_custom_queue_dir_isolated(self, queue_dir):
        """FileQueue with custom queue_dir stays isolated from global QUEUE_DIR."""
        q1 = FileQueue(queue_dir=queue_dir)
        item = QueueItem(job_id="iso1", prompt="p", mood="c", tempo=100, key="C", length=10, seed=5)
        q1.enqueue(item)
        got = q1.dequeue()
        assert got.seed == 5
        # The global QUEUE_DIR should not contain iso1
        from config import QUEUE_DIR
        global_job = QUEUE_DIR / "iso1.json"
        assert not global_job.exists()
        # But the custom dir has it
        assert (queue_dir / "iso1.json").exists()


class TestMockBackend:
    def test_module_loads_without_numpy(self, monkeypatch):
        # Block numpy import to verify fallback path
        import sys as _sys
        monkeypatch.setitem(_sys.modules, "numpy", None)
        # Force reimport
        if "backends.mock_backend" in _sys.modules:
            del _sys.modules["backends.mock_backend"]
        from backends.mock_backend import MockBackend
        b = MockBackend()
        assert b._can_generate_audio is False
        result = b.generate("x", "p", "c", 100, "C", 5)
        assert result["status"] == "success"
        assert result["output_file"].endswith(".txt")

    def test_backend_generates_result(self):
        b = MockBackend()
        result = b.generate("x2", "p", "c", 120, "A", 3)
        assert result["status"] == "success"
        assert "output_file" in result

    def test_backend_receives_seed(self):
        """Backend should receive the seed parameter."""
        b = MockBackend()
        result = b.generate("x3", "p", "c", 120, "C", 5, seed=42)
        assert result["status"] == "success"
        # Metadata should reflect seed if stored; at minimum call succeeds
        assert "output_file" in result


class TestApiEndpoints:
    @pytest.fixture(autouse=True)
    def client(self, monkeypatch):
        monkeypatch.setenv("MG_API_TOKEN", "test-token")
        # Re-import to pick up env var
        if "api_server" in sys.modules:
            del sys.modules["api_server"]
        from api_server import app
        self.app = app
        yield app.test_client()

    def test_health(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        assert r.get_json()["status"] == "ok"

    def test_generate_requires_auth(self, client):
        r = client.post("/generate", json={"prompt": "hi"})
        assert r.status_code == 401

    def test_generate_valid(self, client):
        r = client.post("/generate", json={"prompt": "hi", "seed": 555},
                        headers={"Authorization": "Bearer test-token"})
        assert r.status_code == 202
        data = r.get_json()
        assert data["seed"] == 555
        assert data["status"] == "queued"

    def test_generate_assigns_seed_when_omitted(self, client):
        r = client.post("/generate", json={"prompt": "hi"},
                        headers={"Authorization": "Bearer test-token"})
        assert r.status_code == 202
        data = r.get_json()
        assert data["seed"] is not None
        assert isinstance(data["seed"], int)

    def test_generate_validates_prompt(self, client):
        r = client.post("/generate", json={"prompt": "", "seed": 1},
                        headers={"Authorization": "Bearer test-token"})
        assert r.status_code == 400

    def test_generate_validates_tempo(self, client):
        r = client.post("/generate", json={"prompt": "hi", "tempo": "not-an-int"},
                        headers={"Authorization": "Bearer test-token"})
        assert r.status_code == 400

    def test_generate_validates_length(self, client):
        r = client.post("/generate", json={"prompt": "hi", "length": 0},
                        headers={"Authorization": "Bearer test-token"})
        assert r.status_code == 400

    def test_generate_validates_seed(self, client):
        r = client.post("/generate", json={"prompt": "hi", "seed": "bad"},
                        headers={"Authorization": "Bearer test-token"})
        assert r.status_code == 400

    def test_status_not_found(self, client):
        r = client.get("/status/nope", headers={"Authorization": "Bearer test-token"})
        assert r.status_code == 404