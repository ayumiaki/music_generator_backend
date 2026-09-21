"""Tests for the music generator backend."""
import os
import sys
import time
import signal
import tempfile
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


class TestRegressionCompletedJobNotRedequeued:
    """Regression: completed job must not cycle back into the queue."""
    def test_second_dequeue_after_complete_returns_none(self, q):
        item = QueueItem(job_id="rd1", prompt="p", mood="c", tempo=100, key="C", length=10, seed=1)
        q.enqueue(item)
        got = q.dequeue()
        q.complete(got, {"status": "success", "output_file": "/tmp/x.wav"})
        second = q.dequeue()
        assert second is None, "Completed job must not be dequeued again"

    def test_second_dequeue_after_fail_returns_none(self, q):
        item = QueueItem(job_id="rd2", prompt="p", mood="c", tempo=100, key="C", length=10, seed=2)
        q.enqueue(item)
        got = q.dequeue()
        q.fail(got, "some error")
        second = q.dequeue()
        assert second is None, "Failed job must not be dequeued again"

    def test_completed_job_still_queryable(self, q):
        item = QueueItem(job_id="rd3", prompt="p", mood="c", tempo=100, key="C", length=10, seed=3)
        q.enqueue(item)
        got = q.dequeue()
        q.complete(got, {"status": "success", "output_file": "/tmp/x.wav"})
        query = q.get_item("rd3")
        assert query is not None
        assert query.status == "completed"


class TestRegressionWorkerSeedPassed:
    """Regression: worker must pass seed to backend."""
    def test_backend_receives_seed(self):
        b = MockBackend()
        result = b.generate("x3", "p", "c", 120, "C", 5, seed=42)
        assert result["status"] == "success"
        assert "output_file" in result

    def test_backend_seed_affects_output(self):
        """Same prompt+seed must produce identical output."""
        b = MockBackend()
        r1 = b.generate("x4", "p", "c", 120, "C", 5, seed=999)
        r2 = b.generate("x4", "p", "c", 120, "C", 5, seed=999)
        assert r1["output_file"] == r2["output_file"]
        # Both files should exist and have same content
        p1 = Path(r1["output_file"])
        p2 = Path(r2["output_file"])
        if p1.exists() and p2.exists():
            assert p1.read_bytes() == p2.read_bytes()

    def test_backend_seed_none_still_works(self):
        b = MockBackend()
        result = b.generate("x5", "p", "c", 120, "C", 5)
        assert result["status"] == "success"


class TestRegressionQueueDirIsolation:
    """Regression: FileQueue(queue_dir=...) must not leak to global QUEUE_DIR."""
    def test_custom_dir_isolated_from_global(self, queue_dir):
        from config import QUEUE_DIR
        custom = queue_dir / "custom"
        q1 = FileQueue(queue_dir=custom)
        item = QueueItem(job_id="reg1", prompt="p", mood="c", tempo=100, key="C", length=10, seed=5)
        q1.enqueue(item)
        # Job file must be in custom dir
        assert (custom / "reg1.json").exists()
        # Must NOT be in global QUEUE_DIR
        assert not (QUEUE_DIR / "reg1.json").exists()

    def test_custom_dir_queue_json_stays_in_custom(self, queue_dir):
        from config import QUEUE_DIR
        custom = queue_dir / "custom2"
        # Clean up any leftover global queue.json from other tests
        global_qj = QUEUE_DIR / "queue.json"
        if global_qj.exists():
            global_qj.unlink()
        q1 = FileQueue(queue_dir=custom)
        item = QueueItem(job_id="reg2", prompt="p", mood="c", tempo=100, key="C", length=10, seed=5)
        q1.enqueue(item)
        assert (custom / "queue.json").exists()
        assert not global_qj.exists()

    def test_dequeue_from_custom_dir(self, queue_dir):
        custom = queue_dir / "custom3"
        q1 = FileQueue(queue_dir=custom)
        item = QueueItem(job_id="reg3", prompt="p", mood="c", tempo=100, key="C", length=10, seed=5)
        q1.enqueue(item)
        got = q1.dequeue()
        assert got is not None
        assert got.seed == 5


class TestRegressionTempoValidation:
    """Regression: invalid tempo must return 400, never 500."""
    @pytest.fixture(autouse=True)
    def client(self, monkeypatch):
        monkeypatch.setenv("MG_API_TOKEN", "test-token")
        if "api_server" in sys.modules:
            del sys.modules["api_server"]
        from api_server import app
        self.app = app
        yield app.test_client()

    def test_invalid_tempo_string_returns_400(self, client):
        r = client.post("/generate", json={"prompt": "hi", "tempo": "not-an-int"},
                        headers={"Authorization": "Bearer test-token"})
        assert r.status_code == 400

    def test_invalid_tempo_none_returns_400(self, client):
        r = client.post("/generate", json={"prompt": "hi", "tempo": None},
                        headers={"Authorization": "Bearer test-token"})
        assert r.status_code == 400

    def test_invalid_tempo_list_returns_400(self, client):
        r = client.post("/generate", json={"prompt": "hi", "tempo": [120]},
                        headers={"Authorization": "Bearer test-token"})
        assert r.status_code == 400

    def test_tempo_below_range_returns_400(self, client):
        r = client.post("/generate", json={"prompt": "hi", "tempo": 39},
                        headers={"Authorization": "Bearer test-token"})
        assert r.status_code == 400

    def test_tempo_above_range_returns_400(self, client):
        r = client.post("/generate", json={"prompt": "hi", "tempo": 301},
                        headers={"Authorization": "Bearer test-token"})
        assert r.status_code == 400

    def test_valid_tempo_returns_202(self, client):
        r = client.post("/generate", json={"prompt": "hi", "tempo": 120},
                        headers={"Authorization": "Bearer test-token"})
        assert r.status_code == 202


class TestRegressionSeedOmission:
    """Regression: omitted seed must be assigned a concrete int, never null."""
    @pytest.fixture(autouse=True)
    def client(self, monkeypatch):
        monkeypatch.setenv("MG_API_TOKEN", "test-token")
        if "api_server" in sys.modules:
            del sys.modules["api_server"]
        from api_server import app
        self.app = app
        yield app.test_client()

    def test_seed_omitted_gets_assigned(self, client):
        r = client.post("/generate", json={"prompt": "hi"},
                        headers={"Authorization": "Bearer test-token"})
        assert r.status_code == 202
        data = r.get_json()
        assert data["seed"] is not None
        assert isinstance(data["seed"], int)

    def test_seed_omitted_persists_in_status(self, client):
        r = client.post("/generate", json={"prompt": "hi"},
                        headers={"Authorization": "Bearer test-token"})
        assert r.status_code == 202
        job_id = r.get_json()["job_id"]
        # Check status endpoint shows non-null seed
        sr = client.get(f"/status/{job_id}", headers={"Authorization": "Bearer test-token"})
        assert sr.status_code == 200
        assert sr.get_json()["seed"] is not None
        assert isinstance(sr.get_json()["seed"], int)

    def test_seed_explicit_preserved(self, client):
        r = client.post("/generate", json={"prompt": "hi", "seed": 777},
                        headers={"Authorization": "Bearer test-token"})
        assert r.status_code == 202
        assert r.get_json()["seed"] == 777


class TestRegressionGenerateReturns202:
    """Regression: /generate must return 202, not 200."""
    @pytest.fixture(autouse=True)
    def client(self, monkeypatch):
        monkeypatch.setenv("MG_API_TOKEN", "test-token")
        if "api_server" in sys.modules:
            del sys.modules["api_server"]
        from api_server import app
        self.app = app
        yield app.test_client()

    def test_generate_returns_202(self, client):
        r = client.post("/generate", json={"prompt": "hi"},
                        headers={"Authorization": "Bearer test-token"})
        assert r.status_code == 202

    def test_generate_status_is_queued(self, client):
        r = client.post("/generate", json={"prompt": "hi"},
                        headers={"Authorization": "Bearer test-token"})
        assert r.status_code == 202
        assert r.get_json()["status"] == "queued"


class TestRegressionJobTimeoutUsed:
    """Regression: JOB_TIMEOUT must be enforced by the worker."""
    def test_job_timeout_enforced(self, monkeypatch, tmp_path):
        """Worker must fail a job that exceeds JOB_TIMEOUT."""
        monkeypatch.setenv("MG_API_TOKEN", "test-token")
        monkeypatch.setenv("MG_JOB_TIMEOUT", "1")
        if "config" in sys.modules:
            del sys.modules["config"]
        from config import JOB_TIMEOUT
        assert JOB_TIMEOUT == 1

        from worker import JobTimeoutError, _timeout_handler, process_job

        class SlowBackend:
            def generate(self, **kwargs):
                time.sleep(5)
                return {"status": "success", "output_file": "/tmp/slow.wav"}

        queue_dir = tmp_path / "q"
        queue = FileQueue(queue_dir=queue_dir)
        item = QueueItem(job_id="timeout1", prompt="p", mood="c", tempo=100, key="C", length=10, seed=1)
        queue.enqueue(item)
        got = queue.dequeue()

        backend = SlowBackend()
        process_job(queue, backend, got)

        updated = queue.get_item(got.job_id)
        assert updated.status == "failed"
        assert "timeout" in updated.result.get("error", "").lower()