"""Abstract base class for music generation backends."""

from abc import ABC, abstractmethod
from pathlib import Path

from config import OUTPUT_DIR


class BaseBackend(ABC):
    def __init__(self):
        self.output_dir = Path(OUTPUT_DIR)
        self.output_dir.mkdir(exist_ok=True)

    @abstractmethod
    def generate(self, job_id: str, prompt: str, mood: str, tempo: int, key: str, length: int, seed: int | None = None, output_path: str | None = None) -> dict:
        """Generate audio. If output_path is given, write there; otherwise use default."""