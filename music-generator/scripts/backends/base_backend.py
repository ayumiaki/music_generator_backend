"""Abstract base class for music generation backends."""

from abc import ABC, abstractmethod
from pathlib import Path

from config import OUTPUT_DIR


class BaseBackend(ABC):
    def __init__(self):
        self.output_dir = Path(OUTPUT_DIR)
        self.output_dir.mkdir(exist_ok=True)

    @abstractmethod
    def generate(self, job_id, prompt, mood, tempo, key, length):
        pass