"""
Abstract base class for music generation backends.
"""

from abc import ABC, abstractmethod
from pathlib import Path


class BaseBackend(ABC):
    def __init__(self):
        # Import here to avoid circular imports
        from ..config import OUTPUT_DIR
        self.output_dir = Path(OUTPUT_DIR)
        self.output_dir.mkdir(exist_ok=True)

    @abstractmethod
    def generate(self, job_id: str, prompt: str, mood: str, tempo: int, key: str, length: int) -> dict:
        """
        Generate music based on the given parameters.

        Args:
            job_id: Unique identifier for the job
            prompt: Text description of the desired music
            mood: Mood descriptor (e.g., happy, sad, energetic, chill)
            tempo: Beats per minute (integer)
            key: Musical key (string, e.g., "C", "G#", "F")
            length: Duration in seconds (integer)

        Returns:
            dict: {
                "status": "success" or "error",
                "output_file": absolute path to generated audio file (if success),
                "metadata": dict with extra info (if success),
                "error": error message (if error)
            }
        """
        pass