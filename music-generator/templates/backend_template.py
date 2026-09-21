"""
Template for a custom music generation backend.
Copy this file to scripts/backends/<your_backend_name>.py and implement the generate method.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from ..config import OUTPUT_DIR


class BaseBackend(ABC):
    def __init__(self):
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
        pass  # Implement this method


# Example implementation (remove or replace with your own)
class ExampleBackend(BaseBackend):
    def generate(self, job_id: str, prompt: str, mood: str, tempo: int, key: str, length: int) -> dict:
        # TODO: Implement actual generation logic
        # For now, just return a mock success
        output_file = self.output_dir / f"{job_id}.wav"
        # Write a dummy WAV file or just a placeholder
        output_file.write_text(f"Mock audio for job {job_id}")
        return {
            "status": "success",
            "output_file": str(output_file),
            "metadata": {
                "sample_rate": 44100,
                "duration": length,
                "key": key,
                "tempo": tempo,
                "mood": mood,
                "note": "This is a placeholder. Replace with real generation."
            }
        }