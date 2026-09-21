# Backend Interface

All backends must inherit from `BaseBackend` defined in `scripts/backends/base_backend.py`.

## BaseBackend

```python
class BaseBackend(ABC):
    def __init__(self):
        self.output_dir = OUTPUT_DIR  # from config
        self.output_dir.mkdir(exist_ok=True)

    @abstractmethod
    def generate(self, job_id: str, prompt: str, mood: str, tempo: int, key: str, length: int) -> dict:
        ...
```

### Input Parameters

- `job_id`: Unique identifier for the job (string)
- `prompt`: Text description of the desired music
- `mood`: Mood descriptor (e.g., happy, sad, energetic, chill)
- `tempo`: Beats per minute (integer)
- `key`: Musical key (string, e.g., "C", "G#", "F")
- `length`: Duration in seconds (integer)

### Return Value

The `generate` method must return a dictionary with the following keys:

- `status`: Either `"success"` or `"error"`
- If `status` is `"success"`:
  - `output_file`: Absolute path to the generated audio file (any format, but preferably `.wav` or `.mp3`)
  - `metadata`: Optional dictionary with additional information (e.g., sample rate, duration, generation parameters)
- If `status` is `"error"`:
  - `error`: String describing the error

### Example Success Return

```python
return {
    "status": "success",
    "output_file": "/path/to/output/gen_001.wav",
    "metadata": {
        "sample_rate": 44100,
        "duration": 30,
        "key": "C",
        "tempo": 90,
        "mood": "chill",
        "synthesis": "sine wave mock"
    }
}
```

### Example Error Return

```python
return {
    "status": "error",
    "error": "API rate limit exceeded"
}
```

## Output File Requirements

- The output file must exist and be readable after the method returns.
- For audio files, common formats are `.wav` (uncompressed) or `.mp3` (compressed).
- If generating intermediate files, ensure the final output is placed at the path indicated by `output_file`.

## Thread Safety

Backend instances may be used concurrently by multiple worker threads or processes. Ensure your implementation is thread-safe or stateless.

## Configuration

Backends can read configuration from the `config` module or environment variables as needed.

## See Also

- `scripts/backends/mock_backend.py` for a simple example.
- `templates/backend_template.py` for a starter template.