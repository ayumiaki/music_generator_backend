# Queue Design Choices

The music generator supports two queue implementations: file-based and Redis.

## File-Based Queue

- Each job is stored as a JSON file in the `queue/` directory.
- A master `queue.json` maintains a list of job IDs in FIFO order.
- Simple, zero-dependency, suitable for light usage and development.
- Not suitable for distributed workers (multiple workers on different machines) without additional locking.

## Redis Queue

- Uses Redis hashes to store job data and a Redis list for the queue.
- Enables multiple workers across different machines (if they share the same Redis instance).
- Requires Redis server and the `redis` Python package.
- Configuration via environment variables: `MG_REDIS_HOST`, `MG_REDIS_PORT`, `MG_REDIS_DB`, `MG_QUEUE_TYPE=redis`.

## Choosing a Queue

- For single-machine development and testing, use the file queue (default).
- For production or distributed processing, use Redis.
- The queue type is set via `MG_QUEUE_TYPE` environment variable or in `scripts/config.py`.

## Implementation Details

Both queue types implement the same abstract base class (`BaseQueue`), making it easy to swap or extend with other queue systems (e.g., RabbitMQ, Amazon SQS).

See `scripts/job_queue.py` for the full implementation.