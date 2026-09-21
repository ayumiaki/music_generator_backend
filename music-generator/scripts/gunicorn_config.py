"""
Gunicorn configuration for music generator API.
Usage: gunicorn -c gunicorn_config.py api_server:app
"""
import multiprocessing
import os

# Server socket
bind = f"0.0.0.0:{os.environ.get('PORT', '8000')}"

# Worker processes — use sync workers since rendering is CPU-bound
# For CPU-bound work, workers ≈ CPU cores + 1
# For I/O-bound (e.g., file queue polling), could use gthread
workers = int(os.environ.get("GUNICORN_WORKERS", multiprocessing.cpu_count() + 1))
worker_class = "sync"

# Timeout — generous since WAV rendering can take time
timeout = int(os.environ.get("GUNICORN_TIMEOUT", 120))

# Logging
accesslog = "-"
loglevel = os.environ.get("LOG_LEVEL", "info")

# Don't preload — each worker needs its own queue connection
preload_app = False
