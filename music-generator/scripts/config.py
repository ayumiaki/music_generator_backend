"""
Configuration for the music generator.
"""

import os
from pathlib import Path

# Base directory of the skill
BASE_DIR = Path(__file__).resolve().parent.parent

# Queue settings
QUEUE_TYPE = os.getenv("MG_QUEUE_TYPE", "file")  # file or redis
QUEUE_DIR = Path(os.getenv("MG_QUEUE_DIR", BASE_DIR / "queue"))

# Redis settings (used only if QUEUE_TYPE == "redis")
REDIS_HOST = os.getenv("MG_REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("MG_REDIS_PORT", 6379))
REDIS_DB = int(os.getenv("MG_REDIS_DB", 0))
REDIS_QUEUE_KEY = os.getenv("MG_REDIS_QUEUE_KEY", "music_gen:queue")

# Backend settings
BACKEND_TYPE = os.getenv("MG_BACKEND_TYPE", "mock")  # mock or real (future)

# Worker settings
JOB_TIMEOUT = int(os.getenv("MG_JOB_TIMEOUT", 30))  # seconds
POLL_INTERVAL = float(os.getenv("MG_POLL_INTERVAL", 1.0))  # seconds

# Mock backend settings
MOCK_MIN_DELAY = float(os.getenv("MG_MOCK_MIN_DELAY", 0.5))
MOCK_MAX_DELAY = float(os.getenv("MG_MOCK_MAX_DELAY", 2.0))

# Output directory
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# HTTP API settings (future)
HOST = os.getenv("MG_HOST", "0.0.0.0")
PORT = int(os.getenv("MG_PORT", 8000))

# API authentication
MG_API_TOKEN = os.getenv("MG_API_TOKEN", "")