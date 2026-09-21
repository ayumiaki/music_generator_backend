"""
Simple Flask API server for the music generator.
Exposes:
  POST /generate  - submit a generation job
  GET  /status/<job_id> - check job status
  GET  /artifact/<job_id> - download generated artifact
  GET  /health - health check
"""

import os
import sys
import json
import random
import time
import uuid
from pathlib import Path

from flask import Flask, request, jsonify, send_file

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import OUTPUT_DIR, QUEUE_DIR, HOST, PORT, MG_API_TOKEN, BACKEND_TYPE, QUEUE_TYPE
from job_queue import get_queue, QueueItem

app = Flask(__name__)
queue = get_queue()

if BACKEND_TYPE == "synth":
    from backends.synth_backend import SynthBackend
    backend = SynthBackend()
else:
    from backends.mock_backend import MockBackend
    backend = MockBackend()

def _check_auth() -> bool:
    token = os.environ.get("MG_API_TOKEN", MG_API_TOKEN or "")
    if not token:
        return True  # no token configured = open for dev
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        return auth[7:] == token
    return False

@app.before_request
def _require_auth():
    if request.endpoint == 'health':
        return
    if not _check_auth():
        return jsonify({"error": "unauthorized"}), 401

@app.route('/health', methods=['GET'])
def health():
    backend_type = "synth" if BACKEND_TYPE == "synth" else "mock"
    queue_type = "redis" if QUEUE_TYPE == "redis" else "file"
    return jsonify({
        "status": "ok",
        "queue": queue_type,
        "backend": backend_type,
        "queue_depth": queue.depth(),
    })

@app.route('/queue/status', methods=['GET'])
def queue_status():
    """Return queue metrics for monitoring."""
    items = queue.list_items(limit=200)
    pending = sum(1 for i in items if i.status == "pending")
    processing = sum(1 for i in items if i.status == "processing")
    completed = sum(1 for i in items if i.status == "completed")
    failed = sum(1 for i in items if i.status == "failed")
    return jsonify({
        "queue_type": "redis" if QUEUE_TYPE == "redis" else "file",
        "depth": queue.depth(),
        "pending": pending,
        "processing": processing,
        "completed": completed,
        "failed": failed,
        "total_tracked": len(items),
    })

def _validate_int(value, field_name, min_val=None, max_val=None):
    """Validate and convert a value to int, returning (error_dict, status_code) or (None, int_value)."""
    if value is None:
        return None, None
    try:
        ival = int(value)
    except (TypeError, ValueError):
        return {"error": f"{field_name} must be an integer"}, 400
    if min_val is not None and ival < min_val:
        return {"error": f"{field_name} must be between {min_val} and {max_val}"}, 400
    if max_val is not None and ival > max_val:
        return {"error": f"{field_name} must be between {min_val} and {max_val}"}, 400
    return None, ival

@app.route('/generate', methods=['POST'])
def generate():
    data = request.get_json(force=True)
    if not data:
        return jsonify({"error": "invalid JSON"}), 400

    prompt = data.get('prompt', '').strip()
    if not prompt:
        return jsonify({"error": "prompt is required"}), 400

    mood = data.get('mood', 'neutral')
    key = data.get('key', 'C')
    length_raw = data.get('length', 30)
    tempo_raw = data.get('tempo', 120)
    seed_raw = data.get('seed')  # None if omitted

    # Validate length
    err, length = _validate_int(length_raw, "length", 1, 300)
    if err:
        return jsonify(err), 400
    if length is None:
        return jsonify({"error": "length is required"}), 400

    # Validate tempo
    err, tempo = _validate_int(tempo_raw, "tempo", 40, 300)
    if err:
        return jsonify(err), 400
    if tempo is None:
        return jsonify({"error": "tempo is required"}), 400

    # Validate seed if provided: must be an integer in [0, 2^32-1]
    seed = None
    if seed_raw is not None:
        err, seed = _validate_int(seed_raw, "seed", 0, 2**32 - 1)
        if err:
            return jsonify(err), 400

    job_id = str(uuid.uuid4())
    created_at = time.time()

    # Generate and persist a seed when omitted — must never stay null
    if seed is None:
        seed = random.randint(0, 2**31 - 1)

    item = QueueItem(
        job_id=job_id,
        prompt=prompt,
        mood=mood,
        tempo=tempo,
        key=key,
        length=length,
        seed=seed,
        status='pending',
        result={},
        created_at=created_at,
        queued_at=created_at,
    )

    queue.enqueue(item)

    return jsonify({
        'job_id': job_id,
        'status': 'queued',
        'created_at': created_at,
        'seed': seed
    }), 202

@app.route('/status/<job_id>', methods=['GET'])
def status(job_id):
    item = queue.get_item(job_id)
    if item is None:
        return jsonify({'error': 'not found'}), 404
    return jsonify(item.to_dict())

@app.route('/artifact/<job_id>', methods=['GET'])
def artifact(job_id):
    item = queue.get_item(job_id)
    if item is None:
        return jsonify({'error': 'not found'}), 404

    result = item.result.get('output_file')
    if not result or not Path(result).exists():
        return jsonify({'error': 'artifact not found'}), 404

    if result.endswith('.wav'):
        return send_file(result, mimetype='audio/wav')
    else:
        return send_file(result, mimetype='text/plain')

if __name__ == '__main__':
    app.run(host=HOST, port=PORT, debug=False)