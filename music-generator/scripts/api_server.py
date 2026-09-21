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

from config import OUTPUT_DIR, QUEUE_DIR, HOST, PORT, MG_API_TOKEN
from job_queue import FileQueue, QueueItem
from backends.mock_backend import MockBackend

app = Flask(__name__)
queue = FileQueue()
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
    return jsonify({"status": "ok", "queue": "file", "backend": "mock"})

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
    length = data.get('length', 30)
    tempo = data.get('tempo', 120)
    seed = data.get('seed')  # None if omitted

    try:
        length = int(length)
    except (TypeError, ValueError):
        return jsonify({"error": "length must be an integer"}), 400
    if not (1 <= length <= 300):
        return jsonify({"error": "length must be between 1 and 300"}), 400

    try:
        tempo = int(tempo)
    except (TypeError, ValueError):
        return jsonify({"error": "tempo must be an integer"}), 400
    if not (40 <= tempo <= 300):
        return jsonify({"error": "tempo must be between 40 and 300"}), 400

    if seed is not None:
        try:
            seed = int(seed)
        except (TypeError, ValueError):
            return jsonify({"error": "seed must be an integer"}), 400

    job_id = str(uuid.uuid4())
    created_at = time.time()

    # Generate and persist a seed when omitted
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
        created_at=created_at
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