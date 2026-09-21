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
from typing import Optional

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
    depth = queue.depth()
    # depth can be int (FileQueue) or dict (RedisQueue)
    depth_val = depth if isinstance(depth, int) else depth.get("total_active", 0)
    return jsonify({
        "status": "ok",
        "queue": queue_type,
        "backend": backend_type,
        "queue_depth": depth_val,
    })

@app.route('/queue/status', methods=['GET'])
def queue_status():
    """Return queue metrics for monitoring. Consistent schema across queue types."""
    depth = queue.depth()
    # depth is a dict: {"pending": N, "processing": N, "total_active": N}
    pending = depth.get("pending", 0)
    processing = depth.get("processing", 0)
    total_active = depth.get("total_active", pending + processing)

    # Count terminal states via list_items (best effort)
    items = queue.list_items(limit=500)
    completed = sum(1 for i in items if i.status == "completed")
    failed = sum(1 for i in items if i.status == "failed")
    dead = sum(1 for i in items if i.status == "dead")

    return jsonify({
        "queue_type": "redis" if QUEUE_TYPE == "redis" else "file",
        "queue_depth": total_active,
        "pending": pending,
        "processing": processing,
        "completed": completed,
        "failed": failed,
        "dead": dead,
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

def _parse_wav_header(path: str) -> Optional[dict]:
    """Parse a WAV file header, returning None if invalid or not a WAV."""
    import struct
    try:
        with open(path, 'rb') as f:
            data = f.read(44)
        if len(data) < 44:
            return None
        if data[:4] != b'RIFF' or data[8:12] != b'WAVE':
            return None
        fmt_tag = struct.unpack_from('<H', data, 20)[0]
        channels = struct.unpack_from('<H', data, 22)[0]
        sample_rate = struct.unpack_from('<I', data, 24)[0]
        byte_rate = struct.unpack_from('<I', data, 28)[0]
        bits_per_sample = struct.unpack_from('<H', data, 34)[0]
        file_size = os.path.getsize(path)
        data_size = struct.unpack_from('<I', data, 40)[0]
        duration = data_size / byte_rate if byte_rate > 0 else 0
        return {
            "format": "wav",
            "audio_format": "PCM" if fmt_tag == 1 else f"0x{fmt_tag:04x}",
            "channels": channels,
            "sample_rate": sample_rate,
            "byte_rate": byte_rate,
            "bits_per_sample": bits_per_sample,
            "file_size": file_size,
            "data_size": data_size,
            "duration_sec": round(duration, 3),
        }
    except (OSError, struct.error):
        return None


@app.route('/status/<job_id>', methods=['GET'])
def status(job_id):
    item = queue.get_item(job_id)
    if item is None:
        return jsonify({'error': 'not found'}), 404
    resp = item.to_dict()
    # Add WAV header info for completed jobs with a WAV artifact
    output_file = item.result.get('output_file') if item.result else None
    if item.status == 'completed' and output_file and output_file.endswith('.wav'):
        wav_info = _parse_wav_header(output_file)
        if wav_info:
            resp['wav_header'] = wav_info
    return jsonify(resp)

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