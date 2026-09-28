"""Run-local helpers for decider-2b-vision round 1 (no device access, no shared state)."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / 'out/src/decider-2b-vision'
REVISION = '863e290863655f1d6b69324d77d09ac972d21609'
MODEL_SHA256 = 'ac99c16652524d0c6a8017c987efb8aeaca5050f0bf3c7a3d92b927a57ec17ec'
GRIDS = (256, 512)
ARMS = ('author', 'g256_mrope', 'g256_pos1d', 'g512_mrope', 'g512_pos1d')
TIE_GAP = 1e-4


def write_json(path, obj, indent=1):
    path = ROOT / path
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(obj, indent=indent, ensure_ascii=False, allow_nan=False) + '\n')
    temp.replace(path)


def read_json(path):
    return json.loads((ROOT / path).read_text())


def sha256_file(path):
    with open(path, 'rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_rgb(image):
    """Hash of the decoded RGB pixels plus the size, independent of the PNG encoder."""
    image = image.convert('RGB')
    return sha256_bytes(f'{image.width}x{image.height}:'.encode() + image.tobytes())
