"""Run-local evidence helpers; no shared state or device access."""
import datetime
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REVISION = '1ea54127d3bd52f6d753d9257b32a6380b873907'
PACKAGE_REVISION = 'c4daaac28af9fea95d627015cffa2dd5a5926ee6'
LADDER = [1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1]


def write_json(path, obj):
    path = ROOT / path
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temp.replace(path)


def read_json(path):
    return json.loads((ROOT / path).read_text())


def sha256(path):
    with open(path, 'rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def file_fact(path):
    path = Path(path)
    return dict(path=str(path.relative_to(ROOT)), bytes=path.stat().st_size, sha256=sha256(path))


def snapshot_path():
    return ROOT / read_json('results/snapshot.json')['snapshot_path']


def status(state, message):
    path = ROOT / 'STATUS.md'
    lines = path.read_text().splitlines() if path.exists() else []
    stamp = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).strftime('%Y-%m-%d %H:%M')
    path.write_text(f'STATUS: {state} {stamp} JST — {message}\n' + '\n'.join(lines[1:]) + '\n')
