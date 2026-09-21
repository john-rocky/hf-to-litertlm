"""Fetch the public test JSONL and require the measured dataset fingerprint."""
import argparse
import hashlib
from pathlib import Path
import urllib.request

SHA256 = '3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, default=Path('evaldata/gsm8k_test.jsonl'))
    p.add_argument('--url', default='https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl')
    a = p.parse_args()
    if a.out.exists():
        data = a.out.read_bytes()
    else:
        with urllib.request.urlopen(a.url, timeout=60) as response:
            data = response.read(2 * 1024 * 1024)
    assert hashlib.sha256(data).hexdigest() == SHA256, 'Dataset differs from the measured protocol'
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_bytes(data)
    print('Verified', a.out, SHA256)


if __name__ == '__main__':
    main()
