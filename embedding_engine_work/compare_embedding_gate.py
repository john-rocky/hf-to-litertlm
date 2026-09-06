#!/usr/bin/env python3
"""Compare two embedding JSONs (engine vs tflite) text by text: cosine, and the 3x3 card matrix
when the texts are the verify script's 3 queries + 3 passages (first 6 entries).

    python3 scripts/compare_embedding_gate.py engine.json tflite.json [--card]
"""
import json
import sys

import numpy as np

CARD = np.array([[0.9393, 0.6899, 0.7627], [0.6780, 0.9598, 0.7062], [0.7818, 0.7342, 0.9172]])


def main():
    a = np.array(json.load(open(sys.argv[1]))["embeddings"], dtype=np.float64)
    b = np.array(json.load(open(sys.argv[2]))["embeddings"], dtype=np.float64)
    assert a.shape == b.shape, (a.shape, b.shape)
    an = a / np.linalg.norm(a, axis=1, keepdims=True)
    bn = b / np.linalg.norm(b, axis=1, keepdims=True)
    cos = (an * bn).sum(1)
    for i, c in enumerate(cos):
        print(f"  text {i}: cos {c:.6f}  |a|={np.linalg.norm(a[i]):.4f} |b|={np.linalg.norm(b[i]):.4f}")
    print(f"min cos {cos.min():.6f}  mean {cos.mean():.6f}")
    if "--card" in sys.argv and a.shape[0] >= 6:
        for name, m in (("engine", an), ("tflite", bn)):
            s = m[:3] @ m[3:6].T
            print(f"  {name} 3x3: diag {[round(float(x),4) for x in np.diag(s)]}  "
                  f"max|d| vs card {np.abs(s-CARD).max():.4f}  argmax-diag {(s.argmax(1)==np.arange(3)).all()}")
    return 0 if cos.min() > 0.999 else 1


if __name__ == "__main__":
    sys.exit(main())
