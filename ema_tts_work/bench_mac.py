"""Latency of each exported graph on the Mac CPU (XNNPACK), random inputs, median of N runs after warm-up.

  python bench_mac.py --dir out/fp32 --suffix fp32 --threads 4 --n 20
Prints ms per graph and the implied RTF of one readme2-sized piece (L 150 / T 214 -> text L256, sound L256xT384,
decoder P48 + 2 x P120 = 8.56 s of audio).
"""
import argparse, glob, statistics, time
from pathlib import Path
import numpy as np
from ema_litert import Runner

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="out/fp32")
    ap.add_argument("--suffix", default="fp32")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--n", type=int, default=20)
    a = ap.parse_args()
    res = {}
    for p in sorted(glob.glob(str(HERE / a.dir / f"*_{a.suffix}.tflite"))):
        r = Runner(p, a.threads)
        rng = np.random.default_rng(0)
        feed = {}
        for k, d in r.inputs.items():
            if k == "ids":
                feed[k] = rng.integers(2, 49, size=d["shape"]).astype(np.int32)
            elif k.startswith("m") or k in ("mask", "allow"):
                feed[k] = np.ones(d["shape"], np.float32)
            elif k == "fbias":
                feed[k] = np.zeros(d["shape"], np.float32)
            else:
                feed[k] = rng.standard_normal(size=d["shape"]).astype(np.float32)
        for _ in range(3):
            r(**feed)
        ts = []
        for _ in range(a.n):
            t0 = time.perf_counter(); r(**feed); ts.append((time.perf_counter() - t0) * 1e3)
        name = Path(p).name
        res[name] = statistics.median(ts)
        print(f"{name:36s} median {statistics.median(ts):8.2f} ms  min {min(ts):8.2f} ms  (n={a.n}, threads={a.threads})")
    return res


if __name__ == "__main__":
    main()
