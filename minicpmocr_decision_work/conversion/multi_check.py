#!/usr/bin/env python3
"""Is the multi-signature file the same computation as the single-bucket files? One option at a time, the same option on both sides:
every signature's whole output against its single file on the same input, bit-equal or max|d|; then the first record of every signature again in reverse order on the same
CompiledModel (no state between signatures); and the multi file's own answer against the author's fixtures, the host not in the path.

    <python> conversion/multi_check.py <work> decide --multi out/graphs/multi_decide_phone_fp16fc.tflite --acc gpu_f32 --sharing
    <python> conversion/multi_check.py <work> tower  --multi out/graphs/multi_tower_phone_fp16fc.tflite --acc cpu --weight-cache

Inputs: per signature the largest fixture that fits and the smallest one (out/fix). CPU with --weight-cache: one new XNNPACK cache file per graph file, under a directory this
process removes at the end (the runtime reuses a stale cache unchecked). Writes results/multi_check_<what>_<acc>[_sharing][_wcache][suffix].json.
Never overwrites. The multi file loads first, so its compile's peak is the process's lifetime maximum read right after the load."""
import argparse
import json
import re
import shutil
import sys
import time
from pathlib import Path

import numpy as np


class MultiFile:
    """One CompiledModel over a multi-signature file; buffers per signature on first use and kept."""

    def __init__(self, path, acc, threads, sharing, weight_cache=""):
        from ai_edge_litert.compiled_model import CompiledModel
        from decide_host.litert_backends import options

        t0 = time.perf_counter()
        self.model = CompiledModel.from_file(str(path), options=options(acc, threads, sharing, weight_cache))
        self.load_s = time.perf_counter() - t0
        self.sigs = self.model.get_signature_list()
        self.bufs = {}
        self.last_ms = None

    def __call__(self, sig, feed, out_name):
        if sig not in self.bufs:
            s = self.sigs[sig]
            self.bufs[sig] = ({n: self.model.create_input_buffer_by_name(sig, n) for n in s["inputs"]}, {n: self.model.create_output_buffer_by_name(sig, n) for n in s["outputs"]})
        ins, outs = self.bufs[sig]
        for n, b in ins.items():
            b.write(np.ascontiguousarray(feed[n], dtype=np.float32))
        t0 = time.perf_counter()
        self.model.run_by_name(sig, ins, outs)
        self.last_ms = (time.perf_counter() - t0) * 1000
        shape = [int(x) for x in self.model.get_output_tensor_details(sig)[out_name]["shape"]]
        return np.asarray(outs[out_name].read(int(np.prod(shape)), np.float32), dtype=np.float32).reshape(shape).copy()

    def close(self):
        for ins, outs in self.bufs.values():
            for b in list(ins.values()) + list(outs.values()):
                try:
                    b.destroy()
                except Exception:
                    pass
        self.bufs = {}
        self.model.close()


def diff(a, b, real):
    """a, b [1, rows, d]; real = rows that carry an answer."""
    d = np.abs(a.astype(np.float64) - b.astype(np.float64))
    w = b[0, :real].astype(np.float64)
    return {"bit_equal": bool(np.array_equal(a, b)), "bit_equal_real": bool(np.array_equal(a[0, :real], b[0, :real])), "max_abs_real": float(d[0, :real].max()),
            "rel_rms_real": float(np.linalg.norm(a[0, :real].astype(np.float64) - w) / max(np.linalg.norm(w), 1e-300)), "finite": bool(np.isfinite(a).all()) and bool(np.isfinite(b).all())}


def compare(multi, singles, feeds, out_name, acc, threads, sharing, cache_dir, score=None):
    """singles = {sig: (file, signature in that file)}; feeds = {sig: [(record, real rows, feed)]}; score(sig, record, output) -> dict for the multi file's own answer."""
    import lrt
    from decide_host.litert_backends import Graph

    doc = {"acc": acc, "threads": threads, "sharing": sharing, "weight_cache": bool(cache_dir), "signatures": {}}
    wc = lambda p: str(Path(cache_dir) / (Path(p).stem + ".xnnpack_cache")) if cache_dir else ""   # noqa: E731
    m = MultiFile(multi, acc, threads, sharing, wc(multi))
    doc["multi"] = {"file": Path(multi).name, "bytes": Path(multi).stat().st_size, "load_s": round(m.load_s, 2), "memory_after_load": lrt.memory()}
    first = {}
    for sig in singles:
        rows = []
        for rec, real, feed in feeds[sig]:
            y = m(sig, feed, out_name)
            first[(sig, rec)] = y
            r = {"record": rec, "rows": real, "multi_ms": round(m.last_ms, 1)}
            if score:
                r.update(score(sig, rec, y))
            rows.append(r)
            print(json.dumps({"multi": sig, **{k: (float(f"{v:.4g}") if isinstance(v, float) else v) for k, v in r.items()}}), flush=True)
        doc["signatures"][sig] = {"records": rows}
    for sig in reversed(list(singles)):
        rec, real, feed = feeds[sig][0]
        doc["signatures"][sig]["again_in_reverse"] = diff(m(sig, feed, out_name), first[(sig, rec)], real)
    doc["multi"]["memory_after_runs"] = lrt.memory()
    m.close()
    worst = 0.0
    all_bit = True
    for sig, (path, name) in singles.items():
        g = Graph(path, name, acc, threads, sharing, wc(path))
        doc["signatures"][sig].update(single={"file": Path(path).name, "bytes": Path(path).stat().st_size, "load_s": round(g.load_s, 2)})
        for r, (rec, real, feed) in zip(doc["signatures"][sig]["records"], feeds[sig]):
            y = g(**feed)[out_name]
            c = diff(first[(sig, rec)], y, real)
            r.update(single_ms=round(g.last_ms, 1), **c)
            worst = max(worst, c["max_abs_real"])
            all_bit = all_bit and c["bit_equal"]
            print(json.dumps({"single": sig, "record": rec, "bit_equal": c["bit_equal"], "bit_equal_real": c["bit_equal_real"], "max_abs_real": float(f"{c['max_abs_real']:.4g}"), "ms": r["single_ms"]}), flush=True)
        g.close()
    doc["summary"] = {"all_bit_equal": all_bit, "max_abs_real": worst, "again_all_bit_equal": all(v["again_in_reverse"]["bit_equal"] for v in doc["signatures"].values())}
    return doc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("work", type=Path, help="the work directory")
    ap.add_argument("what", choices=("decide", "tower"))
    ap.add_argument("--multi", required=True)
    ap.add_argument("--acc", default="gpu_f32")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--sharing", action="store_true")
    ap.add_argument("--weight-cache", action="store_true")
    ap.add_argument("--graphs", default="out/graphs")
    ap.add_argument("--form", default="fp16fc")
    ap.add_argument("--suffix", default="")
    a = ap.parse_args()
    WORK = a.work.resolve()
    sys.path.insert(0, str(WORK / "host"))
    sys.path.insert(0, str(WORK / "conversion"))
    sys.path.insert(0, str(WORK / "conversion"))
    from decide_host import formulas as F
    from fc_buffers import buffers
    import lrt

    tag = f"{a.what}_{a.acc}" + ("_sharing" if a.sharing else "") + ("_wcache" if a.weight_cache else "") + a.suffix
    out = WORK / f"results/multi_check_{tag}.json"
    assert not out.exists(), f"never overwrite {out}"
    multi = WORK / a.multi
    sigs = buffers(str(multi))["signatures"]
    singles, feeds, fixes = {}, {}, {}
    fix_dir = WORK / "out/fix"
    if a.what == "decide":
        W = F.Bf16Table(WORK / "snapshot/model.safetensors", "score.weight").rows()
        for p in sorted(fix_dir.glob("rec_*.npz")):
            z = np.load(p)
            fixes[p.stem[4:]] = (int(z["ids"].shape[0]), z)
    else:
        for p in sorted(fix_dir.glob("pic_*.npz")):
            z = np.load(p)
            fixes[p.stem[4:]] = (int(z["pixel_values"].shape[0]), z)

    def decide_feed(z, L):
        T = z["ids"].shape[0]
        e = np.zeros((1, L, 2048), np.float32)
        e[0, :T] = z["inputs_embeds"]
        v = np.zeros((1, L), np.float32)
        v[0, :T] = 1.0
        c, s = F.mrope_tables(F.padded_planes(z["pos3"].astype(np.int64), L))
        c[0, :T], s[0, :T] = z["cos"], z["sin"]
        return {"embeds": e, "valid": v, "cos": c, "sin": s}

    def tower_feed(z, N):
        n = z["pixel_values"].shape[0]
        f = {"patches": np.zeros((1, N, 1536), np.float32), "valid": np.zeros((1, N), np.float32), "pos": np.zeros((1, N, 1024), np.float32),
             "cos": np.ones((1, N, 64), np.float32), "sin": np.zeros((1, N, 64), np.float32)}
        f["patches"][0, :n] = z["pixel_values"]
        f["valid"][0, :n] = 1.0
        f["pos"][0, :n] = z["pos"]
        f["cos"][0, :n], f["sin"][0, :n] = z["cos"], z["sin"]
        return f

    for sig in sigs:
        mt = re.fullmatch(r"(?:decide|tower)_([LN])(\d+)(_qb\d+)?", sig)
        size = int(mt.group(2))
        singles[sig] = (WORK / a.graphs / f"{a.what}_{mt.group(1)}{size}{mt.group(3) or ''}_{a.form}.tflite", a.what)
        fit = sorted((n, name) for name, (n, _) in fixes.items() if n <= size)
        assert fit, f"no fixture fits {sig}"
        picks = [fit[-1], fit[0]] if fit[-1] != fit[0] else [fit[-1]]
        feeds[sig] = []
        for n, name in picks:
            z = fixes[name][1]
            if a.what == "decide":
                feeds[sig].append((name, n, decide_feed(z, size)))
            else:
                feeds[sig].append((name, n // 4, tower_feed(z, size)))

    def score(sig, rec, y):
        n, z = fixes[rec]
        if a.what == "decide":
            logits, p1 = F.softmax_prob1(y[0, n - 1], W)
            return {"prob_1": p1, "d_prob_author": abs(p1 - float(z["prob_1"]))}
        return {"rel_rms_author": lrt.stats(y[0, :n // 4], z["rows"])["rel_rms"]}

    cache_dir = ""
    if a.weight_cache:
        cache_dir = str(WORK / f"out/wcache/multi_check_{tag}")
        shutil.rmtree(cache_dir, ignore_errors=True)
        Path(cache_dir).mkdir(parents=True)
    doc = {"what": a.what, "multi": a.multi, "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), **compare(multi, singles, feeds, "hidden" if a.what == "decide" else "rows", a.acc, a.threads, a.sharing, cache_dir, score)}
    doc["memory_end"] = lrt.memory()
    doc["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    out.write_text(json.dumps(doc, indent=1) + "\n")
    if cache_dir:
        shutil.rmtree(cache_dir, ignore_errors=True)
    print("MULTI_CHECK_DONE " + json.dumps({"tag": tag, **doc["summary"], "load_s": doc["multi"]["load_s"], "footprint_max_after_load_gib": round(doc["multi"]["memory_after_load"].get("lifetime_max_phys_footprint", 0) / 2**30, 2)}), flush=True)


if __name__ == "__main__":
    main()
