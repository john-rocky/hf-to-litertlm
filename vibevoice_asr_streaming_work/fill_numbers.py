#!/usr/bin/env python3
"""Fill the S26 speed placeholders in the card, the curated manifest and FINDINGS from the primary
logs (bench/s26_<file>_<backend>.log) and derive the on-device per-turn latency from the gate JSONs
((process wall - engine-load-only wall) / turns).  Refuses if a log lacks the 256/256 lines."""
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
B = os.path.join(HERE, "bench")


def bench(tag):
    s = open(os.path.join(B, f"s26_{tag}.log")).read()
    pre = re.search(r"Prefill Turn 1: Processed (\d+) tokens.*?Prefill Speed: ([\d.]+)", s, re.S)
    dec = re.search(r"Decode Turn 1: Processed (\d+) tokens.*?Decode Speed: ([\d.]+)", s, re.S)
    ttft = re.search(r"Time to first token: ([\d.]+) s", s)
    assert pre and dec and ttft, tag
    assert pre.group(1) == "256" and dec.group(1) == "256", (tag, pre.group(1), dec.group(1))
    return float(pre.group(2)), float(dec.group(2)), float(ttft.group(1))


def load_only(tag):
    s = open(os.path.join(B, f"s26_{tag}_load.txt")).read()
    return float(re.search(r"LOAD_ONLY_S=([\d.]+)", s).group(1))


vals, turn = {}, {}
for f, key in (("wi8", "WI8"), ("int4", "INT4")):
    for be in ("cpu", "gpu"):
        p, d, t = bench(f"{f}_{be}")
        vals[f"S26_{key}_{be.upper()}_PREFILL"] = f"{p:.0f}"
        vals[f"S26_{key}_{be.upper()}_DECODE"] = f"{d:.1f}"
        vals[f"S26_{key}_{be.upper()}_TTFT"] = f"{t:.2f}"
        gate = "s26_cpu_cpu" if (f == "wi8" and be == "cpu") else "s26_gpu_cpu" if f == "wi8" else f"s26_int4_{be}_cpu"
        g = json.load(open(os.path.join(HERE, f"pixel_gate_{gate}.json")))
        lo = load_only(f"{f}_{be}")
        walls = [r["wall_s"] for r in g["rows"]]
        turns = sum(r["windows"] for r in g["rows"])
        audio = sum(r["dur"] for r in g["rows"])
        per_turn = (sum(walls) - lo * len(walls)) / turns
        turn[(f, be)] = {"per_turn_s": round(per_turn, 2), "load_only_s": round(lo, 1), "turns": turns,
                         "rtf": round(per_turn * turns / audio, 3), "n_procs": len(walls)}
print(json.dumps(vals, indent=1))
print(json.dumps({f"{k[0]}_{k[1]}": v for k, v in turn.items()}, indent=1))
for (f, be), v in turn.items():
    vals[f"S26_TURN_{f.upper()}_{be.upper()}"] = f"{v['per_turn_s']:.2f} s (RTF {v['rtf']:.2f})"
    vals[f"S26_LOAD_{f.upper()}_{be.upper()}"] = f"{v['load_only_s']:.1f} s"
i4g, i4c = turn[("int4", "gpu")], turn[("int4", "cpu")]
vals["S26_TURN_SUMMARY"] = (f"about {i4g['per_turn_s']:.2f} s per 2.93 s turn with the LM on the GPU and {i4c['per_turn_s']:.2f} s on the CPU "
                            f"(int4 file; per-process wall minus a {i4g['load_only_s']:.0f}–{i4c['load_only_s']:.0f} s engine load, averaged over the 73 turns; "
                            f"int8: {turn[('wi8','gpu')]['per_turn_s']:.2f} s / {turn[('wi8','cpu')]['per_turn_s']:.2f} s)")
json.dump({"values": vals, "per_turn": {f"{k[0]}_{k[1]}": v for k, v in turn.items()}},
          open(os.path.join(HERE, "s26_numbers.json"), "w"), indent=1)
for path in (os.path.join(ROOT, "cards", "vibevoice-asr-streaming-1.5b-litert.md"),
             os.path.join(ROOT, "manifest", "curated", "litert-community__VibeVoice-ASR-Streaming-1.5B.json"),
             os.path.join(HERE, "FINDINGS.md")):
    s = open(path).read()
    for k, v in vals.items():
        s = s.replace("{{" + k + "}}", v)
    assert "{{" not in s, (path, re.findall(r"{{[A-Z0-9_]+}}", s))
    open(path, "w").write(s)
    print("filled", path)
if "manifest" in path or True:
    json.load(open(os.path.join(ROOT, "manifest", "curated", "litert-community__VibeVoice-ASR-Streaming-1.5B.json")))
    print("curated JSON parses")
