"""The clock-cap record of one NPU-runner run on the phone, from the host's samples (<dir>/<tag>.samples.txt: blocks
"=== HH:MM:SS uptime <s>" with "kgsl_max <MHz> pwrlevel <n>", "cpucap <policy> <scaling_max> <cpuinfo_max>",
MemAvailable and the runner's VmHWM, and now and then the thermal status; android/measure/README.md has the sampler)
and the runner's wall-clock stamps (stdout: latency_start / done wall_ms, the phone's CLOCK_REALTIME; the phone's local
time is the host's).

    python3 r17_leg_caps.py --tag <tag>   -> <dir>/<tag>.caps.json + one line      (dir = $R17_REL, default device/r17)

A run's latency phase (the samples between latency_start and done, the timed calls) is "cap" when a sample in it shows
the kgsl max clock below 1,300 MHz, any CPU policy capped, or a thermal status above 0; the whole run is reported as
well."""
import argparse
import datetime
import json
import os
import re
from pathlib import Path

K = Path(__file__).resolve().parents[1]
R = K / os.environ.get("R17_REL", "device/r17")


def blocks(path):
    out, cur = [], None
    for line in path.read_text(errors="replace").splitlines():
        if m := re.match(r"=== (\d\d:\d\d:\d\d) uptime ([\d.]+)", line):
            cur = {"t": m.group(1), "uptime": float(m.group(2)), "cpucap": {}}
            out.append(cur)
        elif cur is None:
            continue
        elif m := re.match(r"kgsl_max (\d+) pwrlevel (\d+)", line):
            cur["kgsl_max"], cur["pwrlevel"] = int(m.group(1)), int(m.group(2))
        elif m := re.match(r"cpucap (\S+) (\d+) (\d+)", line):
            cur["cpucap"][m.group(1)] = (int(m.group(2)), int(m.group(3)))
        elif m := re.search(r"Thermal Status: (\d+)", line):
            cur["thermal"] = int(m.group(1))
        elif m := re.match(r"MemAvailable:\s+(\d+)", line):
            cur["mem_avail_kb"] = int(m.group(1))
        elif m := re.match(r"VmHWM:\s+(\d+)", line):
            cur["vmhwm_kb"] = int(m.group(1))
    return out


def summarize(bs):
    if not bs:
        return None
    kg = [b["kgsl_max"] for b in bs if "kgsl_max" in b]
    capped = sorted({f"{p}:{v[0]}/{v[1]}" for b in bs for p, v in b["cpucap"].items() if v[0] != v[1]})
    th = [b["thermal"] for b in bs if "thermal" in b]
    return {"samples": len(bs), "first": bs[0]["t"], "last": bs[-1]["t"], "kgsl_max_min": min(kg) if kg else None,
            "pwrlevel_max": max((b.get("pwrlevel", 0) for b in bs), default=None), "cpu_caps": capped,
            "thermal_max": max(th) if th else None, "vmhwm_kb_max": max((b.get("vmhwm_kb", 0) for b in bs), default=None),
            "mem_avail_kb_min": min((b["mem_avail_kb"] for b in bs if "mem_avail_kb" in b), default=None)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    bs = blocks(R / f"{a.tag}.samples.txt")
    text = (R / f"{a.tag}.stdout.txt").read_text(errors="replace")
    walls = {k: int(v) for k, v in re.findall(r"(latency_start|done) wall_ms=(\d+)", text)}
    lat = None
    if "latency_start" in walls and "done" in walls:
        t0 = datetime.datetime.fromtimestamp(walls["latency_start"] / 1000).strftime("%H:%M:%S")
        t1 = datetime.datetime.fromtimestamp(walls["done"] / 1000 + 2).strftime("%H:%M:%S")
        lat = summarize([b for b in bs if t0 <= b["t"] <= t1]) or {"samples": 0, "window": [t0, t1]}
        lat["window"] = [t0, t1]
    ready = (R / f"{a.tag}.ready.txt").read_text().strip() if (R / f"{a.tag}.ready.txt").exists() else None
    whole = summarize(bs)
    cap = None
    if lat and lat.get("samples"):
        cap = bool((lat["kgsl_max_min"] or 1300) < 1300 or lat["cpu_caps"] or (lat["thermal_max"] or 0) > 0)
    doc = {"tag": a.tag, "ready_gate": ready, "latency_phase": lat, "whole_leg": whole,
           "timing_cap": cap, "rule": "cap = kgsl max clock < 1300 MHz, any CPU policy capped, or thermal status > 0 "
                                      "in a sample inside the latency phase (samples every ~2-3 s)"}
    (R / f"{a.tag}.caps.json").write_text(json.dumps(doc, indent=1) + "\n")
    print(json.dumps({"tag": a.tag, "timing_cap": cap, "latency_phase": lat, "whole_kgsl_min": (whole or {}).get("kgsl_max_min"),
                      "whole_cpu_caps": (whole or {}).get("cpu_caps"), "vmhwm_kb_max": (whole or {}).get("vmhwm_kb_max")}))


if __name__ == "__main__":
    main()
