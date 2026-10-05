"""Splits each phone timing run of the measurement app (modes timing and shared_timing, with per-call records) into the
calls made while neither the GPU nor the CPU clock was capped and the calls made under a cap, call by call. Inputs per
run <tag> in device/r14/: the app's report <tag>.json (timing: [device wall-clock ms at the call's start, ms];
shared_timing: [device wall-clock ms at the request's start, warmup | timed, host | direct, request ms]) and the host's
2 s samples <tag>.samples.txt ("devtime <device epoch s>", "kgsl <clock> <max_clock> <thermal_pwrlevel> <temp>",
"cpu policyN <cur> <scaling_max> <cpuinfo_max>", one block per sample; android/measure/README.md has the sampler).

A call is classified by the sample whose devtime is nearest to the call's start: burst = that sample shows the GPU's
kgsl max_clock at 1,300 MHz with thermal_pwrlevel 0 AND every CPU policy's scaling_max_freq at its cpuinfo_max_freq;
under cap = either is lowered (counted as GPU / CPU / both); a call with no sample within 3 s = unclassified. Calls after
a cap has lifted count as burst again. A request of a multi-row set is classified at its first call's start. Warm-up
calls are reported apart (they include the first call after the compile). gpu_uncapped = the calls with the GPU
uncapped whatever the CPU (burst + CPU-only capped), used by r18_crossover.py.

    python3 scripts/r18_burst.py > cache/r18/burst.md       (also writes results/r18_burst.json)"""
import json
import statistics
from pathlib import Path

K = Path(__file__).resolve().parents[1]
R = K / "device/r14"
FULL = 1300


def samples(tag):
    """[(device ms, gpu_ok, cpu_ok, kgsl max_clock)] per sample block of <tag>.samples.txt."""
    p = R / f"{tag}.samples.txt"
    if not p.exists():
        return []
    out, cur = [], None

    def flush():
        if cur and cur.get("dev") is not None and cur.get("max") is not None:
            out.append((cur["dev"], cur["max"] >= FULL and cur["lvl"] == 0, cur.get("cpu_ok", True), cur["max"]))
    for line in p.read_text(errors="replace").splitlines():
        if line.startswith("=== "):
            flush()
            cur = {}
        elif cur is None:
            continue
        elif line.startswith("devtime "):
            cur["dev"] = int(line.split()[1]) * 1000
        elif line.startswith("kgsl ") and "max" not in cur:
            cur["max"], cur["lvl"] = int(line.split()[2]), int(line.split()[3])
        elif line.startswith("cpu policy"):
            f = line.split()
            cur["cpu_ok"] = cur.get("cpu_ok", True) and int(f[3]) >= int(f[4])
    flush()
    return out


def classify(calls, smp):
    """calls [(t_start_ms, ms)] -> {burst: [ms], gpu: [ms], cpu: [ms], both: [ms], unclassified: n}"""
    res = {"burst": [], "gpu": [], "cpu": [], "both": [], "unclassified": 0}
    for t, ms in calls:
        near = min(smp, key=lambda s: abs(s[0] - t), default=None)
        if near is None or abs(near[0] - t) > 3000:
            res["unclassified"] += 1
            continue
        _, g, c, _ = near
        res["burst" if g and c else "both" if not g and not c else "gpu" if not g else "cpu"].append(ms)
    return res


def st(xs):
    return {"median": round(statistics.median(xs), 1), "n": len(xs), "min": round(min(xs), 1), "max": round(max(xs), 1)} if xs else None


def summarize(calls, smp, extra=None):
    c = classify(calls, smp)
    capped = c["gpu"] + c["cpu"] + c["both"]
    # the crossover tables use the calls with the GPU uncapped (whatever the CPU), the CPU state beside them;
    # gpu_uncapped = burst + cpu-only-capped calls
    return {**(extra or {}), "burst": st(c["burst"]), "under_cap": st(capped),
            "under_cap_by": {"gpu": len(c["gpu"]), "cpu": len(c["cpu"]), "both": len(c["both"])},
            "gpu_uncapped": st(c["burst"] + c["cpu"]), "gpu_uncapped_cpu_capped": st(c["cpu"]),
            "unclassified": c["unclassified"], "all_timed": st([ms for _, ms in calls])}


def main():
    doc = {}
    for rep in sorted(R.glob("kev_s26_r14_*.json")):
        try:
            d = json.loads(rep.read_text())
        except json.JSONDecodeError:      # a leg stopped before the app wrote its report (the 4B trial)
            continue
        if d.get("mode") not in ("timing", "shared_timing") or d.get("status") != "DONE":
            continue
        tag = rep.stem
        smp = samples(tag)
        leg = {"precision": d.get("precision"), "L": d.get("L"), "warmup_setting": d.get("warmup_calls_setting"),
               "reps": d.get("reps"), "cool_ms": d.get("cool_ms"), "gpu_state_before_compile": d.get("gpu_state_before_compile"),
               "kgsl_max_clock_min": min((s[3] for s in smp), default=None), "samples": len(smp), "sets": {}}
        for name, t in (d.get("timing") or {}).items():
            if "timed_calls" in t:          # timing mode: per call of a row set (request sets: calls in row order)
                calls = [(c[0], c[1]) for c in t["timed_calls"]]
                leg["sets"][name] = summarize(calls, smp, {"rows": t.get("rows"), "cool": t.get("cool"),
                                                         "warmup": st([w[1] for w in t.get("warmup_calls", [])])})
                if t.get("rows", 1) > 1 and calls:
                    n = t["rows"]          # also per request (the rows of one request in a row)
                    reqs = [(calls[i][0], sum(ms for _, ms in calls[i:i + n])) for i in range(0, len(calls) - n + 1, n)]
                    leg["sets"][name]["request"] = summarize(reqs, smp)
            elif "calls" in t:              # shared_timing: per request, per hand-over mode
                for mode in ("direct", "host"):
                    calls = [(c[0], c[3]) for c in t["calls"] if c[1] == "timed" and c[2] == mode]
                    leg["sets"][f"{name}/{mode}"] = summarize(calls, smp, {"questions": t.get("questions"), "cool": t.get("cool")})
        if any(("timed_calls" in t or "calls" in t) for t in (d.get("timing") or {}).values()):
            doc[tag] = leg
    (K / "results/r18_burst.json").write_text(json.dumps(doc, indent=1) + "\n")
    print("| leg | set | warm-up(n, median) | burst = GPU も CPU も最大(median / n) | cap 下(median / n; GPU・CPU・両方) | 未分類 | timed 全体の median | cool(待ち ms、方式) | leg 中 kgsl 最小 |")
    print("|---|---|---|---|---|---:|---:|---|---:|")
    for tag, leg in doc.items():
        for name, v in leg["sets"].items():
            for sub, w in (("", v), ("(request)", v.get("request"))):
                if not w:
                    continue
                wu = v.get("warmup") if not sub else None
                b, c, a = w.get("burst"), w.get("under_cap"), w.get("all_timed")
                by = w.get("under_cap_by", {})
                cool = v.get("cool") or {}
                print(f"| {tag.replace('kev_s26_r14_', '')} | {name}{sub} | {f'{wu['n']}, {wu['median']}' if wu else '—'} | "
                      f"{f'{b['median']} / {b['n']}' if b else '— / 0'} | "
                      f"{f'{c['median']} / {c['n']}' if c else '— / 0'}({by.get('gpu', 0)}・{by.get('cpu', 0)}・{by.get('both', 0)}) | "
                      f"{w.get('unclassified', 0)} | {a['median'] if a else '—'} | "
                      f"{cool.get('waited_ms', '—')}, {cool.get('mode', '—')} | {leg['kgsl_max_clock_min']} |")


if __name__ == "__main__":
    main()
