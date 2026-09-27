#!/usr/bin/env python3
"""Round-4 speed numbers from the primary logs (type = vibevoice_asr_streaming_work/fill_numbers.py; nothing typed in).

Mac   bench/mac_{cpu,gpu}_p{256,512}.log   litert-lm 0.17.1 `benchmark` output + /usr/bin/time + load lines
      mac_gate_r12_mac_e2e_{cpu,gpu}.json   end-to-end per-clip walls (engine loaded once)
      bench/mac_encoder.json                encoder alone, 20 warm invokes
S26   bench/s26_{cpu,gpu}.log              v0.16.1 CLI --benchmark (must say Processed 256 for prefill and decode)
      bench/s26_bench_state.json            readiness, exits, load-only runs
      s26_gate_s26_r3_{cpu,gpu}_cpu.json    round-3 per-clip process walls (engine load included) and peak lines

Derived S26 per-clip time (round 3 ran one process per clip, so every wall includes the engine load):
      per_clip_i = wall_i - median(load_only host walls)       (load-only = same flags, --multi_turns=true, empty line)
      median per clip = median_i(per_clip_i); RTF = sum_i(per_clip_i) / sum_i(audio_i)
The decode figure of every Mac cell is checked against the cell's wall time: the tool runs 4 engines (warmup + 3
iterations), each doing init + prefill + decode, so wall >= 4 * 256 / decode_tps must hold (else the cell is flagged).

  python3 bench_numbers.py       -> bench/mac_numbers.json, bench/s26_numbers.json (a missing input = that part absent)
"""
import json
import os
import re
import statistics

HERE = os.path.dirname(os.path.abspath(__file__))
B = os.path.join(HERE, "bench")


def load_note(log):
    m = re.search(r"^(LOAD_OK|CONTENDED) load1=([\d.]+)", log, re.M)
    return (m.group(1), float(m.group(2))) if m else (None, None)


def busiest(log, which):
    blk = re.search(rf"^# {which} busiest processes.*?\n((?:#   .*\n)+)", log, re.M)
    rows = []
    for ln in (blk.group(1).splitlines() if blk else []):
        parts = ln[1:].split()
        if len(parts) >= 3 and parts[0].isdigit():
            rows.append({"pid": int(parts[0]), "command": " ".join(parts[1:-1]), "cpu": float(parts[-1])})
    return rows


RESIDENT = {"dasd", "Finder", "launchd", "Storage", "PerfPowerService", "appstoreagent", "WindowServer", "kernel_task",
            "mds_stores", "fseventsd", "notifyd", "mds", "top", "logd", "coreaudiod", "sysmond", "mediaanalysisd",
            "photoanalysisd", "WallpaperAerials", "mdworker_shared", "spotlightknowledged", "trustd", "cloudd", "bird",
            "siriactionsd", "BackgroundShortc", "WallpaperImageEx"}


def load_owner(rows, min_cpu=10.0):
    """Split the busy processes (>= min_cpu %) of a snapshot into long-running system processes and everything else."""
    busy = [r for r in rows if r["cpu"] >= min_cpu]
    res = [f"{r['command']} {r['cpu']:.0f}%" for r in busy if r["command"] in RESIDENT]
    oth = [f"{r['command']} {r['cpu']:.0f}%" for r in busy if r["command"] not in RESIDENT]
    return {"resident": res, "other": oth}


def uptime_line(log, which):
    m = re.search(rf"^# {which} (\S+ \S+) \| (.*)$", log, re.M)
    return {"time": m.group(1), "uptime": m.group(2)} if m else None


def mac_bench(be, p):
    path = os.path.join(B, f"mac_{be}_p{p}.log")
    if not os.path.exists(path):
        return None
    s = open(path).read()
    g = lambda rx: (lambda m: float(m.group(1)) if m else None)(re.search(rx, s))  # noqa: E731
    r = {"log": os.path.relpath(path, HERE), "backend": be, "prefill_tokens": p, "decode_tokens": 256,
         "prefill_tps": g(r"Prefill speed:\s+([\d.]+) tokens/s"), "decode_tps": g(r"Decode speed:\s+([\d.]+) tokens/s"),
         "init_s": g(r"Init time:\s+([\d.]+) s"), "ttft_s": g(r"Time to first token:\s+([\d.]+) s"),
         "wall_s": g(r"([\d.]+) real"), "max_rss_bytes": g(r"(\d+)\s+maximum resident set size"),
         "max_num_tokens_line": (re.search(r"Max number of tokens\s*:\s*(\d+)", s) or [None, None])[1],
         "exit": (re.search(r"EXIT=(\d+)", s) or [None, None])[1]}
    r["load_state"], r["load1_at_start"] = load_note(s)
    r["before"], r["after"] = uptime_line(s, "before"), uptime_line(s, "after")
    r["busiest_before"], r["busiest_after"] = busiest(s, "before"), busiest(s, "after")
    r["load_owner_before"] = load_owner(r["busiest_before"])
    ok = all(r[k] is not None for k in ("prefill_tps", "decode_tps", "init_s", "ttft_s", "wall_s")) and r["exit"] == "0"
    if ok:
        r["decode_wall_floor_s"] = round(4 * 256 / r["decode_tps"], 2)
        r["decode_consistent_with_wall"] = r["wall_s"] >= r["decode_wall_floor_s"]
    r["valid"] = bool(ok and r.get("decode_consistent_with_wall"))
    return r


def mac_e2e(be):
    path = os.path.join(HERE, f"mac_gate_r12_mac_e2e_{be}.json")
    if not os.path.exists(path):
        return None
    d = json.load(open(path))
    log = open(os.path.join(HERE, f"mac_gate_r12_mac_e2e_{be}.log")).read()
    walls = [r["wall_s"] for r in d["rows"]]
    audio = sum(r["audio_s"] for r in d["rows"])
    r = {"json": os.path.basename(path), "backend": be, "audio_backend": d["audio_backend"], "n": len(walls),
         "audio_s": round(audio, 3), "per_clip_median_s": round(statistics.median(walls), 3),
         "per_clip_min_s": min(walls), "per_clip_max_s": max(walls), "wall_sum_s": round(sum(walls), 3),
         "rtf": round(sum(walls) / audio, 4), "engine_load_s": d["summary"]["engine_load_s"],
         "first_clip_wall_s": d["summary"]["first_clip_wall_s"],
         "cache_dir": d["cache_dir"], "cache_files_before": len(d["cache_files_before"]),
         "oracle_match": d["summary"]["oracle_match"], "wer_en": d["summary"]["wer_en"],
         "compare_file": d["summary"].get("compare_file"), "compare_equal": d["summary"].get("compare_equal"),
         "empty": d["summary"]["empty"], "loop_suspect": d["summary"]["loop_suspect"],
         "bundle_sha256": d["bundle_sha256"], "litert_lm": d["litert_lm"], "threads": d["threads"]}
    r["load_state"], r["load1_at_start"] = load_note(log)
    r["busiest_before"], r["busiest_after"] = busiest(log, "before"), busiest(log, "after")
    r["load_owner_before"] = load_owner(r["busiest_before"])
    return r


def mac():
    out = {"bench": {}, "e2e": {}, "encoder": None}
    for be in ("cpu", "gpu"):
        for p in (256, 512):
            r = mac_bench(be, p)
            if r:
                out["bench"][f"{be}_p{p}"] = r
        e = mac_e2e(be)
        if e:
            out["e2e"][be] = e
    enc = os.path.join(B, "mac_encoder.json")
    if os.path.exists(enc):
        d = json.load(open(enc))
        log = open(os.path.join(B, "mac_encoder.log")).read()
        d["load_state"], d["load1_at_start"] = load_note(log)
        d["busiest_before"] = busiest(log, "before")
        d["load_owner_before"] = load_owner(d["busiest_before"])
        out["encoder"] = d
    return out


def s26():
    st_path = os.path.join(B, "s26_bench_state.json")
    if not os.path.exists(st_path):
        return None
    st = json.load(open(st_path))
    out = {"device": st["device"], "os": st.get("os"), "bin_label": st["bin_label"], "legs": {}}
    for be, leg in st["legs"].items():
        s = open(os.path.join(B, f"s26_{be}.log")).read()
        pre = re.search(r"Prefill Turn 1: Processed (\d+) tokens in ([\d.]+)(ms|s) duration.*?Prefill Speed: ([\d.]+)", s, re.S)
        dec = re.search(r"Decode Turn 1: Processed (\d+) tokens in ([\d.]+)(ms|s) duration.*?Decode Speed: ([\d.]+)", s, re.S)
        ttft = re.search(r"Time to first token: ([\d.]+) s", s)
        init = re.search(r"Init Total: ([\d.]+) ms", s)
        r = {"log": f"bench/s26_{be}.log", "ready": leg["ready"], "ready_line": open(os.path.join(B, f"s26_{be}_ready.txt")).readline().strip(),
             "exit": leg["exit"], "prefill_processed": int(pre.group(1)) if pre else None,
             "decode_processed": int(dec.group(1)) if dec else None,
             "prefill_tps": float(pre.group(4)) if pre else None, "decode_tps": float(dec.group(4)) if dec else None,
             "ttft_s": float(ttft.group(1)) if ttft else None, "init_total_ms": float(init.group(1)) if init else None,
             "uptime": leg["uptime"], "state_after": {k: leg["state_after"].get(k) for k in ("skin_c", "thermal_status", "cpufreq_max_vs_cpuinfo")}}
        r["valid"] = r["prefill_processed"] == 256 and r["decode_processed"] == 256 and r["exit"] == "0"
        loads = [L["host_wall_s"] for L in leg["load_only"] if L["exit"] == "0"]
        r["load_only_host_s"] = loads
        r["load_only_device_s"] = [L["device_s"] for L in leg["load_only"]]
        if len(loads) == 3:
            lo = statistics.median(loads)
            g = json.load(open(os.path.join(HERE, f"s26_gate_s26_r3_{be}_cpu.json")))
            rows = [x for x in g["rows"] if "error" not in x]
            per = [x["wall_s"] - lo for x in rows]
            audio = sum(x["audio_s"] for x in rows)
            r["per_clip"] = {"gate_json": f"s26_gate_s26_r3_{be}_cpu.json", "n": len(rows),
                             "load_only_median_s": round(lo, 3), "wall_median_s": statistics.median(x["wall_s"] for x in rows),
                             "per_clip_median_s": round(statistics.median(per), 2), "per_clip_min_s": round(min(per), 2),
                             "per_clip_max_s": round(max(per), 2), "audio_s": round(audio, 3),
                             "rtf": round(sum(per) / audio, 4),
                             "formula": "per_clip_i = round-3 process wall_i - median(3 load-only host walls); "
                                        "RTF = sum(per_clip_i) / sum(audio_i)"}
        out["legs"][be] = r
    return out


def main():
    m = mac()
    json.dump(m, open(os.path.join(B, "mac_numbers.json"), "w"), indent=1)
    print("mac:", json.dumps({k: {kk: v.get(kk) for kk in ("prefill_tps", "decode_tps", "ttft_s", "init_s", "wall_s", "load1_at_start", "valid")}
                              for k, v in m["bench"].items()}))
    print("mac e2e:", json.dumps({k: {kk: v[kk] for kk in ("per_clip_median_s", "rtf", "engine_load_s", "oracle_match", "compare_equal", "load1_at_start")}
                                  for k, v in m["e2e"].items()}))
    if m["encoder"]:
        print("mac encoder:", {k: m["encoder"][k] for k in ("first_invoke_ms", "warm_median_ms", "load1_at_start")})
    s = s26()
    if s is not None:
        json.dump(s, open(os.path.join(B, "s26_numbers.json"), "w"), indent=1)
        print("s26:", json.dumps({k: {kk: v.get(kk) for kk in ("prefill_tps", "decode_tps", "ttft_s", "valid", "ready")}
                                  for k, v in s["legs"].items()}))
        print("s26 per clip:", json.dumps({k: v.get("per_clip", {}).get("per_clip_median_s") for k, v in s["legs"].items()}))
    else:
        print("s26: no bench/s26_bench_state.json (legs not run)")


if __name__ == "__main__":
    main()
