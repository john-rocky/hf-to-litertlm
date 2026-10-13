#!/usr/bin/env python3
"""Round 3 of the Qwen3-ASR-1.7B speed ladder: one 155-clip runtime jsonl (runtime_gate_r3.py) -> the GATE.md G4
verdict as JSON (stdlib only; the rates come from the lane's pinned scorer score_fleurs.py, imported, not re-written).

G4 (GATE.md, reference = eager fp32 with the litert-torch prompt, gates/ref_eager_litert.jsonl):
  per language   rate <= reference rate + 1.0 pt (en_us WER, cmn_hans_cn / ja_jp CER; score_fleurs.agg_rows)
  empty          empty transcripts per language <= the reference's
  catastrophic   no clip with error > 50 % of its reference length where the reference is <= 50 %
  render         every row's runtime render = common.LITERT_TORCH_PROMPT (audio-only message), 0 runtime errors,
                 155 rows; --render_check adds the 4-case render_check_r3.py result
Also: wall / RTF from the rows' own seconds, with the window guard's pauses (speed/r3/logs/<label>.guard.log, PAUSE /
RESUME epochs) taken out of the clips they overlap; per-clip table; --split_sec S adds the rates for clips <= S and > S
seconds; --pair <jsonl> counts raw-string equality against another run (the same clips).

  cd qwen3_asr17_work && python3 speed/scripts/parity_r3.py --hyp speed/r3/rt_ship_cpu.jsonl --form ship --backend cpu \
      --guard_log speed/r3/logs/ship_cpu.guard.log --out speed/r3/parity_ship_cpu.json
"""
import argparse
import hashlib
import json
import os
import statistics
import sys

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # qwen3_asr17_work
sys.path.insert(0, HERE)
import common  # noqa: E402
import score_fleurs  # noqa: E402

CFGS = ("en_us", "cmn_hans_cn", "ja_jp")
REF = os.path.join(HERE, "gates", "ref_eager_litert.jsonl")
REF_RATES = {"en_us": 4.348, "cmn_hans_cn": 6.039, "ja_jp": 5.244}  # GATE.md "Change 11:06" (10-03)
BAR_PT = 1.0


def jl(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def score(spec, names=None):
    """score_fleurs.py `ref` for one hypothesis file: the same pairs (every FLEURS clip of meta.json, a missing clip
    scores as ''), optionally restricted to `names`. -> (summary per config, per-clip rows, missing)."""
    m = score_fleurs.meta()
    path, field, hyp = score_fleurs.load_hyp(spec)
    pairs = [(n, m[n]["reference_text"]["raw_transcription"], hyp.get(n, ""))
             for n in m if m[n]["path"].startswith("fleurs/") and (names is None or n in names)]
    missing = [n for n, _, _ in pairs if n not in hyp]
    agg, rows = score_fleurs.agg_rows(pairs, m)
    return agg, rows, missing


def pauses(guard_log):
    """[(start, end)] epochs of the guard's SIGSTOP windows (an unclosed one ends at +inf)."""
    out, start = [], None
    if not guard_log or not os.path.exists(guard_log):
        return out
    for line in open(guard_log, encoding="utf-8"):
        parts = line.split()
        if len(parts) > 3 and parts[1] == "epoch" and parts[3] in ("PAUSE", "RESUME"):
            t = float(parts[2])
            if parts[3] == "PAUSE":
                start = t
            elif start is not None:
                out.append((start, t))
                start = None
    if start is not None:
        out.append((start, float("inf")))
    return out


def overlap(a, b, wins):
    return sum(max(0.0, min(b, e) - max(a, s)) for s, e in wins)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hyp", required=True, help="runtime_gate_r3.py jsonl")
    ap.add_argument("--form", required=True)
    ap.add_argument("--backend", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--guard_log", default="")
    ap.add_argument("--render_check", default="", help="render_check_r3.py JSON for this bundle + runtime (optional)")
    ap.add_argument("--split_sec", type=float, default=0.0)
    ap.add_argument("--pair", action="append", default=[], help="another run's jsonl: raw-string equality per clip")
    ap.add_argument("--engine_log", default="", help="the run's stderr log: its ERROR lines (unique, counted) are copied in")
    args = ap.parse_args()

    recs = jl(args.hyp)
    header = next((r for r in recs if r.get("type") == "header"), {})
    footer = next((r for r in recs if r.get("type") == "footer"), {})
    rows = [r for r in recs if r.get("clip")]
    by = {r["clip"]: r for r in rows}

    agg, srows, missing = score(args.hyp + ":text")
    ragg, rrows, rmissing = score(REF + ":text")
    ref_check = {c: ragg[c]["rate_pct"] for c in CFGS}
    assert ref_check == REF_RATES, (ref_check, REF_RATES)
    rerr = {r["clip"]: r for r in rrows}
    delta = {c: round(agg[c]["rate_pct"] - ragg[c]["rate_pct"], 3) for c in CFGS}
    cat = sorted(r["clip"] for r in srows if r["ref_len"] and r["err"] / r["ref_len"] > 0.5)
    ref_cat = sorted(r["clip"] for r in rrows if r["ref_len"] and r["err"] / r["ref_len"] > 0.5)
    new_cat = sorted(set(cat) - set(ref_cat))
    errors = [r["clip"] for r in rows if "error" in r]
    render_equal = sum(r.get("render") == common.LITERT_TORCH_PROMPT for r in rows)
    rc = json.load(open(args.render_check, encoding="utf-8")) if args.render_check else None

    checks = {
        "engine_created": not header.get("engine_error"),
        "rows_155": len(rows) == 155 and not missing,
        "per_language<=ref+1.0pt": all(d <= BAR_PT for d in delta.values()),
        "empty<=ref": all(agg[c]["empty_hyp"] <= ragg[c]["empty_hyp"] for c in CFGS),
        "no_new_catastrophic": not new_cat,
        "errors_0": not errors,
        "render_155": render_equal == len(rows) == 155,
    }
    if rc is not None:
        checks["render_check_all_equal"] = bool(rc.get("all_equal"))
    verdict = "PASS" if all(checks.values()) else "FAIL"

    wins = pauses(args.guard_log)
    per_clip, secs, secs_run = [], [], []
    sm = {r["clip"]: r for r in srows}
    pair_rows = {p: {r["clip"]: r for r in jl(p) if r.get("clip")} for p in args.pair}
    for r in rows:
        p = overlap(r["t_start"], r["t_end"], wins) if "t_start" in r and "t_end" in r else 0.0
        s = sm.get(r["clip"])
        row = {"clip": r["clip"], "config": r["config"], "audio_seconds": r["audio_seconds"],
               "seconds": r.get("seconds"), "paused_s": round(p, 3),
               "language": r.get("language"), "text": r.get("text"), "token_count": r.get("token_count"),
               "render_equal": r.get("render") == common.LITERT_TORCH_PROMPT}
        if "error" in r:
            row["error"] = r["error"]
        if s:
            ref = rerr[r["clip"]]
            row.update({"err": s["err"], "ref_len": s["ref_len"], "ref_eager_err": ref["err"],
                        "catastrophic": r["clip"] in cat, "ref_eager_text": ref["hyp"]})
        for pth, pr in pair_rows.items():
            row.setdefault("raw_equal", {})[os.path.basename(pth)] = (pr.get(r["clip"], {}).get("raw") == r.get("raw"))
        per_clip.append(row)
        if r.get("seconds") is not None:
            secs.append(r["seconds"])
            secs_run.append(r["seconds"] - p)
    audio = sum(r["audio_seconds"] for r in rows if r.get("seconds") is not None)
    wall, paused = sum(secs), sum(secs) - sum(secs_run)
    timing = {"clips_timed": len(secs), "audio_seconds": round(audio, 3), "wall_seconds": round(wall, 3),
              "rtf": round(wall / audio, 4) if audio else None,
              "paused_seconds_in_clips": round(paused, 3), "wall_seconds_run": round(wall - paused, 3),
              "rtf_run": round((wall - paused) / audio, 4) if audio else None,
              "per_clip_s": {"min": min(secs), "median": statistics.median(secs), "max": max(secs)} if secs else None,
              "per_clip_run_s": ({"min": round(min(secs_run), 4), "median": round(statistics.median(secs_run), 4),
                                  "max": round(max(secs_run), 4)} if secs_run else None),
              "guard_pauses": [[s, e] for s, e in wins], "load_seconds": header.get("load_seconds"),
              "footer": footer}
    pairs_summary = {os.path.basename(p): sum(1 for x in per_clip if x.get("raw_equal", {}).get(os.path.basename(p)))
                     for p in args.pair}

    split = None
    if args.split_sec:
        le = {r["clip"] for r in rows if r["audio_seconds"] <= args.split_sec}
        gt = {r["clip"] for r in rows if r["audio_seconds"] > args.split_sec}
        split = {}
        for tag, names in ((f"le_{args.split_sec:g}s", le), (f"gt_{args.split_sec:g}s", gt)):
            a, _, _ = score(args.hyp + ":text", names)
            b, _, _ = score(REF + ":text", names)
            split[tag] = {c: {"n": a[c]["n"], "rate_pct": a[c]["rate_pct"], "ref_rate_pct": b[c]["rate_pct"],
                              "delta_pt": round(a[c]["rate_pct"] - b[c]["rate_pct"], 3),
                              "err": a[c]["err"], "ref_err": b[c]["err"], "ref_len": a[c]["ref_len"]}
                          for c in CFGS if c in a}

    log_errors = {}
    if args.engine_log and os.path.exists(args.engine_log):
        for line in open(args.engine_log, encoding="utf-8", errors="replace"):
            t = line.strip()
            if t.startswith("ERROR") or t.startswith("└") or "Failed to" in t:
                log_errors[t[:400]] = log_errors.get(t[:400], 0) + 1
    doc = {"form": args.form, "backend": args.backend, "verdict": verdict, "checks": checks,
           "engine_log": os.path.relpath(args.engine_log, HERE) if args.engine_log else None,
           "engine_log_errors": [{"line": k, "count": v} for k, v in sorted(log_errors.items(), key=lambda kv: -kv[1])[:12]],
           "hyp": os.path.relpath(args.hyp, HERE), "hyp_sha256": sha256(args.hyp),
           "runtime": header.get("runtime"), "bundle": header.get("bundle"), "bundle_bytes": header.get("bundle_bytes"),
           "engine_error": header.get("engine_error"),
           "engine": {k: header.get(k) for k in ("backend", "audio_backend", "act", "engine_kwargs", "threads",
                                                 "cache_dir", "prompt", "sampler")},
           "reference": {"path": os.path.relpath(REF, HERE), "rates": ref_check, "empty": {c: ragg[c]["empty_hyp"] for c in CFGS},
                         "catastrophic": ref_cat},
           "rates": {c: {k: agg[c][k] for k in ("metric", "rate_pct", "err", "ref_len", "n", "empty_hyp", "exact_equal")}
                     for c in CFGS},
           "delta_vs_ref_pt": delta, "bar_pt": BAR_PT,
           "empty": {c: agg[c]["empty_hyp"] for c in CFGS},
           "catastrophic": {"hyp": cat, "new_vs_ref": new_cat},
           "errors": errors, "render": {"expected": common.LITERT_TORCH_PROMPT, "equal": render_equal, "rows": len(rows)},
           "render_check": ({"path": args.render_check, "all_equal": rc.get("all_equal"),
                             "rows": [{k: x.get(k) for k in ("case", "equal", "runtime_render", "expected",
                                                              "token_count_after_1_output_token", "expected_prompt_tokens")}
                                      for x in rc.get("rows", [])]} if rc else None),
           "languages": {c: {} for c in CFGS},
           "timing": timing, "split": split, "raw_equal_pairs": pairs_summary,
           "scorer": {"module": "score_fleurs.py (score_fleurs.agg_rows, imported)", "formulas": {
               "en_us": score_fleurs.WER_FORMULA, "cmn_hans_cn": score_fleurs.CER_FORMULA, "ja_jp": score_fleurs.CER_FORMULA}},
           "per_clip": per_clip}
    for r in rows:
        if r.get("config") in CFGS:
            doc["languages"][r["config"]][r.get("language") or ""] = doc["languages"][r["config"]].get(r.get("language") or "", 0) + 1
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(doc, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"{args.form} {args.backend} {verdict} | " + " / ".join(
        f"{c} {agg[c]['rate_pct']} (Δ {delta[c]:+.3f})" for c in CFGS) +
        f" | empty {sum(agg[c]['empty_hyp'] for c in CFGS)} | new catastrophic {len(new_cat)} | errors {len(errors)} | "
        f"render {render_equal}/{len(rows)} | wall {timing['wall_seconds']} s RTF {timing['rtf']} "
        f"(paused {timing['paused_seconds_in_clips']} s, RTF run {timing['rtf_run']})"
        + (f" | ENGINE ERROR {header.get('engine_error')[:300]}" if header.get("engine_error") else ""))
    for k, v in checks.items():
        print("  ", k, v)
    if split:
        for tag, d in split.items():
            print("  ", tag, {c: (v["n"], v["rate_pct"], v["delta_pt"]) for c, v in d.items()})
    if pairs_summary:
        print("   raw equal", pairs_summary)


if __name__ == "__main__":
    main()
