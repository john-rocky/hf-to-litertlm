"""Where does the shared-state pair start to beat the row graphs, per bucket pair and precision, from the request-point
legs (r14_device_rows_req.py and r18_req2.py inputs). Primary files only:
  S26  the request block v2 legs through results/r18_burst.json: device/r14/kev_s26_r14_V2R_C7_gpu_<prec>_L<L>_timing.json
       (the rows on the published L128 / L256 files as request sets, each row in its smallest bucket) and
       kev_s26_r14_V2{S,N}_C_gpu_<prec>_Ls<Ls>_timing.json (the pair Ls128 / Ls256 with GPU constant tensor sharing on (S)
       and off (N)): the median of the calls made with the GPU clock uncapped, the CPU
       state beside it, and for the rows also the sum of the uncapped single-call medians of the timing legs
       (kev_s26_r14_T4_C_gpu_<prec>_L<L>_timing); legs of an earlier app build without per-call records
       (kev_s26_r14_TR / PR / PN) are tabled apart when present.
  Mac  results/timing_mac_r14_req.json (Metal float32; rows_best and the pair Ls128 / Ls256 with / without sharing)
A request's row time = the sum of its sets' request medians (a one-row set = its per-call median); pair = the direct
hand-over request median. Crossover = the fewest questions at which the pair is faster than the rows.

    python3 scripts/r18_crossover.py > cache/r18/crossover.md      (also writes results/r18_crossover.json)"""
import json
from pathlib import Path

K = Path(__file__).resolve().parents[1]
R = K / "device/r14"
P = "kev_s26_r14"
POINTS = {128: [("tv4_000", 1), ("own_ticket_01_q2", 2), ("own_ticket_01", 3), ("own_fiveq_09", 5)],
          256: [("tv4_000", 1), ("own_order_06", 2), ("own_ticket_01", 3), ("own_email_03", 3), ("own_fiveq_09", 5)]}


def load(p):
    try:
        return json.loads(Path(p).read_text())
    except FileNotFoundError:
        return None


def cap_info(tag):
    s = R / f"{tag}.samples.txt"
    if not s.exists():
        return None
    mx, t, first_cap = [], None, None
    for line in s.read_text().splitlines():
        if line.startswith("=== "):
            t = line.split()[1]
        elif line.startswith("kgsl "):
            v = int(line.split()[2])
            mx.append(v)
            if v < 1300 and first_cap is None:
                first_cap = t
    ready = (R / f"{tag}.ready.txt").read_text().strip() if (R / f"{tag}.ready.txt").exists() else None
    start = None
    for cl in sorted(R.glob("chain_hold*.log")):
        for line in cl.read_text(errors="replace").splitlines():
            if line.startswith(f"=== leg {tag} start "):
                start = line.split()[4]
    return {"start": start, "ready": ready, "kgsl_max_clock_min": min(mx) if mx else None, "cap_from": first_cap,
            "capped": bool(mx) and min(mx) < 1300}


def leg(tag):
    d = load(R / f"{tag}.json")
    return (d, cap_info(tag)) if d and d.get("status") == "DONE" else (None, None)


def best_leg(tag):
    """The clean one of tag / tag_r2 (a leg taken again after a cap), else the retake, else the first; with both infos."""
    a, ca = leg(tag)
    b, cb = leg(f"{tag}_r2")
    if b and cb and not cb["capped"]:
        return b, cb, "retake (clean)", (a, ca)
    if a and ca and not ca["capped"]:
        return a, ca, "first (clean)", (b, cb)
    if b:
        return b, cb, "retake (capped too)", (a, ca)
    return a, ca, "first (capped)" if a else None, (None, None)


def set_ms(t):
    return (t.get("request_ms_write_run_read") or t["per_call_ms_write_run_read"])["median"]


def s26():
    out = {}
    for prec in ("fp32", "fp16acc32"):
        rows = {}
        legs = {}
        for L in (128, 256):
            d, ci, which, _ = best_leg(f"{P}_TR_C_gpu_{prec}_L{L}_timing")
            legs[f"rows_L{L}"] = {"cap": ci, "which": which}
            if d:
                for name, t in d["timing"].items():
                    rid = name[len("req_"):name.rindex("_L")]
                    rows.setdefault(rid, {})[L] = set_ms(t)
        pair = {}
        for key, tag in (("share128", f"{P}_PR_C_gpu_{prec}_share_timing"), ("share256", f"{P}_PR256_C_gpu_{prec}_share_timing"),
                         ("noshare128", f"{P}_PN_C_gpu_{prec}_noshare_timing")):
            d, ci = leg(tag)
            legs[key] = {"cap": ci}
            if d:
                pair[key] = {rid: t["direct"]["request_ms"]["median"] for rid, t in d["timing"].items()}
        out[prec] = {"rows": rows, "pair": pair, "legs": legs}
    return out


def mac():
    d = load(K / "results/timing_mac_r14_req.json")
    if not d or d.get("status") != "measured":
        return None
    res = d["results"]
    rows, pair = {}, {}
    for k, v in res.items():
        if k.endswith("rows_best"):
            for rid, t in v["requests"].items():
                rows.setdefault(rid, []).append(t["request_ms"]["median"])
        else:
            _, _, Ls, mode = k.split("_")          # pass0_pair_Ls128_share
            for rid, t in v["requests"].items():
                pair.setdefault(f"{mode}{Ls[2:]}", {}).setdefault(rid, []).append(t["direct"]["request_ms"]["median"])
    return {"rows": {r: min(x) for r, x in rows.items()}, "pair": {k: {r: min(x) for r, x in v.items()} for k, v in pair.items()},
            "lock": d.get("lock")}


ROWLENS = {}
for _s in json.loads((R / "timing_rows_r14_req.json").read_text())["sets"]:
    ROWLENS.setdefault(_s["request"], []).extend(len(r["ids"]) for r in _s["rows"])
EXPECT = {}
for _s in json.loads((R / "timing_rows_r14_req.json").read_text())["sets"]:
    EXPECT.setdefault(_s["request"], set()).add(_s["L"])


def row_total(rows, rid):
    """The row-form request time, only when every bucket the request needs was measured."""
    r = rows.get(rid)
    return sum(r.values()) if r and set(r) == EXPECT.get(rid) else None


def crossover(points, row_of, pair_of):
    wins = [n for rid, n in points if pair_of(rid) is not None and row_of(rid) is not None and pair_of(rid) < row_of(rid)]
    return min(wins) if wins else None


def s26_v2():
    """The request block v2 (per-call app build with cool-down): burst medians from results/r18_burst.json.
    Rows: a request's time = the sum over its sets (one per bucket) of the set's burst median per request (one-row sets:
    the per-call burst median). Pair: the direct hand-over burst median of that request."""
    bj = load(K / "results/r18_burst.json") or {}
    out = {}
    for prec in ("fp16acc32", "fp32"):
        rows, pair, legs = {}, {}, {}
        for L in (128, 256):
            tag = f"{P}_V2R_C7_gpu_{prec}_L{L}_timing"     # the rows on the published L128 / L256 files (form C7)
            leg = bj.get(tag)
            legs[f"rows_L{L}"] = {"cap": cap_info(tag)}
            if not leg:
                continue
            for name, v in leg["sets"].items():
                rid = name[len("req_"):name.rindex("_L")]
                w = v.get("request") or v
                b = w.get("gpu_uncapped")      # GPU uncapped, CPU state beside
                rows.setdefault(rid, {})[L] = (b["median"], b["n"], (w.get("burst") or {}).get("n", 0)) if b else (None, 0, 0)
        for key, kind, Ls in (("share128", "S", 128), ("share256", "S", 256), ("noshare128", "N", 128), ("noshare256", "N", 256)):
            tag = f"{P}_V2{kind}_C_gpu_{prec}_Ls{Ls}_timing"
            leg = bj.get(tag)
            legs[key] = {"cap": cap_info(tag)}
            if not leg:
                continue
            for name, v in leg["sets"].items():
                if not name.endswith("/direct"):
                    continue
                b, sb = v.get("gpu_uncapped"), v.get("burst")
                pair.setdefault(key, {})[name[:-len("/direct")]] = ((b["median"], b["n"], (sb or {}).get("n", 0),
                                                                     (sb or {}).get("median")) if b else (None, 0, 0, None))
        comp = {}
        for L in (64, 128, 256, 512):
            leg = bj.get(f"{P}_T4_C_gpu_{prec}_L{L}_timing")
            if leg:
                v = next(iter(leg["sets"].values()))
                b = v.get("burst")          # per call (a request set's calls are row calls too)
                comp[L] = b["median"] if b else None
        out[prec] = {"rows": rows, "pair": pair, "legs": legs, "single_call_burst": comp}
    return out


def composite(rid, single):
    """The row form of a request from the strict-burst single-call values: one call per row in its smallest bucket."""
    lens = ROWLENS.get(rid)
    if not lens:
        return None
    tot = 0.0
    for n in lens:
        L = next(L for L in (64, 128, 256, 512) if n <= L)
        if single.get(L) is None:
            return None
        tot += single[L]
    return tot


def main():
    S, M = s26(), mac()
    doc = {"s26": S, "mac": M, "crossover": {}}
    lines = []
    p = lines.append
    for prec in ("fp16acc32", "fp32"):
        d = S[prec]
        if not d["rows"] and not d["pair"]:
            continue
        p(f"\n#### S26 GPU {prec}、P2(C、旧 APK = 20 call の median、どの leg も開始 5〜20 秒で cap = 同じ cap 条件での比較の記録; "
          "括弧 = 行が入った bucket)\n")
        p("| request | 問 | row | pair Ls128 共有あり | pair Ls128 共有なし | pair Ls256 共有あり |")
        p("|---|---:|---:|---:|---:|---:|")
        allp = {rid: n for Ls in POINTS for rid, n in POINTS[Ls]}
        for rid, n in sorted(allp.items(), key=lambda x: (x[1], x[0])):
            r = d["rows"].get(rid)
            rt = row_total(d["rows"], rid)
            cells = [f"{rt:.1f}({'+'.join(f'L{L}' for L in sorted(r))})" if rt is not None else
                     (f"(L{'/'.join(str(L) for L in sorted(r))} のみ: {sum(r.values()):.1f})" if r else "—")]
            for key in ("share128", "noshare128", "share256"):
                v = (d["pair"].get(key) or {}).get(rid)
                cells.append(f"{v:.1f}" if v is not None else "—")
            p(f"| {rid} | {n} | " + " | ".join(cells) + " |")
        for Ls, key in ((128, "share128"), (128, "noshare128"), (256, "share256")):
            c = crossover(POINTS[Ls], lambda rid: row_total(d["rows"], rid),
                          lambda rid: (d["pair"].get(key) or {}).get(rid))
            doc["crossover"][f"s26_{prec}_{key}"] = c
            p(f"- 分岐点 {key}: " + (f"{c} 問から pair が速い" if c else "測った点(≤ 5 問)では pair が速い点なし"))
        p("- leg の条件: " + "; ".join(f"{k} {v['cap']['start'] if v.get('cap') else '—'} 開始、"
                                       f"{(v['cap'] or {}).get('ready', '—')}、kgsl 最小 {(v['cap'] or {}).get('kgsl_max_clock_min', '—')} MHz"
                                       + (f"(cap {v['cap']['cap_from']} から)" if v.get('cap') and v['cap'].get('cap_from') else "")
                                       + (f"、{v['which']}" if v.get("which") else "")
                                       for k, v in d["legs"].items()))
    V = s26_v2()
    doc["s26_v2"] = V
    for prec in ("fp16acc32", "fp32"):
        d = V[prec]
        if not d["rows"] and not d["pair"]:
            continue
        p(f"\n#### S26 GPU {prec}、request block v2(C、per-call APK + cool-down; 監督 21:5x (a): row も pair も GPU 上限なしの call の "
          "median(CPU の上限は問わない)、括弧 = その n と、うち GPU も CPU も上限なし = 厳密 burst の n; row の合成 = T4 の 1 call の厳密 "
          "burst 値 × 行(各行の最小 bucket))\n")
        p("| request | 問 | row 実測(GPU 上限なしの call、n; うち厳密 burst) | row 厳密 burst の合成(1 call 値 × 行) | pair Ls128 共有あり | "
          "pair Ls128 共有なし | pair Ls256 共有あり | pair Ls256 共有なし |")
        p("|---|---:|---|---:|---:|---:|---:|---:|")
        allp = {rid: n for Ls in POINTS for rid, n in POINTS[Ls]}

        def rowv(rid):
            r = d["rows"].get(rid)
            if not r or set(r) != EXPECT.get(rid) or any(v[0] is None for v in r.values()):
                return None
            return sum(v[0] for v in r.values())
        for rid, n in sorted(allp.items(), key=lambda x: (x[1], x[0])):
            r = d["rows"].get(rid)
            rv = rowv(rid)
            cv = composite(rid, d["single_call_burst"])
            cells = [f"{rv:.1f}({'+'.join(f'L{L}:{r[L][1]};{r[L][2]}' for L in sorted(r))})" if rv is not None else "—",
                     f"{cv:.1f}" if cv is not None else "—"]
            for key in ("share128", "noshare128", "share256", "noshare256"):
                v = (d["pair"].get(key) or {}).get(rid)
                cells.append(f"{v[0]:.1f}({v[1]}; 厳密 {v[2]}" + (f" = {v[3]:.1f}" if v[3] is not None else "") + ")"
                             if v and v[0] is not None else ("—(0)" if v else "—"))
            p(f"| {rid} | {n} | " + " | ".join(cells) + " |")
        for Ls, key in ((128, "share128"), (128, "noshare128"), (256, "share256"), (256, "noshare256")):
            c = crossover(POINTS[Ls], rowv, lambda rid: ((d["pair"].get(key) or {}).get(rid) or (None,))[0])
            doc["crossover"][f"s26v2_{prec}_{key}"] = c
            p(f"- 分岐点 {key}: " + (f"{c} 問から pair が速い" if c else "測った点(≤ 5 問)では pair が速い点なし"))
    if M:
        p(f"\n#### Mac Metal fp32(C、lock 取得時 load {M['lock'].get('load1_at_acquire')}、2 pass の速い方)\n")
        p("| request | 問 | row | pair Ls128 共有あり | pair Ls128 共有なし | pair Ls256 共有あり | pair Ls256 共有なし |")
        p("|---|---:|---:|---:|---:|---:|---:|")
        allp = {rid: n for Ls in POINTS for rid, n in POINTS[Ls]}
        for rid, n in sorted(allp.items(), key=lambda x: (x[1], x[0])):
            cells = [f"{M['rows'][rid]:.1f}" if rid in M["rows"] else "—"]
            for key in ("share128", "noshare128", "share256", "noshare256"):
                v = (M["pair"].get(key) or {}).get(rid)
                cells.append(f"{v:.1f}" if v is not None else "—")
            p(f"| {rid} | {n} | " + " | ".join(cells) + " |")
        for Ls, key in ((128, "share128"), (128, "noshare128"), (256, "share256"), (256, "noshare256")):
            c = crossover(POINTS[Ls], lambda rid: M["rows"].get(rid), lambda rid: (M["pair"].get(key) or {}).get(rid))
            doc["crossover"][f"mac_{key}"] = c
            p(f"- 分岐点 {key}: " + (f"{c} 問から pair が速い" if c else "測った点(≤ 5 問)では pair が速い点なし"))
    (K / "results/r18_crossover.json").write_text(json.dumps(doc, indent=1) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
