#!/usr/bin/env python3
"""Round 5b: the shipped folder end to end on one row-graph length per process — the folder's own host package
(<folder>/host/clef_litert: ClefLiteRT.decide = ClefHost.decide on ClefGraphs) on every reference run of that length
(scripts/r5b_sets.py), then (--ms) the request ms in the same compile.

    TMPDIR=tmp/r5b/t tmp/r5b/venv-rt/bin/python scripts/r5b_e2e.py --folder staging/upload --accelerator gpu \
        --length 1024 [--subset all|cpu20] [--ms] [--deadline-s 1300] [--sums] [--dry]
    -> results/r5b_e2e_<acc>_L<L>.jsonl          one line per run (resumable: a run already in it is skipped)
       results/r5b_e2e_<acc>_L<L>_part<k>.json   per invocation: compile, delegation, memory, files, timings
       results/r5b_ms_<acc>_L<L>.json            with --ms, once every selected run is in the jsonl (never overwritten)
       logs/r5b_e2e_<acc>_L<L>_part<k>.runtime.log   the runtime's own lines (VERBOSE: the delegation lines)

Per run: request (the fixture's raw request, the picture's path in `images`) -> ClefLiteRT.decide(request, tile) with
the graphs tapped (the row graph's hidden and the tower's rows are kept as they pass) -> the response body; then the
same host's scores(encode(request), hidden) -> float32 probabilities (the unrounded values the line reads), and
respond() of those must equal the body. Compared with the reference row (r5b_sets.compare; red arms also against their
base record). Non-finite values in the hidden at the real positions are counted.

Compile: the row graph of --length (and the towers when a picture run or the image ms needs them) is compiled before
the first run and timed; phys_footprint after it; the runtime's `Replacing N out of M node(s) ... yielding P
partitions` lines per compiled model (scripts/clef_rt.py turns VERBOSE on). On the GPU the line is every node in one
partition for every model: anything else ends the part with status DELEGATION_FAIL (exit 4). An exception = FAIL (5).
Files next to the graphs and in TMPDIR are listed before and after (no weight cache may appear).

--sums: after the row graph's compile, a background process re-reads that file from its real path with F_NOCACHE
(scripts/r5b_sums.py, nice 10) while the runs go on; the ms waits for it to end (no disk read overlaps a timed call).

ms (--ms, pre-registered in the round record): text = own_t14's service log cut or repeated by whole lines to the largest T <= L under
own_t14's first question (5 options): T 236 / 459 / 1,007 / 2,037 / 4,083 at L 256 / 512 / 1,024 / 2,048 / 4,096;
at L1024 also photo_01 (960 x 1280) at the 256 and 448 tiles (T 556 / 688, both L1024). Each: 5 warm-up + 20 timed
requests, a request = ClefHost.decide's steps timed one by one (encode = the checks + ids + bucket (+ the picture's
patches); tower; row inputs = table rows + picture rows + cos / sin; row graph = write + run + read; head = JointHead on
the T rows + the lm_head gather; response); then the head alone (scores on the same hidden) 5 + 20; median / min / max /
p90, the load average before and after, the lock line at every timed call.

--dry: the graphs are stand-ins that return the reference's hidden (npz last_hidden, pad rows 7.0) and picture rows
(npz image_embeds): no LiteRT, the plumbing and the comparisons only (results go to results/r5b_dry_*)."""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import importlib.metadata
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

K = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(K / "scripts"))
import r5b_sets as S  # noqa: E402

LOCK = Path(os.environ.get("MEASUREMENT_LOCK_FILE", "/dev/null"))   # the machine-wide measurement lock (its line is recorded)
MS_BASE, MS_IMAGE = "own_t14", "photo_01"
MS_TEXT_T = {256: 236, 512: 459, 1024: 1007, 2048: 2037, 4096: 4083}
CALL_S = {"gpu": {256: 0.344, 512: 0.654, 1024: 1.309, 2048: 2.707, 4096: 5.571},     # the row graph calls measured
          "cpu": {256: 1.289, 512: 2.407, 1024: 4.745, 2048: 9.571, 4096: 20.090}}  # before (results/r2_ms_*.json)


class _RUsageInfoV4(ctypes.Structure):
    _fields_ = [("ri_uuid", ctypes.c_uint8 * 16)] + [(n, ctypes.c_uint64) for n in (
        "ri_user_time", "ri_system_time", "ri_pkg_idle_wkups", "ri_interrupt_wkups", "ri_pageins", "ri_wired_size",
        "ri_resident_size", "ri_phys_footprint", "ri_proc_start_abstime", "ri_proc_exit_abstime",
        "ri_child_user_time", "ri_child_system_time", "ri_child_pkg_idle_wkups", "ri_child_interrupt_wkups",
        "ri_child_pageins", "ri_child_elapsed_abstime", "ri_diskio_bytesread", "ri_diskio_byteswritten",
        "ri_cpu_time_qos_default", "ri_cpu_time_qos_maintenance", "ri_cpu_time_qos_background",
        "ri_cpu_time_qos_utility", "ri_cpu_time_qos_legacy", "ri_cpu_time_qos_user_initiated",
        "ri_cpu_time_qos_user_interactive", "ri_billed_system_time", "ri_serviced_system_time",
        "ri_logical_writes", "ri_lifetime_max_phys_footprint", "ri_instructions", "ri_cycles", "ri_billed_energy",
        "ri_serviced_energy", "ri_interval_max_phys_footprint", "ri_runnable_time")]


def memory() -> dict:
    """ru_maxrss, phys_footprint, its lifetime maximum, resident size (bytes) of this process (proc_pid_rusage v4)."""
    doc = {"ru_maxrss": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)}
    info = _RUsageInfoV4()
    if ctypes.CDLL("/usr/lib/libproc.dylib").proc_pid_rusage(os.getpid(), 4, ctypes.byref(info)) == 0:
        doc.update(phys_footprint=int(info.ri_phys_footprint),
                   lifetime_max_phys_footprint=int(info.ri_lifetime_max_phys_footprint),
                   resident_size=int(info.ri_resident_size))
    return doc


def lock_line() -> str:
    try:
        return LOCK.read_text(errors="replace").strip()
    except OSError:
        return ""


def listing(d: Path) -> list:
    return sorted(f"{p.name} {p.lstat().st_size}" for p in Path(d).iterdir()) if Path(d).is_dir() else []


def ms_stats(ms: list) -> dict:
    a = np.asarray(ms, dtype=np.float64)
    return {"n": int(a.size), "median": float(np.median(a)), "min": float(a.min()), "max": float(a.max()),
            "p90": float(np.percentile(a, 90)), "all": [round(float(x), 3) for x in a]}


def sized_text_request(length: int) -> dict:
    """own_t14's service log, whole lines cut or repeated, under own_t14's first question (MS_TEXT_T = its T)."""
    recs = {r["id"]: r for r in json.loads((K / "fixtures/records.json").read_text())["records"]}
    base = recs[MS_BASE]["request"]
    lines = base["state"].split("\n")
    n = {256: 1, 512: 5, 1024: 15, 2048: 35, 4096: 74}[length]
    return {"model": base["model"], "state": "\n".join(lines[i % len(lines)] for i in range(n)),
            "questions": dict(list(base["questions"].items())[:1])}


class Taps:
    """Wraps a graphs object's row_graph / tower: keeps the last hidden and picture rows and the call times."""

    def __init__(self, graphs):
        self.graphs, self.hidden, self.rows, self.ms_row, self.ms_tower, self.length = graphs, None, None, None, None, None
        self.row_graph_orig, self.tower_orig = graphs.row_graph, graphs.tower
        graphs.row_graph, graphs.tower = self.row_graph, self.tower

    def row_graph(self, length, x):
        t = time.perf_counter()
        h = self.row_graph_orig(length, x)
        self.ms_row, self.hidden, self.length = (time.perf_counter() - t) * 1e3, h, length
        return h

    def tower(self, tile, patches):
        t = time.perf_counter()
        r = self.tower_orig(tile, patches)
        self.ms_tower, self.rows = (time.perf_counter() - t) * 1e3, r
        return r


class StandIn:
    """--dry: graphs that return the reference's hidden / picture rows of the run being decided."""

    def __init__(self, lengths, tiles):
        self.lengths, self.tiles, self.npz, self.compiled = tuple(lengths), tuple(tiles), None, []

    def row_graph(self, length, x):
        z = np.load(self.npz)
        T = z["last_hidden"].shape[0]
        h = np.full((1, length, 4096), 7.0, np.float32)
        h[0, :T] = z["last_hidden"]
        return h

    def tower(self, tile, patches):
        return np.load(self.npz)["image_embeds"].astype(np.float32)

    def close(self):
        pass


def npz_of(run: dict) -> Path:
    d = S.HO / "npz" if run["set"] == "heldout" else K / "ref/npz"
    return d / f"{run['id']}__{run['arm']}.npz"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", required=True, help="K-relative: the staged repository folder")
    ap.add_argument("--accelerator", choices=["gpu", "cpu"], required=True)
    ap.add_argument("--length", type=int, required=True, choices=S.BUCKETS)
    ap.add_argument("--subset", default="all", choices=["all", "cpu20"])
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--ms", action="store_true")
    ap.add_argument("--sums", action="store_true")
    ap.add_argument("--deadline-s", type=float, default=None)
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    t_start = time.monotonic()
    deadline = t_start + a.deadline_s if a.deadline_s else None
    folder = K / a.folder
    sys.path.insert(0, str(folder / "host"))
    import clef_litert as C

    pkg = Path(C.__file__).parent
    assert pkg.resolve() == (folder / "host/clef_litert").resolve(), pkg
    prefix = "r5b_dry" if a.dry else "r5b_e2e"
    tag = f"{a.accelerator}_L{a.length}"
    jsonl = K / f"results/{prefix}_{tag}.jsonl"
    k = 1
    while (K / f"results/{prefix}_{tag}_part{k}.json").exists():
        k += 1
    part_path = K / f"results/{prefix}_{tag}_part{k}.json"
    log_path = K / f"logs/{prefix}_{tag}_part{k}.runtime.log"
    runs = S.select(S.load_runs(), a.length, a.subset)
    done = set()
    if jsonl.exists():
        done = {json.loads(ln)["key"] for ln in jsonl.read_text().splitlines() if ln.strip()}
    todo = [r for r in runs if r["key"] not in done]
    bases = S.red_bases()
    tmp = Path(os.environ.get("TMPDIR", "/nonexistent"))
    row_file = folder / C.FOLDER["row"].format(length=a.length)
    graph_dirs = sorted({str(row_file.resolve().parent), str(folder.resolve())})
    doc = {"what": f"round 5b: {a.folder} end to end on {a.accelerator}, L{a.length} ({a.subset})", "part": k,
           "folder": a.folder, "accelerator": a.accelerator, "length": a.length, "subset": a.subset, "dry": a.dry,
           "package": str(pkg.relative_to(K)), "package_version": C.__version__,
           "package_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(pkg.glob("*.py"))},
           "row_file": row_file.name, "row_file_realpath": str(row_file.resolve()),
           "row_file_bytes": row_file.stat().st_size,
           "selected": len(runs), "already_done": len(runs) - len(todo),
           "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "lock_at_start": lock_line(),
           "loadavg_start": list(os.getloadavg()), "memory_start": memory(), "tmpdir": str(tmp),
           "tmpdir_before": listing(tmp), "graph_dirs_before": {d: listing(Path(d)) for d in graph_dirs},
           "versions": {m: importlib.metadata.version(m) for m in ("numpy", "tokenizers", "pillow")
                        if not (a.dry and m == "ai-edge-litert")}}
    if not a.dry:
        doc["versions"]["ai-edge-litert"] = importlib.metadata.version("ai-edge-litert")
    doc["python"] = sys.version.split()[0]
    rows_written, status, err, sums_proc = 0, "OK", None, None
    log_f = open(log_path, "w")
    saved = os.dup(2)
    os.dup2(log_f.fileno(), 2)
    clef = None
    try:
        from clef_rt import LINE_KEYS, PARTITIONED, REPLACING, runtime_log_verbose

        def delegation(since: int) -> dict:
            sys.stderr.flush()
            text = log_path.read_text(errors="replace")[since:]
            return {"replacing": [{"delegated": int(m[0]), "total": int(m[1]), "delegate": m[2], "partitions": int(m[3])}
                                  for m in REPLACING.findall(text)],
                    "partitioned": [{"subgraph": int(m[0]), "selected": int(m[1]), "total": int(m[2]),
                                     "partitions": int(m[3])} for m in PARTITIONED.findall(text)],
                    "key_lines": [ln for ln in text.splitlines() if LINE_KEYS.search(ln)][:20]}

        def log_size() -> int:
            sys.stderr.flush()
            return log_path.stat().st_size

        t0 = time.perf_counter()
        if a.dry:
            from tokenizers import Tokenizer
            host = C.ClefHost(Tokenizer.from_file(str(folder / C.FOLDER["tokenizer"])),
                              C.Bf16Table(folder / C.FOLDER["embed_tokens"]), C.Bf16Table(folder / C.FOLDER["lm_head"]),
                              C.JointHead.from_files(folder / C.FOLDER["joint_head"], folder / C.FOLDER["joint_head_config"]),
                              buckets=(a.length,))
            clef = C.ClefLiteRT(host, StandIn((a.length,), (256, 448)))
        else:
            doc["logger"] = runtime_log_verbose()
            clef = C.ClefLiteRT.from_dir(folder, a.accelerator, a.threads, lengths=[a.length])
        doc["host_load_seconds"] = round(time.perf_counter() - t0, 3)
        doc["lengths"], doc["tiles"] = list(clef.graphs.lengths), list(clef.graphs.tiles)
        assert doc["lengths"] == [a.length], doc["lengths"]
        need_tiles = sorted({r["tile"] for r in todo if r["image"] is not None}
                            | ({256, 448} if a.ms and a.length == 1024 and a.accelerator == "gpu" else set()))
        models = {}
        if not a.dry and (todo or a.ms):
            for name, fn in [("row", lambda: clef.graphs._row_graph(a.length))] + \
                            [(f"tower_g{t}", (lambda t=t: clef.graphs._tower(t))) for t in need_tiles]:
                at = log_size()
                t1 = time.perf_counter()
                g = fn()
                models[name] = {"file": g.path.name, "compile_seconds": round(time.perf_counter() - t1, 3),
                                "inputs": {n: list(s) for n, s in g.input_shapes.items()},
                                "outputs": {n: list(s) for n, s in g.output_shapes.items()},
                                "fully_accelerated": g.fully_accelerated(), "options": g.options,
                                "memory_after_compile": memory(), "delegation": delegation(at)}
                rep = models[name]["delegation"]["replacing"]
                models[name]["one_partition_all_nodes"] = bool(rep) and all(
                    x["delegated"] == x["total"] and x["partitions"] == 1 for x in rep)
                print(f"compiled {name}: {models[name]['compile_seconds']} s {rep}", flush=True)
                if a.accelerator == "gpu" and not models[name]["one_partition_all_nodes"]:
                    status = "DELEGATION_FAIL"
                if name == "row" and a.sums:
                    sums_proc = subprocess.Popen(
                        ["nice", "-n", "10", sys.executable, str(K / "scripts/r5b_sums.py"), "--file",
                         str(row_file.relative_to(K)), "--out", f"results/r5b_sums_L{a.length}.json"],
                        cwd=K, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    doc["sums_started_at"] = time.strftime("%H:%M:%S")
        doc["models"] = models
        taps = Taps(clef.graphs)
        if status == "OK":
            with open(jsonl, "a") as out:
                for run in todo:
                    if deadline is not None and time.monotonic() > deadline:
                        status = "DEADLINE"
                        break
                    if a.dry:
                        clef.graphs.npz = npz_of(run)
                    req = S.request_of(run)
                    t1 = time.perf_counter()
                    body = clef.decide(req, tile=run["tile"])
                    ms_decide = (time.perf_counter() - t1) * 1e3
                    assert taps.length == a.length, (run["key"], taps.length)
                    enc = clef.host.encode(req, tile=run["tile"])
                    logits = clef.host.scores(enc, taps.hidden)
                    probs = [C.question_probs(z) for z in logits]
                    same_body = clef.host.respond(req, enc, logits) == body
                    T = enc.record.tokens
                    hr = taps.hidden[0, :T]
                    item = S.compare(run, probs, body)
                    item.update(red=S.red_move(run, probs, bases), respond_equals_decide=same_body,
                                nonfinite_real=int((~np.isfinite(hr)).sum()), ms_decide=round(ms_decide, 2),
                                ms_row_graph=round(taps.ms_row, 2),
                                ms_tower=round(taps.ms_tower, 2) if run["image"] is not None else None,
                                tokens_host=T, at=time.strftime("%H:%M:%S"))
                    taps.ms_tower = None
                    out.write(json.dumps(item) + "\n")
                    out.flush()
                    os.fsync(out.fileno())
                    rows_written += 1
                    print(f"{rows_written} {run['key']} T={T} max|dp| {item['max_abs_dp']:.3e} "
                          f"argmax {item['argmax_equal_non_near_tie']} body {item['body_structure_equal']} "
                          f"{ms_decide:.0f} ms", flush=True)
        done = set()
        if jsonl.exists():
            done = {json.loads(ln)["key"] for ln in jsonl.read_text().splitlines() if ln.strip()}
        complete = all(r["key"] in done for r in runs)
        doc["complete"] = complete
        ms_path = K / f"results/{'r5b_dry_ms' if a.dry else 'r5b_ms'}_{tag}.json"
        if sums_proc is not None:
            t1 = time.perf_counter()
            doc["sums_rc"] = sums_proc.wait()
            doc["sums_waited_seconds"] = round(time.perf_counter() - t1, 1)
        if a.ms and complete and status == "OK" and not ms_path.exists():
            n_req = 3 if (a.length == 1024 and a.accelerator == "gpu") else 1
            need_s = 60 + n_req * 25 * (CALL_S[a.accelerator][a.length] * 1.15 + 0.3)
            if deadline is None or time.monotonic() + need_s < deadline:
                ms_path.write_text(json.dumps(ms_protocol(clef, taps, a, C), indent=1) + "\n")
            else:
                doc["ms_deferred"] = "not enough time left in this part"
        doc["memory_end"] = memory()
        clef.close()
        doc["memory_after_close"] = memory()      # does closing the compiled graphs give their memory back?
    except BaseException as e:
        status = "FAIL"
        import traceback
        err = traceback.format_exc()[-3000:]
        print(err, flush=True)
    finally:
        sys.stderr.flush()
        os.dup2(saved, 2)
        log_f.close()
    doc.update(status=status, error=err, runs_this_part=rows_written, finished_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
               seconds_wall=round(time.monotonic() - t_start, 1), lock_at_end=lock_line(),
               loadavg_end=list(os.getloadavg()), tmpdir_after=listing(tmp),
               graph_dirs_after={d: listing(Path(d)) for d in graph_dirs})
    doc["new_files_tmpdir"] = sorted(set(doc["tmpdir_after"]) - set(doc["tmpdir_before"]))
    doc["new_files_next_to_graphs"] = {d: sorted(set(doc["graph_dirs_after"][d]) - set(doc["graph_dirs_before"][d]))
                                       for d in graph_dirs}
    doc["runtime_log_key_lines"] = [ln for ln in log_path.read_text(errors="replace").splitlines()
                                    if __import__("re").search(r"(?i)fail|error|unsupported|abort|reject", ln)][:40]
    part_path.write_text(json.dumps(doc, indent=1, default=str) + "\n")
    print(json.dumps({kk: doc.get(kk) for kk in ("status", "selected", "already_done", "runs_this_part", "complete",
                                                 "seconds_wall", "ms_deferred", "new_files_tmpdir")}), flush=True)
    return {"OK": 0, "DEADLINE": 0, "DELEGATION_FAIL": 4}.get(status, 5)


def ms_protocol(clef, taps, a, C) -> dict:
    host, graphs = clef.host, clef.graphs
    out = {"what": f"round 5b: one decision request on {a.accelerator}, L{a.length} (ClefHost.decide's steps timed)",
           "accelerator": a.accelerator, "length": a.length, "warmup": 5, "timed": 20,
           "lock_before": lock_line(), "loadavg_before": list(os.getloadavg()),
           "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "requests": {}}
    reqs = [("text", sized_text_request(a.length), 448)]
    if a.length == 1024 and a.accelerator == "gpu":
        recs = {r["id"]: r for r in json.loads((K / "fixtures/records.json").read_text())["records"]}
        for tile in (256, 448):
            r = dict(recs[MS_IMAGE]["request"])
            r["images"] = [str(K / "fixtures/images" / recs[MS_IMAGE]["image_files"][0])]
            reqs.append((f"image_g{tile}", r, tile))
    for name, req, tile in reqs:
        body_decide = clef.decide(req, tile=tile)
        parts = {p: [] for p in ("request", "encode", "tower", "row_inputs", "row_graph", "head", "response")}
        labels, lock_ok = set(), True
        for i in range(25):
            ln = lock_line()
            t0 = time.perf_counter()
            enc = host.encode(req, tile=tile)
            t1 = time.perf_counter()
            rows = graphs.tower(enc.tile, enc.patches) if enc.patches is not None else None
            t2 = time.perf_counter()
            x = host.row_inputs(enc, rows)
            t3 = time.perf_counter()
            hidden = graphs.row_graph(enc.length, x)
            t4 = time.perf_counter()
            logits = host.scores(enc, hidden)
            t5 = time.perf_counter()
            body = host.respond(req, enc, logits)
            t6 = time.perf_counter()
            if i >= 5:
                labels.add(ln.split(" timing")[0])
                lock_ok &= f"clef-r5b" in ln
                for p, v in (("request", t6 - t0), ("encode", t1 - t0), ("tower", t2 - t1), ("row_inputs", t3 - t2),
                             ("row_graph", t4 - t3), ("head", t5 - t4), ("response", t6 - t5)):
                    parts[p].append(v * 1e3)
        head = []
        for i in range(25):
            t = time.perf_counter()
            host.scores(enc, hidden)
            if i >= 5:
                head.append((time.perf_counter() - t) * 1e3)
        out["requests"][name] = {
            "tokens": enc.record.tokens, "length": enc.length, "tile": tile if enc.patches is not None else None,
            "questions": len(enc.record.questions), "options": sum(len(q.option_ids) for q in enc.record.questions),
            "body_equals_decide": body == body_decide, "hidden_finite": bool(np.isfinite(hidden[0, :enc.record.tokens]).all()),
            **{p: ms_stats(v) for p, v in parts.items() if v and not (p == "tower" and enc.patches is None)},
            "head_alone": ms_stats(head), "lock_held_by_this_run_every_timed_call": bool(lock_ok),
            "lock_labels_seen": sorted(labels)}
    out.update(lock_after=lock_line(), loadavg_after=list(os.getloadavg()),
               finished_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    return out


if __name__ == "__main__":
    raise SystemExit(main())
