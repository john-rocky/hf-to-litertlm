#!/usr/bin/env python3
"""The host over a set of records, one jsonl row a record, resumable (results/gate/<tag>.jsonl, hidden[pooled] in
out/gate/<tag>/<id>.npy). Fields as conversion/ref_run.py where they mean the same. Needs fetch_inputs.py first.

    <python> conversion/run_set.py <work> --backend litert --form fp16fc --acc gpu_f32 --set A [--arm base|R1|R2|R4] [--tag T]
    <python> conversion/run_set.py <work> --backend torch --set A            (the separation test, the reference environment)

Sets (conversion/sets/, record ids of the benchmark's olmOCR family): examples, A, red, floor, device, ids:1,2,3. A record outside the installed buckets (too_large) goes to
<tag>.skipped.jsonl with its sizes. The order is by decision bucket, then tower bucket: each graph compiles once or close to it.
--stop-after N stops after N new records (dry runs); a picture is never written to git or to the rows."""
import argparse
import hashlib
import json
import os
import resource
import signal
import sys
import time
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("work", type=Path, help="the work directory")
ap.add_argument("--backend", choices=("litert", "torch"), default="litert")
ap.add_argument("--form", default="fp16fc", help="suffix of the graph file names")
ap.add_argument("--tower-form", default="", help="the tower's form when it differs")
ap.add_argument("--acc", choices=("cpu", "gpu_f32", "gpu_default"), default="cpu")
ap.add_argument("--set", default="A")
ap.add_argument("--arm", default="base")
ap.add_argument("--tag", default="")
ap.add_argument("--threads", type=int, default=4)
ap.add_argument("--keep-tower", type=int, default=8)
ap.add_argument("--keep-decide", type=int, default=1)
ap.add_argument("--stop-after", type=int, default=0)
ap.add_argument("--multi", default="", help="the decision graph from one multi-signature file (path from the work directory)")
ap.add_argument("--multi-tower", default="", help="the tower from one multi-signature file (path from the work directory)")
ap.add_argument("--tower-acc", default="", help="the tower's accelerator when it differs from --acc (a decision graph on the default precision, the tower on fp32)")
ap.add_argument("--sharing", action="store_true", help="GpuOptions(constant_tensor_sharing=True)")
ap.add_argument("--retry-skipped", action="store_true", help="the records of <tag>.skipped.jsonl run again (more buckets installed): the old file moves to <tag>.skipped_before.jsonl")
ap.add_argument("--max-n", type=int, default=0, help="skip a record with a picture above this (the torch backbone has no bucket)")
ap.add_argument("--max-seconds", type=int, default=3600)
ap.add_argument("--author-tables", action="store_true", help="torch backend: the author's own rotary tables (torch's float32 cos / sin) in place of the host's: everything else of the host stays under test")
ap.add_argument("--graphs", default="out/graphs")
ap.add_argument("--lengths", default="", help="comma list: only these L buckets")
ap.add_argument("--patches", default="", help="comma list: only these N buckets")
a = ap.parse_args()
WORK = a.work.resolve()
sys.path.insert(0, str(WORK / "host"))
sys.path.insert(0, str(WORK / "conversion"))
from decide_host import host as H  # noqa: E402
from decide_host.litert_backends import LiteRTDecide, LiteRTMultiDecide, LiteRTMultiTower, LiteRTTower, discover  # noqa: E402

SNAP, BENCH = WORK / "snapshot", WORK / "bench"
sizes_l = {int(x) for x in a.lengths.split(",")} if a.lengths else None
sizes_n = {int(x) for x in a.patches.split(",")} if a.patches else None

env = {"backend": a.backend, "pid": os.getpid()}
t_load = time.perf_counter()
if a.backend == "litert":
    d_files = discover(WORK / a.graphs, "decide", a.form, sizes_l)
    t_files = discover(WORK / a.graphs, "tower", a.tower_form or a.form, sizes_n)
    assert d_files and t_files, (d_files, t_files)
    tower = LiteRTMultiTower(WORK / a.multi_tower, a.tower_acc or a.acc, a.threads, a.sharing) if a.multi_tower else LiteRTTower(t_files, a.tower_acc or a.acc, a.threads, a.keep_tower, a.sharing)
    decide = LiteRTMultiDecide(WORK / a.multi, a.acc, a.threads, a.sharing) if a.multi else LiteRTDecide(d_files, a.acc, a.threads, a.keep_decide, a.sharing)
    env.update(multi=a.multi, multi_tower=a.multi_tower, sharing=a.sharing, tower_acc=a.tower_acc or a.acc)
    env.update(form=a.form, tower_form=a.tower_form or a.form, acc=a.acc, threads=a.threads, L=list(d_files), N=list(t_files),
               decide_files={k: v.name for k, v in d_files.items()}, tower_files={k: v.name for k, v in t_files.items()})
    tag = a.tag or f"{a.set}__{a.arm}__{a.form}__{a.acc}".replace(":", "_").replace(",", "_")
else:
    import importlib.util

    import torch
    from transformers import Qwen3_5ForSequenceClassification

    s = importlib.util.spec_from_file_location("fast_dwconv", WORK / "conversion/fast_dwconv.py")
    dw = importlib.util.module_from_spec(s)
    s.loader.exec_module(dw)
    dw.install()
    from torch_backends import TorchDecide, TorchTower

    torch.set_num_threads(a.threads)
    torch.set_grad_enabled(False)
    model = Qwen3_5ForSequenceClassification.from_pretrained(SNAP, dtype=torch.float32).eval()
    if a.author_tables:
        from decide_host import formulas as F
        from transformers.vision_utils import get_vision_position_ids

        vis, rot_author = model.model.visual, model.model.language_model.rotary_emb

        def author_vision(gh, gw):
            rot = vis.rotary_pos_emb(get_vision_position_ids(torch.tensor([[1, gh, gw]]), vis.spatial_merge_size)).reshape(gh * gw, -1)
            emb = torch.cat((rot, rot), dim=-1)
            return emb.cos().float().numpy(), emb.sin().float().numpy()

        def author_text(pos3):
            c, s = rot_author(torch.zeros(1), torch.from_numpy(pos3[:, None, :]).long())
            return c.float().numpy(), s.float().numpy()

        F.vision_rotary, F.mrope_tables = author_vision, author_text
    tower = TorchTower(model.model.visual)
    decide = TorchDecide(model.model.language_model)
    env.update(torch=torch.__version__, threads=torch.get_num_threads(), fast_dwconv=True, author_tables=a.author_tables)
    tag = a.tag or f"{a.set}__{a.arm}__torchbackbone{'_authortables' if a.author_tables else ''}".replace(":", "_").replace(",", "_")
host = H.Host(tower, decide, SNAP)
env["load_s"] = round(time.perf_counter() - t_load, 2)
(WORK / "results/gate").mkdir(parents=True, exist_ok=True)
outp, skipp = WORK / f"results/gate/{tag}.jsonl", WORK / f"results/gate/{tag}.skipped.jsonl"
hdir = WORK / f"out/gate/{tag}"
hdir.mkdir(parents=True, exist_ok=True)


class Stop(Exception):
    pass


def on_alarm(*_):
    raise Stop("watchdog")


signal.signal(signal.SIGALRM, on_alarm)


def label_int(x):
    """score.py label_to_int; an unknown label stays None."""
    return {"1": 1, "pos": 1, "positive": 1, "0": 0, "neg": 0, "negative": 0}.get(str(x).strip().lower())


def row_fields(out: dict) -> dict:
    keep = ("prob_1", "logits", "prob_1_prev", "logits_prev", "T", "N", "grids", "pooled_id", "ids_sha")
    return {k: out[k] for k in keep}


def bucket_of(sizes, n):
    for s in sizes:
        if n <= s:
            return s
    return 10 ** 9


if a.set == "examples":
    rows = []
    for d in sorted((WORK / "author/scripts/minicpm-ocr-jev/examples").iterdir()):
        c = json.loads((d / "case.json").read_text())
        cand = next(x for x in c["candidates"] if x["role"] == c["reference_render_of"])
        t0 = time.perf_counter()
        out = host.score(str(d / c["block_image"]), str(d / c["reference_render"]), cand["text"])
        assert out["status"] == "ok", out
        hid = out.pop("hidden_pooled")
        want = c["observed_prob_yes"][c["reference_render_of"]]
        np.save(hdir / f"{c['case_id']}.npy", hid)
        row = dict(row_fields(out), case=c["case_id"], observed=want, diff=out["prob_1"] - want, p1=abs(out["prob_1"] - want) <= 0.02,
                   seconds=round(time.perf_counter() - t0, 3), ms=out["ms"], hidden_sha=hashlib.sha256(hid.tobytes()).hexdigest()[:16])
        rows.append(row)
        print(json.dumps({k: row[k] for k in ("case", "prob_1", "observed", "diff", "p1", "T", "N", "seconds")}), flush=True)
    (WORK / f"results/gate/{tag}.json").write_text(json.dumps({"env": env, "rows": rows}, indent=1) + "\n")
    print("P1", "PASS" if all(r["p1"] for r in rows) else "FAIL", [round(r["diff"], 5) for r in rows], "DONE", tag, flush=True)
    sys.exit(0)

rows_all = {r["item_id"]: r for r in map(json.loads, (BENCH / "data/jevocrbench.jsonl").read_text(encoding="utf-8").splitlines())}
dist = {r["item_id"]: r for r in map(json.loads, (WORK / "conversion/sets/record_sizes.jsonl").read_text().splitlines())}
red = json.loads((WORK / "conversion/sets/red_arms.json").read_text())
partner = {int(k): v for k, v in red["partner"].items()}
if a.set.startswith("ids:"):
    ids = [int(x) for x in a.set[4:].split(",")]
else:
    name = {"A": "set_A", "red": "red_arms", "floor": "floor_ids", "device": "set_device"}[a.set]
    ids = json.loads((WORK / f"conversion/sets/{name}.json").read_text())["item_ids"]
done = set()
if outp.exists():
    done |= {json.loads(x)["item_id"] for x in outp.read_text().splitlines() if x.strip()}
if a.retry_skipped and skipp.exists():
    skipp.replace(skipp.with_name(f"{tag}.skipped_before.jsonl"))
if skipp.exists():
    done |= {json.loads(x)["item_id"] for x in skipp.read_text().splitlines() if x.strip()}
ls = (sorted(decide.sigs) if a.multi else sorted(decide.files)) if a.backend == "litert" else [10 ** 9]
ns = sorted(tower.files) if a.backend == "litert" else [10 ** 9]


def order_key(i):
    d = dist[i]
    n1 = dist[partner[i]]["N1"] if a.arm == "R4" else d["N1"]
    return (bucket_of(ls, d["T"]), bucket_of(ns, max(n1, d["N2"])), d["T"], i)


order = sorted((i for i in ids if i not in done), key=order_key)
print(json.dumps({"tag": tag, "records": len(ids), "done": len(done), "todo": len(order), "env": env}), flush=True)
t_start = time.time()
n_new = 0
with outp.open("a") as out_f, skipp.open("a") as sk:
    for count, i in enumerate(order):
        d, r = dist[i], rows_all[i]
        orig, rend = str(BENCH / r["image_path"]), str(BENCH / r["render_image_path"])
        text = str(r["ocr_text"] or "")
        if a.arm == "R1":
            text = str(rows_all[partner[i]]["ocr_text"] or "")
            images = [orig, rend]
        elif a.arm == "R2":
            images = [rend, orig]
        elif a.arm == "R4":
            images = [str(BENCH / rows_all[partner[i]]["image_path"]), rend]
        else:
            assert a.arm == "base", a.arm
            images = [orig, rend]
        n1 = dist[partner[i]]["N1"] if a.arm == "R4" else (d["N2"] if a.arm == "R2" else d["N1"])
        if a.max_n and max(n1, d["N1"] if a.arm == "R2" else d["N2"]) > a.max_n:
            sk.write(json.dumps({"item_id": i, "category": r["category"], "status": "above_max_n", "n": [n1, d["N2"]], "T": d["T"], "time": time.strftime("%H:%M:%S")}) + "\n")
            sk.flush()
            continue
        t0 = time.perf_counter()
        signal.alarm(a.max_seconds)
        try:
            out = host.score(images[0], images[1], text)
        except Stop as e:
            out = {"status": "timeout", "why": str(e)}
        signal.alarm(0)
        sec = time.perf_counter() - t0
        if out["status"] != "ok":
            sk.write(json.dumps({"item_id": i, "category": r["category"], **{k: v for k, v in out.items() if k != "hidden_pooled"}, "seconds": round(sec, 2), "time": time.strftime("%H:%M:%S")}) + "\n")
            sk.flush()
            print("SKIP", i, out["status"], out.get("n"), out.get("T"), flush=True)
            continue
        hid = out.pop("hidden_pooled")
        np.save(hdir / f"{i}.npy", hid)
        row = dict(row_fields(out), item_id=i, category=r["category"], label=label_int(r["label"]),
                   family=d["family"], lang=d["lang"], arm=a.arm, seconds=round(sec, 3), ms={k: (round(v, 2) if isinstance(v, float) else [round(x, 2) for x in v]) for k, v in out["ms"].items()},
                   hidden_sha=hashlib.sha256(hid.tobytes()).hexdigest()[:16], time=time.strftime("%H:%M:%S"),
                   peak_rss_gib=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30, 2))
        if a.backend == "litert":
            row.update(L=decide.last_bucket, Ntop=tower.last_bucket)
        out_f.write(json.dumps(row) + "\n")
        out_f.flush()
        n_new += 1
        if count % 10 == 0:
            print(count, i, "T", row["T"], "N", row["N"], f"{sec:.2f} s", "prob", round(row["prob_1"], 5), "rss", row["peak_rss_gib"], f"total {time.time() - t_start:.0f} s", flush=True)
        if a.stop_after and n_new >= a.stop_after:
            break
if a.backend == "litert":
    (WORK / f"results/gate/{tag}.loads.json").write_text(json.dumps({"env": env, "tower": tower.load_log, "decide": decide.load_log}, indent=1) + "\n")
    tower.close()
    decide.close()
print("GATE_DONE", tag, "new", n_new, f"{time.time() - t_start:.0f} s", flush=True)
