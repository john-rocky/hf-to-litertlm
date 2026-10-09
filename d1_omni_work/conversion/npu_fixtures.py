"""Round 10: the fixtures of a C-API runner on LiteRT 2.2.0 (not included) for the d1-omni
decision graph on the Galaxy S26 HTP, and their check on the Mac.

    cd d1_omni_work; PY=~/venvs/lt094dev/bin/python
    $PY scripts/npu_fixtures.py write      # -> device/r10/stage/fx_L{128,256}/ + device/r10/fx_L{128,256}.{manifest.json,sha256}
    ~/code/standup/tools/quiet/quiet_wait.py -- $PY scripts/npu_fixtures.py check     # Mac CPU 8 threads (heavy)
        # -> results/s26_npu_fixture_check_r10.json + device/r10/mac/<graph>/model0/<id3>.f32 (+ the runtime log)

How the runner reads a fixture (the runner's source, read 2026-10-08): for fixture <id3> and every signature input i
(LiteRtGetSignatureInputTensorByIndex(sig, i)), the file <fixtures>/<id3>_<name_i>.f32 is copied byte for byte into
input buffer i (its size must be the tensor's element count x 4; the ids file holds int32 bytes), where name_i =
LiteRtGetSignatureInputName(sig, i) unless --inputs overrides it. The C API's index i runs over the subgraph's input
tensors, not over the SignatureDef's stored (alphabetical) order (Kev round 17: chain A2 died on FIXTURE_ERROR because
its --inputs list was alphabetical; the runner printed `input[i] sig_name=` in subgraph order). This round names every
file by its input name and never passes --inputs, so the files reach their tensors by name; `write` records the
subgraph order (input index, tensor index, name, shape, dtype) next to the SignatureDef's stored order, and the device
scorer checks the runner's own `input[i] sig_name=... elems=... rank=...` lines against it.

write: per L, the gate rows (device/rows_L<L>_sub.json, the rows of rounds 5 and 9) laid out with the host's
build_inputs (host/d1_host.py; media rows get their prefix from device/prefix_<record>.f32) + the question type's
one-hot, as fixtures 000.. in the rows file's order; then the card's timing rows that the rows file does not hold
(timing_rows.json, set (a) card_text/refund and set (b) its three questions; at L128 they are rows 000-002 already, at
L256 they are added as 108-110). Never overwrites a fixture dir.
check: the Mac CompiledModel CPU (XNNPACK, 8 threads, ai-edge-litert 2.2.0) reads the fixture files the way the runner
does (subgraph order, the byte count each tensor needs, ids as int32) and runs them by input name; each fixture's
scores [L] -> device/r10/mac/<graph>/model0/<id3>.f32 (the Mac same-file reference of the device scorer and the source
of the dry run's stand-in runner). The fixtures must equal the host's build_inputs bit for bit, and the scores must
equal the earlier Mac CPU runs of the same graph file bit for bit: round 8's store (out/r8_runs, every real position)
for the f16safe files and round 4's (out/r4_runs) for the pre-rewrite L128 file, plus the marker scores of round 9's /
round 5's Mac CPU gate-app stand-in (device/r9/mac, device/r5/mac: the same rows file run with the app's contract;
round 8 showed the f16safe and the pre-rewrite files give bit-equal CPU scores, so round 5's run of the pre-rewrite
file is a reference for the f16safe file too).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import mmap
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
K = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(K / "scripts"))
sys.path.insert(0, str(K / "host"))
import d1_host as H  # noqa: E402  (numpy only at import)

R10 = K / "device/r10"
STAGE = R10 / "stage"
D = 1024
NAMES = ("ids", "prefix", "media", "pad", "keep_right", "qtype_onehot")
GRAPHS = {   # graph tag -> file, its bytes / sha256 (rounds 4 / 8 records), the fixture dir, the Mac CPU references
    "L128_f16safe_fp16": {
        "L": 128, "file": "out/d1omni_decide_L128_f16safe_fp16.tflite", "bytes": 896250176,
        "sha256": "69720aa44d60a7feb0bd3056b55c7a038e67761b5d810cd5d5b0496ec61273f3", "fx": "fx_L128", "count": None,
        "store": "out/r8_runs/cpu_L128_f16safe_fp16.npz",
        "sels": {"r9_mac_cpu8_same_file": "device/r9/mac/{}mac_d1omni_decide_L128_f16safe_fp16__rows_L128_sub{}",
                 "r5_mac_cpu8_pre_rewrite_file": "device/r5/mac/{}mac_d1omni_decide_L128_fp16__rows_L128_sub{}"}},
    "L256_f16safe_fp16": {
        "L": 256, "file": "out/d1omni_decide_L256_f16safe_fp16.tflite", "bytes": 896315712,
        "sha256": "eebf0dcc2bbefa713c71ce64c5b90600497ff5a26edc3a8ecd0116f99725089b", "fx": "fx_L256", "count": None,
        "store": "out/r8_runs/cpu_L256_f16safe_fp16.npz",
        "sels": {"r9_mac_cpu8_same_file": "device/r9/mac/{}mac_d1omni_decide_L256_f16safe_fp16__rows_L256_sub{}",
                 "r5_mac_cpu8_pre_rewrite_file": "device/r5/mac/{}mac_d1omni_decide_L256_fp16__rows_L256_sub{}"}},
    "L128_fp16": {   # the pre-rewrite file: leg 4 runs its first 20 fixtures (the "collapses on the HTP too" evidence)
        "L": 128, "file": "out/d1omni_decide_L128_fp16.tflite", "bytes": 896201152,
        "sha256": "7f960d2a23d9274a0978f00bb1d6d183beb4cf3edc5f163f5b39c9e4d1ecc74f", "fx": "fx_L128", "count": 20,
        "store": "out/r4_runs/cpu_L128_fp16.npz",
        "sels": {"r5_mac_cpu8_same_file": "device/r5/mac/{}mac_d1omni_decide_L128_fp16__rows_L128_sub{}"}},
}
FX = {128: {"rows": "device/rows_L128_sub.json", "order_graph": "L128_f16safe_fp16"},
      256: {"rows": "device/rows_L256_sub.json", "order_graph": "L256_f16safe_fp16"}}
TIMING_SETS = ("a_one_question", "b_three_questions")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(1 << 24):
            h.update(block)
    return h.hexdigest()


def rel(p) -> str:
    p = Path(p).resolve()
    return str(p.relative_to(K)) if p.is_relative_to(K) else str(p)


def write_new(path: Path, text: str) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {rel(path)}")
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def io_order(path: Path) -> dict:
    """The flatbuffer's signature (mmapped, never runs the model): the subgraph's input / output tensors in order with
    their names, and the SignatureDef's stored order (Kev round 17 r17_device_sig.py, re-written here)."""
    from ai_edge_litert import schema_py_generated as schema

    tname = {v: k for k, v in vars(schema.TensorType).items() if not k.startswith("_")}
    with open(path, "rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    model = schema.Model.GetRootAs(mm, 0)
    assert model.SignatureDefsLength() == 1, model.SignatureDefsLength()
    sd = model.SignatureDefs(0)
    sg = model.Subgraphs(sd.SubgraphIndex())
    stored_in = [(sd.Inputs(i).Name().decode(), int(sd.Inputs(i).TensorIndex())) for i in range(sd.InputsLength())]
    stored_out = [(sd.Outputs(i).Name().decode(), int(sd.Outputs(i).TensorIndex())) for i in range(sd.OutputsLength())]
    by_t_in, by_t_out = {t: n for n, t in stored_in}, {t: n for n, t in stored_out}

    def desc(i, t, name):
        tt = sg.Tensors(int(t))
        return {"index": i, "tensor_index": int(t), "name": name, "tensor_name": tt.Name().decode(),
                "shape": [int(x) for x in tt.ShapeAsNumpy()], "dtype": tname.get(tt.Type(), str(tt.Type()))}

    ins = [desc(i, t, by_t_in[int(t)]) for i, t in enumerate(sg.InputsAsNumpy())]
    outs = [desc(i, t, by_t_out[int(t)]) for i, t in enumerate(sg.OutputsAsNumpy())]
    assert sorted(x["name"] for x in ins) == sorted(n for n, _ in stored_in) == sorted(NAMES), (ins, stored_in)
    assert [x["name"] for x in outs] == ["scores"], outs
    key, sgi, nops = sd.SignatureKey().decode(), int(sd.SubgraphIndex()), int(sg.OperatorsLength())
    del model, sd, sg
    mm.close()
    return {"file": rel(path), "signature": key, "subgraph": sgi, "operators": nops,
            "inputs_subgraph_order": ins, "outputs_subgraph_order": outs,
            "inputs_signaturedef_stored_order": [{"name": n, "tensor_index": t} for n, t in stored_in],
            "note": "the C API (the S26 runner) numbers a signature's inputs / outputs in the subgraph order; the "
                    "SignatureDef's stored order is alphabetical and not used"}


def row_arrays(row: dict, L: int) -> dict:
    """One rows-file row -> the six inputs (host build_inputs + the one-hot)."""
    ids = [int(x) for x in row["ids"]]
    P = int(row.get("P") or 0)
    pre = None
    if P:
        raw = (K / "device" / row["prefix_file"]).read_bytes()
        assert len(raw) == P * D * 4, (row["key"], len(raw), P)
        pre = np.frombuffer(raw, dtype="<f4").reshape(P, D).astype(np.float32)
    x = H.build_inputs(ids, pre, L)
    oh = np.zeros((1, 3), np.float32)
    oh[0, int(row["qtype"])] = 1.0
    x["qtype_onehot"] = oh
    return x


def file_bytes(name: str, arr: np.ndarray) -> bytes:
    assert arr.dtype == (np.int32 if name == "ids" else np.float32), (name, arr.dtype)
    return np.ascontiguousarray(arr).astype("<i4" if name == "ids" else "<f4", copy=False).tobytes()


def concat_sha(d: Path) -> tuple[str, int, int]:
    """sha256 over the files' bytes concatenated in byte-sorted name order (= the phone's `cat $(ls | LC_ALL=C sort)`)."""
    h, n, b = hashlib.sha256(), 0, 0
    for p in sorted(d.iterdir(), key=lambda q: q.name.encode()):
        data = p.read_bytes()
        h.update(data)
        n, b = n + 1, b + len(data)
    return h.hexdigest(), n, b


def write(a) -> int:
    trows = json.loads((K / "device/timing_rows.json").read_text())
    out = {}
    for L in a.L:
        spec = FX[L]
        g = GRAPHS[spec["order_graph"]]
        order = io_order(K / g["file"])
        assert order["signature"] == f"decide_{L}", order["signature"]
        shapes = {x["name"]: x["shape"] for x in order["inputs_subgraph_order"]}
        rows_doc = json.loads((K / spec["rows"]).read_text())
        assert int(rows_doc["L"]) == L and int(rows_doc["hidden"]) == D and int(rows_doc["pad_id"]) == 0
        rows = [dict(r, role="gate") for r in rows_doc["rows"]]
        have = {r["key"]: i for i, r in enumerate(rows)}
        sets = {}
        for s in trows["sets"]:
            if int(s["L"]) != L or s["name"] not in TIMING_SETS:
                continue
            sets[s["name"]] = [r["key"] for r in s["rows"]]
            for r in s["rows"]:
                if r["key"] not in have:
                    have[r["key"]] = len(rows)
                    rows.append(dict(r, role="timing"))
                else:
                    g0 = rows[have[r["key"]]]
                    assert all(g0[k] == r[k] for k in ("ids", "markers", "K", "P", "qtype")), r["key"]
        # set (a) at L256 = card_text/refund laid into the L256 bucket (round 4's a_at_L256, round 5's L256 sets)
        assert set(sets) == set(TIMING_SETS), (L, sorted(sets))
        fdir = STAGE / g["fx"]
        if fdir.exists():
            raise FileExistsError(f"refusing to overwrite {rel(fdir)}")
        tmpdir = fdir.with_name(fdir.name + ".tmp")
        tmpdir.mkdir(parents=True)
        man = []
        for i, r in enumerate(rows):
            id3 = f"{i:03d}"
            x = row_arrays(r, L)
            for name in NAMES:
                assert list(x[name].shape) == shapes[name], (r["key"], name, x[name].shape, shapes[name])
                (tmpdir / f"{id3}_{name}.f32").write_bytes(file_bytes(name, x[name]))
            n = len(r["ids"])
            man.append({"id3": id3, "key": r["key"], "role": r["role"], "P": int(r.get("P") or 0), "n": n,
                        "K": int(r["K"]), "markers": [int(m) for m in r["markers"]], "qtype": int(r["qtype"]),
                        "prefix_file": r.get("prefix_file")})
        os.replace(tmpdir, fdir)
        csha, nfiles, nbytes = concat_sha(fdir)
        shalines = "".join(f"{sha256(p)} {p.name}\n" for p in sorted(fdir.iterdir(), key=lambda q: q.name.encode()))
        write_new(R10 / f"{g['fx']}.sha256", shalines)
        idx = {m["key"]: m["id3"] for m in man}
        doc = {"step": "round 10: the C-API runner's fixtures (scripts/npu_fixtures.py write)",
               "written": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "L": L, "dir": rel(fdir),
               "rows_file": spec["rows"], "rows_sha256": sha256(K / spec["rows"]),
               "timing_rows_file": "device/timing_rows.json", "timing_rows_sha256": sha256(K / "device/timing_rows.json"),
               "order": order, "file_name": "<id3>_<input name>.f32; ids = int32 little-endian, the rest float32",
               "gate_count": sum(1 for m in man if m["role"] == "gate"), "count": len(man),
               "ab_fixture": idx[sets["a_one_question"][0]], "set_a": sets["a_one_question"],
               "set_b": sets["b_three_questions"], "set_b_fixtures": [idx[k] for k in sets["b_three_questions"]],
               "files": nfiles, "bytes": nbytes, "concat_sha256_bytesorted": csha,
               "per_file_sha256": rel(R10 / f"{g['fx']}.sha256"), "rows": man}
        write_new(R10 / f"{g['fx']}.manifest.json", json.dumps(doc, indent=1) + "\n")
        out[L] = {"dir": rel(fdir), "count": len(man), "gate": doc["gate_count"], "files": nfiles, "bytes": nbytes,
                  "concat_sha256": csha[:16], "ab_fixture": doc["ab_fixture"], "set_b_fixtures": doc["set_b_fixtures"],
                  "order": [x["name"] for x in order["inputs_subgraph_order"]]}
    print(json.dumps(out, indent=1))
    return 0


def read_fixture(fdir: Path, id3: str, order: list) -> dict:
    """The runner's read: per input in subgraph order, <id3>_<name>.f32 must hold exactly the tensor's bytes."""
    x = {}
    for t in order:
        p = fdir / f"{id3}_{t['name']}.f32"
        raw = p.read_bytes()
        elems = int(np.prod(t["shape"]))
        assert len(raw) == elems * 4, f"{p.name}: {len(raw)} bytes, the tensor needs {elems * 4}"
        dt = "<i4" if t["dtype"] == "INT32" else "<f4"
        assert t["dtype"] in ("INT32", "FLOAT32"), t
        x[t["name"]] = np.frombuffer(raw, dtype=dt).reshape(t["shape"]).astype(np.int32 if dt == "<i4" else np.float32)
    return x


def sel_slices(report_path: Path, sel_path: Path) -> dict:
    rep = json.loads(report_path.read_text())
    raw = np.fromfile(sel_path, dtype="<f4")
    out, off = {}, 0
    for r in rep["rows"]:
        k = int(r["K"])
        out[r["key"]] = raw[off: off + k].copy()
        off += k
    assert off == raw.size, (sel_path.name, off, raw.size)
    return {"rows": out, "graph": rep.get("graph"), "stand_in": rep.get("stand_in"), "threads": rep.get("threads"),
            "status": rep.get("status")}


def check(a) -> int:
    import importlib.metadata as md

    import litert_run as LR

    doc = {"step": "round 10: the runner fixtures on the Mac CPU vs the earlier Mac CPU runs of the same graph files "
                   "(scripts/npu_fixtures.py check)",
           "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "ai_edge_litert": md.version("ai-edge-litert"),
           "numpy": np.__version__, "python": sys.version.split()[0], "graphs": {}}
    out_path = K / "results/s26_npu_fixture_check_r10.json"
    if out_path.exists():
        raise FileExistsError(f"refusing to overwrite {rel(out_path)}")
    log_dir = K / "logs"
    all_ok = True
    for gtag in a.graphs:
        g = GRAPHS[gtag]
        L = g["L"]
        gpath = K / g["file"]
        assert gpath.stat().st_size == g["bytes"], (gtag, gpath.stat().st_size)
        gsha = sha256(gpath)
        assert gsha == g["sha256"], (gtag, gsha)
        man = json.loads((R10 / f"{g['fx']}.manifest.json").read_text())
        fdir = K / man["dir"]
        csha, nfiles, nbytes = concat_sha(fdir)
        assert csha == man["concat_sha256_bytesorted"], (gtag, "the fixture dir changed since write")
        order_here = io_order(gpath)
        order = order_here["inputs_subgraph_order"]
        same_order = [x["name"] for x in order] == [x["name"] for x in man["order"]["inputs_subgraph_order"]]
        rows = man["rows"][: g["count"]] if g["count"] else man["rows"]
        with np.load(K / g["store"]) as z:
            store = {k: z[k] for k in z.files}
        sels = {}
        for label, pat in g["sels"].items():
            rp, sp = K / pat.format("", ".json"), K / pat.format("sel_", ".f32")
            if rp.exists() and sp.exists():
                sels[label] = sel_slices(rp, sp) | {"report": rel(rp), "sel": rel(sp), "sel_sha256": sha256(sp)}
        mdir = R10 / "mac" / gtag / "model0"
        mdir.mkdir(parents=True, exist_ok=True)
        log = log_dir / f"r10_fxcheck_{gtag}.runtime.log"
        res = {"file": g["file"], "bytes": g["bytes"], "sha256": gsha, "fixtures_dir": man["dir"],
               "fixtures_concat_sha256": csha, "fixtures_files": nfiles, "rows_run": len(rows),
               "order_same_as_manifest": same_order, "inputs_subgraph_order": order,
               "inputs_signaturedef_stored_order": order_here["inputs_signaturedef_stored_order"],
               "operators": order_here["operators"]}
        per, host_eq, st_eq, st_n, st_maxd = [], 0, 0, 0, 0.0
        sel_eq = {k: [0, 0] for k in sels}
        with LR.capture_fd2(log):
            res["logger"] = LR.runtime_log_verbose()
            t0 = time.time()
            cm, desc = LR.open_compiled(gpath, "cpu", threads=8)
            res["compile_seconds"] = round(time.time() - t0, 2)
            res["options"] = desc
            sig = next(iter(cm.get_signature_list()))
            assert sig == f"decide_{L}", sig
            run = LR.Runner(cm, sig)
            assert run.L == L
            rows_doc = {r["key"]: r for r in json.loads((K / man["rows_file"]).read_text())["rows"]}
            trows = {r["key"]: r for s in json.loads((K / "device/timing_rows.json").read_text())["sets"]
                     for r in s["rows"]}
            for m in rows:
                x = read_fixture(fdir, m["id3"], order)
                src = rows_doc.get(m["key"]) or trows[m["key"]]
                ref = row_arrays(src, L)
                heq = all(np.array_equal(x[n], ref[n]) and x[n].dtype == ref[n].dtype for n in NAMES)
                host_eq += heq
                s = run(x)
                s.astype("<f4").tofile(mdir / f"{m['id3']}.f32")
                real = m["P"] + m["n"]
                e = {"id3": m["id3"], "key": m["key"], "role": m["role"], "P": m["P"], "n": m["n"],
                     "fixture_equals_host_build_inputs": heq, "nonfinite": int((~np.isfinite(s)).sum())}
                sk = m["key"].replace("/", "__")
                if sk in store:
                    st_n += 1
                    ref_s = store[sk]
                    ok = ref_s.shape == (real,) and np.array_equal(s[:real], ref_s)
                    st_eq += ok
                    d = float(np.abs(s[:real].astype(np.float64) - ref_s.astype(np.float64)).max()) if ref_s.shape == (real,) else None
                    st_maxd = max(st_maxd, d or 0.0)
                    e["store_bit_equal"], e["store_max_abs_d"] = bool(ok), d
                else:
                    e["store_bit_equal"] = None
                mk = s[[m["P"] + mm for mm in m["markers"][: m["K"]]]]
                for label, sv in sels.items():
                    if m["key"] in sv["rows"]:
                        sel_eq[label][1] += 1
                        eq = bool(np.array_equal(mk, sv["rows"][m["key"]]))
                        sel_eq[label][0] += eq
                        e[f"{label}_markers_bit_equal"] = eq
                per.append(e)
            run.close()
            del cm
        res["delegation"] = LR.delegation_from_log(log)
        res.update(fixtures_equal_host_build_inputs=f"{host_eq}/{len(rows)}",
                   store={"file": g["store"], "rows_compared": st_n, "rows_bit_equal": st_eq,
                          "max_abs_d_real_positions": st_maxd},
                   gate_app_stand_in_markers={k: {"report": v["report"], "sel": v["sel"], "sel_sha256": v["sel_sha256"],
                                                  "graph": v["graph"], "threads": v["threads"],
                                                  "rows_compared": sel_eq[k][1], "rows_bit_equal": sel_eq[k][0]}
                                              for k, v in sels.items()},
                   nonfinite_rows=sum(1 for e in per if e["nonfinite"]), mac_scores_dir=rel(mdir.parent),
                   per_row=per)
        ok = (same_order and host_eq == len(rows) and st_n == len(rows) and st_eq == st_n
              and all(v[0] == v[1] and v[1] > 0 for v in sel_eq.values()) and res["nonfinite_rows"] == 0)
        res["pass"] = bool(ok)
        all_ok &= ok
        doc["graphs"][gtag] = res
        print(f"{gtag}: rows {len(rows)} host {host_eq}/{len(rows)} store {st_eq}/{st_n} "
              + " ".join(f"{k} {v[0]}/{v[1]}" for k, v in sel_eq.items())
              + f" order {[x['name'] for x in order]} compile {res['compile_seconds']} s pass {ok}", flush=True)
    doc["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    doc["pass"] = bool(all_ok)
    write_new(out_path, json.dumps(doc, indent=1) + "\n")
    print(f"check: {'PASS' if all_ok else 'FAIL'} -> {rel(out_path)}")
    return 0 if all_ok else 1


# ================================================================ round 12: any graph, any fixture dir
# (the round 10 subcommands above are unchanged). A fixture dir is named by its spec; its manifest carries the kind
# (text | audio | vision_tower | projector), the subgraph input order of the graph it was made for, the rows, the
# timing fixture (ab_fixture), the element count of output 0 (out_elems: the runner's --dump full writes that many
# float32 per fixture) and the concatenated sha256 the chain checks on the phone.
#   $PY scripts/npu_fixtures.py write-specs --round r12      -> device/r12/stage/<fx>/ + device/r12/<fx>.{manifest.json,sha256}
#   $Q/quiet_wait.py -- $PY scripts/npu_fixtures.py check-specs --round r12
#       -> results/s26_npu_fixture_check_r12.json + device/r12/mac/<graph tag>/model0/<id3>.f32 (Mac CPU, 8 threads)
# check-specs: every graph listed for a fixture dir runs every fixture on the Mac CPU, read the way the runner reads it
# (subgraph order, the byte count each tensor needs), and the outputs are compared with the earlier Mac CPU runs:
# text = round 8's store (out/r8_runs/cpu_L<L>_f16safe_fp16.npz, real positions) and round 9's Mac gate-app stand-in
# (device/r9/mac, marker scores); audio = round 7's store (out/r7_runs/audio_cpu_fp16[_T3001].npz, the host mel, the
# current file; the f16safe file gives the same bits on the CPU, round 12 step 1); vision tower = round 9's Mac generic
# run of the same crops (device/r9/mac/out_mac_d1omni_vision_tower_fp16__g9_vt.f32); projector = the record prefixes of
# round 6's Mac CPU chain (out/r6_runs/cpu_fp32_fp16/<record>.npy). The fixture inputs themselves are compared with the
# host (text: build_inputs) and with round 9's stacked generic inputs where they exist (device/g9_vt_*, g9_au_*).

SPECS = {
    "r12": {
        "fx_L512": {"kind": "text", "L": 512, "rows": "device/rows_L512.json", "ab_key": "tv4_009/answer",
                    "graphs": {"L512_f16safe_fp16": "out/d1omni_decide_L512_f16safe_fp16.tflite"},
                    "store": {"L512_f16safe_fp16": "out/r8_runs/cpu_L512_f16safe_fp16.npz"},
                    "sel": {"L512_f16safe_fp16": "device/r9/mac/{}mac_d1omni_decide_L512_f16safe_fp16__rows_L512{}"}},
        # 20:5x: L512's 8 rows laid into the L1024 bucket (no fixture row has 513..1,024 positions); the scores
        # at the real positions must equal round 11's Mac CPU store of the same file and round 8's of the L512 file
        "fx_L1024": {"kind": "text", "L": 1024, "rows": "device/rows_L512.json", "pack": True, "ab_key": "tv4_009/answer",
                     "graphs": {"L1024_f16safe_fp16": "out/d1omni_decide_L1024_f16safe_fp16.tflite"},
                     "store": {"L1024_f16safe_fp16": "out/r11_runs/cpu_L1024_f16safe_fp16.npz"},
                     "store2": {"L1024_f16safe_fp16": "out/r8_runs/cpu_L512_f16safe_fp16.npz"},
                     "sel": {"L1024_f16safe_fp16": "device/r9/mac/{}mac_d1omni_decide_L512_f16safe_fp16__rows_L512{}"}},
        "fx_L2048": {"kind": "text", "L": 2048, "rows": "device/rows_L2048.json", "ab_key": "own_long_log_10/resolved",
                     "graphs": {"L2048_f16safe_fp16": "out/d1omni_decide_L2048_f16safe_fp16.tflite"},
                     "store": {"L2048_f16safe_fp16": "out/r8_runs/cpu_L2048_f16safe_fp16.npz"},
                     "sel": {"L2048_f16safe_fp16": "device/r9/mac/{}mac_d1omni_decide_L2048_f16safe_fp16__rows_L2048{}"}},
        "fx_A1001": {"kind": "audio", "T": 1001, "ab_key": "aud_01",
                     "graphs": {"audio_T1001_f16safe_fp16": "out/d1omni_audio_T1001_f16safe_fp16.tflite",
                                "audio_T1001_fp16": "out/d1omni_audio_T1001_fp16.tflite",
                                "audio_T1001_clast_fp16": "out/d1omni_audio_T1001_clast_fp16.tflite"}},
        "fx_A501": {"kind": "audio", "T": 501, "ab_key": "aud_01_cut5",
                    "graphs": {"audio_T501_fp16": "out/d1omni_audio_T501_fp16.tflite",
                               "audio_T501_f16safe_fp16": "out/d1omni_audio_T501_f16safe_fp16.tflite"}},
        # the second slot's T2001 legs (card_topic, 10.4 s; the oracle's 19th audio row)
        "fx_A2001": {"kind": "audio", "T": 2001, "ab_key": "card_topic",
                     "graphs": {"audio_T2001_f16safe_fp16": "out/d1omni_audio_T2001_f16safe_fp16.tflite",
                                "audio_T2001_fp16": "out/d1omni_audio_T2001_fp16.tflite"}},
        "fx_A3001": {"kind": "audio", "T": 3001, "ab_key": "long",
                     "graphs": {"audio_T3001_fp16": "out/d1omni_audio_T3001_fp16.tflite",
                                "audio_T3001_f16safe_fp16": "out/d1omni_audio_T3001_f16safe_fp16.tflite"}},
        "fx_VT": {"kind": "vision_tower", "ab_key": "img_dogs_01/c0",
                  "graphs": {"vision_tower_f16safe_fp16": "out/d1omni_vision_tower_f16safe_fp16.tflite",
                             "vision_tower_fp16": "out/d1omni_vision_tower_fp16.tflite"}},
        "fx_PJ": {"kind": "projector", "ab_key": "img_dogs_01/c0", "soft_from": "out/d1omni_vision_tower_fp16.tflite",
                  "graphs": {"projector_fp16": "out/d1omni_projector_fp16.tflite"}},
    }
}
AUDIO_T1001_CLIPS = ("aud_01", "aud_02", "aud_03", "aud_reservation_01", "aud_weather_02", "aud_food_03")


def io_order_any(path: Path) -> dict:
    """io_order() without the decision graph's name asserts: the subgraph's inputs / outputs in order."""
    from ai_edge_litert import schema_py_generated as schema

    tname = {v: k for k, v in vars(schema.TensorType).items() if not k.startswith("_")}
    with open(path, "rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    model = schema.Model.GetRootAs(mm, 0)
    assert model.SignatureDefsLength() == 1, model.SignatureDefsLength()
    sd = model.SignatureDefs(0)
    sg = model.Subgraphs(sd.SubgraphIndex())
    stored_in = [(sd.Inputs(i).Name().decode(), int(sd.Inputs(i).TensorIndex())) for i in range(sd.InputsLength())]
    stored_out = [(sd.Outputs(i).Name().decode(), int(sd.Outputs(i).TensorIndex())) for i in range(sd.OutputsLength())]
    by_t_in, by_t_out = {t: n for n, t in stored_in}, {t: n for n, t in stored_out}

    def desc(i, t, name):
        tt = sg.Tensors(int(t))
        return {"index": i, "tensor_index": int(t), "name": name, "tensor_name": tt.Name().decode(),
                "shape": [int(x) for x in tt.ShapeAsNumpy()], "dtype": tname.get(tt.Type(), str(tt.Type()))}

    ins = [desc(i, t, by_t_in[int(t)]) for i, t in enumerate(sg.InputsAsNumpy())]
    outs = [desc(i, t, by_t_out[int(t)]) for i, t in enumerate(sg.OutputsAsNumpy())]
    key, sgi, nops = sd.SignatureKey().decode(), int(sd.SubgraphIndex()), int(sg.OperatorsLength())
    del model, sd, sg
    mm.close()
    return {"file": rel(path), "signature": key, "subgraph": sgi, "operators": nops,
            "inputs_subgraph_order": ins, "outputs_subgraph_order": outs,
            "inputs_signaturedef_stored_order": [{"name": n, "tensor_index": t} for n, t in stored_in],
            "note": "the C API (the S26 runner) numbers a signature's inputs / outputs in the subgraph order"}


def _audio_inputs(T: int):
    """-> [(key, record, inputs, info)] of an audio bucket's fixtures (host float32 mel, round 7's clips)."""
    import audio_graph as AG
    import d1_audio_host as AH

    out = []
    if T == 1001:
        for rid in AUDIO_T1001_CLIPS:
            x16, src = AG.clip_samples(rid)
            x, info = AH.prepare(x16, bucket=1001)
            out.append((rid, rid, x, {**info, "source": src}))
    elif T == 2001:
        x16, src = AG.clip_samples("card_topic")
        x, info = AH.prepare(x16, bucket=2001)
        out.append(("card_topic", "card_topic", x, {**info, "source": src}))
    elif T == 501:
        x16, src = AG.clip_samples("aud_01", seconds=5.0)
        x, info = AH.prepare(x16)
        out.append(("aud_01_cut5", None, x, {**info, "source": src, "seconds_cut": 5.0}))
    elif T == 3001:
        x16, name = AG.long_samples()
        x, info = AH.prepare(x16)
        out.append(("long", None, x, {**info, "source": name}))
    for _, _, x, info in out:
        assert info["T_b"] == T, (T, info)
    return out


def _vision_crops():
    """-> [(key, record, crop index, tower inputs, grid)] of the 13 crops of the 7 image records (host)."""
    import d1_vision_host as V
    import vision_check as VC

    table = V.read_position_table(__import__("d1_src").WEIGHTS)
    out = []
    recs, _ = VC.image_records()
    for e, rec in recs:
        crops, _ = V.crops_of(V.load_image(VC.image_path(rec)))
        for i, c in enumerate(crops):
            crop = V.to_patches(c)
            out.append((f"{e['id']}/c{i}", e["id"], i, V.tower_inputs(crop, table), tuple(int(v) for v in crop["grid"])))
    return out


def _write_dir(fdir: Path, fixtures: list, order: list) -> tuple:
    """fixtures = [(id3, {name: array})]: <id3>_<name>.f32 per input (int32 ids, float32 the rest), shapes asserted
    against the subgraph order. -> (concat sha256, files, bytes)."""
    if fdir.exists():
        raise FileExistsError(f"refusing to overwrite {rel(fdir)}")
    tmpdir = fdir.with_name(fdir.name + ".tmp")
    tmpdir.mkdir(parents=True)
    shapes = {x["name"]: x["shape"] for x in order}
    dtypes = {x["name"]: x["dtype"] for x in order}
    for id3, x in fixtures:
        assert sorted(x) == sorted(shapes), (id3, sorted(x), sorted(shapes))
        for name, arr in x.items():
            assert list(arr.shape) == shapes[name], (id3, name, arr.shape, shapes[name])
            want = np.int32 if dtypes[name] == "INT32" else np.float32
            assert arr.dtype == want, (id3, name, arr.dtype)
            (tmpdir / f"{id3}_{name}.f32").write_bytes(np.ascontiguousarray(arr).astype(
                "<i4" if want == np.int32 else "<f4", copy=False).tobytes())
    os.replace(tmpdir, fdir)
    return concat_sha(fdir)


def write_specs(a) -> int:
    rd = K / "device" / a.round
    stage = rd / "stage"
    specs = SPECS[a.round]
    summary = {}
    for fx, sp in specs.items():
        if a.only and fx not in a.only:
            continue
        man_path = rd / f"{fx}.manifest.json"
        if man_path.exists():
            raise FileExistsError(f"refusing to overwrite {rel(man_path)}")
        g0 = K / next(iter(sp["graphs"].values()))
        order = io_order_any(g0)
        oshape = order["outputs_subgraph_order"][0]["shape"]
        rows, fixtures = [], []
        if sp["kind"] == "text":
            L = sp["L"]
            rows_doc = json.loads((K / sp["rows"]).read_text())
            assert (int(rows_doc["L"]) == L or (sp.get("pack") and int(rows_doc["L"]) < L)) and int(rows_doc["hidden"]) == D \
                and int(rows_doc["pad_id"]) == 0
            for i, r in enumerate(rows_doc["rows"]):
                id3 = f"{i:03d}"
                fixtures.append((id3, row_arrays(r, L)))
                rows.append({"id3": id3, "key": r["key"], "role": "gate", "P": int(r.get("P") or 0), "n": len(r["ids"]),
                             "K": int(r["K"]), "markers": [int(m) for m in r["markers"]], "qtype": int(r["qtype"]),
                             "prefix_file": r.get("prefix_file")})
        elif sp["kind"] == "audio":
            for i, (key, rid, x, info) in enumerate(_audio_inputs(sp["T"])):
                id3 = f"{i:03d}"
                fixtures.append((id3, {k: np.ascontiguousarray(v, np.float32) for k, v in x.items()}))
                rows.append({"id3": id3, "key": key, "record": rid, "role": "gate", "P": int(info["P"]),
                             "frames": int(info["frames"]), "T_b": int(info["T_b"]), "source": info.get("source"),
                             "seconds_cut": info.get("seconds_cut")})
        elif sp["kind"] == "vision_tower":
            for i, (key, rid, c, x, grid) in enumerate(_vision_crops()):
                id3 = f"{i:03d}"
                fixtures.append((id3, {k: np.ascontiguousarray(v, np.float32) for k, v in x.items()}))
                rows.append({"id3": id3, "key": key, "record": rid, "crop": c, "role": "gate", "grid": list(grid),
                             "patches": grid[0] * grid[1]})
        elif sp["kind"] == "projector":
            import d1_vision_host as V
            import litert_run as LR

            cm, desc = LR.open_compiled(K / sp["soft_from"], "cpu", threads=8)
            sig = next(iter(cm.get_signature_list()))
            ins = {n: cm.create_input_buffer_by_name(sig, n) for n in cm.get_input_tensor_details(sig)}
            outs = {n: cm.create_output_buffer_by_name(sig, n) for n in cm.get_output_tensor_details(sig)}
            for i, (key, rid, c, x, grid) in enumerate(_vision_crops()):
                for n, v in x.items():
                    ins[n].write(np.ascontiguousarray(v, np.float32))
                cm.run_by_name(sig, ins, outs)
                feat = np.asarray(outs["features"].read(1024 * 768, np.float32), np.float32).reshape(1024, 768)
                cells = V.pixel_unshuffle(feat[: grid[0] * grid[1]], grid)
                id3 = f"{i:03d}"
                fixtures.append((id3, {"soft": V.projector_input(cells)}))
                rows.append({"id3": id3, "key": key, "record": rid, "crop": c, "role": "gate", "grid": list(grid),
                             "prefix_rows": int(cells.shape[0])})
            for b in list(ins.values()) + list(outs.values()):
                b.destroy()
        fdir = stage / fx
        csha, nfiles, nbytes = _write_dir(fdir, fixtures, order["inputs_subgraph_order"])
        shalines = "".join(f"{sha256(p)} {p.name}\n" for p in sorted(fdir.iterdir(), key=lambda q: q.name.encode()))
        write_new(rd / f"{fx}.sha256", shalines)
        ab = next(r["id3"] for r in rows if r["key"] == sp["ab_key"])
        doc = {"step": f"round {a.round[1:]}: the C-API runner's fixtures (scripts/npu_fixtures.py write-specs)",
               "written": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "kind": sp["kind"], "fx": fx,
               "L": int(sp.get("L") or 0), "T": sp.get("T"), "out_elems": int(np.prod(oshape)), "out_shape": oshape,
               "dir": rel(fdir), "order": order, "graphs": sp["graphs"],
               "file_name": "<id3>_<input name>.f32; ids = int32 little-endian, the rest float32",
               "rows_file": sp.get("rows"), "rows_sha256": sha256(K / sp["rows"]) if sp.get("rows") else None,
               "gate_count": len(rows), "count": len(rows), "ab_fixture": ab, "ab_key": sp["ab_key"],
               "set_b": [], "set_b_fixtures": [], "files": nfiles, "bytes": nbytes, "concat_sha256_bytesorted": csha,
               "per_file_sha256": rel(rd / f"{fx}.sha256"), "rows": rows}
        if sp["kind"] == "projector":
            doc["soft_from"] = {"file": sp["soft_from"], "sha256": sha256(K / sp["soft_from"]),
                                "how": "the Mac CPU (XNNPACK, 8 threads) tower features of each crop -> the host unshuffle "
                                       "(host/d1_vision_host.py pixel_unshuffle + projector_input)"}
        write_new(man_path, json.dumps(doc, indent=1) + "\n")
        summary[fx] = {"kind": sp["kind"], "count": len(rows), "files": nfiles, "bytes": nbytes, "concat_sha256": csha[:16],
                       "ab_fixture": ab, "order": [x["name"] for x in order["inputs_subgraph_order"]],
                       "out_elems": doc["out_elems"]}
    print(json.dumps(summary, indent=1))
    return 0


def _stacked(path: Path, index: int, shape) -> np.ndarray:
    n = int(np.prod(shape))
    with open(path, "rb") as f:
        f.seek(index * n * 4)
        return np.frombuffer(f.read(n * 4), dtype="<f4").reshape(shape)


def check_specs(a) -> int:
    import importlib.metadata as md

    import litert_run as LR

    rd = K / "device" / a.round
    # a later addition (--only) gets its own file next to the round's check (the first one is never overwritten)
    out_path = K / (f"results/s26_npu_fixture_check_{a.round}_{'_'.join(a.only)}.json" if a.only
                    else f"results/s26_npu_fixture_check_{a.round}.json")
    if out_path.exists():
        raise FileExistsError(f"refusing to overwrite {rel(out_path)}")
    doc = {"step": f"round {a.round[1:]}: the runner fixtures on the Mac CPU (XNNPACK, 8 threads) vs the earlier Mac CPU "
                   "runs and the host (scripts/npu_fixtures.py check-specs)",
           "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "ai_edge_litert": md.version("ai-edge-litert"),
           "numpy": np.__version__, "python": sys.version.split()[0], "fixture_dirs": {}}
    all_ok = True
    vt_stack = {n: K / f"device/g9_vt_{n}.f32" for n in ("pixels", "pos", "mask")}
    au_stack = {n: K / f"device/g9_au_{n}.f32" for n in ("mel", "mel_valid", "v1", "v2", "v3")}
    for fx, sp in SPECS[a.round].items():
        if a.only and fx not in a.only:
            continue
        man = json.loads((rd / f"{fx}.manifest.json").read_text())
        fdir = K / man["dir"]
        csha, nfiles, _ = concat_sha(fdir)
        assert csha == man["concat_sha256_bytesorted"], (fx, "the fixture dir changed since write")
        order_m = man["order"]["inputs_subgraph_order"]
        res = {"kind": man["kind"], "manifest": rel(rd / f"{fx}.manifest.json"), "fixtures": man["count"],
               "concat_sha256": csha, "graphs": {}}
        # the inputs against the host / round 9's stacked inputs
        inputs_eq = []
        if man["kind"] == "text":
            rows_doc = {r["key"]: r for r in json.loads((K / man["rows_file"]).read_text())["rows"]}
            for m in man["rows"]:
                x = read_fixture(fdir, m["id3"], order_m)
                ref = row_arrays(rows_doc[m["key"]], man["L"])
                inputs_eq.append(all(np.array_equal(x[n], ref[n]) and x[n].dtype == ref[n].dtype for n in NAMES))
            res["inputs_vs"] = "host build_inputs (host/d1_host.py) of the rows file"
        elif man["kind"] == "vision_tower":
            g9 = json.loads((K / "device/g9_vt.json").read_text())
            idx = {r["key"]: r["index"] for r in g9["rows"]}
            shapes = {i["name"]: i["shape"] for i in g9["inputs"]}
            for m in man["rows"]:
                x = read_fixture(fdir, m["id3"], order_m)
                inputs_eq.append(all(np.array_equal(x[n], _stacked(vt_stack[n], idx[m["key"]], shapes[n])) for n in x))
            res["inputs_vs"] = "round 9's stacked tower inputs (device/g9_vt_*.f32 = the provider preprocess bit for bit)"
        elif man["kind"] == "audio" and man["T"] == 1001:
            g9 = json.loads((K / "device/g9_au.json").read_text())
            idx = {r["key"]: r["index"] for r in g9["rows"]}
            shapes = {i["name"]: i["shape"] for i in g9["inputs"]}
            for m in man["rows"]:
                x = read_fixture(fdir, m["id3"], order_m)
                inputs_eq.append(all(np.array_equal(x[n], _stacked(au_stack[n], idx[m["key"]], shapes[n])) for n in x))
            res["inputs_vs"] = "round 9's stacked audio inputs (device/g9_au_*.f32, host float32 mel)"
        if inputs_eq:
            res["inputs_equal"] = f"{sum(inputs_eq)}/{len(inputs_eq)}"
        ok = all(inputs_eq)
        for gtag, gfile in man["graphs"].items():
            gpath = K / gfile
            if not gpath.exists():
                res["graphs"][gtag] = {"file": gfile, "missing": True}
                continue
            order = io_order_any(gpath)
            same_order = [x["name"] for x in order["inputs_subgraph_order"]] == [x["name"] for x in order_m]
            mdir = rd / "mac" / gtag / "model0"
            mdir.mkdir(parents=True, exist_ok=True)
            log = K / f"logs/{a.round}_fxcheck_{gtag}.runtime.log"
            g = {"file": gfile, "bytes": gpath.stat().st_size, "sha256": sha256(gpath), "order_same_as_manifest": same_order,
                 "inputs_subgraph_order": [x["name"] for x in order["inputs_subgraph_order"]],
                 "inputs_signaturedef_stored_order": [x["name"] for x in order["inputs_signaturedef_stored_order"]],
                 "operators": order["operators"]}
            outs_all = {}
            with LR.capture_fd2(log):
                g["logger"] = LR.runtime_log_verbose()
                t0 = time.time()
                cm, desc = LR.open_compiled(gpath, "cpu", threads=8)
                g["compile_seconds"] = round(time.time() - t0, 2)
                sig = next(iter(cm.get_signature_list()))
                ins = {n: cm.create_input_buffer_by_name(sig, n) for n in cm.get_input_tensor_details(sig)}
                oname = order["outputs_subgraph_order"][0]["name"]
                outs = {oname: cm.create_output_buffer_by_name(sig, oname)}
                n_out = int(np.prod(order["outputs_subgraph_order"][0]["shape"]))
                for m in man["rows"]:
                    x = read_fixture(fdir, m["id3"], order["inputs_subgraph_order"])
                    for n, v in x.items():
                        ins[n].write(np.ascontiguousarray(v))
                    cm.run_by_name(sig, ins, outs)
                    o = np.asarray(outs[oname].read(n_out, np.float32), np.float32).copy()
                    o.astype("<f4").tofile(mdir / f"{m['id3']}.f32")
                    outs_all[m["id3"]] = o
                for b in list(ins.values()) + list(outs.values()):
                    b.destroy()
                del cm
            g["delegation"] = LR.delegation_from_log(log)
            g["nonfinite_fixtures"] = sum(int(not np.isfinite(o).all()) for o in outs_all.values())
            cmp_ = []
            if man["kind"] == "text":
                sp_store = sp.get("store", {}).get(gtag)
                st = dict(np.load(K / sp_store)) if sp_store else {}
                sp_store2 = sp.get("store2", {}).get(gtag)
                st2 = dict(np.load(K / sp_store2)) if sp_store2 else {}
                pat = sp.get("sel", {}).get(gtag)
                sels = sel_slices(K / pat.format("", ".json"), K / pat.format("sel_", ".f32")) if pat else None
                for m in man["rows"]:
                    s = outs_all[m["id3"]]
                    real = m["P"] + m["n"]
                    e = {"id3": m["id3"], "key": m["key"]}
                    sk = m["key"].replace("/", "__")
                    if sk in st:
                        e["store_bit_equal"] = bool(st[sk].shape == (real,) and np.array_equal(s[:real], st[sk]))
                    if sk in st2:     # the smaller bucket's file (bucket independence on the CPU, round 4)
                        e["store2_bit_equal"] = bool(st2[sk].shape == (real,) and np.array_equal(s[:real], st2[sk]))
                    if sels and m["key"] in sels["rows"]:
                        mk = s[[m["P"] + mm for mm in m["markers"][: m["K"]]]]
                        e["r9_mac_markers_bit_equal"] = bool(np.array_equal(mk, sels["rows"][m["key"]]))
                    cmp_.append(e)
                g["references"] = {"store": sp_store, "store2": sp_store2, "r9_mac_sel": pat.format("sel_", ".f32") if pat else None}
            elif man["kind"] == "audio":
                T = man["T"]
                st_path = K / ("out/r7_runs/audio_cpu_fp16_T3001.npz" if T == 3001 else "out/r7_runs/audio_cpu_fp16.npz")
                st = dict(np.load(st_path))
                key_of = {1001: lambda m: f"{m['key']}__host", 2001: lambda m: f"{m['key']}__host",
                          501: lambda m: "aud_01_cut5__T501", 3001: lambda m: "long__host"}[T]
                for m in man["rows"]:
                    o = outs_all[m["id3"]].reshape(man["out_shape"])[0][: m["P"]]
                    ref = st[key_of(m)]
                    cmp_.append({"id3": m["id3"], "key": m["key"], "store_bit_equal": bool(np.array_equal(o, ref)),
                                 "store_max_abs": float(np.abs(o.astype(np.float64) - ref).max())})
                g["references"] = {"store": rel(st_path), "note": "round 7's Mac CPU store of the current fp16 file, host "
                                                                  "mel (the f16safe and clast files gave the same bits on "
                                                                  "the CPU in round 12 step 1-3)"}
            elif man["kind"] == "vision_tower":
                g9 = json.loads((K / "device/g9_vt.json").read_text())
                idx = {r["key"]: r["index"] for r in g9["rows"]}
                r9 = K / "device/r9/mac/out_mac_d1omni_vision_tower_fp16__g9_vt.f32"
                for m in man["rows"]:
                    o = outs_all[m["id3"]].reshape(1024, 768)
                    ref = _stacked(r9, idx[m["key"]], (1, 1024, 768))[0]
                    cmp_.append({"id3": m["id3"], "key": m["key"], "r9_mac_bit_equal_all_rows": bool(np.array_equal(o, ref)),
                                 "r9_mac_bit_equal_real_rows": bool(np.array_equal(o[: m["patches"]], ref[: m["patches"]]))})
                g["references"] = {"r9_mac": rel(r9)}
            elif man["kind"] == "projector":
                by_rec = {}
                for m in man["rows"]:
                    o = outs_all[m["id3"]].reshape(256, 1024)[: m["prefix_rows"]]
                    by_rec.setdefault(m["record"], []).append((m["crop"], o))
                for rid, parts in by_rec.items():
                    pre = np.concatenate([p for _, p in sorted(parts, key=lambda t: t[0])]).astype(np.float32)
                    ref = np.load(K / f"out/r6_runs/cpu_fp32_fp16/{rid}.npy")
                    cmp_.append({"record": rid, "r6_store_bit_equal": bool(pre.shape == ref.shape and np.array_equal(pre, ref))})
                g["references"] = {"r6_store": "out/r6_runs/cpu_fp32_fp16/<record>.npy (round 6's Mac CPU chain)"}
            g["comparisons"] = cmp_
            flags = [v for e in cmp_ for k, v in e.items() if k.endswith("bit_equal") or k.endswith("bit_equal_real_rows")]
            g["bit_equal"] = f"{sum(bool(v) for v in flags)}/{len(flags)}"
            g["pass"] = bool(same_order and flags and all(flags) and g["nonfinite_fixtures"] == 0)
            g["mac_dir"] = rel(mdir.parent)
            ok &= g["pass"]
            res["graphs"][gtag] = g
            print(f"{fx} {gtag}: bit-equal {g['bit_equal']} order {g['inputs_subgraph_order']} pass {g['pass']}", flush=True)
        res["pass"] = bool(ok)
        all_ok &= ok
        doc["fixture_dirs"][fx] = res
    doc["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    doc["pass"] = bool(all_ok)
    write_new(out_path, json.dumps(doc, indent=1, default=str) + "\n")
    print(f"check-specs: {'PASS' if all_ok else 'FAIL'} -> {rel(out_path)}")
    return 0 if all_ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("write")
    w.add_argument("--L", type=int, nargs="+", default=[128, 256], choices=[128, 256])
    c = sub.add_parser("check")
    c.add_argument("--graphs", nargs="+", default=list(GRAPHS), choices=list(GRAPHS))
    for name in ("write-specs", "check-specs"):
        p = sub.add_parser(name)
        p.add_argument("--round", default="r12", choices=sorted(SPECS))
        p.add_argument("--only", nargs="*", default=[])
    a = ap.parse_args()
    if a.cmd == "write-specs":
        return write_specs(a)
    if a.cmd == "check-specs":
        return check_specs(a)
    return write(a) if a.cmd == "write" else check(a)


if __name__ == "__main__":
    sys.exit(main())
