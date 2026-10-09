"""LiteRT build of D1Decision (one single-signature file per L bucket), run in the exporter venv from K:

    cd d1_omni_work
    ~/venvs/lt094dev/bin/python scripts/graph_build.py --env                      # round 3 step 0: results/export_env.json
    ~/code/standup/tools/quiet/quiet_wait.py -- ~/venvs/lt094dev/bin/python scripts/graph_build.py --L 256   (step 1)
    ~/venvs/lt094dev/bin/python scripts/graph_build.py --dry-run --L 64 --weights random --sample random   (round 1)

Per L: D1Decision(text_config, head_layers, L) with the pinned checkpoint (`encoder.*` / `head.*` tensors only, mapped by
d1_graph.KEY_RULES, every parameter set), the trace sample = fixture row card_text/refund laid out by the host's
build_inputs (47 ids padded to L, P = 0, qtype noul), one eager forward, then
litert_torch.signature(f"decide_{L}", module, sample_kwargs=sample).convert().export(out/d1omni_decide_L{L}_fp32.tflite).
After the export (same process): the static scan (scripts/litert_run.scan) -> results/opscan_L{L}_fp32.json with the
acceptance checks (BROADCAST_TO 0, CUSTOM 0, INT64 tensors <= 1, GATHER_ND 0, EMBEDDING_LOOKUP 1, rank <= 4), the
signature I/O (flatbuffer order and CompiledModel details) -> results/signature_L{L}.json, and a smoke run of the new
file through CompiledModel CPU on the sample vs the eager scores (real positions). rank3_prelint runs separately
(tools/rank3_prelint.py needs only flatbuffers). Existing files are never overwritten.
The command line carries none of the quiet-window guard's words, so the guard cannot see it: wrap it in quiet_wait.py.
--dry-run (round 1): builds the module and the sample, runs the eager forward, prints, stops before signature().

Round 4 (additions; the round-3 calls above behave as before):
    ~/venvs/lt094dev/bin/python scripts/graph_build.py --env-append round_4          # adds key round_4 to export_env.json
    $Q/quiet_wait.py -- ~/venvs/lt094dev/bin/python scripts/graph_build.py --round 4 --L 4096
    $Q/quiet_wait.py -- ~/venvs/lt094dev/bin/python scripts/graph_build.py --round 4 --multi 128,256,512,1024,2048,4096
    ~/venvs/lt094dev/bin/python scripts/graph_build.py --contract replace --mac-recommendation <json>
--round only changes the log / step labels. --multi: ONE fp32 file with a signature decide_<L> per listed L
(out/d1omni_decide_multi6_fp32.tflite for the six buckets): one checkpoint load, and the per-L modules share the very
same parameter modules (embedding, trunk layers, final norm, head; only the per-L RoPE constants differ), so the file
can hold each weight once; litert_torch.signature(...).signature(...)...convert().export(). Same process: the scan
(constant bytes per dtype counted once per buffer vs per tensor), the six signatures' I/O, and a CPU smoke run of every
signature on its fixture sample vs the eager module -> results/multisig_export.json (never overwritten).

Round 8 (addition; every call above behaves as before):
    $Q/quiet_wait.py -- ~/venvs/lt094dev/bin/python scripts/graph_build.py --round 8 --f16safe --L 128
--f16safe swaps the module for d1_graph_f16safe.D1DecisionF16Safe (the per-site fp16-safe norm rewrite, k from
results/norm_range.json; same inputs, outputs and weights) -> out/d1omni_decide_L{L}_f16safe_fp32.tflite
(--f16safe-variant l0sum: the diagnostic form with trunk L00's two RMSNorms in the sum form -> *_f16safe_l0sum_*),
results/opscan_L{L}_f16safe_fp32.json (+ `f16safe`: the op histogram against the plain file's opscan, the expected
MUL increase = one MUL per scaled site, the signature against results/signature_L{L}.json) and
results/signature_L{L}_f16safe.json.
"""
import argparse
import importlib.metadata as md
import json
import platform
import resource
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import d1_src as S  # noqa: E402  (stdlib only at import; sets dont_write_bytecode)

import numpy as np  # noqa: E402
import torch  # noqa: E402

import d1_graph as G  # noqa: E402
import litert_run as R  # noqa: E402

sys.path.insert(0, str(S.K / "host"))
import d1_host as H  # noqa: E402

SAMPLE_ROW = ("card_text", "refund")


def _version(dist):
    try:
        return md.version(dist)
    except md.PackageNotFoundError:
        return None


def out_path(L, form="fp32"):
    return S.K / f"out/d1omni_decide_L{L}_{form}.tflite"


def load_checkpoint_subset():
    """encoder.* / head.* tensors of the pinned safetensors (vision / audio are not read)."""
    from safetensors import safe_open

    sd, dtypes, skipped = {}, {}, 0
    with safe_open(str(S.WEIGHTS), framework="pt") as f:
        for k in f.keys():
            if k.startswith(("encoder.", "head.")):
                t = f.get_tensor(k)
                dtypes[str(t.dtype)] = dtypes.get(str(t.dtype), 0) + 1
                sd[k] = t.float() if t.dtype != torch.float32 else t
            else:
                skipped += 1
    return sd, {"tensors_read": len(sd), "tensors_skipped_vision_audio": skipped, "checkpoint_dtypes": dtypes}


def build(L, weights, cfg_all, f16safe=False, variant=None):
    torch.manual_seed(0)
    if f16safe:   # round 8: the fp16-safe norm rewrite (same parameter keys); variant = a diagnostic form
        import d1_graph_f16safe as F16

        model = F16.D1DecisionF16Safe(cfg_all["text_config"], cfg_all["head_layers"], L, variant=variant).eval()
    else:
        model = G.D1Decision(cfg_all["text_config"], cfg_all["head_layers"], L).eval()
    report = {"weights": weights}
    if f16safe:
        report["f16safe"] = model.f16safe_report
    if weights == "checkpoint":
        if not S.WEIGHTS.exists():
            print(f"weights not present: {S.WEIGHTS}")
            sys.exit(3)
        sd, rd = load_checkpoint_subset()
        report.update(rd)
        report["load"] = G.load_state_dict_from_provider(model, sd)
        del sd
    return model, report


def fixture_sample(L):
    """The card_text/refund row of results/encoded_rows.json through host.build_inputs -> torch tensors."""
    enc = json.loads((S.K / "results/encoded_rows.json").read_text())
    row = next(r for r in enc["rows"] if (r["id"], r["qid"]) == SAMPLE_ROW)
    x = H.build_inputs(row["ids"], None, L)
    oh = np.zeros((1, 3), np.float32)
    oh[0, row["qtype"]] = 1.0
    x["qtype_onehot"] = oh
    kw = {k: torch.from_numpy(x[k]) for k in G.INPUT_NAMES}
    return kw, {"row": "/".join(SAMPLE_ROW), "n_ids": len(row["ids"]), "markers": row["markers"], "qtype": row["qtype"],
                "type": row["type"]}


def env_record():
    from ai_edge_litert.compiled_model import CompiledModel, CpuOptions, GpuOptions, Options
    import ai_edge_litert
    import dataclasses
    import inspect

    pkgs = {}
    for p in ("litert-torch", "litert-converter", "ai-edge-litert", "ai-edge-quantizer", "torch", "numpy", "flatbuffers",
              "safetensors", "transformers"):
        try:
            pkgs[p] = md.version(p)
        except md.PackageNotFoundError:
            pkgs[p] = None
    pkg_dir = Path(ai_edge_litert.__file__).parent
    metal = pkg_dir / "libLiteRtMetalAccelerator.dylib"
    doc = {
        "written": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "step": "round 3 step 0",
        "venv_used": {"name": "lt094dev", "python": sys.version.split()[0], "executable": sys.executable,
                      "platform": platform.platform(), "packages": pkgs},
        "venvs_tried": [{"name": "lt094dev", "result": "used (export / quantize / CompiledModel)"}],
        "fallback_order": ["~/venvs/lt094dev/bin/python", ".venv-092/bin/python3", "~/venvs/ltmain0918/bin/python3"],
        "gpu_options_fields": [{"name": f.name, "default": f.default} for f in dataclasses.fields(GpuOptions)],
        "gpu_options_default_flat": GpuOptions()._as_flat_kwargs(),
        "cpu_options_signature": str(inspect.signature(CpuOptions)),
        "options_signature": str(inspect.signature(Options)),
        "compiled_model_methods": [m for m in dir(CompiledModel) if not m.startswith("_")],
        "metal_dylib": {"path": str(metal), "exists": metal.exists(), "bytes": metal.stat().st_size if metal.exists() else None},
        "weights": {"path": str(S.WEIGHTS), "resolved": str(S.WEIGHTS.resolve()), "bytes": S.WEIGHTS.stat().st_size},
        "note": "lt094dev is shared: run only, nothing installed by this lane.",
    }
    return doc


def env_append(key):
    """Round 4 step 0: add this round's versions under `key` of results/export_env.json (other keys untouched)."""
    path = S.K / "results/export_env.json"
    doc = json.loads(path.read_text())
    if key in doc:
        print(f"{key} already in {path.name}; not rewritten")
        return doc[key]
    rec = env_record()
    rec["step"] = f"{key.replace('_', ' ')} step 0"
    rec["venvs_tried"] = [{"name": "lt094dev", "result": "used (export / quantize / CompiledModel / timing)"}]
    rec["psutil_in_venv"] = _version("psutil")
    rec["memory_method"] = ("proc_pid_rusage(RUSAGE_INFO_V4) ri_phys_footprint / ri_lifetime_max_phys_footprint "
                            "(= vmmap -summary Physical footprint) + ru_maxrss + ps -o rss; psutil is not in the venv")
    rec["machine"] = {"hw.ncpu": _sysctl("hw.ncpu"), "hw.perflevel0.physicalcpu": _sysctl("hw.perflevel0.physicalcpu"),
                      "hw.perflevel1.physicalcpu": _sysctl("hw.perflevel1.physicalcpu"),
                      "hw.memsize": _sysctl("hw.memsize"), "machdep.cpu.brand_string": _sysctl("machdep.cpu.brand_string"),
                      "kern.osproductversion": _sysctl("kern.osproductversion")}
    doc[key] = rec
    R.dump_json(path, doc, overwrite=True)
    return rec


def _sysctl(name):
    import subprocess

    return subprocess.run(["sysctl", "-n", name], capture_output=True, text=True).stdout.strip()


def build_multi(buckets, cfg_all):
    """-> {L: D1Decision} sharing one set of parameter modules (the first L's checkpoint load); report."""
    base, wrep = build(buckets[0], "checkpoint", cfg_all)
    mods = {buckets[0]: base}
    for L in buckets[1:]:
        torch.manual_seed(0)
        m = G.D1Decision(cfg_all["text_config"], cfg_all["head_layers"], L).eval()
        m.encoder.embed_tokens = base.encoder.embed_tokens
        m.encoder.layers = base.encoder.layers
        m.encoder.embedding_norm = base.encoder.embedding_norm
        m.head = base.head
        mods[L] = m
    ids0 = {id(p) for p in base.parameters()}
    shared = {L: all(id(p) in ids0 for p in m.parameters()) and len(list(m.parameters())) == len(ids0)
              for L, m in mods.items()}
    assert all(shared.values()), shared
    rope = {L: [list(m.encoder.cos.shape), list(m.encoder.sin.shape)] for L, m in mods.items()}
    return mods, {"weights": wrep, "parameters_shared_by_identity": shared, "rope_buffers": rope,
                  "parameter_tensors": len(ids0)}


def export_multi(buckets, round_no):
    import litert_torch

    torch.set_num_threads(8)
    cfg_all = S.config()
    tag = f"multi{len(buckets)}"
    path = S.K / f"out/d1omni_decide_{tag}_fp32.tflite"
    res = S.K / "results/multisig_export.json"
    for p in (path, res):
        assert not p.exists(), f"refusing to overwrite {p}"
    t0 = time.time()
    mods, rep = build_multi(buckets, cfg_all)
    samples, eager = {}, {}
    for L in buckets:
        kw, srep = fixture_sample(L)
        samples[L] = (kw, srep)
        with torch.no_grad():
            eager[L] = mods[L](**kw)["scores"].numpy().reshape(-1)
    rec = {"file": str(path.relative_to(S.K)), "buckets": buckets, "signatures": [f"decide_{L}" for L in buckets],
           "build": rep, "python": sys.version.split()[0], "torch": torch.__version__,
           "litert_torch": md.version("litert-torch"), "litert_converter": _version("litert-converter"),
           "convert_call": "litert_torch.signature('decide_<L0>', module_L0, sample_kwargs=...).signature(...)"
                           "... .convert().export(path)"}
    try:
        t1 = time.time()
        chain = None
        for L in buckets:
            kw = samples[L][0]
            chain = (litert_torch.signature(f"decide_{L}", mods[L], sample_kwargs=kw) if chain is None
                     else chain.signature(f"decide_{L}", mods[L], sample_kwargs=kw))
        edge = chain.convert()
        rec["convert_seconds"] = round(time.time() - t1, 1)
        t2 = time.time()
        edge.export(str(path))
        rec["export_seconds"] = round(time.time() - t2, 1)
        del edge, chain
    except BaseException:
        (S.K / f"logs/r{round_no}_graph_build_{tag}.traceback.txt").write_text(traceback.format_exc())
        raise
    rec["peak_rss_bytes"] = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    print(f"wrote {path} ({path.stat().st_size} B) convert {rec['convert_seconds']} s", flush=True)
    sc = R.scan(path)
    singles = {}
    for L in buckets:
        p1 = out_path(L)
        if p1.exists():
            singles[str(L)] = p1.stat().st_size
    checks = {
        "signatures_decide_L": sorted(s["key"] for s in sc["signatures"]) == sorted(rec["signatures"]),
        "subgraphs_eq_signatures": sc["subgraphs"] == len(buckets),
        "BROADCAST_TO_0": sc["op_histogram"].get("BROADCAST_TO", 0) == 0,
        "CUSTOM_0": sc["custom_op_count"] == 0,
        "EMBEDDING_LOOKUP_per_signature": sc["op_histogram"].get("EMBEDDING_LOOKUP", 0) == len(buckets),
        "one_table_buffer": len({e["table_buffer"] for e in sc["embedding_lookup"]}) == 1,
        "tables_FLOAT32_65536x1024": all(e["table_source_dtype"] == "FLOAT32" and e["table_shape"] == [65536, 1024]
                                         for e in sc["embedding_lookup"]),
        "max_tensor_rank_le_4": sc["max_tensor_rank"] <= 4,
    }
    # CPU smoke run of every signature vs its eager module
    cm, desc = R.open_compiled(path, "cpu", threads=8)
    smoke = {}
    for L in buckets:
        kw, srep = samples[L]
        run = R.Runner(cm, f"decide_{L}")
        assert run.L == L, (run.L, L)
        got = run({k: v.numpy() for k, v in kw.items()})
        run.close()
        n = srep["n_ids"]
        smoke[str(L)] = {"max_abs_scores_real_vs_eager": float(np.abs(got[:n].astype(np.float64) - eager[L][:n]).max()),
                         "finite": bool(np.isfinite(got).all())}
    if hasattr(cm, "close"):
        cm.close()
    doc = {"step": f"round {round_no} step 4: multi-signature fp32 export ({len(buckets)} buckets in one file)",
           "written": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "export": rec, "checks": checks,
           "checks_pass": all(checks.values()), "bytes": sc["bytes"], "sha256": sc["sha256"],
           "single_signature_fp32_bytes": singles,
           "bytes_over_single_L256": (sc["bytes"] / singles["256"]) if "256" in singles else None,
           "smoke_cpu_vs_eager": smoke, "scan": {k: sc[k] for k in (
               "subgraphs", "signatures", "operator_count", "op_histogram", "tensor_dtype_histogram",
               "constant_bytes_by_dtype", "constant_unique_buffer_bytes_by_dtype",
               "constant_unique_buffer_count_by_dtype", "embedding_lookup", "fully_connected_count",
               "batch_matmul_shape_groups", "max_tensor_rank", "custom_op_count")},
           "seconds_total": round(time.time() - t0, 1)}
    R.dump_json(res, doc)
    print(json.dumps({k: doc[k] for k in ("bytes", "single_signature_fp32_bytes", "bytes_over_single_L256", "checks",
                                          "smoke_cpu_vs_eager", "seconds_total")}, indent=1), flush=True)
    return 0 if doc["checks_pass"] else 1


def contract_draft(buckets=(256, 512), mac=None):
    """Round 3 step 5: the host <-> graph contract seed (results/contract_draft.json), from the files this round wrote.
    Round 4: `buckets` = every bucket with a signature json; `mac` = the Mac recommendation (a dict, from the timing
    and parity files)."""
    cfg = S.config()
    tok = json.loads((S.K / "results/tokenizer_check.json").read_text())["token_ids"]
    sigs, files = {}, {}
    for L in buckets:
        sd = json.loads((S.K / f"results/signature_L{L}.json").read_text())
        fb = sd["flatbuffer_signatures"][0]
        sigs[str(L)] = {"signature": sd["signature"], "inputs_by_tensor_index": sorted(
            [{"name": i["name"], "dtype": i["dtype"], "shape": i["shape"], "tensor_index": i["tensor_index"]}
             for i in fb["inputs"]], key=lambda i: i["tensor_index"]),
            "outputs": [{"name": o["name"], "dtype": o["dtype"], "shape": o["shape"]} for o in fb["outputs"]]}
        q = json.loads((S.K / f"results/quant_L{L}.json").read_text())
        files[str(L)] = {"fp32": {"file": f"out/d1omni_decide_L{L}_fp32.tflite", "bytes": q["input"]["bytes"]},
                         **{f: {"file": e["output"], "bytes": e.get("bytes"), "sha256": e.get("sha256")}
                            for f, e in q["forms"].items()}}
    gates = {}
    for pth in sorted((S.K / "results").glob("litert_*_parity_L*_*.json")):
        d = json.loads(pth.read_text())
        s = d.get("summary") or {}
        gates[pth.name] = {"status": d.get("status"), "reference": s.get("reference"), "rows": s.get("rows"),
                           "max_abs_dp": s.get("max_abs_dp"), "bar_pass": s.get("bar_pass")}
    return {
        "what": ("d1-omni-600M decision graph (D1Decision) <-> host contract, round 3 draft (L256 / L512 only)"
                 if tuple(buckets) == (256, 512) else
                 f"d1-omni-600M decision graph (D1Decision) <-> host contract, round 4 draft (buckets {list(buckets)})"),
        "written": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "source_model": {"repo": S.REPO, "revision": S.REV, "weights_sha256":
                         "0713bb05270c2685ad106522f4092bceeeb3a93cf79b401f399a712296c911e1"},
        "graph": {"one_file_per_bucket": True, "signature_name": "decide_<L>", "by_L": sigs,
                  "inputs": {
                      "ids": "int32 [1, L]: the encoded question (bos + state + question + options + decide) at "
                             "positions P .. P+n-1; 0 elsewhere (pad id; any value is ignored at pad positions)",
                      "prefix": "float32 [1, L, 1024]: media embeddings (vision / audio graph output) at 0 .. P-1; 0 "
                                "elsewhere (any value is ignored where media = 0)",
                      "media": "float32 [1, L]: 1.0 at 0 .. P-1, else 0",
                      "pad": "float32 [1, L]: 1.0 at the real positions 0 .. P+n-1, else 0",
                      "keep_right": "float32 [1, L]: 0.0 only at P-1 when P > 0 (the last media position never reads "
                                    "the first text token through the ShortConv right tap), else 1.0",
                      "qtype_onehot": "float32 [1, 3]: [choice, score, noul] one-hot of the question type"},
                  "output": {"scores": "float32 [1, L]: the scorer at every position; only P + marker positions "
                                       "are read"},
                  "input_order_note": "CompiledModel / signature list the inputs alphabetically; the flatbuffer "
                                      "tensor order is ids, prefix, media, pad, keep_right, qtype_onehot "
                                      "(bind by name, never by index)",
                  "buckets": list(buckets),
                  **({"buckets_planned_round_4": [128, 1024, 2048, 4096]} if tuple(buckets) == (256, 512) else {}),
                  "bucket_rule": "the smallest L >= P + n"},
        "token_ids": {k: v["id"] for k, v in tok.items()},
        "token_roles": {k: v["role"] for k, v in tok.items()},
        "temperatures": cfg["temperatures"],
        "modes": {"text": {"max_len": cfg["max_length"], "calibrate": True, "noul_default": None, "audio": False},
                  "image": {"max_len": cfg["image_text_length"], "calibrate": False,
                            "noul_default": {"false": "no", "true": "yes"}, "audio": False},
                  "audio": {"max_len": cfg["audio_text_length"], "calibrate": False,
                            "noul_default": {"false": "no", "true": "yes"}, "audio": True,
                            "state_none_becomes": {}},
                  "media_max_len_rule": "max_len = min(mode max_len, max_length - P); refuse when < 64"},
        "host_steps": [
            "q = as_question(question dict); ids, markers = prompt.encode(tok, state, q, max_len, noul_default, audio) "
            "(tokenizer: tokenizer.json, add_special_tokens=False, one bos added by encode)",
            "P = 0 for text, else the media prefix rows (vision / audio graph); L = smallest bucket >= P + len(ids)",
            "inputs = build_inputs(ids, prefix, L) + qtype_onehot (host/d1_host.py)",
            "scores = graph(decide_<L>)(**inputs)['scores'][0]",
            "z = scores[P + markers][:K]; text: z / temperatures[temperature_key(q)]; image / audio: no temperature",
            "p = softmax(z); a noul is reversed to [yes, no]; answer(q, p) gives the provider's response dict"],
        "weight_forms": files,
        "gates_at_writing": gates,
        "notes": ["Metal default precision (fp16 activations) collapses every row (uniform probabilities, 0 "
                  "non-finite) -> GPU runs use fp32 precision (GpuOptions(enforce_f32=True))",
                  "wi8fc is refused by Metal (Unable to parse bc coord for BATCH axis)"],
        **({"mac_recommendation": mac} if mac is not None else {}),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", default="256", help="comma-separated L values")
    ap.add_argument("--weights", choices=("random", "checkpoint"), default="checkpoint")
    ap.add_argument("--sample", choices=("fixture", "random"), default="fixture")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--env", action="store_true", help="write results/export_env.json and stop")
    ap.add_argument("--contract", choices=("print", "write", "replace"),
                    help="step 5: results/contract_draft.json (replace = rewrite after the gates changed; keeps "
                         "the previous `written` stamp in `replaces`)")
    ap.add_argument("--round", type=int, default=3, help="round 4: log / step labels only")
    ap.add_argument("--env-append", metavar="KEY", help="round 4 step 0: add KEY to results/export_env.json")
    ap.add_argument("--multi", metavar="L,L,...", help="round 4 step 4: one fp32 file, one signature per L")
    ap.add_argument("--mac-recommendation", metavar="JSON", help="--contract: the Mac recommendation (round 4)")
    ap.add_argument("--f16safe", action="store_true", help="round 8: D1DecisionF16Safe -> *_f16safe_fp32.tflite")
    ap.add_argument("--f16safe-variant", choices=("l0sum",),
                    help="round 8 diagnostic: d1_graph_f16safe.VARIANTS -> *_f16safe_<variant>_fp32.tflite")
    a = ap.parse_args()
    if a.f16safe_variant:
        a.f16safe = True
    sys.dont_write_bytecode = True

    if a.env_append:
        print(json.dumps(env_append(a.env_append), indent=1))
        return 0

    if a.multi:
        return export_multi([int(x) for x in a.multi.split(",")], a.round)

    if a.contract:
        have = sorted(int(p.stem.split("_L")[1]) for p in (S.K / "results").glob("signature_L*.json"))
        mac = json.loads(Path(a.mac_recommendation).read_text()) if a.mac_recommendation else None
        doc = contract_draft(tuple(have), mac) if a.round >= 4 else contract_draft()
        out = S.K / "results/contract_draft.json"
        if a.contract == "write":
            R.dump_json(out, doc)
        elif a.contract == "replace":
            prev = json.loads(out.read_text())
            doc["replaces"] = {"written": prev.get("written"), "gates_at_writing": prev.get("gates_at_writing")}
            R.dump_json(out, doc, overwrite=True)
        print(json.dumps(doc, indent=1)[:6000])
        return 0

    if a.env:
        doc = env_record()
        R.dump_json(S.K / "results/export_env.json", doc)
        print(json.dumps(doc, indent=1))
        return 0

    import litert_torch  # the exporter package of this venv

    torch.set_num_threads(8)
    cfg_all = S.config()
    for L in [int(x) for x in a.L.split(",")]:
        t0 = time.time()
        vtag = f"f16safe_{a.f16safe_variant}" if a.f16safe_variant else "f16safe"
        tag = f"{vtag}_fp32" if a.f16safe else "fp32"
        path = out_path(L, tag)
        res_scan = S.K / f"results/opscan_L{L}_{tag}.json"
        res_sig = S.K / (f"results/signature_L{L}_{vtag}.json" if a.f16safe else f"results/signature_L{L}.json")
        if not a.dry_run:
            for p in (path, res_scan, res_sig):
                assert not p.exists(), f"refusing to overwrite {p}"
        model, wrep = build(L, a.weights, cfg_all, f16safe=a.f16safe, variant=a.f16safe_variant)
        if a.sample == "fixture":
            kw, srep = fixture_sample(L)
        else:
            kw, srep = G.sample_inputs(L, n_text=L // 2, n_prefix=L // 8, seed=0, qtype=2), {"row": "random"}
        assert tuple(kw) == G.INPUT_NAMES
        with torch.no_grad():
            eager = model(**kw)["scores"]
        rec = {"signature": f"decide_{L}", "L": L, "file": str(path.relative_to(S.K)),
               "inputs": {k: {"dtype": str(v.dtype), "shape": list(v.shape)} for k, v in kw.items()},
               "outputs": {"scores": {"dtype": str(eager.dtype), "shape": list(eager.shape)}},
               "parameters": sum(p.numel() for p in model.parameters()), "weights_report": wrep, "sample": srep,
               "eager_scores_finite": bool(torch.isfinite(eager).all()),
               "python": sys.version.split()[0], "torch": torch.__version__,
               "litert_torch": md.version("litert-torch"), "litert_converter": _version("litert-converter")}
        print(json.dumps({k: rec[k] for k in ("signature", "L", "file", "parameters", "sample", "eager_scores_finite")}),
              flush=True)
        if a.dry_run:
            print(f"dry-run: stopped before litert_torch.signature for decide_{L} (nothing converted)")
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            t1 = time.time()
            edge = litert_torch.signature(rec["signature"], model, sample_kwargs=kw).convert()
            rec["convert_seconds"] = round(time.time() - t1, 1)
            t2 = time.time()
            edge.export(str(path))
            rec["export_seconds"] = round(time.time() - t2, 1)
            del edge
        except BaseException:
            (S.K / f"logs/r{a.round}_graph_build_L{L}.traceback.txt").write_text(traceback.format_exc())
            raise
        rec["peak_rss_bytes"] = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        print(f"wrote {path} ({path.stat().st_size} B) convert {rec['convert_seconds']} s export {rec['export_seconds']} s",
              flush=True)

        # static scan + acceptance checks
        sc = R.scan(path)
        emb = sc["embedding_lookup"]
        n_conv = sum(1 for k in cfg_all["text_config"]["layer_types"] if k != "full_attention")
        n_attn = len(cfg_all["text_config"]["layer_types"]) - n_conv
        fc_expected = {"trunk_conv_in_out": 2 * n_conv, "trunk_attn_qkvo": 4 * n_attn,
                       "trunk_mlp": 3 * len(cfg_all["text_config"]["layer_types"]),
                       "head_layers_qkv_out_ff": 6 * cfg_all["head_layers"], "scorer": 2}
        checks = {
            "BROADCAST_TO_0": sc["op_histogram"].get("BROADCAST_TO", 0) == 0,
            "CUSTOM_0": sc["custom_op_count"] == 0,
            "INT64_tensors_le_1": sc["int64_tensor_count"] <= 1,
            "GATHER_ND_0": sc["op_histogram"].get("GATHER_ND", 0) == 0,
            "EMBEDDING_LOOKUP_1": sc["op_histogram"].get("EMBEDDING_LOOKUP", 0) == 1,
            "embedding_table_65536x1024_FLOAT32": len(emb) == 1 and emb[0]["table_shape"] == [65536, 1024]
            and emb[0]["table_source_dtype"] == "FLOAT32",
            "max_tensor_rank_le_4": sc["max_tensor_rank"] <= 4,
            "one_signature_decide_L": [s["key"] for s in sc["signatures"]] == [rec["signature"]],
        }
        scan_doc = {"step": f"round {a.round} step 1: op scan of the fp32 export, L={L}", "written": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    "export": rec, "fc_expected_by_part": fc_expected, "fc_expected_total": sum(fc_expected.values()),
                    "fc_observed": sc["fully_connected_count"], "checks": checks, "checks_pass": all(checks.values()),
                    "scan": sc}
        # signature I/O through CompiledModel (input order as the runtime reports it) + a smoke run on the sample
        from ai_edge_litert.compiled_model import CompiledModel  # noqa: F401
        cm, desc = R.open_compiled(path, "cpu", threads=8)
        sig = rec["signature"]
        in_det, out_det = cm.get_input_tensor_details(sig), cm.get_output_tensor_details(sig)
        run = R.Runner(cm, sig)
        got = run({k: v.numpy() for k, v in kw.items()})
        run.close()
        n_real = srep.get("n_ids", L)
        e = eager.numpy().reshape(-1)
        smoke = {"backend": desc, "rows": 1, "real_positions": n_real,
                 "max_abs_scores_real_vs_eager": float(np.abs(got[:n_real].astype(np.float64) - e[:n_real]).max()),
                 "max_abs_scores_all_vs_eager": float(np.abs(got.astype(np.float64) - e).max()),
                 "marker_scores_litert": [float(got[m]) for m in srep.get("markers", [])],
                 "marker_scores_eager": [float(e[m]) for m in srep.get("markers", [])],
                 "finite": bool(np.isfinite(got).all())}
        scan_doc["smoke_cpu_vs_eager"] = smoke
        if a.f16safe:   # round 8: against the plain file of the same L
            plain_scan = json.loads((S.K / f"results/opscan_L{L}_fp32.json").read_text())["scan"]
            plain_sig = json.loads((S.K / f"results/signature_L{L}.json").read_text())["flatbuffer_signatures"]
            ph, fh = plain_scan["op_histogram"], sc["op_histogram"]
            delta = {k: fh.get(k, 0) - ph.get(k, 0) for k in sorted(set(ph) | set(fh)) if fh.get(k, 0) != ph.get(k, 0)}
            n_scaled = wrep["f16safe"]["scaled"]
            io = lambda sigs: [(s["key"], [(i["name"], i["shape"], i["dtype"]) for i in s["inputs"]],
                                [(o["name"], o["shape"], o["dtype"]) for o in s["outputs"]]) for s in sigs]
            fchecks = {f"MUL_plus_{n_scaled}_scaled_sites": delta == {"MUL": n_scaled},
                       "signature_io_equal_plain": io(sc["signatures"]) == io(plain_sig),
                       "rank_le_4": sc["max_tensor_rank"] <= 4,
                       "INT64_tensors_equal_plain": sc["int64_tensor_count"] == plain_scan["int64_tensor_count"]}
            scan_doc["f16safe"] = {"module_report": wrep["f16safe"], "plain_opscan": f"results/opscan_L{L}_fp32.json",
                                   "plain_operator_count": plain_scan["operator_count"],
                                   "operator_count": sc["operator_count"], "op_histogram_delta_vs_plain": delta,
                                   "plain_bytes": plain_scan["bytes"], "bytes": sc["bytes"], "checks": fchecks,
                                   "checks_pass": all(fchecks.values())}
            scan_doc["checks_pass"] = scan_doc["checks_pass"] and scan_doc["f16safe"]["checks_pass"]
        sig_doc = {"step": f"round {a.round} step 1: signature of {path.relative_to(S.K)}",
                   "signature": sig, "flatbuffer_signatures": sc["signatures"],
                   "compiled_model_inputs": {n: {k: str(v) for k, v in d.items()} for n, d in in_det.items()},
                   "compiled_model_outputs": {n: {k: str(v) for k, v in d.items()} for n, d in out_det.items()},
                   "inputs_contract": {"ids": "int32 [1, L]: bos + text ids at positions P..P+n-1, 0 elsewhere",
                                       "prefix": "float32 [1, L, 1024]: media embeddings at 0..P-1, 0 elsewhere",
                                       "media": "float32 [1, L]: 1 at 0..P-1",
                                       "pad": "float32 [1, L]: 1 at real positions 0..P+n-1",
                                       "keep_right": "float32 [1, L]: 0 only at P-1 (when P > 0), 1 elsewhere",
                                       "qtype_onehot": "float32 [1, 3]: choice / score / noul"},
                   "output_contract": {"scores": "float32 [1, L]: the scorer at every position; the host reads P + marker"},
                   "file_bytes": sc["bytes"], "sha256": sc["sha256"]}
        rec["seconds_total"] = round(time.time() - t0, 1)
        R.dump_json(res_scan, scan_doc)
        R.dump_json(res_sig, sig_doc)
        print(json.dumps({"L": L, "bytes": sc["bytes"], "ops": sc["operator_count"], "checks": checks,
                          "fc": [sc["fully_connected_count"], sum(fc_expected.values())],
                          "op_histogram": sc["op_histogram"], "smoke": smoke, "seconds": rec["seconds_total"]}, indent=1),
              flush=True)
        del model
    return 0


if __name__ == "__main__":
    sys.exit(main())
