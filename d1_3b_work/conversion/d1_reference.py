"""The reference for the d1-3B LiteRT conversion: the provider's own code (LiquidAI/d1-3B at REV) on the CPU in float32.

    # no weights: the ids and read-out groups the reference will feed, checked against fixtures/rows.json
    $REF scripts/d1_reference.py --dry-run
    # the weights' safetensors header against the Hub's (evidence/) and the source's copy, and the 707 tensors against
    # a meta build of D1Model: results/reference_<tag>_header.json
    $REF scripts/d1_reference.py --check-header --source <snapshot> --tag real
    # the reference (a long CPU job: start it inside a measurement lock; at most 4 threads)
    $REF scripts/d1_reference.py \
        --source <snapshot> --tag real
    # the per-question table: near ties, gold agreement by slice, the red arm, the length join, the other reference
    $REF scripts/d1_reference.py --summarize results/reference_real.json

Model: `AutoModel.from_pretrained(<snapshot>, trust_remote_code=True, dtype=torch.float32)` = D1Model, on the CPU, with
`output_loading_info` asserted clean; `engine = model.engine` (SystemOne with its defaults). Never bfloat16 or MPS: the
provider's card for the sibling model says bfloat16 changes answers.

Per record of fixtures/requests.json (the provider's parser; a question it rejects is left out of its request). The
questions go to the engine as JSON and are parsed by the engine's own copy of prompt.py (transformers' module cache):
its isinstance checks know only its own classes.
  row  (every question)   probs  = engine.probabilities(state, [q])   one plain pass, positions 0..L-1: the oracle
                          logz   = engine._logz_ids([ids])             log-softmax at the answer slot; its pass is
                                   recorded as it runs: the raw logits (`_one_pass` output) and the language model's
                                   final-norm output at every position (a forward hook = what the row graphs return)
  tree (several questions) probs_tree = engine.probabilities(state, qs)  the provider's path for such a request: one
                                   tree, trunk = state, branches = questions; its `_tree_logz` call is recorded as it
                                   runs (trunk, branches, log-probs)
The recorders only read what the provider's code computes; nothing is recomputed in their place.
Checks (all recorded; the run exits 1 if one fails): ids == rows.json ids (before the model loads);
readout(logz) == probs bit for bit; logz == log_softmax(recorded logits) bit for bit; lm_head(hidden at the slot)
within 1e-5 of the recorded logits; the tree's trunk and branches == rows.json's split; readout(tree logz) == probs_tree;
tv4_000 again, bit for bit. Pictures (card_cats_001) wait for their image (`image_pending`).
Outputs, never overwritten: results/reference_<tag>.json and results/reference_<tag>_hidden.npz (`<id>/<qid>` = the
answer slot's hidden [2048]; `<id>/<qid>/full` = every position [L, 2048] for FULL_ROWS). `--only` / `--limit` runs
get `_smoke` in the tag.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import platform
import resource
import struct
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from d1_common import (FIXTURES, HF_SMALL, K, REPO, REV, ROWS, encode, engine_settings, load_fixtures,  # noqa: E402
                       load_tokenizer, provider, questions_of, render_row, sha256_file, split_texts)

WEIGHTS_SHA256 = "50e03317847caf6df9a9aee27ed40f20554a86a21e60d1d47ba41a422b546c0c"
WEIGHTS_BYTES = 6_247_065_504
NEAR_TIE = 0.02
RED_ARM_MIN = 0.02
MAX_THREADS = 4    # the Mac is shared with other lanes' measurements
# every position of these rows goes into the hidden file: the card's text example, and the longest row of the
# mid-length record (L1024 bucket) and of the longest record (L4096 bucket)
FULL_ROWS = (("card_text_001", "refund"), ("own_mid_08k_001", "first_fault"), ("own_long_34k_001", "refunded_twice"))
BUCKETS = (64, 128, 256, 512, 1024, 2048, 4096)   # host/contract.json `buckets.candidates`
HEADER = K / "evidence/hub_safetensors_header_da1fe36a.json"
SOURCE_HEADER = K / "evidence/d1-3b_safetensors_header.json"   # the header copy that came with the source files
LOCK = Path(os.environ.get("D1_GPU_LOCK", str(K / "gpu.lock")))
D1B_DIRS = (K / "external/other_conversion/oracle", K / "external/other_conversion/ref")   # another conversion
# The control arms (fixtures/red_arms.json, `--make-red-arms`): red-arm requests of another conversion of d1-3B,
# adopted on 2026-10-08 after all five of its arms were measured on this float32 path (logs/r6a_red_arms_real_json.log).
# The fixture's own red_arm_000 (= its red_word_tv4_000, "correctly" -> "incorrectly") moves d1-3B by 0.0069 only:
# it stays in requests.json as a record, not as a control.
RED_ARMS = K / "fixtures/red_arms.json"
D1B_RED_ARMS = K / "external/other_conversion/fixtures/red_arms.json"
D1B_RED_ARMS_SHA256 = "54fb9ad6063464e1f30f4d7b45da5e782b82e73b1c8ba8854fb7c055c4353655"
D1B_RECORDS = K / "external/other_conversion/fixtures/records.json"
D1B_RECORDS_SHA256 = "2284bb22e8cdc02a4d1c4a787e34981d2cee4044b32c3b4452abcd61136ef4dd"
ADOPTED_ARMS = ("red_not_qnli_00", "tv4_001_with_state_of_tv4_000",
                "semif_a3f18f3a63d45345942b_with_state_of_semif_0b43ea8e24e74d621f1b",
                "semif_0b43ea8e24e74d621f1b_with_state_of_semif_a3f18f3a63d45345942b")
MEASURED_BEFORE_ADOPTION = "logs/r6a_red_arms_real_json.log"   # read by --make-red-arms (max |dp| per request)


def request_sha256(request: dict) -> str:
    """make_d1_fixtures.request_sha256 (the other conversion's formula): sha256 of the canonical JSON of the request."""
    return hashlib.sha256(json.dumps(request, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def plan(prompt, tok, settings, fixtures, rows_by_key):
    """Per record: the parsed questions, their row ids (asserted equal to rows.json) and the tree split."""
    out = []
    for r in fixtures["records"]:
        parsed = [(qid, q) for qid, q, err in questions_of(prompt, r) if q is not None]
        qdicts = r["request"]["questions"]
        rejected = [(qid, err) for qid, q, err in questions_of(prompt, r) if q is None]
        images = r["request"].get("images") or []
        items = []
        for qid, q in parsed:
            want = rows_by_key[(r["id"], qid)]
            if images:   # the markup only; the processor expands <image> with the picture (round 5)
                items.append({"qid": qid, "q": q, "qd": qdicts[qid], "ids": None, "groups": want["readout_ids"],
                              "image_pending": True})
                continue
            text = render_row(prompt, tok, settings, r["request"]["state"], q)
            ids = encode(tok, text)
            assert ids == want["ids"], (r["id"], qid)
            groups = prompt.readout_ids(tok, q)
            assert groups == want["readout_ids"], (r["id"], qid)
            prefix, suffix = split_texts(prompt, tok, settings, r["request"]["state"], q)
            items.append({"qid": qid, "q": q, "qd": qdicts[qid], "ids": ids, "groups": groups,
                          "trunk": encode(tok, prefix), "branch": encode(tok, suffix)})
        out.append({"record": r, "items": items, "rejected": rejected, "images": images})
    return out


def make_red_arms() -> int:
    """fixtures/red_arms.json: the adopted arms of the other conversion in requests.json's record form, with their
    base record's id. Each base request here equals that file's (JSON value), and each arm differs from its base only
    where that arm says."""
    out = RED_ARMS
    assert not out.exists(), f"refusing to overwrite {out}"
    assert sha256_file(D1B_RED_ARMS) == D1B_RED_ARMS_SHA256 and sha256_file(D1B_RECORDS) == D1B_RECORDS_SHA256
    d1b_arms = json.loads(D1B_RED_ARMS.read_text())
    d1b_records = {r["id"]: r for r in json.loads(D1B_RECORDS.read_text())["records"]}
    ours = {r["id"]: r for r in load_fixtures()["records"]}
    measured = {x.get("request") or x["arm"]: x for x in json.loads((K / MEASURED_BEFORE_ADOPTION).read_text())["arms"]}
    records = []
    for arm in d1b_arms["arms"]:
        if arm["kind"] == "state_swap":
            reqs = [(name, req, name.split("_with_state_of_")[0], next(iter(req["questions"]))) for name, req in arm["requests"].items()]
        else:
            reqs = [(arm["id"], arm["request"], arm["base"], arm["question"])]
        for name, req, base_id, qid in reqs:
            if name not in ADOPTED_ARMS:
                continue
            base = ours[base_id]["request"]
            assert base == d1b_records[base_id]["request"], base_id
            assert list(req["questions"]) == [qid] and list(base["questions"]) == [qid], name
            if arm["kind"] == "state_swap":   # the base's question over the other record's state
                other = arm["pair"][1] if base_id == arm["pair"][0] else arm["pair"][0]
                assert req["questions"] == base["questions"] and req["state"] == ours[other]["request"]["state"], name
            else:                             # one field of the base's question
                field = arm["change"]["field"].split(".")
                assert field[:2] == ["questions", qid] and req["state"] == base["state"], name
                assert {k: v for k, v in req["questions"][qid].items() if k != field[2]} == \
                       {k: v for k, v in base["questions"][qid].items() if k != field[2]}, name
                assert base["questions"][qid][field[2]] == arm["change"]["from"] and req["questions"][qid][field[2]] == arm["change"]["to"]
            records.append({
                "id": name, "source": "red_arm", "base_id": base_id, "question": qid, "request": req, "gold": {qid: None},
                "note": arm.get("note") or arm.get("source") or "",
                "provenance": {"file": "another conversion's control file (red_arms.json)", "sha256": D1B_RED_ARMS_SHA256,
                               "arm": arm["id"], "kind": arm["kind"], "request_name": name,
                               "change": arm.get("change"), "base_gold": arm.get("base_gold"),
                               "base_from": {"file": "another conversion's fixture file (records.json)",
                                             "sha256": D1B_RECORDS_SHA256, "record": base_id},
                               "request_sha256": request_sha256(req), "base_request_sha256": request_sha256(base),
                               "measured_before_adoption": {"max_abs_dp": measured[name]["max_abs_dp"],
                                                            "argmax_base": measured[name]["argmax_base"],
                                                            "argmax_arm": measured[name]["argmax_arm"]}}})
    assert [r["id"] for r in records] == list(ADOPTED_ARMS), [r["id"] for r in records]
    doc = {"version": 1, "created_by": "scripts/d1_reference.py --make-red-arms (round 6a)",
           "what": "the control arms the reference and the gates run beside the fixture: another fixture's red-arm requests that "
                   "move d1-3B's float32 answer by far more than 0.02 against their base record (`base_id`, in requests.json)",
           "rule": d1b_arms["rule"],
           "decision": "2026-10-08, after the five arms of that file were measured on the provider's float32 "
                       "CPU path: these four requests. red_not_paws_00 and tv4_000_with_state_of_tv4_001 sit near the "
                       "line; red_word_tv4_000 (= requests.json red_arm_000, 'correctly' -> 'incorrectly') moves d1-3B "
                       "by 0.0069 only = invalid as a red arm on this model, kept in requests.json as a record",
           "measured_before_adoption": {"evidence": f"W/{MEASURED_BEFORE_ADOPTION}",
                                        "max_abs_dp": {k: v["max_abs_dp"] for k, v in measured.items()}},
           "request_sha256": "sha256 of json.dumps(request, sort_keys=True, ensure_ascii=False, separators=(',', ':')) in UTF-8",
           "record_format": "requests.json's {id, source, request, gold, note, provenance} plus base_id and question "
                            "(the question compared against the base record's same question); gold is null (an arm "
                            "has no intended answer: the gate reads |dp| against the base)",
           "records": records}
    out.write_text(json.dumps(doc, ensure_ascii=False, indent=1) + "\n")
    print(json.dumps({"out": str(out.relative_to(K)), "records": [r["id"] for r in records], "sha256": sha256_file(out)}))
    return 0


def plan_arms(prompt, tok, settings):
    """The control arms' rows (one question each, the plain-row path)."""
    if not RED_ARMS.exists():
        return [], None
    doc = json.loads(RED_ARMS.read_text())
    items = []
    for r in doc["records"]:
        qid = r["question"]
        q = prompt.as_question(r["request"]["questions"][qid])
        ids = encode(tok, render_row(prompt, tok, settings, r["request"]["state"], q))
        items.append({"record": r, "qid": qid, "qd": r["request"]["questions"][qid], "ids": ids,
                      "groups": prompt.readout_ids(tok, q)})
    return items, sha256_file(RED_ARMS)


def dry_run(planned, out_path: Path | None):
    lines, n = [], 0
    for p in planned:
        r = p["record"]
        for it in p["items"]:
            n += 1
            ids = it["ids"]
            lines.append({"id": r["id"], "qid": it["qid"], "type": it["q"].type,
                          "calls": ("tree+row" if len(p["items"]) > 1 else "row"),
                          "row_len": None if ids is None else len(ids),
                          "ids_head": None if ids is None else ids[:6], "ids_tail": None if ids is None else ids[-4:],
                          "readout_ids": it["groups"], "image_pending": it.get("image_pending", False)})
        for qid, err in p["rejected"]:
            lines.append({"id": r["id"], "qid": qid, "rejected": err})
    for x in lines[:6] + [x for x in lines if x.get("rejected") or x.get("image_pending")]:
        print(json.dumps(x, ensure_ascii=False))
    summary = {"records": len(planned), "questions": n,
               "tree_requests": sum(1 for p in planned if len(p["items"]) > 1),
               "rejected": [f"{p['record']['id']}/{q}" for p in planned for q, _ in p["rejected"]],
               "image_pending": [f"{p['record']['id']}/{it['qid']}" for p in planned for it in p["items"] if it.get("image_pending")],
               "ids_equal_rows_json": True}
    print(json.dumps(summary))
    if out_path is not None:
        out_path.write_text(json.dumps({"summary": summary, "rows": lines}, ensure_ascii=False) + "\n")
    return 0


def snapshot_dir(source: str | None) -> Path:
    if source:
        p = Path(source).expanduser()
        return p if p.is_absolute() else (K / p if (K / p).exists() else Path.cwd() / p)
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(REPO, revision=REV, local_files_only=True))


def read_header(path: Path) -> tuple[int, dict]:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        assert 0 < n < 50_000_000, n
        return n, json.loads(f.read(n))


def window_now() -> str | None:
    """The Mac measurement window (quiet_wait.py's rule): the lock names `timing` and its pid is alive."""
    try:
        content = LOCK.read_text(encoding="utf-8", errors="replace").strip()
        age = time.time() - LOCK.stat().st_mtime
    except OSError:
        return None
    if not content or "timing" not in content.lower():
        return None
    words = content.split()
    if "pid" in words and words.index("pid") + 1 < len(words) and words[words.index("pid") + 1].isdigit():
        try:
            os.kill(int(words[words.index("pid") + 1]), 0)
        except ProcessLookupError:
            return None
        except PermissionError:
            pass
    elif age > 3 * 3600:
        return None
    return content


def check_header(a) -> int:
    """Criterion 2: the weights' header == the Hub's (fetched in round 1 by HTTP range) == the source's copy, the data
    regions tile the file, and a meta build of D1Model from the snapshot's code has the header's 707 tensors."""
    import torch
    import transformers
    from transformers import AutoConfig
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    snap = snapshot_dir(a.source)
    w = snap / "model.safetensors"
    out = K / f"results/reference_{a.tag}_header.json"
    assert not out.exists(), f"refusing to overwrite {out}"
    n, local = read_header(w)
    ev = json.loads(HEADER.read_text())
    source_copy = json.loads(SOURCE_HEADER.read_text())
    tensors = {k: v for k, v in local.items() if k != "__metadata__"}
    ends = sorted((v["data_offsets"][0], v["data_offsets"][1]) for v in tensors.values())
    tiled = all(b[0] == a_[1] for a_, b in zip(ends, ends[1:])) and ends[0][0] == 0
    doc = {"weights": str(w), "resolved": str(w.resolve()), "is_symlink": w.is_symlink(), "bytes": w.stat().st_size,
           "header_bytes": n, "evidence": str(HEADER.relative_to(K)), "evidence_header_bytes": ev["header_bytes"],
           "equal_to_evidence": local == ev["header"] and n == ev["header_bytes"],
           "metadata": local.get("__metadata__"),
           "source_copy": str(SOURCE_HEADER), "source_copy_header_equal": tensors == {k: v for k, v in source_copy.items() if k != "__metadata__"},
           "tensors": len(tensors), "dtypes": dict(Counter(v["dtype"] for v in tensors.values())),
           "data_regions_tile": tiled, "data_end_plus_header_eq_file": 8 + n + ends[-1][1] == w.stat().st_size}
    os.environ.setdefault("HF_MODULES_CACHE", str(K / "cache/hf_modules"))
    cls = get_class_from_dynamic_module("modeling_d1.D1Model", str(snap))
    config = AutoConfig.from_pretrained(str(snap))
    with torch.device("meta"):
        model = cls(config)
    model.tie_weights()
    sd = model.state_dict()
    names = {k: list(v.shape) for k, v in sd.items()}
    # transformers 5.x stores Siglip2VisionModel without its `vision_model.` level and renames on load (round 1)
    renamed = {k.replace("model.vision_tower.vision_model.", "model.vision_tower.", 1): v for k, v in tensors.items()}
    doc["meta_build"] = {
        "class": f"{cls.__module__}.{cls.__qualname__}", "transformers": transformers.__version__, "torch": torch.__version__,
        "header_keys_renamed": sum(k != r for k, r in zip(tensors, renamed)),
        "header_tensors": len(renamed), "model_tensors": len(names),
        "matched": sum(1 for k in renamed if k in names and names[k] == renamed[k]["shape"]),
        "missing_in_model": sorted(k for k in renamed if k not in names),
        "not_in_header": sorted(k for k in names if k not in renamed),
        "shape_differences": sorted(k for k in renamed if k in names and names[k] != renamed[k]["shape"]),
        "lm_head_tied": bool(model.lm_head.weight is model.model.language_model.embed_tokens.weight)}
    m = doc["meta_build"]
    doc["all_ok"] = bool(doc["equal_to_evidence"] and doc["source_copy_header_equal"] and tiled and doc["data_end_plus_header_eq_file"]
                         and m["matched"] == len(renamed) == 707 and not m["missing_in_model"]
                         and m["not_in_header"] == ["lm_head.weight"] and m["lm_head_tied"])
    out.write_text(json.dumps(doc, indent=1) + "\n")
    print(json.dumps({k: v for k, v in doc.items() if k != "meta_build"} | {"matched": m["matched"], "not_in_header": m["not_in_header"]}))
    return 0 if doc["all_ok"] else 1


def run(planned, arms, a):
    import numpy as np
    import torch
    import transformers
    from transformers import AutoModel

    assert 1 <= a.threads <= MAX_THREADS, a.threads
    torch.set_num_threads(a.threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.manual_seed(0)
    subset = bool(a.limit or a.only)
    tag = a.tag + ("_smoke" if subset else "")
    out_dir = Path(a.out_dir) if a.out_dir else K / "results"
    out_json, out_npz = out_dir / f"reference_{tag}.json", out_dir / f"reference_{tag}_hidden.npz"
    for p in (out_json, out_npz):
        assert not p.exists(), f"refusing to overwrite {p}"
    t_start = time.time()
    windows: list = []

    def note_window(where: str, always: bool = False) -> None:
        """The start, the end and every change of the measurement window (checked after each record)."""
        w = window_now()
        if always or not windows or w != windows[-1]["lock"]:
            windows.append({"at": time.strftime("%H:%M:%S"), "where": where, "lock": w})

    note_window("start", always=True)
    snap = snapshot_dir(a.source)
    w = snap / "model.safetensors"
    assert w.stat().st_size == WEIGHTS_BYTES, w.stat().st_size
    t = time.time()
    assert sha256_file(w) == WEIGHTS_SHA256, "model.safetensors sha256 differs from the Hub's LFS record"
    sha_s = round(time.time() - t, 1)
    n_hdr, local_header = read_header(w)
    assert local_header == json.loads(HEADER.read_text())["header"], "header differs from the Hub's"
    for name in ("config.json", "tokenizer.json", "prompt.py", "runner.py", "hybrid.py", "lfm2_vl.py", "modeling_d1.py", "api.py"):
        assert (snap / name).read_bytes() == (HF_SMALL / name).read_bytes(), name
    t = time.time()
    model, info = AutoModel.from_pretrained(str(snap), trust_remote_code=True, dtype=torch.float32, output_loading_info=True)
    load_s = round(time.time() - t, 1)
    assert not any(info.get(k) for k in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")), info
    model.eval()
    dtypes = Counter(str(p.dtype) for p in model.parameters())
    devices = Counter(p.device.type for p in model.parameters())
    assert set(dtypes) == {"torch.float32"} and set(devices) == {"cpu"}, (dtypes, devices)
    engine = model.engine
    assert engine.device.type == "cpu" and engine.calibration is None
    lm = model.model.language_model
    # the engine's own prompt module (transformers' copy of prompt.py): its Question classes are the ones its
    # isinstance checks know, so questions go in as JSON and are parsed there
    eprompt = importlib.import_module(type(engine).__module__.rsplit(".", 1)[0] + ".prompt")

    # recorders: they keep what the provider's calls compute, nothing else
    cap: dict = {}
    plain_pass = engine._one_pass

    def recording_pass(*args, **kw):
        out = plain_pass(*args, **kw)
        cap["logits"] = out.logits
        return out

    engine._one_pass = recording_pass
    lm.register_forward_hook(lambda _m, _i, out: cap.__setitem__("hidden", out.last_hidden_state))
    tree_logz_fn, trees = engine._tree_logz, []

    def recording_tree(trunk, rows, **vision):
        z = tree_logz_fn(trunk, rows, **vision)
        trees.append((list(trunk), [list(r) for r in rows], z))
        return z

    engine._tree_logz = recording_tree

    def row_pass(state, qd, ids, groups, key, full_hidden=False):
        """One question's plain row: probabilities() and _logz_ids() with the pass recorded, and the checks."""
        q = eprompt.as_question(qd)
        if eprompt.readout_ids(engine.tokenizer, q) != groups:
            fails["readout_ids_vs_rows_json"].append(key)
        t = time.time()
        probs = engine.probabilities(state, [qd])[0]
        sec_api = time.time() - t
        cap.clear()
        t = time.time()
        logz = engine._logz_ids([ids])[0]
        sec_logz = time.time() - t
        logits = cap["logits"][0, -1].float()
        hidden = cap["hidden"][0]
        if hidden.shape != (len(ids), 2048) or not bool(torch.isfinite(hidden).all()):
            fails["hidden_shape_or_finite"].append(key)
        if not torch.equal(logz, logits - torch.logsumexp(logits, dim=-1)):
            fails["logz_vs_recorded_logits"].append(key)
        if eprompt.readout(engine.tokenizer, q, logz) != probs:
            fails["readout_logz_vs_probabilities"].append(key)
        d_head = float((model.lm_head(hidden[-1]).float() - logits).abs().max())
        worst["head_vs_logits"] = max(worst["head_vs_logits"], d_head)
        if d_head > 1e-5:
            fails["head_vs_logits_1e-5"].append(key)
        npz[key] = hidden[-1].numpy().astype(np.float32).copy()
        if full_hidden:
            npz[key + "/full"] = hidden.numpy().astype(np.float32).copy()
        order = sorted(range(len(probs)), key=lambda i: -probs[i])
        keys = (list(q.criteria) if q.type == "choice" else ["true", "false"] if q.type == "noul"
                else [str(i) for i in range(len(q.criteria))])
        return q, {
            "type": q.type, "keys": keys, "row_len": len(ids), "answer_slot": len(ids) - 1, "ids": ids,
            "readout_ids": groups,
            "logit": [[float(logits[i]) for i in g] for g in groups],
            "logz": [[float(logz[i]) for i in g] for g in groups],
            "lse": float(torch.logsumexp(logits, dim=-1)),
            "score": [max(float(logz[i]) for i in g) for g in groups],
            "probs": probs, "argmax": order[0], "argmax_key": keys[order[0]],
            "top2_gap": probs[order[0]] - probs[order[1]], "head_vs_logits_max_abs": d_head,
            "seconds": {"probabilities": round(sec_api, 4), "logz_ids": round(sec_logz, 4)}}

    if a.only:
        wanted = set(a.only.split(","))
        records = [p for p in planned if p["record"]["id"] in wanted]
        assert {p["record"]["id"] for p in records} == wanted, wanted - {p["record"]["id"] for p in records}
    else:
        records = planned[: a.limit] if a.limit else planned
    fails: dict = defaultdict(list)
    worst = defaultdict(float)
    q_out, r_out, npz = [], [], {}
    full = set(FULL_ROWS)
    t_loop = time.time()
    with torch.inference_mode():
        for n, p in enumerate(records):
            r = p["record"]
            items = [it for it in p["items"] if not it.get("image_pending")]
            rec = {"id": r["id"], "source": r["source"], "request_sha256": request_sha256(r["request"]),
                   "path": None if not items else "tree" if len(items) > 1 else "row",
                   "questions": [it["qid"] for it in items],
                   "image_pending": [it["qid"] for it in p["items"] if it.get("image_pending")],
                   "rejected": [q for q, _ in p["rejected"]]}
            prov = r.get("provenance", {}).get("request_sha256")
            if prov is not None and prov != rec["request_sha256"]:
                fails["request_sha256_vs_provenance"].append(r["id"])
            r_out.append(rec)
            if not items:
                continue
            state = r["request"]["state"]
            probs_tree = logz_tree = None
            if len(items) > 1:
                trees.clear()
                t = time.time()
                probs_tree = engine.probabilities(state, [it["qd"] for it in items])
                rec["seconds_tree"] = round(time.time() - t, 4)
                if len(trees) != 1 or trees[0][0] != items[0]["trunk"] or trees[0][1] != [it["branch"] for it in items]:
                    fails["tree_split_vs_rows_json"].append(r["id"])
                logz_tree = trees[0][2]
            for k, it in enumerate(items):
                key = f"{r['id']}/{it['qid']}"
                q, row = row_pass(state, it["qd"], it["ids"], it["groups"], key, (r["id"], it["qid"]) in full)
                if probs_tree is not None and eprompt.readout(engine.tokenizer, q, logz_tree[k]) != probs_tree[k]:
                    fails["readout_tree_logz_vs_probabilities"].append(key)
                gold = r["gold"].get(it["qid"])
                q_out.append({
                    "id": r["id"], "source": r["source"], "qid": it["qid"], "path": rec["path"], **row,
                    "probs_tree": None if probs_tree is None else probs_tree[k],
                    "logz_tree": None if logz_tree is None else [[float(logz_tree[k][i]) for i in g] for g in it["groups"]],
                    "near_tie": row["top2_gap"] <= NEAR_TIE,
                    "gold_key": gold, "gold_match": None if gold is None else gold == row["argmax_key"],
                })
            note_window(r["id"])
            if n % 25 == 0 or r["source"] in ("own", "red_arm", "card"):
                print(json.dumps({"n": n, "id": r["id"], "path": rec["path"], "rows": [x["row_len"] for x in q_out[-len(items):]],
                                  "s": round(sum(x["seconds"]["probabilities"] + x["seconds"]["logz_ids"] for x in q_out[-len(items):])
                                             + rec.get("seconds_tree", 0), 2),
                                  "elapsed": round(time.time() - t_loop)}), flush=True)
        base = next((p for p in records if p["record"]["id"] == "tv4_000"), None)
        rerun = None
        if base is not None:
            again = engine.probabilities(base["record"]["request"]["state"], [base["items"][0]["qd"]])[0]
            first = next(x["probs"] for x in q_out if x["id"] == "tv4_000")
            rerun = {"record": "tv4_000", "bit_equal": again == first}
            if not rerun["bit_equal"]:
                fails["rerun_tv4_000"].append("tv4_000")
        # the control arms (fixtures/red_arms.json): the plain row of each, against its base record's row
        arm_items, arms_sha = arms
        by_key = {(x["id"], x["qid"]): x for x in q_out}
        arm_out = []
        for it in arm_items:
            r = it["record"]
            base = by_key.get((r["base_id"], it["qid"]))
            if base is None:   # a subset run without the arm's base record
                continue
            key = f"{r['id']}/{it['qid']}"
            state = r["request"]["state"]
            if r["provenance"]["request_sha256"] != request_sha256(r["request"]):
                fails["request_sha256_vs_provenance"].append(r["id"])
            if encode(engine.tokenizer, engine.render(state, eprompt.as_question(it["qd"]))) != it["ids"]:
                fails["arm_ids_vs_engine_render"].append(key)
            _, row = row_pass(state, it["qd"], it["ids"], it["groups"], key)
            pb, pa = dict(zip(base["keys"], base["probs"])), dict(zip(row["keys"], row["probs"]))
            if set(pb) != set(pa):
                fails["arm_keys_vs_base"].append(key)
                continue
            dp = max(abs(pa[k] - pb[k]) for k in pa)
            arm_out.append({"id": r["id"], "source": r["source"], "qid": it["qid"], "path": "row",
                            "arm": r["provenance"]["arm"], "kind": r["provenance"]["kind"], "base_id": r["base_id"],
                            "request_sha256": request_sha256(r["request"]), **row,
                            "probs_base": base["probs"], "keys_base": base["keys"], "argmax_base": base["argmax_key"],
                            "max_abs_dp": dp, "argmax_changed": row["argmax_key"] != base["argmax_key"],
                            "red": dp > RED_ARM_MIN})
            note_window(r["id"])
            print(json.dumps({"arm": r["id"], "base": r["base_id"], "max_abs_dp": round(dp, 6),
                              "argmax": [base["argmax_key"], row["argmax_key"]]}), flush=True)
    note_window("end", always=True)
    loop_s = round(time.time() - t_loop, 1)
    ru = resource.getrusage(resource.RUSAGE_SELF)
    np.savez(out_npz, **npz)
    checks = {name: {"ok": not fails.get(name), "failed": fails.get(name, [])} for name in (
        "request_sha256_vs_provenance", "tree_split_vs_rows_json", "readout_ids_vs_rows_json", "hidden_shape_or_finite",
        "logz_vs_recorded_logits", "readout_logz_vs_probabilities", "head_vs_logits_1e-5",
        "readout_tree_logz_vs_probabilities", "rerun_tv4_000", "arm_ids_vs_engine_render", "arm_keys_vs_base")}
    wall = round(time.time() - t_start, 1)
    doc = {"repo": REPO, "revision": REV, "weights_sha256": WEIGHTS_SHA256, "snapshot": str(snap),
           "path": "AutoModel.from_pretrained(trust_remote_code=True, dtype=float32) on the CPU; model.engine defaults",
           "env": {"python": platform.python_version(), "torch": torch.__version__, "transformers": transformers.__version__,
                   "threads": torch.get_num_threads(), "interop_threads": torch.get_num_interop_threads(),
                   "thread_env": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")},
                   "machine": platform.machine(), "param_dtypes": dict(dtypes), "param_devices": dict(devices),
                   "sha256_seconds": sha_s, "load_seconds": load_s, "loop_seconds": loop_s, "wall_seconds": wall,
                   "cpu_seconds": round(ru.ru_utime + ru.ru_stime, 1),
                   "mean_cores": round((ru.ru_utime + ru.ru_stime) / wall, 2), "max_rss_bytes": ru.ru_maxrss,
                   "started": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t_start)),
                   "ended": time.strftime("%Y-%m-%d %H:%M:%S"), "measurement_window": windows},
           "fixtures_sha256": sha256_file(FIXTURES), "rows_sha256": sha256_file(ROWS), "header_bytes": n_hdr,
           "full_rows": [f"{i}/{q}" for i, q in FULL_ROWS], "worst": dict(worst),
           "checks": checks, "all_ok": all(c["ok"] for c in checks.values()),
           "counts": {"records": len(r_out), "questions": len(q_out),
                      "image_pending": [f"{x['id']}/{q}" for x in r_out for q in x["image_pending"]],
                      "rejected": [f"{x['id']}/{q}" for x in r_out for q in x["rejected"]],
                      "tree_requests": sum(1 for x in r_out if x["path"] == "tree"),
                      "tree_questions": sum(len(x["questions"]) for x in r_out if x["path"] == "tree")},
           "rerun": rerun, "requests": r_out, "questions": q_out,
           "red_arms": None if not arm_items else {
               "file": "fixtures/red_arms.json", "sha256": arms_sha, "threshold": RED_ARM_MIN,
               "rule": "red = some option's probability moves by more than the threshold against the base record's row",
               "arms": len(arm_out), "all_red": bool(arm_out) and all(x["red"] for x in arm_out), "records": arm_out}}
    out_json.write_text(json.dumps(doc, ensure_ascii=False) + "\n")
    print(json.dumps({"questions": len(q_out), "records": len(r_out), "out": str(out_json), "all_ok": doc["all_ok"],
                      "failed": {k: v["failed"][:5] for k, v in checks.items() if not v["ok"]}, "rerun": rerun,
                      "wall_s": wall, "mean_cores": doc["env"]["mean_cores"],
                      "red_arms": None if not arm_out else [round(x["max_abs_dp"], 6) for x in arm_out]}))
    return 0 if doc["all_ok"] else 1


def bucket_of(n: int) -> int | None:
    return next((b for b in BUCKETS if n <= b), None)


def rate(rows: list) -> dict:
    gold = [x for x in rows if x["gold_match"] is not None]
    return {"questions": len(rows), "near_tie": sum(x["near_tie"] for x in rows), "gold_n": len(gold),
            "gold_match": sum(x["gold_match"] for x in gold),
            "gold_rate": round(sum(x["gold_match"] for x in gold) / len(gold), 4) if gold else None}


def d1b_compare(mine: dict) -> dict:
    """The other conversion's float32 reference (oracle/records_oracle.json: the same provider code, its own venv and
    threads) against ours. Its records carry no request_sha256: their ids are mapped through its fixture file (named and hashed in its
    header) to request_sha256, and so to our records. Per common question: row ids, the row probabilities and the
    read-out log-probabilities; for tree requests also the API path (theirs `api.probs`, ours `probs_tree`)."""
    path = D1B_DIRS[0] / "records_oracle.json"
    doc = {"looked_in": [str(d) for d in D1B_DIRS], "checked_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    if not path.exists():
        return doc | {"status": "not arrived"}
    other = json.loads(path.read_text())
    fx = Path(other["fixtures"]["path"])
    fx_ok = fx.exists() and sha256_file(fx) == other["fixtures"]["sha256"]
    sha_of = {r["id"]: request_sha256(r["request"]) for r in json.loads(fx.read_text())["records"]} if fx_ok else {}
    # the same id when both fixtures have it with the same request; else the request alone (their `long_34k` is our
    # `own_long_34k_001`). Kev's fixture holds five pairs of records with one request (tv4x_composition_holdout_00 and
    # _02, ...): matching by the request alone would hide one of each pair.
    ours_sha = {r["id"]: r["request_sha256"] for r in mine["requests"]}
    ours_by_sha = {r["request_sha256"]: r["id"] for r in mine["requests"]}
    mine_q = {(x["id"], x["qid"]): x for x in mine["questions"]}
    rows, unmatched = [], []
    for rec in other["records"]:
        sha_r = sha_of.get(rec["id"])
        oid = rec["id"] if sha_r is not None and ours_sha.get(rec["id"]) == sha_r else ours_by_sha.get(sha_r)
        if oid is None:
            unmatched.append(rec["id"])
            continue
        api = rec.get("api") or {}
        for k, q in enumerate(rec["questions"]):
            x = mine_q.get((oid, q["name"]))
            if x is None:
                unmatched.append(f"{rec['id']}/{q['name']}")
                continue
            zl = {str(i): z for g, zg in zip(x["readout_ids"], x["logz"]) for i, z in zip(g, zg)}
            dz = max(abs(zl[i] - z) for i, z in q["group_logz"].items()) if set(zl) == set(q["group_logz"]) else None
            dp_api = None
            if api.get("path") == "tree" and x["probs_tree"] is not None and len(api.get("probs") or []) > k:
                dp_api = max(abs(a_ - b) for a_, b in zip(x["probs_tree"], api["probs"][k]))
            rows.append({"d1b_id": rec["id"], "id": oid, "qid": q["name"], "ids_equal": q["row_ids"] == x["ids"],
                         "max_abs_dp_row": max(abs(a_ - b) for a_, b in zip(x["probs"], q["probs"])),
                         "max_abs_dlogz_row": dz, "probs_bit_equal": x["probs"] == q["probs"],
                         "argmax_equal": x["argmax"] == max(range(len(q["probs"])), key=q["probs"].__getitem__),
                         "max_abs_dp_tree": dp_api})
    dz_all = [r["max_abs_dlogz_row"] for r in rows if r["max_abs_dlogz_row"] is not None]
    tree = [r["max_abs_dp_tree"] for r in rows if r["max_abs_dp_tree"] is not None]
    sha = sha256_file(path)
    summ_path = path.parents[1] / "results/oracle_summary.json"   # their summary of this file (tied by its sha256)
    summ = json.loads(summ_path.read_text()) if summ_path.exists() else {}
    summ = summ if summ.get("oracle_sha256") == sha else {}
    return doc | {
        "status": "compared" if rows else "arrived, no common question",
        "file": str(path), "sha256": sha, "mtime": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(path.stat().st_mtime)),
        "their_fixtures": other["fixtures"], "their_fixtures_hash_ok": fx_ok, "their_versions": other.get("versions"),
        "their_summary": str(summ_path) if summ else None, "their_threads": summ.get("threads"),
        "their_near_ties": summ.get("near_ties"), "their_tree_vs_row_max_abs_dp": summ.get("tree_vs_row_max_abs_dp"),
        "their_records": len(other["records"]),
        "common_questions": len(rows), "common_pairs_unique": len({(r["id"], r["qid"]) for r in rows}),
        "matched_by_request_only": sorted({f"{r['d1b_id']} -> {r['id']}" for r in rows if r["d1b_id"] != r["id"]}),
        "ours_only": [f"{x['id']}/{x['qid']}" for x in mine["questions"]
                      if (x["id"], x["qid"]) not in {(r["id"], r["qid"]) for r in rows}],
        "unmatched": unmatched,
        "ids_equal": sum(r["ids_equal"] for r in rows), "argmax_equal": sum(r["argmax_equal"] for r in rows),
        "probs_bit_equal": sum(r["probs_bit_equal"] for r in rows),
        "max_abs_dp_row": max((r["max_abs_dp_row"] for r in rows), default=None),
        "mean_abs_dp_row": sum(r["max_abs_dp_row"] for r in rows) / len(rows) if rows else None,
        "max_abs_dlogz_row": max(dz_all, default=None), "logz_compared": len(dz_all),
        "tree_questions": len(tree), "max_abs_dp_tree": max(tree, default=None),
        "rows": rows}


def summarize(path: Path) -> int:
    ref = json.loads(path.read_text())
    out = K / "results/reference_summary.json"
    assert not out.exists(), f"refusing to overwrite {out}"
    qs = ref["questions"]
    per_q = [{"id": x["id"], "qid": x["qid"], "source": x["source"], "type": x["type"], "path": x["path"],
              "row_len": x["row_len"], "bucket": bucket_of(x["row_len"]), "top2_gap": x["top2_gap"],
              "near_tie": x["near_tie"], "argmax_key": x["argmax_key"], "gold_key": x["gold_key"],
              "gold_match": x["gold_match"]} for x in qs]
    gaps = sorted(x["top2_gap"] for x in per_q)

    def pct(q):
        return gaps[min(len(gaps) - 1, int(round(q * (len(gaps) - 1))))]

    by = lambda field: {k: rate([x for x in per_q if x[field] == k]) for k in sorted({x[field] for x in per_q}, key=str)}  # noqa: E731
    base = next(x for x in qs if x["id"] == "tv4_000")
    arm = next(x for x in qs if x["id"] == "red_arm_000")
    dp = max(abs(a_ - b) for a_, b in zip(base["probs"], arm["probs"]))
    doc = {"reference": str(path.relative_to(K) if path.is_relative_to(K) else path), "reference_sha256": sha256_file(path),
           "records": ref["counts"]["records"], "questions": len(per_q),
           "image_pending": ref["counts"]["image_pending"], "rejected": ref["counts"]["rejected"],
           "near_tie": {"threshold": NEAR_TIE, "count": sum(x["near_tie"] for x in per_q),
                        "ids": [f"{x['id']}/{x['qid']}" for x in per_q if x["near_tie"]]},
           "top2_gap": {"min": gaps[0], "p05": pct(0.05), "p10": pct(0.10), "p50": pct(0.5)},
           "gold_overall": rate(per_q), "by_type": by("type"), "by_source": by("source"), "by_bucket": by("bucket"),
           "by_path": by("path"),
           "red_arm_000": {"base": "tv4_000/answer", "arm": "red_arm_000/answer", "keys": base["keys"],
                           "probs_base": base["probs"], "probs_arm": arm["probs"], "max_abs_dp": dp,
                           "argmax_base": base["argmax_key"], "argmax_arm": arm["argmax_key"], "threshold": RED_ARM_MIN,
                           "moves": dp > RED_ARM_MIN,
                           "status": ("invalid as a red arm on d1-3B, recorded only: the one-word change ('correctly' -> "
                                      "'incorrectly') moves the float32 answer by this much; a property of the model, not "
                                      "a defect of the path (decision, 2026-10-08)") if dp <= RED_ARM_MIN else "red"},
           "red_arms": None if not ref.get("red_arms") else {
               "file": ref["red_arms"]["file"], "sha256": ref["red_arms"]["sha256"], "threshold": RED_ARM_MIN,
               "all_red": ref["red_arms"]["all_red"],
               "arms": [{"id": x["id"], "arm": x["arm"], "kind": x["kind"], "base": f"{x['base_id']}/{x['qid']}",
                         "max_abs_dp": x["max_abs_dp"], "argmax_base": x["argmax_base"], "argmax_arm": x["argmax_key"],
                         "argmax_changed": x["argmax_changed"], "red": x["red"], "row_len": x["row_len"]}
                        for x in ref["red_arms"]["records"]]},
           "other_conversion": d1b_compare(ref), "per_question": per_q}
    out.write_text(json.dumps(doc, ensure_ascii=False, indent=1) + "\n")
    print(json.dumps({k: doc[k] for k in ("records", "questions", "gold_overall", "top2_gap")} |
                     {"near_tie": doc["near_tie"]["count"], "red_arm_000_dp": dp, "other_conversion": doc["other_conversion"]["status"],
                      "red_arms": None if not doc["red_arms"] else [(x["id"], round(x["max_abs_dp"], 6)) for x in doc["red_arms"]["arms"]]}))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="no weights: print the ids and read-out groups, then stop")
    ap.add_argument("--dry-run-out", default=None, help="also write the dry-run list to this JSON path")
    ap.add_argument("--check-header", action="store_true", help="the weights' header against the Hub's and a meta build")
    ap.add_argument("--make-red-arms", action="store_true", help="write fixtures/red_arms.json (the adopted arms)")
    ap.add_argument("--summarize", default=None, help="a results/reference_<tag>.json: write results/reference_summary.json")
    ap.add_argument("--source", "--model-dir", dest="source", default=None,
                    help="the snapshot dir with model.safetensors (default: the HF cache at REV)")
    ap.add_argument("--tag", default="fp32")
    ap.add_argument("--limit", type=int, default=0, help="first N records only (outputs get _smoke)")
    ap.add_argument("--only", default="", help="these record ids only, comma-separated (outputs get _smoke)")
    ap.add_argument("--out-dir", default=None, help="where the outputs go (default: results/)")
    ap.add_argument("--threads", type=int, default=MAX_THREADS, help=f"torch intra-op threads, at most {MAX_THREADS}")
    a = ap.parse_args()
    if a.summarize:
        return summarize(Path(a.summarize).resolve())
    if a.make_red_arms:
        return make_red_arms()
    # before torch and transformers are imported: their thread pools and the copy of the provider's code
    for var in ("OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[var] = str(a.threads)
    os.environ.setdefault("HF_MODULES_CACHE", str(K / "cache/hf_modules"))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    if a.check_header:
        return check_header(a)
    prompt = provider("prompt")
    tok = load_tokenizer()
    settings = engine_settings(tok, prompt)
    rows_by_key = {(x["id"], x["qid"]): x for x in json.loads(ROWS.read_text())["rows"]}
    planned = plan(prompt, tok, settings, load_fixtures(), rows_by_key)
    if a.dry_run:
        return dry_run(planned, Path(a.dry_run_out) if a.dry_run_out else None)
    return run(planned, plan_arms(prompt, tok, settings), a)


if __name__ == "__main__":
    sys.exit(main())
