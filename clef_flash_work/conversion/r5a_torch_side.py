#!/usr/bin/env python3
"""Round 5a: the torch-side values the host test compares against, made in the reference venv (torch 2.9.0,
transformers 5.17.0 = the reference's versions) so that the host test itself never imports a framework.

    HF_HUB_OFFLINE=1 venv-ref/bin/python scripts/r5a_torch_side.py      (from K) -> cache/r5a/torch_rope.npz,
                                                                           results/r5a_author_refusals.json

1. cos / sin: for every run of host/test_host.py (214 reference + 3 red arms + 40 held-out), the author's rotary
   (Qwen3_5TextRotaryEmbedding(text_config).forward, as r3's scripts/r3_pixpos.py calls it) on the run's own rotary
   input (npz rope_pos [3, T]) right-padded to the run's bucket L the host's way (maximum + 1, + 2, ... on all three
   planes; written out here again, not imported) -> keys "<set>__<id>__<arm>__cos" / "__sin", float32 [1, L, 64].
2. Refusals: the checkpoint's own systemone() (joint_schema_model.py, sha256 pinned, path-imported unchanged) called
   with model=None and processor=None on every case of host/test_host.py refusal_cases(): its checks run before the
   model is touched, so a ValueError is the author's refusal and its message; a request that passes them stops at
   `processor.tokenizer` (AttributeError) = accepted by the author's checks. For the over-4,096-token case the author's
   encode_record with the processor's tokenizer gives the token count the author's model would read (no cut below
   16,384).
3. Encode cases (host/test_host.py encode_cases()): the author's encode_record with the processor's tokenizer at small
   max_length / max_state_tokens -> ids and spans, or its ValueError message."""
import copy
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

K = Path(__file__).resolve().parents[1]
SNAP = K / "hf/hub/models--Cloudflare--clef-flash/snapshots/f0b7d3cd65296abfe459e6c228b618f5d8b26d2f"
JSM_SHA256 = "0e304cf7c6500e8bb59bef7e2afd2c6373f82596dfb3b57d1aa93c175e2dc3a3"
BUCKETS = (256, 512, 1024, 2048, 4096)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    test = load_module("r5a_host_test", K / "host/test_host.py")          # before torch: its import-time assert holds
    runs = test.load_runs()
    import torch
    import transformers
    from transformers import AutoConfig, AutoProcessor
    from transformers.models.qwen3_5 import modeling_qwen3_5 as M

    t0 = time.time()
    rot = M.Qwen3_5TextRotaryEmbedding(AutoConfig.from_pretrained(str(SNAP)).text_config)
    arrays, lengths = {}, {}
    for run in runs:
        rp = np.load(run["npz"])["rope_pos"].astype(np.int64)
        T = rp.shape[1]
        L = next(b for b in BUCKETS if T <= b)
        planes = np.empty((3, L), np.int64)
        planes[:, :T] = rp
        planes[:, T:] = int(rp.max()) + 1 + np.arange(L - T, dtype=np.int64)[None, :]
        with torch.no_grad():
            c, s = rot(torch.zeros(1, L, 8, dtype=torch.float32), torch.from_numpy(planes)[:, None, :])
        k = f"{run['set']}__{run['id']}__{run['arm']}"
        arrays[k + "__cos"] = c.numpy().astype(np.float32)
        arrays[k + "__sin"] = s.numpy().astype(np.float32)
        lengths[k] = L
    out = K / "cache/r5a/torch_rope.npz"
    np.savez(out, **arrays)
    t_rope = time.time() - t0

    jsm_path = SNAP / "joint_schema_model.py"
    assert hashlib.sha256(jsm_path.read_bytes()).hexdigest() == JSM_SHA256
    jsm = load_module("joint_schema_model", jsm_path)
    tok = AutoProcessor.from_pretrained(str(SNAP)).tokenizer
    cases = []
    for case in test.refusal_cases():
        req = copy.deepcopy(case["request"])
        try:
            jsm.systemone(None, None, req)
            raised, msg = None, None
        except Exception as e:                                           # noqa: BLE001 - the class is the result
            raised, msg = type(e).__name__, str(e)
        item = {"name": case["name"], "raised": raised, "message": msg,
                "author_refuses": raised == "ValueError",
                "passes_author_checks": raised == "AttributeError" and "tokenizer" in (msg or "")}
        if case["name"] == "over_4096_tokens":
            item["tokens"] = len(jsm.encode_record(tok, copy.deepcopy(case["request"])).input_ids)
        cases.append(item)
    encode_cases = []
    for case in test.encode_cases():
        try:
            enc = jsm.encode_record(tok, copy.deepcopy(case["request"]), **case["kwargs"])
            item = {"name": case["name"], "raised": None, "ids": list(enc.input_ids),
                    "questions": [[q.question_id, list(q.question_span), [list(s) for s in q.option_spans],
                                   list(q.option_ids)] for q in enc.questions]}
        except Exception as e:                                           # noqa: BLE001 - the class is the result
            item = {"name": case["name"], "raised": type(e).__name__, "message": str(e)}
        encode_cases.append(item)
    doc = {"what": "round 5a: the author's systemone() checks on the host test's refusal cases (model and processor None) "
                   "and the author's encode_record on its encode cases",
           "versions": {"torch": torch.__version__, "transformers": transformers.__version__},
           "joint_schema_model_sha256": JSM_SHA256, "cases": cases, "encode_cases": encode_cases,
           "torch_rope": {"file": str(out.relative_to(K)), "runs": len(lengths), "seconds": round(t_rope, 1),
                          "lengths": {str(b): sum(v == b for v in lengths.values()) for b in BUCKETS}}}
    (K / "results/r5a_author_refusals.json").write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps({"rope_runs": len(lengths), "rope_seconds": round(t_rope, 1),
                      "cases": [(c["name"], c["raised"], c["message"][:60] if c["message"] else None, c.get("tokens"))
                                for c in cases],
                      "encode_cases": [(c["name"], c["raised"], len(c["ids"]) if "ids" in c else c["message"])
                                       for c in encode_cases]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
