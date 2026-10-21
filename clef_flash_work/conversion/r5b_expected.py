#!/usr/bin/env python3
"""Round 5b: examples/run_example.expected.json of the upload folder, from the reference (no graph is run).

    venv-host/bin/python scripts/r5b_expected.py      -> staging/upload/examples/run_example.expected.json
                                                         staging/upload/examples/request.json (img_10's request)

requests: the four check records (CHECK) with the reference's option ids, probabilities, most likely option and near-tie
flag per question (ref/records_ref.json = the author's float32 systemone() on the CPU), the answer fields of the
reference's response, and the input tokens and row-graph length the staged host gives them (asserted equal to the
reference's token count). refusals: the error name and message the staged host raises on each request of the
example's REFUSALS before any graph runs (message null where it holds a path of this machine)."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

K = Path(__file__).resolve().parents[1]
UP = K / "staging/upload"
sys.path.insert(0, str(UP / "host"))
import clef_litert as C  # noqa: E402

CHECK = [("semif_a3f18f3a63d45345942b", "text", None), ("own_t01", "text", None), ("own_j02", "text", None),
         ("img_02", "g256", "fixtures/images/bars_640x480.png")]
REQUEST_EXAMPLE = "img_10"


def main() -> int:
    from tokenizers import Tokenizer

    ref = {(r["id"], r["arm"]): r for r in json.loads((K / "ref/records_ref.json").read_text())["rows"]}
    public = {r["id"]: r for r in json.loads((UP / "fixtures/requests_public.json").read_text())["records"]}
    host = C.ClefHost(Tokenizer.from_file(str(UP / C.FOLDER["tokenizer"])), C.Bf16Table(UP / C.FOLDER["embed_tokens"]),
                      C.Bf16Table(UP / C.FOLDER["lm_head"]),
                      C.JointHead.from_files(UP / C.FOLDER["joint_head"], UP / C.FOLDER["joint_head_config"]))
    requests = []
    for rid, arm, image in CHECK:
        row = ref[(rid, arm)]
        req = dict(public[rid]["request"])
        assert "withheld" not in req, rid
        tile = {"g256": 256, "g448": 448}.get(arm, 448)
        if image:
            req["images"] = [str(UP / image)]
        enc = host.encode(req, tile=tile)
        assert enc.record.tokens == row["tokens"], (rid, enc.record.tokens, row["tokens"])
        qs = {}
        for q in row["questions"]:
            ans = row["systemone_response"]["answers"][q["question_id"]]
            qs[q["question_id"]] = {"type": q["type"], "option_ids": q["option_ids"], "probabilities": q["probs"],
                                    "argmax": q["argmax_id"], "near_tie": q["near_tie"], "answer_fields": sorted(ans)}
        requests.append({"record": rid, "arm": arm, "tile": tile, "image": image, "input_tokens": enc.record.tokens,
                         "length": enc.length, "questions": qs})
    spec = importlib.util.spec_from_file_location("run_example", UP / "examples/run_example.py")
    ex = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ex)
    refusals = {}
    for name, (req, tile) in ex._refusals().items():
        try:
            host.encode(req, tile=tile)
            raise SystemExit(f"{name}: not refused")
        except C.RequestError as e:
            msg = str(e)
            refusals[name] = {"error": type(e).__name__, "message": None if str(UP) in msg or str(Path.home()) in msg else msg}
    doc = {"what": "what examples/run_example.py --check compares with: the provider's own code (Cloudflare/clef-flash "
                   "at f0b7d3cd, joint_schema_model.py systemone()) in float32 on the CPU, one row per question; "
                   "and the host's refusals",
           "reference": {"python": "3.12.11", "torch": "2.9.0", "transformers": "5.17.0", "device": "cpu",
                         "dtype": "float32", "threads": 10},
           "line": {"near_tie": 0.02, "max_abs_dp": 0.02, "mean_of_request_means": 0.002},
           "requests": requests, "refusals": refusals}
    (UP / "examples/run_example.expected.json").write_text(json.dumps(doc, indent=1) + "\n")
    req = dict(public[REQUEST_EXAMPLE]["request"])
    (UP / "examples/request.json").write_text(json.dumps(req, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps({"requests": [(r["record"], r["input_tokens"], r["length"]) for r in requests],
                      "refusals": {k: v["error"] for k, v in refusals.items()},
                      "request_json": REQUEST_EXAMPLE}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
