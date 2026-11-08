"""Export DecGraph (float32 weights, one length L per file) with litert-torch.

    PYTHONHASHSEED=0 python conversion/export_dec.py --weights merged_fp32.safetensors --L 128 \
        --form k4 --out exports/L128_w32.tflite [--repo .]

--repo holds the host and the files it reads (host/, tokenizer.json, embedding.safetensors, decision_head.safetensors,
temperature.json, examples/): by default the repository this script is in. Run this file as a script: the tensor names in
the output carry the module path of the graph (dec_graph.DecGraph/dec_graph.Layer_6/...), so an import under another
module name writes other bytes.

--form names the form of dec_graph.py: k4, k4_exp_silu or mean_norm.

1. The graph object is checked before the export, on the two short example requests through the host's own inputs, head and
   answers: every winner equal to examples/*.expected.json (the original inference.py) and every probability within 1e-4. A
   request longer than L is skipped and named in the record: default_request has 85 tokens, so the L64 file checks readme_request
   only.
2. litert_torch.convert(graph, sample_kwargs={embeds, cos, sin, mask}).export(out): the signature's input names are the
   keys, its output name is "hidden". The sample is the README request built at L.
3. A static scan of the output (tflite_scan.py): the signature against the contract, no CUSTOM op, no STABLEHLO op, no
   INT64 tensor, no tensor of rank above 4. Prints the bytes and SHA-256 and writes <out>.export.json next to it."""
import argparse
import importlib.metadata
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import dec_graph as G  # noqa: E402

HIDDEN = 896
PROBE_LINE = 1e-4


def winner(answer):
    if answer["type"] == "choice":
        return answer["choice"]
    if answer["type"] == "noul":
        return answer["decision"]
    p = list(answer["probabilities"].values())
    return max(range(len(p)), key=p.__getitem__)


def tensors(feeds):
    return {k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in feeds.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=Path, required=True, help="the file merge_lora.py wrote")
    ap.add_argument("--L", type=int, required=True)
    ap.add_argument("--form", required=True, choices=sorted(G.FORMS))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--repo", type=Path, default=Path(HERE).parent)
    a = ap.parse_args()
    if os.environ.get("PYTHONHASHSEED") != "0":
        raise SystemExit("run with PYTHONHASHSEED=0")
    if a.out.exists():
        raise SystemExit(f"refusing to overwrite {a.out}")
    torch.set_num_threads(4)
    sys.path.insert(0, str(a.repo / "host"))
    from pocket_dec_litert import PocketDecisionLiteRT
    host = PocketDecisionLiteRT(os.path.abspath(a.repo), max_length=a.L)
    graph = G.load_graph(a.form, a.weights)
    checks = []
    sample = None
    for name in ("readme_request", "default_request"):
        request = json.loads((a.repo / f"examples/{name}.json").read_text(encoding="utf-8"))
        expected = json.loads((a.repo / f"examples/{name}.expected.json").read_text(encoding="utf-8"))
        if expected["input_tokens"] > a.L:
            checks.append({"request": name, "tokens": expected["input_tokens"], "skipped": f"longer than L = {a.L}"})
            continue
        encoded = host.encode(request["state"], request["questions"])
        assert len(encoded["ids"]) == expected["input_tokens"], name
        feeds = tensors(host._feeds(encoded, a.L))
        if sample is None:
            sample = feeds
        with torch.no_grad():
            hidden = graph(**feeds)["hidden"][0].numpy()
        answers = host._answers(encoded, host._head(hidden, encoded), host.temperature)
        dp = max(abs(answers[q]["probabilities"][k] - p) for q, e in expected["answers"].items()
                 for k, p in e["probabilities"].items())
        same = all(winner(answers[q]) == winner(e) for q, e in expected["answers"].items())
        checks.append({"request": name, "tokens": len(encoded["ids"]), "max_abs_dp": dp, "winners_equal": same})
    print(json.dumps(checks))
    if not all(c.get("skipped") or (c["winners_equal"] and c["max_abs_dp"] <= PROBE_LINE) for c in checks):
        raise SystemExit("the graph module does not reproduce the original answers")
    import litert_torch
    t0 = time.perf_counter()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    litert_torch.convert(graph, sample_kwargs=sample).export(str(a.out))
    seconds = round(time.perf_counter() - t0, 1)
    from tflite_scan import scan
    s = scan(a.out)
    want_in = sorted([("embeds", [1, a.L, HIDDEN], "FLOAT32"), ("cos", [1, a.L, 64], "FLOAT32"),
                      ("sin", [1, a.L, 64], "FLOAT32"), ("mask", [1, 1, a.L, a.L], "FLOAT32")])
    sig = s["signatures"]
    got_in = sorted((i["name"], i["shape"], i["dtype"]) for i in sig[0]["inputs"]) if len(sig) == 1 else None
    got_out = [(o["name"], o["shape"], o["dtype"]) for o in sig[0]["outputs"]] if len(sig) == 1 else None
    problems = {"signature": got_in != want_in or got_out != [("hidden", [1, a.L, HIDDEN], "FLOAT32")],
                "custom_op": bool(s["custom_op_count"]), "stablehlo": bool(s["stablehlo_ops"]),
                "int64": bool(s["int64_tensor_count"]), "rank_above_4": bool(s["rank_gt4_tensor_count"])}
    doc = {"L": a.L, "form": a.form, "bytes": s["bytes"], "sha256": s["sha256"], "seconds": seconds, "checks": checks,
           "problems": problems, "op_histogram": s["op_histogram"], "signatures": sig,
           "versions": {p: importlib.metadata.version(p) for p in ("torch", "litert-torch", "litert-converter",
                                                                    "ai-edge-litert", "numpy")}}
    a.out.with_name(a.out.name + ".export.json").write_text(json.dumps(doc, indent=1) + "\n")
    print(f"{a.out}: {s['bytes']:,} bytes, SHA-256 {s['sha256']}")
    if any(problems.values()):
        raise SystemExit("export checks failed: " + json.dumps(problems))


if __name__ == "__main__":
    main()
