"""Round 3 acceptance 6 (reference side): the provider's own picture path on the tiny VL model; round 8: the same on
the real weights.

    $REF scripts/d1_vision_ref.py
    $REF scripts/d1_vision_ref.py \
        --source <snapshot> --tag real [--threads 8]

The tiny VL model (d1_vision_graph.load_vl: the provider's D1Model on the tiny config, lm_head tied to the table) with
its SystemOne engine (`model.engine`, the provider's defaults; the processor and tokenizer from hf_small/) answers
REQUESTS, each one question with pictures = `SystemOne._request` -> cap_pixels -> `_image_markup` -> `_image_inputs`
(the processor, then the cut to max(mask.sum(1)) patches) -> `_one_pass(**inputs, logits_to_keep=1)` (the plain pass)
-> the read-out. Hooks record, without changing anything: the text and the processor's tensors (`_image_inputs`),
transformers' image features (`get_image_features`: the tower's last_hidden_state and the per-tile mm rows), the
language model's input embeddings (after `masked_scatter`) and its last hidden state (the answer slot = the last
position), and the response (probabilities, tokens read).
Outputs: results/tiny_vision_e2e_ref.json (requests, texts, ids, probabilities), cache/vision/tiny_e2e_ref.npz.

Round 8 (--source <snapshot> --tag real): the model is loaded as the text reference loads it (scripts/d1_reference.py):
`AutoModel.from_pretrained(<snapshot>, trust_remote_code=True, dtype=torch.float32)` = D1Model on the CPU, after the
weights' bytes and sha256 and the safetensors header are checked against the Hub's and the snapshot's code and
tokenizer files against hf_small/; the loading info must be clean and every parameter float32. `engine = model.engine`
(SystemOne with its defaults; the tokenizer and the processor come from the snapshot). The questions go in through
the engine's own prompt module (transformers' copy of prompt.py: its isinstance checks know only its own classes).
REQUESTS_REAL (pictures K-relative): one_tile = card_cats_001/cats + the card's COCO photo (cache/realv/coco_cats.jpg,
measurement only: never committed, never uploaded), split_thumbnail = tv4x_qnli_07 + the synthetic s1280x853 (2 x 3
tiles + thumbnail), two_pictures = tv4s_00 + the synthetic s512x512 and s333x777. The same hooks as the tiny run; the
npz also keeps every tensor the processor returned (after the provider's cut) and the tower's last hidden state.
Per request: seconds, the measurement window at its start and end. Outputs (never overwritten):
results/real_vision_e2e_ref.json, cache/real/image/real_e2e_ref.npz.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import platform
import resource
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image

K = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(K / "scripts"))
from d1_common import HF_SMALL, load_fixtures, provider, sha256_file  # noqa: E402

OUT = K / "results/tiny_vision_e2e_ref.json"
NPZ = K / "cache/vision/tiny_e2e_ref.npz"
REQUESTS = [  # name, fixture record (state + question), question name, pictures (cache/vision/img_<name>.png)
    ("one_tile", "card_cats_001", "cats", ["s100x60"]),
    ("split_thumbnail", "tv4x_qnli_07", None, ["s1100x520"]),
    ("two_pictures", "tv4s_00", None, ["s512x512", "s100x60"]),
]
OUT_REAL = K / "results/real_vision_e2e_ref.json"
NPZ_REAL = K / "cache/real/image/real_e2e_ref.npz"
REQUESTS_REAL = [  # name, fixture record, question name, pictures (K-relative files)
    ("one_tile", "card_cats_001", "cats", ["cache/realv/coco_cats.jpg"]),
    ("split_thumbnail", "tv4x_qnli_07", None, ["cache/vision/img_s1280x853.png"]),
    ("two_pictures", "tv4s_00", None, ["cache/vision/img_s512x512.png", "cache/vision/img_s333x777.png"]),
]
# round 8: card_cats_001's picture is COCO val2017 000000039769.jpg (two cats)
GOLD_ROUND8 = {"card_cats_001/cats": "two"}
COCO = {"cache/realv/coco_cats.jpg": {"url": "http://images.cocodataset.org/val2017/000000039769.jpg",
                                      "sha256": "dea9e7ef97386345f7cff32f9055da4982da5471c48d575146c796ab4563b04e",
                                      "bytes": 173131}}
MAX_THREADS = 8      # round 8: at most 8 threads on the shared Mac
NEAR_TIE = 0.02


def option_keys(qd: dict) -> list[str]:
    kind = qd.get("type", "choice")
    if kind == "noul":
        return ["true", "false"]
    if kind == "score":
        return [str(i) for i in range(len(qd["criteria"]))]
    return list(qd["criteria"].keys())


def load_real(snap: Path) -> tuple:
    """D1Model as scripts/d1_reference.py loads it, with the same checks; returns (model, engine prompt module, info)."""
    from transformers import AutoModel

    from d1_reference import HEADER, WEIGHTS_BYTES, WEIGHTS_SHA256, read_header

    w = snap / "model.safetensors"
    assert w.stat().st_size == WEIGHTS_BYTES, w.stat().st_size
    t = time.time()
    assert sha256_file(w) == WEIGHTS_SHA256, "model.safetensors sha256 differs from the Hub's LFS record"
    sha_s = round(time.time() - t, 1)
    _, local_header = read_header(w)
    assert local_header == json.loads(HEADER.read_text())["header"], "header differs from the Hub's"
    for name in ("config.json", "processor_config.json", "chat_template.jinja", "tokenizer.json", "tokenizer_config.json",
                 "prompt.py", "runner.py", "hybrid.py", "lfm2_vl.py", "modeling_d1.py", "api.py"):
        assert (snap / name).read_bytes() == (HF_SMALL / name).read_bytes(), name
    t = time.time()
    model, info = AutoModel.from_pretrained(str(snap), trust_remote_code=True, dtype=torch.float32, output_loading_info=True)
    load_s = round(time.time() - t, 1)
    assert not any(info.get(k) for k in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")), info
    model.eval()
    dtypes = Counter(str(p.dtype) for p in model.parameters())
    devices = Counter(p.device.type for p in model.parameters())
    assert set(dtypes) == {"torch.float32"} and set(devices) == {"cpu"}, (dtypes, devices)
    eprompt = importlib.import_module(type(model).__module__.rsplit(".", 1)[0] + ".prompt")
    return model, eprompt, {"source": str(snap), "class": type(model).__name__, "weights_sha256": WEIGHTS_SHA256,
                            "sha256_seconds": sha_s, "load_seconds": load_s, "param_dtypes": dict(dtypes),
                            "param_devices": dict(devices), "attn_implementation": model.config._attn_implementation}


def picture_record(p: str, real: bool) -> dict:
    path = K / p if real else K / f"cache/vision/img_{p}.png"
    im = Image.open(path)
    rec = {"file": str(path.relative_to(K)), "bytes": path.stat().st_size, "sha256": sha256_file(path),
           "format": im.format, "mode": im.mode, "wh": list(im.size)}
    if real and p in COCO:
        assert rec["sha256"] == COCO[p]["sha256"] and rec["bytes"] == COCO[p]["bytes"], rec
        rec["url"] = COCO[p]["url"]
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="tiny", help="tiny | the snapshot directory of LiquidAI/d1-3B (round 8)")
    ap.add_argument("--tag", default="tiny", help="tiny | real")
    ap.add_argument("--threads", type=int, default=0, help="torch threads (default 4 tiny, 8 real; at most 8)")
    a = ap.parse_args()
    real = a.tag != "tiny"
    assert real == (a.source != "tiny"), "--tag real goes with --source <snapshot>"
    out, npz_path = (OUT_REAL, NPZ_REAL) if real else (OUT, NPZ)
    for p in (out, npz_path):
        assert not p.exists(), f"refusing to overwrite {p}"
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    threads = a.threads or (MAX_THREADS if real else 4)
    assert 1 <= threads <= MAX_THREADS, threads
    torch.set_num_threads(threads)
    t0 = time.time()
    if real:
        from d1_reference import window_now

        model, eprompt, info = load_real(Path(a.source).expanduser())
        engine = model.engine
        assert engine.device.type == "cpu" and engine.calibration is None
        parse = eprompt.as_question
        requests = REQUESTS_REAL
    else:
        import d1_vision_graph as VG

        window_now = lambda: None  # noqa: E731
        model, info = VG.load_vl("tiny")
        # where D1Model.engine (model.name_or_path, copied from the config at construction) and SystemOne
        # (config._name_or_path) find the tokenizer and the processor
        model.name_or_path = model.config._name_or_path = str(HF_SMALL)
        engine = model.engine
        parse = provider("prompt").as_question
        requests = REQUESTS
    assert type(engine).__module__.endswith("runner") and engine._one_pass is model
    fixtures = {r["id"]: r for r in load_fixtures()["records"]}
    rec: dict = {}
    orig_inputs = engine._image_inputs

    def image_inputs(text, images):
        out = orig_inputs(text, images)
        rec["text"] = text
        rec["inputs"] = {k: v.detach().clone() for k, v in out.items() if hasattr(v, "detach")}
        return out

    engine._image_inputs = image_inputs
    orig_features = model.model.get_image_features

    def get_image_features(**kw):
        out = orig_features(**kw)
        rec["tower_last_hidden"] = out.last_hidden_state.detach().clone()
        rec["mm"] = [t.detach().clone() for t in out.pooler_output]
        return out

    model.model.get_image_features = get_image_features
    lm = model.model.language_model
    h_pre = lm.register_forward_pre_hook(lambda m, args, kwargs: rec.__setitem__("embeds", kwargs["inputs_embeds"].detach().clone()),
                                         with_kwargs=True)
    h_post = lm.register_forward_hook(lambda m, args, out: rec.__setitem__("hidden", out.last_hidden_state.detach().clone()))
    store, rows = {}, []
    t_loop = time.time()
    for name, rid, qname, pics in requests:
        r = fixtures[rid]
        state = r["request"].get("state")
        qn = qname or next(iter(r["request"]["questions"]))
        qd = r["request"]["questions"][qn]
        images = [Image.open(K / p) if real else Image.open(K / f"cache/vision/img_{p}.png") for p in pics]
        rec.clear()
        win0 = window_now()
        t = time.time()
        with torch.inference_mode():
            probs, read = engine._request(state, [parse(qd)], images)
        sec = round(time.time() - t, 1)
        ids = rec["inputs"]["input_ids"][0].tolist()
        hidden = rec["hidden"][0]
        store[f"ids__{name}"] = np.asarray(ids, np.int64)
        store[f"hidden_slot__{name}"] = hidden[-1].numpy()
        store[f"hidden__{name}"] = hidden.numpy()
        store[f"embeds__{name}"] = rec["embeds"][0].numpy()
        store[f"mm__{name}"] = torch.cat(rec["mm"]).numpy()
        store[f"pixel_values__{name}"] = rec["inputs"]["pixel_values"].numpy()
        store[f"spatial_shapes__{name}"] = rec["inputs"]["spatial_shapes"].numpy()
        if real:
            store[f"pixel_attention_mask__{name}"] = rec["inputs"]["pixel_attention_mask"].numpy()
            store[f"attention_mask__{name}"] = rec["inputs"]["attention_mask"].numpy()
            store[f"tower_last_hidden__{name}"] = rec["tower_last_hidden"].numpy()
            store[f"mm_rows_per_tile__{name}"] = np.asarray([int(m.shape[0]) for m in rec["mm"]], np.int64)
        row = {"name": name, "record": rid, "question": qn, "type": qd.get("type", "choice"), "pictures": pics,
               "request": {"state": state, "question": qd}, "text": rec["text"],
               "text_sha256": hashlib.sha256(rec["text"].encode()).hexdigest(), "tokens": len(ids),
               "tokens_read": read, "image_positions": int(sum(i == model.config.image_token_id for i in ids)),
               "mm_rows": int(store[f"mm__{name}"].shape[0]), "tiles": int(rec["inputs"]["pixel_values"].shape[0]),
               "cut": int(rec["inputs"]["pixel_values"].shape[1]), "probs": probs[0],
               "max_abs_hidden_slot": float(np.abs(store[f"hidden_slot__{name}"]).max())}
        if real:
            keys = option_keys(qd)
            order = sorted(range(len(probs[0])), key=lambda i: -probs[0][i])
            gap = probs[0][order[0]] - probs[0][order[1]]
            gold_fixture = r.get("gold", {}).get(qn)
            gold = GOLD_ROUND8.get(f"{rid}/{qn}", gold_fixture)
            row.update(keys=keys, argmax_key=keys[order[0]], top2_gap=gap, near_tie=gap <= NEAR_TIE,
                       gold_fixture=gold_fixture, gold=gold, gold_source=("round 8 decision" if
                                                                          f"{rid}/{qn}" in GOLD_ROUND8 else "fixture"),
                       argmax_equals_gold=None if gold is None else keys[order[0]] == gold,
                       picture_files=[picture_record(p, True) for p in pics],
                       spatial_shapes=rec["inputs"]["spatial_shapes"].tolist(),
                       real_patches_per_tile=rec["inputs"]["pixel_attention_mask"].sum(1).tolist(),
                       mm_rows_per_tile=store[f"mm_rows_per_tile__{name}"].tolist(),
                       seconds=sec, window_at_start=win0, window_at_end=window_now())
        rows.append(row)
        print(name, row["tokens"], row["image_positions"], row["tiles"], row["cut"], [round(p, 6) for p in row["probs"]],
              row.get("argmax_key"), row.get("seconds"), flush=True)
    loop_s = round(time.time() - t_loop, 1)
    h_pre.remove()
    h_post.remove()
    np.savez(npz_path, **store)
    doc = {"what": ("round 8 reference: the provider's plain pass with pictures on the real weights (float32 CPU)" if real
                    else "round 3 acceptance 6 reference: the provider's plain pass with pictures on the tiny VL model"),
           "model": info, "engine": {"class": type(engine).__name__, "lead": engine.lead, "state_style": engine.state_style,
                                     "system": engine.system, "option_style": engine.option_style},
           "processor": type(engine.processor).__name__, "requests": rows, "npz": str(npz_path.relative_to(K)),
           "npz_sha256": sha256_file(npz_path), "torch": torch.__version__, "seconds": round(time.time() - t0, 1)}
    if real:
        import transformers

        ru = resource.getrusage(resource.RUSAGE_SELF)
        wall = time.time() - t0
        doc.update(npz_bytes=npz_path.stat().st_size, transformers=transformers.__version__,
                   env={"python": platform.python_version(), "machine": platform.machine(), "threads": torch.get_num_threads(),
                        "thread_env": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")},
                        "loop_seconds": loop_s, "cpu_seconds": round(ru.ru_utime + ru.ru_stime, 1),
                        "mean_cores": round((ru.ru_utime + ru.ru_stime) / wall, 2), "max_rss_bytes": ru.ru_maxrss,
                        "started": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t0)),
                        "ended": time.strftime("%Y-%m-%d %H:%M:%S")},
                   fixtures_sha256=sha256_file(K / "fixtures/requests.json"))
    out.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n")
    for r in rows:
        print(r["name"], r["tokens"], r["image_positions"], r["tiles"], r["cut"], [round(p, 6) for p in r["probs"]])
    return 0


if __name__ == "__main__":
    sys.exit(main())
