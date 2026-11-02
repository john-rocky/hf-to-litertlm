"""The provider's own sequences for a request without loading the weights: the `SystemOne` engine (its tokenizer,
prompt, packing and media processor, from the snapshot's code) over a stand-in model that carries only the config and
one parameter. `engine._sequences` is the provider's code path, so ids, state lengths, question tags, slots and media
inputs are exactly what `system_one` builds; nothing runs the backbone.

    from seqs_only import provider_engine, sequences
    engine = provider_engine()
    seqs = sequences(engine, state, questions, media)   # [{ids, state_len, question, ends, image_grid_thw, ...}]

Run from the work directory with $REF. As a script it prints each record's sequence lengths:
    $REF conversion/seqs_only.py --files set/card_omni.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.normpath(os.path.join(HERE, ".."))
SNAP = os.path.join(WORK, "snapshot")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HOME", os.path.join(WORK, "out/hf_home"))
os.environ.setdefault("HF_MODULES_CACHE", os.path.join(WORK, "out/hf_modules"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def provider_engine():
    """The provider's SystemOne over a stand-in model: the snapshot's config and code, no weights."""
    import torch
    from transformers import AutoConfig, AutoTokenizer
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    config = AutoConfig.from_pretrained(SNAP, trust_remote_code=True)
    model_cls = get_class_from_dynamic_module(config.auto_map["AutoModel"], SNAP)
    pkg = model_cls.__module__.rsplit(".", 1)[0]
    __import__(pkg + ".runner")
    runner = sys.modules[pkg + ".runner"]
    calibration = sys.modules.get(pkg + ".calibration") or __import__(pkg + ".calibration", fromlist=["x"])

    class StandIn(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = config
            self.config._name_or_path = SNAP
            self.anchor = torch.nn.Parameter(torch.zeros(1), requires_grad=False)

    tok = AutoTokenizer.from_pretrained(SNAP)
    cal = calibration.Calibration.load(os.path.join(SNAP, "calibration.json"))
    return runner.SystemOne(model=StandIn(), tokenizer=tok, calibration=cal)


def sequences(engine, state, questions, media=None, set_size=None):
    from PIL import Image  # noqa: F401  (media items are PIL images)

    named = engine.named(questions)
    reqs = [(state, list(named.values()), engine_media(media))]
    maps = {}
    seqs = engine._sequences(reqs, position_maps=maps, set_size=set_size)
    out = []
    for n, (i, k, ids, state_len, question, ends, inputs) in enumerate(seqs):
        grid = inputs.get("image_grid_thw") if inputs else None
        out.append({"first_question": k, "ids": list(ids), "state_len": int(state_len),
                    "question": None if question is None else list(question), "ends": [list(e) for e in ends],
                    "image_grid_thw": None if grid is None else grid.tolist(),
                    "pixel_values_shape": None if not inputs or "pixel_values" not in inputs
                    else list(inputs["pixel_values"].shape),
                    "has_position_map": n in maps})
    return out


def engine_media(items):
    from PIL import Image

    out = []
    for m in items or []:
        out.append(Image.open(m) if isinstance(m, str) else m)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="+", required=True)
    ap.add_argument("--media", default=None)
    a = ap.parse_args()
    sys.path.insert(0, HERE)
    from ref_run import load_records

    recs, media = load_records(a.files)
    if a.media:
        media.update(json.load(open(os.path.join(WORK, a.media))))
    engine = provider_engine()
    for rec in recs:
        req = rec["request"]
        paths = [os.path.join(WORK, media[m]["path"]) for m in req.get("media") or []]
        try:
            seqs = sequences(engine, req.get("state"), req["questions"], paths)
            print(rec["id"], [len(s["ids"]) for s in seqs], [s["image_grid_thw"] for s in seqs if s["image_grid_thw"]])
        except Exception as e:
            print(rec["id"], "ERROR", type(e).__name__, e)


if __name__ == "__main__":
    main()
