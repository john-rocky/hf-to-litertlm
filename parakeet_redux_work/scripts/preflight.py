#!/usr/bin/env python3
"""Preflight without running a model (venv .venv-ref): B's expanded state dict against a ParakeetForTDT built on the
meta device (names, shapes, missing / unexpected), O's safetensors keys and shapes against the same, the processor
load, feature shapes and masks for modes (i)-(iv) on one short and the one 34.95 s S32 utterance, decode_text."""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import redux_ref as R  # noqa: E402
import unpack3  # noqa: E402


def main():
    from safetensors import safe_open
    from transformers import ParakeetForTDT, ParakeetTDTConfig
    rep = {}
    cfg = ParakeetTDTConfig.from_pretrained(R.REDUX)
    with torch.device("meta"):
        mb = ParakeetForTDT(cfg)
    want = {k: tuple(v.shape) for k, v in mb.state_dict().items()}
    sd = unpack3.load_redux_state_dict(os.path.join(R.REDUX, "model.safetensors"), os.path.join(R.REDUX, "ternary.json"))
    got = {k: tuple(v.shape) for k, v in sd.items()}
    rep["B"] = {"model_keys": len(want), "file_keys": len(got),
                "missing": sorted(set(want) - set(got)), "unexpected": sorted(set(got) - set(want)),
                "shape_mismatch": [(k, got[k], want[k]) for k in want if k in got and got[k] != want[k]],
                "dtypes": sorted({str(v.dtype) for v in sd.values()})}
    cfg_o = ParakeetTDTConfig.from_pretrained(R.ORIG)
    with torch.device("meta"):
        mo = ParakeetForTDT(cfg_o)
    want_o = {k: tuple(v.shape) for k, v in mo.state_dict().items()}
    with safe_open(os.path.join(R.ORIG, "model.safetensors"), "np") as f:
        got_o = {k: tuple(f.get_slice(k).get_shape()) for k in f.keys()}
    rep["O"] = {"model_keys": len(want_o), "file_keys": len(got_o),
                "missing": sorted(set(want_o) - set(got_o)), "unexpected": sorted(set(got_o) - set(want_o)),
                "shape_mismatch": [(k, got_o[k], want_o[k]) for k in want_o if k in got_o and got_o[k] != want_o[k]]}
    rep["configs_equal_except_ternary"] = {k: (cfg.to_dict().get(k), cfg_o.to_dict().get(k))
                                          for k in set(cfg.to_dict()) | set(cfg_o.to_dict())
                                          if cfg.to_dict().get(k) != cfg_o.to_dict().get(k)}
    proc = R.processor()
    rep["processor"] = {"class": type(proc).__name__, "tokenizer": type(proc.tokenizer).__name__,
                        "fe": type(proc.feature_extractor).__name__, "decoder_type": proc._decoder_type,
                        "blank_id": proc.blank_token_id}
    rows = R.read_jsonl("data/sets/librispeech-test-clean/manifest-S32.jsonl")
    feats = {}
    for r in (rows[0], rows[-1]):
        a = R.load_wav(r["path"])
        d = {"samples": len(a)}
        f, m = R.features(proc, a)
        d["i"] = [list(f.shape), int(m.sum()), str(m.dtype)]
        if len(a) <= 479840:
            f, m = R.features(proc, a, pad_to=479840, mask_padding=True)
            d["ii"] = [list(f.shape), int(m.sum()), float(f[0, int(m.sum()):].abs().max()) if m.sum() < f.shape[1] else 0.0]
            f, m = R.features(proc, a, pad_to=479840, mask_padding=False)
            d["iii"] = [list(f.shape), int(m.sum())]
        w = a[:79840]
        f, m = R.features(proc, w, pad_to=79840, mask_padding=False)
        d["iv_first_window"] = [list(f.shape), int(m.sum())]
        feats[r["id"]] = d
    rep["features"] = feats
    rep["decode_text_sample"] = R.decode_text(proc, proc.tokenizer.encode("hello world", add_special_tokens=False)
                                              + [R.BLANK])
    print(json.dumps(rep, indent=1, default=str))
    ok = (not rep["B"]["missing"] and set(rep["B"]["unexpected"]) == R.VAD_KEYS and not rep["B"]["shape_mismatch"]
          and not rep["O"]["missing"] and not rep["O"]["unexpected"] and not rep["O"]["shape_mismatch"])
    json.dump(rep, open("results/preflight.json", "w"), indent=1, default=str)
    print("PREFLIGHT", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
