#!/usr/bin/env python3
"""The language model of granite-speech-5.0-2b as a standalone GraniteForCausalLM checkpoint, the input of the LM
export (scripts/export_simple_template.py).

  OUT/config.json        the source's text_config (model_type granite, architectures GraniteForCausalLM,
                         tie_word_embeddings false as in the source's top level)
  OUT/model.safetensors  the 363 language_model.* tensors renamed language_model.model.* -> model.*,
                         language_model.lm_head.* -> lm_head.*, BF16 bytes unchanged
  OUT/<tokenizer files, generation_config.json, chat_template.jinja>   copied as they are
  OUT/../build_lm.json   the checks below (next to the checkpoint dir, so the export reads only the model files)

Checks: strict load (loading info empty) and the 363 tensors equal to the source's; the composite model's LM against
the standalone one on a text prompt in fp32 (the composite applies lm_head without logits_scaling and
GraniteForCausalLM divides by logits_scaling 8, so standalone x 8 is compared; transformers 5.22.1); the ids of the
vocabulary's tail that LiteRT-LM's GPU sampler cannot draw when the vocabulary is not a multiple of 4.

  python build_lm.py --src <snapshot dir> --out <dir>
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

COPY = ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "added_tokens.json", "vocab.json",
        "merges.txt", "generation_config.json", "chat_template.jinja"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
    torch.set_num_threads(4)
    src, out = Path(args.src), Path(args.out)
    rep = {}
    cfg = json.load(open(src / "config.json"))
    tc = dict(cfg["text_config"])
    tc.pop("_name_or_path", None)
    tc["architectures"] = ["GraniteForCausalLM"]
    tc["model_type"] = "granite"
    tc["tie_word_embeddings"] = bool(cfg.get("tie_word_embeddings", False))
    tc["transformers_version"] = cfg.get("transformers_version")
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(tc, indent=2))
    sd = {}
    with safe_open(str(src / "model.safetensors"), framework="pt") as f:
        for k in f.keys():
            if k.startswith("language_model.model."):
                sd["model." + k[len("language_model.model."):]] = f.get_tensor(k)
            elif k.startswith("language_model.lm_head."):
                sd["lm_head." + k[len("language_model.lm_head."):]] = f.get_tensor(k)
    rep["lm_tensors"] = len(sd)
    assert len(sd) == 363, len(sd)
    save_file(sd, str(out / "model.safetensors"), metadata={"format": "pt"})
    for n in COPY:
        shutil.copy2(src / n, out / n)
    rep["files"] = {p.name: {"bytes": p.stat().st_size, "sha256": C.sha256(p)} for p in sorted(out.iterdir())
                    if p.is_file()}
    assert rep["files"]["tokenizer.json"]["sha256"] == C.sha256(src / "tokenizer.json")

    from transformers import AutoTokenizer, GraniteForCausalLM, GraniteSpeechForConditionalGeneration
    lm, info = GraniteForCausalLM.from_pretrained(str(out), dtype=torch.float32, output_loading_info=True)
    rep["loading_info"] = {k: [str(x) for x in v] for k, v in info.items()}
    assert all(len(v) == 0 for v in info.values()), info
    lsd = lm.state_dict()
    bad = [k for k, v in sd.items() if not torch.equal(v.to(torch.float32), lsd[k])]
    rep["tensor_value_mismatch"] = bad
    assert not bad and len(lsd) == 363, (bad[:3], len(lsd))
    tcd = lm.config.to_dict()
    rep["standalone_config"] = {k: tcd.get(k) for k in ("embedding_multiplier", "attention_multiplier",
                                                        "residual_multiplier", "logits_scaling", "rope_parameters",
                                                        "rope_scaling", "max_position_embeddings", "tie_word_embeddings",
                                                        "vocab_size")}
    comp = GraniteSpeechForConditionalGeneration.from_pretrained(str(src), dtype=torch.float32).eval()
    tok = AutoTokenizer.from_pretrained(str(out))
    text = "USER: Write one short sentence about the sea.\n ASSISTANT: The sea is wide and deep, and its waves"
    ids = tok(text, return_tensors="pt", add_special_tokens=False)["input_ids"]
    with torch.no_grad():
        a = comp(input_ids=ids).logits[0]
        b = lm(input_ids=ids).logits[0] * float(lm.config.logits_scaling)
    d = (a - b).abs()
    rep["composite_vs_standalone"] = {"tokens": int(ids.shape[1]), "max_abs_logit_diff": float(d.max()),
                                      "top1_equal": int((a.argmax(-1) == b.argmax(-1)).sum()),
                                      "logit_abs_max": float(a.abs().max()),
                                      "note": "standalone logits x logits_scaling (8) vs composite lm_head logits"}
    assert rep["composite_vs_standalone"]["top1_equal"] == ids.shape[1]
    vocab = lm.config.vocab_size
    tail = list(range(vocab - vocab % 4, vocab))
    rep["vocab_mod4"] = {"vocab": vocab, "mod4": vocab % 4,
                         "gpu_sampler_unreachable_ids": {i: tok.convert_ids_to_tokens(i) for i in tail}}
    (out.parent / "build_lm.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1))
    print(json.dumps({k: rep[k] for k in ("lm_tensors", "composite_vs_standalone", "vocab_mod4", "standalone_config")},
                     ensure_ascii=False, indent=1))
    print("BUILD_LM_DONE", flush=True)


if __name__ == "__main__":
    main()
