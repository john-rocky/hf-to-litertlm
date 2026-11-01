#!/usr/bin/env python3
"""export_lm.py HF_DIR WEIGHTS OUT_DIR - the LFM3-Omni-2B language model in float32 through lfm3_work's LFM3 exporter
(`lfm3_ext.export_lfm3`; lfm3_work must be on PYTHONPATH): the weights under `model.language_model.` of WEIGHTS (the
output of smooth_weights.py), the `text_config` block of HF_DIR/config.json, embeddings in (the embedder is a separate
graph, one token per call), the logits and the final-norm `hidden` out (the speech loop reads `hidden`), prefill
signatures 1024 / 256 / 64 / 16 / 4 / 1 and a 4,096-token KV cache, no quantization and no GPU composites.

HF_DIR holds the source repository's code, config and tokenizer files. Its folder name appears in the graph's tensor
names (transformers names the remote code module after the folder and a hash of the code files), so it is `hf_small`.
OUT_DIR gets model.tflite (the decoder), embedder.tflite and the tokenizer files the exporter writes; no .litertlm is
packed here (build_bundle.py does that with the quantized graphs).
"""
import json
import pathlib
import sys
import time

PREFILL = [1024, 256, 64, 16, 4, 1]
CACHE = 4096


def main():
  if len(sys.argv) != 4:
    print(__doc__, file=sys.stderr)
    return 2
  hf, weights, out = (pathlib.Path(p).resolve() for p in sys.argv[1:])
  assert weights.exists() and (hf / "modeling_lfm3.py").exists(), (weights, hf)
  config = json.load(open(hf / "config.json"))["text_config"]
  assert config["model_type"] == "lfm3" and config["hidden_size"] == 2048, config["model_type"]
  from lfm3_ext import export_lfm3  # pylint: disable=g-import-not-at-top
  assert not out.exists() or not any(out.iterdir()), f"refusing to export into a non-empty {out}"
  out.mkdir(parents=True, exist_ok=True)
  t0 = time.time()
  export_lfm3.export_lfm3(str(out), code_dir=str(hf), weights=str(weights), key_prefix="model.language_model.",
                          config=config, tokenizer_dir=str(hf), emit_hidden=True, quantization_recipe="none",
                          cache_length=CACHE, prefill_lengths=PREFILL, externalize_embedder=True,
                          single_token_embedder=True, use_rope_composite=False, bundle_litert_lm=False,
                          keep_temporary_files=True)
  for name in ("model.tflite", "embedder.tflite"):
    p = out / name
    print(f"exported {p} ({p.stat().st_size} B) in {time.time() - t0:.0f} s", flush=True)
  return 0


if __name__ == "__main__":
  sys.exit(main())
