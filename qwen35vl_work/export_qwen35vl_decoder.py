"""Export the Qwen3.5 hybrid text decoder for the fast_vlm bundle:

  EMBEDDER       : token_ids [1,1] -> [1,1,2048]   (single_token_embedder)
  PREFILL_DECODE : embeddings + states -> logits + states   (externalize_embedder)

Runs on the patched worktree (qwen35_work/litert-torch-qwen35 = 115a136 +
qwen35_hybrid_litert_torch.patch, which now includes the position-monotonicity
valid-mask fallback: externalized-embedder graphs carry no token ids, so the
gated-delta pad guard derives the mask from `position_ids[1:] > position_ids[:-1]`
— the same contract upstream 0.9.3 uses, proven in production by LFM2.5-VL).

Float export first (house rule: convs + delta-rule stay float), then post-hoc
wi8fc (FULLY_CONNECTED + EMBEDDING_LOOKUP int8) on BOTH tflites — the embedder
is its own tflite so bundle-level recipes never reach it (the same gap the LFM2.5-VL bundles hit: VLM exports split the embedder into its own tflite).

    CACHE=4096 PYTHONPATH=qwen35_work/litert-torch-qwen35 \
      .venv-092/bin/python qwen35vl_work/export_qwen35vl_decoder.py \
      Qwen/Qwen3.5-2B out/qwen35vl-decoder
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "minicpm5_work"))
from quantize_minicpm5 import build_recipe  # noqa: E402

MODEL = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3.5-2B"
OUT = sys.argv[2] if len(sys.argv) > 2 else "out/qwen35vl-decoder"
os.makedirs(OUT, exist_ok=True)
CACHE = int(os.environ.get("CACHE", "4096"))
PREFILL = [int(x) for x in os.environ.get(
    "PREFILL", "1024,512,256,128,64,32,16,8,4,2,1").split(",")]


def find_tflites(d):
  """Return (embedder, prefill_decode) among raw tflites by kv/state inputs."""
  from ai_edge_litert.interpreter import Interpreter
  emb, dec = None, None
  for f in sorted(os.listdir(d)):
    if not f.endswith(".tflite") or f.endswith("_wi8.tflite"):
      continue
    p = os.path.join(d, f)
    it = Interpreter(model_path=p)
    sigs = it.get_signature_list()
    has_state = any("kv_cache" in n for sig in sigs.values() for n in sig["inputs"])
    if has_state:
      dec = p
    else:
      emb = p
  return emb, dec


def quant_wi8fc(src, dst):
  from ai_edge_quantizer import quantizer
  q = quantizer.Quantizer(src, build_recipe("wi8fc"))
  q.quantize().export_model(dst)
  return round(os.path.getsize(dst) / 1e6, 1)


if os.environ.get("SKIP_EXPORT") != "1":
  from litert_torch.generative.export_hf.export import export

  export(
      model=MODEL,
      output_dir=OUT,
      quantization_recipe="",        # float; post-hoc wi8fc below (house rule)
      externalize_embedder=True,
      single_token_embedder=True,
      cache_length=CACHE,
      prefill_lengths=PREFILL,
      bundle_litert_lm=False,        # raw tflites; hand-assembled FastVlm bundle
      keep_temporary_files=True,
      use_jinja_template=False,
      trust_remote_code=True,
  )
  print("DECODER_EXPORT_DONE")

print("tflites:", [f for f in os.listdir(OUT) if f.endswith(".tflite")])
emb, dec = find_tflites(OUT)
assert emb and dec, (emb, dec)
print("embedder:", emb, "\nprefill_decode:", dec)

res = {"embedder": emb, "prefill_decode": dec}
res["emb_wi8_mb"] = quant_wi8fc(emb, os.path.join(OUT, "embedder_wi8.tflite"))
res["dec_wi8_mb"] = quant_wi8fc(dec, os.path.join(OUT, "prefill_decode_wi8.tflite"))
print("QUANT_DONE", json.dumps(res))
with open(os.path.join(OUT, "export_result.json"), "w") as f:
  json.dump(res, f, indent=2)
