"""Export the standalone Qwen2.5-1.5B decoder (extracted from InternVL3-2B) to the
fast_vlm bundle's EMBEDDER + PREFILL_DECODE tflites.

Matches the FastVLM-0.5B contract:
  EMBEDDER       : token_ids [1,1] -> [1,1,1536]   (single_token_embedder=True)
  PREFILL_DECODE : decode_embeddings [1,1,1536] + KV -> logits + KV   (externalize_embedder)

bundle_litert_lm=False + keep_temporary_files=True -> leaves raw tflites in out_dir
for hand-assembly into a FastVlm .litertlm. BOCTAV4 = blockwise-32 + OCTAV int4,
int8 embedding (the parity recipe used by the 5 prior dense ships).

    CACHE=2048 ~/clipconv/bin/python scripts/export_internvl_decoder.py [llm_dir] [out_dir]
"""
import sys, os, copy

# Import litert_torch FIRST (recipe lib + converter) before any model submodule.
import ai_edge_quantizer.recipe as _aqr  # noqa: E402

LLM = sys.argv[1] if len(sys.argv) > 1 else "src_models/internvl3-2b-llm"
OUT = sys.argv[2] if len(sys.argv) > 2 else "out/internvl-decoder"
os.makedirs(OUT, exist_ok=True)

# BOCTAV4 recipe: blockwise-32 + OCTAV int4 weights, int8 EMBEDDING_LOOKUP.
_I4 = _aqr.dynamic_wi4_afp32()[0]
_I8 = copy.deepcopy(_I4)
_I8["op_config"]["weight_tensor_config"]["num_bits"] = 8
_O4 = copy.deepcopy(_I4)
_O4["algorithm_key"] = _aqr.AlgorithmName.OCTAV
_BO4 = copy.deepcopy(_O4)
_BO4["op_config"]["weight_tensor_config"]["granularity"] = "BLOCKWISE_32"


def _mk_alg(int4_rule, ops_int8):
  rules = [int4_rule]
  for op in ops_int8:
    rr = copy.deepcopy(_I8)
    rr["operation"] = op
    rules.append(rr)
  return rules


_aqr.BOCTAV4 = lambda: _mk_alg(_BO4, ["EMBEDDING_LOOKUP"])

# WI8_FLOAT: int8 weights but FLOAT compute (explicit dequantize) — weight-only
# compression for quantization-sensitive tiny decoders (PaddleOCR-VL 0.36B: the
# default INTEGER compute_precision costs top1 2/10 on teacher-forced logits).
_W8F = copy.deepcopy(_I8)
_W8F["op_config"]["compute_precision"] = "FLOAT"
_W8F["op_config"]["explicit_dequantize"] = True
_aqr.WI8_FLOAT = lambda: [_W8F]

from litert_torch.generative.export_hf.export import export  # noqa: E402

CACHE = int(os.environ.get("CACHE", "2048"))
PREFILL = [int(x) for x in os.environ.get("PREFILL", "128,512").split(",")]
RECIPE = os.environ.get("RECIPE", "BOCTAV4")

export(
    model=LLM,
    output_dir=OUT,
    quantization_recipe=RECIPE,
    externalize_embedder=True,
    single_token_embedder=True,
    cache_length=CACHE,
    prefill_lengths=PREFILL,
    bundle_litert_lm=False,        # keep raw tflites; we hand-assemble a FastVlm bundle
    keep_temporary_files=True,
    use_jinja_template=False,
    trust_remote_code=True,
)
print("DECODER_EXPORT_DONE")
print("tflites:", [f for f in os.listdir(OUT) if f.endswith(".tflite")])
