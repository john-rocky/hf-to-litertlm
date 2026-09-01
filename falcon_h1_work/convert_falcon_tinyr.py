#!/usr/bin/env python3
"""Convert tiiuae/Falcon-H1-Tiny-R-0.6B (44L hybrid, self-emit REASONING) to .litertlm.

  # one-time setup: litert-torch at the pinned base + the hybrid patch
  git clone https://github.com/google-ai-edge/litert-torch litert-torch-falcon
  git -C litert-torch-falcon fetch origin 115a13607c730c81018bb9789138a3e5e5119e3d
  git -C litert-torch-falcon checkout 115a13607c730c81018bb9789138a3e5e5119e3d
  git -C litert-torch-falcon apply "$(pwd)/falcon_h1_litert_torch.patch"

  # download the checkpoint locally first (the template file gets edited)
  hf download tiiuae/Falcon-H1-Tiny-R-0.6B --local-dir src/Falcon-H1-Tiny-R-0.6B
  PYTHONPATH=litert-torch-falcon python convert_falcon_tinyr.py \
      src/Falcon-H1-Tiny-R-0.6B out_tinyr

Same patch and rail as convert_falcon_h1.py (the Instruct family), plus four
Tiny-R-specific measures — each one measured, not assumed:

1. LITERAL-BOS template + NO start_token. Upstream renders ' {{bos_token}} ...'
   but the engine's minijinja leaves `bos_token` unbound (renders empty), so a
   verbatim bundle produces a different token head than HF. At 0.6B that
   one-token difference flips answers (bf16 A/B: 8/8 vs 7/8 — "capital of
   Japan" became Hiroshima). Replacing `{{bos_token}}` with the literal
   `<|begin_of_text|>` and clearing tokenizer.bos_token (so the builder writes
   no start_token) makes the bundle stream byte-identical to HF from BOS on.

2. int8 on FC ONLY — the embedding table stays FLOAT. An int8 embedding table
   (even per-row/channelwise) measurably destabilizes this checkpoint's
   reasoning: runaway thinking past any budget and greedy fact flips that the
   FC-only recipe does not show (float-lm_head does NOT recover it; the input
   embedding is the poison).

3. EXTERNALIZED EMBEDDER. The GPU delegate accepts EMBEDDING_LOOKUP only with
   an int8 table (float crashes engine creation; fp16 weight-only is refused
   as partial delegation), so quality (non-int8 table) and GPU (int8 in-graph
   table) cannot coexist in one graph. --externalize_embedder splits the
   lookup into its own CPU-side section; the decoder graph stays fully
   GPU-delegable and the table stays float.

4. Pad-guard from POSITION MONOTONICITY (in the patch). The externalized
   decoder graph carries no token ids, so the `input_ids != 0` pad mask
   silently disables and CPU pad garbage corrupts the Mamba conv/cumsum state
   (17+25 -> 19.5, float export identical). The patch derives the mask from
   `position_ids[i] > position_ids[i-1]` when token ids are absent.

Reasoning packaging: the model self-emits <think>(131)...</think>(132) with no
template prefill; the thought channel is declared post-hoc (the exporter only
auto-declares channels for jinja templates containing a literal <think>).
Stop tokens [11, 228] come from generation_config via the builder.

Requires litert-lm >= 0.16, litert-torch deps, transformers >= 5.14,
ai-edge-quantizer, litert-lm-builder.
"""
import os
import shutil
import subprocess
import sys
import tempfile
import zlib

src = sys.argv[1] if len(sys.argv) > 1 else "src/Falcon-H1-Tiny-R-0.6B"
outdir = sys.argv[2] if len(sys.argv) > 2 else "out_tinyr"
here = os.path.dirname(os.path.abspath(__file__))

# --- 1. template: literal BOS (HF render byte-identical by construction) ---
tpl_path = os.path.join(src, "chat_template.jinja")
tpl = open(tpl_path).read()
if "{{bos_token}}" in tpl:
  shutil.copy(tpl_path, tpl_path + ".upstream")
  open(tpl_path, "w").write(tpl.replace("{{bos_token}}", "<|begin_of_text|>"))

import transformers  # noqa: E402

_orig = transformers.AutoTokenizer.from_pretrained.__func__


def _no_bos(cls, *a, **k):
  tok = _orig(cls, *a, **k)
  try:
    tok.bos_token = None  # builder writes start_token from bos_token unconditionally
  except Exception as e:  # noqa: BLE001
    print("WARN could not clear bos_token:", e)
  return tok


transformers.AutoTokenizer.from_pretrained = classmethod(_no_bos)

# --- 2. float export with the embedder externalized ---
sys.argv = [
    "litert-torch", "export_hf",
    "--model", src,
    "--output_dir", outdir,
    "--prefill_lengths", "1024,512,256,128,64,32,16,8,4,2,1",
    "--cache_length", "4096",
    "--bundle_litert_lm", "True",
    "--use_jinja_template", "True",
    "--quantization_recipe", "",
    "--externalize_embedder", "True",
]
from litert_torch.cli import main  # noqa: E402

rc = main()
if rc:
  sys.exit(rc)

# --- 3. unpack, quantize the DECODER only (wi8f), repack with float embedder ---
fp = os.path.join(outdir, "model.litertlm")
dump = tempfile.mkdtemp(prefix="tinyr_")
from litert_lm_builder import peek_litertlm_file  # noqa: E402

with open(os.path.join(dump, "peek.txt"), "w") as f:
  peek_litertlm_file(fp, dump, f)
dec = os.path.join(dump, "Section2_TFLiteModel_tf_lite_prefill_decode.tflite")
emb = os.path.join(dump, "Section3_TFLiteModel_tf_lite_embedder.tflite")
meta = os.path.join(dump, "LlmMetadataProto.pbtext")
tokz = os.path.join(dump, "Section1_HF_Tokenizer_Zlib.zlib")
for p in (dec, emb, meta, tokz):
  assert os.path.exists(p), p

from ai_edge_quantizer import quantizer, recipe_manager, qtyping  # noqa: E402

rm = recipe_manager.RecipeManager()
rm.add_dynamic_config(regex=".*",
                      operation_name=qtyping.TFLOperationName.FULLY_CONNECTED,
                      num_bits=8)
qt = quantizer.Quantizer(dec, rm.get_quantization_recipe())
assert not qt.need_calibration
dec_q = os.path.join(dump, "decoder_wi8f.tflite")
qt.quantize().export_model(dec_q)

raw = open(tokz, "rb").read()
tokjson = os.path.join(dump, "tokenizer.json")
open(tokjson, "wb").write(zlib.decompress(raw[8:] if raw[:2] != b"\x78\x9c" else raw))

packed = os.path.join(outdir, "model_wi8f_ext.litertlm")
subprocess.run(["litert-lm-builder",
                "llm_metadata", "--path", meta,
                "hf_tokenizer", "--path", tokjson,
                "tflite_model", "--path", dec_q, "--model_type", "prefill_decode",
                "--prefer_activation_type", "fp32",
                "tflite_model", "--path", emb, "--model_type", "embedder",
                "output", "--path", packed], check=True)

# --- 4. executor metadata (hybrid state binding) + thought channel ---
with_meta = os.path.join(outdir, "model_wi8f_ext_meta.litertlm")
subprocess.run([sys.executable,
                os.path.join(here, "..", "lfm_work", "add_executor_metadata.py"),
                packed, with_meta], check=True)

final = os.path.join(outdir, "Falcon-H1-Tiny-R-0.6B_int8.litertlm")
subprocess.run([sys.executable,
                os.path.join(here, "..", "tools", "add_thought_channel.py"),
                with_meta, final, "--start", "<think>", "--end", "</think>"],
               check=True)
print("DONE:", final)
