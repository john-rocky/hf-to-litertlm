"""pack_dory.py DIR TOKENIZER_JSON OUT.litertlm - pack a quantized dir into a LiteRT-LM bundle the way export_hf packs
one: Authors system metadata + DIR/llm_metadata_t.pb (quantize_dory.py: the runtime chat template) +
DIR/executor_metadata.pb + the repo's tokenizer.json + DIR/model_quantized.tflite (PREFILL_DECODE, no activation type
declared: the GPU runs it with the runtime's default fp16 activations) + DIR/embedder_quantized.tflite (EMBEDDER). The
file is written under a .part name and renamed when complete.

Usage: python dory_work/pack_dory.py <dir> <tokenizer.json> <out.litertlm>
"""
import hashlib
import os
import sys
import time


def sha(path):
  h = hashlib.sha256()
  with open(path, "rb") as f:
    for b in iter(lambda: f.read(64 << 20), b""):
      h.update(b)
  return h.hexdigest()


def main(d, tokenizer, path):
  import litert_lm_builder as lb  # pylint: disable=g-import-not-at-top
  parts = [os.path.join(d, n) for n in ("llm_metadata_t.pb", "executor_metadata.pb", "model_quantized.tflite",
                                        "embedder_quantized.tflite")]
  for p in parts + [tokenizer]:
    if not os.path.exists(p):
      raise SystemExit(f"pack_dory: missing {p}")
  meta_t, executor, model, emb = parts
  b = lb.LitertLmFileBuilder()
  b.add_system_metadata(lb.Metadata(key="Authors", value="ODML", dtype=lb.DType.STRING))
  b.add_llm_metadata(meta_t)
  b.add_executor_metadata(executor)
  b.add_hf_tokenizer(tokenizer)
  b.add_tflite_model(model, lb.TfLiteModelType.PREFILL_DECODE, prefer_activation_type=None)
  b.add_tflite_model(emb, lb.TfLiteModelType.EMBEDDER)
  t0 = time.time()
  with open(path + ".part", "wb") as fh:
    b.build(fh)
  os.rename(path + ".part", path)
  print(f"PACKED {path} {os.path.getsize(path)} B sha256 {sha(path)} ({time.time() - t0:.1f} s)", flush=True)
  return 0


if __name__ == "__main__":
  if len(sys.argv) != 4:
    raise SystemExit(__doc__)
  sys.exit(main(*sys.argv[1:4]))
