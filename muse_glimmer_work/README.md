# Muse-Glimmer-30B → LiteRT-LM (`.litertlm`, int4, one picture at 448×448)

[meta-models/Muse-Glimmer-30B](https://huggingface.co/meta-models/Muse-Glimmer-30B) (revision `cec94fcf`): a 52-layer
dense text decoder (27.85 B parameters) and a 50-layer vision tower. litert-torch 0.10.0's `export_hf` has no entry
for it, so the text decoder goes through `export_hf` behind a small text-only wrapper, and the vision tower is
rewritten for one fixed picture size and converted with `litert_torch.convert`. The result runs on the stock LiteRT-LM
0.19.0 (PyPI wheel), CPU and GPU.

## What had to change

1. **A text-only causal-LM wrapper** (`muse_text_lm.py`) with the checkpoint's own parameter names, registered with
   `AutoModelForCausalLM` (`exist_ok=True`: transformers 5.20 silently skips registering a config class that is defined
   inside transformers). Import `export_hf` before registering: importing it puts a new `transformers` module object
   into `sys.modules`, and a class registered earlier is lost.
2. **The output scale folded into `lm_head`** (`muse_textdir.py --prescale-lm-head`): Muse computes
   `T·tanh(x·m/T)` on the logits. With `lm_head` replaced by `W·(m/T)` after loading, the head is `T·tanh(FC)` and the
   FC weight stays a constant the quantizer reaches; otherwise the decode graph keeps a `MUL(W, scalar)` on the 1.3 B
   parameter table and its FC weight is not a constant.
3. **The lightweight export from an fp32 copy on disk**: `upcast_fp32.py` writes an F32 copy of the checkpoint, so
   transformers' load hands out mmapped tensors instead of an anonymous fp32 copy; `muse_text_export.py --lightweight`
   keeps constants of 1e8 bytes or more lazy through the lowering, and `--readonly-clamp` replaces litert-torch's
   in-place ±inf clamp with a read-only check (the in-place write turns every mmapped page into anonymous memory).
   Peak memory of the export: about 10 GB.
4. **int4 with a floor on the block scales** (`muse_quant.py`): FC weights int4 blockwise-128, `lm_head` int8 per
   channel, block scales floored at 2^-14 while quantizing.
5. **The vision tower for one size** (`muse_vision.py`): the position-embedding interpolation, the 2D RoPE and the
   window order are computed for the 448×448 grid with HF's own helpers and stored as constants; the patch embedding
   becomes a Conv2d and the pixel shuffle four strided slices (no gather). fp32: dynamic int8 moved the image
   embeddings by a relative L2 of 0.20 to 0.24.
6. **A 2,048-token KV cache and a 32-token prefill without a re-export**
   ([`../tools/gpu_graph/cache_variant.py`](../tools/gpu_graph/cache_variant.py),
   [`../tools/gpu_graph/prefill_bucket_clone.py`](../tools/gpu_graph/prefill_bucket_clone.py)): the export takes
   about 3 h, so the cache length is rewritten in the int4 file (every shape and RESHAPE / SLICE constant that carries
   it, and the sliding-window mask's position constant), and a 32-token prefill signature is copied from `prefill_128`.
   The weights are copied byte for byte. On an M4 Max GPU this took decode from 16.7 to 18.3 tokens/s and the first
   text of a picture question from 2.78 to 2.57 s; a follow-up question on the same picture from 0.85 to 0.41 s.
7. **The GPU layout rewrite** ([`../tools/gpu_graph/tfl_rank4.py`](../tools/gpu_graph/tfl_rank4.py)): the GPU
   delegate places rank-2/3 runtime tensors batch-first but right-aligns rank-2/3 constants, so the RoPE multiply
   `[N,h,d] × const[N,1,d]`, the position-table add and MEANs that drop a rank read the wrong rows. The rewrite lifts
   those ops (and the rank-3 attention BATCH_MATMULs) to rank 4; the CPU output stays bit-identical.

## Steps

```
python upcast_fp32.py SNAPSHOT F32 F32.manifest.json                # 119 GB
python muse_textdir.py TEXT --vlm-config SNAPSHOT/config.json --files-from SNAPSHOT --weights-from F32 --prescale-lm-head
PYTHONPATH=shim python muse_text_export.py TEXT EXPORT --prefill 128,1024 --cache 4096 --quant none \
  --lightweight --readonly-clamp                                    # about 3 h from a USB disk
python muse_quant.py EXPORT/model.tflite EXPORT/model_wi4b128_wi8.tflite wi4b128_wi8
python ../tools/gpu_graph/cache_variant.py EXPORT/model_wi4b128_wi8.tflite EXPORT/model_c2048.tflite --cache 2048 \
  --report cache_variant.json                                       # the KV cache 4096 -> 2048, no re-export
python ../tools/gpu_graph/prefill_bucket_clone.py tflite EXPORT/model_c2048.tflite EXPORT/model_p32c2048.tflite \
  --lengths 32 --source prefill_128 --report clone.json             # a 32-token prefill copied from prefill_128
python muse_quant.py EXPORT/embedder.tflite EXPORT/embedder_wi8.tflite emb8
PYTHONPATH=shim python muse_vision.py export --src SNAPSHOT --g 448 --out-dir VISION
python ../tools/gpu_graph/tfl_rank4.py rewrite VISION/vision_encoder_g448_fp32.tflite VISION/vision_encoder_g448_fp32_r4.tflite
python muse_bundle.py --dec EXPORT --decode model_p32c2048.tflite --vis VISION --g 448 \
  --venc vision_encoder_g448_fp32_r4.tflite --vadp vision_adapter_g448_fp32.tflite --tok SNAPSHOT/tokenizer.json \
  --out-dir OUT --name Muse-Glimmer-30B.litertlm --max-tokens 2048
python lm_run.py OUT/Muse-Glimmer-30B.litertlm --backend gpu --image picture_448.png --prompt "How many cats are in this picture?" \
  --system "Reasoning strength: low." --greedy --max-tokens 1024
```

`shim/` stands in for the TensorFlow module `export_hf` 0.10.0 imports without declaring it.

## How it was checked

- The real weights cut to 4 layers (3 sliding-window + 1 global), fp32, the same export path: the same top-1 as HF at
  160 of 160 positions, largest logit difference 5.9e-5.
- The vision tower, fp32, two pictures: relative L2 1.7e-5 and 1.0e-5 against HF's; on the GPU after the rewrite,
  cosine 1.00000 against HF.
- The chat template: the same token ids as HF's processor on 8 conversations.
- The cache rewrite: the real weights cut to 4 layers, exported with a 2,048-token cache and rewritten to 1,024, give
  the original file's logits bit for bit (160 decode positions, and 32 after a 128-token prefill).
- The published file (int4, 2,048-token cache, prefill 32 / 128 / 1,024): on 20 four-option MMStar questions at
  448×448 (greedy, "Reasoning strength: low."), 10 right against 11 for HF's model in Transformers, the same letter on
  19 of 20.
- The two rewrite steps above (`cache_variant.py`, then `prefill_bucket_clone.py tflite`), run again on 2026-10-31 from
  this repository on the int4 export, give the published file's decoder section byte for byte ([REPRODUCE.md](../REPRODUCE.md)).

