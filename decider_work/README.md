# decider-0.8b (Mapika) → LiteRT-LM — a System One decision model, where the readout is the recipe

Published: [litert-community/decider-0.8b-LiteRT](https://huggingface.co/litert-community/decider-0.8b-LiteRT) — `decider-0.8b_fp16.litertlm` (exact, desktop CPU) and `decider-0.8b_int8.litertlm` (dynamic int8, Android CPU/GPU). Source: [Mapika/decider-0.8b](https://huggingface.co/Mapika/decider-0.8b) (revision `1ea54127…`, Apache-2.0), a Qwen3.5-0.8B-Base finetune that answers typed questions (choice / score / noul) with calibrated option probabilities from one forward pass, in the `/v1/systemone` request shape of the decider package. The model never generates text: the answer is the softmax (T = 1.03) over the option-letter logits at the last token of a raw-text row.

## Recipe (the Qwen3.5-0.8B hybrid rail with the HF id swapped)

```bash
# 1. converter: the pinned base is a FORK-ONLY commit — clone the fork, not upstream (upstream/main does not contain 115a136)
git clone https://github.com/john-rocky/litert-torch litert-torch-qwen35
git -C litert-torch-qwen35 checkout 115a13607c730c81018bb9789138a3e5e5119e3d
git -C litert-torch-qwen35 apply "$(pwd)/qwen35_work/qwen35_hybrid_litert_torch.patch"   # registers qwen3_5 AND qwen3_5_text
# 2. float export (same flags as the base): ladder 1024..1, cache 4096, no quantization recipe
PYTHONPATH=litert-torch-qwen35 python qwen35_work/convert_qwen35_hybrid.py Mapika/decider-0.8b out/decider-fp   # stop after the export step
# 3. weights (from the SAME float bundle) + identity template + state metadata + fp32 activations
python decider_work/scripts/quantize_decider_ab.py apply out/decider-fp/model.litertlm out/fp16_raw.litertlm  --recipe wfp16   # fp16 casting, float compute
python decider_work/scripts/quantize_decider_ab.py apply out/decider-fp/model.litertlm out/int8_raw.litertlm  --recipe wi8fc   # dynamic int8 FC + int8 embedding
for f in fp16 int8; do
  python decider_work/scripts/repack_identity.py out/${f}_raw.litertlm out/${f}_tmpl.litertlm --unpack out/${f}_unpack   # identity jinja, no start token, stop 248044
  python scripts/add_executor_metadata.py out/${f}_tmpl.litertlm out/${f}_meta.litertlm
  python scripts/set_activation_type.py  out/${f}_meta.litertlm out/decider-0.8b_${f}.litertlm --type fp32
done
```

Environment: python 3.12, `litert-torch==0.9.2 litert-converter==0.3.0 ai-edge-quantizer==0.8.0 ai-edge-litert==2.1.6 litert-lm==0.15.0 litert-lm-builder==0.15.0 transformers==5.14.1 torch==2.12.1` for the export/repack (`requirements-lock-export.txt`), `ai-edge-litert==2.2.0 litert-lm==0.17.1` for the gates (`requirements-lock-gate.txt`), and `transformers==5.17.0 torch==2.14.0` + the decider package at commit `c4daaac2…` for the oracle (`requirements-lock-oracle.txt`, `--no-deps`: its `flash-linear-attention` dependency has no macOS wheel; transformers falls back to its torch chunked delta rule).

## Readout = the shipped contract

No chat template: the row is `Context:\n<state>\n\nQuestion: …\nOptions:\n(A) …\nAnswer: (` built piecewise by the vendored upstream `prompt.build` (ids, not text: whole-string re-tokenization differs on the 255-option rendering), no BOS. Prefill `ids[:-1]` through the exported prefill signatures, ONE decode of the last token (the prefill signatures of this export output state only), gather `label_table(tok)[1][:nopts]` from the decode logits, softmax at 1.03, upstream `assemble`. Reference: `scripts/systemone_litert.py` (+ `litert_readout.py`, `bundle_cache.py`, `common.py`, `decider_vendored/`). Gate: `fixtures/fixtures.json` (40 synthetic fixtures → 120 rows incl. one 255-option row) against `fixtures/oracle_fp32.json` (the fp32 `Decider.system_one` output, frozen).

**On a GPU feed each row as ONE padded prefill chunk** (smallest signature ≥ len(ids)−1, pad ids = 0, padded query rows fully masked), then decode the last token. Chaining prefill chunks on the GPU corrupts the carried hybrid state: Mac Metal, fp32 activations, multi-chunk max |Δp| 0.177 vs one padded chunk 9.8e-6 vs decode-walk 5.9e-6, while the CPU agrees across all three within 8.9e-6 (`results/gpu_feeding_schemes.json`). Rows longer than 1024 tokens run on CPU.

## What the weights do to the probabilities (same float export, 120 rows, unrounded |Δp| vs the fp32 oracle — `results/quant_ab.json`)

| form | file | argmax | max |Δp| | p95 | rows > 0.02 |
|---|---|---|---|---|---|
| dynamic int8 FC + int8 embedding (house recipe), CPU | 963 MB | 117/120 | 0.160 | 0.068 | 27 |
| the same file on Mac Metal GPU, one padded chunk | | 118/119 | 0.073 | 0.025 | 10 |
| weight-only int8 (float compute), CPU | 965 MB | 119/120 | 0.073 | 0.025 | 10 |
| fp16 float-casting FC + embedding, CPU | 1.63 GB | 120/120 | 6.6e-6 | 3.1e-6 | 0 |
| fp16 FC + dynamic int8 lm_head + int8 embedding, Metal GPU | 1.41 GB | 119/119 | 0.0078 | 0.005 | 0 |

The loss of the house int8 recipe on CPU is the dynamic activation quantization of the FC layers; a GPU dequantizes the int8 weights and computes in float, so the same file lands at the weight-only figure there. The vocab table's int8 is harmless (embedding-only: p95 0.0034). fp16 casting is exact but XNNPACK expands it to fp32 on CPU (weight cache 3.0 GB; a Galaxy S26 needs 6.1 GB at 262 tokens and restarted its framework at 503) — the fp16 file is a desktop file. Two forms that look right but are not: fp16 lm_head + int8 embedding on the tied table makes the quantizer store the int8 table once per subgraph (12×, 4.7 GB); weight-only int8 lm_head sharing the table fails to compile on Metal (shape mismatch). The 1.41 GB form (fp16 FCs, dynamic-int8 lm_head, int8 embedding) compiles on Metal and delegates fully on the S26 CL delegate (6.2 GB peak) — built and gated, not published.

The weight-only int8 file is not published: the Galaxy S26 kernel-panicked about 70 s after engine start on two separate days (`dumpsys dropbox` SYSTEM_LAST_KMSG entries; from the host it looks like a USB drop). The dynamic-int8 file ran four cold S26 legs without incident.

## Device rows (Galaxy S26 SM-S942Q, `litert_lm_advanced_main` v0.16.0, one cold run each — `results/s26_r6_rows.jsonl`)

| file | backend | tokens | prefill tok/s | TTFT | engine init | peak VmHWM |
|---|---|---|---|---|---|---|
| int8 | GPU (188522/188522 ops delegated) | 262 | 300.5 | 0.91 s | 78 s | 4553 MiB |
| int8 | GPU | 503 | 567.2 | 0.97 s | 97 s | 5317 MiB |
| int8 | CPU | 262 | 287.8 | 0.94 s | 14 s | 1941 MiB |
| fp16 | CPU | 262 | 44.9 | 5.92 s | 18 s | 6094 MiB |

For a prefill-dominated decision model the phone GPU is no faster than the CPU at ~260 tokens and costs 78–97 s of engine init and 2–3× the memory; its value is calibration (int8 weights in float). Mac M4 Max CPU (4 threads, quiet, `-p 256 -d 8 --runs 3 --cache no`, litert-lm 0.17.1): fp16 533 tok/s / TTFT 0.53 s / init 42 s; int8 695 tok/s / 0.39 s / 40 s.

## LiteRT-LM engine scoring is not the readout

`RunTextScoring` exists in the C and Python API only (one target per call; not in the CLI, Kotlin or Swift). Scoring advances the session, and a second session on the same engine starts from stale linear-attention state, so on this architecture it works only as one fresh engine per scoring call (fp32 control: hermetic 6/6 argmax with |Δp| ≤ 4.5e-3, shared engine 5/6 with max 0.51 — `results/engine_hermetic_vs_shared.json`). The shipped readout is the graph path above.
