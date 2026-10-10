# Reproduce — Phocinae-Largha-150M-v1 → LiteRT

Everything in this repository was made by the scripts in `conversion/` from the publisher's checkpoint, on an Apple
M4 Max (macOS 27.0) with these Python packages: torch 2.14.0, transformers 5.17.0, litert-torch 0.10.0, litert-converter
0.5.0.dev20260912, ai-edge-litert 2.2.0, ai-edge-quantizer 0.10.0.dev20260914, tokenizers 0.23.2, safetensors 0.8.0,
numpy 2.5.3 (Python 3.12). The scripts take their work directory `W` as the parent of their own folder (`W/scripts/`).

## Sources (pinned)

| what | where | pin |
|---|---|---|
| checkpoint | `Phocinae/Phocinae-Largha-150M-v1` (`hf/`: `model.safetensors`, `encoder/config.json`, `tokenizer/`, `rl_agent_config.json`) | commit `3807157407d14497a7ebd7114a3d71847b5b10a5`; `model.safetensors` sha256 `b6472511eea30729985374f43968cf7f4b16bbe0827de6c6ca07cf92afbb778a` |
| the publisher's runtime (the reference) | `github.com/Phocinae/phocinae-server` (`src/phocinae-server/`, imported from source: `phocinae.engine.Engine`) | commit `705508c33f185aee6a838379502e7a87caa0e138` (0.1.5) |
| test rows | `github.com/Phocinae/Phocinae-Largha-150M-v1` `datasets/typed_test/test_typed_400.jsonl` (the normalized `LocalLLaMA/typed-decisions` test split) and `datasets/flip_recipe/` | commit `f78d42e607912da4f185f774913bcb7190af9c0b` |

## Steps

```bash
cd W   # with hf/, src/phocinae-server/, src/gh_datasets/ in place; P = python3 of the environment above
# 1. fixtures: 2,000 test decisions + the author's flip400 rows (2,400) + 5 control rows; the server-protocol requests
$P scripts/make_fixtures.py
# 2. reference: the publisher's Engine on the CPU in fp32, one row at a time; batch-shape probe; server runs
$P scripts/oracle.py --threads 4
# 3. the export graph in PyTorch vs the reference; fp16 headroom scan; the exact LayerNorm rewrite (x / 64)
$P scripts/torch_parity.py --L 512 --buckets 256,384,512
$P scripts/torch_parity.py --L 512 --scan
SHIFT=$(python3 -c "print(','.join([f'attn{i}=6' for i in range(12,22)]+[f'mlp{i}=6' for i in range(12,22)]+['final=6']))")
for L in 256 384 512; do $P scripts/torch_parity.py --L $L --ln-shift "$SHIFT"; done
# 4. exports (fp32, one file per window; the act head once) and the fp16 token table of the host-lookup form
$P scripts/export.py --act
$P scripts/tables.py
for L in 256 384 512; do for f in ids embeds; do $P scripts/export.py --L $L --form $f --ln-shift "$SHIFT"; done; done
# 5. storage: fp16 FULLY_CONNECTED weights (+ int8 EMBEDDING_LOOKUP for the ids form)
for L in 256 384 512; do
  scripts/quant_all.sh exports/phocinae_L${L}_ids_fp32_shift.tflite v2_fp16fc_i8emb
  scripts/quant_all.sh exports/phocinae_L${L}_embeds_fp32_shift.tflite wfp16
done
# 6. checks on the Mac: CPU (XNNPACK), Metal fp32 and default precision; --timing for the numbers of the card
$P scripts/litert_parity.py --tflite exports/phocinae_L256_ids_v2_fp16fc_i8emb_shift.tflite --L 256 --accel cpu --threads 8
$P scripts/litert_parity.py --tflite exports/phocinae_L256_embeds_wfp16_shift.tflite --L 256 --accel cpu --threads 8 --table exports/tables/tok_embeddings_fp16.bin
scripts/mac_gpu_runs.sh 256 exports/phocinae_L256_ids_v2_fp16fc_i8emb_shift.tflite exports/phocinae_L256_embeds_wfp16_shift.tflite
$P scripts/server_replay.py --tflite exports/phocinae_L512_ids_v2_fp16fc_i8emb_shift.tflite --L 512
# 7. the phone (Galaxy S26; the debug gate app is a CompiledModel Kotlin activity with the same I/O as the SDK's laya family)
for L in 256 384 512; do $P scripts/device_rows.py --L $L; done
scripts/s26_batch.sh 256 phocinae_L256_ids_v2_fp16fc_i8emb_shift.tflite phocinae_L256_embeds_wfp16_shift.tflite
scripts/s26_timing.sh
$P scripts/device_compare.py device/<report>.json
```

The published file names map to the exports as `phocinae_largha_150m_L{L}_fp16fc_int8emb.tflite` =
`exports/phocinae_L{L}_ids_v2_fp16fc_i8emb_shift.tflite` and `phocinae_largha_150m_L{L}_fp16fc_embeds.tflite` =
`exports/phocinae_L{L}_embeds_wfp16_shift.tflite`; `SHA256SUMS` lists every published file.
