# decider-2b-vision → LiteRT-LM: reproduce rounds 1–4 and 6

This directory converts [Mapika/decider-2b-vision](https://huggingface.co/Mapika/decider-2b-vision) at revision `863e290863655f1d6b69324d77d09ac972d21609` (apache-2.0) into three `.litertlm` bundles for the LiteRT-LM `fast_vlm` path: one 256×256 image first, 64 image tokens, the author's 3-channel M-RoPE positions derived inside the decoder graph from the runtime's 1-D position. Rounds 1–4 below are the source-to-bundle chain and its numerical gates, in the order they were run. Every command runs from a clean copy of `decider2bv_work/` that sits inside the repository, next to the repository's `scripts/`, `decider_work/` and `qwen35_work/` directories (section 0 lists the files read from there).

> Public copy. `scripts/make_fixtures*.py` here omit the six internal photo/document rows and the five platformer-style frames, so the fixture and oracle files they write differ from the hashes recorded below; every published row is unchanged (the 42 rows of the model repo's `reference/fixtures/fixtures.json` regenerate byte for byte). The internal round-report renderers are not included. `deps/` holds the exact hybrid patch (sha256 `0a01e2ae…`) and identity template used here. Round-6 steps that read the internal rows or belong to the release tooling (`r6_make_public_fixtures.py`, `r6_card_numbers.py`, `r6_public_from_r4.py`, `r6_make_manifest.py`, `r6_manifest_rows.py`, `r6_sha256sums.py`, `r6_verify_card.py`) are listed for the record; those scripts are not in this copy.

The work ran on an Apple M4 Max (macOS 27.0, arm64). The gates are CPU numbers from ai-edge-litert CompiledModel or from the fp32 PyTorch oracle; every wall-clock figure recorded in `results/` came from a shared machine and is informational only. Large artifacts (weights, venvs, caches, bundles) live under `out/` and are not tracked.

## Final files

| bundle | bytes | sha256 |
| --- | --- | --- |
| `out/bundle_r4/fp16/decider-2b-vision_fp16.litertlm` | 5,509,356,416 | `75b226796c43405b900399551487903dbe4f87d1a2c40aea422b2a9c33c8f62a` |
| `out/bundle_r4/v7c/decider-2b-vision_v7c.litertlm` | 4,498,201,472 | `72f6836051f54ad32f7804e06bc0dae2e022a22f97d864d480f39710f17357bc` |
| `out/bundle_r4/dyn8/decider-2b-vision_dyn8.litertlm` | 3,171,081,088 | `5fb2e19aa2066d2e3955431366bc572edb7abbc6037db53d4eaf4a4a4d0e7bd9` |

The bundle header carries a random uuid and a creation timestamp (`results/bundle_r4_<variant>.json` `system_metadata`), so a rebuilt bundle has a different file sha256 even from identical sections. Compare the sections instead:

| section | fp16 | v7c | dyn8 |
| --- | --- | --- | --- |
| EMBEDDER | `d1485b2ea5d700bf98c4e7fd9b45c928a313f0123820fbb8dc1159933effbfbf` (1,017,121,104 B) | `080dce45fcb89440fe7ea785064bc70a5b8eb24fac0e331a01354191234b2ec1` (511,541,472 B) | same as v7c |
| PREFILL_DECODE | `f8181a9417e686c5738434e2d0700d5f0e8784d4af084d9a7ec1e6c26a1e46e8` (3,829,517,696 B) | `0b8a7b3325cac5dc94dde89c07e6e9aa43dbed00e443da64ee6e8ee2dfc4b93c` (3,323,937,888 B) | `99fcdafe0fb14875871e9796825a5a0311daac03c911f16ee5cfc8d2632481d2` (1,996,826,448 B) |
| VISION_ENCODER | `2577ce50ab75447f5cda98ce45774200f0a0f820d87b8e3c8f3bda9e9e0fbd16` (608,346,896 B) | same | same |
| VISION_ADAPTER | `050121c1fff11716b5e22c4a07e7a8950138fe6897d5f40876e202d4c95b1aad` (50,371,456 B) | same | same |
| HF tokenizer (zlib) | decompresses to the snapshot's `tokenizer.json`, sha256 `b6d27c11283798debbfbcb1bd3bbdbcebac6c9b20d0be0c5b79231c97de97501` | same | same |

Byte determinism of the exports was not tested as such. What was observed: the fp32 embedder came out byte-identical from the round-2 and round-4 exports, and the round-4 fp16 decoder's FC weight bytes equal round 3's (`rounds/r4_notes.md`).

## 0. Files read from outside this directory

Check each hash before running; a different revision of any of these does not reproduce the files above.

| path (relative to the repository root) | sha256 | used by |
| --- | --- | --- |
| `qwen35_work/qwen35_hybrid_litert_torch.patch` | `0a01e2ae9f6bbb0aa79b1ba7de343bad3c4bf17b919a2ccd21c912ede2885aa5` | the converter clone (decoder export) |
| `scripts/add_executor_metadata.py` | `d1f42ab7704c2dc41d16273a6c56a102d0d41a01094b251e1b638557ef036be3` | bundle: ExecutorMetadata for the 48 state buffers |
| `decider_work/scripts/identity_template.py` | `0ab4f0fe0292bde38feb971111b64ca9b7a4a03e7ddc50b3bf8d2f768db7f0cb` | `scripts/build_bundle_g256.py` (identity prompt template) |
| `decider_work/scripts/quantize_decider_ab.py` | `afe0c5f0cf59381e3b7f7eac48a40593e14e1db49c9b2383ee034363a2e9fecc` | recipes `wfp16` (fp16) and `wi8fc` (dyn8) |
| `decider_work/gpu_run/scripts/quantize_decider_ab.py` | `34fee1068dd5e6cd057daa1edb4cb96126c93df0f1387c192f5d620e531de145` | recipe `fp16fc_i8emb_dynhead` (v7c) |

The recipe hashes are also recorded per output in `out/weights_r4/<variant>/<part>.quant.json` (`recipe_provenance.source_sha256`), and the template hash in `out/bundle_r4/<variant>/*_noexec.build.json`.

## 1. Environments

Three venvs from the locks in this directory, all CPython 3.12.13, built with uv (0.11.19 is recorded for the oracle venv). Each recorded `pip freeze` equals its lock (`results/environment_oracle.json`, `logs/venv_export_freeze.txt`, `logs/venv_readout_freeze.txt`).

| venv | lock | key pins | used for |
| --- | --- | --- | --- |
| `out/venv-oracle` | `requirements-lock-oracle.txt` (sha256 `b58a90af…`, compiled from `requirements-oracle.in`) | torch 2.14.0, torchvision 0.29.0, transformers 5.17.0, tokenizers 0.23.2, pillow 12.3.0 | fixtures, fp32 oracle, HF vision dump, derived-rotary unit test |
| `out/venv-export` | `requirements-lock-export.txt` (sha256 `41d4c750…`) | torch 2.12.1, transformers 5.14.1, litert-torch 0.9.2, litert-converter 0.3.0, ai-edge-litert 2.1.6, ai-edge-quantizer 0.8.0, litert-lm 0.15.0 | decoder export (with the patched clone ahead of the wheel), quantization |
| `out/venv-readout` | `requirements-lock-readout.txt` (sha256 `06ec80cd…`) | ai-edge-litert 2.2.0, litert-lm / litert-lm-api / litert-lm-builder 0.17.1, minijinja 2.24.0, tokenizers 0.23.2, numpy 2.5.3, pillow 12.3.0 | vision parity, flatbuffer scans, graph readouts, bundle, runtime |

```sh
cd decider2bv_work
export UV_CACHE_DIR=out/cache/uv
for v in oracle export readout; do
  uv venv out/venv-$v --python 3.12
  uv pip sync requirements-lock-$v.txt --python out/venv-$v/bin/python
done
```

The vision export (round 2) ran in a fourth, pre-existing environment that has no lock file in this directory. Its pins as recorded in `out/vision_g256/result.json`: torch 2.13.0, transformers 5.14.1, litert-torch 0.9.3, litert-converter 0.4.0, ai-edge-litert 2.2.0, numpy 2.5.2. Read from that environment on 2026-09-29 (its site-packages were last modified on 2026-08-31, before this work): Python 3.14.6, torchvision 0.28.0, ai-edge-quantizer 0.9.0, pillow 12.3.0, tokenizers 0.22.2, safetensors 0.8.0, torchao 0.18.0, jax / jaxlib 0.11.1, flatbuffers 25.12.19, protobuf 7.36.0. Below it is called `$VENV_VISION`.

## 2. Source checkpoint

```sh
bash scripts/download_source.sh        # IF=<network interface> for its line-idle check (default en1)
shasum -a 256 out/src/decider-2b-vision/model.safetensors
# ac99c16652524d0c6a8017c987efb8aeaca5050f0bf3c7a3d92b927a57ec17ec
```

The script downloads every file of the revision serially, verifies LFS files against the sha256 the Hub publishes and the others against their git blob id, and writes `out/src/DOWNLOAD_OK`. Everything after this reads the snapshot offline (`HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`; `scripts/oracle.py` and `scripts/dump_hf_vision_g256.py` set both themselves).

## 3. Round 1 — fixtures, fp32 oracle, cost of the position contract and of a fixed grid

```sh
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
out/venv-oracle/bin/python -B scripts/make_fixtures.py
out/venv-oracle/bin/python -B scripts/oracle.py --threads 8 --out fixtures/oracle_fp32.json > logs/oracle.log 2>&1
# C7, fresh-process rerun of six rows (the row ids are in results/c7_rerun_oracle.json):
out/venv-oracle/bin/python -B scripts/oracle.py --threads 8 --rows <six row ids> --out results/c7_rerun_oracle.json
python3 scripts/compare_arms.py        # -> results/contract_cost_r1.json, rounds/r1_tables.generated.md
python3 scripts/controls.py            # -> results/controls_r1.json, rounds/r1_controls.generated.md
```

- `make_fixtures.py` draws every public image with PIL (no randomness, no external data) and writes `fixtures/fixtures.json` (sha256 `a3364f386496911e5736a1f651db4486b9b640bd6c6bafd3b9b84420ed77b283`) with each image's file and decoded-RGB sha256; it refuses to overwrite a PNG whose pixels differ. Six internal photo/document rows use images that are not distributed; the script writes every public PNG before it reaches them, and the public PNG hashes in the fixture JSON are the check for a rebuild.
- `oracle.py` is the checkpoint's own `decider/vision.py` (`VisionDecisionModel.prepare` + `slot_logits`), fp32, CPU, 8 torch threads, one row per forward, softmax at T = 1. Arms: `author` (original image, the processor's dynamic resolution), `g256_mrope` / `g512_mrope` (PIL BICUBIC to G×G first, lossless PNG in `out/fixtures_resized/g<G>/`), `g256_pos1d` / `g512_pos1d` (same pixels, `get_rope_index` replaced by `arange` on all three channels). `fixtures/oracle_fp32.json` sha256 `1c4940d6fb9b612fe49b5c5d0e2dc21ee3db7c645e8483344c4882583cb9e0b0`.

## 4. Round 2 — fp32 graphs at 256: vision tower and the derived-M-RoPE decoder

Fixtures v2 (14 borderline public rows appended; v1 unchanged) and their oracle:

```sh
out/venv-oracle/bin/python -B scripts/make_fixtures_v2.py      # -> fixtures/fixtures_v2.json + the added rows' 256 PNGs
out/venv-oracle/bin/python -B scripts/oracle.py --fixtures fixtures/fixtures_v2.json \
  --rows v2_game_pong_nes_level_multi,v2_game_pong_nes_center_multi,v2_game_breakout_nes_edge_multi,v2_game_pong_atari_near_multi,v2_game_pong_atari_noball,v2_game_breakout_atari_fewbricks,v2_game_breakout_atari_high,v2_color_orange,v2_color_purple,v2_color_cyan,v2_color_split_red_blue \
  --arms author,g256_mrope,g256_pos1d --threads 8 --out fixtures/oracle_fp32_v2add.json
python3 scripts/compare_arms_r2.py     # -> results/contract_cost_r2.json, rounds/r2_contract_tables.generated.md
```

`fixtures/fixtures_v2.json` sha256 `6484e46480481495f9b406bb4b1e84bc042092c3eee3144c02de8907a42a113b` (frozen 2026-09-28T14:26:33Z, before any round-2 inference; `fixtures/fixtures_v2.freeze.json` records both fixture hashes and the added rows' PNG hashes); `fixtures/oracle_fp32_v2add.json` sha256 `d2a0720874842d56ab511f6e94b222c30545a0a78f5e0883a8fa40993a997408`.

Vision tower at 256 (encoder: NHWC `[1,256,256,3]` in [0, 1] → `[1,256,1024]`; adapter: → `[1,64,2048]`):

```sh
out/venv-oracle/bin/python -B scripts/vision_interp_5170.py --img 256          # -> out/vision_g256/interp_5170_g256.npz
PYTHONDONTWRITEBYTECODE=1 $VENV_VISION/bin/python -B scripts/convert_vision_g256.py   # -> out/vision_g256/{vision_encoder,vision_adapter}.tflite + result.json
PYTHONDONTWRITEBYTECODE=1 $VENV_VISION/bin/python -B scripts/interp_diff_5141_vs_5170.py   # -> results/vision_interp_diff.json
out/venv-oracle/bin/python -B scripts/dump_hf_vision_g256.py --threads 8        # -> out/hf_vision_g256/*.npy + index.json
out/venv-readout/bin/python -B scripts/vision_parity_g256.py --threads 8        # -> results/vision_parity_r2.json
```

`convert_vision_g256.py` takes the position-embedding taps from transformers 5.17.0 (the `.npz` above) and calibrates its fp16-safe LayerNorm power-of-two pre-scales on every fixture PNG at 256 plus one uniform-noise image. That calibration set includes the six internal rows; without them the script's image list has to drop those rows, and whether the calibrated scales then change was not measured (the scales used are in `out/vision_g256/result.json` `ln_scales`). fp32 outputs: encoder sha256 `dbbf4bf788e4bf7e9c86155db2dc801705a32d4121d1d432060b6629fe52198e` (1,213,891,408 B), adapter `ef297ea0f0732c48f0e881e2c1cee2acd14d70c0c3eb526df0348a58ddfc3acd` (100,702,864 B).

Derived rotary, unit test and the converter clone (the test writes the HF 5.17.0 reference arrays `out/mrope_ref_5170_*.npy` that the export's pre-check reads, so it runs first):

```sh
out/venv-oracle/bin/python -B scripts/test_mrope_derived.py --step-form clamp --out results/mrope_derived_test.json
git clone https://github.com/john-rocky/litert-torch out/litert-torch-d2bv
git -C out/litert-torch-d2bv checkout --detach 115a13607c730c81018bb9789138a3e5e5119e3d
shasum -a 256 ../qwen35_work/qwen35_hybrid_litert_torch.patch     # must be 0a01e2ae9f6bbb0aa79b1ba7de343bad3c4bf17b919a2ccd21c912ede2885aa5
git -C out/litert-torch-d2bv apply ../../../qwen35_work/qwen35_hybrid_litert_torch.patch
```

The converter commit exists on the fork `john-rocky/litert-torch`; check out the full SHA. The patch is the qwen35 hybrid patch shipped in this repository's `qwen35_work/`. After applying, the recorded clone state is 5 files changed (612 insertions, 6 deletions) plus the new `model_ext/qwen3_5/` directory (`results/decoder_export_r4.json` `export_driver.clone`).

Decoder export and inspection. Round 2 exported the rotary's step as `clamp(x, 0, 1)`; the script's current default is round 4's form, so round 2 is reproduced with `--step-form clamp`:

```sh
PYTHONPATH=out/litert-torch-d2bv PYTHONDONTWRITEBYTECODE=1 \
  out/venv-export/bin/python -B -u scripts/export_decoder_g256.py --out out/decoder_g256_fp32 --step-form clamp
out/venv-readout/bin/python -B scripts/inspect_decoder_export.py        # -> results/decoder_export_r2.json
```

`scripts/mrope_derived.install()` replaces the patch's 1-D rotary forward for the lifetime of the export process only (nothing on disk changes). Export arguments (recorded in `export_driver`): prefill ladder 1024/256/64/16/4/1 + decode, cache 4096, externalized single-token embedder, no quantization recipe. `inspect_decoder_export.py` also measures an op delta against an earlier 1-D-rotary export of the same architecture that is not part of this directory (`../out/qwen35vl-decoder-l6/model.tflite`); that comparison is informational.

Graph readout (fp32 vision tflites + decoder, and HF 5.17.0 vision + decoder), then the report:

```sh
out/venv-readout/bin/python -B -u scripts/graph_readout.py --threads 8 > logs/graph_readout_r2.log 2>&1   # -> results/graph_parity_r2.json
```

With no path flags `graph_readout.py` reads the round-2 fp32 files. Per row it zeroes the 48 state buffers, feeds the oracle's own input ids through the embedder tflite, replaces the 64 image slots with the vision output, and for each answer slot prefills the tokens before it with the largest ladder signatures that fit exactly (no padding), decodes the slot token once, and softmaxes the letter logits A.. over the row's options at T = 1. Image rows use positions 0, 1, 2, …; rows without an image start at 65. Mask: 0 on [first position, own position], −1e30 elsewhere. XNNPACK weight cache `out/xnn_cache/decoder_g256_fp32.xnnpack_cache`.

## 5. Round 3 — fp16 files, the first bundle, the runtime on Mac

```sh
for part in decoder embedder vision_encoder vision_adapter_fp16 vision_adapter_int8; do
  TMPDIR=out/tmp_quant PYTHONDONTWRITEBYTECODE=1 out/venv-export/bin/python -B -u scripts/quantize_fp16_r3.py $part
done
out/venv-readout/bin/python -B -u scripts/graph_readout.py \
  --decoder out/fp16_r3/decoder_fp16.tflite --embedder out/fp16_r3/embedder_fp16.tflite \
  --vision A_fp16=out/fp16_r3/vision_encoder_fp16.tflite:out/fp16_r3/vision_adapter_fp16.tflite \
  --vision B_fp16enc_int8adp=out/fp16_r3/vision_encoder_fp16.tflite:out/fp16_r3/vision_adapter_int8.tflite \
  --xnn-cache out/xnn_cache/decoder_fp16_r3.xnnpack_cache --gate-arm A_fp16 --out results/fp16_readout_r3.json
out/venv-readout/bin/python -B scripts/fp16_parity_r3.py                                   # -> results/fp16_parity_r3.json
# the edited readout, run without flags on two rows, must reproduce round 2 value for value:
out/venv-readout/bin/python -B scripts/graph_readout.py --rows synth_teal,text_finance --out out/r3_regression_readout.json
out/venv-readout/bin/python -B scripts/readout_regression_r3.py                            # -> results/readout_regression_r3.json
```

Recipes: decoder and embedder use `build_recipe('wfp16')` from `decider_work/scripts/quantize_decider_ab.py` (weight-only FLOAT_CASTING, 16-bit, CHANNELWISE on FULLY_CONNECTED and EMBEDDING_LOOKUP, float compute); the vision encoder and adapter use a float-casting JSON recipe on FULLY_CONNECTED + CONV_2D; `vision_adapter_int8` is `ai_edge_quantizer.recipe.dynamic_wi8_afp32()` (measured, not shipped). The fp16 vision files written here (`out/fp16_r3/vision_encoder_fp16.tflite`, `out/fp16_r3/vision_adapter_fp16.tflite`) are the vision sections of every round-4 bundle.

Round-3 bundle (superseded by round 4; its decoder still carries RELU_0_TO_1) and its checks:

```sh
out/venv-readout/bin/python -B scripts/build_bundle_g256.py        # -> out/bundle/decider-2b-vision_fp16_noexec.litertlm
TMPDIR=out/tmp out/venv-readout/bin/python -B ../scripts/add_executor_metadata.py \
  out/bundle/decider-2b-vision_fp16_noexec.litertlm out/bundle/decider-2b-vision_fp16.litertlm \
  --litert-lm out/venv-readout/bin/litert-lm --python out/venv-readout/bin/python
out/venv-readout/bin/litert-lm unpack out/bundle/decider-2b-vision_fp16.litertlm --output-dir out/bundle/unpack_r3 < /dev/null
out/venv-readout/bin/python -B -u scripts/graph_readout.py \
  --decoder out/bundle/unpack_r3/Section4_TFLiteModel_tf_lite_prefill_decode.tflite \
  --embedder out/bundle/unpack_r3/Section3_TFLiteModel_tf_lite_embedder.tflite \
  --vision A_fp16=out/bundle/unpack_r3/Section5_TFLiteModel_tf_lite_vision_encoder.tflite:out/bundle/unpack_r3/Section6_TFLiteModel_tf_lite_vision_adapter.tflite \
  --no-hf-arm --gate-arm A_fp16 --xnn-cache out/xnn_cache/decoder_fp16_bundle_r3.xnnpack_cache --out results/bundle_readout_r3.json
out/venv-readout/bin/python -B scripts/inspect_bundle_r3.py        # -> results/bundle_r3.json
out/venv-readout/bin/python -B scripts/runtime_r3.py run --leg cpu
out/venv-readout/bin/python -B scripts/runtime_r3.py run --leg cpu --text
out/venv-readout/bin/python -B scripts/runtime_r3.py run --leg cpu_visgpu
out/venv-readout/bin/python -B scripts/runtime_r3.py run --leg gpu --rows color_red   # engine creation refused (see FINDINGS.md); a second attempt on synth_teal gave the same refusal
out/venv-readout/bin/python -B scripts/runtime_r3.py judge          # -> results/runtime_r3.json
```

Bundle metadata (`build_bundle_g256.py`): `fast_vlm` with image 256 × 256, max_num_tokens 4096, no start token, stop token 248044 only, an identity jinja template (text parts verbatim, the image part rendered as `<|vision_start|><image_soft_token><|vision_end|>`, nothing added for roles or the generation prompt), structured prompt templates present with empty affixes, the snapshot's `tokenizer.json`, and `prefer_activation_type = fp32` on the decoder section. `add_executor_metadata.py` declares the 48 state buffers (36 linear-attention, 12 K/V).

## 6. Round 4 — RELU-free decoder, three weight forms, bundles, readouts, runtime

```sh
PYTHONPATH=out/litert-torch-d2bv PYTHONDONTWRITEBYTECODE=1 \
  out/venv-export/bin/python -B -u scripts/export_decoder_g256.py --out out/decoder_g256_r4_fp32 --step-form relu_diff
out/venv-oracle/bin/python -B scripts/test_mrope_derived.py --step-form relu_diff --out results/mrope_derived_test_r4.json
out/venv-readout/bin/python -B scripts/inspect_decoder_export.py --export-dir out/decoder_g256_r4_fp32 \
  --result results/decoder_export_r4.json --compare-to results/decoder_export_r2.json
out/venv-readout/bin/python -B scripts/lm_head_scope.py out/decoder_g256_r4_fp32/model.tflite --out results/lm_head_scope_r4.json
out/venv-readout/bin/python -B scripts/fused_activation_scan.py out/decoder_g256_r4_fp32/model.tflite --out results/fused_activation_r4.json
out/venv-readout/bin/python -B scripts/fused_activation_scan.py out/decoder_g256_fp32/model.tflite --out results/fused_activation_r2.json
out/venv-readout/bin/python -B -u scripts/graph_readout.py \
  --decoder out/decoder_g256_r4_fp32/model.tflite --embedder out/decoder_g256_r4_fp32/embedder.tflite \
  --vision i_tflite_vision=out/vision_g256/vision_encoder.tflite:out/vision_g256/vision_adapter.tflite \
  --no-hf-arm --gate-arm i_tflite_vision --xnn-cache out/xnn_cache/decoder_g256_r4_fp32.xnnpack_cache \
  --out results/graph_parity_r4_fp32.json
```

`--step-form relu_diff` writes the rotary's step as `relu(x) − relu(x − 1)`, equal to `clamp(x, 0, 1)` on the integer positions used here, without the RELU_0_TO_1 op. The fp32 decoder: `out/decoder_g256_r4_fp32/model.tflite` 7,591,890,608 B, sha256 `127364fe975469a53d5dea7f1a5b6aca1b6c56f1b962a6cd0b41cb32af4a1db1`; embedder `0c839cefd9f18cf0b30b28dff4ffe7d54995583e36b91f55fc63732eca9c14be` (2,034,239,824 B). `lm_head_scope.py` confirms that the regex `^decode_logits_output;$` used by the v7c recipe matches exactly one FULLY_CONNECTED op, the decode signature's `[248320, 2048]` vocabulary projection.

Weight forms (ai-edge-quantizer 0.8.0, one process per part):

```sh
for v in fp16 dyn8 v7c; do for part in decoder embedder; do
  TMPDIR=out/tmp_quant PYTHONDONTWRITEBYTECODE=1 out/venv-export/bin/python -B -u scripts/quantize_r4.py $v $part
done; done
```

| variant | recipe (file) | decoder | embedder |
| --- | --- | --- | --- |
| fp16 | `wfp16` (`decider_work/scripts/quantize_decider_ab.py`) | weight-only fp16 FLOAT_CASTING on every FC, float compute | fp16 EMBEDDING_LOOKUP |
| v7c | `fp16fc_i8emb_dynhead` (`decider_work/gpu_run/scripts/quantize_decider_ab.py`) | weight-only fp16 on every FC, then dynamic int8 CHANNELWISE on the lm_head (`^decode_logits_output;$`) | int8 CHANNELWISE EMBEDDING_LOOKUP |
| dyn8 | `wi8fc` (`decider_work/scripts/quantize_decider_ab.py`) | dynamic int8 CHANNELWISE on every FC | int8 CHANNELWISE EMBEDDING_LOOKUP |

The vision tower is not re-quantized: every variant uses round 3's fp16 encoder and fp16 adapter.

CPU graph readouts, bundles, runtime, GPU readouts, aggregate and report:

```sh
for v in fp16 dyn8 v7c; do
  out/venv-readout/bin/python -B -u scripts/graph_readout.py \
    --decoder out/weights_r4/$v/decoder.tflite --embedder out/weights_r4/$v/embedder.tflite \
    --vision $v=out/fp16_r3/vision_encoder_fp16.tflite:out/fp16_r3/vision_adapter_fp16.tflite \
    --no-hf-arm --gate-arm $v --xnn-cache out/xnn_cache/${v}_r4.xnnpack_cache --out results/readout_r4_$v.json
done
for v in fp16 dyn8 v7c; do zsh scripts/bundle_r4.sh $v; done
for v in fp16 dyn8 v7c; do for leg in cpu gpu; do
  out/venv-readout/bin/python -B scripts/runtime_r4.py run --variant $v --leg $leg
done; done
out/venv-readout/bin/python -B scripts/runtime_r4.py judge                 # -> results/runtime_r4.json
for v in fp16 dyn8 v7c; do for b in cpu gpu; do
  out/venv-readout/bin/python -B -u scripts/gpu_readout_r4.py --variant $v --backend $b   # -> results/gpu_readout_r4_<v>_<b>.json
done; done
out/venv-readout/bin/python -B scripts/weight_forms_r4.py                  # -> results/weight_forms_r4.json
```

- `bundle_r4.sh <variant>` builds the bundle (`build_bundle_g256.py` with the variant's decoder and embedder and the round-3 fp16 vision files; same metadata as round 3), adds ExecutorMetadata (`../scripts/add_executor_metadata.py`, `TMPDIR=out/tmp`), unpacks it with `litert-lm unpack`, reads the unpacked sections with `graph_readout.py` through a fresh XNNPACK cache, and runs `inspect_bundle_r4.py --variant <variant>` (header, template, stop token, 48 states, fp32 activation preference, section bytes equal to the inputs, tokenizer equal to the snapshot, content readout bit-identical to the variant's CPU graph readout). After that inspection passes, it deletes its own regenerable intermediates (the pre-ExecutorMetadata copy, the unpack dir, the readout cache) and logs each deletion to `logs/r4_deletions.log`.
- `runtime_r4.py run` drives `runtime_row_r3.py --bundle out/bundle_r4/<v>/decider-2b-vision_<v>.litertlm` one row per process, one process at a time, stdin `/dev/null`, one cache dir per bundle (`out/runtime_cache/decider-2b-vision_<v>_r4`). Each process: LiteRT-LM 0.17.1 (PyPI) `Engine(bundle, backend, vision_backend, max_num_tokens=4096, max_num_images=1, cache_dir, enable_benchmark=True)`, `create_conversation(sampler_config=SamplerConfig(top_k=1), max_output_tokens=3)`, one user message `[image (the 256×256 PNG), text]`, streamed with `send_message_async`. `cpu` = `Backend.CPU(thread_count=8)` for the decoder and the vision encoder; `gpu` = `Backend.GPU()` for both (the PyPI build's WebGPU delegate). Rows: the 12 single-question public image rows fixed in `scripts/r3_rows.py`. `judge` compares the prefill token count with the oracle's input length and the first streamed token with the oracle's and the CPU graph's full-vocabulary top-1.
- `gpu_readout_r4.py` runs the decoder on ai-edge-litert 2.2.0 CompiledModel (`--backend gpu` = the Metal accelerator with `GpuOptions(enforce_f32=True)`; `--backend cpu` = the same feeding on CPU as a control) with ONE padded prefill chunk per slot (a fresh zero state per slot; pad rows: embedding 0, position 0, mask −1e30), then one decode. The embedder and the vision tflites run on CPU from the same files as the CPU readout. A three-row CPU smoke run preceded the full runs (`results/gpu_readout_r4_smoke_fp16_cpu.json`, rows `synth_teal`, `text_finance`, `v2_game_breakout_atari_high`).

## Round 6 — public reference, fixture check and Mac measurements

Round 6 converts nothing. It writes the reference readout that ships in the model repo (`publish/reference/`), cuts the public fixture set, checks the reference against round 4 bit for bit, and measures one decision on the Mac.

The reference needs only ai-edge-litert, numpy, Pillow and tokenizers (`requirements-lock-reference.txt`, Python 3.12.13). It reads the `.litertlm` header itself (`bundle_cache.py`: magic, version, FlatBuffer section table; TFLite sections copied byte for byte, the HF tokenizer section decompressed), so it needs neither litert-lm nor torch nor transformers.

```sh
# the minimal environment
UV_CACHE_DIR=out/cache/uv uv venv out/venv-ref --python /opt/homebrew/bin/python3.12
UV_CACHE_DIR=out/cache/uv uv pip install --python out/venv-ref/bin/python ai-edge-litert==2.2.0 numpy==2.5.3 pillow==12.3.0 tokenizers==0.23.2

# the public fixture set: public rows minus the five platformer-style frames; images, 256x256 inputs, three oracle arms
python3 -B scripts/r6_make_public_fixtures.py        # -> publish/reference/fixtures/{fixtures.json,images/,images_256/}

# the reference on every published row, CPU (exact-fit prefill), then the bit comparison with round 4's CPU readouts
scripts/r6_run_checks.sh                             # -> results/r6/check_<v>_cpu.json, results/r6/bit_identity_<v>.json
# the reference's GPU rule (Metal, one padded chunk per slot), informational
scripts/r6_run_gpu_checks.sh                         # -> results/r6/check_<v>_gpu.json

# the card's example
out/venv-ref/bin/python -B publish/reference/decider_litert.py --bundle out/bundle_r4/v7c/decider-2b-vision_v7c.litertlm \
  --request publish/reference/example_request.json --cache-dir out/ref_cache     # -> results/r6/example_cli_v7c.json

# contract-cost numbers for the card (from the oracle files) and round-4 readouts restricted to the published rows
python3 -B scripts/r6_card_numbers.py                # -> results/r6/card_numbers.json
python3 -B scripts/r6_public_from_r4.py              # -> results/r6/public_from_r4.json
```

`check_fixtures.py` rebuilds every request from the original image and the text (PIL bicubic to 256x256; the checkpoint's `build()` -> decode -> encode), asserts that the 256x256 pixels, the token ids and the answer slots equal the oracle's, and reads the letter probabilities. `r6_bit_identity.py` then compares, per published slot, the float32 letter logits and the sha256 of the full-vocabulary logits with `results/readout_r4_<v>.json`, and the text- and vision-embedding hashes per row. Result: 62/62 slots, 42/42 text embeddings and 37/37 vision embeddings identical for fp16 and for v7c, and the extracted sections are byte-identical to the files round 4 read. The GPU rule gave 62/62 letter logits bit-identical to `results/gpu_readout_r4_<v>_gpu.json` (`results/r6/gpu_vs_round4.json`).

Mac measurements (`scripts/r6_mac_bench.py`): one decision per process on the published row `game_pong_atari_up` (150 tokens, one question), three processes per cell, each under `/usr/bin/time -l` with stdin `/dev/null`. Before every process the driver records `uptime` and every other process above 120 % CPU, and waits for them to go (up to 20 minutes before the first process); it samples the same list every 5 s during the run and marks the row contended if anything appeared. GPU processes follow a rest (300 s before the first, 120 s between them). Cells: fp16 and v7c through LiteRT-LM 0.17.1 (`scripts/r6_bench_row.py`: `Engine(..., cache_dir=':nocache')` = `--cache no`, CPU 8 threads or `Backend.GPU()`, greedy, `max_output_tokens=1`; engine-constructor wall, send-to-first-token wall, `get_benchmark_info()`), and the reference readout (`scripts/r6_ref_bench_row.py`, its section folder and weight cache already on disk). A second series (`scripts/r6_bench_disk_after.sh`) repeats the runtime cells with a cache folder, empty before the first run of each cell.

```sh
scripts/r6_bench_after_checks.sh                     # -> results/r6/mac_bench.json, results/r6/bench/*.json, logs/r6/bench/*.log
scripts/r6_bench_disk_after.sh                       # -> results/r6/mac_bench_disk.json
scripts/r6_post_bench.sh                             # re-runs contended rows (--redo-contended; a 4th cache-folder run so each
                                                     # folder cell has 3 warm runs), refreshes the public fixtures, re-runs the
                                                     # checks, the card's usage (r6_usage_test.py) and runtime_example.py
python3 -B scripts/r6_perf_table.py                  # -> results/r6/perf_summary.json (medians over uncontended runs)
python3 -B scripts/r6_perf_markdown.py               # the card's Performance tables
```

The decoder's GPU program cache size in a cache folder was sampled after the GPU runs (`results/r6/gpu_program_cache_growth.json`). The card draft is checked against the result files by `scripts/r6_verify_card.py`, the manifest is generated by `scripts/r6_make_manifest.py` (the repository's `manifest/make_manifest.py`, unmodified, with the Hub file listing answered from the local bundles because the repository does not exist yet; `--public` for the uploadable copy), and `scripts/r6_sha256sums.py` writes `publish/SHA256SUMS`.

The public mirror copy of this directory is staged by `scripts/r6_stage_mirror.py` (curated; notes in `out/mirror_stage/STAGE_NOTES.md`).
