# decider-2b-vision → LiteRT-LM: findings and traps

What rounds 1–4 established, each with the measured number and the file it is read from. Probability differences are per answer slot: |Δp| = the largest absolute difference over that slot's options, softmax at T = 1; p95 is numpy's linear percentile over slots; a tie is a reference top-2 gap ≤ 1e-4 (no slot in these fixtures is a tie). "All rows" includes six internal photo/document rows kept internal; "public" below means the public rows without the five platformer-style frames (a platformer-style frame), which is 37 image rows / 53 answer slots plus 5 rows / 9 slots without an image. The reference is the checkpoint's own `decider/vision.py` in fp32 on CPU (transformers 5.17.0, torch 2.14.0): `fixtures/oracle_fp32.json` + `fixtures/oracle_fp32_v2add.json`.

## 1. The runtime's 1-D positions are not enough; M-RoPE is derived inside the decoder

The LiteRT-LM `fast_vlm` path numbers the prompt 0, 1, 2, … on one channel. The checkpoint uses 3-channel M-RoPE: `<|vision_start|>` at (0, 0, 0), the 64 image tokens at (1, 1 + row, 1 + col) of the 8×8 merged grid, and every later token at p − 56 on all three channels.

| comparison: fixed-256 M-RoPE → fixed-256 1-D positions | slots | non-tie argmax | max \|Δp\| | p95 | slots > 0.02 | source |
| --- | --- | --- | --- | --- | --- | --- |
| v1, all rows | 48 | 47/48 | 0.1983 | 0.0128 | 1 | `results/contract_cost_r1.json` |
| v1 + v2, all rows | 75 | 73/75 | 0.1983 | 0.0275 | 6 | `results/contract_cost_r2.json` |
| v1 + v2, public | 53 | 51/53 | 0.1983 | 0.0432 | 6 | computed from the two oracle files (stored probabilities; the softmax of the stored letter logits gives the same values) |
| v1, all rows, at 512 | 48 | 48/48 | 0.0898 | 0.0520 | 6 | `results/contract_cost_r1.json` |

The largest move is `game_pong_nes_multi` slot 1 (reference top-2 gap 0.237): probabilities [0.612, 0.375, 0.013] become [0.416, 0.573, 0.011], so the answer flips. The second public flip is `v2_game_breakout_atari_high` slot 1 (gap 0.111, |Δp| 0.064). The pre-registered rule (every non-tie argmax equal and max |Δp| ≤ 0.02) failed at both 256 and 512.

The fix is `scripts/mrope_derived.py`: the three channels are computed from the 1-D position with element-wise ops only (steps, a 7-step floor division by the grid width, 0/1-masked inverse-frequency constants), so the graph keeps the op set of the hybrid export and needs no GATHER. Against HF 5.17.0 the derived positions are exact on 48/48 fixture rows and over the whole 4096-position cache (`results/mrope_derived_test.json`, `results/mrope_derived_test_r4.json`). The fp32 graph then reproduces the M-RoPE reference, not the 1-D one: on all 16 slots where the two reference arms differ by more than 0.01 it sits on the M-RoPE side, and its largest distance to the 1-D arm is 0.1982 (`results/graph_parity_r2.json` `falsification_vs_pos1d`).

## 2. The price of a fixed 256×256 input

The author's processor picks a resolution per image (`smart_resize`, factor 32, at least 65,536 pixels); the bundle takes one fixed 256×256 image (64 tokens).

| comparison: author's processing → PIL BICUBIC 256×256 | slots | non-tie argmax | max \|Δp\| | p95 | slots > 0.02 | source |
| --- | --- | --- | --- | --- | --- | --- |
| v1, all rows | 48 | 46/48 | 0.5835 (an internal photo row) | 0.2706 | 9 | `results/contract_cost_r1.json` |
| v1 + v2, all rows | 75 | 72/75 | 0.5835 | 0.3017 | 14 | `results/contract_cost_r2.json` |
| v1 + v2, public | 53 | 51/53 | 0.3729 | 0.3033 | 12 | computed from the two oracle files |

Public flips: `synth_center_circle` slot 0 (300×500 original, gap 0.224, |Δp| 0.222) and `v2_game_breakout_atari_fewbricks` slot 0 (160×210 original, gap 0.259, |Δp| 0.373).

By original size (width × height), public rows (the author's processed size as height × width and its token count, from the oracle's `processor_resized_hw` / `image_grid_thw`):

| original size | author processed size / tokens | slots | max \|Δp\| | argmax equal | slots > 0.02 |
| --- | --- | --- | --- | --- | --- |
| 224×224 | 256×256 / 64 | 7 | 0 | 7/7 | 0 |
| 256×240 | 256×256 / 64 | 13 | 0 | 13/13 | 0 |
| 256×256 | 256×256 / 64 | 1 | 0 | 1/1 | 0 |
| 160×210 (Atari-type frames) | 320×224 / 70 | 13 | 0.3729 | 12/13 | 5 |
| 300×500 | 512×288 / 144 | 3 | 0.2216 | 2/3 | 2 |
| 512×512 | 512×512 / 256 | 2 | 0.0007 | 2/2 | 0 |
| 640×480 | 480×640 / 300 | 8 | 0.0660 | 8/8 | 2 |
| 800×600 | 608×800 / 475 | 6 | 0.3126 | 6/6 | 3 |

On every 224×224, 256×240 and 256×256 original (18 rows including the platformer-style frames; 13 public rows) the author's `pixel_values`, input ids and letter logits are bit-identical to the fixed-256 arm (oracle files; `results/contract_cost_r1.json` `author_grid_equals_G` for the v1 rows). The non-trivial cases are the 256×240 game frames, which both paths stretch to 256 rows: the processor's torchvision bicubic and PIL BICUBIC gave identical pixels on these synthetic frames. That is a measurement on these images, not a general equivalence of the two resamplers. Everything larger or of another shape gets fewer tokens than the author's processing, and the probabilities move as tabled. G = 512 was not better: all rows, v1, 46/48 and max 0.6098 (`results/contract_cost_r1.json`), with two flips of its own on game frames.

The oracle is stable: a fresh-process rerun of 30 forwards was bit-identical (`results/controls_r1.json` C7), and at G×G the processor adds nothing beyond `(x − 127.5) / 127.5` and the patch order (C4: max 2.945e-08 against a float64 hand computation, exact against float32 on 136/136 forwards), so the vision encoder takes [0, 1] pixels and normalizes in the graph.

## 3. transformers 5.14.1 (export) vs 5.17.0 (reference)

- Vision position table: both versions resample the learned 48×48 table with the same bilinear taps in the same order, but 576 of the 1,024 weights are rounded differently, by up to 3.58e-06; on a random table the resampled values differ by up to 1.69e-05 (`results/vision_interp_diff.json`). The vision export therefore uses 5.17.0's taps (`scripts/vision_interp_5170.py` → `out/vision_g256/interp_5170_g256.npz`).
- Text rotary: 5.14.1 recombines the three channels on the frequencies, 5.17.0 on cos/sin. The derived form selects on the frequencies; the selected frequencies are bit-equal to HF's and cos/sin differ by at most 5.96e-08 (bit-equal on 39 of 48 rows, and the same bound over all 4096 positions; `results/mrope_derived_test.json`). The same cos values computed by torch in HF's `[3, B, S, 32]` shape and in `[S, 32]` already differ by that amount (`rounds/r2_notes.md`), so the residue is the evaluation path, not the positions.
- Result: the fp32 vision tflites against HF 5.17.0 over 48 images: correlation ≥ 0.99999992, max |diff| 3.34e-03 (8.5e-04 of the image's largest value), per-token cosine ≥ 0.9999936 (`results/vision_parity_r2.json`). The fp32 graph against the oracle: 75/75, max |Δp| 4.77e-05; with HF's vision output instead of the tflites, 6.67e-06 (`results/graph_parity_r2.json`).

## 4. RELU_0_TO_1 blocks the WebGPU delegate; `relu(x) − relu(x − 1)` does not

- Round 2 wrote the derived rotary's step as `clamp(x, 0, 1)`; the converter lowered it to RELU_0_TO_1, 9 per signature, 63 in the decoder (`results/decoder_export_r2.json`).
- Round 3, LiteRT-LM 0.17.1 (PyPI) with the decoder on GPU: the WebGPU delegate listed `DEQUANTIZE:` (no reason printed) and `RELU_0_TO_1: Not supported op RELU_0_TO_1`, split 25,675 ops to GPU and 252 to CPU (25,927 = the op count of `prefill_1024`, `results/bundle_r3.json`), printed "Hint fully delegated to single delegate is set, but the graph is not fully delegated", and engine creation failed, identically on two attempts (`results/runtime_r3.json`).
- Round 4 wrote the step as `relu(x) − relu(x − 1)` (equal on integer inputs): RELU_0_TO_1 0, per signature RELU +10, SUB +17 (`results/decoder_export_r4.json`); the converter fused 8–9 of the relus into the SUB that forms `x − 1` (`results/fused_activation_r4.json`). The fp32 letter logits are bit-equal to round 2 on 75/75 image and 9/9 image-less slots (`results/weight_forms_r4.json` `fp32_graph.vs_r2_graph`).
- With only that change, the same runtime created the full-GPU engine on 12/12 rows for all three weight forms: 0 unsupported-op lines, 0 partial-delegation lines, 0 `Validation error` lines, 7 delegate kernels per decoder (`results/runtime_r4.json`). The fp16 and v7c decoders still carry DEQUANTIZE (181 per prefill signature, `results/bundle_r4_fp16.json`), so DEQUANTIZE alone did not block this Mac WebGPU path. Whether the Android GPU executors accept it was not measured here.

## 5. Weight forms

| form (decoder / embedder) | CPU graph, image slots: non-tie argmax, max \|Δp\|, p95, > 0.02 | image-less from 65 | CPU one padded chunk vs CPU exact-fit (image / image-less) | Metal GPU one padded chunk: argmax, max, p95 | GPU vs CPU, same padded feeding (84 slots) |
| --- | --- | --- | --- | --- | --- |
| fp16 (fp16 FC + fp16 embedding) | 75/75, 4.06e-05, 2.26e-05, 0 | 9/9, 2.41e-06 | 5.60e-06 / 1.44e-06 | 75/75, 4.07e-05, 2.38e-05 | 6.59e-06 |
| v7c (fp16 FC + dynamic int8 lm_head + int8 embedding) | 74/75, 8.46e-03, 6.17e-03, 0 | 9/9, 3.06e-03 | 9.00e-05 / 2.38e-07 | 75/75, 6.21e-03, 2.04e-03 | 8.59e-03 |
| dyn8 (dynamic int8 FC + int8 embedding) | 72/75, 0.4285, 0.1535, 26 | 9/9, 0.0605 | 0.2493 / 0.0869 | 74/75, 0.0321, 9.91e-03 | 0.2414 |

Sources: `results/readout_r4_<form>.json` (CPU exact-fit), `results/gpu_readout_r4_<form>_{cpu,gpu}.json` (one padded chunk), `results/weight_forms_r4.json` (`padded_readout.gpu_vs_cpu_same_scheme`). Vision is the fp16 encoder + fp16 adapter in every form.

- fp16 is exact to the reference within 4.1e-05 on CPU and on the Metal GPU, and reading the bundle's own sections gives the same full-vocabulary logits bit for bit on 93/93 slots, for all three forms (`results/bundle_r4_<form>.json` `content_readout`).
- v7c's only flip is `v2_game_breakout_atari_fewbricks` slot 0, whose reference top-2 gap is 0.0073 (|Δp| 0.0052); on the Metal GPU it does not flip.
- dyn8 on CPU: the dynamic activation quantization moves the probabilities by up to 0.4285 (`v2_game_breakout_atari_high` slot 1), three non-tie flips with reference gaps 0.11–0.26, and the result depends on how the prompt is split into prefill chunks (0.2493 between two feedings of the same file; the same control moves fp16 by 5.6e-06 and v7c by 9.0e-05). How the dynamic quantization sees a chunk was not measured. On the Metal GPU the same file reads close to the weight-only forms (max 0.0321). In the runtime, `synth_center_circle` gave the reference answer `B` on both legs while the dyn8 CPU graph gives `A` under both feedings; the runtime's own chunk plan is not observed, so which feeding it ran is not known.
- An int8 vision adapter (fp16 encoder) moves the probabilities by up to 9.43e-03, 26 of 75 slots above 1e-3 (`results/fp16_readout_r3.json` arm `B_fp16enc_int8adp`); the bundles keep the fp16 adapter.

## 6. XNNPACK unpacks fp16 weights to fp32 size

The CPU XNNPACK weight cache of the fp16 decoder is 7,540,052,280 B, the same size as the fp32 decoder's (`results/runtime_r3.json` `disk`); v7c's is 6,016,360,768 B and dyn8's 1,896,902,288 B (`results/runtime_r4.json` `disk`). The fp16 file is therefore a desktop file in memory terms even though it is 5.5 GB on disk. The WebGPU runtime writes a program cache of 3,257,608,032 B (fp16, v7c) or 3,276,413,448 B (dyn8) per bundle, and a separate GPU weight cache only for int8 weights (508,563,920 B v7c, 1,881,299,600 B dyn8, 4,408 B fp16).

## 7. Rows without an image need positions from 65

The derived rotary maps every position ≥ 65 to p − 56 on all three channels, so an image-less row read from position 65 has the author's relative positions: 9/9 slots, max |Δp| 2.28e-06 in fp32 (`results/graph_parity_r2.json` `text_from_65`), 2.41e-06 in fp16. Read from position 0 instead, text tokens 1–64 land on image-grid positions: 8/9, max 0.0497 (`text_mario_noimage` slot 1, `text_from_0_informational`). The LiteRT-LM runtime numbers an image-less prompt from 0, so text-only requests belong on a readout that starts at 65. The two image-less runtime rows of round 3 cannot show the difference: their top-1 is the same from 0 and from 65, and their probabilities differ by up to 0.02 (`rounds/r3_notes.md`).

## 8. The runtime on Mac (LiteRT-LM 0.17.1, PyPI)

- All six legs (fp16 / v7c / dyn8 × CPU / GPU) ran 12/12 single-question image rows: prefill token count = the reference input length, and the first streamed token = the reference's full-vocabulary top-1 on 12/12; for dyn8, 11/12 equal its own CPU graph (`results/runtime_r4.json`). The runtime reports no BOS and stop token 248044, renders each message as `<|vision_start|><image_soft_token><|vision_end|>` + the text, and tokenizes the text to the reference ids (`results/runtime_r3_rows/`, `results/bundle_r3.json`).
- The runtime rows were fed the 256×256 PNG already resized with PIL, so the runtime's own resize of other image sizes was not exercised.
- The runtime path is a single question per request and the greedy first token; it gives the argmax, not the option probabilities.
- The bundle requests fp32 activations for the decoder (`prefer_activation_type`); no runtime log line states the precision the GPU executor used.

## 9. Two different GPU paths on one Mac

LiteRT-LM's PyPI build runs its GPU backend through a statically linked WebGPU delegate (log: `Selected adapter: Apple M4 Max, arch=metal-3, vendor=apple, backend=Metal`, accelerator `GPU WebGPU`). ai-edge-litert 2.2.0 CompiledModel with the GPU accelerator loads a Metal accelerator library instead (`Initializing Metal-based API from graph`, 7 per decoder). The probability numbers in section 5 come from the second path, the runtime legs from the first; one does not stand in for the other. Every Metal CompiledModel load printed `AGX: exceeded compiled variants footprint limit` once (`logs/gpu_readout_r4_<form>_gpu.log`) and still reported full acceleration and finite outputs on all 84 slots; what the warning changes, if anything, was not examined.

## 10. Round 6: the public reference reproduces the conversion's readout bit for bit

- `publish/reference/` reads the bundle with ai-edge-litert 2.2.0, numpy, Pillow and tokenizers only. It parses the `.litertlm` header itself (magic `LITERTLM`, major version 1, header end at byte 24, a FlatBuffer section table from byte 32; the HF tokenizer section is an 8-byte uncompressed size followed by a zlib stream), so it needs neither litert-lm nor torch nor transformers. The sections it extracts are byte-identical to the files round 4 read (`results/r6/bit_identity_<v>.json` `section_sha256_equal_round4`).
- On the 42 published requests (62 slots) it rebuilds the upstream token ids and answer slots from the text with the checkpoint's own `build()` (decode, then encode again) and the 256x256 pixels from the original image, all equal to the oracle's, and its letter logits and full-vocabulary logits are bit-identical to round 4's CPU graph readout for fp16 and v7c; text and vision embeddings too (42/42, 37/37). Its GPU rule (Metal, one padded chunk) is bit-identical to round 4's GPU readout on 62/62 slots (`results/r6/gpu_vs_round4.json`).
- Published-row agreement with the fp32 oracle (`results/r6/check_<v>_<backend>.json`): fp16 CPU 53/53, max |dp| 4.06e-05; v7c CPU 52/53, max 0.00846; fp16 Metal 53/53, 4.07e-05; v7c Metal 53/53, 0.00621; text-only 9/9 in all four.
- The runtime's greedy token is the top token over the whole vocabulary. On `v2_game_pong_nes_center_multi` question 1 (two options) the top token is `C` in upstream fp32 and in both files; the runtime path only reads the last question of a request, and on every single-question published row the top token is the option argmax, but a request can in principle get a letter that is not an option.
- The curated fixture generators staged for the public mirror (internal photo/document rows and the five platformer-style frames removed) regenerate the 42 published rows with identical image bytes, RGB hashes, contexts, questions and options (run in a scratch copy).
- The public mirror's copy of the hybrid patch (`qwen35_work/qwen35_hybrid_litert_torch.patch`, sha256 `997bfc4d…`) lacks the externalized-embedder pad guard (valid rows from position monotonicity) that this VLM decoder was exported with (`0a01e2ae…`); the staged mirror copy carries the exact patch under `deps/`.

## 11. Round 6: one decision on the Mac through LiteRT-LM 0.17.1

One decision per fresh process (published row `game_pong_atari_up`, 150 prompt tokens, greedy, one output token), three uncontended processes per cell; medians (`results/r6/perf_summary.json`, from `results/r6/mac_bench.json` and `results/r6/mac_bench_disk.json`):

| file | backend | cache folder | TTFT s | engine creation s | prefill tok/s | peak RSS GB | peak footprint GB |
| --- | --- | --- | --- | --- | --- | --- | --- |
| fp16 | CPU | none | 1.68 | 16.8 | 132 | 53.7 | 47.9 |
| fp16 | CPU | warm | 1.72 | 13.7 | 115 | 21.2 | 6.6 |
| fp16 | GPU | none | 0.29 | 28.5 | 826 | 15.6 | 19.3 |
| fp16 | GPU | warm | 0.32 | 29.2 | 811 | 15.8 | 19.6 |
| v7c | CPU | none | 1.54 | 16.3 | 140 | 45.5 | 41.3 |
| v7c | CPU | warm | 4.42 | 13.4 | 49 | 12.5 | 1.5 |
| v7c | GPU | none | 0.26 | 26.8 | 843 | 7.0 | 10.6 |
| v7c | GPU | warm (2 runs) | 0.47 | 30.2 | 688 | 6.7 | 10.2 |

- `cache_dir=':nocache'` (the house `--cache no`) makes the CPU process peak at 47.9 GB (fp16) / 41.3 GB footprint and 53.7 / 45.5 GB RSS; with a cache folder the footprint is 6.6 / 1.5 GB. The `--cache no` protocol row is therefore not what a default user sees on this 7-signature decoder; both are tabled. Cause not measured here (the house note on per-signature repacking without a shared cache is the likely one).
- With a warm cache folder, CPU TTFT splits into two groups: 1.16–1.72 s when the cache file was resident, 4.4–7.6 s with 463,370–556,840 page faults (the file read back from disk; `/usr/bin/time -l` in `logs/r6/bench/*_disk.log`), against 78–173 page faults in the fast runs. The v7c warm-folder median (4.42 s) comes from two such runs.
- The WebGPU program cache in a cache folder grows by 270,663,680 B at every engine creation and a warm folder did not shorten GPU engine creation (26.5–31.1 s over all GPU runs, either way); see the correction below.
- The runtime's `init_time_in_second` reads about twice the Engine constructor wall in every run (e.g. 34.96 s vs 17.18 s); the card quotes the constructor wall.
- Every one of the 32 runtime processes (including the re-run ones) gave `A`, and none printed a `Validation error` line.
- The reference readout (CompiledModel CPU, warm section folder and weight cache): constructor 12.8–12.9 s median; first decision in a process 1.00–11.69 s over six uncontended runs, second decision 0.89–1.08 s.
- Peer load: other processes on the same Mac ran intermittently. Six runs were marked contended (a process above 120 % CPU before, during or after): five were re-run (the originals are kept under `superseded_runs`) and the sixth, the v7c GPU warm-folder run 3, is left out of the medians.

## Open questions

- Android GPU executors: engine creation, delegation and memory for fp16, v7c and dyn8 on a phone are not measured in rounds 1–4; the Mac WebGPU result does not transfer.
- Prompts longer than 2,048 tokens with fp16 activations: the derived rotary computes positions in float, and fp16 cannot represent every integer above 2,048. Every fixture row is at most 150 tokens and the bundle requests fp32 activations; this was not exercised.
- The runtime's own image resize for inputs that are not 256×256 (see section 8).
- A public rebuild of the vision tower calibrates its fp16-safe LayerNorm pre-scales without the six internal rows; whether any power-of-two scale changes was not measured.
- Why the WebGPU delegate counts 96 (six kernels) and 97 (one kernel) external tensors for signatures that declare 48 state inputs, 48 state outputs and 3 more inputs was not examined.

## Corrections to earlier accounts

- Section 6 (and round 4's disk table) gives the WebGPU program cache as 3,257,608,032 B per bundle. That is its size after round 4's 12 GPU engine creations with one cache folder. In round 6 the decoder's program cache grew by 270,663,680 B at every engine creation (550,971,232 B after two, 821,634,912 B after three; `results/r6/gpu_program_cache_growth.json`); 3,257,608,032 / 12 = 271,467,336 B per creation matches. A warm folder did not shorten GPU engine creation (see section 11).

- The round log summarizes control C4 as "max 5.9e-8"; the stored values are max 2.945e-08 against a float64 hand computation and exactly 0 against float32 on all 136 (v1) and 28 (v2) forwards that carry the check (`results/controls_r1.json`, the `c4` records in both oracle files).
