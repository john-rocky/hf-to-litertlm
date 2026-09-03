# Qwen3.5 MTP — speculative decoding bundles for LiteRT-LM (experimental, not shipped)

This directory converts Qwen3.5-0.8B / 2B into `.litertlm` bundles that run under LiteRT-LM's speculative decoding (`--speculative-decoding true`, litert-lm ≥ 0.16.0): the base graph gains a `verify` signature, and the checkpoint's own 1-layer MTP (multi-token prediction) head is exported as the drafter section the runtime consumes. It is the first third-party MTP bundle for this runtime that we know of.

**Status (2026-09-04): built and gated, not shipped.** Two runtime findings hold the ship: (1) the runtime does not commit the turn-final stop token under the flag, so multi-turn conversations see a malformed transcript (LiteRT-LM [#3439](https://github.com/google-ai-edge/LiteRT-LM/issues/3439), reproduces on litert-lm 0.16.0 and 0.16.1); (2) on iPhone the flag-on leg decodes slower than flag-off in our harness (iOS-runtime per-round overhead; the fp16-activation hypothesis was refuted with an fp32-activation pair). Single-turn generation and benchmarks are unaffected by (1). A ready-made 0.8B bundle for reproducing #3439 is public at [mlboydaisuke/Qwen3.5-0.8B-MTP-repro-LiteRT](https://huggingface.co/mlboydaisuke/Qwen3.5-0.8B-MTP-repro-LiteRT).

## Recipe (three steps)

```bash
# 0. litert-torch at the Qwen3.5 pin + the combined patch (hybrid recipe + MTP additions; apply INSTEAD of
#    qwen35_work/qwen35_hybrid_litert_torch.patch, on a clean 115a136)
git clone https://github.com/google-ai-edge/litert-torch litert-torch-mtp
git -C litert-torch-mtp checkout 115a136
git -C litert-torch-mtp apply "$(pwd)/mtp_work/qwen35_mtp_litert_torch.patch"

# 1. base graph: prefill ladder + decode + verify, embeddings externalized (mandatory: verify must read the
#    same table as decode or greedy equivalence breaks), cache 4096, G=3 draft steps, ring R=G+2
PYTHONPATH=litert-torch-mtp MTP_BUNDLE=1 python mtp_work/convert_qwen35_mtp.py Qwen/Qwen3.5-0.8B mtp_work/out_08b
python mtp_work/finalize_mtp_bundle.py mtp_work/out_08b/model.litertlm mtp_work/out_08b/Qwen3.5-0.8B_mtp_int8.litertlm

# 2. drafter: the checkpoint's mtp.* tensors + tied head as a single-signature tflite (equivalence-gated at
#    build time against a transformers reference), int8, packed as a tf_lite_mtp_drafter section
PYTHONPATH=litert-torch-mtp python mtp_work/export_mtp_drafter.py Qwen/Qwen3.5-0.8B mtp_work/out_08b/drafter.tflite --quantize
python mtp_work/pack_mtp_drafter.py mtp_work/out_08b/Qwen3.5-0.8B_mtp_int8.litertlm mtp_work/out_08b/drafter_int8.tflite mtp_work/out_08b/Qwen3.5-0.8B_mtp_drafter_int8.litertlm

# 3. run
litert-lm run mtp_work/out_08b/Qwen3.5-0.8B_mtp_drafter_int8.litertlm --backend cpu --speculative-decoding true --prompt "..."
```

Environment used: transformers 5.14.1, torch 2.13, ai-edge-quantizer 0.9.0, litert-lm 0.16.0 CLI for `unpack`/`pack`. `export_mtp_drafter.py --topk 32768` builds a top-K drafter head (one sliced tied-head matmul, re-expanded in-graph to full-vocab logits) that cuts the drafter section 278 → 57 MB (0.8B) / 573 → 132 MB (2B) with verify untouched.

## What the patch adds (design: `P1_DESIGN.md`)

- **Ring-addressed gated-delta state** (`ring_size = G+2`) so verify rounds can be rewound by overwrite: the runtime drafts G tokens past the accepted prefix, and the rejected tail's linear-attention state must not survive into the next round. The legacy single-slot path stays untouched for non-MTP exports.
- **A `verify` signature** — decode-shaped graph over G+1 positions, embeddings in, per-position fp32 logits and `activations` out; `activations` is also added to decode (the drafter consumes the previous hidden state).
- **Drafter contract** (`llm_litert_mtp_drafter.cc` at the 0.16.0 tag): inputs `activations [1,1,2H]` (embedding first, previous hidden second), `input_pos [1]`, `mask [1,1,1,cache_len]` (declared but unused: the released runtime initializes it unconditionally and dies without it), pair K/V `kv_cache_{k,v}_mtp` bound by name from the base bundle's state buffers; outputs `logits` and `projected_activations`.
- Two traps worth knowing: an in-place `conv_state.copy_()` is a dead write against a ring slice (rewritten functionally), and `tfl_dus` start indices are 0-d int32 with only one mismatching dim per call — block writes must never straddle slot −1.

## Verification record (0.8B unless noted)

- Interpreter-level gate (float tflite vs stock HF fp32, forced rejection at every draft index, continuation prefill over the garbage tail): 33/33 checks, logits max|diff| ≤ 6e-5, argmax equal at every committed position, through ring wraps and rewinds. 2B: 33/33.
- Released-runtime gates (litert-lm 0.16.0, CPU): 8-question chat gate 7/8 (2B 8/8); hermetic single-word fills 1..16 under the flag 16/16; multi-turn sessions with every turn ending mid-round: 2B all green both flags.
- Output equality vs flag-off: 1/4 prompts byte-identical per size; the rest diverge where an HF-fp32 referee shows top-2 gaps of 0.125–0.375 (median 1.5–9.1), i.e. int8 near-ties, not a contract error. Verify greedy is exact w.r.t. its own logits.
- Speed, Apple M4 Max CPU, `litert-lm benchmark -p 128 -d 256 --cache no`, flag off → on: 0.8B decode 52.3 → 58.4 tok/s (1.12×), prefill 611.5 → 627.1; 2B decode 41.4 → 51.3 (1.24×). Engine acceptance on the 0.8B: 2.43–3.13 tokens per round across 4 prompts (mean 2.76), matching the desktop oracle (2.95).
- Galaxy S26 CPU, capped 150-token decode, legs interleaved with cooling waits, median of 3: 0.8B full-head 32.4 → 22.9 tok/s (0.71×) on a document-continuation prompt; math prompts go the other way (higher acceptance). Phones are where MTP should win on paper (decode is weight-read-bound), and this is the number that currently says otherwise for the small model.

## The #3439 probe

`stop_token_probe_3439.py <bundle>` runs one 5-turn greedy conversation with the flag off and on and prints, per turn, the engine's committed decode-token count, that turn's prefill size, and the conversation's `token_count`. Flag off commits reply + stop (3 for "MANGO"); flag on commits 2, the next prefill is identical, and `token_count` drifts by −1 per turn. Measured identically on litert-lm 0.16.0 and 0.16.1 (python API, CPU, macOS).
