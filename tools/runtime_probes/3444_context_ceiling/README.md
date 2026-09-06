# LiteRT-LM #3444 — context ceiling re-check on the PyPI runtime

Issue: https://github.com/google-ai-edge/LiteRT-LM/issues/3444 (gemma-4-E2B: `maxNumTokens: 4096` accepted at init, conversation dies near `getTokenCount()` ~2000 on the CLiteRTLM_mac xcframework, v0.16.0).

This probe opens the same bundle (`litert-community/gemma-4-E2B-it-litert-lm/gemma-4-E2B-it.litertlm`) through the PyPI `litert-lm` Python API with `max_num_tokens=4096`, sends five ~785-token user turns with 4-token replies (greedy), and prints `token_count` after each turn.

Result (2026-09-06, macOS arm64, M4 Max):

| runtime | backend | turns 1–5 | turn 6 |
|---|---|---|---|
| litert-lm 0.16.1 | CPU | OK, token_count 786 → 3926 | `FAILED_PRECONDITION: Prefill input length exceeds available state entries (remaining capacity: 170)` |
| litert-lm 0.17.0 | CPU | same | same |
| litert-lm 0.16.1 | GPU (WebGPU) | same | same |
| litert-lm 0.17.0 | GPU (WebGPU) | same | same |

So on the PyPI runtime the bundle serves the full 4096 budget and fails only at the real wall (3926 + 785 > 4096). The ~2048 ceiling reported in the issue was measured on the Swift/Metal xcframework path, which had no 0.17.0 build at the time of this run.

Files: `probe_3444_ceiling.py` (the script, ~30 lines), `logs/` (full stdout/stderr of the four runs).
