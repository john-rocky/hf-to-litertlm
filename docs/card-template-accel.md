# Card template — on-device accelerator section

Template for the accelerator section of the model cards published from this repo.
Rules for filling it:

- **Every number carries its conditions** on the same row or in the line directly under
  the table: device + SoC, runtime version, N, thermal state (+ headroom), and for LLMs
  the prompt length. A number without conditions does not go on a card.
- Numbers come from primary measurement logs only. Never join measurements to a repo by
  model *name* — join on the exact file name (same-name exports from different sources
  differ).
- One verdict per model, from the fixed vocabulary below. Verdicts are spelled out in
  prose on the card; the token goes in the tracking table.
- The recipe itself is never copied onto a card — cards link the canonical docs:
  - NPU: <https://github.com/john-rocky/hf-to-litertlm/blob/main/docs/android-npu.md>
  - GPU: <https://github.com/john-rocky/hf-to-litertlm/blob/main/docs/android-gpu.md>

## Verdict vocabulary (classic models)

`NPU_FASTER(x.xx×)` · `GPU_FASTER(x.xx×)` · `NPU_ONLY` · `GPU_ONLY` ·
`NPU_NOT_COMPILABLE` · `RUNTIME_FAIL(npu|gpu)` · `UNOBTAINABLE`

The multiplier is (slower backend median) / (faster backend median). `RUNTIME_FAIL` names
the side that compiled but failed to run. Report losing verdicts as plainly as winning
ones — GPU-faster rows and does-not-compile rows stay on the card.

## Classic model block (`.tflite`)

```markdown
## On-device performance (NPU / GPU)

| backend | inference (median) | load |
|---|---:|---:|
| Qualcomm NPU (HTP) | {npu_ms} ms | {npu_load_ms} ms |
| GPU | {gpu_ms} ms | {gpu_load_ms} ms |

Measured on {device} ({soc}), LiteRT {litert_ver} + QAIRT {qairt_ver}, N={n} median,
thermal status NONE (headroom {headroom}) on both runs. The NPU row was
{compiled_ahead_of_time_for_{soc} | compiled on the device at first load}.
Verdict: {verdict_prose}.

To run this model on the NPU or GPU in your own app, see the
[NPU recipe](https://github.com/john-rocky/hf-to-litertlm/blob/main/docs/android-npu.md)
and [GPU recipe](https://github.com/john-rocky/hf-to-litertlm/blob/main/docs/android-gpu.md).
```

**Say which compile path the NPU row used.** On the one model measured both ways, a
precompiled (AOT) artifact and an on-device (JIT) compile agreed on inference and on
steady-state load, and differed only on the **first** load — 0.16 s against 11.6 s. They
also differ in what the reader has to build: AOT means the file they download is not the
file you measured. Naming the wrong path is the easiest way to put a number on a card
nobody can reproduce. Check the harness, not the summary table: if the NPU and GPU rows
were benched from two different files, the NPU file is an AOT artifact.

Both load columns are mandatory whenever both backends run — load is a headline
difference between the backends, not a footnote. For a one-sided verdict
(`NPU_ONLY`, `RUNTIME_FAIL`, …) keep the surviving row and state the other side's failure
in the verdict prose, with the error one-liner if there is one.

## LLM block (`.litertlm`)

```markdown
## On-device performance

| backend | prefill (tok/s) | decode (tok/s) | load |
|---|---:|---:|---:|
| CPU | {cpu_prefill} | {cpu_decode} | {cpu_load} s |
| GPU | {gpu_prefill} | {gpu_decode} | {gpu_load} s |
| NPU (JIT) | {npu_prefill} | {npu_decode} | {npu_load} s (compiles on every load) |

Measured on {device} ({soc}), LiteRT-LM {runtime_ver}, prompt {prompt_tok} tokens,
{max_out} output tokens, thermal status NONE.

**Run on Android:** import into Google AI Edge Gallery and tick **GPU** under
"Compatible accelerators" in the import dialog (the default is CPU-only) — see the
[GPU recipe](https://github.com/john-rocky/hf-to-litertlm/blob/main/docs/android-gpu.md).
NPU: [recipe](https://github.com/john-rocky/hf-to-litertlm/blob/main/docs/android-npu.md).
```

Drop rows that were not measured; never fill a row from another model or another export
of the same model. Short-prompt prefill numbers are a floor, not a throughput claim —
if the prompt was under ~100 tokens, say so in the conditions line. A GPU gate failure is
recorded as `GPU_FAIL({reason})` (or `OOM`) in the tracking table, and the card keeps the
CPU row.
