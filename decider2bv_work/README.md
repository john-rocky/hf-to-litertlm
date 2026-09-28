# decider-2b-vision (Mapika) → LiteRT-LM — a vision System One model with the upstream M-RoPE positions derived inside the graph

Published: [litert-community/decider-2b-vision-LiteRT](https://huggingface.co/litert-community/decider-2b-vision-LiteRT) — `decider-2b-vision_fp16.litertlm` (desktop, closest to upstream), `decider-2b-vision_fp16-int8vocab.litertlm` (desktop GPU at half the memory) and `decider-2b-vision_int8.litertlm` (phones; Galaxy S26 CPU and GPU). The model reads letter logits at one answer slot per question from one image plus lettered options and turns them into option probabilities; nothing is generated.

- [REPRODUCE.md](REPRODUCE.md) — the conversion by round: pinned environments (`requirements-lock-*.txt`), the checkpoint download, the upstream fp32 oracle, the vision tower at 256×256, the decoder export with the derived M-RoPE rotary (`scripts/mrope_derived.py`, `scripts/export_decoder_g256.py`), the weight forms, the bundle, the graph readout and the LiteRT-LM runtime checks.
- [FINDINGS.md](FINDINGS.md) — what was measured and the traps met on the way.
- [deps/](deps/) — the exact litert-torch exporter patch this decoder used (`qwen35_hybrid_litert_torch.patch`, applied on the fork commit `115a13607c730c81018bb9789138a3e5e5119e3d` of `john-rocky/litert-torch`) and the identity-template helper.
- [reference/](reference/) — the torch-free readout (`decider_litert.py`, ai-edge-litert + numpy + Pillow + tokenizers), the runtime example, the vendored upstream prompt code and the 42 public fixture rows with the upstream fp32 letter logits; the same files ship in the model repository.
- [scripts/](scripts/) — every script the rounds ran, in the order REPRODUCE.md gives.

The section in the top-level [REPRODUCE.md](../REPRODUCE.md) summarises the findings that generalise (the 1-D position cost, the RELU_0_TO_1 refusal on GPU delegates, the fixed-input cost, the weight forms on a 12 GB phone).
