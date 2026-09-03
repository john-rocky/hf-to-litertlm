# P1 design — Qwen3.5 ring + verify re-export (frozen 2026-08-31, sources read at LiteRT-LM 90f42140 / litert-torch 115a136)

Everything below was derived from source, not the handoff doc. Line refs: `llm_litert_mtp_drafter.cc`,
`llm_litert_compiled_model_executor.cc` (executor), `litert/state.cc`, all at FETCH_HEAD 90f42140.

## Runtime round flow (exact, verified)

- `current_step` invariant = number of ACCOUNTED tokens (processed + 1 pending). Decode runs at
  `step = current_step - 1` (DecodeInternal).
- **Round 1 of a decode turn** (`ran_decode == false`): one normal decode at slot `s` (consumes pending
  token `t_s`, writes KV/state slot `s`, outputs `logits` + `activations` = post-final-norm `h_s`);
  token_id `t_{s+1}` sampled host-side; then `Draft(position = s+1, token_id = t_{s+1}, activations = h_s)`.
- **Rounds 2+**: NO decode. `Draft(position = p, token_id = pending bonus t_p, activations = nullopt)` —
  drafter seed comes from the PREVIOUS verify's `activations[last_verified_token_id_idx_]`.
- Inside Draft(p, t_p): drafter `input_pos = p-1` fixed for all G chained steps (mask filled by runtime =
  attend cache slots 0..p-1 — but the correct teacher range is ≤ p-2, see below: drafter graph must build
  its own mask from input_pos and NOT declare a mask input). Drafter kv inputs = Duplicate()d handles of
  the BASE state buffers (same memory), names starting `kv_cache_`; drafter outputs get fresh buffers
  (drafter writes never land in state — chained-step self k/v is in-graph only).
- Verify: `input_pos = [p..p+G]` rank-1 int32, embeddings input = table rows of `[t_p, draft_1..draft_G]`
  (host LookupPrefill), mask = runtime-filled standard causal (start p, steps G+1), outputs `logits`
  [1,G+1,V] + `activations` [1,G+1,H] fp32 (required, fp32-checked at Create). Greedy argmax per position;
  accepted c = longest prefix match; bonus = argmax at index c; `current_step += c+1`. Verify writes cache
  slots p..p+G; slots p+c+1..p+G are garbage and are overwritten by the NEXT round's verify (p' = p+c+1,
  writes p'..p'+G ⊇ garbage) before any read.

## Ring rule for the 18 gated-delta layers (and why R = G+2)

State after position q lives at ring slot `q mod R`. Every signature reads slot `(input_pos[0]-1) mod R`
(fresh start: (-1) mod R = R-1, engine-zeroed ✓ torch.remainder gives R-1) and writes:
- prefill: final chunk state at `(last_valid_pos) mod R` (last_valid = input_pos[0]+V-1, V from pad guard);
- decode: slot `input_pos[0] mod R`;
- verify: per-position states at `(input_pos[i]) mod R` for all G+1 positions (needs chunk_size=1 kernel
  variant that returns stacked per-position states; modeling passes `last_recurrent_state` STRAIGHT to
  `update_recurrent_state`, so a [T,H,Dk,Dv] return flows into the cache layer unmodified).
Read slot p-1 is never clobbered by the verify write range [p..p+G] iff R ≥ G+2 (p-1 ≡ p+G+1 mod G+2).
Every read targets the immediately-preceding call's last written slot — stale intermediate slots are never
read. Same rule for the conv window ring (kernel K=4; per-position windows in verify = slices [j+1..j+K]
of [cached_window | new_cols]).
Buffer shapes: `kv_cache_lr_i` [R, 16, 128, 128], `kv_cache_lc_i` [R, 6144, 4], ring slice-read returns
[1,...] = the exact shape the modeling already consumes. R=5 (G=3): ~95 MB fp32 for 0.8B/2B.

## Drafter teacher-forcing (oracle `fixed` mode, measured E[G=3]=2.95/3.06)

Slot convention (= P0 oracle `teacher_and_step1`): **drafter KV slot q holds k/v of
x_q = fc(cat(ne(emb(t_{q+1})), nh(h_q))) at RoPE position q.** Query at round p = x_{p-1} (host concat
gives emb(t_p)‖h_{p-1}), rope p-1 = runtime input_pos ✓; attends teacher slots ≤ p-2 + own in-graph k/v
(slot p-1 in the cache is garbage at drafter time — its true pair needs the bonus token; it becomes
correct when the round's verify runs).
- k = rope(k_norm(k_proj(input_layernorm(x))), pos), v = v_proj(input_layernorm(x)); weights from the 15
  `mtp.*` tensors (transformers discards them on load — load from safetensors directly).
- Base graphs write, for each input position q, drafter slot q-1 = pair (emb-input row q, h_{q-1}):
  prefill block DUS at input_pos[0] for slots [a..b-1] (pairs 1..T-1) + one single-row DUS at
  `(a-1) mod cache_len` (pair 0, h from hidden ring) — mod so the a=0 phantom lands at slot 4095,
  never attended; decode: single row at `(s-1) mod L`; verify: one block DUS at p-1 (p ≥ 1 always).
- Pad-tail pairs in prefill write garbage at slots ≥ last_valid; provably overwritten before any drafter
  read (drafter attend limit p-2 never passes the verified-correct frontier). No masking needed.
- Buffers: `kv_cache_k_24` / `kv_cache_v_24`, base-attention layout (k seq axis 2, v seq axis 3),
  full cache_length 4096, 2 kv heads × 256 head dim. Names start with `kv_cache_` (drafter Duplicate
  lookup) and contain `kv` (verify state-bind skip list) — both runtime requirements.

## Hidden ring

`kv_cache_h_0` [R, 1, 1024] fp32: h_q (post-final-norm) at slot q mod R. Read (input_pos[0]-1) mod R
(first pair + nothing else); write: prefill = last-valid h only; decode = h_s; verify = all G+1 h's
(accept point c is unknown at graph time). This is the state the P0 doc didn't spell out; forced by
chunk boundaries + verify's first pair.

## Export decisions

- ALL signatures embeddings-input (external_emb modules) + EMBEDDER section: verify ≡ decode bit-exact
  requires both to read the same table (in-graph int8 table vs embedder-section int8 table would break
  greedy equivalence). Pad-guard valid falls back to position monotonicity (already in the patch, proven
  LFM2.5-VL). `externalize_embedder=True` in export config; embedder quantized post-hoc with the bundle.
- verify = prefill-shaped T=G+1=4 module, embeddings in, logits+activations (fp32, post-final-norm,
  computed as text_model → norm → lm_head in-module) out. Signature literally "verify";
  num_draft_steps inferred by runtime from input_pos dim = G+1.
- decode adds `activations` output (RET_CHECK'd by runtime on MTP path; harmless extra output otherwise).
- Runtime name constants: tokens/input_pos/mask("mask" accepted)/embeddings/logits — all matched by our
  existing conventions. `activations`/`projected_activations` looked up literally.
- ExecutorMetadata: extend local script classifier for kinds `h` (TYPE_LINEAR_ATTENTION); `k_24`/`v_24`
  classify as global KV automatically. lr/lc stay TYPE_LINEAR_ATTENTION (opaque, shapes free).
- G=3 → R=5 first; 0.8B then 2B. Dev exports on short ladder; ship exports full ladder.

## Gates (P1 kill-gates)

A. Interpreter-level (ai_edge_litert, float tflite, no runtime): simulate the EXACT runtime call sequence
   (prefill chunks incl. partial, decode, verify rounds with forced rejections) vs patched-HF reference:
   teacher-forced logits + ring state contents + drafter-KV slots. The rewind gate: force wrong drafts,
   check CONTENT of every state after the next round (gates-can-lie: content, not crash-free).
B. `scripts/verify_quality.py` on the packed bundle (litert-lm CPU run) — normal path E2E parity.
C. Hermetic multi-turn (banana sweep pattern) on the bundle.
