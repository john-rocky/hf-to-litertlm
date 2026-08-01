"""Logit-parity check for the Nanbeige 4.2 looped-export port (convert_nanbeige42.py).

Loads ONE fp32 model instance and compares, on the same weights:
  A) reference: the ORIGINAL HF modeling forward (eager attention, use_cache=False,
     the model's own loop/cache-free path) — ground truth;
  B) port one-shot: the patched export forward over a zeroed 44-slot LiteRT static
     cache, full prompt in one pass (validates loop unroll + rope + transposed attn
     + within-chunk causal attention over the DUS'd cache);
  C) port prefill+decode: prompt[:-1] prefilled, last token decoded (validates
     cross-call cache persistence exactly as the runtime drives it).

Run from the repo root (the converter's template default resolves relative to CWD):
    python scripts/parity_nanbeige42.py

Env: NB42_MODEL (default Nanbeige/Nanbeige4.2-3B), NB42_CACHE_LEN (default 512),
     NB42_PROMPT to override the probe text.
"""

import os
import sys

import torch

MODEL_ID = os.environ.get("NB42_MODEL", "Nanbeige/Nanbeige4.2-3B")
CACHE_LEN = int(os.environ.get("NB42_CACHE_LEN", "512"))
PROMPT = os.environ.get(
    "NB42_PROMPT",
    "Natalia sold clips to 48 of her friends in April, and then she sold half as"
    " many clips in May. How many clips did Natalia sell altogether?",
)
MASK_NEG = -1e9


def build_mask(t_rows, first_row_pos, cache_len):
  """Runtime-style additive mask [1, 1, t_rows, cache_len]: row r (absolute position
  first_row_pos + r) may attend cache slots 0 .. first_row_pos + r."""
  mask = torch.full((1, 1, t_rows, cache_len), MASK_NEG, dtype=torch.float32)
  for r in range(t_rows):
    mask[0, 0, r, : first_row_pos + r + 1] = 0.0
  return mask


def main():
  import transformers

  # Install the export patches BEFORE loading: the rope fix must be active during
  # from_pretrained (transformers 5.x zeroes the init-computed inv_freq buffer on
  # the meta device — with the fix, rope is recomputed on the fly, so BOTH the
  # original forward (our reference) and the port see correct positions).
  sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
  saved_argv, sys.argv = sys.argv, [sys.argv[0]]  # converter reads argv at import
  import convert_nanbeige42 as nb42conv
  sys.argv = saved_argv
  nb42conv.install_nanbeige_loop_patches(MODEL_ID)

  print(f"Loading {MODEL_ID} fp32 (eager attention)...")
  tok = transformers.AutoTokenizer.from_pretrained(MODEL_ID)
  cfg = transformers.AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
  model = transformers.AutoModelForCausalLM.from_pretrained(
      MODEL_ID,
      config=cfg,
      torch_dtype=torch.float32,
      trust_remote_code=True,
      attn_implementation="eager",
  )
  model.eval()

  ids = tok(PROMPT, return_tensors="pt").input_ids
  t = ids.shape[1]
  print(f"prompt tokens: {t}, cache_len: {CACHE_LEN}")
  assert t + 1 < CACHE_LEN, "prompt longer than parity cache"

  # --- A) reference: ORIGINAL modeling forward (saved by the installer), no cache ---
  base_cls = type(model).__mro__[1]
  with torch.no_grad():
    ref = base_cls._nb42_original_forward(
        model, input_ids=ids, use_cache=False
    ).logits  # [1, T, V]

  from litert_torch.generative.export_hf.core import cache as lrt_cache

  n_slots = model.config.num_hidden_layers * nb42conv._nb_num_loops(model.config)
  n_kv = model.config.num_key_value_heads
  head_dim = model.config.head_dim

  def fresh_cache():
    layers = [
        lrt_cache.LiteRTLMCacheLayer(
            key_cache=torch.zeros((1, n_kv, CACHE_LEN, head_dim)),
            value_cache=torch.zeros((1, n_kv, head_dim, CACHE_LEN)),
            k_ts_idx=2,
            v_ts_idx=3,
        )
        for _ in range(n_slots)
    ]
    return lrt_cache.LiteRTLMCache(layers)

  def port_forward(kv, token_ids, first_pos):
    n = token_ids.shape[1]
    pos = torch.arange(first_pos, first_pos + n, dtype=torch.int32)
    kv.set_cache_runtime_args({"cache_position": pos})
    with torch.no_grad():
      out = model(
          input_ids=token_ids.to(torch.int32),
          position_ids=pos.unsqueeze(0),
          past_key_values=kv,
          attention_mask=build_mask(n, first_pos, CACHE_LEN),
          k_ts_idx=2,
          v_ts_idx=3,
      )
    return out.logits

  def report(name, got, want):
    d = (got - want).abs()
    top1_g = got.argmax(-1).item()
    top1_w = want.argmax(-1).item()
    top20_g = set(got.topk(20).indices.tolist())
    top20_w = set(want.topk(20).indices.tolist())
    ok = top1_g == top1_w
    print(f"[{name}] max|dlogit|={d.max().item():.4e}  top1 {top1_g}"
          f"{'==' if ok else '!='}{top1_w} ({tok.decode([top1_g])!r})  "
          f"top20 overlap {len(top20_g & top20_w)}/20")
    return ok, d.max().item()

  # --- B) one-shot ---
  logits_b = port_forward(fresh_cache(), ids, 0)
  full_d = (logits_b - ref).abs().max().item()
  print(f"[one-shot] full-sequence max|dlogit| over all {t} positions: {full_d:.4e}")
  ok_b, _ = report("one-shot last", logits_b[0, -1], ref[0, -1])

  # --- C) prefill + decode ---
  kv = fresh_cache()
  port_forward(kv, ids[:, :-1], 0)
  logits_c = port_forward(kv, ids[:, -1:], t - 1)
  ok_c, _ = report("prefill+decode", logits_c[0, -1], ref[0, -1])

  passed = ok_b and ok_c and full_d < 5e-2
  print("PARITY:", "PASS" if passed else "FAIL")
  return 0 if passed else 1


if __name__ == "__main__":
  sys.exit(main())
