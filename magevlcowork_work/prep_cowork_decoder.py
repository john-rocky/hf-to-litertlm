"""Extract Mage-VL-Cowork's text decoder as a standalone Qwen3ForCausalLM and prove it is the full model's text path.

Copied from magevl_work/prep_magevl_decoder.py and changed for this checkpoint:
  * the text keys are already in Qwen3ForCausalLM's layout (`model.layers.*`, `model.embed_tokens.weight`,
    `model.norm.weight`, `lm_head.weight`), so no prefix is rewritten; `visual.*` is skipped;
  * text_config has no `rope_parameters`: `rope_theta` (5,000,000) is read from the flat key;
  * the lm_head is untied (it differs from embed_tokens), so the standalone model keeps the checkpoint's lm_head;
  * tokenizer / template / generation files are copied from the checkpoint as bytes (sha256 checked), never re-written
    by `save_pretrained` (a tokenizer written by save_pretrained is not the repository's tokenizer.json byte for byte).
    config.json and model.safetensors in the output are the standalone Qwen3 model's own (save_pretrained of the model
    built from text_config).

Proof (fp32, CPU, eager attention, no image): the same prompt (the OSWorld GUI agent's system prompt from the
checkpoint's osworld_eval/mm_agents/mage_gui_agent.py + a `Task: ...` user turn, rendered by the vendor chat template
with the generation prompt) through (a) the full LlavaOnevision2ForConditionalGeneration (trust_remote_code) and (b) the
standalone decoder reloaded from the saved directory: max|dlogit| over every position, argmax agreement, next-token
top-5, and a short greedy continuation of both.

    HF_HUB_OFFLINE=1 HF_MODULES_CACHE=<dir> python -I magevlcowork_work/prep_cowork_decoder.py <checkpoint> OUT \\
        [REPORT.json]
Writes OUT/ (the standalone decoder) and REPORT.json (default: prep_decoder.json next to OUT).
"""
import ast
import gc
import glob
import hashlib
import json
import os
import platform
import sys
import time

import torch
from safetensors import safe_open

if len(sys.argv) not in (3, 4):
  sys.exit(__doc__)
SRC = os.path.abspath(sys.argv[1])
OUT = os.path.abspath(sys.argv[2])
RES = os.path.abspath(sys.argv[3]) if len(sys.argv) > 3 else os.path.join(os.path.dirname(OUT), "prep_decoder.json")
os.makedirs(os.path.dirname(RES), exist_ok=True)
TASK = "Open the Settings app and turn on Wi-Fi."
GEN_TOKENS = 48
COPY_FILES = ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "special_tokens_map.json",
              "added_tokens.json", "chat_template.jinja", "generation_config.json")

res = {"started": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "src": SRC, "out": OUT, "task": TASK}


def log(*a):
  print(*a, flush=True)


def sha256(p):
  h = hashlib.sha256()
  with open(p, "rb") as f:
    for b in iter(lambda: f.read(1 << 24), b""):
      h.update(b)
  return h.hexdigest()


def versions():
  import importlib.metadata as m
  v = {"python": platform.python_version()}
  for p in ("torch", "transformers", "tokenizers", "safetensors", "jinja2", "numpy"):
    try:
      v[p] = m.version(p)
    except Exception:  # noqa: BLE001
      v[p] = None
  return v


def gui_system_prompt():
  """MAGE_GUI_SYSTEM_PROMPT read out of the vendor's agent file (a module-level string literal)."""
  path = os.path.join(SRC, "osworld_eval", "mm_agents", "mage_gui_agent.py")
  tree = ast.parse(open(path).read())
  for node in tree.body:
    if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "MAGE_GUI_SYSTEM_PROMPT" for t in node.targets):
      return ast.literal_eval(node.value)
  raise SystemExit("MAGE_GUI_SYSTEM_PROMPT not found")


def build_config():
  from transformers import Qwen3Config
  tcfg = json.load(open(os.path.join(SRC, "config.json")))["text_config"]
  assert "rope_parameters" not in tcfg and tcfg.get("rope_scaling") is None, "expected a flat rope_theta"
  assert set(tcfg["layer_types"]) == {"full_attention"} and not tcfg.get("use_sliding_window")
  cfg = Qwen3Config(
      vocab_size=tcfg["vocab_size"],
      hidden_size=tcfg["hidden_size"],
      intermediate_size=tcfg["intermediate_size"],
      num_hidden_layers=tcfg["num_hidden_layers"],
      num_attention_heads=tcfg["num_attention_heads"],
      num_key_value_heads=tcfg["num_key_value_heads"],
      head_dim=tcfg["head_dim"],
      hidden_act=tcfg["hidden_act"],
      max_position_embeddings=tcfg["max_position_embeddings"],
      rms_norm_eps=tcfg["rms_norm_eps"],
      rope_theta=float(tcfg["rope_theta"]),
      attention_bias=tcfg["attention_bias"],
      tie_word_embeddings=False,
      attention_dropout=0.0,
      bos_token_id=tcfg["bos_token_id"],
      eos_token_id=tcfg["eos_token_id"],
      use_sliding_window=False,
  )
  cfg.architectures = ["Qwen3ForCausalLM"]
  rp = getattr(cfg, "rope_parameters", None)
  theta = (rp or {}).get("rope_theta", getattr(cfg, "rope_theta", None))
  assert float(theta) == float(tcfg["rope_theta"]), f"rope_theta did not reach the config: {rp}"
  res["config"] = {"rope_parameters": rp, "rope_theta": float(theta), "tie_word_embeddings": cfg.tie_word_embeddings,
                   "layers": cfg.num_hidden_layers, "hidden": cfg.hidden_size, "heads": cfg.num_attention_heads,
                   "kv_heads": cfg.num_key_value_heads, "head_dim": cfg.head_dim, "vocab": cfg.vocab_size}
  return cfg


def text_state_dict():
  idx = json.load(open(os.path.join(SRC, "model.safetensors.index.json")))["weight_map"]
  want = sorted(k for k in idx if not k.startswith("visual."))
  sd = {}
  for shard in sorted(glob.glob(os.path.join(SRC, "model-*.safetensors"))):
    with safe_open(shard, framework="pt") as f:
      for k in f.keys():
        if k.startswith("visual."):
          continue
        assert k.startswith("model.") or k == "lm_head.weight", f"unexpected text key {k}"
        sd[k] = f.get_tensor(k)
  assert sorted(sd) == want, "shards and index disagree"
  dtypes = sorted({str(v.dtype) for v in sd.values()})
  res["checkpoint"] = {"text_tensors": len(sd), "visual_tensors": sum(1 for k in idx if k.startswith("visual.")),
                       "dtypes": dtypes,
                       "lm_head_equals_embed": bool(torch.equal(sd["lm_head.weight"], sd["model.embed_tokens.weight"]))}
  return sd


def extract():
  from transformers import Qwen3ForCausalLM
  cfg = build_config()
  log("building Qwen3ForCausalLM (bf16)...")
  lm = Qwen3ForCausalLM._from_config(cfg, attn_implementation="eager").to(torch.bfloat16).eval()
  sd = text_state_dict()
  missing, unexpected = lm.load_state_dict(sd, strict=False)
  res["strict"] = {"loaded": len(sd), "missing_raw": list(missing), "unexpected_raw": list(unexpected)}
  bad_missing = [m for m in missing if "rotary_emb" not in m and "inv_freq" not in m]
  assert not bad_missing, f"missing: {bad_missing[:8]}"
  assert not unexpected, f"unexpected: {unexpected[:8]}"
  model_keys = set(lm.state_dict())
  res["strict"]["model_state_keys"] = len(model_keys)
  res["strict"]["one_to_one"] = (model_keys == set(sd))
  assert model_keys == set(sd), "state_dict keys != checkpoint text keys"
  for k, v in lm.state_dict().items():
    assert torch.equal(v, sd[k]), f"tensor {k} changed on load"
  log(f"loaded {len(sd)} tensors, strict 1:1 OK (missing raw {list(missing)}, unexpected 0)")
  os.makedirs(OUT, exist_ok=True)
  lm.save_pretrained(OUT, safe_serialization=True)
  copies = {}
  for f in COPY_FILES:
    s = os.path.join(SRC, f)
    d = os.path.join(OUT, f)
    with open(s, "rb") as fi, open(d, "wb") as fo:
      fo.write(fi.read())
    copies[f] = {"sha256": sha256(d), "same_as_src": sha256(d) == sha256(s), "bytes": os.path.getsize(d)}
    assert copies[f]["same_as_src"]
  res["copied_from_hf"] = copies
  res["saved"] = {f: {"bytes": os.path.getsize(os.path.join(OUT, f)), "sha256": sha256(os.path.join(OUT, f))}
                  for f in sorted(os.listdir(OUT)) if f.endswith((".safetensors", "config.json"))}
  del lm, sd
  gc.collect()


def prompt_ids():
  from tokenizers import Tokenizer
  from transformers import AutoTokenizer
  system = gui_system_prompt()
  messages = [{"role": "system", "content": system}, {"role": "user", "content": f"Task: {TASK}"}]
  hf_tok = AutoTokenizer.from_pretrained(SRC)
  text = hf_tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
  tok = Tokenizer.from_file(os.path.join(SRC, "tokenizer.json"))
  ids = tok.encode(text, add_special_tokens=False).ids
  ids_hf = hf_tok(text, add_special_tokens=False).input_ids
  res["prompt"] = {"system_chars": len(system), "system_sha256": hashlib.sha256(system.encode()).hexdigest(),
                   "text_chars": len(text), "tail": text[-80:], "n_ids": len(ids),
                   "ids_tokenizers_equal_autotokenizer": ids == list(ids_hf),
                   "autotokenizer_class": type(hf_tok).__name__}
  assert text.endswith("<|im_start|>assistant\n<think>\n"), repr(text[-60:])
  return tok, torch.tensor([ids])


def greedy(model, ids, n):
  out = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=n, do_sample=False,
                       num_beams=1, pad_token_id=151643)
  return out[0, ids.shape[1]:].tolist()


def main():
  res["versions"] = versions()
  torch.set_grad_enabled(False)
  t0 = time.time()
  extract()
  res["extract_s"] = round(time.time() - t0, 1)
  tok, ids = prompt_ids()

  # (a) full model, fp32, trust_remote_code. transformers 5.x does not read the remote class attribute
  # `_checkpoint_conversion_mapping` (conversion_mapping.get_model_conversion_mapping skips custom code and has no
  # legacy read of the attribute): without key_mapping the 398 text tensors stay randomly initialized (n_missing 398 /
  # n_unexpected 398). The vendor's own mapping is passed as key_mapping.
  from transformers import AutoModelForCausalLM
  key_mapping = {r"^model(?!\.(language_model|visual))": "model.language_model"}
  res["full_key_mapping"] = key_mapping
  t0 = time.time()
  full, info = AutoModelForCausalLM.from_pretrained(SRC, trust_remote_code=True, dtype=torch.float32,
                                                    attn_implementation="eager", output_loading_info=True,
                                                    key_mapping=key_mapping)
  full.eval()
  sd = text_state_dict()
  lmw = full.get_output_embeddings().weight
  emw = full.get_input_embeddings().weight
  res["full_load"] = {
      "class": type(full).__name__,
      "s": round(time.time() - t0, 1),
      "missing": sorted(info.get("missing_keys", []))[:20], "n_missing": len(info.get("missing_keys", [])),
      "unexpected": sorted(info.get("unexpected_keys", []))[:20], "n_unexpected": len(info.get("unexpected_keys", [])),
      "mismatched": [str(x) for x in info.get("mismatched_keys", [])][:20],
      "lm_head_is_embed_tensor": lmw.data_ptr() == emw.data_ptr(),
      "lm_head_equals_ckpt_lm_head": bool(torch.equal(lmw.detach(), sd["lm_head.weight"].float())),
      "embed_equals_ckpt_embed": bool(torch.equal(emw.detach(), sd["model.embed_tokens.weight"].float())),
      "config_tie_word_embeddings": getattr(full.config, "tie_word_embeddings", None),
      "text_config_tie_word_embeddings": getattr(full.config.text_config, "tie_word_embeddings", None),
  }
  log("full load:", json.dumps(res["full_load"]))
  assert res["full_load"]["n_missing"] == 0 and res["full_load"]["n_unexpected"] == 0, "full model did not load 1:1"
  assert res["full_load"]["lm_head_equals_ckpt_lm_head"] and not res["full_load"]["lm_head_is_embed_tensor"]
  # every text tensor of the full model equals the checkpoint
  fsd = full.model.language_model.state_dict()
  n_eq = sum(1 for k, v in fsd.items() if torch.equal(v, sd["model." + k].float()))
  res["full_load"]["language_model_tensors_equal_ckpt"] = f"{n_eq}/{len(fsd)}"
  del sd
  gc.collect()
  t0 = time.time()
  la = full(input_ids=ids, use_cache=False).logits.float()[0]
  res["full_forward_s"] = round(time.time() - t0, 1)
  ga = greedy(full, ids, GEN_TOKENS)
  del full, fsd
  gc.collect()

  # (b) standalone decoder reloaded from disk, fp32
  from transformers import Qwen3ForCausalLM
  t0 = time.time()
  lm = Qwen3ForCausalLM.from_pretrained(OUT, dtype=torch.float32, attn_implementation="eager").eval()
  res["standalone_load_s"] = round(time.time() - t0, 1)
  t0 = time.time()
  lb = lm(input_ids=ids, use_cache=False).logits.float()[0]
  res["standalone_forward_s"] = round(time.time() - t0, 1)
  gb = greedy(lm, ids, GEN_TOKENS)
  del lm
  gc.collect()

  assert torch.isfinite(la).all() and torch.isfinite(lb).all(), "non-finite logits"
  d = (la - lb).abs()
  am_a, am_b = la.argmax(-1), lb.argmax(-1)
  top_a = la[-1].topk(5)
  top_b = lb[-1].topk(5)
  dec = lambda i: tok.decode([int(i)], skip_special_tokens=False)
  res["parity"] = {
      "positions": int(la.shape[0]),
      "max_abs_dlogit_all": float(d.max()),
      "max_abs_dlogit_last": float(d[-1].max()),
      "mean_abs_dlogit_all": float(d.mean()),
      "argmax_equal_positions": f"{int((am_a == am_b).sum())}/{la.shape[0]}",
      "argmax_last": [int(am_a[-1]), int(am_b[-1])],
      "top5_full": [[int(i), dec(i), round(float(v), 4)] for v, i in zip(top_a.values, top_a.indices)],
      "top5_standalone": [[int(i), dec(i), round(float(v), 4)] for v, i in zip(top_b.values, top_b.indices)],
      "greedy_full": tok.decode(ga, skip_special_tokens=False),
      "greedy_standalone": tok.decode(gb, skip_special_tokens=False),
      "greedy_ids_equal": ga == gb,
      "greedy_n": len(ga),
  }
  res["ok"] = (res["parity"]["argmax_equal_positions"] == f"{la.shape[0]}/{la.shape[0]}") and ga == gb
  res["finished"] = time.strftime("%Y-%m-%d %H:%M:%S %Z")
  json.dump(res, open(RES, "w"), indent=1, ensure_ascii=False)
  log("PARITY", json.dumps(res["parity"], ensure_ascii=False))
  log("PREP_DONE ok=" + str(res["ok"]))


if __name__ == "__main__":
  try:
    main()
  except BaseException as e:  # noqa: BLE001
    res["error"] = f"{type(e).__name__}: {e}"
    os.makedirs(os.path.dirname(RES), exist_ok=True)
    json.dump(res, open(RES, "w"), indent=1, ensure_ascii=False, default=str)
    raise
