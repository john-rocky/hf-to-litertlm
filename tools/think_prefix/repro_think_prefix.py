#!/usr/bin/env python3
"""Show what litert-torch's structured-template extraction writes as the assistant
prefix, next to the generation prompt the same chat template renders.

`parse_chat_template` (litert_torch/generative/export_hf/core/litert_lm_builder.py)
derives `prompt_templates.model.prefix` from an assistant HISTORY turn
(`add_generation_prompt=False`, `enable_thinking=False`). LiteRT-LM sends that
prefix as the generation prompt. This script prints both strings for each
tokenizer so the two can be compared byte for byte.

    python repro_think_prefix.py Qwen/Qwen3-4B-Thinking-2507 ibm-granite/granite-4.2-3b
    python repro_think_prefix.py --template my_template.jinja Qwen/Qwen3-4B-Thinking-2507

Needs: transformers, litert-torch (0.9.3 / 0.9.4 / main behave the same).
"""
import argparse
import importlib.metadata as md

from transformers import AutoTokenizer
from litert_torch.generative.export_hf.core.litert_lm_builder import parse_chat_template


def generation_prompt(tok, **kw):
  msgs = [{"role": "user", "content": "Q"}]
  a = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False, **kw)
  b = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **kw)
  assert b.startswith(a)
  return b[len(a):]


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("repos", nargs="+", help="HF repo ids or local tokenizer dirs")
  ap.add_argument("--template", help="override the chat template with this jinja file")
  args = ap.parse_args()
  print("litert-torch", md.version("litert-torch"), "| transformers", md.version("transformers"))
  for repo in args.repos:
    tok = AutoTokenizer.from_pretrained(repo)
    if args.template:
      tok.chat_template = open(args.template).read()
    gen = generation_prompt(tok)
    parts = parse_chat_template(tok)
    prefix = parts[2][0] if parts else None
    print(f"\n[{repo}{' + ' + args.template if args.template else ''}]")
    print(f"  vendor generation prompt (add_generation_prompt=True): {gen!r}")
    print(f"  extracted prompt_templates.model.prefix:               {prefix!r}")
    print("  MATCH" if gen == prefix else "  MISMATCH")


if __name__ == "__main__":
  main()
