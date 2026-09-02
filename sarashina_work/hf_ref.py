#!/usr/bin/env python3
"""bf16 PyTorch reference for the sarashina2.2 8Q gates (EN + JA), greedy, with a
BOS A/B column.

The bundle carries no start_token (NO_START_TOKEN, matching add_bos_token=false
and the official template). This script feeds the SAME rendered prompt to the
bf16 model (a) exactly as apply_chat_template renders it and (b) with a leading
<s>, so the recipe decision rests on a measurement rather than on the
tokenizer_config flag ([[bundle-prepends-bos-add-bos-false]]: eligibility is
not damage — sweep per model).

    ~/venvs/ltconv040dev/bin/python sarashina_work/hf_ref.py <hf_model_or_dir> <out.json>
"""
import json
import re
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import verify_quality as vq  # noqa: E402
import verify_quality_ja as vja  # noqa: E402

model_id, out_path = sys.argv[1], sys.argv[2]
dev = "mps" if torch.backends.mps.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16).to(dev).eval()
print(f"loaded {model_id} on {dev}; add_bos_token={getattr(tok, 'add_bos_token', None)} "
      f"bos={tok.bos_token!r}({tok.bos_token_id}) eos={tok.eos_token!r}({tok.eos_token_id})")


def gen(prompt_text, bos, max_new=160):
    ids = tok(prompt_text, add_special_tokens=False, return_tensors="pt")["input_ids"]
    if bos:
        ids = torch.cat([torch.tensor([[tok.bos_token_id]]), ids], dim=1)
    ids = ids.to(dev)
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=max_new, do_sample=False,
                             eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id)
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)


report = {"model": model_id, "device": dev, "sets": {}}
for set_name, questions, suffix in (("en", vq.QUESTIONS, vq.SUFFIX),
                                     ("ja", vja.QUESTIONS, vja.SUFFIX)):
    rows = []
    for label, q, pat in questions:
        rendered = tok.apply_chat_template([{"role": "user", "content": q + suffix}],
                                           tokenize=False, add_generation_prompt=True)
        row = {"label": label}
        for col, bos in (("official", False), ("with_bos", True)):
            t0 = time.time()
            ans = gen(rendered, bos)
            row[col] = {"ok": bool(re.search(pat, ans.lower())), "answer": ans,
                        "s": round(time.time() - t0, 1)}
        rows.append(row)
        print(f"  [{set_name}] {label:16s} official={'✓' if row['official']['ok'] else '·'} "
              f"{' '.join(row['official']['answer'].split())[:60]!r}  | +BOS="
              f"{'✓' if row['with_bos']['ok'] else '·'} "
              f"{' '.join(row['with_bos']['answer'].split())[:60]!r}", flush=True)
    report["sets"][set_name] = {
        "official": sum(r["official"]["ok"] for r in rows),
        "with_bos": sum(r["with_bos"]["ok"] for r in rows),
        "rows": rows,
    }
    print(f"== {set_name}: official {report['sets'][set_name]['official']}/8, "
          f"+BOS {report['sets'][set_name]['with_bos']}/8", flush=True)

# streaming probe text for reference (what the model itself writes at bf16)
rendered = tok.apply_chat_template([{"role": "user", "content": vja.PROBE[1] + vja.SUFFIX}],
                                   tokenize=False, add_generation_prompt=True)
report["utf8_probe_official"] = gen(rendered, False)
print("probe:", repr(report["utf8_probe_official"]))
Path(out_path).write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
print("wrote", out_path)
