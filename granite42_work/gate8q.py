#!/usr/bin/env python3
"""8Q gate for granite-4.2-3b — scripts/verify_quality.py with one scoring fix.

granite-4.2 bundles PREFILL the think opener (`<|im_start|>assistant\n<think>\n`
is the generation prompt), so the model's OUTPUT never contains a literal
`<think>` — it emits `reasoning...</think>answer`. verify_quality.strip_think
only strips a *balanced* `<think>...</think>` block, so on such output the
reasoning text would be scored too — and the reasoning routinely contains the
expected string (it repeats "0.9" while comparing 0.9 vs 0.11), which would
fake a pass (a gate must not be able to pass by accident).

This wrapper monkeypatches strip_think to also drop everything up to the LAST
`</think>` when the opener is absent, then delegates to verify_quality.main().
Shared scripts stay untouched.

    ~/venvs/<any py3>/bin/python granite42_work/gate8q.py <model.litertlm> \
        --backend cpu --max-tokens 2048 --json granite42_work/gate8q_int8_cpu.json
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import verify_quality as vq  # noqa: E402


def strip_think_prefilled(text):
  if "<think>" in text and "</think>" not in text:
    return ""  # opened in-output, never closed -> no final answer
  out = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
  # Prefilled opener: no literal <think> in the output, reasoning ends at </think>.
  if "</think>" in out:
    out = out.rsplit("</think>", 1)[-1]
  return out.strip()


vq.strip_think = strip_think_prefilled

if __name__ == "__main__":
  sys.exit(vq.main())
