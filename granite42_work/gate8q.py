#!/usr/bin/env python3
"""8Q gate for granite-4.2-3b — runs scripts/verify_quality.py.

granite-4.2 bundles PREFILL the think opener (`<|im_start|>assistant\n<think>\n`
is the generation prompt), so the model's OUTPUT never contains a literal
`<think>` — it emits `reasoning...</think>answer`. Scoring that whole output
would score the reasoning too — and the reasoning routinely contains the
expected string (it repeats "0.9" while comparing 0.9 vs 0.11), which would
fake a pass (a gate must not be able to pass by accident).

verify_quality.strip_think drops everything up to the LAST closing tag itself
(this wrapper used to patch that in), so this file only runs verify_quality.main().

    ~/venvs/<any py3>/bin/python granite42_work/gate8q.py <model.litertlm> \
        --backend cpu --max-tokens 2048 --json granite42_work/gate8q_int8_cpu.json
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import verify_quality as vq  # noqa: E402


if __name__ == "__main__":
  sys.exit(vq.main())
