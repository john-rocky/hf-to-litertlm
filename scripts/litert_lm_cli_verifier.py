#!/usr/bin/env python3
"""litert-mac-verify's command line on the litert-lm CLI, for `scripts/verify_quality.py --verifier`.

    litert_lm_cli_verifier.py <model.litertlm> <prompt> [--max-tokens N] [--backend cpu|gpu]

The same contract as litert-mac-verify's --backend path: one engine per call, the KV cache sized to
--max-tokens (max_num_tokens is the whole context in both), greedy sampling, and one line
`OUTPUT: [<answer, newlines as ⏎>]` on stdout, the answer without the CLI's channel blocks (the
thought of a thinking model), as litert-mac-verify's response carried no thought channel either.
verify_quality.py scores that line as before.
The CLI is $LITERT_LM_CLI, else the litert-lm next to this interpreter, else litert-lm on PATH.
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_bundle import strip_channels  # noqa: E402


def main(argv):
    positional, max_tokens, backend = [], 256, "cpu"
    i = 0
    while i < len(argv):
        if argv[i] in ("--max-tokens", "-n") and i + 1 < len(argv):
            max_tokens = int(argv[i + 1]); i += 2
        elif argv[i] == "--backend" and i + 1 < len(argv):
            backend = argv[i + 1]; i += 2
        else:
            positional.append(argv[i]); i += 1
    if len(positional) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    here = Path(sys.executable).parent / "litert-lm"
    cli = (os.environ.get("LITERT_LM_CLI") or (str(here) if here.exists() else None)
           or shutil.which("litert-lm"))
    if not cli:
        print("FAILED: litert-lm CLI not found", file=sys.stderr)
        return 2
    model, prompt = positional
    proc = subprocess.run(
        [cli, "run", model, "--prompt", prompt, "--backend", backend,
         "--max-num-tokens", str(max_tokens), "--top-k", "1", "--cache", "no"],
        capture_output=True, text=True)
    sys.stderr.write(proc.stderr)
    text = strip_channels(proc.stdout).rstrip("\n")
    if proc.returncode != 0 and not text.strip():
        print(f"FAILED: litert-lm exit {proc.returncode}")
        return 1
    # a run that filled the cache still hands back what it generated, as litert-mac-verify does
    print("OUTPUT: [" + text.replace("\n", "⏎") + "]")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
