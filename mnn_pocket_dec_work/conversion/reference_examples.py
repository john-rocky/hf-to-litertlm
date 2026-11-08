"""The original inference.py's answers for the example requests (the *.expected.json files of examples/).

    python conversion/reference_examples.py --src <MNN-Pocket-0.3B-Dec download> --examples examples [--write]

Run it in the reference environment (requirements-reference.txt). inference.py is imported from --src, unchanged, and
runs in float32 on the CPU with one torch thread. For each <name>_request.json in --examples, predict() is printed in the
form of the original command line (json.dumps(..., ensure_ascii=False, indent=2, allow_nan=False)). With --write the text
goes to <name>_request.expected.json (an existing file with other text stops the script); without it, the text is
compared with that file and the exit code is 1 on a difference."""
import argparse
import importlib.util
import json
from pathlib import Path

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True, help="the downloaded taobao-mnn/MNN-Pocket-0.3B-Dec directory")
    ap.add_argument("--examples", type=Path, required=True)
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    torch.set_num_threads(1)
    spec = importlib.util.spec_from_file_location("mnn_pocket_dec_inference", a.src / "inference.py")
    inference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(inference)
    model = inference.PocketDecisionModel.from_pretrained(a.src, "cpu", "float32")
    failed = False
    for path in sorted(a.examples.glob("*_request.json")):
        request = json.loads(path.read_text(encoding="utf-8"))
        result = model.predict(request["state"], request["questions"])
        text = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        expected = path.with_suffix(".expected.json")
        if expected.exists():
            same = expected.read_text(encoding="utf-8") == text
            print(f"{path.name}: {result['input_tokens']} tokens, {'same as' if same else 'DIFFERS from'} {expected.name}")
            failed |= not same
        elif a.write:
            expected.write_text(text, encoding="utf-8")
            print(f"{path.name}: {result['input_tokens']} tokens, wrote {expected.name}")
        else:
            print(f"{path.name}: {result['input_tokens']} tokens, no {expected.name}")
            failed = True
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
