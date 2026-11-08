"""Run the example requests through the LiteRT graphs and compare them with the original model's answers.

Run from the repository root:
    pip install -r host/requirements.txt
    python examples/run_examples.py                       # CPU and GPU, prints each result
    python examples/run_examples.py --backend cpu --check # also compares with examples/*.expected.json

readme_request.json is the call in the "How to run" section of the original model card, default_request.json is the
DEFAULT_REQUEST of the original inference.py, and long_request.json is an invented incident log of 1,024 tokens. Each
*.expected.json is what the original inference.py answers for that request (float32, CPU, one thread). --check passes
when every question has the same winner (choice, decision, or the most likely score level) and every probability is
within 0.02 of the expected one; it prints the largest difference.
"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "host"))

from pocket_dec_litert import PocketDecisionLiteRT  # noqa: E402

EXAMPLES = ("readme_request", "default_request", "long_request")
TOLERANCE = 0.02


def winner(answer):
    if answer["type"] == "choice":
        return answer["choice"]
    if answer["type"] == "noul":
        return answer["decision"]
    probabilities = list(answer["probabilities"].values())
    return max(range(len(probabilities)), key=probabilities.__getitem__)


def compare(result, expected):
    """-> (largest probability difference, list of problems)."""
    problems, largest = [], 0.0
    if result["input_tokens"] != expected["input_tokens"]:
        problems.append(f"input_tokens {result['input_tokens']} != {expected['input_tokens']}")
    if list(result["answers"]) != list(expected["answers"]):
        return largest, problems + ["different question ids"]
    for qid, want in expected["answers"].items():
        got = result["answers"][qid]
        if list(got["probabilities"]) != list(want["probabilities"]):
            problems.append(f"{qid}: different option keys")
            continue
        for key, p in want["probabilities"].items():
            largest = max(largest, abs(got["probabilities"][key] - p))
        if winner(got) != winner(want):
            problems.append(f"{qid}: winner {winner(got)!r} != {winner(want)!r}")
    if largest > TOLERANCE:
        problems.append(f"largest probability difference {largest:.4f} > {TOLERANCE}")
    return largest, problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=("cpu", "gpu", "both"), default="both")
    ap.add_argument("--check", action="store_true", help="compare with examples/*.expected.json")
    a = ap.parse_args()
    failed = False
    for backend in (("cpu", "gpu") if a.backend == "both" else (a.backend,)):
        model = PocketDecisionLiteRT.from_pretrained(ROOT, backend=backend)
        for name in EXAMPLES:
            with open(os.path.join(ROOT, "examples", f"{name}.json"), encoding="utf-8") as f:
                request = json.load(f)
            result = model.predict(request["state"], request["questions"])
            print(f"{backend} {name}:", json.dumps(result, ensure_ascii=False, indent=2))
            if a.check:
                with open(os.path.join(ROOT, "examples", f"{name}.expected.json"), encoding="utf-8") as f:
                    expected = json.load(f)
                largest, problems = compare(result, expected)
                print(f"{backend} {name}: {'matches' if not problems else 'DIFFERS'} {name}.expected.json "
                      f"(largest probability difference {largest:.2e})" + "".join(f"\n  {p}" for p in problems))
                failed |= bool(problems)
        del model
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
