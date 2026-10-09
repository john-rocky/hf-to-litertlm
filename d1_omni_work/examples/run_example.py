"""Example: the three requests of the model card's Quick start — a text request with two questions, a photo and a
voice note — answered through host/d1_omni.py on the CPU (or the GPU with --gpu).

Run with the downloaded model repository (contract.json, the .tflite files, host/ and fixtures/):
    pip install -r <repo>/host/requirements.txt
    python examples/run_example.py --repo <repo>            # prints the three responses
    python examples/run_example.py --repo <repo> --check    # also compares them with run_example.expected.json
    python examples/run_example.py --repo <repo> --gpu      # the GPU, each graph at its precision from contract.json
(--repo defaults to this folder's parent, so the script also runs from inside the repository's own examples/.)

The expected file holds the CPU responses. --check passes when every answer (choice, label or level) and every
usage count is the same and every probability is within 1e-4 of the expected one (two CPU runs printed identical
values). With --gpu the tolerance is 0.02, the per-probability bar the shipped graphs were checked against: the
audio graph runs at the GPU's default precision, which moved this voice note's probability by 1.3e-3 on an Apple
M4 Max, while the text and photo answers stayed within 1e-5 of the CPU.
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TOL = 1e-4
TOL_GPU = 0.02
TEXT_STATE = "I was charged twice this month, please refund one of them."
TEXT_QUESTIONS = {
    "refund": {"type": "noul", "instructions": "Is the customer asking for a refund?"},
    "team": {"type": "choice", "instructions": "Which team should handle this?",
             "criteria": {"billing": "Charges, refunds, invoices", "technical": "App or site faults",
                          "fraud": "Suspected unauthorised use"}},
}
IMAGE_QUESTIONS = {"count": {"type": "choice", "instructions": "How many dogs are in the photo?",
                             "criteria": {"one": "One", "two": "Two", "more": "Three or more"}}}
AUDIO_STATE = "Voice note from a user."
AUDIO_QUESTIONS = {"wants": {"type": "noul", "instructions": "Is the speaker asking for something to be done?"}}


def requests(repo):
    return [("text", TEXT_STATE, TEXT_QUESTIONS, {}),
            ("image", None, IMAGE_QUESTIONS, {"images": [repo / "fixtures/media/img_dogs_01.png"]}),
            ("audio", AUDIO_STATE, AUDIO_QUESTIONS, {"audio": repo / "fixtures/media/aud_01.wav"})]


def close(a, b, tol):
    """Same structure, same strings and integers, floats within tol."""
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(close(a[k], b[k], tol) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(close(x, y, tol) for x, y in zip(a, b))
    if isinstance(a, float) or isinstance(b, float):
        return isinstance(a, (int, float)) and isinstance(b, (int, float)) and abs(a - b) <= tol
    return a == b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="compare with examples/run_example.expected.json")
    ap.add_argument("--gpu", action="store_true", help="the GPU, each graph at its precision from contract.json")
    ap.add_argument("--repo", default=str(HERE.parent), help="the model repository (default: this folder's parent)")
    ap.add_argument("--write-expected", action="store_true", help="write the responses to run_example.expected.json")
    a = ap.parse_args()
    repo = Path(a.repo).resolve()
    sys.path.insert(0, str(repo / "host"))
    from d1_omni import D1Omni

    out = {}
    with D1Omni(repo, accelerator="gpu" if a.gpu else "cpu") as model:
        for name, state, questions, media in requests(repo):
            out[name] = model.system_one(state, questions, **media)
    print(json.dumps(out, indent=2, ensure_ascii=False))
    if a.write_expected:
        (HERE / "run_example.expected.json").write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n",
                                                        encoding="utf-8")
    if a.check:
        expected = json.loads((HERE / "run_example.expected.json").read_text(encoding="utf-8"))
        ok = close(out, expected, TOL_GPU if a.gpu else TOL)
        print("matches run_example.expected.json" if ok else "DIFFERS from run_example.expected.json")
        if not ok:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
