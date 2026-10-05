"""Example: one support ticket, three questions (choice, noul, score), answered on the CPU.

Run from the repository root:
    pip install -r host/requirements-host.txt
    python examples/run_example.py               # prints the /v1/systemone response
    python examples/run_example.py --check       # also compares it with examples/run_example.expected.json
    python examples/run_example.py --mode row    # only the row graphs (default: auto)

The expected file holds the response without "latency_ms" (a timing, different on every run). The ticket and every
name in it are invented. In the default mode ("auto") the host runs the three questions through the shared-state pair
when its file is present (the state once, then one question step per question: 128 + 3 x 64 = 320 positions, against
3 x 256 = 768 through the row graphs) and through the row graphs otherwise; "row" uses only the row graphs (here L256),
"pair" only the pair.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "host"))

from kev_litert import KevLiteRT  # noqa: E402

REQUEST = {
    "state": "Ticket #48213, opened by Mara Quellen.\n\nHi, I ordered the Thistlebeam desk lamp (order TB-20931) on "
             "September 14 and was charged twice on my card: two identical charges of 64.90 appeared on September 15. "
             "The lamp arrived yesterday and works perfectly, no complaints there. Could you refund the duplicate "
             "charge? I need it sorted before my card statement closes on Friday. No rush on anything else, and thanks "
             "for the lovely lamp.",
    "questions": {
        "team": {"type": "choice", "instructions": "Which team should handle this ticket?",
                 "criteria": {"billing": "Charges, refunds and invoices",
                              "shipping": "Deliveries, tracking and lost parcels",
                              "returns": "Exchanges and sending a product back",
                              "technical": "Product faults and setup help"}},
        "deadline": {"type": "noul", "instructions": "Does the customer ask for action by a specific deadline?",
                     "criteria": {"true": "The ticket names a day or date by which something must happen",
                                  "false": "No deadline is stated"}},
        "mood": {"type": "score", "instructions": "How upset is the customer?", "criteria": ["Calm", "Annoyed", "Angry"]},
    },
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="compare with examples/run_example.expected.json")
    ap.add_argument("--mode", choices=["auto", "row", "pair"], default="auto", help="the host's route (default auto)")
    a = ap.parse_args()
    with KevLiteRT.from_dir(ROOT, mode=a.mode) as kev:
        response = kev.decide(REQUEST)
    print(json.dumps(response, indent=2, ensure_ascii=False))
    if a.check:
        expected = json.loads((ROOT / "examples/run_example.expected.json").read_text(encoding="utf-8"))
        got = {k: v for k, v in response.items() if k != "latency_ms"}
        print("matches run_example.expected.json" if got == expected else
              f"DIFFERS from run_example.expected.json:\n{json.dumps(expected, indent=2, ensure_ascii=False)}")
        if got != expected:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
