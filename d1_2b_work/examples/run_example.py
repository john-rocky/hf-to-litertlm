"""The source model card's examples on this repository's LiteRT files, through the PyTorch-free host.

    python examples/run_example.py                         # CPU (XNNPACK, 8 threads), the single files
    python examples/run_example.py --accel gpu             # the GPU at float32 (Metal on a Mac)
    python examples/run_example.py --check                 # compare with examples/run_example.expected.json
    python examples/run_example.py --form multi8 --accel gpu --check   # the eight-signature file instead

The examples of the LiquidAI/d1-2B model card: a support ticket with four questions (yes/no, a pick, several picks, a
rating), values from an invoice (text, number, date, and a number that is absent), the card's photo with a count, one
box and every cat, and two tickets in one call. Then a state too long for the largest graph, which the host refuses.
The photo (COCO val2017 000000039769.jpg, 640 x 480) is not in this repository: the script downloads it into memory and
checks its SHA-256 first (--image reads a local copy instead).

--check compares every answer with run_example.expected.json, the answers of the provider's own code in float32 on the
CPU: probabilities, P(found) and P(yes) within 1e-4, box edges within 0.002 (two of the model's 0.001 steps), box
confidence within 0.02, and everything else equal (the picked options, texts, spans, numbers, dates, counts and the
token counts); and the refusal. It prints CHECK PASS or CHECK FAIL and exits 1 on a failure.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import sys
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "host"))

PHOTO_URL = "http://images.cocodataset.org/val2017/000000039769.jpg"
PHOTO_SHA256 = "dea9e7ef97386345f7cff32f9055da4982da5471c48d575146c796ab4563b04e"

QUESTIONS = {
    "refund": {"type": "noul", "instructions": "Is the customer asking for a refund?"},
    "team": {"type": "choice", "instructions": "Which team should handle this?",
             "criteria": {"billing": "Charges, refunds, invoices", "shipping": "Delivery and parcels",
                          "technical": "App or site faults"}},
    "topics": {"type": "choice", "multiple": True, "instructions": "Which topics does the message raise?",
               "criteria": {"billing": "Payments and charges", "delivery": "Shipping and parcels",
                            "login": "Signing in to the account"}},
    "urgency": {"type": "score", "instructions": "How urgent is this ticket?", "criteria": ["low", "medium", "high"]},
}
TICKET = "I was charged twice and my parcel never arrived. Please refund one charge."
INVOICE = "ACME SA\nInvoice 2026-117\nTotal due: 1 234,50 EUR\nPay before 03/04/2026 (day/month/year)"
INVOICE_QUESTIONS = {
    "payee": {"type": "text", "instructions": "Who issued the invoice?"},
    "total": {"type": "number", "instructions": "How much is due?"},
    "due": {"type": "date", "instructions": "By when must it be paid?"},
    "vat": {"type": "number", "instructions": "How much VAT is charged?"},
}
PHOTO_QUESTIONS = {
    "cats": {"type": "number", "instructions": "How many cats are there?", "range": [0, 10], "step": 1},
    "remote": {"type": "box", "instructions": "Where is the remote control on the left?"},
    "every_cat": {"type": "boxes", "instructions": "Every cat.", "max": 4},
}
TICKETS = ["Where is my parcel? It was due Monday.", "The app crashes when I open settings."]
LONG_STATE = "The parcel left the depot on time and reached the sorting centre a day later. " * 600

EDGES = ("top", "bottom", "left", "right")


def photo(path: str | None):
    from PIL import Image

    data = open(path, "rb").read() if path else urllib.request.urlopen(PHOTO_URL, timeout=60).read()
    digest = hashlib.sha256(data).hexdigest()
    if digest != PHOTO_SHA256:
        raise SystemExit(f"the photo's SHA-256 is {digest}, not {PHOTO_SHA256}")
    return Image.open(io.BytesIO(data))


def compare(ref, got, path, out, qtype=""):
    """Append to `out` every difference outside the tolerances of the module docstring. `qtype` is the answer's type:
    the box confidence (a fraction of 512 sample boxes) is compared within 0.02, a choice's confidence within 1e-4."""
    key = path.rsplit(".", 1)[-1]
    if isinstance(ref, dict) and isinstance(got, dict):
        qtype = ref.get("type", qtype) if isinstance(ref.get("type"), str) else qtype
        if set(ref) != set(got):
            out.append(f"{path}: keys {sorted(got)} != {sorted(ref)}")
        for k in ref:
            if k in got:
                compare(ref[k], got[k], f"{path}.{k}", out, qtype)
    elif isinstance(ref, list) and isinstance(got, list):
        if len(ref) != len(got):
            out.append(f"{path}: {len(got)} items != {len(ref)}")
        for i, (r, g) in enumerate(zip(ref, got)):
            compare(r, g, f"{path}[{i}]", out, qtype)
    elif isinstance(ref, float) and key not in ("number", "score") and ".interval" not in path:
        tol = 0.002 if key in EDGES else 0.02 if key == "confidence" and qtype in ("box", "boxes") else 1e-4
        if isinstance(got, bool) or not isinstance(got, (int, float)) or not (math.isfinite(got) and abs(got - ref) <= tol):
            out.append(f"{path}: {got!r} vs {ref!r} (tolerance {tol})")
    elif ref != got:
        out.append(f"{path}: {got!r} != {ref!r}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--accel", choices=("cpu", "gpu"), default="cpu")
    ap.add_argument("--form", choices=("files", "multi8"), default="files")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--weight-cache", default="", help="the XNNPACK weight cache file of --form multi8 on the CPU")
    ap.add_argument("--image", default="", help="a local copy of the card's photo")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--out", default="", help="write the answers as JSON")
    a = ap.parse_args()
    from d1_litert import load_host

    t0 = time.time()
    results, refusal = {}, None
    with load_host(ROOT, accelerator=a.accel, form=a.form, threads=a.threads,
                   weight_cache=a.weight_cache or None) as host:
        for name, call in (("ticket", lambda: host.system_one(TICKET, QUESTIONS)),
                           ("invoice", lambda: host.system_one(INVOICE, INVOICE_QUESTIONS)),
                           ("photo", lambda: host.system_one(None, PHOTO_QUESTIONS, media=[photo(a.image or None)])),
                           ("tickets", lambda: host.system_one_batch([(t, {"team": QUESTIONS["team"]})
                                                                       for t in TICKETS]))):
            t = time.time()
            results[name] = call()
            print(f"{name}: {time.time() - t:.2f} s (compiles included)")
        try:
            host.system_one(LONG_STATE, {"refund": QUESTIONS["refund"]})
            refusal = {"refused": False}
        except ValueError as e:
            refusal = {"refused": True, "type": "ValueError", "message": str(e)}
    wall = time.time() - t0

    a_t = results["ticket"]["answers"]
    print("ticket:", round(a_t["refund"]["noul"], 2), a_t["team"]["choice"], a_t["topics"]["choices"],
          a_t["urgency"]["legend"][str(int(a_t["urgency"]["score"]))])
    a_i = results["invoice"]["answers"]
    print("invoice:", a_i["payee"]["text"], a_i["total"]["number"], a_i["due"]["date"], a_i["vat"]["number"])
    a_p = results["photo"]["answers"]
    print("photo:", a_p["cats"]["number"], a_p["every_cat"]["count"],
          {k: round(v, 2) for k, v in a_p["remote"]["box"].items()})
    print("tickets:", [r["answers"]["team"]["choice"] for r in results["tickets"]])
    print("refusal:", refusal.get("message", refusal))
    print(f"wall {wall:.1f} s, accel {a.accel}, form {a.form}")
    if a.out:
        with open(a.out, "w") as f:
            json.dump({"accel": a.accel, "form": a.form, "results": results, "refusal": refusal}, f, indent=1)
    if a.check:
        exp = json.load(open(os.path.join(ROOT, "examples", "run_example.expected.json")))
        diffs = []
        for name in ("ticket", "invoice", "photo"):
            compare(exp["examples"][name]["answers"], results[name]["answers"], name, diffs)
            compare(exp["examples"][name]["usage"], results[name]["usage"], f"{name}.usage", diffs)
        for i, r in enumerate(results["tickets"]):
            compare(exp["examples"]["tickets"][i]["answers"], r["answers"], f"tickets[{i}]", diffs)
            compare(exp["examples"]["tickets"][i]["usage"], r["usage"], f"tickets[{i}].usage", diffs)
        if not refusal.get("refused") or exp["refusal"]["contains"] not in refusal.get("message", ""):
            diffs.append(f"refusal: {refusal}")
        print("\n".join(diffs) if diffs else "", end="\n" if diffs else "")
        print("CHECK PASS" if not diffs else f"CHECK FAIL ({len(diffs)} differences)")
        sys.exit(1 if diffs else 0)


if __name__ == "__main__":
    main()
