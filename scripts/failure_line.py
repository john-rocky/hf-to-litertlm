"""The one failure line that scripts/convert.py and scripts/check_bundle.py print:

    FAIL <stage>: <op or first error line> -> next: <one hint>

Stages (docs/one-command-lock.md, "When it fails"): fetch, gate, load, trace, convert, quantize,
bundle, engine-load, generate. The hint comes from scripts/failure_hints.json (the first row whose
stage list contains the stage and whose regex matches the text), else from the caller (a refusal's
own action), else it asks for a model request with the line.
"""
import json
import re
from pathlib import Path

HINTS = Path(__file__).resolve().parent / "failure_hints.json"
REQUEST_HINT = ("open a model request with this line: "
                "https://github.com/john-rocky/hf-to-litertlm/issues/new?template=model-request.yml")

# Most specific first: a converter's own "failed to legalize" names the op it could not lower.
OP_PATTERNS = [
    r"failed to legalize operation '([^']+)'",
    r"[Uu]nsupported op(?:eration)?[:\s]+'?([A-Za-z0-9_.:]+)",
    r"\b(aten\.[A-Za-z0-9_]+(?:\.[A-Za-z0-9_]+)?)",
    r"\b(stablehlo\.[A-Za-z0-9_]+)",
    r"\b(tfl\.[A-Za-z0-9_]+)",
]


def extract_op(text):
    """The op name a traceback or an error message talks about, or None."""
    for pat in OP_PATTERNS:
        m = re.search(pat, text or "")
        if m:
            return m.group(1)
    return None


def first_line(text, limit=200):
    for line in (text or "").splitlines():
        line = line.strip()
        if line:
            return line[:limit]
    return "(no message)"


def hint_for(stage, text):
    """The hint of the first scripts/failure_hints.json row that matches, else None."""
    try:
        rows = json.loads(HINTS.read_text())["hints"]
    except (OSError, ValueError, KeyError):
        return None
    for row in rows:
        if stage in row["stages"] and re.search(row["match"], text or "", flags=re.IGNORECASE):
            return row["hint"]
    return None


def format_line(stage, what, hint):
    what = " ".join(str(what).split())
    return f"FAIL {stage}: {what} -> next: {hint}"


def failure(stage, what, context="", hint=None, **fields):
    """The `failure` dict for a report: stage, what, op, hint, line (+ `fields`, e.g. seconds and
    toolchain). `context` (a traceback, model_type=<type>) is matched against the hints and
    searched for an op, but not printed."""
    text = f"{what}\n{context}"
    op = extract_op(text)
    shown = op or first_line(what)
    hint = hint_for(stage, text) or hint or REQUEST_HINT
    return {"stage": stage, "what": shown, "op": op, "hint": hint,
            "line": format_line(stage, shown, hint), **fields}
