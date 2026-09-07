#!/usr/bin/env python3
"""Re-run of the #3510 probe against `litert-lm serve --api openai` (0.16.1 pip wheel),
plus three extra grammars that test the U+2581 / <0xNN> table hypothesis, and a
max_tokens check. One JSON line per case in the results file."""
import json, sys, time, argparse, requests

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True, help='"/path/to/x.litertlm,gpu,8192"')
ap.add_argument("--cases", default="all")
ap.add_argument("--timeout", type=float, default=900)
ap.add_argument("--out", default="probe_results.jsonl")
ap.add_argument("--url", default="http://127.0.0.1:8093/v1/chat/completions")
ap.add_argument("--max-tokens-field", default="max_tokens")
args = ap.parse_args()

Q = ("Answer with exactly two lowercase English words separated by one space: "
     "the common pet that meows and the common pet that barks. Nothing else.")
SCHEMA = {"type": "json_schema", "json_schema": {"name": "w", "schema": {
    "type": "object", "properties": {"meows": {"type": "string"}, "barks": {"type": "string"}},
    "required": ["meows", "barks"], "additionalProperties": False}, "strict": True}}
CASES = [
    # reporter's six
    ("none",                   None,                                              Q, 40),
    ("regex_nl_or_space",      {"type": "regex", "regex": "[a-z]+[ \\n][a-z]+"},   Q, 40),
    ("regex_newline",          {"type": "regex", "regex": "[a-z]+\\n[a-z]+"},      Q, 40),
    ("json_schema",            SCHEMA,                                            Q, 40),
    ("regex_space",            {"type": "regex", "regex": "[a-z]+ [a-z]+"},        Q, 40),
    ("regex_two_spaces",       {"type": "regex", "regex": "[a-z]+  [a-z]+"},       Q, 40),
    # hypothesis tests: the grammar is written in the table's spelling of a space
    ("regex_u2581",            {"type": "regex", "regex": "[a-z]+▁[a-z]+"},   Q, 40),
    ("regex_byte_piece_0x20",  {"type": "regex", "regex": "cat<0x20>dog"},         Q, 40),
    ("regex_literal_space",    {"type": "regex", "regex": "cat dog"},              Q, 40),
    ("regex_u0120",            {"type": "regex", "regex": "[a-z]+\u0120[a-z]+"},   Q, 40),
    # max_tokens check on the unconstrained path
    ("none_count_maxtok3",     None, "Count from 1 to 40, separated by single spaces. Nothing else.", 3),
    ("regex_count_maxtok3",    {"type": "regex", "regex": "[0-9\\n]+"},
                               "Count from 1 to 40, one number per line. Nothing else.", 3),
]
want = None if args.cases == "all" else set(args.cases.split(","))
with open(args.out, "a") as out:
    for name, rf, prompt, mt in CASES:
        if want and name not in want:
            continue
        body = {"model": args.model, "messages": [{"role": "user", "content": prompt}],
                args.max_tokens_field: mt, "temperature": 0}
        if rf:
            body["response_format"] = rf
        t0 = time.time()
        rec = {"case": name, "model": args.model, "response_format": rf, "prompt": prompt,
               args.max_tokens_field: mt, "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}
        try:
            r = requests.post(args.url, json=body, timeout=args.timeout)
            rec["wall_s"] = round(time.time() - t0, 1)
            rec["http"] = r.status_code
            try:
                j = r.json()
                rec["content"] = j["choices"][0]["message"]["content"]
                rec["finish_reason"] = j["choices"][0].get("finish_reason")
                rec["usage"] = j.get("usage")
            except Exception:
                rec["raw"] = r.text[:600]
        except requests.RequestException as e:
            rec["wall_s"] = round(time.time() - t0, 1)
            rec["error"] = repr(e)[:300]
        print(json.dumps(rec, ensure_ascii=False), flush=True)
        out.write(json.dumps(rec, ensure_ascii=False) + "\n"); out.flush()
