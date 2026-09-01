#!/usr/bin/env python3
"""IFEval-lite — instruction-following A/B gate for LFM2.5-230M quant recipes.

GSM8K is at floor for this model (bf16 ~10/100 — the card itself scopes the
model away from math), so the family's conv-int8-vs-wi8fc A/B needs the
model's actual competence: verifiable instruction following (its headline
card number is IFEval 71.71). This harness re-implements the mechanically
checkable instruction types of google/IFEval (evaldata/ifeval.jsonl, 541
rows) and scores prompt-level strict accuracy (all instructions in a prompt
must pass) on the first --n rows whose instructions are all implemented.

evaldata/ is gitignored — regenerate the dataset file (541 rows) by paging
https://datasets-server.huggingface.co/rows?dataset=google%2FIFEval&config=default&split=train
in offset steps of 100 and writing each row["row"] as a JSONL line.

The word/sentence tokenizers are simplified vs the official
instruction_following_eval package, so numbers from here are comparable ONLY
within this harness (bf16 row vs bundle rows) — never against published
IFEval scores.

  # HF bf16 reference row
  ~/venvs/ltconv040dev/bin/python lfm230_work/ifeval_lite.py --hf \
      --json-out lfm230_work/ifeval_bf16.json
  # bundle row (fresh engine per prompt, explicit backend)
  LITERT_LM=~/venvs/lt0161run/bin/litert-lm python3 lfm230_work/ifeval_lite.py \
      --engine lfm230_work/LFM2.5-230M_int8.litertlm --backend cpu \
      --json-out lfm230_work/ifeval_int8.json
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "evaldata", "ifeval.jsonl")
MODEL = "LiquidAI/LFM2.5-230M"
LITERT_LM = os.environ.get(
    "LITERT_LM", os.path.expanduser("~/venvs/lt0161run/bin/litert-lm"))


def _words(text):
    return re.findall(r"\w+", text)


def _sentences(text):
    parts = re.split(r"[.!?]+", text)
    return [p for p in parts if p.strip()]


def _rel(count, relation, n):
    if relation == "at least":
        return count >= n
    if relation == "less than":
        return count < n
    raise ValueError(f"unknown relation {relation}")


# instruction_id -> checker(response, kwargs) -> bool. Faithful to the logic
# of google-research/instruction_following_eval, minus its NLTK tokenizers.
def chk_no_comma(r, kw):
    return "," not in r

def chk_number_words(r, kw):
    return _rel(len(_words(r)), kw.get("relation", "at least"), kw["num_words"])

def chk_number_sentences(r, kw):
    return _rel(len(_sentences(r)), kw.get("relation", "at least"),
                kw["num_sentences"])

def chk_forbidden_words(r, kw):
    low = r.lower()
    return not any(re.search(rf"\b{re.escape(w.lower())}\b", low)
                   for w in kw["forbidden_words"])

def chk_highlighted(r, kw):
    n = len([m for m in re.findall(r"\*[^\n\*]+\*", r) if m.strip("* \n")])
    return n >= kw["num_highlights"]

def chk_keyword_frequency(r, kw):
    c = len(re.findall(rf"\b{re.escape(kw['keyword'].lower())}\b", r.lower()))
    return _rel(c, kw.get("relation", "at least"), kw["frequency"])

def chk_repeat_prompt(r, kw):
    return r.strip().lower().startswith(kw["prompt_to_repeat"].strip().lower())

def chk_quotation(r, kw):
    s = r.strip()
    return len(s) >= 2 and s.startswith('"') and s.endswith('"')

def chk_lowercase(r, kw):
    return r == r.lower()

def chk_capital(r, kw):
    return r == r.upper()

def chk_keywords_existence(r, kw):
    low = r.lower()
    return all(re.search(rf"\b{re.escape(k.lower())}\b", low)
               for k in kw["keywords"])

def chk_title(r, kw):
    return re.search(r"<<[^\n<>]+>>", r) is not None

def chk_bullets(r, kw):
    n = len(re.findall(r"^\s*\*[^\*].*$", r, flags=re.MULTILINE)) + \
        len(re.findall(r"^\s*-.*$", r, flags=re.MULTILINE))
    return n == kw["num_bullets"]

def chk_end_phrase(r, kw):
    return r.strip().strip('"').lower().endswith(
        kw["end_phrase"].strip().strip('"').lower())

def chk_json(r, kw):
    s = r.strip()
    s = re.sub(r"^```(?:json)?\s*", "", s)
    s = re.sub(r"\s*```$", "", s)
    try:
        json.loads(s)
        return True
    except ValueError:
        return False

def chk_number_paragraphs(r, kw):
    parts = [p for p in re.split(r"\s?\*\*\*\s?", r) if p.strip()]
    return len(parts) == kw["num_paragraphs"]

def chk_placeholders(r, kw):
    return len(re.findall(r"\[[^\[\]]+\]", r)) >= kw["num_placeholders"]

def chk_postscript(r, kw):
    marker = kw["postscript_marker"]
    if marker == "P.P.S":
        pat = r"\s*p\.\s?p\.\s?s.*$"
    elif marker == "P.S.":
        pat = r"\s*p\.\s?s\..*$"
    else:
        pat = r"\s*" + re.escape(marker.lower()) + r".*$"
    return re.search(pat, r.lower(), flags=re.MULTILINE) is not None

def chk_letter_frequency(r, kw):
    c = r.lower().count(kw["letter"].lower())
    return _rel(c, kw.get("let_relation", "at least"), kw["let_frequency"])

def chk_capital_word_frequency(r, kw):
    c = len([w for w in _words(r) if w.isupper()])
    return _rel(c, kw.get("capital_relation", "at least"),
                kw["capital_frequency"])

def chk_two_responses(r, kw):
    parts = [p for p in r.split("******") if p.strip()]
    return len(parts) == 2

def chk_multiple_sections(r, kw):
    sp = kw.get("section_spliter") or kw.get("section_splitter") or "Section"
    n = len(re.findall(rf"{re.escape(sp)}\s+\d", r))
    return n >= kw["num_sections"]

def chk_constrained_response(r, kw):
    opts = ("My answer is yes.", "My answer is no.", "My answer is maybe.")
    return any(o in r for o in opts)

def chk_nth_paragraph_first_word(r, kw):
    paras = [p for p in re.split(r"\n\n", r) if p.strip()]
    if len(paras) != kw["num_paragraphs"]:
        return False
    idx = kw["nth_paragraph"] - 1
    if idx >= len(paras):
        return False
    w = _words(paras[idx])
    return bool(w) and w[0].lower() == kw["first_word"].lower()


CHECKERS = {
    "punctuation:no_comma": chk_no_comma,
    "length_constraints:number_words": chk_number_words,
    "length_constraints:number_sentences": chk_number_sentences,
    "keywords:forbidden_words": chk_forbidden_words,
    "detectable_format:number_highlighted_sections": chk_highlighted,
    "keywords:frequency": chk_keyword_frequency,
    "combination:repeat_prompt": chk_repeat_prompt,
    "startend:quotation": chk_quotation,
    "change_case:english_lowercase": chk_lowercase,
    "change_case:english_capital": chk_capital,
    "keywords:existence": chk_keywords_existence,
    "detectable_format:title": chk_title,
    "detectable_format:number_bullet_lists": chk_bullets,
    "startend:end_checker": chk_end_phrase,
    "detectable_format:json_format": chk_json,
    "length_constraints:number_paragraphs": chk_number_paragraphs,
    "detectable_content:number_placeholders": chk_placeholders,
    "detectable_content:postscript": chk_postscript,
    "keywords:letter_frequency": chk_letter_frequency,
    "change_case:capital_word_frequency": chk_capital_word_frequency,
    "combination:two_responses": chk_two_responses,
    "detectable_format:multiple_sections": chk_multiple_sections,
    "detectable_format:constrained_response": chk_constrained_response,
    "length_constraints:nth_paragraph_first_word": chk_nth_paragraph_first_word,
}


def load_rows(n):
    rows = []
    for line in open(DATA):
        d = json.loads(line)
        if all(iid in CHECKERS for iid in d["instruction_id_list"]):
            rows.append(d)
        if len(rows) >= n:
            break
    return rows


def score(row, response):
    per = []
    for iid, kw in zip(row["instruction_id_list"], row["kwargs"]):
        kw = {k: v for k, v in (kw or {}).items() if v is not None}
        try:
            ok = bool(CHECKERS[iid](response, kw))
        except Exception as e:  # checker bug — count as fail, keep the trace
            ok = False
            per.append({"id": iid, "ok": False, "err": repr(e)})
            continue
        per.append({"id": iid, "ok": ok})
    return all(p["ok"] for p in per), per


def gen_hf(rows, max_new_tokens):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16).to(device).eval()
    for row in rows:
        rendered = tok.apply_chat_template(
            [{"role": "user", "content": row["prompt"]}],
            add_generation_prompt=True, tokenize=False)
        ids = tok(rendered, return_tensors="pt",
                  add_special_tokens=False).input_ids.to(device)
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=max_new_tokens,
                                 do_sample=False, eos_token_id=[7, 2],
                                 pad_token_id=0)
        yield tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)


def gen_engine(rows, bundle, backend, max_num_tokens):
    for row in rows:
        p = subprocess.run(
            [LITERT_LM, "run", bundle, "--prompt", row["prompt"],
             "--backend", backend, "--cache", "no",
             "--temperature", "0", "--seed", "0",
             "--max-num-tokens", str(max_num_tokens)],
            capture_output=True, text=True, timeout=900,
            stdin=subprocess.DEVNULL)
        yield p.stdout.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf", action="store_true")
    ap.add_argument("--engine", default=None)
    ap.add_argument("--backend", default="cpu", choices=["cpu", "gpu"])
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--max-tokens", type=int, default=1400)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()
    if args.hf == bool(args.engine):
        sys.exit("pass exactly one of --hf / --engine <bundle>")

    rows = load_rows(args.n)
    gen = (gen_hf(rows, args.max_tokens) if args.hf
           else gen_engine(rows, args.engine, args.backend, args.max_tokens))
    tag = args.tag or ("bf16_hf" if args.hf else os.path.basename(args.engine))

    t0, strict, out_rows = time.time(), 0, []
    for i, (row, resp) in enumerate(zip(rows, gen)):
        ok, per = score(row, resp)
        strict += ok
        out_rows.append({"key": row["key"], "ok": ok, "per": per,
                         "resp": resp[:1500]})
        print(f"[{i+1}/{len(rows)}] {'OK ' if ok else 'NG '}"
              f" {','.join(p['id'] for p in per)}  strict={strict}/{i+1}",
              flush=True)

    summary = {"tag": tag, "n": len(rows), "strict": strict,
               "acc": strict / len(rows),
               "max_tokens": args.max_tokens,
               "seconds": round(time.time() - t0, 1)}
    print(json.dumps(summary))
    if args.json_out:
        json.dump({"summary": summary, "rows": out_rows},
                  open(args.json_out, "w"), indent=1)


if __name__ == "__main__":
    main()
