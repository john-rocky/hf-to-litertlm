"""Round 1 (tokenizer stage): fixtures/token_probes.json and fixtures/rows.json from fixtures/requests.json.

    $REF scripts/build_rows.py [--replace]

Everything runs the provider's own code from hf_small/ (d1_common.provider) with the tokenizer `D1Model.engine` loads
(`AutoTokenizer.from_pretrained(<model dir>)`); no weights.

token_probes.json
  `encode(add_special_tokens=False)` of A..Z, " A".." Z", 0..9, the yes / no forms, eight special tokens, and (the
  fixture's native-letter labels) a..z and " a".." z", each with "single token"; the provider's fallback pool scanned the
  same way; for every choice question of the fixture, `prompt.option_codes` / `prompt.aliases` / `prompt.readout_ids`
  against the probe table (a question whose alias is not its own code fell back to the pool and is listed); the noul
  and score groups against the probes.
rows.json
  One row per (record, question) the provider's parser accepts: `text` = `prompt.render` with the arguments
  SystemOne passes (d1_common.engine_settings), `ids` = encode(text) (the BOS is in the text), `answer_slot` = len - 1,
  `row_len`, `readout_ids` (one group per option), `state_len` = len(encode(prefix_text)), and three tokenizer checks:
  `prefix_equal` (the row starts with the encoded prefix), `split_equal` (encode(prefix) + encode(suffix) = ids, the
  ids `SystemOne._request` feeds a request of several questions), `raw_equal` (tokenizers.Tokenizer.from_file on
  tokenizer.json gives the same ids). card_cats_001 carries its picture markup (`SystemOne._image_markup(1)`, run on a
  stand-in object holding the provider's processor); its `<image>` is expanded only with the picture (round 5), so its
  ids are the unexpanded ones and it is left out of the length table. Questions the parser rejects are listed under
  `skipped`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import types
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from d1_common import (FIXTURES, HF_SMALL, K, PROBES, REV, ROWS, encode, engine_settings, load_fixtures,  # noqa: E402
                       load_tokenizer, provider, questions_of, render_row, sha256_file, split_texts)

LETTERS = [chr(c) for c in range(ord("A"), ord("Z") + 1)]
LOWER = [chr(c) for c in range(ord("a"), ord("z") + 1)]
SPECIALS = ["<|startoftext|>", "<|im_start|>", "<|im_end|>", "<image>", "<|image_start|>", "<|image_end|>",
            "<|img_thumbnail|>", "<|img_row_1_col_1|>"]
BUCKETS = (256, 512, 1024, 2048, 4096)


def pct(values, q):
    """Nearest rank on the sorted values (the Kev scripts' rule)."""
    s = sorted(values)
    return s[min(len(s) - 1, int(round(q * (len(s) - 1))))]


def probe_table(tok) -> list[dict]:
    groups = [("A..Z", LETTERS), (' " A".." Z"', [" " + c for c in LETTERS]), ("0..9", [str(i) for i in range(10)]),
              ("yes forms", ["yes", "Yes", "YES"]), ("no forms", ["no", "No", "NO"]), ("special", SPECIALS),
              ("a..z (native labels)", LOWER), ('" a".." z" (native labels)', [" " + c for c in LOWER])]
    rows = []
    for group, texts in groups:
        for t in texts:
            ids = encode(tok, t)
            rows.append({"group": group.strip(), "text": t, "ids": ids, "single": len(ids) == 1})
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--replace", action="store_true", help="overwrite token_probes.json and rows.json")
    a = ap.parse_args()
    for p in (PROBES, ROWS):
        if p.exists() and not a.replace:
            raise SystemExit(f"refusing to overwrite {p} (pass --replace)")
    import tokenizers
    import transformers
    from tokenizers import Tokenizer
    from transformers import AutoProcessor

    prompt, runner = provider("prompt"), provider("runner")
    tok = load_tokenizer()
    raw = Tokenizer.from_file(str(HF_SMALL / "tokenizer.json"))
    settings = engine_settings(tok, prompt)
    fixtures = load_fixtures()
    fixtures_sha = sha256_file(FIXTURES)

    # ---------------------------------------------------------------- probes
    probes = probe_table(tok)
    by_text = {p["text"]: p for p in probes}
    pool = [{"text": t, "ids": encode(tok, t)} for t in prompt._FALLBACK_POOL]
    pool_families = [("A..Z", 0, 26), ("00..99", 26, 126), ("a..z", 126, 152), ("#0..#199", 152, 352), ("AA..ZZ", 352, len(pool))]
    pool_summary = {name: {"entries": hi - lo, "single": sum(len(p["ids"]) == 1 for p in pool[lo:hi])}
                    for name, lo, hi in pool_families}
    assert sum(v["entries"] for v in pool_summary.values()) == len(pool)
    yes_ids = [by_text[t]["ids"][0] for t in prompt.YES_FORMS if by_text[t]["single"]]
    no_ids = [by_text[t]["ids"][0] for t in prompt.NO_FORMS if by_text[t]["single"]]

    choice_checks, fallbacks, group_mismatch = Counter(), [], []
    noul_checked = score_checked = 0
    for r in fixtures["records"]:
        for qid, q, err in questions_of(prompt, r):
            if q is None:
                continue
            groups = prompt.readout_ids(tok, q)
            if isinstance(q, prompt.Choice):
                labels = list(q.criteria.keys())
                codes = prompt.option_codes(labels)
                alias = prompt.aliases(tok, labels)
                for code, (acode, aid), g in zip(codes, alias, groups):
                    p = by_text.get(code)
                    if acode != code:
                        fallbacks.append({"id": r["id"], "qid": qid, "code": code, "alias": acode, "alias_id": aid})
                        continue
                    assert p is not None and p["single"] and p["ids"][0] == aid, (r["id"], qid, code, aid, p)
                    sp = by_text.get(" " + code)
                    want = [aid] + ([sp["ids"][0]] if sp and sp["single"] and sp["ids"][0] != aid else [])
                    if g != want:
                        group_mismatch.append({"id": r["id"], "qid": qid, "code": code, "group": g, "probe_group": want})
                choice_checks[tuple(codes)] += 1
            elif isinstance(q, prompt.Noul):
                assert groups == [yes_ids, no_ids], (r["id"], qid, groups)
                noul_checked += 1
            else:
                want = [[by_text[str(i)]["ids"][0]] for i in range(len(q.criteria))]
                assert groups == want, (r["id"], qid, groups, want)
                score_checked += 1
    code_table = {}
    for codes in choice_checks:
        for c in codes:
            sp = by_text.get(" " + c)
            code_table[c] = {"id": by_text[c]["ids"][0], "space_id": sp["ids"][0] if sp and sp["single"] else None}
    probes_doc = {
        "what": "token ids of the option forms and special tokens under the d1-3B tokenizer, and the provider's verbalizer "
                "checked against them on every question of fixtures/requests.json",
        "tokenizer": {"loader": "transformers AutoTokenizer.from_pretrained(hf_small) (what D1Model.engine loads)",
                      "class": type(tok).__name__, "len": len(tok), "vocab_size": tok.vocab_size,
                      "added_tokens": len(tok.get_added_vocab()), "bos": [tok.bos_token, tok.bos_token_id],
                      "eos": [tok.eos_token, tok.eos_token_id], "pad": [tok.pad_token, tok.pad_token_id],
                      "add_special_tokens_adds_nothing": encode(tok, "hi") == tok.encode("hi", add_special_tokens=True),
                      "embedding_rows": 128000, "transformers": transformers.__version__,
                      "tokenizers": tokenizers.__version__, "tokenizer_json_sha256": sha256_file(HF_SMALL / "tokenizer.json"),
                      "repo": "LiquidAI/d1-3B", "revision": REV},
        "ids": "tokenizer.encode(text, add_special_tokens=False)",
        "probes": probes,
        "summary": {g: {"probes": sum(p["group"] == g for p in probes), "single": sum(p["group"] == g and p["single"] for p in probes)}
                    for g in dict.fromkeys(p["group"] for p in probes)},
        "fallback_pool": {"source": "prompt._FALLBACK_POOL", "families": pool_summary,
                          "not_single": [p["text"] for p in pool if len(p["ids"]) != 1][:50]},
        "verbalizer_check": {
            "fixtures_sha256": fixtures_sha,
            "choice_code_sets": {" ".join(k): v for k, v in sorted(choice_checks.items(), key=lambda x: -x[1])},
            "code_table": code_table, "fallbacks": fallbacks, "group_mismatches": group_mismatch,
            "noul_questions": noul_checked, "noul_groups": [yes_ids, no_ids],
            "score_questions": score_checked,
        },
    }

    # ------------------------------------------------------------------ rows
    proc = AutoProcessor.from_pretrained(str(HF_SMALL))
    stand_in = types.SimpleNamespace(processor=proc)   # `_image_markup` reads only self.processor
    rows, skipped = [], []
    for r in fixtures["records"]:
        images = r["request"].get("images") or []
        markup = runner.SystemOne._image_markup(stand_in, len(images)) if images else ""
        parsed = questions_of(prompt, r)
        n_ok = sum(q is not None for _, q, _ in parsed)
        state = r["request"]["state"]
        for qid, q, err in parsed:
            if q is None:
                skipped.append({"id": r["id"], "qid": qid, "error": err})
                continue
            text = render_row(prompt, tok, settings, state, q, markup)
            ids = encode(tok, text)
            prefix, suffix = split_texts(prompt, tok, settings, state, q, markup)
            assert prefix + suffix == text
            p_ids, s_ids = encode(tok, prefix), encode(tok, suffix)
            row = {"id": r["id"], "source": r["source"], "qid": qid, "type": q.type, "questions_in_request": n_ok,
                   "path": "tree" if n_ok > 1 else "row",
                   "keys": (list(q.criteria) if q.type == "choice" else ["true", "false"] if q.type == "noul"
                            else [str(i) for i in range(len(q.criteria))]),
                   "gold": r["gold"].get(qid),
                   "readout_ids": prompt.readout_ids(tok, q),
                   "row_len": len(ids), "answer_slot": len(ids) - 1, "state_len": len(p_ids),
                   "prefix_equal": ids[:len(p_ids)] == p_ids, "split_equal": p_ids + s_ids == ids,
                   "raw_equal": raw.encode(text, add_special_tokens=False).ids == ids,
                   "text": text, "ids": ids}
            if q.type == "choice":
                row["codes"] = [c for c, _ in prompt.aliases(tok, list(q.criteria))]
            if images:
                row["images"] = images
                row["image_markup"] = markup
                row["image_expansion_pending"] = True
            rows.append(row)

    sized = [x for x in rows if not x.get("image_expansion_pending")]
    lens = [x["row_len"] for x in sized]
    by_source = {}
    for src in dict.fromkeys(x["source"] for x in sized):
        ls = [x["row_len"] for x in sized if x["source"] == src]
        by_source[src] = {"rows": len(ls), "p50": pct(ls, 0.5), "p99": pct(ls, 0.99), "max": max(ls)}
    lengths = {
        "rows": len(sized), "excluded": [f"{x['id']}/{x['qid']}" for x in rows if x.get("image_expansion_pending")],
        "p50": pct(lens, 0.5), "p99": pct(lens, 0.99), "max": max(lens), "min": min(lens),
        "mean": round(statistics.fmean(lens), 1),
        "fit": {str(L): sum(n <= L for n in lens) for L in BUCKETS},
        "smallest_bucket": {str(L): sum(1 for n in lens if n <= L and all(n > b for b in BUCKETS if b < L)) for L in BUCKETS},
        "over_largest": [f"{x['id']}/{x['qid']}" for x in sized if x["row_len"] > BUCKETS[-1]],
        "state_len": {"p50": pct([x["state_len"] for x in sized], 0.5), "max": max(x["state_len"] for x in sized)},
        "by_source": by_source,
        "percentile_rule": "nearest rank: sorted[round(q * (n - 1))]",
    }
    checks = {k: sum(not x[k] for x in rows) for k in ("prefix_equal", "split_equal", "raw_equal")}
    rows_doc = {
        "what": "one row per (record, question): the provider's render, its ids under the provider's tokenizer, the answer "
                "slot, the read-out token groups and the state length",
        "fixtures_sha256": fixtures_sha, "settings": settings,
        "render": "prompt.render(tokenizer, state, q, bos, lead, style, system, option_style, images) = SystemOne.render",
        "ids": "tokenizer.encode(text, add_special_tokens=False)",
        "path": "the provider's runner: a request of one question is one plain pass (_one_pass; with pictures, of the "
                "processor's ids); several questions run as a tree (trunk = encode(prefix), or the processor's ids when there "
                "are pictures; branches = encode(suffix))",
        "rows_count": len(rows), "skipped": skipped, "check_failures": checks, "lengths": lengths,
        "rows": rows,
    }
    PROBES.write_text(json.dumps(probes_doc, indent=1, ensure_ascii=False) + "\n")
    ROWS.write_text(json.dumps(rows_doc, ensure_ascii=False) + "\n")
    print(json.dumps({"probes": probes_doc["summary"], "fallback_pool": pool_summary,
                      "choice_code_sets": probes_doc["verbalizer_check"]["choice_code_sets"],
                      "fallbacks": len(fallbacks), "group_mismatches": len(group_mismatch),
                      "noul": noul_checked, "score": score_checked,
                      "rows": len(rows), "skipped": skipped, "check_failures": checks,
                      "lengths": {k: v for k, v in lengths.items() if k != "by_source"},
                      "sha256": {"token_probes": sha256_file(PROBES), "rows": sha256_file(ROWS)}}, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
