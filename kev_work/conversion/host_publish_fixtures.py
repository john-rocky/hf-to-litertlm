"""The published fixture files -> host/public/fixtures/{requests_public.json, oracle_probs.json}.

    python scripts/host_publish_fixtures.py

requests_public.json = fixtures/requests.json with every record taken from the kev repository's transfer-v4 file
reduced to a reference: tv4 / tv4x / tv4s (220 records) keep id, source, gold, note and the provenance fields file, repo,
tag, line, line_sha256, meta_id (= _meta.id); red_arm_000 (tv4_000 with one word changed) keeps derived_from and edit.
No "request" and no other _meta field (composition_holdout's _meta carries the generated facts). SemIf (144, MIT) and
the 12 records written for this conversion keep everything. The header fields of fixtures/requests.json are kept
verbatim so that fixtures/rebuild_requests.py can rebuild fixtures/requests.json byte for byte (sha256 asserted).

oracle_probs.json = the author's fp32 reference for all 402 questions without any text: keys, row length, readout
indices, z before / after the temperature, probabilities, argmax, top-2 gap, near-tie flag, gold key and the answer
object of kev.api.to_answers minus its "legend" (the legend repeats the level texts). No token ids (they spell the text).
Both files are rebuilt from the inputs on every run (refuses to overwrite)."""
import hashlib
import json
from pathlib import Path

K = Path(__file__).resolve().parents[1]
FIXTURES, ORACLE = K / "fixtures/requests.json", K / "oracle/oracle_0.8b.json"
OUT = K / "host/public/fixtures"
REFERENCE_SOURCES = ("tv4", "tv4x", "tv4s", "red_arm")
REFERENCE_PROVENANCE = ("file", "repo", "tag", "line", "line_sha256", "meta_id")
HEADER = ("version", "created_by", "record_format", "gold_keys", "sources", "counts", "question_types")


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def main():
    import argparse
    global ORACLE, OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["0.8b", "4b"], default="0.8b")
    ap.add_argument("--out", default="")
    ap.add_argument("--created-by", default="")
    ap.add_argument("--only", choices=["oracle_probs", "requests_public"], default="")
    a = ap.parse_args()
    if a.model == "4b":
        ORACLE = K / "oracle/oracle_4b.json"
    if a.out:
        OUT = K / a.out
    raw = FIXTURES.read_bytes()
    full = json.loads(raw)
    assert list(full) == [*HEADER, "records"], list(full)
    assert json.dumps(full, indent=1, ensure_ascii=False) + "\n" == raw.decode(), "requests.json is not in make_fixtures.py's format"
    records, n_ref = [], 0
    for r in full["records"]:
        assert list(r) == ["id", "source", "request", "gold", "note", "provenance"], (r["id"], list(r))
        if r["source"] in REFERENCE_SOURCES:
            prov = r["provenance"]
            keep = ("derived_from", "edit") if r["source"] == "red_arm" else REFERENCE_PROVENANCE
            records.append({"id": r["id"], "source": r["source"], "gold": r["gold"], "note": r["note"],
                            "provenance": {k: prov[k] for k in keep}})
            n_ref += 1
        else:
            assert r["source"] in ("semif", "own"), r["source"]
            records.append(r)
    rebuilt = raw
    if a.created_by:   # the published header; rebuild_requests.py then gives these bytes
        rebuilt = (json.dumps({**full, "created_by": a.created_by}, indent=1, ensure_ascii=False) + "\n").encode()
        assert json.loads(rebuilt)["records"] == full["records"]
    full_pub = json.loads(rebuilt)
    pub = {
        "what": ("The fixture requests of the Kev LiteRT checks (the same requests for Kev-0.8B and Kev-4B). " if a.created_by
                 else "The fixture requests of the Kev-0.8B LiteRT checks. ") + "Records from the kev repository's transfer-v4 "
                "development file (sources tv4, tv4x, tv4s, and red_arm, derived from tv4_000) are listed by reference "
                "only: no 'request'; rebuild_requests.py restores them from that file at tag kev-1.0. SemIf and the "
                "records written for this conversion carry their requests.",
        "full_requests_json_sha256": sha256_bytes(rebuilt),
        "full_requests_json_bytes": len(rebuilt),
        "rebuild": "python rebuild_requests.py --out requests.json [--kev-repo <clone of github.com/jaredpalmer/kev at tag kev-1.0>]",
        "record_counts": {"with_request": len(records) - n_ref, "reference_only": n_ref},
        "header": {k: full_pub[k] for k in HEADER},
        "records": records,
    }
    oracle = json.loads(ORACLE.read_text())
    assert oracle["fixtures_sha256"] == sha256_bytes(raw)
    keep = ("id", "qid", "source", "type", "keys", "row_len", "decide_idx", "opt_idx", "z_pre", "z_post", "probs",
            "argmax_key", "top2_gap", "near_tie", "gold_key")
    questions = []
    for q in oracle["questions"]:
        row = {k: q[k] for k in keep}
        row["answer"] = {k: v for k, v in q["answer"].items() if k != "legend"}
        questions.append(row)
    probs = {
        "what": "The author's fp32 reference for every fixture question (kev 1.0 code on the CPU, torch 2.8.0, "
                "transformers 5.17.0): probabilities after the checkpoint's temperature, the logits before (z_pre) and "
                "after it (z_post), argmax, top-2 gap (near_tie = gap <= 0.02), gold key, the answer of kev.api.to_answers "
                "without its legend, row length and readout indices (row = state + one question's branch). No text and no token ids.",
        "checkpoint": oracle["checkpoint"], "kev": oracle["kev"], "temperature": oracle["temperature"],
        "special_token_ids": oracle["special_token_ids"], "pad_token_id": oracle["pad_token_id"],
        "context": oracle["context"], "fixtures_sha256": oracle["fixtures_sha256"],
        **({"fixtures_note": "fixtures_sha256 is the requests.json the reference read; rebuild_requests.py gives the same "
                             "377 records under the published header (requests_public.json full_requests_json_sha256)"}
           if a.created_by else {}),
        "oracle_json_sha256": sha256_bytes(ORACLE.read_bytes()),
        "requests": [{k: r[k] for k in ("id", "source", "questions", "usage", "state_tokens", "row_lens")} for r in oracle["requests"]],
        "questions": questions,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    for name, doc in (("requests_public.json", pub), ("oracle_probs.json", probs)):
        if a.only and not name.startswith(a.only):
            continue
        path = OUT / name
        assert not path.exists(), f"refusing to overwrite {path}"
        path.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n")
    # no text of a reference record may remain anywhere in the public files
    texts = []
    for r in full["records"]:
        if r["source"] in REFERENCE_SOURCES:
            req = r["request"]
            state = req["state"] if isinstance(req["state"], str) else json.dumps(req["state"], ensure_ascii=False)
            texts.append(state)
            for q in req["questions"].values():
                if isinstance(q.get("instructions"), str):
                    texts.append(q["instructions"])
                crit = q.get("criteria")
                vals = crit.values() if isinstance(crit, dict) else (crit or [])
                texts += [v for v in vals if isinstance(v, str)]
    blob = "\n".join((OUT / n).read_text() for n in ("requests_public.json", "oracle_probs.json") if (OUT / n).exists())
    leaked = [t for t in texts if len(t) >= 12 and t in blob] + [w for w in ('"_meta"', '"certificate"', '"src"') if w in blob]
    assert not leaked, f"{len(leaked)} reference texts found in the public files, e.g. {leaked[0][:80]!r}"
    print(json.dumps({"requests_public": {"records": len(records), "with_request": len(records) - n_ref, "reference_only": n_ref,
                                          "rebuilt_sha256": sha256_bytes(rebuilt), "rebuilt_bytes": len(rebuilt),
                                          "bytes": (OUT / "requests_public.json").stat().st_size if (OUT / "requests_public.json").exists() else None},
                      "oracle_probs": {"questions": len(questions), "oracle": str(ORACLE.relative_to(K)),
                                       "bytes": (OUT / "oracle_probs.json").stat().st_size if (OUT / "oracle_probs.json").exists() else None},
                      "reference_texts_checked": len(texts), "leaked": len(leaked)}, indent=1))


if __name__ == "__main__":
    main()
