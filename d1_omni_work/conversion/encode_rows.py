"""Step 4: every fixture question through the provider's encode() -> results/encoded_rows.json,
results/fixture_summary.json

    HF_HUB_DISABLE_XET=1 venv-ref/bin/python scripts/encode_rows.py

Mode per record (modeling_d1.probabilities_batch): text = max_len 16384, noul_default None, audio False, temperature;
image = 896, YES_NO, False, no temperature; audio = 15360, YES_NO, True, state None -> {}, no temperature.
max_len is min(mode max_len, 16384 - P); with the prefix sizes used here (image 144 / 256, audio <= 375) that never
binds (asserted). budget / per / truncation flags are recomputed with encode()'s own formulas and cross-checked
against its output (marker positions, total length).
"""
import json
import time
from collections import Counter, defaultdict

import numpy as np

import d1_src as S

BUCKETS = [128, 256, 512, 1024, 2048, 4096]
PREFIX = {"image": {"384px": 144, "512px": 256}, "audio": {"10s": 125, "30s": 375}}


def pct(xs, q):
    return float(np.percentile(np.asarray(xs, dtype=np.float64), q)) if xs else None


def within(xs):
    return {f"le_{b}": int(sum(1 for x in xs if x <= b)) for b in BUCKETS} | {"over_4096": int(sum(1 for x in xs if x > 4096))}


def smallest_bucket(n):
    return next((b for b in BUCKETS if n <= b), None)


def bucket_proposal(text, image, audio):
    """Rows per smallest fitting bucket (text rows; media rows with their prefix), and the proposal line.
    The attention-score size per layer is arithmetic ([1, 8, 2L, L] fp32 in the design-3 GQA form), not a measurement."""
    occ = lambda xs: dict(Counter(smallest_bucket(n) for n in xs))  # noqa: E731
    media = {"image_384px_plus_144": occ([n + 144 for n in image]), "image_512px_plus_256": occ([n + 256 for n in image]),
             "audio_10s_plus_125": occ([n + 125 for n in audio]), "audio_30s_plus_375": occ([n + 375 for n in audio])}
    return {
        "line": "text L128 / L256 / L512 / L1024 / L2048 / L4096 (smallest that fits); image: 384 px -> L256, "
                "512 px -> L512, tiled originals -> L2048 / L4096; audio: <= 10 s -> L256, <= 30 s -> L512 "
                "(one trunk graph per L serves text and media rows alike)",
        "text_rows_by_smallest_bucket": dict(sorted(occ(text).items())),
        "media_rows_by_smallest_bucket": media,
        "empty_buckets_in_this_fixture": [b for b in BUCKETS if b not in occ(text)
                                          and all(b not in v for v in media.values())],
        "attention_scores_fp32_bytes_per_layer": {f"L{b}": 8 * 2 * b * b * 4 for b in BUCKETS},
        "note": "L1024 holds no fixture row (no row between 513 and 1,024 tokens); it is kept for real states of that "
                "size and can be checked with shorter rows padded into it. L4096 holds only own_long_3400 (3 rows) and "
                "its fp32 attention scores are 1.07 GB per attention layer in this form (arithmetic, not measured).",
    }


def main():
    t0 = time.time()
    tok = S.tokenizer()
    P = S.provider_prompt()
    cfg = S.config()
    temps = cfg["temperatures"]
    doc = json.loads((S.K / "fixtures/requests.json").read_text())
    ids_of = tok.convert_tokens_to_ids
    enc = lambda s: tok(P.escape(s), add_special_tokens=False)["input_ids"]  # noqa: E731  (encode()'s own enc)

    rows, skipped, errors = [], [], []
    for r in doc["records"]:
        mode = S.mode_of(r)
        state = S.state_of(r, mode)
        for qid, qd in r["request"]["questions"].items():
            try:
                q = P.as_question(qd)
            except ValueError as e:
                skipped.append({"id": r["id"], "qid": qid, "error": str(e)})
                continue
            max_len = mode["max_len"]
            if mode["mode"] != "text":
                big = max(PREFIX[mode["mode"]].values())
                assert min(max_len, S.MAX_LENGTH - big) == max_len  # the prefix never shortens max_len here
            try:
                ids, markers = P.encode(tok, state, q, max_len, mode["noul_default"], mode["audio"])
            except ValueError as e:
                errors.append({"id": r["id"], "qid": qid, "source": r["source"], "error": str(e)})
                continue
            # encode()'s internals, recomputed
            opts = P.render_options(q, mode["noul_default"], mode["audio"])
            K = len(opts)
            budget = max(96, min(K * 24 + 32, max_len // 2))
            per = max(2, (budget - 3 * K) // K)
            instr = enc(q.instructions)
            q_cut = 1 + len(instr) > max(16, budget)
            opt_tok = [len(enc(" " + t)) for t in opts]
            opt_cut = [n > per for n in opt_tok]
            question_len = min(1 + len(instr), max(16, budget)) + sum(3 + min(n, per) for n in opt_tok) + 1
            room = max(0, max_len - question_len - 2)
            state_tok = len(enc(P.serialize(state)))
            state_cut = state_tok > room
            assert len(ids) == 1 + 1 + min(state_tok, room) + question_len, (r["id"], qid)
            assert all(ids[m] == S.TOKEN_IDS["<|mask|>"] for m in markers) and len(markers) == K
            assert ids[0] == 1 and ids[1] == S.TOKEN_IDS["<|reserved_7|>"] and ids[-1] == S.TOKEN_IDS["<|reserved_11|>"]
            texts = [P.serialize(state), q.instructions] + opts
            escaped = any(P.escape(t) != t for t in texts)
            row = {"id": r["id"], "qid": qid, "source": r["source"], "mode": mode["mode"], "type": q.type,
                   "qtype": P.QTYPES[q.type], "K": K, "len": len(ids), "budget": budget, "per": per,
                   "state_tokens": state_tok, "room": room, "state_truncated": state_cut,
                   "instructions_truncated": q_cut, "options_truncated": sum(opt_cut), "escaped": escaped,
                   "temperature_key": P.temperature_key(q) if mode["calibrate"] else None,
                   "temperature": (temps.get(P.temperature_key(q), temps.get(q.type, 1.0)) if mode["calibrate"]
                                   else None),
                   "noul_flip": q.type == "noul", "markers": markers, "ids": ids}
            rows.append(row)

    all_ids = [[x["id"], x["qid"], x["ids"]] for x in rows]
    out_rows = {"written": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "requests_sha256": S.sha256_file(
        S.K / "fixtures/requests.json"), "tokenizer": {"repo": S.REPO, "revision": S.REV},
        "n_rows": len(rows), "skipped_by_as_question": skipped, "encode_errors": errors,
        "ids_sha256": S.sha256_json(all_ids), "rows": rows}
    (S.K / "results/encoded_rows.json").write_text(json.dumps(out_rows, ensure_ascii=False) + "\n")

    # ---- summary
    by_src = defaultdict(list)
    recs_by_src = Counter(r["source"] if r["id"] != "own_long_3400" else "own_long" for r in doc["records"])
    for x in rows:
        by_src[x["source"] if x["id"] != "own_long_3400" else "own_long"].append(x)
    per_source = {}
    for src, xs in by_src.items():
        lens = [x["len"] for x in xs]
        per_source[src] = {"records": recs_by_src[src], "questions": len(xs),
                           "types": dict(Counter(x["type"] for x in xs)), "max_K": max(x["K"] for x in xs),
                           "len_p50": pct(lens, 50), "len_p99": pct(lens, 99), "len_max": max(lens),
                           "len_min": min(lens), "buckets_text_len": within(lens)}
    text = [x["len"] for x in rows if x["mode"] == "text"]
    image = [x["len"] for x in rows if x["mode"] == "image"]
    audio = [x["len"] for x in rows if x["mode"] == "audio"]
    lengths = {
        "text": {"rows": len(text), "p50": pct(text, 50), "p99": pct(text, 99), "max": max(text),
                 "buckets": within(text)},
        "image_text_only": {"rows": len(image), "max": max(image), "lens": image},
        "image_plus_144_384px": {"max": max(image) + 144, "buckets": within([n + 144 for n in image])},
        "image_plus_256_512px": {"max": max(image) + 256, "buckets": within([n + 256 for n in image])},
        "audio_text_only": {"rows": len(audio), "max": max(audio), "lens": audio},
        "audio_plus_125_10s": {"max": max(audio) + 125, "buckets": within([n + 125 for n in audio])},
        "audio_plus_375_30s": {"max": max(audio) + 375, "buckets": within([n + 375 for n in audio])},
    }
    pub_text = [x["len"] for x in rows if x["mode"] == "text"
                and next(r for r in doc["records"] if r["id"] == x["id"])["publishable"]]
    longest = sorted(rows, key=lambda x: -x["len"])[:8]
    own_long = next(x for x in rows if x["id"] == "own_long_3400")
    summary = {
        "written": out_rows["written"], "requests_sha256": out_rows["requests_sha256"],
        "n_records": len(doc["records"]), "n_questions_in_fixture": sum(len(r["request"]["questions"])
                                                                        for r in doc["records"]),
        "n_rows_encoded": len(rows), "skipped_by_as_question": skipped, "encode_errors": errors,
        "by_source": per_source,
        "types_all": dict(Counter(x["type"] for x in rows)),
        "modes": dict(Counter(x["mode"] for x in rows)),
        "max_K": max(x["K"] for x in rows), "max_markers_per_row": max(len(x["markers"]) for x in rows),
        "K_histogram": dict(sorted(Counter(x["K"] for x in rows).items())),
        "lengths": lengths,
        "publishable_text_rows": {"rows": len(pub_text), "p50": pct(pub_text, 50), "p99": pct(pub_text, 99),
                                  "max": max(pub_text), "buckets": within(pub_text)},
        "longest_rows": [{"id": x["id"], "qid": x["qid"], "len": x["len"]} for x in longest],
        "state_truncated_rows": [f"{x['id']}/{x['qid']}" for x in rows if x["state_truncated"]],
        "instructions_truncated_rows": [f"{x['id']}/{x['qid']}" for x in rows if x["instructions_truncated"]],
        "options_truncated_rows": [{"row": f"{x['id']}/{x['qid']}", "n": x["options_truncated"], "per": x["per"]}
                                   for x in rows if x["options_truncated"]],
        "escape_effective_rows": [f"{x['id']}/{x['qid']}" for x in rows if x["escaped"]],
        "own_long_3400": {"state_tokens": own_long["state_tokens"], "row_lens": [x["len"] for x in rows
                                                                               if x["id"] == "own_long_3400"]},
        "temperatures_used": dict(Counter(f"{x['temperature_key']}={x['temperature']}" for x in rows
                                          if x["temperature_key"])),
        "bucket_proposal": bucket_proposal(text, image, audio),
        "percentile_method": "numpy.percentile default (linear interpolation)",
        "ids_sha256": out_rows["ids_sha256"],
        "seconds": round(time.time() - t0, 1),
    }
    (S.K / "results/fixture_summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False) + "\n")
    show = {k: summary[k] for k in ("n_records", "n_questions_in_fixture", "n_rows_encoded", "encode_errors",
                                    "types_all", "modes", "max_K", "K_histogram", "state_truncated_rows",
                                    "instructions_truncated_rows", "options_truncated_rows", "own_long_3400",
                                    "temperatures_used", "longest_rows", "ids_sha256", "seconds")}
    show["escape_effective_rows_n"] = len(summary["escape_effective_rows"])
    show["by_source"] = {k: {kk: v[kk] for kk in ("records", "questions", "len_p50", "len_p99", "len_max",
                                                   "buckets_text_len")} for k, v in per_source.items()}
    show["lengths"] = {k: {kk: vv for kk, vv in v.items() if kk != "lens"} for k, v in lengths.items()}
    print(json.dumps(show, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
