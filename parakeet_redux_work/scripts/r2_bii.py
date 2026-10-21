#!/usr/bin/env python3
"""Reference B_ii for r2 (venv .venv-ref; LANE §1b): reference B in r1's mode ii = 30 s windows (479,840 samples,
zeros after the audio) + the valid-length mask, transformers generate() per window, window texts decoded separately
(TDT decode = tokenizers Tokenizer.decode(ids, skip_special_tokens=True), no grouping of repeated ids: LANE §0b-8)
and joined with a space. Host rule of r2: a window shorter than 320 samples (fewer than 2 mel frames: the extractor's
unbiased std divides by zero) is dropped.

  --part long     every utterance over 479,840 samples (LS 9, FLEURS de 22 / fr 1 / ru 1) in mode ii
  --part check    200 utterances of <= 479,840 samples drawn with random.Random(20261020) from the 4 sets: mode ii
                  tokens and text against mode i (r1's full run, out/runs/run_B_<set>.full.jsonl)
  --part full     every utterance in mode ii (only needed when check finds a difference)
  --part assemble results/ref_bii_<set>.jsonl (id, text, win_tokens, source) for the 4 sets = mode i for the
                  <= 30 s utterances when check found 200 / 200 equal (else the full mode-ii rows), the long rows
                  from --part long; and results/ref_bii_score.json (L and J per set, with B mode i beside them and
                  the P1 lines of LANE §1b).
--shard k/n splits long / check / full; each shard writes results/r2/bii_<part>.s<k>.jsonl (resumable).
"""
import argparse
import glob
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r2_graphs as G  # noqa: E402
import redux_ref as R  # noqa: E402

W30 = 479840
SETS = {"ls": ("data/sets/librispeech-test-clean/manifest.jsonl", "en"),
        "de": ("data/sets/fleurs-de/manifest.jsonl", "de"),
        "fr": ("data/sets/fleurs-fr/manifest.jsonl", "fr"),
        "ru": ("data/sets/fleurs-ru/manifest.jsonl", "ru")}
B_I = {"ls": ("L", 1.957, "J", 2.377), "de": ("L", 5.611, "J", 6.681), "fr": ("L", 7.435, "J", 8.141),
       "ru": ("L", 6.717, "J", 7.253)}   # B mode i with the TDT decode (LANE §1b as rewritten 20:5x)
_TK = None


def tdt_text(ids):
    """TDT decode of one window: no grouping of repeated ids (LANE §0b-8)."""
    global _TK
    if _TK is None:
        from tokenizers import Tokenizer
        _TK = Tokenizer.from_file("hf/redux/tokenizer.json")
    return _TK.decode([i for i in ids if i != R.BLANK], skip_special_tokens=True)
LINE = {"ls": 0.15, "de": 0.30, "fr": 0.30, "ru": 0.30}


def n_samples(path):
    import soundfile as sf
    return sf.info(path).frames


def work_list(part):
    items = []
    for s, (man, _) in SETS.items():
        for r in R.read_jsonl(man):
            items.append((s, r))
    if part == "full":
        return items
    sized = [(s, r, n_samples(r["path"])) for s, r in items]
    if part == "long":
        return [(s, r) for s, r, n in sized if n > W30]
    pool = [(s, r) for s, r, n in sized if n <= W30]
    return random.Random(20261020).sample(pool, 200)


def mode_ii(model, proc, audio):
    toks, frames, durs, texts, dropped = [], [], [], [], 0
    for s0 in range(0, max(len(audio), 1), W30):
        w = audio[s0:s0 + W30]
        if len(w) < G.MIN_SAMPLES:
            dropped += 1
            continue
        feats, mask = R.features(proc, w, pad_to=W30, mask_padding=True)
        seq, dur = R.gen(model, feats, mask)
        steps = R.steps_from_generate(seq, dur)
        toks.append([tk for (_, tk, _) in steps if tk != R.BLANK])
        frames.append([t for (t, tk, _) in steps if tk != R.BLANK])
        durs.append([d for (_, tk, d) in steps if tk != R.BLANK])
        texts.append(tdt_text(toks[-1]))
    return toks, frames, durs, " ".join(x for x in texts if x), dropped


def run_part(a):
    k, n = (int(x) for x in a.shard.split("/"))
    items = work_list(a.part)[k::n]
    out = f"results/r2/bii_{a.part}.s{k}.jsonl"
    os.makedirs("results/r2", exist_ok=True)
    done = {(x["set"], x["id"]) for x in R.read_jsonl(out)}
    todo = [(s, r) for s, r in items if (s, r["id"]) not in done]
    if not todo:
        print(f"{out}: nothing to do", flush=True)
        return 0
    if a.deadline and time.time() > a.deadline - 90:
        return 0
    R.set_threads(a.threads)
    model, rep = R.build("B")
    proc = R.processor()
    modes_i = {}
    if a.part == "check":
        for s in SETS:
            for x in R.read_jsonl(f"out/runs/run_B_{s}.full.jsonl"):
                modes_i[(s, x["id"])] = x
    per, m = 0.0, 0
    with open(out, "a") as f:
        for s, r in todo:
            if a.deadline and time.time() + max(3 * per, 10) > a.deadline:
                break
            t0 = time.time()
            audio = R.load_wav(r["path"])
            toks, frames, durs, text, dropped = mode_ii(model, proc, audio)
            row = {"set": s, "id": r["id"], "samples": len(audio), "win_tokens": toks, "win_frames": frames,
                   "win_durs": durs, "text": text, "n_windows": len(toks), "dropped_windows": dropped}
            if a.part == "check":
                mi = modes_i[(s, r["id"])]
                row["eq_tokens"] = len(toks) == 1 and toks[0] == mi["tokens"]
                row["eq_frames_durs"] = len(toks) == 1 and frames[0] == mi["frames"] and durs[0] == mi["durs"]
                row["eq_text"] = text == tdt_text(mi["tokens"])
                # diagnostic only: mode i again in this process (same threads) tells a padding effect from a
                # thread-count effect if a row differs (r1's mode i ran 2-3 threads per job)
                feats, mask = R.features(proc, audio)
                seq, _ = R.gen(model, feats, mask)
                fresh = [x for x in seq[1:] if x != R.BLANK]
                row["fresh_mode_i_eq_r1_mode_i"] = fresh == mi["tokens"]
                row["mode_ii_eq_fresh_mode_i"] = len(toks) == 1 and toks[0] == fresh
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            dt = time.time() - t0
            per = dt if per == 0 else 0.8 * per + 0.2 * dt
            m += 1
    print(f"{out}: +{m}, {len(todo) - m} left", flush=True)
    return 0


def shard_rows(part):
    rows = []
    for p in sorted(glob.glob(f"results/r2/bii_{part}.s*.jsonl")):
        rows += R.read_jsonl(p)
    return rows


def assemble():
    import wer as W
    check = shard_rows("check")
    longs = {(x["set"], x["id"]): x for x in shard_rows("long")}
    full = {(x["set"], x["id"]): x for x in shard_rows("full")}
    n_check = len({(x["set"], x["id"]) for x in check})
    diff = [x for x in check if not (x["eq_tokens"] and x["eq_text"])]
    fd_diff = [x["id"] for x in check if x["eq_tokens"] and not x.get("eq_frames_durs", True)]
    want_long = len(work_list("long"))
    rep = {"check": {"n": n_check, "token_and_text_equal": n_check - len(diff),
                     "differ": [{k: x[k] for k in ("set", "id", "eq_tokens", "eq_text")} for x in diff],
                     "by_set": {s: sum(1 for x in check if x["set"] == s) for s in SETS},
                     "tokens_equal_but_frames_or_durs_differ": fd_diff,
                     "fresh_mode_i_ne_r1_mode_i": [x["id"] for x in check if not x.get("fresh_mode_i_eq_r1_mode_i", True)],
                     "mode_ii_ne_fresh_mode_i": [x["id"] for x in check if not x.get("mode_ii_eq_fresh_mode_i", True)]},
           "long": {"n": len(longs), "want": want_long,
                    "dropped_windows": sum(x["dropped_windows"] for x in longs.values()),
                    "rows_with_dropped": [(x["set"], x["id"], x["samples"]) for x in longs.values()
                                          if x["dropped_windows"]]}, "sets": {}}
    if n_check != 200 or len(longs) != want_long:
        rep["stop"] = "check or long incomplete"
        json.dump(rep, open("results/ref_bii_score.json", "w"), indent=1, ensure_ascii=False)
        print(json.dumps(rep)[:800])
        return 1
    use_full = bool(diff)
    for s, (man, lang) in SETS.items():
        rows = R.read_jsonl(man)
        mode_i = {x["id"]: x for x in R.read_jsonl(f"out/runs/run_B_{s}.full.jsonl")}
        out = []
        for r in rows:
            if (s, r["id"]) in longs:
                x = longs[(s, r["id"])]
                out.append({"id": r["id"], "text": x["text"], "win_tokens": x["win_tokens"],
                            "win_frames": x["win_frames"], "win_durs": x["win_durs"], "source": "mode_ii_long",
                            "dropped_windows": x["dropped_windows"]})
            elif use_full:
                x = full.get((s, r["id"]))
                if x is None:
                    rep["stop"] = "full mode ii rows missing (the check found a difference)"
                    json.dump(rep, open("results/ref_bii_score.json", "w"), indent=1, ensure_ascii=False)
                    return 4
                out.append({"id": r["id"], "text": x["text"], "win_tokens": x["win_tokens"],
                            "win_frames": x["win_frames"], "win_durs": x["win_durs"], "source": "mode_ii"})
            else:
                x = mode_i[r["id"]]
                out.append({"id": r["id"], "text": tdt_text(x["tokens"]), "win_tokens": [x["tokens"]],
                            "win_frames": [x["frames"]], "win_durs": [x["durs"]], "source": "mode_i"})
        with open(f"results/ref_bii_{s}.jsonl", "w") as f:
            for x in out:
                f.write(json.dumps(x, ensure_ascii=False) + "\n")
        res, _ = W.score([r["text"] for r in rows], [x["text"] for x in out], lang, [r["id"] for r in rows])
        ri, _ = W.score([r["text"] for r in rows], [tdt_text(mode_i[r["id"]]["tokens"]) for r in rows], lang,
                        [r["id"] for r in rows])
        bi = B_I[s]
        if (ri["L"]["wer"], ri["J"]["wer"]) != (bi[1], bi[3]):
            rep.setdefault("warn", []).append(f"{s}: mode i recomputed L/J {ri['L']['wer']}/{ri['J']['wer']} "
                                              f"!= r1 {bi[1]}/{bi[3]}")
        rep["sets"][s] = {"utterances": res["utterances"], "L": res["L"]["wer"], "J": res["J"]["wer"],
                          "J_SDI": [res["J"]["S"], res["J"]["D"], res["J"]["I"]], "ref_words": res["J"]["ref_words"],
                          "empty_hyp": res["empty_hyp"], "B_mode_i": {"L": bi[1], "J": bi[3]},
                          "P1_line": {"L": round(res["L"]["wer"] + LINE[s], 3), "J": round(res["J"]["wer"] + LINE[s], 3)},
                          "long_rows": sum(1 for x in out if x["source"] == "mode_ii_long"),
                          "long_text_differs_from_mode_i": sum(1 for x in out if x["source"] == "mode_ii_long"
                                                               and x["text"] != tdt_text(mode_i[x["id"]]["tokens"]))}
    rep["source_of_le30s"] = "mode_ii (full)" if use_full else "mode_i (check 200/200 equal)"
    json.dump(rep, open("results/ref_bii_score.json", "w"), indent=1, ensure_ascii=False)
    print(json.dumps(rep, ensure_ascii=False)[:1500], flush=True)
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True, choices=["long", "check", "full", "assemble"])
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--deadline", type=float, default=0.0)
    a = ap.parse_args()
    if a.part == "assemble":
        return assemble()
    return run_part(a)


if __name__ == "__main__":
    import torch
    torch.set_grad_enabled(False)
    sys.exit(main())
