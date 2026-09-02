#!/usr/bin/env python3
"""Japanese 8Q quality gate for a .litertlm — the same harness as
scripts/verify_quality.py (litert-mac-verify, one question per process, same
PASS rule: >= min-correct AND no degenerate answer) with the 8 checks asked in
Japanese, plus one UNSCORED streaming probe.

Why a separate file: the English 8Q measures instruction-following in the
model's second language; a Japanese-tuned checkpoint is gated on what it is
for. The extra probe asks for emoji + a rare kanji so the per-token streaming
decode of ByteFallback pieces (4 byte tokens per emoji on this vocab) is
exercised — a U+FFFD in that answer means the runtime emitted a partial UTF-8
sequence, which the English gates never see.

    python3 sarashina_work/verify_quality_ja.py <model.litertlm> [--backend cpu|gpu] [--json out.json]
"""
import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import verify_quality as vq  # noqa: E402

SUFFIX = " 簡潔に答えてください。"
QUESTIONS = [
    ("17+25=42",       "17 + 25 はいくつですか？",                       r"42"),
    ("capital=東京",    "日本の首都はどこですか？",                        r"東京"),
    # 涼しい (cool) is a standard antonym of 暑い too — the 1b bf16 gives it.
    ("opp(暑い)=寒い",  "「暑い」の反対の言葉は何ですか？",                  r"寒い|冷たい|涼しい"),
    ("days/week=7",    "一週間は何日ありますか？",                        r"7|７|七"),
    ("thanks(en)",     "「ありがとう」を英語で言うと何ですか？",             r"thank"),
    ("8*7=56",         "8 かける 7 はいくつですか？",                      r"56"),
    ("0.9>0.11",       "0.9 と 0.11 では、どちらが大きいですか？",          r"0\.9"),
    ("Fuji=静岡/山梨",  "富士山はどの都道府県にありますか？",                r"静岡|山梨"),
]
# Unscored: exercises multi-byte streaming (emoji = 4 ByteFallback tokens, 𠮷 = 4).
PROBE = ("utf8-stream", "次の文をそのまま書き写してください: 「今日は😀の日、𠮷野家で鬱陶しい雨☔」", None)


def degenerate_ja(text):
    """verify_quality.degenerate() splits on whitespace, which Japanese lacks —
    add a character-level loop check (any 6-char chunk repeated 4+ times)."""
    if vq.degenerate(text):
        return True
    t = re.sub(r"\s+", "", text)
    if len(t) >= 48:
        chunks = [t[i:i + 6] for i in range(0, len(t) - 5)]
        from collections import Counter
        if chunks and Counter(chunks).most_common(1)[0][1] >= 4:
            return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--min-correct", type=int, default=6)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--verifier", default=str(vq.DEFAULT_VERIFIER))
    ap.add_argument("--backend", choices=["cpu", "gpu"])
    ap.add_argument("--json")
    ap.add_argument("--name")
    args = ap.parse_args()
    model = Path(args.model)
    if not model.exists():
        print(f"ERROR: model not found: {model}", file=sys.stderr)
        return 2
    name = args.name or model.stem
    print(f"== JA quality gate: {name} ({args.backend or 'default'}) ==\n   model: {model}\n")
    results, toks = [], []
    for label, q, pat in QUESTIONS + [PROBE]:
        try:
            raw, tok = vq.run_verifier(args.verifier, model, q + SUFFIX, args.max_tokens,
                                       backend=args.backend)
        except Exception as e:  # noqa: BLE001
            print(f"FAIL (harness) on '{label}': {e}", file=sys.stderr)
            return 2
        ans = vq.strip_think(raw)
        scored = pat is not None
        ok = bool(re.search(pat, ans.lower())) if scored else ("�" not in ans and bool(ans.strip()))
        degen = (not ans.strip()) or degenerate_ja(ans)
        if tok:
            toks.append(tok)
        results.append({"label": label, "question": q, "ok": ok, "scored": scored,
                        "degenerate": degen, "answer": ans, "raw": raw, "tok_s": tok})
        mark = "✓" if ok else "·"
        print(f"   [{mark}]{' ⚠️degen' if degen else '       '} {label:16s}"
              f"{'' if scored else ' (probe)'} -> {' '.join(ans.split())[:90]!r}")
    score = sum(r["ok"] for r in results if r["scored"])
    any_degen = any(r["degenerate"] for r in results if r["scored"])
    probe = next(r for r in results if not r["scored"])
    median_tok = sorted(toks)[len(toks) // 2] if toks else None
    passed = (score >= args.min_correct) and (not any_degen)
    print(f"\n   correct: {score}/8   degenerate: {'⚠️ YES' if any_degen else 'no'}"
          f"   utf8-probe: {'clean' if probe['ok'] else '⚠️ U+FFFD or empty'}"
          + (f"   decode~{median_tok:.0f} tok/s" if median_tok else ""))
    print(f"   VERDICT: {'✅ PASS' if passed else '❌ FAIL'}  (threshold {args.min_correct}/8, non-degenerate)")
    if args.json:
        Path(args.json).write_text(json.dumps({
            "name": name, "model": str(model), "backend": args.backend,
            "score": score, "of": 8, "degenerate": any_degen, "passed": passed,
            "utf8_probe_clean": probe["ok"], "median_decode_tok_s": median_tok,
            "questions": [{k: r[k] for k in ("label", "ok", "scored", "degenerate", "answer")}
                          for r in results],
        }, indent=2, ensure_ascii=False) + "\n")
        print(f"   wrote {args.json}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
