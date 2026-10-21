#!/usr/bin/env python3
"""Round 1 step 6: the author's encode_record (joint_schema_model.py on processor.tokenizer, transformers 5.17.0) against
the torch-free copy (scripts/clef_tok.py on tokenizers.Tokenizer.from_file(<snapshot>/tokenizer.json)).

    HF_HUB_OFFLINE=1 venv-ref/bin/python scripts/tok_compare.py      -> results/tok_compare.json (refuses to overwrite)

1. Every record of fixtures/records.json (186), fixtures/heldout/records.json (30) and fixtures/red_arms.json (3): the
   text part of the encoding (no processor, no images: the author's media block sits after the prefix and is round 3's)
   from both sides; equal = the same input_ids, and per question the same id, order, type, question_span, option_spans,
   option_ids.
2. Strings with combining marks (Hindi, Thai, Tamil, Arabic with harakat), CJK, emoji, code and plain English: ids from
   processor.tokenizer(text, add_special_tokens=False) and from the file's tokenizer; every string that differs is listed
   with both id lists and the token pieces of each.
3. The normalizer / pre_tokenizer / decoder / post_processor / model type of processor.tokenizer's backend against the
   file's own JSON (equal or not, both texts kept when they differ), and the added-token counts."""
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

K = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(K / "scripts"))
import clef_tok  # noqa: E402

REV = "f0b7d3cd65296abfe459e6c228b618f5d8b26d2f"
SNAP = K / "hf/hub/models--Cloudflare--clef-flash/snapshots" / REV
JSM_SHA256 = "0e304cf7c6500e8bb59bef7e2afd2c6373f82596dfb3b57d1aa93c175e2dc3a3"
STRINGS = {
    "english": "The customer asks for a refund of the order placed on 2026-09-14.",
    "hindi": "यह वाक्य हिन्दी में लिखा गया है और इसमें मात्राएँ हैं।",
    "hindi_2": "क्या आप मेरी सहायता कर सकते हैं? धन्यवाद।",
    "thai": "ภาษาไทยมีวรรณยุกต์และสระที่อยู่เหนือหรือใต้พยัญชนะ",
    "tamil": "தமிழ் மொழி மிகவும் பழமையான மொழிகளில் ஒன்று.",
    "arabic_harakat": "اَلْعَرَبِيَّةُ لُغَةٌ جَمِيلَةٌ وَمُفِيدَةٌ.",
    "arabic_plain": "اللغة العربية لغة جميلة ومفيدة.",
    "cjk_ja": "日本語の文章です。注文番号は四七一一です。",
    "cjk_zh": "这是一个中文句子，用于测试分词。",
    "cjk_ko": "한국어 문장도 확인합니다.",
    "emoji": "Great job 👍🏽 — the family 👨‍👩‍👧 says thanks 🎉",
    "code": "def f(x):\n    return x ** 2  # square\n\nprint(f(12))",
    "json_like": '{"order_id":"A-1029","total":201.5,"items":["paper","toner"]}',
    "combining_latin": "Café naïve façade — é (e + combining acute)",
    "vietnamese": "Tiếng Việt có nhiều dấu thanh như ắ, ẫ, ợ.",
    "devanagari_numbers": "मूल्य ₹४५० है",
}


def load_author():
    jsm_path = SNAP / "joint_schema_model.py"
    assert hashlib.sha256(jsm_path.read_bytes()).hexdigest() == JSM_SHA256
    spec = importlib.util.spec_from_file_location("joint_schema_model", jsm_path)
    jsm = importlib.util.module_from_spec(spec)
    sys.modules["joint_schema_model"] = jsm
    spec.loader.exec_module(jsm)
    return jsm


def author_form(enc) -> dict:
    return {"ids": list(enc.input_ids),
            "questions": [{"question_id": q.question_id, "type_id": q.question_type, "question_span": list(q.question_span),
                           "option_spans": [list(s) for s in q.option_spans], "option_ids": list(q.option_ids)}
                          for q in enc.questions]}


def main() -> int:
    out = K / "results/tok_compare.json"
    assert not out.exists(), f"refusing to overwrite {out}"
    jsm = load_author()
    import tokenizers
    import transformers
    from tokenizers import Tokenizer
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(str(SNAP))
    ta = processor.tokenizer
    tb = Tokenizer.from_file(str(SNAP / "tokenizer.json"))
    sets = [("fixture", K / "fixtures/records.json"), ("heldout", K / "fixtures/heldout/records.json"),
            ("red_arm", K / "fixtures/red_arms.json")]
    rows, n_eq = [], 0
    for name, path in sets:
        for r in json.loads(path.read_text())["records"]:
            req = {k: v for k, v in r["request"].items() if k not in ("images", "videos")}
            a = author_form(jsm.encode_record(ta, req))
            b = clef_tok.encode_record(tb, req)
            b = {"ids": b["ids"], "questions": b["questions"]}
            eq_ids = a["ids"] == b["ids"]
            eq_q = a["questions"] == b["questions"]
            n_eq += eq_ids and eq_q
            row = {"set": name, "id": r["id"], "has_image": bool(r.get("image_files")), "tokens_text_part": len(a["ids"]),
                   "questions": len(a["questions"]), "ids_equal": eq_ids, "questions_equal": eq_q}
            if not (eq_ids and eq_q):
                first = next((i for i, (x, y) in enumerate(zip(a["ids"], b["ids"])) if x != y), min(len(a["ids"]), len(b["ids"])))
                row.update(first_difference=first, author_len=len(a["ids"]), copy_len=len(b["ids"]))
            rows.append(row)
    strings = []
    for key, text in STRINGS.items():
        ia = list(ta(text, add_special_tokens=False).input_ids)
        ib = list(tb.encode(text, add_special_tokens=False).ids)
        item = {"key": key, "text": text, "equal": ia == ib, "author_len": len(ia), "file_len": len(ib)}
        if ia != ib:
            item.update(author_ids=ia, file_ids=ib, author_pieces=ta.convert_ids_to_tokens(ia),
                        file_pieces=[tb.id_to_token(i) for i in ib],
                        author_decodes_back=ta.decode(ia) == text, file_decodes_back=tb.decode(ib) == text)
        strings.append(item)
    file_json = json.loads((SNAP / "tokenizer.json").read_text())
    back_json = json.loads(ta.backend_tokenizer.to_str())
    parts = {}
    for k in ("normalizer", "pre_tokenizer", "decoder", "post_processor"):
        parts[k] = {"equal": file_json.get(k) == back_json.get(k)}
        if not parts[k]["equal"]:
            parts[k].update(file=file_json.get(k), author_backend=back_json.get(k))
    parts["model_type"] = {"file": file_json["model"]["type"], "author_backend": back_json["model"]["type"]}
    parts["vocab_equal"] = file_json["model"].get("vocab") == back_json["model"].get("vocab")
    parts["merges_equal"] = file_json["model"].get("merges") == back_json["model"].get("merges")
    parts["added_tokens"] = {"file": len(file_json.get("added_tokens", [])), "author_backend": len(back_json.get("added_tokens", [])),
                             "only_in_author_backend": sorted({t["content"] for t in back_json.get("added_tokens", [])}
                                                              - {t["content"] for t in file_json.get("added_tokens", [])}),
                             "only_in_file": sorted({t["content"] for t in file_json.get("added_tokens", [])}
                                                    - {t["content"] for t in back_json.get("added_tokens", [])})}
    doc = {"what": "round 1 step 6: author encode_record (processor.tokenizer) vs the torch-free copy (tokenizers + clef_tok)",
           "versions": {"transformers": transformers.__version__, "tokenizers": tokenizers.__version__},
           "author_tokenizer_class": type(ta).__name__, "tokenizer_json_sha256": hashlib.sha256((SNAP / "tokenizer.json").read_bytes()).hexdigest(),
           "records": len(rows), "records_equal": n_eq, "all_records_equal": n_eq == len(rows),
           "by_set": {s: {"records": sum(r["set"] == s for r in rows),
                          "equal": sum(r["set"] == s and r["ids_equal"] and r["questions_equal"] for r in rows)}
                      for s, _ in sets},
           "strings": len(strings), "strings_equal": sum(s["equal"] for s in strings),
           "strings_differing": [s["key"] for s in strings if not s["equal"]],
           "backend_vs_file": parts, "string_rows": strings, "record_rows": rows}
    out.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps({k: doc[k] for k in ("records", "records_equal", "by_set", "strings", "strings_equal", "strings_differing")},
                     indent=1, ensure_ascii=False))
    print(json.dumps({k: (v if k in ("model_type", "vocab_equal", "merges_equal", "added_tokens") else v["equal"])
                      for k, v in parts.items()}, indent=1, ensure_ascii=False))
    return 0 if doc["all_records_equal"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
