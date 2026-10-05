"""Tokenizer probes for the Python host -> host/tokenizer_probes.json (published as fixtures/tokenizer_probes.json).

    HF_HOME=$W/hf HF_HUB_OFFLINE=1 $ORACLE scripts/host_tokenizer_probes.py

The expected ids come from the tokenizer the author's code uses: transformers 5.17.0 AutoTokenizer for
Qwen/Qwen3.5-0.8B-Base@dc7cdfe2 (class Qwen2Tokenizer; transformers rebuilds the pipeline in code: 248,077 entries,
33 added tokens). Per probe: "ids" = tok(text, add_special_tokens=False), "user_ids" = kev.model.user_tokens(tok, text)
(`<|name|>` rewritten to `<¦name¦>` first, what encode() calls). Recorded beside them: whether the Kev repository's
tokenizer.json (jaredpalmer/kev-0.8b@v1.0, the host's file) and the base repository's own tokenizer.json, both read with
`tokenizers.Tokenizer.from_file`, give the same plain ids. host_parity.py asserts the host's tokenizer on all of them."""
import hashlib
import json
from pathlib import Path

K = Path(__file__).resolve().parents[1]
BASE, BASE_REV = "Qwen/Qwen3.5-0.8B-Base", "dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68"
BASE_JSON = K / "hf/hub/models--Qwen--Qwen3.5-0.8B-Base/snapshots" / BASE_REV / "tokenizer.json"
KEV_JSON = K / "hf/hub/models--jaredpalmer--kev-0.8b/snapshots/788ddbdd65715bb03a56788c822f6c632c9a551d/tokenizer.json"
OUT = K / "host/tokenizer_probes.json"
PROBES = ["Hello world", "नमस्ते दुनिया", "a<think>b", "x<tool_response>y", "<tts_pad>", "café naïve",
          "日本語のテキストです。", "  leading spaces", "emoji 😀 test", "tab\tnew\nline", "<|fim_prefix|>state",
          "Ticket #48213, opened by Mara Quellen."]


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    assert not OUT.exists(), f"refusing to overwrite {OUT}"
    import tokenizers
    import transformers
    from tokenizers import Tokenizer
    from transformers import AutoTokenizer
    from kev.model import user_tokens

    tok = AutoTokenizer.from_pretrained(BASE, revision=BASE_REV)
    kev_json, base_json = Tokenizer.from_file(str(KEV_JSON)), Tokenizer.from_file(str(BASE_JSON))
    rows = []
    for text in PROBES:
        ids = tok(text, add_special_tokens=False).input_ids
        rows.append({"text": text, "ids": ids, "user_ids": user_tokens(tok, text),
                     "kev_repo_tokenizer_json_equal": kev_json.encode(text, add_special_tokens=False).ids == ids,
                     "base_repo_tokenizer_json_equal": base_json.encode(text, add_special_tokens=False).ids == ids})
    doc = {
        "what": "expected token ids of 12 probe strings under the tokenizer the author's code uses",
        "reference": {"tokenizer": f"transformers AutoTokenizer.from_pretrained('{BASE}', revision='{BASE_REV}')",
                      "class": type(tok).__name__, "len": len(tok), "added_tokens": len(tok.get_added_vocab()),
                      "transformers": transformers.__version__, "tokenizers": tokenizers.__version__},
        "ids": "tok(text, add_special_tokens=False).input_ids",
        "user_ids": "kev.model.user_tokens(tok, text): <|name|> -> <¦name¦>, then the same call",
        "files": {"kev_repo_tokenizer_json": {"repo": "jaredpalmer/kev-0.8b", "revision": "v1.0 (bf75a6a8848ea6960ff2ed108d9ed44c2941174f)",
                                              "bytes": KEV_JSON.resolve().stat().st_size, "sha256": sha256_file(KEV_JSON)},
                  "base_repo_tokenizer_json": {"repo": BASE, "revision": BASE_REV,
                                               "bytes": BASE_JSON.resolve().stat().st_size, "sha256": sha256_file(BASE_JSON)}},
        "probes": rows,
        "summary": {"probes": len(rows), "kev_repo_tokenizer_json_equal": sum(r["kev_repo_tokenizer_json_equal"] for r in rows),
                    "base_repo_tokenizer_json_equal": sum(r["base_repo_tokenizer_json_equal"] for r in rows)},
    }
    OUT.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps(doc["summary"]))


if __name__ == "__main__":
    main()
