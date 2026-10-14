# tokenizer_parity — BPE → SentencePiece conversion check

Companion scripts for the litert-torch issue on `tokenizer_to_sentencepiece_lib`
(byte-level BPE vocabularies lose their byte semantics in the converted
SentencePiece proto: standalone Latin-1/Ext-A characters get the byte token's
id, unencodable characters become the UNK piece, and with no `unk_token` that
piece is pad/eos). Only `swap_tokenizer_section.py` and the bundle check at the end need a
`.litertlm` bundle and the LiteRT-LM CLI.

| file | what it does |
|---|---|
| `repro_min.py` | converts a public HF tokenizer with the stock `tokenizer_to_sentencepiece_lib.convert` and encodes a few strings with SentencePiece next to the HF tokenizer. Output: `repro_min.out`. |
| `fix_candidate_lib.py` | a patched copy of the lib: the 256 byte-level tokens become `<0xXX>` BYTE pieces with `byte_fallback` on, only the tokenizer's own `unk_token` is typed UNKNOWN (a dedicated `<unk>` is appended when there is none), specials stay USER_DEFINED. Changes are marked `# FIX`. |
| `fix_candidate_test.py` | converts with the original lib and with the patch, then counts mismatches against the HF tokenizer over the probes, the 223 standalone characters U+00A1–U+017F and every added token. Output: `fix_candidate_test.out`, `fix_candidate_results.json`. |
| `probes.py` | the shared probe strings. |
| `swap_tokenizer_section.py` | repairs an already-built `.litertlm`: replaces only its tokenizer section with an HF `tokenizer.json` (`litert-lm unpack` → edit `model.toml` → `litert-lm pack`) and checks every other section stayed byte-identical. `python swap_tokenizer_section.py in.litertlm tokenizer.json out.litertlm` |

## Run

```
python3.12 -m venv sp020
sp020/bin/pip install "sentencepiece==0.2.0" protobuf "transformers==5.14.1"   # 0.2.0 = the version LiteRT-LM releases pin; 0.2.2 refuses the NUL piece the stock converter emits
export HF_HUB_DISABLE_XET=1
# with litert-torch installed the lib is imported from the package; otherwise point at a checkout:
export LITERT_TORCH_SRC=~/code/litert-torch
sp020/bin/python3 repro_min.py HuggingFaceTB/SmolLM3-3B Qwen/Qwen2.5-0.5B-Instruct
sp020/bin/python3 fix_candidate_test.py      # SmolLM3-3B, Qwen2.5-0.5B-Instruct, SmolVLM2-500M, Ministral-3-3B-Reasoning-2512
```

Measured with the pins above (`fix_candidate_test.out`): standalone-character mismatches 198/163/204/196 → 0 on the four vocabularies, characters mapped to UNK 71/36/92/69 → 0, no special token splits (SmolVLM2's `<|endoftext|>` is that tokenizer's real unk and stays a SentencePiece UNK piece). What remains is what a SentencePiece BPE cannot express: the pre-tokenizer regex (digit grouping, leading-space rules) and merges over partial UTF-8 sequences.

## Is a bundle's tokenizer the repo's own?

`litert-lm unpack in.litertlm --output-dir u` writes an HF tokenizer section as `u/Section<N>_HF_Tokenizer_Zlib.zlib`: an 8-byte little-endian length, then the zlib-compressed `tokenizer.json`. Compare it with the repo's own `tokenizer.json` at the revision the bundle was converted from; when the two differ, compare the pipelines, because `AutoTokenizer.save_pretrained` can write the same vocabulary with another normalizer and pre-tokenizer:

```
python3 -c "import sys, zlib; b = open(sys.argv[1], 'rb').read(); open('bundle_tokenizer.json', 'wb').write(zlib.decompress(b[8:]))" u/Section*_HF_Tokenizer_Zlib.zlib
shasum -a 256 bundle_tokenizer.json tokenizer.json
python3 -c "import json, sys; a, b = (json.load(open(p)) for p in sys.argv[1:3]); [print(k, 'same' if a.get(k) == b.get(k) else 'DIFFERS') for k in ('normalizer', 'pre_tokenizer', 'model', 'added_tokens')]" bundle_tokenizer.json tokenizer.json
```

`swap_tokenizer_section.py in.litertlm tokenizer.json out.litertlm` then puts the repo's file in. North-Micro-Vision-Instruct's bundles carried the copy `save_pretrained` writes (every digit its own token) until 2026-10-15; REPRODUCE.md, "Metadata re-packs (2026-10)", block G.
