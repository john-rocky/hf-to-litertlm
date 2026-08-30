# tokenizer_parity — BPE → SentencePiece conversion check

Companion scripts for the litert-torch issue on `tokenizer_to_sentencepiece_lib`
(byte-level BPE vocabularies lose their byte semantics in the converted
SentencePiece proto: standalone Latin-1/Ext-A characters get the byte token's
id, unencodable characters become the UNK piece, and with no `unk_token` that
piece is pad/eos). Nothing here needs a `.litertlm` bundle or LiteRT-LM.

| file | what it does |
|---|---|
| `repro_min.py` | converts a public HF tokenizer with the stock `tokenizer_to_sentencepiece_lib.convert` and encodes a few strings with SentencePiece next to the HF tokenizer. Output: `repro_min.out`. |
| `fix_candidate_lib.py` | a patched copy of the lib: the 256 byte-level tokens become `<0xXX>` BYTE pieces with `byte_fallback` on, only the tokenizer's own `unk_token` is typed UNKNOWN (a dedicated `<unk>` is appended when there is none), specials stay USER_DEFINED. Changes are marked `# FIX`. |
| `fix_candidate_test.py` | converts with the original lib and with the patch, then counts mismatches against the HF tokenizer over the probes, the 223 standalone characters U+00A1–U+017F and every added token. Output: `fix_candidate_test.out`, `fix_candidate_results.json`. |
| `probes.py` | the shared probe strings. |

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
