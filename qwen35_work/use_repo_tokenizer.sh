#!/usr/bin/env bash
# use_repo_tokenizer.sh IN OUT TOKENIZER_JSON - replace ONLY the tokenizer section of a built .litertlm with the model repo's tokenizer.json
# bytes, then check the result; the step after convert_qwen35_9b.sh for Qwen/Qwen3.5-9B.
# Why: the export writes the tokenizer.json that transformers builds from the Qwen2Tokenizer class (its pre-tokenizer regex has no
# \p{M} class, and it adds the 7 audio / TTS tokens of tokenizer_config.json), not the repo's tokenizer.json (whose regex has \p{M}).
# The two give different ids for text with combining marks (Hindi, Thai, Tamil, Arabic with harakat); English, CJK, code and emoji
# tokenize the same. The weights are untouched. Not needed when the repo's tokenizer.json already is the class form (Ornith-1.0-9B).
#   1 ../tools/tokenizer_parity/swap_tokenizer_section.py IN TOKENIZER_JSON OUT (`litert-lm unpack` -> tokenizer.json replaced ->
#     `litert-lm pack`; it asserts every other section byte-identical and prints the section list)
#   2 tok_section.py OUT OUT.tok TOKENIZER_JSON: the new section's json bytes equal TOKENIZER_JSON (sha256), and the tokenizers
#     backend built from the section equals the one built from the file
# Exit 0 = both checks pass (OUT kept); otherwise OUT is removed and the exit is non-zero. Refuses to overwrite OUT.
# env: RUN_PY (a Python with litert-lm-builder 0.18.0), LITERT_LM (the litert-lm 0.18.0 CLI), CHECK_PY (a Python with
#      litert-lm-builder and tokenizers; default RUN_PY), EXPECT_TOKENIZER_SHA256 (optional: refuse a TOKENIZER_JSON with another sha256;
#      Qwen/Qwen3.5-9B @ c202236235762e1c871ad0ccb60c8ee5ba337b9a = 5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42).
set -uo pipefail
[ $# -eq 3 ] || { echo "usage: $0 IN OUT TOKENIZER_JSON"; exit 2; }
Q=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$Q/.." && pwd)
IN=$1; OUT=$2; TOK=$3
: "${RUN_PY:?set RUN_PY}" "${LITERT_LM:?set LITERT_LM}"
CHECK_PY=${CHECK_PY:-$RUN_PY}
[ -s "$IN" ] || { echo "no $IN"; exit 2; }
[ -e "$OUT" ] && { echo "refusing to overwrite $OUT"; exit 2; }
[ -s "$TOK" ] || { echo "no $TOK"; exit 2; }
TOK_SHA=$(shasum -a 256 "$TOK" | cut -d' ' -f1)
if [ -n "${EXPECT_TOKENIZER_SHA256:-}" ] && [ "$TOK_SHA" != "$EXPECT_TOKENIZER_SHA256" ]; then
  echo "$TOK has sha256 $TOK_SHA, expected $EXPECT_TOKENIZER_SHA256"; exit 2
fi
fail() { rm -f "$OUT"; echo "FAILED: $1"; exit "${2:-1}"; }
echo "### use_repo_tokenizer: IN=$IN OUT=$OUT tokenizer=$TOK ($TOK_SHA)"
env LITERT_LM_CLI="$LITERT_LM" "$RUN_PY" -I "$ROOT/tools/tokenizer_parity/swap_tokenizer_section.py" "$IN" "$TOK" "$OUT" < /dev/null \
  || fail "swap_tokenizer_section.py exit $?"
"$CHECK_PY" -I "$Q/tok_section.py" "$OUT" "$OUT.tok" "$TOK" || fail "tok_section.py exit $?"
grep -q '"json_bytes_equal_repo": true' "$OUT.tok.json" || fail "the new tokenizer section is not the repo file's bytes"
grep -q '"backend_equal_repo": true' "$OUT.tok.json" || fail "the backend built from the section differs from the repo file's"
shasum -a 256 "$OUT"
echo "### use_repo_tokenizer done: $(stat -f %z "$OUT" 2>/dev/null || stat -c %s "$OUT") B; other sections identical; tokenizer json = $TOK"
