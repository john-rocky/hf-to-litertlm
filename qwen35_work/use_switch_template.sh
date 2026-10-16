#!/usr/bin/env bash
# use_switch_template.sh IN OUT - give a built Qwen/Qwen3.5-9B .litertlm the thinking-switch metadata of litert-community/Qwen3.5-4B:
# the prompt template qwen35_text_T1.jinja (enable_thinking off by default: the generation prompt opens with an empty think block, past
# assistant turns render without one; enable_thinking=true pre-fills `<think>\n`), a `thought` channel (`<think>\n` / `\n</think>`),
# supports_thinking true, supports_function_calling false. Only the LlmMetadata section changes. The step after convert_qwen35_9b.sh
# and use_repo_tokenizer.sh for Qwen/Qwen3.5-9B.
#   1 dump_llm_metadata.py IN OUT.meta: the bundle's LlmMetadata section (the header and that section only)
#   2 make_new_pbtext.py OUT.meta qwen35_text_T1.jinja OUT.pbtext (no --template-only): the four fields above; stop tokens,
#     max_num_tokens, the model type and everything else kept
#   3 ../g9v3_work/repack_metadata.py IN OUT --pbtext OUT.pbtext --expect-changed <the fields step 2 changed>: `litert-lm unpack` ->
#     the new LlmMetadataProto.pbtext -> `litert-lm pack`, then every other section asserted unchanged (TFLite raw bytes, the tokenizer's
#     inflated bytes, ExecutorMetadata parsed), the section list and IN's sha256 too
# Every command is printed as `### CMD: ...` before it runs. Exit 0 = the repack report PASS (OUT kept); otherwise OUT is removed and the
# exit is non-zero. Refuses to overwrite OUT. The file sha256 changes from run to run (the packer writes a uuid and a timestamp):
# compare sections.
# env: RUN_PY (a Python with litert-lm-builder 0.18.0), LITERT_LM (the litert-lm 0.18.0 CLI)
set -uo pipefail
[ $# -eq 2 ] || { echo "usage: $0 IN OUT"; exit 2; }
Q=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$Q/.." && pwd)
IN=$1; OUT=$2
: "${RUN_PY:?set RUN_PY}" "${LITERT_LM:?set LITERT_LM}"
[ -s "$IN" ] || { echo "no $IN"; exit 2; }
for p in "$OUT" "$OUT.meta" "$OUT.pbtext" "$OUT.repack.json"; do [ -e "$p" ] && { echo "refusing to overwrite $p"; exit 2; }; done
fail() { rm -f "$OUT"; echo "FAILED: $1"; exit "${2:-1}"; }
run() { echo "### CMD: $*"; "$@"; }
run "$RUN_PY" -I "$Q/dump_llm_metadata.py" "$IN" "$OUT.meta" || fail "dump_llm_metadata.py exit $?"
run "$RUN_PY" -I "$Q/make_new_pbtext.py" "$OUT.meta" "$Q/qwen35_text_T1.jinja" "$OUT.pbtext" || fail "make_new_pbtext.py exit $?"
EXPECT=$(python3 -c 'import json,sys; print(",".join(json.load(open(sys.argv[1]))["changed_fields"]))' "$OUT.pbtext.check.json")
run "$RUN_PY" -I "$ROOT/g9v3_work/repack_metadata.py" "$IN" "$OUT" --pbtext "$OUT.pbtext" --report "$OUT.repack.json" \
  --expect-changed "$EXPECT" --cli "$LITERT_LM" || fail "repack_metadata.py exit $?"
run "$LITERT_LM" describe "$OUT" < /dev/null
shasum -a 256 "$OUT"
echo "### use_switch_template done: $(stat -f %z "$OUT" 2>/dev/null || stat -c %s "$OUT") B; changed $EXPECT; every other section unchanged"
