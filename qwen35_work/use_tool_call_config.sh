#!/usr/bin/env bash
# use_tool_call_config.sh IN OUT - make LiteRT-LM 0.19.0 parse the tool calls of a .litertlm of the Qwen3.5 architecture (the model writes
# <tool_call><function=NAME><parameter=ARG>VALUE</parameter></function></tool_call>; a generic_model bundle without a tool-call
# configuration returns that as text). Adds the tool_call_config of LiteRT-LM's models/qwen3_5/LlmMetadataProto.pbtext at v0.19.0
# and sets supports_function_calling true; nothing else changes (the chat template, stop tokens, channels, sampler, model type and
# max_num_tokens stay as IN has them). Only the LlmMetadata section changes. For the litert-community Qwen3.5-4B, Qwen3.5-9B,
# Agents-A1-4B, Qwen3.8-27B and Qwen3.5-4B-3D-Coder files, after each repo's own metadata step (REPRODUCE.md names it).
#   1 curl the official pbtext at the pinned commit (OUT.official.pbtext; make_tool_call_pbtext.py checks its sha256)
#   2 dump_llm_metadata.py IN OUT.meta: the bundle's LlmMetadata section (the header and that section only)
#   3 make_tool_call_pbtext.py OUT.meta OUT.official.pbtext OUT.pbtext: the two fields above; refuses a bundle that is not
#     generic_model or already has a tool_call_config
#   4 ../g9v3_work/repack_metadata.py IN OUT --pbtext OUT.pbtext --expect-changed tool_call_config,supports_function_calling:
#     `litert-lm unpack` -> the new LlmMetadataProto.pbtext -> `litert-lm pack`, then every other section asserted unchanged
#     (TFLite raw bytes, the tokenizer's inflated bytes, ExecutorMetadata parsed), the section list and IN's sha256 too
#   5 bundle_sections.py OUT.sections.json OUT: the sha256 of every section of OUT
# Every command is printed as `### CMD: ...` before it runs. Exit 0 = the repack report PASS (OUT kept); otherwise OUT is removed and
# the exit is non-zero. Refuses to overwrite OUT. The file sha256 changes from run to run (the packer writes a uuid and a timestamp):
# compare sections (scripts/section_identity.py, or the sha256 lines step 5 prints).
# env: RUN_PY (a Python with litert-lm-builder 0.19.0), LITERT_LM (the litert-lm 0.19.0 CLI)
set -uo pipefail
[ $# -eq 2 ] || { echo "usage: $0 IN OUT"; exit 2; }
Q=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$Q/.." && pwd)
IN=$1; OUT=$2
OFFICIAL_URL=https://raw.githubusercontent.com/google-ai-edge/LiteRT-LM/771b06e1a2c7f70a399a718599109facf37e38fd/models/qwen3_5/LlmMetadataProto.pbtext
: "${RUN_PY:?set RUN_PY}" "${LITERT_LM:?set LITERT_LM}"
[ -s "$IN" ] || { echo "no $IN"; exit 2; }
for p in "$OUT" "$OUT.official.pbtext" "$OUT.meta" "$OUT.pbtext" "$OUT.repack.json" "$OUT.sections.json"; do
  [ -e "$p" ] && { echo "refusing to overwrite $p"; exit 2; }
done
fail() { rm -f "$OUT"; echo "FAILED: $1"; exit "${2:-1}"; }
run() { echo "### CMD: $*"; "$@"; }
run curl -fsSL -o "$OUT.official.pbtext" "$OFFICIAL_URL" || fail "curl exit $?"
run "$RUN_PY" -I "$Q/dump_llm_metadata.py" "$IN" "$OUT.meta" || fail "dump_llm_metadata.py exit $?"
run "$RUN_PY" -I "$Q/make_tool_call_pbtext.py" "$OUT.meta" "$OUT.official.pbtext" "$OUT.pbtext" || fail "make_tool_call_pbtext.py exit $?"
run "$RUN_PY" -I "$ROOT/g9v3_work/repack_metadata.py" "$IN" "$OUT" --pbtext "$OUT.pbtext" --report "$OUT.repack.json" \
  --expect-changed tool_call_config,supports_function_calling --cli "$LITERT_LM" || fail "repack_metadata.py exit $?"
run "$RUN_PY" -I "$Q/bundle_sections.py" "$OUT.sections.json" "$OUT" || fail "bundle_sections.py exit $?"
shasum -a 256 "$OUT"
echo "### use_tool_call_config done: $(stat -f %z "$OUT" 2>/dev/null || stat -c %s "$OUT") B; changed tool_call_config, supports_function_calling; every other section unchanged"
