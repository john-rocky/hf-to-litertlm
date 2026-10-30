#!/bin/bash
# externalize_repo.sh — one Hub .litertlm file → the same file with its TFLite weights moved into TFLiteWeights
# sections at the end of the container, through LiteRT's public tools (externalize_tflite_flatbuffer.py +
# externalize_litertlm_sections.py, fetched by fetch_tools.sh), checked section by section and tensor by tensor,
# smoke-run on CPU, staged for review. One repo, one file per call. bash 3.2 or newer (macOS /bin/bash works).
# Disk: about four times the file size during the run (input, unpacked, externalized, output).
#
#   externalize_repo.sh <org/repo> <file.litertlm> [options]
#     --source PATH               use this local copy instead of downloading; size / sha256 are still checked against
#                                 the Hub
#     --no-hub                    the --source file is new and not on the Hub (a fresh export): no model info, no Hub
#                                 comparison; <org/repo> only names the staging directory and the report
#     --revision REV              the Hub revision (branch or commit) whose file is the input; default main
#     --api-json PATH             read the Hub model info (/api/models/<repo>/revision/<REV>?blobs=true) from this file
#     --keep-inline-under-mb N    TFLiteModel sections of N MiB or less stay inline (results row with an empty weights
#                                 path); default 0 = every TFLiteModel section is externalized
#     --num-elements-threshold N  the tool's --num_elements_threshold: tensors of N elements or fewer stay in the graph;
#                                 default 256 (the tool's own). Per section it is raised to the largest INT8 / UINT8
#                                 per-channel DEPTHWISE_CONV_2D filter (lib/dw_floor.py): LiteRT's GPU delegate refuses
#                                 such a filter as an external buffer, and the file would get no GPU engine
#     --executor-metadata auto|off  auto = a bundle with a prefill_decode section and no ExecutorMetadata section gets
#                                 one generated from the signatures (lib/executor_skeleton.py); default off
#     --smoke-attachment PNG      the smoke asks about this image instead of the text question (vision bundles)
#     --no-smoke                  skip the smoke; run it later with --smoke-only
#     --smoke-only                only smoke the staged file (sha256 re-checked) and rewrite its report
#     --workdir DIR               default $XR_WORK_ROOT/<org>__<repo>/<file>/ (XR_WORK_ROOT default ./externalize_work)
#     --staging DIR               default $XR_STAGING_ROOT/<org>__<repo>/ (XR_STAGING_ROOT default ./externalize_staging)
#     --keep-work                 keep the download, the unpacked / externalized files and the intermediate TOMLs
#   Staged: <staging>/<file>, <file>.sha256, <file>.REPORT.md, <file>.model.toml (the TOML it was packed from).
#   Exit: 0 staged | 1 a chain step failed (log named) | 2 usage or precondition | 3 disk guard (free < 4 × file size)
#     | 4 input does not match the Hub (size / sha256) or the Hub could not be read | 5 a check failed, or nothing was
#     externalized (nothing staged) | 6 smoke failed: non-zero exit, empty reply or "An error occurred" (nothing staged;
#     with --smoke-only the report is marked)
#   Env: XR_PY (python3), XR_LITERT_LM (litert-lm), XR_BUILDER (litert-lm-builder), XR_PEEK (litert-lm-peek), XR_HF
#     (hf), XR_TOOLS_PATH (the directory fetch_tools.sh printed), XR_TOOLS_COMMIT override the defaults below;
#     HF_TOKEN reaches the Hub (never written); XR_ALLOW_UNPINNED_TOOLS=1 accepts tool files whose sha256 differ from
#     the pins (recorded); XR_SMOKE_TIMEOUT (s, default 600); XR_FAULT=section|weights plants one fault before the
#     checks, XR_FAULT=depthwise skips the depthwise raise above (self-tests of exit 5 only).
set -uo pipefail
LIB=$(cd "$(dirname "$0")" && pwd)/lib
PY=${XR_PY:-python3}
CLI=${XR_LITERT_LM:-litert-lm}
BUILDER=${XR_BUILDER:-litert-lm-builder}
PEEK=${XR_PEEK:-litert-lm-peek}
HF=${XR_HF:-hf}
TOOLS_COMMIT=${XR_TOOLS_COMMIT:-8761012bb42439f0ac1df55f144d6ace47c2b35c}
TOOLS_PATH=${XR_TOOLS_PATH:-$HOME/.cache/litert-externalize/$TOOLS_COMMIT}
WORK_ROOT=${XR_WORK_ROOT:-$PWD/externalize_work}
STAGING_ROOT=${XR_STAGING_ROOT:-$PWD/externalize_staging}
PIN_FLATBUFFER=5fa96fa600131b3964221dad0ff823358f2794e51084a70baf660b441dd4762c
PIN_SECTIONS=0aa2b9e1eba0427a127a1b22916fb98ae440bd9a8376dc50ed15d65261b6c367
SMOKE_TIMEOUT=${XR_SMOKE_TIMEOUT:-600}
FAULT=${XR_FAULT:-}

usage() { awk 'NR >= 7 && /^#/ { sub(/^# ?/, ""); print; next } NR >= 7 { exit }' "$0"; exit 2; }
[ $# -ge 2 ] || usage
REPO=$1
FILE=$2
shift 2
SOURCE= API_JSON= REVISION=main KEEP_MB=0 EXECMD=off ATTACH= NO_SMOKE=0 SMOKE_ONLY=0 W= S= KEEP_WORK=0 NO_HUB=0 THRESH=256
need() { [ $# -ge 2 ] && [ -n "$2" ] || { echo "$1 needs a value"; exit 2; }; }
while [ $# -gt 0 ]; do
  case $1 in
    --source) need "$@"; SOURCE=$2; shift 2 ;;
    --no-hub) NO_HUB=1; shift ;;
    --num-elements-threshold) need "$@"; THRESH=$2; shift 2 ;;
    --api-json) need "$@"; API_JSON=$2; shift 2 ;;
    --revision) need "$@"; REVISION=$2; shift 2 ;;
    --keep-inline-under-mb) need "$@"; KEEP_MB=$2; shift 2 ;;
    --executor-metadata) need "$@"; EXECMD=$2; shift 2 ;;
    --smoke-attachment) need "$@"; ATTACH=$2; shift 2 ;;
    --no-smoke) NO_SMOKE=1; shift ;;
    --smoke-only) SMOKE_ONLY=1; shift ;;
    --workdir) need "$@"; W=$2; shift 2 ;;
    --staging) need "$@"; S=$2; shift 2 ;;
    --keep-work) KEEP_WORK=1; shift ;;
    -h|--help) usage ;;
    *) echo "unknown option: $1"; exit 2 ;;
  esac
done
echo "$REPO" | grep -Eq '^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$' || { echo "repo must be <org>/<name>: $REPO"; exit 2; }
case $FILE in *.litertlm) ;; *) echo "file must end in .litertlm: $FILE"; exit 2 ;; esac
echo "$KEEP_MB" | grep -Eq '^[0-9]+([.][0-9]+)?$' || { echo "--keep-inline-under-mb takes a number: $KEEP_MB"; exit 2; }
echo "$THRESH" | grep -Eq '^[0-9]+$' || { echo "--num-elements-threshold takes a whole number: $THRESH"; exit 2; }
[ $NO_HUB = 1 ] && [ -z "$SOURCE" ] && [ $SMOKE_ONLY = 0 ] && { echo "--no-hub needs --source (the new file)"; exit 2; }
[ $NO_HUB = 1 ] && [ -n "$API_JSON" ] && { echo "--no-hub and --api-json exclude each other"; exit 2; }
echo "$REVISION" | grep -Eq '^[A-Za-z0-9._-]+$' || { echo "--revision takes a branch name or a commit: $REVISION"; exit 2; }
case $EXECMD in auto|off) ;; *) echo "--executor-metadata takes auto or off: $EXECMD"; exit 2 ;; esac
[ $NO_SMOKE = 1 ] && [ $SMOKE_ONLY = 1 ] && { echo "--no-smoke and --smoke-only exclude each other"; exit 2; }
abspath() { case $1 in /*) echo "$1" ;; *) echo "$PWD/$1" ;; esac; }
[ -n "$ATTACH" ] && { [ -f "$ATTACH" ] || { echo "no such attachment: $ATTACH"; exit 2; }; ATTACH=$(abspath "$ATTACH"); }
[ -n "$SOURCE" ] && { [ -f "$SOURCE" ] || { echo "no such source file: $SOURCE"; exit 2; }; SOURCE=$(abspath "$SOURCE"); }
[ -n "$API_JSON" ] && { [ -f "$API_JSON" ] || { echo "no such model-info file: $API_JSON"; exit 2; }; API_JSON=$(abspath "$API_JSON"); }
BASE=$(basename "$FILE")
SLUG=$(echo "$REPO" | sed 's#/#__#')
W=$(abspath "${W:-$WORK_ROOT/$SLUG/$BASE}")
S=$(abspath "${S:-$STAGING_ROOT/$SLUG}")
U=$W/unpack
E=$W/externalized
B=$W/build
RP=$W/report
LG=$W/logs
KEEP_BYTES=$(awk -v n="$KEEP_MB" 'BEGIN { printf "%d", n * 1048576 }')

fsize() { stat -L -f %z "$1" 2>/dev/null || stat -L -c %s "$1"; }   # -L: a --source symlink is measured through the link
sha() { if command -v shasum > /dev/null; then shasum -a 256 "$1" | cut -d' ' -f1; else sha256sum "$1" | cut -d' ' -f1; fi; }
log() { echo "[$(date '+%F %T')] $*" | tee -a "$RP/run.log"; }
kv() { printf '%s\t%s\n' "$1" "$2" >> "$RP/run.tsv"; }
qline() { local l; l=$(printf '%q ' "$@"); echo "${l% }"; }
run() {   # run <log> <command...>: command line + output into <log>, line appended to commands.txt; returns its exit
  local lg=$1 rc l; shift
  l=$(qline "$@")
  echo "$l" >> "$RP/commands.txt"
  { echo "### $(date '+%F %T') CMD: $l"; "$@" < /dev/null; rc=$?; echo "### exit: $rc"; } > "$lg" 2>&1
  return $rc
}
pyrun() { local lg=$1 s=$2; shift 2; run "$lg" "$PY" -I "$LIB/$s" "$@"; }
pytool() { local lg=$1 m=$2; shift 2; run "$lg" $TIMEL "$PY" -I "$LIB/run_tool.py" "$TOOLS_PATH" "$m" "$@"; }
report() { kv finished "$(date '+%F %T %Z')"; "$PY" -I "$LIB/make_report.py" --workdir "$W" --out "$1" > "$LG/make_report.log" 2>&1; }
fail() {  # fail <exit> <message>: record, write the report into the work dir, keep every file for inspection
  log "FAILED (exit $1): $2"
  kv result "FAILED (exit $1): $2"
  report "$RP/REPORT.md"
  log "report: $RP/REPORT.md; work files kept in $W"
  exit "$1"
}
filelist() { (cd "$1" && find . -type f | sort | while read -r f; do
  printf '%s\t%s\t%s\n' "${f#./}" "$(fsize "$f")" "$(sha "$f")"; done); }
TIMEL=
if /usr/bin/time -l true > /dev/null 2>&1; then TIMEL="/usr/bin/time -l"; elif /usr/bin/time -v true > /dev/null 2>&1; then TIMEL="/usr/bin/time -v"; fi

smoke() {  # smoke <file>: 0 = PASS, 6 = FAILED; report/smoke.{out,tsv}, logs/smoke.err
  local f=$1 q rc v t0 l
  if [ -n "$ATTACH" ]; then q="What is the dominant color of this image? Answer briefly."
  else q="What is the capital of France? Answer in one word."; fi
  local cmd=("$CLI" run "$f" --backend cpu --top-k 1 --thinking false --cache no --prompt "$q")
  [ -n "$ATTACH" ] && cmd+=(--attachment "$ATTACH")
  l="$(qline "${cmd[@]}") < /dev/null"
  echo "$l" >> "$RP/commands.txt"
  t0=$(date +%s)
  perl -e 'alarm shift @ARGV; exec @ARGV or die "exec failed: $!\n"' "$SMOKE_TIMEOUT" "${cmd[@]}" \
    < /dev/null > "$RP/smoke.out" 2> "$LG/smoke.err"
  rc=$?
  if [ $rc = 0 ] && grep -q '[^[:space:]]' "$RP/smoke.out" && ! grep -q 'An error occurred' "$RP/smoke.out"; then v=PASS
  else v=FAILED; fi
  { printf 'command\t%s\n' "$l"; printf 'rc\t%s\n' "$rc"; printf 'seconds\t%s\n' "$(( $(date +%s) - t0 ))"
    printf 'verdict\t%s\n' "$v"; printf 'prompt\t%s\n' "$q"; printf 'attachment\t%s\n' "${ATTACH:-none}"; } > "$RP/smoke.tsv"
  log "smoke: exit $rc after $(( $(date +%s) - t0 )) s, verdict $v, reply: $(head -c 300 "$RP/smoke.out" | tr '\n' ' ')"
  [ $v = PASS ] && return 0
  log "smoke stderr tail: $(tail -3 "$LG/smoke.err" | tr '\n' ' ')"
  return 6
}

# --- smoke only: the staged file, inside the caller's window --------------------------------------------------------
if [ $SMOKE_ONLY = 1 ]; then
  [ -s "$S/$BASE" ] && [ -s "$S/$BASE.sha256" ] && [ -f "$RP/run.tsv" ] || {
    echo "--smoke-only needs $S/$BASE, $S/$BASE.sha256 and the run's $RP/run.tsv"; exit 2; }
  log "smoke-only: $S/$BASE"
  want=$(cut -d' ' -f1 "$S/$BASE.sha256")
  got=$(sha "$S/$BASE")
  [ "$want" = "$got" ] || { log "staged file sha256 $got != $S/$BASE.sha256 ($want)"; exit 2; }
  kv smoke_mode "--smoke-only on the staged file (sha256 re-checked)"
  smoke "$S/$BASE"
  src=$?
  if [ $src = 0 ]; then kv result "STAGED, smoke PASS"; else kv result "STAGED, SMOKE FAILED — do not upload"; fi
  report "$S/$BASE.REPORT.md"
  log "report rewritten: $S/$BASE.REPORT.md"
  exit $src
fi

# --- preconditions -----------------------------------------------------------------------------------------------------
[ -e "$S/$BASE" ] && { echo "refusing: $S/$BASE is already staged (move it away or pass another --staging)"; exit 2; }
for d in "$U" "$E" "$B" "$W/int"; do
  [ -e "$d" ] && { echo "refusing: $d is left from an earlier run (inspect it, then remove it)"; exit 2; }
done
mkdir -p "$W" "$S" || exit 2
ts=$(date +%Y%m%d-%H%M%S)
[ -e "$RP" ] && mv "$RP" "$RP.$ts"
[ -e "$LG" ] && mv "$LG" "$LG.$ts"
mkdir -p "$RP" "$LG" "$B" "$E" || exit 2
log "start: $REPO $FILE → work $W, staging $S"
kv repo "$REPO"; kv file "$FILE"; kv out_name "$BASE"; kv started "$(date '+%F %T %Z')"; kv revision_asked "$REVISION"
kv host "$(uname -sm) $(sw_vers -productVersion 2>/dev/null || true)"
kv keep_inline_under_mib "$KEEP_MB"; kv executor_metadata "$EXECMD"; kv num_elements_threshold "$THRESH"
if [ $NO_SMOKE = 1 ]; then kv smoke_mode "skipped (--no-smoke)"; else kv smoke_mode "cpu, in this run"; fi
for x in "$PY" "$CLI" "$BUILDER" "$PEEK"; do command -v "$x" > /dev/null || fail 2 "not found: $x"; done
[ -z "$SOURCE" ] && { command -v "$HF" > /dev/null || fail 2 "not found: $HF (needed to download; or give --source)"; }
TF=$TOOLS_PATH/litert/tools/externalize_tflite_flatbuffer.py
TS=$TOOLS_PATH/litert/tools/externalize_litertlm_sections.py
[ -f "$TF" ] && [ -f "$TS" ] || fail 2 "LiteRT tools not found under $TOOLS_PATH/litert/tools (run fetch_tools.sh)"
sf=$(sha "$TF"); ss=$(sha "$TS")
if [ "$sf" = "$PIN_FLATBUFFER" ] && [ "$ss" = "$PIN_SECTIONS" ]; then pinned="yes (commit $TOOLS_COMMIT)"
elif [ "${XR_ALLOW_UNPINNED_TOOLS:-0}" = 1 ]; then pinned="NO (XR_ALLOW_UNPINNED_TOOLS=1)"
else fail 2 "tool files differ from the pinned sha256 (flatbuffer $sf, sections $ss); XR_ALLOW_UNPINNED_TOOLS=1 to run anyway"; fi
kv tools_path "$TOOLS_PATH"; kv tools_commit "$TOOLS_COMMIT"; kv tools_pinned "$pinned"
kv tool_sha_flatbuffer "$sf"; kv tool_sha_sections "$ss"
"$PY" -I -c 'import sys; sys.path.insert(0, sys.argv[1]); from litert.python import schema_py_generated as s; s.ExternalBufferT; s.ExternalBufferGroupT' \
  "$TOOLS_PATH" > "$LG/schema_check.log" 2>&1 || fail 2 "the tools cannot import litert.python.schema_py_generated with ExternalBuffer (see $LG/schema_check.log)"
kv python "$PY"; kv cli "$CLI"; kv cli_version "$("$CLI" --version 2>&1 | tail -1)"; kv builder "$BUILDER"; kv peek "$PEEK"; kv hf "$HF"

hub_meta() {  # model info → HUB_SIZE HUB_LFS_SIZE HUB_SHA REV
  local api=$RP/hub_model_info.json url line
  if [ -n "$API_JSON" ]; then
    cp "$API_JSON" "$api" || return 1
    kv api_json "$API_JSON (given with --api-json)"
  else
    url="https://huggingface.co/api/models/$REPO/revision/$REVISION?blobs=true"
    echo "curl -fsSL '$url' -o $api${HF_TOKEN:+   # Authorization header from HF_TOKEN, given on stdin}" >> "$RP/commands.txt"
    if [ -n "${HF_TOKEN:-}" ]; then
      printf 'header = "Authorization: Bearer %s"\n' "$HF_TOKEN" | curl -fsSL -K - "$url" -o "$api" 2> "$LG/hub_api.err" || return 1
    else
      curl -fsSL "$url" -o "$api" 2> "$LG/hub_api.err" || return 1
    fi
    kv api_json "$api"
  fi
  kv api_read_at "$(date '+%F %T %Z')"
  line=$("$PY" -I "$LIB/hub_file.py" "$api" "$FILE") || { log "hub: $line"; return 1; }
  HUB_SIZE=$(echo "$line" | cut -f1); HUB_LFS_SIZE=$(echo "$line" | cut -f2)
  HUB_SHA=$(echo "$line" | cut -f3); REV=$(echo "$line" | cut -f4)
  kv hub_size "$HUB_SIZE"; kv hub_lfs_size "$HUB_LFS_SIZE"; kv hub_sha256 "$HUB_SHA"; kv revision "$REV"
  log "hub: $REPO@$REV $FILE size $HUB_SIZE lfs.sha256 $HUB_SHA"
}

# --- 0 size + disk guard -------------------------------------------------------------------------------------------
HUB_READ=0
[ $NO_HUB = 1 ] && kv hub "not read (--no-hub: a new file, not on the Hub)"
if [ -n "$SOURCE" ]; then SIZE=$(fsize "$SOURCE")
else hub_meta || fail 4 "cannot read the Hub model info for $REPO / $FILE"; HUB_READ=1; SIZE=$HUB_SIZE; fi
needb=$(( SIZE * 4 ))
avail=$(( $(df -Pk "$W" | awk 'NR == 2 { print $4 }') * 1024 ))
log "disk guard: file $SIZE B × 4 = $needb B needed, $avail B free under $W"
[ "$avail" -lt "$needb" ] && fail 3 "disk guard: $avail B free under $W < 4 × $SIZE B"
if [ "$(df -P "$S" | awk 'NR == 2 { print $6 }')" != "$(df -P "$W" | awk 'NR == 2 { print $6 }')" ]; then
  savail=$(( $(df -Pk "$S" | awk 'NR == 2 { print $4 }') * 1024 ))
  [ "$savail" -lt "$SIZE" ] && fail 3 "disk guard: $savail B free under $S < $SIZE B"
fi

# --- 1 the input, checked against the Hub ------------------------------------------------------------------------------
DOWNLOADED=0
if [ -n "$SOURCE" ]; then
  INT=$SOURCE
  kv source_kind "local copy (--source)"
  [ $HUB_READ = 1 ] || [ $NO_HUB = 1 ] || hub_meta || fail 4 "cannot read the Hub model info for $REPO / $FILE"
else
  INT=$W/int/$FILE
  run "$LG/download.log" env HF_HUB_DISABLE_XET=1 "$HF" download "$REPO" "$FILE" --revision "$REV" --local-dir "$W/int" \
    || fail 4 "download failed ($LG/download.log)"
  [ -f "$INT" ] || fail 4 "download left no $INT"
  DOWNLOADED=1
  kv source_kind "downloaded at revision $REV"
fi
isz=$(fsize "$INT"); ish=$(sha "$INT")
kv int_path "$INT"; kv int_size "$isz"; kv int_sha256 "$ish"
if [ $NO_HUB = 1 ]; then
  kv input_check "not compared (--no-hub: a new file)"; log "input: $isz B sha256 $ish (--no-hub: not compared with the Hub)"
elif [ "$isz" = "$HUB_SIZE" ] && [ "$isz" = "$HUB_LFS_SIZE" ] && [ "$ish" = "$HUB_SHA" ]; then
  kv input_check "PASS (size and sha256 = the Hub's)"; log "input: $isz B sha256 $ish = Hub"
else
  kv input_check "FAIL (local $isz B $ish vs Hub $HUB_SIZE B $HUB_SHA)"
  fail 4 "input does not match the Hub: local $isz B sha256 $ish, Hub $HUB_SIZE B (lfs $HUB_LFS_SIZE) sha256 $HUB_SHA"
fi

# --- 2 unpack → manifest → tool per TFLiteModel section → results → rewrite → (executor metadata) → pack → verify ------
run "$LG/unpack.log" "$CLI" unpack "$INT" --output-dir "$U" || fail 1 "unpack failed ($LG/unpack.log)"
grep -q '^Successfully unpacked' "$LG/unpack.log" && [ -f "$U/model.toml" ] || fail 1 "unpack printed no success line or wrote no model.toml ($LG/unpack.log)"
filelist "$U" > "$RP/unpack.files.txt"
run "$RP/peek_orig.txt" "$PEEK" --litertlm_file "$INT" || fail 1 "peek of the input failed ($RP/peek_orig.txt)"
grep -q 'Data Type:[[:space:]]*TFLiteWeights' "$RP/peek_orig.txt" && fail 2 "the input already has TFLiteWeights sections"
if [ "$FAULT" = section ]; then
  [ -f "$U/LlmMetadataProto.pbtext" ] || fail 2 "XR_FAULT=section needs an LlmMetadata section"
  printf 'stop_tokens {\n  token_str: "<xr-fault>"\n}\n' >> "$U/LlmMetadataProto.pbtext"
  log "XR_FAULT=section: one stop token appended to the unpacked LlmMetadata (dry run of exit 5)"
fi
pytool "$LG/manifest.log" litert.tools.externalize_litertlm_sections manifest --toml "$U/model.toml" --output "$B/manifest.tsv" \
  || fail 1 "manifest failed ($LG/manifest.log)"
N=$(( $(wc -l < "$B/manifest.tsv") - 1 ))
log "manifest: $N TFLiteModel sections: $(tail -n +2 "$B/manifest.tsv" | cut -f1,2 | tr '\t\n' ': ')"
[ "$N" -ge 1 ] || fail 5 "no TFLiteModel section to externalize"
printf 'section_index\tmodel_type\tmodel_path\tweights_path\n' > "$B/results.tsv"
while IFS=$'\t' read -r idx mt inp; do
  isec=$(fsize "$inp")
  if [ "$KEEP_BYTES" -gt 0 ] && [ "$isec" -le "$KEEP_BYTES" ]; then
    printf '%s\t%s\t%s\t%s\n' "$idx" "$mt" "$inp" "" >> "$B/results.tsv"
    log "section $idx $mt: $isec B <= $KEEP_BYTES B (--keep-inline-under-mb $KEEP_MB): kept inline, tool not run"
    continue
  fi
  od=$E/$mt
  th=$THRESH
  fl=$("$PY" -I "$LIB/dw_floor.py" "$inp" 2> "$LG/dw_floor_$mt.err") || fail 1 "dw_floor.py failed on $mt ($LG/dw_floor_$mt.err)"
  fmax=$(echo "$fl" | cut -f1); fn=$(echo "$fl" | cut -f2)
  if [ "$FAULT" = depthwise ]; then
    log "XR_FAULT=depthwise: $fn INT8 per-channel DEPTHWISE_CONV_2D filters, threshold left at $th (dry run of exit 5)"
  elif [ "$fmax" -gt "$th" ]; then
    th=$fmax
    log "section $idx $mt: $fn INT8 per-channel DEPTHWISE_CONV_2D filters (largest $fmax elements): threshold $THRESH -> $th, they stay in the graph"
  fi
  kv "threshold_$mt" "$th (asked $THRESH; INT8 per-channel DEPTHWISE_CONV_2D filters: $fn, largest $fmax elements)"
  pytool "$LG/externalize_$mt.log" litert.tools.externalize_tflite_flatbuffer --input_model "$inp" --output_dir "$od" \
    --group_name tflite_weights --num_elements_threshold "$th" || fail 1 "the tool failed on $mt ($LG/externalize_$mt.log)"
  if [ -s "$od/tflite_weights" ]; then
    if [ "$FAULT" = weights ]; then
      p=$(( $(fsize "$od/tflite_weights") / 2 ))
      v=$(dd if="$od/tflite_weights" bs=1 skip=$p count=1 2> /dev/null | od -An -tu1 | tr -d ' ')
      printf "\\$(printf '%03o' $(( v ^ 1 )))" | dd of="$od/tflite_weights" bs=1 seek=$p count=1 conv=notrunc 2> /dev/null
      log "XR_FAULT=weights: byte $p of $mt's weights flipped (dry run of exit 5)"
      FAULT=done
    fi
    printf '%s\t%s\t%s\t%s\n' "$idx" "$mt" "$od/model.tflite" "$od/tflite_weights" >> "$B/results.tsv"
  else
    printf '%s\t%s\t%s\t%s\n' "$idx" "$mt" "$inp" "" >> "$B/results.tsv"
    log "section $idx $mt: the tool externalized nothing (0 B weights): kept inline"
  fi
  log "externalize $mt: $(grep -E '^Externalized|^Cleared|^Original|^Processed|^### exit' "$LG/externalize_$mt.log" | tr '\n' ' ')"
done < <(tail -n +2 "$B/manifest.tsv")
M=$(tail -n +2 "$B/results.tsv" | awk -F'\t' '$4 != ""' | wc -l | tr -d ' ')
[ "$M" -ge 1 ] || fail 5 "nothing externalized (0 rows with a weights file)"
pytool "$LG/rewrite.log" litert.tools.externalize_litertlm_sections rewrite --toml "$U/model.toml" --results "$B/results.tsv" \
  --output "$B/model.toml" || fail 1 "rewrite failed ($LG/rewrite.log)"
PACK_TOML=$B/model.toml
ADDED=
if [ "$EXECMD" = auto ]; then
  pd=$(awk -F'\t' 'NR > 1 && $2 == "prefill_decode" { print $3 }' "$B/manifest.tsv")
  if grep -q 'Data Type:[[:space:]]*ExecutorMetadataProto' "$RP/peek_orig.txt"; then
    kv executor_metadata "auto: the input has an ExecutorMetadata section, nothing added"; log "executor metadata: present, nothing added"
  elif [ -z "$pd" ]; then
    kv executor_metadata "auto: no prefill_decode section, nothing added"; log "executor metadata: no prefill_decode section, nothing added"
  else
    pyrun "$LG/executor_metadata.log" executor_skeleton.py gen --tflite "$pd" --out "$B/executor_metadata.pbtext" \
      || fail 1 "executor metadata not generated ($LG/executor_metadata.log)"
    pyrun "$LG/executor_insert.log" executor_skeleton.py insert --toml "$B/model.toml" --pbtext "$B/executor_metadata.pbtext" \
      --out "$B/model.exec.toml" || fail 1 "executor metadata not inserted ($LG/executor_insert.log)"
    PACK_TOML=$B/model.exec.toml
    ADDED=$B/executor_metadata.pbtext
    kv executor_metadata "auto: generated from $(basename "$pd") and added"; log "executor metadata: generated and added"
  fi
fi
OUT=$B/$BASE
run "$LG/pack.log" $TIMEL "$BUILDER" toml --path "$PACK_TOML" output --path "$OUT" || fail 1 "pack failed ($LG/pack.log)"
grep -q 'successfully created' "$LG/pack.log" && [ -s "$OUT" ] || fail 1 "pack printed no success line ($LG/pack.log)"
run "$RP/peek_new.txt" "$PEEK" --litertlm_file "$OUT" || fail 1 "peek of the output failed ($RP/peek_new.txt)"
if pytool "$LG/verify_peek.log" litert.tools.externalize_litertlm_sections verify-peek --peek "$RP/peek_new.txt" \
     --expected-models "$N" --expected-weights "$M"; then kv verify_peek "PASS ($N TFLiteModel, $M TFLiteWeights at the end)"
else kv verify_peek "FAIL ($(tail -2 "$LG/verify_peek.log" | head -1))"; fi
log "verify-peek ($N / $M): $(tail -2 "$LG/verify_peek.log" | tr '\n' ' ')"

# --- 3 checks --------------------------------------------------------------------------------------------------------
extra=()
[ -n "$ADDED" ] && extra=(--added-executor-metadata "$ADDED")
pyrun "$LG/sections_check.log" sections_check.py --orig "$INT" --new "$OUT" --results "$B/results.tsv" \
  ${extra[@]+"${extra[@]}"} --out "$RP/sections.md" --json "$RP/sections.json"
rc1=$?
kv sections_check "$(grep -E '^VERDICT|^ERROR' "$LG/sections_check.log" | head -1 | sed 's/^VERDICT: //')"
bargs=()
while IFS=$'\t' read -r idx mt mp wp; do
  inp=$(awk -F'\t' -v m="$mt" 'NR > 1 && $2 == m { print $3 }' "$B/manifest.tsv")
  bargs+=(--orig "$mt=$inp")
  [ -n "$wp" ] && bargs+=(--ext "$mt=$mp" --blob "$mt=$wp")
done < <(tail -n +2 "$B/results.tsv")
pyrun "$LG/breakdown.log" tflite_breakdown.py "${bargs[@]}" --out "$RP/breakdown.md" --json "$RP/breakdown.json" \
  --tsv "$RP/externalized_by_dtype.tsv" --graph-mark-mib 50
rc2=$?
kv breakdown "$(grep -E '^VERDICT|^ERROR' "$LG/breakdown.log" | head -1 | sed 's/^VERDICT: //')"
kv float_external "$(sed -n 's/^FLOAT_EXTERNAL=//p' "$LG/breakdown.log")"
kv dw_int8_external "$(sed -n 's/^DW_INT8_EXTERNAL=//p' "$LG/breakdown.log")"
kv graph_over_50mib "$(sed -n 's/^GRAPH_OVER_MARK=//p' "$LG/breakdown.log")"
log "checks: sections rc $rc1, tensors rc $rc2, $(grep -E '^FLOAT_EXTERNAL|^DW_INT8_EXTERNAL|^GRAPH_OVER' "$LG/breakdown.log" | tr '\n' ' ')"
grep -q '^### exit: 0' "$LG/verify_peek.log" && [ $rc1 = 0 ] && [ $rc2 = 0 ] \
  || fail 5 "a check failed: verify-peek $(grep '^### exit' "$LG/verify_peek.log"), sections rc $rc1, tensors rc $rc2"

# --- 4 smoke ---------------------------------------------------------------------------------------------------------
if [ $NO_SMOKE = 0 ]; then
  smoke "$OUT" || fail 6 "smoke failed (reply in $RP/smoke.out, stderr in $LG/smoke.err)"
fi

# --- 5 staging -------------------------------------------------------------------------------------------------------
mv "$OUT" "$S/$BASE" || fail 1 "could not move $OUT to $S"
echo "$(sha "$S/$BASE")  $BASE" > "$S/$BASE.sha256"
cp "$PACK_TOML" "$S/$BASE.model.toml"
kv out_size "$(fsize "$S/$BASE")"; kv out_sha256 "$(cut -d' ' -f1 "$S/$BASE.sha256")"; kv staged_path "$S/$BASE"
if [ $NO_SMOKE = 1 ]; then kv result "STAGED (smoke not run yet)"; else kv result "STAGED, smoke PASS"; fi

# --- 6 clean up (only what this run made; never the --source file) -----------------------------------------------------
if [ $KEEP_WORK = 0 ]; then
  rm -rf "${U:?}" "${E:?}"
  rm -f "${B:?}/model.toml" "${B:?}/model.exec.toml"
  [ $DOWNLOADED = 1 ] && rm -rf "${W:?}/int"
  kv cleanup "removed unpack/, externalized/, build/model*.toml$([ $DOWNLOADED = 1 ] && echo ', int/ (the download)')"
else
  kv cleanup "kept (--keep-work)"
fi
report "$S/$BASE.REPORT.md"
log "staged: $S/$BASE ($(cut -d' ' -f1 "$S/$BASE.sha256")), report $S/$BASE.REPORT.md"
exit 0
