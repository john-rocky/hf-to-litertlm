#!/usr/bin/env bash
# Re-pack a published .litertlm file with the official LiteRT-LM configuration (models/<family>/LlmMetadataProto.pbtext + chat_template.jinja)
# at a pinned LiteRT-LM commit, keeping every other section byte-identical and the file's own max_num_tokens.
#
#   bash scripts/repack_official_metadata.sh <hf-repo> <file.litertlm> <family> <hub-commit> [max_num_tokens] [litert-lm-commit]
#   e.g. bash scripts/repack_official_metadata.sh litert-community/MiniCPM5-1B minicpm_wi4b32_wi8_afp32_gpu_opt.litertlm minicpm5 f6a837aa9437aa389f0161f2a1a65c715ae1c17a 1024
#
# Output: out/repack/<file>  (sha256 printed last). Needs uv (for the pinned litert-lm 0.18.0 CLI), curl, shasum, diff, perl.
set -euo pipefail
REPO=${1:?hf repo, e.g. litert-community/Qwen3-4B}; FILE=${2:?file name}; FAMILY=${3:?minicpm5 | qwen3 | qwen2_5}; HUB=${4:?hub commit of the original}
MAXTOK=${5:-}; LM=${6:-4a363c0728d4461eaf05c4d87250ef3e90deb035}
ROOT=$(cd "$(dirname "$0")/.." && pwd); WORK=$ROOT/out/repack/work_${FILE%.litertlm}; OFF=$ROOT/out/repack/official_$LM; OUT=$ROOT/out/repack
litert-lm() { uvx --from litert-lm==0.18.0 litert-lm "$@"; }
litert-lm --version
mkdir -p "$OFF/$FAMILY" "$WORK/src" "$OUT"
curl -fsSL -o "$OFF/$FAMILY/LlmMetadataProto.pbtext" "https://raw.githubusercontent.com/google-ai-edge/LiteRT-LM/$LM/models/$FAMILY/LlmMetadataProto.pbtext"
curl -fsSL -o "$OFF/$FAMILY/chat_template.jinja" "https://raw.githubusercontent.com/google-ai-edge/LiteRT-LM/$LM/models/$FAMILY/chat_template.jinja"
[ -s "$WORK/src/$FILE" ] || curl -fsSL -o "$WORK/src/$FILE" "https://huggingface.co/$REPO/resolve/$HUB/$FILE"
shasum -a 256 "$WORK/src/$FILE"
rm -rf "$WORK/work" "$WORK/chk"
litert-lm unpack "$WORK/src/$FILE" --output-dir "$WORK/work"
grep '^max_num_tokens' "$WORK/work/LlmMetadataProto.pbtext" || echo "max_num_tokens: not set in the original"
cp "$OFF/$FAMILY/LlmMetadataProto.pbtext" "$WORK/work/LlmMetadataProto.pbtext"
if [ -n "$MAXTOK" ]; then perl -pi -e "s/^max_num_tokens: \d+\$/max_num_tokens: $MAXTOK/" "$WORK/work/LlmMetadataProto.pbtext"; fi
grep '^max_num_tokens' "$WORK/work/LlmMetadataProto.pbtext"
litert-lm pack "$WORK/work" --chat-template "$OFF/$FAMILY/chat_template.jinja" --output "$OUT/$FILE"
litert-lm unpack "$OUT/$FILE" --output-dir "$WORK/chk" --chat-template "$WORK/chk.jinja" && diff "$WORK/chk.jinja" "$OFF/$FAMILY/chat_template.jinja"
# every section other than LlmMetadata must be byte-identical (the 0.18.0 packer lower-cases a section label, so compare by content, not by name)
python3 - "$WORK/work" "$WORK/chk" <<'PY'
import sys, pathlib, hashlib
a, b = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
def digests(d): return sorted(hashlib.sha256(p.read_bytes()).hexdigest() for p in d.iterdir() if p.is_file() and p.name not in ("LlmMetadataProto.pbtext", "model.toml"))
da, db = digests(a), digests(b)
assert da == db, f"section mismatch: {len(da)} vs {len(db)} files"
print(f"sections byte-identical: {len(da)} files")
PY
shasum -a 256 "$OUT/$FILE"
