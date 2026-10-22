#!/usr/bin/env bash
# fetch_pinned.sh REPO REVISION HF_CACHE FILE... - the named files of one Hugging Face model repo at a pinned commit,
# downloaded with curl into HF_CACHE/models--<org>--<name>/snapshots/REVISION/ (the hub cache layout, plain files),
# one file at a time over one connection. A file counts as downloaded when it has the size the Hub reports for it at
# that commit; until then curl resumes (-C -), because a long transfer can end early with exit status 0. Files that
# already have their size are skipped. verify_pinned.py then checks the content against the Hub's record of the commit
# (sha256 for LFS files, git blob sha1 for the others).
set -euo pipefail
[ $# -ge 4 ] || { echo "usage: $0 REPO REVISION HF_CACHE FILE..." >&2; exit 2; }
repo=$1; rev=$2; dir="$3/models--${repo//\//--}/snapshots/$rev"; shift 3
mkdir -p "$dir"
for f in "$@"; do
  url="https://huggingface.co/$repo/resolve/$rev/$f"
  want=$(curl -sSIL --fail --retry 3 "$url" | tr -d '\r' | awk 'tolower($1) == "content-length:" {n = $2} END {print n}')
  [ -n "$want" ] || { echo "fetch_pinned.sh: the Hub reports no size for $f" >&2; exit 1; }
  out="$dir/$f"
  if [ -f "$out" ] && [ "$(wc -c < "$out" | tr -d ' ')" = "$want" ]; then echo "have $f ($want B)"; continue; fi
  tries=0
  while :; do
    have=0
    [ -f "$out.part" ] && have=$(wc -c < "$out.part" | tr -d ' ')
    [ "$have" = "$want" ] && break
    [ "$have" -gt "$want" ] && { echo "fetch_pinned.sh: $out.part has $have B, more than the Hub's $want" >&2; exit 1; }
    tries=$((tries + 1))
    [ $tries -le 50 ] || { echo "fetch_pinned.sh: $f has $have of $want B after 50 tries" >&2; exit 1; }
    curl -sS -L --fail -C - -o "$out.part" "$url" || sleep 5
  done
  mv "$out.part" "$out"
  echo "fetched $f ($want B)"
done
