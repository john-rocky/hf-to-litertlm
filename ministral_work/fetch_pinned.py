#!/usr/bin/env python3
"""fetch_pinned.py CACHE_DIR MANIFEST.json - mistralai/Ministral-3-8B-Instruct-2512-BF16 (the bf16 repo; the plain repo
mistralai/Ministral-3-8B-Instruct-2512 ships FP8 weights) at a pinned commit into a scratch HF cache (never the shared
~/.cache/huggingface/hub: step 2 below rewrites refs/main inside CACHE_DIR). IGNORE = consolidated.safetensors is not
fetched (the Mistral-format copy of the same weights, 17.8 GB; the HF shards are the source); files are fetched one at a
time (max_workers 1) and the convert script sets HF_HUB_DISABLE_XET=1, so a long transfer that stops early shows up as a
size / hash mismatch in step 3 instead of a silently short file.

  python ministral_work/fetch_pinned.py hf_cache hf_cache_manifest.json

1. snapshot_download(repo, revision=PINNED, cache_dir=CACHE_DIR, ignore_patterns=IGNORE) - every file of the commit but IGNORE.
2. refs/main := PINNED inside CACHE_DIR, so `HF_HUB_CACHE=CACHE_DIR HF_HUB_OFFLINE=1` + `from_pretrained(REPO)` (and
   `litert-torch export_hf --model REPO`, which takes no revision) resolve exactly this commit.
3. verify every file against the Hub's record for that commit (HF API ?blobs=true): LFS files by sha256, the others by
   git blob sha1 (sha1(b"blob <len>\\0" + data)); sizes too; an IGNORE file must be absent. Writes MANIFEST.json (repo, revision,
   per-file hashes, PASS).
Exit 0 = all files match, 1 = a mismatch or a missing file, 2 = usage. Refuses the shared cache path.
"""
import hashlib
import json
import os
import sys
import urllib.request

REPO = "mistralai/Ministral-3-8B-Instruct-2512-BF16"
PINNED = "f6fae9795746f63c9be8344932f01275f3c63734"
IGNORE = ["consolidated.safetensors"]
SHARED = os.path.realpath(os.path.expanduser("~/.cache/huggingface/hub"))


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(4 << 20), b""):
            h.update(c)
    return h.hexdigest()


def git_blob_sha1(p):
    data = open(p, "rb").read()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def main():
    if len(sys.argv) != 3:
        print(__doc__, file=sys.stderr)
        return 2
    cache, manifest = os.path.abspath(sys.argv[1]), sys.argv[2]
    if os.path.realpath(cache).startswith(SHARED):
        print(f"refusing the shared HF cache {cache}", file=sys.stderr)
        return 2
    from huggingface_hub import snapshot_download
    snap = snapshot_download(REPO, revision=PINNED, cache_dir=cache, max_workers=1, ignore_patterns=IGNORE)
    ref = os.path.join(cache, "models--" + REPO.replace("/", "--"), "refs", "main")
    os.makedirs(os.path.dirname(ref), exist_ok=True)
    with open(ref, "w") as f:
        f.write(PINNED)
    url = f"https://huggingface.co/api/models/{REPO}/revision/{PINNED}?blobs=true"
    api = json.load(urllib.request.urlopen(url, timeout=60))
    assert api["sha"] == PINNED, api.get("sha")
    rows, ok = [], True
    for s in api["siblings"]:
        p = os.path.join(snap, s["rfilename"])
        row = {"file": s["rfilename"], "hub_size": s.get("size"), "hub_blob_sha1": s.get("blobId"),
               "hub_lfs_sha256": (s.get("lfs") or {}).get("sha256")}
        if s["rfilename"] in IGNORE:
            row["ignored"] = True
            row["PASS"] = not os.path.exists(p)  # not fetched on purpose; present = the ignore did not hold
            if not row["PASS"]:
                row["error"] = "ignored file present"
        elif not os.path.exists(p):
            row["PASS"] = False
            row["error"] = "missing"
        else:
            row["size"] = os.path.getsize(p)
            if row["hub_lfs_sha256"]:
                row["sha256"] = sha256_file(p)
                row["PASS"] = row["sha256"] == row["hub_lfs_sha256"] and row["size"] == row["hub_size"]
            else:
                row["git_blob_sha1"] = git_blob_sha1(p)
                row["PASS"] = row["git_blob_sha1"] == row["hub_blob_sha1"] and row["size"] == row["hub_size"]
        ok = ok and row["PASS"]
        rows.append(row)
        print(f"{'OK ' if row['PASS'] else 'BAD'} {row['file']} {row.get('size')} B")
    doc = {"repo": REPO, "revision": PINNED, "cache_dir": cache, "snapshot": snap, "refs_main": PINNED,
           "files": rows, "verdict": "PASS" if ok else "FAIL"}
    with open(manifest, "w") as f:
        json.dump(doc, f, indent=1)
    print(doc["verdict"], snap)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
