#!/usr/bin/env python3
"""fetch_pinned.py CACHE_DIR MANIFEST.json - ibm-granite/granite-4.1-8b at a pinned commit into a scratch HF cache (never
the shared ~/.cache/huggingface/hub: step 2 below rewrites refs/main inside CACHE_DIR). Files are fetched one at a time
(max_workers 1) and the convert script sets HF_HUB_DISABLE_XET=1, so a long transfer that stops early shows up as a size /
hash mismatch in step 3 instead of a silently short file.

  python granite41_work/fetch_pinned.py hf_cache hf_cache_manifest.json

1. snapshot_download(repo, revision=PINNED, cache_dir=CACHE_DIR) - every file of the commit.
2. refs/main := PINNED inside CACHE_DIR, so `HF_HUB_CACHE=CACHE_DIR HF_HUB_OFFLINE=1` + `from_pretrained(REPO)` (and
   `litert-torch export_hf --model REPO`, which takes no revision) resolve exactly this commit.
3. verify every file against the Hub's record for that commit (HF API ?blobs=true): LFS files by sha256, the others by
   git blob sha1 (sha1(b"blob <len>\\0" + data)); sizes too. Writes MANIFEST.json (repo, revision, per-file hashes, PASS).
Exit 0 = all files match, 1 = a mismatch or a missing file, 2 = usage. Refuses the shared cache path.
"""
import hashlib
import json
import os
import sys
import urllib.request

REPO = "ibm-granite/granite-4.1-8b"
PINNED = "1504002f650e656a0a3789d99574df12e3e94ed0"
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
    snap = snapshot_download(REPO, revision=PINNED, cache_dir=cache, max_workers=1)
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
        if not os.path.exists(p):
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
