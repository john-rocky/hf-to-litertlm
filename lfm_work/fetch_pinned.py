#!/usr/bin/env python3
"""fetch_pinned.py REPO REVISION CACHE_DIR [MANIFEST.json] - one commit of a Hugging Face model repo into a dedicated cache.

`litert-torch export_hf --model REPO` takes no revision: it loads whatever the cache's refs/main names (with
HF_HUB_OFFLINE=1) or the Hub's latest commit (online). To convert exactly the commit a published file came from:

1. snapshot_download(REPO, revision=REVISION, cache_dir=CACHE_DIR) - every file of that commit.
2. refs/main := REVISION inside CACHE_DIR, so `HF_HUB_CACHE=CACHE_DIR HF_HUB_OFFLINE=1` + `--model REPO` resolve this commit.
3. Check every file against the Hub's record for the commit (HF API ?blobs=true): LFS files by sha256, the others by git
   blob sha1, sizes too. A silently truncated download cannot pass.

CACHE_DIR must not be the default shared cache (~/.cache/huggingface/hub): step 2 rewrites refs/main there.
Writes MANIFEST.json (default CACHE_DIR/fetch_manifest.json): repo, revision, per-file hashes, verdict.
Exit 0 = every file matches, 1 = a mismatch or a missing file, 2 = usage.
Environment: huggingface_hub (any recent version; comes with transformers).
"""
import hashlib
import json
import os
import sys
import urllib.request

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")  # plain HTTPS; set before the hub import

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
    if len(sys.argv) not in (4, 5):
        print(__doc__, file=sys.stderr)
        return 2
    repo, revision, cache = sys.argv[1], sys.argv[2], os.path.abspath(sys.argv[3])
    manifest = sys.argv[4] if len(sys.argv) == 5 else os.path.join(cache, "fetch_manifest.json")
    if os.path.realpath(cache).startswith(SHARED):
        print(f"refusing the shared Hugging Face cache {cache}: refs/main is rewritten here", file=sys.stderr)
        return 2
    from huggingface_hub import snapshot_download
    snap = snapshot_download(repo, revision=revision, cache_dir=cache, max_workers=1)
    ref = os.path.join(cache, "models--" + repo.replace("/", "--"), "refs", "main")
    os.makedirs(os.path.dirname(ref), exist_ok=True)
    with open(ref, "w") as f:
        f.write(revision)
    url = f"https://huggingface.co/api/models/{repo}/revision/{revision}?blobs=true"
    api = json.load(urllib.request.urlopen(url, timeout=60))
    if api["sha"] != revision:
        print(f"the Hub resolved {revision} to {api['sha']}: pass the full 40-character commit", file=sys.stderr)
        return 2
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
    doc = {"repo": repo, "revision": revision, "cache_dir": cache, "snapshot": snap, "refs_main": revision,
           "files": rows, "verdict": "PASS" if ok else "FAIL"}
    with open(manifest, "w") as f:
        json.dump(doc, f, indent=1)
    print(doc["verdict"], snap)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
