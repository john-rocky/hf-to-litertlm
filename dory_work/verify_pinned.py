#!/usr/bin/env python3
"""verify_pinned.py REPO REVISION HF_CACHE MANIFEST.json FILE... - check the files fetch_pinned.sh wrote into
HF_CACHE/models--<org>--<name>/snapshots/REVISION/ against the Hub's record of that commit (HF API ?blobs=true):
LFS files by sha256, the others by git blob sha1 (sha1(b"blob <len>\\0" + data)), sizes too. Writes MANIFEST.json
(repo, revision, snapshot dir, per-file hashes, verdict). Exit 0 = every file matches, 1 = a mismatch or a missing
file, 2 = usage."""
import hashlib
import json
import os
import sys
import urllib.request


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
    if len(sys.argv) < 6:
        print(__doc__, file=sys.stderr)
        return 2
    repo, revision, cache, manifest = sys.argv[1:5]
    snap = os.path.join(os.path.abspath(cache), "models--" + repo.replace("/", "--"), "snapshots", revision)
    api = json.load(urllib.request.urlopen(
        f"https://huggingface.co/api/models/{repo}/revision/{revision}?blobs=true", timeout=60))
    assert api["sha"] == revision, api.get("sha")
    hub = {s["rfilename"]: s for s in api["siblings"]}
    rows, ok = [], True
    for name in sys.argv[5:]:
        s = hub.get(name) or {}
        p = os.path.join(snap, name)
        row = {"file": name, "hub_size": s.get("size"), "hub_blob_sha1": s.get("blobId"),
               "hub_lfs_sha256": (s.get("lfs") or {}).get("sha256")}
        if not s or not os.path.exists(p):
            row.update(PASS=False, error="not in the commit" if not s else "missing")
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
        print(f"{'OK ' if row['PASS'] else 'BAD'} {repo} {name} {row.get('size')} B", flush=True)
    doc = {"repo": repo, "revision": revision, "snapshot": snap, "files": rows, "verdict": "PASS" if ok else "FAIL"}
    with open(manifest, "w") as f:
        json.dump(doc, f, indent=1)
    print(doc["verdict"], snap, flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
