#!/usr/bin/env python3
"""One file's size, LFS sha256 and the repo revision, from a Hub model-info JSON (`/api/models/<repo>?blobs=true`).

  hub_file.py <api.json> <file>   ->  <size> TAB <lfs size> TAB <lfs sha256> TAB <repo revision>
Exit 0 = listed, 1 = the file is not in the repo's siblings (or has no LFS entry), 2 = unreadable JSON.
"""
import json
import sys


def main():
    try:
        info = json.load(open(sys.argv[1], encoding="utf-8"))
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}")
        return 2
    for s in info.get("siblings") or []:
        if s.get("rfilename") == sys.argv[2]:
            lfs = s.get("lfs") or {}
            if not lfs.get("sha256"):
                print(f"ERROR: {sys.argv[2]} has no LFS sha256 in the model info")
                return 1
            print(f"{s.get('size', '')}\t{lfs.get('size', '')}\t{lfs['sha256']}\t{info.get('sha', '')}")
            return 0
    print(f"ERROR: {sys.argv[2]} is not listed in {info.get('id', '?')} at {info.get('sha', '?')}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
