#!/usr/bin/env python3
"""Run manifest/make_manifest.py for a model repo that is not on the Hub yet.

make_manifest.py lists the repo's files through the HF API before it reads anything, also with --local-file (the
tree request is unconditional). litert-community/Fun-ASR-Nano-2512 is created only after the user's GO, so that
request answers 401 today. This wrapper replaces only the file listing, and only when the Hub answers 401/404: the
listing is then the set of --local-file names. Every derived value (sha256, size, sections, context length,
capabilities) is still read out of the local bundle by make_manifest.py's own code. make_manifest.py is not edited.

  .venv-092/bin/python3 funasr_nano_work/make_manifest_local.py litert-community/Fun-ASR-Nano-2512 \\
      --curated manifest/curated/litert-community__Fun-ASR-Nano-2512.json \\
      --local-file Fun-ASR-Nano-2512.litertlm=funasr_nano_work/out/bundle/Fun-ASR-Nano-2512.litertlm \\
      [--public --out funasr_nano_work/ship/litertlm_manifest.json]
"""
import os
import sys
import urllib.error

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "manifest"))
import make_manifest as mm  # noqa: E402


def local_names(argv):
    names = []
    for i, a in enumerate(argv):
        if a == "--local-file" and i + 1 < len(argv):
            names.append(argv[i + 1].split("=", 1)[0])
        elif a.startswith("--local-file="):
            names.append(a.split("=", 2)[1])
    return names


NAMES = local_names(sys.argv[1:])
_http_json = mm.http_json


def http_json(url):
    try:
        return _http_json(url)
    except urllib.error.HTTPError as e:
        if "/api/models/" in url and url.endswith("/tree/main") and e.code in (401, 404) and NAMES:
            print(f"NOTE: {url} -> HTTP {e.code} (repo not on the Hub yet); file list = --local-file names {NAMES}",
                  file=sys.stderr)
            return [{"path": n, "type": "file"} for n in NAMES]
        raise


mm.http_json = http_json

if __name__ == "__main__":
    mm.main()
