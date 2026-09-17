#!/usr/bin/env python3
"""Upload the VibeVoice-ASR-Streaming-1.5B bundles + card + LICENSE to litert-community, verifying the
remote LFS sha256 against the local file (a truncated transfer looks like a model bug) and reading
the card back byte-identical.  The manifest is uploaded separately (make_manifest.py --public).
    ~/venvs/lt094dev/bin/python3 vibevoice_asr_streaming_work/upload_hf.py [--dry-run]
"""
import hashlib
import os
import sys

from huggingface_hub import HfApi, hf_hub_download

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
REPO = "litert-community/VibeVoice-ASR-Streaming-1.5B"
CARD = os.path.join(ROOT, "cards", "vibevoice-asr-streaming-1.5b-litert.md")
FILES = [
    (os.path.join(HERE, "ship/VibeVoice-ASR-Streaming-1.5B.litertlm"), "VibeVoice-ASR-Streaming-1.5B.litertlm",
     "VibeVoice-ASR-Streaming-1.5B: int4-b128 LM + int8 audio encoder (3.47 s window), streaming multi-turn protocol"),
    (os.path.join(HERE, "ship/VibeVoice-ASR-Streaming-1.5B_int8.litertlm"), "VibeVoice-ASR-Streaming-1.5B_int8.litertlm",
     "VibeVoice-ASR-Streaming-1.5B int8 LM (desktop build)"),
    (os.path.join(HERE, "ship/LICENSE"), "LICENSE", "MIT LICENSE (inherited from microsoft/VibeVoice)"),
]


def sha256(path, chunk=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main():
    dry = "--dry-run" in sys.argv
    for local, dest, _ in FILES:
        assert os.path.exists(local), local
        print(f"{dest}: {os.path.getsize(local)} bytes sha256 {sha256(local)}", flush=True)
    assert "{{" not in open(CARD).read(), "card still has placeholders"
    if dry:
        print("DRY RUN — nothing uploaded")
        return
    api = HfApi()
    api.create_repo(REPO, repo_type="model", exist_ok=True)
    api.upload_file(path_or_fileobj=CARD, path_in_repo="README.md", repo_id=REPO,
                    commit_message="VibeVoice-ASR-Streaming-1.5B LiteRT-LM card")
    back = open(hf_hub_download(REPO, "README.md", force_download=True), "rb").read()
    assert back == open(CARD, "rb").read(), "card read-back differs"
    print("card uploaded + read back identical", flush=True)
    for local, dest, msg in FILES:
        local_sha = sha256(local)
        api.upload_file(path_or_fileobj=local, path_in_repo=dest, repo_id=REPO, commit_message=msg)
        info = api.get_paths_info(REPO, [dest])
        if info and info[0].lfs:
            remote = info[0].lfs.sha256
            print(f"{dest}: remote sha256 {remote} size {info[0].lfs.size}", flush=True)
            if remote != local_sha or info[0].lfs.size != os.path.getsize(local):
                print("SHA/SIZE MISMATCH — upload not trustworthy", flush=True)
                sys.exit(1)
        else:
            print(f"{dest}: non-LFS, size {info[0].size}", flush=True)
    print("UPLOAD_OK https://huggingface.co/" + REPO)


if __name__ == "__main__":
    main()
