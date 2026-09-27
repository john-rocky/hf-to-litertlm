#!/usr/bin/env python3
"""Run the ```python block of a VibeVoice-ASR card (Hub README or local card) as written, in a given litert-lm-api venv.

The block is extracted from the markdown, compiled under the name `card_python_block` (so a traceback names the
block's own line numbers) and exec'd in a scratch directory that holds the bundle (symlink, under the file name the
block opens) and `clip.wav` (a copy of one fixture from ../vibevoice_asr_work/fixtures/). Nothing in the block is
edited except the substitutions listed in the JSON output under "substitutions":

- streaming card: none. The block calls two functions it tells the reader to supply ("your decoder"),
  `load_mono_float_24k(path)` and `write_wav(path, samples, sr)`; this harness defines them in the exec namespace
  (pure python, 16-bit PCM in/out, exact round trip for the fixtures).
- bitnet card: the placeholder path "/abs/path/clip.wav" becomes the scratch clip's absolute path, and `dur = 5.86`
  (the card's example value = clip00's length) is rewritten to the chosen clip's length when another clip is used.
- `--cpu`: `backend=GPU()` -> `backend=CPU()` (the streaming block puts the LM on the GPU).

The printed transcript (one line per window for the streaming card, one line for the bitnet card) is compared with a
gate JSON of this repo (`--expect`, rows[].chunks / rows[].hyp for the same clip id). Any exception inside the block is
caught and reported with the failing block line. The scratch directory (the runtime writes its cache next to the
model path) is deleted at the end.

  ~/venvs/lt0171run/bin/python verify_card_snippet.py --card streaming --readme <README.md> --bundle <.litertlm> \
      --expect mac_gate_int4_gpu_cpu_0161.json [--clip clip00] [--cpu] --out <result.json>
"""
import argparse
import array
import contextlib
import io
import json
import os
import re
import shutil
import struct
import sys
import tempfile
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURES = os.path.join(os.path.dirname(HERE), "vibevoice_asr_work", "fixtures")
BUNDLE_NAME = {"streaming": "VibeVoice-ASR-Streaming-1.5B.litertlm", "bitnet": "VibeVoice-ASR-BitNet.litertlm"}
SR = 24000


def read_wav_int16(path):
    raw = open(path, "rb").read()
    assert raw[:4] == b"RIFF" and raw[8:12] == b"WAVE", path
    i, sr, ch, bits, data = 12, None, None, None, None
    while i + 8 <= len(raw):
        cid, sz = raw[i:i + 4], struct.unpack("<I", raw[i + 4:i + 8])[0]
        body = raw[i + 8:i + 8 + sz]
        if cid == b"fmt ":
            _tag, ch, sr, _br, _ba, bits = struct.unpack("<HHIIHH", body[:16])
        elif cid == b"data":
            data = body
        i += 8 + sz + (sz & 1)
    assert sr == SR and ch == 1 and bits == 16, (path, sr, ch, bits)
    a = array.array("h")
    a.frombytes(data[:len(data) // 2 * 2])
    if sys.byteorder != "little":
        a.byteswap()
    return a


def load_mono_float_24k(path):
    """The reader's decoder in the streaming card: mono float samples at 24 kHz, any length."""
    return [s / 32768.0 for s in read_wav_int16(path)]


def write_wav(path, samples, sr):
    """The reader's writer in the streaming card: 16-bit PCM mono."""
    a = array.array("h", [max(-32768, min(32767, int(round(x * 32768.0)))) for x in samples])
    if sys.byteorder != "little":
        a.byteswap()
    data = a.tobytes()
    with open(path, "wb") as f:
        f.write(b"RIFF" + struct.pack("<I", 36 + len(data)) + b"WAVE")
        f.write(b"fmt " + struct.pack("<IHHIIHH", 16, 1, 1, sr, sr * 2, 2, 16))
        f.write(b"data" + struct.pack("<I", len(data)) + data)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--card", choices=["streaming", "bitnet"], required=True)
    ap.add_argument("--readme", required=True, help="markdown file whose first ```python block is run")
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--clip", default="clip00")
    ap.add_argument("--expect", help="gate JSON (rows[].chunks or rows[].hyp) to compare the printed transcript with")
    ap.add_argument("--cpu", action="store_true", help="substitute backend=GPU() with backend=CPU() in the block")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    md = open(args.readme).read()
    block = re.search(r"```python\n(.*?)```", md, re.S).group(1)
    meta = {os.path.basename(m["file"]).split(".")[0]: m for m in json.load(open(os.path.join(FIXTURES, "meta.json")))}
    clip = meta[args.clip]
    src_wav = os.path.join(os.path.dirname(FIXTURES), clip["file"])
    n = len(read_wav_int16(src_wav))
    dur = n / SR
    tmp = tempfile.mkdtemp(prefix=f"vv_card_{args.card}_", dir=os.environ.get("TMPDIR"))
    clip_path = os.path.join(tmp, "clip.wav")
    subs = []
    code = block
    if args.card == "bitnet":
        code = code.replace('"/abs/path/clip.wav"', json.dumps(clip_path))
        subs.append(f'"/abs/path/clip.wav" -> {clip_path}')
        if f"dur = {dur:.2f}" not in code:
            code = re.sub(r"dur = [0-9.]+", f"dur = {dur:.2f}", code, count=1)
            subs.append(f"dur = {dur:.2f} (clip length)")
    if args.cpu:
        assert "backend=GPU()" in code, "block has no backend=GPU() to substitute"
        code = code.replace("backend=GPU()", "backend=CPU()", 1)
        subs.append("backend=GPU() -> backend=CPU()")
    import importlib.metadata
    doc = {
        "card": args.card, "readme": os.path.abspath(args.readme), "bundle": os.path.abspath(args.bundle),
        "clip": args.clip, "clip_id": clip["id"], "clip_seconds": round(dur, 3), "substitutions": subs,
        "litert_lm_api": importlib.metadata.version("litert-lm-api"), "python": sys.executable,
        "block_sha1": __import__("hashlib").sha1(block.encode()).hexdigest()[:12],
    }
    cwd = os.getcwd()
    printed, err = "", None
    try:
        os.symlink(os.path.abspath(args.bundle), os.path.join(tmp, BUNDLE_NAME[args.card]))
        shutil.copy(src_wav, clip_path)
        os.chdir(tmp)
        ns = {"__name__": "__card__", "load_mono_float_24k": load_mono_float_24k, "write_wav": write_wav}
        buf = io.StringIO()
        t0 = time.time()
        try:
            with contextlib.redirect_stdout(buf):
                exec(compile(code, "card_python_block", "exec"), ns)
        except Exception as e:  # report the block line that raised
            tb = traceback.extract_tb(sys.exc_info()[2])
            frames = [f for f in tb if f.filename == "card_python_block"]
            line_no = frames[-1].lineno if frames else None
            line_src = code.splitlines()[line_no - 1] if line_no else None
            err = {"type": type(e).__name__, "message": str(e), "block_line": line_no, "block_source": line_src}
        doc["wall_s"] = round(time.time() - t0, 2)
        printed = buf.getvalue()
        if "resp" in ns:  # the last send_message() return value, as the venv's API produced it
            r = ns["resp"]
            doc["response_type"] = type(r).__module__ + "." + type(r).__name__
            doc["response_json"] = json.dumps(r, default=repr, ensure_ascii=False)
        if "conv" in ns and err is not None:
            with contextlib.suppress(Exception):
                ns["conv"].close()
    finally:
        os.chdir(cwd)
        doc["scratch_files_at_end"] = sorted(os.listdir(tmp))
        shutil.rmtree(tmp)
    doc["printed"] = printed
    doc["error"] = err
    lines = printed.splitlines()
    if args.expect:
        rows = {r["id"]: r for r in json.load(open(args.expect))["rows"]}
        row = rows[clip["id"]]
        if args.card == "streaming":
            want = row["chunks"]
        else:
            want = [row["hyp"]]
        # the block prints one turn per print() call; a turn's text may itself contain a newline (" \n Speaker 0:")
        doc["expect"] = {"file": os.path.abspath(args.expect), "turns": want, "printed": "".join(t + "\n" for t in want)}
        doc["matches_expect"] = printed == doc["expect"]["printed"]
    status = "ERROR" if err else ("PASS" if doc.get("matches_expect", True) else "MISMATCH")
    doc["status"] = status
    json.dump(doc, open(args.out, "w"), ensure_ascii=False, indent=1)
    print(f"[{status}] litert-lm-api {doc['litert_lm_api']} card={args.card} clip={args.clip} subs={subs}")
    if err:
        print(f"  block line {err['block_line']}: {err['block_source']!r}\n  {err['type']}: {err['message']}")
    print(f"  printed: {printed!r}")
    if args.expect:
        print(f"  expect : {doc['expect']['printed']!r}")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
