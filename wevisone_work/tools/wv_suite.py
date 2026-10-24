#!/usr/bin/env python3
"""Round-2 runtime gate on the Mac: a WeVisOne-VL-2B bundle through PyPI litert-lm 0.18.1, one case per process.

Image rows (29): `litert-lm run <bundle> --prompt <row prompt> --attachment <png> --backend B --vision-backend V
--cache no --temperature 0 --seed 0 [--cpu-thread-count N] < /dev/null`, attachment = the B arm's 512 PNG
(out/r2/png512, PIL BICUBIC) or, with --resize runtime, the original fixture image (the runtime's stb CATMULLROM
resize). Text rows: the 9 single-turn questions through the same CLI (no attachment); two_turn through the Python API
of the same wheel in one process (two send_message calls on one conversation, the CLI's own Engine /
create_conversation / SamplerConfig arguments: tools/wv_rt_api.py); `long` = the paragraph summary (> 300 tokens).
Outputs r2/suite/<label>/<row>.json in the shape tools/score_r2.py reads (response, new_ids = the HF tokenizer's ids of
the answer + <|im_end|>, n_new, eos, truncated) plus the command, rc, wall time and the stderr scan (error lines,
`not fully delegated`, delegate / fallback lines). Rows whose json exists are skipped.

    python3 tools/wv_suite.py --bundle out/r2/bundle/X.litertlm --label wi8_cpu --backend cpu --vision-backend cpu \
        [--rows all] [--text] [--long] [--resize png|runtime] [--jobs 3] [--verbose]
"""
import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

K = Path(__file__).resolve().parents[1]
RUN = Path.home() / "venvs/lt0181run/bin/litert-lm"
API_PY = Path.home() / "venvs/lt0181run/bin/python3"
TOKENIZER = Path("~/.cache/huggingface/hub/models--tencent--WeVisOne-VL-2B/snapshots/"
                 "3dad9ce8bcb6fbe0aea945bf8860bd2cfc3663bf/tokenizer.json").expanduser()
IM_END = 151645
SCAN = re.compile(r"not fully delegated|fully delegated|Replacing \d+|delegat|fallback|Unsupported|unsupported|"
                  r"Validation error|Shape mismatch|ERROR|Error|error|INVALID|INTERNAL|nan|NaN", re.I)


def tok():
    from tokenizers import Tokenizer
    return Tokenizer.from_file(str(TOKENIZER))


def text_rows():
    sys.path.insert(0, str(K / "tools"))
    rows = []
    for p in sorted((K / "ref" / "T0").glob("*.json")):
        t0 = json.loads(p.read_text())
        rows.append((t0["row"], t0["turns"], t0["check_regex"], t0))
    return rows


def scan(stderr):
    hits = [l for l in stderr.splitlines() if SCAN.search(l)]
    return {"n_lines": len(stderr.splitlines()), "hits": hits[:60], "n_hits": len(hits),
            "not_fully_delegated": sum("not fully delegated" in l for l in stderr.splitlines()),
            "validation_error": sum("Validation error" in l for l in stderr.splitlines())}


def run_cli(a, prompt, attachment, log_path):
    cmd = [str(RUN), "run", str(K / a.bundle), "--prompt", prompt, "--backend", a.backend, "--cache", "no",
           "--temperature", "0", "--seed", "0"]
    if attachment:
        cmd += ["--attachment", str(attachment)]
    if a.vision_backend:
        cmd += ["--vision-backend", a.vision_backend]
    if a.backend == "cpu" and a.cpu_threads:
        cmd += ["--cpu-thread-count", str(a.cpu_threads)]
    if a.verbose:
        cmd += ["--verbose"]
    t0 = time.time()
    pr = subprocess.run(cmd, capture_output=True, text=True, timeout=1800, stdin=subprocess.DEVNULL)
    wall = round(time.time() - t0, 1)
    log_path.write_text(pr.stderr)
    return cmd, pr.returncode, pr.stdout, pr.stderr, wall


def record(tk, answer, extra):
    ids = tk.encode(answer, add_special_tokens=False).ids + [IM_END]
    rec = {"response": answer + "<|im_end|>", "answer": answer, "new_ids": ids, "n_new": len(ids), "eos": True,
           "truncated": False}
    rec.update(extra)
    return rec


def image_case(a, tk, row, out_dir, logs):
    path = out_dir / f"{row['id']}.json"
    if path.exists():
        return None
    if a.resize == "runtime":
        att = K / "fixtures" / row["image"]
    else:
        att = K / a.png_dir / f"{Path(row['image']).stem}.png"
    cmd, rc, out, err, wall = run_cli(a, row["prompt"], att, logs / f"{row['id']}.stderr.log")
    answer = out.strip()
    rec = record(tk, answer, {"row": row["id"], "task": row["task"], "cmd": cmd, "rc": rc, "wall_s": wall,
                              "attachment": str(att.relative_to(K)),
                              "attachment_sha256": hashlib.sha256(att.read_bytes()).hexdigest(),
                              "stderr_scan": scan(err)})
    path.write_text(json.dumps(rec, indent=1, ensure_ascii=False) + "\n")
    return f"{row['id']}: rc {rc} {wall} s | {answer[:110]!r}"


def text_case(a, tk, rid, turns, pat, t0, out_dir, logs):
    path = out_dir / "text" / f"{rid}.json"
    if path.exists():
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    if len(turns) == 1:
        cmd, rc, out, err, wall = run_cli(a, turns[0], None, logs / f"text_{rid}.stderr.log")
        answers = [out.strip()]
    else:
        cmd = [str(API_PY), "-I", str(K / "tools" / "wv_rt_api.py"), "--bundle", str(K / a.bundle), "--backend",
               a.backend, "--turns-json", json.dumps(turns)]
        if a.vision_backend:
            cmd += ["--vision-backend", a.vision_backend]
        if a.backend == "cpu" and a.cpu_threads:
            cmd += ["--cpu-thread-count", str(a.cpu_threads)]
        t1 = time.time()
        pr = subprocess.run(cmd, capture_output=True, text=True, timeout=1800, stdin=subprocess.DEVNULL)
        wall, rc, err = round(time.time() - t1, 1), pr.returncode, pr.stderr
        (logs / f"text_{rid}.stderr.log").write_text(err)
        try:
            answers = json.loads(pr.stdout.strip().splitlines()[-1])["answers"]
        except Exception:  # noqa: BLE001
            answers = [pr.stdout.strip()]
    ref_answers = [r["answer"] for r in t0["records"]]
    final = answers[-1]
    rec = {"row": rid, "turns": turns, "answers": answers, "ref_answers": ref_answers,
           "pass": bool(re.search(pat, final.lower())), "ref_pass": t0["pass"],
           "same_text_as_T0": [x.strip() == y.strip() for x, y in zip(answers, ref_answers)],
           "ids": [tk.encode(x, add_special_tokens=False).ids for x in answers], "cmd": cmd, "rc": rc, "wall_s": wall,
           "stderr_scan": scan(err)}
    path.write_text(json.dumps(rec, indent=1, ensure_ascii=False) + "\n")
    return f"text {rid}: pass {rec['pass']} (T0 {t0['pass']}) same {rec['same_text_as_T0']} | {final[:80]!r}"


def long_case(a, tk, out_dir, logs):
    path = out_dir / "text" / "long_summary.json"
    if path.exists():
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    t0 = json.loads((K / "r2" / "ref" / "T0" / "long_summary.json").read_text())
    t2 = json.loads((K / "r2" / "ref" / "T2" / "long_summary.json").read_text())
    cmd, rc, out, err, wall = run_cli(a, t0["turns"][0], None, logs / "text_long_summary.stderr.log")
    answer = out.strip()
    ids = tk.encode(answer, add_special_tokens=False).ids
    rec = {"row": "long_summary", "answer": answer, "ids": ids,
           "same_text_as_T0": answer == t0["records"][0]["answer"],
           "same_text_as_T2": answer == t2["records"][0]["answer"],
           "T0_answer": t0["records"][0]["answer"], "T2_answer": t2["records"][0]["answer"], "cmd": cmd, "rc": rc,
           "wall_s": wall, "stderr_scan": scan(err)}
    path.write_text(json.dumps(rec, indent=1, ensure_ascii=False) + "\n")
    return f"long: same as T0 {rec['same_text_as_T0']} T2 {rec['same_text_as_T2']} | {answer[:100]!r}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--backend", default="cpu")
    ap.add_argument("--vision-backend", default=None)
    ap.add_argument("--cpu-threads", type=int, default=4)
    ap.add_argument("--rows", default="")
    ap.add_argument("--text", action="store_true")
    ap.add_argument("--long", action="store_true")
    ap.add_argument("--resize", choices=("png", "runtime"), default="png")
    ap.add_argument("--png-dir", default="out/r2/png512", help="pre-resized attachments (out/r2/png768 for 768)")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()
    out_dir = K / "r2" / "suite" / a.label
    logs = out_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    bundle = K / a.bundle
    meta = {"label": a.label, "bundle": a.bundle, "bundle_bytes": bundle.stat().st_size, "backend": a.backend,
            "vision_backend": a.vision_backend, "cpu_threads": a.cpu_threads if a.backend == "cpu" else None,
            "resize": a.resize, "runtime": subprocess.run([str(RUN), "--version"], capture_output=True, text=True,
                                                          stdin=subprocess.DEVNULL).stdout.strip(),
            "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    tk = tok()
    jobs = []
    rows = json.loads((K / "fixtures" / "fixtures.json").read_text())["rows"]
    want = None if a.rows in ("", "all") else set(a.rows.split(","))
    for row in rows if a.rows else []:
        if not want or row["id"] in want:
            jobs.append(("image", row))
    if a.text:
        for rid, turns, pat, t0 in text_rows():
            jobs.append(("text", (rid, turns, pat, t0)))
    if a.long:
        jobs.append(("long", None))

    def run(job):
        kind, x = job
        if kind == "image":
            return image_case(a, tk, x, out_dir, logs)
        if kind == "text":
            return text_case(a, tk, *x, out_dir, logs)
        return long_case(a, tk, out_dir, logs)
    with cf.ThreadPoolExecutor(max_workers=a.jobs) as ex:
        for msg in ex.map(run, jobs):
            if msg:
                print(time.strftime("%H:%M:%S"), a.label, msg, flush=True)
    meta["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    print("SUITE_DONE", a.label, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
