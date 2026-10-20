#!/usr/bin/env python3
"""Rebuild the bundle a lock describes and compare the bytes: MATCH or DIFF.

    uv run python scripts/verify_lock.py out/<model>/<model>.lock.json [--keep] [--json OUT]
    hf-to-litertlm verify out/<model>/<model>.lock.json    (the same, from the package)

1. This interpreter's toolchain must be the lock's (toolchain id, python, every installed
   package version, uv.lock sha256). Otherwise it stops with FAIL and lists what differs: a lock
   is only a test of the toolchain that wrote it (to move to another one, raise the toolchain as
   docs/one-command-lock.md says).
2. scripts/convert.py <repo> --revision <sha> --recipe <id> --no-gate, into a fresh dir under out/
   (from the package: out/ in the current directory).
3. The whole-file sha256 equal -> MATCH. Otherwise DIFF, with each section's sha256 and the
   header's uuid / creation_timestamp of both files; the rebuilt file is kept for a closer look
   (tools/tfl_compare/ compares two TFLite sections).

Exit 0 = MATCH, 1 = DIFF or FAIL.
"""
import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import convert  # noqa: E402  (toolchain_info, converter_info, REPO_ROOT, OUT_ROOT)


def toolchain_diffs(want, have):
    out = [f"{k}: lock {want.get(k)!r}, here {have.get(k)!r}"
           for k in ("id", "python", "pinned_env", "uv_lock_sha256") if want.get(k) != have.get(k)]
    wp, hp = want.get("packages", {}), have.get("packages", {})
    out += [f"package {n}: lock {wp.get(n)}, here {hp.get(n)}"
            for n in sorted(set(wp) | set(hp)) if wp.get(n) != hp.get(n)]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("lock")
    ap.add_argument("--keep", action="store_true", help="keep the rebuilt dir after a MATCH too")
    ap.add_argument("--json", help="write the comparison here")
    args = ap.parse_args()
    if not Path(args.lock).is_file():
        print(f"FAIL lock: no such file {args.lock} -> next: pass the <model>.lock.json "
              "that convert.py wrote next to the bundle")
        return 1
    lock = json.loads(Path(args.lock).read_text())
    repo, rev, rid = lock["source"]["repo"], lock["source"]["revision"], lock["recipe"]["id"]
    name = repo.split("/")[-1]
    result = {"lock": args.lock, "repo": repo, "revision": rev, "recipe": rid, "verdict": None}

    diffs = toolchain_diffs(lock["toolchain"], convert.toolchain_info())
    if diffs:
        print(f"FAIL toolchain: this environment is not the lock's ({len(diffs)} differences)")
        for d in diffs[:40]:
            print("  " + d)
        result.update(verdict="FAIL", toolchain_diffs=diffs)
        _write(args.json, result)
        return 1
    here = convert.converter_info()
    if here["sha256"] != lock["converter"]["sha256"]:
        print(f"WARNING: scripts/convert.py is not the file that wrote the lock "
              f"({lock['converter']['sha256'][:12]} at {lock['converter']['repo_commit']}, "
              f"here {here['sha256'][:12]}); its guards change bytes — rebuilding anyway")

    out_root = convert.OUT_ROOT
    out_root.mkdir(exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=f"{name}.verify-", dir=out_root))
    cmd = [sys.executable, str(convert.REPO_ROOT / "scripts" / "convert.py"), repo,
           "--revision", rev, "--recipe", rid, "--out", str(tmp), "--no-gate"]
    print("rebuild: " + " ".join(cmd[1:]))
    t0 = time.time()
    proc = subprocess.run(cmd)
    result["rebuild_seconds"] = round(time.time() - t0)
    new_lock = tmp / f"{name}.lock.json"
    if proc.returncode != 0 or not new_lock.exists():
        print(f"FAIL rebuild: scripts/convert.py exit {proc.returncode} (output kept in {tmp})")
        result.update(verdict="FAIL", rebuild_dir=str(tmp))
        _write(args.json, result)
        return 1
    new = json.loads(new_lock.read_text())
    a, b = lock["bundle"], new["bundle"]
    result["sha256"] = {"lock": a["sha256"], "rebuild": b["sha256"]}
    for part in ("source", "recipe"):
        if {k: v for k, v in lock[part].items()} != {k: v for k, v in new[part].items()}:
            print(f"NOTE: the rebuild's {part} differs from the lock's")
            result[f"{part}_differs"] = True

    if a["sha256"] == b["sha256"]:
        result["verdict"] = "MATCH"
        print(f"MATCH {a['file']} {a['size']:,} B sha256 {a['sha256']} "
              f"(rebuilt in {result['rebuild_seconds']} s)")
        if not args.keep:
            shutil.rmtree(tmp, ignore_errors=True)
        else:
            result["rebuild_dir"] = str(tmp)
        _write(args.json, result)
        return 0

    result["verdict"] = "DIFF"
    result["rebuild_dir"] = str(tmp)
    print(f"DIFF {a['file']}\n  lock    {a['sha256']} {a['size']:,} B\n"
          f"  rebuild {b['sha256']} {b['size']:,} B  ({tmp})")
    rows = []
    for i in range(max(len(a["sections"]), len(b["sections"]))):
        sa = a["sections"][i] if i < len(a["sections"]) else {}
        sb = b["sections"][i] if i < len(b["sections"]) else {}
        same = sa.get("sha256") == sb.get("sha256") and sa.get("type") == sb.get("type")
        rows.append({"index": i, "type": sa.get("type") or sb.get("type"), "same": same,
                     "lock": sa, "rebuild": sb})
        print(f"  section {i} {rows[-1]['type']:<22} {'SAME' if same else 'DIFF'}  "
              f"{sa.get('size')} -> {sb.get('size')}  "
              f"{(sa.get('sha256') or '')[:16]} {(sb.get('sha256') or '')[:16]}")
    result["sections"] = rows
    for k in ("uuid", "creation_timestamp"):
        va, vb = a["system_metadata"].get(k), b["system_metadata"].get(k)
        print(f"  system_metadata {k}: lock {va} | rebuild {vb}")
    result["system_metadata"] = {"lock": a["system_metadata"], "rebuild": b["system_metadata"]}
    _write(args.json, result)
    return 1


def _write(path, result):
    if path:
        Path(path).write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    sys.exit(main())
