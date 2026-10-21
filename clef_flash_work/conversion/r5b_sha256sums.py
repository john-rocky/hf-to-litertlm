#!/usr/bin/env python3
"""Round 5b: staging/upload/SHA256SUMS (the form of the earlier repositories: `<sha256>  ./<path>`, sorted).

    python3 scripts/r5b_sha256sums.py [--write]     -> results/r5b_sha256sums.json (+ staging/upload/SHA256SUMS with --write)

Every file of the folder but .venv, __pycache__, .cache and SHA256SUMS itself. A file that is a symlink to a large file
takes its sha256 from the F_NOCACHE re-reads of this round (results/r5b_sums_L<L>.json, results/r5b_sums_misc.json:
the bytes read back from the real copy, the HDD's included) and that value must equal the work directory's ledger
(results/files.json, files_r3.json, files_r5a.json, host/contract.json for the checkpoint's joint head); a symlink
without a re-read is reported as pending and --write refuses. The small files are hashed here."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

K = Path(__file__).resolve().parents[1]
UP = K / "staging/upload"
sys.path.insert(0, str(K / "scripts"))
SKIP = {".venv", "__pycache__", ".cache"}


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    reread = {}
    for f in sorted((K / "results").glob("r5b_sums_*.json")):
        for path, r in json.loads(f.read_text())["files"].items():
            reread[Path(r["realpath"]).resolve()] = dict(r, record=f.name, staged=path)
    stage = json.loads((K / "results/r5b_stage.json").read_text())
    recorded = {it["path"]: it for it in stage["core"]["items"] if it["kind"] == "link"}
    rows, pending, mismatch = [], [], []
    for p in sorted(UP.rglob("*")):
        rel = p.relative_to(UP)
        if any(part in SKIP for part in rel.parts) or p.name == "SHA256SUMS" or p.is_dir():
            continue
        row = {"path": f"./{rel.as_posix()}", "bytes": p.stat().st_size}
        if p.is_symlink():
            real = p.resolve()
            rec = recorded.get(rel.as_posix())
            row.update(link_target=os.readlink(p), ledger_sha256=rec["sha256"] if rec else None,
                       ledger=rec["sha256_from"] if rec else None)
            r = reread.get(real)
            if r is None:
                pending.append(row["path"])
                row["sha256"] = None
            else:
                row.update(sha256=r["sha256"], read_back=r["record"], f_nocache=r["f_nocache"],
                           bytes_read=r["bytes"])
                if rec is None or r["sha256"] != rec["sha256"] or r["bytes"] != row["bytes"]:
                    mismatch.append(row["path"])
        else:
            row.update(sha256=sha256(p), read_back="hashed here")
        rows.append(row)
    doc = {"what": "round 5b: SHA256SUMS of staging/upload", "files": len(rows),
           "bytes": sum(r["bytes"] for r in rows), "pending": pending, "mismatch": mismatch, "rows": rows}
    (K / "results/r5b_sha256sums.json").write_text(json.dumps(doc, indent=1) + "\n")
    print(json.dumps({k: doc[k] for k in ("files", "bytes", "pending", "mismatch")}, indent=1))
    if a.write:
        if pending or mismatch:
            raise SystemExit("not written: pending or mismatching files")
        lines = sorted(f"{r['sha256']}  {r['path']}" for r in rows)
        lines.sort(key=lambda ln: ln.split("  ", 1)[1].encode())
        (UP / "SHA256SUMS").write_text("\n".join(lines) + "\n")
        print(f"wrote staging/upload/SHA256SUMS ({len(lines)} lines)")
    return 0 if not mismatch else 1


if __name__ == "__main__":
    raise SystemExit(main())
