#!/usr/bin/env python3
"""Round 5b: lay out the upload folder (staging/upload) in the order of the earlier LiteRT decision-model repositories
(graphs at the root, tables/, head/, tokenizer/, host/, examples/, fixtures/, conversion/), the large files as symlinks
to their real copies (the row graphs on the HDD, the towers and the vocabulary tables on this disk, the joint head on
the HDD under the checkpoint's blobs) and the small ones copied or written here.

    venv-host/bin/python scripts/r5b_stage.py core      # graphs, tables, head, tokenizer, host package, requirements
    venv-host/bin/python scripts/r5b_stage.py check     # every staged file against its source: bytes, link target,
                                                        # the ledgers' sha256 for the large files (not re-read here)
    -> results/r5b_stage.json (what was staged, from where, with which recorded sha256)

The row graphs and towers keep their bytes; only the names change (FOLDER in host/clef_litert/runtime.py). The towers'
position rows (cache/r3/consts_g{256,448}.npz `pos_embeds`, the round-3 tower input `pos`) are written as one-key F32
safetensors without metadata. Nothing in out/, cache/ or the HDD is written."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

K = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(K / "host"))
import clef_litert as C  # noqa: E402

UP = K / "staging/upload"
SNAP = K / "hf/hub/models--Cloudflare--clef-flash/snapshots/f0b7d3cd65296abfe459e6c228b618f5d8b26d2f"
ROWS = {L: f"out/r2/clef_rowprefill_L{L}_v2_fp16fc.tflite" for L in (256, 512, 1024, 2048, 4096)}
TOWERS = {t: f"out/r3/clef_tower_g{t}_v2_fp16fc.tflite" for t in (256, 448)}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def ledgers() -> dict:
    """path -> (bytes, sha256) from the work directory's ledgers (results/files.json, files_r3.json, files_r5a.json)."""
    out = {}
    for x in json.loads((K / "results/files.json").read_text())["files"]:
        if isinstance(x.get("sha256"), str):
            out[x["path"]] = (x["bytes"], x["sha256"])
    for p, x in json.loads((K / "results/files_r3.json").read_text())["files"].items():
        out[p] = (x["bytes"], x["sha256"])
    for x in json.loads((K / "results/files_r5a.json").read_text())["files"]:
        if isinstance(x.get("sha256"), str):
            out[x["path"]] = (x["bytes"], x["sha256"])
    return out


def link(dst: Path, target: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink() or dst.exists():
        if dst.is_symlink() and os.readlink(dst) == str(target):
            return
        raise SystemExit(f"{dst} exists and is not the link to {target}")
    os.symlink(target, dst)


def copy(dst: Path, src: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.read_bytes() == src.read_bytes():
        return
    shutil.copyfile(src, dst)


def plan() -> list[dict]:
    """Every core file: staged path, kind (link / copy / write), source, the recorded size and sha256 if any."""
    led = ledgers()
    items = []
    for L, src in ROWS.items():
        b, s = led[src]
        items.append({"path": C.FOLDER["row"].format(length=L), "kind": "link", "source": src,
                      "target": str((K / src).resolve()), "bytes": b, "sha256": s, "sha256_from": "results/files.json"})
    for t, src in TOWERS.items():
        b, s = led[src]
        items.append({"path": C.FOLDER["tower"].format(tile=t), "kind": "link", "source": src,
                      "target": str((K / src).resolve()), "bytes": b, "sha256": s, "sha256_from": "results/files_r3.json"})
    for name in ("embed_tokens", "lm_head"):
        src = f"out/r5a/{name}.safetensors"
        b, s = led[src]
        items.append({"path": C.FOLDER[name], "kind": "link", "source": src, "target": str((K / src).resolve()),
                      "bytes": b, "sha256": s, "sha256_from": "results/files_r5a.json"})
    hub = {r["file"]: r for r in json.loads((K / "results/checkpoint_check.json").read_text())["files"]}
    items.append({"path": C.FOLDER["joint_head"], "kind": "link", "source": "checkpoint joint_head.safetensors",
                  "target": str((SNAP / "joint_head.safetensors").resolve()),
                  "bytes": hub["joint_head.safetensors"]["bytes"], "sha256": hub["joint_head.safetensors"]["sha256"],
                  "sha256_from": "results/checkpoint_check.json (= the Hub's LFS oid at f0b7d3cd)"})
    for name, fn in (("joint_head_config", "joint_head_config.json"), ("tokenizer", "tokenizer.json")):
        items.append({"path": C.FOLDER[name], "kind": "copy", "source": f"checkpoint {fn}", "src": str(SNAP / fn),
                      "bytes": hub[fn]["bytes"], "sha256": hub[fn]["sha256"],
                      "sha256_from": "results/checkpoint_check.json"})
    for t in (256, 448):
        items.append({"path": C.FOLDER["tower_pos"].format(tile=t), "kind": "write",
                      "source": f"cache/r3/consts_g{t}.npz pos_embeds", "src": str(K / f"cache/r3/consts_g{t}.npz")})
    for p in sorted((K / "host/clef_litert").glob("*.py")):
        items.append({"path": f"host/clef_litert/{p.name}", "kind": "copy", "source": f"host/clef_litert/{p.name}",
                      "src": str(p)})
    items.append({"path": "host/requirements-host.txt", "kind": "copy", "source": "host/requirements-host.txt",
                  "src": str(K / "host/requirements-host.txt")})
    items.append({"path": "LICENSE", "kind": "copy", "source": "checkpoint LICENSE", "src": str(SNAP / "LICENSE")})
    return items


def do_core() -> dict:
    items = plan()
    for it in items:
        dst = UP / it["path"]
        if it["kind"] == "link":
            link(dst, Path(it["target"]))
        elif it["kind"] == "copy":
            copy(dst, Path(it["src"]))
        else:
            arr = np.load(it["src"])["pos_embeds"].astype(np.float32)
            if dst.exists():
                if not np.array_equal(C.load_f32_tensor(dst), arr):
                    raise SystemExit(f"{dst} exists with other values")
            else:
                dst.parent.mkdir(parents=True, exist_ok=True)
                C.runtime.write_f32_tensor(dst, "pos", arr)
            it["shape"] = list(arr.shape)
    return {"items": items}


def do_check() -> dict:
    rows = []
    for it in plan():
        dst = UP / it["path"]
        r = {"path": it["path"], "kind": it["kind"], "present": dst.exists() or dst.is_symlink()}
        if not r["present"]:
            rows.append(r)
            continue
        if it["kind"] == "link":
            r["link_target"] = os.readlink(dst)
            r["link_ok"] = r["link_target"] == it["target"] and Path(it["target"]).exists()
            r["bytes_ok"] = dst.stat().st_size == it["bytes"]
            r["sha256_recorded"] = it["sha256"]
        elif it["kind"] == "copy":
            r["sha256"] = sha256(dst)
            r["equal_to_source"] = r["sha256"] == sha256(Path(it["src"]))
            if "sha256" in it and it.get("sha256_from"):
                r["equal_to_record"] = r["sha256"] == it["sha256"]
        else:
            arr = np.load(it["src"])["pos_embeds"].astype(np.float32)
            got = C.load_f32_tensor(dst)
            r["values_equal_to_source"] = bool(np.array_equal(got.view(np.uint32), arr.view(np.uint32)))
            r["shape"] = list(got.shape)
            r["sha256"] = sha256(dst)
        rows.append(r)
    ok = all(r["present"] and r.get("link_ok", True) and r.get("bytes_ok", True) and r.get("equal_to_source", True)
             and r.get("equal_to_record", True) and r.get("values_equal_to_source", True) for r in rows)
    return {"rows": rows, "pass": ok}


def main() -> int:
    what = sys.argv[1] if len(sys.argv) > 1 else "check"
    t = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    out = K / "results/r5b_stage.json"
    doc = json.loads(out.read_text()) if out.exists() else {"what": "round 5b: the staged upload folder"}
    if what == "core":
        doc["core"] = do_core() | {"at": t}
    doc["check"] = do_check() | {"at": t}
    out.write_text(json.dumps(doc, indent=1) + "\n")
    bad = [r for r in doc["check"]["rows"] if not (r["present"] and r.get("link_ok", True) and r.get("bytes_ok", True)
                                                   and r.get("equal_to_source", True) and r.get("equal_to_record", True)
                                                   and r.get("values_equal_to_source", True))]
    print(json.dumps({"files": len(doc["check"]["rows"]), "pass": doc["check"]["pass"], "bad": bad}, indent=1))
    return 0 if doc["check"]["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
