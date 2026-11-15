#!/usr/bin/env python3
"""The `sha256` item of TFLiteModel sections, for externalize_repo.sh --tflite-sha256-item.

  sha256_item.py set --toml IN.toml --out OUT.toml
  sha256_item.py check --litertlm FILE

LiteRT-LM's builder on main since 2026-10-31 (google-ai-edge/LiteRT-LM commit 7f9ae298; litert-lm-nightly
0.20.0.dev20261103 packs this way) writes a `sha256` item on every section, the sha256 of the section's bytes. The
unpack of litert-lm 0.19.1 and 0.19.2 copies that item into model.toml like any other item, and litert-lm-builder
0.19.1 and 0.19.2 write the TOML's items as they are. After the weights move, a TFLiteModel section holds the graph
without its weights, so the copied item is the hash of the section before the move. LiteRT-LM's ModelSync (sync/ on
main, in no release on 2026-11-15) hashes each section against its item by default (verify_checksums) and fails the
update on a mismatch (read in its source, not run); the runtime does not read the item.

set:   in every [[section]] block of IN.toml whose section_type is TFLiteModel and that carries a sha256 item, replace
       the item's value with the sha256 of the block's data_path file (the graph the section will hold: the packer
       writes that file's bytes as the section). A block without a sha256 item is left alone. OUT.toml must be in the
       same directory as IN.toml (data_path is relative to it). The rest of the text stays as it was. Prints one line
       per TFLiteModel block. Exit 0; 1 = OUT.toml is in another directory, the [[section]] lines do not match the
       parsed sections, a data_path file is missing, a block's parsed sha256 item is not found once in its text, or
       the new text does not parse.
check: read FILE's header; every TFLiteModel section that carries a sha256 item must carry the sha256 of its bytes
       [begin, end). Prints one line per TFLiteModel section and `SHA256-ITEM: PASS` or `SHA256-ITEM: FAIL`.
       Exit 0 = PASS, 1 = FAIL, 2 = unreadable.
"""
import argparse
import hashlib
import os
import re
import struct
import sys
import tomllib

SECTION_RE = re.compile(r"(?m)^\[\[section\]\][ \t]*$")
ITEM_RE = re.compile(r'(\{\s*key\s*=\s*"sha256"\s*,\s*value_type\s*=\s*"String"\s*,\s*value\s*=\s*")([^"]*)("\s*\})')


def sha_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(64 << 20), b""):
            h.update(b)
    return h.hexdigest()


def cmd_set(a):
    src = os.path.abspath(a.toml)
    out = os.path.abspath(a.out)
    if os.path.dirname(src) != os.path.dirname(out):
        print("--out must be in the same directory as --toml (data_path is relative to it)", file=sys.stderr)
        return 1
    text = open(src, encoding="utf-8").read()
    sections = tomllib.loads(text).get("section", [])
    starts = [m.start() for m in SECTION_RE.finditer(text)]
    if not starts or len(starts) != len(sections):
        print(f"{len(starts)} [[section]] lines but {len(sections)} parsed sections in {src}", file=sys.stderr)
        return 1
    bounds = list(zip(starts, starts[1:] + [len(text)]))
    pieces = [text[:starts[0]]]
    n_tfl = n_set = 0
    for i, ((b, e), sec) in enumerate(zip(bounds, sections)):
        block = text[b:e]
        if sec.get("section_type") == "TFLiteModel":
            n_tfl += 1
            dp = sec.get("data_path")
            path = os.path.join(os.path.dirname(src), dp) if isinstance(dp, str) else None
            if not path or not os.path.isfile(path):
                print(f"section {i}: TFLiteModel data_path {dp!r} is not a file", file=sys.stderr)
                return 1
            want = [m for m in sec.get("additional_metadata", []) if m.get("key") == "sha256"]
            found = ITEM_RE.findall(block)
            if len(want) != len(found) or len(found) > 1:
                print(f"section {i}: {len(want)} parsed sha256 items, {len(found)} in the text", file=sys.stderr)
                return 1
            if not found:
                print(f"section {i} TFLiteModel {dp}: no sha256 item, left alone")
            else:
                new = sha_file(path)
                old = found[0][1]
                block = ITEM_RE.sub(lambda m: m.group(1) + new + m.group(3), block)
                n_set += 1
                print(f"set section {i} TFLiteModel {dp}: {old or '(empty)'} -> {new}"
                      f"{' (unchanged)' if old.lower() == new else ''}")
        pieces.append(block)
    new_text = "".join(pieces)
    try:
        parsed = tomllib.loads(new_text).get("section", [])
    except tomllib.TOMLDecodeError as ex:
        print(f"the new TOML does not parse: {ex}", file=sys.stderr)
        return 1
    if len(parsed) != len(sections):
        print("the new TOML has another number of sections", file=sys.stderr)
        return 1
    with open(out, "w", encoding="utf-8") as f:
        f.write(new_text)
    print(f"TFLiteModel blocks {n_tfl}, sha256 items set {n_set}: {out}")
    return 0


def cmd_check(a):
    try:
        from litert_lm_builder import litertlm_core, litertlm_peek
        from litert_lm_builder import litertlm_header_schema_py_generated as H
        bad = n = n_item = 0
        with open(a.litertlm, "rb") as f:
            f.seek(24)
            hend = struct.unpack("<Q", f.read(8))[0]
            f.seek(32)
            md = H.LiteRTLMMetaData.GetRootAs(f.read(hend - 32), 0)
            sm = md.SectionMetadata()
            for i in range(sm.ObjectsLength()):
                o = sm.Objects(i)
                if litertlm_core.any_section_data_type_to_string(o.DataType()) != "TFLiteModel":
                    continue
                n += 1
                items = [litertlm_peek.kvp_to_dict(o.Items(j)) for j in range(o.ItemsLength())]
                item = next((x["value"] for x in items if x.get("key") == "sha256"), None)
                b, e = o.BeginOffset(), o.EndOffset()
                f.seek(b)
                h = hashlib.sha256()
                left = e - b
                while left > 0:
                    d = f.read(min(left, 64 << 20))
                    if not d:
                        raise ValueError(f"section {i} truncated")
                    h.update(d)
                    left -= len(d)
                got = h.hexdigest()
                if item is None:
                    print(f"section {i} TFLiteModel {e - b} B sha256 {got}: no item")
                else:
                    n_item += 1
                    ok = str(item).lower() == got
                    bad += not ok
                    print(f"section {i} TFLiteModel {e - b} B sha256 {got}: item {item} {'=' if ok else 'DIFFERS'}")
    except Exception as ex:  # noqa: BLE001
        print(f"unreadable: {type(ex).__name__}: {ex}", file=sys.stderr)
        return 2
    print(f"SHA256-ITEM: {'PASS' if not bad else 'FAIL'} (TFLiteModel sections {n}, with a sha256 item {n_item}, "
          f"differing {bad})")
    return 1 if bad else 0


def main():
    ap = argparse.ArgumentParser(description="The sha256 item of TFLiteModel sections (externalize_repo.sh "
                                             "--tflite-sha256-item).")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("set")
    s.add_argument("--toml", required=True)
    s.add_argument("--out", required=True)
    c = sub.add_parser("check")
    c.add_argument("--litertlm", required=True)
    a = ap.parse_args()
    return cmd_set(a) if a.cmd == "set" else cmd_check(a)


if __name__ == "__main__":
    sys.exit(main())
