#!/usr/bin/env python3
"""Run one tiled fixture through the bundle with the LiteRT-LM Python API: the thumbnail and the slices of
make_tiles.py as image items of ONE user message, a fresh Engine in this process. Appends a row in hf_oracle.py's
shape (mode "t448") to --out.

  --mark ctx  (default, the shipped template): plain image items, the bundle's own template, and
              create_conversation(extra_context={"slice_cols": <grid columns>}) (chat_template_ocr.jinja)
  --mark keys (the first form, per-item keys): slices marked {"tile": "slice", "row_break": ...} and --template (chat_template_ocr_tiles.jinja)
              given at conversation creation

  python rt_tiles.py BUNDLE --tiles DIR/tiles.json --name page_en --out runs.jsonl [--backend cpu] [--vision-backend cpu]
"""
import argparse
import hashlib
import json
import os
import time

import litert_lm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bundle")
    ap.add_argument("--tiles", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--backend", default="cpu")
    ap.add_argument("--vision-backend", default="cpu")
    ap.add_argument("--template", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                       "chat_template_ocr_tiles.jinja"))
    ap.add_argument("--max-output-tokens", type=int, default=1536)
    ap.add_argument("--mark", choices=("ctx", "keys"), default="ctx")
    ap.add_argument("--cache-dir", default=None, help="the runtime's cache dir (default: beside the bundle)")
    a = ap.parse_args()
    man = json.load(open(a.tiles))
    d = os.path.dirname(os.path.abspath(a.tiles))
    it = next(x for x in man["items"] if x["name"] == a.name)
    rows, cols = it["grid"]
    content = []
    for k, f in enumerate(it["files"]):
        p = os.path.join(d, f["file"])
        assert hashlib.sha256(open(p, "rb").read()).hexdigest() == f["sha256"], p
        item = {"type": "image", "path": p}
        if k > 0 and a.mark == "keys":
            r, c = divmod(k - 1, cols)
            item.update({"tile": "slice", "row_break": bool(c == 0 and r > 0)})
        content.append(item)
    content.append({"type": "text", "text": it["prompt"]})
    t0 = time.time()
    be = {"cpu": litert_lm.interfaces.Backend.CPU, "gpu": litert_lm.interfaces.Backend.GPU}
    engine = litert_lm.Engine(a.bundle, backend=be[a.backend](), vision_backend=be[a.vision_backend](),
                              max_num_images=len(it["files"]), cache_dir=a.cache_dir)
    t1 = time.time()
    if a.mark == "keys":
        conv = engine.create_conversation(chat_template=open(a.template).read())
    else:
        conv = engine.create_conversation(extra_context={"slice_cols": cols})
    resp = conv.send_message({"role": "user", "content": content}, max_output_tokens=a.max_output_tokens)
    t2 = time.time()
    text = "".join(c.get("text", "") for c in resp.get("content", []) if isinstance(c, dict)) \
        if isinstance(resp.get("content"), list) else str(resp.get("content", ""))
    row = {"name": a.name, "mode": f"t{man['g']}", "label": f"rt_{a.backend}", "runtime": litert_lm.__version__
           if hasattr(litert_lm, "__version__") else "?", "grid": it["grid"], "n_images": len(it["files"]),
           "backend": a.backend, "vision_backend": a.vision_backend, "mark": a.mark, "engine_s": round(t1 - t0, 1),
           "gen_s": round(t2 - t1, 1), "text": text, "text_raw": text}
    with open(a.out, "a") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"{a.name} grid {it['grid']} images {len(it['files'])} engine {row['engine_s']}s gen {row['gen_s']}s | "
          f"{text[:100]!r}", flush=True)


if __name__ == "__main__":
    main()
