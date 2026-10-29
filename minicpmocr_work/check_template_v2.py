#!/usr/bin/env python3
"""Desk check of chat_template_ocr.jinja (the ship template: the dual template + page tiles through the extra-context
variable `slice_cols`) against upstream's token ids, in jinja2 and in minijinja (the engine LiteRT-LM renders with;
v0.18.1 runtime/components/rust/minijinja_template.rs puts every extra-context key at the template's top level).

  python check_template_v2.py --src SNAPSHOT --fixtures DIR --tiles DIR/tiles.json --out OUT.json [--g 448]

Checks:
  A. the 9 template cases (build_ocr_bundle.cases) without slice_cols: the render equals chat_template_ocr_dual.jinja's (the
     template shipped with MiniCPM-V-4.7-1B and in the first OCR bundles), and its ids equal MiniCPMOcrProcessor's.
  B. the same 9 cases with slice_cols = 3: the render is unchanged (no message holds two images).
  C. the 6 tiled fixtures of make_tiles.py (thumbnail + slices in one user message, then the prompt) with
     slice_cols = the grid's columns: the ids equal hf_tiles.py's upstream prompt (the one the t448 runs saw), and the
     render equals the item-key template (chat_template_ocr_tiles.jinja) on the same message.
  D. slice_cols given as "2" / 0 / None / absent on a 2x2 tiling: "2" = 2; 0, None and absent = no slices.
"""
import argparse
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from build_ocr_bundle import cases  # noqa: E402

SOFT = "<image_soft_token>"


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--src", required=True)
  ap.add_argument("--fixtures", required=True)
  ap.add_argument("--tiles", required=True)
  ap.add_argument("--out", required=True)
  ap.add_argument("--g", type=int, default=448)
  a = ap.parse_args()
  import jinja2
  import minijinja
  import importlib.metadata as md
  from PIL import Image
  from transformers import AutoProcessor

  new_src = open(os.path.join(HERE, "chat_template_ocr.jinja")).read()
  dual_src = open(os.path.join(HERE, "chat_template_ocr_dual.jinja")).read()
  tiles_src = open(os.path.join(HERE, "chat_template_ocr_tiles.jinja")).read()
  n_soft = (a.g // 28) ** 2
  proc = AutoProcessor.from_pretrained(a.src)
  tok = proc.tokenizer
  pad_id = tok.convert_tokens_to_ids("<|image_pad|>")

  j2 = jinja2.Environment(undefined=jinja2.StrictUndefined)
  mj = minijinja.Environment()
  mj.add_template("new", new_src)

  def r_j2(src, msgs, gen=True, **ctx):
    return j2.from_string(src).render(messages=msgs, add_generation_prompt=gen, **ctx)

  def r_mj(msgs, gen=True, **ctx):
    return mj.render_template("new", messages=msgs, add_generation_prompt=gen, **ctx)

  def bundle_ids(text):
    out, pieces = [], text.split(SOFT)
    for k, piece in enumerate(pieces):
      out += tok(piece, add_special_tokens=False).input_ids
      if k < len(pieces) - 1:
        out += [pad_id] * n_soft
    return out

  res = {"jinja2": jinja2.__version__, "minijinja": md.version("minijinja"),
         "template_sha256": hashlib.sha256(new_src.encode()).hexdigest(), "template_bytes": len(new_src.encode()),
         "A": [], "B": [], "C": [], "D": []}
  ok = True
  imgs = [os.path.join(a.fixtures, "para_en_p448.png"), os.path.join(a.fixtures, "table_p448.png")]
  for name, msgs, gen in cases(*imgs):
    hf_msgs, b_msgs = [], []
    for m in msgs:
      if isinstance(m["content"], str):
        hf_msgs.append(dict(m))
        b_msgs.append(dict(m))
        continue
      hc, bc = [], []
      for it in m["content"]:
        if it["type"] == "image":
          hc.append({"type": "image", "image": Image.open(it["fixture"]).convert("RGB")})
          bc.append({"type": "image"})
        else:
          hc.append(dict(it))
          bc.append(dict(it))
      hf_msgs.append({"role": m["role"], "content": hc})
      b_msgs.append({"role": m["role"], "content": bc})
    hf = proc.apply_chat_template(hf_msgs, add_generation_prompt=gen, tokenize=True, return_dict=True,
                                  enable_thinking=False)
    hf_ids = hf["input_ids"]
    hf_ids = hf_ids.tolist() if hasattr(hf_ids, "tolist") else hf_ids
    hf_ids = hf_ids[0] if hf_ids and isinstance(hf_ids[0], list) else hf_ids
    t_new, t_mj, t_dual = r_j2(new_src, b_msgs, gen), r_mj(b_msgs, gen), r_j2(dual_src, b_msgs, gen)
    row = {"case": name, "eq_dual": t_new == t_dual, "eq_minijinja": t_new == t_mj,
           "eq_hf_ids": bundle_ids(t_new) == list(hf_ids), "n_ids": len(hf_ids)}
    res["A"].append(row)
    ok &= row["eq_dual"] and row["eq_minijinja"] and row["eq_hf_ids"]
    t3, t3m = r_j2(new_src, b_msgs, gen, slice_cols=3), r_mj(b_msgs, gen, slice_cols=3)
    rowb = {"case": name, "eq_without": t3 == t_new, "eq_minijinja": t3 == t3m}
    res["B"].append(rowb)
    ok &= rowb["eq_without"] and rowb["eq_minijinja"]
    print(f"A {name:24s} dual {row['eq_dual']} mj {row['eq_minijinja']} hf {row['eq_hf_ids']} n {len(hf_ids)} | "
          f"B same {rowb['eq_without']} mj {rowb['eq_minijinja']}", flush=True)

  man = json.load(open(a.tiles))
  pad = "<|image_pad|>" * n_soft
  for it in man["items"]:
    rows_n, cols = it["grid"]
    n = len(it["files"])
    content = [{"type": "image"} for _ in range(n)] + [{"type": "text", "text": it["prompt"]}]
    msgs = [{"role": "user", "content": content}]
    keyed = [{"type": "image"}] + [{"type": "image", "tile": "slice", "row_break": bool((k - 1) % cols == 0 and k > 1)}
                                   for k in range(1, n)] + [{"type": "text", "text": it["prompt"]}]
    t_new, t_mj = r_j2(new_src, msgs, slice_cols=cols), r_mj(msgs, slice_cols=cols)
    t_keyed = r_j2(tiles_src, [{"role": "user", "content": keyed}])
    hf_rows = ["".join(f"<slice>{pad}</slice>" for _ in range(cols)) for _ in range(rows_n)]
    hf_prompt = (f"<|im_start|>user\n<image_id>0</image_id><image>{pad}</image>" + "\n".join(hf_rows) +
                 f"\n{it['prompt']}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")
    hf_ids = tok(hf_prompt, add_special_tokens=False).input_ids
    row = {"name": it["name"], "grid": it["grid"], "n_images": n, "eq_hf_ids": bundle_ids(t_new) == hf_ids,
           "eq_minijinja": t_new == t_mj, "eq_item_keys": t_new == t_keyed, "n_ids": len(hf_ids),
           "render": t_new.replace(SOFT, "S")}
    res["C"].append(row)
    ok &= row["eq_hf_ids"] and row["eq_minijinja"] and row["eq_item_keys"]
    print(f"C {it['name']:14s} grid {it['grid']} hf {row['eq_hf_ids']} mj {row['eq_minijinja']} "
          f"keys {row['eq_item_keys']} n {len(hf_ids)}", flush=True)

  msgs = [{"role": "user", "content": [{"type": "image"} for _ in range(5)] + [{"type": "text", "text": "T"}]}]
  want2 = r_j2(new_src, msgs, slice_cols=2)
  plain = r_j2(new_src, msgs)
  for label, ctx in (("str2", {"slice_cols": "2"}), ("zero", {"slice_cols": 0}), ("none", {"slice_cols": None}),
                     ("absent", {})):
    tj, tm = r_j2(new_src, msgs, **ctx), r_mj(msgs, **ctx)
    want = want2 if label == "str2" else plain
    row = {"ctx": label, "eq_expected": tj == want, "eq_minijinja": tj == tm}
    res["D"].append(row)
    ok &= row["eq_expected"] and row["eq_minijinja"]
    print(f"D {label:6s} expected {row['eq_expected']} mj {row['eq_minijinja']}", flush=True)
  res["all_ok"] = bool(ok)
  with open(a.out, "w") as f:
    json.dump(res, f, indent=1, ensure_ascii=False)
    f.write("\n")
  print("TEMPLATE_V2", "ALL_OK" if ok else "MISMATCH", a.out)
  return 0 if ok else 1


if __name__ == "__main__":
  raise SystemExit(main())
