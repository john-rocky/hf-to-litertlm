#!/usr/bin/env python3
"""Read a picture with Xiaomi-OCR-0's LiteRT-LM bundle (fast_vlm, one picture of 768 x 1344 per request).

  python ocr_xiaomi.py Xiaomi-OCR-0_int8.litertlm picture.jpg [--task document|text|table|formula|kie]
      [--keys date,total_amount,...] [--prompt TEXT] [--backend cpu|gpu] [--vision-backend cpu|gpu]
      [--max-output-tokens N] [--cache-dir DIR] [--keep-picture OUT.png]

Needs litert-lm (pip install litert-lm) and Pillow. The bundle resizes every picture to 768 wide x 1344 tall without
keeping its aspect ratio and without reading EXIF orientation, so this script applies the EXIF orientation, pastes the
picture centred on a white canvas of that tall 4:7 shape that just holds it, and resizes the canvas to 768 x 1344
(bicubic) itself. The task prompts are the five of the model card (xiaomi-research/Xiaomi-OCR-0, "Task prompts");
--task kie writes the card's key-information-extraction prompt with the keys of --keys. One Engine per call: the text
decoder carries a recurrent state, and a second conversation on the same Engine continues from the first one's state
(LiteRT-LM issue #3165).
"""
import argparse
import os
import sys
import tempfile

from PIL import Image, ImageOps

H, W = 1344, 768   # the bundle's picture: FastVlm image_tensor_height x image_tensor_width (1,008 image tokens)

TASKS = {
    "document": "Extract all information from the main body of the document image and represent it in markdown "
                "format, ignoring headers and footers. Tables should be expressed in OTSL format, formulas in the "
                "document should be represented using LATEX format, and the parsing should be organized according "
                "to the reading order.",
    "text": "Extract the text in the image.",
    "table": "Parse the table in the image into OTSL.",
    "formula": "Identify the formula in the image and represent it using LATEX format.",
}
KIE_HEAD = "Extract key information in the image"


def kie_prompt(keys):
  """The model card's key-information-extraction prompt, with one empty string per key."""
  body = "{\n" + ",\n".join(f'    "{k}": ""' for k in keys) + "\n}"
  return f"{KIE_HEAD}\n\nPlease output the key information in JSON format according to the following schema:\n{body}"


def pad_to(img, h=H, w=W):
  """Centre img on a white canvas of aspect h:w that just holds it, then resize to w x h."""
  if img.height * w > img.width * h:      # taller than the canvas shape: keep the full height
    ch, cw = img.height, round(img.height * w / h)
  else:                                   # wider: keep the full width
    cw, ch = img.width, round(img.width * h / w)
  canvas = Image.new("RGB", (cw, ch), "white")
  canvas.paste(img, ((cw - img.width) // 2, (ch - img.height) // 2))
  return canvas.resize((w, h), Image.BICUBIC)


def main():
  ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
  ap.add_argument("bundle")
  ap.add_argument("image")
  ap.add_argument("--task", choices=sorted(TASKS) + ["kie"], default="document")
  ap.add_argument("--keys", default=None, help="comma-separated keys for --task kie")
  ap.add_argument("--prompt", default=None, help="your own prompt (overrides --task)")
  ap.add_argument("--backend", choices=("cpu", "gpu"), default="cpu")
  ap.add_argument("--vision-backend", choices=("cpu", "gpu"), default=None, help="default: the --backend value")
  ap.add_argument("--max-output-tokens", type=int, default=3000,
                  help="the KV cache holds 4,096 tokens; a picture and the prompt take about 1,030-1,100 of them")
  ap.add_argument("--cache-dir", default=None, help="where the runtime writes its caches (default: beside the bundle)")
  ap.add_argument("--keep-picture", default=None, help="also save the 768 x 1344 picture the bundle reads")
  a = ap.parse_args()

  if a.prompt is not None:
    prompt = a.prompt
  elif a.task == "kie":
    keys = [k.strip() for k in (a.keys or "").split(",") if k.strip()]
    if not keys:
      sys.exit("--task kie needs --keys, e.g. --keys date,seller_name,total_amount")
    prompt = kie_prompt(keys)
  else:
    prompt = TASKS[a.task]
  if not os.path.isfile(a.bundle):
    sys.exit(f"no such bundle: {a.bundle!r}")
  try:
    img = ImageOps.exif_transpose(Image.open(a.image)).convert("RGB")
  except (OSError, ValueError) as e:
    sys.exit(f"cannot read the picture {a.image!r}: {e}")
  picture = pad_to(img)

  import litert_lm
  backend = {"cpu": litert_lm.interfaces.Backend.CPU, "gpu": litert_lm.interfaces.Backend.GPU}
  with tempfile.TemporaryDirectory() as tmp:
    path = a.keep_picture or os.path.join(tmp, "picture.png")
    picture.save(path, format="PNG")
    engine = litert_lm.Engine(a.bundle, backend=backend[a.backend](),
                              vision_backend=backend[a.vision_backend or a.backend](),
                              max_num_images=1, cache_dir=a.cache_dir)
    conv = engine.create_conversation()
    message = {"role": "user", "content": [{"type": "image", "path": path}, {"type": "text", "text": prompt}]}
    print(f"[{img.width}x{img.height} -> {W}x{H}, {a.backend} (vision {a.vision_backend or a.backend})]",
          file=sys.stderr)
    # streamed: a blocking send_message gives up after 10 minutes, and a long page on a phone CPU can take longer
    for chunk in conv.send_message_async(message, max_output_tokens=a.max_output_tokens):
      content = chunk.get("content") if isinstance(chunk, dict) else getattr(chunk, "content", None)
      if isinstance(content, list):
        for c in content:
          if isinstance(c, dict) and c.get("type", "text") == "text":
            print(c.get("text", ""), end="", flush=True)
      elif isinstance(content, str):
        print(content, end="", flush=True)
    print()


if __name__ == "__main__":
  main()
