#!/usr/bin/env python3
"""Read document pages with the LiteRT-LM bundle of baidu/Unlimited-OCR-2 (fast_vlm, one 1,024 x 1,024 page per
request).

  python ocr_unlimited.py Unlimited-OCR-2_int8.litertlm page.png [more pages ...] [--backend cpu|gpu]
      [--vision-backend cpu|gpu] [--threads N] [--max-tokens N] [--cache-dir DIR | --no-cache] [--out FILE]

Needs litert-lm (pip install litert-lm) and Pillow.

Pictures: PNG or JPEG. Each one gets its EXIF orientation applied and is fitted, centred, onto a 1,024 x 1,024 gray
(127, 127, 127) square (PIL ImageOps.pad, bicubic). That is the model's own preparation of a page in base mode, the
mode of its multi-page inference: _uocr_build_inputs in the checkpoint's modeling_unlimitedocr.py pads every page
into 1,024 x 1,024 with the gray of its 0.5 normalisation mean. A page prepared this way passes LiteRT-LM's own resize
to 1,024 x 1,024 unchanged; a picture of another shape would be stretched.
Prompt: none to choose. The bundle's template writes <PAGE>, the page, then the model's instruction
"Multi page parsing.".
Decoding: the most likely token at each step (the bundle sets no sampler; LiteRT-LM 0.19.0 then takes top-p with
k = 1) with the runtime's no-repeat n-gram constraint, n = 35 within the last 128 generated tokens
(NoRepeatNgramConfig(35, 128)), the model's own setting for one page.
Several pictures: one request per page, in the order given, each in a new conversation on one Engine. The texts are
printed in that order, joined as they are; each page's text starts with <PAGE>, as in the model's multi-page output.
Output: the model's raw text with its tags kept (<PAGE>, <BLOCK>, <FORK>, ...), as the checkpoint's infer() returns it.
Caches: by default the runtime keeps the weights it prepares in cache files beside the bundle (about 4.4 GB for the
CPU, 2.8 GB for the GPU; --cache-dir DIR puts them elsewhere), and on a Mac M4 Max one page then peaked at a 2.3-2.6
GB memory footprint on the CPU and 5.9 GB on the GPU. --no-cache writes no file: the runtime prepares the weights in
memory at every start, and the process peaked at 9.1 GB (CPU) and 7.9 GB (GPU).

Exit status: 0 = every page ended on the model's end token; 3 = a page stopped at --max-tokens (its text is cut; the
other pages are complete); 2 = a picture or the bundle could not be read, or a bad option (nothing was run);
1 = the runtime failed.
"""
import argparse
import base64
import io
import os
import sys
import time

from PIL import Image, ImageOps

SIZE = 1024               # the bundle's page: 1,024 x 1,024 (273 image tokens)
GRAY = (127, 127, 127)    # the checkpoint's pad colour: int(0.5 * 255) per channel
PROMPT = "Multi page parsing."
CONTEXT = 4096            # the bundle's max_num_tokens
PROMPT_TOKENS = 279       # one page's prompt: BOS + <PAGE> + 273 image tokens + the 4 tokens of the instruction
NGRAM = (35, 128)         # no-repeat n-gram size, window of generated tokens


def fail(message):
  """A picture or the bundle cannot be read: say so and exit with 2 before anything is run."""
  print(f"error: {message}", file=sys.stderr)
  sys.exit(2)


def page_picture(path):
  """The 1,024 x 1,024 RGB picture the bundle reads for the picture at path (EXIF orientation applied, gray pad)."""
  img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
  return ImageOps.pad(img, (SIZE, SIZE), color=GRAY)


def png_bytes(img):
  buf = io.BytesIO()
  img.save(buf, format="PNG")
  return buf.getvalue()


def read_pages(paths):
  """-> [(path, original size, PNG bytes of the 1,024 x 1,024 page)]; exits with 2 on a picture it cannot read."""
  pages = []
  for p in paths:
    if not os.path.isfile(p):
      fail(f"no such file: {p!r}")
    try:
      with Image.open(p) as im:
        size = im.size
      pages.append((p, size, png_bytes(page_picture(p))))
    except (OSError, ValueError, Image.DecompressionBombError) as e:
      fail(f"cannot read {p!r} as a picture: {e}")
  return pages


def stream_text(conv, message, max_tokens, ngram_config, out):
  """Sends one page; writes the text to out as it arrives and returns it."""
  text = ""
  # streamed: a blocking send_message gives up after 10 minutes, and a long page on a slow CPU can take longer
  for chunk in conv.send_message_async(message, max_output_tokens=max_tokens, no_repeat_ngram_config=ngram_config):
    content = chunk.get("content") if isinstance(chunk, dict) else getattr(chunk, "content", None)
    parts = content if isinstance(content, list) else [{"type": "text", "text": content or ""}]
    for c in parts:
      if isinstance(c, dict) and c.get("type", "text") == "text":
        out.write(c.get("text", ""))
        out.flush()
        text += c.get("text", "")
  return text


def main():
  ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
  ap.add_argument("bundle", help="the Unlimited-OCR-2 .litertlm file")
  ap.add_argument("images", nargs="+", help="one picture per page (PNG or JPEG), in page order")
  ap.add_argument("--backend", choices=("cpu", "gpu"), default="cpu", help="the text decoder's backend")
  ap.add_argument("--vision-backend", choices=("cpu", "gpu"), default=None, help="default: the --backend value")
  ap.add_argument("--threads", type=int, default=4, help="CPU threads (default 4)")
  ap.add_argument("--max-tokens", type=int, default=CONTEXT - PROMPT_TOKENS,
                  help=f"tokens to generate per page at most (default {CONTEXT - PROMPT_TOKENS} = the "
                       f"{CONTEXT:,}-token context minus the page's {PROMPT_TOKENS}-token prompt)")
  cache = ap.add_mutually_exclusive_group()
  cache.add_argument("--cache-dir", default=None,
                     help="where the runtime keeps its cache files (default: beside the bundle; about 4.4 GB for the "
                          "CPU, 2.8 GB for the GPU)")
  cache.add_argument("--no-cache", action="store_true",
                     help="write no cache file: the runtime then prepares the weights in memory at every start (Mac M4 "
                          "Max: a 9.1 GB peak memory footprint on the CPU and 7.9 GB on the GPU, against 2.3-2.6 GB "
                          "and 5.9 GB with the cache)")
  ap.add_argument("--out", default=None, help="also write the joined text to this file")
  a = ap.parse_args()
  if not 0 < a.max_tokens <= CONTEXT - PROMPT_TOKENS:
    ap.error(f"--max-tokens must be 1-{CONTEXT - PROMPT_TOKENS}")
  if not os.path.isfile(a.bundle):
    fail(f"no such bundle: {a.bundle!r}")
  pages = read_pages(a.images)

  import litert_lm
  vision = a.vision_backend or a.backend
  make = {"cpu": lambda: litert_lm.interfaces.Backend.CPU(thread_count=a.threads),
          "gpu": lambda: litert_lm.interfaces.Backend.GPU()}
  t0 = time.time()
  try:
    engine = litert_lm.Engine(a.bundle, backend=make[a.backend](), vision_backend=make[vision](), max_num_images=1,
                              cache_dir=":nocache" if a.no_cache else a.cache_dir, enable_benchmark=True)
  except Exception as e:  # noqa: BLE001
    print(f"error: the runtime could not open {a.bundle!r} on {a.backend} (vision {vision}): {e}", file=sys.stderr)
    return 1
  print(f"[engine {a.backend} (vision {vision}): {time.time() - t0:.1f} s]", file=sys.stderr)
  ngram_config = litert_lm.NoRepeatNgramConfig(no_repeat_ngram_size=NGRAM[0], window_size=NGRAM[1])

  texts, cut = [], []
  for i, (path, size, png) in enumerate(pages, 1):
    message = {"role": "user", "content": [{"type": "image", "blob": base64.b64encode(png).decode("ascii")},
                                           {"type": "text", "text": PROMPT}]}
    t1 = time.time()
    try:
      conv = engine.create_conversation()
      texts.append(stream_text(conv, message, a.max_tokens, ngram_config, sys.stdout))
      info = conv.get_benchmark_info()
    except Exception as e:  # noqa: BLE001
      print(f"\nerror: page {i} ({path}): the runtime failed: {e}", file=sys.stderr)
      return 1
    n = info.last_decode_token_count
    tps = info.last_decode_tokens_per_second
    stopped = n >= a.max_tokens
    if stopped:
      cut.append(i)
    print(f"\n[page {i}/{len(pages)} {os.path.basename(path)} {size[0]}x{size[1]} -> {SIZE}x{SIZE}: {n} tokens, "
          f"{tps:.1f} tok/s, {time.time() - t1:.1f} s"
          f"{', stopped at --max-tokens: the text is cut' if stopped else ''}]", file=sys.stderr)
  sys.stdout.write("\n")
  if a.out:
    with open(a.out, "w", encoding="utf-8") as f:
      f.write("".join(texts))
  if cut:
    print(f"warning: page(s) {', '.join(map(str, cut))} stopped at --max-tokens {a.max_tokens}; their text is cut",
          file=sys.stderr)
    return 3
  return 0


if __name__ == "__main__":
  sys.exit(main())
