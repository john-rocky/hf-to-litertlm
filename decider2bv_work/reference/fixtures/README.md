# Fixtures

42 requests (37 with an image, 5 text-only; 62 answer slots) and the upstream model's float32 answers for each, used by [check_fixtures.py](../check_fixtures.py).

Every image here was drawn for this check with PIL by `make_fixtures.py` and `make_fixtures_v2.py` (in [hf-to-litertlm `decider2bv_work/scripts/`](https://github.com/john-rocky/hf-to-litertlm/tree/main/decider2bv_work/scripts)): colour cards, shapes, arrows, bar charts, a word, and simple Pong- and Breakout-style frames at 160×210 and 256×240. None is a photograph, a screenshot or an emulator render. The generators are deterministic, and `fixtures.json` records each image's file SHA-256 and decoded-RGB SHA-256.

- `images/` — the original images, at the size in each file name.
- `images_256/` — each original resized to 256×256 with PIL `Image.BICUBIC`, the input the LiteRT files take.
- `fixtures.json` — per request: context, questions and options, the image files with their hashes, the first position (0 with an image, 65 without), and the upstream values of three arms. Each arm holds `input_ids`, `slot_idx`, and per slot the letter logits (`letter_logits_fp32`), the probabilities (softmax at T = 1), the argmax, the top-two probability gap and the full-vocabulary top-1 token id.

| arm | what upstream was given |
| --- | --- |
| `g256_mrope` | the image from `images_256/` — the reference the LiteRT files are checked against |
| `author` | the original image, at upstream's own dynamic resolution — shows the cost of the fixed 256×256 input |
| `g256_pos1d` | the `images_256/` image with plain 1-D positions instead of the model's 3-channel M-RoPE positions — shows why the decoder derives the M-RoPE positions |

The upstream values come from the checkpoint's own `decider/vision.py` (`prepare()` + `slot_logits()`, unmodified) at revision `863e290863655f1d6b69324d77d09ac972d21609`, in float32 on an Apple M4 Max CPU (torch 2.14.0, transformers 5.17.0, 8 threads), one request per forward.

The contexts and questions of the game and colour rows follow the upstream repository's own prompt shapes (its game intros and option lists, its colour example); four of the five text-only rows reuse its `infer.py` and `vision.py` demo requests.
