# demo_scene_check — score a VLM demo scene before you film it

Fixed photos and requests, truth regions as masks, every model asked in its own author's prompt form, the `.litertlm` run the way the app sends it, a detector + fixed-rule arm, one score file, and a table whose two lines say whether the scene works in the shipped form and whether a cheaper route makes the same clip. The method, the worked example and the commands: [docs/demo_scene_check.md](../../docs/demo_scene_check.md).

| File | Does |
|---|---|
| `scene_where_to_put.py` | The example scene as data: 10 Wikimedia Commons photos (CC0 / public domain; title, author, licence, sha1), surfaces and objects as polygons, 30 placement requests with their zones, 23 reference objects |
| `regions.py` | Polygons → 1000 × 1000 truth masks (zone ∧ surface shrunk by 4 units ∧ ¬ other objects grown by 4 units) |
| `fixtures.py` | `write`: download the photos (sha1 checked), the source picture (long side 1024), its 512 × 512 stretch, every mask, overlays to look at; `check`: recompute every hash |
| `plan.py` | One plan per model: every case and prompt text fixed (pointing / grounding forms of LFM3-VL, Qwen3-VL, LFM2.5-VL, Gemma); `--objects` for the reference objects; `--cllm` for the detector's boxes as text |
| `run_hf.py` | A plan through transformers: greedy, float32, one JSON line per case, resumable |
| `run_litertlm.py` | A `.litertlm` VLM on the plan's 512 × 512 pictures with the LiteRT-LM Python package: image + text in one message, thinking per flag, the output budget from the context, one process per case |
| `detect.py` | `fetch`: the pinned Grounding DINO base with sha256 checks; `run`: one forward pass per object phrase, every box with its score |
| `crule.py` | The fixed rule: a point from the detector's boxes (beside, in front, behind, between, on) |
| `score.py` | Every answer read (the model's own format, then favourable readings for the rivals) and scored against its mask; `--selftest` |
| `table.py` | Summary, best condition per group, the work line and the swap line, candidates to film, objects pointed vs. places, budget |
| `misses.py` | Where each wrong point landed (on the named object, on another object, the wrong side, off the surface) |
| `read.py` | Every answer as text next to its verdict, and the points drawn on the picture |

Needs numpy and Pillow; torch and transformers for `run_hf.py` and `detect.py`; `litert-lm` for `run_litertlm.py`. Photos, masks and model files stay under the output directory (`out/` is not committed).
