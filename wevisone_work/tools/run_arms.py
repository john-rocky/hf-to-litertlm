#!/usr/bin/env python3
"""WeVisOne-VL-2B round 1: fp32 CPU reference arms on the fixtures.

Arms (all fp32, CPU, one thread, greedy, max_new_tokens 1024, the checkpoint template + the package's system turn):
  A0  the package's WeVisOneVL.predict() as shipped (engine min/max pixels, repetition_penalty 1.05, masks on <OBJ>)
  A1  A0's inputs with repetition_penalty 1.0
  B   image squashed to 512x512 (PIL BICUBIC), processor min = max = 262,144 px (16x16 merged grid), full model
  C   B + DeepStack folded: the three deepstack merger outputs summed into the image embedding, no layer 0-2 injection
  D   C + 1-D positions (all three M-RoPE channels = arange)
  E   C + derived M-RoPE (1-D positions in, (t, h, w) derived inside the rotary from P and the grid)
  F   768x768 full (24x24 = 576 tokens);  G  768 fold + derived M-RoPE
  T0  text-only prompts, plain positions;  T1  text-only prompts, derived M-RoPE counted from 0 (the runtime's numbering)

Modes (json files that exist are skipped; one model load per process):
  --arm X --rows all|id,id   generation -> ref/<X>/<row>.json (--out-dir for the fresh-process reproduction)
  --text T0|T1                text-only prompts -> ref/<T>/<id>.json
  --tf --rows ...             teacher-forced pairs on the reference sequences -> ref/tf/<row>.json
  --masks B --rows ...        B-arm mask references for <OBJ> rows -> ref/masks_B/<row>.json + .npz (decoder inputs)
  --mask-selfcheck ROW        this file's mask path vs the package's decode_masks on A0's inputs (must be equal)
  --rope-check                derived (t, h, w) vs Qwen3VLModel.get_rope_index on every image row, 512 and 768
"""
import os

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_k, "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import hashlib
import importlib.util
import json
import platform
import re
import resource
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

K = Path(__file__).resolve().parents[1]
SNAP = Path(os.environ.get("WEVISONE_SNAPSHOT", "~/.cache/huggingface/hub/models--tencent--WeVisOne-VL-2B/"
                           "snapshots/3dad9ce8bcb6fbe0aea945bf8860bd2cfc3663bf")).expanduser()
FIXTURES = K / "fixtures" / "fixtures.json"
REF = K / "ref"
MAX_NEW = 1024
SYSTEM_PREFIX_P = 15        # tokens before the first <|image_pad|> with the package's system turn (checked per row)


def _load_local(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().parent / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MR = _load_local("mrope_derived_wv")
WP = _load_local("wv_parse")

FIXED = {"B": (512, False, "hf"), "C": (512, True, "hf"), "D": (512, True, "1d"), "E": (512, True, "derived"),
         "F": (768, False, "hf"), "G": (768, True, "derived")}
PACKAGE = {"A0": (1.05, True), "A1": (1.0, False)}

SUFFIX = " Answer briefly."
TEXT_ROWS = [
    ("q_add", "What is 17 + 25?" + SUFFIX, r"\b42\b"),
    ("q_tokyo", "What is the capital of Japan?" + SUFFIX, r"tokyo"),
    ("q_cold", 'What is the opposite of "hot"?' + SUFFIX, r"\bcold\b"),
    ("q_week", "How many days are in a week?" + SUFFIX, r"\bseven\b|\b7\b"),
    ("q_merci", 'How do you say "thank you" in French?' + SUFFIX, r"merci"),
    ("q_mul", "What is 8 times 7?" + SUFFIX, r"\b56\b"),
    ("q_larger", "Which is larger: 0.9 or 0.11?" + SUFFIX, r"0\.9"),
    ("q_rhyme", 'Complete the rhyme: "Roses are red, violets are ___"' + SUFFIX, r"\bblue\b"),
    ("q_france", "What is the capital of France? Answer in one word.", r"paris"),
]
TWO_TURN = ("two_turn", ["My name is Ada.", "What is my name?"], r"\bada\b")


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, ensure_ascii=False) + "\n")
    tmp.replace(path)


def rows_of(spec):
    rows = json.loads(FIXTURES.read_text())["rows"]
    if spec in (None, "all"):
        return rows
    want = spec.split(",")
    by_id = {r["id"]: r for r in rows}
    return [by_id[w] for w in want]


def open_image(row):
    path = K / "fixtures" / row["image"]
    with Image.open(path) as im:
        image = im.convert("RGB")
    digest = hashlib.sha256(np.asarray(image).tobytes()).hexdigest()
    if digest != row["pixel_sha256"]:
        raise SystemExit(f"{row['id']}: pixel sha256 mismatch for {path}")
    return image


def peak_rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9   # bytes on macOS


def versions():
    import transformers
    return {"torch": torch.__version__, "transformers": transformers.__version__, "python": platform.python_version(),
            "threads": torch.get_num_threads(), "snapshot": SNAP.name, "machine": platform.machine()}


class Engine:
    """The package's engine (model + processor + mask decoder) and the arm patches on it."""

    def __init__(self):
        from transformers import AutoProcessor
        from wevisone_vl.engine import WeVisOneVL
        t0 = time.perf_counter()
        self.pkg = WeVisOneVL(str(SNAP), device="cpu", dtype="float32")
        self.model = self.pkg.model
        self.core = self.model.model
        self.visual = self.core.visual
        self.lm = self.core.language_model
        self.rotary = self.lm.rotary_emb
        self.tok = self.pkg.processor.tokenizer
        assert float(self.rotary.inv_freq.abs().max()) > 0, "text rotary inv_freq is zero (meta-load trap)"
        assert float(self.visual.rotary_pos_emb.inv_freq.abs().max()) > 0, "vision rotary inv_freq is zero"
        assert next(self.model.parameters()).dtype == torch.float32
        self.procs = {side: AutoProcessor.from_pretrained(str(SNAP), min_pixels=side * side, max_pixels=side * side,
                                                          trust_remote_code=True) for side in (512, 768)}
        self.image_pad = self.tok.convert_tokens_to_ids("<|image_pad|>")
        self._undo = []
        log(f"engine loaded in {time.perf_counter() - t0:.1f} s, rss peak {peak_rss_gb():.1f} GB")

    # ---- patches -------------------------------------------------------------------------------------------
    def clear(self):
        for undo in reversed(self._undo):
            undo()
        self._undo = []
        self.core.rope_deltas = None

    def set_arm(self, fold, pos, grid=None):
        self.clear()
        if fold:
            visual = self.visual
            cls_forward = type(visual).forward

            def fold_forward(*a, **k):
                out = cls_forward(visual, *a, **k)
                out.pooler_output = out.pooler_output + sum(out.deepstack_features)
                out.deepstack_features = []
                return out
            visual.forward = fold_forward
            self._undo.append(lambda: visual.__dict__.pop("forward", None))
        if pos in ("1d", "derived"):
            core = self.core

            def one_d_rope_index(input_ids, mm_token_type_ids=None, image_grid_thw=None, video_grid_thw=None,
                                 attention_mask=None, **kw):
                b, length = input_ids.shape
                p = torch.arange(length, dtype=input_ids.dtype, device=input_ids.device)
                return (p.view(1, 1, length).expand(3, b, length).contiguous(),
                        torch.zeros(b, 1, dtype=input_ids.dtype, device=input_ids.device))
            core.get_rope_index = one_d_rope_index
            self._undo.append(lambda: core.__dict__.pop("get_rope_index", None))
        if pos == "derived":
            self._undo.append(MR.install(self.rotary, grid))
        self.core.rope_deltas = None

    # ---- inputs --------------------------------------------------------------------------------------------
    def fixed_inputs(self, image, side, prompt, response=None):
        vl_image = image.resize((side, side), Image.Resampling.BICUBIC)
        msgs = self.pkg._messages(vl_image, prompt)
        if response is not None:
            msgs = msgs + [{"role": "assistant", "content": response}]
        proc = self.procs[side]
        text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=response is None)
        return proc(text=[text], images=[vl_image], padding=True, return_tensors="pt"), vl_image

    def package_inputs(self, image, prompt):
        return self.pkg._inputs(self.pkg._messages(image, prompt), image, generation=True)

    def text_inputs(self, turns, answers=()):
        msgs = [{"role": "system", "content": "You are a helpful assistant."}]
        for i, user in enumerate(turns):
            msgs.append({"role": "user", "content": user})
            if i < len(answers):
                msgs.append({"role": "assistant", "content": answers[i]})
        text = self.pkg.processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        return self.pkg.processor(text=[text], padding=True, return_tensors="pt")

    def describe(self, inputs):
        ids = inputs["input_ids"][0].tolist()
        grid = inputs.get("image_grid_thw")
        out = {"n_prompt_tokens": len(ids), "prompt_ids": ids}
        if grid is not None:
            g = grid[0].tolist()
            out.update({"image_grid_thw": g, "processor_hw": [g[1] * 16, g[2] * 16],
                        "n_image_tokens": sum(1 for t in ids if t == self.image_pad),
                        "P": ids.index(self.image_pad)})
        return out

    # ---- generation ----------------------------------------------------------------------------------------
    def greedy(self, inputs, rep=1.0, max_new=MAX_NEW):
        self.core.rope_deltas = None
        t0 = time.perf_counter()
        with torch.inference_mode():
            seq = self.model.generate(**inputs, max_new_tokens=max_new, repetition_penalty=rep, use_cache=True,
                                      do_sample=False)
        wall = time.perf_counter() - t0
        new = seq[0, inputs["input_ids"].shape[1]:].tolist()
        return new, wall

    def decode(self, ids):
        return self.tok.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)

    # ---- teacher forcing -------------------------------------------------------------------------------------
    def forced_logits(self, inputs, new_ids):
        """Logits predicting each of new_ids given the prompt in `inputs` (one forward, no cache)."""
        n = len(new_ids)
        extra = torch.tensor([new_ids], dtype=inputs["input_ids"].dtype)
        kw = {"input_ids": torch.cat([inputs["input_ids"], extra], dim=1),
              "attention_mask": torch.ones(1, inputs["input_ids"].shape[1] + n, dtype=torch.long),
              "mm_token_type_ids": torch.cat([inputs["mm_token_type_ids"],
                                              torch.zeros(1, n, dtype=inputs["mm_token_type_ids"].dtype)], dim=1)}
        for key in ("pixel_values", "image_grid_thw"):
            if key in inputs:
                kw[key] = inputs[key]
        self.core.rope_deltas = None
        with torch.inference_mode():
            out = self.model(**kw, use_cache=False, logits_to_keep=n + 1)
        self.core.rope_deltas = None
        return out.logits[0, :-1].float()


def result_record(eng, row, arm, inputs, new_ids, wall, extra=None):
    raw = eng.decode(new_ids)
    eos = bool(new_ids) and new_ids[-1] in WP.EOS_IDS
    rec = {"row": row["id"], "arm": arm, "task": row["task"], "prompt": row["prompt"], "image_size": row["size"]}
    rec.update(eng.describe(inputs))
    rec.update({"new_ids": new_ids, "n_new": len(new_ids), "eos": eos,
                "truncated": (len(new_ids) >= MAX_NEW) and not eos, "response": raw,
                "parsed": WP.parse(raw, row["task"]), "wall_s": round(wall, 2),
                "peak_rss_gb": round(peak_rss_gb(), 2), "versions": versions()})
    if extra:
        rec.update(extra)
    return rec


def run_generation(eng, arm, rows, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    for row in rows:
        path = out_dir / f"{row['id']}.json"
        if path.exists():
            continue
        image = open_image(row)
        t0 = time.perf_counter()
        if arm in PACKAGE:
            rep, masks = PACKAGE[arm]
            eng.clear()
            captured = {}
            original = eng.model.generate

            def wrapped(**kw):
                captured["inputs"] = {k: v for k, v in kw.items() if isinstance(v, torch.Tensor)}
                t1 = time.perf_counter()
                seq = original(**kw)
                captured["gen_s"] = time.perf_counter() - t1
                captured["sequences"] = seq
                return seq
            eng.model.generate = wrapped
            try:
                result = eng.pkg.predict(image, task=row["task"], query=row["query"], prompt=row["prompt"],
                                         decode_matting=masks, max_new_tokens=MAX_NEW, repetition_penalty=rep)
            finally:
                eng.model.__dict__.pop("generate", None)
            inputs = captured["inputs"]
            new_ids = captured["sequences"][0, inputs["input_ids"].shape[1]:].tolist()
            extra = {"package_response": result.response, "gen_s": round(captured["gen_s"], 2),
                     "predict_s": round(result.latency_s, 2), "repetition_penalty": rep,
                     "vl_image_size": list(image.size)}
            if masks and result.masks:
                alphas = np.stack(result.masks).astype(np.float32)
                np.savez_compressed(out_dir / f"{row['id']}.masks.npz", alpha=alphas.astype(np.float16),
                                    boxes=np.array(result.boxes, dtype=np.float32))
                extra["masks"] = summarize_masks(result.boxes, alphas)
            rec = result_record(eng, row, arm, inputs, new_ids, time.perf_counter() - t0, extra)
        else:
            side, fold, pos = FIXED[arm]
            g = side // 32
            eng.set_arm(fold, pos, MR.Grid(SYSTEM_PREFIX_P, g, g))
            inputs, vl_image = eng.fixed_inputs(image, side, row["prompt"])
            desc = eng.describe(inputs)
            assert desc["P"] == SYSTEM_PREFIX_P and desc["image_grid_thw"] == [1, 2 * g, 2 * g], desc
            new_ids, wall = eng.greedy(inputs, rep=1.0)
            eng.clear()
            rec = result_record(eng, row, arm, inputs, new_ids, wall,
                                {"fold": fold, "positions": pos, "side": side, "vl_image_size": list(vl_image.size),
                                 "repetition_penalty": 1.0})
        write_json(path, rec)
        log(f"{arm} {row['id']}: n_new {rec['n_new']} trunc {rec['truncated']} wall {rec['wall_s']} s "
            f"| {rec['parsed']['answer'][:100]!r}")


def summarize_masks(boxes, alphas):
    return {"n_instances": int(alphas.shape[0]), "boxes": [[round(float(v), 4) for v in b] for b in boxes],
            "mean_alpha": [round(float(a.mean()), 6) for a in alphas],
            "frac_gt_half": [round(float((a > 0.5).mean()), 6) for a in alphas]}


def run_text(eng, arm, out_dir):
    grid = MR.Grid(SYSTEM_PREFIX_P, 16, 16)
    jobs = [(rid, [q], pat) for rid, q, pat in TEXT_ROWS] + [TWO_TURN]
    for rid, turns, pat in jobs:
        path = out_dir / f"{rid}.json"
        if path.exists():
            continue
        if arm == "T1":
            eng.set_arm(False, "derived", grid)
        else:
            eng.clear()
        answers, records = [], []
        for t in range(len(turns)):
            inputs = eng.text_inputs(turns[:t + 1], answers)
            new_ids, wall = eng.greedy(inputs, rep=1.0)
            raw = eng.decode(new_ids)
            answers.append(WP.strip_answer(raw))
            records.append({"turn": t + 1, "prompt_ids": inputs["input_ids"][0].tolist(),
                            "n_prompt_tokens": int(inputs["input_ids"].shape[1]), "new_ids": new_ids,
                            "n_new": len(new_ids), "response": raw, "answer": answers[-1], "wall_s": round(wall, 2)})
        eng.clear()
        final = answers[-1].lower()
        rec = {"row": rid, "arm": arm, "turns": turns, "check_regex": pat, "pass": bool(re.search(pat, final)),
               "records": records, "positions": "derived from 0 (P=15, 16x16)" if arm == "T1" else "plain",
               "versions": versions()}
        write_json(path, rec)
        log(f"{arm} {rid}: pass {rec['pass']} | {answers[-1][:80]!r}")


def softmax64(logits):
    return torch.softmax(logits.double(), dim=-1)


def compare_logits(ref_tokens, ref_logits, test_logits):
    ref_tokens = torch.tensor(ref_tokens)
    pr, pt = softmax64(ref_logits), softmax64(test_logits)
    top_t = test_logits.argmax(-1)
    top_r = ref_logits.argmax(-1)
    agree = (top_t == ref_tokens)
    ranks = (test_logits > test_logits.gather(1, ref_tokens[:, None])).sum(-1)
    coord = torch.tensor([WP.is_coord(t) for t in ref_tokens.tolist()])
    coord_ok = []
    for i in torch.nonzero(coord).flatten().tolist():
        a, b = int(top_t[i]), int(ref_tokens[i])
        coord_ok.append(WP.is_coord(a) and abs(WP.coord_bin(a) - WP.coord_bin(b)) <= 5)
    dp = (pr - pt).abs().max(-1).values
    first_dis = int(torch.nonzero(~agree).flatten()[0]) if bool((~agree).any()) else None
    return {"n": int(len(ref_tokens)), "top1_agree": float(agree.double().mean()),
            "n_disagree": int((~agree).sum()), "first_disagree": first_dis,
            "ref_self_top1": float((top_r == ref_tokens).double().mean()),
            "n_coord": int(coord.sum()),
            "coord_within5": (float(np.mean(coord_ok)) if coord_ok else None),
            "max_abs_dp": float(dp.max()), "mean_abs_dp": float(dp.mean()),
            "worst_rank": int(ranks.max()),
            "nll_ref": float(-torch.log(pr.gather(1, ref_tokens[:, None])).mean()),
            "nll_test": float(-torch.log(pt.gather(1, ref_tokens[:, None])).mean())}


def run_tf(eng, rows, out_dir):
    for row in rows:
        path = out_dir / f"{row['id']}.json"
        if path.exists():
            continue
        refs = {a: json.loads((REF / a / f"{row['id']}.json").read_text()) for a in ("A0", "A1", "B", "F")}
        image = open_image(row)
        t0 = time.perf_counter()
        out = {"row": row["id"], "task": row["task"], "pairs": {}, "n_new": {a: refs[a]["n_new"] for a in refs}}

        def logits_for(arm, ref_arm):
            new_ids = refs[ref_arm]["new_ids"]
            if arm in PACKAGE:
                eng.clear()
                inputs = eng.package_inputs(image, row["prompt"])
            else:
                side, fold, pos = FIXED[arm]
                g = side // 32
                eng.set_arm(fold, pos, MR.Grid(SYSTEM_PREFIX_P, g, g))
                inputs, _ = eng.fixed_inputs(image, side, row["prompt"])
            if arm in refs:
                assert inputs["input_ids"][0].tolist() == refs[arm]["prompt_ids"], f"{arm}: prompt ids differ"
            lg = eng.forced_logits(inputs, new_ids)
            eng.clear()
            return lg

        # G (768 fold) is not scored: F is forced on its own path only
        for ref_arm, tests in (("B", ("C", "D", "E")), ("F", ()), ("A1", ("B", "F"))):
            new_ids = refs[ref_arm]["new_ids"]
            if not new_ids:
                continue
            ref_lg = logits_for(ref_arm, ref_arm)
            for t in tests:
                out["pairs"][f"{t}_on_{ref_arm}"] = compare_logits(new_ids, ref_lg, logits_for(t, ref_arm))
            out["pairs"][f"{ref_arm}_self"] = compare_logits(new_ids, ref_lg, ref_lg)
            del ref_lg
        # A0 generated with repetition_penalty 1.05: how often plain greedy (A1's rule) picks another token on A0's path
        a0_lg = logits_for("A0", "A0")
        out["pairs"]["A1rule_on_A0"] = compare_logits(refs["A0"]["new_ids"], a0_lg, a0_lg)
        del a0_lg
        out["wall_s"] = round(time.perf_counter() - t0, 1)
        out["versions"] = versions()
        write_json(path, out)
        log(f"tf {row['id']}: " + ", ".join(f"{k} {v['top1_agree']:.3f}" for k, v in out["pairs"].items())
            + f" ({out['wall_s']} s)")


def run_text_tf(eng, out_dir):
    path = out_dir / "text.json"
    if path.exists():
        return
    grid = MR.Grid(SYSTEM_PREFIX_P, 16, 16)
    out = {}
    for rid in [r[0] for r in TEXT_ROWS] + [TWO_TURN[0]]:
        t0 = json.loads((REF / "T0" / f"{rid}.json").read_text())
        rec = t0["records"][-1]
        inputs = {"input_ids": torch.tensor([rec["prompt_ids"]]),
                  "mm_token_type_ids": torch.zeros(1, len(rec["prompt_ids"]), dtype=torch.int32)}
        eng.clear()
        ref_lg = eng.forced_logits(inputs, rec["new_ids"])
        eng.set_arm(False, "derived", grid)
        test_lg = eng.forced_logits(inputs, rec["new_ids"])
        eng.clear()
        out[rid] = compare_logits(rec["new_ids"], ref_lg, test_lg)
        log(f"tf text {rid}: T1 on T0 top1 {out[rid]['top1_agree']:.3f} max|dp| {out[rid]['max_abs_dp']:.4f}")
    write_json(path, {"pairs": out, "note": "T1 (derived M-RoPE from 0) forced on T0's last-turn tokens",
                      "versions": versions()})


def mask_inputs(eng, vl_image, proc, cnn_source, prompt, response):
    """The package's decode_masks (engine.py L434-516) with the VLM image / processor and the 1024 canvas source
    passed separately. Returns (boxes, t_seg, visual maps, cnn image, grid_hw, teacher ids)."""
    from wevisone_vl.engine import _pil_to_float_rgb, parse_seg_boxes
    from wevisone_vl.feature_bridge import (LastHiddenCapture, VisualFeatureCapture, assert_cnn_rgb, extract_t_seg,
                                            per_sample_premerge_feature_maps, premerge_grid_hw)
    pkg = eng.pkg
    boxes = parse_seg_boxes(response)
    if not boxes:
        return None
    msgs = pkg._messages(vl_image, prompt) + [{"role": "assistant", "content": response}]
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
    teacher = proc(text=[text], images=[vl_image], padding=True, return_tensors="pt")
    decoder_image = cnn_source.convert("RGB").resize((pkg.cnn_size, pkg.cnn_size), resample=Image.Resampling.BICUBIC)
    cnn_image = assert_cnn_rgb(_pil_to_float_rgb(decoder_image), cnn_image_size=pkg.cnn_size)
    eng.core.rope_deltas = None
    old = getattr(eng.model.config, "use_cache", True)
    eng.model.config.use_cache = False
    try:
        with torch.inference_mode():
            with VisualFeatureCapture(eng.model, feature_layers=pkg.cfg.vision_feature_layers) as vcap:
                with LastHiddenCapture(eng.model) as hcap:
                    eng.model(**teacher, output_hidden_states=False, return_dict=True, use_cache=False)
    finally:
        eng.model.config.use_cache = old
        eng.core.rope_deltas = None
    t_seg = extract_t_seg(hcap.hidden, teacher["input_ids"], pkg.seg_token_id)[0]
    grid_thw = teacher["image_grid_thw"]
    grid_hw = premerge_grid_hw(grid_thw)[0]
    maps = per_sample_premerge_feature_maps(vcap.premerge_features, pkg.cfg.vision_feature_layers, grid_thw,
                                            pkg.spatial_merge_size)[0]
    count = min(len(boxes), int(t_seg.shape[0]))
    return boxes[:count], t_seg[:count], maps, cnn_image, grid_hw, teacher["input_ids"][0].tolist()


def decode_alpha(eng, boxes, t_seg, maps, cnn_image, grid_hw, height, width):
    with torch.inference_mode():
        out = eng.pkg.mask_decoder(image=cnn_image, visual_features_2d=maps, t_seg=t_seg,
                                   boxes01=torch.tensor(boxes, dtype=torch.float32), grid_hw=grid_hw)
    logits = out["mask_logits"]
    alpha = eng.pkg._masks_from_logits(logits, height, width)
    return logits, np.stack(alpha)


def run_masks_b(eng, rows, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    for row in rows:
        path = out_dir / f"{row['id']}.json"
        if path.exists() or row["task"] not in ("segmentation", "matting"):
            continue
        b = json.loads((REF / "B" / f"{row['id']}.json").read_text())
        response = WP.strip_answer(b["response"])
        image = open_image(row)
        eng.set_arm(False, "hf")
        vl_image = image.resize((512, 512), Image.Resampling.BICUBIC)
        t0 = time.perf_counter()
        got = mask_inputs(eng, vl_image, eng.procs[512], image, row["prompt"], response)
        rec = {"row": row["id"], "arm": "B", "response": response}
        if got is None:
            rec.update({"n_instances": 0, "note": "no box followed by <OBJ> in the B answer"})
        else:
            boxes, t_seg, maps, cnn_image, grid_hw, teacher_ids = got
            tm = time.perf_counter()
            logits, alpha = decode_alpha(eng, boxes, t_seg, maps, cnn_image, grid_hw, image.height, image.width)
            mask_s = time.perf_counter() - tm
            gen_ids = b["prompt_ids"] + b["new_ids"]
            rec.update(summarize_masks(boxes, alpha))
            rec.update({"grid_hw": list(grid_hw), "teacher_len": len(teacher_ids),
                        "retokenized_equals_generated": teacher_ids[:len(gen_ids)] == gen_ids[:len(teacher_ids)],
                        "mask_decoder_s": round(mask_s, 2), "total_s": round(time.perf_counter() - t0, 2)})
            np.savez_compressed(out_dir / f"{row['id']}.npz",
                                taps=np.stack([m[0].numpy() for m in maps]).astype(np.float32),   # [4,1024,32,32]
                                t_seg=t_seg.float().numpy(), boxes=np.array(boxes, dtype=np.float32),
                                cnn_image=cnn_image.numpy().astype(np.float32), mask_logits=logits.float().numpy(),
                                alpha=alpha.astype(np.float16))
        eng.clear()
        write_json(path, rec)
        log(f"masks B {row['id']}: {rec.get('n_instances')} instance(s)")


def mask_selfcheck(eng, row_id):
    row = rows_of(row_id)[0]
    a0 = json.loads((REF / "A0" / f"{row_id}.json").read_text())
    image = open_image(row)
    eng.clear()
    boxes_pkg, masks_pkg = eng.pkg.decode_masks(image, row["prompt"], a0["package_response"])
    got = mask_inputs(eng, image, eng.pkg.processor, image, row["prompt"], a0["package_response"])
    boxes, t_seg, maps, cnn_image, grid_hw, _ = got
    _, alpha = decode_alpha(eng, boxes, t_seg, maps, cnn_image, grid_hw, image.height, image.width)
    diff = float(np.abs(np.stack(masks_pkg) - alpha).max())
    stored = np.load(REF / "A0" / f"{row_id}.masks.npz")["alpha"].astype(np.float32)
    rec = {"row": row_id, "max_abs_vs_package_decode_masks": diff,
           "max_abs_vs_stored_A0_fp16": float(np.abs(stored - alpha).max()),
           "boxes_equal": [list(map(float, b)) for b in boxes_pkg] == [list(map(float, b)) for b in boxes]}
    write_json(REF / "masks_B" / f"selfcheck_{row_id}.json", rec)
    log(f"mask selfcheck {row_id}: {rec}")


def run_rope_check(eng, rows):
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel
    out = {"function": "transformers 5.12.1 Qwen3VLModel.get_rope_index (+ get_vision_position_ids); decode "
                       "positions continue +1 per channel (GenerationMixin._update_model_kwargs_for_generation)",
           "derived": "tools/mrope_derived_wv.py derived_thw / derived_cos_sin", "rows": []}
    total = equal = 0
    cos_max = sin_max = 0.0
    for row in rows:
        image = open_image(row)
        for side in (512, 768):
            g = side // 32
            grid = MR.Grid(SYSTEM_PREFIX_P, g, g)
            inputs, _ = eng.fixed_inputs(image, side, row["prompt"])
            ids = inputs["input_ids"]
            length = ids.shape[1]
            hf_pos, deltas = Qwen3VLModel.get_rope_index(eng.core, ids, mm_token_type_ids=inputs["mm_token_type_ids"],
                                                         image_grid_thw=inputs["image_grid_thw"],
                                                         attention_mask=inputs["attention_mask"])
            hf = hf_pos[:, 0, :]                                             # [3, L]
            cont = hf[:, -1:] + torch.arange(1, MAX_NEW + 1)[None, :]       # generate's +1 per channel
            hf_all = torch.cat([hf, cont], dim=1)                          # [3, L + 1024]
            p = torch.arange(length + MAX_NEW, dtype=torch.float32)
            t, h, w = MR.derived_thw(p, grid)
            der = torch.stack([t, h, w]).to(torch.long)
            plain = torch.tensor(MR.reference_thw(length + MAX_NEW, grid)).T
            eq = (der == hf_all).all(0)
            first_bad = int(torch.nonzero(~eq).flatten()[0]) if bool((~eq).any()) else None
            # cos / sin: HF rotary on HF positions vs the derived rotary on the 1-D positions
            x = torch.zeros(1, 1, dtype=torch.float32)
            cos_hf, sin_hf = type(eng.rotary).forward(eng.rotary, x, hf_all[:, None, :])
            cos_d, sin_d = MR.derived_cos_sin(p[None, :].to(torch.long), eng.rotary.inv_freq,
                                              eng.rotary.mrope_section, grid, eng.rotary.attention_scaling)
            dc = float((cos_hf - cos_d).abs().max())
            ds = float((sin_hf - sin_d).abs().max())
            bit = float(((cos_hf == cos_d) & (sin_hf == sin_d)).all(-1).double().mean())
            rec = {"row": row["id"], "side": side, "grid": repr(grid), "prompt_len": length,
                   "positions": int(length + MAX_NEW), "equal": int(eq.sum()), "first_mismatch": first_bad,
                   "plain_formula_equal": bool((plain == hf_all).all()), "rope_delta": int(deltas[0, 0]),
                   "cos_max_abs": dc, "sin_max_abs": ds, "cos_sin_bit_equal_frac": bit}
            out["rows"].append(rec)
            total += rec["positions"]
            equal += rec["equal"]
            cos_max, sin_max = max(cos_max, dc), max(sin_max, ds)
            log(f"rope {row['id']} {side}: {rec['equal']}/{rec['positions']} cos {dc:.2e} sin {ds:.2e}")
    out["summary"] = {"rows": len(out["rows"]), "positions": total, "equal": equal,
                      "all_equal": equal == total, "cos_max_abs": cos_max, "sin_max_abs": sin_max}
    write_json(REF / "mrope_check.json", out)
    log(f"rope check: {equal}/{total} equal")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm")
    ap.add_argument("--rows", default="all")
    ap.add_argument("--out-dir")
    ap.add_argument("--text", choices=("T0", "T1"))
    ap.add_argument("--tf", action="store_true")
    ap.add_argument("--text-tf", action="store_true")
    ap.add_argument("--masks", choices=("B",))
    ap.add_argument("--mask-selfcheck")
    ap.add_argument("--rope-check", action="store_true")
    args = ap.parse_args()
    eng = Engine()
    if args.arm:
        out_dir = Path(args.out_dir) if args.out_dir else REF / args.arm
        run_generation(eng, args.arm, rows_of(args.rows), out_dir)
    if args.text:
        run_text(eng, args.text, Path(args.out_dir) if args.out_dir else REF / args.text)
    if args.tf:
        run_tf(eng, rows_of(args.rows), REF / "tf")
    if args.text_tf:
        run_text_tf(eng, REF / "tf")
    if args.masks:
        run_masks_b(eng, rows_of(args.rows), REF / "masks_B")
    if args.mask_selfcheck:
        mask_selfcheck(eng, args.mask_selfcheck)
    if args.rope_check:
        run_rope_check(eng, rows_of(args.rows))
    log(f"done, peak rss {peak_rss_gb():.1f} GB")


if __name__ == "__main__":
    sys.exit(main())
