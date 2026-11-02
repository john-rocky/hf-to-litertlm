"""The provider's reference for d1-Omni-2B: LiquidAI/d1-Omni-2B @ 431f1d66 with its own code, float32 (or float64)
on the CPU, one request per call (batch 1: no padding unless a request has more questions than the set size),
unrounded. The d1-2B reference with the Omni snapshot, the loader of records.py (pictures as PIL images, clips and
spoken questions as {"type": "audio", "path": ...}) and two more taps (the audio inputs of each sequence and the
provider's `get_audio_features` output of each forward).

For every record of the given files it writes, by record id:
  answers  the dict `system_one` returns (composed exactly as `SystemOneApi.system_one_batch` composes it)
  probs    what `SlotReader._padded` returns per question (the input of `api.answer`)
  raw      for option questions, the readout before the temperature
  seqs     per sequence: ids, state_len, question tags, slot positions (`ends`), picture grids, audio inputs
  usage, time_s, error (a refusal is a result)
and in the npz: `<id>::slots<j>` (slot hidden states of forward j), `<id>::positions<j>` (read slots' position
logits), `<id>::audio<j>` (the projected audio features of forward j, every clip's valid tokens in order).

--card-exact runs the model card's four code blocks as the card writes them (the beep clip made by the card's own
numpy code, the text questions, the invoice, the COCO picture through transformers.image_utils.load_image of the local
copy, sha256 checked) and compares the printed values.

Run from the work directory with $REF:
  $REF conversion/ref_run.py --card-exact --out out/ref/card_exact_fp32.json
  $REF conversion/ref_run.py --files set/requests.json set/extra_g96x72.json set/red_arms.json \
      --media set/media.json set/media_extra.json --out out/ref/ref_fp32.json
  $REF conversion/ref_run.py --files set/requests_audio.json set/red_arms_audio.json set/card_omni.json \
      --media set/media_audio.json --out out/ref/ref_fp32.json
Without --files: the five set files of records.py (not the card's examples) and their media tables.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.normpath(os.path.join(HERE, ".."))
SNAP = os.path.join(WORK, "snapshot")
COCO = os.path.join(WORK, "out/media/coco_val2017_000000039769.jpg")
COCO_SHA256 = "dea9e7ef97386345f7cff32f9055da4982da5471c48d575146c796ab4563b04e"
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HOME", os.path.join(WORK, "out/hf_home"))          # never the shared cache
os.environ.setdefault("HF_MODULES_CACHE", os.path.join(WORK, "out/hf_modules"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
sys.path.insert(0, os.path.join(WORK, "host"))


def jsonable(x):
    import numpy as np
    import torch

    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, torch.Tensor):
        return jsonable(x.tolist())
    if isinstance(x, np.ndarray):
        return jsonable(x.tolist())
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, float) or isinstance(x, int) or isinstance(x, str) or x is None or isinstance(x, bool):
        return x
    return repr(x)


class Tap:
    """Instruments the provider's engine without changing what it computes."""

    def __init__(self, engine, model):
        import torch

        self.engine = engine
        pkg = type(engine).__module__.rsplit(".", 1)[0]
        self.slots_mod = sys.modules[pkg + ".slots"]
        self.prompt_mod = sys.modules[pkg + ".prompt"]
        self.reset()
        orig_split = self.slots_mod.split

        def split(slots, states, plan, project):
            self.hidden.append(slots.detach().to(torch.float32).cpu().numpy().copy())
            return orig_split(slots, states, plan, project)

        self.slots_mod.split = split
        orig_sequences = engine._sequences

        def sequences(requests, **kw):
            seqs = orig_sequences(requests, **kw)
            for (i, k, ids, state_len, question, ends, inputs) in seqs:
                grid = inputs.get("image_grid_thw") if inputs else None
                feats = inputs.get("input_features") if inputs else None
                fmask = inputs.get("feature_attention_mask") if inputs else None
                self.seqs.append({"first_question": k, "ids": list(ids), "state_len": int(state_len),
                                  "question": None if question is None else list(question),
                                  "ends": [list(e) for e in ends],
                                  "image_grid_thw": None if grid is None else grid.tolist(),
                                  "audio": None if feats is None else {
                                      "input_features_shape": list(feats.shape),
                                      "frames": [int(v) for v in fmask.sum(-1).tolist()]},
                                  "media_keys": sorted(inputs) if inputs else []})
            return seqs

        engine._sequences = sequences
        orig_slot_logits = engine._slot_logits

        def slot_logits(seqs, batch, slots, pointer):
            logz, positions = orig_slot_logits(seqs, batch, slots, pointer)
            if positions is not None:
                self.positions.append(positions.detach().float().cpu().numpy().copy())
            return logz, positions

        engine._slot_logits = slot_logits
        orig_readouts = engine._readouts

        def readouts(qs, logz, tempered):
            out = orig_readouts(qs, logz, tempered)
            self.raw.append(self.prompt_mod.readouts(engine.tokenizer, qs, logz, [None] * len(qs)))
            self.readout_kinds.append([type(q).__name__ for q in qs])
            return out

        engine._readouts = readouts
        body = model.model
        orig_audio = body.get_audio_features

        def get_audio_features(input_features, feature_attention_mask):
            out = orig_audio(input_features, feature_attention_mask)
            self.audio.append(out.detach().to(torch.float32).cpu().numpy().copy())
            return out

        body.get_audio_features = get_audio_features

    def reset(self):
        self.hidden, self.seqs, self.positions, self.raw, self.readout_kinds, self.audio = [], [], [], [], [], []


def card_exact(model) -> dict:
    """The card's four code blocks (README.md of the snapshot) as written, with their printed values."""
    import numpy as np
    from transformers.image_utils import load_image

    out = {}
    # A recording: three short beeps over quiet noise, 16 kHz mono
    rate = 16000
    clip = np.random.default_rng(0).normal(0, 0.003, 4 * rate).astype(np.float32)
    for start in (0.5, 1.5, 2.5):
        clip[int(start * rate):int(start * rate) + 3200] += 0.3 * np.sin(2 * np.pi * 880 * np.arange(3200) / rate)
    o = model.system_one(None, {
        "beeps": {"type": "number", "instructions": "How many beeps are there?", "range": [0, 10], "step": 1},
        "speech": {"type": "noul", "instructions": "Does someone speak in the recording?"},
    }, media=[{"type": "audio", "audio": clip, "sampling_rate": rate}])
    a = o["answers"]
    out["beeps"] = {"answers": jsonable(a), "usage": o["usage"], "got": [a["beeps"]["number"], a["speech"]["noul"] > 0.5],
                    "want": [3.0, False], "clip_sha256": hashlib.sha256(clip.tobytes()).hexdigest()}
    questions = {
        "refund": {"type": "noul", "instructions": "Is the customer asking for a refund?"},
        "team": {"type": "choice", "instructions": "Which team should handle this?",
                 "criteria": {"billing": "Charges, refunds, invoices", "shipping": "Delivery and parcels",
                              "technical": "App or site faults"}},
        "topics": {"type": "choice", "multiple": True, "instructions": "Which topics does the message raise?",
                   "criteria": {"billing": "Payments and charges", "delivery": "Shipping and parcels",
                                "login": "Signing in to the account"}},
    }
    o = model.system_one("I was charged twice and my parcel never arrived. Please refund one charge.", questions)
    a = o["answers"]
    out["text"] = {"answers": jsonable(a), "usage": o["usage"],
                   "got": [round(a["refund"]["noul"], 2), a["team"]["choice"], a["topics"]["choices"]],
                   "want": [0.94, "billing", ["billing", "delivery"]]}
    invoice = "ACME SA\nInvoice 2026-117\nTotal due: 1 234,50 EUR\nPay before 03/04/2026 (day/month/year)"
    o = model.system_one(invoice, {
        "payee": {"type": "text", "instructions": "Who issued the invoice?"},
        "total": {"type": "number", "instructions": "How much is due?"},
        "due": {"type": "date", "instructions": "By when must it be paid?"},
        "vat": {"type": "number", "instructions": "How much VAT is charged?"},
    })
    a = o["answers"]
    out["invoice"] = {"answers": jsonable(a), "usage": o["usage"],
                      "got": [a["payee"]["text"], a["total"]["number"], a["due"]["date"], a["vat"]["number"]],
                      "want": ["ACME SA", 1234.5, "2026-04-03", None]}
    with open(COCO, "rb") as f:
        assert hashlib.sha256(f.read()).hexdigest() == COCO_SHA256, "COCO picture sha256"
    image = load_image(COCO)  # the card: load_image("http://images.cocodataset.org/val2017/000000039769.jpg")
    o = model.system_one(None, {
        "cats": {"type": "number", "instructions": "How many cats are there?", "range": [0, 10], "step": 1},
        "remote": {"type": "box", "instructions": "Where is the remote control on the left?"},
        "every_cat": {"type": "boxes", "instructions": "Every cat.", "max": 4},
    }, media=[image])
    a = o["answers"]
    out["coco"] = {"answers": jsonable(a), "usage": o["usage"],
                   "got": [a["cats"]["number"], a["every_cat"]["count"],
                           {k: round(v, 2) for k, v in a["remote"]["box"].items()}],
                   "want": [2.0, 2, {"top": 0.15, "bottom": 0.24, "left": 0.06, "right": 0.27}]}
    for v in out.values():
        v["match"] = jsonable(v["got"]) == jsonable(v["want"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="*", default=None, help="request files (default: records.py's five set files)")
    ap.add_argument("--media", nargs="*", default=None, help="media tables (default: those of the five set files)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--hidden", default=None, help="npz of slot hidden states (default: --out with _slots.npz)")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--ids", default=None, help="comma-separated ids to run (default: all)")
    ap.add_argument("--ids-file", default=None, help="a JSON list of ids (or {ids: [...]}) to run")
    ap.add_argument("--card-exact", action="store_true", help="only the card's four code blocks")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--fp64", action="store_true", help="the provider in float64 end to end (needs fp64_patch.py, "
                                                        "not in this folder): weights bf16 -> float64")
    a = ap.parse_args()

    import numpy as np
    import torch
    from transformers import AutoModel

    from records import load_records, open_request

    torch.set_num_threads(a.threads)
    out_path = os.path.join(WORK, a.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    t0 = time.time()
    model = AutoModel.from_pretrained(SNAP, trust_remote_code=True,
                                      dtype=torch.float64 if a.fp64 else torch.float32)
    model.eval()
    engine = model.engine
    patched = []
    if a.fp64:
        sys.path.insert(0, HERE)
        import fp64_patch

        patched = fp64_patch.patch(model, engine)
    load_s = time.time() - t0
    env = {"dtype": "float64" if a.fp64 else "float32", "fp64_patched": patched, "torch": torch.__version__,
           "threads": torch.get_num_threads(), "load_s": round(load_s, 2), "set_size": engine.set_size,
           "attention": engine.attention, "max_pixels": engine.max_pixels, "d1_codes": engine.tokenizer.d1_codes,
           "d1_numbers": engine.tokenizer.d1_numbers, "box_edges": engine.tokenizer.d1_box_edges, "lead": engine.lead,
           "temperature": getattr(engine.calibration, "default", None),
           "model_class": type(model).__module__ + "." + type(model).__name__,
           "audio_attn_implementation": model.model.audio_encoder.lfm.config._attn_implementation,
           "transformers_lfm3": "vendored" if "transformers_modules" in type(model.model.language_model).__module__
           else type(model.model.language_model).__module__}

    if a.card_exact:
        with torch.inference_mode():
            res = card_exact(model)
        doc = {"what": "the model card's four examples, run as the card writes them", "snapshot": SNAP, "env": env,
               "results": res, "all_match": all(v["match"] for v in res.values()), "wall_s": round(time.time() - t0, 2)}
        json.dump(doc, open(out_path, "w"), indent=1)
        for k, v in res.items():
            print("CARD", k, "match" if v["match"] else "MISMATCH", json.dumps(jsonable(v["got"])), "want",
                  json.dumps(jsonable(v["want"])))
        print("CARD_ALL", "MATCH" if doc["all_match"] else "MISMATCH")
        return

    recs, media = load_records(a.files, a.media)
    want = None
    if a.ids:
        want = set(a.ids.split(","))
    if a.ids_file:
        ids = json.load(open(os.path.join(WORK, a.ids_file)))
        want = (want or set()) | set(ids["ids"] if isinstance(ids, dict) else ids)
    if want is not None:
        recs = [r for r in recs if r["id"] in want]
    if a.limit:
        recs = recs[: a.limit]
    hid_path = os.path.join(WORK, a.hidden) if a.hidden else out_path.replace(".json", "_slots.npz")
    doc = json.load(open(out_path)) if os.path.exists(out_path) else {
        "what": f"provider reference, {env['dtype']} CPU, one request per call", "snapshot": SNAP, "records": {}}
    arrays = dict(np.load(hid_path)) if os.path.exists(hid_path) else {}
    doc["env"] = env
    tap = Tap(engine, model)
    todo = [r for r in recs if r["id"] not in doc["records"]]
    for n, rec in enumerate(todo):
        rid = rec["id"]
        tap.reset()
        t1 = time.time()
        try:
            state, questions, items = open_request(rec, media)
            named = engine.named(questions)
            with torch.inference_mode():
                done = engine.run([(state, list(named.values()), items)])
            probs, read = done[0]
            answers = {name: engine.answer(state, q, p) for (name, q), p in zip(named.items(), probs)}
            err = None
        except Exception as e:  # a refusal is a result too (e.g. a box over two pictures)
            probs, read, answers, err = None, None, None, f"{type(e).__name__}: {e}"
        dt = time.time() - t1
        row = {"answers": jsonable(answers), "probs": jsonable(probs), "usage": {"input_tokens": read, "output_tokens": 0},
               "raw": jsonable(tap.raw), "readout_kinds": list(tap.readout_kinds), "seqs": list(tap.seqs),
               "time_s": round(dt, 4), "error": err, "question_names": list(rec["request"]["questions"])}
        for j, h in enumerate(tap.hidden):
            arrays[f"{rid}::slots{j}"] = h
        for j, pz in enumerate(tap.positions):
            arrays[f"{rid}::positions{j}"] = pz
        for j, au in enumerate(tap.audio):
            arrays[f"{rid}::audio{j}"] = au
        doc["records"][rid] = row
        if n % 10 == 0 or n == len(todo) - 1:
            json.dump(doc, open(out_path + ".tmp", "w"))
            os.replace(out_path + ".tmp", out_path)
            np.savez(hid_path + ".tmp.npz", **arrays)
            os.replace(hid_path + ".tmp.npz", hid_path)
        print(f"{rid} {dt:.2f}s tokens={read} err={err}", flush=True)
    doc["wall_s"] = round(time.time() - t0, 2)
    json.dump(doc, open(out_path + ".tmp", "w"))
    os.replace(out_path + ".tmp", out_path)
    np.savez(hid_path + ".tmp.npz", **arrays)
    os.replace(hid_path + ".tmp.npz", hid_path)
    print("REF_DONE", len(doc["records"]), "records", round(time.time() - t0, 1), "s")


if __name__ == "__main__":
    main()
