"""The provider's reference for d1-2B: LiquidAI/d1-2B @ 10e7671d with its own code, float32 on the CPU, one request
per call (batch 1: no padding unless a request has more questions than the set size), unrounded.

For every record of the given files (the fixtures' requests.json form) it writes, by record id:
  answers  the dict `system_one` returns (composed exactly as `SystemOneApi.system_one_batch` composes it)
  probs    what `SlotReader._padded` returns per question (the input of `api.answer`)
  raw      for option questions (noul / choice / score / multiple), the readout before the temperature
  seqs     per sequence: ids, state_len, question tags, slot positions (`ends`), media grids
  usage, time_s
and the hidden states at every slot (`slots.split`'s input, [S, 2048] per batch) to an npz, plus the pointer position
logits of read slots.

Run from the work directory with $REF:
  $REF conversion/ref_run.py --files set/card.json --out out/ref/card_fp32.json --card
  $REF conversion/ref_run.py --files set/requests.json set/red_arms.json --media set/media.json \
      --out out/ref/ref_fp32.json
  $REF conversion/ref_run.py --files set/extra_g96x72.json --media set/media_extra.json --out out/ref/ref_fp32.json
Resumes: ids already in --out are skipped (the file is rewritten after every record).
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
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HOME", os.path.join(WORK, "out/hf_home"))          # never the shared cache
os.environ.setdefault("HF_MODULES_CACHE", os.path.join(WORK, "out/hf_modules"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


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


def load_records(paths):
    recs, media = [], {}
    for p in paths:
        doc = json.load(open(os.path.join(WORK, p) if not os.path.isabs(p) else p))
        if isinstance(doc, list):  # red_arms.json: a list of arms with a full request each
            for arm in doc:
                recs.append({"id": arm["id"], "source": "red_arm", "kind": arm.get("kind"), "request": arm["request"]})
            continue
        recs.extend(doc["records"])
        media.update(doc.get("media", {}))
    return recs, media


def load_media_table(path):
    if not path:
        return {}
    return json.load(open(os.path.join(WORK, path)))


class Tap:
    """Instruments the provider's engine without changing what it computes."""

    def __init__(self, engine):
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
                self.seqs.append({"first_question": k, "ids": list(ids), "state_len": int(state_len),
                                  "question": None if question is None else list(question),
                                  "ends": [list(e) for e in ends],
                                  "image_grid_thw": None if grid is None else grid.tolist(),
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

    def reset(self):
        self.hidden, self.seqs, self.positions, self.raw, self.readout_kinds = [], [], [], [], []


def card_check(rec, answers):
    a = answers
    rid = rec["id"]
    if rid == "card_ticket":
        u = a["urgency"]
        got = (round(a["refund"]["noul"], 2), a["team"]["choice"], a["topics"]["choices"],
               u["legend"][str(int(u["score"]))])
        want = (1.0, "billing", ["billing", "delivery"], "medium")
    elif rid == "card_invoice":
        got = (a["payee"]["text"], a["total"]["number"], a["due"]["date"], a["vat"]["number"])
        want = ("ACME SA", 1234.5, "2026-04-03", None)
    elif rid == "card_coco":
        got = (a["cats"]["number"], a["every_cat"]["count"], {k: round(v, 2) for k, v in a["remote"]["box"].items()})
        want = (2.0, 2, {"top": 0.15, "bottom": 0.24, "left": 0.06, "right": 0.28})
    elif rid in ("card_batch_1", "card_batch_2"):
        got = (a["team"]["choice"],)
        want = ("shipping",) if rid == "card_batch_1" else ("technical",)
    else:
        return None
    return {"got": jsonable(got), "want": jsonable(want), "match": jsonable(got) == jsonable(want)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="+", required=True)
    ap.add_argument("--media", default=None, help="media.json (records' media ids); card.json carries its own")
    ap.add_argument("--out", required=True)
    ap.add_argument("--hidden", default=None, help="npz of slot hidden states (default: --out with _slots.npz)")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--ids", default=None, help="comma-separated ids to run (default: all)")
    ap.add_argument("--card", action="store_true", help="also run model.system_one / system_one_batch on the card "
                                                        "examples and check the card's printed values")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--fp64", action="store_true", help="the provider in float64 end to end (needs fp64_patch.py, "
                                                        "not in this folder): weights bf16 -> float64")
    a = ap.parse_args()

    import numpy as np
    import torch
    from PIL import Image
    from transformers import AutoModel

    torch.set_num_threads(a.threads)
    recs, media = load_records(a.files)
    media.update(load_media_table(a.media))
    if a.ids:
        want = set(a.ids.split(","))
        recs = [r for r in recs if r["id"] in want]
    if a.limit:
        recs = recs[: a.limit]
    out_path = os.path.join(WORK, a.out)
    hid_path = os.path.join(WORK, a.hidden) if a.hidden else out_path.replace(".json", "_slots.npz")
    doc = json.load(open(out_path)) if os.path.exists(out_path) else {
        "what": "provider reference, float32 CPU, one request per call", "snapshot": SNAP, "records": {}}
    arrays = dict(np.load(hid_path)) if os.path.exists(hid_path) else {}

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
    doc["env"] = {"dtype": "float64" if a.fp64 else "float32", "fp64_patched": patched, "torch": torch.__version__, "threads": torch.get_num_threads(), "load_s": round(load_s, 2),
                  "set_size": engine.set_size, "attention": engine.attention, "max_pixels": engine.max_pixels,
                  "d1_codes": engine.tokenizer.d1_codes, "d1_numbers": engine.tokenizer.d1_numbers,
                  "box_edges": engine.tokenizer.d1_box_edges, "lead": engine.lead,
                  "temperature": getattr(engine.calibration, "default", None),
                  "model_class": type(model).__module__ + "." + type(model).__name__,
                  "transformers_lfm3": "vendored" if "transformers_modules" in type(model.model.language_model).__module__
                  else type(model.model.language_model).__module__}
    tap = Tap(engine)

    def open_media(ids):
        items = []
        for mid in ids or []:
            m = media[mid]
            p = os.path.join(WORK, m["path"])
            data = open(p, "rb").read()
            if m.get("sha256") and hashlib.sha256(data).hexdigest() != m["sha256"]:
                raise SystemExit(f"media {mid}: sha256 mismatch")
            items.append(Image.open(p))
        return items

    for n, rec in enumerate(recs):
        rid = rec["id"]
        if rid in doc["records"]:
            continue
        req = rec["request"]
        state, questions, items = req.get("state"), req["questions"], open_media(req.get("media"))
        tap.reset()
        t1 = time.time()
        try:
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
               "raw": jsonable(tap.raw), "readout_kinds": list(tap.readout_kinds), "seqs": list(tap.seqs), "time_s": round(dt, 4),
               "error": err, "question_names": list(questions)}
        for j, h in enumerate(tap.hidden):
            arrays[f"{rid}::slots{j}"] = h
        for j, pz in enumerate(tap.positions):
            arrays[f"{rid}::positions{j}"] = pz
        if a.card and rec.get("source") == "card" and answers is not None:
            row["card_check"] = card_check(rec, answers)
            with torch.inference_mode():
                direct = model.system_one(state, questions, media=open_media(req.get("media")))
            row["system_one_equal"] = jsonable(direct["answers"]) == row["answers"]
        doc["records"][rid] = row
        if n % 10 == 0 or n == len(recs) - 1:
            json.dump(doc, open(out_path + ".tmp", "w"))
            os.replace(out_path + ".tmp", out_path)
            np.savez(hid_path + ".tmp.npz", **arrays)
            os.replace(hid_path + ".tmp.npz", hid_path)
        print(f"{rid} {dt:.2f}s tokens={read} err={err}", flush=True)

    if a.card:
        batch = [r for r in recs if r["id"] in ("card_batch_1", "card_batch_2")]
        if len(batch) == 2:
            with torch.inference_mode():
                outs = model.system_one_batch([(r["request"]["state"], r["request"]["questions"]) for r in batch])
            doc["card_batch_call"] = {"choices": [o["answers"]["team"]["choice"] for o in outs],
                                      "probabilities": [jsonable(o["answers"]["team"]["probabilities"]) for o in outs],
                                      "want": ["shipping", "technical"]}
    doc["wall_s"] = round(time.time() - t0, 2)
    json.dump(doc, open(out_path + ".tmp", "w"))
    os.replace(out_path + ".tmp", out_path)
    np.savez(hid_path + ".tmp.npz", **arrays)
    os.replace(hid_path + ".tmp.npz", hid_path)
    if a.card:
        for rid, row in doc["records"].items():
            if "card_check" in row:
                print("CARD", rid, json.dumps(row["card_check"]), "system_one_equal", row.get("system_one_equal"))
        if "card_batch_call" in doc:
            print("CARD batch", json.dumps(doc["card_batch_call"]))


if __name__ == "__main__":
    main()
