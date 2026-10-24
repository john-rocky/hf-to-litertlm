#!/usr/bin/env python3
"""Round 2: the two-turn text row through the fp32 graphs (CompiledModel CPU, tools/wv_readout.py's classes), two ways:

  carried   the runtime's schedule: turn 1 = prefill prompt[:-1], decode from the last prompt token, greedy to the
            stop token; turn 2 = the stop token (the pending token) + T0's turn-2 increment through prefill, decode
            from the increment's last token, greedy — the KV of turn 1's answer comes from decode steps
  fresh     T0's own turn-2 prompt (55 ids) prefilled at once, decode, greedy (= what ref/T0 measured in HF)
Prints both turn-2 answers next to T0's and the runtime's (r2/suite/fp32_gpu). -> r2/twoturn_sim.json

    ~/venvs/lt0100dev/bin/python -I tools/wv_twoturn_sim.py
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

K = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("wv_readout", Path(__file__).resolve().parent / "wv_readout.py")
RO = importlib.util.module_from_spec(spec)
spec.loader.exec_module(RO)
STOP = (151645, 151643)


class Args:
    dec, emb, enc, ada, vision, threads = ("out/r2/dec_fp32/model.tflite", "out/r2/dec_fp32/embedder.tflite", "",
                                           "", "hf", 8)
    label = "fp32_graph"         # reuse the readout's XNNPACK weight cache file


def greedy(ro, states, pos, cur_emb, max_new=128):
    out = []
    for _ in range(max_new):
        states, lg = ro.dec.step("decode", cur_emb, np.array([pos]), states)
        t = int(lg.reshape(-1).argmax())
        out.append(t)
        if t in STOP:
            break
        cur_emb, pos = ro.embed(t)[None], pos + 1
    return out, states, pos


def prefill(ro, ids, start, states):
    emb = np.stack([ro.embed(t) for t in ids])
    for key, a, n in RO.chunk_plan(len(ids)):
        states, _ = ro.dec.step(key, emb[a:a + n], np.arange(start + a, start + a + n), states)
    return states


def main():
    from tokenizers import Tokenizer
    tk = Tokenizer.from_file(str(Path("~/.cache/huggingface/hub/models--tencent--WeVisOne-VL-2B/snapshots/"
                                      "3dad9ce8bcb6fbe0aea945bf8860bd2cfc3663bf/tokenizer.json").expanduser()))
    ro = RO.Readout(Args())
    t0 = json.loads((K / "ref" / "T0" / "two_turn.json").read_text())["records"]
    p1, p2 = t0[0]["prompt_ids"], t0[1]["prompt_ids"]
    # carried
    st = prefill(ro, p1[:-1], 0, ro.dec.zero_states())
    gen1, st, pos = greedy(ro, st, len(p1) - 1, ro.embed(p1[-1])[None])
    inc = p2[len(p1) + len(gen1):]                      # T0's turn-2 increment after prompt 1 + answer + stop
    assert p2[:len(p1) + len(gen1)] == p1 + gen1, "turn-1 answer differs from T0's turn-2 history"
    # the pending stop token sits at position len(p1) + len(gen1) - 1; the increment follows it
    start = len(p1) + len(gen1) - 1
    st = prefill(ro, [gen1[-1]] + inc[:-1], start, st)
    gen2c, _, _ = greedy(ro, st, start + len(inc), ro.embed(inc[-1])[None])
    # fresh
    st2 = prefill(ro, p2[:-1], 0, ro.dec.zero_states())
    gen2f, _, _ = greedy(ro, st2, len(p2) - 1, ro.embed(p2[-1])[None])
    rt = json.loads((K / "r2" / "suite" / "fp32_gpu" / "text" / "two_turn.json").read_text())["answers"]
    res = {"turn1_equals_T0": gen1 == t0[0]["new_ids"], "turn1": tk.decode(gen1),
           "carried_turn2": tk.decode(gen2c), "fresh_turn2": tk.decode(gen2f), "T0_turn2": t0[1]["answer"],
           "runtime_turn2": rt[1], "carried_equals_T0": gen2c == t0[1]["new_ids"],
           "fresh_equals_T0": gen2f == t0[1]["new_ids"], "increment_len": len(inc),
           "increment_text": tk.decode(inc)}
    # hypotheses for the runtime's turn 2 (each from the same post-turn-1 states)
    st1 = prefill(ro, p1[:-1], 0, ro.dec.zero_states())
    gen1b, st1, _ = greedy(ro, st1, len(p1) - 1, ro.embed(p1[-1])[None])
    assert gen1b == gen1
    width = ro.dec.width

    def run_turn2(first_rows, rest_ids, first_pos):
        emb = np.concatenate([first_rows, np.stack([ro.embed(t) for t in rest_ids[:-1]])]) if len(rest_ids) > 1 \
            else first_rows
        s = st1
        for key, a, n in RO.chunk_plan(len(emb)):
            s, _ = ro.dec.step(key, emb[a:a + n], np.arange(first_pos + a, first_pos + a + n), s)
        g, _, _ = greedy(ro, s, first_pos + len(emb), ro.embed(rest_ids[-1])[None])
        return tk.decode(g).strip()
    stop_row = ro.embed(gen1[-1])[None]
    hyp = {
        "as_runtime_should": run_turn2(stop_row, inc, start),
        "stop_embedding_zero": run_turn2(np.zeros((1, width), np.float32), inc, start),
        "stop_embedding_token0": run_turn2(ro.embed(0)[None], inc, start),
        "stop_token_missing": run_turn2(np.zeros((0, width), np.float32), inc, start),
        "positions_plus1": run_turn2(stop_row, inc, start + 1),
        "stop_token_twice": run_turn2(np.concatenate([stop_row, stop_row]), inc, start),
        "newline_missing": run_turn2(stop_row, inc[1:], start) if inc[0] == 198 else None,
    }
    res["hypotheses"] = {k: {"text": v, "equals_runtime": v == rt[1].strip(), "equals_T0": v == t0[1]["answer"].strip()}
                         for k, v in hyp.items() if v is not None}
    (K / "r2" / "twoturn_sim.json").write_text(json.dumps(res, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps(res, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
