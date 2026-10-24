#!/usr/bin/env python3
"""Round 6: is LiteRT-LM's second-turn defect (the previous turn's stop token enters the KV cache with a zero
embedding, r2/twoturn_sim.json on WeVisOne-VL-2B) also on a text bundle with an EMBEDDER section?

Bundle: litert-community/Qwen2.5-Coder-1.5B-Instruct `Qwen2.5-Coder-1.5B-Instruct_int4.litertlm` (rev 2bdadb12, sha256
273ecc77…, sections LlmMetadata + HF tokenizer + tf_lite_prefill_decode + tf_lite_embedder), unpacked with
`litert-lm-peek --dump_files_dir` into out/r6/coder/unpack.

  runtime  (litert-lm 0.18.1 Python API, CPU, 1 thread, no cache, greedy) one conversation of two user turns per
           process: each turn's own render, token counts, prefill / decode counts and answer -> --out JSON
  replay   (ai-edge-litert CompiledModel, CPU, 1 thread) the two graphs of the same file on the runtime's schedule
           (wv_readout.Decoder / chunk_plan: prompt[:-1] through prefill signatures largest first, the last token
           decoded): turn 1 from the runtime's turn-1 render, greedy to a stop id; turn 2 = the pending stop token +
           the runtime's turn-2 render, with the stop token's row as (a) its embedder row, (b) zeros, (c) the runtime's
           write for a token without an embedding (the int32 id in the first 4 bytes, zeros after:
           FillInputBufferWithToken, v0.18.1 llm_litert_compiled_model_executor.cc L413-447) -> --out JSON with
           each turn-2 text and whether it equals the runtime's

    ~/venvs/lt0181run/bin/python3 -I tools/r6_twoturn_probe.py runtime --pair 0 --out r6/twoturn/rt_p0_a.json
    ~/venvs/lt0100dev/bin/python -I tools/r6_twoturn_probe.py replay --runtime r6/twoturn/rt_p0_a.json ... \
        --out r6/twoturn/replay.json
"""
import argparse
import json
import sys
import zlib
from pathlib import Path

K = Path(__file__).resolve().parents[1]
UNPACK = K / "out" / "r6" / "coder" / "unpack"
BUNDLE = K / "out" / "r6" / "coder" / "Qwen2.5-Coder-1.5B-Instruct_int4.litertlm"
STOP = (151645, 151643)
PAIRS = [
    ["My name is Ada.", "What is my name?"],
    ["Remember this number: 4182.", "What number did I ask you to remember?"],
    ["List three fruits.", "Which of them is yellow?"],
    ["My favorite color is green.", "What is my favorite color? Answer in one word."],
]


def run_runtime(a):
    import litert_lm
    turns = PAIRS[a.pair]
    res = {"bundle": str(BUNDLE), "pair": a.pair, "turns_text": turns, "runtime": litert_lm.__name__,
           "backend": "cpu", "threads": 1, "turns": []}

    def counts(conv):
        lib = conv._lib
        ptr = lib.litert_lm_conversation_get_benchmark_info(conv._ptr)
        try:
            return ([lib.litert_lm_benchmark_info_get_prefill_token_count_at(ptr, i)
                     for i in range(lib.litert_lm_benchmark_info_get_num_prefill_turns(ptr))],
                    [lib.litert_lm_benchmark_info_get_decode_token_count_at(ptr, i)
                     for i in range(lib.litert_lm_benchmark_info_get_num_decode_turns(ptr))])
        finally:
            lib.litert_lm_benchmark_info_delete(ptr)

    with litert_lm.Engine(str(BUNDLE), backend=litert_lm.Backend.CPU(thread_count=1), cache_dir=":nocache",
                          enable_benchmark=True) as eng:
        with eng.create_conversation(sampler_config=litert_lm.SamplerConfig(temperature=0.0, seed=0)) as conv:
            for i, user in enumerate(turns):
                msg = {"role": "user", "content": [{"type": "text", "text": user}]}
                rendered = conv.render_message_to_string(msg)
                before = conv.token_count
                reply = conv.send_message(msg)
                if isinstance(reply, dict):
                    text = "".join(p.get("text", "") for p in reply.get("content", []) if isinstance(p, dict))
                else:
                    parts = getattr(reply, "content", None)
                    text = "".join(getattr(p, "text", "") or "" for p in parts) if parts else str(reply)
                pre, dec = counts(conv)
                res["turns"].append({"turn": i + 1, "user": user, "render": rendered,
                                     "render_ids": list(eng.tokenize(rendered)), "token_count_before": before,
                                     "token_count_after": conv.token_count, "prefill_counts": pre,
                                     "decode_counts": dec, "answer": text})
    Path(K / a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(K / a.out).write_text(json.dumps(res, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps([(t["turn"], t["token_count_before"], t["token_count_after"], t["prefill_counts"],
                       t["decode_counts"], t["answer"][:100]) for t in res["turns"]], ensure_ascii=False))
    return 0


def run_replay(a):
    import importlib.util
    import numpy as np
    from tokenizers import Tokenizer
    spec = importlib.util.spec_from_file_location("wv_readout", Path(__file__).resolve().parent / "wv_readout.py")
    RO = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(RO)
    raw = (UNPACK / "Section1_HF_Tokenizer_Zlib.zlib").read_bytes()
    tk = Tokenizer.from_str(zlib.decompress(raw[8:]).decode("utf-8"))
    dec = RO.Decoder(UNPACK / "Section2_TFLiteModel_tf_lite_prefill_decode.tflite", 1, None)
    emb_model = RO.compiled(UNPACK / "Section3_TFLiteModel_tf_lite_embedder.tflite", 1)
    key = "decode_embedder"
    ein = emb_model.get_input_tensor_details(key)
    eout = emb_model.get_output_tensor_details(key)
    (iname, idet), = ein.items()
    (oname, odet), = eout.items()
    rows = {}

    def embed(t):
        if t not in rows:
            ib = emb_model.create_input_buffer_by_name(key, iname)
            ob = emb_model.create_output_buffer_by_name(key, oname)
            try:
                ib.write(np.array([[t]], dtype=np.int32))
                emb_model.run_by_name(key, {iname: ib}, {oname: ob})
                rows[t] = ob.read(int(np.prod(odet["shape"])), odet["dtype"]).reshape(-1).copy()
            finally:
                ib.destroy()
                ob.destroy()
        return rows[t]

    def prefill_rows(rows_arr, start, states):
        for k, s, n in RO.chunk_plan(len(rows_arr)):
            states, _ = dec.step(k, rows_arr[s:s + n], np.arange(start + s, start + s + n), states)
        return states

    def greedy(states, pos, cur_row, max_new):
        out = []
        for _ in range(max_new):
            states, lg = dec.step("decode", cur_row[None], np.array([pos]), states)
            t = int(lg.reshape(-1).argmax())
            out.append(t)
            if t in STOP:
                break
            cur_row, pos = embed(t), pos + 1
        return out, states

    report = {"bundle": str(BUNDLE), "replay": "ai-edge-litert CompiledModel CPU, 1 thread", "pairs": []}
    for path in a.runtime:
        rt = json.loads((K / path).read_text())
        t1, t2 = rt["turns"]
        p1, inc = t1["render_ids"], t2["render_ids"]
        width = dec.width
        st = prefill_rows(np.stack([embed(t) for t in p1[:-1]]), 0, dec.zero_states())
        gen1, st1 = greedy(st, len(p1) - 1, embed(p1[-1]), a.max_new)
        text1 = tk.decode(gen1, skip_special_tokens=True)
        start = len(p1) + len(gen1) - 1          # the pending stop token's position
        exact = np.zeros(width, np.float32)
        exact.view(np.int32)[0] = gen1[-1]
        hyp = {}
        for name, first in (("stop_row_embedded", embed(gen1[-1])), ("stop_row_zero", np.zeros(width, np.float32)),
                            ("stop_row_runtime_write", exact)):
            rows_arr = np.stack([first] + [embed(t) for t in inc[:-1]])
            s2 = prefill_rows(rows_arr, start, st1)
            g2, _ = greedy(s2, start + len(inc), embed(inc[-1]), a.max_new)
            tx = tk.decode(g2, skip_special_tokens=True)
            hyp[name] = {"text": tx, "n_tokens": len(g2), "equals_runtime_turn2": tx.strip() == t2["answer"].strip()}
        entry = {"runtime_json": path, "pair": rt["pair"], "turns_text": rt["turns_text"],
                 "turn1_replay": text1, "turn1_runtime": t1["answer"],
                 "turn1_equal": text1.strip() == t1["answer"].strip(), "turn1_ids": gen1,
                 "kv_after_turn1_replay": len(p1) + len(gen1), "kv_after_turn1_runtime": t1["token_count_after"],
                 "turn2_increment_ids": len(inc), "turn2_prefill_count_runtime": t2["prefill_counts"],
                 "turn2_runtime": t2["answer"], "hypotheses": hyp}
        report["pairs"].append(entry)
        print(json.dumps({k: entry[k] for k in ("pair", "turn1_equal", "kv_after_turn1_replay",
                                                "kv_after_turn1_runtime")},
                         ensure_ascii=False), {k: v["equals_runtime_turn2"] for k, v in hyp.items()}, flush=True)
    Path(K / a.out).write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n")
    return 0


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)
    r = sub.add_parser("runtime")
    r.add_argument("--pair", type=int, required=True)
    r.add_argument("--out", required=True)
    p = sub.add_parser("replay")
    p.add_argument("--runtime", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--max-new", type=int, default=160)
    a = ap.parse_args()
    return run_runtime(a) if a.mode == "runtime" else run_replay(a)


if __name__ == "__main__":
    sys.exit(main())
