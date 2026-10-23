#!/usr/bin/env python3
"""make_int4_27b.py EXPORT_DIR OUT_DIR b32|b128 TEMPLATE.jinja - one Mixed INT4 .litertlm from the float export of
convert_qwen38_hybrid_lw.py (its prefill_decode TFLite, without a float .litertlm); one step of convert_qwen38_27b.sh. The Mixed INT4
recipe of make_int4_9b.py with the TFLite as the input:
  1 quantize EXPORT_DIR's prefill_decode TFLite (export_artifacts.json) with the recipe object of ../minicpm5_work/quantize_minicpm5.py
    build_recipe("wi4b32_wi8" | "wi4b128_wi8", "minmax", [QWEN35_FLOAT_FC_REGEX]) (loaded by path) = int4 blockwise-32 / -128
    min-max on every FULLY_CONNECTED, int8 channelwise on the lm_head FC and the embedding, NO_QUANTIZE on 'Linear_in_proj_(a|b);'
    (the GatedDeltaNet decay / gate projections); clamp-before-quantize: ai-edge-quantizer's
    uniform_quantize_tensor.tensor_zp_scale_from_min_max is wrapped so a blockwise 4-bit symmetric scale is max(scale, 2^-14) and the
    weights are quantized with that floor (a block of zeros or near-zeros otherwise gets a zero or fp16-subnormal scale; the naive
    min-max path calls the function through the module, so the wrapper is seen); prints CLAMP_TOTAL; the quantized model is written
    by `qt.quantize(serialize_to_path=...)` into the output file's mmap (the same _serialize_model bytes as
    `quantize().export_model()`, without a bytearray of the whole output in RAM: for the 27B b128 the peak memory footprint was
    46.2 GB);
  2 LlmMetadata = the packager's own (EXPORT_DIR/llm_metadata.pb: stop tokens / max_num_tokens / llm_model_type / sampler as the
    export built them; <|im_end|> 248046 added to the stop tokens only if missing) -> make_new_pbtext.py (no --template-only:
    jinja_prompt_template = TEMPLATE byte for byte, channels thought "<think>\\n" / "\\n</think>", supports_thinking true,
    supports_function_calling false; its check.json must say PASS);
  3 litert-lm-builder pack: that pbtext + the export's tokenizer.json + the quantized TFLite (prefill_decode);
  4 ../scripts/add_executor_metadata.py (the state buffers); 5 ../scripts/set_activation_type.py --type fp32.
convert_qwen38_27b.sh then puts the model repo's tokenizer.json in the tokenizer section (use_repo_tokenizer.sh) and runs the checks
(fc_dtypes.py / const_ops.py / scan_scales.py / describe). Each intermediate is removed once the next step has read it.
Output OUT_DIR/<QWEN35_NAME>_mixed_int4.litertlm (b32) / _mixed_int4_b128.litertlm; records in OUT_DIR/work_<block>/.
Env: QWEN35_NAME (default Qwen3.8-27B), QWEN35_FLOAT_FC_REGEX (';;'-separated, default 'Linear_in_proj_(a|b);'), RUN_PY (a Python
with litert-lm-builder 0.18.1) and LITERT_LM (the litert-lm 0.18.1 CLI), LITERT_LM_BUILDER (default: the litert-lm-builder next to
RUN_PY), TMPDIR (unpack / pack scratch of steps 4-5), KEEP_TFLITE=1 (keep the quantized TFLite, moved to QTFL_KEEP_DIR when set).
Run with the export's Python (ai-edge-quantizer 0.8.0)."""
import atexit

import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RPY, CLI = os.environ["RUN_PY"], os.environ["LITERT_LM"]
BUILDER = os.environ.get("LITERT_LM_BUILDER") or os.path.join(os.path.dirname(RPY), "litert-lm-builder")


def load_by_path(name, path):
    """Load a sibling tool by its path; undo the sys.path entry it inserts at import."""
    before = list(sys.path)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sys.path[:] = before
    return mod


def clamp_patch(stats):
    from ai_edge_quantizer.algorithms.uniform_quantize import uniform_quantize_tensor as u
    floor = np.float32(2.0 ** -14)
    orig = u.tensor_zp_scale_from_min_max

    def clamped(min_value, max_value, num_bits, symmetric, granularity, clipping_values=None):
        zp, scale = orig(min_value, max_value, num_bits, symmetric, granularity, clipping_values)
        if u.is_blockwise(granularity) and num_bits == 4:
            if not symmetric:
                raise ValueError("make_int4_27b: asymmetric blockwise int4 is not covered by the floor")
            low = scale < floor
            k = int(np.count_nonzero(low))
            stats["blockwise_int4_calls"] += 1
            if k:
                scale = np.where(low, floor, scale).astype(scale.dtype)
                stats["tensors"] += 1
                stats["scales"] += k
                print(f"CLAMP floor 2^-14: {k} of {scale.size} block scales", flush=True)
        return zp, scale

    u.tensor_zp_scale_from_min_max = clamped
    atexit.register(lambda: print(f"CLAMP_TOTAL tensors={stats['tensors']} scales={stats['scales']} "
                                  f"(blockwise int4 calls {stats['blockwise_int4_calls']})", flush=True))


def run(cmd, log, **kw):
    print("### CMD:", " ".join(cmd), flush=True)
    with open(log, "a") as f:
        f.write("### CMD: " + " ".join(cmd) + "\n")
    subprocess.run(cmd, check=True, **kw)


def drop(path):
    if os.path.isdir(path):
        n = sum(os.path.getsize(os.path.join(r, x)) for r, _, fs in os.walk(path) for x in fs)
        shutil.rmtree(path)
    elif os.path.exists(path):
        n = os.path.getsize(path)
        os.remove(path)
    else:
        return
    print(f"removed {n} {path}", flush=True)


def main():
    if len(sys.argv) != 5:
        print(__doc__, file=sys.stderr)
        return 2
    exp, out, block, tpl = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2]), sys.argv[3], os.path.abspath(sys.argv[4])
    assert block in ("b32", "b128"), block
    name = os.environ.get("QWEN35_NAME", "Qwen3.8-27B") + ("_mixed_int4" if block == "b32" else "_mixed_int4_b128")
    regexes = [r for r in os.environ.get("QWEN35_FLOAT_FC_REGEX", "Linear_in_proj_(a|b);").split(";;") if r]
    final = os.path.join(out, name + ".litertlm")
    work = os.path.join(out, f"work_{block}")
    os.makedirs(work, exist_ok=False)
    log = os.path.join(work, "cmd.log")
    art = json.load(open(os.path.join(exp, "export_artifacts.json")))
    tfl = art["artifacts"]["prefill_decode_model_path"]
    tok = art["artifacts"]["tokenizer_model_path"]
    rec = {"export_dir": exp, "tflite": tfl, "tflite_bytes": os.path.getsize(tfl), "tokenizer": tok, "block": block,
           "float_fc_regexes": regexes, "template": tpl, "run_py": RPY, "steps": {}}
    t0 = time.time()

    # 1. quantize (the recipe + the clamp floor)
    stats = {"tensors": 0, "scales": 0, "blockwise_int4_calls": 0}
    clamp_patch(stats)
    q5 = load_by_path("quantize_minicpm5", os.path.join(ROOT, "minicpm5_work", "quantize_minicpm5.py"))
    recipe = q5.build_recipe({"b32": "wi4b32_wi8", "b128": "wi4b128_wi8"}[block], "minmax", regexes)
    json.dump(recipe, open(os.path.join(work, "recipe.json"), "w"), indent=1, default=str)
    from ai_edge_quantizer import quantizer
    qt = quantizer.Quantizer(tfl, recipe)
    if qt.need_calibration:
        raise SystemExit("recipe unexpectedly needs calibration")
    qpath = os.path.join(work, "model_q.tflite")
    qt.quantize(serialize_to_path=qpath)  # serialized into the output file's mmap (no bytearray of the whole output)
    rec["steps"]["quantize"] = {"s": round(time.time() - t0, 1), "bytes": os.path.getsize(qpath), "clamp": dict(stats)}
    print(f"quantized {os.path.getsize(qpath)} B in {time.time() - t0:.0f} s; clamp {stats}", flush=True)
    del qt

    # 2. LlmMetadata: the export's + 248046 if missing -> make_new_pbtext (switch form)
    from litert_lm_builder.runtime.proto import llm_metadata_pb2
    raw = open(art["llm_metadata_pb"], "rb").read()
    md = llm_metadata_pb2.LlmMetadata()
    md.ParseFromString(raw)
    ids = [i for s in md.stop_tokens for i in s.token_ids.ids]
    added = []
    if 248046 not in ids:
        md.stop_tokens.add().token_ids.ids.append(248046)
        added.append(248046)
    meta_dir = os.path.join(work, "meta_export")
    os.makedirs(meta_dir)
    open(os.path.join(meta_dir, "LlmMetadataProto.bin"), "wb").write(md.SerializeToString())
    rec["steps"]["metadata"] = {"export_llm_metadata_sha256": hashlib.sha256(raw).hexdigest(), "export_stop_ids": ids,
                                "stop_ids_added": added}
    pbtext = os.path.join(work, f"{name}.pbtext")
    run([RPY, "-I", os.path.join(HERE, "make_new_pbtext.py"), meta_dir, tpl, pbtext], log)
    chk = json.load(open(pbtext + ".check.json"))
    if chk.get("status") != "PASS":
        raise SystemExit(f"make_new_pbtext check {chk.get('status')}")
    rec["steps"]["metadata"]["pbtext_check"] = {k: chk[k] for k in ("changed_fields", "template_sha256", "new_pbtext_sha256")}

    # 3. pack
    packed = os.path.join(work, "packed.litertlm")
    run([BUILDER, "llm_metadata", "--path", pbtext, "hf_tokenizer", "--path", tok,
         "tflite_model", "--path", qpath, "--model_type", "prefill_decode", "output", "--path", packed], log)
    if not os.path.exists(packed) or os.path.getsize(packed) == 0:
        raise SystemExit("pack wrote nothing")
    if os.environ.get("KEEP_TFLITE") == "1":  # e.g. for a decode-walk check of the quantized graph (= the bundle's TFLite section)
        h = hashlib.sha256()
        with open(qpath, "rb") as f:
            for c in iter(lambda: f.read(64 << 20), b""):
                h.update(c)
        keep_dir = os.environ.get("QTFL_KEEP_DIR")  # e.g. another disk
        if keep_dir:
            os.makedirs(keep_dir, exist_ok=True)
            dst = os.path.join(keep_dir, f"{name}.tflite")
            shutil.move(qpath, dst)
            qpath = dst
        rec["steps"]["quantize"]["tflite_kept"] = qpath
        rec["steps"]["quantize"]["tflite_sha256"] = h.hexdigest()
    else:
        drop(qpath)
    # 4. ExecutorMetadata
    meta = os.path.join(work, "meta.litertlm")
    run([sys.executable, os.path.join(ROOT, "scripts", "add_executor_metadata.py"), packed, meta, "--litert-lm", CLI], log)
    drop(packed)
    # 5. fp32 activations
    run([sys.executable, os.path.join(ROOT, "scripts", "set_activation_type.py"), meta, final, "--type", "fp32",
         "--litert-lm", CLI], log)
    drop(meta)
    rec["final"] = final
    rec["final_bytes"] = os.path.getsize(final)
    rec["total_s"] = round(time.time() - t0, 1)
    json.dump(rec, open(os.path.join(work, "build.json"), "w"), indent=1)
    print("DONE:", final, rec["final_bytes"], "B", f"{rec['total_s']} s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
