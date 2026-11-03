"""convert.py's speech recognition route: a standalone ASR checkpoint -> one .litertlm with Speech2TextMetadata.

    uv run python scripts/convert.py <org>/<model> --task asr [--revision SHA] [--smoke] [--check]

Taken with --task asr, or without it when config.json's model_type is a standalone speech recognition type. Today
it converts transformers' ParakeetForTDT checkpoints (model_type parakeet_tdt: nvidia/parakeet-tdt-0.6b-v3 and its
finetunes saved by transformers); every other speech recognition type is refused with its reason. Design, the field
by field match with litert-torch PR #1426 and the measurements: docs/asr-route.md.

  1. entry gate   config.json and the repo's files (convert.py's gated check has already run)
  2. fetch        the configs, tokenizer.json and the safetensors at the pinned commit, each checked against the Hub
  3. convert      asr/export_parakeet_tdt.py in this environment (litert-torch 0.9.4, transformers 5.14.1): the
                  encode / decode / decode_1 graphs, int8 weights (dynamic_wi8_afp32), PyTorch checks
  4. bundle       asr/pack_speech2text.py in the asr/ environment (asr/uv.lock: litert-lm-builder 0.19.0) with the
                  header's uuid and timestamp derived from the source
  5. gate         asr/smoke.py in the asr/ environment (mediapipe 1.2.0 AudioTranscriber, CPU): --smoke = one
                  LibriSpeech clip, WER <= 0.2; otherwise 73 clips, WER <= 0.10 (litert-community's own
                  parakeet_tdt_0.6b_v3_5s_i8_stateful.litertlm scores 0.0609 on them, 2026-11-03)

Writes out/<name>/<name>_int8.litertlm, <name>.lock.json, convert_report.json, asr_work/ (the .tflite files and
the export record) and asr_gate.json. The report's status words and its check / export_seconds / gate_seconds
fields are the LLM path's (smoke_pass, converted_pass, built, ...), so a tool that reads convert_report.json
reads both routes the same way; the first clip is the check. Exit codes as convert.py: 0 converted and gated, 1 the gate failed, 2 refused, 3 failed.
"""
import json
import os
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ASR_ENV = REPO_ROOT / "asr"
EXPORTERS = {"parakeet_tdt": ASR_ENV / "export_parakeet_tdt.py"}
# Standalone speech recognition model types this route answers for (with a refusal when unsupported).
# LLM-based speech recognition (qwen3_asr, granite_speech, voxtral, ...) is not here: those are audio-in
# LLM bundles (LlmMetadata + an audio encoder), REPRODUCE.md "Qwen3-ASR-1.7B" and the other audio-in entries.
STANDALONE_ASR = {
    "parakeet_tdt": None,
    "parakeet_ctc": "an encoder-only CTC graph (Speech2TextMetadata TF_LITE_AUDIO_ENCODER); this route exports "
                    "TDT only so far",
    "parakeet_rnnt": "RNN-T without durations: LiteRT-LM v0.19.0's Parakeet runtime decodes TDT logits",
    "whisper": "Whisper's encoder-decoder graph is not exported by this route",
    "moonshine": "Moonshine's encoder-decoder graph is not exported by this route",
    "nemotron_asr_streaming": "a cache-aware streaming RNN-T; Speech2TextModelType has no entry for it "
                              "(nemotron_asr_stream_work/ converts it to plain .tflite graphs for a host loop)",
    "nemotron3_5_asr": "a cache-aware streaming RNN-T; Speech2TextModelType has no entry for it "
                       "(nemotron_asr_stream_work/ converts it to plain .tflite graphs for a host loop)",
}
SOURCE_FILES = ("config.json", "generation_config.json", "processor_config.json", "preprocessor_config.json",
                "tokenizer.json", "tokenizer_config.json")
INPUT_SEC, STATEFUL_AFTER = 5.0, 4  # litert-community/parakeet-tdt-0.6b-v3's 5 s stateful export
SMOKE_CLIPS, SMOKE_BAR = 1, 0.2
GATE_CLIPS, GATE_BAR = 73, 0.10
# the gate's clips (asr/smoke.py checks the file's sha256): 73 LibriSpeech dev-clean utterances, CC BY 4.0
CLIPS = {"repo": "hf-internal-testing/librispeech_asr_dummy", "revision": "5be91486e11a2d616f4ec5db8d3fd248585ac07a",
         "file": "clean/validation-00000-of-00001.parquet"}


def wants_asr(args, cfg):
    return args.task == "asr" or (args.task is None and (cfg or {}).get("model_type") in STANDALONE_ASR)


def clip_file():
    """The gate's clip parquet through the Hugging Face cache (HF_HUB_CACHE decides where; a batch runner can
    fill its cache beforehand with this function)."""
    from huggingface_hub import hf_hub_download
    return hf_hub_download(CLIPS["repo"], CLIPS["file"], repo_type="dataset", revision=CLIPS["revision"])


def side_env(cmd):
    """A command in the asr/ environment (uv run --frozen --project asr)."""
    uv = os.environ.get("UV") or shutil.which("uv")
    if not uv:
        return None
    return [uv, "run", "--frozen", "--project", str(ASR_ENV), "python", *map(str, cmd)]


def side_env_vars():
    """This process's environment without the variables that point uv at the converter's own environment:
    with UV_PROJECT_ENVIRONMENT set, `uv run --project asr` would install asr/uv.lock's packages into it."""
    return {k: v for k, v in os.environ.items() if k not in ("VIRTUAL_ENV", "UV_PROJECT_ENVIRONMENT")}


def run(cv, args, report, info, api, out_dir):
    """The route after convert.py's fetch of the repo info and its gated check. Returns the exit code."""
    report["decisions"]["task"] = "asr"
    sha = report["revision"]
    name = args.model.split("/")[-1]
    files = {s.rfilename for s in (info.siblings or [])}
    report["stage"] = "gate"
    cfg_text = cv.hub_text(args.model, "config.json", sha)
    if not cfg_text:
        cv.refuse(report, "asr_no_config", "no config.json: this route reads transformers checkpoints",
                  "point at a repo saved with transformers (a .nemo-only repo is NeMo's format)")
    cfg = json.loads(cfg_text)
    mt = cfg.get("model_type")
    report["decisions"]["model_type"] = mt
    if cfg.get("auto_map") and not cv.native_model_type(mt):
        cv.refuse(report, "remote_code", f"config.json declares auto_map and model_type {mt!r} is not in "
                  "transformers", "not converted: remote-code architectures are out of scope")
    if cfg.get("quantization_config"):
        cv.refuse(report, "pre_quantized", "weights are already quantized",
                  "convert from the full-precision source repo instead")
    if mt not in EXPORTERS:
        why = STANDALONE_ASR.get(mt) or f"model_type {mt!r} is not a standalone speech recognition type this " \
                                        "route knows"
        cv.refuse(report, "asr_unsupported", f"{mt}: {why}",
                  "supported: model_type parakeet_tdt (transformers ParakeetForTDT, e.g. "
                  "nvidia/parakeet-tdt-0.6b-v3); docs/asr-route.md lists the rest")
    if args.recipe != "int8":
        cv.refuse(report, "asr_recipe", f"--recipe {args.recipe}: the ASR route writes int8 weights only "
                  "(dynamic_wi8_afp32, the form of litert-community/parakeet-tdt-0.6b-v3's i8 file)",
                  "drop --recipe / --int4")
    # convert.py's own pick, narrowed to what this route reads (so a prefetch by pick_source_files covers it)
    picked = cv.pick_source_files(files)
    weights = [f for f in picked if f.endswith(".safetensors")]
    if not weights:
        cv.refuse(report, "no_weights", "no top-level safetensors file",
                  "point at a repo with transformers weights (a .nemo-only repo is NeMo's format)")
    if "tokenizer.json" not in files:
        cv.refuse(report, "asr_no_tokenizer", "no tokenizer.json (the bundle carries the repo's own file)",
                  "point at a repo that ships tokenizer.json")
    names = sorted(set(weights) | {f for f in picked if f in SOURCE_FILES})
    report["decisions"]["source_files"] = names
    report["decisions"]["params"] = getattr(getattr(info, "safetensors", None), "total", None)
    report["decisions"]["window"] = {"input_sec": INPUT_SEC, "stateful_after": STATEFUL_AFTER}
    if side_env(["-V"]) is None:
        cv.fail("uv not found: the bundle and gate steps run in the asr/ environment", stage="gate",
                code=2)
    if args.check:
        report["status"] = "accepted"
        cv.write_report(report)
        print(json.dumps(report, indent=2))
        return 0

    report["stage"] = "fetch"
    snap, cached = cv.download_source(report, args.model, sha, names)
    source_files = cv.verify_source(report, api, args.model, sha, snap, names, cached)
    toolchain = cv.toolchain_info()
    report["toolchain"] = {k: toolchain[k] for k in ("id", "python", "pinned_env", "uv_lock_sha256")}

    report["stage"] = "convert"
    work = out_dir / "asr_work"
    exporter = EXPORTERS[mt]
    export_args = [str(snap), str(work), "--input-sec", str(INPUT_SEC), "--stateful-after", str(STATEFUL_AFTER),
                   "--recipe", "int8"]
    cv.write_report(report)
    t0 = time.time()
    rc = subprocess.call([sys.executable, str(exporter), *export_args])
    exp_path = work / "export.json"
    exp = json.loads(exp_path.read_text()) if exp_path.exists() else {}
    if rc == 2:
        cv.refuse(report, "asr_unsupported", "; ".join(exp.get("unsupported") or ["unsupported checkpoint"]),
                  "docs/asr-route.md lists what the runtime reads")
    if rc != 0:
        cv.fail(f"{exporter.name} exit {rc}" + (": a PyTorch check failed" if rc == 1 else ""), stage="convert",
                context=json.dumps(exp.get("checks", {})))
    report["export"] = {k: exp.get(k) for k in ("shapes", "front_end", "checks", "convert_s", "quantize_s",
                                                "speech2text_metadata", "signatures")}
    if (exp.get("front_end") or {}).get("assumed"):
        print("NOTE: the repo has no processor_config.json; the bundle relies on the runtime's Parakeet front end "
              "(128 mel bins, 16 kHz, 25 ms / 10 ms framing, pre-emphasis 0.97), with the encoder's num_mel_bins "
              "checked")
    report["export"]["seconds"] = round(time.time() - t0, 1)
    report["export_seconds"] = round(time.time() - t0)

    report["stage"] = "bundle"
    bundle = out_dir / f"{name}_int8.litertlm"
    header = {"uuid": str(uuid.uuid5(uuid.NAMESPACE_URL, f"{args.model}@{sha}#asr-int8#{toolchain['id']}")),
              "creation_timestamp": report["decisions"]["commit_date"]}
    pack_cmd = side_env([ASR_ENV / "pack_speech2text.py", work / "asr_model_int8.tflite", snap / "tokenizer.json",
                         work / "speech2text_metadata.json", bundle, "--uuid", header["uuid"],
                         "--timestamp", header["creation_timestamp"]])
    cv.write_report(report)
    p = subprocess.run(pack_cmd, capture_output=True, text=True, env=side_env_vars())
    sys.stderr.write(p.stderr[-4000:])
    if p.returncode != 0 or not bundle.exists():
        cv.fail(f"pack_speech2text.py exit {p.returncode}: {(p.stderr.strip().splitlines() or [''])[-1]}",
                stage="bundle", context=p.stderr[-4000:])
    packed = json.loads(p.stdout.strip().splitlines()[-1])
    (Path(str(bundle) + ".speech2text_metadata.pb")).unlink(missing_ok=True)
    identity = cv.bundle_identity(bundle)
    report["bundle"] = {"path": str(bundle), "sha256": identity["sha256"], "size": identity["size"],
                        "sections": [s["type"] for s in identity["sections"]]}
    print(f"bundle: {bundle} ({identity['size']:,} B, sha256 {identity['sha256']})")

    gate = None
    if not args.no_gate:
        report["stage"] = "gate"
        clips, bar = (SMOKE_CLIPS, SMOKE_BAR) if args.smoke else (GATE_CLIPS, GATE_BAR)
        gate_json = out_dir / "asr_gate.json"
        gate_json.unlink(missing_ok=True)
        cv.write_report(report)
        try:
            parquet = clip_file()
        except Exception as e:  # noqa: BLE001 — one FAIL line
            cv.fail(f"the gate's clips ({CLIPS['repo']}): {type(e).__name__}: {str(e)[:200]}", stage="fetch")
        t0 = time.time()
        rc = subprocess.call(side_env([ASR_ENV / "smoke.py", bundle, "--clips", clips, "--bar", bar,
                                       "--parquet", parquet, "--json", gate_json]),
                             env=side_env_vars())
        g = json.loads(gate_json.read_text()) if gate_json.exists() else {}
        first = (g.get("clips") or [{}])[0]
        gate = {"kind": "smoke" if args.smoke else "gate", "runtime": g.get("runtime"), "clips": clips, "bar": bar,
                "wer": g.get("wer"), "pass": bool(g.get("pass")) and rc == 0, "exit": rc, "first": first}
        report["gate"] = gate
        report["gate_seconds"] = round(time.time() - t0)
        report["check"] = {"engine": g.get("runtime"), "backend": "cpu", "seconds": first.get("transcribe_s"),
                           "answer_head": (first.get("text") or "")[:120] or None, "pass": gate["pass"],
                           "arithmetic": None}

    # the keys scripts/verify_lock.py reads are the LLM lock's: source, recipe.id, toolchain, converter, bundle
    lock = {
        "lock_version": cv.LOCK_VERSION,
        "command": f"{cv.COMMAND} {args.model} --revision {sha} --task asr",
        "task": "asr",
        "source": {"repo": args.model, "revision": sha, "commit_date": report["decisions"]["commit_date"],
                   "files": source_files},
        "recipe": {
            "id": "int8",
            "export": {"script": str(exporter.relative_to(REPO_ROOT)), "script_sha256": cv.sha256_file(exporter),
                       "args": export_args[2:], "quantization_recipe": "dynamic_wi8_afp32",
                       "speech2text_metadata": exp.get("speech2text_metadata")},
            "bundle_step": {"script": "asr/pack_speech2text.py",
                            "script_sha256": cv.sha256_file(ASR_ENV / "pack_speech2text.py"),
                            "system_metadata": header,
                            "asr_uv_lock_sha256": cv.sha256_file(ASR_ENV / "uv.lock"),
                            "toolchain": packed.get("toolchain")},
        },
        "toolchain": toolchain,
        "process": {"PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED")},
        "converter": cv.converter_info(),
        "bundle": identity,
        "gate": gate,
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "host": cv.host_info(),
    }
    cv.write_lock(out_dir / f"{name}.lock.json", lock)
    report["lock"] = str(out_dir / f"{name}.lock.json")
    if gate is not None and not gate["pass"]:
        report["status"] = "smoke_failed" if args.smoke else "converted_gate_failed"
        if gate["wer"] is not None:  # the runtime transcribed, the words are wrong: a verdict, not an op
            report["failure"] = cv.failure("gate", f"WER {gate['wer']} > {gate['bar']} on {gate['clips']} clip(s)",
                                           hint="read asr_gate.json: each clip's text against its transcript",
                                           **cv.failure_fields())
        else:
            report["failure"] = cv.failure("engine-load", f"asr/smoke.py exit {gate['exit']}: no transcript",
                                           hint="the FAIL smoke line above names the error",
                                           **cv.failure_fields())
        cv.write_report(report)
        print(report["failure"]["line"], file=sys.stderr)
        return 1
    if gate is None:
        report["status"] = "built"
    else:
        report["status"] = "smoke_pass" if args.smoke else "converted_pass"
    report["stage"] = None
    cv.write_report(report)
    print(f"DONE: {bundle}" + (f" ({gate['kind']}: WER {gate['wer']} on {gate['clips']} clip(s), bar "
                               f"{gate['bar']})" if gate else " (built, no gate)"))
    return 0
