#!/usr/bin/env python3
"""Thin gated wrapper around the stock litert-torch export.

    uv sync
    uv run python scripts/convert.py <hf_model_id> [--revision SHA] [--recipe int8|int4]
                                     [--out DIR] [--min-correct N] [--smoke] [--check]

-> out/<name>/<name>_<recipe>.litertlm, <name>.lock.json, convert_report.json.

The lock (docs/one-command-lock.md) pins the source (repo, commit, size and sha256 of every
file the export read, each checked against the Hub's tree), the export call, the toolchain
(every installed package, uv.lock's sha256) and the output (file and section sha256). The
builder's random uuid and now() timestamp are replaced in this process by values derived from
the source, so `scripts/verify_lock.py <lock>` can rebuild the same bytes. A failure ends with
one line on stderr, `FAIL <stage>: <op or first error line> -> next: <hint>` (hints:
scripts/failure_hints.json), and the same fields in convert_report.json["failure"].

Measured basis (2026-08-24): stock `export(model, output_dir)` with no other
arguments converts untouched finetunes correctly on both the model_ext path
(qwen3: QVikhr-3-1.7B) and the generic path (smollm3: a SmolLM3-3B finetune;
2026-08-25 adds MiniCPM5-1B derivatives — plain llama, default path, 7/8,
template byte-equal, A≡B) —
the derivative's own chat template is embedded verbatim (including a
`chat_template.jinja` that tokenizer_config.json doesn't reference) and the
runtime applies it (greedy A/B/C proof). So this wrapper adds NO per-model
template or architecture handling (item 6 is one generic, proven content-form
adaptation for the runtime). It adds only:

  1. an entry gate — refuse, with a structured reason, what stock export
     cannot convert honestly (gated repos, remote-code architectures,
     pre-quantized weights); never convert on a guess. "Remote-code" means
     auto_map for a model_type transformers does NOT register: when it does
     register one, the repo's Python is a back-compat shim that never loads
     (measured 2026-08-25 on Nemotron-3-Nano-4B) and the model converts.
     Adapter (LoRA/PEFT)
     repos are no longer refused: the adapter is merged into its base
     (peft merge_and_unload, measured 2026-08-24 bitwise == W + (a/r)*BA)
     into a temporary hub-format dir and stock-exported from there — unless
     the BASE is gated/remote-code/pre-quantized, which still refuses;
  2. two opt-ins — EXTERNALIZE_EMBEDDER when the model is >=3B params (keeps
     the main weights section under the iOS ~2GiB mmap limit), and the proven
     int4 recipe (blockwise-32 OCTAV + int8 embedding) behind --int4;
  3. an exit gate — scripts/verify_quality.py (8 questions, bar 6/8), with a
     3200-token budget for <think> models so a long thought is not scored as
     degeneracy; plus a template lint that WARNS (does not block) when the
     embedded Jinja needs Python-style methods — current runtimes render
     them (verified 2026-08-24 on Mac 0.16.0 and an Android main build),
     pre-2026 runtimes may die on the first message;
  4. hybrid routing (HYBRID_STOCK) — architectures the stock export handles
     only in specific envs (Qwen3.5/GatedDeltaNet: litert-torch main, not any
     released wheel) are refused with the env action when the installed
     litert-torch lacks the model_ext, gate on --backend cpu (their stock
     bundles are not GPU-delegable), and at >=3B export with the shipped 4B's
     reduced 7-signature prefill ladder. Measured 2026-08-24: base 0.8B 6/8;
     numind/NuExtract3 (4.5B finetune) 8/8 with template byte-equal and
     greedy A/B/C proven.

  5. one source normalization — hunyuan_v1_dense's `rope_scaling` (dynamic +
     alpha) resolves to a STATIC rope in transformers, but the leftover
     data-dependent growth branch kills torch.export; the baked-theta
     snapshot it exports instead is bitwise-equal to the HF reference
     (measured 2026-08-27, see normalize_static_alpha_rope).
  6. the template the 0.18 runtime can render — litert-lm 0.18.0 renders on
     native minijinja values: content is a list of text parts and a map has no
     Python methods, so a transformers template drops the message (Qwen3-1.7B:
     the user turn rendered empty), raises (SmolLM2: str + list; gemma-4-E2B-it:
     "map has no method named get") or prints the list. Only when it must, the
     template gets .get('k') -> ['k'] and then the dual form (LiteRT-LM's
     format_content on every content read), and goes into the export only once
     the runtime's render equals transformers' byte for byte on four
     conversations; after the export the runtime's own render must equal it too
     (prepare_template, check_runtime_template, scripts/chat_template_dual.py).
  7. one toolchain-skew shim — litert-torch 0.9.4 writes a Gemma4 metadata
     field litert-lm-builder 0.18.0 dropped; the one assignment is skipped
     when both are installed, as litert-torch main does (builder_compat).

Exit codes: 0 = converted and gate passed, 1 = converted but gate (or check) failed,
2 = stopped at the entry gate (fetch / gate stage; refusals also print their JSON),
3 = failed in the pipeline (load / trace / convert / quantize / bundle, or past --max-minutes).
Every run writes <out>/convert_report.json with the decisions taken.
"""
import argparse
import datetime
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from failure_line import failure, first_line  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
INT4_RECIPE = REPO_ROOT / "recipes" / "qwen3_int4_block32_octav.json"
VERIFY = REPO_ROOT / "scripts" / "verify_quality.py"
CLI_VERIFIER = REPO_ROOT / "scripts" / "litert_lm_cli_verifier.py"
CURATED_INDEX = REPO_ROOT / "recipes" / "curated_index.json"
PINNED_ENV = REPO_ROOT / ".venv"
# The hf-to-litertlm package (pypi/) runs these files from its own unpacked copy and sets
# HF_TO_LITERTLM_PACKAGE to its version: outputs then go under the caller's directory.
PACKAGE = os.environ.get("HF_TO_LITERTLM_PACKAGE")
OUT_ROOT = Path.cwd() / "out" if PACKAGE else REPO_ROOT / "out"
COMMAND = "hf-to-litertlm" if PACKAGE else "uv run python scripts/convert.py"
REPO_URL = "https://github.com/john-rocky/hf-to-litertlm"
CAPPED_CHILD_ENV = "HF_TO_LITERTLM_CAPPED_CHILD"
# PYTHONHASHSEED of every run (see the __main__ block); the lock records it
HASH_SEED = os.environ.get("HF_TO_LITERTLM_HASH_SEED", "0")
LOCK_VERSION = 1

# --recipe -> the export's quantization_recipe. int8 is the export's own default
# (dynamic_wi8_afp32); int4 is the proven blockwise-32 OCTAV recipe file.
RECIPES = {"int8": "dynamic_wi8_afp32", "int4": str(INT4_RECIPE)}
# The export's defaults (litert-torch 0.9.4 ExportableModuleConfig), passed explicitly so the
# lock pins them even if a later release moves a default.
EXPORT_DEFAULTS = {"prefill_lengths": [128], "cache_length": 4096}
# Top-level files the stock export reads besides the weights (configs, tokenizer, template).
SOURCE_SUFFIXES = (".json", ".txt", ".model", ".jinja", ".tiktoken")
# The decisions of the post-export guards; they change bytes, so the lock carries them.
POST_EXPORT_KEYS = ("stop_tokens_added", "start_token", "start_token_bos", "executor_metadata")

_REPORT = None  # the run's report, so fail() can record the failure in it
_T0 = time.time()

# Python-style constructs in vendor Jinja that old minijinja builds could not
# render. Current runtimes render all of these (measured 2026-08-24).
PY_METHOD_RE = re.compile(
    r"\.(get|startswith|endswith|split|rsplit|replace|strip|lstrip|rstrip|items|keys|values|append)\s*\("
)

# Hybrid architectures the stock export converts correctly, but only in specific
# environments — keyed by config.json model_type, value = the model_ext module
# the export needs. Measured 2026-08-24 on Qwen/Qwen3.5-0.8B:
#   - released litert-torch 0.9.3 has no qwen3_5 model_ext; its generic path
#     dies mid-trace (LiteRTLMConvCacheLayer.update_conv_state TypeError).
#     litert-torch main (0.10.0.dev, pip install from git) exports it stock,
#     template embedded verbatim, ExecutorMetadata emitted, CPU gate 6/8 PASS.
#   - released 0.9.4 DOES ship the qwen3_5 exportables (the ext check passes
#     there), but its stock output is degenerate — endless token repetition on
#     a 2B export, cause not isolated (measured 2026-08-25). The exit gate
#     catches it; until isolated, export Qwen3.5 from main.
#   - the stock bundle is CPU-only: the GPU delegate rejects the GatedDeltaNet
#     kernel (invalid TRANSPOSE permutation) and the verifier's default engine
#     path fails outright instead of falling back — the exit gate must run
#     --backend cpu. (The GPU-delegable shipped models come from the per-model
#     recipe in qwen35_work/, not from this stock path.)
HYBRID_STOCK = {
    "qwen3_5": "litert_torch.generative.export_hf.model_ext.qwen3_5",
    "qwen3_5_text": "litert_torch.generative.export_hf.model_ext.qwen3_5",
}

def native_model_type(model_type):
    """True when the installed transformers registers this model_type natively.
    A repo that declares auto_map AND a registered model_type ships the repo
    code as a back-compat shim for older transformers: without
    trust_remote_code the library implementation loads and the repo code is
    never imported (measured 2026-08-25 on nvidia/NVIDIA-Nemotron-3-Nano-4B-
    BF16 — AutoConfig returns transformers' native NemotronHConfig). Only an
    UNREGISTERED model_type actually requires executing the repo's code."""
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES
    return model_type in CONFIG_MAPPING_NAMES


# Hybrid families NO released litert-torch converts at all — the wheels ship
# no Mamba2 cache layer (0.9.4: zero mamba support; measured 2026-08-25 on a
# granite-4.0-h-350m finetune: the generic path maps the mamba layers to the
# qwen3.5-era linear-attention cache and dies before tracing with
# AttributeError: 'GraniteMoeHybridConfig' object has no attribute
# 'linear_key_head_dim'). This repo carries a pinned patched checkout + a
# family recipe script instead; convert.py routes to it when the checkout is
# present and refuses with the exact setup command when it is not.
HYBRID_RECIPE = {
    "granitemoehybrid": {
        "script": "granite_work/convert_granite4h.py",
        "checkout": "granite_work/litert-torch-granite",
        "setup": (
            "cd granite_work && "
            "git clone https://github.com/google-ai-edge/litert-torch litert-torch-granite && "
            "git -C litert-torch-granite fetch origin 115a13607c730c81018bb9789138a3e5e5119e3d && "
            "git -C litert-torch-granite checkout 115a13607c730c81018bb9789138a3e5e5119e3d && "
            "git -C litert-torch-granite apply \"$(pwd)/granite_hybrid_litert_torch.patch\""
        ),
        # granite's template has no leading BOS; the engine-prepended
        # start_token measurably flips 350M-scale greedy decoding
        # (REPRODUCE: the granite-4.0-h-350m start_token lesson)
        "drop_start_token": "granite_work/drop_start_token.py",
    },
    "falcon_h1": {
        "script": "falcon_h1_work/convert_falcon_h1.py",
        "checkout": "falcon_h1_work/litert-torch-falcon",
        "setup": (
            "cd falcon_h1_work && "
            "git clone https://github.com/google-ai-edge/litert-torch litert-torch-falcon && "
            "git -C litert-torch-falcon fetch origin 115a13607c730c81018bb9789138a3e5e5119e3d && "
            "git -C litert-torch-falcon checkout 115a13607c730c81018bb9789138a3e5e5119e3d && "
            "git -C litert-torch-falcon apply \"$(pwd)/falcon_h1_litert_torch.patch\""
        ),
        # no start_token drop: the shipped Falcon-H1 bundles keep it and gate
        # clean (6-8/8 with HF-identical text) at every size
    },
    "zamba2": {
        "script": "zamba2_work/convert_zamba2.py",
        "checkout": "zamba2_work/litert-torch-zamba2",
        "setup": (
            "cd zamba2_work && "
            "git clone https://github.com/google-ai-edge/litert-torch litert-torch-zamba2 && "
            "git -C litert-torch-zamba2 fetch origin 115a13607c730c81018bb9789138a3e5e5119e3d && "
            "git -C litert-torch-zamba2 checkout 115a13607c730c81018bb9789138a3e5e5119e3d && "
            "git -C litert-torch-zamba2 apply \"$(pwd)/zamba2_litert_torch.patch\""
        ),
    },
    "nemotron_h": {
        "script": "nemotron_h_work/convert_nemotron_h.py",
        "checkout": "nemotron_h_work/litert-torch-nemotron",
        "setup": (
            "cd nemotron_h_work && "
            "git clone https://github.com/google-ai-edge/litert-torch litert-torch-nemotron && "
            "git -C litert-torch-nemotron fetch origin 115a13607c730c81018bb9789138a3e5e5119e3d && "
            "git -C litert-torch-nemotron checkout 115a13607c730c81018bb9789138a3e5e5119e3d && "
            "git -C litert-torch-nemotron apply \"$(pwd)/nemotron_h_litert_torch.patch\""
        ),
    },
}


def zamba2_preport_guard(report, cfg):
    """Zamba2 derivatives finetuned from Zyphra's ORIGINAL release are
    serialized in the pre-transformers-port format: config vocabulary from the
    4.43 era (layers_block_type ['m','g'], state_size, use_mamba_kernels...)
    AND the old weight layout (model.mamba_layers.*). Current transformers
    rejects the config outright (strict validate_layer_type), and under a
    translated config the state_dict is still a different generation
    (measured 2026-08-25 on ssmits/Zamba2-1.2B-instruct-Dutch: 684 missing /
    404 unexpected keys). No thin bridge is honest here — refuse before
    downloading a checkpoint no modern stack can construct."""
    lbt = (cfg or {}).get("layers_block_type") or []
    if set(lbt) & {"m", "g"}:
        refuse(report, "preport_serialization",
               "the checkpoint is serialized in the pre-port Zamba2 format "
               "(transformers 4.4x-era config vocabulary and model.mamba_layers.* "
               "weight layout) — current transformers cannot construct it, so no "
               "converter downstream of transformers can either",
               "ask the model author to re-serialize: load with transformers <=4.48 "
               "and save_pretrained with >=4.49, then convert the re-serialized repo")

def normalize_static_alpha_rope(report, model_id, cfg, out_dir, revision=None):
    """hunyuan_v1_dense ships `rope_scaling: {type: dynamic, alpha: ...}`, but
    transformers 5.14.1 resolves it STATICALLY at init —
    modeling_hunyuan_v1_dense ("DynamicNTKAlphaRotary - unique to this model")
    computes base = rope_theta * alpha**(head_dim/(head_dim-2)) once. The
    forward still carries the generic @dynamic_rope_update growth branch, and
    its data-dependent `seq_len > max_seq_len_cached` guard is what kills
    torch.export (GuardOnDataDependentSymNode on Eq(u0, 1), measured
    2026-08-27 on tencent/Hy-MT2-1.8B). Baking the resolved base into
    rope_theta and dropping rope_scaling is bitwise-identical to the HF
    reference (inv_freq and teacher-forced logits, measured) for all
    seq_len <= max_position_embeddings — the dropped branch only fires past
    that, far beyond any bundle context. Returns a local hub-format dir whose
    config carries the baked value; everything else symlinks into the HF
    cache."""
    from huggingface_hub import snapshot_download
    snap = Path(snapshot_download(model_id, revision=revision,
                                  ignore_patterns=["*.png", "*.jpg", "train/*"]))
    norm = out_dir / "rope_normalized_src"
    norm.mkdir(parents=True, exist_ok=True)
    for f in snap.iterdir():
        if f.name == "config.json" or f.is_dir():
            continue
        dst = norm / f.name
        if not dst.exists():
            dst.symlink_to(f.resolve())
    dim = cfg["head_dim"]
    baked = cfg["rope_theta"] * cfg["rope_scaling"]["alpha"] ** (dim / (dim - 2))
    norm_cfg = {k: v for k, v in cfg.items() if k != "rope_scaling"}
    norm_cfg["rope_theta"] = baked
    (norm / "config.json").write_text(json.dumps(norm_cfg, indent=1))
    report["decisions"]["rope_normalization"] = {
        "from": dict(cfg["rope_scaling"]),
        "baked_rope_theta": baked,
        "faithful_up_to": cfg.get("max_position_embeddings"),
    }
    print(f"NOTE: rope_scaling type=dynamic alpha={cfg['rope_scaling']['alpha']} "
          f"resolves statically in transformers — baked rope_theta={baked:.6g} and "
          "dropped the untraceable growth branch (bitwise-equal below "
          f"{cfg.get('max_position_embeddings')} positions)")
    return norm


# State-carrying hybrids whose stock export succeeds on the RELEASED litert-torch
# (the lfm2 model_ext ships since 0.9.1) but whose 0.9.3 bundling emits no
# ExecutorMetadata section — litert-lm >= 0.15 then fails at inference with
# "missing some output TensorBuffers". Measured 2026-08-24 on an LFM2.5-1.2B
# finetune: export 64 s, 3 sections; after the retrofit the SAME bundle gates
# 8/8 (CPU ~75 tok/s, GPU probe fine — no gate backend override needed).
# NOTE: transformers 5.15.x breaks the lfm2 export on 0.9.3 AND 0.9.4
# (Lfm2ShortConv.L_cache AttributeError) — pin transformers 5.14.x. Released
# 0.9.4 emits ExecutorMetadata natively (measured 2026-08-25); the retrofit
# then detects the existing section and no-ops.
HYBRID_EXEC_RETROFIT = {"lfm2"}


def hybrid_gate_backend(report, cfg):
    """Entry check for HYBRID_STOCK architectures. Returns the exit-gate
    backend override (None = verifier default), refusing when the installed
    litert-torch lacks the model_ext the architecture needs — before any
    download or merge work."""
    model_type = (cfg or {}).get("model_type")
    report["decisions"]["model_type"] = model_type
    ext = HYBRID_STOCK.get(model_type)
    if not ext:
        return None
    import importlib.util
    if importlib.util.find_spec(ext) is None:
        refuse(report, "model_ext_missing",
               f"model_type {model_type} needs the {ext.rsplit('.', 1)[-1]} model_ext, "
               "which this litert-torch does not ship (the released 0.9.3 generic path "
               "dies mid-trace: LiteRTLMConvCacheLayer.update_conv_state TypeError)",
               "install litert-torch from main into a fresh venv and rerun there: "
               "pip install 'litert-torch @ git+https://github.com/google-ai-edge/"
               "litert-torch.git' (measured 2026-08-24 @ 8379afb)")
    report["decisions"]["gate_backend"] = "cpu"
    return "cpu"


def ensure_turn_end_stop(bundle, tokenizer_src, report):
    """Post-export guard: the tokenizer's eos_token_id must be in the bundle's
    stop_tokens. Qwen3.5 checkpoints declare only <|endoftext|> (measured
    2026-08-24: text_config.eos_token_id=248044, no generation_config.json,
    and the stock template's probe render raise_exceptions so the exporter's
    template-derived stop never fires) while the tokenizer's eos is
    <|im_end|>=248046 — the bundle then never stops at the turn end, and a
    thinking derivative burns the whole budget (measured gate 0/8, fixed
    bundle answers and stops). No-op when the stop list is already right."""
    from transformers import AutoTokenizer

    eos_id = AutoTokenizer.from_pretrained(tokenizer_src).eos_token_id
    if eos_id is None:
        return
    import re as _re
    import tempfile
    from litert_lm_builder import pack_litertlm_file, unpack_litertlm_file

    with tempfile.TemporaryDirectory(dir=bundle.parent) as td:
        unpack_litertlm_file(str(bundle), td)
        pb = Path(td) / "LlmMetadataProto.pbtext"
        text = pb.read_text()
        declared = {int(m) for m in _re.findall(r"ids:\s*(\d+)", text)}
        if eos_id in declared:
            return
        m = _re.search(r"stop_tokens \{.*?\n\}", text, flags=_re.DOTALL)
        if not m:
            print(f"WARNING: no stop_tokens block found; not adding eos {eos_id}")
            return
        block = m.group(0)
        text = text.replace(
            block, block + "\nstop_tokens {\n  token_ids {\n    ids: %d\n  }\n}" % eos_id, 1)
        pb.write_text(text)
        fixed = bundle.parent / (bundle.name + ".stopfix")
        pack_litertlm_file(str(Path(td) / "model.toml"), str(fixed))
        fixed.replace(bundle)
    report["decisions"]["stop_tokens_added"] = [eos_id]
    print(f"stop-token guard: added tokenizer eos {eos_id} to the bundle's stop_tokens")


def in_pinned_env():
    """True when this interpreter is the uv.lock environment (`uv run` from the repo)."""
    return PINNED_ENV.exists() and Path(sys.prefix).resolve() == PINNED_ENV.resolve()


def resolve_litert_lm_cli():
    """A litert-lm >= 0.15 CLI: $LITERT_LM_CLI, else the pinned environment's
    own CLI when convert.py runs in it (litert-lm 0.18.0 from uv.lock), else
    the ~/venvs/lt0160run default, else `litert-lm` on PATH (last — which() can
    surface a pyenv shim that exits 127, so the known venv wins)."""
    pinned_cli = Path(sys.executable).parent / "litert-lm"
    default_cli = Path.home() / "venvs/lt0160run/bin/litert-lm"
    return (os.environ.get("LITERT_LM_CLI")
            or (str(pinned_cli) if in_pinned_env() and pinned_cli.exists() else None)
            or (str(default_cli) if default_cli.exists() else None)
            or shutil.which("litert-lm"))


def ensure_executor_metadata(bundle, report):
    """Post-export guard for HYBRID_EXEC_RETROFIT: retrofit the ExecutorMetadata
    section the released exporter omits (see the dict's comment), reusing the
    shipped lfm_work/add_executor_metadata.py. No-op when the section exists."""
    cli = resolve_litert_lm_cli()
    tool = REPO_ROOT / "lfm_work" / "add_executor_metadata.py"
    fixed = bundle.parent / (bundle.name + ".execfix")
    proc = subprocess.run(
        [sys.executable, str(tool), str(bundle), str(fixed),
         "--litert-lm", cli, "--python", sys.executable],
        capture_output=True, text=True)
    if proc.returncode == 0:
        fixed.replace(bundle)
        report["decisions"]["executor_metadata"] = "retrofitted"
        print("executor-metadata guard: section retrofitted "
              "(released stock bundling omits it for this architecture)")
    elif "already has" in (proc.stdout + proc.stderr):
        report["decisions"]["executor_metadata"] = "present"
    else:
        fail("add_executor_metadata failed: " + (proc.stdout + proc.stderr)[-500:], stage="bundle")

def convert_via_recipe(report, recipe, export_src, out_dir, args):
    """HYBRID_RECIPE path: run the family recipe converter as a subprocess with
    the pinned patched litert-torch checkout on PYTHONPATH, then the family's
    post-steps (start_token drop). Returns the bundle path; these families gate
    on CPU (the Mamba2 5-D SSM ops exceed the GPU delegate's rank limit)."""
    import os
    checkout = REPO_ROOT / recipe["checkout"]
    script = REPO_ROOT / recipe["script"]
    if args.int4:
        refuse(report, "recipe_no_int4",
               "this family's recipe ships post-hoc int8 over linears+embedding only "
               "(export-time conv-int8 measurably costs quality on it)",
               "rerun without --int4")
    if not (checkout / "litert_torch").exists():
        refuse(report, "recipe_checkout_missing",
               f"model_type {report['decisions']['model_type']} converts only via the "
               f"pinned patched litert-torch checkout in {recipe['checkout']} — no "
               "released litert-torch ships a Mamba2 cache layer (measured 2026-08-25 "
               "on 0.9.4: dies at cache construction, GraniteMoeHybridConfig has no "
               "attribute 'linear_key_head_dim')",
               "one-time setup: " + recipe["setup"] + " ; then rerun")
    print(f"recipe export ({recipe['script']}, pinned checkout): {export_src} -> {out_dir}")
    env = dict(os.environ, PYTHONPATH=str(checkout))
    # the recipe's sub-steps shell out to litert-lm-builder / litert-lm by bare
    # name — put the interpreter's venv bin and the resolved CLI's dir on PATH
    # so the routed path works from a clean shell (measured: exit 127 otherwise)
    bindirs = [str(Path(sys.executable).parent)]
    cli = resolve_litert_lm_cli()
    if cli:
        bindirs.append(str(Path(cli).parent))
    env["PATH"] = os.pathsep.join(bindirs + [env.get("PATH", "")])
    # ≥3B recipe models get the same reduced 7-signature prefill ladder as the
    # stock path (convert.py's stock-path comment has the three measured
    # reasons; the 12 GB-iPhone jetsam datapoint there IS a hybrid 4B of this
    # family). The drivers read PREFILL_LENGTHS and default to the full ladder.
    if (report["decisions"].get("params") or 0) >= 3e9:
        env["PREFILL_LENGTHS"] = "1024,256,64,16,4,1"
        report["decisions"]["prefill_ladder"] = "reduced_7sig_3b"
    # the recipe drivers keep the embedding in the main section — record that
    # honestly instead of the stock-path opt-in set earlier
    if report["decisions"].get("externalize_embedder"):
        report["decisions"]["externalize_embedder"] = "not_on_recipe_path"
    t0 = time.time()
    proc = subprocess.run([sys.executable, str(script), export_src, str(out_dir)],
                          env=env, text=True)
    if proc.returncode:
        report["status"] = "export_failed"
        write_report(report)
        fail(f"recipe converter failed (exit {proc.returncode})", stage="convert")
    report["export_seconds"] = round(time.time() - t0)
    bundle = out_dir / (Path(export_src).name + "_int8.litertlm")
    if not bundle.exists():
        fail(f"recipe converter finished without producing {bundle.name}", stage="bundle")
    if recipe.get("drop_start_token"):
        dropped = bundle.parent / (bundle.name + ".nobos")
        proc = subprocess.run(
            [sys.executable, str(REPO_ROOT / recipe["drop_start_token"]),
             str(bundle), str(dropped)], env=env, text=True)
        if proc.returncode:
            fail("drop_start_token failed", stage="bundle")
        dropped.replace(bundle)
        report["decisions"]["start_token"] = "dropped"
        print("start-token guard: dropped (family template has no leading BOS)")
    report["decisions"]["recipe"] = recipe["script"]
    return bundle


def _render_template(source, bos, msgs):
    """Render a chat template the way a given consumer binds `bos_token`.

    `{% generation %}` is a transformers-only tag (assistant masking) that
    emits nothing, so dropping it leaves the rendered text unchanged.
    """
    import jinja2

    env = jinja2.Environment(trim_blocks=True, lstrip_blocks=True,
                             extensions=["jinja2.ext.loopcontrols"])
    env.globals["raise_exception"] = lambda m: (_ for _ in ()).throw(RuntimeError(m))
    env.globals["strftime_now"] = lambda f: "01 Jan 2026"
    source = re.sub(r"\{%-?\s*(end)?generation\s*-?%\}", "", source)
    return env.from_string(source).render(messages=msgs, add_generation_prompt=True,
                                          bos_token=bos, eos_token="", tools=None)


def _leading(text, token):
    """How many times `token` repeats at the very start of `text`."""
    n = 0
    while token and text.startswith(token):
        text = text[len(token):]
        n += 1
    return n


def drop_start_token_in_process(bundle):
    """granite_work/drop_start_token.py's edit (remove the leading start_token
    block of LlmMetadata) through litert_lm_builder's unpack/pack in this
    process — the functions the `litert-lm unpack` / `pack` CLI wraps — so the
    pinned system metadata (pin_system_metadata) applies to the repack too.
    False when the bundle has no leading start_token."""
    import tempfile
    from litert_lm_builder import litertlm_builder

    with tempfile.TemporaryDirectory(dir=bundle.parent) as td:
        toml = litertlm_builder.unpack(str(bundle), td)
        pb = Path(td) / "LlmMetadataProto.pbtext"
        text = pb.read_text()
        if not text.startswith("start_token {"):
            return False
        depth, quoted, prev = 0, False, ""
        for i, ch in enumerate(text):  # the block's own closing brace, quotes skipped
            if ch == '"' and prev != "\\":
                quoted = not quoted
            elif not quoted:
                depth += (ch == "{") - (ch == "}")
                if ch == "}" and depth == 0:
                    break
            prev = "" if prev == "\\" else ch
        pb.write_text(text[i + 1:].lstrip("\n"))
        dropped = bundle.parent / (bundle.name + ".nobos")
        dropped.unlink(missing_ok=True)
        litertlm_builder.pack(toml, str(dropped))
        dropped.replace(bundle)
    return True


def ensure_no_spurious_start_token(bundle, tokenizer_src, report, in_process=False):
    """Post-export guard: the bundling writes LlmMetadata start_token from
    tokenizer.bos_token without consulting add_bos_token or the chat template,
    and the engine prepends that start_token to every rendered prompt.

    Everything below is measured on the runtime itself (2026-08-27 audit of all
    88 published bundles, reports/bos_audit_20260827.md in the source repo):

      R1  the assembled prompt is [start_token] + tokenize(render), the start
          token contributing exactly ONE token (prefill 16 vs 15 with/without).
      R2  the engine renders the template with minijinja, which leaves
          `bos_token` UNBOUND — `{{ bos_token }}` renders EMPTY. Only a BOS
          written as a LITERAL in the template survives into the prompt.
      R3  the engine's own encode adds no special tokens, so the tokenizer's
          add_bos_token / post-processor never applies on device.

    So the count the model actually receives is

        (1 if start_token is the bos) + literal BOS at the head of the render

    and the reference it should equal is what transformers serves, which is the
    same template with `bos_token` BOUND (the chat path encodes with
    add_special_tokens=False, so add_bos_token does not enter there either).

    Rendering the template twice — bos bound and bos empty — separates the two
    shapes that used to be conflated:
      - `{{ bos_token }}` (Falcon-H1, LFM2.5, gemma): the renders differ, the
        engine contributes nothing, the start_token supplies the single BOS.
        KEEP — dropping it would ship a bundle with no BOS at all.
      - a literal BOS (Hy-MT2, Zamba2, SmolVLM2): the renders are identical,
        the template supplies one and the start_token a second. DROP.
      - no BOS anywhere upstream (granite, Nemotron-H, Phi-4): reference 0,
        bundle 1. DROP.
    """
    import os
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tokenizer_src)
    if tok.bos_token is None:
        return
    bos = tok.bos_token
    msgs = [{"role": "user", "content": "x"}]
    source = getattr(tok, "chat_template", None)
    try:
        reference = _render_template(source, bos, msgs) if source else None
        runtime = _render_template(source, "", msgs) if source else None
    except Exception as e:
        reference = runtime = None
        report["decisions"]["start_token_guard"] = f"unverified ({type(e).__name__})"
    if reference is None or runtime is None:
        print("start-token guard: template did not render — left alone")
        return

    want = _leading(reference, bos)          # what the model was served
    from_template = _leading(runtime, bos)   # what the engine's render supplies
    have = from_template + 1                 # + the metadata start_token (R1)
    report["decisions"]["start_token_bos"] = {"reference": want, "bundle": have}
    if have == want:
        report["decisions"].setdefault("start_token", "kept_matches_reference")
        print(f"start-token guard: kept (BOS count {have} matches the reference)")
        return
    if have < want:
        # Cannot happen while a start_token exists and the template renders the
        # BOS as a literal; surfaced rather than silently shipped.
        fail(f"start-token guard: bundle would feed {have} BOS but the reference "
             f"has {want} — the template renders BOS through a variable and no "
             f"start_token supplies it", stage="bundle")
    decision = ("dropped_duplicate_bos" if want >= 1 else "dropped_spurious_bos")
    note = (f"start-token guard: dropped (bundle would feed {have} leading BOS, "
            f"the reference has {want})")
    if in_process:
        if drop_start_token_in_process(bundle):
            report["decisions"]["start_token"] = decision
            print(note)
        else:
            report["decisions"].setdefault("start_token", "absent")
        return
    cli = resolve_litert_lm_cli()
    env = dict(os.environ)
    if cli:
        env["PATH"] = str(Path(cli).parent) + os.pathsep + env.get("PATH", "")
    dropped = bundle.parent / (bundle.name + ".nobos")
    proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / "granite_work" / "drop_start_token.py"),
         str(bundle), str(dropped)], env=env, text=True, capture_output=True)
    if proc.returncode == 0:
        dropped.replace(bundle)
        report["decisions"]["start_token"] = decision
        print(note)
    elif "no leading start_token" in (proc.stdout + proc.stderr):
        # setdefault, not assignment: on the recipe path a family that already
        # dropped its start token must keep saying "dropped", not "absent"
        report["decisions"].setdefault("start_token", "absent")
    else:
        fail("drop_start_token failed: " + (proc.stdout + proc.stderr)[-300:], stage="bundle")


# The adapter merge MUST run in its own process: loading/merging the model with
# peft+transformers and then calling the stock export in the SAME process
# produces a corrupted bundle (measured 2026-08-24 — identical merged dir,
# deterministic garbage output; the same dir exported by a fresh process is
# correct). argv: base_dir adapter_dir merged_dir info_json_path
MERGE_CHILD = """
import json, shutil, sys
from pathlib import Path
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base_dir, adapter_dir, merged_dir, info_path = map(Path, sys.argv[1:5])
model = AutoModelForCausalLM.from_pretrained(base_dir, dtype="auto")
model = PeftModel.from_pretrained(model, adapter_dir)
model = model.merge_and_unload()
model.save_pretrained(merged_dir)
tok_src = adapter_dir if (adapter_dir / "tokenizer_config.json").exists() else base_dir
AutoTokenizer.from_pretrained(tok_src).save_pretrained(merged_dir)
for extra in ("chat_template.jinja", "generation_config.json"):
    if (adapter_dir / extra).exists():
        shutil.copy(adapter_dir / extra, merged_dir / extra)

# template inheritance: if neither the adapter tokenizer nor its repo carried
# a chat template, the derivative renders with the base's
template_source = "adapter"
tc = json.loads((merged_dir / "tokenizer_config.json").read_text())
if not tc.get("chat_template") and not (merged_dir / "chat_template.jinja").exists():
    template_source = "base"
    if (base_dir / "chat_template.jinja").exists():
        shutil.copy(base_dir / "chat_template.jinja", merged_dir / "chat_template.jinja")
    else:
        bt = json.loads((base_dir / "tokenizer_config.json").read_text()).get("chat_template")
        if isinstance(bt, str) and bt.strip():
            (merged_dir / "chat_template.jinja").write_text(bt)

info_path.write_text(json.dumps({
    "tokenizer_source": "adapter" if tok_src == adapter_dir else "base",
    "template_source": template_source,
}))
"""


def fail(msg, *, stage, context="", code=3):
    """Stop with the FAIL line of docs/one-command-lock.md (stage + op or first
    error line + hint), recorded in the open report as `failure`."""
    f = failure(stage, msg, context=context, **failure_fields())
    if _REPORT is not None:
        _REPORT["status"] = "failed"
        _REPORT["failure"] = f
        write_report(_REPORT)
    print(f"ERROR: {msg}", file=sys.stderr)
    print(f["line"], file=sys.stderr)
    sys.exit(code)


def failure_fields():
    """seconds since the start and the toolchain id, for a table of failures."""
    tid = ((_REPORT or {}).get("toolchain") or {}).get("id")
    if tid is None:
        try:
            tid = toolchain_info()["id"]
        except Exception:  # noqa: BLE001 — the failure line must still print
            tid = None
    return {"seconds": round(time.time() - _T0), "toolchain": tid}


# Refusals that happen before any file is read are the fetch stage; the rest are the gate.
REFUSAL_STAGE = {"gated": "fetch", "adapter_base_gated": "fetch", "adapter_base_missing": "fetch"}


# The entry gate reads the repo's own gated flag, so a token does not change a gated refusal.
GATED_ACTION = ("gated repos are not converted, with or without a token; for a model that "
                "needs no sign-in, start from a converted one at "
                "https://huggingface.co/litert-community, such as "
                "litert-community/gemma-4-E2B-it-litert-lm")


def refuse(report, reason, detail, action):
    report["status"] = "refused"
    report["refusal"] = {"reason": reason, "detail": detail, "action": action}
    report["failure"] = failure(
        REFUSAL_STAGE.get(reason, "gate"), f"{reason}: {detail}",
        context=f"model_type={report['decisions'].get('model_type')}", hint=action,
        **failure_fields())
    print(json.dumps(report["refusal"], indent=2))
    write_report(report)
    print(report["failure"]["line"], file=sys.stderr)
    sys.exit(2)


def write_report(report):
    out = Path(report["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    (out / "convert_report.json").write_text(json.dumps(report, indent=2))


def hub_text(model_id, filename, revision):
    """A small file of the repo at `revision`, through hf_hub_download (it
    sends the stored token and caches; anonymous raw/ requests get
    rate-limited, HTTP 429 measured 2026-10-19). None when it is not there."""
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import EntryNotFoundError, HfHubHTTPError
    try:
        return Path(hf_hub_download(model_id, filename, revision=revision)).read_text()
    except (EntryNotFoundError, HfHubHTTPError, OSError):
        return None


def get_chat_template(src, revision=None):
    """The template stock export will embed: tokenizer_config's chat_template,
    else chat_template.jinja (both observed in the wild). `src` is the local
    dir the export reads, or a Hub id read at `revision`."""
    local = Path(src).is_dir()
    read = ((lambda name: (Path(src) / name).read_text() if (Path(src) / name).exists() else None)
            if local else (lambda name: hub_text(src, name, revision)))
    tc = read("tokenizer_config.json")
    if tc:
        try:
            t = json.loads(tc).get("chat_template")
            if isinstance(t, str) and t.strip():
                return t
        except json.JSONDecodeError:
            pass
    return read("chat_template.jinja")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def pick_source_files(files):
    """The top-level files the stock export reads: configs, tokenizer files and
    the chat template (SOURCE_SUFFIXES), and the weights — model*.safetensors,
    else any top-level *.safetensors, else pytorch_model*.bin. Subfolders
    (onnx/, original/, runs/) are never fetched (SmolLM2-1.7B-Instruct carries
    15 GB of ONNX files next to its 3.4 GB checkpoint)."""
    top = sorted(f for f in files if "/" not in f)
    st = [f for f in top if f.endswith(".safetensors")]
    weights = ([f for f in st if f.startswith("model")] or st
               or [f for f in top if f.startswith("pytorch_model") and f.endswith(".bin")])
    return sorted(set(weights) | {f for f in top if f.endswith(SOURCE_SUFFIXES)})


def download_source(report, model_id, revision, names):
    """snapshot_download of exactly `names` at the pinned commit (xet off: it
    stalls on multi-GB pulls on this host). Returns the snapshot dir."""
    from huggingface_hub import snapshot_download, try_to_load_from_cache
    cached = {n for n in names
              if isinstance(try_to_load_from_cache(model_id, n, revision=revision), str)}
    t0 = time.time()
    snap = Path(snapshot_download(model_id, revision=revision,
                                  allow_patterns=[re.sub(r"([*?\[])", r"[\1]", n) for n in names]))
    report["download"] = {"seconds": round(time.time() - t0, 1), "files": len(names),
                          "files_already_cached": len(cached)}
    return snap, cached


def verify_source(report, api, model_id, revision, snap, names, cached):
    """Size and hash of every fetched file against the Hub's tree at the
    commit: an LFS file by its sha256 (the LFS oid), any other by its git blob
    sha1. Returns the lock's source.files (path, size, sha256, lfs)."""
    tree = {f.path: f for f in api.list_repo_tree(model_id, revision=revision)
            if hasattr(f, "size")}
    rows, fetched = [], 0
    for name in names:
        path, entry = snap / name, tree.get(name)
        if entry is None or not path.is_file():
            fail(f"{name}: not in the snapshot or not in the Hub tree at {revision}", stage="fetch")
        size = path.stat().st_size
        h256 = hashlib.sha256()
        h1 = None if entry.lfs else hashlib.sha1(f"blob {size}\0".encode())
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 22), b""):
                h256.update(chunk)
                if h1:
                    h1.update(chunk)
        if size != entry.size:
            fail(f"{name}: size mismatch (local {size:,} B, the Hub {entry.size:,} B)", stage="fetch")
        if entry.lfs and h256.hexdigest() != entry.lfs.sha256:
            fail(f"{name}: sha256 mismatch (local {h256.hexdigest()}, the Hub {entry.lfs.sha256})",
                 stage="fetch")
        if h1 and h1.hexdigest() != entry.blob_id:
            fail(f"{name}: blob sha1 mismatch (local {h1.hexdigest()}, the Hub {entry.blob_id})",
                 stage="fetch")
        if name not in cached:
            fetched += size
        rows.append({"path": name, "size": size, "sha256": h256.hexdigest(), "lfs": bool(entry.lfs)})
    report["download"]["bytes_fetched"] = fetched
    report["download"]["verified"] = len(rows)
    return rows


def pin_system_metadata(uuid_value, timestamp):
    """litert_lm_builder 0.18.0 writes a fresh uuid4() and now() into every
    bundle header (populate_system_metadata, which LitertLmFileBuilder's
    _build_sections looks up as a module global), so two honest runs never
    match byte for byte. Replace it in this process with fixed values; every
    pack in this process goes through it (the export's own build and the
    guards' unpack/pack). The caller reads the header back to confirm."""
    from litert_lm_builder import litertlm_builder as lb

    def populate_system_metadata(system_metadata):
        kept = [m for m in system_metadata if m.key not in ("uuid", "creation_timestamp")]
        return kept + [lb.Metadata(key="uuid", value=uuid_value, dtype=lb.DType.STRING),
                       lb.Metadata(key="creation_timestamp", value=timestamp,
                                   dtype=lb.DType.STRING)]

    lb.populate_system_metadata = populate_system_metadata


def read_system_metadata(path):
    """{key: value} of a bundle header's system metadata (string values)."""
    from litert_lm_builder import litertlm_core
    from litert_lm_builder import litertlm_header_schema_py_generated as schema
    end_at = litertlm_core.HEADER_END_LOCATION_BYTE_OFFSET
    with open(path, "rb") as f:
        head = f.read(1 << 16)
        hdr_end = int.from_bytes(head[end_at:end_at + 8], "little")
        if hdr_end > len(head):
            f.seek(0)
            head = f.read(hdr_end + 64)
    meta = schema.LiteRTLMMetaData.GetRootAs(
        bytearray(head[litertlm_core.HEADER_BEGIN_BYTE_OFFSET:hdr_end]), 0)
    out, sm = {}, meta.SystemMetadata()
    for i in range(sm.EntriesLength()):
        kv = sm.Entries(i)
        value = None
        if kv.ValueType() == schema.VData.StringValue:
            sv = schema.StringValue()
            sv.Init(kv.Value().Bytes, kv.Value().Pos)
            value = sv.Value().decode()
        out[kv.Key().decode()] = value
    return out


def bundle_identity(path):
    """sha256, size, per-section sha256 (scripts/section_identity.py) and the
    header's system metadata of a bundle."""
    from section_identity import sections
    return {
        "file": Path(path).name,
        "size": Path(path).stat().st_size,
        "sha256": sha256_file(path),
        "sections": [{"index": i, "type": t, "size": s, "sha256": h}
                     for i, (t, s, h) in enumerate(sections(str(path)))],
        "system_metadata": read_system_metadata(path),
    }


def toolchain_info():
    """The exporter stack of this interpreter: id lt<litert-torch>-b<builder>-t<torch>,
    every installed distribution, python, and uv.lock's sha256 when this is the
    pinned environment."""
    import importlib.metadata as md
    import platform
    pkgs = {}
    for d in md.distributions():
        name = re.sub(r"[-_.]+", "-", d.metadata["Name"] or "").lower()
        if name:
            pkgs[name] = d.version
    v = lambda n: pkgs.get(n, "none")  # noqa: E731
    pinned = in_pinned_env()
    lock = REPO_ROOT / "uv.lock"
    return {
        "id": f"lt{v('litert-torch')}-b{v('litert-lm-builder')}-t{v('torch')}",
        "python": platform.python_version(),
        "pinned_env": pinned,
        "uv_lock_sha256": sha256_file(lock) if pinned and lock.exists() else None,
        "packages": dict(sorted(pkgs.items())),
    }


def converter_info():
    """convert.py's own sha256 (its guards change bytes) and the repo commit; from the
    package, its version instead (the unpacked copy is no checkout)."""
    if PACKAGE:
        return {"path": "scripts/convert.py", "sha256": sha256_file(Path(__file__)),
                "repo_commit": None, "package": f"hf-to-litertlm {PACKAGE}"}
    head = subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
                          capture_output=True, text=True)
    return {"path": "scripts/convert.py", "sha256": sha256_file(Path(__file__)),
            "repo_commit": head.stdout.strip() if head.returncode == 0 else None}


# One field litert-torch 0.9.4 writes that litert-lm-builder 0.18.0 no longer has: the gemma4
# metadata builder sets llm_model_type.gemma4.use_template_for_fc_format (0.9.4
# export_hf/model_ext/gemma4/metadata_builder.py l.54), 0.18.0's Gemma4 message dropped the field
# (0.16.1 and 0.17.1 carry it), and every gemma4 export of the pair dies at the packaging step
# (AttributeError, measured 2026-10-19; google-ai-edge/litert-torch#1329). litert-torch main deleted
# the line (#1241), so the shim goes away with the next release: after raising the toolchain,
# scripts/verify_refs.sh shows it doing nothing (recipe.compat empty in the gemma-4 lock).
GEMMA4_FC_LINE = "  llm_metadata.llm_model_type.gemma4.use_template_for_fc_format = True\n"


def builder_compat(report):
    """When the installed builder's Gemma4 message lacks use_template_for_fc_format and the
    installed litert-torch still writes it, run the gemma4 metadata builder without that one
    line (the function recompiled from its own source minus the line, as litert-torch main has
    it). Nothing changes for any other pair; the report and the lock record the skip when the
    gemma4 metadata builder actually runs, so only a gemma4 lock carries it."""
    import importlib
    import inspect
    from litert_lm_builder.runtime.proto import llm_model_type_pb2
    if "use_template_for_fc_format" in llm_model_type_pb2.Gemma4.DESCRIPTOR.fields_by_name:
        return
    try:
        mod = importlib.import_module(
            "litert_torch.generative.export_hf.model_ext.gemma4.metadata_builder")
    except ImportError:
        return
    src = inspect.getsource(mod.build_llm_metadata)
    if src.count(GEMMA4_FC_LINE) != 1:
        return
    scope = {}
    exec(compile(src.replace(GEMMA4_FC_LINE, ""), mod.__file__, "exec"), mod.__dict__, scope)
    without_line = scope["build_llm_metadata"]
    label = "gemma4-use_template_for_fc_format (litert-torch#1329, fixed on main by #1241)"

    def build_llm_metadata(*args, **kwargs):
        compat = report["decisions"].setdefault("compat", [])
        if label not in compat:
            compat.append(label)
        return without_line(*args, **kwargs)

    mod.build_llm_metadata = build_llm_metadata


# Gemma 4 configs written by newer transformers carry the full-attention layers' head size per
# layer: text_config.per_layer_config {"04": {"head_dim": 512}, ...} and no global_head_dim
# (rizhsh/gemma-4-E2B-patent-claim-revision, saved by 5.17.0). That is what 5.19.0's
# Gemma4TextConfig.__post_init__ makes of global_head_dim (and of num_global_key_value_heads when
# attention_k_eq_v is set): one override per full_attention layer. The pinned 5.14.1 reads
# global_head_dim (modeling_gemma4.py l.1194) and stops while it validates such a config:
# validate_architecture's hasattr(self, "head_dim") meets the heterogeneity mixin, which raises
# AmbiguousGlobalPerLayerAttributeError (the nightly probe's FAIL fetch on one E2B and two E4B
# finetunes, 2026-10-23 to 10-26).
# The same configs name the vision tower's rotary type "axial": 5.19.0's Gemma4VisionConfig has
# default_rope_type "axial" and renames a saved "default" to it; 5.14.1 has no "axial" (KeyError when
# the vision tower is built, 2026-10-27) and computes the same inverse frequencies under "default"
# (Gemma4VisionRotaryEmbedding.compute_default_rope_parameters = 5.19.0's
# compute_axial_rope_parameters).
GEMMA4_PER_LAYER_FIELDS = {"head_dim": "global_head_dim",
                           "num_key_value_heads": "num_global_key_value_heads"}


def gemma4_per_layer_fields(text_config):
    """The global fields of the pinned transformers that a Gemma 4 text config's per_layer_config
    stands for ({"global_head_dim": 512}), or None when it is not exactly the newer form: one
    identical override on every full_attention layer and on no other, of head_dim (and of
    num_key_value_heads only when attention_k_eq_v is set), with no global field already set."""
    plc, types = text_config.get("per_layer_config"), text_config.get("layer_types")
    if not isinstance(plc, dict) or not plc or not isinstance(types, list):
        return None
    try:
        layers = {int(k): v for k, v in plc.items()}
    except (TypeError, ValueError):
        return None
    full = {i for i, t in enumerate(types) if t == "full_attention"}
    overrides = list(layers.values())
    if set(layers) != full or not all(isinstance(o, dict) and o == overrides[0] for o in overrides):
        return None
    keys = set(overrides[0])
    if "head_dim" not in keys or not keys <= set(GEMMA4_PER_LAYER_FIELDS):
        return None
    if "num_key_value_heads" in keys and text_config.get("attention_k_eq_v") is not True:
        return None
    fields = {GEMMA4_PER_LAYER_FIELDS[k]: v for k, v in overrides[0].items()}
    if any(f in text_config for f in fields):
        return None
    return fields


def gemma4_config_compat(report):
    """Load configs in this process with the newer Gemma 4 forms turned back into what the pinned
    transformers has: a per_layer_config into the global fields (gemma4_per_layer_fields) and the
    vision rotary type "axial" into "default", so the source loads as google/gemma-4-E2B-it's own
    config does. The files on disk stay as the Hub has them (the lock's source hashes hold); the
    report and the lock record each rewrite when it happens, so only such a lock carries it. Any
    other per_layer_config or rotary type is left for transformers to judge."""
    from transformers import configuration_utils, modeling_rope_utils
    from transformers.models.gemma4 import configuration_gemma4
    pinned = configuration_gemma4.Gemma4TextConfig.__dataclass_fields__
    per_layer = all(f in pinned for f in GEMMA4_PER_LAYER_FIELDS.values())
    axial = ("axial" not in modeling_rope_utils.ROPE_INIT_FUNCTIONS and getattr(
        configuration_gemma4.Gemma4VisionConfig, "default_rope_type", "default") != "axial")
    if not (per_layer or axial):
        return
    base = configuration_utils.PreTrainedConfig
    get_config_dict = base.get_config_dict.__func__

    def record(label):
        compat = report["decisions"].setdefault("compat", [])
        if label not in compat:
            compat.append(label)
            print(f"NOTE: {label}")

    def compat_get_config_dict(cls, *args, **kwargs):
        config_dict, rest = get_config_dict(cls, *args, **kwargs)
        model_type, version = config_dict.get("model_type"), config_dict.get("transformers_version")
        text_config = {"gemma4": config_dict.get("text_config"),
                       "gemma4_text": config_dict}.get(model_type)
        vision_config = {"gemma4": config_dict.get("vision_config"),
                         "gemma4_vision": config_dict}.get(model_type)
        fields = (gemma4_per_layer_fields(text_config)
                  if per_layer and isinstance(text_config, dict) else None)
        if fields:
            layers = sorted(int(k) for k in text_config.pop("per_layer_config"))
            text_config.update(fields)
            record(f"gemma4-per_layer_config {fields} on layers {layers} (written by transformers "
                   f"{version}; read as global fields)")
        rope = vision_config.get("rope_parameters") if isinstance(vision_config, dict) else None
        if axial and isinstance(rope, dict) and rope.get("rope_type") == "axial":
            rope["rope_type"] = "default"
            record(f"gemma4-vision rope_type axial (written by transformers {version}; read as "
                   "default, the pinned name of the same rotary)")
        return config_dict, rest

    base.get_config_dict = classmethod(compat_get_config_dict)


class StageTracker:
    """Which stage the stock export is in — load, trace, convert, quantize,
    bundle — written to convert_report.json when it changes, so a failure or a
    --max-minutes timeout names its stage. The export calls each hooked function
    through its module attribute (export.py: export_lib.load_model; core.py:
    torch.export.export and litert_converter.exported_programs_to_flatbuffer;
    export_lib: quantize_model), so wrapping the attribute is enough."""

    def __init__(self, report):
        self.report = report

    def enter(self, stage):
        if self.report.get("stage") != stage:
            self.report["stage"] = stage
            write_report(self.report)

    def install(self):
        import functools
        import importlib
        import torch
        from litert_torch.generative.export_hf.core import export_lib
        from litert_torch.generative.export_hf.core import litert_lm_builder as hf_bundle
        converter = importlib.import_module("litert_torch._convert.litert_converter")

        def wrap(owner, name, stage, after=None):
            fn = getattr(owner, name)

            @functools.wraps(fn)
            def staged(*a, **k):
                self.enter(stage)
                out = fn(*a, **k)
                if after:
                    self.enter(after)
                return out
            setattr(owner, name, staged)

        wrap(export_lib, "load_model", "load")
        wrap(torch.export, "export", "trace", after="convert")
        wrap(converter, "exported_programs_to_flatbuffer", "convert")
        wrap(export_lib, "quantize_model", "quantize")
        wrap(export_lib, "export_tokenizer", "bundle")
        wrap(hf_bundle, "package_model", "bundle")


def run_capped(args, out_dir):
    """--max-minutes: rerun this command as a child and kill its process group
    at the cap. The child keeps convert_report.json's `stage` current, so a kill
    still names the stage it was in."""
    import signal
    env = dict(os.environ, **{CAPPED_CHILD_ENV: "1"})
    t0 = time.time()
    proc = subprocess.Popen([sys.executable, *sys.argv], env=env, start_new_session=True)
    try:
        return proc.wait(timeout=args.max_minutes * 60)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()
    path = out_dir / "convert_report.json"
    report = (json.loads(path.read_text()) if path.exists()
              else {"model": args.model, "output_dir": str(out_dir), "decisions": {}})
    report["status"] = "failed"
    report["failure"] = failure(report.get("stage") or "fetch",
                                f"timeout after {args.max_minutes:g} min (--max-minutes)",
                                seconds=round(time.time() - t0),
                                toolchain=(report.get("toolchain") or {}).get("id"))
    write_report(report)
    print(report["failure"]["line"], file=sys.stderr)
    return 3


def curated_key(model_id):
    """The scripts/reproduce_llm.sh key of a repo this repo ships with a
    hand-made recipe (recipes/curated_index.json), else None."""
    try:
        return json.loads(CURATED_INDEX.read_text())["models"].get(model_id)
    except (OSError, ValueError, KeyError):
        return None


def vet_adapter_base(report, model_id, files, api, revision):
    """Vetting half of the adapter path: resolve the base from
    adapter_config.json and refuse what stock export could not convert
    honestly (gated / remote-code / pre-quantized BASE). Downloads no weights —
    `--check` stops here. Returns (base_id, base_info, base_cfg, gate_backend)."""
    from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError

    ac_text = hub_text(model_id, "adapter_config.json", revision)
    if not ac_text:
        fail("could not fetch adapter_config.json", stage="fetch")
    base_id = json.loads(ac_text).get("base_model_name_or_path")
    if not base_id:
        refuse(report, "adapter_no_base",
               "adapter_config.json has no base_model_name_or_path",
               "merge the adapter manually onto its base, then convert the merged repo")
    report["decisions"]["adapter_base"] = base_id
    if not any(f.startswith("adapter_model.") for f in files):
        refuse(report, "no_weights",
               "adapter_config.json present but no adapter_model.* weight file",
               "point at a repo with adapter weights or full PyTorch weights")

    try:
        base_info = api.model_info(base_id, files_metadata=False)
    except GatedRepoError:
        refuse(report, "adapter_base_gated",
               f"base model {base_id} is gated — its weights are not openly downloadable",
               "adapters on a gated base are not converted, with or without a token")
    except RepositoryNotFoundError:
        refuse(report, "adapter_base_missing",
               f"base model {base_id} not found on the Hub",
               "the adapter's base pointer is dangling; merge manually from a real base")
    if getattr(base_info, "gated", False):
        refuse(report, "adapter_base_gated",
               f"base model {base_id} is gated (gated={base_info.gated!r})",
               "adapters on a gated base are not converted, with or without a token")

    report["decisions"]["adapter_base_revision"] = base_info.sha
    base_cfg_text = hub_text(base_id, "config.json", base_info.sha)
    if not base_cfg_text:
        fail(f"could not fetch base config.json for {base_id}", stage="fetch")
    base_cfg = json.loads(base_cfg_text)
    if base_cfg.get("auto_map"):
        if native_model_type(base_cfg.get("model_type")):
            report["decisions"]["auto_map"] = "compat_shim_ignored_native_arch"
            print(f"NOTE: base {base_id} declares auto_map but model_type "
                  f"{base_cfg.get('model_type')!r} is native in transformers — the repo "
                  "code is a back-compat shim and is never loaded")
        else:
            refuse(report, "remote_code",
                   f"base {base_id} declares auto_map ({list(base_cfg['auto_map'])}) and "
                   f"model_type {base_cfg.get('model_type')!r} is not in transformers — "
                   "the architecture lives in repo code",
                   "not converted: remote-code architectures are out of scope for the stock exporter")
    if base_cfg.get("quantization_config"):
        refuse(report, "pre_quantized",
               f"base {base_id} is already quantized "
               f"({base_cfg['quantization_config'].get('quant_method', '?')}) — "
               "LoRA cannot merge into quantized weights honestly",
               "convert from the full-precision base instead")
    gate_backend = hybrid_gate_backend(report, base_cfg)
    if base_cfg.get("model_type") == "zamba2":
        zamba2_preport_guard(report, base_cfg)

    return base_id, base_info, base_cfg, gate_backend


def download_and_merge_adapter(report, model_id, base_id, out_dir, revision, base_revision):
    """Download half of the adapter path: snapshot the base and the adapter,
    then merge_and_unload (subprocess-isolated) into a temporary hub-format
    dir that the stock export takes as its model arg. Only reached once
    vet_adapter_base has already cleared the base.

    A full model.safetensors sitting in an adapter repo is deliberately
    ignored: measured 2026-08-24, such a dump can be a different training
    checkpoint than the published adapter weights.

    Returns merged_dir. Tokenizer comes from the adapter repo when it ships
    tokenizer_config.json, else from the base; if the adapter's tokenizer
    carries no chat template, the base's template is inherited."""
    from huggingface_hub import snapshot_download

    print(f"adapter repo: merging into base {base_id} (merge-first, in a subprocess)")
    base_dir = Path(snapshot_download(base_id, revision=base_revision))
    adapter_dir = Path(snapshot_download(model_id, revision=revision, allow_patterns=[
        "adapter_config.json", "adapter_model.*", "tokenizer*", "*.model",
        "special_tokens_map.json", "vocab*", "merges.txt",
        "chat_template.jinja", "generation_config.json"]))

    merged_dir = out_dir / "merged"
    out_dir.mkdir(parents=True, exist_ok=True)
    info_path = out_dir / "merge_info.json"
    proc = subprocess.run(
        [sys.executable, "-c", MERGE_CHILD,
         str(base_dir), str(adapter_dir), str(merged_dir), str(info_path)],
        text=True)
    if proc.returncode != 0 or not info_path.exists():
        fail(f"adapter merge subprocess failed (exit {proc.returncode}); "
             "if the traceback above is an ImportError: uv sync (peft is in uv.lock)",
             stage="load")
    merge_info = json.loads(info_path.read_text())
    info_path.unlink()

    report["decisions"]["adapter"] = {
        "base": base_id, **merge_info, "merged_dir": str(merged_dir),
    }
    return merged_dir


def main():
    global _REPORT
    sys.stdout.reconfigure(line_buffering=True)  # keep stdout in step with the exporter's stderr in a log
    ap = argparse.ArgumentParser()
    ap.add_argument("model", help="HF model id, e.g. org/name")
    ap.add_argument("--revision",
                    help="commit sha, branch or tag to convert (default: main, resolved "
                         "to its commit sha; the lock records the sha)")
    ap.add_argument("--recipe", choices=sorted(RECIPES), default="int8",
                    help="int8 = the export's dynamic_wi8_afp32 (default); int4 = the "
                         "proven blockwise-32 OCTAV recipe with an int8 embedding")
    ap.add_argument("--int4", action="store_true", help="alias of --recipe int4")
    ap.add_argument("--out", help="output dir (default: out/<name>)")
    ap.add_argument("--min-correct", type=int, default=6, help="exit-gate bar (of 8)")
    ap.add_argument("--keep-merged", action="store_true",
                    help="keep the temporary merged dir of an adapter conversion")
    ap.add_argument("--gate-script",
                    help="task-model gate: run this script (args: bundle --backend B "
                         "--litert-lm CLI --out JSON; exit 0 = pass) instead of the "
                         "generic 8-question gate — for finetunes that transform their "
                         "input rather than answer questions (s1-mini/Tashkeel), where "
                         "the generic gate certifies nothing")
    ap.add_argument("--check", action="store_true",
                    help="run the entry gate only — write convert_report.json with "
                         "status 'accepted' and the decisions made so far (revision, "
                         "route, adapter base, params), print it, exit 0; refusals still "
                         "exit 2. No weights are downloaded")
    ap.add_argument("--smoke", action="store_true",
                    help="scripts/check_bundle.py's one-prompt check instead of the 8-question "
                         "gate, the run capped at --max-minutes (45 unless given), the failure "
                         "in convert_report.json — for converting a batch of models")
    ap.add_argument("--max-minutes", type=float,
                    help="kill the run after this many minutes (the report names the stage)")
    ap.add_argument("--curated", action="store_true",
                    help="run this repo's hand-made recipe (recipes/curated_index.json -> "
                         "scripts/reproduce_llm.sh) instead of the generic path; no lock")
    ap.add_argument("--no-gate", action="store_true",
                    help="build and lock only, no exit gate and no check "
                         "(scripts/verify_lock.py rebuilds this way)")
    args = ap.parse_args()
    if args.int4:
        args.recipe = "int4"
    args.int4 = args.recipe == "int4"

    name = args.model.split("/")[-1]
    out_dir = Path(args.out) if args.out else OUT_ROOT / name
    if args.smoke and args.max_minutes is None:
        args.max_minutes = 45
    if args.max_minutes and not os.environ.get(CAPPED_CHILD_ENV):
        sys.exit(run_capped(args, out_dir))
    report = {
        "model": args.model,
        "revision": None,
        "recipe": args.recipe,
        "output_dir": str(out_dir),
        "status": "started",
        "stage": "fetch",
        "decisions": {},
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    _REPORT = report
    write_report(report)
    # hf_xet stalls on multi-GB downloads on this host; plain HTTP is set before the import
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

    # ---------------- entry gate ----------------
    from huggingface_hub import HfApi
    from huggingface_hub.utils import (GatedRepoError, HfHubHTTPError,
                                       RepositoryNotFoundError, RevisionNotFoundError)
    api = HfApi()
    try:
        info = api.model_info(args.model, revision=args.revision, files_metadata=False)
    except GatedRepoError:
        refuse(report, "gated",
               "the repo requires accepting a license/terms before download",
               GATED_ACTION)
    except RevisionNotFoundError as e:
        fail(f"revision {args.revision} not found in {args.model}", stage="fetch",
             context=f"RevisionNotFoundError {e}", code=2)
    except RepositoryNotFoundError as e:
        fail(f"model not found on the Hub: {args.model}", stage="fetch",
             context=f"RepositoryNotFoundError {e}", code=2)
    except HfHubHTTPError as e:
        fail(f"{type(e).__name__}: {first_line(str(e))}", stage="fetch", context=str(e), code=2)
    sha = info.sha
    commit_date = info.last_modified.astimezone(datetime.timezone.utc).isoformat()
    report["revision"] = sha
    report["decisions"]["commit_date"] = commit_date
    print(f"source: {args.model} @ {sha} ({commit_date})")

    key = curated_key(args.model)
    if key:
        report["decisions"]["curated_recipe"] = key
        if args.curated and PACKAGE:
            refuse(report, "curated_needs_repo",
                   f"the hand-made recipe {key} runs from a clone of {REPO_URL}",
                   f"git clone {REPO_URL} && cd hf-to-litertlm && "
                   f"bash scripts/reproduce_llm.sh {key} (REPRODUCE.md)")
        if args.curated:
            print(f"--curated: bash scripts/reproduce_llm.sh {key} (no lock on this path)")
            write_report(report)
            os.execvp("bash", ["bash", str(REPO_ROOT / "scripts" / "reproduce_llm.sh"), key])
        print(f"NOTE: this repo ships a hand-made recipe for {args.model}: "
              f"bash scripts/reproduce_llm.sh {key} (REPRODUCE.md); continuing on the "
              "generic path (--curated runs that recipe instead)")
    elif args.curated:
        fail(f"no curated recipe for {args.model} in recipes/curated_index.json", stage="gate",
             code=2)

    if getattr(info, "gated", False):
        refuse(report, "gated",
               f"gated={info.gated!r} — weights are not openly downloadable",
               GATED_ACTION)

    report["stage"] = "gate"
    files = {s.rfilename for s in (info.siblings or [])}
    merged_dir = base_info = gate_backend = normalized_dir = None
    is_adapter = "adapter_config.json" in files
    if is_adapter:
        base_id, base_info, base_cfg, gate_backend = vet_adapter_base(
            report, args.model, files, api, sha)
        if not args.check:
            merged_dir = download_and_merge_adapter(report, args.model, base_id, out_dir,
                                                    sha, base_info.sha)
    elif not any(f.endswith(".safetensors") or f.endswith(".bin") for f in files):
        refuse(report, "no_weights",
               "no safetensors/bin weight files in the repo",
               "point at a repo with full PyTorch weights")

    if not is_adapter:  # adapter path already vetted the BASE's config
        cfg_text = hub_text(args.model, "config.json", sha)
        if not cfg_text:
            fail("could not fetch config.json", stage="fetch", code=2)
        cfg = json.loads(cfg_text)
        if cfg.get("auto_map"):
            if native_model_type(cfg.get("model_type")):
                report["decisions"]["auto_map"] = "compat_shim_ignored_native_arch"
                print(f"NOTE: config declares auto_map but model_type "
                      f"{cfg.get('model_type')!r} is native in transformers — the repo "
                      "code is a back-compat shim and is never loaded")
            else:
                refuse(report, "remote_code",
                       f"config.json declares auto_map ({list(cfg['auto_map'])}) and model_type "
                       f"{cfg.get('model_type')!r} is not in transformers — the architecture "
                       "lives in repo code",
                       "not converted: remote-code architectures are out of scope for the stock exporter")
        if cfg.get("quantization_config"):
            refuse(report, "pre_quantized",
                   f"weights are already quantized ({cfg['quantization_config'].get('quant_method', '?')})",
                   "convert from the full-precision source repo instead")
        gate_backend = hybrid_gate_backend(report, cfg)
        if cfg.get("model_type") == "zamba2":
            zamba2_preport_guard(report, cfg)
        rope = cfg.get("rope_scaling") or {}
        if (cfg.get("model_type") == "hunyuan_v1_dense" and cfg.get("head_dim")
                and rope.get("type") == "dynamic" and rope.get("alpha")):
            if args.check:
                report["decisions"]["rope_normalization"] = "would_apply_static_alpha_bake"
            else:
                normalized_dir = normalize_static_alpha_rope(report, args.model, cfg, out_dir, sha)

    # ---------------- opt-ins ----------------
    params = getattr(getattr(base_info or info, "safetensors", None), "total", None)
    externalize = bool(params and params >= 3e9)
    report["decisions"]["params"] = params
    report["decisions"]["externalize_embedder"] = externalize
    if params is None:
        print("NOTE: param count unavailable from the Hub; not externalizing the embedder")

    export_kwargs = dict(EXPORT_DEFAULTS)
    if externalize:
        export_kwargs["externalize_embedder"] = True
    export_kwargs["quantization_recipe"] = RECIPES[args.recipe]
    report["decisions"]["quantization"] = "int4_block32_octav" if args.int4 else "default_int8"

    # ≥3B models get the reduced prefill ladder the shipped Qwen3.5-4B and
    # granite-4.1-3b use (6 prefill lengths + decode = 7 signatures instead of
    # 12). Three measured reasons: every signature costs engine RAM even if
    # never called (a 12 GB iPhone jetsam-kills a 248k-vocab hybrid 4B at Metal
    # program init with the full ladder — REPRODUCE 2026-08-14 — and iOS kills
    # the DENSE granite-4.1-3b at Metal init with 11 signatures too), and the
    # export's converter passes scale with the merged module, which OOM-killed
    # a full-ladder 4B export on a 128 GB host twice (2026-08-24). No
    # correctness cost: the runtime plans coarser prefill chunks.
    if externalize:
        export_kwargs["prefill_lengths"] = [1024, 256, 64, 16, 4, 1]
        report["decisions"]["prefill_ladder"] = "reduced_7sig_3b"

    recipe = HYBRID_RECIPE.get(report["decisions"].get("model_type"))
    if recipe and PACKAGE:
        refuse(report, "recipe_needs_repo",
               f"model_type {report['decisions']['model_type']} converts only through its family "
               "recipe and a patched litert-torch checkout, which the package does not carry",
               f"in a clone of {REPO_URL}: {recipe['setup']} ; then "
               f"uv run python scripts/convert.py {args.model}")
    # The lock covers the generic stock path from a plain repo; an adapter merge,
    # a rope-normalized source and a family recipe carry none yet.
    lockable = not (is_adapter or normalized_dir or recipe or report["decisions"].get(
        "rope_normalization"))
    report["decisions"]["lock"] = lockable

    if args.check:
        report["status"] = "accepted"
        write_report(report)
        print(json.dumps(report, indent=2))
        sys.exit(0)

    # ---------------- source (pinned commit, every file checked) ----------------
    source_files = None
    if lockable:
        report["stage"] = "fetch"
        names = pick_source_files(files)
        snap, cached = download_source(report, args.model, sha, names)
        source_files = verify_source(report, api, args.model, sha, snap, names, cached)
        d = report["download"]
        print(f"source: {d['verified']} files verified against the Hub tree "
              f"({d['bytes_fetched']:,} B fetched, {d['files_already_cached']} already cached, "
              f"{d['seconds']} s)")
        export_src = snap
    else:
        export_src = merged_dir or normalized_dir or args.model
    # every config load below (template, export, runtime check) goes through it
    gemma4_config_compat(report)

    # ---------------- template lint (warn only) ----------------
    template = get_chat_template(export_src, sha)
    if template is None and base_info is not None:
        template = get_chat_template(report["decisions"]["adapter_base"], base_info.sha)
    methods = sorted(set(PY_METHOD_RE.findall(template))) if template else []
    uses_strftime = bool(template and "strftime_now" in template)
    is_think = bool(template and ("<think>" in template or "enable_thinking" in template))
    report["decisions"]["template_lint"] = {
        "python_methods": methods, "strftime_now": uses_strftime, "think_model": is_think,
    }
    if methods or uses_strftime:
        print(f"WARNING: chat template uses {methods + (['strftime_now'] if uses_strftime else [])}; "
              "renders on current runtimes (verified 2026-08-24), pre-2026 runtimes may fail "
              "on the first message")

    # ---------------- chat template: the form the 0.18 runtime renders ----------------
    # litert-lm 0.18.0 renders the template on native values: content is a list
    # of text parts and a map has no Python methods. A transformers template then
    # drops the message (Qwen3-1.7B: the runtime's own render of the user turn
    # came back empty, 2026-10-19), raises (SmolLM2: str + list; gemma-4-E2B-it:
    # "map has no method named get", template line 238) or prints the list. The
    # edited template goes into the export only once its runtime render equals
    # transformers' byte for byte (scripts/chat_template_dual.py).
    template_record = template_path = None
    if not recipe:
        template_record, template_path = prepare_template(report, export_src, out_dir)
        if template_path:
            export_kwargs["jinja_chat_template_override"] = str(template_path)

    # ---------------- export (stock, or family recipe) ----------------
    toolchain = toolchain_info()
    report["toolchain"] = {k: toolchain[k] for k in ("id", "python", "pinned_env", "uv_lock_sha256")}
    lock = lock_path = None
    if recipe:
        gate_backend = "cpu"
        report["decisions"]["gate_backend"] = "cpu"
        bundle = convert_via_recipe(report, recipe, str(export_src), out_dir, args)
        report["bundle_bytes"] = bundle.stat().st_size
        ensure_turn_end_stop(bundle, export_src, report)
        # The same generic guard the stock path runs. Its condition reproduces
        # every hand-made per-family choice here (measured 2026-08-25: fires on
        # granite — which the family flag above already dropped — and on
        # nemotron_h; stays silent on Falcon-H1 and Zamba2, whose templates
        # render their BOS themselves). nemotron_h was the family with the
        # condition and no guard: its bundles (the shipped 4B included) prepend
        # <s> to a template that starts <SPECIAL_10>/<|im_start|>, with
        # add_bos_token False. At 4B the model is robust — gate 7/8 either way,
        # HF greedy identical on 2 of 3 probes — but the mismatch is real and
        # granite-350m showed it bites at small scale.
        ensure_no_spurious_start_token(bundle, export_src, report)
    else:
        print(f"exporting {export_src} -> {out_dir}  (stock, opts={export_kwargs})")
        system_metadata = {
            "uuid": str(uuid.uuid5(uuid.NAMESPACE_URL,
                                   f"{args.model}@{sha}#{args.recipe}#{toolchain['id']}")),
            "creation_timestamp": commit_date,
        }
        pin_system_metadata(system_metadata["uuid"], system_metadata["creation_timestamp"])
        builder_compat(report)
        StageTracker(report).install()
        report["stage"] = "load"
        from litert_torch.generative.export_hf.export import export
        t0 = time.time()
        try:
            export(model=str(export_src), output_dir=str(out_dir), **export_kwargs)
        except Exception as e:  # noqa: BLE001 — every failure becomes one FAIL line
            import traceback
            tb = traceback.format_exc()
            sys.stderr.write(tb)
            report["export_seconds"] = round(time.time() - t0)
            report["status"] = "export_failed"
            fail(f"{type(e).__name__}: {e}", stage=report.get("stage") or "load",
                 context=f"{tb}\nmodel_type={report['decisions'].get('model_type')}")
        report["export_seconds"] = round(time.time() - t0)
        report["stage"] = "bundle"
        exported = out_dir / "model.litertlm"
        if not exported.exists():
            report["status"] = "export_failed"
            fail("export finished without producing model.litertlm", stage="bundle")
        ensure_turn_end_stop(exported, export_src, report)
        ensure_no_spurious_start_token(exported, export_src, report, in_process=True)
        if report["decisions"].get("model_type") in HYBRID_EXEC_RETROFIT:
            ensure_executor_metadata(exported, report)
        bundle = out_dir / f"{name}_{args.recipe}.litertlm"
        exported.replace(bundle)
        report["bundle_export_name"] = exported.name
        report["bundle_bytes"] = bundle.stat().st_size
        if template_record and template_record["form"] != "none" and not args.no_gate:
            template_record["runtime"] = check_runtime_template(report, bundle, export_src,
                                                                template_record)
        if lockable:
            ident = bundle_identity(bundle)
            pinned = {k: ident["system_metadata"].get(k) for k in system_metadata}
            ident["system_metadata_pinned"] = pinned == system_metadata
            if not ident["system_metadata_pinned"]:
                print(f"WARNING: the bundle header carries {pinned}, not the pinned "
                      f"{system_metadata} — a pack outside this process; the bytes will "
                      "not repeat")
            rq = RECIPES[args.recipe]
            lock = {
                "lock_version": LOCK_VERSION,
                "command": f"{COMMAND} {args.model} --revision {sha} --recipe {args.recipe}",
                "source": {"repo": args.model, "revision": sha, "commit_date": commit_date,
                           "files": source_files},
                "recipe": {
                    "id": args.recipe,
                    "export_kwargs": {k: (str(Path(v).relative_to(REPO_ROOT))
                                          if k == "quantization_recipe" and v.endswith(".json")
                                          else Path(v).name if k == "jinja_chat_template_override"
                                          else v) for k, v in export_kwargs.items()},
                    "quantization_recipe_sha256": recipe_sha256(rq),
                    "post_export": {k: report["decisions"][k] for k in POST_EXPORT_KEYS
                                    if k in report["decisions"]},
                    "compat": report["decisions"].get("compat", []),
                },
                "template": template_record,
                "toolchain": toolchain,
                "process": {"PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED")},
                "converter": converter_info(),
                "bundle": ident,
                "check": None,
                "built_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "host": host_info(),
            }
            lock_path = out_dir / f"{name}.lock.json"
            write_lock(lock_path, lock)
            report["lock"] = str(lock_path)
            print(f"lock: {lock_path} (bundle sha256 {ident['sha256']})")
        else:
            print("NOTE: no lock for this source (adapter merge / rope-normalized source)")
    report["bundle"] = str(bundle)

    if args.no_gate:
        report["status"] = "built"
        report["stage"] = None
        write_report(report)
        print(f"built (no gate): {bundle} ({report['bundle_bytes']:,} B)")
        sys.exit(0)

    # ---------------- exit gate ----------------
    from check_bundle import check as check_bundle
    cli = resolve_litert_lm_cli()
    gate_json = out_dir / "gate.json"
    t0 = time.time()
    proc = None
    if args.smoke:
        report["stage"] = "engine-load"
        result = check_bundle(bundle)
        report["gate"] = {"smoke": result}
        passed = result["pass"]
    elif args.gate_script:
        print(f"exit gate: task gate {args.gate_script}"
              + (f" --backend {gate_backend}" if gate_backend else ""))
        gate_cmd = [sys.executable, args.gate_script, str(bundle),
                    "--backend", gate_backend or "cpu", "--out", str(gate_json)]
        if cli:
            gate_cmd += ["--litert-lm", cli]
        proc = subprocess.run(gate_cmd, text=True)
        gate = json.loads(gate_json.read_text()) if gate_json.exists() else None
        report["gate"] = {"script": args.gate_script, "passed": proc.returncode == 0}
        if gate:
            report["gate"].update({k: gate[k] for k in ("match_hf", "n") if k in gate})
        passed = proc.returncode == 0
    else:
        max_tokens = 3200 if is_think else 512
        gate_cmd = [sys.executable, str(VERIFY), str(bundle),
                    "--max-tokens", str(max_tokens), "--min-correct", str(args.min_correct),
                    "--json", str(gate_json)]
        env = None
        if recipe:
            if gate_backend:
                gate_cmd += ["--backend", gate_backend]
        else:
            # the stock path gates on the CLI of this environment (CPU, greedy):
            # litert-mac-verify's --backend contract through scripts/litert_lm_cli_verifier.py
            gate_cmd += ["--verifier", str(CLI_VERIFIER), "--backend", gate_backend or "cpu"]
            env = dict(os.environ, LITERT_LM_CLI=cli or "")
        print("exit gate: " + " ".join(gate_cmd[1:]))
        proc = subprocess.run(gate_cmd, text=True, env=env)
        gate = json.loads(gate_json.read_text()) if gate_json.exists() else None
        report["gate"] = {k: gate[k] for k in ("score", "of", "degenerate", "passed")} if gate else None
        if report["gate"] is not None and not recipe:
            report["gate"]["engine"] = "litert-lm CLI (scripts/litert_lm_cli_verifier.py)"
            report["gate"]["backend"] = gate_backend or "cpu"
        passed = proc.returncode == 0
    report["gate_seconds"] = round(time.time() - t0)

    # ---------------- check (the pinned CLI loads it and the user's text reaches the model) ----------------
    checked = report["gate"]["smoke"] if args.smoke else check_bundle(bundle)
    report["check"] = {k: checked[k] for k in ("engine", "backend", "seconds", "answer_head", "pass",
                                               "arithmetic")}
    if lock is not None:
        lock["check"] = {k: checked[k] for k in ("engine", "backend", "prompt", "answer_head",
                                                 "pass", "seconds", "arithmetic")}
        write_lock(lock_path, lock)

    if args.smoke:
        report["status"] = "smoke_pass" if passed else "smoke_failed"
    elif not passed:
        report["status"] = "converted_gate_failed"
    elif not checked["pass"]:
        report["status"] = "converted_check_failed"
    else:
        report["status"] = "converted_pass"
    report["stage"] = None
    if args.smoke or (passed and not checked["pass"]):
        report["failure"] = checked.get("failure") and {**checked["failure"], **failure_fields()}
    elif not passed:
        g = report["gate"] or {}
        what = (f"8-question gate {g['score']}/8 (bar {args.min_correct}/8"
                + (", degenerate)" if g.get("degenerate") else ")")
                if "score" in g else f"gate {args.gate_script or 'harness'} failed (exit "
                f"{proc.returncode if proc else '?'})")
        report["failure"] = failure("generate", what, **failure_fields())
    if merged_dir and not args.keep_merged:
        shutil.rmtree(merged_dir, ignore_errors=True)
        print(f"removed temporary merged dir {merged_dir} (--keep-merged retains it)")
    if normalized_dir and not args.keep_merged:
        shutil.rmtree(normalized_dir, ignore_errors=True)
        print(f"removed temporary rope-normalized dir {normalized_dir} (--keep-merged retains it)")
    write_report(report)
    print(f"{report['status']}: {bundle} ({report['bundle_bytes']:,} B); "
          f"report: {out_dir / 'convert_report.json'}")
    if report.get("failure"):
        print(report["failure"]["line"], file=sys.stderr)
    sys.exit(0 if report["status"] in ("converted_pass", "smoke_pass") else 1)


def prepare_template(report, export_src, out_dir):
    """Prove the template the export would embed (the tokenizer's chat_template)
    renders on the runtime as transformers renders it (scripts/chat_template_dual.py),
    with the edits that needs (.get -> indexing, the dual form), and write the
    edited template for the export's jinja_chat_template_override. Returns (lock
    record, path or None); a template that cannot be proven stops the run before
    the export (no bundle)."""
    from transformers import AutoTokenizer
    from chat_template_dual import prove
    tmpl = AutoTokenizer.from_pretrained(str(export_src)).chat_template
    if not isinstance(tmpl, str) or not tmpl.strip():
        report["decisions"]["template"] = {"form": "none"}
        return {"form": "none", "reason": "the tokenizer has no single string chat template"}, None
    report["stage"] = "template"
    proof, final = prove(tmpl)
    record = {k: proof.get(k) for k in ("form", "edits", "original_sha256", "final_sha256", "classes",
                                        "basis", "cases", "oracle", "macro", "macro_sha256")}
    report["decisions"]["template"] = {k: record[k] for k in ("form", "edits", "classes", "basis")}
    if not proof["pass"]:
        fail(f"content read left unwrapped: {proof['unwrapped']}" if proof.get("unwrapped")
             else proof["reason"], stage="template")
    if final == tmpl:
        print(f"chat template: unchanged (the runtime renders it as transformers does; basis "
              f"{proof['basis']})")
        return record, None
    path = out_dir / "chat_template.bundle.jinja"
    out_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(final)
    print(f"chat template: {proof['form']} ({proof['edits']}; the original on the runtime: "
          f"{proof['classes']}; the edited template renders transformers' prompt byte for byte, "
          f"basis {proof['basis']}) -> {path.name}")
    return record, path


def check_runtime_template(report, bundle, export_src, record):
    """The prompt the runtime builds is the prompt the proof compared with:
    litert-lm's Conversation.render_message_to_string (scripts/chat_template_dual.py
    --runtime-render, its own process) against transformers' apply_chat_template
    of the model's own template on c1-c3 — with string content, or with parts for
    a case whose basis is parts (a template with its own list branch) — the
    bundle's start token in front when it keeps one, whole or as the increment
    after the history. Anything else stops the run."""
    from transformers import AutoTokenizer
    from chat_template_dual import RUNTIME_CASES, cases
    tok = AutoTokenizer.from_pretrained(str(export_src))
    proc = subprocess.run([sys.executable, str(REPO_ROOT / "scripts" / "chat_template_dual.py"),
                           "--runtime-render", str(bundle)], capture_output=True, text=True)
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout).strip().splitlines()
        fail("runtime render failed: " + (err[-1] if err else f"exit {proc.returncode}"),
             stage="engine-load", context=proc.stderr[-4000:])
    got = json.loads(proc.stdout.strip().splitlines()[-1])
    start = tok.bos_token if report["decisions"].get("start_token") == "kept_matches_reference" else ""
    result = {}
    for name, turns, _ in cases(""):
        if name not in RUNTIME_CASES:
            continue
        form = record["basis"].get(name, "string")
        msgs = [{"role": r, "content": t if form == "string" else [{"type": "text", "text": t}]}
                for r, t in turns]
        full = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        hist = (tok.apply_chat_template(msgs[:-1], tokenize=False, add_generation_prompt=False)
                if len(msgs) > 1 else "")
        if got[name] and start + got[name] == full:
            result[name] = form
        elif got[name] and hist and full.startswith(hist) and got[name] == full[len(hist):]:
            result[name] = form + " increment"
        else:
            fail(f"the runtime's prompt for {name} is not what transformers renders ({form} "
                 f"basis): {got[name][:120]!r}", stage="template")
    report["decisions"]["template_runtime"] = result
    print(f"chat template: the runtime's prompt equals transformers' render ({result})")
    return result


def recipe_sha256(recipe):
    """sha256 of the quantization recipe the export resolves: the file's bytes
    for a .json recipe, else the canonical JSON of ai_edge_quantizer's named
    recipe (the export calls recipe_lib.<name>())."""
    if recipe.endswith(".json"):
        return sha256_file(recipe)
    from ai_edge_quantizer import recipe as recipe_lib
    canon = json.dumps(getattr(recipe_lib, recipe)(), sort_keys=True, default=str)
    return hashlib.sha256(canon.encode()).hexdigest()


def host_info():
    import platform
    return {"platform": platform.platform(), "machine": platform.machine()}


def write_lock(path, lock):
    Path(path).write_text(json.dumps(lock, indent=2) + "\n")


if __name__ == "__main__":
    # The export orders sets of strings by their hash (litert-torch 0.9.4
    # build_llm_metadata iterates `expanded_stop_tokens = set()` into
    # LlmMetadata.stop_tokens), and Python randomizes string hashes per
    # process: two runs wrote the same stop tokens in two orders (2026-10-19,
    # Qwen3-1.7B, LlmMetadata 4,449 B both times, every other section equal).
    # Rerun under a fixed seed before anything else happens.
    if os.environ.get("PYTHONHASHSEED") != HASH_SEED:
        os.execve(sys.executable, [sys.executable, *sys.argv],
                  dict(os.environ, PYTHONHASHSEED=HASH_SEED))
    try:
        main()
    except Exception as e:  # noqa: BLE001 — anything not already a FAIL line becomes one
        import traceback
        tb = traceback.format_exc()
        sys.stderr.write(tb)
        fail(f"{type(e).__name__}: {e}", stage=(_REPORT or {}).get("stage") or "fetch",
             context=tb)
