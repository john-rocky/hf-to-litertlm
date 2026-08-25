#!/usr/bin/env python3
"""Thin gated wrapper around the stock litert-torch export.

    ~/venvs/ltconv040dev/bin/python scripts/convert.py <hf_model_id> [--out DIR]
                                                       [--int4] [--min-correct N]

Measured basis (2026-08-24): stock `export(model, output_dir)` with no other
arguments converts untouched finetunes correctly on both the model_ext path
(qwen3: QVikhr-3-1.7B) and the generic path (smollm3: a SmolLM3-3B finetune;
2026-08-25 adds MiniCPM5-1B derivatives — plain llama, default path, 7/8,
template byte-equal, A≡B) —
the derivative's own chat template is embedded verbatim (including a
`chat_template.jinja` that tokenizer_config.json doesn't reference) and the
runtime applies it (greedy A/B/C proof). So this wrapper adds NO template or
architecture handling. It adds only:

  1. an entry gate — refuse, with a structured reason, what stock export
     cannot convert honestly (gated repos, remote-code architectures,
     pre-quantized weights); never convert on a guess. Adapter (LoRA/PEFT)
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

Exit codes: 0 = converted and gate passed, 1 = converted but gate failed,
2 = refused at the entry gate (see the printed JSON), 3 = harness error.
Every run writes <out>/convert_report.json with the decisions taken.
"""
import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
INT4_RECIPE = REPO_ROOT / "recipes" / "qwen3_int4_block32_octav.json"
VERIFY = REPO_ROOT / "scripts" / "verify_quality.py"

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


def resolve_litert_lm_cli():
    """A litert-lm >= 0.15 CLI: $LITERT_LM_CLI, else the ~/venvs/lt0160run
    default, else `litert-lm` on PATH (last — which() can surface a pyenv shim
    that exits 127, so the known venv wins)."""
    import os
    default_cli = Path.home() / "venvs/lt0160run/bin/litert-lm"
    return (os.environ.get("LITERT_LM_CLI")
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
        fail("add_executor_metadata failed: " + (proc.stdout + proc.stderr)[-500:])

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
    t0 = time.time()
    proc = subprocess.run([sys.executable, str(script), export_src, str(out_dir)],
                          env=env, text=True)
    if proc.returncode:
        report["status"] = "export_failed"
        write_report(report)
        fail(f"recipe converter failed (exit {proc.returncode})")
    report["export_seconds"] = round(time.time() - t0)
    bundle = out_dir / (Path(export_src).name + "_int8.litertlm")
    if not bundle.exists():
        fail(f"recipe converter finished without producing {bundle.name}")
    if recipe.get("drop_start_token"):
        dropped = bundle.parent / (bundle.name + ".nobos")
        proc = subprocess.run(
            [sys.executable, str(REPO_ROOT / recipe["drop_start_token"]),
             str(bundle), str(dropped)], env=env, text=True)
        if proc.returncode:
            fail("drop_start_token failed")
        dropped.replace(bundle)
        report["decisions"]["start_token"] = "dropped"
        print("start-token guard: dropped (family template has no leading BOS)")
    report["decisions"]["recipe"] = recipe["script"]
    return bundle


def ensure_no_spurious_start_token(bundle, tokenizer_src, report):
    """Post-export guard: the bundling writes LlmMetadata start_token from
    tokenizer.bos_token without consulting add_bos_token or the chat template.
    When the template's own rendering never leads with BOS and the tokenizer
    says add_bos_token: False — or bos == eos, granite's shape, where the
    prepended token reads as 'this document already ended' — the engine feeds
    a token stream the model never saw in training. Measured on the base
    granite-4.1-3b: 8/8 -> 5/8 with echo-the-question failures that look
    exactly like quantization damage; measured on granite-4.0-h-350m and its
    Tashkeel derivative: greedy flips to garbage. No-op for every other shape
    (bos None, add_bos_token True, or a template that renders BOS itself)."""
    import os
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tokenizer_src)
    if tok.bos_token is None:
        return
    add_bos = getattr(tok, "add_bos_token", None)
    bos_eq_eos = tok.bos_token_id is not None and tok.bos_token_id == tok.eos_token_id
    if add_bos is not False and not bos_eq_eos:
        return
    try:
        rendered = tok.apply_chat_template([{"role": "user", "content": "x"}],
                                           tokenize=False, add_generation_prompt=True)
        if rendered.startswith(tok.bos_token):
            return  # the template leads with BOS on purpose
    except Exception:
        pass
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
        report["decisions"]["start_token"] = "dropped_spurious_bos"
        print("start-token guard: dropped (template never leads with BOS and "
              "add_bos_token is False / bos == eos)")
    elif "no leading start_token" in (proc.stdout + proc.stderr):
        report["decisions"]["start_token"] = "absent"
    else:
        fail("drop_start_token failed: " + (proc.stdout + proc.stderr)[-300:])


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


def fail(msg):
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(3)


def refuse(report, reason, detail, action):
    report["status"] = "refused"
    report["refusal"] = {"reason": reason, "detail": detail, "action": action}
    print(json.dumps(report["refusal"], indent=2))
    write_report(report)
    sys.exit(2)


def write_report(report):
    out = Path(report["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    (out / "convert_report.json").write_text(json.dumps(report, indent=2))


def hf_fetch(url):
    import urllib.request
    req = urllib.request.Request(url, headers={"User-Agent": "hf-to-litertlm-convert"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.read().decode()
    except Exception:
        return None


def get_chat_template(model_id):
    """The template stock export will embed: tokenizer_config's chat_template,
    else chat_template.jinja (both observed in the wild)."""
    tc = hf_fetch(f"https://huggingface.co/{model_id}/raw/main/tokenizer_config.json")
    if tc:
        try:
            t = json.loads(tc).get("chat_template")
            if isinstance(t, str) and t.strip():
                return t
        except json.JSONDecodeError:
            pass
    j = hf_fetch(f"https://huggingface.co/{model_id}/raw/main/chat_template.jinja")
    return j


def merge_adapter_first(report, model_id, files, out_dir, api):
    """Merge-first path for adapter (LoRA/PEFT) repos: resolve the base from
    adapter_config.json, refuse what stock export could not convert honestly
    (gated / remote-code / pre-quantized BASE), else merge_and_unload into a
    temporary hub-format dir that the stock export takes as its model arg.

    A full model.safetensors sitting in an adapter repo is deliberately
    ignored: measured 2026-08-24, such a dump can be a different training
    checkpoint than the published adapter weights.

    Returns (merged_dir, base_info). Tokenizer comes from the adapter repo
    when it ships tokenizer_config.json, else from the base; if the adapter's
    tokenizer carries no chat template, the base's template is inherited."""
    from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError

    ac_text = hf_fetch(f"https://huggingface.co/{model_id}/raw/main/adapter_config.json")
    if not ac_text:
        fail("could not fetch adapter_config.json")
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
               "accept the base model's terms on huggingface.co and retry with a token; "
               "not converted on a guess")
    except RepositoryNotFoundError:
        refuse(report, "adapter_base_missing",
               f"base model {base_id} not found on the Hub",
               "the adapter's base pointer is dangling; merge manually from a real base")
    if getattr(base_info, "gated", False):
        refuse(report, "adapter_base_gated",
               f"base model {base_id} is gated (gated={base_info.gated!r})",
               "accept the base model's terms on huggingface.co and retry with a token")

    base_cfg_text = hf_fetch(f"https://huggingface.co/{base_id}/raw/main/config.json")
    if not base_cfg_text:
        fail(f"could not fetch base config.json for {base_id}")
    base_cfg = json.loads(base_cfg_text)
    if base_cfg.get("auto_map"):
        refuse(report, "remote_code",
               f"base {base_id} declares auto_map ({list(base_cfg['auto_map'])}) — the "
               "architecture lives in repo code, outside transformers",
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

    from huggingface_hub import snapshot_download

    print(f"adapter repo: merging into base {base_id} (merge-first, in a subprocess)")
    base_dir = Path(snapshot_download(base_id))
    adapter_dir = Path(snapshot_download(model_id, allow_patterns=[
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
             "if the traceback above is an ImportError: pip install peft")
    merge_info = json.loads(info_path.read_text())
    info_path.unlink()

    report["decisions"]["adapter"] = {
        "base": base_id, **merge_info, "merged_dir": str(merged_dir),
    }
    return merged_dir, base_info, gate_backend


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", help="HF model id, e.g. org/name")
    ap.add_argument("--out", help="output dir (default: out/<name>)")
    ap.add_argument("--int4", action="store_true",
                    help="quantize with the proven int4 recipe instead of default int8")
    ap.add_argument("--min-correct", type=int, default=6, help="exit-gate bar (of 8)")
    ap.add_argument("--keep-merged", action="store_true",
                    help="keep the temporary merged dir of an adapter conversion")
    ap.add_argument("--gate-script",
                    help="task-model gate: run this script (args: bundle --backend B "
                         "--litert-lm CLI --out JSON; exit 0 = pass) instead of the "
                         "generic 8-question gate — for finetunes that transform their "
                         "input rather than answer questions (s1-mini/Tashkeel), where "
                         "the generic gate certifies nothing")
    args = ap.parse_args()

    out_dir = Path(args.out) if args.out else REPO_ROOT / "out" / args.model.split("/")[-1]
    report = {
        "model": args.model,
        "output_dir": str(out_dir),
        "status": "started",
        "decisions": {},
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    # ---------------- entry gate ----------------
    from huggingface_hub import HfApi
    from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError
    api = HfApi()
    try:
        info = api.model_info(args.model, files_metadata=False)
    except GatedRepoError:
        refuse(report, "gated",
               "the repo requires accepting a license/terms before download",
               "accept the terms on huggingface.co and retry with a token; not converted on a guess")
    except RepositoryNotFoundError:
        fail(f"model not found on the Hub: {args.model}")

    if getattr(info, "gated", False):
        refuse(report, "gated",
               f"gated={info.gated!r} — weights are not openly downloadable",
               "accept the terms on huggingface.co and retry with a token")

    files = {s.rfilename for s in (info.siblings or [])}
    merged_dir = base_info = gate_backend = None
    if "adapter_config.json" in files:
        merged_dir, base_info, gate_backend = merge_adapter_first(
            report, args.model, files, out_dir, api)
    elif not any(f.endswith(".safetensors") or f.endswith(".bin") for f in files):
        refuse(report, "no_weights",
               "no safetensors/bin weight files in the repo",
               "point at a repo with full PyTorch weights")

    if merged_dir is None:  # adapter path already vetted the BASE's config
        cfg_text = hf_fetch(f"https://huggingface.co/{args.model}/raw/main/config.json")
        if not cfg_text:
            fail("could not fetch config.json")
        cfg = json.loads(cfg_text)
        if cfg.get("auto_map"):
            refuse(report, "remote_code",
                   f"config.json declares auto_map ({list(cfg['auto_map'])}) — the architecture "
                   "lives in repo code, outside transformers",
                   "not converted: remote-code architectures are out of scope for the stock exporter")
        if cfg.get("quantization_config"):
            refuse(report, "pre_quantized",
                   f"weights are already quantized ({cfg['quantization_config'].get('quant_method', '?')})",
                   "convert from the full-precision source repo instead")
        gate_backend = hybrid_gate_backend(report, cfg)
        if cfg.get("model_type") == "zamba2":
            zamba2_preport_guard(report, cfg)

    # ---------------- opt-ins ----------------
    params = getattr(getattr(base_info or info, "safetensors", None), "total", None)
    externalize = bool(params and params >= 3e9)
    report["decisions"]["params"] = params
    report["decisions"]["externalize_embedder"] = externalize
    if params is None:
        print("NOTE: param count unavailable from the Hub; not externalizing the embedder")

    export_kwargs = {}
    if externalize:
        export_kwargs["externalize_embedder"] = True
    if args.int4:
        export_kwargs["quantization_recipe"] = str(INT4_RECIPE)
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

    # ---------------- template lint (warn only) ----------------
    template = get_chat_template(args.model)
    if template is None and base_info is not None:
        template = get_chat_template(report["decisions"]["adapter_base"])
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

    # ---------------- export (stock, or family recipe) ----------------
    export_src = str(merged_dir) if merged_dir else args.model
    recipe = HYBRID_RECIPE.get(report["decisions"].get("model_type"))
    if recipe:
        gate_backend = "cpu"
        report["decisions"]["gate_backend"] = "cpu"
        bundle = convert_via_recipe(report, recipe, export_src, out_dir, args)
        report["bundle_bytes"] = bundle.stat().st_size
        ensure_turn_end_stop(bundle, export_src, report)
    else:
        print(f"exporting {export_src} -> {out_dir}  (stock defaults"
              + (f", opts={export_kwargs}" if export_kwargs else "") + ")")
        from litert_torch.generative.export_hf.export import export
        t0 = time.time()
        export(model=export_src, output_dir=str(out_dir), **export_kwargs)
        report["export_seconds"] = round(time.time() - t0)
        bundle = out_dir / "model.litertlm"
        if not bundle.exists():
            report["status"] = "export_failed"
            write_report(report)
            fail("export finished without producing model.litertlm")
        report["bundle_bytes"] = bundle.stat().st_size
        ensure_turn_end_stop(bundle, export_src, report)
        ensure_no_spurious_start_token(bundle, export_src, report)
        if report["decisions"].get("model_type") in HYBRID_EXEC_RETROFIT:
            ensure_executor_metadata(bundle, report)

    # ---------------- exit gate ----------------
    gate_json = out_dir / "gate.json"
    if args.gate_script:
        print(f"exit gate: task gate {args.gate_script}"
              + (f" --backend {gate_backend}" if gate_backend else ""))
        gate_cmd = [sys.executable, args.gate_script, str(bundle),
                    "--backend", gate_backend or "cpu", "--out", str(gate_json)]
        cli = resolve_litert_lm_cli()
        if cli:
            gate_cmd += ["--litert-lm", cli]
        proc = subprocess.run(gate_cmd, text=True)
        gate = json.loads(gate_json.read_text()) if gate_json.exists() else None
        report["gate"] = {"script": args.gate_script, "passed": proc.returncode == 0}
        if gate:
            report["gate"].update({k: gate[k] for k in ("match_hf", "n") if k in gate})
    else:
        max_tokens = 3200 if is_think else 512
        print(f"exit gate: verify_quality --max-tokens {max_tokens} --min-correct {args.min_correct}"
              + (f" --backend {gate_backend}" if gate_backend else ""))
        gate_cmd = [sys.executable, str(VERIFY), str(bundle),
                    "--max-tokens", str(max_tokens), "--min-correct", str(args.min_correct),
                    "--json", str(gate_json)]
        if gate_backend:
            gate_cmd += ["--backend", gate_backend]
        proc = subprocess.run(gate_cmd, text=True)
        gate = json.loads(gate_json.read_text()) if gate_json.exists() else None
        report["gate"] = {k: gate[k] for k in ("score", "of", "degenerate", "passed")} if gate else None
    report["status"] = "converted_pass" if proc.returncode == 0 else "converted_gate_failed"
    if merged_dir and not args.keep_merged:
        shutil.rmtree(merged_dir, ignore_errors=True)
        print(f"removed temporary merged dir {merged_dir} (--keep-merged retains it)")
    write_report(report)
    print(f"{report['status']}: {bundle} ({report['bundle_bytes']:,} B); "
          f"report: {out_dir / 'convert_report.json'}")
    sys.exit(0 if proc.returncode == 0 else 1)


if __name__ == "__main__":
    main()
