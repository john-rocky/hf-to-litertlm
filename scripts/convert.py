#!/usr/bin/env python3
"""Thin gated wrapper around the stock litert-torch export.

    ~/venvs/ltconv040dev/bin/python scripts/convert.py <hf_model_id> [--out DIR]
                                                       [--int4] [--min-correct N]

Measured basis (2026-08-24): stock `export(model, output_dir)` with no other
arguments converts untouched finetunes correctly on both the model_ext path
(qwen3: QVikhr-3-1.7B) and the generic path (smollm3: a SmolLM3-3B finetune) —
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
     pre-2026 runtimes may die on the first message.

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
    return merged_dir, base_info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", help="HF model id, e.g. org/name")
    ap.add_argument("--out", help="output dir (default: out/<name>)")
    ap.add_argument("--int4", action="store_true",
                    help="quantize with the proven int4 recipe instead of default int8")
    ap.add_argument("--min-correct", type=int, default=6, help="exit-gate bar (of 8)")
    ap.add_argument("--keep-merged", action="store_true",
                    help="keep the temporary merged dir of an adapter conversion")
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
    merged_dir = base_info = None
    if "adapter_config.json" in files:
        merged_dir, base_info = merge_adapter_first(report, args.model, files, out_dir, api)
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

    # ---------------- stock export ----------------
    export_src = str(merged_dir) if merged_dir else args.model
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

    # ---------------- exit gate ----------------
    max_tokens = 3200 if is_think else 512
    gate_json = out_dir / "gate.json"
    print(f"exit gate: verify_quality --max-tokens {max_tokens} --min-correct {args.min_correct}")
    proc = subprocess.run(
        [sys.executable, str(VERIFY), str(bundle),
         "--max-tokens", str(max_tokens), "--min-correct", str(args.min_correct),
         "--json", str(gate_json)],
        text=True)
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
