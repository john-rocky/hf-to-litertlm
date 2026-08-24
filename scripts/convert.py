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
     cannot convert honestly (adapter-only repos, gated repos, remote-code
     architectures, pre-quantized weights); never convert on a guess;
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", help="HF model id, e.g. org/name")
    ap.add_argument("--out", help="output dir (default: out/<name>)")
    ap.add_argument("--int4", action="store_true",
                    help="quantize with the proven int4 recipe instead of default int8")
    ap.add_argument("--min-correct", type=int, default=6, help="exit-gate bar (of 8)")
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
    if "adapter_config.json" in files:
        refuse(report, "adapter_only",
               "the repo holds a LoRA/PEFT adapter, not full weights",
               "merge the adapter into its base model first (peft merge_and_unload), "
               "then convert the merged full-weight repo")
    if not any(f.endswith(".safetensors") or f.endswith(".bin") for f in files):
        refuse(report, "no_weights",
               "no safetensors/bin weight files in the repo",
               "point at a repo with full PyTorch weights")

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
    params = getattr(getattr(info, "safetensors", None), "total", None)
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
    print(f"exporting {args.model} -> {out_dir}  (stock defaults"
          + (f", opts={export_kwargs}" if export_kwargs else "") + ")")
    from litert_torch.generative.export_hf.export import export
    t0 = time.time()
    export(model=args.model, output_dir=str(out_dir), **export_kwargs)
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
    write_report(report)
    print(f"{report['status']}: {bundle} ({report['bundle_bytes']:,} B); "
          f"report: {out_dir / 'convert_report.json'}")
    sys.exit(0 if proc.returncode == 0 else 1)


if __name__ == "__main__":
    main()
