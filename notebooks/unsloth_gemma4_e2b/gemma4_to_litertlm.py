#!/usr/bin/env python3
"""Turn a merged Gemma 4 E2B finetune into a .litertlm file for LiteRT-LM, and try it.

    python gemma4_to_litertlm.py versions REQUIREMENTS.txt
    python gemma4_to_litertlm.py export MODEL_DIR BF16.litertlm [--work DIR]
    python gemma4_to_litertlm.py fix BF16.litertlm OUT.litertlm
    python gemma4_to_litertlm.py ask OUT.litertlm [PROMPT]

Run it with the Python of the environment requirements.txt lists (the notebook makes one with venv), not the
training environment.

versions: compare the installed packages with the pins of REQUIREMENTS.txt; exit 1 on any difference.
export:   litert-torch's Hugging Face export of MODEL_DIR (the folder save_pretrained_merged writes): text only,
          int4 weights (blocks of 128, the embedding tables per channel), prefill signatures of 1024 and 128
          tokens, a 2048-token KV cache, the embedders as their own sections, and the checkpoint read in bfloat16
          straight from the safetensors (use_v2, delete_in_memory_params, model_dtype="bfloat16"), which keeps
          the peak memory of the export down. Two fixes on the way:
          - the per-layer embedder returns its output uncast, so in a bfloat16 export it stays bfloat16 and
            LiteRT-LM refuses the file ("The output tensor from the Embedding model must be of type float32.");
            it is cast to float32 here, as litert-torch already does for the main embedder;
          - LiteRT-LM renders the chat template with minijinja on native values: every message content is a list
            of text parts and a map has no Python methods, so a template written for transformers can raise
            (`x.get('key')`) or print the list where the text belongs. The template is rendered that way and the
            way transformers renders it, on five conversations, and edited only as far as every render needs to
            match: `x.get('key')` becomes `x['key']`, then, if that is not enough, every content read goes
            through LiteRT-LM's format_content macro. A template these edits do not fix is refused.
fix:      bf16to32.py: the BFLOAT16 tensors the bfloat16 read leaves in the graph (KV cache, composite inputs and
          outputs, some constants) become FLOAT32 and the casts that turn into no-ops are removed, because
          LiteRT-LM's GPU delegate does not take BFLOAT16 tensors; then the result is checked against its input.
ask:      load the file on the CPU and answer one prompt greedily.
"""
import argparse
import hashlib
import json
import os
import re
import resource
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent

# int4 weights: symmetric, blocks of 128; the embedding lookups int4 per channel
RECIPE = """[
 {
  "regex": ".*",
  "operation": "*",
  "algorithm_key": "min_max_uniform_quantize",
  "op_config": {
   "weight_tensor_config": {
    "num_bits": 4,
    "symmetric": true,
    "granularity": "BLOCKWISE_128",
    "dtype": "INT"
   },
   "compute_precision": "INTEGER",
   "explicit_dequantize": false,
   "skip_checks": false,
   "min_weight_elements": 0
  }
 },
 {
  "regex": ".*",
  "operation": "EMBEDDING_LOOKUP",
  "algorithm_key": "min_max_uniform_quantize",
  "op_config": {
   "weight_tensor_config": {
    "num_bits": 4,
    "symmetric": true,
    "granularity": "CHANNELWISE",
    "dtype": "INT"
   },
   "compute_precision": "INTEGER",
   "explicit_dequantize": false,
   "skip_checks": false,
   "min_weight_elements": 0
  }
 }
]"""

EXPORT_ARGS = dict(
    prefill_lengths=[1024, 128], cache_length=2048, externalize_embedder=True,
    use_v2=True, delete_in_memory_params=True, model_dtype="bfloat16",
    # the GPU graph options of the tested form: every composite and fusion off
    apply_gpu_composites=True, use_qkv_norm_rope_composite=False, use_sdpa_composite=False, fuse_gate_up=False,
    fuse_qkv=False, use_rope_composite=False, use_swiglu_composite=False, use_short_conv_composite=False,
    use_bool_mask=False, enable_gpu_dynamic_cache=False)

E2B = {"hidden_size": 1536, "num_hidden_layers": 35}   # gemma-4-E2B's text_config


def fail(msg, code=1):
    print(f"ERROR: {msg}", file=sys.stderr, flush=True)
    sys.exit(code)


def peak_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 / 1e9   # Linux: kB


def norm(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def versions(req):
    import importlib.metadata as md
    want = {}
    for line in open(req, encoding="utf-8"):
        m = re.match(r"\s*([A-Za-z0-9][A-Za-z0-9_.\-]*)(?:\[[^\]]*\])?\s*==\s*([^\s;#]+)", line)
        if m:
            want[norm(m.group(1))] = m.group(2)
    have = {norm(d.metadata["Name"]): d.version for d in md.distributions() if d.metadata["Name"]}
    bad = [(n, v, have.get(n)) for n, v in want.items() if have.get(n) != v]
    for n, v, h in bad:
        print(f"  {n}: requirements.txt pins {v}, installed {h or 'nothing'}")
    print(f"versions: {len(want) - len(bad)} of {len(want)} pinned packages as pinned", flush=True)
    return 1 if bad else 0


# The chat template check: the proof of scripts/chat_template_dual.py in github.com/john-rocky/hf-to-litertlm, with
# the runtime renderer of its tools/chat_template_prefix/desk_render.py. The runtime's render = LiteRT-LM 0.18.0's
# (runtime/components/prompt_template.cc, runtime/components/rust/minijinja_template.rs): the template's Python
# idioms rewritten, minijinja 2.14.0 keeping trailing newlines and trimming / lstripping blocks, its filters, a
# `none` test that is true for an undefined value only, bos_token = eos_token = "", every content a list of text
# parts. The reference = transformers' own render of the original template, with the same bos / eos.

# LiteRT-LM a8178d7d, models/qwen3/chat_template.jinja (Apache-2.0): the macro, byte for byte.
FORMAT_CONTENT = """{%- macro format_content(content) -%}
    {%- if content is string -%}
        {{- content -}}
    {%- elif content is sequence -%}
        {%- for item in content -%}
            {%- if item['type'] == 'text' -%}
                {{- item['text'] -}}
            {%- endif -%}
        {%- endfor -%}
    {%- endif -%}
{%- endmacro -%}"""
MACRO_SOURCE = "LiteRT-LM a8178d7d models/qwen3/chat_template.jinja (format_content, Apache-2.0)"

# prompt_template.cc v0.18.0 l.42-72, in order (RE2::GlobalReplace == re.sub for these patterns)
MINIJINJA_EDITS = (
    (r"\.startswith\((.*?)\)", r" is startingwith \1"),
    (r"\.endswith\((.*?)\)", r" is endingwith \1"),
    (r"\.replace\((.*?),(.*?)\)", r" | replace(\1,\2)"),
    (r"\.split\((.*?)\)\[0\]", r" | split(\1) | first"),
    (r"\.split\((.*?)\)\[-1\]", r" | split(\1) | last"),
    (r"\.split\((.*?)\)", r" | split(\1)"),
    (r"\.join\((.*?)\)", r" | join(\1)"),
    (r"\.lstrip\(\)", " | lstrip"),
    (r"\.lstrip\((.*?)\)", r" | lstrip(\1)"),
    (r"\.rstrip\(\)", " | rstrip"),
    (r"\.rstrip\((.*?)\)", r" | rstrip(\1)"),
    (r"\.strip\(\)", " | trim"),
    (r"\.strip\((.*?)\)", r" | trim(\1)"),
    (r"\.items\(\)", " | items"),
    (r"{% generation %}", ""),
    (r"{% endgeneration %}", ""),
)
NONE_TEST = re.compile(r"\bis(\s+not)?\s+none\b")
NOW_TS = 1767225600   # 2026-01-01T00:00:00Z: a fixed `now`, so strftime_now renders the same every time

QUOTED = r"'(?:\\.|[^'\\])*'|\"(?:\\.|[^\"\\])*\""
ACCESS = r"\b(?:messages\[0\]|message)(?:\.content|\[\s*['\"]content['\"]\s*\])"
GET = (r"(?P<obj>\b[A-Za-z_]\w*(?:\.[A-Za-z_]\w*|\[[^\[\]]*\])*)\.get\(\s*(?P<key>'[^'\\]*'|\"[^\"\\]*\")\s*\)"
       r"(?P<none>\s+is\s+(?:not\s+)?none\b)?")
# a content read of any name, for naming what a failed proof left unwrapped
ANY_ACCESS = re.compile(r"\b[A-Za-z_][\w\[\]0-9]*(?:\.content|\[\s*['\"]content['\"]\s*\])")

Q = "What is the capital of France?"
SYS = "You are a concise, helpful assistant. Preserve system marker SYS_AUDIT_314159."
ANS = "Paris is the capital of France. ASSISTANT_AUDIT_271828."
Q2 = "Which country is Paris in? USER_SECOND_161803."
LEAK = re.compile(r"\[\s*\{\s*['\"](?:type|text)['\"]\s*:")

EDITS = {"get->index": "x.get('key') written as x['key']",
         "get->defined-or-none": "x.get('key') before an `is none` test written as "
                                 "(x['key'] if x['key'] is defined else none)",
         "format_content": "content reads routed through LiteRT-LM's format_content macro"}


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def edit_expressions(source, pattern, replace):
    """Apply `replace` to the matches of `pattern` inside {{ }} / {% %} only, never inside a comment, a string
    literal or the text between tags. Returns (text, count)."""
    token = re.compile(pattern + "|" + QUOTED)
    pieces, pos, count = [], 0, 0
    while pos < len(source):
        start = re.search(r"\{[{%#]", source[pos:])
        if start is None:
            pieces.append(source[pos:])
            break
        a = pos + start.start()
        pieces.append(source[pos:a])
        opener = source[a:a + 2]
        closer = {"{{": "}}", "{%": "%}", "{#": "#}"}[opener]
        quote, escaped, b = None, False, a + 2
        while b < len(source):
            char = source[b]
            if escaped:
                escaped = False
            elif quote:
                if char == "\\":
                    escaped = True
                elif char == quote:
                    quote = None
            elif opener != "{#" and char in "'\"":
                quote = char
            elif source.startswith(closer, b):
                break
            b += 1
        body = source[a + 2:b]

        def sub(match):
            nonlocal count
            if match.group(0).startswith(('"', "'")):
                return match.group(0)
            count += 1
            return replace(match)

        pieces.append(opener + (body if opener == "{#" else token.sub(sub, body)) + source[b:b + 2])
        pos = b + 2
    return "".join(pieces), count


def rewrite_get(source):
    """`X.get('k')` -> `X['k']`; before an `is none` test, the defined-or-none form."""
    counts = {"get->index": 0, "get->defined-or-none": 0}

    def rep(m):
        index = f"{m.group('obj')}[{m.group('key')}]"
        if m.group("none"):
            counts["get->defined-or-none"] += 1
            return f"({index} if {index} is defined else none){m.group('none')}"
        counts["get->index"] += 1
        return index

    text, _ = edit_expressions(source, GET, rep)
    return text, {k: v for k, v in counts.items() if v}


def wrap_content_reads(source):
    """The dual form: every direct content read through format_content, the macro in front."""
    body, n = edit_expressions(source, ACCESS, lambda m: "format_content(" + m.group(0) + ")")
    return FORMAT_CONTENT + "\n" + body, n


def template_cases(source):
    c = [("c1", [("user", Q)], {}),
         ("c2", [("system", SYS), ("user", Q)], {}),
         ("c3", [("user", Q), ("assistant", ANS), ("user", Q2)], {})]
    if "enable_thinking" in source:
        c += [("c4_true", [("user", Q)], {"enable_thinking": True}),
              ("c4_false", [("user", Q)], {"enable_thinking": False})]
    return c


def as_messages(turns, form):
    return [{"role": r, "content": t if form == "string" else [{"type": "text", "text": t}]} for r, t in turns]


def literal(value):
    """A minijinja literal for a JSON-like value (native maps and lists: no Python methods)."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "none"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, list):
        return "[" + ", ".join(literal(v) for v in value) + "]"
    if isinstance(value, dict):
        return "{" + ", ".join(f"{json.dumps(k)}: {literal(v)}" for k, v in value.items()) + "}"
    raise TypeError(type(value).__name__)


def _tojson(value, *args, **kwargs):   # the runtime's tojson: ", " / ": " spacing, no extra arguments
    import jinja2
    if args or kwargs:
        raise jinja2.TemplateError("tojson: too many arguments")
    if isinstance(value, jinja2.Undefined):
        raise jinja2.TemplateError("tojson: undefined value")
    return json.dumps(value, ensure_ascii=False, separators=(", ", ": "))


def _lstrip(s, chars=None):
    return str(s).lstrip(chars)


def _rstrip(s, chars=None):
    return str(s).rstrip(chars)


def _raise_exception(msg):
    import jinja2
    raise jinja2.TemplateError(str(msg))


def _strftime_now(fmt):
    import datetime
    return datetime.datetime.fromtimestamp(NOW_TS, datetime.timezone.utc).strftime(fmt)


def _result(fn):
    try:
        return {"ok": True, "text": fn()}
    except Exception as e:  # a template error is a result here
        return {"ok": False, "error": f"{type(e).__name__}: {str(e).splitlines()[0][:200]}"}


def runtime_source(source):
    """The runtime's rewrites of Python idioms, then `is none` / `is not none` as `is undefined` /
    `is not undefined` (minijinja's Python binding hands undefined and none to a Python test as the same None, and
    the runtime's `none` test is is_undefined)."""
    for pat, rep in MINIJINJA_EDITS:
        source = re.sub(pat, rep, source)
    return NONE_TEST.sub(lambda m: "is" + (m.group(1) or "") + " undefined", source)


def runtime_render(source, turns, form, extra):
    """The runtime's way: its edits of the source, its environment, the messages as literals at the head."""
    def go():
        import minijinja
        body = runtime_source(source)
        env = minijinja.Environment()
        env.pycompat = False
        env.trim_blocks = True
        env.lstrip_blocks = True
        env.keep_trailing_newline = True
        env.add_function("raise_exception", _raise_exception)
        env.add_function("strftime_now", _strftime_now)
        env.add_filter("tojson", _tojson)
        env.add_filter("lstrip", _lstrip)
        env.add_filter("rstrip", _rstrip)
        env.add_template("t", "{% set messages = " + literal(as_messages(turns, form)) + " %}" + body)
        return env.render_template("t", add_generation_prompt=True, now=NOW_TS, bos_token="", eos_token="",
                                   **extra)
    return _result(go)


def reference_render(source, turns, form, extra):
    """transformers' own render of the template (jinja2, Python methods), bos / eos bound to ""."""
    from transformers.utils.chat_template_utils import render_jinja_template

    def go():
        out = render_jinja_template(conversations=[as_messages(turns, form)], chat_template=source,
                                    add_generation_prompt=True, bos_token="", eos_token="", **extra)
        rendered = out[0] if isinstance(out, tuple) else out
        return rendered[0]
    return _result(go)


def same(a, b):
    return (a["ok"] and b["ok"] and a["text"] == b["text"]) or (
        not a["ok"] and not b["ok"] and a["error"] == b["error"])


def classify(turns, ref, got):
    """equal / drop / raise / leak / differs: what the runtime render does to the reference."""
    if same(ref, got):
        return "equal"
    if not got["ok"]:
        return "raise"
    if LEAK.search(got["text"]):
        return "leak"
    if ref["ok"] and any(t in ref["text"] and t not in got["text"] for _, t in turns):
        return "drop"
    return "differs"


def retains(turns, r):
    return r["ok"] and not LEAK.search(r["text"]) and all(t in r["text"] for _, t in turns)


def _h(r):
    return {"sha256": sha(r["text"]), "bytes": len(r["text"].encode())} if r["ok"] else {"error": r["error"]}


def prove_template(source):
    """(record, final template). record: form, edits, per-case classes, basis and hashes, pass."""
    import importlib.metadata
    record = {"original_sha256": sha(source), "form": None, "edits": {}, "final_sha256": None,
              "oracle": f"minijinja {importlib.metadata.version('minijinja')} (runtime settings, "
                        "literal messages) against transformers render_jinja_template",
              "macro": MACRO_SOURCE, "macro_sha256": sha(FORMAT_CONTENT),
              "classes": {}, "basis": {}, "cases": {}, "pass": False, "reason": None, "unwrapped": None}
    all_cases = template_cases(source)
    ref = {n: {f: reference_render(source, t, f, x) for f in ("string", "parts")} for n, t, x in all_cases}
    record["classes"] = {n: classify(t, ref[n]["string"], runtime_render(source, t, "parts", x))
                         for n, t, x in all_cases}
    # each case is compared with transformers' string render, or with its parts render when the template renders
    # parts itself (every text kept) and the string render differs from it
    basis = {n: ("string" if same(ref[n]["string"], ref[n]["parts"]) or not retains(t, ref[n]["parts"])
                 else "parts") for n, t, x in all_cases}

    def matches(text):
        out = {}
        for n, t, x in all_cases:
            got_p = runtime_render(text, t, "parts", x)
            want = ref[n][basis[n]]
            out[n] = {"basis": basis[n], "reference": _h(want), "runtime_parts": _h(got_p),
                      "equal": same(want, got_p) and want["ok"]}
        return out

    form, final = [], source
    checked = matches(final)
    if not all(c["equal"] for c in checked.values()):
        trial = [runtime_render(final, t, "parts", x) for n, t, x in all_cases]
        if any(not r["ok"] and "no method named get" in r["error"] for r in trial):
            final, counts = rewrite_get(final)
            record["edits"].update(counts)
            form.append("get")
            checked = matches(final)
    if not all(c["equal"] for c in checked.values()):
        final, n = wrap_content_reads(final)
        if n:
            record["edits"]["format_content"] = n
            form.append("dual")
            basis = {k: "string" for k in basis}  # the dual form gives the string prompt
            checked = matches(final)
            for name, t, x in all_cases:  # and the string form still renders the same
                got_s = runtime_render(final, t, "string", x)
                checked[name]["runtime_string"] = _h(got_s)
                checked[name]["equal"] = checked[name]["equal"] and same(ref[name]["string"], got_s)
    record["form"] = "+".join(form) or "unchanged"
    record["basis"] = {n: c["basis"] for n, c in checked.items()}
    record["cases"] = checked
    record["final_sha256"] = sha(final)
    record["pass"] = all(c["equal"] for c in checked.values())
    if not record["pass"]:
        bad = [n for n, c in checked.items() if not c["equal"]]
        unwrapped = [m.group(0) for m in ANY_ACCESS.finditer(final)
                     if not final[max(0, m.start() - 15):m.start()].endswith("format_content(")]
        record["unwrapped"] = unwrapped[0] if unwrapped else None
        record["reason"] = (f"the runtime's render differs from transformers' in {', '.join(bad)}"
                            + (f"; content read left unwrapped: {unwrapped[0]}" if unwrapped else ""))
    return record, final


def read_template(model_dir):
    path = model_dir / "chat_template.jinja"
    if path.is_file():
        return path.read_text(encoding="utf-8")
    cfg = model_dir / "tokenizer_config.json"
    tmpl = json.loads(cfg.read_text(encoding="utf-8")).get("chat_template") if cfg.is_file() else None
    return tmpl if isinstance(tmpl, str) and tmpl.strip() else None


def export(model_dir, out, work):
    model_dir, out, work = Path(model_dir), Path(out), Path(work)
    if not (model_dir / "config.json").is_file():
        fail(f"{model_dir} has no config.json: save the merged model there first (save_pretrained_merged)")
    if not list(model_dir.glob("*.safetensors")):
        fail(f"{model_dir} has no .safetensors weights: save the merged 16-bit model there (save_pretrained_merged)")
    cfg = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    tc = cfg.get("text_config") or {}
    if cfg.get("model_type") != "gemma4" or any(tc.get(k) != v for k, v in E2B.items()):
        fail(f"{model_dir} is model_type {cfg.get('model_type')!r} with text_config "
             f"{ {k: tc.get(k) for k in E2B} }; this export is made and tested for Gemma 4 E2B {E2B}")
    tmpl = read_template(model_dir)
    if tmpl is None:
        fail(f"{model_dir} has no chat template (chat_template.jinja or tokenizer_config.json)")
    proof, fixed = prove_template(tmpl)
    if not proof["pass"]:
        left = " It still calls .get(...) in a form this script does not rewrite." if ".get(" in fixed else ""
        fail("the chat template does not render on LiteRT-LM as transformers renders it, and the edits this "
             f"script makes do not fix that: {proof['reason']}.{left}")
    work.mkdir(parents=True, exist_ok=True)
    (work / "chat_template.jinja").write_text(fixed, encoding="utf-8")
    (work / "chat_template_proof.json").write_text(json.dumps(proof, indent=1) + "\n", encoding="utf-8")
    (work / "recipe.json").write_text(RECIPE, encoding="utf-8")
    edits = "; ".join(f"{n} {EDITS[k]}" for k, n in proof["edits"].items()) or "no edit needed"
    print(f"chat template: {edits}; LiteRT-LM's render matches transformers' on all {len(proof['cases'])} test "
          f"conversations -> {work / 'chat_template.jinja'}", flush=True)

    import torch
    from litert_torch.generative.export_hf.export import export as hf_export
    from litert_torch.generative.export_hf.model_ext.gemma4 import exportable_module as g4
    cls = g4.LiteRTExportableModuleForPerLayerEmbedder
    orig = cls.forward

    def forward(self, token_ids):   # the per-layer embedder's output in float32, as the main embedder's
        return {k: v.to(torch.float32) for k, v in orig(self, token_ids).items()}
    cls.forward = forward

    export_dir = work / "export"
    shutil.rmtree(export_dir, ignore_errors=True)
    t0 = time.time()
    hf_export(model=str(model_dir), output_dir=str(export_dir), quantization_recipe=str(work / "recipe.json"),
              jinja_chat_template_override=str(work / "chat_template.jinja"), **EXPORT_ARGS)
    made = export_dir / "model.litertlm"
    if not made.is_file():
        fail(f"the export finished without {made}")
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(made), str(out))
    shutil.rmtree(export_dir, ignore_errors=True)
    print(f"export: {out} {out.stat().st_size:,} B in {time.time() - t0:.0f} s, peak memory {peak_gb():.1f} GB",
          flush=True)


def fix(src, dst):
    sys.path.insert(0, str(HERE))
    import bf16to32
    if not Path(src).is_file():
        fail(f"{src} does not exist: run the export first")
    with open(src, "rb") as f:
        if f.read(8) != b"LITERTLM":
            fail(f"{src} is not a .litertlm file")
    _, sections = bf16to32.tflite_sections(src)
    n = {i: sum(m.Subgraphs(s).Tensors(t).Type() == bf16to32.BF16 for s in range(m.SubgraphsLength())
                for t in range(m.Subgraphs(s).TensorsLength())) for i, m in sections.items()}
    del sections
    if not any(n.values()):
        fail(f"{src} has no BFLOAT16 tensors (TFLite sections {sorted(n)}): it is not the bfloat16 export this "
             "step rewrites, so there is nothing to do", code=2)
    print(f"BFLOAT16 tensors per TFLite section: {n}", flush=True)
    t0 = time.time()
    bf16to32.main(str(src), str(dst), str(Path(dst).with_suffix(".bf16to32.json")))
    if bf16to32.check(str(dst), str(src)) != 0:
        fail(f"{dst} did not pass the check against {src} (see the lines above)")
    h = hashlib.sha256()
    with open(dst, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    print(f"fix: {dst} {os.path.getsize(dst):,} B sha256 {h.hexdigest()} in {time.time() - t0:.0f} s", flush=True)


def ask(bundle, prompt):
    import tempfile
    import litert_lm
    t0 = time.time()
    with tempfile.TemporaryDirectory() as cache:
        engine = litert_lm.Engine(str(bundle), backend=litert_lm.Backend.CPU(), cache_dir=cache)
        conv = engine.create_conversation(
            sampler_config=litert_lm.SamplerConfig(top_k=1, top_p=1.0, temperature=0.0), max_output_tokens=64)
        t1 = time.time()
        reply = conv.send_message({"role": "user", "content": prompt})
        t2 = time.time()
        del conv, engine
    print(f"Q: {prompt}\nA: {str(reply).strip()}\n(load {t1 - t0:.1f} s, answer {t2 - t1:.1f} s, CPU)", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("versions").add_argument("requirements")
    p = sub.add_parser("export")
    p.add_argument("model_dir")
    p.add_argument("out")
    p.add_argument("--work", default="litertlm_work")
    p = sub.add_parser("fix")
    p.add_argument("src")
    p.add_argument("dst")
    p = sub.add_parser("ask")
    p.add_argument("bundle")
    p.add_argument("prompt", nargs="?", default="What is 17 + 25?")
    a = ap.parse_args()
    if a.cmd == "versions":
        return versions(a.requirements)
    if a.cmd == "export":
        return export(a.model_dir, a.out, a.work)
    if a.cmd == "fix":
        return fix(a.src, a.dst)
    return ask(a.bundle, a.prompt)


if __name__ == "__main__":
    sys.exit(main())
