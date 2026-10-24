#!/usr/bin/env python3
"""Round 2: assemble the WeVisOne-VL-2B fast_vlm `.litertlm` (shape = decider2bv_work/scripts/build_bundle_g256.py,
template form = minicpmv47_work/scripts/build_mv47_bundle.py's dual jinja).

  EMBEDDER       : token_ids [1, 1] -> [1, 1, 8196]
  PREFILL_DECODE : embeddings [1, T, 8196] + input_pos + mask + 56 KV states (-> logits)   prefer_activation_type = arg
  VISION_ENCODER : image NHWC [1, 512, 512, 3] in [0, 1] -> [1, 1024, 4096]
  VISION_ADAPTER : [1, 1024, 4096] -> [1, 256, 8196]
  HF tokenizer   : the pinned snapshot's tokenizer.json, bytes as on the Hub (sha256 3bb17242...fc5e)
  LlmMetadata    : fast_vlm 512 x 512, max_num_tokens 2048 (= the export's KV cache), NO start token (bos_token null),
                   stop = <|im_end|> 151645 and <|endoftext|> 151643, STRUCTURED prompt_templates present and empty
                   (memory vlm-fastvlm-ride: jinja-only metadata gives a null conversation), the jinja below.

Template contract: the system turn is always "You are a helpful assistant." (a passed system message is
not rendered: the derived image positions start at P = 15); the image of the FIRST user turn, its first image item,
renders first as <|vision_start|><image_soft_token><|vision_end|> (the runtime splits on <image_soft_token> and
injects the adapter's 256 rows); any other image item is not rendered, and the runtime then refuses the request
("Provided more images than expected in the prompt", multimodal_processor_helper.cc v0.18.1) instead of placing it
at wrong positions; text items follow in order, concatenated (the checkpoint template's form); user content may be
a string or a list (0.18 content parts). Assistant turns close at <|im_end|> and the next turn opens with "\\n"
(the KV holds <|im_end|> as the pending token, not the "\\n": memory conversation-prefix-contract).

Modes:
  --check-template OUT.json  (reference venv: jinja2 + tokenizers) desk-render against ref/B (29 image rows),
                             ref/T0 (10 text rows, two_turn both turns), and the contract cases; ids = the pieces
                             between <image_soft_token> tokenized alone + 256 image slots per soft token
  --build                    (export venv: litert_lm_builder 0.18.1) write <out-dir>/<name>_noexec.litertlm + record

    K/.venv-ref/bin/python -I tools/build_wv_bundle.py --check-template r2/template_check.json
    ~/venvs/lt0100dev/bin/python -I tools/build_wv_bundle.py --build --name wv2b_wi8 --decoder ... --embedder ... \
        --encoder ... --adapter ... --act fp32
"""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

K = Path(__file__).resolve().parents[1]
SNAP = Path(os.environ.get("WEVISONE_SNAPSHOT", "~/.cache/huggingface/hub/models--tencent--WeVisOne-VL-2B/"
                           "snapshots/3dad9ce8bcb6fbe0aea945bf8860bd2cfc3663bf")).expanduser()
TOKENIZER = SNAP / "tokenizer.json"
TOKENIZER_SHA256 = "3bb172429b952469b51c64783f52fa01c26d1b30f1f31a4693e385a43f9cfc5e"
IMAGE_SIZE = 512
MAX_TOKENS = 2048
STOP_IDS = [151645, 151643]
SOFT = "<image_soft_token>"
N_IMAGE = 256
IMAGE_PAD = 151655
SYSTEM = "You are a helpful assistant."


def _lit(s):
    assert "'" not in s and "\\" not in s, s
    return "{{- '" + s.replace("\n", "\\n") + "' -}}"


TEXT_PARTS = ("{%- if message['content'] is string -%}{{- message['content'] -}}"
              "{%- else -%}{%- for item in message['content'] -%}"
              "{%- if item['type'] == 'text' -%}{{- item['text'] -}}{%- endif -%}"
              "{%- endfor -%}{%- endif -%}")

JINJA = (
    "{%- set ns = namespace(nl=false, first_user=true) -%}"
    + _lit("<|im_start|>system\n" + SYSTEM + "<|im_end|>\n") +
    "{%- for message in messages -%}"
    "{%- set role = message['role'] -%}"
    "{%- if role == 'user' or role == 'assistant' or role == 'model' -%}"
    "{%- if ns.nl -%}" + _lit("\n") + "{%- set ns.nl = false -%}{%- endif -%}"
    "{%- if role == 'user' -%}"
    + _lit("<|im_start|>user\n") +
    "{%- if ns.first_user and message['content'] is not string -%}"
    "{%- set img = namespace(done=false) -%}"
    "{%- for item in message['content'] -%}"
    "{%- if item['type'] == 'image' and not img.done -%}"
    + _lit("<|vision_start|>" + SOFT + "<|vision_end|>") +
    "{%- set img.done = true -%}"
    "{%- endif -%}"
    "{%- endfor -%}"
    "{%- endif -%}"
    + TEXT_PARTS + _lit("<|im_end|>\n") +
    "{%- set ns.first_user = false -%}"
    "{%- else -%}"
    + _lit("<|im_start|>assistant\n") + TEXT_PARTS + _lit("<|im_end|>") +
    "{%- set ns.nl = true -%}"
    "{%- endif -%}"
    "{%- endif -%}"
    "{%- endfor -%}"
    "{%- if add_generation_prompt -%}"
    "{%- if ns.nl -%}" + _lit("\n") + "{%- endif -%}"
    + _lit("<|im_start|>assistant\n") +
    "{%- endif -%}"
)


def sha256_file(p):
    with open(p, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


# ------------------------------------------------------------------------------------------------- template check
def check_template(out_path):
    import jinja2
    from tokenizers import Tokenizer
    assert sha256_file(TOKENIZER) == TOKENIZER_SHA256
    tk = Tokenizer.from_file(str(TOKENIZER))
    env = jinja2.Environment(undefined=jinja2.StrictUndefined)
    tmpl = env.from_string(JINJA)

    def ids_of(text):
        out = []
        pieces = text.split(SOFT)
        for k, piece in enumerate(pieces):
            out += tk.encode(piece, add_special_tokens=False).ids
            if k < len(pieces) - 1:
                out += [IMAGE_PAD] * N_IMAGE
        return out

    def render(msgs, gen=True):
        return tmpl.render(messages=msgs, add_generation_prompt=gen)

    rows, ok_all = [], True

    def case(name, msgs, want_ids=None, want_text=None, gen=True, prefix_of=None):
        nonlocal ok_all
        text = render(msgs, gen)
        ids = ids_of(text)
        rec = {"case": name, "render": text, "n_ids": len(ids), "n_soft": text.count(SOFT)}
        ok = True
        if want_ids is not None:
            rec["ids_equal"] = ids == want_ids
            if not rec["ids_equal"]:
                rec["first_div"] = next((i for i, (a, b) in enumerate(zip(ids, want_ids)) if a != b),
                                        min(len(ids), len(want_ids)))
            ok &= rec["ids_equal"]
        if want_text is not None:
            rec["text_equal"] = text == want_text
            ok &= rec["text_equal"]
        if prefix_of is not None:
            prev = render(prefix_of, gen=False)
            rec["prefix_contract_ok"] = text.startswith(prev)
            ok &= rec["prefix_contract_ok"]
        rec["ok"] = ok
        ok_all &= ok
        rows.append(rec)
        print(("ok " if ok else "NG ") + name, {k: v for k, v in rec.items() if k not in ("render", "case")},
              flush=True)
        return text

    fixtures = {r["id"]: r for r in json.loads((K / "fixtures" / "fixtures.json").read_text())["rows"]}
    for rid, row in fixtures.items():
        b = json.loads((K / "ref" / "B" / f"{rid}.json").read_text())
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": row["prompt"]}]}]
        case(f"B:{rid}", msgs, want_ids=b["prompt_ids"])
    for p in sorted((K / "ref" / "T0").glob("*.json")):
        t0 = json.loads(p.read_text())
        msgs = []
        for i, user in enumerate(t0["turns"]):
            msgs.append({"role": "user", "content": [{"type": "text", "text": user}]})
            want = t0["records"][i]["prompt_ids"]
            case(f"T0:{t0['row']}:turn{i + 1}", list(msgs), want_ids=want,
                 prefix_of=msgs[:-1] if i else None)
            msgs.append({"role": "assistant", "content": [{"type": "text", "text": t0["records"][i]["answer"]}]})
    q = "What is the capital of France? Answer in one word."
    base = render([{"role": "user", "content": q}])
    case("string_equals_parts", [{"role": "user", "content": [{"type": "text", "text": q}]}], want_text=base)
    case("system_ignored", [{"role": "system", "content": "Be terse."}, {"role": "user", "content": q}],
         want_text=base)
    case("system_parts_ignored", [{"role": "system", "content": [{"type": "text", "text": "Be terse."}]},
                                  {"role": "user", "content": [{"type": "text", "text": q}]}], want_text=base)
    one = render([{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Describe."}]}])
    case("text_before_image_item", [{"role": "user", "content": [{"type": "text", "text": "Describe."},
                                                                  {"type": "image"}]}], want_text=one)
    t = case("two_images_first_turn", [{"role": "user", "content": [{"type": "image"}, {"type": "image"},
                                                                     {"type": "text", "text": "Describe."}]}],
             want_text=one)
    hist = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Describe."}]},
            {"role": "assistant", "content": [{"type": "text", "text": "A red square."}]}]
    t2 = case("image_then_followup", hist + [{"role": "user", "content": [{"type": "text", "text": "Where?"}]}],
              prefix_of=hist)
    case("image_in_second_turn_not_rendered",
         hist + [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Where?"}]}],
         want_text=t2)
    res = {"jinja": JINJA, "jinja2": jinja2.__version__, "tokenizer_sha256": TOKENIZER_SHA256,
           "rule": "ids = render split on <image_soft_token>, each piece tokenized alone (add_special_tokens False), "
                   "256 image slots per soft token", "all_ok": ok_all, "cases": rows}
    Path(K / out_path).write_text(json.dumps(res, indent=1, ensure_ascii=False) + "\n")
    print("TEMPLATE_CHECK", "ALL_OK" if ok_all else "MISMATCH", out_path, flush=True)
    return 0 if ok_all else 1


# ------------------------------------------------------------------------------------------------- build
def metadata(image_size=IMAGE_SIZE):
    from litert_lm_builder.runtime.proto import llm_metadata_pb2, llm_model_type_pb2
    from tokenizers import Tokenizer
    tk = Tokenizer.from_file(str(TOKENIZER))
    assert [tk.id_to_token(i) for i in STOP_IDS] == ["<|im_end|>", "<|endoftext|>"]
    for tok, want in (("<|vision_start|>", 151652), ("<|vision_end|>", 151653), ("<|image_pad|>", IMAGE_PAD)):
        assert tk.token_to_id(tok) == want, (tok, tk.token_to_id(tok))
    assert tk.token_to_id(SOFT) is None
    md = llm_metadata_pb2.LlmMetadata()
    md.max_num_tokens = MAX_TOKENS
    for affix in (md.prompt_templates.user, md.prompt_templates.model, md.prompt_templates.system):
        affix.SetInParent()
    md.jinja_prompt_template = JINJA
    md.llm_model_type.CopyFrom(llm_model_type_pb2.LlmModelType(fast_vlm=llm_model_type_pb2.FastVlm()))
    md.llm_model_type.fast_vlm.image_tensor_height = image_size
    md.llm_model_type.fast_vlm.image_tensor_width = image_size
    for sid in STOP_IDS:
        md.stop_tokens.add().token_ids.ids.append(sid)
    assert not md.HasField("start_token")
    return md


def build(a):
    import importlib.metadata as imd
    import litert_lm_builder as lb
    from google.protobuf import text_format
    assert sha256_file(TOKENIZER) == TOKENIZER_SHA256
    out_dir = K / a.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{a.name}_noexec.litertlm"
    assert not out_path.exists(), f"refusing to overwrite {out_path}"
    md = metadata(a.image_size)
    md_path = out_dir / f"{a.name}.llm_metadata.pb"
    md_path.write_bytes(md.SerializeToString())
    (out_dir / f"{a.name}.llm_metadata.pbtext").write_text(text_format.MessageToString(md))
    (out_dir / f"{a.name}.jinja").write_text(JINJA)
    b = lb.LitertLmFileBuilder()
    b.add_system_metadata(lb.Metadata(key="Authors", value="", dtype=lb.DType.STRING))
    b.add_llm_metadata(str(md_path))
    b.add_hf_tokenizer(str(TOKENIZER))
    sections = [("EMBEDDER", a.embedder, None), ("PREFILL_DECODE", a.decoder, a.act or None),
                ("VISION_ENCODER", a.encoder, None), ("VISION_ADAPTER", a.adapter, None)]
    inputs = {}
    for kind, rel, act in sections:
        p = K / rel
        inputs[kind] = {"path": rel, "bytes": p.stat().st_size, "sha256": sha256_file(p), "prefer_activation_type": act}
        b.add_tflite_model(str(p), getattr(lb.TfLiteModelType, kind), prefer_activation_type=act)
    with open(out_path, "wb") as f:
        b.build(f)
    rec = {"status": "DONE", "output": str(out_path.relative_to(K)), "bytes": out_path.stat().st_size,
           "sha256": sha256_file(out_path), "builder": imd.version("litert-lm-builder"), "inputs": inputs,
           "tokenizer": {"path": str(TOKENIZER), "sha256": TOKENIZER_SHA256},
           "metadata_pbtext": text_format.MessageToString(md), "jinja": JINJA}
    (out_dir / f"{a.name}_noexec.build.json").write_text(json.dumps(rec, indent=1) + "\n")
    print("BUNDLE_BUILT", rec["output"], rec["bytes"], rec["sha256"], flush=True)
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-template", default=None)
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--name")
    ap.add_argument("--out-dir", default="out/r2/bundle")
    ap.add_argument("--decoder")
    ap.add_argument("--embedder")
    ap.add_argument("--encoder")
    ap.add_argument("--adapter")
    ap.add_argument("--act", default="", help="PREFILL_DECODE prefer_activation_type ('' = not declared)")
    ap.add_argument("--image-size", type=int, default=IMAGE_SIZE, help="fast_vlm image tensor side (512 or 768)")
    ap.add_argument("--print-jinja", action="store_true")
    a = ap.parse_args()
    if a.print_jinja:
        print(JINJA)
        return 0
    if a.check_template:
        return check_template(a.check_template)
    if a.build:
        return build(a)
    ap.error("one of --check-template / --build / --print-jinja")


if __name__ == "__main__":
    sys.exit(main())
