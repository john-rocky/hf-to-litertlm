"""Exports an LFM3 decoder with litert-torch's `export_hf` pipeline.

    PYTHONPATH=<lfm3_work> <venv>/bin/python -m lfm3_ext.export_lfm3 \
        --code_dir=<repo dir> --output_dir=<out> [--weights=<file|dir>] \
        [--key_prefix=model.] [--config_json=<file> --config_key=text_config] \
        [--emit_hidden] [--export key=value ...]

`export_lfm3()` is the same entry in Python. It builds `Lfm3ForCausalLM` from
the repo's remote code with the decoder weights under `key_prefix`
(`loader.py`), hands that model to `export_hf.export.export` in place of its
checkpoint loader, and passes `Lfm3Extension(emit_hidden=...)` as the model
extension, so every other stage (exportables, conversion, quantization,
tokenizer, LiteRT-LM packaging, ExecutorMetadata) is the stock one. Keyword
arguments other than the ones below go to `export.export` unchanged
(`prefill_lengths`, `cache_length`, `quantization_recipe`,
`externalize_embedder`, `bundle_litert_lm`, `jinja_chat_template_override`,
`tokenizer_path_override`, ...).
"""

import argparse
import ast
import json
import os
from unittest import mock

from litert_torch.generative.export_hf import export as export_hf
from litert_torch.generative.export_hf.core import export_lib
import transformers

from . import extension
from . import loader


def export_lfm3(
    output_dir,
    *,
    code_dir,
    weights=None,
    key_prefix="model.",
    config=None,
    tokenizer_dir=None,
    emit_hidden=False,
    **export_kwargs,
):
  """Exports an LFM3 decoder; returns `export.export`'s result.

  Args:
    output_dir: Output directory (`model.litertlm` when bundling).
    code_dir: Repo directory with the LFM3 remote code (see `loader.py`).
    weights: `.safetensors` file / directory / list; default `code_dir`.
    key_prefix: Decoder key prefix (`model.`, `model.language_model.`,
      `lfm.model.`).
    config: Decoder config dict / `Lfm3Config`; default `code_dir/config.json`.
    tokenizer_dir: Tokenizer directory; default `code_dir`.
    emit_hidden: Add the `hidden` (final-norm output) output to prefill and
      decode.
    **export_kwargs: Passed to `litert_torch...export_hf.export.export`.
  """
  model = loader.load_lfm3_causal_lm(
      code_dir, weights, key_prefix=key_prefix, config=config
  )
  tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_dir or code_dir, trust_remote_code=True)
  artifacts = export_lib.SourceModelArtifacts(
      model=model,
      model_config=model.config,
      text_model_config=model.config,
      tokenizer=tokenizer,
  )

  def _load_model(*args, **kwargs):
    del args, kwargs
    return artifacts

  with mock.patch.object(export_lib, "load_model", _load_model):
    return export_hf.export(
        model=code_dir,
        output_dir=output_dir,
        model_extension=extension.Lfm3Extension(emit_hidden=emit_hidden),
        **export_kwargs,
    )


def _parse_value(text):
  try:
    return ast.literal_eval(text)
  except (ValueError, SyntaxError):
    return text


def main():
  ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
  ap.add_argument("--code_dir", required=True)
  ap.add_argument("--output_dir", required=True)
  ap.add_argument("--weights")
  ap.add_argument("--key_prefix", default="model.")
  ap.add_argument("--config_json", help="JSON file holding the decoder config")
  ap.add_argument("--config_key", help="key of the decoder block in --config_json")
  ap.add_argument("--tokenizer_dir")
  ap.add_argument("--emit_hidden", action="store_true")
  ap.add_argument(
      "--export",
      action="append",
      default=[],
      metavar="KEY=VALUE",
      help="export.export keyword (Python literal values), repeatable",
  )
  args = ap.parse_args()
  config = None
  if args.config_json:
    with open(args.config_json) as f:
      config = json.load(f)
    if args.config_key:
      config = config[args.config_key]
  kwargs = {}
  for item in args.export:
    key, _, value = item.partition("=")
    kwargs[key] = _parse_value(value)
  print("export_lfm3 kwargs:", json.dumps({k: repr(v) for k, v in kwargs.items()}))
  export_lfm3(
      os.path.abspath(args.output_dir),
      code_dir=os.path.abspath(args.code_dir),
      weights=args.weights,
      key_prefix=args.key_prefix,
      config=config,
      tokenizer_dir=args.tokenizer_dir,
      emit_hidden=args.emit_hidden,
      **kwargs,
  )


if __name__ == "__main__":
  main()
