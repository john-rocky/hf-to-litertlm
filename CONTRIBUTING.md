# Contributing

## Want a model converted? Paste a link.

Open a [model request](https://github.com/john-rocky/hf-to-litertlm/issues/new?template=model-request.yml)
with the Hugging Face URL. That is the whole ask. The bundle, the recipe, and the measured
numbers come back on the issue, or the reason the model could not be converted.

## A published bundle misbehaves

Open a plain issue with the repo and file name, the runtime and its version (`litert-lm --version`,
or the app and its version), the device and backend, the prompt, and what came out. Check the
model's `known_issues` in its `litertlm_manifest.json` and its section in
[REPRODUCE.md](REPRODUCE.md) first; many device walls are already recorded there.

## Pull requests

Welcome. From easiest to hardest:

- **Manifest readers** ([`readers/`](readers/)): a new language, or a fix. Mirror the TypeScript
  and Dart tests; they run against the shipped manifests in `manifest/examples/`.
- **Docs and scripts**: setup on another OS, a clearer error message, a flag that saves a step.
- **A new conversion**: the script, a `case` in `scripts/reproduce_llm.sh` or
  `scripts/reproduce_vlm.sh`, and a REPRODUCE.md entry with the gate result.

Issues labeled [good first issue](https://github.com/john-rocky/hf-to-litertlm/labels/good%20first%20issue)
are scoped for a first PR.

Rules that keep the record honest:

- **No weights, bundles, or checkouts in git.** `.gitignore` blocks them; reproduce, do not vendor.
- **Measured numbers only.** A number in a card, manifest, or README comes with the run that
  produced it: device, runtime version, prompt and decode tokens, cache off. A benchmark figure
  from a backend that produced no text is not a measurement.
- **Manifests are generated** by `manifest/make_manifest.py`, never hand-edited; derived fields
  must match the bundle header.
- **Reader semantics** ([`readers/README.md`](readers/README.md)) change only with tests updated
  in both readers.
- Code and comments in English. Apache-2.0; converted bundles inherit their base model's license.

This is an independent project, not affiliated with Google. Runtime bugs go to
[LiteRT-LM](https://github.com/google-ai-edge/LiteRT-LM/issues).
