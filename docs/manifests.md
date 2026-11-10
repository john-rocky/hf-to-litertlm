# Deployment manifests and audits

## Deployment manifests

A `.litertlm` bundle carries the conversation contract in its header, but nothing
machine-readable says which of a repo's files fits which device, backend, and RAM budget, at
what measured speed. `manifest/` defines a repo-level `litertlm_manifest.json` for that
deployment layer: [`manifest/SCHEMA.md`](../manifest/SCHEMA.md) (spec),
[`manifest/make_manifest.py`](../manifest/make_manifest.py) (generator),
[`manifest/examples/`](../manifest/examples/) (finished manifests for two of the published repos — 60 repos ship one as of 2026-09-05), and [`readers/`](../readers/) (dependency-free TypeScript and Dart reference readers).
The schema is registered in [SchemaStore](https://www.schemastore.org/litertlm_manifest.json), so editors that use its catalog (VS Code, JetBrains, and others) validate and autocomplete any file named `litertlm_manifest.json` with no setup.

The same manifest drives Google Play delivery: [`tools/play_ai_pack/`](../tools/play_ai_pack/) turns
it into a Play for On-device AI pack — one device group per manifest recommendation, the
device-targeting XML, and a host app that fetches the pack and runs the file on the backend the
manifest names.

## Audit an org's files

[`tools/community_audit/`](../tools/community_audit/) reads the header of every `.litertlm` file of a Hugging Face org (HTTP ranges, no weights) and writes one table of what each file carries: inline or external weights, the ExecutorMetadata section, vision and audio sections, the model type, whether the chat template reads `tools` and `enable_thinking`, the channels, the sampler, and flags for the files to rebuild. Each repo is read at a recorded commit, and `--pin` reads the same commits again, so a second read gives the same table. `run_check.py` runs a file from each of the most-downloaded repos, plus files named on the command line, on the CPU and the GPU (a text question, a tool-call exchange, and a photo question and thinking on and off where the file has them) and notes where the run and the header disagree. The current `litert-community` table is [`tools/community_audit/table/audit.md`](../tools/community_audit/table/audit.md).

## How the README's speeds are read

Decode speed is tokens per second, read from each repo's `litertlm_manifest.json`: the fastest
verified backend per device class, compiled-model cache off, prefill and conditions in the
manifest. A range is the spread across runs. A dash means no measured row for that class yet.

## What people download

[`tools/coverage/COVERAGE.md`](../tools/coverage/COVERAGE.md) lines this up against what people download: the Hub's 100 most-downloaded `text-generation` and 100 most-downloaded `image-text-to-text` models (30-day count, quantized re-uploads left out), each joined to the `litert-community` repos whose card names it as `base_model`, with what the conversion still lacks: a `.litertlm` file, a manifest, a file verified on GPU, external weights, image input. In the 2026-11-01 (UTC) fetch, 36 of the 200 have a conversion there and 4 lack none of these. `python3 tools/coverage/make_coverage_table.py` rebuilds it from the public Hub API.

## Projects that read the manifests

- [react-native-litert-lm](https://github.com/hung-yueh/react-native-litert-lm) — on-device LLM
  inference for React Native on LiteRT-LM, on npm as
  [`react-native-litert-lm`](https://www.npmjs.com/package/react-native-litert-lm). Since 0.7.0 its
  `resolveFromManifest()` takes a repo name from the README's table and picks the
  `.litertlm` file, backend, sampler defaults, and stream channels from that repo's
  [`litertlm_manifest.json`](#deployment-manifests), so a published bundle of a family its
  engine knows (its README lists them) loads without a conversion step
  ([source](https://github.com/hung-yueh/react-native-litert-lm/blob/568552814d9e512a09763b547261918dccd5224b/src/manifestResolver.ts)).
- [flutter_edge_ai](https://github.com/DenisovAV/flutter_edge_ai) (formerly flutter_gemma) —
  on-device AI for Flutter. Its LiteRT-LM package, on pub.dev as
  [`flutter_edge_ai_litertlm`](https://pub.dev/packages/flutter_edge_ai_litertlm), has shipped
  `LitertlmManifestResolver` since `flutter_gemma_litertlm` 1.6.0. `FlutterEdgeAi.resolveHuggingFace("<repo>")`
  reads a repo's manifest and returns the `.litertlm` file that fits the platform, so a
  published bundle from the README's table installs in a Flutter app without a conversion step
  ([source](https://github.com/DenisovAV/flutter_edge_ai/tree/712ff53881695c8866eb7a0ae56e9b72ee85dc12/packages/flutter_edge_ai_litertlm/lib/src/manifest)).
- [local_assistant_v2](https://github.com/taslimmuhammed/local_assistant_v2) — an Android
  assistant app with on-device memory search. Its EmbeddingGemma converter, which builds a
  LiteRT-LM EmbeddingEngine bundle, follows the recipe in
  [`embedding_engine_work/`](../embedding_engine_work/)
  ([source](https://github.com/taslimmuhammed/local_assistant_v2/blob/e352d9536fb86ff7f31b9b56258342db7dad9d2b/tools/embedder/convert_embeddinggemma.py)).

Both manifest resolvers vendor the reference readers in [`readers/`](../readers/) and were
contributed from this repository as pull requests
([#25](https://github.com/hung-yueh/react-native-litert-lm/pull/25),
[#469](https://github.com/DenisovAV/flutter_edge_ai/pull/469)). The SDKs' maintainers merged
them and now ship and maintain them.
