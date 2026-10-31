# Reproducing the shipped LLM conversions

Every dense / reasoning LLM `.litertlm` shipped from this repo is reproducible with one command:

```bash
bash scripts/reproduce_llm.sh <model-key>     # -> out/<key>/model.litertlm
bash scripts/reproduce_llm.sh --list          # all keys
bash scripts/reproduce_llm.sh --all           # everything (heavy)
```

The engine is `scripts/export_simple_template.py`
(`<hf_model> out/<name> templates/<template>.jinja <quant>` + env). Recipes reconstructed **2026-07-06**
from `cards/*.md` + auto-memory + `reports/*` while the memory was fresh (a few env/template fields are
best-inference, flagged below). The tables here are the dense/reasoning LLMs; the **Vision-language
models** section at the bottom covers the VLMs (`scripts/reproduce_vlm.sh`).

## Metadata re-packs (2026-10)

From 2026-10-13 to 2026-10-15, 28 published `.litertlm` files were re-packed with the litert-lm 0.18.0 CLI, each from the file at the Hub commit in the table below. In 26 files only LlmMetadata changed (the model type, the chat template or a thought channel), in 2 only the tokenizer section; every other section is byte-identical to the file it was built from. Each repo's model card says what changed for its users.

The blocks after the table are the commands the files were built with. Run them from the repository root in bash or zsh, with `REPO`, `FILE` and `SRC` set from a row of the table and `WORK` an empty directory for that file, after this start:

```sh
litert-lm() { uvx --from litert-lm==0.18.0 litert-lm "$@"; }
litert-lm --version
mkdir -p $WORK/src $WORK/out
[ -f $WORK/src/$FILE ] || curl -fsSL -o $WORK/src/$FILE https://huggingface.co/$REPO/resolve/$SRC/$FILE
shasum -a 256 $WORK/src/$FILE
```

The CLI prints its errors and still exits 0: read the `Successfully` line of every unpack and pack and the counts of the `grep -c` lines (the old value 1 → 0, the new one 0 → 1); every `diff` prints nothing. The packer writes a new uuid and creation timestamp into the container header, so a rebuilt file's sha256 differs from the table; compare it section by section with `python scripts/section_identity.py <published> <rebuilt>` (every section `SAME`). The one-command tools named in blocks E, F and G need the litert-lm CLI and litert-lm-builder in the Python that runs them (`pip install litert-lm==0.18.0`).

| block | repo | file | change | built from (Hub commit) | result sha256 (first 16) | Hub commit of the result |
|---|---|---|---|---|---|---|
| A | litert-community/LFM2.5-1.2B-Instruct | LFM2.5-1.2B-Instruct_int4.litertlm | `lfm2` + tool rendering | a997cb6f49a991c0d9b541d3dc4b6ffaa405fb3c | e904729a8a8a0160 | d23bed4d |
| A | litert-community/LFM2.5-1.2B-Instruct | LFM2.5-1.2B-Instruct_int4_gpu.litertlm | `lfm2` + tool rendering | a997cb6f49a991c0d9b541d3dc4b6ffaa405fb3c | 8d3a4d73e121166a | a872e98e |
| A | litert-community/LFM2.5-1.2B-Instruct | LFM2.5-1.2B-Instruct_int8.litertlm | `lfm2` + tool rendering | a997cb6f49a991c0d9b541d3dc4b6ffaa405fb3c | ed8201f88a1a1162 | b2614265 |
| A | litert-community/LFM2.5-1.2B-JP | LFM2.5-1.2B-JP_int4.litertlm | `lfm2` + tool rendering | 20a41f29c1e61485a28eaf783c3ca3e796d685f9 | eed10977c555e7a7 | 896c0aab |
| A | litert-community/LFM2.5-1.2B-JP | LFM2.5-1.2B-JP_int4_gpu.litertlm | `lfm2` + tool rendering | 20a41f29c1e61485a28eaf783c3ca3e796d685f9 | 2f70967b635324bc | fc1951ca |
| A | litert-community/LFM2.5-1.2B-JP | LFM2.5-1.2B-JP_int8.litertlm | `lfm2` + tool rendering | 20a41f29c1e61485a28eaf783c3ca3e796d685f9 | d4b0af0842582646 | f1de60e7 |
| A | litert-community/LFM2.5-1.2B-JP | LFM2.5-1.2B-JP_int8_gpu.litertlm | `lfm2` + tool rendering | 20a41f29c1e61485a28eaf783c3ca3e796d685f9 | a3248222f6cb0c62 | 2c0464e4 |
| A | litert-community/LFM2.5-1.2B-Thinking | LFM2.5-1.2B-Thinking_int4.litertlm | `lfm2` + tool rendering | 773d2ccb1807ed28867da924a4cb7a6112ef3501 | da3c0ad154b78f33 | b96ac1f5 |
| A | litert-community/LFM2.5-1.2B-Thinking | LFM2.5-1.2B-Thinking_int4_gpu.litertlm | `lfm2` + tool rendering | 773d2ccb1807ed28867da924a4cb7a6112ef3501 | 51130e6e57a940bb | ea485b08 |
| A | litert-community/LFM2.5-1.2B-Thinking | LFM2.5-1.2B-Thinking_int8.litertlm | `lfm2` + tool rendering | 773d2ccb1807ed28867da924a4cb7a6112ef3501 | 6c9fb75c38a5d3c3 | 66d411fb |
| A | litert-community/LFM2.5-1.2B-Thinking | LFM2.5-1.2B-Thinking_int8_gpu.litertlm | `lfm2` + tool rendering | 773d2ccb1807ed28867da924a4cb7a6112ef3501 | 73122d7c8e72009d | d72ba82f |
| A | litert-community/LFM2.5-2.6B | LFM2.5-2.6B_int4.litertlm | `lfm2` + tool rendering | 0a88561cd1ddc82ab8badff4e641bd54236a4867 | d18381eaee160ede | e6f5ac5d |
| A | litert-community/LFM2.5-2.6B | LFM2.5-2.6B_int8.litertlm | `lfm2` + tool rendering | 0a88561cd1ddc82ab8badff4e641bd54236a4867 | 16460604711efd81 | e3cbd311 |
| B | litert-community/LFM2.5-230M | LFM2.5-230M_int4.litertlm | `lfm2` | eef2b56ccb220f8bd9e2c4ecd6a06fd8728776b0 | c9065e28358a1f2f | d145aeda |
| B | litert-community/LFM2.5-230M | LFM2.5-230M_int8.litertlm | `lfm2` | eef2b56ccb220f8bd9e2c4ecd6a06fd8728776b0 | 3c891cb3ab299385 | aa064b96 |
| C | mlboydaisuke/MiniCPM5-2B-LiteRT | MiniCPM5-2B_int4.litertlm | official `minicpm5` configuration | fc93c93a1be1c27c863cef06fa10f30585120df2 | 7234c22920b42798 | 967ffe12 |
| C | mlboydaisuke/MiniCPM5-2B-LiteRT | MiniCPM5-2B_int8.litertlm | official `minicpm5` configuration | fc93c93a1be1c27c863cef06fa10f30585120df2 | 702b28cf111bedd2 | cb9a20cf |
| D | litert-community/Falcon-H1-3B-Instruct | Falcon-H1-3B-Instruct_int8.litertlm | `qwen2p5` + tool results | 5979909b36447edc3b685e7ad93101ab9acb5a43 | 2b03ccab37a26c4e | d8c58dfb |
| D | litert-community/granite-4.0-h-1b | granite-4.0-h-1b_int8.litertlm | `qwen2p5` + tool results | 09a702fac64abf082a9ca9c842782fd121160b48 | a38b9af8d8fe150d | a3ef2ced |
| D | litert-community/granite-4.0-h-350m | granite-4.0-h-350m_fp16.litertlm | `qwen2p5` + tool results | 808e03b1ea0b1c23981fce68498650ef17346ead | a48f4471dce39abe | 5307042a |
| D | litert-community/granite-4.0-h-350m | granite-4.0-h-350m_int8.litertlm | `qwen2p5` + tool results | 808e03b1ea0b1c23981fce68498650ef17346ead | 46d2781278e77d96 | baeac72e |
| D | litert-community/granite-4.0-h-350m | granite-4.0-h-350m_int8_gpu.litertlm | `qwen2p5` + tool results | 808e03b1ea0b1c23981fce68498650ef17346ead | 81020b48fa6384b8 | 482b1786 |
| D | mlboydaisuke/Falcon3-3B-Instruct-LiteRT | model.litertlm | `qwen2p5` + tool results | 3190c7349875ca6059a542e31a56ca649e6a7806 | 00fe6ef173363ba8 | a553da43 |
| E | mlboydaisuke/Qwen3-0.6B-wo4-float-repro-LiteRT | qwen3-0.6b_weight_only_wi4_afp32.litertlm | content parts | 8897b5efd1babebf191fff925b3366119c1a85a6 | 4cc6201f38aa6655 | 6432f5b8 |
| F | litert-community/Ovis2.5-2B | Ovis2.5-2B.litertlm | `thought` channel | d36697ad01dca6962117b1ae3101da44e3b1ed22 | f31d0ef62545bf31 | ccb79915 |
| F | mlboydaisuke/DeepSeek-R1-Distill-Qwen-7B-LiteRT | model.litertlm | `thought` channel | e3608156cdbcf020f16d3cfeca2268ba100253c6 | b4b373725e074096 | d66b93a8 |
| G | litert-community/North-Micro-Vision-Instruct | North-Micro-Vision-Instruct_int4.litertlm | tokenizer section | 54df6991b685dc0c2aa933b15990fab3d2b50951 | 0518a6fe7c73dd04 | c97bb1e8 |
| G | litert-community/North-Micro-Vision-Instruct | North-Micro-Vision-Instruct_wi8.litertlm | tokenizer section | 54df6991b685dc0c2aa933b15990fab3d2b50951 | 6ce917e1eec510f0 | d812a8a7 |

**A — `lfm2` and tool rendering (13 files).** The model type changes from `generic_model` to `lfm2`, so litert-lm 0.18.0 and newer parse the model's `<|tool_call_start|>` calls instead of returning them as text, and the file's own template gets the tool rendering of LiteRT-LM's `models/lfm2/chat_template.jinja`. `add_tool_branch.py` writes that template from the one the source file carries and prints `MATCH` when the result is the packed template.

```sh
curl -fsSL -o $WORK/official_lfm2.jinja https://raw.githubusercontent.com/google-ai-edge/LiteRT-LM/a484d21d5113886348fe4c56af61830e45e664a8/models/lfm2/chat_template.jinja
litert-lm unpack $WORK/src/$FILE --output-dir $WORK/work --chat-template $WORK/orig.jinja
python tools/chat_template_parts/add_tool_branch.py lfm2 $WORK/orig.jinja $WORK/tools.jinja --official $WORK/official_lfm2.jinja
grep '^max_num_tokens' $WORK/work/LlmMetadataProto.pbtext
grep -c '^  generic_model {$' $WORK/work/LlmMetadataProto.pbtext
grep -c '^  lfm2 {$' $WORK/work/LlmMetadataProto.pbtext
perl -0pi -e 's/^llm_model_type \{\n  generic_model \{\n  \}\n\}$/llm_model_type {\n  lfm2 {\n  }\n}/m' $WORK/work/LlmMetadataProto.pbtext
grep -c '^  generic_model {$' $WORK/work/LlmMetadataProto.pbtext
grep -c '^  lfm2 {$' $WORK/work/LlmMetadataProto.pbtext
grep '^max_num_tokens' $WORK/work/LlmMetadataProto.pbtext
litert-lm pack $WORK/work --chat-template $WORK/tools.jinja --output $WORK/out/$FILE
litert-lm unpack $WORK/out/$FILE --output-dir $WORK/chk --chat-template $WORK/chk.jinja && diff $WORK/chk.jinja $WORK/tools.jinja
diff -r -x LlmMetadataProto.pbtext -x model.toml $WORK/work $WORK/chk
shasum -a 256 $WORK/out/$FILE
```

**B — `lfm2` only (2 files).** The template these files carry already renders tool calls and tool results, so the model type is the whole change.

```sh
litert-lm unpack $WORK/src/$FILE --output-dir $WORK/work --chat-template $WORK/orig.jinja
grep '^max_num_tokens' $WORK/work/LlmMetadataProto.pbtext
grep -c '^  generic_model {$' $WORK/work/LlmMetadataProto.pbtext
grep -c '^  lfm2 {$' $WORK/work/LlmMetadataProto.pbtext
perl -0pi -e 's/^llm_model_type \{\n  generic_model \{\n  \}\n\}$/llm_model_type {\n  lfm2 {\n  }\n}/m' $WORK/work/LlmMetadataProto.pbtext
grep -c '^  generic_model {$' $WORK/work/LlmMetadataProto.pbtext
grep -c '^  lfm2 {$' $WORK/work/LlmMetadataProto.pbtext
grep '^max_num_tokens' $WORK/work/LlmMetadataProto.pbtext
litert-lm pack $WORK/work --output $WORK/out/$FILE
litert-lm unpack $WORK/out/$FILE --output-dir $WORK/chk --chat-template $WORK/chk.jinja && diff $WORK/chk.jinja $WORK/orig.jinja
diff -r -x LlmMetadataProto.pbtext -x model.toml $WORK/work $WORK/chk
shasum -a 256 $WORK/out/$FILE
```

**C — the official `minicpm5` configuration (2 files).** LiteRT-LM's `models/minicpm5` metadata and chat template replace the file's (the re-packed files need litert-lm 0.18.0 or newer), with the steps of `scripts/repack_official_metadata.sh`: download, unpack, the official pbtext in place, pack with the official template, the section check. `max_num_tokens` is 4096 before and after.

```sh
bash scripts/repack_official_metadata.sh mlboydaisuke/MiniCPM5-2B-LiteRT MiniCPM5-2B_int4.litertlm minicpm5 fc93c93a1be1c27c863cef06fa10f30585120df2 4096 a484d21d5113886348fe4c56af61830e45e664a8
bash scripts/repack_official_metadata.sh mlboydaisuke/MiniCPM5-2B-LiteRT MiniCPM5-2B_int8.litertlm minicpm5 fc93c93a1be1c27c863cef06fa10f30585120df2 4096 a484d21d5113886348fe4c56af61830e45e664a8
# -> out/repack/<file>
```

**D — `qwen2p5` and tool results (6 files).** The model type changes from `generic_model` to `qwen2p5`, so the runtime parses the JSON calls these models write in `<tool_call>`, and the file's own template also renders the tool result, which litert-lm 0.18 passes as a `tool_response` content part. In Falcon3's template `tojson` also loses its `indent` argument (the runtime's `tojson` takes none, so passing `tools` failed), and in Falcon-H1's the tool-call schema line is written in JSON (the runtime's parser rejects single quotes). The granite files get `tools/chat_template_parts/granite-4.0-h_tools.jinja`; for the Falcon files `add_tool_branch.py` writes the template from the source file's own.

```sh
litert-lm unpack $WORK/src/$FILE --output-dir $WORK/work --chat-template $WORK/orig.jinja
TPL=$PWD/tools/chat_template_parts/granite-4.0-h_tools.jinja   # granite-4.0-h; for Falcon, the two lines below instead
# python tools/chat_template_parts/add_tool_branch.py falcon-h1 $WORK/orig.jinja $WORK/tools.jinja && TPL=$WORK/tools.jinja   # Falcon-H1-3B-Instruct
# python tools/chat_template_parts/add_tool_branch.py falcon3 $WORK/orig.jinja $WORK/tools.jinja && TPL=$WORK/tools.jinja     # Falcon3-3B-Instruct
shasum -a 256 $TPL
grep '^max_num_tokens' $WORK/work/LlmMetadataProto.pbtext
grep -c '^  generic_model {$' $WORK/work/LlmMetadataProto.pbtext
grep -c '^  qwen2p5 {$' $WORK/work/LlmMetadataProto.pbtext
perl -0pi -e 's/^llm_model_type \{\n  generic_model \{\n  \}\n\}$/llm_model_type {\n  qwen2p5 {\n  }\n}/m' $WORK/work/LlmMetadataProto.pbtext
grep -c '^  generic_model {$' $WORK/work/LlmMetadataProto.pbtext
grep -c '^  qwen2p5 {$' $WORK/work/LlmMetadataProto.pbtext
grep '^max_num_tokens' $WORK/work/LlmMetadataProto.pbtext
litert-lm pack $WORK/work --chat-template $TPL --output $WORK/out/$FILE
litert-lm unpack $WORK/out/$FILE --output-dir $WORK/chk --chat-template $WORK/chk.jinja && diff $WORK/chk.jinja $TPL
diff -r -x LlmMetadataProto.pbtext -x model.toml $WORK/work $WORK/chk
shasum -a 256 $WORK/out/$FILE
```

**E — content parts (1 file).** litert-lm 0.18 hands every message content to the template as a list of parts; this file's template read strings only, so every turn rendered empty. `tools/chat_template_parts/qwen3-0.6b-wo4_parts.jinja` reads both forms; nothing else changes. `python scripts/set_prompt_template.py $WORK/src/$FILE $WORK/out/$FILE $TPL` gives the same sections.

```sh
TPL=$PWD/tools/chat_template_parts/qwen3-0.6b-wo4_parts.jinja
shasum -a 256 $TPL
litert-lm unpack $WORK/src/$FILE --output-dir $WORK/work --chat-template $WORK/orig.jinja
grep '^max_num_tokens' $WORK/work/LlmMetadataProto.pbtext
grep -c '^channels {$' $WORK/work/LlmMetadataProto.pbtext
litert-lm pack $WORK/work --chat-template $TPL --output $WORK/out/$FILE
litert-lm unpack $WORK/out/$FILE --output-dir $WORK/chk --chat-template $WORK/chk.jinja && diff $WORK/chk.jinja $TPL
diff -r -x LlmMetadataProto.pbtext -x model.toml $WORK/work $WORK/chk
shasum -a 256 $WORK/out/$FILE
```

**F — a thought channel (2 files).** On litert-lm 0.18.0 the reasoning came back inside the message text. A `thought` channel with the bare markers `<think>` and `</think>` (single tokens in both tokenizers) returns it on the thought channel; nothing else changes. DeepSeek-R1-Distill-Qwen-7B's file carries no Jinja template: unpack it without `--chat-template` and skip the template diff. `python tools/add_thought_channel.py $WORK/src/$FILE $WORK/out/$FILE --start '<think>' --end '</think>'` gives the same sections.

```sh
litert-lm unpack $WORK/src/$FILE --output-dir $WORK/work --chat-template $WORK/orig.jinja
grep '^max_num_tokens' $WORK/work/LlmMetadataProto.pbtext
grep -c '^channels {$' $WORK/work/LlmMetadataProto.pbtext
[ "$(tail -c 1 $WORK/work/LlmMetadataProto.pbtext | xxd -p)" = 0a ]
printf 'channels {\n  channel_name: "thought"\n  start: "<think>"\n  end: "</think>"\n}\n' >> $WORK/work/LlmMetadataProto.pbtext
grep -c '^channels {$' $WORK/work/LlmMetadataProto.pbtext
grep -c '^  channel_name: "thought"$' $WORK/work/LlmMetadataProto.pbtext
grep -c '^  start: "<think>"$' $WORK/work/LlmMetadataProto.pbtext
grep -c '^  end: "</think>"$' $WORK/work/LlmMetadataProto.pbtext
grep '^max_num_tokens' $WORK/work/LlmMetadataProto.pbtext
litert-lm pack $WORK/work --output $WORK/out/$FILE
litert-lm unpack $WORK/out/$FILE --output-dir $WORK/chk --chat-template $WORK/chk.jinja && diff $WORK/chk.jinja $WORK/orig.jinja
grep -c '^channels {$' $WORK/chk/LlmMetadataProto.pbtext
diff -r -x LlmMetadataProto.pbtext -x model.toml $WORK/work $WORK/chk
shasum -a 256 $WORK/out/$FILE
```

**G — the tokenizer section (2 files).** The section held the `tokenizer.json` that `AutoTokenizer.save_pretrained` writes, which splits every digit; it is now CohereLabs/North-Micro-Vision-Instruct's own `tokenizer.json` at `fa548d9a` (sha256 `6fcc5292…`), and every other section, LlmMetadata included, is byte-identical. `python tools/tokenizer_parity/swap_tokenizer_section.py $WORK/src/$FILE $WORK/tokenizer.json $WORK/out/$FILE` gives the same sections.

```sh
curl -fsSL -o $WORK/tokenizer.json https://huggingface.co/CohereLabs/North-Micro-Vision-Instruct/resolve/fa548d9aa62c1220bfd21fc6cfc465f1bd2a1a40/tokenizer.json
shasum -a 256 $WORK/tokenizer.json
litert-lm unpack $WORK/src/$FILE --output-dir $WORK/work --chat-template $WORK/orig.jinja
grep '^max_num_tokens' $WORK/work/LlmMetadataProto.pbtext
grep -c '^section_type = "HF_Tokenizer"$' $WORK/work/model.toml
grep -c '^data_path = "Section1_HF_Tokenizer_Zlib.zlib"$' $WORK/work/model.toml
mv $WORK/work/Section1_HF_Tokenizer_Zlib.zlib $WORK/orig_tokenizer_section.zlib
cp $WORK/tokenizer.json $WORK/work/tokenizer.json
perl -pi -e 's/^data_path = "Section1_HF_Tokenizer_Zlib\.zlib"$/data_path = "tokenizer.json"/' $WORK/work/model.toml
grep -c '^data_path = "tokenizer.json"$' $WORK/work/model.toml
grep -c 'Section1_HF_Tokenizer_Zlib' $WORK/work/model.toml
litert-lm pack $WORK/work --output $WORK/out/$FILE
litert-lm unpack $WORK/out/$FILE --output-dir $WORK/chk --chat-template $WORK/chk.jinja && diff $WORK/chk.jinja $WORK/orig.jinja
diff $WORK/work/LlmMetadataProto.pbtext $WORK/chk/LlmMetadataProto.pbtext
diff -r -x LlmMetadataProto.pbtext -x model.toml -x tokenizer.json -x Section1_HF_Tokenizer_Zlib.zlib $WORK/work $WORK/chk
shasum -a 256 $WORK/out/$FILE
```

Checked on 2026-10-15 on a Mac with litert-lm 0.18.0: run as printed from the source commit, blocks A, B, D, E, F and G rebuilt LFM2.5-1.2B-Instruct_int4, LFM2.5-230M_int4, granite-4.0-h-350m_int8, qwen3-0.6b_weight_only_wi4_afp32, Ovis2.5-2B and North-Micro-Vision-Instruct_int4 with every section equal to the published file, and `set_prompt_template.py`, `add_thought_channel.py` and `swap_tokenizer_section.py` gave the same sections for the E, F and G files. Block C was not re-run.

## External weights (2026-10)

A first published `.litertlm` is re-packed so that the weights of its TFLiteModel section sit in a TFLiteWeights section at the end of the file, with LiteRT's `externalize_tflite_flatbuffer.py` and `externalize_litertlm_sections.py` (commit `c5c0d28f`; `tools/externalize` pins `8761012b` since 2026-10-30, which gives the same sections) and litert-lm-builder 0.18.0 (Hub PR https://huggingface.co/litert-community/Qwen3-0.6B/discussions/13, open). The Android GPU path of the LiteRT-LM v0.18.0 and v0.18.1 builds crashes on such a file (google-ai-edge/LiteRT-LM#3986); LiteRT-LM 0.19.0 runs it. Every other section is byte-identical to the file it was built from; `tools/externalize/README.md` lists the checks each file passed.

| repo | file | built from (Hub commit) | source sha256 (first 16) | TFLiteModel / TFLiteWeights sections | result sha256 (first 16) | Hub commit of the result |
|---|---|---|---|---|---|---|
| litert-community/Qwen3-0.6B | Qwen3-0.6B_dynamic_wi4b32_afp32.litertlm | a3c5d805ae362dff7f580bc25f2dfb9a5a7eaa76 | 03e7da1eb1108b50 | 1 / 1 | 96491bb0f78fa680 | PR #13 (open 2026-10-15, refs/pr/13) |

Rebuild a row from the repository root, after the `pip install` line and `fetch_tools.sh` in `tools/externalize/README.md`:

```sh
export XR_TOOLS_PATH=$HOME/.cache/litert-externalize/8761012bb42439f0ac1df55f144d6ace47c2b35c
bash tools/externalize/externalize_repo.sh litert-community/Qwen3-0.6B Qwen3-0.6B_dynamic_wi4b32_afp32.litertlm \
  --revision a3c5d805ae362dff7f580bc25f2dfb9a5a7eaa76
```

The builder writes a new uuid and creation timestamp into the container header, so a rebuilt file's sha256 differs from the table. Compare it with the published file section by section: `python scripts/section_identity.py <published> externalize_staging/litert-community__Qwen3-0.6B/Qwen3-0.6B_dynamic_wi4b32_afp32.litertlm` prints SAME for every section. The run's report (`externalize_staging/litert-community__Qwen3-0.6B/Qwen3-0.6B_dynamic_wi4b32_afp32.litertlm.REPORT.md`) gives the raw sha256 of each section: TFLiteModel `dcb1e3d69326af0d…`, TFLiteWeights `21011d25361c728d…`. Run as written on 2026-10-30 with the tools at `8761012b`: SAME for all five sections against the published file.


## 2026-10-12 — metadata re-ship to the official LiteRT-LM configuration, second batch (3 files, weights unchanged)

Same recipe and script as the 2026-10-10 batch (`scripts/repack_official_metadata.sh`, official config at LiteRT-LM commit `4a363c0728d4461eaf05c4d87250ef3e90deb035`, byte-identical to main `03d3979f` on 2026-10-10). Every section other than LlmMetadata is byte-identical to the published file; `max_num_tokens` was left at the official 4096 because each original either set 4096 or left the field unset. Proposed as Hub pull requests (metadata only); until a PR merges, `main` serves the original.

| repo | file | family | original (Hub commit) | max_num_tokens | sha256 of the file in the PR (first 16) | PR |
|---|---|---|---|---|---|---|
| litert-community/Qwen3-4B | qwen3_4b_channelwise_int8_float32kv.litertlm | qwen3 | 2f7415b4e3a149c16b929659d146a6eaf6d99a34 | 4096 (unchanged) | fe7483c117978061 | https://huggingface.co/litert-community/Qwen3-4B/discussions/4 |
| litert-community/Qwen3-0.6B | Qwen3-0.6B.litertlm | qwen3 | a3c5d805ae362dff7f580bc25f2dfb9a5a7eaa76 | 4096 (unchanged) | 346f9b2e76b95ed9 | https://huggingface.co/litert-community/Qwen3-0.6B/discussions/7 |
| litert-community/Qwen2.5-1.5B-Instruct | Qwen2.5-1.5B-Instruct_multi-prefill-seq_f32_ekv4096.litertlm | qwen2_5 | 5e6e7a58d03777898b31330a15d2e0a5a9107de3 | 4096 (official; the original left it unset) | 13f79f0d0d2d6d68 | https://huggingface.co/litert-community/Qwen2.5-1.5B-Instruct/discussions/8 |

What differs from the first batch, learned on these files:

- Two of the three originals had no chat template at all (`Qwen3-0.6B.litertlm`: no model type either; the Qwen2.5 f32 file: a default system turn baked into its user prefix). A file with no template is not broken on litert-lm 0.18.0 — the runtime renders it from its prefix/suffix templates — so for such a file the official configuration is a behaviour change, not a fix: the baked default system turn goes (the official qwen2_5 template writes a system block only for a system message or tools), the qwen2_5 pbtext declares `min_runtime_version` 0.18.0 and `supports_function_calling`, and the qwen3 pbtext adds the thought channel. Run the original on the current runtime first and describe the change as a change.
- `Qwen3-0.6B.litertlm` and the Qwen2.5 f32 file carry a SentencePiece tokenizer section (not an HF tokenizer.json); the re-pack leaves it byte-identical, and with it an emoji defect: asked to repeat `Good morning 😊 🚀`, both files answer `Good morningĠðŁĺĬ🚀` before and after the update (the space + emoji come out as GPT-2 byte-alphabet characters), while the same repos' HF-tokenizer files return the emoji. The fix for that is a tokenizer-section swap (the 2026-08-30 batch above), not a metadata change. Read the tokenizer section type after the unpack and run an emoji repeat-back prompt (thinking off) on every re-pack.
- `litert-community/Qwen3-0.6B/qwen3_0_6b_mixed_int4.litertlm` could not be re-packed with the public tools: `litert-lm unpack` (0.18.0, and the builder on LiteRT-LM main) stops with `DecodeError … 'litert.lm.proto.EmbeddingMetadata': Wire format was corrupt` and exits 0; the section the header labels EmbeddingMetadataProto parses byte-exactly as an ExecutorMetadata. Check the `Successfully` line of every unpack / pack, not the exit code.
- Checks on litert-lm 0.18.0 (python API, CPU, one process per check) on each result: plain prompt, 17 × 23 with thinking on and off (no thought channel for qwen2_5), an automatic tool call answered from its result, a tool result read from the tool message, two turns; on litert-lm 0.17.0 all three render the user turn empty and answer something unrelated, exit 0. The Qwen3-4B int8 result also answered 8 of 8 on an eight-question check on a Mac's CPU and GPU (WebGPU on Metal, every prefill and decode subgraph delegated), the same as the repo's `qwen3_4b_mixed_int4.litertlm`.

## 2026-10-10 — metadata re-ship to the official LiteRT-LM configuration (8 files, weights unchanged)

Eight published `.litertlm` files get the LiteRT-LM official configuration of their family, `models/<family>/LlmMetadataProto.pbtext` + `chat_template.jinja` at LiteRT-LM commit `4a363c0728d4461eaf05c4d87250ef3e90deb035`, with the file's own `max_num_tokens` kept and every other section byte-identical (the script compares the unpacked section files by content). litert-lm 0.18.0 hands the chat template each message's content as a list of parts; the official templates read parts, so the re-packed files need litert-lm 0.18.0 or newer (on 0.17.x the user turn renders empty).

```bash
bash scripts/repack_official_metadata.sh <hf-repo> <file.litertlm> <family> <hub-commit-of-the-original> [max_num_tokens]
# -> out/repack/<file.litertlm>, sha256 printed last; the two diffs / the section check print nothing on success
```

| repo | file | family | original (Hub commit) | max_num_tokens | sha256 of the file in the PR (first 16) |
|---|---|---|---|---|---|
| litert-community/MiniCPM5-1B | minicpm_wi4b32_wi8_afp32_gpu_opt.litertlm | minicpm5 | f6a837aa9437aa389f0161f2a1a65c715ae1c17a | 1024 | 40ffba999ca45223 |
| litert-community/MiniCPM5-1B | minicpm_wi4b32_wi8_afp32.litertlm | minicpm5 | f6a837aa9437aa389f0161f2a1a65c715ae1c17a | 1024 | ee1b70697ef583c4 |
| litert-community/MiniCPM5-1B | MiniCPM5-1B_dynamic_wi8_afp32.litertlm | minicpm5 | f6a837aa9437aa389f0161f2a1a65c715ae1c17a | 4096 (unchanged) | 13ec3c4372dd4172 |
| litert-community/MiniCPM5-2B | MiniCPM5-2B_int4.litertlm | minicpm5 | 8f5f487a72141710604742866bc6f67afdffc798 | 4096 (unchanged) | 05b78bb57dd5ded8 |
| litert-community/Qwen3-1.7B | Qwen3-1.7B_dynamic_wi4b32_afp32.litertlm | qwen3 | 73fbc3fe8271c162a603ee66f6e7ed25b6211195 | 4096 (unchanged) | 2a72330b504a6596 |
| litert-community/Qwen3-1.7B | Qwen3_1.7B.litertlm | qwen3 | 73fbc3fe8271c162a603ee66f6e7ed25b6211195 | 4096 (unchanged) | 362ec1c906a49bbe |
| litert-community/Qwen3-4B | qwen3_4b_mixed_int4.litertlm | qwen3 | 84cc5a35c9c65cd18fcd65bb1f3a7d77a4acfe6e | 2048 | 569fe029f51e55b6 |
| litert-community/Qwen2.5-1.5B-Instruct | Qwen2.5-1.5B-Instruct_multi-prefill-seq_q8_ekv4096.litertlm | qwen2_5 | 19edb84c69a0212f29a6ef17ba0d6f278b6a1614 | 4096 (official; the original left it unset) | c4a9b7289365ab80 |

A re-run reproduces every section byte for byte but not the file's sha256: the packer writes a new uuid and creation timestamp into the container header (37 bytes differed between two runs on the first file above, same size). The resulting files are proposed to the repos as Hub pull requests (metadata only); until a PR merges, `main` serves the original. Checks that ran on each result with litert-lm 0.18.0 (python API, CPU, one process per check): describe, a plain prompt, an automatic tool call answered from its result, thinking on and off, two turns, a tool result read from the tool message, and a system-prompt instruction. Exceptions worth knowing: `qwen3_4b_mixed_int4.litertlm` keeps a 2,048-token context, and with thinking on (the official qwen3 default) and no `thinking_token_budget` the thought can use it up before the answer; the Qwen2.5 official configuration sets `min_runtime_version 0.18.0` and `supports_function_calling`.

## 2026-10-06 — GPU graph re-ship (8 files, weights unchanged)

The eight files below were exported without litert-torch's `--apply_gpu_composites`. Their prefill/decode graph writes the KV cache with two DYNAMIC_UPDATE_SLICE ops per layer and runs attention with BATCH_MATMUL(adjY). A graph exported with the flag has the same KV layout and weights, but it expresses both steps as composites.

**The re-ship rewrites only the prefill/decode graph into the flag's form; every weight buffer keeps its bytes.** The two KV-cache writes per layer become one STABLEHLO_COMPOSITE `odml.cache_update`, and each BATCH_MATMUL(adjY) attention product becomes `odml.runtime_bmm`. A signature input `param_tensor` INT32[1,1,1,7] is added, and the runtime fills in start and end. The attention mask stays a FLOAT32 input and is added to the logits by broadcast.

Seven of the files also get `prefill_16`, a 16-token prefill signature copied from the existing one (`prefill_128`, or `prefill_256` in `SmolLM3-3B.litertlm`). It shares every weight buffer, and every existing subgraph and SignatureDef keeps its index and bytes. On the Mac GPU, the new files decode 1.22–1.46× as fast as the previous ones. On the seven with `prefill_16`, the Mac GPU time to first token for a 16-token prompt drops to 0.18–0.40 of the previous value (last table).

Step 1 ran on all eight files; `previous.litertlm` is the file at the Hub commit in the first table below. Step 2 ran on all but Phi-4-mini-reasoning, with `--source prefill_256` for `SmolLM3-3B.litertlm`. The scripts unpack and pack the bundle with the CLI named in `LITERT_LM_CLI`.

```
export LITERT_LM_CLI=/path/to/litert-lm
python tools/gpu_graph/gpu_graph_retrofit.py previous.litertlm retrofit.litertlm --decomp exporter --mask add_bcast
python tools/gpu_graph/prefill_bucket_clone.py build retrofit.litertlm new.litertlm --lengths 16 --source prefill_128 --report bucket.json
```

Phi-4-mini-reasoning ships without `prefill_16`. With it, on the Mac GPU with fp16 activations, the file got 7 of 8 questions and 26 of 30 GSM8K questions right. The plain retrofit got 8 of 8 and 28 of 30, the same as the previous file.

The copies in [`tools/gpu_graph/`](tools/gpu_graph/) rebuild all eight files from the previous files, with every bundle section byte-identical to the shipped file (sha256 per section). Only the bundle header differs because `litert-lm pack` writes a new uuid and timestamp. So the whole-file sha256 of a rebuild does not match the table.

Checks:
- All eight: every bundle section other than the prefill/decode graph is byte-identical to the previous file, and so is the graph's weight region (checked per section and per buffer).
- The seven with `prefill_16`, on CPU: `prefill_16` and the signature it was copied from give bit-identical results on the same tokens (KV cache at the valid positions and 3 decode steps; 2 of 2 cases).
- The same seven: answers are byte-identical to the plain retrofit's (step 1 only) on 9 prompts (8 questions and 1 long prompt). This holds on CPU with the `litert-lm` CLI and on the Mac GPU with fp32 activations.

| Repo (`litert-community/…`) | File | Size (bytes) | sha256 (first 12) | Built from Hub commit |
|---|---|---:|---|---|
| SmolLM3-3B | `SmolLM3-3B_q4_block32_ekv4096.litertlm` | 2,004,633,520 | `bdd7eb05b676` | `c29e7655125f` |
| SmolLM3-3B | `SmolLM3-3B.litertlm` (int8) | 3,125,204,208 | `c1c6c6351947` | `c29e7655125f` |
| Ministral-3-3B-Instruct-2512 | `Ministral-3-3B-Instruct-2512_q4_block32_ekv4096.litertlm` | 2,342,866,928 | `13c47bcbaf30` | `613e437fea97` |
| VibeThinker-3B | `model.litertlm` | 2,059,236,272 | `7885496ccd8f` | `087512d2eba6` |
| Hy-MT2-1.8B | `Hy-MT2-1.8B_int8.litertlm` | 1,827,026,224 | `275ff52bf04c` | `18cc1a6794d7` |
| Phi-4-mini-reasoning | `model.litertlm` | 2,784,187,376 | `d507262ca3d0` | `ea890746c4f4` |
| Qwen3-4B-Thinking-2507 | `model.litertlm` | 2,477,192,112 | `8a8a6a419a55` | `92751be3c031` |
| DeepSeek-R1-Distill-Qwen-7B | `DeepSeek-R1-Distill-Qwen-7B_q4_block32_ekv4096.litertlm` | 4,533,682,160 | `050787a59d80` | `96991f3653ce` |

Mac GPU with fp16 activations, previous → new file. Decode and TTFT (time to first token): Mac Studio M4 Max, `litert-lm` 0.17.1 CLI `benchmark --cache no` (WebGPU on Metal), both files run in the same window, median of 3 runs. Decode uses a 256-token prompt and 256 decode tokens. GSM8K: 30 questions, greedy, up to 2048 tokens.

| Model | Decode tok/s | TTFT, 16-token prompt (s) | GSM8K (of 30) |
|---|---:|---:|---:|
| SmolLM3-3B q4 | 90.5 → 131.8 | 0.111 → 0.035 | 26 → 26 |
| SmolLM3-3B int8 | 77.0 → 106.0 | 0.205 → 0.038 | — |
| Ministral-3-3B-Instruct-2512 | 92.4 → 129.5 | 0.121 → 0.038 | 26 → 26 |
| VibeThinker-3B | 92.0 → 125.6 | 0.107 → 0.035 | 29 → 30 |
| Hy-MT2-1.8B | 103.5 → 150.7 | 0.076 → 0.031 | — |
| Phi-4-mini-reasoning (no `prefill_16`) | 80.6 → 114.9 | 0.130 → 0.113 | 28 → 28 |
| Qwen3-4B-Thinking-2507 | 66.6 → 92.1 | 0.152 → 0.037 | 26 → 26 |
| DeepSeek-R1-Distill-Qwen-7B | 65.0 → 79.1 | 0.220 → 0.046 | 27 → 28 |


## 2026-08-30 — tokenizer section re-ship (17 files, weights unchanged)

An audit of every published `.litertlm` (91 files) against the upstream tokenizers found that the 14 bundles whose tokenizer was a SentencePiece conversion of a byte-level BPE vocabulary (litert-torch `tokenizer_to_sentencepiece_lib`, our `FORCE_SPM` recipe) mis-tokenize standalone accented/special characters, turn characters without a whole-character piece (emoji, most of Latin Extended-A) into the token the conversion had reused as UNK, and — for `SmolLM3-3B.litertlm` — never match `<|im_end|>` in the prompt (reported upstream as [litert-torch #1205](https://github.com/google-ai-edge/litert-torch/issues/1205); repro and a measured converter patch in [`tools/tokenizer_parity/`](tools/tokenizer_parity/)). Two OLMo-2 bundles carried a re-serialized `tokenizer.json` with the GPT-2 default pre-tokenizer instead of the model's regex, and PaddleOCR-VL's vendor SentencePiece model typed `</s>` as a control symbol the template writes as text.

**Fix = replace only the tokenizer section**, everything else byte-identical (checked per section):

```
# SP-converted bundles and OLMo-2: embed the upstream tokenizer.json (HF tokenizer path)
python tools/tokenizer_parity/swap_tokenizer_section.py in.litertlm <upstream>/tokenizer.json out.litertlm
# PaddleOCR-VL: retype the CONTROL-typed added tokens of the vendor .spiece as USER_DEFINED, then repack
```

Gate on the LiteRT-LM runtime (0.16.1, CPU), old file vs new file: default turn, 7 probe strings, the 223 standalone characters U+00A1–U+017F and every special token tokenize identically to the upstream tokenizer; prefill token count equals the upstream count; the ASCII-only test questions whose ids do not change answer byte-identically old→new; VLM bundles caption an image. Files: InternVL3-1B/2B, InternVL3_5-1B/2B/4B, LLaVA-OneVision-0.5B, Mage-VL, Ministral-3-3B-Reasoning-2512, Ovis2.5-2B, Polaris-4B-Preview, Qwen2-VL-2B, SmolLM3-3B (`SmolLM3-3B.litertlm`), SmolVLM2-2.2B/500M, PaddleOCR-VL-1.6, OLMo-2-1B-Instruct (litert-community and mlboydaisuke). On-device rows on the cards were measured on the previous files; the runtime's tokenizer code is the same on macOS and on device, and the weights and graph are byte-identical.

**Recipe change going forward:** `FORCE_SPM` is retired for byte-level BPE vocabularies — export with the HF tokenizer path (the default), or, if a SentencePiece section is required, convert with `tools/tokenizer_parity/fix_candidate_lib.py` and gate the built bundle with `engine.tokenize` parity on specials, accented characters and emoji.

## Running the converted models on-device (Android)

- **[docs/android-npu.md](docs/android-npu.md)** — Qualcomm NPU (HTP) with on-device JIT
  compilation: the 10 runtime libraries, the two Environment options, the silent-CPU-fallback
  trap, and the SRQ export recipe for LLM bundles (verified on SM8850 / Hexagon v81).
- **[docs/android-gpu.md](docs/android-gpu.md)** — GPU: the AI Edge Gallery import toggle
  (default is CPU-only), the LiteRT-LM API path, and what to expect (prefill gains,
  decode ≈ CPU, ~2× model size in RAM).

## Verified by running (2026-07-07)

Every recipe below was **actually executed** and gated (`verify_quality.py`, 8-question gate, max-tokens 2048):
**16/18 reproduce + pass the gate.** The two exceptions are not recipe errors:
- **`smollm3-3b`** — reproduces + answers **8/8 correct**, but the gate flags one verbose thinking-model
  answer as degenerate (a gate-strictness artifact on the `smollm3_think` template; the shipped model is
  GSM8K 0-pt parity). Recipe is correct.
- **`fastcontext-4b`** — the source repo `microsoft/FastContext-1.0-4B-SFT` is now **private/gated**; it
  can't be re-downloaded without an HF token. Recipe is verbatim-confirmed from `reports/fastcontext-4b-parity.md`.
- **`qwen25-3b`** — the *original* recipe (`gptqrec` dequant-recovery) was found to **fail on the current
  `ai_edge_quantizer`** during this run; switched to the version-robust **`BMIX4_128`** path (now passes 8/8).
- `falcon3-3b` passes 7/8 (its int4 is known-not-parity; shipped withheld). `r1-distill-qwen-1.5b` passes 6/8
  (shallow 1.5B). All others pass 8/8.

**Recipe legend:** `BOCTAV4` = blockwise-32 int4 + OCTAV + int8 embedding (best quality, Mac/Android) ·
`BOCTAV4_128` = blockwise-128 variant (iPhone / 4B, fits the ~2 GiB section) · `BMIX4[_128]` = blockwise
int4 min-max (no OCTAV) + int8 embedding. **Env:** `FORCE_SPM` (BPE→SP tokenizer; auto-enables
`FIX_ADDED_TOKENS` for `<think>` models) · `EXTERNALIZE_EMBEDDER` (split embedding so 3B+ loads on iPhone) ·
`PHI3_STATIC_ROPE` (Phi longrope→static) · `GPTQREC_GCD_FIX` (GPTQ ingest) · `CACHE` / `PREFILL`.

**Template safety:** every recipe here exports with `use_jinja_template=False` — the script swaps the vendor's HF chat template for a minimal ChatML one (`templates/*.jinja`), and the converter packs it as plain prefix/suffix turn markers, so the bundle embeds **no Jinja at all**. A plain litert-torch export instead defaults to `use_jinja_template=True` and embeds the vendor template verbatim. On older runtimes that was a crasher: vendor templates often call Python-style methods (`.get()`, `.startswith()`) the old minijinja renderer didn't implement, so the bundle imported fine and died on the first message (an earlier Edge Gallery build: `Failed to apply template: unknown method: map has no method named get`). Current runtimes render these templates. Measured 2026-08-24 with a `.startswith()`/`.split()`-heavy vendor template: litert-lm 0.16.0 CLI and litert-mac-verify on Mac, and a 2026-08-21 `litert_lm_main` build on Android CPU all applied the template (prefill length matches the templated prompt) and answered correctly; `strftime_now()` also renders on 0.16.0. Treat embedded Python-method Jinja as a warning for old runtimes, not a blocker; the current Edge Gallery build was not retested. Inspect any bundle's template with `pip install litert-lm-builder && python -m litert_lm_builder.litertlm_peek_main --litertlm_file model.litertlm`.

## Single-command models

| key | HF source | template | quant | env | shipped to |
|---|---|---|---|---|---|
| `fastcontext-4b` | microsoft/FastContext-1.0-4B-SFT | chatml_simple | BOCTAV4 | EXTERNALIZE_EMBEDDER, CACHE=4096 | litert-community/FastContext-1.0-4B-SFT |
| `nanbeige4.1-3b` | Nanbeige/Nanbeige4.1-3B | chatml_simple | BOCTAV4 | FORCE_SPM, EXTERNALIZE_EMBEDDER, CACHE=4096 | litert-community/Nanbeige4.1-3B |
| `nanbeige4.2-3b` | Nanbeige/Nanbeige4.2-3B | chatml_think | BOCTAV4 | FORCE_SPM, EXTERNALIZE_EMBEDDER, CACHE=4096; dedicated `convert_nanbeige42.py` (looped transformer: 22 shared-weight layers ×2 loops → 44 KV slots) | litert-community/Nanbeige4.2-3B |

#### Nanbeige4.2 on a GPU: one TOML line, and it is not the graph

`Nanbeige4.2-3B` shipped CPU-only because its GPU output was a flood of `<unk>` on every backend tried. The
graph was never the problem — it delegates in full, 4716 of 4716 ops, on Adreno and on Metal alike. What
breaks is the **activation dtype**: at fp16 the unrolled loop's second pass reads the graph input instead of
the first pass's output, so the model computes from nothing. Declaring fp32 activations fixes it with the
weights untouched:

```bash
python scripts/set_activation_type.py model.litertlm model_fp32act.litertlm --type fp32
```

Published as `model_fp32act.litertlm` alongside the original. Measured the same day on the same machines:
Apple M4 Max Metal 409 tok/s prefill and 39.7 decode against 57 and 12.9 on CPU — a GPU win both ways; on a
Galaxy S26 the GPU decodes at 3.9 tok/s against the CPU's 3.45, a wash, with the GPU's only clear advantage
being peak memory (2590 MB against 4152 MB).

Two traps this lane paid for, both worth copying:

- **`litert-lm benchmark` reports throughput for a backend that produced no text.** The published card carried
  556 / 52.6 tok/s for the file that emits `<unk>`; the number is real and meaningless. Confirm generation
  before quoting any figure.
- **Neither file starts on an iPhone 17 Pro** at a 512-token budget under an 8-question on-device harness: the
  GPU leg fails at state-buffer allocation and the CPU leg is killed by the OS after engine init. Both fail
  identically, so fp32 activations are not the cause — a 3B at this cache size does not fit that app's budget.
  A smaller cache or a lower token budget may well succeed; this is one configuration, not a verdict on the
  device.
| `olmo2-1b` | allenai/OLMo-2-0425-1B-Instruct | olmo2_simple | BOCTAV4 | CACHE=4096 | mlboydaisuke/OLMo-2-1B-Instruct-LiteRT |
| `olmo2-7b` | allenai/OLMo-2-1124-7B-Instruct | olmo2_simple | BOCTAV4 | CACHE=4096, EXTERNALIZE_EMBEDDER | *(desktop-only, not published — >2 GiB section)* |
| `polaris-4b` | POLARIS-Project/Polaris-4B-Preview | qwen3_think | BOCTAV4_128 | **FORCE_SPM**, EXTERNALIZE_EMBEDDER, CACHE=4096 | litert-community/Polaris-4B-Preview |
| `qwen3-1.7b` | Qwen/Qwen3-1.7B | qwen3_think | BOCTAV4 | CACHE=4096 | mlboydaisuke *(dropped→private)* |
| `qwen3-4b-thinking` | Qwen/Qwen3-4B-Thinking-2507 | qwen3_think | **BOCTAV4_128** | EXTERNALIZE_EMBEDDER, CACHE=4096 | litert-community/Qwen3-4B-Thinking-2507 |
| `r1-distill-qwen-1.5b` | deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B | deepseek_r1_simple | BOCTAV4 | CACHE=4096 | mlboydaisuke/DeepSeek-R1-Distill-Qwen-1.5B-LiteRT |
| `r1-distill-qwen-7b` | deepseek-ai/DeepSeek-R1-Distill-Qwen-7B | deepseek_r1_simple | BOCTAV4 | CACHE=4096, EXTERNALIZE_EMBEDDER | mlboydaisuke/DeepSeek-R1-Distill-Qwen-7B-LiteRT *(desktop)* |
| `smollm3-3b` | HuggingFaceTB/SmolLM3-3B | smollm3_think | BOCTAV4 | CACHE=4096, EXTERNALIZE_EMBEDDER | mlboydaisuke/SmolLM3-3B-LiteRT |
| `twil-lm3` | webAI-Official/TwIL-LM3 | smollm3_think | BOCTAV4 | CACHE=4096, EXTERNALIZE_EMBEDDER | mlboydaisuke/TwIL-LM3-LiteRT ⚠ non-commercial |
| `jan-nano` | Menlo/Jan-nano | qwen3_think ⚠ | BOCTAV4_128 | EXTERNALIZE_EMBEDDER, CACHE=4096 | litert-community/Jan-nano |
| `vibethinker-3b` | WeiboAI/VibeThinker-3B | chatml_simple | **BOCTAV4** (block32 ONLY) | CACHE=4096, EXTERNALIZE_EMBEDDER | litert-community/VibeThinker-3B |
| `falcon3-3b` | tiiuae/Falcon3-3B-Instruct | falcon_simple | BMIX4_128 | CACHE=2048 | *(withheld/private — int4 ≠ parity)* |
| `llama32-3b` | meta-llama/Llama-3.2-3B-Instruct | llama_simple | BMIX4 | EXTERNALIZE_EMBEDDER, CACHE=4096 | mlboydaisuke/Llama-3.2-3B-Instruct-LiteRT |

## Models with a prep step (the runner does it automatically)

| key | prep | then export |
|---|---|---|
| `ministral3-3b` | `extract_ministral3_text.py mistralai/Ministral-3-3B-Instruct-2512 → src_models/ministral3-3b-text` (drop pixtral vision) | mistral_simple, BOCTAV4, EXTERNALIZE_EMBEDDER, CACHE=4096 → litert-community/Ministral-3-3B-Instruct-2512 |
| `ministral3-3b-reasoning` | `extract_text_backbone.py mistralai/Ministral-3-3B-Reasoning-2512 → src_models/…-reasoning-text` | mistral_simple, BOCTAV4, **FORCE_SPM**, EXTERNALIZE_EMBEDDER, CACHE=4096 → litert-community/Ministral-3-3B-Reasoning-2512 |
| `phi4-mini-reasoning` | download microsoft/Phi-4-mini-reasoning, set `config.sliding_window=None` | phi_simple, BOCTAV4, **PHI3_STATIC_ROPE**, EXTERNALIZE_EMBEDDER, CACHE=4096 → litert-community/Phi-4-mini-reasoning |
| `qwen25-3b` | `ingest_gptq_dequant.py Qwen/Qwen2.5-3B-Instruct-GPTQ-Int4 Qwen/Qwen2.5-3B-Instruct … fp32clip` (dequantize GPTQ) | chatml_simple, **BMIX4_128**, CACHE=4096 → mlboydaisuke *(NC license, personal only)* |

## Caveats / lower-confidence fields (verify against the card + memory before quoting)

- **`jan-nano` template** ⚠ — no source names it. It's a Qwen3-4B thinking model, so `qwen3_think.jinja`
  (matches the sibling Qwen3-4B-Thinking and the reasoning-template-parity note); `chatml_simple.jinja`
  (what same-base FastContext used) is the alternative. Swap and re-gate if output rambles.
- **`polaris-4b` FORCE_SPM** — added **2026-08-25**; the row shipped without it, and without it the
  tokenizer section does not match the published file. The evidence is in the published bundle's own
  header, which anyone can read with a ranged HTTP request: it carries an `SP_Tokenizer` section, and
  `FORCE_SPM` is the only thing in `export_simple_template.py` that produces one from a BPE source (it
  rebinds `export_lib.export_tokenizer`). Checked against four published controls read the same way —
  `Nanbeige4.1-3B` and `Ministral-3-3B-Reasoning` (FORCE_SPM in recipe) carry `SP_Tokenizer`;
  `Qwen3-4B-Thinking` and `FastContext-4B` (no FORCE_SPM) carry `HF_Tokenizer_Zlib`. `jan-nano` carries
  `HF_Tokenizer_Zlib`, so its row is right as written.
- **`polaris-4b` and `jan-nano` carry one post-export step** that these recipes do not perform: on
  2026-08-22 both published files were repacked to declare a thought channel, so that a reasoning model's
  `<think>...</think>` chain is addressable instead of streaming inline into the answer (without
  `LlmMetadata.channels` a thinking budget is silently ignored). The step adds a `channels` block naming
  the `<think>` / `</think>` markers and repacks; it is metadata-only — every tflite section comes out
  byte-identical — so quality and speed are unaffected, but the published file's sha256 differs from what
  these recipes produce. Bundle `creation_timestamp` reads 2026-08-22 for both.
- **`llama32-3b`** was originally exported through the **official litert-torch main** (BPE patch upstream) —
  re-running with the current default env reproduces the same `BMIX4` recipe; expect equivalent, not bit-identical.
- **`vibethinker-3b`** needs a runtime stop-token fix (`generation_config.eos_token_id=[151643,151645]` so
  `<|im_end|>` ends ChatML turns) and is **block32-only** (block128 collapses 90→64% GSM8K).
- **`qwen3-4b-thinking`** is **block128-only** (block32 corrupts on iPhone GPU). Reasoning models: eval at
  max-tokens ≥ 2048 or int4 falsely looks degraded.
- **Ministral source ids** — the extract scripts accept an HF id or a local dir; if the plain id 404s, use the
  `*-BF16` variant of the repo (the bf16 weights the extraction was run on).
- **`qwen25-3b` GPTQ path** — the original ship used `recipes/gptqrec_int4_block128.json` + `GPTQREC_GCD_FIX`
  (dequantized-weight-recovery), but the **current `ai_edge_quantizer` rejects blockwise + recovery**
  (`dequantized_weight_recovery.py:163`). The runner therefore uses **`BMIX4_128`** on the `fp32clip`
  dequantized checkpoint — the ingest script's own docstring path (min-max lands the grid on the ±7 levels
  → same recovery, version-robust). Use the recovery recipe only on an ai_edge_quantizer that supports it.
- **Not published** (reproducible, but were withheld): `olmo2-7b` / `r1-distill-qwen-7b` (desktop-only size),
  `falcon3-3b` (naive int4 ≠ bf16 parity), `qwen3-1.7b` (dropped), `qwen25-3b` (Qwen NC license → personal
  namespace only).

## Vision-language models (`fast_vlm`)

VLMs reproduce via `bash scripts/reproduce_vlm.sh <key>` (each runs a `ship_*.sh` that downloads the
source, converts the vision encoder+adapter and the decoder — int4 unless noted — and assembles the
bundle). Details per model in `cards/<name>-litert.md`.

| key | vision | decoder | ship script |
|---|---|---|---|
| `granite-docling-258m` | SigLIP-512 (SmolVLM2 rail, 64 tok) | granite Llama 576h/30L (**wi8-float** — int4/int8-int corrupt DocTags), cache 4096 | `ship_granite_docling.sh` |
| `internvl3-1b` | InternViT-448 | Qwen2.5-0.5B | `ship_internvl_1b.sh` |
| `internvl3.5-1b` / `-2b` / `-4b` | InternViT-448 | Qwen3-0.6B / 1.7B / 4B | `ship_internvl3_5_{1b,2b,4b}.sh` |
| `llava-onevision-0.5b` | SigLIP-384 (730 tok) | Qwen2-0.5B | `ship_llavaov.sh` |
| `mage-vl` | **static-448, no GATHER_ND** (196 tok) | Qwen3-4B **int4-b128**, cache 2048 | `ship_magevl.sh` |
| `north-micro-vision` | **static-512, deepstack folded, no GATHER_ND** (256 tok) | Cohere2-rehost 2B **int8** + in-bundle `fp32_fp16` hint | `ship_northmv.sh` |
| `ovis2.5-2b` | **static-NaViT-512** (256 tok) | Qwen3-1.7B | `ship_ovis_2b.sh` |
| `paddleocr-vl-1.6` | **static-NaViT-560** (400 tok) | ERNIE-4.5-0.3B (**fp16** — int4/int8 corrupt OCR) | `ship_paddleocr_vl.sh` |
| `qwen2-vl-2b` | **static-672, no GATHER_ND** (576 tok) | Qwen2-1.5B int4 | `ship_qwen2vl_2b.sh` |
| `smolvlm2-500m` / `-2.2b` | SigLIP + pixel-shuffle | SmolLM2 / SmolLM2-1.7B | `ship_smolvlm2{,_22b}.sh` |
| `wevisone-vl-2b` | **static-512, DeepStack kept in an 8,196-wide embedding + image flag, no GATHER_ND** (256 tok) | the Qwen3-VL 2B text decoder, **fp16** weights, M-RoPE derived in the graph | `ship_wevisone.sh` |

`granite-docling-258m` is IBM's document-conversion VLM (the Docling model): page image → DocTags markup (layout + OTSL table structure + formulas), convertible to Markdown/HTML with docling-core. It rides the SmolVLM2 rail **unchanged** (idefics3's vision tower is architecturally SmolVLM2's). Three things its scripts encode. (1) **Quantization: wi8-float only** — int4 and integer-compute int8 corrupt DocTags structure on the 258M decoder (PaddleOCR-VL pattern); int8 weights with float compute keep the table gate exact. (2) **The bundle jinja protects the newline after `<|end_of_text|>`** (`{% endfor -%}`, no leading trim): the model is format-exact, and a `{%-` that eats that one `\n` token turns every image turn into a hallucinated blank page — at every decoder precision, while text-only turns stay fine. (3) **App contract: pre-resize pages to 512×512 BILINEAR** before sending. The base model is trained with tiling; the single-global-512 path this bundle runs is resampling-sensitive *in eager too*, and the runtime's own downscale from larger inputs lands on the bad side. Backend is CPU (the GPU delegate rejects the quantized 576×576 FC at kernel init on macOS WebGPU and Android OpenCL alike; an fp16 decoder runs on Android OpenCL but ~4× slower than CPU). Device-verified: Galaxy S26 CPU converts the synthetic 5×6 table page exactly (25/25 cells, byte-identical across runs) at 35.4 s/page.

`qwen2-vl-2b` is the general-purpose Qwen2-VL VLM (describe / VQA / OCR). Two gotchas baked into its scripts: (a) reordering patches into the merger's 2×2-block order with a gather emits a `GATHER_ND` op that the mobile GPU delegate can't compile (the vision executor then fails to create on-device) — so the encoder keeps raster order and the 2×2 merge is done with strided slices + concat in the adapter; (b) the fast_vlm runtime feeds 1-D positions (no M-RoPE), which preserves describe/VQA/OCR/count but degrades cross-cell *ranking* over 2-D tables.

**Qwen2-VL derivative intake** (`scripts/ship_qwen2vl_derivative.sh`, measured 2026-08-25 on
[numind/NuExtract-2.0-2B](https://huggingface.co/numind/NuExtract-2.0-2B) — 32k downloads, the
top generative Qwen2-VL-2B derivative). Two findings, one of them a wall:

- **Family fact — VLM derivatives train the vision tower too.** Per-tensor diff vs the base:
  339/391 `visual.*` and 315/338 `model.*` tensors differ. Unlike every LLM family measured,
  a derivative conversion re-exports vision from the derivative's own weights (the ship script
  and `convert_qwen2vl_vision.py`'s `MODEL` env encode this). The rails otherwise carry over
  unchanged: decoder re-host is bit-exact (`text-only logits maxdiff=0.000000`), vision
  static-672 export fp32 corr 0.99999998 / int8 end-to-end corr 0.896 — the same 0.90-class
  int8 behavior as the shipped base. Tokenizer files differ only by serialization era
  (added_tokens.json split out, a `#version` merges header; probe tokenization identical).
- **The wall was ours, not the runtime's** (reported as a runtime defect 2026-08-25, retracted on
  LiteRT-LM #3348 on 2026-08-27, corrected here 2026-09-04). The measured facts stand: bundle
  greedy output on a specials-free prompt is byte-identical to HF fp32 for 500+ tokens, and the
  moment the prompt carries `<|im_start|>`/`<|im_end|>` the bundle answers `{"names": []}` where
  HF extracts correctly. The attribution did not: every exclusion experiment was judged by
  comparing greedy outputs, and that measure is degenerate — several *different* wrong-id
  substitutions all reproduced the bundle's output, which was over-read as localizing the
  runtime's encode path. Rebuilt against the published `litert-community/Qwen2-VL-2B` (a fast_vlm
  bundle) with a discriminating measure — teacher-forced `run_text_scoring` after prefill, plus
  prefill-token counts against `engine.tokenize` — the runtime encodes every ChatML special to its
  single correct id on the templated and the raw path alike, and every marker substitution moves
  the score (harness and numbers are on the issue). The runtime's llm / fast_vlm / `tokenize`
  paths share one encode call. What was defective is the bundle's own SentencePiece tokenizer
  section, written by litert-torch's BPE→SentencePiece conversion (litert-torch #1205: no byte
  fallback, pad/eos typed as the UNK piece). `scripts/gate_specials.py` now measures exactly
  that — the engine's encode against the upstream `tokenizer.json` on every added token alone and
  mid-string, plus Latin-1, Ext-A, emoji, CJK, digits and whitespace probes (litert-lm 0.16.0):

  | bundle | tokenizer section | result |
  |---|---|---|
  | `litert-community/Qwen2-VL-2B` as published before 2026-08-30 (sha256 `a9fd5053…`) | SP, converted | **FAIL 6/33** — `<|endoftext|>` splits into 7 tokens, `é` → byte id 165, `ă` → 151643 (it *becomes* `<|endoftext|>`), leading whitespace differs; `<|im_start|>` / `<|im_end|>` correct |
  | `litert-community/Qwen2-VL-2B` as published now (sha256 `cf481776…`) | HF `tokenizer.json` | **PASS 33/33** |
  | gemma-3-270m re-export (SP-native vocab, 6,417 added tokens) | SP, vendor | **PASS 12838/12838** |

  The NuExtract bundle was deleted before the retraction, so which of the #1205 defects hit it
  is inference (its tokenizer serialization splits the added tokens out into `added_tokens.json`,
  the shape that loses them). `ship_qwen2vl_derivative.sh` now ends with this gate, and the note
  above its step 5 gives the HF `tokenizer.json` rebuild for when it fails. The builder also no
  longer writes `start_token = "None"` — that literal went out as the word `None` at the head of
  every prompt on ten published fast_vlm bundles (BOS audit, 2026-08-27). **Verdict: Qwen2-VL
  derivative conversions are mechanically complete; a format-exact derivative ships once it
  passes both `gate_specials.py` and its task gate (`qwen2vl_work/gate_nuextract.py`).
  NuExtract-2.0-2B has not been re-converted and remains unpublished.**

`north-micro-vision` is Cohere's North-Micro-Vision-Instruct (2.48B, apache-2.0) — the first Cohere-family model on this rail, and its vision tower is the Qwen3-VL design (SigLIP2-SO400M dims, 27 blocks, three *deepstack* mergers). Three things the scripts encode. (1) **Deepstack folds into the single fast_vlm image embedding**: HF adds three extra vision embeddings to the residual stream after decoder layers 0/1/2; the fast_vlm contract carries one embedding, so the encoder emits `concat(h27, h8, h16, h24)` and the adapter computes `merger(h27) + Σ deepstack_merger_i(h_i)` — exactly representable, and the ablation (`northmv_work/phase0_deepstack_ablation.py`, `phase0_teacher_forced.py`) shows fold beats drop on every metric (teacher-forced top-1 0.96 fold / 0.93 fold+1-D positions vs the released model, no collapse). (2) **The decoder re-hosts as `Cohere2ForCausalLM` with a rope-layout patch** (`northmv_work/northmv_rope_patch.py`): CohereCompass rotates Llama-style half-split pairs while stock Cohere2 rotates interleaved pairs; patching Cohere2 to the Llama-style math loads the weights verbatim (text logits maxdiff 0.0) *and* replaces the `BROADCAST_TO` + 5-D `CONCATENATION` that stock Cohere2's rope lowers to — both rejected by the GPU delegate. (3) **Vision GPU rules learned the hard way** (all in `convert_northmv_vision.py`): keep every activation rank ≥3 with a leading batch dim (the Metal delegate computes rank-2 elementwise ops against a rank-2 constant *wrong*, silently — the symptom is "this image is a blank canvas"); pre-scale LayerNorm inputs by a calibrated power of two from block 9 on (register-scale activations |x|≈2000 overflow fp16 (x−m)²); and a `clamp` between LN and the following FC stops the converter folding LN-gamma into the int8 weight. Ship quant is int8 vision + int8 (dynamic) decoder with `prefer_activation_type="fp32_fp16"` declared on the decoder section — Mali-class GPUs accumulate fp16 and overflow at the 256 image-token positions, so an fp16 decoder answers image turns as if blind (echoes the question) while text-only stays perfect; the in-bundle declaration fixes it on every runtime without flags. Vision fp16 is desktop-only (compiling it on a Mali GPU hard-crashes the phone). Tokenizer: bundle the repo's own `tokenizer.json` bytes at a pinned revision — the sentencepiece conversion crashes on null-byte pieces, and `AutoTokenizer.save_pretrained` rewrites the file with transformers' `CohereTokenizer` pipeline, which splits every digit while the trained tokenizer groups digits in threes. Files built before 2026-10-15 carried that per-digit section; the published files were fixed by a tokenizer-section swap (`tools/tokenizer_parity/swap_tokenizer_section.py`). The runtime's 1-D positions replace M-RoPE, with the same 2-D-table-ranking caveat as `qwen2-vl-2b`. Needs transformers ≥5.16 (`cohere_compass`) for the vision/prep side and the released litert-torch 0.9.3 stack for the decoder export.

`mage-vl` is Microsoft's 4.7B general VLM (describe / VQA / full-page OCR). Its vision tower is the Qwen2-VL design, but two things make it *simpler* to convert: `temporal_patch_size=1` (the patch-embed already is a stride-16 `Conv2d` — no Conv3d fold) and the text decoder is a **stock Qwen3-4B fed plain 1-D positions**, so the fast_vlm runtime contract is mathematically exact (no M-RoPE caveat at all — the 2-D-table-ranking limitation of `qwen2-vl-2b` does not apply). Its 3-D rope splits head_dim 4:6:6 over (t,h,w) with *interleaved* rotation (not half-split); the scripts precompute it from raster positions with t=0 and verify the whole patch pipeline bit-identical against the model's own processor. Cache is 2048 on purpose: at 4096 the 36-layer/kv8 fp32 KV cache is ~1.2 GB, which more than doubles the on-device session footprint; 2048 keeps the whole phone session ~1.5 GiB. Env gotchas baked into the ship script: python 3.14 breaks torchao's import (use ≤3.13), and torchvision must be the torch-matched pair (`torch==2.12.1 torchvision==0.27.1`) or pip upgrades torch past litert-torch's pin. `magevl_work/precheck_magevl_vision.py` runs the full static-rewrite export precheck on a random-init tower — no checkpoint download — in minutes.

`paddleocr-vl-1.6` is the OCR/document-parsing specialist (task prompts `OCR:` / `Table Recognition:` /
`Formula Recognition:` / …). Two conversion gotchas are baked into its scripts: transformers ≥5.12 loads
remote-code rotary modules with a ZEROED non-persistent `inv_freq` (fix: `rope_init()` after load — and
validate against the native `paddleocr_vl` port, not the remote code), and its 0.36B decoder must ship
fp16 (`RECIPE=WF16`): int4 and dynamic-int8 both measurably corrupt transcription.

VLM quality is gated on vision end-to-end corr (≈1.0 fp32) + eager image grounding, not the 8-question
text gate (image input is device-only on this toolchain). `internvl3-2b` has a card but is reproduced by
adapting `ship_internvl_1b.sh` (model id + dims) — no dedicated script.

## Text-to-speech (Qwen3-TTS, host-loop tflite)

`qwen3tts_work/` converts [Qwen/Qwen3-TTS-12Hz-0.6B-Base](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base) (Apache-2.0 speech LM, 10 languages, x-vector voice cloning) into three LiteRT graphs plus host tables, and includes the runnable host-side pipeline. Published artifacts: [litert-community/Qwen3-TTS-12Hz-0.6B-Base](https://huggingface.co/litert-community/Qwen3-TTS-12Hz-0.6B-Base). This one is **not** a `.litertlm` — the speech-LM decode loop (16-codebook embedding sum, 15-step inner AR sub-loop per frame, PCM output) is outside the LiteRT-LM Engine's text loop, so the graphs run under a Compiled Model host loop instead.

```bash
cd qwen3tts_work
# reference dumps (env with qwen-tts==0.1.1, transformers==4.57.3):
python dump_talker_ref.py && python dump_mtp_ref.py && python dump_codec_ref.py
# conversion + verification (env with litert-torch 0.9.1, transformers 5.12.x):
python extract_talker_ckpt.py && python export_talker.py && python verify_talker.py
RECIPE=BOCTAV4 python export_talker.py          # int4 variant
python export_mtp.py && python export_codec.py && python extract_host_tables.py
python assemble_release.py                      # -> out/release/ (published layout)
# synthesize (auto-downloads the published models if out/release is absent):
python synthesize.py --text "Hello from LiteRT." --output hello.wav --model_dir out/release
```

Gates: talker tflite corr 1.0 / top-1 100% (and the synthesized Qwen3 checkpoint is bit-exact vs the talker — the TTS mrope reduces to standard RoPE); MTP 15/15 greedy tokens; codec corr 1.0; end-to-end with `--talker fp32 --greedy` = token-for-token vs the PyTorch reference, waveform corr 1.0, ASR round-trip exact. Known trap: channelwise int8 (tooling default) degenerates — use `RECIPE=BOCTAV4` (blockwise-32).

**Fast MTP (single-graph fold + dynamic int8).** The default `export_mtp.py` graph is a decode step run 17× per audio frame — weight-streaming-bound and the whole pipeline's bottleneck (M4 Max RTF ≈ 2.5). `export_mtp_folded.py` folds all 16 inner steps × 5 layers into ONE graph (in-graph argmax + embedding-table gather, KV kept internal): token-identical (52/52 e2e frames), 41 ms/frame vs 148 (M4 Max), end-to-end RTF 2.18. Data-free int8/int4 on the fold both fail (blockwise-int8 is a no-op; int4 collapses — the 15 lm-heads can't survive 4 bits); calibrated static int16 runs away. What works is **GPTQ int8 → dynamic int8 (DRQ)** so the weights stay int8 in RAM:

```bash
# calibration frames from the fp32 pipeline (extend the sentence list as needed):
TEXT="The quick brown fox jumps over the lazy dog." DUMP_MTP=out/qwen3tts-mtp/calib/calib_1.npz \
  python hostloop_e2e.py
python export_mtp_folded.py                                   # fp32 fold (desktop-exact, 559 MB)
BITS=8 GROUP=32 TAG=int8g32 python gptq_mtp_folded.py         # torch GPTQ (ai-edge-quantizer's is a stub)
python quantize_mtp_folded_v2.py fp16                         # or a plain fp16 fold (desktop only)
```

`gptq_mtp_folded.py` bakes the GPTQ int8 grid; a `dynamic_wi8_afp32` re-quant of that export (`recipe.dynamic_wi8_afp32`) yields a 218 MB graph whose FC weights are int8 in RAM (no explicit-dequantize — that trap re-materializes fp32 at load and defeats the size win). Gate = frame count + no runaway + ASR round-trip (not token match; the int trajectory diverges yet stays intelligible). On a Pixel 8a this cuts the MTP from ≈333 to ≈68 ms/frame (~5×) and the end-to-end RTF from ≈6.7 to ≈3.3; the codec decoder then dominates. `verify_codec_chunking.py` documents a related limit: the fixed-T codec graph's left-context chunking is only seam-clean up to that T, so long utterances want a larger-T codec export.

**Fast codec (mixed-precision split).** With the MTP folded, the codec decoder is the dominant stage (FLOP-bound: upsampling convs). ai-edge-quantizer can't int8 those convs (CONV_2D axis-remap error, TRANSPOSE_CONV unsupported — only the small transformer quantizes), and a plain fp16 cast is a load-time no-op. What works is running the graph with **XNNPACK FORCE_FP16** (`CpuOptions(xnnpack_flags=4)`), ARM-native fp16, no requant. Global FORCE_FP16 is ~2.2× but breaks quality — the 8-layer pre-transformer's large-magnitude activations don't survive fp16 (ASR unintelligible). `export_codec_split.py` splits the decoder at that boundary into Part A (RVQ + pre_conv + pre_transformer, run fp32) and Part B (upsample + SEANet convnet — nearly all the FLOPs, run FORCE_FP16). Part B in fp16 keeps the full 2.2× while waveform corr recovers to 0.997 and ASR is identical to fp32:

```bash
python export_codec_split.py                                 # -> codec_partA/partB (T=64) + bench
CODEC_SPLIT=1 python hostloop_e2e.py                         # end-to-end with the split codec
```

The host loop runs Part A (fp32) → hidden [1,512,T] → Part B (fp16) → PCM. On a Pixel 8a the codec drops from ≈114 to ≈40 ms/frame (2.5×); combined with the folded int8 MTP the end-to-end **RTF falls from ≈6.7 to ≈2.06** (~3.2×), ASR-lossless. Gate the codec on the "Hello…" e2e_ref codes (`hostloop_e2e.py` writes the waveform) — the standalone `codec_equiv_ref` clip isn't speech and won't ASR. `quantize_codec.py` records the int8/fp16 attempts that don't work.


## Audio8-TTS-Preview-0.6b (DualAR speech LM, 44.1 kHz codec, zero-shot voice cloning) — host-loop tflite set

`audio8_tts_work/` converts [Edge0/Audio8-TTS-Preview-0.6b](https://huggingface.co/Edge0/Audio8-TTS-Preview-0.6b) (Apache-2.0, 601M + 337M codec, 11 languages) into four LiteRT graphs plus a Python host loop. The architecture is Fish-Audio-S2-Pro-style DualAR: a 24-layer "slow" transformer (Qwen2.5-0.5B geometry) emits one semantic token per 46 ms frame, a 4-layer "fast" transformer fills the frame's 10 codebooks one at a time, and a causal codec (RVQ + 8-layer windowed transformer + SEANet upsampler) turns frames into 44.1 kHz audio.

```bash
cd audio8_tts_work
# 1. oracle dumps (env with transformers 4.57.x = the vendor's family; soundfile, scipy):
python oracle_ref.py                                   # 14 seeded cases -> out/oracle/*.npz + .wav, fixtures/ref_*_codes.npy
# 2. port parity, then export (env with litert-torch 0.9.4, ai-edge-litert 2.2.0, ai-edge-quantizer):
python verify_port.py 3                                # torch port vs oracle: slow/fast logits ~3e-5, codec bit-exact
PREFILL=256 SUFFIX=_p256 python export_slow.py         # slow AR: prefill_256 + decode, KV 2048, fp32 (2.15 GB)
SUFFIX=_v3 python export_fast.py                       # fast AR step graph, fp32
T=128 SUFFIX=_g2 python export_codec.py && T=192 SUFFIX=_g2 python export_codec.py
SEC=10 SUFFIX=_g2 python export_codec_encoder.py       # voice registration encoder (10 s bucket)
# 3. quantize (post-hoc ai-edge-quantizer) and gate
python quantize_ar.py out/slow/slow_fp32_c2048_p256.tflite drq8     # dynamic int8 (ships)
python quantize_ar.py out/slow/slow_fp32_c2048_p256.tflite bo4      # blockwise-32 OCTAV int4 (small option)
python quantize_ar.py out/fast/fast_fp32_v3.tflite drq8
python quantize_codec.py out/codec/codec_decoder_fp32_T128_g2.tflite fp16   # exact; GPU-capable
python quantize_codec.py out/codec/codec_decoder_fp32_T192_g2.tflite fp16
python quantize_encoder.py out/codec/codec_encoder_fp32_10s_g2.tflite fp16
T=128 SUFFIX=_g2 python export_codec_native_i8.py && python fix_native_i8_tables.py \
  out/codec/codec_decoder_i8native_T128_g2.tflite out/codec/codec_decoder_i8nativefix_T128_g2.tflite   # CPU int8 codec
python verify_tflite_slow.py out/slow/slow_drq8_c2048_p256.tflite 3 && python verify_tflite_fast.py out/fast/fast_drq8_v3.tflite 3
SLOW=out/slow/slow_drq8_c2048_p256.tflite FAST=out/fast/fast_drq8_v3.tflite CODEC=out/codec/codec_decoder_fp16_T128_g2.tflite TAG=q1 python hostloop_e2e.py
python gate_audio.py asr out/e2e/q1; python gate_audio.py spk out/e2e/q1; python gate_audio.py report out/e2e/q1   # whisper turbo / TitaNet
python assemble_ship.py out/ship                       # published layout (+ voices/, tokenizer.json, host loop)
python out/ship/audio8_tts_litert.py --model-dir out/ship --text "Hello from LiteRT." --voice out/ship/voices/en_librispeech_1272 --out hello.wav
```

Gates: fp32 graphs reproduce the PyTorch oracle's code sequence frame-for-frame on all 14 cases (same seeded sampler); codec decoder bit-exact at torch level, 1.9e-6 as .tflite; int8 slow + int8 fast + fp16 codec keeps the oracle's WER (en 1.1%, ja 0.0%, whisper large-v3-turbo) and speaker cosine (TitaNet-L, en 0.68 / ja 0.76 vs oracle 0.66 / 0.74).

Walls and fixes recorded in `audio8_tts_work/FINDINGS.md`: `repeat_interleave` on the KV cache lowers to BROADCAST_TO and costs 615 ms/step (fold GQA heads into the matmul rows: 16 ms); `torch.polar` has no lowering (eager-cached real tables); the vendor's bf16-rounded RoPE table is reproduced exactly by permuting q/k rows to the rotate-half layout; litert-torch's PT2E dynamic mode also quantizes unannotated embedding tables asymmetrically, which TFLite's EMBEDDING_LOOKUP refuses (write the fp32 codebooks back); ai-edge-quantizer's >32 MiB chunked path breaks on 4-D conv weights; every gather index is clamped in-graph so random-input benchmark tools do not fault.

## MiniCPM family (new-style: full-jinja LlmMetadata + post-hoc quantization)

`minicpm_work/` converts the MiniCPM family with the newer packaging used by litert-community/MiniCPM5-1B: the LlmMetadata carries the model's **full `chat_template.jinja` verbatim** plus a `thought` channel (`<think>`/`</think>`), so hybrid-reasoning works natively in LiteRT-LM ≥0.14 (`enable_thinking` via conversation extra context; thinking text arrives on a separate channel). Export unquantized (`--quantization_recipe=""`), then quantize with ai-edge-quantizer, then repackage with `litert-lm-builder`:

```bash
cd minicpm_work
./convert_minicpm.sh minicpm5-1b      # openbmb/MiniCPM5-1B    -> wi8 (int8)
./convert_minicpm.sh minicpm4-0.5b    # openbmb/MiniCPM4-0.5B  -> OCTAV int4-b32 + int8 embed/head
./convert_minicpm.sh minicpm4.1-8b    # openbmb/MiniCPM4.1-8B  -> int4-b32 + int8 embed/head
```

Needs python ≥3.11, `pip install litert-torch litert-lm "transformers==5.6.2"` (litert-torch 0.9.1 breaks with transformers 5.14+). Run/eval with the `litert-lm` CLI or `eval_gsm8k_api.py` (the old prefix/suffix-template harnesses don't apply jinja-only metadata).

What the scripts encode (the non-obvious parts):

- **MiniCPM5-1B** is stock `LlamaForCausalLM`. Its metadata lists 14 `X<|im_end|>\n` **string** stop tokens (`LlmMetaProto.pbtext` has them all, matching the published artifact). *Correction 2026-08-25*: an earlier revision of this note attributed them to "fused merge tokens" in the tokenizer — measured, the tokenizer has **no** vocab entry containing `<|im_end|>` besides the special token itself (id 130073; `.<|im_end|>\n` encodes as 3 tokens). The 14 strings are generated by litert-torch's exporter: the template-derived turn-end stop `<|im_end|>\n` expanded with 13 punctuation prefixes (`_STOP_TOKEN_PREFIXES` in `export_hf/core/litert_lm_builder.py`, a defense against SentencePiece greedy token merging). A stock 0.9.3 export of a MiniCPM5 checkpoint emits the same 16-entry stop set as the published artifact by construction.
- **MiniCPM4/4.1** are custom `MiniCPMForCausalLM` (muP + longrope; remote code needs transformers 4.46 and won't load in 5.x). `prep_minicpm4_as_llama.py` folds them to a stock llama checkpoint (untie; `emb ×scale_emb`; `o/down ×scale_depth/√L`; `lm_head ÷(hidden/dim_model_base)`), and `export_static_longrope.py` strips transformers' `@dynamic_rope_update` (a data-dependent branch torch.export rejects; exact here since long==short factors and factor==1).
- **MiniCPM4 tokenizer**: the HF `tokenizer.json` bundle loses all spaces on decode; the raw `tokenizer.model` bundle lacks the added-token `<|im_end|>`(73440) stop. `fix_sp_added_tokens.py` appends the added tokens to the SP model as USER_DEFINED pieces and the fixed `.spiece` is what gets bundled.
- **Recipe choice (GSM8K n=100, greedy, thinking off)**: MiniCPM5-1B — wi8 **63** vs official artifact 61; data-free int4-b32 lands 48 (both min-max and OCTAV; the official artifact's int4 quantizer is not reproducible from released ai-edge-quantizer, so int8 is the recommended repro target). MiniCPM4-0.5B — bf16 57; OCTAV int4 **50**; min-max int4 collapses to 38 (sub-1B int4 sensitivity), int8 47. MiniCPM4.1-8B — int4-b32 **44/50 (88%)** (8B is int4-robust; desktop-class at 4.9GB).

### 2026-09-08 — MiniCPM5-2B (hybrid thinking, 2.5B): verbatim jinja + zero-scale fix + per-file activation dtype

**Published:** [mlboydaisuke/MiniCPM5-2B-LiteRT](https://huggingface.co/mlboydaisuke/MiniCPM5-2B-LiteRT) — `MiniCPM5-2B_int4.litertlm` (1.55 GB, block-32 int4, zero-scale fix, fp16-default activations) and `MiniCPM5-2B_int8.litertlm` (2.60 GB, int8 with fp32 activations declared), card `cards/minicpm5-2b-litert.md`, `litertlm_manifest.json` in the repo.

`openbmb/MiniCPM5-2B` (released 2026-09-06) is stock `LlamaForCausalLM` — 42 layers, hidden 2048, GQA 16:2, untied 130,560-vocab embeddings, no rope scaling — and the same tokenizer family as MiniCPM5-1B. One command:

```bash
bash scripts/reproduce_llm.sh minicpm5-2b          # -> out/minicpm5-2b/MiniCPM5-2B_{int4,int8}.litertlm
# or directly: bash minicpm_work/convert_minicpm5_2b.sh   (RECIPES="int4 int8" by default)
```

Standard dense env (litert-torch 0.9.3 / ai-edge-quantizer 0.9.0 / litert-lm-builder 0.16.1 / transformers 5.14.1), not the 0.9.1 env of `convert_minicpm.sh`. What the script encodes:

- **Template = the checkpoint's `chat_template.jinja` verbatim on the jinja path** (`USE_JINJA=1`; the script downloads it from the checkpoint so it cannot drift), the packaging of litert-community/MiniCPM5-1B. The exporter auto-declares the `thought` channel because the template carries a literal `<think>`; `enable_thinking` stays a runtime knob and both ways of passing it reach the template (`ThinkingConfig` and conversation `extra_context`, measured on litert-lm 0.17.0): unset → the model thinks by default, true → `<think>\n` prefill, false → empty think block. The 2B template's assistant-history branch (verbatim when the turn carries think tags, an empty think block otherwise) held 3-turn conversations under all three modes with no prefix-contract error.
- **`start_token <s>` is correct here** — the template's own `{{ bos_token }}` renders empty at runtime and the engine prepends the metadata start token, so the prompt sees exactly one `<s>` as upstream. Stops = generation_config eos `[1, 130073]` plus the exporter's 14 punctuation-prefix string stops (see the 1B note above).
- **int4 needs the zero-scale post-fix.** Decoder layer 0's MLP has 13 all-zero rows (both `gate_proj` and `up_proj`), so blockwise-32 quantization emits 832 + 832 zero scales and XNNPACK refuses to prepare (`unsupported scale value (0.000000) … INT4`); the GPU delegate does not care, which is how a GPU-only gate would have shipped a CPU-dead file. `minicpm_work/fix_zero_scales_inplace.py` patches the fp16 scale words at their absolute offsets inside the bundle (3,328 bytes changed, size and every other section identical) — the in-place variant of `bonsai_work/fix_zero_block_scales.py`, needed because that tool repacks a single tflite section and would drop the externalized embedder. int8 has no zero scales.
- **The GPU executor's activation dtype must be decided per file, by measurement — and read against the fp32 CPU reference.** With the default fp16 activations the int8 file's thinking on the rhyme gate question ran 2021 tokens without closing `</think>` on Metal (CPU: `blue`); declaring `prefer_activation_type = "fp32"` (`scripts/set_activation_type.py`, weights untouched) closes it in 452 tokens and the file passes 8/8 (−14 % GPU decode). On a 10-question thinking-ON GSM8K subset the two int8 dtypes were close (fp16 finished 10/10, fp32 7/10, CPU reference 9/10), so for int8 this is a trade that buys the gate, not a cure. The int4 file looked like the mirror image at first (fp16 GPU finished 22/30 inside 3584 tokens, fp32 GPU 0/30) — but **int4 on the CPU, i.e. the fp32 reference, also finishes 0/10 with ~13.7k-character chains**, the same as its fp32 GPU rows. So the int4 fp16 path is the one that diverges from the weights (toward shorter, terminating chains), and the real finding is about the quantization: **int4 thinks ~4.5× longer than int8 (median 13.7k vs 3.2k chars on CPU; bf16 itself 3.0k) and fails to terminate at 3584 tokens; int8 terminates 9/10 — the same 9 as bf16.** Ship reading: int8 with fp32 declared; int4 at the fp16 default, labelled as the direct-answer / short-thinking file.
- **Quality (GSM8K n=100, greedy, thinking OFF = the family's card protocol, max 2048):** bf16 (MPS, upstream template) **92**; int8 CPU **91**; int4 CPU **86** / GPU-fp16 **87** / GPU-fp32act **88** — int4 costs ~5 points on this 42-layer model, and the activation dtype costs nothing measurable in no-think mode. The 8-question gate: bf16 itself scores 6/8 with thinking off ("opposite of hot" → "Cool.", rhyme → "Green."), 8/8 with thinking on or unset — read a thinking-off gate against that, not against 8/8.
- **Block-128 int4 (`RECIPES=int4b128`) was tried and rejected**: GSM8K thinking-off 79 vs block-32's 87, 8Q CPU 7/8 with a non-terminating think, and the thinking chains stay ~13k characters — block size is not the lever for this model's int4 thinking damage.
- Harness beside the script: `hf_oracle_minicpm5_2b.py` (bf16 8Q × three thinking modes), `probe_toggle_minicpm5_2b.py` (toggle + multi-turn through the python API), `gsm8k_bf16_minicpm5_2b.py` / `gsm8k_engine_minicpm5_2b.py` (the parity rows above; `--thinking on|off|unset`, `--ids` subsets, resumable).

### 2026-09-13 — MiniCPM5-2B int8 for litert-community: the LiteRT-LM `models/minicpm5` README command, as written

**Shipped as a card PR** (2026-09-13, [litert-community/MiniCPM5-2B#5](https://huggingface.co/litert-community/MiniCPM5-2B/discussions/5)): `MiniCPM5-2B_int8.litertlm` (2,546,937,984 B, sha256 `61dafe88…`) replaces the previous int8 once the PR lands; the card's int4 stays the `minicpm5-2b` recipe above. The same command's int4 recipe is the subject of [LiteRT-LM#3577](https://github.com/google-ai-edge/LiteRT-LM/issues/3577).

```bash
bash scripts/reproduce_llm.sh minicpm5-2b-readme-int8      # -> out/minicpm5-2b-readme-int8/model.litertlm
# or directly: bash minicpm_work/export_minicpm5_2b_readme.sh dynamic_wi8_afp32 out/x
```

The [README block](https://github.com/google-ai-edge/LiteRT-LM/blob/main/models/minicpm5/README.md#model-conversion) run verbatim on the released wheels (litert-torch **0.9.4**, litert-converter 0.4.0, ai-edge-quantizer 0.9.0, litert-lm-builder 0.16.1, transformers 5.14.1): `--quantization_recipe=dynamic_wi8_afp32 --use_bool_mask=True --apply_gpu_composites=True --cache_length=32771 --experimental_lightweight_conversion` + the repository's `LlmMetadataProto.pbtext` as the metadata override. Only the model id (2B instead of the README's 1B), the pbtext path and the output dir differ. 51 s on an M4 Max.

- **Pin the pbtext.** The script fetches `LlmMetadataProto.pbtext` at commit `b5e34ab1` (2026-09-01) and checks its sha256; the copy on `main` changed on 2026-09-11 (content is rendered as multimodal parts only), so an unpinned copy produces a different bundle from the one that was measured.
- **What the pbtext packaging changes vs the verbatim-jinja recipe above:** the LiteRT-LM canonical `chat_template.jinja` (ChatML, `<tools>` block, `enable_thinking | default(false)` → with nothing set the model answers directly; `enable_thinking=true` via `ThinkingConfig` or the conversation's extra context pre-opens `<think>\n`), the `thought` channel declared as `<think>\n` / `</think>`, start `<s>`, stops `</s>` + `<|im_end|>` (no punctuation-prefix string stops), `max_num_tokens 4096`, ONE prefill signature (128; longer prompts are prefilled in 128-token chunks). `--cache_length=32771` is the runtime's magic number: every run logs `magic_number=32771, target_number=4096` and the cache is sized to 4096 at load.
- **Graph:** int8 per-channel on every linear including the in-graph 130560×2048 embedding and lm_head (no embedder section), `odml.cache_update` ×42, `odml.rms_norm`, `odml.runtime_bmm`, boolean mask; `prefill_128` 1836 ops / `decode` 1655 ops; no custom op. Full LITERT_CL delegation on a Galaxy S26 (1836/1836 + 1655/1655) and full WebGPU delegation on a Mac.
- **Quality (same harness as the 09-08 rows):** GSM8K thinking-off n=100 greedy max 2048: **93** on the Mac GPU (bf16 92, inside one standard error); 8-question gate 6/8 at the no-think default (= the bf16 model's own thinking-off score) and 8/8 with thinking on, both backends; thinking-on 10-question subset 10/10 closed, 10/10 correct with the runtime-default fp16 GPU activations — no fp32 declaration needed on this file.
- **Speed (litert-lm 0.17.0 benchmark `-p 256 -d 256 --runs 3 --cache no --max-num-tokens 1024`, M4 Max):** GPU 2084 prefill / 124.2 decode tok/s, CPU 175 / 37.1.
- **The same command's int4 (`dynamic_wi4c_hr_afp32`, channelwise int4 + Hadamard rotation)** runs fully delegated but wanders off-topic on the Galaxy S26 GPU (OpenCL) while its CPU run answers — on the v0.16.0 kit CLI and on the litertlm-android 0.17.0 AAR alike; the same file answers on the Mac GPU. Plain `dynamic_wi4c_afp32` (no rotation) collapses on both backends. Not shipped; kept for the int8 story only.

### 2026-08-25 — Finetune intake: MiniCPM5-1B derivatives ride the DEFAULT convert.py path

MiniCPM5-1B is stock `LlamaForCausalLM` (`model_type: llama`), so its finetunes go through
`scripts/convert.py` with **no routing, no env change, no retrofit** — the plain dense rail in the
standard `ltconv040dev`-class env (litert-torch 0.9.3). The per-model recipe above remains what
reproduces the *published base* artifact (its post-hoc wi8 quantization); derivatives don't need it.
Measured on huihui-ai/Huihui-MiniCPM5-1B-abliterated (a real behavioral finetune — range-probed:
attention/MLP projections differ from base, embeddings untouched, structure identical):

- One command → `converted_pass`: stock export (export-time int8, cache 4096), gate **7/8**
  non-degenerate at ~165 tok/s decode on the verifier's default backend — no gate-backend
  override, and the post-export stop-token guard is a no-op (generation_config already declares
  `[1, 130073]`).
- The stock bundle's metadata converges with the published litert-community/MiniCPM5-1B artifact
  **by construction**: stops = ids {1, 130073} + the same 14 `X<|im_end|>\n` string stops (the
  exporter's punctuation-prefix expansion — see the corrected note above), thought channel
  auto-added (`<think>` in the template), `max_num_tokens` 4096 (the artifact says 1024).
- Template proof: embedded jinja **byte-equal 9062/9062** to the checkpoint's
  `chat_template.jinja`; greedy **A ≡ B** (litert-lm 0.16.0 CLI, cpu, top-k 1: think content
  832/832 chars + final answer verbatim-identical after normalizing the runtime's
  `[thought]`/`[/thought]` channel rendering to the raw `<think>` markers). Two B-lane
  requirements: strip the render's leading `<s>` (the engine prepends BOS itself — double-BOS
  shifts the greedy continuation on this family exactly as the LFM2.5 lane measured), and the
  channel normalization above for thinking outputs.
- C-lane contrast is structurally N/A for this family too: every usable top derivative probed
  (7 of 7) ships the base's `chat_template.jinja` byte-identical (md5 65d7b88e).
- **The exit gate refuses defective derivatives here too**: a tool-use DPO derivative of the
  same base converts cleanly but fails the gate 4/8-degenerate — greedy on simple arithmetic
  enters a repetition loop inside `<think>` and never closes it, and the same defect class
  reproduces in HF transformers on the source checkpoint (bf16 greedy degenerates into
  `32<|im_end|>` spam on the same prompt), so the conversion is faithful and the subject
  itself is broken. Second measured case of the publish bar doing its job (after the Qwen3.5
  adapter case above).

**litert-torch 0.9.4 notes (measured 2026-08-25)**: 0.9.4 ships the `qwen3_5` exportables and
emits the `ExecutorMetadata` section natively — for **LFM2.5**, new 0.9.4 exports no longer need
the retrofit (convert.py's guard detects the existing section and no-ops), while the Qwen3.5
env guidance is **unchanged**: the 0.9.4 native path exports and loads, but its output is
degenerate (endless `<|iim_ending|>` repetition on a 2B stock export; cause not yet isolated) —
keep exporting Qwen3.5 from litert-torch main. Also measured: transformers 5.15.x breaks the
lfm2 export on both 0.9.3 and 0.9.4 (`Lfm2ShortConv.L_cache` AttributeError) — pin
transformers 5.14.x for this family.

## LFM2.5 family (hybrid ShortConv + attention)

`lfm_work/` converts LiquidAI's LFM2/2.5 hybrid models (gated short-conv blocks + GQA attention). litert-torch 0.9.1 ships the `lfm2` model support and litert-lm ≥0.14 runs it, but the 0.9.1 exporter has a correctness bug on real prompts: the runtime right-pads prefill chunks whenever the prompt doesn't exactly fill a prefill signature, and the exported ShortConv block saves its conv state from the **last (padded) columns** of the chunk — attention can mask padding, a causal conv cannot — so the first generated token of nearly every reply is corrupted (`"to is a vast..."`). Easy to miss: the model recovers after ~1 token (conv window is 2 wide) and GSM8K still parses answers, it just silently loses ~20pt. **litert-torch 0.9.2 fixes this upstream** (valid-token masking + one-hot matmul state select) and `convert_lfm25.py` auto-detects the version: on 0.9.1 it applies the local `lfm_short_conv_patch.py` (mask-derived valid length + gather from the last *valid* columns); on ≥0.9.2 it skips the patch. Either way engine output is token-identical to an exact per-token decode loop at every prompt length (verified with a padded-prefill harness against the raw loop).

```bash
cd lfm_work
python convert_lfm25.py LiquidAI/LFM2.5-1.2B-Instruct out_lfm25_12b        # -> int8 (export-time recipe, convs included)
python convert_lfm25.py LiquidAI/LFM2.5-1.2B-Instruct out_lfm25_12b_fp --fp  # unquantized, for post-hoc int4:
python ../minicpm_work/quantize_litertlm.py apply out_lfm25_12b_fp/model.litertlm lfm25_int4.litertlm --recipe wi4b32_wi8 --algo octav
```

Same env as MiniCPM (python 3.10+, `litert-torch litert-lm "transformers==5.6.2"`). The non-obvious parts:

- **Quantize convs at export time only.** The export-time dynamic-int8 recipe (the default) safely includes the conv layers. Post-hoc `ALL_SUPPORTED` int8 via ai-edge-quantizer kills them (no output) — post-hoc recipes must stick to linears + embedding (`wi8fc`, `wi4b32_wi8`).
- **Multi-length prefill signatures (1..1024)** let the runtime pick tight chunks (less padding waste); they do NOT fix the conv-state bug by themselves — the remainder chunk still pads unless the length happens to exactly match a signature.
- **Metadata**: bos `<|startoftext|>`, stops `[7 (<|im_end|>), 2]`, the HF `chat_template.jinja` verbatim (tool-calling works), and a `thought` channel declaration (`<think>`/`</think>`).
- **Recipe results (GSM8K n=100, greedy; Instruct/JP at max-tokens 1024, Thinking at 2048)**: Instruct — bf16 **79** · export-time int8 **81** (parity) · post-hoc int8-linears-only 79 · OCTAV int4-b32 72. Thinking — bf16 81 · int8-linears-only **77** · int4 72. JP — bf16 63 · int8-linears-only **65** (parity) · int4 55. Without the conv-state fix the Instruct int8 file lands at 59 — that's the bug, not the quantization.
- **Conv-int8 sensitivity varies by finetune**: export-time conv-int8 is free on Instruct (+2), neutral on Thinking (−1), and costs the JP tune **9 points** (56 vs 65) — A/B the export-time recipe against post-hoc linears-only (`wi8fc`) for every new finetune before picking.
- **GPU works with litert-torch ≥ 0.9.2 exports** (it did not with 0.9.1: the local fix's `index_select` lowered to GATHER_ND and its mask sum brought int64 ops — both rejected by GPU delegates; the upstream 0.9.2 rework emits neither). One catch: the 0.9.2 exporter wraps attention softmax in an `odml.softmax` StableHLO composite that the GPU delegate in released litert-lm 0.14 doesn't accept (engine creation fails with "graph is not fully delegated"; CPU unaffected). `convert_lfm25.py` therefore strips the composite marker by default — the math is unchanged, plain softmax stays — which makes the export fully GPU-delegable on today's runtime (Mac WebGPU, 1.2B int8: ~4750 tok/s prefill / ~195 tok/s decode, 8/8 on the quality gate). Pass `--keep-softmax-composite` to keep the marker for newer runtimes that fuse it.
- **The published 1.2B files predate that 0.9.2 rework, so each repo carries a separate GPU file.** They are 0.9.1-lineage: on a GPU delegate they reach 536 of 579 ops and the engine then aborts (`Hint fully delegated to single delegate is set, but the graph is not fully delegated`), the rejected ops being the local patch's `GATHER_ND` plus the INT64 `ADD`/`CAST`/`SUM` of its mask sum. Measured on a Pixel 8a with `litert_lm_main` built from the litert-lm v0.16.0 tag, and identical to the macOS signature. Rather than replace files that are already device-verified on CPU, each repo now also ships `<Name>_int4_gpu.litertlm`, built with litert-torch 0.9.3 + litert-converter 0.3.1:

```bash
cd lfm_work
python convert_lfm25.py LiquidAI/LFM2.5-1.2B-Instruct out_fp --fp
python ../minicpm_work/quantize_litertlm.py apply out_fp/model.litertlm int4.litertlm \
    --recipe wi4b32_wi8 --algo octav
python add_executor_metadata.py int4.litertlm LFM2.5-1.2B-Instruct_int4_gpu.litertlm
```

- **The same GPU re-export carries to int8, and for JP/Thinking the recipe must change with it.** The `_gpu` files above are int4; an int8 GPU file is built the same way but the quantizer step is `--recipe wi8fc`, not the exporter's default export-time int8. On Instruct the export-time recipe is the better one (+2 GSM8K) and it is what that repo ships; on JP it costs **9 points** (56 vs 65) and on Thinking 1 point, so those two repos ship post-hoc linears+embedding with convs left float, and their GPU file has to reproduce the same weights. `fix_zero_block_scales.py` has no role here — `wi8fc` has no block scales at all. The retrofit step is not optional: litert-torch 0.9.3 does not emit `ExecutorMetadata`, and `quantize_litertlm.py` rebuilds the bundle with only three sections.

```bash
cd lfm_work
python convert_lfm25.py LiquidAI/LFM2.5-1.2B-JP out_jp_fp --fp
python ../minicpm_work/quantize_litertlm.py apply out_jp_fp/model.litertlm jp_wi8fc.litertlm --recipe wi8fc
python add_executor_metadata.py jp_wi8fc.litertlm LFM2.5-1.2B-JP_int8_gpu.litertlm   # "22 state buffers: 10 linear-attn, 12 kv"
```

  Delete the quantizer's output before a rebuild — `add_executor_metadata.py` removes its own target, but `quantize_litertlm.py` does not, so a stale intermediate can survive a re-run and get packed. Built on litert-torch 0.9.3 / litert-converter 0.3.1 / ai-edge-quantizer 0.8.0 / litert-lm 0.15.0 / transformers 5.14.1 (pin transformers at 5.14.x — 5.15 renames `Lfm2ShortConv.L_cache` and the lfm2 export dies). Measured against the CPU-lineage int8 it replaces, same harness and same day: GSM8K identical (JP 65 vs 65, Thinking 77 vs 77, n=100 greedy), 8-question gate 8/8 (JP) and 7/8 (Thinking) on both backends, and on CPU the two files generate byte-identical text on a fixed greedy prompt — the re-export changes the graph, not the weights. On a Galaxy S26 (SM8850, litert-lm v0.16.0) the new file delegates fully (542/542 across ten prefill signatures, 501/501 and 519/519) and generates, where the file it replaces reaches 536 of 579 and aborts; 205-token prompt, prefill ~940-980 tok/s, decode ~42 tok/s, peak RSS ~570 MB. Worth noting because it contradicts a common rule of thumb: the int8 file's GPU-side peak came in *below* the 736 MB int4 file's, not at the ~4x-file-size fp32 residency the heuristic predicts. Against the int4 GPU file it is ~23% slower to decode and ~10 GSM8K points more accurate, so the two are a real speed/quality pair rather than a strict upgrade. The `_int8_gpu` files were not gated on iOS.

  The 2.6B pipeline's `fix_zero_block_scales.py` step is a no-op on this family (measured: 0 zero scales across 0 tensors on all three 1.2B checkpoints) — harmless to run, and the 2.6B does need it. The result delegates **fully** on Android OpenCL (501/501 and 519/519 nodes, zero rejected ops) and runs on the macOS GPU backend; **iOS Metal runs this family** on litert-lm >= 0.16.0 — but only when `maxNumTokens` is set to **1024**. What was tracked as LiteRT-LM#3129 turned out to be context sizing on the integration side, not a runtime bug: the Metal delegate compiles its kernels against the KV cache `maxNumTokens` pre-allocates, and a value that does not match the exported plan fails engine creation with a shader-compile error. Verified by generation on an iPhone 17 Pro (`_int4_gpu`, v0.16.0 xcframework): decode ~70 tok/s, peak ~560 MB. On a Pixel 8a the GPU's win is prefill and time-to-first-token (263-token prompt: ~190 tok/s vs ~38-54 on CPU; TTFT ~1.4 s vs ~5 s), while decode is bandwidth-bound and roughly equal (~21 tok/s).
- **`litert_lm_main --benchmark_prefill_tokens` / `--benchmark_decode_tokens` are silently ignored** (measured on the v0.16.0 tag build, both backends): runs that pass them still report `Processed 19 tokens` and ~50 decode tokens. A 19-token prefill reads about 3x slower than a 256-token one on the same file, so a row labelled "prefill 256" that came from those flags is mislabelled. Drive a real prompt with `--input_prompt_file` and cap generation with `--max_output_tokens`.
- **litert-lm ≥ 0.15 needs an `ExecutorMetadata` section** for state-carrying (hybrid) models: 0.15 binds the per-layer conv/attention state buffers through a new `ExecutorMetadataProto` section, and files exported with litert-torch ≤ 0.9.2 don't have it — they run fine on 0.14 but fail at inference on 0.15 with `missing some output TensorBuffers` (attention-only models are unaffected). Fix an existing file in place with `python add_executor_metadata.py in.litertlm out.litertlm` (weights unchanged; the result runs on both 0.14 and 0.15 — this is how the published LFM2.5 repos were updated on 2026-08-04). Checking on the composite from the previous bullet: the 0.15 GPU delegate still rejects `odml.softmax`, so the strip default stays.

### 2026-08-24 — Finetune intake: LFM2.5 derivatives ride the RELEASED litert-torch

Finetunes of LFM2.5 go through `scripts/convert.py` in the standard `ltconv040dev`-class
env (litert-torch 0.9.3 — the lfm2 model_ext ships in releases; litert-torch *main*'s
lfm2 patch is currently incompatible with transformers 5.15, `Lfm2ShortConv` has no
`L_cache`, so do NOT use the Qwen3.5 main-venv for this family). Measured on
huihui-ai/Huihui-LFM2.5-1.2B-Instruct-abliterated (real behavioral finetune, old-
generation 1,783 B chat template — the top derivatives all carry it):

- Stock export succeeds in ~64 s, but **0.9.3's bundling emits no ExecutorMetadata
  section** (3 sections), so litert-lm ≥ 0.15 fails at inference — convert.py now
  retrofits it automatically after export (`ensure_executor_metadata()`, reusing
  `lfm_work/add_executor_metadata.py`; 22 state buffers on the 1.2B; needs a
  litert-lm ≥ 0.15 CLI: `$LITERT_LM_CLI`, else PATH, else `~/venvs/lt0160run`).
- Proof: one command → `converted_pass`, gate **8/8** (verifier default backend — the
  bundle runs on WebGPU at ~162 tok/s decode; the odml.softmax-composite GPU rejection
  recorded above for litert-lm 0.14/0.15 does not reproduce on the 0.16 runtime),
  embedded template byte-equal 1783/1783, greedy **A ≡ B** byte-identical.
- Two method notes: the derivative's old-generation template renders IDENTICALLY to the
  base's current one for every CLI-reachable input (system/user/assistant turns — they
  diverge only in the tools path), so a C-lane contrast is structurally N/A for this
  family's typical derivatives; and the B lane must strip the template's leading
  `bos_token` before `--no-template` (the engine prepends BOS itself — double-BOS
  changes the token stream and the greedy output, the `granite-4.1-3b` BOS trap in
  A/B-proof form).

### LFM2.5-2.6B (the thinking flagship)

`lfm_work/convert_lfm25_26b.py` converts [LiquidAI/LFM2.5-2.6B](https://huggingface.co/LiquidAI/LFM2.5-2.6B) (22 ShortConv + 8 attention layers, vocab 128000, reasons in a `<think>` block by default). Published: [litert-community/LFM2.5-2.6B](https://huggingface.co/litert-community/LFM2.5-2.6B). Same env and mechanics as the 1.2B family above (litert-torch ≥ 0.9.2), plus three 2.6B-specific facts:

```bash
cd lfm_work
python convert_lfm25_26b.py LiquidAI/LFM2.5-2.6B out_26b            # int8 (then add_executor_metadata.py)
python convert_lfm25_26b.py LiquidAI/LFM2.5-2.6B out_26b_fp --fp    # for the int4 pipeline in the script docstring
```

- **Regenerate the metadata for the 2.6B tokenizer.** Stop ids are per-tokenizer, not per-family: `<|im_end|>` is id 7 in the 1.2B and **124900** here (`<|endoftext|>` 124895). `lfm25_26b_LlmMetaProto.pbtext` also replaces the vendor chat template: the vendor one both uses minijinja-fatal Python methods and strips past `<think>` blocks keyed on the last user index, which violates the runtime's incremental render contract (turn-3 failure). The bundled template renders history verbatim and **pre-fills `<think>` in the generation prompt** — load-bearing for quantized builds: with a bare assistant prompt, int4 loses think-discipline first (2,500-token unscaffolded deliberations instead of answers from turn 2); with the pre-fill all variants answer tersely and the thought channel routes cleanly.
- **The zero-scale int4 wall is not ternary-only.** This dense-trained checkpoint carries ~746k all-zero 32-blocks across 26 tensors; OCTAV int4-b32 emits zero scales and the file dies at engine invoke on every runtime version. Pipeline order: quantize → `fix_zero_block_scales.py` → `add_executor_metadata.py`.
- **The int8 file is desktop/Android-class** (~2.7 GB single weight section exceeds what iOS memory-maps for one section); the int4-b32 file (1.67 GB) is the iPhone variant.

**Results (GSM8K n=100, greedy, max-tokens 2048, identical harness both sides)**: bf16 reference **92** · export-time int8 **88** · post-hoc int8-linears-only 89 (conv-int8 harmless on this model — matches Instruct, unlike the JP tune) · OCTAV int4-b32 **83**. Structure gates: 8Q 8/8 CPU (both files), 42-length first-token sweep clean (fresh engine per length), 3-turn conversation exact, Mac WebGPU 8Q 7/8 (int8). Mac M4 Max CPU: int8 434 tok/s prefill@1024 / 38 decode; int4 43.7 decode.

### LFM2.5-230M (the smallest decoder) — a template the runtime cannot parse, and a shape that kills the GPU shader compile

`lfm_work/convert_lfm25_230m.py` converts [LiquidAI/LFM2.5-230M](https://huggingface.co/LiquidAI/LFM2.5-230M) (14L: 8 ShortConv + 6 attention, 230M params, tied embeddings). Published: [litert-community/LFM2.5-230M](https://huggingface.co/litert-community/LFM2.5-230M). Same env as the family above (litert-torch 0.9.3, **transformers pinned 5.14.x**). Three facts none of the larger LFM2.5 ships hit:

```bash
cd lfm_work
python convert_lfm25_230m.py out_230m           # export-time int8 (NOT what ships — see below)
python convert_lfm25_230m.py out_230m_fp --fp   # unquantized, the ship pipeline's source
python fix_230m_template.py out_230m_fp/model.litertlm fp_fixed.litertlm
python ../minicpm_work/quantize_litertlm.py apply fp_fixed.litertlm int8.litertlm --recipe wi8fc
python ../minicpm_work/quantize_litertlm.py apply fp_fixed.litertlm int4.litertlm --recipe wi4b32_wi8 --algo octav
python ../bonsai_work/fix_zero_block_scales.py int4.litertlm int4_z.litertlm   # measured no-op here, family pipeline order
python add_executor_metadata.py int8.litertlm LFM2.5-230M_int8.litertlm
python add_executor_metadata.py int4_z.litertlm LFM2.5-230M_int4.litertlm
```

- **The checkpoint's chat template does not run on litert-lm's minijinja — every naive bundle dies on its first message.** Two constructs: HF's `{% generation %}`/`{% endgeneration %}` assistant-mask markers are a *parse* error ("unknown statement generation"), and `message.get("content")` is a *render* error ("map has no method named get"). `fix_230m_template.py` strips the markers and rewrites the two `.get` sites to plain indexing inside the packed bundle (weight-identical); `make_metadata_230m.py` proves the repaired template renders byte-identically to the original through HF `apply_chat_template` on 6 conversation shapes (user/system/multi-turn/past-think/closed/tools) — run it before trusting any edit. String methods (`.split`, `.endswith`) sit in branches the runtime never reaches and need no rewrite.
- **cache_length 4099 + the 1024 prefill signature = GPU engine-creation failure at shader compile** (`CreateShaderModule` validation error, measured on macOS WebGPU). Isolated to the pair: ladder+4096 compiles, 128-only+4099 compiles, 1024-only+4099 fails. The 230M ships **cache 4096** — a clean multiple of the widest signature — and delegates fully on Adreno OpenCL (492/492 nodes on all 11 prefill signatures + decode, Galaxy S26) and on Metal/WebGPU.
- **This tune sits on the JP/Thinking side of the family conv-int8 law.** GSM8K is floor for a 230M non-reasoning model (bf16 23/100 — useless for the A/B), so the recipe decision ran on instruction following (`ifeval_lite_230m.py`, a re-implementation of google/IFEval's mechanically checkable types; numbers comparable only within the harness): bf16 60.0 / export-time conv-int8 53.3 / **post-hoc wi8fc 58.3** (n=120 prompt-level strict). Convs stay float; wi8fc ships as the int8 file.
- One more per-device fact worth recording: **at 230M, int4 is not the fast file everywhere.** iPhone 17 Pro Metal decodes int4 13% faster than int8 (162 vs 143 tok/s), but Galaxy S26 Adreno decodes it 2.8× *slower* (43 vs 122) and Mac is a wash — the blockwise dequant overhead dominates at this size. int8 is the primary recommendation.

### LFM2.5-350M — the LiteRT-LM `models/lfm2` configuration replaces the vendor template, and the 4-section export needs a quantizer that keeps every section

`lfm_work/convert_lfm25_350m.sh` converts [LiquidAI/LFM2.5-350M](https://huggingface.co/LiquidAI/LFM2.5-350M) (16 layers: 10 double-gated short-conv blocks + 6 GQA blocks, hidden 1024, tied embeddings, 65,536 vocab) into the two files of [litert-community/LFM2.5-350M](https://huggingface.co/litert-community/LFM2.5-350M). Environments (released wheels): litert-torch 0.9.4, litert-converter 0.4.0, ai-edge-quantizer 0.9.0, transformers 5.14.1 (5.15 breaks the lfm2 export), torch 2.13.0; litert-lm 0.18.0 for unpack / pack and for running the files.

```bash
EXPORT_PY=/path/to/python-with-litert-torch-0.9.4 \
RUN_PY=/path/to/python-with-litert-lm-0.18.0 \
LITERT_LM=/path/to/litert-lm \
  bash lfm_work/convert_lfm25_350m.sh out_350m
```

| file | section | bytes | sha256 |
|---|---|---|---|
| `LFM2.5-350M_int8.litertlm` (403,416,064 B) | LlmMetadataProto | 3,862 | `f01d215c91cf7e7c59968db83183899fa997a0d07aecbce635b17636bbd030e0` |
| | ExecutorMetadataProto | 1,421 | `91a4cbfc72328a45d16d88c38a3a08438be05c3438a012283a5d14a3dff1301e` |
| | HF_Tokenizer_Zlib | 906,962 | `c06bb6a9a576b1348a9f423971a4894aa17aada66282c1d1df206c74102ab657` |
| | TFLiteModel | 402,449,408 | `ee9605c6f2d60eb9c9b0025a7922b3ae1144610c232e1884e142c4599d904022` |
| `LFM2.5-350M_int4.litertlm` (247,747,696 B) | LlmMetadataProto | 3,862 | `f01d215c91cf7e7c59968db83183899fa997a0d07aecbce635b17636bbd030e0` |
| | ExecutorMetadataProto | 1,421 | `91a4cbfc72328a45d16d88c38a3a08438be05c3438a012283a5d14a3dff1301e` |
| | HF_Tokenizer_Zlib | 906,962 | `c06bb6a9a576b1348a9f423971a4894aa17aada66282c1d1df206c74102ab657` |
| | TFLiteModel | 246,781,040 | `81d165c9170e2ddadb296227a6c27aad2a502bd6acb3fb1515bbb7067d4c9537` |

The script printed these section hashes for both files when it was run from a copy of this repository's layout on 2026-10-12 (205 s on an M4 Max: 18:04:44–18:08:09; every section equal to the published files' sections by raw bytes); the file sha256 differs from run to run (the packer writes a new uuid and creation timestamp), so compare sections.

- **The vendor chat template does not run on the runtime's Jinja engine, so the whole LlmMetadata is replaced with the LiteRT-LM `models/lfm2` configuration** at commit [86a7e6d0](https://github.com/google-ai-edge/LiteRT-LM/blob/86a7e6d098c3a6578208fc8538228fa704ca2e8b/models/lfm2/chat_template.jinja). The export embeds LiquidAI's `chat_template.jinja`, which uses `{% generation %}` markers: every message fails with `Failed to apply template: Template creation failed: syntax error: unknown statement generation`. `repack_official_lfm2.py` reads the configuration at the pinned commit, checks both files by sha256, changes two lines (`max_num_tokens` 32768 → 4096, `supports_thinking` true → false) and proves every other section unchanged. Rendered conversations match the vendor template byte for byte for user, system, multi-turn and one-tool prompts; after a tool call two things differ: arguments in double quotes and the tool result wrapped in `[…]`. Stops `<|im_end|>` and `<|endoftext|>`, start token `<|startoftext|>`.
- **Tested on litert-lm 0.18.0.** The configuration's template reads message content as a list of parts. litert-lm 0.17.1 passes a string: the engine prefilled 9 tokens for a one-question turn, exactly an empty user turn, and the model asked what the question was.
- **litert-torch 0.9.4 writes four sections (LlmMetadata, ExecutorMetadata, HF tokenizer, TFLite); `minicpm_work/quantize_litertlm.py apply` rebuilt three.** It took the last `*.pbtext` of the builder's dump as the LLM metadata, so on macOS it handed `ExecutorMetadataProto.pbtext` to the builder and stopped with `has no field named "llm_executor_metadata"`; in the other order it would have dropped the ExecutorMetadata section, which binds the 22 conv / KV state buffers. `lfm_work/quantize_keep_sections.py` quantizes only the TFLite section of an unpacked bundle and packs the same `model.toml` back, checking each section. `quantize_litertlm.py` now takes the metadata files by name and passes the ExecutorMetadata section to the builder; on this export its `apply --recipe wi8fc` output matches the int8 file's sections before the metadata swap byte for byte.
- **KV budget 4096 with the 11-signature prefill ladder (1024 … 1)**, as for LFM2.5-230M (4099 fails the WebGPU shader compile next to the 1024 signature).
- **int8 = post-hoc dynamic int8 on the linears + embedding, convs float; int4 = OCTAV int4 blockwise-32 on the linears, int8 lm_head + embedding**, zero-scale patch a no-op here. On the IFEval-lite harness (`lfm_work/ifeval_lite_230m.py` rows and checkers; numbers comparable only within the harness): bf16 81, int8 82, int4 79 of 120 prompts; the export-time int8 arm that also quantizes the convs scored 84, indistinguishable from the post-hoc file on paired prompts (8 differ, p 0.73), so the int8 file keeps the convs float like the 230M. GSM8K says little about this model: bf16 writes the answer line without working in most replies and scores 2 / 100 (7 with a step-by-step suffix).
- Measured (prefill 1024 / decode 256 / context 2048, litert-lm 0.18.0 on the Mac, prebuilt `litert_lm_advanced_main` 2026-09-18 on the phones): Galaxy S26 GPU int4 4391 / 106.9 tok/s, int8 5661 / 92.9; Pixel 8a GPU int8 2411 / 47.9, int4 1246 / 50.1; Mac M4 Max GPU (WebGPU) int8 13556 / 423.7, int4 13965 / 425.9; every GPU signature fully delegated on all three.

### LFM2-700M — the first-generation LFM2 template fails at render, and on this checkpoint the int8 file keeps the exporter's default recipe (convolutions quantized)

`lfm_work/convert_lfm2_700m.sh` converts [LiquidAI/LFM2-700M](https://huggingface.co/LiquidAI/LFM2-700M) at commit 86f49fc9 (742,489,344 parameters in 16 layers: 10 double-gated short-convolution blocks + 6 GQA blocks, hidden 1536, tied embeddings, 65,536 vocab) into the two files of [litert-community/LFM2-700M](https://huggingface.co/litert-community/LFM2-700M). Environments (released wheels): litert-torch 0.9.4, litert-converter 0.4.0, ai-edge-quantizer 0.9.0, transformers 5.14.1 (5.15 breaks the lfm2 export), torch 2.13.0; litert-lm 0.18.0 for unpack / pack and for running the files.

```bash
EXPORT_PY=/path/to/python-with-litert-torch-0.9.4 \
RUN_PY=/path/to/python-with-litert-lm-0.18.0 \
LITERT_LM=/path/to/litert-lm \
  bash lfm_work/convert_lfm2_700m.sh out_700m
```

| file | section | bytes | sha256 |
|---|---|---|---|
| `LFM2-700M_int8.litertlm` (808,281,472 B) | LlmMetadataProto | 3,862 | `f01d215c91cf7e7c59968db83183899fa997a0d07aecbce635b17636bbd030e0` |
| | ExecutorMetadataProto | 1,421 | `91a4cbfc72328a45d16d88c38a3a08438be05c3438a012283a5d14a3dff1301e` |
| | HF_Tokenizer_Zlib | 906,839 | `0205d1d26f53c7f2ab0d3183ce2e31c2189c03b5bef7efe01f316a0895310f3a` |
| | TFLiteModel | 807,314,816 | `b0144c2a7e44ba0ad7cc77951fe1d27789ea384a43e9564c6cb6dad11fcaf0fd` |
| `LFM2-700M_int4.litertlm` (480,794,160 B) | LlmMetadataProto | 3,862 | `f01d215c91cf7e7c59968db83183899fa997a0d07aecbce635b17636bbd030e0` |
| | ExecutorMetadataProto | 1,421 | `91a4cbfc72328a45d16d88c38a3a08438be05c3438a012283a5d14a3dff1301e` |
| | HF_Tokenizer_Zlib | 906,839 | `0205d1d26f53c7f2ab0d3183ce2e31c2189c03b5bef7efe01f316a0895310f3a` |
| | TFLiteModel | 479,827,504 | `ec44bb1b09f42346436c37bebe64789bb8bcb4e612a7e260a6965fa743937d14` |

The script printed these section hashes for both files when it was run from a copy of this repository's layout on 2026-10-12 (550 s on an M4 Max; a second run after a docstring wording fix printed the same; every section equal to the published files' sections by raw bytes); the file sha256 differs from run to run (the packer writes a new uuid and creation timestamp), so compare sections.

- **The checkpoint is pinned.** `litert-torch export_hf` takes no revision, so `lfm_work/fetch_pinned.py` downloads commit 86f49fc9 into a separate cache, checks every file against the Hub's record for that commit, and both exports run offline from it. The checkpoint's config.json is the transformers 4.54 form (`full_attn_idxs`, no `layer_types`); transformers 5.14.1 reads it as 10 short-convolution + 6 attention layers.
- **The int8 file is the exporter's default recipe, convolutions included.** `dynamic_wi8_afp32` stores channelwise int8 weights for every linear, the 10 short-convolution kernels and the embedding. The post-hoc int8 that LFM2.5-230M and LFM2.5-350M ship (linears + embedding int8, convolutions float) scored lower on this checkpoint: on the IFEval-lite harness (rows and checkers of `lfm_work/ifeval_lite_230m.py`; numbers comparable only within the harness) bf16 passed 77 of 120 prompts, the export-time int8 75 and 74 in two runs (149/240), the post-hoc int8 69 and 71 (140/240). A rerun of the same file in fresh processes flipped 4 of 24 prompts, so the choice used both runs. The int4 file (OCTAV int4 blockwise-32 on the linears of the unquantized export, int8 lm_head + embedding, convolutions float; no zero or subnormal block scale, so the zero-scale patch changes nothing) passed 67. GSM8K (test questions 0–99, two step-by-step prompt wordings): bf16 59 / 59, int8 58 / 62, int4 52 / 57 of 100; unlike LFM2.5-350M, no reply was the bare answer line.
- **The vendor chat template does not render on litert-lm 0.18.0, so the whole LlmMetadata is replaced with the LiteRT-LM `models/lfm2` configuration** at commit [86a7e6d0](https://github.com/google-ai-edge/LiteRT-LM/blob/86a7e6d098c3a6578208fc8538228fa704ca2e8b/models/lfm2/chat_template.jinja). The export embeds Liquid AI's first-generation LFM2 `chat_template.jinja`; its `message.get("content")` fails every message with `Failed to apply template: unknown method: map has no method named get`, and the CLI still exits 0. `repack_official_lfm2.py` (the same file as for LFM2.5-350M) changes two lines of the configuration (`max_num_tokens` 32768 → 4096, `supports_thinking` true → false) and proves every other section unchanged. Plain chat renders byte for byte like the vendor template; with tools the configuration writes `List of tools: […]` where the LFM2 template wraps the list in `<|tool_list_start|>…<|tool_list_end|>` and the tool result in `<|tool_response_start|>…<|tool_response_end|>`, and it quotes call arguments with `"` where the LFM2 template uses `'`; on 0.18.0 both files still returned a parsed tool call. Stops `<|im_end|>` and `<|endoftext|>`, start token `<|startoftext|>`. `<think>` is not a token of this vocabulary and the files are marked non-thinking.
- **Tested on litert-lm 0.18.0.** The configuration's template reads message content as a list of parts; on litert-lm 0.17.1 the int8 file answered as if no question had been asked.
- **KV budget 4096 with the 11-signature prefill ladder (1024 … 1)**, as for the LFM2.5 files; a 6-signature twin was never more than 3 % faster on any device measured.
- Measured (prefill 1024 / decode 256 / context 2048, litert-lm 0.18.0 on the Mac, prebuilt `litert_lm_advanced_main` 2026-09-18 on the phones): Galaxy S26 GPU int4 2427 / 73.3 tok/s, int8 3396 / 59.0; Pixel 8a GPU int8 1451 / 34.8, int4 620 / 39.9; Mac M4 Max GPU (WebGPU) int8 7870 / 306.3, int4 7839 / 348.3; every GPU signature fully delegated on all three.

### LFM2-2.6B — a model that reasons when it chooses, a thought channel kept on, and an int4 zero-scale fix that has to clear the blocks too

`lfm_work/convert_lfm2_26b.sh` converts [LiquidAI/LFM2-2.6B](https://huggingface.co/LiquidAI/LFM2-2.6B) at commit `36ed799f` (30 layers: 22 short-conv blocks + 8 grouped-query attention layers, hidden 2048, tied embeddings, 65,536 vocab) into the two files of [litert-community/LFM2-2.6B](https://huggingface.co/litert-community/LFM2-2.6B). Environments (released wheels): litert-torch 0.9.4, litert-converter 0.4.0, ai-edge-quantizer 0.9.0, transformers 5.14.1 (5.15 breaks the lfm2 export), torch 2.13.0; litert-lm 0.18.0 for unpack / pack and for running the files.

```bash
EXPORT_PY=/path/to/python-with-litert-torch-0.9.4 \
RUN_PY=/path/to/python-with-litert-lm-0.18.0 \
LITERT_LM=/path/to/litert-lm \
  bash lfm_work/convert_lfm2_26b.sh out_lfm2_26b
```

| file | section | bytes | sha256 |
|---|---|---|---|
| `LFM2-2.6B_int8.litertlm` (2,723,809,744 B) | LlmMetadataProto | 3,862 | `b8771d3dcd862593a9eef0c933353dc8d7888bdee09cca7373a37c02d3237375` |
| | ExecutorMetadataProto | 2,465 | `b9672684fb34216314d1c7cb4045f7520d056325ccc48d17a50d0ed80a0cd40e` |
| | HF_Tokenizer_Zlib | 906,872 | `f7d531f34af15ec3a8ee935071d06f670a90018aa65c3e5ec9bcce41ad97d29b` |
| | TFLiteModel | 2,722,843,088 | `d5aa29585c8723d910b5b04932c2b8238bc14067e3438b4c032ac3cf1765686e` |
| `LFM2-2.6B_int4.litertlm` (1,529,581,168 B) | LlmMetadataProto | 3,862 | `b8771d3dcd862593a9eef0c933353dc8d7888bdee09cca7373a37c02d3237375` |
| | ExecutorMetadataProto | 2,465 | `b9672684fb34216314d1c7cb4045f7520d056325ccc48d17a50d0ed80a0cd40e` |
| | HF_Tokenizer_Zlib | 906,872 | `f7d531f34af15ec3a8ee935071d06f670a90018aa65c3e5ec9bcce41ad97d29b` |
| | TFLiteModel | 1,528,614,512 | `763f1d02ffe30d7aefa059c1a567e611e1bf209fb6a2a3d1c052ecd0055efe13` |

The script printed these section hashes for both files when it was run from a copy of this repository's layout on 2026-10-12 (972 s on an M4 Max, 22:30:58–22:47:10; every section equal to the published files' sections by raw bytes); the file sha256 differs from run to run (the packer writes a new uuid and creation timestamp), so compare sections.

- **The checkpoint is pinned.** `litert-torch export_hf` takes no revision, so `lfm_work/fetch_pinned.py` downloads commit `36ed799f` into a separate cache, points that cache's `refs/main` at it, and checks every file against the Hub's record (LFS sha256 or git blob sha1, and the size) before the export runs offline from it. Set `LFM2_HF_CACHE` to reuse a cache that already holds the commit; the script refuses the shared `~/.cache/huggingface/hub`, whose `refs/main` it would rewrite.
- **Liquid AI's chat template does not render on the runtime's Jinja engine, so the whole LlmMetadata is replaced with the LiteRT-LM `models/lfm2` configuration** at commit [86a7e6d0](https://github.com/google-ai-edge/LiteRT-LM/blob/86a7e6d098c3a6578208fc8538228fa704ca2e8b/models/lfm2/chat_template.jinja). With the metadata the export writes, litert-lm 0.18.0 fails every message: `Failed to apply template: unknown method: map has no method named get (in template:53)` — the template's `message.get("content")` — and the CLI still exits 0. `repack_official_lfm2.py --supports-thinking true` reads the configuration at the pinned commit, checks both files by sha256, changes one line (`max_num_tokens` 32768 → 4096) and proves every other section unchanged. Without tools the rendered conversation matches Liquid AI's template byte for byte; with tools the list is written as `List of tools: […]` where Liquid AI's template wraps it in `<|tool_list_start|>…<|tool_list_end|>`, and after a tool call the argument quotes and the result wrapping differ. Stops `<|im_end|>` and `<|endoftext|>`, start token `<|startoftext|>`.
- **The model decides when to reason, so `supports_thinking` stays true and the configuration's `thought` channel stays.** Liquid AI describes LFM2-2.6B as the family's one model with "dynamic hybrid reasoning (traces between `<think>` and `</think>` tokens) for complex or multilingual prompts"; `<think>` is plain text in this vocabulary, and the runtime's channel takes the trace out of the reply. Measured: "What is the capital of France?" opens a trace; 1 of 120 IFEval-lite prompts and 0 of 100 GSM8K questions did. On litert-lm 0.18.0, `--thinking-budget 0` and `--thinking-budget 32` left this bundle's output unchanged.
- **Tested on litert-lm 0.18.0.** The configuration's template reads message content as a list of parts; on litert-lm 0.17.1 the int8 file answered "It seems like your message might be incomplete…" to a one-question prompt.
- **OCTAV int4 block-32 writes 746,432 zero block scales on this checkpoint** (26 tensors: the short-conv input projection of 20 layers and the MLP w1 / w3 of layers 1, 2 and 5; no subnormal scale). The quantizer computes the scale before it quantizes, so those blocks were quantized as w / 0 and hold saturated nibbles; `minicpm_work/fix_zero_scales_inplace.py --zero-payload --require-normal` gives each zero scale the tensor's smallest nonzero scale, zeroes the block's payload (11,942,912 bytes), and refuses to write a file that still has a zero or subnormal scale. Without `--zero-payload` (the tool's earlier behaviour, which the MiniCPM5-2B recipe uses) only the scale words change and the int4 file differs from the published one.
- **litert-torch 0.9.4 writes four sections; `lfm_work/quantize_keep_sections.py` quantizes only the TFLite section** and packs the same `model.toml` back, as in the LFM2.5-350M entry. ExecutorMetadata binds 38 state buffers: 22 short-conv states, 8 key and 8 value caches.
- **KV budget 4096 with the 11-signature prefill ladder (1024 … 1).** The int8 file's TFLite section is 2,722,843,088 bytes, over 2 GiB; it was not run on iOS.
- **int8 = post-hoc dynamic int8 on the linears + embedding, convs float; int4 = OCTAV int4 blockwise-32 on the linears, int8 lm_head + embedding.** GSM8K (questions 0–99, greedy, at most 2048 output tokens, scored on the reply without the trace; litert-lm rows on the Mac GPU, one process per question): bf16 88, int8 89, int4 87. The export-time int8 arm that also quantizes the convolutions scored 89 too, 2 questions apart from the post-hoc file. IFEval-lite (`lfm_work/ifeval_lite_230m.py` rows and checkers; comparable only within the harness): bf16 85, int8 85, int4 87 of 120 prompts.
- Measured (prefill 1024 / decode 256 / context 2048, litert-lm 0.18.0 on the Mac, prebuilt `litert_lm_advanced_main` 2026-09-18 on the phones): Mac M4 Max GPU (WebGPU) int8 2466 / 125.6 tok/s, int4 2453 / 163.8; Galaxy S26 GPU int8 1122 / 20.4, int4 682 / 27.8; Pixel 8a GPU int8 504 / 11.1, int4 172 / 15.8 — the 2.7 GB int8 file runs on the Pixel 8a GPU with 419,100 kB of MemAvailable left at its lowest; every GPU signature fully delegated on all three.

### sarashina2.2-0.5b / 1b instruct (SB Intuitions, Japanese) — a SentencePiece vocab whose chat specials are CONTROL pieces, and a BOS the model never saw

`sarashina_work/convert_sarashina.py` converts [sbintuitions/sarashina2.2-0.5b-instruct-v0.1](https://huggingface.co/sbintuitions/sarashina2.2-0.5b-instruct-v0.1) and [-1b-instruct-v0.1](https://huggingface.co/sbintuitions/sarashina2.2-1b-instruct-v0.1) (plain `LlamaForCausalLM`, 24 layers, GQA 16:8, **102,400-entry untied vocab**, MIT). Published: [litert-community/sarashina2.2-0.5b-instruct-v0.1](https://huggingface.co/litert-community/sarashina2.2-0.5b-instruct-v0.1), [litert-community/sarashina2.2-1b-instruct-v0.1](https://huggingface.co/litert-community/sarashina2.2-1b-instruct-v0.1).

```bash
# int8 ship (int8 dynamic on linears + embedding)
python sarashina_work/convert_sarashina.py sbintuitions/sarashina2.2-0.5b-instruct-v0.1 out/s05_int8 templates/sarashina_simple.jinja dynamic_wi8_afp32
# int4 ship (blockwise-32 + OCTAV linears, int8 embedding)
python sarashina_work/convert_sarashina.py sbintuitions/sarashina2.2-0.5b-instruct-v0.1 out/s05_int4 templates/sarashina_simple.jinja BOCTAV4
# same two lines with sarashina2.2-1b-instruct-v0.1 for the 1b
# gates: scripts/verify_quality.py (EN 8Q) + sarashina_work/verify_quality_ja.py (JA 8Q + UTF-8 streaming probe),
#        sarashina_work/tokenizer_parity.py (engine vs HF ids), sarashina_work/gate_multiturn_ja.py,
#        sarashina_work/jcqa_eval.py (JCommonsenseQA, bf16 vs bundle in one harness)
```

The wrapper is `scripts/export_simple_template.py` (structured prompt templates, prefill ladder 1024..1, KV 4096) plus two facts this checkpoint needs:

- **Embed the HF `tokenizer.json`, not the vendor `tokenizer.model`.** litert-torch copies `tokenizer.model` verbatim whenever the HF tokenizer exposes `vocab_file` ending in it. sarashina's SentencePiece model types every chat special (`<|user|>` `<|assistant|>` `<|system|>` `</s>`) as **CONTROL**, and a bare sentencepiece `Encode` never matches CONTROL pieces from text — the template's `<|user|>` comes out as `< | user | >` (5 pieces) and the model never sees its role markers. The HF fast tokenizer matches them as added tokens. Clearing `vocab_file` makes the exporter fall through to `save_pretrained` → `tokenizer.json`. Its decoder chain has no `Strip` step, so the Metaspace per-token-strip trap does not apply. Engine-vs-HF ids: identical on 7 probes (specials-as-text, Japanese, emoji/rare-kanji byte-fallback, Latin-1, whitespace).
- **Write no `start_token`.** `add_bos_token` is false and the official template emits no `<s>`, but the exporter sets `start_token` from `tokenizer.bos_token` regardless, and the runtime then prepends it. Measured on bf16 with the same rendered prompt ± `<s>`: the 0.5b drops the 8-question gate from 6→5 (English) and 8→7 (Japanese); the 1b is unmoved. `NO_START_TOKEN=1` (set by the wrapper) clears `bos_token` before the metadata is built.
- No post-processing: plain attention (no ExecutorMetadata retrofit), template renders on the runtime as-is.
- Per-device fact: **on the Galaxy S26 int4 is not faster than int8 for these files** (0.5b GPU decode 32–40 vs 36 tok/s; CPU 19.5 vs 20–21) — the untied 102,400 vocab makes the int8 embedding + lm_head the dominant per-token cost, so the int4 dequant overhead on the linears buys nothing. int8 is the recommended file; int4 is the size option.

### Spark-X2.5-1.7B / 4B (SparkLLM Team, thinking) — a remote-code architecture exported by patching the vendor file, not re-implementing it

`spark_work/convert_spark.py` converts [XHToken/Spark-X2.5-1.7B](https://huggingface.co/XHToken/Spark-X2.5-1.7B) and [XHToken/Spark-X2.5-4B](https://huggingface.co/XHToken/Spark-X2.5-4B) (`model_type spark2_5`, remote code; 28L / 36L dense, sliding-window(512)×3 : full×1 hybrid attention, per-head sigmoid attention-output gate, exact-GELU gated MLP, partial rotary 0.25 on the full layers, 131,072-entry tied vocab; **Apache-2.0**; thinking by default). Published: `litert-community/Spark-X2.5-1.7B`, `litert-community/Spark-X2.5-4B`.

```bash
# 1. make an export copy of the checkpoint (vendor modeling file patched; weights symlinked)
python3 spark_work/patch_modeling.py src_models/Spark-X2.5-1.7B src_models/Spark-X2.5-1.7B-export
# 2. int8 ship (int8 dynamic on linears + embedding)
python spark_work/convert_spark.py src_models/Spark-X2.5-1.7B-export out/spark-1.7b-int8 templates/spark25_think.jinja dynamic_wi8_afp32
# 3. int4 ship (blockwise-32 + OCTAV linears, int8 embedding) — EXTERNALIZE_EMBEDDER is REQUIRED (tied vocab)
EXTERNALIZE_EMBEDDER=1 python spark_work/convert_spark.py src_models/Spark-X2.5-1.7B-export out/spark-1.7b-int4 templates/spark25_think.jinja BOCTAV4
# 4B: the same three lines with PREFILL=1024,256,64,16,4,1 (six signatures) and BOCTAV4_128 for the int4 file
# gates: spark_work/gate8q_cli.py (8Q, thinking-aware), scripts/gate_specials.py (tokenizer parity),
#        spark_work/multiturn_gate.py, spark_work/eval_gsm8k_engine.py + spark_work/gsm8k_bf16.py (GSM8K, 3584-token budget)
```

**Tool-calling variant (agent use).** The published bundles carry the plain-chat subset of the vendor template. For
function calling, repack a bundle with `templates/spark25_tools.jinja` — the vendor template verbatim (tools rendered as
`## Tools … <tools>{tool.function|tojson}</tools>`, calls as `<tool_call>NAME<arg_key>K</arg_key><arg_value>V</arg_value></tool_call>`,
tool role as `<|Tool|><tool_response>…</tool_response>`) plus one edit so the tool turn LiteRT-LM hands over as content
blocks (`{type: tool_response, name, response}`) renders too:

```
python scripts/set_prompt_template.py out/spark-4b-int4/model.litertlm out/spark-4b-int4-tools.litertlm templates/spark25_tools.jinja
```

Metadata only; weights byte-identical (and the 4B's `prefer_activation_type = "fp32"` survives). The runtime's own
tool-call parser does not know the `<arg_key>` form, so an app parses the calls itself and answers with one `tool`
message carrying one `tool_response` block per call; the tool list can go in through `extra_context={"tools": [...]}`
(python) / `ConversationConfig(extraContext = mapOf("tools" to …))` (Kotlin). A phone-agent app built this way lives in
[edge-agent-lab/android/phone-agent](https://github.com/john-rocky/edge-agent-lab/tree/main/android/phone-agent).

`convert_spark.py` wraps `scripts/export_simple_template.py` (`USE_JINJA=1`, `NO_START_TOKEN=1`, KV 4096, prefill ladder 1024..1) plus the facts this checkpoint needs:

- **The vendor modeling file is patched for export (`spark_work/patch_modeling.py`, exact-string edits, each asserted unique).** `modeling_spark.py` computes attention through its own eager function and never reads `config._attn_implementation`, while litert-torch's export path works by registering `lrt_transposed_attention` and handing the model a KV cache whose `update()` returns k/v in a transposed layout only that interface understands. The patch dispatches through `ALL_ATTENTION_FUNCTIONS[impl]` when a non-eager implementation is set (eager stays byte-identical), applies the per-head sigmoid gate in the interface's `[B,T,N,H]` layout, threads `**kwargs` down to the attention call, declares `_supports_attention_backend` / `_supports_sdpa` / `_can_compile_fullgraph`, and marks `is_sliding`. Two further edits are needed just to load the vendor file under transformers 5.x (`_tied_weights_keys` list → mapping form; `create_causal_mask` kwargs `inputs_embeds`, no `cache_position`) — `--ref-only` produces a copy with only those, which is the parity reference. Measured: patched-eager vs reference-eager **max |Δlogit| 0.0** on 24 random tokens (both sizes); sdpa 2.5e-4 / 3.6e-4.
- **Jinja template carried verbatim** (`templates/spark25_think.jinja`): the vendor format always opens with a default system block, which the structured prefix/suffix templates cannot express. Byte-identical to the vendor `chat_template.jinja` on 18 shape × flag combinations (user / system+user / user+assistant+user × add_generation_prompt × enable_thinking); prefix-contract safe (assistant history renders the vendor's no-reasoning form `<|Bot|></think>{content}`); the literal `<think>` makes litert-torch declare the `thought` channel.
- **No `start_token`**: `add_bos_token` is false and the template carries its own `<｜start▁of▁sentence｜>` per message; a bf16 A/B with an extra BOS was 8/8 either way on the 8-question gate, but it is not the vendor prompt.
- **int4 must externalize the embedder.** The vocab is tied; int4 on the lm_head FULLY_CONNECTED + int8 on EMBEDDING_LOOKUP make the quantizer copy the 131,072-row table once per signature (tiny checkpoint: 4.58 vs 1.68 bytes/parameter, `scripts/check_bundle_sanity.py`). int8 needs no split.
- **Tokenizer = upstream `tokenizer.json`** (byte-level BPE, three Split regexes + Digits + ByteLevel); engine-vs-`tokenizers` ids identical on 234/234 probe rows.
- **Reasoning budget**: GSM8K at 2048 output tokens left a third of the bf16 model's questions mid-think; the parity rows use 3584 (the largest budget under the bundle's 4096 KV). 1.7B: bf16 76/100, int8 76/100 (paired: both 71, 5/5 split).
- **Measuring bf16 on MPS**: judge memory by `vmmap` physical footprint, not RSS; `generate` with the default DynamicCache re-allocates the KV cache every step and the caching allocator parks all of it (a 4B run peaked at 130 GB) — `gsm8k_bf16.py` uses a static cache, the MPS watermark cap and a per-question `empty_cache`.

### LFM2-2.6B-Exp — LFM2-2.6B's recipe unchanged, for a checkpoint that reasons although its card does not say so

`lfm_work/convert_lfm2_26b_exp.sh` converts [LiquidAI/LFM2-2.6B-Exp](https://huggingface.co/LiquidAI/LFM2-2.6B-Exp) at commit `2251c865` — "an experimental checkpoint built on LFM2-2.6B using pure reinforcement learning", with LFM2-2.6B's configuration: 30 layers (22 short-conv blocks + 8 grouped-query attention layers), hidden 2048, tied embeddings, 65,536 vocab — into the two files of [litert-community/LFM2-2.6B-Exp](https://huggingface.co/litert-community/LFM2-2.6B-Exp). The steps are the LFM2-2.6B entry's with the model, the commit and the file names changed. Environments (released wheels): litert-torch 0.9.4, litert-converter 0.4.0, ai-edge-quantizer 0.9.0, transformers 5.14.1 (5.15 breaks the lfm2 export), torch 2.13.0; litert-lm 0.18.0 for unpack / pack and for running the files.

```bash
EXPORT_PY=/path/to/python-with-litert-torch-0.9.4 \
RUN_PY=/path/to/python-with-litert-lm-0.18.0 \
LITERT_LM=/path/to/litert-lm \
  bash lfm_work/convert_lfm2_26b_exp.sh out_lfm2_26b_exp
```

| file | section | bytes | sha256 |
|---|---|---|---|
| `LFM2-2.6B-Exp_int8.litertlm` (2,723,809,744 B) | LlmMetadataProto | 3,862 | `b8771d3dcd862593a9eef0c933353dc8d7888bdee09cca7373a37c02d3237375` |
|  | ExecutorMetadataProto | 2,465 | `b9672684fb34216314d1c7cb4045f7520d056325ccc48d17a50d0ed80a0cd40e` |
|  | HF_Tokenizer_Zlib | 906,872 | `f7d531f34af15ec3a8ee935071d06f670a90018aa65c3e5ec9bcce41ad97d29b` |
|  | TFLiteModel | 2,722,843,088 | `ec42f6d866b57c230280f01d84458c7fbdc5980c8fa57e9b0e304ef04b29f6b7` |
| `LFM2-2.6B-Exp_int4.litertlm` (1,529,581,168 B) | LlmMetadataProto | 3,862 | `b8771d3dcd862593a9eef0c933353dc8d7888bdee09cca7373a37c02d3237375` |
|  | ExecutorMetadataProto | 2,465 | `b9672684fb34216314d1c7cb4045f7520d056325ccc48d17a50d0ed80a0cd40e` |
|  | HF_Tokenizer_Zlib | 906,872 | `f7d531f34af15ec3a8ee935071d06f670a90018aa65c3e5ec9bcce41ad97d29b` |
|  | TFLiteModel | 1,528,614,512 | `902fc8210c90984188d83773f419c5166a0d94e885d81bc08d04f2e14baa6e4e` |

The script printed these section hashes for both files when it was run from a copy of this repository's layout on 2026-10-13; every section equal to the published files' sections by raw bytes. The file sha256 differs from run to run (the packer writes a new uuid and creation timestamp), so compare sections.

- **The same steps as LFM2-2.6B.** The checkpoint is fetched at the pinned commit and every file is checked against the Hub's record. The whole LlmMetadata is replaced with the LiteRT-LM `models/lfm2` configuration at commit [86a7e6d0](https://github.com/google-ai-edge/LiteRT-LM/blob/86a7e6d098c3a6578208fc8538228fa704ca2e8b/models/lfm2/chat_template.jinja), with one line changed (`max_num_tokens` 32768 → 4096): Liquid AI's template for this checkpoint is the LFM2-2.6B file byte for byte and fails on litert-lm 0.18.0 the same way (`map has no method named get`). The metadata, executor and tokenizer sections come out byte-identical to the LFM2-2.6B files'.
- **The checkpoint reasons when it chooses.** Liquid AI's README for this checkpoint has no reasoning statement. Measured: "What is the capital of France?" opens a trace, and so did 1 of 120 IFEval-lite prompts; none of 100 GSM8K questions did. `supports_thinking` stays true and the configuration's `thought` channel takes the trace out of the reply; on litert-lm 0.18.0, `--thinking-budget 0` and `--thinking-budget 32` left this bundle's output unchanged.
- **OCTAV int4 block-32 writes 746,432 zero block scales on this checkpoint as well**, in the LFM2-2.6B layout: the short-conv input projection of 20 layers and the MLP w1 / w3 of layers 1, 2 and 5 (26 tensors, no subnormal scale). `minicpm_work/fix_zero_scales_inplace.py --zero-payload --require-normal` fixes them as in the LFM2-2.6B entry.
- **KV budget 4096 with the 11-signature prefill ladder (1024 … 1)**. The int8 file's TFLite section is 2,722,843,088 bytes, over 2 GiB; not run on iOS.
- **int8 = post-hoc dynamic int8 on the linears + embedding, convs float; int4 = OCTAV int4 blockwise-32 on the linears, int8 lm_head + embedding.** GSM8K (questions 0–99, greedy, at most 2048 output tokens, scored on the reply without the trace; litert-lm rows on the Mac GPU, one process per question): bf16 89, int8 87, int4 89. IFEval-lite (comparable only within the harness): bf16 102, int8 102, int4 101 of 120 prompts.
- **On CPU the int4 file does not always stop after its answer.** In 9 runs of one extraction prompt on the Mac (litert-lm 0.18.0 CLI) 3 kept repeating lines until prompt and reply filled the 4096-token KV budget, with the bundle's sampler and with greedy decoding; on the GPU the same prompt ended normally, and so did the int8 file on CPU. Long generation on CPU was not evaluated, and the cause is not established.
- Measured (prefill 1024 / decode 256 / context 2048, litert-lm 0.18.0 on the Mac, prebuilt `litert_lm_advanced_main` 2026-09-18 on the phones): Mac M4 Max GPU (WebGPU) int8 2441 / 123.4 tok/s, int4 2476 / 161.8; Galaxy S26 GPU int8 1141 / 20.4, int4 664 / 27.6; Pixel 8a GPU int8 499 / 11.1, int4 172 / 15.9 — the 2.7 GB int8 file runs on the Pixel 8a GPU with 501,160 kB of MemAvailable left at its lowest; every GPU signature fully delegated on all three.

## granite-4.0-h (Mamba2 + attention hybrid) — first Mamba2 hybrid on the released runtime

`granite_work/convert_granite4h.py` converts IBM's granite-4.0-h dense-hybrid models (Mamba2 selective-scan blocks interleaved with grouped-query attention) to `.litertlm`. Published: [litert-community/granite-4.0-h-1b](https://huggingface.co/litert-community/granite-4.0-h-1b). **Requires litert-lm ≥ 0.15 to run** (the hybrid conv/SSM states bind through the `ExecutorMetadata` section).

```bash
cd granite_work
git clone https://github.com/john-rocky/litert-torch litert-torch-granite
# the pinned base is no longer reachable from main (measured 2026-08-25), so a
# plain short-SHA checkout fails — fetch the commit by full SHA first
git -C litert-torch-granite fetch origin 115a13607c730c81018bb9789138a3e5e5119e3d
git -C litert-torch-granite checkout 115a13607c730c81018bb9789138a3e5e5119e3d
git -C litert-torch-granite apply "$(pwd)/granite_hybrid_litert_torch.patch"
PYTHONPATH=litert-torch-granite python convert_granite4h.py ibm-granite/granite-4.0-h-1b out_granite_1b
# GPU ship shape (2026-08-13): declare fp32 activations in the bundle TOML (repack, no re-export)
python ../scripts/set_activation_type.py out_granite_1b/granite-4.0-h-1b_int8.litertlm granite-4.0-h-1b_int8.litertlm --type fp32
```

The patch (against the pinned litert-torch base above) is what makes a state-carrying hybrid convert *and generate correctly* end to end. The non-obvious parts, in the order they were needed:

- **A Mamba2 export-cache layer** (conv `[B, conv_dim, K]` + SSM recurrent `[B, heads, head_dim, state]`) registered for Granite's layer types, so `torch.export` traces the model's own state contract instead of failing on the attention-only default cache.
- **State-continuation tracing.** Prefill graphs trace the chunk-continuation branch (previous conv/SSM state consumed) and the decode graph traces the single-step branch (conv window rolled by one, `has_previous_state` truthy at trace time). Without this the decode graph has no state continuity: it converts, loads and runs — and generates garbage from the second token on. Trap: `torch.export`'s pytree flatten/unflatten REBUILDS cache-layer objects mid-trace, so any mode flag must ride in the pytree context or it silently never reaches the graph.
- **The mamba prefill-pad guard.** The runtime's chunk planner runs prefill chunks *partially filled* (the remainder chunk is padded), and granite's own `apply_mask_to_padding_states` is a no-op at batch 1 — so pads poison the conv/SSM state at chunk-plan-dependent prompt lengths (the Mamba2 analog of the LFM2 ShortConv prefill-pad bug). The guard derives a valid mask from `tokens != 0` (the engine zero-fills pads), makes pad positions identity SSM steps (dt → ~0), and gathers the stored conv window at the last *valid* column with an in-graph one-hot matmul.
- **Quantize convs never.** Post-hoc dynamic int8 on linears + embedding only (`wi8fc`); export-time conv-int8 measurably costs quality on this family (8Q sanity 5/8 vs 6/8 on the 350m — same rule as the LFM2.5 convs).
- Verification bar for the ship: float-graph teacher-forced parity vs the HF reference = correlation 1.000000 with identical top-1 at every checked decode position; int8 sanity gate 8/8 (= the PyTorch reference); first-token robustness sweep against the runtime's real chunk plans at every prompt length 12–60, all clean.

### 2026-08-13 update — the folded scan: 3.5× on CPU, and the GPU wall falls

The published 1b was re-converted from the same weights with the patch's selective scan re-expressed: every contraction that the HF reference spells as broadcast-multiply-reduce (materializing intermediates that scale as `chunks × chunk_len² × heads × d_state`, some at rank 5–6) is rewritten as a **batched matmul with the chunk and head axes folded into the batch axis** — every tensor in the scan is rank ≤ 4, there is no `BROADCAST_TO`, no rank-2 `PAD`, and no int64 index math. The decode path drops its per-token `expand`-materializations for implicit broadcasting. The patch self-checks the folded form against the reference formulation numerically on every export.

What this buys, same weights, same recipe (Mac M4 Max, `litert-lm benchmark -p 256 -d 256 --runs 3 --cache no`, litert-lm 0.16.0):

| granite-4.0-h-1b int8 | prefill | decode |
|---|---|---|
| previous file, CPU | 69 tok/s | 11.4 tok/s |
| re-converted, CPU | **247 tok/s** | **39.2 tok/s** |
| re-converted, GPU | **1684 tok/s** | **134.7 tok/s** |

The CPU speedup is pure graph shape — the matmul form never builds the giant intermediates. The GPU part additionally needed two things:

- **Keep the batch dimension in the pad guard's reductions** (`valid.sum(-1, keepdim=True)`, not `sum(-1)[0]`). A rank-0 tensor entering broadcast arithmetic is *silently miscomputed* by the GPU delegate — the graph is accepted, `is_fully_accelerated` stays true, and every layer's conv state quietly zeroes. If a hybrid delegates 100% and still generates garbage, suspect rank-0 scalars before anything else.
- **Declare fp32 activations** (`scripts/set_activation_type.py`, above). The GPU executor's default fp16 activations overflow on the scan's intermediates; the failure mode is the engine's "Invalid decode and sample result" / all-zero tokens, not an error message.

Ship verification on the re-converted file: float parity vs HF teacher-forced across 8 decode positions — per-position max|logit diff| ≤ 1.1e-4, correlation 1.000000, top-1 and top-5 identical; int8 8-question gate **8/8 on CPU and GPU, on litert-lm 0.15.0 and 0.16.0**; on iPhone 17 Pro (Metal) 8/8 with decode 29.8 tok/s vs 15.9 CPU; on Pixel 8a (OpenCL) every subgraph fully delegated (zero rejections) with correct output. Low-end Android GPU decode does not beat CPU (decode is bandwidth-bound and the fp32-activation path reads 4× the bytes) — the on-device GPU win is Apple hardware, plus prefill/TTFT everywhere.

### granite-4.0-h-350m (fp16 + int8) — and the start_token lesson

Published: [litert-community/granite-4.0-h-350m](https://huggingface.co/litert-community/granite-4.0-h-350m) (fp16 723 MB + int8 436 MB). The 350m initially looked like it had "quantization jitter" (replies ending after a few tokens at some prompt lengths; multi-question prompts answered only at the last question). The actual root cause was **our packaging, not the model or the runtime**: the bundle's `LlmMetadata start_token` makes the engine prepend `<|end_of_text|>` to every prompt, Granite's official chat template has no leading BOS, and at 350M scale that one token flips the greedy trajectory (prepending the same token to the HF reference reproduces the degraded output verbatim — the graph was faithful all along). The 1b is robust to it; the 350m is not.

*Followed up 2026-08-27:* robustness is not fidelity, and the 1b's bundle was
left carrying that start_token. The audit of all published bundles confirms
`granite-4.0-h-1b` still prepends `<|end_of_text|>` where the reference stream
has no BOS at all — the same defect the 350m was fixed for, on a model that
merely tolerates it better. Same one-command fix (`drop_start_token.py`).

```bash
cd granite_work
PYTHONPATH=litert-torch-granite python convert_granite4h.py ibm-granite/granite-4.0-h-350m out_granite_350m
python drop_start_token.py out_granite_350m/granite-4.0-h-350m_int8.litertlm granite-4.0-h-350m_int8.litertlm
python ../lfm_work/add_executor_metadata.py out_granite_350m/model.litertlm out_granite_350m/model_fp_meta.litertlm
python make_fp16_variant.py out_granite_350m/model_fp_meta.litertlm granite-4.0-h-350m_fp16.litertlm --drop-start-token
```

Gates on the published files (litert-lm 0.15 CPU, Mac + iPhone 17 Pro): fp16 = 8/8 sanity, greedy output token-identical to the HF fp32 reference on our probes, prompt-length sweep 12–200 clean; int8 = 8/8 sanity, sweep clean except a known 33–37-token band where replies can end early (int8 noise interacting with one prefill chunk shape — fp16 is clean there; documented on the card). On phones ship int8: the CPU runtime unpacks fp16 weights to fp32 in RAM (~3.7 GB peak on iPhone vs ~2.1 GB for int8).

#### The GPU variant, and why the published two are not it

The two files above are CPU files, and the reason is in the graph rather than in the runtime. `transformers`
writes the Mamba2 SSD contractions as broadcast-multiply-reduce, which materialises rank-5 and rank-6
intermediates; every mobile GPU delegate refuses tensors above rank 4. Measured on the published lineage,
`prefill_1024` carries 411 rank-6 and 739 rank-5 tensors, and the Adreno delegate takes 109 of its 3673 ops
before the engine gives up.

The exporter patch has since rewritten those five contractions as batched matmuls with the chunk and head axes
folded into the batch axis. Re-exporting on it gives a graph that is rank ≤ 4 everywhere, and one extra
declaration takes it the rest of the way:

```bash
cd granite_work
PYTHONPATH=litert-torch-granite python convert_granite4h.py ibm-granite/granite-4.0-h-350m out_granite_350m
python drop_start_token.py out_granite_350m/granite-4.0-h-350m_int8.litertlm g350_nobos.litertlm
python ../scripts/set_activation_type.py g350_nobos.litertlm granite-4.0-h-350m_int8_gpu.litertlm --type fp32
```

`prefer_activation_type = "fp32"` is not optional at this size. Without it the file still delegates, but int8
activation noise on the GPU costs real quality and speed — measured on an M4 Max, 5/8 on the 8-question gate
at 76 tok/s decode, against **8/8 at 148 tok/s** with the declaration. The archive had this recorded as a
1b-only rescue; that was single-gate evidence and it did not survive re-measurement.

Gate, Galaxy S26 (Adreno), `litert_lm_advanced_main` v0.16.0: **every one of 12 subgraphs fully delegated in a
single partition, no XNNPack split, correct answer.** 205-token benchmark, n=2: 486 / 484 tok/s prefill,
48.8 / 47.9 tok/s decode, TTFT 0.48 s, peak VmHWM ~2.13 GB. Engine init is 8.6–9.6 s and is separate from TTFT.

Two things this does **not** fix:

- **fp16 stays CPU-only.** The rank wall is gone from it too, but float-casting `EMBEDDING_LOOKUP` leaves
  `Empty quantization params` and a `DEQUANTIZE` the delegate will not take — 3326 of 3536 ops, engine refused.
  The int8 recipe quantizes the same table and delegates. There is no fp16 GPU variant.
- **The engine-reuse band moves, it does not close.** The published int8 file breaks at chat-templated lengths
  33–37 when one `Engine` is reused across a growing shared prefix; on the re-export the band is **37–41**. It
  is the runtime carrying recurrent state across conversations ([LiteRT-LM#3165](https://github.com/google-ai-edge/LiteRT-LM/issues/3165)),
  not the weights, and a fresh `Engine` per conversation is clean at every length tested either way (35–45 on the re-export;
  the hermetic probe's template overhead puts its own floor at 18).

The cheapest way to check a re-export before spending device time is to unpack it and histogram tensor ranks
per subgraph: rank ≥ 5 anywhere in a prefill graph means the delegate will refuse, and no phone is needed to
know it.

### granite-4.0-h finetune intake (Tashkeel-350M-v2) — the recipe rides derivatives unchanged

Measured 2026-08-25 on [Etherll/Tashkeel-350M-v2](https://huggingface.co/Etherll/Tashkeel-350M-v2)
(Arabic diacritization SFT of granite-4.0-h-350m; the most-downloaded granite-4.0-h derivative
on the Hub). Published: [mlboydaisuke/Tashkeel-350M-v2-LiteRT](https://huggingface.co/mlboydaisuke/Tashkeel-350M-v2-LiteRT)
(fp16 exact-parity + int8, measured card + litertlm_manifest.json). Family fact, same as
every LLM family measured so far: the finetune's
`chat_template.jinja` is byte-equal to the base's, and the tokenizer/generation config diffs are
cosmetic (`model_max_length`, eos as list) — so the base recipe applies verbatim:

```bash
cd granite_work
PYTHONPATH=litert-torch-granite python convert_granite4h.py Etherll/Tashkeel-350M-v2 out_tashkeel
python drop_start_token.py out_tashkeel/Tashkeel-350M-v2_int8.litertlm Tashkeel-350M-v2_int8.litertlm
# exact-parity variant (the int8 flips two diacritics on the 10-case gate; fp16 is clean)
python ../lfm_work/add_executor_metadata.py out_tashkeel/model.litertlm out_tashkeel/model_fp_meta.litertlm
python make_fp16_variant.py out_tashkeel/model_fp_meta.litertlm Tashkeel-350M-v2_fp16.litertlm --drop-start-token
python gate_tashkeel.py Tashkeel-350M-v2_fp16.litertlm --backend cpu
```

Three-point proof (litert-lm 0.16.0 CLI, CPU): (a) export ✓ — 481 MB int8, 64 state buffers
(56 conv/SSM + 8 KV), ExecutorMetadata present, stop token = the tokenizer's
`<|end_of_text|>` 100257; (b) embedded template byte-equal 6418/6418 to the checkpoint's;
(c) greedy A/B vs HF fp32 on the model's own task (`gate_tashkeel.py`: the card's worked
example + 9 undiacritized MSA probes, both sides rendering the identical string):
**float 10/10 and fp16 10/10 byte-exact**, int8 8/10 — the two int8 misses are
single-diacritic greedy flips (one also over-runs past the natural end), and both probes are
byte-exact on the float parent, so they are quantization cost, not conversion error — the same
350M-scale int8 sensitivity the base model's card documents (fp16 is the exact-parity variant
there too).
The start_token drop is REQUIRED here as on the base: an HF BOS A/B flips the greedy output
from correct diacritization to garbage (`صلى السلام عليكم`).
Honest note: the model card's own worked example (`اَلسَلَامُ عَلَيْكُمْ`) matches neither the
bundle nor HF fp32 greedy (`السَّلَامُ عَلَيْكُمْ`) — the card example disagrees with the
checkpoint's own greedy output, so HF parity, not the card string, is the certification.

Stock-export failure shape for the record (why the pinned checkout stays): released
litert-torch 0.9.4 has zero mamba support in the wheel; its generic path maps granite's mamba
layers onto the qwen3.5-era linear-attention cache and dies before tracing —
`AttributeError: 'GraniteMoeHybridConfig' object has no attribute 'linear_key_head_dim'`
(`export_hf/core/cache.py`, `create_from_config`). `scripts/convert.py` now routes
`model_type: granitemoehybrid` to this recipe automatically (HYBRID_RECIPE) when the pinned
checkout is present, and refuses with the exact setup command when it is not; task-specific
derivatives gate through `--gate-script` (the generic 8-question gate certifies nothing on a
model that diacritizes its input instead of answering). One-command run, measured end-to-end
from a clean shell:

```bash
python scripts/convert.py Etherll/Tashkeel-350M-v2 --gate-script granite_work/gate_tashkeel.py
```

recipe export 454 s → start_token dropped → task gate on CPU → int8 8/10 (the same two
single-diacritic flips, reproduced deterministically) → exit 1 `converted_gate_failed` with the
bundle and `convert_report.json` written. The exit code is the honest int8-at-350M verdict, not
a conversion failure — the fp16 flow above is the exact-parity finish.

## Qwen3.5 (GatedDeltaNet + attention hybrid) — first Qwen3.5 in LiteRT form

`qwen35_work/convert_qwen35_hybrid.py` converts Qwen3.5 hybrids (gated-delta-rule linear attention interleaved with gated full attention; the 0.8B is 18 + 6 layers) to `.litertlm`, text decoder only (the multimodal checkpoint's vision tower and MTP heads are dropped, matching upstream's own `Qwen3_5ForCausalLM` load contract). Published: [litert-community/Qwen3.5-0.8B](https://huggingface.co/litert-community/Qwen3.5-0.8B). **Requires litert-lm ≥ 0.15 to run** (the conv/recurrent states bind through the `ExecutorMetadata` section).

```bash
cd qwen35_work
git clone https://github.com/john-rocky/litert-torch litert-torch-qwen35
git -C litert-torch-qwen35 fetch --depth 1 origin 115a13607c730c81018bb9789138a3e5e5119e3d  # no ref reaches this commit any more
git -C litert-torch-qwen35 checkout --detach 115a13607c730c81018bb9789138a3e5e5119e3d
git -C litert-torch-qwen35 apply "$(pwd)/qwen35_hybrid_litert_torch.patch"
PYTHONPATH=litert-torch-qwen35 python convert_qwen35_hybrid.py Qwen/Qwen3.5-0.8B out_qwen35_08b
# GPU ship shape (2026-08-13): declare fp32 activations in the bundle TOML (repack, no re-export)
python ../scripts/set_activation_type.py out_qwen35_08b/Qwen3.5-0.8B_int8.litertlm Qwen3.5-0.8B_int8.litertlm --type fp32
```

The patch (against the pinned litert-torch base, transformers ≥ 5.14) shares the granite hybrid machinery — export-cache layers for `layer_types == "linear_attention"`, state-continuation tracing (fused single-step decode branch + chunked-continuation prefill branch), and the prefill-pad guard (pads become identity delta-rule steps; the conv window is gathered at the last valid column). Qwen3.5-specific parts, in the order they were needed:

- **Constant-eye chunk kernel.** The reference chunked delta rule builds `torch.eye` inside the traced function; it lowers to `STABLEHLO_IOTA`, which no released TFLite kernel set registers — the exported file will not even load into an Interpreter. The kernel is vendored with the identity matrix lifted as a graph constant (`torch.tensor` from python data). The triu masks and aranges constant-fold on their own; only `eye` survives as a runtime op.
- **Unwrap `@force_accelerate_hooks`.** The mixer's `forward` is wrapped without `functools.wraps`, so `inspect.getsource` returns the accelerate wrapper; the real function is fished out of the wrapper's closure before the anchored source replacements.
- **Delta-rule pad identity is on `a`, not dt.** Zeroed pad input alone still decays the recurrent state (softplus(0 + dt_bias) ≠ 0); the guard forces the pre-softplus gate input to −30 on pads so per-token decay is exp(0) = 1, and re-zeroes q/k/v after the conv so pads inject nothing.
- **Chat template replaced.** The stock template strips the `<think>` block from history assistant turns, which violates LiteRT-LM's incremental conversation rendering (each turn's render must string-extend the previous one) and hard-fails turn 2. The bundled simplified ChatML template keeps the empty think block in both the generation prompt and history renders — multi-turn then works exactly (`chat_template_simple.jinja`).
- **Quantize convs never**: post-hoc dynamic int8 on linears + embedding (`wi8fc`), convs and the delta rule stay float — same rule as LFM2.5 / granite.

Verification bar for the ship: float-graph teacher-forced parity vs the HF reference = correlation 1.000000 with identical top-1 at every checked decode position (max|logit diff| ≤ 3.7e-5); an 8-question sanity gate answered **word-for-word identically to HF fp32 by both the float and the published int8 build**; first-token robustness sweep against the runtime's real chunk plans at every prompt length 12–60, all clean; multi-turn state continuity verified.

### 2026-08-13 update — the GPU wall falls (and a delegate bug found on the way)

The published 0.8B was re-converted from the same weights with the vendored chunk kernel re-expressed the same way as granite's scan: contractions as **batched matmuls with chunk and head axes folded into the batch axis**, every tensor rank ≤ 4, no `BROADCAST_TO`, no int64 index math. Two GPU-specific lessons came out of getting it not just delegated but *correct*:

- **Never `PAD` a rank-3 tensor on a non-final axis.** The GPU delegate executes it wrong — a bit-exact one-row shift along the outermost axis (row 0 zeros), silently, with `is_fully_accelerated` still true. In this kernel that shifted every gate to the neighbouring head. Reported upstream with a 1-op repro: [LiteRT#9272](https://github.com/google-ai-edge/LiteRT/issues/9272). The kernel now writes every tail-pad as `torch.cat` with a zeros constant (unsqueezing to rank 4 before padding is also exact, if you prefer `PAD`).
- **Declare fp32 activations** (`scripts/set_activation_type.py`, above). At fp16 the remaining failure is not the converter: one layer-0 head's real weights push fp16 intermediates over range — a property of the checkpoint.

Re-ship verification: parity re-run on the ship export (48 positions, top-1/top-5 100%, Pearson 1.0000); 8-question gate **8/8 on CPU and GPU on both litert-lm 0.15.0 and 0.16.0**; prompt-length robustness with a fresh engine per length 40/40 CPU + 20/20 GPU; on iPhone 17 Pro (Metal) 8/8 with decode 41.4 tok/s vs 14.6 CPU. CPU speed is unchanged (the kernel was already matmul-form on CPU): Mac M4 Max CPU 666/46.7 tok/s, GPU 1972/161.8 (`-p 256 -d 256 --runs 3 --cache no`, 0.16.0). One honest limit: a Pixel 8a cannot compile the full prefill-ladder ship file on its GPU (fp32-expanded weights + ten compiled programs exceed ~3.8 GB available); a reduced dev shape of the same graph runs there fully delegated and correct, and CPU is unaffected.

### 2026-08-14 — Qwen3.5-4B: two walls the 0.8B never hit

Same driver, same patch, one command (`convert_qwen35_hybrid.py Qwen/Qwen3.5-4B out_qwen35_4b`, then the fp32act repack). Published: [litert-community/Qwen3.5-4B](https://huggingface.co/litert-community/Qwen3.5-4B). The 4B (24 linear-attention + 8 attention layers) is architecturally the 0.8B scaled up — but it trips two size-dependent walls, both now handled in the patch/driver:

- **Grouped value heads reject on GPU.** The 4B is the first Qwen3.5 with `linear_num_value_heads` (32) ≠ `linear_num_key_heads` (16); upstream expands q/k with `repeat_interleave(ratio, dim=2)`, guarded by `if ratio > 1` — so the 0.8B (ratio 1) never traces it. Its rank-5 `unsqueeze→expand→reshape` lowering is rejected by the GPU delegate (`RESHAPE: Tensor dimensions must be less than 5`) and engine creation fails outright. The patch re-expresses it rank-≤4: fold the batch dim, materialize copies with `torch.cat`, re-fold — bitwise-identical to the stock op.
- **Signature count is charged in RAM, and a 248k-vocab 4B pays too much per signature.** With the 0.8B's full prefill ladder (11 lengths + decode = 12 signatures), a 12 GB iPhone jetsam-kills the app at Metal program init #9 — before any inference. Every signature costs engine memory even if never called. The 4B ships a reduced ladder (`QWEN35_PREFILL_LADDER=1024,256,64,16,4,1`, 7 signatures): GPU peak drops to ~4.9 GB on iPhone 17 Pro and the engine creates cleanly, at no correctness cost (the runtime plans slightly coarser chunks; the prompt-length sweep below re-verified against the new plans).

Ship verification (all on the shipped file or its float parent): teacher-forced parity vs HF fp32 across 48 positions — top-1/top-5 100%, Pearson 1.0000, KL ≈ 0; 8-question gate **8/8 on CPU and GPU on both litert-lm 0.15.0 and 0.16.0**; prompt-length robustness, fresh engine per length, 40/40 CPU + 20/20 GPU; iPhone 17 Pro (Metal) answers the composite 8-question probe **word-for-word identically to HF fp32** at 9.3 tok/s decode / 87 tok/s prefill / 4.98 GB peak (CPU: 5.4 / 24.0 / 1.70 GB). Mac M4 Max (`-p 256 -d 256 --runs 3 --cache no`, 0.16.0): GPU 672 / 62.0 tok/s, CPU 243 / 19.9. Honest limits: on the composite probe the CPU int8 path answers 6 of 8 (everything it says is correct; individual questions are 8/8) — the GPU path is the fidelity path; and 8 GB-class Android phones cannot fit the 4.1 GB file (Apple-hardware-first release).

### 2026-08-14 — Qwen3.5-2B: the easy sibling (both 4B walls stay un-tripped)

Same driver, same patch, one command (`convert_qwen35_hybrid.py Qwen/Qwen3.5-2B out_qwen35_2b`, then the fp32act repack). Published: [litert-community/Qwen3.5-2B](https://huggingface.co/litert-community/Qwen3.5-2B). The 2B (18 linear-attention + 6 attention layers, hidden 2048) sits between the 0.8B and 4B and needs **no model-specific work**:

- Its linear-attention heads are ungrouped (`linear_num_key_heads` = `linear_num_value_heads` = 16, ratio 1), so the 4B's grouped-value-head interleave rewrite never traces — same as the 0.8B.
- The **full prefill ladder fits on-phone** (11 prefill lengths + decode = 12 signatures): iPhone 17 Pro GPU peaks at ~5.3 GB during engine creation, comfortably under the 12 GB device's jetsam ceiling — the 4B's reduced-ladder workaround is not needed at this size.

Ship verification (all on the shipped file or its float parent): teacher-forced parity vs HF fp32 across 48 positions — top-1/top-5 100%, Pearson 1.0000, KL ≈ 0; 8-question gate **8/8 on CPU and GPU on both litert-lm 0.15.0 and 0.16.0**; prompt-length robustness, fresh engine per length, 40/40 CPU + 20/20 GPU; iPhone 17 Pro (Metal) answers the composite 8-question probe **word-for-word identically to HF fp32 through answer 7** — including the model's own arithmetic slip on question 1 — then adds a correct 8th answer where fp32 ends its turn one token earlier, at 24.3 tok/s decode / 237.7 tok/s prefill / TTFT 0.73 s / 5.33 GB peak (CPU: 16.2 / 206.5 / 0.77 s / 1.52 GB). Mac M4 Max (`-p 256 -d 256 --runs 3 --cache no`, 0.16.0): GPU 1486 / 114.3 tok/s, CPU 592 / 37.6. Honest limits: on the composite probe the CPU int8 path degrades (3 of 8, deterministic, identical on Mac and iPhone; individual questions are 8/8 everywhere) — the GPU path is the fidelity path; and on a Pixel 8a the OpenCL delegate accepts the whole graph (zero rejections, full delegation on every compiled signature) but engine creation exhausts an 8 GB phone's memory before finishing — Apple-hardware-first, 12 GB+ for Android GPU.

### 2026-08-24 — Finetune intake: stock litert-torch *main* converts Qwen3.5 derivatives

The recipe above reproduces the shipped checkpoints; **finetunes of Qwen3.5 now go through
`scripts/convert.py`** instead. Measured on the 0.8B base plus a real 4B product finetune:

- **litert-torch main ships a `qwen3_5` model_ext** (absent from every released wheel — PyPI
  tops out at 0.9.3, no nightly package, no tag contains it): full reauthored exportables
  routed on `model_type in ('qwen3_5', 'qwen3_5_text')`, no patch, no template surgery.
  One-command install into a **fresh venv**: `pip install 'litert-torch @
  git+https://github.com/google-ai-edge/litert-torch.git'` (measured @ `8379afb` =
  0.10.0.dev; its setup.py pins nightly ai-edge deps — do not share a venv with the released
  0.9.3 pipelines). On 0.9.3 the generic path dies mid-trace
  (`LiteRTLMConvCacheLayer.update_conv_state() takes 2 positional arguments but 3 were
  given`); convert.py pre-empts that with a `model_ext_missing` refusal carrying the install
  action, before any download.
- **Stock export embeds the derivative's own template verbatim and emits the
  ExecutorMetadata section** (conv/recurrent state binding, litert-lm ≥ 0.15) that the
  recipe era added by hand. Qwen/Qwen3.5-0.8B: 226 s, 754 MiB int8, template byte-equal
  7755/7755, CPU gate 6/8 PASS ~47 tok/s. The 6/8 — the shipped wi8fc build is 8/8
  word-for-word — is the stock export-time int8 quantizing convs + delta rule against the
  family rule; that quality notch is the price of the no-recipe path.
- **CPU-only.** The GPU delegate rejects the stock GatedDeltaNet lowering (`TRANSPOSE:
  Permutation for transpose is invalid`, 195/20239 ops delegated) and engine creation fails
  outright — no fallback — so convert.py's exit gate runs `--backend cpu`
  (verify_quality.py grew that flag). GPU-delegable Qwen3.5 remains the per-model recipe
  above.
- **≥3B derivatives get the 4B ship's reduced prefill ladder** (1024,256,64,16,4,1 + decode
  = 7 signatures): signatures are charged in engine RAM even if never called (2026-08-14
  above), and the full-ladder 4B export's converter passes were OOM-killed twice on a
  128 GB host before the reduction (even reduced, the export peaks > 84 GB and swaps hard;
  ~26 min).
- **Proof finetune — numind/NuExtract3** (4.54B multimodal checkpoint, MTP weights in a
  separate file, 123k DL): one command → `converted_pass`, gate **8/8** non-degenerate
  ~20 tok/s CPU, embedded template byte-equal (6,692 chars — including its
  `{% generation %}` blocks, which the current runtime parses), and greedy **A ≡ B**
  byte-identical (runtime-applied template == HF-side render fed raw) with **C ≠ A** (the
  base template's thinking-ON default yields a different answer — the template diff is
  behavior-bearing, and the runtime is applying the derivative's).
- Multi-turn honesty: the stock Qwen3.5 template strips `<think>` from history assistant
  turns, which violates the runtime's incremental render contract (see the recipe above) —
  derivatives inheriting it are single-turn-proven only.
- ⭐ **Qwen3.5 checkpoints under-declare their stop tokens, and both of the stock
  exporter's derivations miss the turn end.** The repos ship no generation_config.json and
  declare only `text_config.eos_token_id = 248044` (`<|endoftext|>`); the exporter's
  template-derived stop never fires because the stock template's probe render hits its own
  `raise_exception` (the "Failed to parse chat template" line in the export log is
  load-bearing). Every stock Qwen3.5 bundle stops ONLY at `<|endoftext|>` —
  `<|im_end|>` = 248046 missing — which the base/NuExtract3 gates masked (those models
  happen to emit `<|endoftext|>` after answering) and a thinking derivative exposed:
  generation runs straight through `<|im_end|>` and the gate reads 0/8 empty.
  `convert.py` now runs `ensure_turn_end_stop()` after every export: if the checkpoint
  tokenizer's `eos_token_id` is missing from the bundle's stop_tokens, it is appended
  (litert_lm_builder unpack → LlmMetadataProto.pbtext → repack); no-op when stops are
  declared correctly (the dense rail).
- **Adapter derivatives ride merge-first unchanged** (cs552-the-expendables/
  patientagent-sft-glm5, r16 α32 targeting the GatedDeltaNet projections):
  `AutoModelForCausalLM` on the multimodal base loads text-only (426 tensors, 0 vision;
  config rewritten to `qwen3_5_text` — why HYBRID_STOCK keys both model_types), and the
  merged hybrid-layer weights verify bitwise as `W + (α/r)·B@A`. That subject then FAILED
  the gate 0/8 for a model-own reason worth knowing: it inherits the base 4B template
  (thinking-ON default) but its SFT never learned to close `<think>` — measured identical
  in HF (greedy emits the answer immediately and never `</think>`), so the runtime files
  the whole answer into the filtered thought channel. The conversion is faithful; the gate
  correctly refuses a model that cannot exit its think block. peft must be installed in
  the export venv (peft 0.20.0 + accelerate 1.14.0 alongside the main stack import-clean).

### 2026-09-04 — Qwen3.5-0.8B / 2B MTP (speculative decoding): built and gated, held by two runtime findings

`mtp_work/` exports Qwen3.5 with a `verify` signature and packs the checkpoint's own 1-layer MTP head as the runtime's drafter section, so the bundle runs under `litert-lm run --speculative-decoding true` (≥ 0.16.0). Recipe, patch, gates and numbers: `mtp_work/README.md`. The short version: Mac CPU decode 1.12× (0.8B) / 1.24× (2B) with acceptance matching the desktop oracle; Galaxy S26 CPU 0.71× on the 0.8B for document text; and a runtime defect — under the flag the turn-final stop token is never committed, so multi-turn transcripts lose `<|im_end|>` after every assistant turn (LiteRT-LM #3439, reproduces on 0.16.0 and 0.16.1; probe script in the directory). Not shipped; a repro bundle is public at `mlboydaisuke/Qwen3.5-0.8B-MTP-repro-LiteRT`.

### 2026-10-11 — Qwen3.5-2B int8 re-quantized: the CPU repetition loop was activation quantization of the GatedDeltaNet decay/gate heads

The shipped `Qwen3.5-2B_int8.litertlm` (2026-08-14 recipe) looped on multi-step arithmetic at its default (thinking off) on the **CPU backend only**: `What is 17 times 23?` gave `340 + 17 = 340 + 17 = …` to the token cap in 3 of 3 fresh litert-lm 0.18.0 processes on a Mac M4 Max, and `391` followed by `$391$` repeated to the cap on a Pixel 8a. The same file answered 391 on the GPU, and the float export of the same graph answered 391 on the CPU — so the weights were not the problem. LiteRT-LM's CPU path runs dynamic-int8 FULLY_CONNECTED ops with the input row quantized to int8 at run time (XNNPACK `qd8`, enabled by the runtime's `DYNAMIC_FULLY_CONNECTED` delegate flag); the GPU dequantizes the weights and multiplies in float. Bisecting one FC group at a time (each group left float via an ai-edge-quantizer `NO_QUANTIZE` override, everything else the shipped int8, two fresh CPU processes per build) found the culprit: the GatedDeltaNet decay/gate projections `in_proj_a` / `in_proj_b` (16 outputs each, 36 FCs, 3.4 MB). Float `down_proj` (including the layer with a 128× massive activation channel), float attention projections, float gate/up or SmoothQuant-style per-channel scaling did not fix it; float `in_proj_a` + `in_proj_b` alone did.

The re-quantized file keeps only those 36 projections in float; every other tensor is the previous quantization byte for byte (the baseline rebuilt from the same export reproduces the shipped TFLite section byte-identically, so the control is exact). Measured (one fresh process per run): Mac CPU six digit questions 17/18 at a 400-token cap (the 18th a correct derivation cut by the cap; uncapped 3/3), gate questions 8/8 on CPU and GPU, Pixel 8a CPU 3/3 clean, Galaxy S26 CPU 4/4 and GPU 3/3; the shipped control on the same phones: Mac CPU 5/18, Pixel 8a 0/3 and Galaxy S26 CPU 0/1 (degenerate: `391`, then `$391$` repeated to the cap), gate 8/8. Speed is unchanged: on the Galaxy S26 (prebuilt `litert_lm_advanced_main` of 2026-09-18, prefill 256 / decode 64, median of three runs) CPU 567.3 / 22.90 tok/s against the shipped file's 562.1 / 22.75, GPU 308.2 / 19.17 tok/s. File: 2,120,039,328 B (+3,446,512 B), sha256 `e7473f57…`. GPU activations stay fp32 (the fp16 override emits `!` from token 0 on this family).

Reproduce (the 2B; the 0.8B / 4B recipes are unchanged until measured):

```bash
cd qwen35_work   # litert-torch-qwen35 at the pinned base + patch as above
QWEN35_FLOAT_FC_REGEX='Linear_in_proj_(a|b);' PYTHONPATH=litert-torch-qwen35 \
  python convert_qwen35_hybrid.py Qwen/Qwen3.5-2B out_qwen35_2b
python ../scripts/set_activation_type.py out_qwen35_2b/Qwen3.5-2B_int8.litertlm Qwen3.5-2B_int8.litertlm --type fp32
```

The command above reproduces the Hub file's TFLite section byte for byte. The Hub bundle's metadata carries one detail added on 2026-09-21 that this driver does not write (a dual-form ChatML template that also renders content-part messages); it is a metadata re-pack of the same sections. (Corrected 2026-10-12: this paragraph first also listed `<|im_end|>` as a stop token the driver does not declare — the driver has declared it itself since 2026-08-07.)

`quantize_minicpm5.py apply --float-fc-regex RE` (new, repeatable) adds a `NO_QUANTIZE` config for every FULLY_CONNECTED whose scope matches `RE`; the scope string ai-edge-quantizer matches against is the op's **output tensor name followed by `;`**, so anchor with the `;` and never with `$` (a `$`-anchored regex silently matches nothing and the build is the baseline again — check `inspect` for the float FC count before trusting a candidate). Rule of thumb for hybrid (GatedDeltaNet / Mamba) models: the tiny recurrence-parameter heads (decay / gate projections) are the sensitive int8 consumers on the CPU path; keeping them float costs megabytes. When a file misbehaves on CPU but not on GPU, test the GPU and the float export before touching the weights — the defect is in the activation path, and a one-group-float bisection finds the consumer in a handful of 40-second builds.

### 2026-10-12 — Qwen3.5-0.8B int8 and VL int8 re-quantized the same way

The shipped 0.8B files had the 2B's CPU defect, intermittently (litert-lm 0.18.0, Mac M4 Max CPU, one fresh process per run): with ThinkingConfig unset, `What is 17 times 23?` repeated `17 × 20 + 17 × 3 = ` to the cap in 1 of 3 runs of each file, and six multiplication questions were right in 7 of 18 runs (text) and 12 of 18 (VL). A Pixel 8a CPU got the prompt right in every run (text 3 of 3, VL 2 of 2), too few to say the phone differs. Both files now keep the 36 GatedDeltaNet decay/gate FCs float (why: *2026-10-11 — Qwen3.5-2B int8 re-quantized* above); the VL file only in its decoder section. Measured: Mac CPU 18 of 18 on both files, Galaxy S26 CPU and GPU 3 of 3 each, GSM8K (n = 100, text) 11 → 49 on CPU and 52 on GPU, bf16 51.

```bash
cd qwen35_work   # litert-torch-qwen35 at the pinned base + patch as above
QWEN35_FLOAT_FC_REGEX='Linear_in_proj_(a|b);' PYTHONPATH=litert-torch-qwen35 \
  python convert_qwen35_hybrid.py Qwen/Qwen3.5-0.8B out_qwen35_08b
python ../scripts/set_activation_type.py out_qwen35_08b/Qwen3.5-0.8B_int8.litertlm Qwen3.5-0.8B_int8.litertlm --type fp32
```

```bash
cd qwen35vl_work   # vision and bundle steps: "The 0.8B rides the same script"
QWEN35_FLOAT_FC_REGEX='Linear_in_proj_(a|b);' PREFILL=1024,256,64,16,4,1 CACHE=4096 \
  PYTHONPATH=../qwen35_work/litert-torch-qwen35 \
  python export_qwen35vl_decoder.py Qwen/Qwen3.5-0.8B out/qwen35vl08-decoder
```

The text command, run end to end on 2026-10-12 (the pinned worktree, ai-edge-quantizer 0.8.0 or 0.10.0 — both give the same bytes — and `litert-lm-builder` 0.18.0 on `PATH`), writes the Hub file's TFLite section byte for byte; the Hub bundle's chat template is the 2026-09-21 metadata re-pack that also accepts the 0.18 content-parts form. The VL command with `SKIP_EXPORT=1` on the 2026-08-28 fp32 decoder and embedder (ai-edge-quantizer 0.8.0) writes the Hub VL file's decoder and embedder sections byte for byte; a fresh float export on 2026-10-12 from the same worktree gives a decoder section with different bytes (sha256 `466c8560…`), so the published VL decoder is tied to its 2026-08-28 export.

### 2026-10-13 — Qwen3.5-4B int8 re-quantized the same way

The shipped `Qwen3.5-4B_int8.litertlm` had the 2B's CPU defect, intermittently (litert-lm 0.18.0, Mac M4 Max CPU, one fresh process per run, 400-token cap): six multiplication questions, three runs each, reached the cap in 7 of 18 runs, 6 of them repeating one line after the right product; with ThinkingConfig unset, `What is 17 times 23?` reached the cap after its answer in 1 of 3 runs; and with thinking on through a raw `<think>` prefill and no budget, 0 of 3 runs closed the thought. The file now keeps the 48 GatedDeltaNet decay/gate FCs (`in_proj_a` / `in_proj_b` of the 24 linear-attention layers) float (why: *2026-10-11 — Qwen3.5-2B int8 re-quantized* above); a fresh float export, quantized with the previous recipe, reproduces the shipped TFLite section byte for byte. Measured on the new file: Mac CPU 18 of 18 with no repeated line, thinking on without a budget 3 of 3 closed, the 8-question sanity gate 8/8 on CPU and GPU, Mac GPU 6 of 6, Galaxy S26 CPU 4 of 4 and 178.9 / 8.93 tok/s prefill / decode there (median of iterations 2–3, clocks capped).

```bash
cd qwen35_work   # litert-torch-qwen35 at the pinned base + patch as above
QWEN35_PREFILL_LADDER=1024,256,64,16,4,1 QWEN35_FLOAT_FC_REGEX='Linear_in_proj_(a|b);' PYTHONPATH=litert-torch-qwen35 \
  python convert_qwen35_hybrid.py Qwen/Qwen3.5-4B out_qwen35_4b
python ../scripts/set_activation_type.py out_qwen35_4b/Qwen3.5-4B_int8.litertlm Qwen3.5-4B_int8.litertlm --type fp32
```

The command above, run end to end on 2026-10-13 (the pinned worktree, ai-edge-quantizer 0.8.0 and `litert-lm-builder` 0.18.0 on `PATH`), writes the new Hub file's TFLite section byte for byte (sha256 `f9f6fc26…`); the Hub bundle's metadata is the 2026-10-09 re-pack with the thinking switch.

### 2026-10-16 — Ornith-1.0-9B: a Qwen3.5-9B finetune on the 9B hybrid recipe, with thinking off in the template

`qwen35_work/convert_qwen35_9b.sh` converts [ornith-ai/Ornith-1.0-9B](https://huggingface.co/ornith-ai/Ornith-1.0-9B) at commit `83dc1f5e` — Ornith's finetune of Qwen3.5-9B, "the most lightweight member of the Ornith family" — into the two files of [litert-community/Ornith-1.0-9B](https://huggingface.co/litert-community/Ornith-1.0-9B). The old id `deepreinforce-ai/Ornith-1.0-9B` redirects to it. Environments: for the export and quantize steps transformers 5.14.1, torch 2.12.1, ai-edge-quantizer 0.8.0 and litert-torch at `115a13607c73` with `decider2bv_work/deps/qwen35_hybrid_litert_torch.patch` applied (the same bytes as the `qwen35_work/` copy since 2026-10-17; before that the `qwen35_work/` copy was an older version, 14 lines shorter in `model_ext/qwen3_5/patch.py`); litert-lm 0.18.0 for the metadata step and for running the files.

```bash
git clone https://github.com/john-rocky/litert-torch qwen35_work/litert-torch-qwen35
git -C qwen35_work/litert-torch-qwen35 fetch --depth 1 origin 115a13607c730c81018bb9789138a3e5e5119e3d
git -C qwen35_work/litert-torch-qwen35 checkout --detach 115a13607c730c81018bb9789138a3e5e5119e3d
git -C qwen35_work/litert-torch-qwen35 apply "$(pwd)/decider2bv_work/deps/qwen35_hybrid_litert_torch.patch"
QWEN35_PY=/path/to/python-with-transformers-5.14.1-torch-2.12.1-ai-edge-quantizer-0.8.0 \
RUN_PY=/path/to/python-with-litert-lm-0.18.0 \
LITERT_LM=/path/to/litert-lm-0.18.0 \
CHECK_PY=/path/to/python-with-ai-edge-litert-and-litert-lm-builder \
  bash qwen35_work/convert_qwen35_9b.sh ornith-ai/Ornith-1.0-9B out_ornith10_9b 83dc1f5e24ef8527af019a6b3bf66ac0f1c2c999
```

| file | section | bytes | sha256 |
|---|---|---|---|
| `Ornith-1.0-9B_int8.litertlm` (9,201,127,808 B) | LlmMetadataProto | 729 | `edc4ab993144df3f8a5e9833c3db500ffe9ef8e2b57ae19db7ad8a9e03c6fdc4` |
|  | ExecutorMetadataProto | 4,293 | `8a366d569ef88de755e22c623aa25e905dcfed55e2b2652f29feb437efea3d30` |
|  | HF_Tokenizer_Zlib | 3,913,268 | `4b9bc2f069b49a2ec3019a51bc625b6b568d33a124a599f7f29de009227905a1` |
|  | TFLiteModel | 9,197,162,880 | `80237a807bf28b9cc24b4132bfc7189fe56af71ab84b5e2015a322924caec398` |
| `Ornith-1.0-9B_mixed_int4.litertlm` (6,060,652,448 B) | LlmMetadataProto | 729 | `edc4ab993144df3f8a5e9833c3db500ffe9ef8e2b57ae19db7ad8a9e03c6fdc4` |
|  | ExecutorMetadataProto | 4,293 | `8a366d569ef88de755e22c623aa25e905dcfed55e2b2652f29feb437efea3d30` |
|  | HF_Tokenizer_Zlib | 3,913,268 | `4b9bc2f069b49a2ec3019a51bc625b6b568d33a124a599f7f29de009227905a1` |
|  | TFLiteModel | 6,056,687,520 | `60b4047d98c6d521518fbd86941ac87e263d1109e07bcfa6b5582286543f9cdf` |

Run as published on 2026-10-16 (one command, the environment of its header, 18 minutes on a 128 GB Mac, peak 48.3 GB), this script rebuilt every one of these sections byte for byte from the pinned checkpoint; so had the conversion run of 2026-10-15 behind the shipped files. The file sha256 differs from run to run (the packer writes a new uuid and creation timestamp), so compare sections.

- **The 9B recipe, unchanged.** The same script converts Qwen/Qwen3.5-9B; for Ornith only the HF id and the revision change. Ornith's text_config equals Qwen3.5-9B's in every shared key but `use_cache` (false; the export sets it) and its four added keys repeat values Qwen3.5-9B sets elsewhere; it ships no MTP weights, and its generation_config names two eos ids (248044 and 248046), which the export writes as the two stop tokens.
- **The lightweight export path.** `convert_qwen35_hybrid_lw.py` is `convert_qwen35_hybrid.py` plus `experimental_lightweight_conversion` with `RESOURCE_CONSTANT_NUMEL_THRESHOLD=100000000`; no op has all-constant inputs and every FULLY_CONNECTED weight is a constant. The float export peaked at 49.4 GB and each quantize step at 36.5 GB on a 128 GB Mac. Run one export at a time.
- **int8 = `wi8fc` (channelwise int8 weights, float activations); Mixed INT4 = blockwise-32 min-max int4 FCs + int8 channelwise lm_head and embedding**; in both files the 48 GatedDeltaNet decay/gate projections (`in_proj_a`, `in_proj_b`) stay float, and every int4 scale is normal.
- **fp32 GPU activations, declared.** `set_activation_type.py --type fp32` writes `prefer_activation_type: fp32` into the TFLite section; without it the int8 weights answered 0 of 8 check questions on the Mac GPU, every reply a run of `!`.
- **Thinking off in the template.** `make_new_pbtext.py --template-only` replaces only the prompt template with `qwen35_text_current.jinja`, the dual-form ChatML template of the earlier Qwen3.5 text files: every assistant turn and the generation prompt carry an empty think block. Its renders equal transformers' render of Ornith's template with `enable_thinking=False` in six conversation shapes; tool lists are not rendered. The thinking-switch template of the Qwen3.5-4B bundle was not used: it writes past assistant turns without the think block that Ornith's template always writes.
- **The tokenizer is the repo's file; no start token.** The bundle's tokenizer section inflates to Ornith's tokenizer.json byte for byte, and the engine's prefill count equals transformers' token count of the vendor render.
- **KV budget 4096, six prefill signatures (1024, 256, 64, 16, 4, 1), 64 state buffers** (8 + 8 KV caches, 48 GatedDeltaNet states). The TFLite section is 9,197,162,880 bytes (int8) and 6,056,687,520 bytes (Mixed INT4), both over 2 GiB; not run on iOS.
- GSM8K (questions 0–99, 0-shot step by step, greedy, at most 2048 output tokens, one process per question, thinking off): bf16 95, int8 97, Mixed INT4 95 on the Mac GPU.
- Measured (prefill 1024 / decode 256 / context 2048, litert-lm 0.18.0, Mac M4 Max): GPU (WebGPU) int8 581.4 / 39.99 tok/s, Mixed INT4 585.4 / 44.93; CPU int8 227.4 / 15.17, Mixed INT4 68.4 / 15.77; every signature fully delegated on the Mac GPU. On litert-lm 0.17.1 both files answered a one-word question on the Mac CPU (one `run` each). Phones: the Mixed INT4 file's GPU engine does not build on a 12 GB-class Galaxy S26 (on a device-lab Galaxy S26 the build ran out of memory in four sessions, the shipped file and a three-signature variant alike); the int8 file is not started on that phone's GPU and the CPU path needs more memory than the phone has; no Qwen3.5 file on the Hub built a GPU engine on the Pixel 8a.

### 2026-10-16 — Qwen3.5-9B: the 9B hybrid recipe, the repo's tokenizer.json in the tokenizer section, the Qwen3.5-4B thinking switch in the metadata

`qwen35_work/convert_qwen35_9b.sh` converts [Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) at commit `c2022362` (the script's default revision for this id) — the post-trained 9B of Qwen3.5, a "Causal Language Model with Vision Encoder" whose text decoder these files hold — into the int8 and Mixed INT4 files of [litert-community/Qwen3.5-9B](https://huggingface.co/litert-community/Qwen3.5-9B); `qwen35_work/use_repo_tokenizer.sh` then replaces each file's tokenizer section with the repo's `tokenizer.json`, and `qwen35_work/use_switch_template.sh` gives each file the metadata of the litert-community/Qwen3.5-4B files: the template `qwen35_text_T1.jinja` with a thinking switch that is off by default, a `thought` channel, `supports_thinking` true. Environments: for the export and quantize steps transformers 5.14.1, torch 2.12.1, ai-edge-quantizer 0.8.0 and litert-torch at `115a13607c73` with `decider2bv_work/deps/qwen35_hybrid_litert_torch.patch` applied (the same bytes as the `qwen35_work/` copy since 2026-10-17; before that the `qwen35_work/` copy was an older version, 14 lines shorter in `model_ext/qwen3_5/patch.py`); litert-lm 0.18.0 for the metadata steps, the tokenizer swap and running the files.

```bash
git clone https://github.com/john-rocky/litert-torch qwen35_work/litert-torch-qwen35
git -C qwen35_work/litert-torch-qwen35 fetch --depth 1 origin 115a13607c730c81018bb9789138a3e5e5119e3d
git -C qwen35_work/litert-torch-qwen35 checkout --detach 115a13607c730c81018bb9789138a3e5e5119e3d
git -C qwen35_work/litert-torch-qwen35 apply "$(pwd)/decider2bv_work/deps/qwen35_hybrid_litert_torch.patch"
export QWEN35_PY=/path/to/python-with-transformers-5.14.1-torch-2.12.1-ai-edge-quantizer-0.8.0
export RUN_PY=/path/to/python-with-litert-lm-0.18.0
export LITERT_LM=/path/to/litert-lm-0.18.0
export CHECK_PY=/path/to/python-with-ai-edge-litert-litert-lm-builder-and-tokenizers
bash qwen35_work/convert_qwen35_9b.sh Qwen/Qwen3.5-9B out_qwen35_9b
SNAP=out_qwen35_9b/hf_cache/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a
for f in Qwen3.5-9B_int8 Qwen3.5-9B_mixed_int4; do
  EXPECT_TOKENIZER_SHA256=5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42 \
    bash qwen35_work/use_repo_tokenizer.sh out_qwen35_9b/$f.litertlm out_qwen35_9b/$f.swap.litertlm "$SNAP/tokenizer.json"
  bash qwen35_work/use_switch_template.sh out_qwen35_9b/$f.swap.litertlm out_qwen35_9b/$f.switch.litertlm
  mv out_qwen35_9b/$f.switch.litertlm out_qwen35_9b/$f.litertlm && rm out_qwen35_9b/$f.swap.litertlm
done
```

| file | section | bytes | sha256 |
|---|---|---|---|
| `Qwen3.5-9B_int8.litertlm` (9,200,832,896 B) | LlmMetadataProto | 4,314 | `7fc50e38ed0bcdebf42fde57a216193962c14271ddfbaac466c68d4cd3601691` |
|  | ExecutorMetadataProto | 4,293 | `8a366d569ef88de755e22c623aa25e905dcfed55e2b2652f29feb437efea3d30` |
|  | HF_Tokenizer_Zlib | 3,614,174 | `f7099e031c8ef292b5b5d49c36b75dfb39c81f215ccdaf3fd87d5071ab6fe684` |
|  | TFLiteModel | 9,197,162,880 | `f4c77e3760190eb690f62e9dfb31ecd223c3b25e530e972e8b6abf4ab38fabd4` |
| `Qwen3.5-9B_mixed_int4.litertlm` (6,060,357,536 B) | LlmMetadataProto | 4,314 | `7fc50e38ed0bcdebf42fde57a216193962c14271ddfbaac466c68d4cd3601691` |
|  | ExecutorMetadataProto | 4,293 | `8a366d569ef88de755e22c623aa25e905dcfed55e2b2652f29feb437efea3d30` |
|  | HF_Tokenizer_Zlib | 3,614,174 | `f7099e031c8ef292b5b5d49c36b75dfb39c81f215ccdaf3fd87d5071ab6fe684` |
|  | TFLiteModel | 6,056,687,520 | `13312a1045cfc93e6ab05c2c9338ab68bfd2b5da0484028e77d64207d16ce7b0` |

How these sections were reproduced: the export, quantize and metadata steps, run again from the pinned checkpoint on 2026-10-14 (the same steps as `convert_qwen35_9b.sh`), rebuilt the four sections of the files before the tokenizer swap byte for byte; the swap applied to those files gives their four sections after it; `use_switch_template.sh` as printed, run on those swapped files, gives the four sections above, byte for byte (it prints each command it runs as `### CMD:`). `use_repo_tokenizer.sh` as printed, run on a file that already carries the repo's tokenizer, returns its four sections unchanged. `convert_qwen35_9b.sh` itself ran as published on 2026-10-16 for Ornith-1.0-9B (the same script bytes on that repo's id and revision, one command, 18 minutes on a 128 GB Mac): all four sections of both of that repo's files byte for byte; the three commands above were not run end to end on this repo's id in the form printed here. The file sha256 differs from run to run (the packer writes a new uuid and creation timestamp), so compare sections.

- **The tokenizer section is the repo's `tokenizer.json`.** The export writes the tokenizer that transformers builds from the Qwen2Tokenizer class: its pre-tokenizer regex has no `\p{M}`, and it adds the 7 audio / TTS tokens of `tokenizer_config.json`. The repo's file has the `\p{M}` regex; the two give different ids for text with combining marks (Hindi, Thai, Tamil, Arabic with harakat), while English, CJK, code and emoji tokenize the same. After the swap, litert-lm 0.18.0 `Engine.tokenize` matches `tokenizers` on the repo file in 346 of 346 test strings, and every evaluation prompt below has the same ids as before it. The swap tool is `tools/tokenizer_parity/swap_tokenizer_section.py`.
- **The lightweight export path, as for every 9B of this family.** `convert_qwen35_hybrid_lw.py` is `convert_qwen35_hybrid.py` plus `experimental_lightweight_conversion` with `RESOURCE_CONSTANT_NUMEL_THRESHOLD=100000000`; no op has all-constant inputs and every FULLY_CONNECTED weight is a constant. The float export peaked at 49.4 GB and each quantize step at 35.4 GB on a 128 GB Mac. Run one export at a time.
- **int8 = `wi8fc` (channelwise int8 weights, float activations); Mixed INT4 = blockwise-32 min-max int4 FCs + int8 channelwise lm_head and embedding**; in both files the 48 GatedDeltaNet decay/gate projections (`in_proj_a`, `in_proj_b`) stay float, and every int4 scale is normal.
- **fp32 GPU activations, declared.** Without the declaration the int8 weights answered 0 of 8 check questions on the Mac GPU, every reply a run of `!`.
- **A thinking switch, off by default.** `make_new_pbtext.py` without `--template-only` writes the four LlmMetadata fields of the Qwen3.5-4B files: the template `qwen35_text_T1.jinja` (LiteRT-LM's `models/qwen3_5/chat_template.jinja` at 87ab9334 with two edits), the `thought` channel (`<think>\n` / `\n</think>`), `supports_thinking` true and `supports_function_calling` false; stops and everything else stay. With nothing set the generation prompt carries an empty think block and past assistant turns render without one, as Qwen's template writes them; `enable_thinking=true` (`ThinkingConfig`, or `--thinking true` on the CLI) pre-fills `<think>\n` and the reasoning comes back on the channel. The template renders 6 of 7 checked conversation shapes byte for byte as transformers renders Qwen's template. With thinking on and a 2,048-token cap the thought does not always close; with `thinking_token_budget=300` it closed in every run on both files, and a three-turn conversation passed on the CPU and the GPU.
- **Tools.** The template renders tool definitions; `llm_model_type` stays `generic_model`, which has no parser for Qwen3.5's XML tool call, and `supports_function_calling` is false.
- **Stops and start token.** The repo has no `generation_config.json`; the export declares config.json's eos (248044, `<|endoftext|>`) and the rail adds 248046 (`<|im_end|>`); no start token, and the engine's prefill count equals transformers' token count of the vendor render.
- **KV budget 4096, six prefill signatures (1024, 256, 64, 16, 4, 1), 64 state buffers** (8 + 8 KV caches, 48 GatedDeltaNet states); the TFLite section is over 2 GiB in both files; not run on iOS.
- GSM8K (questions 0–99, 0-shot step by step, greedy, at most 2048 output tokens, one process per question, thinking off): bf16 97, int8 98, Mixed INT4 95 on the Mac GPU.
- Measured (prefill 1024 / decode 256 / context 2048, litert-lm 0.18.0, Mac M4 Max): GPU (WebGPU) int8 593.1 / 38.37 tok/s, Mixed INT4 603.6 / 46.07; CPU int8 228.1 / 15.06, Mixed INT4 66.25 / 14.25; every signature fully delegated on the Mac GPU. On litert-lm 0.17.1 both files answered a one-word question on the Mac CPU. Phones: the Mixed INT4 file's GPU engine does not build on a 12 GB-class Galaxy S26 (on a device-lab Galaxy S26 the build ran out of memory); the int8 file is not started on that phone's GPU and the CPU path needs more memory than the phone has; no Qwen3.5 file on the Hub built a GPU engine on the Pixel 8a.

### 2026-10-16 — Ornith-1.5-9B: the same 9B hybrid recipe, plus one step that puts the model repo's tokenizer.json in the bundle

`qwen35_work/convert_qwen35_9b.sh` converts [ornith-ai/Ornith-1.5-9B](https://huggingface.co/ornith-ai/Ornith-1.5-9B) at commit `489cb979` — Ornith's post-training of Qwen3.5-9B that extends Ornith-1.0, "the most lightweight member of the Ornith-1.5 family" — into the two files of [litert-community/Ornith-1.5-9B](https://huggingface.co/litert-community/Ornith-1.5-9B). One step runs after the script: each file's tokenizer section is replaced by the model repo's `tokenizer.json`, because the export bundles the tokenizer as transformers writes it for the Qwen2Tokenizer class, which is not this repo's file. Environments (the same as for Qwen3.5-9B and Ornith-1.0-9B): transformers 5.14.1, torch 2.12.1, ai-edge-quantizer 0.8.0 and litert-torch at `115a13607c73` with `decider2bv_work/deps/qwen35_hybrid_litert_torch.patch` applied; litert-lm 0.18.0 (CLI and builder) for the metadata step, the swap and running the files.

```bash
git clone https://github.com/john-rocky/litert-torch qwen35_work/litert-torch-qwen35
git -C qwen35_work/litert-torch-qwen35 fetch --depth 1 origin 115a13607c730c81018bb9789138a3e5e5119e3d
git -C qwen35_work/litert-torch-qwen35 checkout --detach 115a13607c730c81018bb9789138a3e5e5119e3d
git -C qwen35_work/litert-torch-qwen35 apply "$(pwd)/decider2bv_work/deps/qwen35_hybrid_litert_torch.patch"
QWEN35_PY=/path/to/python-with-transformers-5.14.1-torch-2.12.1-ai-edge-quantizer-0.8.0 \
RUN_PY=/path/to/python-with-litert-lm-0.18.0 \
LITERT_LM=/path/to/litert-lm-0.18.0 \
CHECK_PY=/path/to/python-with-ai-edge-litert-and-litert-lm-builder \
  bash qwen35_work/convert_qwen35_9b.sh ornith-ai/Ornith-1.5-9B out_ornith15_9b 489cb97981b8654bcfcf30ce1f94ed1b62e07b53
TOK=out_ornith15_9b/hf_cache/models--ornith-ai--Ornith-1.5-9B/snapshots/489cb97981b8654bcfcf30ce1f94ed1b62e07b53/tokenizer.json
shasum -a 256 "$TOK"   # 5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42
mkdir -p ornith15_9b
for f in Ornith-1.5-9B_int8 Ornith-1.5-9B_mixed_int4; do
  LITERT_LM_CLI=/path/to/litert-lm-0.18.0 /path/to/python-with-litert-lm-0.18.0 \
    tools/tokenizer_parity/swap_tokenizer_section.py out_ornith15_9b/$f.litertlm "$TOK" ornith15_9b/$f.litertlm
done
```

| file | section | bytes | sha256 |
|---|---|---|---|
| `Ornith-1.5-9B_int8.litertlm` (9,200,832,896 B) | LlmMetadataProto | 720 | `998962b1831fe4da10f92b1d7317e9cb003f6bcbe158e1e29c554dcae4d00c09` |
|  | ExecutorMetadataProto | 4,293 | `8a366d569ef88de755e22c623aa25e905dcfed55e2b2652f29feb437efea3d30` |
|  | HF_Tokenizer_Zlib | 3,614,174 | `f7099e031c8ef292b5b5d49c36b75dfb39c81f215ccdaf3fd87d5071ab6fe684` |
|  | TFLiteModel | 9,197,162,880 | `4d6d2197af22bca87173926ee0826fee7437ce178cf3c731713ae5406c6bd476` |
| `Ornith-1.5-9B_mixed_int4.litertlm` (6,060,357,536 B) | LlmMetadataProto | 720 | `998962b1831fe4da10f92b1d7317e9cb003f6bcbe158e1e29c554dcae4d00c09` |
|  | ExecutorMetadataProto | 4,293 | `8a366d569ef88de755e22c623aa25e905dcfed55e2b2652f29feb437efea3d30` |
|  | HF_Tokenizer_Zlib | 3,614,174 | `f7099e031c8ef292b5b5d49c36b75dfb39c81f215ccdaf3fd87d5071ab6fe684` |
|  | TFLiteModel | 6,056,687,520 | `dda8d148266fc1a4ab2ed3e0439e7cb555dd9c7a3c399eba0a44103596eeb1b8` |

The recipe and the swap rebuilt every one of these sections byte for byte when they were run again from the pinned checkpoint on 2026-10-15; the script as published (the same bytes) rebuilt the Ornith-1.0-9B files from their pinned checkpoint on 2026-10-16 (one command, the environment of its header, 18 minutes on a 128 GB Mac, all four sections byte for byte). The file sha256 differs from run to run (the packer writes a new uuid and creation timestamp), so compare sections.

- **The 9B recipe, unchanged.** The script is byte-identical to the one that converts Qwen/Qwen3.5-9B and Ornith-1.0-9B; only the HF id and the revision change. Ornith-1.5 keeps every Qwen3.5-9B text_config value, and its checkpoint carries the vision tower and an MTP head that the text export does not read.
- **The tokenizer swap.** `swap_tokenizer_section.py` unpacks the bundle with the litert-lm CLI, replaces the tokenizer file, packs it again and asserts that the other three sections are byte-identical; the new section inflates to the repo's `tokenizer.json` byte for byte. The two tokenizers give the same ids on English, code, CJK and every check and eval prompt; they differ on Hindi, Thai, Tamil and Arabic with diacritics, where the bundle now matches the vendor's file. litert-lm 0.18.0 and 0.17.1 tokenize as `tokenizers` does with the repo file, 174 of 174 probes.
- **One stop token.** The repo has no `generation_config.json`, so the export declares `config.json`'s eos, id 248046 (`<|im_end|>`), as the only stop token; the Qwen3.5-9B and Ornith-1.0-9B bundles of the same recipe carry 248044 and 248046.
- **The lightweight export path.** `convert_qwen35_hybrid_lw.py` is `convert_qwen35_hybrid.py` plus `experimental_lightweight_conversion` with `RESOURCE_CONSTANT_NUMEL_THRESHOLD=100000000`; no op has all-constant inputs and every FULLY_CONNECTED weight is a constant. The float export peaked at 49.4 GB and each quantize step at 35–37 GB on a 128 GB Mac. Run one export at a time.
- **int8 = `wi8fc` (channelwise int8 weights, float activations); Mixed INT4 = blockwise-32 min-max int4 FCs + int8 channelwise lm_head and embedding**; in both files the 48 GatedDeltaNet decay/gate projections (`in_proj_a`, `in_proj_b`) stay float, and every int4 scale is normal.
- **fp32 GPU activations, declared.** `set_activation_type.py --type fp32` writes `prefer_activation_type: fp32` into the TFLite section; without it the int8 weights answered 0 of 8 check questions on the Mac GPU, every reply a run of `!`.
- **Thinking off in the template.** `make_new_pbtext.py --template-only` replaces only the prompt template with `qwen35_text_current.jinja` (the same 701-byte file as in the Qwen3.5-9B and Ornith-1.0-9B files): every assistant turn and the generation prompt carry an empty think block. Its renders equal transformers' render of Ornith's `chat_template.jinja` with `enable_thinking=False` in six conversation shapes; tool lists are not rendered. The repo also carries a different template inside `tokenizer_config.json`; transformers loads `chat_template.jinja`.
- **No start token; KV budget 4096, six prefill signatures (1024, 256, 64, 16, 4, 1), 64 state buffers** (8 + 8 KV caches, 48 GatedDeltaNet states). The TFLite section is 9,197,162,880 bytes (int8) and 6,056,687,520 bytes (Mixed INT4), both over 2 GiB; not run on iOS.
- GSM8K (questions 0–99, 0-shot step by step, greedy, at most 2048 output tokens, one process per question, thinking off): bf16 94, int8 95, Mixed INT4 92 on the Mac GPU. On the Mac CPU the int8 file answered 15 of 17 prompt lengths of a one-word check, with a polite refusal at 23 and 24 tokens; the GPU and the Mixed INT4 file answered all 17.
- Measured (prefill 1024 / decode 256 / context 2048, litert-lm 0.18.0, Mac M4 Max): GPU (WebGPU) int8 609.0 / 39.57 tok/s, Mixed INT4 615.5 / 45.58; CPU int8 234.1 / 14.98, Mixed INT4 68.1 / 14.78; every signature fully delegated on the Mac GPU. On litert-lm 0.17.1 both files answered a one-word question on the Mac CPU. Phones: the Mixed INT4 file's GPU engine does not build on a 12 GB-class Galaxy S26 (on a device-lab Galaxy S26 the build ran out of memory); the int8 file is not started on that phone's GPU and the CPU path needs more memory than the phone has; no Qwen3.5 file on the Hub built a GPU engine on the Pixel 8a.

### 2026-10-22 — Qwen3.8-27B: the 9B hybrid recipe at 27B, exported from an F32 copy of the checkpoint and quantized from the float TFLite without a float .litertlm

`qwen35_work/convert_qwen38_27b.sh` converts [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) at commit `1d4bf0f2` (the script's default revision for this id) — Qwen3.8's 27B, "a native vision-language model" built "on the architectural foundation of Qwen3.5", whose text decoder this file holds — into the Mixed INT4 file of [litert-community/Qwen3.8-27B](https://huggingface.co/litert-community/Qwen3.8-27B) in one command: the pinned checkpoint, an F32 copy of it, the float export, the int4 build, the repo's `tokenizer.json` in the tokenizer section, the thinking-off metadata, and the checks. Environments: for the copy, export and int4 steps transformers 5.14.1, torch 2.12.1, ai-edge-quantizer 0.8.0 and litert-torch at `115a13607c73` with `qwen35_work/qwen35_hybrid_litert_torch.patch` applied; litert-lm 0.18.1 for packing, the tokenizer swap and running the file; ai-edge-litert 2.2.0 for the checks. Disk for the 27B: about 310 GB (snapshot 56 GB, F32 copy 111 GB, float TFLite 108 GB, the int4 file twice).

```bash
git clone https://github.com/john-rocky/litert-torch qwen35_work/litert-torch-qwen35
git -C qwen35_work/litert-torch-qwen35 fetch --depth 1 origin 115a13607c730c81018bb9789138a3e5e5119e3d
git -C qwen35_work/litert-torch-qwen35 checkout --detach 115a13607c730c81018bb9789138a3e5e5119e3d
git -C qwen35_work/litert-torch-qwen35 apply "$(pwd)/qwen35_work/qwen35_hybrid_litert_torch.patch"
export QWEN35_PY=/path/to/python-with-transformers-5.14.1-torch-2.12.1-ai-edge-quantizer-0.8.0
export RUN_PY=/path/to/python-with-litert-lm-0.18.1
export LITERT_LM=/path/to/litert-lm-0.18.1
export CHECK_PY=/path/to/python-with-ai-edge-litert-2.2.0-litert-lm-builder-and-tokenizers
bash qwen35_work/convert_qwen38_27b.sh Qwen/Qwen3.8-27B out_qwen38_27b
```

| file | section | bytes | sha256 |
|---|---|---|---|
| `Qwen3.8-27B_mixed_int4.litertlm` (15,379,451,056 B) | LlmMetadataProto | 4,781 | `6ceb8e6dadc40ff181cb4e99d624d5e7775f23fff640e38e59f93480574944f8` |
|  | ExecutorMetadataProto | 8,661 | `09b6c9037088ffd913ad046b0ad4865714c460cb25e531a449a79e2b79d57bdf` |
|  | HF_Tokenizer_Zlib | 3,614,237 | `bb631427decb5bab029f3be0d437684f6cdf5db4569e9e7ed798ecbefcf4678c` |
|  | TFLiteModel | 15,375,781,040 | `4ad0b70a3d565356fa96baafc2731f3e78f311012fe3b8cda795a3cd521aa8af` |

How this file was made, and what ran as printed: the four steps up to the tokenizer swap ran one at a time on 2026-10-20 (`upcast_fp32.py`) and 2026-10-21 (`convert_qwen38_hybrid_lw.py`, `make_int4_27b.py`, then the tokenizer swap that `use_repo_tokenizer.sh` runs) with the same arguments and environments, and gave a file with a thinking switch in its metadata; `use_noswitch_template.sh` as printed turned that file into the published one on 2026-10-23 (every section but the LlmMetadata byte-identical). The copies of the three Python scripts that ran then differ from the ones here only in comments and messages, in how they find the repository and the Python environments, in one key of the int4 build's JSON record, and in the export driver, which now checks its arguments, imports two modules earlier and no longer carries two imports, a directory variable and two helpers that were unused there; the tokenizer swap ran the same `tools/tokenizer_parity/swap_tokenizer_section.py` with the same arguments through a different wrapper. The 27B was not exported again for this entry. `convert_qwen38_27b.sh` as printed ran from the download to the checks for Qwen/Qwen3.5-2B at `15852e8c` (a smaller checkpoint of the same architecture), in three invocations with the same arguments: `STOP_AFTER=upcast` on 2026-10-22 (the download and the F32 copy), `STOP_AFTER=export` on 2026-10-23 (the export, 9 minutes on a 128 GiB Mac), then the rest (int4, tokenizer, thinking-off metadata and checks, 37 s); each exited 0. Its file: LlmMetadataProto 4,781 B `6ceb8e6dadc40ff181cb4e99d624d5e7775f23fff640e38e59f93480574944f8` (byte for byte the published file's: the same template, stop tokens, KV budget and model type), ExecutorMetadataProto 3,201 B `616a19e00ee4aa8286646823c406738c6670f3befe9d8d15fc7027522d097630`, HF_Tokenizer_Zlib 3,614,174 B `f7099e031c8ef292b5b5d49c36b75dfb39c81f215ccdaf3fd87d5071ab6fe684` (that repo's `tokenizer.json`), TFLiteModel 1,304,326,992 B `f18b3e0af0d088f5d615aa43c4e5ba4f5c9431b541683e70c40b96d85784049f`. That output carries Qwen3.8's chat template, so it tests the steps and is not a Qwen3.5-2B file to use. The file sha256 differs from run to run (the packer writes a new uuid and creation timestamp), so compare sections.

- **No float `.litertlm`.** In float32 the 27B is a 108 GB TFLite, and the 9B recipe packs that into a float `.litertlm` (another 108 GB) before it quantizes. `convert_qwen38_hybrid_lw.py` is the export of `convert_qwen35_hybrid_lw.py` with the packager's builder writing no bytes (the packager still selects the chat template and builds the LlmMetadata), and `make_int4_27b.py` quantizes the TFLite itself and packs the int4 file.
- **An F32 copy of the checkpoint.** transformers 5.14.1 loads a bf16 checkpoint for a float32 export by converting every tensor (`_materialize_copy` → `.to(dtype=float32)`), a float32 copy of every weight in RAM: the export from the bf16 checkpoint reached a 61 GB memory footprint while loading and was stopped. From an F32 checkpoint the same call returns the memory-mapped tensor. `upcast_fp32.py` writes that copy (BF16 → F32 by a 16-bit shift, exact; every tensor checked against the source bytes, every shard read back after writing).
- **A read-only ±inf check.** litert-torch's `_clamp_inf_values` (`backend/inline_consts.py`) clamps every float constant in place while it builds the MLIR module, which writes into every page of the memory-mapped weights (a 4-layer cut of the checkpoint: peak memory footprint 17.7 GB). The driver replaces it with a read-only check that calls the original only when a tensor holds ±inf: the same TFLite byte for byte on the 4-layer cut at a 1.6 GB peak; on the 27B, 1,573 float constants checked, none clamped. The 27B export: 66 minutes, peak memory footprint 11.8 GB, a 107.7 GB TFLite.
- **The int4 build writes into its output file.** `qt.quantize(serialize_to_path=...)` serializes the quantized model into the output file's memory map instead of a bytearray of the whole output (the same bytes): the 27B build took 18 minutes at a 46.2 GB peak memory footprint.
- **`scan_scales.py` reads the file through a memory map** instead of reading it whole and then copying each section (two copies of the 15 GB file in RAM); on the Qwen3.5-2B file above, the earlier version and this one print the same output, text and JSON, byte for byte.
- **Mixed INT4 with int4 blocks of 128**: blockwise-128 min-max int4 FCs, int8 channelwise lm_head and embedding, the 96 GatedDeltaNet decay/gate projections (`in_proj_a`, `in_proj_b`) float; block scales floored at 2^-14 (694 of 190,054,400 raised; every scale normal); fp32 GPU activations declared (with the GPU's default fp16 activations the first check question came back as a run of `!`). The block-32 file of the same export was 1.14 GB larger and slower on both backends (Mac GPU 21.31 vs 19.87 s, CPU 103.34 vs 92.74 s for a 1024 + 256 workload) with the same 8-question results (8/8 on both backends) and the same top-1 count on the decode walk (23 of 24).
- **Thinking off only.** The int4 step packs the switch template `qwen38_text_Q1.jinja` (Qwen3.8's chat format with `enable_thinking` defaulting to false, past assistant turns with their think block as with Qwen's `preserve_thinking` default, Qwen's tool-call layout and its reasoning-effort text when thinking is on; rendered with minijinja 2.14.0 and LiteRT-LM's template rewrites it equals transformers' render of Qwen's template at `enable_thinking=False` and `=True` in nine conversation shapes) with a `thought` channel and `supports_thinking` true. With that switch, thinking on broke three-turn conversations at litert-lm 0.18.1's default conversation settings: the third reply of "My name is Kenji." / "What is 17 times 23?" / "What is my name?" (thinking on, `thinking_token_budget=300`, Mac GPU) came back empty in 3 of 3 runs, with "I don't know your name." inside the thought; on the CPU (one run, 2026-10-23) the third reply was "I don't know your name.". The file with `filter_channel_content_from_kv_cache` off (one GPU run) and the bf16 model in transformers with Qwen's template at `enable_thinking=True` (one run) each answered "Your name is Kenji.". The published Qwen3.5-9B and 4B files (int8 and Mixed INT4) kept the name in the turn-3 reply of the same three turns at the default in 30 of 30 runs on the Mac CPU and GPU (2026-10-24; 29 answered it, one 4B int8 CPU run repeated the turns to the 2,048-token cap), so this check is per file. So `use_noswitch_template.sh` gives the published file `qwen38_text_noswitch.jinja` (the switch template's thinking-off branch: an empty think block at every generation prompt; its renders equal the switch template's thinking-off renders, 216 of 216), no channel and `supports_thinking` false. `supports_function_calling` stays false and `llm_model_type` `generic_model`, which in litert-lm 0.18.1 has no parser for Qwen's XML tool call.
- **The tokenizer section is the repo's `tokenizer.json`** (`use_repo_tokenizer.sh`): the export writes the tokenizer transformers builds from the tokenizer class, which tokenizes Hindi, Thai, Tamil and Arabic with harakat differently; after the swap litert-lm 0.18.1 `Engine.tokenize` matches `tokenizers` on the repo file in 292 of 292 test strings.
- **KV budget 4096, six prefill signatures (1024, 256, 64, 16, 4, 1), 128 state buffers** (16 + 16 KV caches, 96 GatedDeltaNet states); stops 248044 and 248046, both declared by the export; no start token.
- **GSM8K** (questions 0–99, the question with a step-by-step instruction and a final `#### <number>` line, thinking off, greedy, at most 2048 output tokens, read with `scripts/parity_gsm8k.py`): the int4 file 97 on the Mac GPU (litert-lm 0.18.1, a fresh process per question, on the file before the metadata re-pack, whose thinking-off prompts are the published file's), the bf16 model 96 (transformers 5.14.1 on MPS, Qwen's template at `enable_thinking=False`). The answers differ on two questions: 12, right only in the int4 reply, which reached the 2048-token cap, and 62, wrong in both.
- Measured (prefill 1024 / decode 256 / context 2048, litert-lm 0.18.1, Mac M4 Max): GPU (WebGPU) 175.5 / 18.24 tok/s; CPU with the XNNPACK weight cache 21.82 / 5.59 tok/s. Without the weight cache the CPU engine packs the int4 weights once per signature: on the block-32 file it passed 91 GB of memory during the engine build and was stopped. Not run on a phone: the file is larger than the phones' memory.
- **Count the WebGPU validation errors in every Mac GPU log.** A `litert-lm benchmark` process builds one Engine per run, the warmup included (four for `--runs 3`). Across 11 benchmark processes on this file and its variants (44 engines), WebGPU `Validation error` lines (`[Buffer (unlabeled)] usage (Storage(read-write)|Storage(read-only)) includes writable usage and another usage in the same synchronization scope`, then `[Invalid CommandBuffer] is invalid due to a previous error`) appeared in 2 engines, each the third engine of its process, while the process exited 0 and printed rates; 130 single-engine processes (the 100 GSM8K questions among them) printed none. The litert-lm 0.18.1 CLI prints only ERROR lines unless `--verbose` is given; these are ERROR lines, so they show without it. The GPU speeds in this entry come from benchmark processes without such a line.

## Qwen3.5-2B VISION build — the shipped text model was half of a VLM

The Qwen3.5 checkpoints are multimodal, and the text conversion above deliberately drops the vision tower. `qwen35vl_work/` keeps it: the checkpoint's own 24-layer, 1024-dim ViT (no DeepStack — `deepstack_visual_indexes` is empty upstream), wired to LiteRT-LM's `fast_vlm` contract at a static 512x512. Published alongside the text file: [litert-community/Qwen3.5-2B](https://huggingface.co/litert-community/Qwen3.5-2B) (`Qwen3.5-2B-VL_int8.litertlm`).

```bash
cd qwen35vl_work
# vision tower -> two tflites (encoder + adapter); ~10 min on an M4 Max
IMG=512 python convert_qwen35_vision.py out/qwen35vl-vision
python vision_quant_ab.py out/qwen35vl-vision          # fp32/fp16/int8 A/B on real images

# decoder for the fast_vlm contract (externalised embedder), SIX prefill signatures
PREFILL=1024,256,64,16,4,1 CACHE=4096 \
  PYTHONPATH=../qwen35_work/litert-torch-qwen35 \
  python export_qwen35vl_decoder.py Qwen/Qwen3.5-2B out/qwen35vl-decoder

# assemble + bind the hybrid states
DEC=out/qwen35vl-decoder VIS=out/qwen35vl-vision TOK=<tokenizer.json> \
  VENC=vision_encoder_fp16.tflite VADP=vision_adapter_int8.tflite DEC_ACT=fp32 \
  python build_qwen35vl_bundle.py
python ../scripts/add_executor_metadata.py out/.../Qwen3.5-2B-VL.litertlm <final>.litertlm
```

Uses the same patched `litert-torch` worktree as the text build (clone + checkout + apply from the section above).

**Six prefill signatures, not the text build's eleven.** Every exported signature is charged engine memory whether or not it is called. The text build's full 1–1024 ladder peaks around 5.3 GB on an iPhone 17 Pro and fits; attaching the vision tower pushes the same ladder past a 12 GB phone's jetsam ceiling — all three Metal legs were killed after initialising 11 of 12 signatures. Cutting to `1024,256,64,16,4,1` fixed it with no other change, and **also tripled CPU decode** (3.8 → 14.1 tok/s, first token 23.7 s → 5.1 s), because the runtime's XNNPACK weight repack is charged per signature too.

**The pad guard stops arming when the embedder is externalised.** `fast_vlm` moves the embedder into its own tflite, which removes token ids from the decoder graph — so the gated-delta prefill-pad guard, which derived its valid mask from `input_ids != 0`, silently did nothing, and padding would have poisoned the recurrent state during multi-chunk prefill. It now falls back to position monotonicity (`pos[1:] > pos[:-1]`, first slot valid), the same contract the upstream exportable module uses. A hermetic prompt-length sweep (40/40 clean) is what proves the guard armed; nothing else in the gate set catches it.

**Vision export traps, all encoded in `convert_qwen35_vision.py`.** The upstream tower is dynamic-resolution and its `grid_thw` preprocessing aborts `torch.export` outright, so it is re-authored for one static image: Conv3d patch-embed folded to Conv2d over the duplicated temporal frame, learned position embeddings bilinearly resampled to the static grid ahead of time, 2-D rope precomputed, explicit full attention instead of the variable-length split. Patches stay in raster order and the 2x2 merge happens in the adapter as four strided slices and a concat, so no `GATHER_ND` reaches the mobile GPU delegate. Two numerical guards matter: every activation keeps a leading batch dimension (Metal computes rank-2 elementwise against a rank-2 constant silently wrong), and LayerNorm is pre-scaled by a calibrated power of two, because from block 11 the residual stream carries values large enough to overflow fp16 accumulation.

**Ship shape: fp16 encoder, int8 adapter.** The exported tower matches the model's own vision path at correlation 0.9999999992 in fp32. Quantizing the encoder to int8 drops that to 0.97–0.98 on real photographs while the adapter survives int8 at 0.9998, so only the adapter is quantized. An int8-encoder build is 300 MB smaller and still answers correctly — that is the variant to use on Mali, where an fp16 vision encoder is known to crash devices of this class.

**Contract cost.** `fast_vlm` feeds sequential positions, so the checkpoint's M-RoPE collapses to plain RoPE. `hf_oracle_gen.py` measures it: against the full M-RoPE reference over nine image/prompt pairs, one generation is identical and the other eight are fluent same-content paraphrases that diverge deep in the answer. `suite_gate_vl.py` scores a bundle against both references.



### The 0.8B rides the same script — but not the same numbers

Published as `Qwen3.5-0.8B-VL_int8.litertlm` on [litert-community/Qwen3.5-0.8B](https://huggingface.co/litert-community/Qwen3.5-0.8B) (1.30 GB). Identical commands, with the model id and output dirs swapped:

```bash
cd qwen35vl_work
MODEL=Qwen/Qwen3.5-0.8B IMG=512 python convert_qwen35_vision.py out/qwen35vl08-vision
MODEL=Qwen/Qwen3.5-0.8B python vision_quant_ab.py out/qwen35vl08-vision

PREFILL=1024,256,64,16,4,1 CACHE=4096 \
  PYTHONPATH=../qwen35_work/litert-torch-qwen35 \
  python export_qwen35vl_decoder.py Qwen/Qwen3.5-0.8B out/qwen35vl08-decoder

DEC=out/qwen35vl08-decoder VIS=out/qwen35vl08-vision TOK=<tokenizer.json> \
  VENC=vision_encoder_fp16.tflite VADP=vision_adapter_int8.tflite DEC_ACT=fp32 \
  OUT_NAME=Qwen3.5-0.8B-VL_int8.litertlm python build_qwen35vl_bundle.py
python ../scripts/add_executor_metadata.py <out>.litertlm <final>.litertlm
```

**The tower is a different size, so two measured values must be re-derived rather than inherited.** The 0.8B ViT is **12 layers at 768 dim** against the 2B's 24 at 1024. The script calibrates the fp16-safe LayerNorm scales per run, so nothing needs editing — but the resulting table is genuinely different (16 from block 6 on, 512 at the final norm, versus 32 from block 11 for the 2B), and hand-copying a sibling's table would silently mis-scale the tower. The int8 encoder also behaves better here: correlation **0.9916-0.9951** on real photographs against the 2B's 0.974-0.984, so the int8-vision variant is on firmer ground at this size. fp32 parity is 0.9999999999.

**Build the vision tflites with litert-converter 0.4.0, not 0.3.1.** On 0.3.1 the adapter's 2x2 merge (`f[:, 0::2, 0::2, :]`) lowers to **six `GATHER_ND` ops** — a mobile-GPU hard wall — where 0.4.0 gives six `STRIDED_SLICE`, matching the shipped 2B adapter. This happens at every shape including the 2B's, so it is a toolchain variable rather than a geometry one, and **parity is bit-identical across both builds**: no accuracy check catches it. Gate the op histogram (the script prints `enc ops` / `adp ops`) and require `gather`, `flex` and `custom` all empty before anything downstream consumes the file.

**`add_executor_metadata.py` is not optional on a litert-torch 0.9.3 export.** 0.9.3 emits no `ExecutorMetadataProto`, and a fresh bundle dies at engine creation with `INTERNAL: ... No KV cache inputs found.` even though its decoder tflite carries all 48 `kv_cache` inputs. That is what the `_final` suffix means on these artifacts. When it happens, run a known-good sibling bundle on the same CLI first — that exonerates the runtime and the venv in one command.

**The six-signature ladder fit iPhone Metal on the first export**, with no jetsam and no `maxNumTokens` override — unlike the 2B, which needed the ladder cut after three killed Metal legs. Measured on an iPhone 17 Pro: vision turns 65.8 tok/s with grounding 2/2, text probe 45.7 tok/s on Metal at ~3.8 GB peak.

**Where the device and the desktop disagree, run the same prompt on the desktop before blaming the file.** On the composite 8-question probe this bundle answers 8/8 on Mac GPU *and* 8/8 on Mac CPU, while the phone scores 6/8 on Metal and 3/8 on CPU. None of the device misses reproduce on Mac, so they are device-side and not the conversion — the opposite of the 2B, whose arithmetic slip did reproduce on Mac and was therefore the model. The iPhone CPU result was re-run cold three days later after an app reinstall and came back byte-identical, so it is deterministic rather than thermal.

### OvisOCR2 — an OCR finetune rides the 0.8B vision rail unchanged

Published as [mlboydaisuke/OvisOCR2-LiteRT](https://huggingface.co/mlboydaisuke/OvisOCR2-LiteRT)
(1.30 GB). [ATH-MaaS/OvisOCR2](https://huggingface.co/ATH-MaaS/OvisOCR2) is a page-level
document-parsing post-train of Qwen3.5-0.8B, and its `config.json`, `chat_template.jinja`,
`tokenizer_config.json` and `preprocessor_config.json` are **byte-identical** to the base
checkpoint's — only the weights differ. So the 0.8B VL commands apply with the model id
swapped:

```bash
cd qwen35vl_work
MODEL=ATH-MaaS/OvisOCR2 IMG=512 python convert_qwen35_vision.py out/ovisocr2-vision
# OCR is the stricter consumer: A/B the vision quant on DOCUMENT fixtures, not photos
MODEL=ATH-MaaS/OvisOCR2 FIXTURES=<dir> FIXTURE_FILES=table_512.png,formula_512.png,arxiv_512.png \
  python vision_quant_ab.py out/ovisocr2-vision

PREFILL=1024,256,64,16,4,1 CACHE=4096 \
  PYTHONPATH=../qwen35_work/litert-torch-qwen35 \
  python export_qwen35vl_decoder.py ATH-MaaS/OvisOCR2 out/ovisocr2-decoder

DEC=out/ovisocr2-decoder VIS=out/ovisocr2-vision TOK=out/ovisocr2-decoder/tokenizer.json \
  VENC=vision_encoder_fp16.tflite VADP=vision_adapter_int8.tflite DEC_ACT=fp32 \
  OUT_DIR=out/ovisocr2-bundle OUT_NAME=OvisOCR2_int8.litertlm python build_qwen35vl_bundle.py
python ../scripts/add_executor_metadata.py out/ovisocr2-bundle/OvisOCR2_int8.litertlm OvisOCR2_int8_final.litertlm
```

What is checkpoint-specific and had to be re-measured rather than inherited:

- **The fp16-safe LayerNorm table.** Same shape as base 0.8B (16 from block 6, 512 at the
  final norm) but the absolute magnitudes moved (LN absmax 2390.9 → 2803.9). The script
  calibrates per run, so nothing is edited — but a hand-copied sibling table would have been
  silently wrong, again.
- **The vision-quant A/B on document fixtures.** A rendered table page, a formula page and a
  dense arXiv page at 512×512. Ship shape (fp16 encoder + int8 adapter) holds 0.9998+
  end-to-end correlation on all three; the int8/int8 Mali variant holds 0.994–0.998.
  Photo-fixture numbers must not be quoted for an OCR tower — the fixture set is part of
  the measurement.
- **Stop tokens without a `generation_config.json`.** The upstream repo ships none; the ids
  come from `config.json` (`eos_token_id` 248044) plus the tokenizer's `<|im_end|>` (248046),
  and the bundle builder verifies both against the actual tokenizer before baking them.

Two behaviors that belong to the checkpoint, not the conversion — measure the HF fp32
original before blaming the file:

- **Dense pages exceed 512×512.** On a 9-pt arXiv page the transcription starts correct,
  then paraphrases and finally repeats; upstream's own reference inference code ships a
  repeat-cleanup post-processor for this. On pages legible at 512² (the table and formula
  fixtures) the transcription is exact — all 20 table cells, symbol-for-symbol LaTeX — and
  the GPU backend's output is byte-identical to CPU's.
- **General chat is eroded by the OCR post-training.** On a generic 8-question sanity gate
  the model transcribes the prompt instead of answering; the fp32 original does the same.
  It is a document parser; prompt it as one.

## Falcon-H1 (attention ∥ Mamba2 in parallel, every layer) — first fully-hybrid family in LiteRT form

`falcon_h1_work/convert_falcon_h1.py` converts TII's Falcon-H1 Instruct models to `.litertlm`. Every layer (36 on the 0.5B, 24 on the 1.5B, 32 on the 3B, 66 on the 1.5B-Deep) runs a grouped-query attention branch and a Mamba2 selective-scan branch **in parallel** on the same normalized input and sums them — so every layer carries a KV cache *and* conv/SSM recurrent state. Published: [litert-community/Falcon-H1-0.5B-Instruct](https://huggingface.co/litert-community/Falcon-H1-0.5B-Instruct), [litert-community/Falcon-H1-1.5B-Instruct](https://huggingface.co/litert-community/Falcon-H1-1.5B-Instruct), [litert-community/Falcon-H1-3B-Instruct](https://huggingface.co/litert-community/Falcon-H1-3B-Instruct) and [litert-community/Falcon-H1-1.5B-Deep-Instruct](https://huggingface.co/litert-community/Falcon-H1-1.5B-Deep-Instruct). **Requires litert-lm ≥ 0.15 to run.**

```bash
cd falcon_h1_work
git clone https://github.com/john-rocky/litert-torch litert-torch-falcon
git -C litert-torch-falcon fetch origin 115a13607c730c81018bb9789138a3e5e5119e3d   # base unreachable from main since 2026-08
git -C litert-torch-falcon checkout 115a13607c730c81018bb9789138a3e5e5119e3d
git -C litert-torch-falcon apply "$(pwd)/falcon_h1_litert_torch.patch"
PYTHONPATH=litert-torch-falcon python convert_falcon_h1.py tiiuae/Falcon-H1-0.5B-Instruct out_falcon_05b
# GPU ship shape: declare fp32 activations in the bundle TOML (repack, no re-export)
python ../scripts/set_activation_type.py out_falcon_05b/Falcon-H1-0.5B-Instruct_int8.litertlm Falcon-H1-0.5B-Instruct_int8.litertlm --type fp32
```

The patch shares the granite/Qwen3.5 hybrid machinery — the folded rank ≤ 4 scan, state-continuation tracing, and the prefill-pad guard ride verbatim (the scan body is byte-identical to granite's, asserted at patch time). What Falcon-H1 needed on top:

- **A composite hybrid cache layer.** KV and conv/SSM state live at the SAME layer index, flattened as `k_i/v_i/mc_i/mr_i`. The class inherits the attention cache layer first (mask/timestamp semantics) and transformers' `LinearAttentionCacheLayerMixin` second — without the mixin, `Cache.has_previous_state(i)` refuses the layer as "an Attention layer". The runtime binds states by tensor name, so co-residency is just packaging.
- **Re-inject the exporter's kwargs.** `FalconH1Model.forward`'s layer loop drops `**kwargs`, so the timestamp indices the export cache needs never reach attention. The patched model pops them and re-injects them at each attention call.
- **`mamba_d_ssm` is not `expand × hidden`.** `FalconH1Config` defaults the SSM intermediate to its own width (and `mamba_d_head` resolves from `'auto'` against it) — shape inference must read the config, not assume the granite/bamba relation.
- **`mup_vector` is a non-persistent model-level buffer** — a prime candidate for transformers 5.x meta-load zeroing (and a silently zeroed buffer would still "pass" parity when reference and export share the model object). The patch zero-checks it after load; the check cannot live in `forward()` because fake-tensor tracing cannot concretize `float(buffer.max())`.

Ship verification (0.5B): float parity vs HF teacher-forced across 8 decode positions — max|logit diff| 7.6e-05, correlation 1.000000, top-1/top-5 identical; 8-question gate **int8 = float = GPU** (all 6/8 with near-verbatim identical text; the two misses are the 0.5B's own level — the float graph misses them the same way); hermetic prompt-length sweep 41/41 clean; Pixel 8a (OpenCL) delegates every subgraph with zero rejections and correct output; iPhone 17 Pro (Metal) decode 52.7 tok/s vs 36.2 CPU. Mac M4 Max: GPU 2650 prefill / 127.5 decode tok/s, CPU 473 / 59.0 (`-p 256 -d 256 --runs 3 --cache no`, litert-lm 0.16.0).

The 1.5B rides the same driver unchanged: parity max|logit diff| 1.6e-04 / correlation 1.000000 / top-1+top-5 identical; 8-question gate CPU 7/8 (the miss is answered identically by the HF bf16 reference) and GPU 6/8 (one borderline int8 greedy flip); hermetic sweep clean; on iPhone 17 Pro the composite quality probe answers 8/8 on BOTH backends with identical text — Metal decode 25.8 tok/s vs 15.5 CPU, TTFT 0.88 s vs 3.16 s. Mac M4 Max: GPU 1447 / 102.9, CPU 238 / 33.7.

The 3B rides the same driver unchanged too — one command, zero code changes (32 all-hybrid layers → 128 state buffers, full 1–1024 prefill ladder, int8 3.15 GiB). Its fp32 export exceeds the TFLite Python Interpreter's flatbuffer cap, so parity runs split: teacher-forced HF fp32 logits in the torch venv, a CompiledModel decode-walk over the same ids in the litert venv (`scripts/parity_logits_bigmodel.py`) — 48 positions, top-1/top-5 100%, Pearson 1.0000, KL ≈ 0. The 3B is the first Falcon-H1 size where int8 drops nothing: the 8-question gate is **8/8 on every lane** (GPU/CPU × litert-lm 0.15.0/0.16.0), hermetic sweep 40/40 CPU + 20/20 GPU, and on iPhone 17 Pro the composite probe answers 8/8 on both backends — Metal decode 14.0 tok/s / prefill 111.5 / TTFT 1.49 s / 3.03 GB peak (CPU: 7.8 / 48.7 / 3.14 s / 1.46 GB); the full 12-signature ladder loads on a 12 GB phone with no memory pressure (falcon's 65k vocab keeps the per-signature RAM tax light — a 248k-vocab 4B needed its ladder cut to 7 signatures). Mac M4 Max: GPU 979 / 65.3, CPU 121 / 20.9. One family-normal log line to expect on GPU: `TopK requires src tensor C dimension to be divisible by 4` (vocab 65537) — generation is unaffected on every size.

The 1.5B-Deep variant takes the same one command as well. It is the deepest shape in the family — 66 all-hybrid layers, so the bundle packages **264 state buffers** (132 conv/SSM + 132 KV, the most on this rail) and quantizes to 1.83 GB int8 from a 5.91 GiB float graph. Same gates, same answers: 48-position parity top-1/top-5 100% with Pearson 1.0000 and KL ≈ 0, 8-question gate 8/8 on all four lanes (GPU/CPU × 0.15.0/0.16.0), hermetic sweep 40/40 CPU + 20/20 GPU. Its bench shows the depth trade running both ways on an M4 Max: prefill **1134 tok/s — above the 3B's 979 —** and decode **51.7, below its 65.3**, because decode walks 66 sequential attention+scan blocks per token while prefill batches over the sequence. If you are timing a conversion of your own, read the stage breakdown and not the wall clock: of this one's 88 minutes, 69 were the HF download and 19 were the graph export — a *faster* export than the 3B's ~26 minutes despite twice the layers.

On an iPhone 17 Pro the Deep answers the composite probe **8/8 on GPU and 8/8 on CPU**, word for word identical apart from one function word: Metal decode 16.5 tok/s / prefill 140.3 / TTFT 1.19 s / 4.89 GB peak (CPU: 9.9 / 103.2 / 1.53 s / 1.74 GB). All 264 state buffers and the full ladder load with room to spare — depth costs nothing on the device side. Budget about a minute for GPU engine creation (8 s on CPU): every prefill signature's Metal kernels are built up front.

### 2026-09-01 — Falcon-H1-Tiny-R-0.6B: the family's first reasoning ship, and the size where two family assumptions break

[litert-community/Falcon-H1-Tiny-R-0.6B](https://huggingface.co/litert-community/Falcon-H1-Tiny-R-0.6B) — 44 all-hybrid layers (176 state buffers), self-emitting `<think>…</think>` reasoner, **requires litert-lm ≥ 0.16**. It does NOT ride `convert_falcon_h1.py` unchanged; `falcon_h1_work/convert_falcon_tinyr.py` carries four measures the Instruct sizes never needed, each one measured rather than assumed (the script's docstring has the details):

```bash
cd falcon_h1_work   # same pinned litert-torch base + falcon_h1_litert_torch.patch as above
hf download tiiuae/Falcon-H1-Tiny-R-0.6B --local-dir src/Falcon-H1-Tiny-R-0.6B
PYTHONPATH=litert-torch-falcon python convert_falcon_tinyr.py src/Falcon-H1-Tiny-R-0.6B out_tinyr
```

1. **The bundle stream must be byte-identical to HF, BOS included.** The upstream template renders `{{bos_token}}`, which the engine's template renderer leaves empty; the builder's start_token puts BOS in a different spot than HF's render. At 0.6B that one-token head difference flips greedy answers (bf16 A/B: 8/8 vs 7/8 — "capital of Japan" became *Hiroshima*). The script hardcodes the literal `<|begin_of_text|>` into the template and suppresses start_token.
2. **int8 stops at the FC boundary — the embedding table stays float.** The family rule "int8 on linears + embedding" fails here: an int8 embedding table (even per-row) gives runaway thinking and greedy fact flips; FC-only int8 restores bf16-class behavior (a float lm_head does NOT — the input embedding is the poison).
3. **The embedder is externalized into its own CPU-side section**, because the GPU delegate accepts an in-graph EMBEDDING_LOOKUP only with an int8 table (a float table crashes engine creation, an fp16 cast is refused as partial delegation) — quality and GPU delegation cannot coexist in one graph.
4. **The pad guard now derives its mask from position monotonicity** (patch updated in place): the externalized decoder graph carries no token ids, so the previous `input_ids != 0` mask silently disabled and CPU pad garbage corrupted the Mamba conv/cumsum state (`17+25 → 19.5`, float export identical — not a quantization bug).

The reasoning is packaged, not just tolerated: the `thought` channel (`<think>`/`</think>`) is declared post-hoc with `tools/add_thought_channel.py` (verified: reasoning arrives on the channel on Mac CPU/GPU and iPhone, and `--thinking-budget 16` cuts at exactly 16 tokens), and both upstream stop tokens `[11, 228]` ride in from `generation_config`. Give sessions a ≥ 2048-token output budget — truncated mid-thought a reasoner produces no final answer.

Ship verification: **GSM8K 100-question greedy at 2048 tokens, scored after `</think>`: the int8 bundle 76/100 vs the bf16 PyTorch model 71/100** on the identical stream and protocol (per-question flips run in both directions — quantization noise, not degradation). 8-question gate: Mac CPU 6/8 / GPU 7/8, iPhone 17 Pro CPU 6/8 / Metal 6/8, zero unclosed thinks; the recurring misses are two fact-recall knife-edges the bf16 model also flips under one-token perturbations. Mac M4 Max (`-p 256 -d 256 --runs 3 --cache no`, litert-lm 0.16.0): GPU 2216 prefill / 97.8 decode / TTFT 0.13 s, CPU 381 / 47.8 / 0.69 s. iPhone 17 Pro (cold, single runs, 148-token composite prompt): **CPU decodes faster than Metal at this size** — 29.4 vs 22.8 tok/s, at 0.82 vs 3.01 GB peak — the GPU-overhead side of the small-model law, opposite to every larger sibling.

### 2026-10-15 — Falcon-H1R-7B: the family's reasoning 7B on the same patch, a lightweight export, and int4 scales clamped before quantizing

`falcon_h1_work/convert_falcon_h1r_7b.sh` converts [tiiuae/Falcon-H1R-7B](https://huggingface.co/tiiuae/Falcon-H1R-7B) at commit `a6f74bf1` — TII's reasoning-specialized Falcon-H1 7B, 44 layers of attention ∥ Mamba2 — into the two files of [litert-community/Falcon-H1R-7B](https://huggingface.co/litert-community/Falcon-H1R-7B). Environments: litert-torch at the pinned base `115a136` + `falcon_h1_litert_torch.patch` (as above), torch 2.12.1, transformers 5.14.1, ai-edge-quantizer 0.8.0, ai-edge-litert 2.1.6, litert-converter 0.3.0 for the export and the quantization; litert-lm 0.18.0 for the metadata steps and for running the files.

```bash
EXPORT_PY=/path/to/python-with-transformers-5.14.1-and-ai-edge-quantizer-0.8.0 \
RUN_PY=/path/to/python-with-litert-lm-0.18.0 \
LITERT_LM=/path/to/litert-lm \
CHECK_PY=/path/to/python-with-ai-edge-litert \
  bash falcon_h1_work/convert_falcon_h1r_7b.sh out_falcon_h1r_7b
```

| file | section | bytes | sha256 |
|---|---|---|---|
| `Falcon-H1R-7B_int8.litertlm` (7,810,102,560 B) | LlmMetadataProto | 3,909 | `4357d902b8bba60c04772ea93df3f51b8e7d5b63778562bf124d9513f1cca65d` |
|  | ExecutorMetadataProto | 11,901 | `46ce186bbd0653fc8bd389b9e3ac08b7360f8d3b9f91a2efe65a172301bcbf09` |
|  | HF_Tokenizer_Zlib | 2,165,329 | `daf71f7283b2f4d6562aa7fa66f1db1e14af3c00658bb7ffbf857efe5f42090a` |
|  | TFLiteModel | 7,807,874,336 | `0e4138cdf09ee010db0ee1e3d930cac106cc6aab5249dc8d868899ca5009bc16` |
| `Falcon-H1R-7B_int4.litertlm` (4,689,037,856 B) | LlmMetadataProto | 3,909 | `4357d902b8bba60c04772ea93df3f51b8e7d5b63778562bf124d9513f1cca65d` |
|  | ExecutorMetadataProto | 11,901 | `46ce186bbd0653fc8bd389b9e3ac08b7360f8d3b9f91a2efe65a172301bcbf09` |
|  | HF_Tokenizer_Zlib | 2,165,329 | `daf71f7283b2f4d6562aa7fa66f1db1e14af3c00658bb7ffbf857efe5f42090a` |
|  | TFLiteModel | 4,686,809,632 | `9d0a95684e2c698a49689f6c2a23f092190f5d124c3d95972c6813e1c5fbb530` |

Run as published on 2026-10-16 (one command, the environment of its header, 74 minutes on a 128 GB Mac, peak 60.1 GB; the export 71 minutes, 55 of them the converter's constant folding, each quantize step 1.5 minutes), this script rebuilt every one of these sections byte for byte from the pinned checkpoint; so had the conversion run of 2026-10-14 behind the shipped files (22:23 → 01:01). The file sha256 differs from run to run (the packer writes a new uuid and creation timestamp), so compare sections.

- **Same patch, three changes for a 7B.** The float export takes six prefill signatures (1024, 256, 64, 16, 4, 1) over a 4096-position cache instead of the rail's ladder, and litert-torch's `--experimental_lightweight_conversion=True` with `RESOURCE_CONSTANT_NUMEL_THRESHOLD=100000000` (constants of 1e8 bytes or more stay lazy through the MLIR lowering): the default path of an 8B model reached a 98.8 GB peak on a 128 GiB Mac. The export peaks at about 61 GB and takes ~80 minutes, ~60 of them in litert-torch's runtime constant folding of 10 small constants (the folding pass copies every composite decomposition of the graph); each quantization ~30 GB. Run one export or quantization at a time.
- **A loud, random export failure.** One run failed in the MLIR verifier with `redefinition of symbol named 'jax_lowering_9dc46fef'`: litert-torch's JAX bridge names each lowering with 32 random bits; re-running draws new names.
- **int8 = `wi8fc`** (dynamic int8 on every FULLY_CONNECTED + int8 channelwise embedding; the Mamba conv and the scan stay float). **int4 = `wi4b32_wi8`** (int4 blockwise-32 min-max FCs + int8 lm_head and embedding) run through `quantize_clamped.py`: a plain build leaves 96 zero and 748 subnormal fp16 block scales in 7 Mamba-mixer tensors, and XNNPACK prepares a blockwise int4 FC only when every scale is normal, so the CPU path of such a file is lost; the wrapper raises every blockwise int4 scale below 2^-14 to 2^-14 before quantizing; `scan_scales.py` then finds every scale normal.
- **Reasoning packaging.** `falcon_h1r_dual.jinja` is the vendor's ChatML template with a content-parts macro (string or list-of-parts messages), the vendor's past-turn form for every history turn (the answer, without the thought), and TII's default system message verbatim; its renders equal transformers' vendor render in 24 cells (8 conversation shapes × 3 thinking states). The prompt ends with the bare assistant header and the model writes `<think>` itself; the bundle declares the `thought` channel (`<think>` / `</think>`), `supports_thinking`, stop ids 11 and 228, and no start token (the vendor stream has no BOS; the engine's prefill counts equal transformers'). A thinking budget cuts the thought.
- **fp32 GPU activations.** With the GPU decoder's default fp16 activations the int8 file wrote `<|pad|>` from the first token on all 8 check questions; `set_activation_type.py --type fp32` fixes it at about 1 % of the GPU benchmark time.
- **State and tokenizer.** 176 state buffers = 44 K + 44 V + 88 Mamba conv / recurrent states. The tokenizer section is the vendor's `tokenizer.json` with an identity `post_processor` written by the exporter (ids identical on 15 test sentences). One Engine per conversation: the running state is the shape of LiteRT-LM #3165.
- **Checks** (litert-lm 0.18.0, Mac M4 Max): GSM8K questions 0–99, greedy, at most 2,048 output tokens, scored after `</think>` (a reply that never closed its thought is scored on its whole text): int8 96 / 100, int4 98 / 100 on the GPU; the bf16 reference ran questions 0–20 only (21 / 21); the 8-question check 8/8 on CPU and GPU for both files. Prefill 1024 / decode 256 / context 2048: GPU int8 565.05 / 35.88 tok/s (8.95 s), int4 565.78 / 39.71 tok/s (8.26 s); CPU int8 135.64 / 11.59 tok/s (29.65 s), int4 60.33 / 12.35 tok/s (37.72 s).
- **Phones** (prebuilt `litert_lm_advanced_main` of 2026-09-18): the OpenCL delegate took all seven signatures of the int4 file on a Galaxy S26 and on a Pixel 8a, and neither built the GPU engine — the S26 rebooted during the warm-up (its boot log says `reboot`, no kernel panic) and the Pixel 8a ran short of memory; the shipped rows are the Mac rows (docs/android-gpu.md "7–9B models on a 12 GB phone").

### 2026-08-27 — Qwen3.5-4B Mixed INT4: the block size is the only knob that matters

Requested in [LiteRT-LM #1658](https://github.com/google-ai-edge/LiteRT-LM/issues/1658) for 8 GB Android phones: the int8 file is 4.10 GB and cannot fit; a Mixed INT4 build is 2.57 GB. Two commands on top of the recipe above (same pinned base + patch, same float export):

```bash
QWEN35_PREFILL_LADDER=1024,256,64,16,4,1 PYTHONPATH=litert-torch-qwen35 \
  python convert_qwen35_hybrid.py Qwen/Qwen3.5-4B out_qwen35_4b   # float export (reuse if you have it)
python make_int4_4b.py out_qwen35_4b/model.litertlm out_qwen35_4b b32   # -> Qwen3.5-4B_mixed_int4_b32.litertlm
```

`make_int4_4b.py` (this directory) applies `minicpm5_work/quantize_minicpm5.py` post-hoc — int4 **blockwise-32 min-max** on every FULLY_CONNECTED, overridden to int8 channelwise on the lm_head and embedding lookup — then the same template/stop-token/ExecutorMetadata/fp32act steps as the int8 ship. Two things worth copying from how this was chosen:

- **Pick the block size by measurement, not by family habit.** All four candidates (b32/b128 × min-max/OCTAV) were built from the *same* float export and scored on GSM8K n=100, greedy, 2048-token budget, identical harness: int8 control 97, **b32 min-max 93**, b32 OCTAV 92, b128 min-max 90, b128 OCTAV 90. OCTAV bought nothing here; blockwise-128 (the faster-decode habit from dense 4B ships) costs 3 more points on this hybrid. And on today's runtime the iPhone decode gap between b32 and b128 has vanished (11.4 vs 11.9 tok/s Metal) — the historical reason to prefer b128 no longer holds on this rail.
- **Budget ≥2048 tokens when you score a Qwen3.5 on GSM8K, even with thinking disabled.** At 512 the model's verbose step-by-step runs past the budget before the `#### N` line and the extractor grabs stray numbers — it reads as a quantization collapse and is actually truncation (measured: the same questions come back correct at 2048).

Ship verification (all on the shipped file): 8-question gate **8/8 on CPU and GPU on both litert-lm 0.15.0 and 0.16.0**; prompt-length robustness, fresh engine per length, 40/40 CPU + 20/20 GPU; iPhone 17 Pro answers the composite 8-question probe **8/8 on both Metal and CPU** (the int8 file's CPU path answers 6 of 8 on this probe) — Metal decode 11.4 tok/s / prefill 88.7 / TTFT 1.89 s / 5.74 GB peak, CPU 8.9 / 46.7 / 3.12 s / 1.68 GB. Mac M4 Max (`-p 256 -d 256 --runs 3 --cache no`, 0.16.0): GPU 669 / 68.5 tok/s, CPU 100 / 20.1. Honest limits: the GSM8K gap vs int8 is real (93 vs 97); Mac CPU prefill is ~2.4× slower than int8's (int4 unpack cost — decode is equal or faster everywhere); and we have not yet run it on an actual 8 GB Android phone — the 2.57 GB file + ~1.7 GB CPU working set is the sizing evidence, measured on iPhone.

### 2026-09-21 — Agents-A1-4B (InternScience): the Qwen3.5-4B rail on an agentic finetune, tools + thinking template

<a id="agents-a1-4b-reproduction"></a>

Same 4B hybrid patch and reduced ladder, applied to
[InternScience/Agents-A1-4B](https://huggingface.co/InternScience/Agents-A1-4B).
The checkpoint's config is byte-identical to Qwen/Qwen3.5-4B's. This builds the
**text decoder only (vision tower dropped)**, with two files from one float
parent: dynamic int8 (4,407,426,400 bytes) and mixed int4 block-32 min-max
(2,754,365,536 bytes), both declaring fp32 activations. Bundles:
[litert-community/Agents-A1-4B](https://huggingface.co/litert-community/Agents-A1-4B).

Set the interpreter/CLI variables using [agents_a1_work/README.md](agents_a1_work/README.md).
One-command build: `bash scripts/reproduce_llm.sh agents-a1-4b`. The explicit
steps, from the repository root, are:

```bash
git clone --no-checkout --depth 1 https://github.com/google-ai-edge/litert-torch "$LITERT_TORCH_DIR"
git -C "$LITERT_TORCH_DIR" fetch --depth 1 https://github.com/google-ai-edge/litert-torch 115a13607c730c81018bb9789138a3e5e5119e3d
git -C "$LITERT_TORCH_DIR" checkout --detach 115a13607c730c81018bb9789138a3e5e5119e3d
git -C "$LITERT_TORCH_DIR" apply "$(pwd)/qwen35_work/qwen35_hybrid_litert_torch.patch"
if git -C "$LITERT_TORCH_DIR" apply --check "$(pwd)/agents_a1_work/qwen35_export_compat.patch" 2>/dev/null; then git -C "$LITERT_TORCH_DIR" apply "$(pwd)/agents_a1_work/qwen35_export_compat.patch"; fi  # a no-op since 2026-10-17: qwen35_work's patch carries the branch; patch_state.json checks the seven files either way
"$PY" -B agents_a1_work/export_float.py --download --model "$AGENTS_A1_CHECKPOINT" --output "$AGENTS_A1_OUTPUT/float" --litert-torch-dir "$LITERT_TORCH_DIR"
"$PY" -B agents_a1_work/build_bundles.py --model "$AGENTS_A1_CHECKPOINT" --float-bundle "$AGENTS_A1_OUTPUT/float/model.litertlm" --output "$AGENTS_A1_OUTPUT"
# Float CPU parity first (pt and lt run serially in their respective environments):
"$PY" -B agents_a1_work/download_gsm8k.py
"$PACKAGER" unpack "$AGENTS_A1_OUTPUT/float/model.litertlm" --output-dir "$AGENTS_A1_OUTPUT/float/unpacked"
"$PY" -B agents_a1_work/parity.py pt --hf "$AGENTS_A1_CHECKPOINT"
"$LT_PY" -B agents_a1_work/parity.py lt --tflite "$AGENTS_A1_OUTPUT/float/unpacked/"*TFLiteModel*.tflite
"$PY" -B agents_a1_work/parity.py cmp
# Runtime cells: repeat per file/backend and both CLIs; complete loops are in agents_a1_work/README.md.
"$PY" -B agents_a1_work/gate8q.py --model "$AGENTS_A1_OUTPUT/Agents-A1-4B_int8.litertlm" --cli "$LITERT_LM" --runtime 0.17.1 --backend cpu --out "$AGENTS_A1_OUTPUT/results/gate8q_int8_cpu_0.17.1.json"
"$PY" -B agents_a1_work/banana_hermetic_sweep.py --model "$AGENTS_A1_OUTPUT/Agents-A1-4B_int8.litertlm" --cli "$LITERT_LM" --runtime 0.17.1 --backend cpu --out "$AGENTS_A1_OUTPUT/results/banana_int8_cpu.json"
```

The small `qwen35_export_compat.patch` adds the position-mask fallback missing
from this repository’s older Qwen3.5 patch; hashes verify that all seven patched files
match the measured export source.

The [directory instructions](agents_a1_work/README.md#gates-in-order) give the
complete eight-cell matrix, both length ranges, thinking/conversation/tools
probes and the ordered GSM8K A/B commands. Every runtime request uses a fresh
process, closed stdin, greedy sampling and seed 0; the multi-turn API probe
intentionally keeps one Conversation. Cache is off except during the serial
GSM8K pass, after cache-off generation gates on the same file. No GPU number
is inferred from conversion or delegation alone.

What changes from the older Qwen3.5 template: the pinned canonical Qwen3.5
form accepts **strings and content parts**; the vendor's default system prompt
is embedded **verbatim**, including `Current date: 2026-07-14`, and any explicit
system message replaces it. Assistant history renders position-independently.
Thinking is on by default with the `thought` channel; disabling it emits the
empty think scaffold. Native XML `<tool_call><function=...>` calls are rendered
into the prompt, but the tested runtime does **not** parse them into tool-call
events; the application parses the emitted block. Template checks match HF
byte-for-byte for the basic cases and preserve the prefix contract; tool JSON
spacing differences are retained and reported. Both stop ids, 248044 and
248046, are declared.

Measured on Apple M4 Max:

- Float CPU logits, 48 teacher-forced positions: top-1/top-5 **48/48**, minimum
  Pearson **0.999999999862**, mean KL **−1.64e−8 nats**, max absolute difference
  **0.000221253**. Acceptance: 48/48, Pearson ≥0.9999 at every position and mean
  KL ≤0.001, with finite logits. The large-file lt stage uses CompiledModel 2.2.0
  and needed approximately 73 GB peak RSS; run it alone.
- Both files: **8/8 in all eight CPU/GPU × 0.17.1/0.18.0.dev20260919 cells**,
  zero degenerate answers and all answers on topic; **40/40 CPU and 20/20 GPU**
  prompt-length sweeps per file on 0.17.1.
- GSM8K first 100, GPU 0.17.1, thinking off, context 4096 and no decode cap:
  **93/100 int8, 93/100 mixed int4** with the original extractor; **94/100 and
  94/100** after numeric normalization of the same saved answers. Context-limit
  counts: **1 int8, 2 mixed int4**.
- Thinking/channel/budget-64 probes pass on both runtimes; the three-turn
  conversation recalls Osaka and Ken with thinking off and on. Native XML tools
  pass **12/12** fixed requests, including three with thinking enabled.

Mac benchmark: `-p 256 -d 256 --runs 3 --cache no`, litert-lm 0.17.1, default warmup. All four rows were taken on a quiet machine (1-minute load under 3.0, at least 300 s of rest before a GPU cell); three of them are a second pass that agreed within 2 % with a first pass taken on a busier machine.

| File / backend | Prefill tok/s | Decode tok/s | TTFT s |
|---|---:|---:|---:|
| int8 / GPU | 672 | 61.2 | 0.40 |
| mixed int4 / GPU | 676 | 67.2 | 0.39 |
| int8 / CPU | 255 | 20.6 | 1.05 |
| mixed int4 / CPU | 99.1 | 20.2 | 2.63 |

Two traps to preserve in a reproduction: **fetch the full 115a136 hash before
checkout** (it is no longer reachable by a plain clone plus abbreviated
checkout), and **keep the original and corrected GSM8K columns separate**.
The old extractor's `rstrip(".0")` turns `20.00` into `2`; the correction keeps
the same extraction precedence and normalizes numerically, without rerunning
generation.

Phone check (Galaxy S26, SM8850, Android 16, LiteRT-LM Android CLI build of 2026-09-18; the same 8 questions with thinking off, one fresh session per question, the prompt rendered from the bundled template): mixed int4 **8/8 on CPU** (peak 4.94 GB; 47.1 prefill / 11.9 decode tok/s, TTFT 5.52 s at 256/256, thermal status 0, frequencies uncapped) and **8/8 on the OpenCL GPU** with every subgraph delegated (peak 4.16 GB; speed not measured). With thinking on, the final answer arrives after a 350–400-token thought. The int8 file is a desktop-class file: on this phone its CPU answers are correct but 2 of 8 run on into repetitive self-talk, the thinking-on probe gives no final answer, and starting its GPU engine took the phone off USB — the published card tells Android users to take the mixed int4 file. The Android CLI has no thinking switch and no sampler flags, so a phone gate that leaves thinking on needs a token cap in the thousands and a watchdog in minutes, not seconds.

<!-- evidence: agents_a1_work/measured_results.json; raw evidence fingerprints and exact observed values are recorded there. -->

### Falcon-H1 finetune intake — one command, and a gate refusal worth reading

`scripts/convert.py` routes `model_type: falcon_h1` to this recipe the same way as granite
(HYBRID_RECIPE; no start_token drop — the shipped Falcon-H1 bundles keep it and gate clean at
every size). Measured 2026-08-25 on
[megabytes/Falcon-H1-0.5B-Instruct-heretic](https://huggingface.co/megabytes/Falcon-H1-0.5B-Instruct-heretic)
(Heretic v1.1.0 abliteration of the 0.5B, the most-downloaded real Falcon-H1 derivative):

```bash
python scripts/convert.py megabytes/Falcon-H1-0.5B-Instruct-heretic
```

One command from a clean shell: recipe export 796 s → 650 MB int8, 144 state buffers
(72 conv/SSM + 72 KV) → generic 8-question gate on CPU → **5/8, verdict "do NOT publish",
exit 1**. That verdict is the gate working, not the conversion failing — the misses decompose,
measured:

- The graph is faithful: on the divergence probes, the float export ≡ HF fp32 greedy once both
  sides use the bundle's BOS convention (the metadata start_token `<|begin_of_text|>`,
  prepended on the HF side too): `17+25 → "42"`, `thank-you-in-French → "Thank you."` —
  token-identical.
- Two misses are the abliterated model's own answers (`8×7 → "28."` verbatim on HF fp32; the
  French miss is model-level under either BOS convention — abliteration's measured KL 0.0954
  vs base costs real capability at 0.5B scale).
- One miss is an int8 greedy flip (`17+25`: int8 `"22."` vs float/HF `"42"`) — the base 0.5B's
  int8 did not flip this, so the ablated weights quantize worse.

Family fact for derivative templates: the Heretic tool re-serializes `chat_template.jinja`
with CRLF line endings (1167 bytes vs the base's 1151) — content-identical after
universal-newline normalization, which is exactly the form the exporter embeds (bundle
template = 1151 bytes = base's). Compare templates modulo newline convention, or you will
flag re-serialized derivatives as template forks. `tokenizer.json` is byte-equal to the base.

**2026-10-13 — int8 re-quantized.** The shipped int8 file (sha256 `0001573f…`) ran on after the right answer on the CPU backend: on a Mac M4 Max CPU (litert-lm 0.18.0, greedy (top-k 1), one fresh process per run, 400-token cap), six multiplication questions (17 × 23, 23 × 17, 12 × 34, 9 × 87, 45 × 12, 36 × 25; thinking disabled, three runs each) got the right product in 18 of 18 runs but ran on into self-talk until the cap in 14, and `What is 17 times 23?` with thinking on and no budget (the bundle's default) did not close its thought in 3 of 3 runs. The int8 file now keeps the 48 GatedDeltaNet decay/gate FCs (`in_proj_a` / `in_proj_b` of the 24 linear-attention layers) float and is otherwise unchanged: run `agents_a1_work/build_bundles.py` with `QWEN35_FLOAT_FC_REGEX='Linear_in_proj_(a|b);'` set, which it passes to the int8 `quantize_minicpm5.py apply` step (the mixed_int4 recipe is unchanged; the same cause and the same fix as the 2026-10-11 note on the [Qwen3.5-2B card](https://huggingface.co/litert-community/Qwen3.5-2B)). The new file is 4,419,086,560 bytes, sha256 `1481f76d…`; its other sections are byte-identical to the shipped bundle's. Exact bytes need the 2026-09-21 environment: the litert-torch 0.9.3 package with ai-edge-quantizer 0.8.0, ai-edge-litert 2.1.6, torch 2.12.1 and transformers 5.14.1 beside the pinned litert-torch checkout, with the checkpoint read from a local directory. A float export made there on 2026-10-13 (16,909,163,968 bytes, sha256 `59f6076c…`), quantized with the previous recipe, reproduces the shipped TFLite section byte for byte (sha256 `e0839d95…`, 4,403,461,472 bytes). Beside litert-torch 0.9.2, with the checkpoint read by its Hub id, the same recipe gives a section 2,064 bytes larger, with ai-edge-quantizer 0.8.0 and 0.10.0 alike; which of the two differences moves the bytes was not isolated. Run end to end on 2026-10-13 from that export, `build_bundles.py` writes the new Hub file's TFLite section byte for byte (sha256 `6daa0173…`) and the Hub mixed_int4 file's (sha256 `9c81c446…`). Measured on the new file: on the Mac CPU (greedy (top-k 1)) the six questions end by themselves with the right product in 18 of 18 runs, and a raw `<think>` prefill closes the thought in 3 of 3; the 8-question gate (temperature 0) passes 8/8 on CPU and GPU; the Mac GPU gives 6 of 6 (greedy (top-k 1), thinking disabled, one run per question); and on a Galaxy S26 CPU at the bundle's default (thinking on, the bundle's default sampler, 1024-token cap), `What is 17 times 23?` closed the thought and was answered in 4 of 4 processes.

## Zamba2 (Mamba2 backbone + a shared, LoRA-specialized transformer block) — and the metaspace tokenizer trap

`zamba2_work/convert_zamba2.py` converts Zyphra's Zamba2 instruct models to `.litertlm`. The 1.2B is 38 layers: 32 Mamba2 selective-scan layers plus 6 `hybrid` positions where ONE shared transformer block — a single set of attention+MLP weights tied across all six positions, specialized by per-position LoRA adapters (rank 128) — attends over `concat(hidden, original_embedding)` and projects into that position's mamba input. Published: [litert-community/Zamba2-1.2B-instruct](https://huggingface.co/litert-community/Zamba2-1.2B-instruct) and [litert-community/Zamba2-2.7B-instruct](https://huggingface.co/litert-community/Zamba2-2.7B-instruct). **Requires litert-lm ≥ 0.15 to run.**

```bash
cd zamba2_work
git clone https://github.com/john-rocky/litert-torch litert-torch-zamba2
git -C litert-torch-zamba2 fetch origin 115a13607c730c81018bb9789138a3e5e5119e3d   # base unreachable from main since 2026-08
git -C litert-torch-zamba2 checkout 115a13607c730c81018bb9789138a3e5e5119e3d
git -C litert-torch-zamba2 apply "$(pwd)/zamba2_litert_torch.patch"
PYTHONPATH=litert-torch-zamba2 python convert_zamba2.py Zyphra/Zamba2-1.2B-instruct out_zamba2_12b
# GPU ship shape: declare fp32 activations in the bundle TOML (repack, no re-export)
python ../scripts/set_activation_type.py out_zamba2_12b/Zamba2-1.2B-instruct_int8.litertlm Zamba2-1.2B-instruct_int8.litertlm --type fp32
```

The scan needed nothing new: Zamba2's reference scan is the same Mamba2 SSD math as granite's in a different spelling, its wrapper is NemotronH's shape (min-only `time_step_min` clamp — the pad-decay floor trap — and the same width-0 `d_mlp` split), and every mixer attribute matches NemotronH's names, so the patch reuses the NemotronH folded rank ≤ 4 forward verbatim and asserts the reuse contract on the upstream source at patch time (layer checker: `zamba2_folded_cache_check.py`, five cache paths, worst 2.26e-04). The cache side adds Zamba2's third config vocabulary for the same mixer geometry (`n_mamba_heads` / `mamba_headdim` / `mamba_ngroups`) to the shape inference, routes the 6 shared-attention positions to the composite hybrid cache layer (KV + conv/SSM state at one index, as Falcon-H1), and the shared block itself traces cleanly: tied weights are stored once, and each position's LoRA adapter is selected by a Python int at trace time. What Zamba2 DID need was new:

- **The metaspace tokenizer trap.** Zamba2 is a Llama/Mistral SP-BPE (metaspace `▁`) tokenizer serialized as `tokenizer.json` — and the runtime executes the HF decoder pipeline **per streamed token**, so the trailing `Strip(" ", start=1)` decoder (meant to trim the ONE artificial leading space the `Prepend("▁")` normalizer adds to a whole sequence) eats the leading space of EVERY piece. Every word starts with `▁` in SP-BPE, so generations come out with all interior spaces missing ("Thesumof17and25is42.") — content correct, formatting destroyed, quality gates failing on word-boundary checks with zero degeneration. Byte-level-BPE models (Falcon/Qwen/Llama-3-style `Ġ` pieces) never hit this: their spaces live inside the piece bytes and their decoders carry no Strip. Fix: drop the Strip decoder from the bundled tokenizer (`fix_tokenizer_strip.py`, wired into the converter) — interior spaces are exact under per-token decode, and the one behavior change is a sequence-initial space the runtime already trims.

Ship verification (1.2B, on the shipped file or its float parent): teacher-forced parity vs HF fp32 across 48 positions (split harness — the fp32 export is 4.98 GB) — top-1/top-5 100%, Pearson 1.0000, KL ≈ 0; 8-question gate **8/8 on every lane** (GPU/CPU × litert-lm 0.15.0/0.16.0); hermetic prompt-length sweep 40/40 CPU + 20/20 GPU; iPhone 17 Pro composite probe **8/8 on CPU**, 6/8 on GPU where both misses (8×7, the rhyme) are questions the **HF fp32 reference itself answers incorrectly** on this composite — the GPU path tracks the reference's own behavior, deterministically and with identical text on Mac and iPhone Metal. iPhone 17 Pro: Metal decode 12.8 tok/s / prefill 96.6 / TTFT 1.63 s / 5.43 GB peak (CPU: 7.4 / 87.3 / 1.73 s / 1.82 GB) — the full 12-signature ladder loads on a 12 GB phone with no memory pressure; the GPU peak includes the six shared-attention positions' wide KV (32 KV heads × 128 head dim at 4096 context) in fp32. Mac M4 Max (`-p 256 -d 256 --runs 3 --cache no`, 0.16.0): GPU 1033 / 74.0 tok/s, CPU 450 / 22.7.

The **2.7B** is the same driver with two differences worth knowing before you run it. First, it is a *two-block* Zamba2 — 54 Mamba2 layers with TWO shared transformer blocks applied alternately at 9 interleaved positions — and **current `transformers` (5.14.x, and main as of 2026-08-14) cannot construct any two-block Zamba2 at all**: `Zamba2Model.get_layers` assigns `block_id` by global layer index while the checkpoint layout and the model's own weight-tie cycle follow hybrid *occurrence* order, so construction raises before weights are read. The patch carries a structure fix (`block_id` by hybrid order, with an anchor assertion), verified key-exact against the published checkpoint; the parity leg drives the reference through `parity_pt_structfix.py` for the same reason. Second, its prefill ladder is cut to six signatures — `ZAMBA2_PREFILL_LADDER=1024,256,64,16,4,1 python convert_zamba2.py Zyphra/Zamba2-2.7B-instruct out_27b` — because a signature costs memory whether or not it is called.

Ship verification (2.7B): parity 48 positions top-1/top-5 100%, Pearson 1.0000, KL ≈ 0; 8-question gate 8/8 on all four lanes; hermetic sweep 40/40 CPU + 20/20 GPU; Mac M4 Max GPU 603 / 43.6, CPU 282 / 14.7. On an iPhone 17 Pro the composite probe is **8/8 on CPU** (prefill 14.3 tok/s, decode 3.6, TTFT 10.2 s, 2.19 GB peak) — and **Metal does not load this size on a 12 GB phone**: engine creation is killed by the OS, deterministically, twice, warm cache included.

That last result is worth a measurement habit, because it is cheap and it saves a conversion. **The Metal path resides the weights in fp32, so phone GPU peak memory runs at a multiple of the int8 file** — 3.98× on the 1.2B (1364 MB file → 5432 MB peak; its CPU run is 1.34×), and 2.80× on Falcon-H1-1.5B-Deep. The multiple is architecture-dependent — Zamba2's wide shared-attention KV inflates its own — so measure it on a sibling that fits, and budget the pessimistic end. At 2.80 GB the 2.7B projects to ≈ 11 GB against 11.7 GB of phone RAM, which is what happens. The tempting fix — cut the prefill ladder — does nothing here: going from 12 signatures to 6 changed neither the failure nor its timing, because the weight term, not the variant term, is what overflows.

### Zamba2 finetune intake — the pre-port serialization wall (measured refusal)

`scripts/convert.py` routes `model_type: zamba2` through HYBRID_RECIPE like the other pinned
families. But the only real Zamba2 derivative on the Hub today
([ssmits/Zamba2-1.2B-instruct-Dutch](https://huggingface.co/ssmits/Zamba2-1.2B-instruct-Dutch))
is serialized in **Zyphra's pre-transformers-port format**, and that is a wall no converter
can cross, measured 2026-08-25 on transformers 5.14.1:

- Its `config.json` speaks the 4.43-era vocabulary (`layers_block_type: ['m','g',…]`,
  `state_size`, `use_mamba_kernels`, …) — current transformers rejects it at construction
  (strict `validate_layer_type`).
- The key VALUES all pair off exactly with the base's modern re-serialization (state_size ↔
  mamba_d_state 128, lora_rank ↔ adapter_rank 128, …), so we tried the honest translation:
  construct under the base's config and load the derivative's weights. The state_dict is a
  different generation too — `model.mamba_layers.*` layout, **684 missing / 404 unexpected
  keys**. Bridging that layout is the transformers port itself, not a converter's thin layer
  (and the port still has open construction bugs upstream:
  [transformers#47994](https://github.com/huggingface/transformers/issues/47994), which breaks
  every `num_mem_blocks > 1` checkpoint even in the modern format).

`convert.py` therefore refuses pre-port checkpoints at the entry gate, before any download
(`preport_serialization`, exit 2), with the concrete action: the model author re-serializes by
loading with transformers ≤ 4.48 and `save_pretrained` with ≥ 4.49. A future new-format
derivative rides the same pinned recipe that shipped the two litert-community Zamba2 models.


## Nemotron-H (Mamba2 + MLP + attention, three layer kinds) — and the registry trap

`nemotron_h_work/convert_nemotron_h.py` converts NVIDIA's Nemotron-H (the 4B: 24 Mamba2 + 24 plain-MLP + 4 attention layers) to `.litertlm`. Published: [litert-community/Nemotron-H-4B-Instruct-128K](https://huggingface.co/litert-community/Nemotron-H-4B-Instruct-128K). **Requires litert-lm ≥ 0.15 to run.**

```bash
cd nemotron_h_work
git clone https://github.com/john-rocky/litert-torch litert-torch-nemotron
git -C litert-torch-nemotron fetch origin 115a13607c730c81018bb9789138a3e5e5119e3d   # base unreachable from main since 2026-08
git -C litert-torch-nemotron checkout 115a13607c730c81018bb9789138a3e5e5119e3d
git -C litert-torch-nemotron apply "$(pwd)/nemotron_h_litert_torch.patch"
PYTHONPATH=litert-torch-nemotron python convert_nemotron_h.py nvidia/Nemotron-H-4B-Instruct-128K out_nemotron_4b
# GPU ship shape: declare fp32 activations in the bundle TOML (repack, no re-export)
python ../scripts/set_activation_type.py out_nemotron_4b/Nemotron-H-4B-Instruct-128K_int8.litertlm Nemotron-H-4B-Instruct-128K_int8.litertlm --type fp32
```

The folded scan is a **port, not reuse**: NemotronH's `torch_forward` is an older SSD spelling, not byte-identical to granite's. The lessons, in the order they cost time:

- **Check how the block CONSTRUCTS its mixer before trusting a class swap.** `NemotronHBlock` picks mixer classes from a module-level `MIXER_TYPES` dict bound at import time, so swapping the module's class attribute exports the *unrewritten reference scan* — with no pad guard, and with every parity gate green (reference math is correct math). **A parity gate can never prove which scan form got traced**; only a pad sweep or the GPU sieve can. The patch swaps the dict entry and adds a loud zero-patched-mixers guard.
- **A min-only dt clamp floors the pads.** transformers' NemotronH torch path clamps `softplus(dt + dt_bias)` at `time_step_min` with no upper bound, so a zeroed pad input still decays the recurrent state by `exp(time_step_min · A)` per pad. The pad guard forces pads to exact identity (`dt = 0` applied *after* the clamp). That floor is not NVIDIA's arithmetic: NVIDIA's modeling code, and transformers' own CUDA path, clamp to `time_step_limit` (default (0, inf), no floor). The 4B exports above were made through transformers' torch-path arithmetic; the 9B entry below follows NVIDIA's limit (`nemotron_h_vendor_dt.patch`), and a new Nemotron-H conversion should too.
- **`mlp` = a cache-less layer type.** Mamba/attention layers address the cache by absolute layer index, so MLP layers need a layer object that keeps the index while contributing ZERO tensors to the flatten. The dict fallback would have allocated 24 phantom KV buffers — and signatures pay RAM for every buffer that exists, called or not.
- **Config vocabulary.** NemotronH spells the mamba geometry `mamba_num_heads`/`mamba_head_dim`/`ssm_state_size`/`conv_kernel`/`n_groups`, and the SSM intermediate is `heads × head_dim` — equal to `expand × hidden` on granite/bamba by construction, NOT here (7168 vs 6144 on the 4B).
- **Reference forwards need `use_cache=False`** — transformers' own DynamicCache has no `mlp` layer type and KeyErrors on it.
- **Sweep state-carrying models with a fresh engine per prompt length.** Reusing one engine across sweep conversations rewinds a shared prefix only partially; that corrupts linear-attention state and produces false failures indistinguishable from conversion bugs.
- **Read an HF reference per scan chunk.** Without the CUDA kernels (CPU, Apple MPS) transformers runs its naive chunked scan ("Falling back to the naive implementation"), and on prompts longer than `chunk_size` (128 here) it lost part of the question: on NVIDIA-Nemotron-Nano-9B-v2 the 16 GSM8K prompts over 128 tokens scored 6/16 in transformers fp32 on MPS against 15/16 for the bundles, and 15/16 in transformers once `chunk_size` was raised to 256 (one chunk). Split a GSM8K comparison at `chunk_size`, or run the reference with `chunk_size` ≥ the longest prompt, before calling a gap a conversion loss.

Ship verification (4B): float parity vs HF teacher-forced across 8 decode positions — max|logit diff| 5.8e-05, correlation 1.000000, top-1/top-5 identical; 8-question gate **8/8 on CPU and GPU**; hermetic sweep clean at the ship shape; iPhone 17 Pro (Metal) runs with GPU == CPU answers (peak 4.02 GB, increased-memory entitlement). Honest limit: a 4B does not fit an 8 GB Android phone — on a Pixel 8a, engine creation aborts on both backends. Mac M4 Max: GPU 724 prefill / 75.0 decode tok/s, CPU 99 / 20.1.

Derivative intake: `scripts/convert.py` routes `model_type: nemotron_h` through HYBRID_RECIPE
like the other pinned families. The 4B above still has no real derivative, but the family
gained a **new base** in 2026 — `nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16` (3.4M downloads,
19 finetunes + 14 adapters, the most-downloaded ≤4B model released in 2026 by a wide margin)
— and it rides the same recipe unchanged. Measured 2026-08-26, one command, no new patch.
Published: [litert-community/Nemotron-3-Nano-4B](https://huggingface.co/litert-community/Nemotron-3-Nano-4B)
(2026-08-27; to our knowledge the first Nemotron-3-Nano in LiteRT form — the Hub otherwise
carries only GGUF conversions).

```bash
python scripts/convert.py nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16
# ship step (as for the 4B above): declare fp32 activations so the GPU executor keeps range
python scripts/set_activation_type.py out/.../NVIDIA-Nemotron-3-Nano-4B-BF16_int8.litertlm \
    Nemotron-3-Nano-4B_int8.litertlm --type fp32
```

export 1645 s → 3.84 GiB int8 (4,126,697,184 B), 50 state buffers (42 mamba conv+SSM,
8 KV; 21 mamba + 17 MLP + 4 GQA layers). Gates on the **shipped file**: **CPU 7/8 PASS**
(the one miss is the rhyme item — "violets are purple"), **GPU 8/8** with `--cache no`.
M4 Max `benchmark -p 256 -d 256 --runs 3`: GPU 696 prefill / 74.5 decode tok/s,
CPU 395 / 28.4.

**GPU on this bundle needs `--cache no`** — one-variable measurement, same runner and file
with only the cache flag flipped: with the compiled-graph cache, `litert-lm run --backend
gpu` dies on WebGPU `Invalid BindGroup` validation errors and the 8-question sweep through
`litert-mac-verify` returns token soup (0/8); with `--cache no` the same file answers 8/8.
The cache path is where it goes wrong; the root cause is not isolated further. Note the
exit gate calls `litert-mac-verify`, which has no cache flag — so a routed `convert.py`
run gates this family on CPU and a green CPU gate says nothing about GPU here. The sibling
4B's card documents the same `--cache no` invocation, so treat it as a family-level caveat
rather than a property of this checkpoint.

Chat template embedded
**byte-equal** to the repo's `chat_template.jinja` (10,504/10,504 — note the repo's
`tokenizer_config.json` copy is a *different* 10,497-byte string; the exporter embeds the
one `AutoTokenizer` actually resolves). Reasoning model: ChatML turns, `<think>`,
`eos = <|im_end|>` (id 11) added by the stop-token guard on top of the exported `ids: 2`.

**2026-10-13 — chat template re-pack (metadata only).** The template NVIDIA ships keeps the full `<think>…</think>` block on the latest assistant turn and rewrites it to an empty `<think></think>` on every assistant turn that a newer user message follows (`truncate_history_thinking`, default true). LiteRT-LM re-renders the conversation before each message and requires the new render to extend the previous one; on litert-lm 0.18.0, which passes the model's reasoning to the template, the second message of every conversation with thinking on therefore failed with `INTERNAL: The new rendered template string does not start with the previous rendered template string` (own report [LiteRT-LM#3166](https://github.com/google-ai-edge/LiteRT-LM/issues/3166); litert-lm 0.17.1 did not pass the reasoning and was unaffected). `templates/nemotron3nano_history.jinja` is that template with the position condition removed on its two lines (134 and 172), so every assistant turn in the history renders as `<think></think>` plus its answer; the generation prompt is unchanged.

```bash
# litert-lm 0.18.0 (CLI and builder); <in> = the file published until 2026-10-13 (sha256 feefdcd5…)
python scripts/set_prompt_template.py <in>/Nemotron-3-Nano-4B_int8.litertlm out/Nemotron-3-Nano-4B_int8.litertlm \
    templates/nemotron3nano_history.jinja --litert-lm litert-lm
python scripts/section_identity.py <in>/Nemotron-3-Nano-4B_int8.litertlm out/Nemotron-3-Nano-4B_int8.litertlm
```

Result: `LlmMetadataProto 11006 -> 10932 DIFF`, `ExecutorMetadataProto 3365 -> 3365 SAME`, `HF_Tokenizer_Zlib 2806488 -> 2806488 SAME`, `TFLiteModel 4123829984 -> 4123829984 SAME` — the only field that differs inside `LlmMetadataProto` is `jinja_prompt_template`. File size unchanged (4,126,697,184 B). The published file's sha256 is `e9274fb21c95944c5ad76b10a8294ebfe5b54a359d6d6166b1879a958331ca6a`; a re-run gives a different file sha256 (the 0.18 packer writes a fresh uuid and timestamp into the header) with all four sections byte-identical to it. Checked on the new file: three turns ("My name is Kenji." / "What is my name? Answer with the name only." / "What is 17 times 23? Answer with the number only.") answer "Kenji" and "391" on litert-lm 0.18.0 with thinking on and off and on 0.17.1 (CPU, greedy, one run each), on a Galaxy S26 CPU with thinking on and off, and the card's 8-question gate is 8/8 on CPU and on GPU with `--cache no` on the 0.18.0 CLI (the same as the previous file).

Three things this intake fixed or established, each measured:

- **`auto_map` is not proof of remote code.** The repo declares `auto_map`, and the entry
  gate refused it outright (exit 2) — wrongly. transformers 5.14.1 registers `nemotron_h`
  natively, so without `trust_remote_code` the library class loads and the repo's Python is
  never imported (`AutoConfig.from_pretrained` returns transformers' own `NemotronHConfig`;
  the export ran with `trust_remote_code: False`). The gate now refuses only when `auto_map`
  is present **and** the model_type is absent from `CONFIG_MAPPING` — verified in both
  directions: this model converts, and `kakaocorp/kanana-2-1.3b-instruct` (`kanana2_tiny`,
  unregistered) still refuses with the same structured reason.
- **The ≥3B reduced prefill ladder now reaches the recipe path too.** It was wired only into
  the stock export, so a 4B hybrid — exactly the shape whose signature-count RAM law is
  documented above — was still getting the full 11-signature ladder. `convert_via_recipe`
  passes `PREFILL_LENGTHS` to the driver, and all four drivers read it (Zamba2 keeps its
  older `ZAMBA2_PREFILL_LADDER` name as an alias). This export used 6 prefill + decode.
- **nemotron_h was the one recipe family with a spurious start token and no guard.** Its
  tokenizer says `add_bos_token: False` and its template opens with `<|im_start|>`
  (`<SPECIAL_10>` on the 4B), yet the bundle's `LlmMetadata` carries `start_token: "<s>"` —
  **including the published 4B bundle**, checked directly. Running the stock path's generic
  guard over all four families reproduces every hand-made choice here: it fires on granite
  (which the family flag already dropped) and on nemotron_h, and stays silent on Falcon-H1
  and Zamba2, whose templates render their own BOS. So the guard is now wired into the
  recipe path instead of a fifth per-family flag. Honest limit on the impact: at 4B this
  model is **robust** to the mismatch — the gate scores 7/8 with the start token and 7/8
  without it (same single miss, a rhyme judgment), and HF greedy is byte-identical on 2 of
  3 probes, diverging mid-thought on the third but reaching the same answer. The fix aligns
  the runtime stream with the training stream; it does not rescue a broken model. The
  granite-350m lesson is why it still matters — that mismatch bites at small scale.

Trap already recorded above and confirmed again here: HF reference forwards for this family
need `use_cache=False` (transformers' generic `DynamicCache` KeyErrors on the `mlp` layer
type), which is what the greedy A/B above ran with.

Coverage boundary, stated rather than implied: **RWKV-7 is the one shipped LLM family with
no derivative intake here.** The shipped 0.1B-world size has zero Hub derivatives (measured
2026-08-25; the rwkv7-g1 line's ~27 "finetunes" are almost all the author's own size
re-serializations), the architecture is not transformers-native, and its conversion never
rode this repo's rails — so there is nothing to route until real derivative demand appears.

### 2026-08-27 — Hy-MT2-1.8B intake: one config bake closes the sweep's real gap, and the engine's start_token prepend gets proven

The sweep below left one in-demand family untried with no blocker found:
`tencent/Hy-MT2-1.8B` (`hunyuan_v1_dense`, 226k downloads, 1.2k likes — a 33-language
translation model). It now converts with the standard one command:

```bash
python scripts/convert.py tencent/Hy-MT2-1.8B
```

`converted_pass`: export 120 s → 1.69 GiB int8 (1,815,622,960 B, 2.04B params), gate
6/8 non-degenerate (~50 tok/s decode on the verifier's default backend), template
byte-equal 654/654 (this repo carries only `chat_template.jinja` — no
tokenizer_config copy to disagree with).

Published: [litert-community/Hy-MT2-1.8B](https://huggingface.co/litert-community/Hy-MT2-1.8B)
(2026-08-27; to our knowledge the first Hy-MT2 in LiteRT form — the Hub otherwise carries
GGUF, MLX and FP8 conversions). Gates on the shipped file: 6/8 on CPU **and** GPU with the
same two misses, so the drops are a property of this translation-tuned 1.8B, not a backend.
M4 Max `benchmark -p 256 -d 256 --runs 3 --cache no`: GPU 2008 prefill / 105.8 decode
tok/s (TTFT 0.14 s, after a ≥300 s rest), CPU 211 / 34.0 (TTFT 1.24 s); both backends
gated on real generations before quoting. Unlike the Nemotron hybrids, GPU works with the
default compiled-graph cache on this dense bundle.

**Why stock failed, and why the fix is exact.** The stock export dies in
`torch.export` with `GuardOnDataDependentSymNode` at transformers'
`dynamic_frequency_update` (`if seq_len > max_seq_len_cached`). But transformers
5.14.1 resolves this family's rope STATICALLY at init — its hunyuan modeling computes
`base = rope_theta * alpha**(head_dim/(head_dim-2))` once ("DynamicNTKAlphaRotary")
and never rescales below `max_position_embeddings`. Only the dead growth branch is
untraceable. `convert.py` therefore bakes the resolved base into `rope_theta`
(11,158,839.925) and drops `rope_scaling` before export
(`normalize_static_alpha_rope`). Measured equivalence: `inv_freq` bitwise-equal,
teacher-forced logits bitwise-equal (max |diff| 0.0). Faithful for all positions
≤ 262144 — far past the bundle's 4096 context.

**The engine prepends `start_token` unconditionally — proven, not assumed.** This
family's template renders `<|hy_begin_of_sentence|>` itself and its tokenizer adds no
BOS at encode time, so the exported metadata `start_token` looked like a double BOS.
Output-vs-HF comparison could not decide it: int8 greedy divergence coincidentally
matched the extra-BOS reference on 2 of 3 probes even after the drop. The
discriminator that decides runs entirely inside the runtime: `[start_token]+rest`
(original bundle, `--no-template`, rendered text minus its BOS string) generates
byte-identical greedy output to `[template-BOS]+rest` (dropped bundle, normal
templating) on 3 of 3 probes — same id, same stream, so the default export was
feeding BOS twice. This supersedes the 2026-08-25 note that the guard's silence
on BOS-rendering templates reproduced the right choice — that silence kept a
double BOS.

*Corrected 2026-08-27, after auditing all 88 published bundles.* "The template
renders BOS" is not one shape but two, and only one of them duplicates. The
engine renders the template with **minijinja, which leaves `bos_token`
unbound**, so `{{ bos_token }}` renders **empty** on device and only a BOS
written as a **literal** survives. Read straight out of the runtime:
`Conversation.render_message_to_string` returns `<start_of_turn>user…` for a
template whose source begins `{{ bos_token }}`, and the benchmark prefill
counter reads 16 tokens with a `start_token` against 15 without — one token,
always prepended. The engine's own `tokenize()` adds no special tokens either,
so a tokenizer's `add_bos_token` / post-processor never applies on device.

So Hy-MT2 (a literal) really was doubled, but **Falcon-H1 and LFM2.5 write
`{{ bos_token }}` and are faithful** — their `start_token` supplies the single
BOS the model expects, and dropping it would ship a bundle with no BOS at all.
The families that were actually doubled are the ones with a literal BOS at the
head of the template: Zamba2, SmolVLM2, and Nanbeige (via its affixes).

Honest numbers on the fix: the double-BOS bundle scored 8/8 on the gate; the
stream-faithful bundle scores 6/8 (misses: "Cool" for opposite-of-hot, "pink" for
the rhyme — noise-level for a translation-tuned 1.8B, no degeneration). The ship
follows the training stream, not the lucky gate. On the model's actual task,
runtime greedy matches HF bf16 byte-for-byte on 1 of 3 translation probes; the
other two are valid alternates of the int8-vs-bf16 class (cross-checked: the
no-BOS control arm lands on the plain-HF string on one of them).

Trap for the next family: do not diagnose BOS handling by comparing outputs
against HF arms — quantization noise can mimic the wrong arm. Compare token
streams inside the runtime (`--no-template` against normal templating).

### What a 2026 sweep of the phone band actually finds

To keep the coverage claim honest rather than asserted, the ≤4.5B text-generation models
created in 2026 were enumerated mechanically (2026-08-25/26, HfApi, five angles — trending,
downloads, likes, new-and-recent, and a direct sweep of ~70 major model orgs; parameter
counts read from each repo's own safetensors metadata). Union: **835 models across 93
`model_type` values**, of which **589 (70.5%)** route through a path documented above. Of
the rest, most are research checkpoints (`gpt2`, `opensci`, `ceno`, one-off student models).
What is genuinely outside, in demand order, with the reason:

| model | why it is out |
|---|---|
| `sapientinc/HRM-Text-1B` (248k dl) | transformers-native, but hierarchical recurrence (`H_cycles`/`L_cycles`) with `prefix_lm` — not a prefill/decode contract |
| ~~`tencent/Hy-MT2-1.8B`~~ | was listed here as "no blocker found — untried"; converted 2026-08-27, see the intake above |
| `CohereLabs/tiny-aya-*` (55 ft + 55 ad) | the bases are **gated**; the entry gate refuses on principle, not on capability |
| `ai21labs/AI21-Jamba2-3B` (65k dl) | Mamba-**1** hybrid; the four pinned patches are all Mamba2/SSD, so this needs a new port rather than a recipe |
| `kakaocorp/kanana-2-*`, `ibm-granite/granite-swash-*` | genuinely remote-code / not yet registered in transformers 5.14.1 |
| `Zyphra/ZUNA`, `stabilityai/SAME-*` | not transformers-format checkpoints |

Two things this refuted about the previous session's summary: the phone band's gaps are
**not** only MoE and Gemma (Hy-MT2 and Jamba2 were dense-ish, ungated, and in demand;
Hy-MT2 has since been converted — the intake above — and Jamba2 still stands), and
one apparent gap — the most-downloaded 2026 model in the whole band — was this converter's
own false refusal, fixed above. Separately, the MoE boundary is getting load-bearing:
`facebook/MobileMoE-S/M`, `LFM2.5-8B-A1B`, `ibm-granite/granite-switch-4.1-3b`, and
`arcee-ai` afmoe all landed in the phone band in 2026, and the runtime still ships no MoE
kernel.

## Bamba (arch reference — no published artifact)

`bamba_work/bamba_litert_torch.patch` carries the same folded-scan machinery to IBM's Bamba. `BambaMixer.torch_forward` is byte-identical to granite's, so the extension is a thin wrapper that asserts the source identity at patch time. Arch-verified at tiny scale (E2E float parity correlation 1.000000; 5-path cache checker worst 5.5e-07), and the 9B converts and runs — but every published Bamba checkpoint is a 9B BASE model (no chat template), desktop-class, so nothing is shipped. The patch is here as the third data point on how far the granite scan form travels.

## LFM2.5-Encoder (bidirectional encoders → plain .tflite)

`lfm_work/convert_lfm25_encoder.py` converts LiquidAI's LFM2.5-Encoder-350M/230M — multilingual (15-language) bidirectional masked-LM encoders on the same LFM2 hybrid backbone — to plain LiteRT `.tflite` for embeddings / retrieval / fill-mask, fully on CPU. These are NOT `export_hf` runs: the models have no KV cache, so the HF eager model (remote code = stock `Lfm2Model` + bidirectional patches) is traced directly with `litert_torch` multi-signature convert. Signatures: `encode_{64,128,256,512}` → `last_hidden_state` `[1,S,1024]` (padded positions zeroed), plus `mlm_128` → masked-LM logits. Published: [litert-community/LFM2.5-Encoder-350M](https://huggingface.co/litert-community/LFM2.5-Encoder-350M), [litert-community/LFM2.5-Encoder-230M](https://huggingface.co/litert-community/LFM2.5-Encoder-230M).

```bash
cd lfm_work
python convert_lfm25_encoder.py LiquidAI/LFM2.5-Encoder-230M out_lfm25_encoder_230m   # -> fp32 + wi8fc int8 + fp16
python verify_lfm25_encoder.py  LiquidAI/LFM2.5-Encoder-230M out_lfm25_encoder_230m   # 15-language parity report
```

Env: litert-torch ≥ 0.9.2, transformers ≥ 5.12 (the encoders' remote code needs 5.x; tested 5.14.1), torch 2.12. The non-obvious parts:

- **Mobile-GPU delegation needs a rank-4 respelling of transformers' GQA `repeat_kv` (2026-08-13 re-export, whole family).** The stock HF spelling — `x[:, :, None, :, :].expand(...).reshape(...)` — lowers to a rank-5 `BROADCAST_TO` + rank-5 `RESHAPE`, which the Metal/OpenCL/WebGPU delegates refuse; the original exports ran only ~136/828 ops on GPU and were *slower* than CPU there. `lfm_work/_rank4_repeat_kv.py` respells it as an outer product against a ones row through a matmul (bitwise-identical — asserted at import) and every convert script in this family now calls `install()` before `from_pretrained` and `rebind_all()` after (remote code imports symbols by value, so patching transformers by module name can miss the live call site — the same trap as `apply_mask_to_padding_states`). The graph gate is `assert "BROADCAST_TO" not in hist`. Result: the int8 files fully delegate on Pixel 8a OpenCL (852/852 nodes, `encode_512` ~10 ms) and iPhone 17 Pro Metal, with GPU-vs-fp32-reference cosine at the int8 noise floor (≥0.9948).
- **GPU precision must be fp32.** At fp16 GPU precision the norm reductions overflow (|h| reaches ~50) and every output is NaN — set the GPU accelerator's precision to fp32 (`precision = 2` in the LiteRT GPU options); correctness was verified on-device only in that mode.
- **The 5-signature 350M file cannot GPU-compile on phones** — every signature subgraph materializes its own fp32 copy of the weights on the GPU, and 5×350M exceeds phone memory (the 230M fits). The 350M repo therefore also ships `LFM2.5-Encoder-350M_wi8fc_single-sig.tflite` (encode_512 only) as the phone-GPU artifact; multi-signature use stays on CPU. The PII-Detector and Spellchecker int8 files do not GPU-compile at all (a runtime kernel limit on their 161-label / 128802-vocab int8 heads — consistent across Metal and OpenCL); CPU remains their mobile path.
- **transformers' `apply_mask_to_padding_states` is a no-op at batch==1** (`shape[0] > 1` guard) — and batch-1 right-padded input is exactly what static tflite signatures produce. Without a fix, the symmetric (non-causal) ShortConv reads pad-token embeddings past the sequence end and the last valid tokens diverge. The script rebinds the symbol **inside the remote-code module** (patching `transformers.models.lfm2.modeling_lfm2` does nothing — the remote module imported the function by value) to an unconditional mask-multiply; zeroed pad inputs make the padded forward token-exact vs the unpadded one.
- **Gate on pad-CONTENT invariance, not on padded-vs-unpadded diff.** With complete masking, changing the pad-region token ids leaves valid positions bitwise unchanged (the script asserts exactly 0.0). Padded-vs-unpadded absolute diff is shape-dependent reduction-order float noise (~1e-4 on the 350M) — a threshold on it false-fails.
- **Signature inputs must be passed as `sample_kwargs`** — positional `sample_args` names the tflite inputs `args_0/args_1` instead of `input_ids`/`attention_mask`.
- **`wi8fc` int8 = linears + embedding only, convs float** (same rule as the LFM2.5 decoders — `ALL_SUPPORTED` int8 post-hoc kills the conv layers). The embedding lowers to `EMBEDDING_LOOKUP` (not GATHER), so the int8 recipe reaches the 65536×1024 vocab table, and the tied `lm_head` FC dedupes against it in the flatbuffer.
- **Parity (16 sentences over all 15 languages, mean-pooled-embedding cosine vs PyTorch)**: fp32/fp16 **1.000000** (fill-mask top-5 sets identical); wi8fc ≥ **0.9948** (230M) / **0.9966** (350M) min, fill-mask top-1 4/4 / 3/4. Cross-signature outputs (encode_64/128/256 on the same sentence) agree bitwise.
- **int8 is the mobile artifact; fp16 is desktop-only.** On an iPhone 17 Pro the int8 files reproduce desktop outputs bit-exactly (cosine 1.000000, max diff 0.0; encode_512 93 ms on the 230M, 145 ms on the 350M, 6 threads). The fp16 files jetsam at interpreter init even with the increased-memory-limit entitlement — XNNPACK unpacks fp16→fp32 per signature subgraph (×5, no weights cache). Also load one model per process on iOS; two sequential interpreter loads stack XNNPACK's repacked weights and hit the memory limit.
- **transformers 5.x remote-code load check**: assert `rotary_emb.inv_freq.min() > 0` after `from_pretrained` — 5.x meta-device materialization can silently zero init-computed buffers, and since reference and export share the model object, parity would still "pass" on a broken load.

### Task finetunes (Prompt-Router / Policy-Linter / PII-Detector / Spellchecker)

`lfm_work/convert_lfm25_encoder_finetunes.py router|linter|pii|gec` converts LiquidAI's four 350M encoder finetunes; `verify_lfm25_finetunes.py <kind>` runs task-level parity (route probabilities / rule flags / entity tags / edit tags) for fp32 + fp16 + int8. Published: [Prompt-Router](https://huggingface.co/litert-community/LFM2.5-Encoder-350M-Prompt-Router) · [Policy-Linter](https://huggingface.co/litert-community/LFM2.5-Encoder-350M-Policy-Linter) · [PII-Detector](https://huggingface.co/litert-community/LFM2.5-Encoder-350M-PII-Detector) · [Spellchecker](https://huggingface.co/litert-community/LFM2.5-Encoder-350M-Spellchecker). Each has a custom remote-code head — none is "the encoder script unchanged":

- **Router / Linter take pooling matrices as extra float32 graph inputs** (`text_pool` [1,1,S], `category_pool`/`rule_pool` [1,8,S]) — the offset-derived mean-pool weights the vendors' `route()` helpers compute; building them host-side keeps the graph static. All-zero rows for unused lane/rule slots are exact no-ops (their logit collapses to the score bias; ignore beyond your real N).
- **The batch-1 padding rebind must go through `Lfm2ShortConv.slow_forward.__globals__`** — each finetune repo inlines its own copy of the bidirectional patches, so patching a module by name misses them; the installed function's globals always hit the right namespace.
- **Spellchecker's tied vocab heads quantize for free**: `tie_replace` computes `$REPLACE`/`$APPEND` logits as `proj(h) @ embed[:64400].T`, which lowers to FULLY_CONNECTED with a constant weight — the int8 recipe reaches it and the embedding slice dedupes (429 MB total, not +264 MB). Its iterative decode and optional reranker stay host-side.
- **Task parity**: router/linter/gec are task-exact through int8; PII is exact at fp32/fp16, and int8 flipped exactly one tag in our test — an entity-end token with a 0.53-logit fp32 margin (inherently borderline; documented on the card).

### LFM2.5-Embedding-350M (the retrieval bi-encoder on the same backbone)

`lfm_work/convert_lfm25_embedding.py` converts [LiquidAI/LFM2.5-Embedding-350M](https://huggingface.co/LiquidAI/LFM2.5-Embedding-350M) — a multilingual (11-language) dense bi-encoder built on the same `Lfm2BidirectionalModel` backbone as the encoders above — to plain LiteRT `.tflite` for retrieval / RAG / semantic search. Signatures `embed_{64,128,256,512}` → `output_0` `[1,1024]`, **already CLS-pooled and L2-normalized**. Published: [litert-community/LFM2.5-Embedding-350M](https://huggingface.co/litert-community/LFM2.5-Embedding-350M) (int8 371 MB + fp16 712 MB).

```bash
cd lfm_work
python convert_lfm25_embedding.py LiquidAI/LFM2.5-Embedding-350M out_lfm25_embed_350m  # fp32 + wi8fc int8 + fp16
python verify_lfm25_embedding.py  out_lfm25_embed_350m --gates A,B,C,D --prefix-ab     # card oracle + STS17 + retrieval + mechanics
python bench_lfm25_embedding.py   out_lfm25_embed_350m
```

Env: litert-torch ≥ 0.9.2, transformers 5.14.1 (see the version-drift note below), torch 2.12. The non-obvious parts:

- **The repo's remote code does not load on transformers 5.14.1.** It targets 4.56.2 and installs `Lfm2ShortConv.forward = _shortconv_forward`, which forwards `**kwargs` straight into a `_noncausal_shortconv_forward(hidden_states, past_key_values, cache_position, attention_mask)` that does not declare `seq_idx` — which 5.14.1's `Lfm2DecoderLayer` now passes. Every forward dies with `TypeError: ... unexpected keyword argument 'seq_idx'`. The fix is to rebind `forward` to a wrapper that filters kwargs down to `inspect.signature(slow_forward).parameters`, **not** to edit the cached remote code — the filter survives a vendor update.
- **Pick the FA2 pad semantics, and pass the masks yourself.** The vendor supports two behaviours: the sdpa default lets the symmetric ShortConv read pad-token embeddings ("matching the behavior the checkpoints were trained with"), FA2 zeroes pads first ("closest match to the unpadded forward"), and the card reports the two equivalent within 0.002 nDCG across 11 languages. For a static multi-signature export only the FA2 behaviour is viable: under the default, the *same text* routed to `embed_128` vs `embed_512` returns a different vector (max abs 0.005). Force the zeroing and length invariance becomes exact.
- **The vendor's own comment about which mask arrives is already stale.** It says eager/sdpa hand the ShortConv a 4D additive mask "on which this is a no-op". On 5.14.1, `Lfm2Model.forward` builds `{"full_attention": create_causal_mask(...), "conv": create_recurrent_attention_mask(...)}` and the conv layers receive the **2D** pad mask. It is still a no-op, but because of the other half of the guard — `apply_mask_to_padding_states` needs `attention_mask.shape[0] > 1`, and static tflite signatures are exactly batch 1. Read the installed source, not the comment about it.
- **`Lfm2Model.forward` accepts `attention_mask` as a dict** (`if not isinstance(causal_mask_mapping := attention_mask, dict)`) and uses it verbatim, so the script builds both masks and hands them over — the same escape hatch ModernBERT offers. That keeps `create_recurrent_attention_mask` out of the trace, whose "return `None` when the mask is all-ones" branch would otherwise specialize at export time into a graph that ignores `attention_mask` (the identical hazard documented for the Nemotron embedder). Gate it: shortening the mask must move the output (4.8e-02 here).
- **Every invariance check lands on exactly 0.0**, which is what makes the multi-signature file safe to use: hand-built masks vs the vendor's own mask path 0.0, pad-content invariance 0.0, padded-vs-unpadded 0.0, and cross-signature agreement bitwise 0.0 at 64/128/256/512 on fp32, int8 and fp16 alike. (The encoders above only reach ~1e-4 on padded-vs-unpadded, because there the pooled output aggregates every position; here the CLS vector does not.)
- **Read `config_sentence_transformers.json` for the prompt strings; do not retype them from the prose.** The contract is `query: ` / `document: ` **with the trailing space** — prose and config agree, and the space changes tokenization (`Ġ`-prefixed content token, e.g. 3747 vs 3493). Note the sibling ColBERT repo stores a completely different contract (`query_prefix: "[Q] "`, `similarity_fn_name: "MaxSim"`, `do_query_expansion`, a punctuation skiplist) — reading the wrong repo's config is an easy way to ship the wrong prefix.
- **A 50-query retrieval set cannot settle a prefix question.** On NanoSciFact: no prefix 0.8668, no-space 0.8726, documented 0.8540 nDCG@10 — a ~0.019 spread on n=50, inside one standard error, with recall@5 differing by a single query. That is not evidence against the documented prefix; it is a set without the resolution to say anything. Use the graded STS-style gate for resolution, and follow the upstream contract for the prefix.
- **`wi8fc` int8 = FC dynamic-range + EMBEDDING_LOOKUP channelwise, convs float** (same rule as every LFM2 hybrid). Confirm it in the flatbuffer rather than trusting the recipe: 92/92 FULLY_CONNECTED int8, EMBEDDING_LOOKUP int8, all 10 DEPTHWISE_CONV_2D still float32. 354.5M params → fp32 1420.6 MB, int8 371.2 MB, fp16 711.9 MB. Unlike granite-embedding-r2 (65% vocab table) the table is only 19% of the parameters here, so the win comes from the linears.
- **Quality is quantization-flat.** STS17 over 11 language pairs: fp32 mean 0.6720, int8 0.6721. NanoSciFact: fp32 nDCG@10 0.8540 → int8 0.8494. fp16 is bitwise identical to the PyTorch reference on every task metric. (`en-tr` 0.083 and `nl-en` 0.484 are identical on the reference too — Turkish and Dutch are not among the model's 11 languages, so they are a control that the gate discriminates rather than a defect.)
- **int8 is the mobile artifact; fp16 runs on a large phone but is not a phone build.** On an iPhone 17 Pro (6 threads) int8 reproduces desktop outputs **bit-exactly** (cosine 1.000000, max diff 0.0) across en/ja/de/ar/hi, both prompt forms, short and long inputs — embed_128 40–44 ms, embed_512 137–142 ms, peak footprint 1245 MiB for the 4-signature file. fp16 also passes (max diff 2e-07) but peaks at **5799 MiB**, and its first call on each signature costs 355–538 ms of one-time XNNPACK fp16→fp32 unpacking against 73–98 ms warm. Note the per-signature memory cost is model-specific: 4 signatures cost far less here than the ~846 MiB/signature measured on the 2048-d Nemotron embedder.
- **fp16 is slower than fp32 on CPU** (Mac, embed_512: 200.8 ms vs 168.8 ms; int8 123.0 ms). XNNPACK unpacks fp16 weights to fp32 at run time, so the smaller file buys disk, not latency.

### LFM2.5-ColBERT-350M (late interaction, one vector per token)

`lfm_work/convert_lfm25_colbert.py` converts [LiquidAI/LFM2.5-ColBERT-350M](https://huggingface.co/LiquidAI/LFM2.5-ColBERT-350M) — the late-interaction sibling of the embedder above, on the same `Lfm2BidirectionalModel` backbone — to plain LiteRT `.tflite`. Signatures `encode_{32,128,256,512}` → `output_0` `[1,S,128]`: a pylate `Dense(1024→128, bias=False)` head plus a per-token L2 normalize, folded into the graph. Published: [litert-community/LFM2.5-ColBERT-350M](https://huggingface.co/litert-community/LFM2.5-ColBERT-350M) (int8 370 MB + fp16 710 MB).

```bash
cd lfm_work
python convert_lfm25_colbert.py LiquidAI/LFM2.5-ColBERT-350M out_lfm25_colbert_350m  # fp32 + wi8fc int8 + fp16
python verify_lfm25_colbert.py  out_lfm25_colbert_350m --gates A,B,C --unpadded-ref  # ranking + MaxSim retrieval + mechanics
python bench_lfm25_colbert.py   out_lfm25_colbert_350m
```

Env: litert-torch ≥ 0.9.2, transformers 5.14.1, torch 2.12. The non-obvious parts:

- **Zeroing the padding states is exactly WRONG here — the opposite of the embedder above, on the same backbone.** The two repos ship *different* `modeling_lfm2_bidirectional.py` under the same filename: the ColBERT copy gates `apply_mask_to_padding_states` on `_attn_implementation == "flash_attention_2"` and leaves hidden states untouched on eager/sdpa, with the vendor's own note that transformers >=5.x routes the raw 2D pad mask there and "would zero padding/query-expansion states and shift per-token embeddings (hurts ColBERT MaxSim)". `config.json` also sets `disable_flash_attention: true`. **Diff the remote code between sibling repos; never assume a shared filename means shared behaviour.** The converter asserts the gate is present before exporting — without it, query-expansion vectors are silently zeroed with no exception and no NaN, and only retrieval quality suffers.
- **The host contract is most of the work, and it is read out of pylate 1.6.0 rather than guessed.** Queries: tokenize to `query_length - 1` = 31 with `padding="max_length"`, padding with the **EOS token (id 7)** — pylate uses the mask token if the tokenizer has one and EOS otherwise, and this tokenizer has no mask token; insert `[Q] ` (id 64400) at **position 1**, after BOS; leave `attention_mask` **0** on the expansion positions (`attend_to_expansion_tokens: false`); and **score all 32 vectors**, expansion included. Documents: insert `[D] ` (id 64401) at position 1 and keep `skiplist_mask AND attention_mask` (32 punctuation ids). Scoring is MaxSim over unit vectors. `verify_lfm25_colbert.py` is a working reference implementation of all of it.
- **The pad token id is load-bearing.** Because padding states are not zeroed, padding with 0 instead of 7 moves the output by 0.34 max abs. Note `config.json` says `pad_token_id: 0`; pylate does not use it.
- **Cross-signature agreement is bitwise at every kept position**, on fp32, int8 and fp16 alike — so pick a signature for speed and memory, not for quality. The reason is mechanical: the ShortConv kernel is 3, so a token reads one neighbour on each side, and once there is a single pad to the right, more pads change nothing. The drift from padding is binary, not length-proportional. The one position that legitimately differs between signature lengths is the **last** one (its right neighbour is absent at the end of a short signature and a pad inside a long one); it is always masked out host-side, so gate the kept positions, not the whole tensor.
- **Pad the reference the same way you pad the thing you are judging.** Running the torch reference unpadded while the tflite side pads to a signature reports the padding delta as conversion error (it showed up as 0.9957 per-token cos and −0.0152 nDCG@10 before the reference was fixed). Attributed properly on NanoSciFact (600 docs, 50 queries): torch unpadded 0.8777, torch padded 0.8625, fp32 **0.8625**, fp16 **0.8625**, int8 0.8626 — conversion and quantization are both free and the whole delta is the static shape. And −0.0152 on 50 queries is inside one standard error, so report it as unresolved rather than as a measured quality loss.
- **`wi8fc` int8 = FC dynamic-range + EMBEDDING_LOOKUP channelwise, convs float.** 353.5M params, vocab 64402 (the embedder's is 65536 — a bench script that borrowed the sibling's vocab bound produced out-of-range ids). fp32 1416.5 MB, int8 370.1 MB, fp16 709.8 MB. No `GATHER_ND`, no `LOGICAL_AND`.
- **int8 is the mobile artifact.** iPhone 17 Pro, 6 threads: int8 reproduces desktop outputs **bit-exactly** (cosine 1.000000, max diff 0.0) across a real 32-token query with 23 expansion positions, documents in three scripts, and the same document at two signature lengths — encode_32 30 ms, encode_128 40–64 ms, encode_512 180 ms, peak 1254 MiB. fp16 also passes but peaks at 5786 MiB, and is slower than fp32 on CPU because XNNPACK unpacks fp16 weights at run time.


### NVIDIA-Nemotron-Nano-9B-v2 — the 9B on the Nemotron-H rail, NVIDIA's dt limit instead of transformers' lower clamp, and a template litert-lm 0.18 can parse

`nemotron_h_work/convert_nemotron_nano_9b.sh` converts [nvidia/NVIDIA-Nemotron-Nano-9B-v2](https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-9B-v2) at commit `6533e8de` — a 56-layer three-kind hybrid, 27 Mamba2 + 25 MLP + 4 attention layers, which NVIDIA compressed and distilled from Nemotron-Nano-12B-v2 — into the two files of [litert-community/NVIDIA-Nemotron-Nano-9B-v2](https://huggingface.co/litert-community/NVIDIA-Nemotron-Nano-9B-v2). Environments: the export and quantization python had torch 2.12.1, transformers 5.14.1, ai-edge-quantizer 0.8.0, litert-converter 0.3.0 and ai-edge-litert 2.1.6, with litert-torch from commit 115a136 plus the two patches on `PYTHONPATH`; litert-lm 0.18.0 packs the metadata and runs the files.

```bash
EXPORT_PY=/path/to/python-with-transformers-5.14.1-and-ai-edge-quantizer-0.8.0 \
RUN_PY=/path/to/python-with-litert-lm-0.18.0 \
LITERT_LM=/path/to/litert-lm \
  bash nemotron_h_work/convert_nemotron_nano_9b.sh out_nemotron_nano_9b
```

| file | section | bytes | sha256 |
|---|---|---|---|
| `NVIDIA-Nemotron-Nano-9B-v2_int8.litertlm` (9,072,259,712 B) | LlmMetadataProto | 4,043 | `d877751e12c04c36a8dd74c7b61d454945bf5f2f4fc0c70c0962f78bdfa425b9` |
|  | ExecutorMetadataProto | 4,181 | `a59a735f28ac13a15c6d1a7c49d4754190ae5ee1cd486a3f000898a18b2c8d60` |
|  | HF_Tokenizer_Zlib | 2,806,525 | `4265cd3c805431c7c95219f59c51ea138a9eebe16414abb61f8e654a15945a94` |
|  | TFLiteModel | 9,069,392,512 | `9b9deafb4d20bbec7e73b2204a0337f86338266e2edffab7460e1b28cb4efa97` |
| `NVIDIA-Nemotron-Nano-9B-v2_int4.litertlm` (5,592,589,856 B) | LlmMetadataProto | 4,043 | `d877751e12c04c36a8dd74c7b61d454945bf5f2f4fc0c70c0962f78bdfa425b9` |
|  | ExecutorMetadataProto | 4,181 | `a59a735f28ac13a15c6d1a7c49d4754190ae5ee1cd486a3f000898a18b2c8d60` |
|  | HF_Tokenizer_Zlib | 2,806,525 | `4265cd3c805431c7c95219f59c51ea138a9eebe16414abb61f8e654a15945a94` |
|  | TFLiteModel | 5,589,722,656 | `ea67042adf37666e37d0df331b99858ec6c8187e70e7abde5b2e432048d64cf4` |

The conversion run that produced these files (the same steps as this script, with the local paths and memory gates of a shared machine) rebuilt every one of these sections byte for byte from the pinned checkpoint on 2026-10-15. Run as published on 2026-10-16 (one command, the environment of its header, 37 minutes on a 128 GB Mac including the Hub fetch, peak 40.8 GB), this script rebuilt every one of them byte for byte as well. The file sha256 differs from run to run (the packer writes a new uuid and creation timestamp), so compare sections.

- **NVIDIA's dt limit, not transformers' lower clamp.** transformers 5.14.1 clamps the Mamba2 `softplus(dt + dt_bias)` from below at `time_step_min` (0.001); NVIDIA's modeling code clamps to `time_step_limit`, default (0.0, inf) = no clamp. In this checkpoint 339 of 3,456 heads have softplus(dt_bias) below 0.001; with the same bf16 weights the two forms agree on 94.3–94.8 % of the teacher-forced argmax at about 1k tokens. `nemotron_h_vendor_dt.patch` (on top of `nemotron_h_litert_torch.patch`) makes the folded scan follow NVIDIA's limit: the 27 MAXIMUM ops per signature that the lower clamp adds are gone, and the patched mixer matches NVIDIA's torch forward within 1.4e-7 relative. On GSM8K both forms scored 93 on the Mac GPU with the same questions wrong; the files use NVIDIA's.
- **The lightweight export path for a 9B.** `experimental_lightweight_conversion` with `RESOURCE_CONSTANT_NUMEL_THRESHOLD=100000000`, six prefill signatures (1024, 256, 64, 16, 4, 1) + decode over a 4096-token KV cache; `const_ops.py` / `fc_dtypes.py`: every FULLY_CONNECTED weight a constant, only the scan's chunk masks have all-constant inputs. Export peak 40.8 GB, each quantize step 35.5 GB on a 128 GB Mac. Run one export at a time.
- **int8 = `wi8fc` (dynamic int8 on every FULLY_CONNECTED + int8 embedding; the convolutions and the selective scan stay float); int4 = `wi4b32_wi8` (int4 blockwise-32 FCs + int8 lm_head and embedding) through `quantize_clamped.py`, which floors blockwise scales at 2^-14 — 66,388 floored here; every scale normal after.**
- **fp32 GPU activations are declared** (`scripts/set_activation_type.py --type fp32`): with the runtime's fp16 default the int8 file answered `<unk>` repeated on all eight check questions; the fp16 file was 1.5 % faster.
- **The template.** NVIDIA's template does not parse on litert-lm 0.18.0 (`.split('</think>')[1]` becomes `| split('</think>')[1]`), and its trailing-assistant branch renders an assistant turn differently by position. `make_template.py` rebuilds it as `nemotron_nano_dual.jinja` with 11 asserted edits: content parts through `format_content_dual_tool.jinja`, filters for the Python string methods, the trailing-assistant branch removed; its renders equal transformers' render of NVIDIA's template in 13 conversation shapes × 3 thinking states. Reasoning stays NVIDIA's switch: `/no_think` in the system or a user message turns it off, otherwise the prompt ends with `<think>` + newline.
- **Metadata.** No start token: the prompt stream starts at `<SPECIAL_10>` and the engine's prefill count equals transformers'; stop ids 2, 11, 12; the thought channel `<think>\n` … `\n</think>`; 62 state buffers = 4 key + 4 value caches + one conv and one SSM state per Mamba2 layer; the tokenizer section is the repo's tokenizer.json byte for byte.
- Measured (prefill 1024 / decode 256 / context 2048, litert-lm 0.18.0, Mac M4 Max): GPU (WebGPU) int8 480.18 / 41.02 tok/s, int4 478.27 / 45.52; CPU int8 127.93 / 12.39, int4 56.87 / 13.99; every signature fully delegated on the Mac GPU. GSM8K (questions 0–99, thinking off, greedy, at most 1024 output tokens): int8 93, int4 92 on the Mac GPU; transformers on MPS scored 83 (bf16) and 84 (fp32), the whole gap on the 16 prompts longer than 128 tokens, the chunk size of transformers' torch scan; those 16 re-run in fp32 as one chunk answered 15.
- No phone row: the Galaxy S26 was not used for this file after a Mamba2-hybrid sibling dropped it off USB during its GPU warm-up, and the Pixel 8a int4 GPU try was stopped by the harness's RAM guard. One Engine per conversation: the 27 Mamba2 layers keep a recurrent state (LiteRT-LM #3165).

## Nemotron-3-Embed-1B (a bidirectional embedder wearing a decoder's clothes)

`nemotron_work/convert_nemotron3_embed.py` converts [nvidia/Nemotron-3-Embed-1B-BF16](https://huggingface.co/nvidia/Nemotron-3-Embed-1B-BF16) — NVIDIA's multilingual retrieval/RAG embedding model (34 languages, 2048-d, pruned+distilled from Ministral-3-3B) — to plain LiteRT `.tflite`. Like the LFM2.5 encoders this is NOT an `export_hf` run: there is no KV cache in the embedding path, so the HF eager model is traced directly with `litert_torch` multi-signature convert. Signatures `embed_{64,128,256,512}` → `output_0` `[1,2048]`, **already mean-pooled and L2-normalized** — one call in, one finished embedding out.

```bash
cd nemotron_work
python convert_nemotron3_embed.py nvidia/Nemotron-3-Embed-1B-BF16 out_nemotron3_embed_1b  # fp32 + wi8fc int8 + fp16
python verify_nemotron3_embed.py out_nemotron3_embed_1b                                   # card oracle + STS17 + retrieval + mechanics
python bench_nemotron3_embed.py  out_nemotron3_embed_1b
```

Env: litert-torch ≥ 0.9.2, transformers 5.14.1, torch 2.12. The non-obvious parts:

- **`architectures: ["Ministral3Model"]` is the stock causal-LM class — `config.is_causal: false` is what makes it bidirectional.** transformers honours it: `create_causal_mask()` falls back to `create_bidirectional_mask()` and the flag reaches `sdpa_attention_forward` as an explicit `is_causal=False`. `Ministral3Attention.__init__` hardcodes `self.is_causal = True` and is **not** the switch — flipping it changes the output by exactly 0.0. Verify the real thing instead: change the LAST token and check that position 0 moves (it moves by 5.96 here; under causal attention it could not move at all).
- **The bidirectional mask is skipped entirely when nothing is padded.** `create_bidirectional_mask` → `sdpa_mask(allow_is_bidirectional_skip=True)` returns `None` for an all-ones mask. Harmless eagerly, *not* harmless under `torch.export`: the branch specializes at trace time, so tracing with an unpadded sample bakes a graph that ignores `attention_mask` and lets pad tokens into both the attention and the mean. Always trace with right-padded samples, and gate on **pad-content invariance** — scribble random ids over the pad region and require the embedding to be bitwise unchanged (0.0 here, on every variant).
- **Pooling and normalization are not in the HF module.** `config.pooling: "avg"` and `modules.json`'s `2_Normalize` are sentence-transformers metadata that `Ministral3Model` ignores; mean-pool (with `include_prompt: true`) and L2-normalize have to be added to the traced module or you ship raw hidden states.
- **The `query: ` / `passage: ` prefix is mandatory — including for symmetric similarity.** This is an E5-style model; raw text is out of distribution. On STS17 (100 pairs, int8), no prefix vs `query: ` on both sides: en-en 0.636→**0.865**, es-en 0.064→**0.803**, en-ar −0.029→**0.724**. Cross-lingual collapses to noise without it, which is exactly the capability the model is bought for — and the upstream `config_sentence_transformers.json` sets `default_prompt_name: null`, so a plain `SentenceTransformer.encode()` lands in the bad column. Use `encode_query` / `encode_document` or prefix by hand.
- **Build the attention bias yourself; don't let `sdpa_mask` do it.** Feeding a 2D mask routes through transformers' index-based mask construction, which lowers to `GATHER_ND` -> `LOGICAL_AND` -> `SELECT_V2`. Read the constants out of an actual export and it is an identity gather at batch 1 (index grid `[[0,0],[0,1],...]`), an all-True q-side mask, and `where(mask, 0.0, -inf)` — i.e. `bias[0,0,q,k] = 0 if valid else -inf`, with no q dependence. `_preprocess_mask_arguments` early-exits on any 4D mask, so emitting a `[1,1,1,S]` bias directly removes `GATHER_ND` and `LOGICAL_AND` and never materializes the `[1,1,S,S]` mask. Output is **bitwise identical** either way, and it is speed-neutral (an apparent -8% reversed when the A/B was run in the other order — thermal drift, not the change).
- **The GPU wall here is not `GATHER_ND`, it is 5-D tensors.** Compiling for Metal, 256 ops are rejected with `RESHAPE: Tensor dimensions must be less than 5` — the `[1,8,1,S,128] -> [1,8,3,S,128]` expansion inside GQA `repeat_kv` — plus 4 unsupported `BROADCAST_TO`. Only ~62 of ~1100 ops offload and compilation then fails, with or without the `GATHER_ND`. A GPU lane needs `repeat_kv` re-expressed in 4-D (or SDPA's `enable_gqa`); the mask cleanup above is hygiene, not the fix. Checked on litert-converter 0.3.0 and 0.3.1 — identical op histogram, identical rejection profile, bitwise identical weights, so this is a property of the graph rather than of one converter build.
- **`wi8fc` int8 = FC dynamic-range + EMBEDDING_LOOKUP channelwise.** The embedding lowers to `EMBEDDING_LOOKUP` (not GATHER), so the recipe reaches the 131072×2048 vocab table — 24% of all parameters. 1140.9M params → fp32 4567 MB (a fine intermediate; the writer uses buffer offsets, so the 2 GiB flatbuffer figure is not a wall), int8 1167 MB, fp16 2286 MB.
- **Quality is quantization-flat.** Against the base card's own documented similarity matrix all variants reproduce it within the card's own bf16-vs-fp32 gap (max abs 0.0041 for fp32/fp16, 0.0075 for int8) and rank the right document first 4/4. STS17 over 11 language pairs: fp32 mean 0.7292, int8 mean 0.7296 — task-lossless. (`en-tr` scores ~0.04 for every variant: Turkish is not one of the 34 languages NVIDIA evaluated. It is a useful control that the gate discriminates.)
- **Signature count is a memory decision, not a quality one.** Every build returns bitwise identical vectors, and the same text routed through `embed_64/128/256/512` is bitwise identical too. But the interpreter delegates *every* signature subgraph at creation, so peak RAM scales with the signatures **present in the file**, not the ones you call — measured on an iPhone 17 Pro at ~846 MiB per signature: 1 sig → 1028 MiB peak, 2 sigs → 1901 MiB, 4 sigs → 3593 MiB, while the file itself only grows 1148 → 1167 MB. Build `--seqs 512,128` for phones; the 4-signature file wants desktop or the `increased-memory-limit` entitlement.
- **int8 is the mobile artifact.** On an iPhone 17 Pro the int8 builds reproduce desktop outputs **bit-exactly** (cosine 1.000000, max diff 0.0) across five scripts and both signatures; embed_128 103 ms, embed_512 ~400 ms at 6 threads — faster than an M4 Max Mac on the identical file (150 / 561 ms), which does not speed up with more threads. The fp16 file is **desktop-only**: on device it is killed by the OS during interpreter creation, before the first inference.
- **Retrieval sanity** (SciFact-derived, 50 queries / 600 docs, subsampled so not BEIR-comparable): fp32 nDCG@10 0.8683, int8 0.8628, fp16 0.8683. At 50 queries this gate cannot resolve deltas below a couple of percent — Gate B (1100 graded pairs) is the higher-resolution evidence.

There is no `.litertlm` here on purpose: `litert-lm-builder` will happily pack one (`embedding_metadata` + `tflite_model --model_type text_encoder`, and it peeks back clean), but the released runtime has no embedding executor to run it — `Engine()` on such a file aborts. Plain `.tflite` is the working artifact today.

## EmbeddingEngine bundles (`.litertlm`, litert-lm >= 0.17.0) — granite-embedding-311m-r2 / LFM2.5-Embedding-350M / Nemotron-3-Embed-1B

`embedding_engine_work/` re-exports the four pooled embedders above into the bundle the LiteRT-LM 0.17.0 `EmbeddingEngine` loads (embedder lookup graph + `encoder_<S>` graphs with pooling in-graph, plus tokenizer and `EmbeddingMetadata`). Same weights and quantization as the `.tflite` files; vectors match them at cosine 1.000000. Read `embedding_engine_work/README.md` for the contract and the two traps, then per model:

```
python embedding_engine_work/convert_granite_embedding_r2_engine.py ibm-granite/granite-embedding-311m-multilingual-r2 out_granite_engine
BUILDER_PY=<venv with litert-lm-builder>=0.17.0>/bin/python embedding_engine_work/pack_granite_embedding_r2.sh out_granite_engine out_granite_engine/embedder_wi8.tflite out_granite_engine/encoder_wi8fc.tflite <tokenizer.json> out_granite_engine/granite-embedding-311m-r2_wi8fc.litertlm
```
(`convert_lfm25_embedding_engine.py` / `pack_lfm25_embedding.sh`, and `convert_nemotron3_embed_engine.py` / `pack_nemotron3_embed.sh` take the same arguments; a pack for a model whose tokenizer post-processor adds both bos and eos declares both, so the engine's `insert_special_tokens` reproduces it.) Use with `litert_lm.embedding_engine.EmbeddingEngine` (Python) or `com.google.ai.edge.litertlm.EmbeddingEngine` (Kotlin, litertlm-android 0.17.0); keep `insert_special_tokens` at its default.

## granite-embedding-*-multilingual-r2 (ModernBERT encoders → plain .tflite)

`granite_embed_work/convert_granite_embedding_r2.py` converts IBM's multilingual ModernBERT bi-encoders to plain LiteRT `.tflite` for retrieval / RAG / semantic search on CPU. Same encoder lane as the LFM2.5 and Nemotron embedders — no KV cache, so the HF eager model is traced directly with `litert_torch` multi-signature convert. Signatures `embed_{64,128,256,512}` → `output_0` `[1,768]`, **already CLS-pooled and L2-normalized**. Published: [litert-community/granite-embedding-311m-multilingual-r2](https://huggingface.co/litert-community/granite-embedding-311m-multilingual-r2) (int8 336 MB + fp16 629 MB).

```bash
cd granite_embed_work
python convert_granite_embedding_r2.py ibm-granite/granite-embedding-311m-multilingual-r2 out_granite_311m
python verify_granite_embedding_r2.py out_granite_311m --prefix-ab
```

Env: litert-torch ≥ 0.9.2, transformers 5.14.1, torch 2.12. The non-obvious parts:

- **Read the pooling config; do not assume.** This model is **CLS pooling** (`1_Pooling/config.json` → `pooling_mode_cls_token: true`, so `hidden[:, 0]`) with **no prompt prefixes**. The Nemotron embedder in this same repo is mean pooling with a *mandatory* `query: `/`passage: ` prefix. Two embedders, opposite contracts — carrying one recipe to the other silently produces wrong vectors.
- **ModernBERT alternates local and global attention.** 22 layers = 8 `full_attention` + 14 `sliding_attention` (every 3rd is global) with a 64-wide half-window, and **each layer type has its own rope frequency set** (global θ 150000 / local 160000) — so assert *both* `inv_freq` buffers are non-zero after load, not just one.
- **Build both masks yourself.** `ModernBertModel.forward` accepts `attention_mask` as a **dict** (`{"full_attention": ..., "sliding_attention": ...}`) and uses it verbatim, which bypasses transformers' `sdpa_mask` entirely: the resulting graph has **zero int64 tensors and no GATHER_ND**. Semantics verified bitwise against transformers at S=128 and S=512 — `full` = pad mask broadcast over queries, `sliding` = pad mask AND `|q − k| ≤ 64`.
- ⚠️ **A sliding window plus right padding creates fully-masked query rows → NaN.** Once a pad position is further than the window from every real token (`q ≥ n_valid + 64`), the sliding row is entirely False and softmax runs over all `-inf`. This is the common case, not an edge case: a 6-token query padded into `embed_512` leaves **442** such rows. It hides well — **eager PyTorch SDPA absorbs it** (so a torch-vs-torch comparison is silent), the lowered SOFTMAX emits **NaN** in fp32/fp16, and **int8 masks it completely** behind a healthy-looking `cos 0.999234`. *An int8 variant scoring better than fp32 is the tell.* Fix: OR the diagonal into the sliding mask (`(valid & band) | (q == k)`) — output-neutral, because for a valid query `k=q` is already allowed and pad rows never reach a valid row (pad **keys** stay masked everywhere). Verified bitwise identical at S=64/128/256/512 with the NaN gone. Gate it with an `isfinite` assert per variant per signature, plus a short-text-in-longest-signature case.
- **`ai_edge_quantizer.export_model` refuses to overwrite**, so a second run leaves *stale* int8/fp16 files beside a freshly exported fp32 — which then get "verified". Delete prior outputs before quantizing.
- **Quality**: reproduces the base card's documented 3×3 cross-lingual matrix to **every published digit** (fp32/fp16 max abs 0.0000, int8 0.0039). STS17 over 11 language pairs: fp32 0.7363 / int8 0.7327 / fp16 0.7364. Retrieval (SciFact-derived, subsampled so not BEIR-comparable) nDCG@10 0.8400 / 0.8360 / 0.8400 with identical recall@5 and hit@1. Cross-signature outputs agree **bitwise**.
- **Prefix contract is per-model — measure it.** An unofficial `query: ` prefix *raises* en-en STS17 from 0.783 to 0.826, which looks like grounds to override the vendor's empty prompts. On retrieval it reverses: nDCG@10 0.836 → 0.832/0.830 and recall@5 0.900 → 0.860. It is a symmetric-similarity artifact; bare text stays the contract. `--prefix-ab` runs this control.
- **int8 is the device artifact.** On an iPhone 17 Pro both files reproduce desktop outputs **bit-exactly** (cosine 1.000000, max diff 0.0), but int8 peaks at **518 MiB** for all four signatures (embed_128 58–120 ms, embed_512 331–354 ms) against fp16's 3700 MiB and up to 6× the latency — XNNPACK expands fp16 weights to fp32 while packing each signature subgraph. At 311M the per-signature packing cost is ~118 MiB, so unlike the 1.14B Nemotron build there is no need to trim signatures or ask for a raised memory limit.

## bekko-embedding-v1-a25m (mmBERT Japanese/multilingual embedder — when 80% of the model is the embedding table)

`bekko_work/convert_bekko_embed.py` converts [hotchpotch/bekko-embedding-v1-a25m](https://huggingface.co/hotchpotch/bekko-embedding-v1-a25m) — an ultra-compact multilingual retrieval model (100+ languages, strong Japanese, 384-d, Matryoshka 256/128/64) — to plain LiteRT `.tflite`. Same ModernBERT encoder lane as the granite-embedding-r2 section above (hand-built `{layer_type: bias}` mask dict, sliding-window diagonal fix, isfinite gates — read that section first; it all transfers). Signatures `embed_{64,128,256,512}` → `output_0` `[1,384]`, **already mean-pooled and L2-normalized**, **no prefixes** (the README FAQ is explicit — a third pooling/prefix contract in three embedders shipped from this repo: read `1_Pooling/config.json` per model, never carry the last model's recipe).

```bash
cd bekko_work
python convert_bekko_embed.py hotchpotch/bekko-embedding-v1-a25m out_bekko_a25m  # fp32 + wi8fc + embt8 + fp16
python verify_bekko_embed.py out_bekko_a25m                # card oracle + JSTS/Matryoshka/STS17 + JSQuAD + mechanics
python verify_bekko_embed.py out_bekko_a25m --gates E --report verify_report_e.json  # cross-variant retrieval
python bench_bekko_embed.py  out_bekko_a25m
```

Env: litert-torch ≥ 0.9.2, transformers 5.14.1, torch 2.12. What is new relative to the granite-r2 section:

- **The model is mostly embedding table, so the quantization question inverts.** 123.2M params, 13L/384h mmBERT (pruned from jhu-clsp/mmBERT-small), Gemma-style vocab 256000: the 256000×384 table is 80% of all parameters and only 25M are "active". The upstream repo's own default ONNX/OpenVINO artifacts quantize ONLY that table to row-wise int8 and call transformer-weight int8 "experimental, not recommended". `embt8` reproduces exactly that recipe (embedding int8 channelwise, float compute: 214 MB, per-vector cos ≥ 0.9999 to torch); `wi8fc` additionally puts dynamic-range int8 on the FCs (142 MB).
- **The vendor's "transformer int8 not recommended" verdict does not transfer to DRQ int8 — but you must gate on tasks, not vectors.** wi8fc's per-vector smoke cos on random ids is 0.985–0.992, which reads as damage. On real tasks it is lossless: JSTS (JGLUE v1.3, 300 pairs) Spearman 0.8179 vs torch 0.8175, with Matryoshka truncation to 256/128/64 intact; JSQuAD retrieval (800 paragraphs, 150 questions) at or above the reference; STS17 spot-checks within 0.0025. Per-vector cosine and task score are different quantities, and at 4× less RAM than the vendor-analog recipe the task numbers are the ones that matter.
- **Gate the RAG deployment shape explicitly: index built with upstream weights, queried with the quantized artifact.** Cross-variant retrieval (documents encoded by PyTorch fp32, queries by wi8fc) scores nDCG@10 0.9186 vs the 0.9179 all-PyTorch control — the int8 embedding space is compatible with a server-built index. A variant can pass same-variant gates while sitting in a slightly rotated space; this is the gate that would catch it.
- **fp16 is strictly dominated here — build it, measure it, don't ship it.** XNNPACK expands fp16 weights to fp32 while packing every signature subgraph: 2387 MiB peak for 4 signatures vs embt8's 575 MiB and wi8fc's 225 MiB, at identical task scores (M4 Max, 8 threads, all signatures invoked).
- **Japanese gates**: JGLUE v1.3 (`datasets/{jsts,jsquad}-v1.3` on github.com/yahoojapan/JGLUE — the v1.1 paths in older writeups are gone). The base card's quickstart scores carry a ~0.028 backend residual that the PyTorch fp32 reference itself reproduces, so the oracle is argmax + the documented cross-lingual similarities (which match to 0.003), not bit-distance to the card.
- **Mechanics**: cross-signature outputs bitwise identical on every variant (the 11-token probe in `embed_512` is also the empty-sliding-window NaN shape — finite everywhere, so the diagonal fix holds), pad-content invariance exactly 0.0, zero int64 tensors, no `GATHER_ND`.


## harrier-oss-v1-0.6b (a causal decoder pooled at the last token — the e5-mistral shape, statically)

`harrier_work/convert_harrier_embed.py` converts [microsoft/harrier-oss-v1-0.6b](https://huggingface.co/microsoft/harrier-oss-v1-0.6b) — Microsoft's multilingual (100+ languages) instruction-prompted embedding model, 596M params, 1024-d — to plain LiteRT `.tflite`. Same encoder lane (direct multi-signature trace), but the opposite attention contract to the bidirectional embedders above: the body is a bare `Qwen3Model` and attention stays **causal** — nothing is flipped; the embedding is the LAST non-padding token's hidden state, L2-normalized, both in-graph. Signatures `embed_{64,128,256,512}` → `output_0` `[1,1024]`.

```bash
cd harrier_work
python convert_harrier_embed.py microsoft/harrier-oss-v1-0.6b out_harrier_0p6b   # fp32 + wi8fc + fp16
python -u verify_harrier_embed.py out_harrier_0p6b --prefix-ab                   # README example + JSTS/STS17 + JSQuAD + mechanics
python -u verify_harrier_embed.py out_harrier_0p6b --gates E --report verify_report_e.json
python bench_harrier_embed.py  out_harrier_0p6b
```

Env: litert-torch ≥ 0.9.2, transformers 5.14.1, torch 2.12. The non-obvious parts:

- **`Qwen3Model` accepts the same `{layer_type: bias}` mask dict as ModernBERT** (`causal_mask_mapping := attention_mask` walrus, key `"full_attention"`), so the hand-built causal+padding bias `[1,1,S,S]` (`allowed(q,k) = (k <= q) & valid[k]`) bypasses `sdpa_mask` entirely — no GATHER of any kind in the graph. No NaN guard is needed in the causal direction: every query row can reach position 0.
- **A last-token pool must rebind to the mask, not to position S−1.** With right padding the pooled position is `sum(mask) − 1`. The static-graph form — `onehot = (arange(S) == sum(mask) − 1); pooled = sum(h * onehot)` — lowers to a single `ONE_HOT` + `SUM`, no dynamic gather. Gate both directions: scribbling pad ids must move nothing (0.0 measured), dropping the last REAL token must move the output a lot (0.13 measured). A graph that unconditionally reads position S−1 passes every all-valid smoke test and silently pools pad garbage in production.
- **The tokenizer appends `<|endoftext|>` itself** (a `TemplateProcessing` post-processor: `tok("summit define")` → `[1242, 1763, 6979, 151643]`), and that appended token is what gets pooled — the e5-mistral EOS convention, delivered by the tokenizer while the vendor's own usage snippet never mentions it. Tokenize with the repo's `tokenizer.json` and the contract is automatic; a hand-rolled BPE without the post-processor pools the last *text* token instead, which is a different (untrained) contract. It is also the same id as PAD — the mask is what distinguishes them.
- **The instruction prefix is mandatory and the degradation is now a number.** Queries carry `Instruct: {task}\nQuery: ` (documents raw; prompts ship in `config_sentence_transformers.json`). Measured with `--prefix-ab` on both torch and the int8 artifact: dropping it costs 0.018 JSTS Spearman and **0.028 JSQuAD nDCG@10**. Note the contrast with granite-r2 above, where an *uninvited* prefix helped STS and hurt retrieval — the prefix contract is per-model in sign as well as size; measure it, never assume it.
- **Quality is quantization-flat, and fp32/fp16 are digit-exact.** fp32 and fp16 tflite reproduce the torch reference to every printed digit on JSTS (0.8553), STS17 (en-en 0.8893 / es-en 0.8345 / ko-ko 0.8874 / en-ar 0.8517) and JSQuAD (nDCG@10 0.9363); wi8fc lands within 0.004 everywhere (JSTS 0.8559, nDCG@10 0.9360), and querying a torch-built index with int8 embeddings scores at the all-torch control (0.9389 vs 0.9363) — the RAG shape of index-on-server, query-on-device holds.
- **wi8fc = FC int8 DRQ + EMBEDDING_LOOKUP int8 channelwise.** 596M params → fp32 2390 MB (fine intermediate), int8 626 MB, fp16 1199 MB. Per-vector smoke cos on random ids reads 0.989–0.997 for int8 — the task gates above are the ones that matter.
- **Signature count is NOT always the RAM lever.** Unlike the 1.14B Nemotron build (~846 MiB/signature), the int8 peak here is signature-independent: 2401 MiB with 4 signatures vs 2351 MiB with 2 (Mac, load+invoke). Ship the flexible 4-signature file. fp16 is still the RAM disaster (11.9 GiB, XNNPACK expands fp16→fp32 per subgraph) — desktop only.


## voyage-4-nano (a "causal" config that is actually a bidirectional embedder)

`voyage_work/convert_voyage_embed.py` converts [voyageai/voyage-4-nano](https://huggingface.co/voyageai/voyage-4-nano) — Voyage AI's multilingual retrieval embedder with a shared embedding space across the voyage-4 series (a device-side nano can query an index built by the larger cloud models), 347M params, 2048-d, MRL-truncatable to 1024/512/256 — to plain LiteRT `.tflite`. Signatures `embed_{64,128,256,512}` → `output_0` `[1,2048]`, mean pool + 1024→2048 projection + L2 normalize in-graph.

```bash
cd voyage_work
python convert_voyage_embed.py voyageai/voyage-4-nano out_voyage_nano   # fp32 + wi8fc + fp16
python -u verify_voyage_embed.py out_voyage_nano --prefix-ab            # README example + JSTS/STS17 + JSQuAD + mechanics
python -u verify_voyage_embed.py out_voyage_nano --gates E --report verify_report_e.json
python bench_voyage_embed.py  out_voyage_nano
```

Env: litert-torch ≥ 0.9.2, transformers 5.14.1, torch 2.12. The non-obvious parts:

- **`architectures: ["Qwen3ForCausalLM"]` is a lie.** `auto_map.AutoModel` points at remote-code `Qwen3BidirectionalModel`: a bare `Qwen3Model` body with every `self_attn.is_causal` flipped False plus a per-token bias-free `linear` 1024→2048 (`config.num_labels`) applied to `last_hidden_state` BEFORE pooling. Convert the config's nominal architecture and you ship a causal model that was never trained that way.
- **The remote code targets transformers 4.51 and breaks twice under 5.14.** It sets no `config_class` (AutoModel.register crashes with `AttributeError: 'NoneType' object has no attribute '__name__'`) and calls `create_causal_mask(input_embeds=…, cache_position=…)` — both renamed/removed in 5.x. The converter imports a vendored copy of the modeling file (shipped next to the script), sets `config_class = Qwen3Config`, and shims the mask call. `trust_remote_code` is never used.
- **The export mask bypasses the vmap path entirely.** The remote forward builds `create_causal_mask(..., or_mask_function=bidirectional)` — vmap machinery that specializes on `attention_mask=None` and emits rank-5 intermediates at trace time. Its semantics reduce to `(causal | valid[k]) & valid[k] = valid[k]`, a key-only gate — so the hand-built `[1,1,S,S]` bias into `Qwen3Model`'s `{"full_attention": bias}` dict is exactly equivalent (gated: cos ≥ 0.9999995 vs the remote path, padded and unpadded) and trace-safe. Full attention + ≥1 valid key ⇒ no empty rows ⇒ no NaN guard needed.
- **Bidirectionality is gated, not assumed** — the inverse of the harrier gate above: editing the LAST valid token must move position 0's hidden state (7.9e-02 measured; a causal graph gives exactly 0).
- **Prompts ride on BOTH sides** (`config_sentence_transformers.json`): queries `"Represent the query for retrieving supporting documents: "`, documents `"Represent the document for retrieval: "`. Measured with `--prefix-ab`: on JSQuAD the query prompt is score-neutral (+0.005 nDCG@10 *without* it, torch included) — opposite sign to harrier's +0.028-with. Third data point that the prefix contract is per-model in sign; measure, never assume.
- **Quality is quantization-flat — consistent with the vendor's QAT claim.** fp32/fp16 reproduce torch to every printed digit (JSTS 0.8406, JSQuAD nDCG@10 0.9292); wi8fc lands within 0.008 everywhere and half the deltas are positive. The shared-space deployment shape holds under int8: torch-built index × wi8fc queries = 0.9283 vs 0.9292 control, and at MRL-256 (truncate + renormalize) 0.8989 vs 0.8990.
- **wi8fc = FC int8 DRQ + EMBEDDING_LOOKUP int8 channelwise.** 347M params → fp32 1389 MB, int8 364 MB (1.15 GiB peak RSS, phone-viable), fp16 696 MB but **7.1 GiB peak RSS** (XNNPACK fp16→fp32 per-signature expansion) — desktop only.

## mLateOn (multilingual ModernBERT ColBERT — late interaction on device)

`mlateon_work/convert_mlateon_colbert.py` converts [lightonai/mLateOn](https://huggingface.co/lightonai/mLateOn) — LightOn's multilingual late-interaction retriever (mmBERT-base, 307M params, 128-d per token, MaxSim), the strongest long-document retriever in its class and trained on nine languages yet generalizing to unseen scripts — to plain LiteRT `.tflite`. Signatures `encode_{32,128,256,512}` → `output_0` `[1,S,128]`, per-token L2-normalized; MaxSim scoring is host-side.

```bash
cd mlateon_work
python convert_mlateon_colbert.py lightonai/mLateOn out_mlateon   # fp32 + wi8fc + fp16
python -u verify_mlateon_colbert.py out_mlateon --gates A,D,B,J   # card oracle + mechanics + NanoSciFact + JSQuAD
python -u verify_mlateon_colbert.py out_mlateon --gates E --bank-cache --report verify_report_e.json
python bench_mlateon_colbert.py out_mlateon
```

Env: litert-torch ≥ 0.9.2, transformers 5.14.1, torch 2.12 (+ pyarrow for the NanoSciFact loader). The non-obvious parts:

- **The pad id is 4 (`<mask>`), not `config.json`'s 0.** PyLate sets `pad_token_id = mask_token_id` whenever the tokenizer has a mask token; the vendor's own `onnx_config.json` says 4. With padding masked out of attention the graph output on real tokens is provably pad-id-independent here (gated 0.0) — but a host should still pad with 4 to match the reference stack.
- **The PyLate head is three all-linear Dense modules and folds exactly.** `use_residual: true` means a PARALLEL bias-free `residual` Linear (`out = linear(x) + residual(x)`), so 768→1536(res) → 1536→768(res) → 768→128 collapses to one 128×768 matrix `W3 @ (W2l+W2r) @ (W1l+W1r)` — folded in float64, gated at 2.4e-6 against the module stack. One FC in the graph instead of five.
- **The sliding-window NaN wall, and why int8 hides it.** ModernBERT alternates full and ±64-window attention (`config.sliding_window` = `local_attention // 2` = 64, inclusive). Under right padding, a pad query row ≥64 past the last valid token has an empty window → all-masked row → SDPA NaN in fp32. There is no pooling to poison — but the NaN sits in the `[1,S,128]` output, and int8 kernels can round it into finite-looking garbage, so **assert finiteness on the fp32 artifact** (the variant that cannot fake it). The hand bias allows the diagonal (`valid[k] | q==k`) — a no-op for valid rows that keeps pad rows finite; the host drops them anyway.
- **No query expansion, unlike classic ColBERT.** `do_query_expansion: false` ⇒ queries are natural-length with `[Q] ` (id 256000) inserted at position 1 after `<bos>`; documents get `[D] ` (256001); `skiplist_words` is empty. Every contract knob differs from LFM2.5-ColBERT above (pad id, expansion, pad-id load-bearing-ness, skiplist) — same architecture class, opposite host contracts, each read out of the shipped config + PyLate source, never assumed.
- **The base card publishes exact MaxSim scores — use them as the oracle.** The converted fp32 reference reproduces `[9.6029, 9.5838, 9.5877, 9.4578]` to all four printed decimals, which anchors tokenizer, prefix insertion, padding, masks, head fold, normalization and MaxSim in a single gate before anything is traced.
- **Quality is quantization-flat, including on an unseen language.** fp32/fp16 match torch to every digit; wi8fc: NanoSciFact nDCG@10 0.8951 vs 0.8882 torch, JSQuAD (Japanese — NOT in the nine training languages) 0.9719 vs 0.9710, and torch-index × int8-queries is identical to control on JSQuAD in every metric. Per-token smoke cos on random ids reads as low as 0.974 for int8 — random ids abuse the 256k SP table; the task gates are the ones that matter.
- **wi8fc = FC int8 DRQ + EMBEDDING_LOOKUP int8 channelwise** (the 256002×768 table is ~64% of params). 307M params → fp32 1233 MB, int8 331 MB (678 MiB peak RSS — the smallest-footprint retriever in this repo), fp16 619 MB but ~6 GiB RSS — desktop only. Encoding runs ~4,600 tok/s at `encode_512` int8 on an M4 Max (late interaction pays at scoring time, not encoding time).


## ettin-reranker-400m-v1 (true cross-encoder reranking on device — and a score that must not be sigmoided)

`ettin_work/convert_ettin_reranker.py` converts [cross-encoder/ettin-reranker-400m-v1](https://huggingface.co/cross-encoder/ettin-reranker-400m-v1) — the Sentence Transformers org's ModernBERT cross-encoder (ettin-400m backbone, 396M params, MSE-distilled from mxbai-rerank-large-v2) — to plain LiteRT `.tflite`. Query and passage go through the model **together**; signatures `score_{128,256,512}` → one raw relevance logit per invoke. The first true cross-encoder in this repo (everything else reranks with bi-encoders or decoder LMs).

```bash
cd ettin_work
python convert_ettin_reranker.py cross-encoder/ettin-reranker-400m-v1 out_ettin   # fp32 + wi8fc + fp16
python -u verify_ettin_reranker.py out_ettin      # card oracle + mechanics + NanoSciFact rerank
python bench_ettin_reranker.py out_ettin
```

Env: litert-torch ≥ 0.9.2, transformers 5.14.1, torch 2.12 (+ pyarrow for the NanoSciFact loader). The non-obvious parts:

- **This is NOT `AutoModelForSequenceClassification`.** It is the sentence-transformers v5 *module-stack* CrossEncoder (`modules.json`): Transformer → Pooling(cls) → Dense 1024→1024 no-bias + **GELU** → LayerNorm → Dense 1024→1 → score. The GELU means the head does NOT fold to one matrix (unlike the all-linear PyLate heads) — implement the stack in-graph as-is. Also note the backbone config's `classifier_pooling: "mean"` is a base-model leftover; the ST module config says `cls`, and the ST path is the vendor path.
- **The score is a raw logit — the activation is part of the model config, not the API.** `CrossEncoder.predict` defaults to sigmoid for single-label models, but this repo pins `activation_fn: torch.nn.Identity` in `config_sentence_transformers.json`, so the published card scores (`[3.6875, 11.6875, 4.75, 9.375]`) are raw. A host that assumes the API default gets card-incomparable scores. Read the config, not the docs.
- **The card's example scores were printed from a bfloat16 run** (the usage snippet passes `dtype: bfloat16`; every value is a multiple of 0.0625). fp32 reproduces them to bf16 tolerance only — max |Δ| 0.054, ranking exact. Gate rank order + tolerance, not decimals, when a card example was run in bf16.
- **Pairs are the tokenizer's own pair template**: `tokenizer(query, passage, truncation="longest_first")` → `[CLS] q [SEP] d [SEP]`, no token_type_ids, pad 50283.
- **The sliding-window NaN wall reaches the score through CLS pooling.** ModernBERT alternates full and ±64-window attention; under right padding an empty pad row goes NaN, and the next layer's `0·NaN` value mixing carries it into every row — including row 0 that the head pools. The hand mask allows the diagonal (`valid[k] | q==k`); gate = padded score equals unpadded score (2.9e-6 eager, exactly 0 across signatures on the artifact).
- **Quantization on a 28L/1024h deep-narrow body is measurable**: NanoSciFact rerank (50 queries × 20 candidates) — torch/fp32/fp16 all nDCG@10 0.9637 / hit@1 0.920 with fp16 at zero pairwise inversions; wi8fc 0.9563 / 0.900 with 2.8% of pairs inverted, max raw-score drift 0.84. Both ship; the card carries the table. 396M → fp32 1588 MB, int8 413 MB (1469 MiB peak RSS), fp16 797 MB but ~6.4 GiB RSS — desktop only. Mac 16-thread: 89.5 ms per candidate at score_256 int8.

## mxbai-edge-colbert-v0-32m (a 32M ColBERT at 39 MB — and the strongest card oracle yet)

`mxbai_work/convert_mxbai_colbert.py` converts [mixedbread-ai/mxbai-edge-colbert-v0-32m](https://huggingface.co/mixedbread-ai/mxbai-edge-colbert-v0-32m) — Mixedbread's edge ColBERT (ettin-32m ModernBERT backbone, 64-d per token, PyLate head) — to plain LiteRT `.tflite`. Signatures `encode_{48,128,256,512}` → `[1,S,64]`, per-token L2-normalized; MaxSim host-side. 48 mirrors the vendor's query length.

```bash
cd mxbai_work
python convert_mxbai_colbert.py mixedbread-ai/mxbai-edge-colbert-v0-32m out_mxbai   # fp32 + wi8fc + fp16
python -u verify_mxbai_colbert.py out_mxbai       # card oracle + mechanics + NanoSciFact + cross-variant
python bench_mxbai_colbert.py out_mxbai
```

Env: same as ettin above. The non-obvious parts:

- **The pad id is 50284 (`[MASK]`), not `config.json`'s 50283.** Same PyLate rule as mLateOn (pad = mask when the tokenizer has one); the vendor's own `onnx_config.json` says 50284. Pad-id-independence on real tokens is gated (0.0), but hosts should match the reference stack.
- **Lowercase first.** `do_lower_case` is set in `sentence_bert_config.json` and the ST Transformer module lowers the text before tokenizing — neither mLateOn nor ettin does this. Same architecture class, third distinct host contract; every knob read out of the shipped configs + PyLate source, never assumed.
- **The skiplist is live here** (unlike mLateOn's empty one): 32 ASCII punctuation token ids, applied to DOCUMENT vectors host-side after encoding; `[CLS]`/`[D]`/`[SEP]` are kept, queries keep everything. Truncation is `query_length−1`/`document_length−1` = 47/511 *before* the `[Q]`/`[D]` insert at position 1.
- **The card publishes scores AND shapes — use both.** The reimplementation reproduces the MaxSim scores `[11.2081, 11.5308, 11.4104, 11.4756]` to all four printed decimals *and* the printed embedding shapes ((12,64) query, (18,64) first doc) pre-export. At 32M the margins are ~0.1, so Mars-on-top is a genuinely tight rank check — and it pins tokenizer, lowercase, prefixes, pad, skiplist, head fold and normalize in one gate.
- **Smoke and task disagree at 32M — trust the task, but run both.** int8's worst per-token smoke cos on random ids at encode_512 is **−0.27** (a destroyed token vector; the worst smoke of any encoder here), yet NanoSciFact damage is only −0.008 nDCG@10 with recall@5/hit@1 unchanged, and torch-index × int8-queries is within 0.007 of control. The 64-d head amplifies per-token noise on abusive inputs; real text does not hit it.
- **wi8fc = FC int8 DRQ + embedding int8 channelwise**: 32M → fp32 131 MB, int8 **39 MB (144 MiB peak RSS — the smallest retriever footprint in this repo)**, fp16 67 MB but 676 MiB RSS (the per-signature fp16→fp32 expansion law holds even at 32M). ~14,800 tok/s at encode_512 int8 on an M4 Max — int8 and fp16 bench identical, so int8 wins on RSS alone.


## BitCPM-CANN (ternary / 1.58-bit LLMs — int4 blockwise is a lossless container)

`bitcpm_work/` converts openbmb's BitCPM-CANN family (BitNet-b1.58-style ternary QAT LLMs, Apache-2.0). LiteRT has no native 1-bit/ternary type ([feature request #7713](https://github.com/google-ai-edge/LiteRT/issues/7713)), but it doesn't need one to run these losslessly: BitCPM's QAT quantizer produces weights in {-α, 0, +α} per group of 128 input channels (α from absmean), released "pseudo-quantized" — the ternary values materialized in a plain bf16 checkpoint. Min-max symmetric int4 `BLOCKWISE_32` (blocks subdivide the 128-groups along the same axis) maps every weight onto {-7, 0, +7} with **zero rounding decisions**; the only residual is fp16 rounding of the per-block scale (≤4e-4 relative). `verify_ternary.py` checks both properties on the actual checkpoint before you convert. You pay 4 bits/weight instead of the native ~1.6 (≈2.5× their packed GGUF), but 4× under f16, on stock CPU/GPU int4 kernels.

```bash
cd bitcpm_work
./convert_bitcpm.sh                    # openbmb/BitCPM-CANN-1B -> bitcpm-cann-1b_wi4b32_wi8.litertlm (1.05 GB)
```

Same env as MiniCPM (`litert-torch litert-lm "transformers==5.6.2"`). The non-obvious parts:

- **The 1B is a stock `LlamaForCausalLM`** (no remote code, no muP) in the MiniCPM4 tokenizer family — the pipeline reuses `../minicpm_work`'s longrope-static export wrapper, the ChatML metadata pbtext verbatim (stops `[2, 73440]`), and the SP added-tokens fix (raw `tokenizer.model` lacks `<|im_end|>`=73440 → stop never fires; the 1B repo also ships no `added_tokens.json`, so `prep_bitcpm_as_llama.py` synthesizes one from `tokenizer_config.json`). The 0.5B is MiniCPM4-0.5B-arch (muP) — run it through the `prep_minicpm4_as_llama.py` fold first if you want it.
- **Use `--algo minmax`, not OCTAV.** For ternary blocks, min-max is exact by construction (scale = α/7); OCTAV optimizes MSE for continuous distributions and can pick a non-exact grid. This is the reverse of the usual recommendation for non-ternary models.
- **Embeddings + lm_head are NOT ternary** (full bf16, untied) — they take the usual int8 channelwise treatment.
- **Results (GSM8K n=100, greedy, identical prompt/extraction)**: torch bf16 **63** · .litertlm int4-b32 **60** (parity within n=100 noise; openbmb's own harness reports 61.56). The proof the container is lossless: the same data-free `wi4b32_wi8` recipe on the non-ternary MiniCPM5-1B loses ~13 points (61 → 48); on ternary it's free. 8-question gate 7/8. Mac bench (prefill 256/decode 256): CPU 285/57 tok/s, GPU (WebGPU) 1137/54.

## Ternary-Bonsai-1.7B (PrismML ternary Qwen3 — second ternary family, one new wall)

`bonsai_work/` converts PrismML's Ternary-Bonsai LLMs (ternary g128 with FP16 group scales — vendor-documented — on a stock `Qwen3ForCausalLM`; Apache-2.0; "unpacked" fp16 release = ternary values materialized, same pattern as BitCPM-CANN). The same int4-lossless-container argument applies and `verify_ternary.py` confirms it on the real weights (Bonsai preserves ~0.003% salient outliers in fp, so a handful of blocks quantize normally instead of exactly).

```bash
cd bonsai_work
./convert_bonsai.sh                    # prism-ml/Ternary-Bonsai-1.7B-unpacked -> ternary-bonsai-1.7b_wi4b32_wi8.litertlm (1.11 GB)
```

The non-obvious parts:

- **Stock Qwen3**: yarn rope is static at init (no `@dynamic_rope_update` wrapper needed, unlike longrope); BPE tokenizer bundles as `hf_tokenizer` (no SP fix); metadata pbtext is generated from the repo's `chat_template.jinja` (always-empty `<think>` prefill = non-thinking 2507 style → no thought channel), stops `[151645, 151643]`, **no start_token** (Qwen has no BOS).
- **⭐ Verbatim Qwen-style templates break MULTI-TURN on the runtime (minijinja)**: the runtime's minijinja implements `startswith`/`endswith` (and `[::-1]` slicing) but NOT `.strip()/.lstrip()/.split()`, and the engine's incremental assistant-turn render always executes the template's reasoning-rerender branch — so the FIRST message works and the SECOND dies with `unknown method: string has no method named strip`. Single-turn evals (GSM8K, quality gates) structurally cannot catch this. `convert_bonsai.sh` therefore rewrites the assistant-history branch (lines 34–51 of the upstream template) to a plain verbatim emit — faithful for this non-thinking model; multi-turn re-verified on Mac and Android.
- **⭐ Zero-scale wall (new)**: ternary weights are sparse enough that a 32-weight block can be ALL zeros → ai-edge-quantizer min-max blockwise emits **scale = 0** → XNNPACK refuses to prepare (`unsupported scale value (0.000000) ... for INT4 tensor`). This model hits it (14 blocks across 6 tensors); BitCPM-CANN-1B happened not to. `fix_zero_block_scales.py` patches it post-quant: LiteRT stores blockwise scales in **separate FLOAT16 tensors referenced by the BlockwiseQuantization details table** (not `QuantizationParameters.scale`), so the script walks the raw (lazy) flatbuffers API and replaces zero scales with the tensor's min nonzero scale in place — the affected blocks are all-zero, so dequantization is unchanged. Seconds, vs ~12 min for an object-API round-trip.
- **Results (GSM8K n=100, greedy, identical prompt/extraction)**: torch bf16 **77** · .litertlm int4-b32 **74** — the same −3-within-noise parity as BitCPM-CANN-1B, now across two vendors and two architectures. 8-question gate **8/8**. Mac bench (matched back-to-back): GPU 1576/62 tok/s, CPU 222/33.
- **On-device (Pixel 8a, Google AI Edge Gallery 1.0.15, official in-app benchmark, p256/d256×3)**: Ternary-Bonsai-1.7B CPU **86.5 prefill / 16.3 decode tok/s** (TTFT 3.0s), multi-turn chat correct with the patched template. BitCPM-CANN-1B GPU **133.1 / 7.7 tok/s** (GPU init ~50s) — int4-b32 blockwise runs on the mobile GPU. Ternary-Bonsai crashes at GPU engine creation on the same device (SIGSEGV in the runtime's weight loading; BitCPM with the identical recipe works, so it's file-specific — tied embeddings / odd vocab 151669 / qwen3 q-k-norms / yarn are the differing suspects). Note the Gallery accelerator list is fixed at import time: enable GPU in the import dialog, or the benchmark silently runs CPU.
- **The image sibling**: `prism-ml/bonsai-image-ternary-4B-unpacked` (FLUX.2-Klein pipeline) — the diffusion transformer's 100 block linears verify ternary-g128/int4-exact the same way (`verify_ternary.py` accepts a group-size arg and matches `blocks` names), while the text encoder is a stock fp16 Qwen3-4B (only the DiT is ternarized). The full pipeline conversion is the next section.

## Bonsai Image 4B (ternary DiT text-to-image — the full diffusion pipeline)

`bonsai_image_work/` converts the whole FLUX.2-klein pipeline to three fixed-shape tflite graphs plus a torch-free host loop. Published: [litert-community/Bonsai-Image-ternary-4B](https://huggingface.co/litert-community/Bonsai-Image-ternary-4B); sample PR: [litert-samples#244](https://github.com/google-ai-edge/litert-samples/pull/244).

```bash
cd bonsai_image_work                       # checkpoint auto-downloads; BONSAI_SNAPSHOT overrides
python export_dit.py                       # fp64-rope fix + export -> dit_fp32.tflite (14.4 GiB)
python quantize_dit.py                     # int4-b32 ternary linears + int8 rest -> 2.11 GiB
python fix_zero_block_scales.py dit_int4b32.tflite dit_int4b32.tflite   # zero-scale patch (REQUIRED, 15 scales here)
python export_textenc.py && python quant_textenc.py                     # prompt embedder -> 1.68 GiB int4
python export_vae.py                       # -> 0.19 GiB fp32
python generate.py --model-dir . --prompt "a bonsai tree"               # host loop: no torch, no diffusers
```

The non-obvious parts:

- **⭐ Flux2 rope builds its frequency table in float64** → `tfl.pow` on f64 fails to legalize at the very last conversion stage. `export_dit.py` forces f32 (cost: max rel 2.9e-6, verified with REAL position ids — an all-zero-ids check is vacuous since cos=1/sin=0 at position 0 for any dtype).
- **The ternary container claim survives the full chain**: the finished int4-b32 file uses only {-7, 0, +7} in all 100 block linears — zero rounding decisions, measured in the artifact.
- **Zero-scale wall again**: the DiT hits 15 zero scales across 4 tensors (the LLM sibling hit 14/6) — this is the generic sparse-ternary case, not a model quirk. Patch is mandatory before XNNPACK will prepare.
- **Text encoder exports as a prompt embedder**: the pipeline reads hidden layers (9, 18, 27) only → litert-torch prunes the top 9 of 36 layers + LM head automatically (4.02 B → 3.11 B). Recipe note: dynamic-range quantization floors image fidelity regardless of bit width (activation quantization dominates); weight-only (`quantize_weight_only.py`) is the quality lane, and int8 weight-only is the practical variant.
- **The host loop needs three exact formulas**, all verified against an instrumented reference run before any device work: the sigma schedule is Flux2's `compute_empirical_mu` fit (NOT the scheduler config's linspace+shift — that gives different sigmas), the DiT timestep input equals sigma, and the packed-latent→VAE transform is a per-PACKED-channel affine (the VAE BatchNorm running stats) applied BEFORE 2×2 unpatchify.
- **⚠ Runtime integration**: attach the XNNPACK delegate explicitly (with `num_threads`) when using the C API — the reference-kernel fallback is orders of magnitude slower AND numerically wrong on blockwise int4.
- **Measured**: Mac CPU 8 threads ≈ 19 s/image (512×512, 4 steps); iPhone 17 Pro (XNNPACK, 6 threads) ≈ 64 s/image, 2.9 GiB peak, bit-exact vs desktop (51.2 dB PSNR on the final PNG).

### 256×256 pair (same weights; DiT step 3.4× and VAE 4.3× faster on a Galaxy S26)

`bonsai_image_work/size256/` re-traces the DiT at 256 image tokens (16×16 patch grid) and the VAE decoder at a 32×32 latent — nothing else changes: the text encoder, tokenizer and latent constants are shared, and the int4 recipe is `quantize_dit.py` unmodified. Published next to the 512 pair on the same card (`dit_256_int4b32.tflite`, `vae_dec_256_fp32.tflite`, `pipeline_meta.json` → `variants.256`, `generate.py --size 256`).

```bash
cd bonsai_image_work
SIZE=256 WORK=out256 python size256/export_dit_256.py        # fp32 export, rope check with REAL ids -> out256/dit_fp32.tflite
(cd out256 && python ../quantize_dit.py)                     # same recipe as 512 -> dit_int4b32.tflite
python size256/fix_scales.py out256/dit_int4b32.tflite out256/dit_256_int4b32.tflite   # 15 zero scales, as at 512
SIZE=256 WORK=out256 python size256/export_vae_256.py        # -> out256/vae_dec_256_fp32.tflite
SIZE=256 WORK=out256 python size256/verify_dit_256.py        # tflite vs torch fp64 with REAL ids, inputs mapped by position
python size256/reference_torch.py --hub-dir <card dir> --size 256 --out ref256.png --noise-out noise256.bin
BONSAI_INIT_LATENTS=noise256.bin python size256/generate.py --model-dir <card dir> --size 256 --out gen256.png
python size256/compare.py ref256.png gen256.png              # PSNR + Laplacian variance
```

Measured (2026-09-27):

- **Galaxy S26, CPU/XNNPACK 6 threads, `benchmark_model` (the binary behind `litert benchmark --android --cpu`), one call per graph**: DiT 14.6 s → 4.3 s per step, VAE decoder 4.9 s → 1.1 s, text encoder 1.8 s either way; the sum for a 4-step image goes ≈ 65 s → ≈ 20 s. **Peak memory does not move** (4.08 GB → 4.06 GB on the DiT): the file is the same 2.11 GiB, so 256 helps slow phones, not phones that run out of memory. `size256/bench_android.sh` is the runner.
- **Fidelity**: with the same prompt embeds and the same initial noise, the all-tflite 256 image scores **39.2 dB PSNR** against the torch pipeline (Laplacian variance 873 vs 865); the random-input DiT check gives cosine 0.9986 (the shipped 512 file: 0.9993 under the same check). The pipeline's sigma schedule and `flowmatch_sigmas(steps, tokens)` agree to 6e-8 at 256 tokens — `mu` is a function of the image-token count, so a host port must recompute it for the new size.
- **Trap**: at 256×256 `img_ids` and `txt_ids` are both (256, 4). A host that maps DiT inputs by shape (as the first 512 verifier did) silently swaps them; map by `args_<n>`.
- `size256/sweep_sizes.py` renders the six stress prompts at both sizes into one grid (`grid_512_vs_256.png` on the card); the OPEN sign, the face and the feather survive at 256.

### macOS GPU demo app (DiT on the Apple GPU — 6.6 s/image)

`device/BonsaiAppMac/` is a SwiftUI app that runs the DiT on the Apple GPU through the LiteRT Metal accelerator: **0.76 s/DiT-step, 6.6 s/image total** (vs ≈19 s all-CPU), after a one-time ~40 s Metal compile per launch. Text encoder and VAE stay on CPU, where the CompiledModel path is bit-exact vs the fixture set. GPU vs same-latent CPU pipeline: 31.6 dB PSNR, visually identical.

```bash
cd bonsai_image_work
python export_dit_gpu.py                        # GPU-shaped DiT (rope constants folded, gather-free) -> dit_gpu_fp32.tflite
python quantize_dit.py dit_gpu_fp32.tflite      # -> dit_gpu_int4b32.tflite
python fix_zero_block_scales.py dit_gpu_int4b32.tflite dit_gpu_int4b32.tflite
# (or skip the three steps above: the pre-quantized dit_gpu_int4b32.tflite is published on the model card)
cd device/BonsaiAppMac
./prep_resources.sh && xcodegen generate
xcodebuild -project BonsaiMac.xcodeproj -scheme Bonsai -configuration Release build
```

The pitfalls that cost real time (details in `device/BonsaiAppMac/README.md`): the runtime dylib pair must be **same-generation** (ai-edge-litert 2.1.6 works; the 2026-07-31 LiteRT-main macos_arm64 prebuilt pair SIGSEGVs in an XNNPACK transpose microkernel inside the Metal accelerator during delegate init on this 2.27 GiB DiT — repro CLI in `device/BonsaiAppMac/smoke/`); **fp32 precision is mandatory** (default fp16 corrupts this model's activations), passed as a hand-built TOML opaque option because the prebuilt exports no options helpers; and the GPU-shaped export is required — the CPU-shaped DiT does not run on the Metal delegate.

### Shieldstral-1.0-3B (a single-token safety classifier, not a chat model)

`shieldstral_work/convert_shieldstral.sh` converts [mistralai/Shieldstral-1.0-3B](https://huggingface.co/mistralai/Shieldstral-1.0-3B). Published: [litert-community/Shieldstral-1.0-3B](https://huggingface.co/litert-community/Shieldstral-1.0-3B).

The checkpoint is `Mistral3ForConditionalGeneration` — a pixtral vision tower plus a Ministral3 text decoder — and this recipe ships the **text lane**: `scripts/extract_ministral3_text.py` drops the tower, which is output-neutral for text input (bit-identical yes/no logits on the floor set). The decoder is then the ordinary dense lane, same recipe family as `Ministral-3-3B-Instruct-2512`.

```bash
bash shieldstral_work/convert_shieldstral.sh     # -> out_int4/model.litertlm, out_int8/model.litertlm
```

Three things are specific to this model class and worth knowing before you gate it:

**It answers with one token.** The reply is literally `yes` or `no`, and the published quantity is a softmax over those two logits, thresholded at 0.5. A generic 8-question quality gate is nearly useless here — obvious cases carry ±9 to ±14 logit margins, so every build scores 8/8, including one whose measured numbers were wrong. Build the floor gate from **borderline** items, and treat label agreement plus correlation of the raw logit margin against the source model as the real verdict.

**The scoring API has two traps, both silent.** `Session.run_text_scoring` *mutates* the session, so scoring a second candidate after the same prefill returns a plausible-but-wrong number — give each candidate its own session and prefill. And `create_session(apply_prompt_template=True)` prefills the user **prefix only** (no user suffix, no start token), which scores the model mid-prompt: on this bundle that read 1.060 instead of 8.087 and inverted a clearly-unsafe floor item. Pre-render the whole prompt yourself and pass `apply_prompt_template=False`. The ordinary generate path is unaffected — a bundle can generate perfectly while every scored number is wrong.

**The fixed system prompt is baked into the user prefix** (`templates/shieldstral_simple.jinja`). The model card fixes it as part of the inference contract, and the scoring entry point carries no role information, so shipping it as a separate system slot would make the model's own reference prompt unreachable from the API you actually classify with.

Gate results at n=200 (stratified from the public OpenAI moderation evaluation set, threshold 0.5, identical prompt and extraction on both sides): source fp32 F1 85.0 · bf16 85.7 · int8 CPU 86.1 · int8 GPU 86.1 · int4-b32 CPU 85.7 · int4-b32 GPU 86.2. Every row lands inside a 1.2-point band, and every label flip sits where the reference margin is < 0.7 from zero. int4-b32's main section is 1.82 GiB and runs on an iPhone 17 Pro (~1.8 s per verdict warm, ~1.2 GB peak); int8's 3.33 GiB section is desktop-only.

#### Shieldstral text+image (first pixtral tower here)

`shieldstral_work/vision/` builds the multimodal bundle. Pixtral turns out to be an unusually shallow dynamic-resolution tower: at a fixed square size the per-image crop is the identity, the block-diagonal mask for one image is all zeros (so it collapses to plain full attention), and the meshgrid position ids become a constant. Patchify is a single Conv2d and the sequence stays raster-ordered, so no `GATHER_ND` is introduced — the usual mobile-GPU vision blocker never arises.

```bash
python shieldstral_work/vision/static_pixtral.py  src_models/shieldstral-3b   # corr 1.0 vs eager
python shieldstral_work/vision/verify_adapter.py  src_models/shieldstral-3b   # corr 1.0 vs inputs_embeds
python shieldstral_work/vision/convert_shieldstral_vision.py src_models/shieldstral-3b out_vision --size 560
CACHE=4096 python scripts/export_internvl_decoder.py src_models/shieldstral-3b-text out_decoder
DEC=out_decoder VIS=out_vision python shieldstral_work/vision/build_shieldstral_bundle.py
```

#### The vision bundle needs a different exporter flag than the text one

The published `Shieldstral-1.0-3B-vision_int4.litertlm` does not load on a mobile GPU: the delegate takes 52
of 1187 ops on `prefill_2048` and the engine is refused, on `STABLEHLO_COMPOSITE: odml.softmax`. The text
bundles are unaffected — and the difference is not the vision tower, it is which exporter each lane used.
litert-torch 0.9.2 marks the attention softmax as an `odml.softmax` composite unconditionally, and
litert-converter 0.3.0 has no lowering for it. `scripts/export_simple_template.py` carries an env-gated strip
for exactly this; `scripts/export_internvl_decoder.py`, which the vision lane calls, does not.

Two ways out. The strip is one. The better one needs no source change: **litert-converter 0.3.1 lowers the
composite natively**, so re-exporting the decoder on it produces a builtin `SOFTMAX` and an otherwise
identical graph.

```bash
CACHE=4096 PREFILL=2048,1024,512,256,128,64,32,16,8,4,2,1 RECIPE=BOCTAV4 \
  python scripts/export_internvl_decoder.py src_models/shieldstral-3b-text out_decoder   # on a 0.3.1 venv
DEC=out_decoder VIS=out_vision python shieldstral_work/vision/build_shieldstral_bundle.py
```

⚠ The `PREFILL=` and `RECIPE=` above are not optional. The exporter defaults to `128,512`, which does not
reproduce the shipped bundle — its largest subgraph is `prefill_2048`.

Check the result without a device: the re-exported decoder's subgraphs should be 1187 ops for `prefill_2048`
through `prefill_2`, 1058 for `prefill_1` and 1087 for `decode` — identical, one for one, to the text sibling
that already delegates in full. A whole-file scan should find zero `odml.softmax`; the ~2660 `odml.rms_norm`
occurrences stay, and are not a blocker. On a Galaxy S26 the rebundled file then delegates 15202 of 15202 ops
across all 13 subgraphs, the same numbers as the text file.

⚠ **The position ids must keep indexing the trained grid.** They are `h * max_width + w` with `max_width = config.image_size // patch_size` (110 here) — substituting the *new* grid width silently re-indexes the rope table and the output stays plausible.

⚠ **The token expansion does not fit the runtime's injection contract as-is.** Pixtral expands one `[IMG]` into a grid of image tokens with `[IMG_BREAK]` ending every row and one `[IMG_END]` at the end, and those markers keep their ordinary *text* embeddings — but the runtime injects one contiguous block at the soft-token position. The escape is to fold the marker embeddings into the adapter: they are constant rows of the decoder's embedding table, so the adapter emits the full block (a `cat` with a constant column — no gather, no dynamic shape). Verify against `inputs_embeds` for the same image.

Two runtime facts: `vision_backend` must be named explicitly (without it the engine loads, the conversation is created, and only the first image message fails), and the scoring API is text-only (`run_prefill(list[str])`), so image documents get the generated verdict but no continuous score.

**Letterbox images before passing them.** The runtime resizes to the declared H×W with no padding. On 100 labelled images, stretching cost 8.5 F1 against the source model where letterboxing cost 3.0, and agreement rose from 92% to 96% — a larger effect than the int8-vs-int4 choice.

## LFM2.5-VL (3B / 1.6B / 450M — LFM2 hybrid text + SigLIP2 vision, native runtime image support)

The first family where the vanilla `litert-torch` VLM path and the runtime's own image pipeline line up end-to-end: `LiquidAI/LFM2.5-VL-3B`, `-1.6B` and `-450M` convert with stock `export_hf --task image_text_to_text` and run **text + image on the released `litert-lm==0.16.0` pip runtime** (`litert-lm run model.litertlm --prompt "..." --attachment img.png`).

```bash
python -m venv .venv && .venv/bin/pip install litert-torch==0.9.3 "transformers==5.14.1" pillow "torch==2.12.1" "torchvision==0.27.1" litert-lm
.venv/bin/python lfm_work/convert_lfm25_vl.py LiquidAI/LFM2.5-VL-3B out_vl3b_int8            # int8 (text+vision dynamic_wi8)
.venv/bin/python lfm_work/convert_lfm25_vl.py LiquidAI/LFM2.5-VL-3B out_vl3b_fp --fp         # fp text for int4
.venv/bin/python lfm_work/quantize_lfm25_vl.py out_vl3b_fp/model.litertlm LFM2.5-VL-3B_int4.litertlm          # int4-b32 octav + wi8 embedder + zero-scale fix + executor metadata
.venv/bin/python lfm_work/quantize_lfm25_vl.py out_vl3b_int8/model.litertlm LFM2.5-VL-3B_int8.litertlm --recipe none  # executor metadata only
LITERT_LM=.venv/bin/litert-lm python lfm_work/gate_lfm25_vl_text.py LFM2.5-VL-3B_int4.litertlm cpu
LITERT_LM=.venv/bin/litert-lm python lfm_work/gate_lfm25_vl_image.py LFM2.5-VL-3B_int4.litertlm cpu
```

⚠ **Pin transformers==5.14.1.** 5.15.0 renamed `Lfm2ShortConv.L_cache` → `conv_kernel_size` and the 0.9.3 export subclass still reads the old attribute (`AttributeError` at model load). torchvision must stay 0.27.x or it drags torch past litert-torch's `<2.13` pin.

⚠ **The metadata pbtext needs `llm_model_type { lfm2 {} }` — empty is correct.** The runtime's `Lfm2DataProcessor` proto defaults already equal this family's processor config (512×512, patch 16, max 1024 patches, pooling 2, mean/std 0.5, boi `<|image_start|>` / eoi `<|image_end|>`). The chat template must render a bare `<image>` per image content part and nothing else: the runtime splits the rendered prompt on it and inserts boi + preprocessed pixels + eoi itself, so a template that emits boi/eoi too would double them.

⚠ **The externalized embedder dodges every text recipe.** VLM exports split the embedder into its own tflite; an `--fp` text export leaves it float32 (1 GB at the 3B's 128k vocab) and a post-hoc recipe applied to `prefill_decode` never reaches it — `quantize_lfm25_vl.py` runs the wi8 recipe on the embedder tflite as a separate pass (1049→264 MB). It also goes through `litert-lm pack/unpack` instead of a single-tflite rebuild, because the older quantize/zero-scale scripts would silently drop the vision sections.

⚠ **The zero-scale int4 wall is inherited by the VL text stack.** The 3B checkpoint carries 746,432 all-zero 32-blocks across 26 tensors — the same dense-checkpoint signature as LFM2.5-2.6B — so the zero-scale fix (folded into `quantize_lfm25_vl.py`) is mandatory; the 450M has none.

Vocab split inside one family: the 3B uses the 128k vocab (stops 124900/124895), the 1.6B and 450M the older 64k one (stops 7/2, same as the 1.2B family) — the converter picks the matching pbtext by model name and refuses sizes it does not know (check `generation_config.json` before adding one).

Gates on the 0.16.0 pip CLI (temperature 0, `--cache no`): 3B int8 AND int4 pass text 7/8 + image 5/5 on cpu and macOS gpu, with image answers verbatim identical to the HF bf16 reference (`verify_lfm25_vl_hf.py`); 1.6B passes text 8/8 on all four configs; 450M int4 passes 8/8 + 4/5. iPhone GPU was not re-tested for these VL files. LiteRT-LM#3129, once read as an iOS block for the ShortConv family, turned out to be an integration-side `maxNumTokens` sizing issue and the text-only 1.2B GPU files do run on iOS Metal at `maxNumTokens` 1024 — but that is measured on the text family, not here, so treat iOS for VL as unmeasured rather than blocked.

⚠ **The 64k-vocab models (1.6B, 450M) deterministically miss fine shape/counting fixtures on-device while the torch reference answers them** — and every conversion-side suspect has been eliminated by experiment: exported vision tower AND projector match torch at cosine 1.0000 on identical inputs; an unquantized bundle reproduces the identical misses; runtime patchify layout equals the HF processor's (code-read, both (ph,pw,c) raster); the prompt/token streams match token-for-token (boi/eoi are single tokens in every vocab); double-BOS, fp16 decode, jpeg-level pixel perturbation and prefill-chunk-boundary alignment (probed by forcing the image block exactly onto a 256-token chunk edge via a custom chat template) all change nothing; geometry probes on-device (corner localization, stripe orientation) come back CORRECT. Coarse understanding, OCR and localization are intact — the residue is engine-internal precision on contour/counting answers, the 3B is unaffected, and the next step is instrumenting the runtime to diff the spliced embedding sequence against HF `inputs_embeds`.

## Verify a reproduction

```bash
~/venvs/ltconv040dev/bin/python scripts/verify_quality.py out/<key>/model.litertlm --json   # 8-question gate
# parity (dense): scripts/parity_gsm8k.py  ·  reasoning models: run at --max-tokens 2048
# single-token classifiers: label F1 + margin correlation vs the source model, not the 8-question gate
```

## granite-4.1-3b (dense) — and the BOS a converted bundle must not prepend

`granite41_work/convert_granite41_3b.sh` converts IBM's dense 3.4B instruct model to `.litertlm` on a pristine released stack (litert-torch 0.9.3 / litert-converter 0.3.1 / ai-edge-quantizer 0.8.0 / litert-lm-builder 0.16.0, transformers 5.14.1) — no patched checkout. Published: [litert-community/granite-4.1-3b](https://huggingface.co/litert-community/granite-4.1-3b). **Requires litert-lm ≥ 0.16 to run.**

```bash
cd granite41_work
pip install "litert-torch==0.9.3" "litert-lm-builder==0.16.0" "transformers==5.14.1"
python -c "from huggingface_hub import snapshot_download; snapshot_download('ibm-granite/granite-4.1-3b', local_dir='src_models/granite-4.1-3b')"
./convert_granite41_3b.sh          # int4 (ship) + int8, six prefill signatures, NO_START_TOKEN=1
```

**The one thing that will bite you on this family.** `litert-lm-builder` writes a `start_token` into the bundle metadata from `tokenizer.bos_token` whenever that field is set, without consulting `add_bos_token`. Granite declares `add_bos_token: False` **and** its BOS is the same token as its EOS (`<|end_of_text|>`), so the runtime prepends a token the model reads as "this document has already ended": it answers by echoing the question back, and the 8-question sanity gate drops from 8/8 to 5/8. It looks exactly like quantization damage and is not — feeding the same rendered prompt to the bf16 PyTorch model with and without that leading token reproduces both halves.

The conversion script sets `NO_START_TOKEN=1`, which clears `bos_token` before the metadata is built. For a bundle you have already built, `strip_start_token.py in.litertlm out.litertlm` unpacks it, drops the field and repacks — weights untouched, about 30 seconds. Check any model whose `tokenizer_config.json` says `add_bos_token: False`; bos == eos is the shape that hurts.

**Six prefill signatures, not eleven.** Every exported signature is charged engine memory whether or not it is called. An eleven-signature build of these weights is killed by iOS during Metal initialisation unless the app caps the context at 1024; `1024,256,64,16,4,1` creates its engine at the bundle's full 4096 on an iPhone 17 Pro, for 12 MB more on disk.

`hf_oracle.py` runs the same eight questions through the HF model in bf16 first, so a later miss on the converted file can be attributed to the conversion rather than to the model.

### granite-4.1 finetune intake — the BOS guard goes generic, and the family fact meets its first exception

Measured 2026-08-25 on [Kezmark/Mordant-3B-Think](https://huggingface.co/Kezmark/Mordant-3B-Think)
(full SFT for image-prompt composition with chain-of-thought; the highest-download real
granite-4.1-3b derivative after excluding a name-declared Claude distill and the
unsloth/mlx mirrors). Hub demand: 19 finetunes + 15 adapters.
Published: [mlboydaisuke/Mordant-3B-Think-LiteRT](https://huggingface.co/mlboydaisuke/Mordant-3B-Think-LiteRT)
(int8, with the measured card and litertlm_manifest.json).

```bash
python scripts/convert.py Kezmark/Mordant-3B-Think
```

One command, clean shell: export 1,621 s → 3.76 GB int8, reduced 7-signature ladder,
externalized embedder, think-aware gate → **8/8 PASS, decode ~71 tok/s CPU**, template
byte-equal 1474/1474, bundle stop = the tokenizer's `<|end_of_text|>` 100257.

Two things this subject taught the intake:

- **The BOS trap generalizes, so the guard did too.** Mordant has `bos == eos ==
  <|end_of_text|>` and a template that never leads with BOS — the shape from the base
  granite-4.1-3b lesson above. Measured on this subject: prepending that one token flips HF
  bf16 greedy from a correct answer to an endless ``` ``` ``` loop. `convert.py` now runs
  `ensure_no_spurious_start_token` after every stock export. *Rewritten 2026-08-27:* it
  renders the chat template **twice** — once with `bos_token` bound (what transformers
  serves, the reference) and once with it empty (what minijinja does on device) — and drops
  the metadata start_token only when `1 + literal-BOS-in-the-device-render` exceeds the
  reference count. That separates `{{ bos_token }}` from a literal BOS, which the earlier
  `add_bos_token` / auto-add test conflated in both directions: it would have dropped
  Falcon-H1's start token (leaving zero BOS) and left Zamba2's genuine duplicate in place.
  The e2e report records `start_token: dropped_spurious_bos` / `dropped_duplicate_bos` /
  `kept_matches_reference`, plus the two counts under `start_token_bos`.
- **The template family fact has its first measured exception.** Every LLM derivative
  measured until now kept the base template byte-for-byte (modulo CRLF re-serialization);
  Mordant *replaces* it with its own 1,474-byte think-form template. The stock path is
  indifferent — it embeds the derivative's own template verbatim either way (byte-equal
  measured) — but "derivative templates always equal the base" is now demonstrated folklore,
  not fact. Same lesson one layer down: the trainer's re-save collapsed `tokenizer.json`'s
  pre_tokenizer (granite's contraction-aware Split → plain ByteLevel); the derivative's own
  tokenizer is what gets bundled and what both sides of any A/B must use.

The reduced 7-signature prefill ladder is now applied to **every ≥3B export**, not just
hybrids — the dense granite-4.1-3b's own 11-signature iOS Metal kill (documented above) is
the same signature-memory law the hybrid 4B hit.

### 2026-08-27 — auditing every published bundle's first token, and what the engine really does

The Hy-MT2 double-BOS finding raised an obvious question about everything already
shipped, so all **88 published `.litertlm` files across 64 repos** were swept. Only the
header is needed — `LlmMetadata` sits in the first few KB, so two HTTP range requests per
file read the `start_token` and the embedded template without downloading any weights
(`make_manifest.py` uses the same pattern). Zero files failed to parse.

**Read the engine, not the output.** The earlier discriminator compared generated text,
which is why it took three probes to survive int8 coincidence. The runtime exposes the
answer directly: `Conversation.render_message_to_string` returns the minijinja render
verbatim, `Engine.tokenize` returns the exact ids, and with `enable_benchmark=True` the
prefill counter returns the assembled prompt length. Repacking one local bundle's metadata
three ways settles the mechanics with no generation at all:

| `start_token` | resolves to | prefill tokens |
|---|---|---|
| `<bos>` | id 2 | **16** |
| the string `"None"` | **id 9336, the word `None`** | **16** |
| absent | — | **15** |

`len(tokenize(render))` is 15 in all three. So: the engine prepends the start token and it
costs exactly one token; the render it prepends onto **does not contain the BOS** even
though the template source begins `{{ bos_token }}` — minijinja leaves `bos_token`
unbound; and `tokenize("hello world")` returns `[23391, 1902]`, so the engine's encode
adds no special tokens regardless of what the tokenizer's post-processor says. (The
bundle's own `tokenizer.json` *does* carry a `TemplateProcessing` that would add `<bos>`;
it is simply not applied.)

The count a bundle actually feeds is therefore `(1 if start_token is the bos) + literal
BOS at the head of the render`, and the reference it must equal is the same template
rendered with `bos_token` bound — what transformers serves, since its chat path encodes
with `add_special_tokens=False`.

**19 of 88 bundles were feeding a first token the model was never trained on**, in three
shapes:

- **The literal string `"None"` (10 bundles)** — `str(None)` reached the metadata for
  models whose tokenizer has no BOS at all, and the engine resolves it to a real vocab
  token. Measured on the published `InternVL3-1B`, the prompt began
  `['None', '<|im_start|>', 'user', …]`. InternVL3-1B/2B, InternVL3_5-1B/2B/4B,
  LLaVA-OneVision-0.5B, Mage-VL, Ovis2.5-2B, Qwen2-VL-2B, SmolLM3-3B.
- **A genuine double BOS (6 bundles)** — the template writes the BOS as a *literal*, so
  the start token lands on top of it. Measured on the published `SmolVLM2-500M`:
  `['<|im_start|>', '<|im_start|>', 'User', …]` against a reference of one. Zamba2-1.2B/2.7B,
  SmolVLM2-500M/2.2B, Nanbeige4.1-3B/4.2-3B.
- **A BOS where the reference has none (3 bundles)** — granite-4.0-h-1b, Nemotron-H-4B,
  Phi-4-mini-reasoning. The 350m was fixed for exactly this in August; the 1b was left
  because it tolerates the token, which is not the same as being faithful to training.

**Falcon-H1, LFM2.5 and gemma-3-270m are clean** — they write `{{ bos_token }}`, which
renders empty on device, so their start token supplies the one BOS the model expects.
Dropping it would have shipped a bundle with *no* BOS. Two more read as defects until the
reference was taken from real token ids rather than declared strings: North-Micro-Vision's
`start_token: ids[2]` **is** `<BOS_TOKEN>`, and PaddleOCR-VL's canonical head is
`<|begin_of_sentence|>`, not the `<s>` its config declares.

**The fix is metadata-only.** `drop_start_token.py` unpacks, removes the block, and
repacks; `section_identity.py` then sha256s every section and requires everything except
`LlmMetadataProto` to be byte-identical, so no requantization can hide in a repack and the
card's measured numbers still stand — only the file's own sha256 changes. Nanbeige needs a
second edit: its affix `system.prefix` is `system\n`, i.e. the leading `<|im_start|>` was
coming from the start token, so a bare drop fixes the no-system path and breaks the system
one. Dropping the start token *and* rewriting that prefix to `<|im_start|>system\n` leaves
the system path's prompt identical (prefill 24 before and after) while fixing the other.

### granite-4.0-micro — a granitemoehybrid checkpoint with no mamba layer, rewritten to the dense Granite layout first

`granite_work/convert_granite40_micro.sh` converts [ibm-granite/granite-4.0-micro](https://huggingface.co/ibm-granite/granite-4.0-micro) at commit `56111ae1` — "a 3B parameter long-context instruct model" — into the two files of [litert-community/granite-4.0-micro](https://huggingface.co/litert-community/granite-4.0-micro). Environments (released wheels): litert-torch 0.9.4, litert-converter 0.4.0, ai-edge-quantizer 0.9.0, transformers 5.14.1, torch 2.13.0; litert-lm 0.18.0 for the metadata step and for running the files.

```bash
EXPORT_PY=/path/to/python-with-litert-torch-0.9.4 \
RUN_PY=/path/to/python-with-litert-lm-0.18.0 \
LITERT_LM=/path/to/litert-lm \
  bash granite_work/convert_granite40_micro.sh out_granite40_micro
```

| file | section | bytes | sha256 |
|---|---|---|---|
| `granite-4.0-micro_int8.litertlm` (3,768,357,216 B) | LlmMetadataProto | 6,441 | `783b0ea54f5a5e7627a76f1f561abe5f351abea01c45ed506ba3cd385904bd1d` |
|  | ExecutorMetadataProto | 5,446 | `d6a12663c3b913d5dc4e21f17e8330ea0ef21a9e4e1f251144c52170d77fa346` |
|  | HF_Tokenizer_Zlib | 1,404,724 | `7d73e13c3e3c6e082ca8e3bdae4c212936b191fc0f57e94b087e205399e38ee6` |
|  | TFLiteModel | 3,501,542,592 | `e66176bfb26014665602e93c9ebec02e85d072e9f09fcf2913fc0a2a1698a916` |
|  | TFLiteModel | 265,343,328 | `1331235f15723f89a20be5f2a9346da46cb494b789b459752b1d90f9192159d8` |
| `granite-4.0-micro_int4.litertlm` (2,197,590,368 B) | LlmMetadataProto | 6,441 | `783b0ea54f5a5e7627a76f1f561abe5f351abea01c45ed506ba3cd385904bd1d` |
|  | ExecutorMetadataProto | 5,446 | `d6a12663c3b913d5dc4e21f17e8330ea0ef21a9e4e1f251144c52170d77fa346` |
|  | HF_Tokenizer_Zlib | 1,404,724 | `7d73e13c3e3c6e082ca8e3bdae4c212936b191fc0f57e94b087e205399e38ee6` |
|  | TFLiteModel | 1,930,787,712 | `e7404f3470e9a2c506f2a9df5e77faebc399566de6a6938c1822da2687098835` |
|  | TFLiteModel | 265,343,328 | `1331235f15723f89a20be5f2a9346da46cb494b789b459752b1d90f9192159d8` |

The script wrote these section hashes (`repack_int8.json` / `repack_int4.json`, `sections[].new_sha256`) when it was run from a copy of this repository's layout on 2026-10-13; every section equal to the published files' sections by raw bytes. The file sha256 differs from run to run (the packer writes a new uuid and creation timestamp), so compare sections.

- **A dense model under a hybrid name.** The checkpoint's config.json says `granitemoehybrid`, but all 40 layers are attention layers and there are 0 experts. `to_dense_granite.py` moves the weights byte for byte into the dense Granite layout that the export path already handles: `shared_mlp.input_linear` rows 0..8191 → `mlp.gate_proj`, rows 8192..16383 → `mlp.up_proj`, `shared_mlp.output_linear` → `mlp.down_proj`, every other tensor as it was, config = granite-4.1-3b's key set.
- **The rewrite is proven before anything is exported.** `dense_check.py` runs the original and the rewritten checkpoint in fp32 on the CPU over three prompts (37, 77 and 839 tokens): max |logit delta| 1.922e-04, 3.338e-04 and 9.127e-04, argmax equal at every position, 50 greedy tokens identical; in float64 the delta is 0.0. transformers 5.14.1 cannot run `generate()` on the original checkpoint (`has_previous_state` raises `ValueError` when no mamba layer exists), so the script wraps the recurrent-mask builder to return `None` when the config has no linear-attention layer. Exporting the original checkpoint directly also works (one fused gate|up matmul followed by two slices; within 3 % of the rewrite's speed on a Mac).
- **IBM's template, no start token, one stop.** `USE_JINJA=1` embeds IBM's chat_template.jinja byte for byte, default system message included; its renders on the engine's template renderer equal transformers' in 9 of 10 conversation shapes. `bos == eos == <|end_of_text|>` with `add_bos_token: false`, so `NO_START_TOKEN=1` (the granite-4.1-3b lesson above) and the prompt starts with `<|start_of_role|>`. `make_pbtext.py` keeps only the stop id 100257 of the exporter's 15 entries and sets `supports_thinking` and `supports_function_calling` false.
- **Tool calls come back as text; send tool results back as text.** The bundle's model type has no tool-call parser: the model's `<tool_call>{…}</tool_call>` JSON arrives in the reply text. A tool result passed to the Python API as a tool-response content part renders empty with this template and the model invented a report; the same result as the tool message's text was used.
- **KV budget 4096, six prefill signatures (1024, 256, 64, 16, 4, 1), embedder externalised**. The int8 file's main TFLite section is 3,501,542,592 bytes, over 2 GiB; not run on iOS.
- **int8 = `dynamic_wi8_afp32` (channelwise int8 weights, float activations); int4 = `BOCTAV4` (blockwise-32 OCTAV int4 + int8 embedding), every scale normal.** GSM8K (questions 0–99, 0-shot step by step, greedy, at most 1024 output tokens, one process per question): bf16 86, int8 83 on the CPU and 90 on the GPU, int4 84 on the GPU; the files keep the runtime's default fp16 GPU activations.
- Measured (prefill 1024 / decode 256 / context 2048, litert-lm 0.18.0 on the Mac, prebuilt `litert_lm_advanced_main` 2026-09-18 on the phones): Mac M4 Max GPU (WebGPU) int8 1425 / 69.8 tok/s, int4 1455 / 87.3; Galaxy S26 GPU 27.75–29.17 s for int8 and 21.25–22.17 s for int4 over 3 runs each; Pixel 8a GPU int4 98.2 / 8.24 tok/s — the int8 file has no Pixel 8a GPU row: the harness stops the runtime below 400,000 kB of MemAvailable, and it stopped every process of that try; every decoder signature fully delegated on all three GPUs.

**Re-shipped 2026-10 in the GPU-composite form (retrofit).** The two published files were rebuilt from the files above without re-exporting: `tools/gpu_graph/gpu_graph_retrofit.py` rewrites the prefill and decode graphs into the shape litert-torch's `--apply_gpu_composites` produces (one `odml.cache_update` composite per layer instead of two DYNAMIC_UPDATE_SLICE writes, `odml.runtime_bmm` composites for the attention products, a `param_tensor` signature input) and keeps every weight buffer byte for byte; the metadata, tokenizer and embedder sections are the previous files' bytes.

```bash
# LITERT_LM_CLI must point at a litert-lm 0.18.0 CLI before the two commands run
python tools/gpu_graph/gpu_graph_retrofit.py granite-4.0-micro_int8.litertlm granite-4.0-micro_int8.retrofit.litertlm --decomp exporter --mask add_bcast --report retrofit_int8.json
python tools/gpu_graph/gpu_graph_retrofit.py granite-4.0-micro_int4.litertlm granite-4.0-micro_int4.retrofit.litertlm --decomp exporter --mask add_bcast --report retrofit_int4.json
```

| file | bytes | sha256 |
|---|---|---|
| `granite-4.0-micro_int8.litertlm` (re-shipped) | 3,769,241,952 | `04ee6ad9e905448f02fa58ad3757cd3a32a773ce7b908fafb73cf8d9b5d84df5` |
| `granite-4.0-micro_int4.litertlm` (re-shipped) | 2,198,475,104 | `b931889ac060fb563f5ae20b3f4ebc772297f15f77cdb9b0f55b1b25a160b325` |

- **What changed and what did not.** The report (`retrofit_*.json`) lists every buffer of the previous main section unchanged in both files (716 in the int4 file, 435 in the int8 file); the main TFLite section grew from 1,930,787,712 to 1,931,672,448 bytes (int4) by the composite graph alone. The attention mask stays a FLOAT32 input added by broadcast (`--mask add_bcast`); on the CPU this form runs within 1.4 % of the previous files (Mac M4 Max, prefill 1024 / decode 256), while `--mask select_bool` (SELECT_V2) cost the Mac CPU 8 % here and the tool's docstring records `--mask add` as broken on the Mac GPU.
- **Why.** On GPUs the KV-cache write and the attention product reach the runtime as composite ops: Mac M4 Max (WebGPU, the previous files in the same windows) decode 71.6 → 89.3 tok/s (int8) and 86.1 → 111.3 tok/s (int4), prefill 1471 → 1381 and 1478 → 1391 tok/s; Galaxy S26 (Adreno 840) int4 decode 15.7 → 20.2 tok/s, prefill 335.6 → 235.6 tok/s, so the 1024 + 256 workload takes 17.0 s instead of 19.5 s (mean of two holds, both orders) while the time to the first token grows from 3.19 to 4.41 s. On the Pixel 8a (Mali-G715) the int4 file's GPU time is unchanged within the order effect. The Mac CPU rows are within 1.4 % of the previous files'; on the phones the CPU was not measured in same-hold pairs.
- **Same answers.** The re-shipped files give the previous files' texts on the 8-question gate (CPU and GPU) and the BOS check and pass the 3-turn conversation (the int8 file's CPU replies to turns 2 and 3 are worded differently); GSM8K (questions 0–99, the same protocol as above): int8 90 on the GPU and 84 on the CPU, int4 84 on the GPU — the previous files' 90 / 83 / 84, with 0 discordant questions on the GPU.
- **Verifying the retrofit output.** `litert-lm unpack` of the new file and of the previous file give the same bytes for every section but the main TFLite one; the tool's `--report` JSON lists, per signature, the 40 layers it rewrote and their attention products, the composites it wrote (280 `odml.cache_update`, 40 per signature; 548 `odml.runtime_bmm`, 80 in decode and 78 in each prefill signature) and the unchanged buffers, and `litert-lm run --backend gpu --verbose` on the new file shows every decoder subgraph delegated.
- **Re-run 2026-10-16.** The two commands above, run again from this repository's clone against the 2026-10-13 files, reproduced the published files section by section: the five section sha256 values, the buffer counts (int8 435 → 736, int4 716 → 1017) and the composite counts (280 `odml.cache_update`, 548 `odml.runtime_bmm` per file) all equal, at the published sizes; the file-level sha256 differs (`b5007379…` and `6ae8e6cc…` against the published `b931889a…` and `04ee6ad9…`) while every section matches, so compare sections.

### granite-4.1-8b — the lightweight export path for an 8B, and the GPU-composite graph by rewrite

`granite41_work/convert_granite41_8b.sh` converts [ibm-granite/granite-4.1-8b](https://huggingface.co/ibm-granite/granite-4.1-8b) at commit `1504002f` — "a 8B parameter long-context instruct model" — into the two files of [litert-community/granite-4.1-8b](https://huggingface.co/litert-community/granite-4.1-8b). Environments (released wheels): litert-torch 0.9.4, litert-converter 0.4.0, ai-edge-quantizer 0.9.0, transformers 5.14.1, torch 2.13.0; litert-lm 0.18.0 for the metadata step, the graph rewrite's pack and running the files.

```bash
EXPORT_PY=/path/to/python-with-litert-torch-0.9.4 \
RUN_PY=/path/to/python-with-litert-lm-0.18.0 \
LITERT_LM=/path/to/litert-lm \
  bash granite41_work/convert_granite41_8b.sh out_granite41_8b
```

| file | section | bytes | sha256 |
|---|---|---|---|
| `granite-4.1-8b_int8.litertlm` (8,949,862,832 B) | LlmMetadataProto | 6,122 | `7db3ad38d53da7ef22963c575df91540ab90a69baffec62145df4847f9dbb3b1` |
|  | ExecutorMetadataProto | 5,446 | `d6a12663c3b913d5dc4e21f17e8330ea0ef21a9e4e1f251144c52170d77fa346` |
|  | HF_Tokenizer_Zlib | 1,404,724 | `7d73e13c3e3c6e082ca8e3bdae4c212936b191fc0f57e94b087e205399e38ee6` |
|  | TFLiteModel | 8,528,914,704 | `b298d333ff77cc673a5451eb9217a4b792ec40e8fdbb3a67d7dc954e727621d8` |
|  | TFLiteModel | 419,484,080 | `7f239a4fefee591ff51112fce2af15342f8917452d35c703d10f2609cb1eb4ab` |
| `granite-4.1-8b_int4.litertlm` (5,151,986,096 B) | LlmMetadataProto | 6,122 | `7db3ad38d53da7ef22963c575df91540ab90a69baffec62145df4847f9dbb3b1` |
|  | ExecutorMetadataProto | 5,446 | `d6a12663c3b913d5dc4e21f17e8330ea0ef21a9e4e1f251144c52170d77fa346` |
|  | HF_Tokenizer_Zlib | 1,404,724 | `7d73e13c3e3c6e082ca8e3bdae4c212936b191fc0f57e94b087e205399e38ee6` |
|  | TFLiteModel | 4,731,041,056 | `77b54f7a1725be9660d8455597f5ee0bfbf1f82db89b40543eac7f2abb9f8a06` |
|  | TFLiteModel | 419,484,080 | `7f239a4fefee591ff51112fce2af15342f8917452d35c703d10f2609cb1eb4ab` |

Run as published on 2026-10-16 (one command, the environment of its header, 33 minutes on a 128 GB Mac including the Hub fetch), this script rebuilt every one of these sections byte for byte from the pinned checkpoint, and the two exported-graph files as well; so had the conversion run of 2026-10-14 behind the shipped files. The file sha256 differs from run to run (the packer writes a new uuid and creation timestamp), so compare sections.

- **The lightweight export path.** An export of this model through the default path reached 98.8 GB on a 128 GB Mac in the converter passes before a RAM guard stopped it. `export_lightweight.py` is `scripts/export_simple_template.py` plus `experimental_lightweight_conversion` (`LIGHTWEIGHT=1`). With litert-torch's 1 MiB default for lazy constants the converter left the 40 o_proj weights unquantized behind a constant-only MUL (× residual_multiplier 0.22), so `RESOURCE_CONSTANT_NUMEL_THRESHOLD=100000000` keeps only constants above 1e8 bytes (the MLP weights and the embedding table) lazy; every FULLY_CONNECTED weight is then an int8 / int4 constant and no op has all-constant inputs, and the residual scale stays a scalar MUL on each down_proj output. Peaks 50–56 GiB per export. Run one export at a time.
- **IBM's template, no start token, one stop.** `USE_JINJA=1` embeds IBM's chat_template.jinja byte for byte; unlike granite-4.0-micro's it has no default system message, so a conversation without one has no system turn. Its renders on the engine's template renderer equal transformers' in 9 of 10 conversation shapes. `bos == eos == <|end_of_text|>` with `add_bos_token: false`, so `NO_START_TOKEN=1` (the granite-4.1-3b lesson above): the engine's prefill count equals transformers' and the prompt starts with `<|start_of_role|>`. `make_pbtext.py` keeps only the stop id 100257 of the exporter's 15 entries and sets `supports_thinking` and `supports_function_calling` false.
- **Tool calls come back as text; send tool results back as text.** The bundle's model type has no tool-call parser: the model's `<tool_call>{…}</tool_call>` JSON arrives in the reply text. A tool result passed to the Python API as a tool-response content part renders empty with this template and the reply did not use it; the same result as the tool message's text was used.
- **KV budget 4096, six prefill signatures (1024, 256, 64, 16, 4, 1), embedder externalised**. The main TFLite section is 8,528,914,704 bytes (int8) and 4,731,041,056 bytes (int4), both over 2 GiB; not run on iOS.
- **int8 = `dynamic_wi8_afp32` (channelwise int8 weights, float activations); int4 = `BOCTAV4` (blockwise-32 OCTAV int4 + int8 embedding), every scale normal.** GSM8K (questions 0–99, 0-shot step by step, greedy, at most 1024 output tokens, one process per question): bf16 96, int8 95, int4 93 on the Mac GPU; the same weights before the graph rewrite: 95 and 93, with no question changing between right and wrong.
- **The GPU-composite graph by rewrite.** `tools/gpu_graph/gpu_graph_retrofit.py X_plain.litertlm X.litertlm --decomp exporter --mask add_bcast` (step 4, once per file) turns each layer's two DYNAMIC_UPDATE_SLICE KV writes into one `odml.cache_update` and the attention products into `odml.runtime_bmm` (280 and 548 composites), keeps every weight byte and every other section. On the Mac GPU the 1024 / 256 / 2048 workload went from 8.20 to 6.95 s (int8) and from 6.93 to 5.78 s (int4); on the CPU the difference stayed inside the run-to-run spread.
- Measured (prefill 1024 / decode 256 / context 2048): Mac M4 Max, litert-lm 0.18.0, two passes per file, the median — GPU (WebGPU) int8 718.0 / 46.07 tok/s, int4 719.6 / 58.31; CPU int8 110.60 / 9.82, int4 51.50 / 11.68; every decoder signature fully delegated on the Mac GPU. Galaxy S26 (the prebuilt litert_lm_advanced_main of 2026-09-18, OpenCL), the int4 file in two sessions: prefill 91.43 / 90.43, decode 5.40 / 5.30 tok/s. On litert-lm 0.17.1 both files answered a one-word question on the Mac CPU (one `run` each). The int8 file was not run on the Galaxy S26 after another 8B int8 file dropped that phone during its GPU engine warm-up; the Pixel 8a int4 GPU try was stopped by the harness's guard at 400,000 kB of MemAvailable.

### Ministral-3-8B-Instruct-2512 — a text tower out of a vision checkpoint, the lightweight export path for an 8B, and the GPU-composite graph by rewrite

`ministral_work/convert_ministral3_8b.sh` converts [mistralai/Ministral-3-8B-Instruct-2512-BF16](https://huggingface.co/mistralai/Ministral-3-8B-Instruct-2512-BF16) at commit `f6fae979` — the BF16 weights of [mistralai/Ministral-3-8B-Instruct-2512](https://huggingface.co/mistralai/Ministral-3-8B-Instruct-2512), whose repo ships FP8 weights — into the two files of [litert-community/Ministral-3-8B-Instruct-2512](https://huggingface.co/litert-community/Ministral-3-8B-Instruct-2512). Environments (released wheels): litert-torch 0.9.4, litert-converter 0.4.0, ai-edge-quantizer 0.9.0, transformers 5.14.1, torch 2.13.0; litert-lm 0.18.0 for the metadata step, the graph rewrite's pack and running the files.

```bash
EXPORT_PY=/path/to/python-with-litert-torch-0.9.4 \
RUN_PY=/path/to/python-with-litert-lm-0.18.0 \
LITERT_LM=/path/to/litert-lm \
  bash ministral_work/convert_ministral3_8b.sh out_ministral3_8b
```

| file | section | bytes | sha256 |
|---|---|---|---|
| `Ministral-3-8B-Instruct-2512_int8.litertlm` (8,638,820,784 B) | LlmMetadataProto | 3,734 | `4fa8c1545105d9ecb22c2d84acb58293b58b7b179d24bc00c1a5e4f4fd4272b5` |
|  | ExecutorMetadataProto | 4,618 | `a3db49d0f81caf3130c521ca2e672a9ddd032b52c2c86b0744e2de6a349b8aa4` |
|  | HF_Tokenizer_Zlib | 2,806,681 | `d30768fefb41588cee87f2f0aa1c1fe7883c6650864be23f0285bfc5842ca5fc` |
|  | TFLiteModel | 8,088,047,040 | `91c219c33b9e865cf03fab00f89ded7052e296032e9fe77b1fafa61d793cd57c` |
|  | TFLiteModel | 547,893,680 | `4be81d8d5f7e6fc108b9486eeae109a8b2b569861aeb9a2f0261c0fcf736b1ba` |
| `Ministral-3-8B-Instruct-2512_int4.litertlm` (5,039,124,912 B) | LlmMetadataProto | 3,734 | `4fa8c1545105d9ecb22c2d84acb58293b58b7b179d24bc00c1a5e4f4fd4272b5` |
|  | ExecutorMetadataProto | 4,618 | `a3db49d0f81caf3130c521ca2e672a9ddd032b52c2c86b0744e2de6a349b8aa4` |
|  | HF_Tokenizer_Zlib | 2,806,681 | `d30768fefb41588cee87f2f0aa1c1fe7883c6650864be23f0285bfc5842ca5fc` |
|  | TFLiteModel | 4,488,362,336 | `7872d3ab4a8ececed80e54b676bf54c4887e1510539623a2130538d4d86c2dc4` |
|  | TFLiteModel | 547,893,680 | `4be81d8d5f7e6fc108b9486eeae109a8b2b569861aeb9a2f0261c0fcf736b1ba` |

Run as published on 2026-10-16 (one command, the environment of its header, 24 minutes on a 128 GB Mac), this script rebuilt every one of these sections byte for byte from the pinned checkpoint; so had the conversion run of 2026-10-14 behind the shipped files (40 minutes). The file sha256 differs from run to run (the packer writes a new uuid and creation timestamp), so compare sections.

- **A text tower out of a vision checkpoint.** The checkpoint is `Mistral3ForConditionalGeneration`: a Pixtral vision tower, a projector and a Ministral 3 text decoder. `scripts/extract_ministral3_text.py` keeps the decoder and its untied lm_head as a standalone `Ministral3ForCausalLM` (0 missing, 0 unexpected keys; the YaRN rope parameters kept) and drops the rest; the full model and the text tower give the same bf16 logits at every position of three 528–533-token prompts, max |delta| 0.0.
- **The lightweight export path.** An 8B export through the default path (granite-4.1-8b on the same 128 GB Mac) reached 98.8 GB in the converter passes before a RAM guard stopped it. `export_lightweight.py` is `scripts/export_simple_template.py` plus `experimental_lightweight_conversion` (`LIGHTWEIGHT=1`), and `RESOURCE_CONSTANT_NUMEL_THRESHOLD=100000000` keeps only constants above 1e8 bytes (the MLP weights and the lm_head) lazy; peaks 46 GB (int8) and 52 GB (int4). `const_ops.py` and `fc_dtypes.py` check the result: no op with all-constant inputs, no fp32 FULLY_CONNECTED weight. Run one export at a time.
- **The 3B bundle's template with the 8B system message.** `ministral3_8b_dual.jinja` is the template of the Ministral-3-3B-Instruct-2512 bundle (string or content-parts messages) with Mistral AI's default system message for the 8B verbatim; `{today}` and `{yesterday}` stay literal text, as in the vendor template. The template writes no BOS; the bundle's start token `<s>` adds one, and the engine's prefill count equals transformers' for the vendor template. Its renders equal transformers' vendor render in 6 conversation shapes; the vendor's tool list is not rendered. `make_pbtext.py` keeps only the stop id 2 (`</s>`) of the exporter's 15 entries and sets `supports_thinking` and `supports_function_calling` false.
- **KV budget 4096, six prefill signatures (1024, 256, 64, 16, 4, 1), embedder externalised**. The main TFLite section is 8,088,047,040 bytes (int8) and 4,488,362,336 bytes (int4), both over 2 GiB; not run on iOS.
- **int8 = `dynamic_wi8_afp32` (channelwise int8 weights, float activations); int4 = `BOCTAV4` (blockwise-32 OCTAV int4 + int8 embedding), every scale normal.** GSM8K (questions 0–99, 0-shot step by step, greedy, at most 1024 output tokens, one process per question): bf16 95, int8 97, int4 96 on the Mac GPU; the same weights before the graph rewrite: 95 and 94.
- **The GPU-composite graph by rewrite.** `tools/gpu_graph/gpu_graph_retrofit.py X_plain.litertlm X.litertlm --decomp exporter --mask add_bcast` (step 5, once per file) turns each layer's two DYNAMIC_UPDATE_SLICE KV writes into one `odml.cache_update` and the attention products into `odml.runtime_bmm` (238 and 464 composites), keeps every weight byte and every other section. On the Mac GPU the 1024 / 256 / 2048 workload went from 7.78 to 6.76 s (int8) and from 6.40 to 5.35 s (int4); on the CPU within 1 %.
- Measured (prefill 1024 / decode 256 / context 2048): Mac M4 Max, litert-lm 0.18.0, two passes per file, the median — GPU (WebGPU) int8 784.8 / 46.66 tok/s, int4 787.5 / 61.74; CPU int8 124.05 / 11.49, int4 55.39 / 12.93; every decoder signature fully delegated on the Mac GPU. Galaxy S26 (the prebuilt litert_lm_advanced_main of 2026-09-18, OpenCL), the int4 file in two sessions: prefill 99.98 / 98.36, decode 5.72 / 5.57 tok/s. On litert-lm 0.17.1 both files answered a one-word question on the Mac GPU (one `run` each). The int8 file's Galaxy S26 GPU try ended with the phone restarting during the engine warm-up (2026-10-14), and the Pixel 8a int4 GPU try was stopped by the harness's guard at 400,000 kB of MemAvailable.

### Olmo-3-7B-Think — a sliding-window dense 7B whose int4 scales need a lower bound, and a GPU-composite rewrite that does not apply

`olmo3_work/convert_olmo3_7b_think.sh` converts [allenai/Olmo-3-7B-Think](https://huggingface.co/allenai/Olmo-3-7B-Think) at commit `d97e442d` — Ai2's "Olmo 3 … Think" 7B, the final RLVR model after SFT and DPO — into the three files of [litert-community/Olmo-3-7B-Think](https://huggingface.co/litert-community/Olmo-3-7B-Think): the int8 file, the int4 file and the int4 weights with a 2,048-token cache (the phone form). Environments (released wheels): litert-torch 0.9.4, litert-converter 0.4.0, ai-edge-quantizer 0.9.0, transformers 5.14.1, torch 2.13.0; litert-lm 0.18.0 for the metadata step, the pack and running the files.

```bash
EXPORT_PY=/path/to/python-with-litert-torch-0.9.4 \
RUN_PY=/path/to/python-with-litert-lm-0.18.0 \
LITERT_LM=/path/to/litert-lm \
  bash olmo3_work/convert_olmo3_7b_think.sh out_olmo3_7b_think
```

| file | section | bytes | sha256 |
|---|---|---|---|
| `Olmo-3-7B-Think_int8.litertlm` (7,434,672,480 B) | LlmMetadataProto | 2,547 | `a5241fd35a249b90e4279fd5cbbbb8c6271434dc580bbd3819de13cebb0e6217` |
|  | ExecutorMetadataProto | 4,342 | `8204f6768cf688ea74ad882370d702cd3e2a7e64fd029f2574e089ce1905c50a` |
|  | HF_Tokenizer_Zlib | 1,403,566 | `16308de7d8fb86b623319e70f47466cce8e6c820a49be62cc79c73db35b8b84f` |
|  | TFLiteModel (tf_lite_prefill_decode) | 7,014,027,584 | `7bccf11343093aaa79b6d7eb7aa5ef5db96161076462f4d57225d03e27e1534b` |
|  | TFLiteModel (tf_lite_embedder) | 419,174,752 | `54858deca8f8027e7e0c9e17fff83dac6dcef767b4b00cfcf50423b527b413b3` |
| `Olmo-3-7B-Think_int4.litertlm` (4,309,457,248 B) | LlmMetadataProto | 2,547 | `a5241fd35a249b90e4279fd5cbbbb8c6271434dc580bbd3819de13cebb0e6217` |
|  | ExecutorMetadataProto | 4,342 | `8204f6768cf688ea74ad882370d702cd3e2a7e64fd029f2574e089ce1905c50a` |
|  | HF_Tokenizer_Zlib | 1,403,566 | `16308de7d8fb86b623319e70f47466cce8e6c820a49be62cc79c73db35b8b84f` |
|  | TFLiteModel (tf_lite_prefill_decode) | 3,888,809,568 | `cef4605667184f1bdfbb4373e449000bc3420121ead70fb065744e3e2e0af60d` |
|  | TFLiteModel (tf_lite_embedder) | 419,174,752 | `54858deca8f8027e7e0c9e17fff83dac6dcef767b4b00cfcf50423b527b413b3` |
| `Olmo-3-7B-Think_int4_c2048.litertlm` (4,309,440,864 B) | LlmMetadataProto | 2,547 | `24577128e747b23411b4dff6561abbfaa8874fddce2d20a720c3134c803e57fc` |
|  | ExecutorMetadataProto | 4,342 | `11c3dd26fd5db88ab88d8648ba9e2d78d1e98e8d4a6a8d8c94f87210f94f0ffa` |
|  | HF_Tokenizer_Zlib | 1,403,566 | `16308de7d8fb86b623319e70f47466cce8e6c820a49be62cc79c73db35b8b84f` |
|  | TFLiteModel (tf_lite_prefill_decode) | 3,888,801,952 | `26c7e0b919700535a4adaf2c7977f20ccdd7e744f7f9a7b7c324c6a8c0ff5393` |
|  | TFLiteModel (tf_lite_embedder) | 419,174,752 | `54858deca8f8027e7e0c9e17fff83dac6dcef767b4b00cfcf50423b527b413b3` |

The conversion run that produced these files (the same steps as this script, with the local paths and RAM gates of a shared machine) was rebuilt from the pinned checkpoint in one command on 2026-10-16: the int8 file and the cache-2048 file came back with all five sections raw-identical; the int4 file came back with four of five raw-identical, and its main graph the same size with 17,917 differing bytes — the two RoPE inverse-frequency constants (sliding default, full-layer YaRN) stored in the other order and the operator order of the seven signature subgraphs, the 879 decomposition subgraphs identical; a structural comparison that ignores tensor indices and names finds the seven signatures equal. A run of this script as published is pending and will be recorded here. The file sha256 differs from run to run (the packer writes a new uuid and creation timestamp), so compare sections. Run as published on 2026-10-16 (one command, the environment of its header, 34 minutes on a 128 GB Mac, peak 55.9 GB; the checkpoint from a local cache), this script rebuilt the cache-2048 file with all five sections byte for byte and the int8 and int4 files with four of five; their decoder sections came back the same size and equal by the structural comparison, in the other of the two orders an export writes the RoPE constants and the signature subgraphs in: int8 `4875e643c90fb4f39dce99728b041c84b99eac04b057e91296d4de9b2d845e07` (the shipped file: `7bccf11343093aaa79b6d7eb7aa5ef5db96161076462f4d57225d03e27e1534b`), int4 `35fcfe45352f713465fe5da7be918f09f278b8738aeec4da65f2b880995eeafb` (shipped: `cef4605667184f1bdfbb4373e449000bc3420121ead70fb065744e3e2e0af60d`; this form also came out of the recipe re-run, bit for bit) - a run of yours may land on either hash.

- **Files of this repo the script calls** (as they are at the time of writing; their code is that of the files the conversion run used, apart from comments and the rounding of `repack_metadata.py`'s reported timings): `granite41_work/export_lightweight.py` (sha256 `fef0bb83fd9cb4090c7ed4ffbb4770946b6b07bf713e586bc8cd6fb1cc5a9268`), `granite_work/scan_scales.py` (`e742ab3b0bfaa662e143e68addd2588ecfa3a5542fdb22efa517aeef0e26702e`), `granite_work/repack_metadata.py` (`0f1b2ded56d561fd1b018aee168740711a50cb00812982e3cbb456dc6db4a297`), `tools/tokenizer_parity/swap_tokenizer_section.py` (`2f8749beba92124c652b6a3b9fff90d111a0b52ad3303c5adeb6b3e1bb171476`), `tools/gpu_graph/gpu_graph_retrofit.py` (`ce32549cfceeb5eb0cfe713bc730d954344c76ec7b1592d8dfea341efba60e72`, step 4 only); the rest is in `olmo3_work/`.
- **Stock export path, lightweight conversion.** `Olmo3ForCausalLM` exports through litert-torch's `export_hf` with no custom code: the sliding-window layers get litert-torch's sliding mask (`create_sliding_mask(...) + mask`) over a full-length cache, which for a cache no longer than the 4,096-token window is the causal mask, and the eight full-attention layers get YaRN cos / sin. `granite41_work/export_lightweight.py` with `LIGHTWEIGHT=1` and `RESOURCE_CONSTANT_NUMEL_THRESHOLD=100000000`: every FULLY_CONNECTED weight is an int8 / int4 constant and no op has all-constant inputs. Peaks 52.6–56.9 GB per export on a 128 GB Mac; run one export at a time.
- **A lower bound on the int4 block scales.** Plain `BOCTAV4` wrote 1,574,919 zero and 2,031,784 subnormal fp16 block scales (of 215,210,752) into this model's FC weights, most in the late layers' q_proj; XNNPACK refuses non-normal scales. `olmo3_work/export_clamped_lw.py` floors every blockwise int4 scale at 2^-14 before the weights are rounded: `CLAMP_TOTAL tensors=55 scales=3606703`, and the scan of the result finds every scale normal (3,611,960 at exactly 2^-14).
- **Ai2's template, edited for the runtime's template engine.** `olmo3_think_dual.jinja` (built by `make_template.py`) is Ai2's chat_template.jinja with the content-parts macro and `.get` rewritten as `is defined` tests; its renders equal transformers' render of Ai2's template in 9 conversation shapes × 3 `enable_thinking` states × both history forms, as strings and as content parts. Its default system message is the template's ("You are OLMo, a helpful function-calling AI assistant built by Ai2. …"), which differs from the README's "Default System Message" section. The generation prompt ends `<|im_start|>assistant\n<think>`: every reply starts in the thought, and `--thinking false` does not change that. No start token: the engine's prefill count equals transformers'. `make_pbtext.py` keeps the stop ids 100265 and 100257 of the exporter's 16 entries and sets `supports_thinking` true and `supports_function_calling` false; the thought channel `<think>` / `</think>` is declared by the exporter.
- **Tokenizer section = the repo's tokenizer.json.** The export writes transformers' re-serialization (the post_processor differs); `tools/tokenizer_parity/swap_tokenizer_section.py` puts the repo's bytes in, `tok_section.py` checks.
- **The GPU-composite rewrite does not apply.** `tools/gpu_graph/gpu_graph_retrofit.py --decomp exporter --mask add_bcast` (step 4, `GPU_GRAPH=1`) refuses this graph — the attention-mask input also feeds the sliding-window mask construction — so the published files are the exporter's graphs.
- **KV budget 4096, six prefill signatures (1024, 256, 64, 16, 4, 1), embedder externalised.** The main TFLite section is 7,014,027,584 bytes (int8) and 3,888,809,568 bytes (int4), both over 2 GiB; not run on iOS. The KV cache is multi-head (32 × 128 per layer, full length on every layer).
- **int8 = `dynamic_wi8_afp32`; int4 = `BOCTAV4` with every block scale at least 2^-14.** GSM8K (questions 0–99, 0-shot step by step, greedy, at most 2,048 output tokens, one process per question; the answer after `</think>`, a reply whose thought did not close scored on its whole text): bf16 84 (26 replies with the thought still open at 2,048 tokens), int8 81 (32), int4 80 (37) on the Mac GPU; the cache-2048 file 77 at 1,900 / 2,048 tokens (47).
- Measured (prefill 1024 / decode 256 / context 2048): Mac M4 Max, litert-lm 0.18.0, two passes per file, the median — GPU (WebGPU) int8 838.81 / 30.40 tok/s, int4 836.55 / 34.76; CPU int8 131.36 / 6.70, int4 64.28 / 7.11; every decoder signature fully delegated on the Mac GPU. On litert-lm 0.17.1 both files answered a one-word question on the Mac CPU (one `run` each). The int8 file was not run on the Galaxy S26 (another 8B int8 file dropped that phone during its GPU engine warm-up), and the Pixel 8a was not run (the int4 file plus its KV cache is more than that phone's free RAM); the Galaxy S26 int4 GPU rows are pending.

## granite-4.2-3b (dense, thinking) — the think prefill an extracted template silently drops

`granite42_work/convert_granite42_3b.sh` converts IBM's compact reasoning model (3.66B dense, Aug 2026) on a pristine released stack (litert-torch 0.9.3 / litert-converter 0.4.0 / ai-edge-quantizer 0.9.0 / litert-lm-builder 0.16.1, transformers 5.14.1). Published: [litert-community/granite-4.2-3b](https://huggingface.co/litert-community/granite-4.2-3b). **Requires litert-lm ≥ 0.16 to run.**

```bash
pip install "litert-torch==0.9.3" "litert-converter==0.4.0" "ai-edge-quantizer==0.9.0" "litert-lm-builder==0.16.1" "transformers==5.14.1"
python -c "from huggingface_hub import snapshot_download; snapshot_download('ibm-granite/granite-4.2-3b', local_dir='src_models/granite-4.2-3b')"
granite42_work/convert_granite42_3b.sh    # int8 + int4, six prefill signatures, NO_START_TOKEN=1
python tools/add_thought_channel.py granite42_work/out_int8/model.litertlm granite-4.2-3b_int8.litertlm --start '<think>' --end '</think>'
```

4.2 returns from the 4.0/4.1 hybrid line to full attention, so the dense 4.1 rail carries over — six-signature prefill ladder (an eleven-signature build of this shape is killed by iOS at Metal engine init), externalized embedder, `NO_START_TOKEN=1` (4.2 declares `<s>` as BOS but its tokenizer never prepends it — the post-processor adds nothing — so the builder's unconditional `start_token` write must be suppressed). Both quantizations gate **8/8 on CPU and GPU** (Mac, litert-lm 0.16.0), answers arriving cleanly through the thought channel; measured throughput is on the model card.

**What 4.2 adds is the reasoning machinery, and two conversion steps are load-bearing:**

- **The think prefill must survive template extraction.** IBM's template ends the generation prompt with `<|im_start|>assistant\n<think>\n`. The exporter's `parse_chat_template` derives the assistant prefix from an assistant *history* message (`add_generation_prompt=False`), so a jinja that opens `<think>` only in its generation branch exports a bundle whose `model.prefix` is a bare `<|im_start|>assistant\n` — the scaffold is silently gone, and think-discipline is left to the quantized model's choice. `templates/granite42_think.jinja` therefore puts the opener in the assistant history branch too (the chatml_think shape). After any export of a thinking model, read `prompt_templates.model.prefix` out of the bundle before trusting it.
- **The thought channel must be declared.** The structured-template path emits no `LlmMetadata.channels`; without it the runtime streams raw reasoning into the answer and silently ignores any thinking budget. `tools/add_thought_channel.py` adds the block post-export and refuses to pass if anything but `channels` changed.

`granite42_work/hf_oracle.py` is the bf16 reference for the same eight gate questions (scored on the text after `</think>` — the reasoning routinely contains the expected string, e.g. it repeats "0.9" while comparing 0.9 vs 0.11, so scoring the whole output would fake a pass). `granite42_work/gate8q.py` applies the same prefilled-opener-aware scoring to the converted bundle.

### 2026-09-15 — bundle the canonical Jinja chat template (metadata-only repack)

The files published on 2026-08-31 carried only the structured `prompt_templates` (no `jinja_prompt_template`). On LiteRT-LM 0.16.0 and 0.17.0 a bundle without a Jinja template gets its history re-rendered by the runtime's default template, which renders history turns with role `model` while the engine stores its replies as `assistant`, so every assistant turn drops out of the re-rendered history and long conversations drift (LiteRT-LM #3445). The fix is to bundle a Jinja template. `granite42_work/models_granite4_2/` is the canonical template directory in the shape of LiteRT-LM `models/lfm2/` (template, `LlmMetadataProto.pbtext`, README, BUILD, testdata + goldens; `bazel test //models/granite4_2:chat_template_test` passes in a LiteRT-LM checkout at a4673247 with the directory copied in), and the published int4/int8 were re-packed with it, weights and tokenizer byte-identical:

```bash
LITERT_LM_BIN=~/venvs/lt0170run/bin/litert-lm python3 tools/chat_template/set_jinja_template.py \
  granite-4.2-3b_int4.litertlm out/granite-4.2-3b_int4.litertlm granite42_work/models_granite4_2/chat_template.jinja
```

The template follows IBM's `chat_template.jinja` (ChatML with the always-present system turn, IBM's `<tool_call>`/`<function=…>` XML, `enable_thinking` toggle) with one deliberate difference: past assistant turns always render as `<think></think>` + answer and never carry their reasoning, because a position-dependent history fails the runtime's prefix check. Probe (issue's 3-turn script, 0.17.0, greedy): before `OK` / `OK 42` / `42 Alfred orange`, after `OK` / `42` / `Alfred, orange.`.

## Qwen2.5-Coder-1.5B-Instruct — a 1.5B code model at 1.12 GB, and why the size is the recipe

`qwen25coder_work/convert_qwen25_coder.sh`. Published: [litert-community/Qwen2.5-Coder-1.5B-Instruct](https://huggingface.co/litert-community/Qwen2.5-Coder-1.5B-Instruct). **Requires litert-lm ≥ 0.16.**

Decode on this runtime is memory-bandwidth-bound, which has a blunt consequence: putting a bigger model on the GPU does not make it fast, and halving the weights does. Measured on one machine and one protocol (M4 Max, `-p 256 -d 256 --cache no`, quiet, serialized) — this 1.12 GB build against a 2.19 GB 3B-class int4 build converted the same day:

| | Qwen2.5-Coder-1.5B (1.12 GB) | 3B-class int4 (2.19 GB) |
|---|---|---|
| prefill | **3037 tok/s** | 1241 |
| decode | **137.8 tok/s** | 86.3 |
| TTFT | **0.099 s** | 0.233 s |
| init | **2.38 s** | 5.96 s |

**`EXTERNALIZE_EMBEDDER=1` is required.** This model ties its embedding and `lm_head`, so `BOCTAV4` — int4 on every FULLY_CONNECTED, int8 on EMBEDDING_LOOKUP — describes one tensor two ways. The quantizer resolves the conflict by copying the 151936×1536 table once per prefill signature: seven copies, 1.63 GB, **65% of a 2.53 GB file**, with nothing logged and "int4" still true of the linears. `check_bundle_sanity.py <bundle> --params N` makes that a one-line check (exit 1 on duplication); a healthy int4 build lands near 0.7 bytes/param.

**The chat template carries a default system prompt.** Upstream inserts `You are Qwen, created by Alibaba Cloud. You are a helpful assistant.` whenever the caller supplies no system message. A generic ChatML template renders the markers without it, so the shipped bundle would run the model outside the state it was tuned in on every default-path request — quietly. `qwen25_coder_simple.jinja` bakes it into the user prefix and renders byte-identical to upstream for a single turn.

**Gate a code model on code.** An 8-question general-knowledge check certifies nothing here. `eval_code.py` asks for six small functions and **executes** each against assertions — a task passes only if the code imports and every assertion holds, because generated code that reads correctly and raises on the second assertion is exactly how a damaged quantization survives a skim. This build scores 6/6, and 8/8 on the general gate.

## S1-mini (0.6B ASR-transcript normalizer) — the template flag a runtime can't pass, and the stop strings that eat your punctuation

`s1mini_work/convert_s1mini.py` converts [superwhisper/s1-mini](https://huggingface.co/superwhisper/s1-mini) (published: [mlboydaisuke/S1-mini-LiteRT](https://huggingface.co/mlboydaisuke/S1-mini-LiteRT)) — Superwhisper's Qwen3-0.6B finetune that rewrites raw dictation into clean written text — on the released stack (litert-torch 0.9.3, litert-lm-builder ≥ 0.15). One command, int8, 688 MB. License note: Apache 2.0 plus a naming clause — any distribution must keep identifying the model as "S1-mini" by "Superwhisper".

```bash
pip install "litert-torch==0.9.3" "litert-lm-builder>=0.15" "transformers>=4.51" torch
python s1mini_work/convert_s1mini.py            # download (pinned rev) + export + stop-token fix
python s1mini_work/gate_normalize.py out/s1-mini-int8/model.litertlm --backend cpu
```

**The model needs `enable_thinking=False`, and no runtime will ever pass it.** S1-mini keeps Qwen3's template, whose default renders the thinking-mode prompt; the card warns that form usually yields no usable output, and the trained form is the `enable_thinking=False` render (assistant turn opens with an empty `<think>` scaffold). A LiteRT-LM engine applies the embedded template with no such kwarg — so the verbatim vendor template ships a broken model. The fix is resolved at export time: `s1mini_bundle.jinja` hardcodes the non-thinking render, and also bakes in the exact system prompt the card requires (clients that can't send a system message — most simple chat UIs — get the trained format anyway; clients that send the card's wording lose nothing, duplicates are dropped). Render equivalence to the vendor template with `enable_thinking=False` was verified 12/12 before export, and a follow-up turn's render is a string-extension of the previous turn plus reply, so KV reuse stays valid.

**The exporter's composite stop strings silently eat final punctuation.** litert-torch expands the template's turn suffix into punctuation-prefixed stop variants (`".<|im_end|>\n"`, `"?<|im_end|>\n"`, …) to cope with SentencePiece greedy merging. Qwen3 is BPE — that merge never happens — but the runtime still matches the multi-token stop sequence and strips the **whole** match, so a reply ending `…Thursday."` comes back `…Thursday`. On a punctuation-restoration model this is fatal, and a general-knowledge gate can never see it (regexes match content words, not a trailing period). Measured cleanly: 7 of 7 task cases whose reference ends in "." lost exactly that period, and a metadata-only rewrite (drop every `token_str` stop, keep the two real stop ids) flipped all 7 back. The convert script does that rewrite automatically. If you export other BPE models with this stack, check your bundles.

**int8 only, and that is a measurement, not a guess.** The int8 build reproduces the HF fp32 greedy outputs **byte-for-byte, 10/10 task cases, on CPU and GPU** (`gate_normalize.py` feeds both sides the identical rendered string). int4-b32 was built and rejected: it drops the comma before "and" and discourse words like "hmm" (identically on both backends), and decodes *slower* than int8 at 0.6B (24 vs 31 tok/s CPU, 88 vs 105 tok/s Mac GPU) — block-dequant overhead beats the bandwidth saving at this size. Also note s1-mini ties its embedding and lm_head, so an int4-FC + int8-embedding recipe triggers the vocab-table duplication defect described in the Qwen2.5-Coder section (656 MB "int4", larger than int8; `externalize_embedder=True` clears it) — another reason int8 is the honest recipe here.

**Gate it on its own task.** S1-mini is not a chat model: asked "What is 17 + 25?" it will normalize the question, not answer it, so the 8-question gate certifies nothing. `gate_normalize.py` runs 10 dictation-normalization cases (all four registers, lists, email, numbers/time/currency, and the pure-filler case whose correct output is the empty string) against HF fp32 on the identical rendered string, with a BOS A/B check first. Mac numbers for this build (p256/d256, runs 3, `--cache no`, litert-lm 0.16.0, M4 Max): GPU 3777 prefill / 146.7 decode tok/s, TTFT 0.075 s; CPU 466 / 34.6, TTFT 0.58 s.

## granite-speech-5.0-470m-turboctc (CTC conformer ASR → plain .tflite)

First ASR in the catalog. `granite_speech_work/` converts IBM's 473M CTC conformer to plain LiteRT `.tflite` — one encoder forward, no decoder loop; the host does argmax-collapse-decode in ~10 lines. Published: [litert-community/granite-speech-5.0-470m-turboctc](https://huggingface.co/litert-community/granite-speech-5.0-470m-turboctc) (int8 518 MB + fp16 994 MB).

```sh
hf download ibm-granite/granite-speech-5.0-470m-turboctc --local-dir granite_speech_work/hf_model
python3 granite_speech_work/fetch_fixtures.py         # 20 LibriSpeech dev-clean clips
python3 granite_speech_work/eager_gate.py             # oracle: WER 3.79% + buffer checks
python3 granite_speech_work/convert_granite_speech_ctc.py
python3 granite_speech_work/verify_tflite.py          # 20/20 transcript match per variant
```

Env: torch 2.12 + torchaudio 2.11 (the processor's mel front-end needs torchaudio) + transformers 5.14 + litert_torch 0.9.3 + ai-edge-quantizer + ai-edge-litert.

**The Hub repo is two-faced (as of 2026-09-01), and `AutoModel` fails on released transformers.** `config.json`/`model.safetensors` are re-exported for the unreleased stock `GraniteSpeech5ForCTC` (model_type `granite_speech5_ctc`, HF-standard tensor names, no `auto_map`), while the bundled remote-code `.py` files are the older packaged `CtcConformerForCTC` with stock-3.3 granite_speech naming. `load_model.py` loads through the remote code with a 16-rule full-key state-dict rename; the nasty pair is `conv.norm.*` (BatchNorm) → `conv.batch_norm.*` while `norm_conv.*` (LayerNorm) → `conv.norm.*` — rename per full key in one pass, never cascading prefix rewrites. `load_state_dict(strict=True)` over all 550 tensors proves the mapping; a 20-clip WER gate proves the wiring.

**Shaw's re-association is the difference between a 923 MB and a 518 MB int8 file.** The encoder's rel-pos bias `einsum(q, rel_pos_emb(dists[:blk,:blk]))` constant-folds under `torch.export` into a `[blk, blk, 128]` fp32 constant per layer × per distinct block length × per signature (~0.5 GB across the 3-signature export) that FC-only dynamic int8 cannot reach (it feeds BATCH_MATMUL). Rewriting as `gather(q @ W.T, dists)` (`shaw_patch.py`) keeps the single shared `[1025, 128]` table and is **bitwise identical** in eager on all three windows. Cost: 17 GATHER_ND ops — the mobile GPU delegate then refuses to compile the graph (measured on a Galaxy S26), which is fine because this is a CPU ship.

**Fixed windows are semantically cheap because attention is block-local (128 frames), but not free.** Signatures `transcribe_{5,10,30}s` zero-pad audio to the window; padding shifts the 128-frame block boundaries, and on the 20-clip fixture set 19/20 transcribe identically vs unpadded with one single-word change (corpus WER 3.79% → 4.24%). Route each clip to the smallest window that fits.

**Verification is transcript-exact, not just cosine-close.** int8 and fp16 both match the fp32 eager transcripts 20/20; on a Galaxy S26 (CompiledModel CPU) the transcripts are again exact on every signature and the int8 logits are bit-identical to the desktop run of the same file (max abs diff 0.0). Speed: int8 transcribes a 30 s window in 239 ms on an M4 Max (8 threads) and 666 ms on the S26 — ~126× / ~45× real-time; fp16 is ~3× slower on CPU (XNNPACK fp32 repack), use int8 on device.

## VibeVoice-ASR-BitNet (speech-to-text LLM, audio-in `.litertlm`) — first audio-in bundle in the catalog

`vibevoice_asr_work/` converts Microsoft's [VibeVoice-ASR-BitNet](https://huggingface.co/microsoft/VibeVoice-ASR-BitNet) (σ-VAE acoustic + semantic conv encoders at 24 kHz → a Qwen2.5-1.5B-shaped **BitNet/ternary** LM, MIT) into one `.litertlm` that LiteRT-LM's released runtime (≥ 0.16.1) drives end-to-end through its **generic audio path** — the runtime decodes and resamples the clip, frames raw PCM into 3200-sample frames, runs the bundled audio encoder, and splices the 7.5 Hz embeddings into the ChatML prompt. No runtime patch, no litert-torch patch.

```sh
hf download microsoft/VibeVoice-ASR-BitNet model-00001-of-00003.safetensors model-00002-of-00003.safetensors \
   model-00003-of-00003.safetensors config.json tokenizer.json tokenizer_config.json vocab.json generation_config.json \
   --local-dir vibevoice_asr_work/hf
python3 vibevoice_asr_work/fetch_fixtures.py                    # 20 LibriSpeech dev-clean clips at 24 kHz
python3 vibevoice_asr_work/load_bitnet.py                       # legacy -> transformers-native, ternarize, strict load; hf_native/ + lm_native/
python3 vibevoice_asr_work/eager_gate.py                        # fp32 oracle: WER 2.68 % (12/448)
python3 vibevoice_asr_work/export_audio_encoder.py --seconds 30 --quant fp32,wi8fc
EXTERNALIZE_EMBEDDER=1 CACHE=2048 PREFILL=512,128,32 \
  python3 scripts/export_simple_template.py vibevoice_asr_work/lm_native out/vibevoice-lm templates/chatml_simple.jinja BMIX4_128
litert-lm unpack out/vibevoice-lm/model.litertlm --output-dir out/vibevoice-lm/unpack   # the two LM tflites
python3 vibevoice_asr_work/patch_tokenizer.py vibevoice_asr_work/lm_native/tokenizer.json vibevoice_asr_work/lm_native/tokenizer_audio.json
DEC=out/vibevoice-lm/unpack EMB=tf_lite_embedder DECODE=tf_lite_prefill_decode \
  ENC=vibevoice_asr_work/out/audio_encoder/audio_encoder_30s_wi8fc.tflite \
  TOK=vibevoice_asr_work/lm_native/tokenizer_audio.json OUT=out/vibevoice-bundle \
  python3 vibevoice_asr_work/build_bundle.py
python3 vibevoice_asr_work/mac_gate.py --bundle out/vibevoice-bundle/VibeVoice-ASR-BitNet.litertlm   # runtime gate: WER 2.68 %
```

Env: torch 2.13 + transformers 5.14.1 (has `vibevoice_asr` natively) + litert-torch 0.9.4 + ai-edge-quantizer 0.9.0 + litert-lm-builder 0.16.1 for everything but the LM export, which goes through the lane's usual `export_simple_template.py` (litert-torch 0.9.3); the gate needs `litert-lm-api` 0.16.1 (python) — `litert-lm` 0.16.0 supplies `unpack`/`benchmark`.

**"BitNet" is applied at conversion, per tensor, and it is the model.** The Hub checkpoint stores fp32 latent weights; VibeASR.cpp (the vendor runtime) ternarizes the 7 projections per LM layer with a per-tensor absmean scale (`s = 1/mean|w|`, `round(w·s).clamp(−1,1)/s`) at GGUF-conversion time. `load_bitnet.py` does exactly that (α 0.030–0.070, 37 % zeros); run un-ternarized, the LM emits `<|im_end|>` spam. A per-tensor ternary is exactly representable by int4 blockwise min-max at any block size (every block is {−α, 0, +α} → scale α/7), so the LM ships as `BMIX4_128` — the same driver every dense LLM in this repo uses — and the Mac runtime transcript WER equals the fp32 oracle to the word.

**The tokenizer has no string for the speech markers.** The repo ships the Qwen2 tokenizer (added tokens end at 151645) while the model was trained with 151646/151647/151648 (`<|object_ref_start|>` / `<|object_ref_end|>` / `<|box_start|>` in Qwen2.5) as speech start/end/pad — VibeASR.cpp hard-codes the ids. The generic runtime path tokenizes `start_of_audio_token` and `audio_suffix` as text, so `patch_tokenizer.py` adds the three strings as special tokens; they land on exactly those ids by construction (asserted), ordinary text tokenizes identically.

**The audio encoder is one single-signature tflite with the vendor's normaliser folded in.** `audio` f32 `[1, 225, 3200]` (30 s of raw 24 kHz PCM as the runtime frames it, zero-padded) → `features` f32 `[1, 225, 1536]`: RMS-normalise to −25 dBFS over the non-zero frames + peak clip, acoustic encoder (mean latents — the HF class adds `vae_std·randn` even at inference; the vendor runtime uses the mean), semantic encoder, projector. The output must be named literally `features` (return a dict — positional returns become `output_0`) and every reduction keeps `keepdim=True` (the GPU delegate rejects rank-0 tensors). int8 dynamic on the linears (`wi8fc`, 951 MB; the vendor ships the VAE at int8) is WER-lossless; the window is 30 s because a 10 s window costs 7 words on the 20-clip set from context resets at chunk boundaries (the engine chunks by window with no overlap).

**Bundle metadata is the runtime contract, written by hand in `build_bundle.py`:** `GenericModel{audio_enabled, delimiter_regex/audio_token_regex on <|box_start|>, start_of_audio_token, audio_suffix, skip_mel_spectrogram_extraction, 24000 Hz, frame = hop = 3200}` + ChatML templates + a jinja that emits the vendor system prompt and renders an audio item as `<|box_start|>\n` (default instruction "Please transcribe it." when the turn has no text). Without an audio adapter section the executor treats the encoder as a streaming encoder with window = 225 frames and (buffering off by default) zero-pads short clips; valid tokens = valid frames.

**Gates.** fp32 oracle 12/448 = 2.68 % (vendor prompt; adding the generation prompt changes nothing; dropping the duration sentence 2.90 %); encoder int8 through the eager LM 2.68 %; LiteRT-LM 0.16.1 python CPU 2.68 %, LM on Metal GPU 2.46 %; Pixel 8a `litert_lm_advanced_main` v0.16.1 CPU 3.12 % (one clip flips "On the" → "Under" — Arm int8 numerics), Galaxy S26 CPU see the card. LM speed: M4 Max CPU 357 / 59.3 tok/s, Metal 2045 / 138.6 tok/s (`-p 256 -d 256 --cache no`). The audio encoder is **CPU-only** on Metal/WebGPU: the 30 s stem dispatches 90 001 workgroups against a 65 535 limit and the engine returns empty text — a shape problem (a ≤ 5.5 s window would fit), not an op problem.

## VibeVoice-ASR-Streaming-1.5B (streaming speech-to-text LLM, audio-in `.litertlm`, one window per turn)

`vibevoice_asr_streaming_work/` converts Microsoft's [VibeVoice-ASR-Streaming-1.5B](https://huggingface.co/microsoft/VibeVoice-ASR-Streaming-1.5B) (the same σ-VAE acoustic + semantic conv encoders as VibeVoice-ASR-BitNet, a dense Qwen2.5-1.5B-shaped LM, MIT) into a `.litertlm` that LiteRT-LM's released runtime (≥ 0.16.1) drives through its **generic audio path as a multi-turn conversation** — one 3.47 s audio window per user turn, the model's chunk text + `<|text_chunk_end|>` as the model turn. No runtime patch, no litert-torch patch. Published: [litert-community/VibeVoice-ASR-Streaming-1.5B](https://huggingface.co/litert-community/VibeVoice-ASR-Streaming-1.5B).

```sh
hf download microsoft/VibeVoice-ASR-Streaming-1.5B --include '*.safetensors' '*.json' merges.txt --local-dir vibevoice_asr_streaming_work/hf
python3 vibevoice_asr_work/fetch_fixtures.py                    # 20 LibriSpeech dev-clean clips at 24 kHz (shared with the BitNet lane)
python3 vibevoice_asr_streaming_work/precheck.py --src vibevoice_asr_streaming_work/hf --stage load,enc,eager --n-clips 20 --out vibevoice_asr_streaming_work/out
#   load  : legacy checkpoint -> transformers-native class (strict), saves out/lm_native/ (dense fp32 Qwen2ForCausalLM)
#   enc   : audio_encoder_26f_fp32.tflite (26-frame window, no normaliser), parity vs eager cos 1.00000
#   eager : the vendor streaming protocol in PyTorch -> WER 7.59 % (34/448); the one-shot variant transcribes the last window only
python3 vibevoice_asr_streaming_work/quant_encoder.py           # -> audio_encoder_26f_wi8fc.tflite (int8 dynamic FC, cos 0.99954)
EXTERNALIZE_EMBEDDER=1 CACHE=2048 PREFILL=512,128,32 \
  python3 scripts/export_simple_template.py vibevoice_asr_streaming_work/out/lm_native out/vvs-lm-int4 templates/chatml_simple.jinja BMIX4_128
EXTERNALIZE_EMBEDDER=1 CACHE=2048 PREFILL=512,128,32 \
  python3 scripts/export_simple_template.py vibevoice_asr_streaming_work/out/lm_native out/vvs-lm-int8 templates/chatml_simple.jinja dynamic_wi8_afp32
litert-lm unpack out/vvs-lm-int4/model.litertlm --output-dir out/vvs-lm-int4/unpack   # the two LM tflites (same for int8)
DEC=out/vvs-lm-int4/unpack ENC=vibevoice_asr_streaming_work/out/audio_encoder_26f_wi8fc.tflite \
  TOK=vibevoice_asr_streaming_work/out/lm_native/tokenizer.json OUT=out/vvs-bundle-int4 \
  python3 vibevoice_asr_streaming_work/build_bundle.py            # bare-prompt jinja, model suffix <|text_chunk_end|>, stops 151665/151643
python3 vibevoice_asr_streaming_work/mac_gate.py --bundle out/vvs-bundle-int4/VibeVoice-ASR-Streaming-1.5B.litertlm --limit 20   # runtime gate, one turn per window: WER 7.81 % (int4) / 7.14 % (int8)
python3 vibevoice_asr_streaming_work/pixel_gate.py --push-windows --backend gpu --audio-backend cpu                              # Android CLI, --multi_turns, one [audio:] line per window
```

Env: torch 2.13 + transformers 5.14.1 (native `vibevoice_asr`) + litert-torch 0.9.4 + ai-edge-quantizer 0.9.0 + litert-lm-builder 0.16.1 for everything but the LM export (`export_simple_template.py`, litert-torch 0.9.3); `mac_gate.py` is pure Python (litert-lm-api 0.16.1 or 0.17.1; no numpy needed) — `litert-lm` 0.16.0 supplies `unpack`/`benchmark`.

**It is a protocol, not a prompt — and the single-turn bundle contract does not fit it.** The vendor's `streaming_generate` feeds a bare text prompt (no ChatML), then per window `<|object_ref_start|>` + 26 audio embeddings + `<|object_ref_end|>`, decodes until `<|text_chunk_end|>` (151665) or EOS, force-feeds `<|text_chunk_end|>` and continues on the same KV cache. Measured in eager: given all of a clip's audio inside one marker pair and then asked to generate, the model transcribes the LAST 3.5 s only (3/3 clips). So `build_bundle.py` sets every `prompt_templates` prefix/suffix to the empty string except `model.suffix = <|text_chunk_end|>`, a jinja that renders the vendor prompt once and an audio item as `<|box_start|>`, and stop tokens {151665, 151643}. In the runtime that reproduces the loop exactly: the sampled stop token never enters the KV, the next turn's render appends the suffix (= the vendor's force-feed), and prefix dedup keeps the earlier audio turns (token_count grows 73 → 113 → 152 → 191 → 222 over five windows; nothing is re-encoded).

**Window 26 frames, overlap on the app side, no normaliser.** `preprocessor_config.json`: `chunk_frames 22`, `lookahead_frames 4`, `normalize_audio false`. The encoder tflite is stateless per window (`audio` f32 `[1, 26, 3200]` → `features` `[1, 26, 1536]`); the app sends 83 200 samples per turn and advances 70 400 — the runtime's own overlapped windowing exists only for the Gemma-3n adapter + `feature_states` contract, so without an adapter stride == window. Latents are the mean (the vendor Python demo samples `fix_std 0.5` noise at inference). The -25 dBFS RMS block of the BitNet encoder is dropped.

**Dense LM, standard recipe.** Same Qwen2.5-1.5B shape as the BitNet ship but dense bf16 weights, no ternarization: int4 blockwise-128 costs 3 words / 448 against int8 on the Mac CPU (7.81 % vs 7.14 %; fp32 eager 7.59 %) and saves 780 MB and ~650 MB of peak RAM on the phone, so int4 is the phone file and int8 the desktop file.

**Gates (20 clips, 448 words, 73 windows).** fp32 eager 34/448; Mac 0.16.1 int8 CPU 32 / LM-on-Metal 33, int4 CPU 35 / Metal 31; Galaxy S26 `litert_lm_advanced_main` v0.16.1 `--multi_turns` int8 CPU 32 / OpenCL-LM 31, int4 CPU 35 / OpenCL-LM 32; 0.17.1 Python identical to 0.16.1 on the clips checked. The ~7 % is the model's own 3.47 s-window streaming trade-off (the non-streaming BitNet bundle scores 12/448 on the same set). The audio encoder is CPU-only: on Metal/WebGPU and Adreno it returns empty text even with fp32 activations (the 26-frame window is inside the Metal dispatch limit, so this is a conv-stack issue, not the BitNet lane's workgroup-count wall).

## Bonsai 2 27B (PrismML ternary Qwen3.8-27B, Hadamard-rotated) — 2026-09-19

`bonsai2_work/` converts [prism-ml/Ternary-Bonsai-2-27B-mlx-2bit](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-mlx-2bit) (the exact 2-bit g128 values; there is no unpacked release) to int4-blockwise `.litertlm` bundles for [LiteRT-LM](https://github.com/google-ai-edge/LiteRT-LM) ≥ 0.15. Text only; Mac-class files (16.5 GB at block 32, 15.3 GB at block 128). One command, the steps it runs and every trap are in `bonsai2_work/FINDINGS.md`:

```bash
bash bonsai2_work/convert_bonsai2.sh     # dl (8.6 GB) -> dequant -> float export (26 min, 119 GB peak RAM) -> int4 bundle
```

Same rail as the Qwen3.5 hybrids above (`qwen35_work/litert-torch-qwen35` = litert-torch 115a136 + the hybrid patch) plus `hadamard_export.py`, which puts PrismML's block-1024 activation rotation into the graph as MUL + RESHAPE + FULLY_CONNECTED(H/32) + RESHAPE. Three things that are not in any earlier lane: the MLX pack stores the RMSNorm weights as `1 + w` (transformers' `Qwen3_5RMSNorm` adds the 1 itself), mlx `Conv1d` weights are `[out, k, in]`, and the quantizer's exclusion regex must name the rotation FC only (`Linear_hadamard_rotation`), not the wrapper class. Gates on the 27B: 8-question check 8/8 on CPU and GPU (both block sizes), two-turn chat, logits vs PrismML's bundled MLX loader (24/24 top-1, Pearson 0.99988), GSM8K n=100 greedy thinking-off 96/100 against 95/100 on PrismML's own runtime (same prompt, the 4 misses shared), `litert-lm benchmark` on an M4 Max (b32 GPU 133 / 14.2 tok/s, b128 GPU 136 / 16.5, CPU 14.6 / 4.79 and 19.1 / 5.32). Evaluate this family one process per question (LiteRT-LM #3165).

## Chat templates vs the 0.18 content-parts form — metadata-only re-ship of the string-only bundles (2026-09-21)

LiteRT-LM 0.18 passes every message `content` to the bundle's Jinja template as a list of typed parts; a string-only template then drops the prompt, raises on `+`, or prints the list into the prompt (details, per-family templates and the swap tool in [`tools/chat_template_parts/`](tools/chat_template_parts/)). The bundles listed in `tools/chat_template_parts/families.tsv` are re-published with their template swapped for the dual-form version and nothing else changed:

```bash
python -m venv .venv-swap && .venv-swap/bin/pip install litert-lm==0.17.1
.venv-swap/bin/python tools/chat_template_parts/swap_template.py in.litertlm out.litertlm --jinja tools/chat_template_parts/dual/<sha12>.jinja
# proves: section list unchanged, TFLite bytes identical, tokenizer + executor metadata identical, LlmMetadata equal except jinja_prompt_template
```

Gate per file before upload: `litert-lm run` on 0.17.1 and on the 0.18 nightly answer the same one-turn question; the 0.17.1 answer of the swapped bundle is byte-identical to the original's (greedy). The card of each repo carries the line "chat template updated to accept the 0.18 content-parts form (string form unchanged); weights, tokenizer and executor metadata byte-identical".

## decider-0.8b (Mapika) — a System One decision model on the Qwen3.5-0.8B hybrid rail: the readout is the recipe (2026-09-23)

Published: [litert-community/decider-0.8b-LiteRT](https://huggingface.co/litert-community/decider-0.8b-LiteRT) — `decider-0.8b_fp16.litertlm` (fp16 float-casting, exact: 120/120 rows, max |Δp| 6.6e-6 vs the fp32 oracle; a desktop CPU file) and `decider-0.8b_int8.litertlm` (the house dynamic-int8 recipe, 963 MB; Android CPU or GPU). The model answers typed questions with calibrated option probabilities from one forward pass (no generation), so the gate is the probability, not a generated text: 40 synthetic fixtures → 120 rows, unrounded |Δp| against the frozen fp32 `Decider.system_one` output. Recipe, scripts, fixtures, evidence and the one-command rebuild: [`decider_work/`](decider_work/) (`bash decider_work/reproduce.sh`). Three findings that generalise:

- **Dynamic int8 costs calibration on CPU, not on GPU.** The same file: CPU 117/120, max |Δp| 0.160, p95 0.068 (the FC layers quantize activations dynamically); Mac Metal GPU with one padded prefill chunk 118/119, 0.073, 0.025 (the GPU runs the int8 weights in float). fp16 casting is exact but XNNPACK expands it to fp32 on CPU (3.0 GB weight cache; 6.1 GB on a Galaxy S26 at 262 tokens), so fp16 stays a desktop file.
- **On a GPU, chained prefill chunks corrupt the carried state of this hybrid.** Mac Metal, fp32 activations: multi-chunk max |Δp| 0.177 vs one padded chunk 9.8e-6 vs decode-walk 5.9e-6, while the CPU agrees across the three schemes within 8.9e-6. A GPU readout feeds one padded chunk (rows ≤ 1024 tokens); longer rows run on CPU.
- **Two weight forms that look right and are not**: fp16 lm_head + int8 embedding on the tied vocab table makes the quantizer store the int8 table once per subgraph (12×, 4.7 GB); a weight-only int8 lm_head sharing the table fails to compile on Metal (shape mismatch). fp16 FCs + dynamic-int8 lm_head + int8 embedding (1.41 GB) compiles on Metal (p95 0.005) and delegates fully on the S26 CL delegate — built and gated, not published.

Galaxy S26 (`litert_lm_advanced_main` v0.16.0, one cold run each): int8 GPU 262 tokens 300 tok/s / TTFT 0.91 s / engine init 78 s / 4553 MiB (188522/188522 ops delegated), 503 tokens 567 tok/s / 0.97 s / 97 s / 5317 MiB; int8 CPU 262 tokens 288 tok/s / 0.94 s / 14 s / 1941 MiB. For a prefill-dominated decision model the phone GPU buys no time at ~260 tokens; its value is calibration. A weight-only int8 build (p95 0.025 on CPU) is not published: the S26 kernel-panicked ~70 s after engine start on two days (`dumpsys dropbox` SYSTEM_LAST_KMSG), which from the host looks like a USB drop.

Recipe note that applies to every hybrid recipe in this file: the pinned litert-torch base `115a13607c730c81018bb9789138a3e5e5119e3d` exists only on the `john-rocky/litert-torch` fork (upstream/main never contained it); the clone lines above were corrected to the fork on 2026-09-23.

## Qwen3-0.6B — the dynamic INT4 (block-32) GPU bundle on litert-community, refreshed: a stock litert-torch export with the GPU composites and a chat template that takes content parts (2026-09-20)

Published: [litert-community/Qwen3-0.6B](https://huggingface.co/litert-community/Qwen3-0.6B) — `Qwen3-0.6B_dynamic_wi4b32_afp32.litertlm` (344,671,744 B, sha256 `03e7da1e…`), shipped as [PR #6](https://huggingface.co/litert-community/Qwen3-0.6B/discussions/6) on 2026-09-20 and on the repo's main since 2026-09-21. It replaces the 2026-08-04 file of the same name, whose string-only chat template stops on current LiteRT-LM main builds (`tried to use + operator on unsupported types string and sequence`; the 0.18 nightly drops the prompt instead — [LiteRT-LM #3688](https://github.com/google-ai-edge/LiteRT-LM/issues/3688); the metadata-only re-ship two entries up is the same fix for our own bundles). No converter of ours is involved: this is litert-torch's own `export_hf`, run as written, with LiteRT-LM's Qwen3 metadata pinned to one commit — so the entry is a recipe pin plus the measurement record.

```bash
PY=<python of a litert-torch main venv> bash scripts/reproduce_llm.sh qwen3-0.6b-wi4b32     # -> out/qwen3-0.6b-wi4b32/model.litertlm
# or directly, with the section-by-section comparison against the published file:
REF=Qwen3-0.6B_dynamic_wi4b32_afp32.litertlm bash qwen3_work/export_qwen3_06b_wi4b32.sh out/x
```

The command ([`qwen3_work/export_qwen3_06b_wi4b32.sh`](qwen3_work/export_qwen3_06b_wi4b32.sh)): `python -m litert_torch.generative.export_hf --model=Qwen/Qwen3-0.6B --quantization_recipe=dynamic_wi4b32_afp32 --prefill_lengths=128,1024 --cache_length=32768 --enable_gpu_dynamic_cache=True --apply_gpu_composites=True --use_bool_mask=True --fuse_qkv=True --fuse_gate_up=True --use_swiglu_composite=True --use_rope_composite=True --bundle_litert_lm=True --litert_lm_llm_metadata_override=LlmMetadataProto.pbtext`, the pbtext being LiteRT-LM's `models/qwen3/LlmMetadataProto.pbtext` at [`a8178d7d`](https://github.com/google-ai-edge/LiteRT-LM/blob/a8178d7dca6ac4d472476c9d76e482d8c8cbee3c/models/qwen3/LlmMetadataProto.pbtext) (the script fetches it and checks its sha256). Toolchain: litert-torch main `731ef0a` (2026-09-17, installed from source as 0.10.0), ai-edge-quantizer 0.9.0, litert-converter 0.4.0, ai-edge-litert 2.2.0, litert-lm-builder 0.16.1, torch 2.13.0, torchao 0.18.0, transformers 5.14.1, Python 3.14. 1 min 31 s on an M4 Max with the checkpoint cached.

- **A reproduction is section-identical, never file-identical.** Two exports of one recipe differ in the bundle's uuid and creation_timestamp. The literal command re-run on 2026-09-20, and the script on 2026-09-24, reproduce the published file section for section (`litert-lm unpack`, 0.17.1): `Section3_TFLiteModel_tf_lite_prefill_decode.tflite` 342,476,288 B sha256 `082dd282…`, `LlmMetadataProto.pbtext` `662580ef…`, `ExecutorMetadataProto.pbtext` `55f67399…`, `Section2_HF_Tokenizer_Zlib.zlib` `7e610629…`, and `model.toml` equal apart from those two keys. The script performs this comparison when `REF=` names the published file, and exits non-zero on any difference.
- **Pin the exporter and the pbtext.** The released litert-torch 0.9.4 has no `--use_swiglu_composite` (the script refuses to run on it), and main moves, so the commit is the pin. LiteRT-LM's pbtext moved on 2026-09-11 to a template that takes content parts only (see the MiniCPM5 entry); `a8178d7d` (2026-09-01) takes a string or a list of parts, and that is the generation which answers on litert-lm 0.17.1, on the 0.18 nightly, on litertlm-android 0.17.0 and on a 2026-09-18 main build alike. `export_hf` is a Python Fire CLI that accepts unknown flags without complaint (`apply_gpu_composites` and `use_bool_mask` are not config fields; they travel as extra kwargs and are read by name), so a misspelled flag is silently ignored — read the graph, not the command line, when a composite is missing.
- **What the graph is.** prefill_128 / prefill_1024 / decode = 916 / 916 / 830 ops; `odml.rope` ×171, `odml.swiglu` ×87, `odml.rms_norm` ×342, `odml.runtime_bmm` ×171, `odml.cache_update` ×87; one fused QKV FC and one fused gate/up FC per layer; a BOOL mask; INT4 block-32 on the 113 FULLY_CONNECTED weights and on the embedding table, fp32 activations, float KV. `--cache_length=32768` with `enable_gpu_dynamic_cache` becomes a KV axis of 32771 (the exporter's next-prime magic number); the pbtext's `max_num_tokens` is 4096. Against the 2026-08-04 file: all 114 quantized weight tensors and their scales are byte-identical, the graph gains the SwiGLU composite, and the prefill signatures now return logits (the lm_head FC sits in prefill_128 / prefill_1024 — exporter behaviour at `731ef0a`; on the Pixel 8a GPU the prefill is lower than the old graph's in every iteration, 2–3 % on process medians, and the decode steadier, 34.7–35.9 tok/s across iterations against 25–35). Left out on purpose: `use_qkv_norm_rope_composite` (the OpenCL program fails to build at engine creation on both phones — `unexpected type name 'float4'`) and the SDPA composites (they build, then give invalid output: `clEnqueueNDRangeKernel … Invalid command queue` on the Pixel 8a, 28 × `Shape mismatch` and the text "!!!!" on the S26). With the SDPA composite on the prefill signature only, the benchmark runs and reports **higher** prefill (S26 3,030 against 2,707 tok/s) while the text is "!!!!" on both phones — a speed row for a graph that computes garbage. Gate the text before quoting a number.
- **Speed** — the prebuilt Android `litert_lm_advanced_main` from `gs://litert/binaries/latest/android_arm64/litert_lm/` (2026-09-18 build, sha256 `adac974b…`); 1024 prefill / 256 decode / max_num_tokens 1280, five iterations per process, median of iterations 2–5, several processes shown as a range; every CPU leg ends with capped clocks, so the CPU cells are sustained-load numbers. Prefill / decode tok/s:

| file | Pixel 8a GPU (Mali-G715) | Pixel 8a CPU | Galaxy S26 GPU (Adreno) | Galaxy S26 CPU |
|---|---:|---:|---:|---:|
| this file | 714–723 / 35.5–35.8 | 101–136 / 17.9–22.9 | 2,693–2,704 / 72.1–73.5 | 298–301 / 36.3–37.4 |
| the repo's `qwen3_0_6b_mixed_int4.litertlm` (2026-09-19 version; TorchAO recipe, INT8 embedding table) | 575–576 / 33.1–33.6 | 106 / 3.4–3.5 | 1,965–1,970 / 63.7–70.8 | 246–254 / 5.5 |
| the 2026-08-04 file this one replaces | template error, exit 13 | template error | template error | — |

  Peak private footprint (one iteration): 997 MB GPU / 1,222 MB CPU on the Pixel 8a, 538 / 1,393 MB on the S26 (the mixed INT4 file: 892 / 2,883 MB on the Pixel 8a). The mixed INT4 graph decodes at 3.4–5.5 tok/s on XNNPACK with this binary; on CPU this file decodes 5–7× faster, and it leads on both GPUs as well.
- **Quality.** 8-question gate through the runtime's Session API on the phones (each bundle's own sampler, TOP_P k20 p0.95 t0.6): 7/8 on the Pixel 8a, 6–7/8 on the S26 (the mixed INT4 file 6–7/8). GSM8K through the released litert-lm 0.17.1 CLI, Mac GPU, one process per question, greedy: non-thinking, first 500 test questions, **46.8 %** (mixed INT4 46.6 %, exact McNemar p = 1.0 — two graphs on the same weights; the 10-question spread is the kernel-numerics floor). Thinking mode at a 2,048-token cap, 200 questions: reaches a final answer on 70 (the 2026-08-04 file 62, the mixed INT4 file 110); of the 38 questions all three answer, 35 / 37 / 36 are correct. The lower completion rate is a property of the recipe (INT4 embedding table — measured, not isolated as the cause), not of the refresh: same weights, and the two template generations give identical rows.
- **Where it runs, measured:** PyPI litert-lm 0.17.1 (Mac CPU and WebGPU), litert-lm-nightly 0.18.0.dev20260918, litertlm-android 0.17.0 (Galaxy S26, GPU and CPU), the 2026-09-18 Android build above (both phones, CPU and GPU) — all on-prompt. The full flag set (qkv_norm_rope + SDPA composites) also fails the desktop WebGPU path of 0.17.1 and of the nightly with about 16,300 WGSL validation errors: a recipe for newer runtimes, not a portable one yet.

## Fun-ASR-Nano-2512 (speech-to-text LLM, audio-in `.litertlm`, one utterance of up to 30 s per message)

`funasr_nano_work/` converts Tongyi Lab's [Fun-ASR-Nano-2512](https://huggingface.co/FunAudioLLM/Fun-ASR-Nano-2512) (SenseVoice SAN-M encoder, 70 layers, + 2-layer adaptor + Qwen3-0.6B, zh / en / ja, Apache-2.0) into one `.litertlm` that LiteRT-LM's released runtime (v0.16.1 CLI, 0.17.1 Python) drives through its **generic audio path**: the runtime frames 16 kHz PCM into 60 ms frames, runs the bundled encoder on a 30.24 s window, and splices the embeddings into a ChatML prompt at `<|AUDIO|>`. No runtime patch, no litert-torch patch. Hub repo: `litert-community/Fun-ASR-Nano-2512`.

```sh
cd funasr_nano_work
MODE=serial ./dl_weights.sh both                  # -vllm model.safetensors (export source) + official model.pt (reference), sha256-checked
HF_HUB_DISABLE_XET=1 hf download FunAudioLLM/Fun-ASR-Nano-2512-vllm --revision a4362c943d48951f98ca2a62181cc028970270c5 \
  --include "config.json" "config.yaml" "configuration.json" "tokenizer.json" "tokenizer_config.json" "vocab.json" "merges.txt" \
  "generation_config.json" "preprocessor_config.json" "LICENSE" "MODEL_PROVENANCE.json" "README.md" "example/*" --local-dir out/hf_vllm
HF_HUB_DISABLE_XET=1 hf download FunAudioLLM/Fun-ASR-Nano-2512 --revision 272c57b82523ada6fd87095e955f8e29100979ab \
  --include "README.md" "config.yaml" "configuration.json" "Qwen3-0.6B/*" "multilingual.tiktoken" "example/*" --local-dir out/hf_official
python make_fixtures.py                           # 5 official examples + 20 LibriSpeech dev-clean clips -> 16 kHz s16 mono WAV
python oracle_funasr.py                           # reference: funasr 1.4.16, fp32, dither 0, greedy, one discarded warm-up call
python build_lm_native.py                         # out/lm_native: Qwen3ForCausalLM from the llm.* tensors (tied head) + tokenizer
python port_eval.py                               # torch port of fbank + LFR + SAN-M + adaptor: bit-exact vs funasr on 25/25 clips
python export_audio_encoder.py                    # audio_encoder_504f_fp32.tflite: audio [1,504,960] -> features [1,63,1024] + mask [1,63]
python quant_encoder.py --variants fp16           # fp16 weights (FLOAT_CASTING), fp32 compute: transcripts 25/25 = reference
cd .. && EXTERNALIZE_EMBEDDER=1 CACHE=2048 PREFILL=512,128,32 python scripts/export_simple_template.py \
  funasr_nano_work/out/lm_native funasr_nano_work/out/lm_int8 templates/chatml_simple.jinja dynamic_wi8_afp32 && cd funasr_nano_work
litert-lm unpack out/lm_int8/model.litertlm --output-dir out/lm_int8/unpack          # tf_lite_prefill_decode + tf_lite_embedder
python build_bundle.py --enc out/audio_encoder/audio_encoder_504f_fp16.tflite --lm out/lm_int8/unpack \
  --out-name Fun-ASR-Nano-2512.litertlm --prefer-act fp32                              # GenericModel audio metadata + ChatML jinja
python mac_gate.py --bundle out/bundle/Fun-ASR-Nano-2512.litertlm --backend cpu --tag cpu   # 25-clip runtime gate: 21/25 = reference, WER 19/448
python mac_gate.py --bundle out/bundle/Fun-ASR-Nano-2512.litertlm --backend gpu --tag gpu   # LM on the GPU, audio on the CPU: 24/25, WER 20/448
```

Env: the reference runs in a separate venv (python 3.12, torch 2.14, torchaudio 2.11, funasr 1.4.16, transformers 5.17); the port, the encoder export and the bundle use torch 2.13 + litert-torch 0.9.4 + ai-edge-quantizer 0.9.0 + ai-edge-litert 2.2.0 + litert-lm-builder 0.16.1; the LM export (`export_simple_template.py`) uses litert-torch 0.9.3 + ai-edge-quantizer 0.8.0 + transformers 5.14.1; `mac_gate.py` is pure Python on `litert-lm-api` 0.17.1.

**The encoder graph starts at raw PCM.** The runtime's generic audio path hands the encoder framed PCM (`skip_mel_spectrogram_extraction`, frame = hop = 960 samples, `audio_input_scale` 1.0) and no clip length. So the Kaldi fbank (torchaudio's, which funasr calls; DFT and mel as constant matmuls), the LFR stacking (m 7, n 6), the SAN-M encoder and the adaptor are one graph, and the graph takes the length from the last non-zero sample. It emits a uint8 `mask`; the runtime keeps the embeddings up to its last non-zero entry, so the number of audio tokens per clip (ceil(L/8)) is decided in the graph. Consequence: exact digital silence at the end of a clip counts as padding (2 of the 25 clips lose one 60 ms LFR row; the transcripts do not change).

**One message = one window; long audio = one conversation per piece.** Without an adapter section the runtime treats the encoder as streaming with window = stride = 504 frames (30.24 s) and concatenates the windows' embeddings, but the model transcribes only the first window's speech (60.2 s clip: WER 83/143). Cutting the clip at 30.24 s and sending each piece in a new conversation gives 10/143, the reference's whole-clip figure; two turns of one conversation give 22/143 (the second turn repeats the first).

**The GPU needs fp32 activations for the LM.** With the GPU's default fp16 activations the Qwen3 LM emits token 0 (`!`) at every step, also for a text-only prompt, on the Mac (WebGPU) and on the Galaxy S26 (OpenCL). `build_bundle.py --prefer-act fp32` writes `prefer_activation_type: fp32` on the prefill/decode section; LiteRT-LM applies it when the app sets no activation type (v0.16.1 and 0.17.1 both read it), and the GPU transcripts then match the CPU's (24/25 = reference on both GPUs). The audio encoder stays on the CPU: the Mac GPU delegate does not take its DEQUANTIZE / PAD / SELECT_V2 ops.

**Quantization.** fp16 encoder weights are transcript-lossless (25/25); int8 dynamic encoder weights are not (19/25). The LM is the standard int8 dynamic export with an int8 embedder: 21/25 transcripts equal the reference (a proper-noun spelling, two punctuation differences, kana on the Korean smoke clip), English WER 19/448 against the reference's 20/448, and the unquantized LM gives 25/25, so every difference is the int8 LM. FLEURS (50 clips per language): en WER 5.48 % vs 5.13 %, zh CER 6.86 % vs 6.86 %, ja CER 6.81 % vs 6.95 %. An int4 LM (blockwise) gives 14/25 and is not shipped.

## decider-2b-vision (Mapika) — a vision System One model on the Qwen3.5-2B hybrid rail: the author's M-RoPE positions are derived inside the graph (2026-09-29)

Published: [litert-community/decider-2b-vision-LiteRT](https://huggingface.co/litert-community/decider-2b-vision-LiteRT) — `decider-2b-vision_fp16.litertlm` (5.51 GB, sha256 `75b22679…`; fp16 casting, float compute: 53/53 public image answers, max |Δp| 4.1e-05 vs the upstream fp32 readout; desktop CPU or GPU), `decider-2b-vision_fp16-int8vocab.litertlm` (4.50 GB, `72f68360…`; fp16 FCs + dynamic-int8 output head + int8 embedding: 52/53, 0.0085 on CPU, 53/53, 0.0062 on Mac Metal; desktop GPU at half the memory of fp16) and `decider-2b-vision_int8.litertlm` (3.17 GB, `5fb2e19a…`; the house dynamic-int8 recipe: 6/6 answer letters on a Galaxy S26 CPU and GPU, probabilities calibrated on a GPU only — CPU max |Δp| 0.43, Mac Metal 0.032). The model ([Mapika/decider-2b-vision](https://huggingface.co/Mapika/decider-2b-vision), apache-2.0, revision `863e290…`) reads letter logits at one answer slot per question from an image plus lettered options; the bundle is the fast_vlm shape (vision encoder + adapter fp16 at a fixed 256×256 = 64 image tokens, the Qwen3.5 hybrid decoder, identity template, no start token, stop 248044). Recipe by round, the exact exporter patch it used, the torch-free reference readout and the 42 public fixture rows with the upstream fp32 values: [`decider2bv_work/`](decider2bv_work/) (`REPRODUCE.md`, `FINDINGS.md`, `deps/`, `reference/`).

- **The runtime's 1-D image positions are a measurable cost for a probability model, and the graph can pay it itself.** Upstream uses 3-channel M-RoPE positions (image tokens on an 8×8 grid). Feeding plain 1-D positions to the upstream fp32 model on the same 256×256 pixels flipped 2 of 53 public image answers and moved a probability by 0.198 (measured before any conversion, on the HF model alone). The shipped decoder derives (t, h, w) from the runtime's single position inside the graph — image first, fixed grid: p = 0 → (0,0,0); 1 ≤ p ≤ 64 → (1, 1 + (p−1)//8, 1 + (p−1)%8); p ≥ 65 → p − 56 on all channels — written element-wise (channel masks × inv_freq, min/max steps, no GATHER), `decider2bv_work/scripts/mrope_derived.py`. fp32 graph vs upstream: 75/75 image slots, max |Δp| 4.8e-05; on every slot where the M-RoPE and 1-D upstream arms differ by > 0.01 the graph sits on the M-RoPE side.
- **Text-only rows on such a decoder start at position N + 1 (65 here).** All three derived channels then carry the same constant shift, and RoPE only sees relative positions: 9/9 text-only slots, max |Δp| 2.4e-06. Started at 0 (what the runtime does for a prompt without an image) the same rows read 8/9, 0.05 — so text-only requests go through the reference readout, and the card says so.
- **`clamp(x, 0, 1)` lowers to `RELU_0_TO_1`, which the LiteRT-LM GPU delegates refuse** (Mac WebGPU listed it together with `DEQUANTIZE` and left 252 ops on the CPU; the "not fully delegated" refusal followed). `relu(x) − relu(x − 1)` is the same step on integer inputs and delegates fully — all three files then ran on Mac GPU (12/12 first tokens), and `DEQUANTIZE` on its own did not block that path.
- **The fixed 256×256 input costs nothing where upstream's own processing lands on a 16×16 patch grid (224×224, 256×240, 256×256: identical pixels, |Δp| 0 on 13 requests) and up to 0.373 on larger synthetic images (24 public requests, 30/32 answers) — 0.58 on photographs kept private.** Both sides are upstream fp32; the size, not the conversion, moves the probability.
- **Weight forms on a 12 GB phone.** XNNPACK unpacks fp16 weights to fp32 (7.5 GB cache for the fp16 file, 6.0 GB for fp16-int8vocab), so neither desktop file fits a Galaxy S26 (the fp16-int8vocab GPU engine creation stopped at 6.55 GB VmHWM under a 6.5 GB guard). The dynamic-int8 file fits both backends (CPU VmHWM 4.54 GB, GPU 4.32 GB with the phone's MemAvailable down to 0.77 GB); its CPU probabilities are not calibrated (activations are quantized on the fly; the numbers also move with the prefill chunk plan, 0.43 exact-fit vs 0.23 one padded chunk), its GPU ones are (0.032 on Mac Metal).
- **Two transformers versions resample the vision position table with the same bilinear taps but different weight rounding (3.6e-6 at 256).** The parity target was the checkpoint's version (5.17.0); the export venv had 5.14.1, so the taps were computed in 5.17.0 and handed to the converter (`scripts/vision_interp_5170.py`).

Galaxy S26 (`litert_lm_advanced_main` v0.16.0, one cold run each, 150-token prompt with the image): int8 CPU prefill 136.7 tok/s / TTFT 1.22 s / engine init 17.7 s / VmHWM 4.54 GB; int8 GPU (all 7 decoder signatures on LITERT_CL) 340.8 tok/s / 0.56 s / 64.4 s / 4.32 GB. Mac M4 Max, LiteRT-LM 0.17.1, one decision per process: GPU TTFT 0.26–0.47 s, CPU 1.5–1.7 s (tables on the card).

## Confucius4-R2T2 (streaming speech-to-text LLM, audio-in `.litertlm`, one clip of up to 30 s per message)

`confucius4_r2t2_work/` converts NetEase Youdao's [Confucius4-R2T2](https://huggingface.co/netease-youdao/Confucius4-R2T2) (Qwen3-ASR-1.7B fine-tune: 24-layer audio encoder + 28-layer Qwen3 LM, 2.04 B parameters, 30 languages, NetEase Youdao Model Use License Agreement) into one `.litertlm` that the released LiteRT-LM runtime (`litert-lm-api` 0.17.1) drives through its **generic audio path**: the runtime frames 16 kHz PCM into 10 ms frames, the bundled encoder computes the original log-mel and the audio embeddings for a 30 s window, and the runtime splices them into the prompt at `<|audio_pad|>`. No runtime patch; the two litert-torch changes are process-local monkeypatches in the scripts. Hub repo: `<NAMESPACE>/Confucius4-R2T2`.

```sh
cd confucius4_r2t2_work
HF_HUB_DISABLE_XET=1 hf download netease-youdao/Confucius4-R2T2 --revision 185ce639118ad1362d049ca0d8ed04b6ec5cd6c9
python convert_layout.py                      # out/hf_layout: key rename to the transformers v5 -hf layout (payload byte-identical);
                                              # config / generation / processor_config from Qwen/Qwen3-ASR-1.7B-hf @ bcd2b5b7
python eager_hf.py --mode full                # reference: transformers 5.14.1, fp32, the litert-torch prompt, 155 clips
PYTHONPATH=out/pyoverlay_builder017 python export_bundle_r2.py --model out/hf_layout --out out/export/c4r2_f32_30s \
  --quantization_recipe none --input_sec 30 --cache_length 1024      # fp32 export, encoder attention in the original 104-token windows
litert-lm unpack out/export/c4r2_f32_30s/model.litertlm --output-dir out/unpack/c4r2_f32_30s
python quant_sections.py --src out/unpack/c4r2_f32_30s --tag c4r2_30s --arm c4r2_30s_C_lmi8 \
  --lm dynamic_wi8_afp32 --emb dynamic_wi8_afp32 --enc fp16 --pack   # LM int8 dynamic + int8 embedder (+ an fp16 encoder that the next step replaces)
litert-lm unpack out/export/c4r2_30s_C_lmi8/model.litertlm --output-dir out/unpack/r3_src_C
PYTHONPATH=out/pyoverlay_builder017 python export_bundle_r3.py --window 30 --src out/unpack/r3_src_C \
  --name c4r3_30s_C_lt --prompt litert        # encoder audio [1,3000,160] -> features [1,390,2048] + mask, fp16; generic_model metadata + template; pack
python r3_render_check.py --mode expected && python r3_render_check.py --mode runtime --prompt litert \
  --bundle out/export/c4r3_30s_C_lt/model.litertlm                  # the runtime's render of the template
python r3_runtime.py --bundle out/export/c4r3_30s_C_lt/model.litertlm --clips all --prompt bundle \
  --out out/r3_rt/c4r3_30s_C_lt_full155_bundle_cpu.jsonl            # released runtime, CPU, 155 clips
python score_confucius4.py ref --hyp out/r3_rt/c4r3_30s_C_lt_full155_bundle_cpu.jsonl:text   # FLEURS en 4.43 / zh 6.40 / ja 6.21
python verify_card_snippet.py --prompt bundle                       # the card's Python block, run as written
python r3_stream.py --backend runtime --prompt litert --bundle out/export/c4r3_30s_C_lt/model.litertlm \
  --out out/r3_stream/runtime_s1_litert.jsonl                       # the vendor's streaming loop through the same API
```

Env: export, quantization and the eager reference run in one venv (python 3.14, torch 2.13, transformers 5.14.1, litert-torch 0.10.0 at 731ef0a, ai-edge-quantizer 0.9.0, ai-edge-litert 2.2.0) with litert-lm-builder 0.17.0 on `PYTHONPATH` (litert-torch writes `LlmMetadata.pad_token`, which builder 0.16.1 lacks); `r3_runtime.py`, `r3_render_check.py --mode runtime`, `verify_card_snippet.py` and the streaming worker run on `litert-lm-api` 0.17.1 alone (pure Python). Fixtures: the 155-clip set of the Fun-ASR-Nano entry (FLEURS test en_us / cmn_hans_cn / ja_jp, 50 clips each, plus 5 example clips) under `$C4_FIXTURES`.

**Encoder attention windows.** Transformers attends inside 104-token (8 s) windows; litert-torch's Qwen3-ASR export patches the attention to 13-token (1 s) chunks, which cost 1.6 / 5.0 / 9.0 points (en / zh / ja, transcripts against the original on 5 s clips). `encoder_window.py` replaces that patched forward in-process with one SDPA and a constant block-diagonal mask built in numpy (so the graph gets one constant, not run-time index ops). The patched encoder equals the transformers encoder with max |Δ| 0 on 5 s and 30 s inputs.

**The encoder graph starts at raw PCM.** The generic path's built-in mel (HTK-style filterbank on the magnitude, natural log, frames not centred) is not the Qwen3-ASR / Whisper log-mel, so the bundle declares `skip_mel_spectrogram_extraction` with frame = hop = 160 samples and the encoder computes the extractor's log-mel in the graph: reflect pad + one strided conv over the windowed DFT basis (= `torch.stft(center=True)`), the extractor's Slaney filterbank, `log10`, `max − 8`, `(x + 4) / 4`. The runtime zero-fills the 30 s window after a clip, which is the input the 30 s single-window recipe was gated on. Output `features` [1, 390, 2048] plus a uint8 `mask` of ones: without a mask the runtime would take `ceil(3000 / (3000 // 390)) = 429` rows of a 390-row buffer (GENERIC_CONTRACT.md).

**Template.** The bundle renders the litert-torch prompt (`<|im_start|>user<|audio_start|><|audio_pad|><|audio_end|><|im_end|><|im_start|>assistant\n`). The original repository's prompt (an empty system turn) drops FLEURS zh / ja to 10.75 / 9.33 % CER on this file (language tags `English` / `None`, Chinese numerals, one translated clip per language) and also scores worse in fp32 PyTorch (7.78 / 7.47 against 6.24 / 6.21). Text items of the user message go after `assistant\n`: that is where the original code puts a forced language (`language X<asr_text>`) and where its streaming loop puts the committed prefix.

**Quantization.** Chosen on the 155 full clips: int8 dynamic LM + int8 embedder + fp16 encoder. All-int8 translated one Mandarin clip into English (zh 9.67 %), an fp16 LM stopped after `language None` on one English clip (en 6.26 %). Mac CPU (M4 Max, 4 threads): RTF 0.124, 4.55 GB peak RSS with warm caches; LM on the Mac GPU (WebGPU over Metal, `prefill_128` and `decode` fully delegated): RTF 0.053, FLEURS 4.52 / 6.45 / 6.06. The audio encoder does not run on the GPU delegate (`BATCH_MATMUL: Not supported batched mat mul case: non-constant tensor`).

## Qwen3-ASR-1.7B (speech-to-text LLM, audio-in `.litertlm`, one clip of up to 30 s per message)

`qwen3_asr17_work/` converts [Qwen/Qwen3-ASR-1.7B-hf](https://huggingface.co/Qwen/Qwen3-ASR-1.7B-hf) (24-layer audio encoder + 28-layer Qwen3 LM, 2.04 B parameters, Apache 2.0) into one `.litertlm` that the released LiteRT-LM runtime (`litert-lm-api` 0.17.1) drives through its **generic audio path**: the runtime frames 16 kHz PCM into 10 ms frames, the bundled encoder computes the original log-mel and the audio embeddings for a 30 s window, and the runtime splices them into the prompt at `<|audio_pad|>`. Same recipe as the Confucius4-R2T2 entry (a fine-tune of this model with the same audio encoder), without its layout conversion. Hub repo: [litert-community/Qwen3-ASR-1.7B](https://huggingface.co/litert-community/Qwen3-ASR-1.7B).

```sh
cd qwen3_asr17_work
HF_HUB_DISABLE_XET=1 hf download Qwen/Qwen3-ASR-1.7B-hf --revision bcd2b5b7f32b480ab5790554cfa8347f246a14f3
python eager_hf.py --prompt litert --out out/ref_eager_litert.jsonl       # reference: transformers 5.14.1, fp32, 155 clips
python make_crops.py --samples 79999 --out_dir out/crops79999 && python encoder_window_parity.py   # 104-token windows = transformers
PYTHONPATH=out/pyoverlay_builder017 python export_fp32.py --out out/export/q17_f32_30s       # fp32 export, 30 s, KV 1024
litert-lm unpack out/export/q17_f32_30s/model.litertlm --output-dir out/unpack/q17_f32_30s
python quant_sections.py --src out/unpack/q17_f32_30s --tag q17_30s --arm q17_30s_C \
  --lm dynamic_wi8_afp32 --emb dynamic_wi8_afp32 --enc fp16 --pack       # LM int8 dynamic + int8 embedder
litert-lm unpack out/export/q17_30s_C/model.litertlm --output-dir out/unpack/q17_30s_C_src
PYTHONPATH=out/pyoverlay_builder017 python export_bundle.py --window 30 --src out/unpack/q17_30s_C_src \
  --name q17_30s_C_lt --prompt litert        # encoder audio [1,3000,160] -> features [1,390,2048] + mask, fp16; generic_model metadata; pack
python render_check.py --mode expected && python render_check.py --mode runtime --prompt litert \
  --bundle out/export/q17_30s_C_lt/model.litertlm                       # the runtime's render of the template
python runtime_gate.py --bundle out/export/q17_30s_C_lt/model.litertlm --clips all \
  --out out/rt/q17_30s_C_lt_full155_cpu.jsonl                          # released runtime, CPU, 155 clips
python score_fleurs.py ref --hyp out/rt/q17_30s_C_lt_full155_cpu.jsonl:text    # FLEURS en 4.43 / zh 6.19 / ja 5.73
python verify_card_snippet.py --clip zh --language Chinese             # the card's Python block, run as written
```

Env: export, quantization and the eager reference run in one venv (python 3.14, torch 2.13, transformers 5.14.1, litert-torch 0.10.0 at 731ef0a, ai-edge-quantizer 0.9.0, ai-edge-litert 2.2.0) with litert-lm-builder 0.17.0 on `PYTHONPATH` (`out/pyoverlay_builder017/litert_lm_builder` → a 0.17.0 install; litert-torch writes `LlmMetadata.pad_token`, which builder 0.16.1 lacks); `runtime_gate.py`, `render_check.py --mode runtime` and `verify_card_snippet.py` run on `litert-lm-api` 0.17.1 alone (pure Python). Fixtures: the 155-clip set of the Fun-ASR-Nano entry (FLEURS test en_us / cmn_hans_cn / ja_jp, 50 clips each, plus 5 example clips) under `$C4_FIXTURES`.

**Encoder attention windows.** Transformers attends inside 104-token (8 s) windows; litert-torch's Qwen3-ASR export patches the attention to 13-token (1 s) chunks (encoder output cosine 0.81 on 5 s, 0.76 on 30 s against transformers). `encoder_window.py` replaces that patched forward in-process with one SDPA and a constant block-diagonal mask built in numpy; the patched encoder equals the transformers encoder with max |Δ| 0 on 5 s and 30 s inputs.

**The encoder graph starts at raw PCM.** The bundle declares `skip_mel_spectrogram_extraction` with frame = hop = 160 samples and the encoder computes the extractor's log-mel in the graph (reflect pad + one strided conv over the windowed DFT basis, the extractor's Slaney filterbank, `log10`, `max − 8`, `(x + 4) / 4`), then outputs `features` [1, 390, 2048] and a uint8 `mask` of ones.

**Template.** The bundle renders the litert-torch Qwen3-ASR prompt (`<|im_start|>user<|audio_start|><|audio_pad|><|audio_end|><|im_end|><|im_start|>assistant\n`, no system turn) message by message and puts the user's text items after `assistant\n`, where `qwen-asr` puts a forced `language X<asr_text>`. `export_bundle.py --prompt official` bakes the checkpoint's own chat template instead (empty system turn, newlines): with the same weights it scored Mandarin / Japanese CER 6.65 / 6.06 against 6.19 / 5.73 and tagged 8 of 50 Mandarin clips `English` (fp32 PyTorch: 6.86 / 5.50 against 6.04 / 5.24, 9 of 50).

**Quantization.** int8 dynamic LM + int8 embedder + fp16 encoder (the Confucius4 choice). On the 155 clips through 0.17.1: Mac CPU en 4.43 / zh 6.19 / ja 5.73 against PyTorch fp32 with the same prompt 4.35 / 6.04 / 5.24; LM on the Mac GPU 4.52 / 6.19 / 5.32; Galaxy S26 CPU (v0.16.1 CLI, `--benchmark`, weight caches loaded) prefill 538–547 tokens/s for the 398-token prompt, decode 25.0–25.6 tokens/s.


### The 15 s file (`Qwen3-ASR-1.7B-15s.litertlm`, 2026-10-13, the speed ladder's ship form)

Same weights and recipes as the 30 s file; the encoder window is 15 s and the LM carries prefill signatures for 128 and 256 tokens. The commands below are the ones that built the shipped file (run 2026-10-13 in `qwen3_asr17_work/`, the same venv as above with litert-lm-builder 0.17.0 on `PYTHONPATH`; the lane copies of the scripts live under `speed/scripts/`):

```sh
cd qwen3_asr17_work
PYTHONPATH=out/pyoverlay_builder017 python export_fp32.py --out out/r5/f32_pb128_256 --input_sec 30 --cache_length 1024 --prefill_lengths 128,256   # fp32 LM export, KV 1024, prefill 128 and 256 (805 s wall on a shared Mac)
litert-lm unpack out/r5/f32_pb128_256/model.litertlm --output-dir out/r5/unpack/f32_pb128_256
PYTHONPATH=out/pyoverlay_builder017 python speed/scripts/quant_r5.py --src out/r5/unpack/f32_pb128_256 --tag F6_30s_pb --arm F6_C \
  --lm dynamic_wi8_afp32 --emb dynamic_wi8_afp32 --enc fp16 --pack            # quant_sections.py with its output root under out/r5: LM int8 dynamic + int8 embedder (the 30 s file's recipe)
litert-lm unpack out/r5/export/F6_C/model.litertlm --output-dir out/r5/unpack/F6_C_src
PYTHONPATH=out/pyoverlay_builder017 python speed/scripts/export_bundle_r3.py --window 15 --kv 1024 --enc_name enc15 --stages encoder,tflite   # the 15 s encoder: audio [1,1500,160] -> features [1,195,2048] + mask, fp16 (round 3; sha256 of enc__fp16.tflite 3f6394ff…)
PYTHONPATH=out/pyoverlay_builder017 python speed/scripts/pack_r5.py --src out/r5/unpack/F6_C_src --enc out/r3/enc/enc15/enc__fp16.tflite \
  --expect_enc_sha256 3f6394ff85db33f3c2b6560c1fefb626d3b8e2610b21667ed4f92aa462fa1385 --name F6_win15_pb128_256 --kv 1024 --prompt litert \
  --form_out speed/forms/F6_win15_pb128_256/model.litertlm                # export_bundle.pack() with the 15 s encoder: 2,701,438,880 B, sha256 fbec6d12… (the bundle's sha256 changes with each pack: the header carries a uuid and a timestamp; the sections are what to compare)
~/venvs/lt0180run/bin/python speed/scripts/runtime_gate_r3.py --bundle speed/forms/F6_win15_pb128_256/model.litertlm --clips all --backend cpu --out speed/r5/rt_F6_cpu.jsonl   # litert-lm-api 0.18.0, 155 clips
python speed/scripts/parity_r3.py --hyp speed/r5/rt_F6_cpu.jsonl --out speed/r5/parity_F6_cpu.json   # FLEURS en 4.35 / zh 6.19 / ja 5.84 (GPU fp16: 4.44 / 6.24 / 5.21)
```

Speed rows (response = audio encoder + prefill + decode, the 6.4 s English demo clip, warm): Mac M4 Max GPU 0.341 s / CPU 0.847 s (litert-lm-api 0.18.0), Galaxy S26 GPU 1.40–1.51 s / CPU 1.55 s, Pixel 8a GPU 5.03–6.33 s / CPU 5.10 s (v0.18.0 prebuilt CLI); the 30 s file on the same days 0.538 / 1.191, 2.11–2.81 / 2.11, 9.86–9.98 / 8.39 s. The full ladder with every form tried (int4 LM = parity FAIL, encoder int8 = runtime refusal unless the log-mel stays float, KV 512 = GPU parity FAIL on Japanese, prefill buckets on the 30 s window) is `qwen3_asr17_work/SPEED_LADDER.md` (rows 1–67, `speed/ROUND1–6.md`).

## Kev-0.8B and Kev-4B (Jared Palmer) — decision models that never generate text, as raw LiteRT graphs: row graphs at six lengths and shared-state pairs; Kev-0.8B on the Galaxy S26 GPU and, up to 256 tokens, on its NPU (2026-10-05)

Published: [litert-community/Kev-0.8B-LiteRT](https://huggingface.co/litert-community/Kev-0.8B-LiteRT) with six row graphs `kev-0.8b_rowprefill_L{64,128,256,512,1024,2048}_fp16fc_i8emb.tflite` (1,258,031,552 B, sha256 `86c5c915…`; 1,258,444,912 B, `64fbd072…`; 1,259,246,704 B, `436d8321…`; 1,261,233,328 B, `3ca6d890…`; 1,266,799,568 B, `e3c04777…`; 1,284,223,520 B, `43d9d551…`), two shared-state pairs `kev-0.8b_sharedstate_Ls{128,256}_Lq64_fp16fc_i8emb.tflite` (1,261,368,160 B, `f06672e0…`; 1,261,918,016 B, `15197866…`) and `head/kev_0.8b_pointer_head.safetensors` (2,099,632 B, plus a JSON with the temperature 2.3510958125672174), and [litert-community/Kev-4B-LiteRT](https://huggingface.co/litert-community/Kev-4B-LiteRT) with the same eight files for Kev-4B, `kev-4b_rowprefill_L{64,128,256,512,1024,2048}_fp16fc_i8emb.tflite` (7,785,929,712 B, `9c6b332e…`; 7,786,450,960 B, `fdd03b44…`; 7,787,432,576 B, `e039d17b…`; 7,789,789,264 B, `db9650b7…`; 7,796,075,120 B, `439f2c36…`; 7,814,938,320 B, `37561dfa…`) and `kev-4b_sharedstate_Ls{128,256}_Lq64_fp16fc_i8emb.tflite` (7,789,323,680 B, `6df68c4c…`; 7,789,936,224 B, `a7fe4ba5…`), and `head/kev_4b_pointer_head.safetensors` (5,245,360 B, temperature 2.406050072164233). The L512, L1024 and L2048 files keep their 2026-10-04 names with new contents. The sources are `jaredpalmer/kev-0.8b` and `jaredpalmer/kev-4b` at tag `v1.0` (Apache-2.0): `Qwen/Qwen3.5-0.8B-Base` and `Qwen/Qwen3.5-4B-Base` adapted with a rank-16 LoRA, plus a pointer head that reads the hidden states. Given a text (the state) and typed questions about it (noul, choice, score) in the `/v1/systemone` shape of the author's server, they return an answer with probabilities per question; they never generate text. The files are raw LiteRT graphs, not `.litertlm` bundles: no LM head, the hidden states after the final RMSNorm as the output, the same weights in all eight files of a model, and the pointer head applied by the host in float32. The reference is the author's fp32 code at tag `kev-1.0` (post-temperature probabilities; 377 requests with 402 questions). A near tie is a question whose two most likely options in the reference are 0.02 or less apart, and the tolerance, set before the runs, is the same most likely option on every other question, max |Δp| ≤ 0.02 and mean ≤ 0.002. Every Kev-0.8B file gives the reference's most likely option on every question that is not a near tie, with max |Δp| 0.0104 on the Mac CPU and on Metal at float32 with ai-edge-litert 2.2.0 (L512: 377/377 + 13/15, mean 9.1e-4), 0.0110 on the Galaxy S26 GPU at `FP16_WITH_FP32_ACCUM` and 0.0134 on the S26 NPU (L64, L128 and L256 only). Every Kev-4B file gives it on every question, near ties included, with max |Δp| 0.0152 on Metal at float32 (L512: 383/383 + 9/9, mean 4.5e-4) and the same numbers on the Mac CPU for L1024 and both pairs. On the Galaxy S26, a Kev-0.8B call of about 130 tokens takes 199.4 ms on L256 at `FP16_WITH_FP32_ACCUM` while no clock cap is lowered ("cool"; 220.5 ms "sustained", after a cap) and 126.7 ms on the NPU (the L256 graph computes all 256 positions at any row length). The 2026-10-04 L512 file took 630 ms at FP32, sustained load included. On the S26, a 5-question request takes 624.8 ms through the Ls128 pair, or 430.9 ms with constant tensor sharing off. Kev-4B did not fit the 12 GB phone.

- **A graph computes all of its positions whatever the row length, so each model now has row graphs at six lengths, and the host runs each row on the smallest one that holds it.** The row form is unchanged. One row is the state plus one question: `ids` int32 [1, L] (right-padded with 248044) and `valid` float32 [1, L] in, `hidden` float32 [1, L, d] out, positions the constants 0 to L−1, and a mask that is a constant causal mask plus (1 − valid) × (−1e4). No KV or Gated DeltaNet state goes in or out, so the graph needs no cache-update operators, and the forms that mobile GPUs reject in the stock exporters (masked_fill, in-place row updates with index_put, a rank-5 repeat_interleave on the 4B, rank-4 PAD) stay out of it. The host reads the hidden states at the decide token and at each option's closing 248050, then applies the pointer head. On Metal at float32, a Kev-0.8B call takes 24.8 to 24.9 ms on L64 (a 64-token row) and 398.7 to 398.8 ms on L2048 (1,805 tokens). A row of 128 to 142 tokens takes 54.4 ms on L256; the 2026-10-04 files had nothing below L512, which took 139.9 ms for about 130 tokens. On Kev-4B, an 80-token row takes 118.5 ms on L128; the earlier release had nothing below L512 (464.4 to 464.7 ms). No file of either model has a CUSTOM operator, an int64 tensor or a tensor of rank above 4. The Galaxy S26 OpenCL delegate took every node of each Kev-0.8B file in one partition (3,912 for L64 to 23,635 for L2048).
- **The final Gated DeltaNet kernel computes the same fp32 result with fewer operators, and it is written so that float16 storage does not overflow or lose precision in it.** It forms the in-chunk inverse (I − A)⁻¹ by recursive doubling (a masked copy, then 5 levels of two matrix products per 64-token chunk) instead of a 63-step forward substitution. Softplus becomes relu(x) + log1p(exp(x − 2 relu(x))), so its EXP argument stays ≤ 0. Every EXP argument in the chunk kernel is clamped at −80. The decays inside a chunk become sums over their own window, formed with constant 0/1 matrices. The kernel scales v by 2⁶ and the gated norm's eps by 4⁶, which lifts the norm's input out of the float16 subnormal range. Kev-4B also runs every input_layernorm on 2⁻¹ x and the final RMSNorm on 2⁻⁵ x, with eps scaled to match. In PyTorch fp32 on all 402 questions, the probabilities stay within 4.7e-6 (0.8B) and 5.3e-6 (4B) of the earlier kernel's, with the same most likely option. The 512-token fp32 graph has 8,327 operators instead of 20,873 (0.8B) and 11,226 instead of 27,921 (4B). Float16 storage needed the changes because the converter lowers every RMS-type norm's mean(x²) to a SUM: a sum of squares above 65,504 becomes inf, and the row comes out as zeros. Without its two power-of-two scales, Kev-4B's final norm input is above that maximum on every test row (up to 145,930.4). With them, the smallest margin to 65,504 is 6.55× on the 4B (post_attention_layernorm) and 10.97× on the 0.8B (final norm); "Float16 headroom of the norms" in `conversion/README.md` gives every norm.
- **At the GPU's default precision (float16 activations) the new files stay finite but fall outside the tolerance; float16 storage with float32 accumulation (`FP16_WITH_FP32_ACCUM`) keeps every Kev-0.8B file within it on the Galaxy S26 GPU.** The 2026-10-04 files gave NaN on 18 of 393 test rows at the default precision. At that precision on Metal, the new Kev-0.8B L128 and L2048 files give max |Δp| 0.0332 and 0.0352 (mean 3.6e-3 and 4.1e-3), and the Kev-4B L128 and L512 files 0.0456 and 0.0675. Python's `GpuOptions` offers only float32 (`enforce_f32=True`) and the default, so the desktop GPU runs at float32. The Kotlin and C APIs add `FP16_WITH_FP32_ACCUM`. With it, every Kev-0.8B file passes on the Galaxy S26 GPU (LiteRT 2.2.0, Kotlin CompiledModel API, OpenCL): L512 gives 377/377 + 13/15, max 0.0091, mean 1.06e-3, and the largest max is 0.0110, on the Ls256 pair. A cool call there takes 57.4, 102.4 and 199.4 ms against 85.3, 137.4 and 270.2 ms at FP32 (L64, L128, L256). FP32 stays closer to the reference on the 9 rows longer than 1,024 tokens: max 0.0015 and mean 4.3e-4, against 0.0093 and 1.84e-3 at `FP16_WITH_FP32_ACCUM`, which leaves little room under the 0.002 bound on the mean. No Kev-4B output has been checked at `FP16_WITH_FP32_ACCUM`.
- **A shared-state pair runs the state once and each question from it; constant tensor sharing keeps one copy of the weights for its two signatures, at a cost in time.** A pair is one file. Its `state_prefill_<Ls>` signature takes `[248060]` and the state's tokens (Ls 128 or 256) and returns the state of every layer: 48 tensors on the 0.8B (23,347,200 bytes per request at Ls 128) and 64 on the 4B. Its `question_step_<Ls>_64` signature takes one question's branch of up to 64 tokens and those tensors, handed over directly as its input buffers, and places the branch's positions after the state. A state belongs to the file that made it, because the kernel carries the recurrent state at 2⁶ times its stock value. On Kev-0.8B, a pair and the row route differ by at most 3.7e-6 on the CPU and 4.4e-6 on Metal. With sharing off, the Kev-0.8B 5-question request on the Ls128 pair takes 163.6 to 164.0 ms instead of 199.4 to 200.1 ms on Metal at float32, but the process footprint after one request is 6.3 GB instead of 3.0 GB. On the Galaxy S26 at `FP16_WITH_FP32_ACCUM`, the same request takes 430.9 ms instead of 624.8 ms, with a VmHWM of 6.6 to 7.0 GB instead of 3.1 to 3.2 GB and the phone's smallest MemAvailable at 2.7 to 3.1 GB instead of 5.6 to 6.1 GB. Sharing is the host's default. In the default `auto` mode, the host takes the smallest pair whose Ls holds the state and compares, for the n questions whose branches fit, R (the sum of the lengths of the smallest row graphs that fit their rows) with P = Ls + n × 64. Those questions take the pair only when R > 1.5 × P; of the 377 test requests, 9 go through a pair (3 on Ls128, 6 on Ls256) and 368 through the row graphs.
- **On the Galaxy S26's Qualcomm HTP, a sigmoid that reads a FULLY_CONNECTED output directly loses precision; a SUM over an axis of size 1 between them changes no value and makes the Kev-0.8B 64-, 128- and 256-token files pass.** The runner, `android/measure/kev_npu_runner.cc` on the LiteRT 2.2.0 C API, uses the accelerators NPU and CPU. The int8 embedding lookup runs on the CPU and every other operator on the NPU (the dispatch delegate takes 2 of the 3 nodes); with the NPU alone, LiteRT 2.2.0 fails to compile the graphs. Two kinds of gate, the Gated DeltaNet output norm's gate and the attention output gate, are FULLY_CONNECTED outputs that a sigmoid reads. On the HTP, such a sigmoid was off by about 2.3e-3 (absolute, root mean square), and graphs without the SUMs moved the probabilities by up to 0.085. The three files carry 24 SUMs each, one after each in_proj_z and q_proj linear. They give the same bits as the graphs without them on the Mac CPU and GPU and on the S26 GPU. On the NPU, L64 gives 71/71 + 1/1 (max |Δp| 0.0105), L128 311/311 + 10/10 (0.0134, mean 1.29e-3) and L256 371/371 + 13/14 (0.0134, mean 1.24e-3). In the same runner, one row takes 44.0, 67.8 and 126.7 ms on the NPU and 56.9, 101.9 and 194.9 ms on the GPU at `FP16_WITH_FP32_ACCUM` (L64, L128, L256; median of 20 calls). NPU times move by about 5% from process to process. The GPU times for L64 and L256 are of graphs with the same weights but without the SUMs; on L128, the published file took 101.9 ms and the graph without them 101.1 ms. The initial load compiles each file on the phone, in 55.7, 191.2 and 263.3 s (VmHWM 5.5 to 6.2 GB), and leaves a cache of 1.27 to 1.29 GB; later loads read it in 1.5 to 1.7 s. Not measured: the NPU through the Kotlin CompiledModel API, other phones or SoCs, the initial compile inside an app, and files compiled ahead of time (on small graphs, AOT gave the same bits as the JIT). The NPU runtime libraries of LiteRT 2.2.0 and the QAIRT 2.47 HTP libraries are not included.
- **Compiling a graph is enough to make the phone lower its GPU clock cap, so each timing run waits after the compile, and every call is classed as cool or sustained by the clock caps the host reads.** In a run on a 128-token graph at FP32, the 7.6 s compile took the GPU from 41 to 60 °C. The 17 calls made while the GPU cap was lowered took 143.4 to 161.5 ms; the 8 made after it was lifted took 134.6 to 137.3 ms. Each timing run now waits until the GPU's clock cap and temperature are back to their values from before the compile, and the host reads the phone's clocks every 2 seconds. Cool is the median of the calls made while neither the GPU cap (1,300 MHz) nor any CPU cap was lowered; sustained is the median of the calls made after a cap was lowered. Kev-0.8B at `FP16_WITH_FP32_ACCUM`, cool and sustained: L256 199.4 and 220.5 ms (rows of 128 to 142 tokens; the 5-question request on five L256 calls, 992.9 and 1,088.3 ms), L512 387.0 and 393.4 ms (300 tokens), L1024 819.8 and 847.7 ms (1,000 tokens), L2048 1,808.2 and 2,147.6 ms (1,805 tokens). L64 (57.4 ms, 64 tokens) and L128 (102.4 ms, 80 tokens) ran with no cap lowered. In a run where the GPU cap fell to 578 MHz, the 5-question request on the pair with sharing took 1,081.0 ms instead of 624.8 ms.
- **Kev-4B ran on the Mac only: it did not fit the 12 GB Galaxy S26 in two tries.** On 2026-10-03, during the GPU compile of the earlier release's L1024 file (7.8 GB) at FP32, the delegate took all 34,313 nodes, and 9 s later the low-memory killer stopped the process (VmHWM 5.7 GB, 7.2 GB in swap on the killer's line). On 2026-10-05, during the GPU compile of this release's L64 file (7,785,929,712 bytes) at `FP16_WITH_FP32_ACCUM`, the delegate took all 5,307 nodes in one partition; 6 s later MemAvailable had fallen from 6.8 GB to 1.2 GB, with the process at 5.9 GB VmHWM and the GPU's allocation still at 0.3 GB, and the host stopped the app. The CPU was not tried. On the Apple M4 Max (128 GB), Metal at float32 takes 70.8 ms for 64 tokens (L64), 118.5 ms for 80 (L128), 396.0 to 396.5 ms for 300 (L512) and 1,705.4 ms for 1,805 (L2048; the earlier file took 1,824.2 ms in the same timing run). The 5-question request takes 864.1 ms through the row graphs, 871.8 ms through the Ls128 pair with sharing and 492.6 ms without. With one row graph, the process footprint is 17.0 to 20.8 GB after compile with a peak of 37.0 to 38.1 GB on the GPU, and 14.7 to 16.4 GB on the CPU. With a pair on the GPU, it is 15.9 GB (peak 17.2 GB) with sharing and 36.9 GB (peak 54.9 GB) without. An fp32 export takes about 50 GB of memory at its peak.
- **The storage form is unchanged: FULLY_CONNECTED weights in float16 behind DEQUANTIZE and an int8 embedding table with one scale per row, because dynamic int8 broke the calibration and a float16 table does not compile on the GPU.** For the 2026-10-04 release, the storage forms were compared on the Kev-0.8B L1024 graph with 392 questions and the same tolerance. Dynamic int8 FC with the int8 table gave 376/377 + 11/15 and max |Δp| 0.0701 on Metal, and 373/377 + 9/15 and 0.1465 on the Mac CPU, where the control moved by only 0.0159 and went undetected. The int8 table alone, with FC in fp32, gave 0.0103 and the same two near-tie flips as the shipped form, so the 0.0104 comes from the table. A float16 table gave 0.0010 on the CPU, but the GPU refuses to compile it (`EMBEDDING_LOOKUP: Empty quantization params`). At float32 the new files give the same numbers as the 2026-10-04 files: max 0.0104 on the 0.8B, where the int8 table sets the difference, and 0.0152 on the 4B, where its source was not measured separately. Activations, the causal convolutions and the delta rule stay in float32, and each pair keeps its RoPE table in float32.
- **The tokenizer that matches the author's loader is the Kev repo's `tokenizer.json`, not the base repo's.** With the Kev repo's file (sha256 `06b95093…`), the Python host passes 12/12 tokenizer probes, and its token ids equal the author's `encode` on all 402 questions for both models. The base repo's own `tokenizer.json`, read with `tokenizers.Tokenizer.from_file`, differs on 4 of 12 probes (Devanagari combining marks, `<think>`-like strings), and the host refuses it. Run on the published files without PyTorch or the kev package, the host reproduces the desktop runs bit for bit for each model: 452 of 452 questions through the row graphs, 314 through the Ls128 pair and 341 through the Ls256 pair.

Scripts, fixtures, the Python host, the Kotlin snippet, the sources behind the phone numbers and both per-model reproduction guides: [`kev_work/`](kev_work/) (`README.md`; `REPRODUCE-Kev-0.8B.md` and `REPRODUCE-Kev-4B.md`; `conversion/README.md` with the commands in order, the kernel forms and the float16 headroom of the norms, where the Kev-4B commands take `--model 4b`; `android/measure/README.md` with the build and run commands of the measurement activity and of the NPU runner).

## d1-omni-600M (Liquid AI) — a decision model that answers text, image and audio questions with probabilities and never generates text, as raw LiteRT graphs: decision graphs at six lengths, a vision tower and projector, audio encoders for clips of up to 5, 10, 20 and 30 s (2026-10-09)

Published: [litert-community/d1-omni-600M-LiteRT](https://huggingface.co/litert-community/d1-omni-600M-LiteRT) (2026-10-09) with six decision graphs `d1-omni-600M_decide_L{128,256,512,1024,2048,4096}_fp16.tflite` (896,250,176 to 898,281,808 B), `d1-omni-600M_vision_tower_fp16.tflite` (171,563,424 B), `d1-omni-600M_projector_fp16.tflite` (16,791,504 B), four audio encoders `d1-omni-600M_audio_T{501,1001,2001,3001}_fp16.tflite` (221,138,784 to 242,933,600 B), `contract.json` (every file's SHA-256, signature, bucket rule, token id, temperature, host step and GPU precision), the provider's `tokenizer.json`, a Python host without PyTorch and the public check set and the model card: 43 files, 6,503,130,609 B. The source is `LiquidAI/d1-omni-600M` at `414f8d6438174f5b2133a9c21a478fc42625e308` (LFM Open License v1.0). Given a state and typed questions (noul, choice, score), it returns a probability per option; images and audio enter as prefix rows ahead of the text. The reference is the provider's own float32 code; a file passes when it gives the reference's most likely option on every question that is not a near tie (the two most likely options 0.02 or less apart), with max |Δp| ≤ 0.02 and mean ≤ 0.002. Through the host on the Mac (Apple M4 Max, ai-edge-litert 2.2.0), the 303 public questions give max |Δp| 0.00191 (text), 0.00324 (image) and 0.00154 (audio) on the CPU, and 0.00192, 0.00331 and 0.00318 on Metal with each graph at its precision from `contract.json`.

- **Every graph stores its fully connected weights in fp16, and its norms run in an fp16-safe form that changes no float32 bit.** The decision graphs rewrite 36 of their 50 norms, the vision tower 13 of its 25 LayerNorms and each audio encoder 31 of its 87: the norm's input is multiplied by 2^-k and its epsilon by 4^-k. Without the rewrite the decision graphs gave uniform probabilities on every row under float16 storage, on Metal at the default precision and on the Galaxy S26 GPU at `FP16_WITH_FP32_ACCUM`.
- **The host picks the smallest decision graph that holds the row, because a graph computes all of its positions.** On Metal at float32 one call takes 11.5 ms on L128 and 444.4 ms on L4096; one image crop takes 33.15 ms in the tower and 1.06 ms in the projector; the audio encoders take 9.83 to 18.70 ms at the default precision, and a 10 s clip with one question takes 30.80 ms end to end.
- **On the Galaxy S26 (LiteRT 2.2.0), FP32 on the GPU keeps every shipped graph it ran (all but L4096) within the bar; `FP16_WITH_FP32_ACCUM` did so on the conversion lane's rows (101 to 108 per decision bucket) and is faster, but one of the 210 public L128 text rows moves by 0.040 at that precision (FP32: 0.0019 on the same rows); the NPU passed L128's 101 gate rows but fails the full 210-row public set (4 rows over 0.02, one argmax change) and the other graphs, so it is not a recommended path.** One call takes 42.0 ms on L128 and 778.1 ms on L2048 at `FP16_WITH_FP32_ACCUM` (46.5 and 1,321.6 ms at FP32), the vision tower 137.1 ms, the audio encoders 19.1 to 79.7 ms. On the NPU, L128 takes 29.6 ms and passes its 101 gate rows but not the full public set; L256, the vision tower and the T1001 audio encoder fall outside the bar there too. The shipped L4096 file has not run on the phone.

Scripts, the host, the check set, an example, the measurement app and the reproduction guide: [`d1_omni_work/`](d1_omni_work/) (`README.md`, `REPRODUCE.md`, `conversion/README.md` with the commands in order).

## d1-3B (Liquid AI) — a decision model that answers typed questions about text, JSON values and pictures with probabilities and never generates text, as raw LiteRT graphs for desktops: row graphs at six lengths, three shared-state pairs, a picture tower and projector (2026-10-09)

Published: [litert-community/d1-3B-LiteRT](https://huggingface.co/litert-community/d1-3B-LiteRT) (2026-10-09) with six row graphs `d1-3b_rowprefill_embeds_L{128,256,512,1024,2048,4096}_fp16fc.tflite` (4,871,345,568 to 4,940,420,512 B), three shared-state pairs `d1-3b_sharedstate_embeds_Ls64_Lq64_fp16fc.tflite`, `d1-3b_sharedstate_embeds_Ls128_Lq64_fp16fc.tflite` and `d1-3b_sharedstate_embeds_Ls256_Lq128_fp16fc.tflite` (4,871,972,336 to 4,872,620,816 B), `d1-3b_vision_tower_fp16fc.tflite` (825,613,520 B), `d1-3b_projector_fp16fc.tflite` (27,281,424 B), the tables the host reads (the tied embedding in bfloat16, 524,288,104 B; the read-out rows; the tower's position table), the provider's `tokenizer.json`, `contract.json`, a Python host without PyTorch, the test fixtures and the model card: 45.3 GB. The source is `LiquidAI/d1-3B` at `da1fe36a861f24690f27f622dca1d8688503d113` (LFM Open License v1.0), post-trained from `LiquidAI/LFM2.5-VL-3B`. Given a state (text, a JSON value or pictures) and typed questions (noul, choice, score), it returns a probability per option. The reference is the provider's own code in float32 on the CPU, one row per question: 415 text questions and three requests with pictures. A file passes when it gives the reference's most likely option on every question that is not a near tie (the two most likely options 0.02 or less apart), with max |Δp| ≤ 0.02 and mean ≤ 0.002, and when each of four control requests moves some option by more than 0.02. On an Apple M4 Max (macOS 27.0, ai-edge-litert 2.2.0), every text file passes on the CPU (XNNPACK, 8 threads) and on Metal at float32, with every operator on the delegate in one partition. The probabilities stay within 1.02e-5 of the reference on text and within 3.39e-5 on the three picture requests.

- **The text graphs take embeddings, not token ids.** The host writes the rows of the bfloat16 embedding table as float32 into the graph's `embeds` input. An earlier form kept an int8 copy of the table inside the graph (5.14 GB for the 256-token row graph, against 4.87 GB now) and is not published.
- **A graph computes all of its positions, so the host sends each request to the form with the smaller expected time.** The expected times come from call times measured on Metal at float32 and stored in `contract.json`. One call takes 61.8 ms on L128 and 2,050.0 ms on L4096; a pair costs one state call plus one call per question (37.80 + 38.72 ms for Ls64). The source card's three-question example takes 153.9 ms through the Ls64 pair and 185.5 ms on three L128 row graphs. A five-question request takes 255.1 ms through the Ls128 pair and 544.0 ms on five L256 row graphs, and the source card's photo with one question takes 389.3 ms, 156.2 ms of it in the tower. Compiling on Metal takes 20.3 to 28.3 s per row graph and 39.2 to 45.5 s per pair. Each compiled graph holds its own copy of the weights: a process footprint of 13.9 to 14.1 GB per row graph and 23.2 GB per pair on Metal.
- **Run the GPU at float32 precision.** At Metal's default precision (float16 storage) every text file stays finite but moves the probabilities by 0.044 to 0.086, and the picture requests by up to 0.52, which flips one answer.
- **A 12 GB Galaxy S26 did not hold the model.** With LiteRT 2.2.0 through the Kotlin CompiledModel API, the earlier 5.14 GB form of the 256-token row graph was delegated whole on the GPU (OpenCL, `FP16_WITH_FP32_ACCUM`, 1,694 of 1,694 nodes) and 1,692 of 1,694 nodes on the CPU (XNNPACK, 4 threads). In both, a memory guard stopped the app 8 s into the compile, when MemAvailable fell below 1,500,000 kB. A 2.71 GB form with dynamic int8 FULLY_CONNECTED weights loaded on the CPU but moved the probabilities by up to 0.157. The published files did not run on the phone, and its NPU was not tried.
- **Reproduce it on a Mac with two Python environments.** The provider's reference runs on Python 3.12.11 with torch 2.14.1 and transformers 5.14.1 (`conversion/requirements-venv-ref.txt`). Export, quantization, the LiteRT checks and the timing run on Python 3.14.6 with litert-torch 0.9.4, torch 2.13.0, transformers 5.14.1, ai-edge-litert 2.2.0 and ai-edge-quantizer 0.9.0 (`conversion/requirements-lt094dev.txt`). `conversion/README.md` gives the commands in order: restore the test requests (`fixtures/rebuild_requests.py` reads 221 of them from the kev repository at tag `kev-1.0`) and their rows, cut the tables out of `model.safetensors`, run the reference, then export, store and gate each row graph (`d1_run_L.sh` per length), each pair and the picture graphs, write the contract, time the requests and hash the shipped set.
- **Check a download with the example.** `examples/run_example.py --check` (CPU) and `--check --accel gpu` (Metal at float32) send the source card's two examples and a request too long for any graph. Every probability must be within 1e-5 of the provider's float32 answers, and the most likely options, the input token counts, the routes and the refusal must match.

Scripts, the host, the test fixtures, an example and the reproduction guide: [`d1_3b_work/`](d1_3b_work/) (`REPRODUCE.md` with the sources, the steps and where each number comes from; `conversion/README.md` with the commands in order). The graphs, the tables and the tokenizer are on the Hub only.

## Phocinae-Largha-150M-v1 (Phocinae) — a typed-decision model on mmBERT-small in the laya contract: three windows, two table forms, every form within the bar on the Galaxy S26 GPU, NPU and CPU (2026-10-10)

Published: [litert-community/Phocinae-Largha-150M-LiteRT](https://huggingface.co/litert-community/Phocinae-Largha-150M-LiteRT) with `phocinae_largha_150m_L{256,384,512}_fp16fc_int8emb.tflite` (fp16 FULLY_CONNECTED weights + an int8 embedding table in the graph, `input_ids`), `phocinae_largha_150m_L{256,384,512}_fp16fc_embeds.tflite` (fp16 FULLY_CONNECTED weights, `inputs_embeds` from the float16 token table `tables/tok_embeddings_fp16.bin`), `phocinae_act_head_fp32.tflite`, the source tokenizer and `rl_agent_config.json`, an hfmodels-android descriptor (family laya), a Python host and an example.

- **The reference is the publisher's runtime on the public test split, and it reproduces the card.** `phocinae-server` 0.1.5 (`Engine`, pure PyTorch, CPU fp32) on the 2,000 decisions of `LocalLLaMA/typed-decisions` test, rendered as the author's eval harness renders them, agrees with gold on 0.9055 (the author's own re-run) and gives the flip400 counts 13 / 11 / 9 / 6 — the published numbers to the count. Every check below compares against that run, row by row: argmax on the non-near-tie rows, max |Δp| ≤ 0.02 and mean |Δp| ≤ 0.002 over every option after the publisher's temperatures, all rows finite, and five control rows (one state with three numbers changed) that must move.
- **The graph is the laya contract** (`input_ids` or `inputs_embeds`, `attention_mask`, `qtype_onehot` → `token_logits`, `pooled_cls`; the host gathers the option markers, computes the four act features and runs the act head graph), written out from the publisher's modules with additive masks (−1e4, the diagonal always allowed) and the attention and head layers explicit.
- **fp16 arithmetic needs the LayerNorm inputs of encoder layers 12–21 and the final norm scaled by 2⁻⁶ (eps by 2⁻¹²).** Their squares reach 2.0e7 on the test rows, over float16's 65,504; without the rewrite Metal's default precision returns max |Δp| 0.287. The rewrite is bit-identical in fp32 and every published graph carries it; with it Metal default, the Galaxy S26's FP16 and FP16_WITH_FP32_ACCUM, and the Qualcomm HTP pass the bar.
- **The int8 table is the whole residual of the one-file form; the float16 table on the host is exact.** int8 EMBEDDING_LOOKUP in the graph: max |Δp| 0.0066 at window 256 and 0.0107 at 384 / 512 (fp32 FULLY_CONNECTED with the same table gives the same numbers). float16 rows looked up on the host: 3.8e-4 / 8.0e-4. A float16 table inside the graph passes on the CPU but the GPU delegate refuses to compile it; dynamic int8 FULLY_CONNECTED fails the bar (0.0835, a probability model).
- **Galaxy S26 (SM-S942Q, LiteRT 2.2.0 CompiledModel, Kotlin): every form passes on every row of the test split at every window.** One question's row on a rested phone (median of 20 after 5 warm-ups): GPU FP16 15.3 / 22.4 / 30.7 ms, NPU (HTP, on-device JIT, the host-lookup form in one partition) 16.8 / 19.7 / 26.9 ms, GPU FP16_WITH_FP32_ACCUM 20.4 / 30.0 / 40.4 ms, GPU FP32 26.9 / 42.1 / 62.7 ms, CPU 4 threads 39.0 / 63.0 / 96.6 ms at windows 256 / 384 / 512. The NPU compile takes 3.9 / 11.0 / 42.7 s cold and 0.2 s from LiteRT's cache. A 2,000-row parity leg drops the GPU clock from 1,300 to 461 MHz by its end, so the timing came from separate short legs after the phone cooled.
- **On the Mac (M4 Max, ai-edge-litert 2.2.0)**: Metal fp32 7.43 / 9.44 / 11.85 ms, Metal default 7.66 / 9.49 / 10.80 ms, CPU 8 threads 31.1 / 38.2 / 48.1 ms per question at the three windows.

Scripts, the Python host, the example and the gate activity: [`phocinae_work/`](phocinae_work/) (`README.md`, `REPRODUCE.md`, `conversion/`, `host/`, `examples/`, `android/gate/`).

## EMA Lightning (Turkish TTS, 8.6M: 4-step flow-matching DiT + HiFi-GAN, 48 kHz) — four static-shape graphs with the bookkeeping on the host — 2026-10-11

`ema_tts_work/` converts [canberkkkkkk/ema-lightning](https://huggingface.co/canberkkkkkk/ema-lightning) (Apache-2.0) into `text` / `align` / `step` (×4, Euler on the host) / `decoder` graphs, one multi-signature `.tflite` per kind (every length bucket a signature, weights shared; 27.3 MB in all), plus a torch-free host loop. Published: [litert-community/EMA-Lightning-LiteRT](https://huggingface.co/litert-community/EMA-Lightning-LiteRT). Details and the measured tables: [`ema_tts_work/README.md`](ema_tts_work/README.md).

```bash
cd ema_tts_work
python3.12 -m venv .venv && .venv/bin/pip install ema-lightning==1.1.2 litert-torch==0.9.4 ai-edge-quantizer soundfile scipy
mkdir -p precheck/weights && (cd precheck/weights && curl -sLO https://huggingface.co/canberkkkkkk/ema-lightning/resolve/main/ema.pt && curl -sLO https://huggingface.co/canberkkkkkk/ema-lightning/resolve/main/decoder.pt)
python oracle_ref.py 4                                     # the pip package on CPU (seed 0): six sentences -> fixtures/
python export_graphs.py --out out/fp32 --text 32,64,128,256 --align 32x48,64x96,128x192,256x384,256x512 --step 48,96,192,384,512 --decoder 48,120 --multisig
python verify_tflite.py --split --multi --forms text=fp32,sound=fp32,decoder=fp32     # min SNR 66 dB, frames identical
python quantize.py --mode fp16 --only decoder --match multi                           # decoder fp16 weights (lossless-grade)
python verify_tflite.py --split --multi --forms text=fp32,sound=fp32,decoder=fp16     # the ship set: min SNR 56 dB
python assemble_ship.py --out out/ship/EMA-Lightning-LiteRT --forms text=fp32,sound=fp32,decoder=fp16 --split --multi
python out/ship/EMA-Lightning-LiteRT/ema_host.py "Merhaba, bu ses telefonunuzun üzerinde üretiliyor." --out merhaba.wav
```

- Converter rewrites: the decoder's `ConvTranspose1d(k 10, s 5, p 2)` (neither SAME nor VALID) exported as VALID + crop; the aligner's `expand` (BROADCAST_TO, refused by the GPU delegate) replaced by a broadcast through the RoPE multiply; the ConvNeXt layer scale folded into the following linear (its rank-2 × rank-3-constant MUL is miscomputed by the ML Drift GPU backend, LiteRT #10445); one DiT step per graph (an unrolled 4-step graph duplicates every FC weight per step in the file).
- Measured on a Galaxy S26 (LiteRT 2.2.0): one 3.5 s sentence in 52.5 ms (text + align CPU, 4 steps on the HTP at 1.56 ms each, decoder on the GPU at fp16; RTF 0.015, first audio 21.7 ms), 56.5 ms without the NPU libraries, 149 ms all on the CPU. Dynamic int8 fails the 30 dB bar everywhere (the distilled DiT amplifies weight noise); fp16 activations fail on the text graph and on four unrolled steps.

## G9v3-3B — a dense 3B with a thinking switch: GPTQ-calibrated int4 on a Mac, and a GPU activation type declared per file

`g9v3_work/convert_g9v3_3b.sh` converts [ai9stars/G9v3-3B](https://huggingface.co/ai9stars/G9v3-3B) at commit `d9553445` — "a dense 3B causal language model … built for local deployment and resource-constrained scenarios"; 52 layers, hidden 2048, 16 attention heads / 2 KV heads, 130,560-token vocabulary, untied embeddings — into the two files of [litert-community/G9v3-3B](https://huggingface.co/litert-community/G9v3-3B). Environments (released wheels): litert-torch 0.9.4, litert-converter 0.4.0, ai-edge-quantizer 0.9.0, transformers 5.14.1 for the export; litert-lm 0.18.0 for the metadata step and for running the files; gptqmodel 7.3.5 on torch 2.14.1 under Python 3.12 for the int4 calibration.

```bash
# once: a Python 3.12 environment with gptqmodel 7.3.5, datasets, requests and torchvision, patched for Apple silicon (MPS)
/path/to/py312-gptq/bin/python g9v3_work/patch_gptqmodel_mps.py
EPY=/path/to/python-with-litert-torch-0.9.4 \
RPY=/path/to/python-with-litert-lm-0.18.0 \
CLI=/path/to/litert-lm \
GPY=/path/to/py312-gptq/bin/python \
  bash g9v3_work/convert_g9v3_3b.sh out_g9v3
```

| file | section | bytes | sha256 |
|---|---|---|---|
| `G9v3-3B_int8.litertlm` (3,102,493,024 B) | LlmMetadataProto | 5,201 | `4b6b001395b61a9ed65a4cca48fe42bffce09966212a2eab0468e1d0af679a28` |
|  | ExecutorMetadataProto | 7,102 | `fe71235f270549a4d0143191ef4be8a37c7958725588ee0274d421ceb0c0bcc7` |
|  | HF_Tokenizer_Zlib | 2,048,362 | `61a39298f9decdb79edb2ed38eead3cee1b30e6803d932f7e4db24b714339f9b` |
|  | TFLiteModel | 2,822,005,664 | `f540fbb1e489b48159d0a38bd95be69d079755d71956bccc23db535fafc27bc7` |
|  | TFLiteModel | 278,366,560 | `a71c9fa0791206e58c34fb6f7233351985a7bfa5444ba5972ad741c3dfe75081` |
| `G9v3-3B_int4.litertlm` (1,834,436,960 B) | LlmMetadataProto | 5,201 | `4b6b001395b61a9ed65a4cca48fe42bffce09966212a2eab0468e1d0af679a28` |
|  | ExecutorMetadataProto | 7,102 | `fe71235f270549a4d0143191ef4be8a37c7958725588ee0274d421ceb0c0bcc7` |
|  | HF_Tokenizer_Zlib | 2,048,362 | `61a39298f9decdb79edb2ed38eead3cee1b30e6803d932f7e4db24b714339f9b` |
|  | TFLiteModel | 1,553,951,792 | `8e151c6cab115b8ce4b1d7912209fdac3a6e999d36e698444f45d295c3b9e77b` |
|  | TFLiteModel | 278,366,560 | `a71c9fa0791206e58c34fb6f7233351985a7bfa5444ba5972ad741c3dfe75081` |

When the script was run from a copy of this repository's layout on 2026-10-13, building the calibration set from the datasets on the way, every section of both files came out byte-identical to the published files' sections listed here. The file sha256 differs from run to run: the builder writes a new `uuid` and `creation_timestamp` into the header, so compare sections.

- **GPTQ int4, calibrated on Apple silicon.** `gptq_quantize.py` runs gptqmodel 7.3.5 (4 bits, group 128, symmetric, desc_act off, damp 0.01, true-sequential) on 256 chat samples, each a user turn and an assistant turn rendered with the checkpoint's chat template with thinking off: 128 from HuggingFaceH4/ultrachat_200k, 64 from tatsu-lab/alpaca, 64 from the GSM8K training split. It reads the datasets at pinned revisions and checks that the built set hashes to `26e9cc95…` before quantizing; built that way, the set matched byte for byte the one the published file was calibrated on. The set itself is not in this repository: tatsu-lab/alpaca is CC BY-NC 4.0. `patch_gptqmodel_mps.py` makes five edits to gptqmodel 7.3.5 for Apple silicon (MPS); every run here used them. On one M4 Max the GPTQ checkpoint came out bit-identical in four runs, the published file's and three re-runs; other Apple silicon was not tried.
- **The GPTQ grid written as blockwise-128 int4, with a 2^-14 minimum on the block scales.** `scripts/ingest_gptq_dequant.py … fp32clip` writes the dequantized checkpoint and `recipes/gptqrec_int4_block128.json` (`dequantized_weight_recovery`) writes it back as blockwise-128 int4 with an int8 lm_head and int8 embedding. On this checkpoint that yields 781 block scales below 2^-14: 365 GPTQ group scales in layer 51 (down_proj 253, o_proj 112) and 416 all-zero blocks in layer 0 (gate_proj 208, up_proj 208). `export_clamped.py` raises them to 2^-14 before the payload is written, and `scan_scales.py` finds no zero or subnormal scale in the result; an earlier data-free int4 export of this checkpoint without that minimum stopped on the Mac CPU with XNNPACK's `unsupported scale value (0.000000)`.
- **Accuracy picks each file's GPU activation type.** GSM8K (the first 100 test questions, thinking off, greedy, at most 1024 output tokens, one process per question): bf16 82; int8 85 on the CPU, 84 on the GPU with fp32 activations and 69 with the runtime's default fp16; int4 78 on the CPU, 81 on the GPU with fp16 activations and 78 with fp32. So the int8 file declares fp32 and the int4 file fp16 (`prefer_activation_type` on both TFLiteModel sections; the declaration changes no section's bytes). `scripts/set_activation_type.py` does not rewrite a declaration that is already there, so the recipe declares once, on the undeclared file. The CPU path does not read the declaration.
- **Thinking is a switch, off by default.** The bundle embeds `g9v3.jinja`, built on LiteRT-LM's `models/qwen3` chat template with message content read as typed parts; on jinja2 and on minijinja it renders the same characters as the vendor template through transformers for 10 of 11 prompt shapes — the exception is a prompt with the thinking switch unset, which this template treats as thinking off. The generation prompt ends with `<think>\n\n</think>\n\n`; `--thinking true` on the CLI (or `enable_thinking` in the API) ends it with `<think>\n`, and the trace arrives on the `thought` channel. With thinking on, three single questions closed their trace and answered on both files; in a three-turn conversation the int4 file's third trace ran past the 1536-token reply limit without closing, while the int8 file answered all three turns. GSM8K was run with thinking off only.
- **Start token, stops, tool calls, sampling.** The runtime prefills the same token ids as transformers, with one `<s>` (id 0) at the start; the stops are ids 1 and 130073. The bundle declares no function calling: given a tool, the model's XML call `<function name="get_weather"><param name="city">Paris</param></function>` comes back as reply text. The bundle sets no sampler; on the 0.18.0 CLI, `--temperature` and `--top-p` take effect only together with `--top-k`.
- **KV budget 4096, six prefill signatures (1024, 256, 64, 16, 4, 1), embedder externalised.** The int8 file's decoder section is 2,822,005,664 bytes, over 2 GiB; not run on iOS. Both files also load and answer on litert-lm 0.17.1.
- Measured (prefill 1024 / decode 256 / context 2048, litert-lm 0.18.0 on the Mac, prebuilt `litert_lm_advanced_main` 2026-09-18 on the phones): Mac M4 Max GPU (WebGPU) int8 1429.0 / 62.43 tok/s, int4 1506.9 / 61.49; Galaxy S26 GPU int8 113.9 / 12.22, int4 402.9 / 18.46; Pixel 8a int8 on the CPU 20.85 / 3.53, int4 on the GPU 125.1 / 8.96. The harness stops a process when MemAvailable falls below 400,000 kB; on the Pixel 8a GPU it stopped every process of the fp32-declared int8 file, while the int4 file's lowest MemAvailable was 2,179,860 kB. Every decoder signature fully delegated on all three GPUs.

**Re-shipped 2026-10-18: the int8 file in the GPU-composite form (retrofit); the int4 file unchanged.** The published int8 file was rebuilt from the 2026-10-13 int8 file above without re-exporting: `tools/gpu_graph/gpu_graph_retrofit.py` rewrites the prefill and decode graphs into the shape litert-torch's `--apply_gpu_composites` produces (one `odml.cache_update` composite per layer instead of two DYNAMIC_UPDATE_SLICE writes, `odml.runtime_bmm` composites for the attention products, a `param_tensor` signature input) and keeps every weight buffer byte for byte; the metadata, tokenizer and embedder sections, and the `prefer_activation_type = "fp32"` item on both TFLite sections, are the previous file's bytes. The tool first reads the bundle's ExecutorMetadata state buffers into its report (`state_buffers`, `linear_attention_state`) and refuses a linear-attention state on LiteRT-LM 0.18.0 (README, Limits); this bundle declares 104 KV-cache buffers (52 `TYPE_GLOBAL_KEY_CACHE`, 52 `TYPE_GLOBAL_VALUE_CACHE`) and no other state, so it runs without a flag.

```bash
# LITERT_LM_CLI must point at a litert-lm 0.18.0 CLI before the command runs
python tools/gpu_graph/gpu_graph_retrofit.py G9v3-3B_int8.litertlm G9v3-3B_int8.retrofit.litertlm --decomp exporter --mask add_bcast --report retrofit_int8.json
```

| file | bytes | sha256 |
|---|---|---|
| `G9v3-3B_int8.litertlm` (re-shipped) | 3,103,656,288 | `8d9f5e8522e553458ab3fb7f4a461359367934d0741c5e82cd777da55ce69dc5` |
| `G9v3-3B_int4.litertlm` (unchanged, 2026-10-13) | 1,834,436,960 | `ea2c773ec678d44ffca43c0fca21a46622fd812fb9b53706f82d966d02e3c468` |

| re-shipped int8 file | section | bytes | sha256 |
|---|---|---|---|
| `G9v3-3B_int8.litertlm` (3,103,656,288 B) | LlmMetadataProto | 5,201 | `4b6b001395b61a9ed65a4cca48fe42bffce09966212a2eab0468e1d0af679a28` |
|  | ExecutorMetadataProto | 7,102 | `fe71235f270549a4d0143191ef4be8a37c7958725588ee0274d421ceb0c0bcc7` |
|  | HF_Tokenizer_Zlib | 2,048,362 | `61a39298f9decdb79edb2ed38eead3cee1b30e6803d932f7e4db24b714339f9b` |
|  | TFLiteModel (prefill_decode, `prefer_activation_type fp32`) | 2,823,168,928 | `6044d072c7797c578f6f5311698ed1c2152d2b3aed91f45bf894149334e73ff2` |
|  | TFLiteModel (embedder) | 278,366,560 | `a71c9fa0791206e58c34fb6f7233351985a7bfa5444ba5972ad741c3dfe75081` |

- **What changed and what did not.** Only the prefill_decode section is new: it grew from 2,822,005,664 to 2,823,168,928 bytes (+1,163,264) by the composite graph alone. The report lists the previous section's 540 buffers against the new section's 925: the 470 external (weight) buffers have the same sizes and the whole weight region is byte-identical (the same region sha256 on both sides), the 68 inline buffers are the same, 2 are empty, none mismatch. Per signature (prefill 1024, 256, 64, 16, 4, 1 and decode) the tool rewrote all 52 layers: 364 `odml.cache_update` composites (52 per signature) and 716 `odml.runtime_bmm` (104 in decode, 102 in each prefill signature) replace every DYNAMIC_UPDATE_SLICE and BATCH_MATMUL of the attention; the `param_tensor` input is INT32[1, 1, 1, 7]; the attention mask stays a FLOAT32 input added by broadcast over the 8 query heads of each KV head (`--mask add_bcast`). On the Mac CPU this form runs within 4 % of the previous file (M4 Max, prefill 1024 / decode 256, two passes in alternating order: 14.18 s against 14.71 s); on the CPU of the Galaxy S26 named below it took 4–10 % longer in two session pairs.
- **Why.** On GPUs the KV-cache write and the attention product reach the runtime as composite ops. Mac M4 Max (WebGPU, the previous file in the same windows): decode 62.8 → 84.2 tok/s, prefill 1428 → 1340 tok/s, so the 1024 + 256 workload takes 3.81 s instead of 4.80 s. Galaxy S26 (Google's Developer Device Platform device m1q-36, one session per file back to back, the median of cycles 2 and 3): prefill 122.6 → 137.0 tok/s, decode 13.1 → 17.3 tok/s, time to first token 8.43 → 7.54 s, the workload 27.89 → 22.27 s. The int8 file has no Pixel 8a GPU row (the 2026-10-13 card's memory guard; not retried).
- **Same answers.** The re-shipped file gives the previous file's texts on the 8-question gate (CPU 8/8, GPU 7/8, the same miss), the BOS check (20 / 15 / 20 prefilled tokens), the 3-turn conversation on CPU and GPU and the 39-length prompt check; GSM8K (questions 0–99, the same procedure as above, Mac GPU with the declared fp32 activations): 84, the previous file's 84 with 0 discordant questions. On the platform's Galaxy S26 the re-shipped file answered "What is the capital of France? Answer in one sentence." with "The capital of France is Paris." on the CPU and on the GPU (one prompt session each, two processes per session).
- **Why the int4 file is unchanged.** The same rewrite of the int4 file (1,835,600,224 bytes, `9c23ec0f…`) ran the Mac GPU 21 % faster (4.46 → 3.54 s) and the Galaxy S26 GPU 6.6 % faster (15.14 → 14.14 s), 10 % slower on the Pixel 8a GPU (36.7 → 40.4 s, one same-hold pair, Mali-G715), and scored 79 on GSM8K against the published file's 81 on the same Mac GPU with 8 questions differing (both files reproduce their own answers run to run; all 5 questions only the published file gets right run into the 1024-token limit on the rewritten file). With the runtime's fp16 GPU activations, which this file declares, the composite graph changes long greedy generations; on the fp32-declared int8 file the answers are identical. The int4 file therefore stays as published on 2026-10-13.
- **Verifying the retrofit output.** `litert-lm unpack` of the new file and of the previous file give the same bytes for every section but the prefill_decode one; the tool's `--report` JSON lists, per signature, the 52 layers it rewrote and their attention products, the composites it wrote, the unchanged buffers and the state buffers it read, and `litert-lm run --backend gpu --verbose` on the new file shows all 7 decoder subgraphs delegated (the embedder section's 7 small subgraphs run on the CPU as before).
- **Re-run 2026-10-18.** The command above, run again from this repository's clone against the 2026-10-13 int8 file: the five section sha256 values, the buffer counts (540 → 925) and the composite counts (364 / 716) equal at the published size; the file-level sha256 differs (`f1a97eaa7ad703369fb496db38f293770905590d309fee06cd69c22e0fd34b10`; only the packer's uuid and timestamp differ) while every section matches, so compare sections.

## 2026-10-13 — MiniCPM-V-4.7-1B: a Qwen3.5-text hybrid VLM as a fast_vlm bundle, the window-merger ViT re-authored in plain torch, and a chat template whose second turn matches upstream token for token

`scripts/ship_minicpmv47_1b.sh` (key `minicpm-v-4.7-1b` of `scripts/reproduce_vlm.sh`) converts [openbmb/MiniCPM-V-4.7-1B](https://huggingface.co/openbmb/MiniCPM-V-4.7-1B) at revision `984a89aa` (apache-2.0; "the latest model in the MiniCPM-V series, available in a 1B variant"; 1,554,476,272 parameters: a 24-layer text decoder in the Qwen3.5 text architecture — 18 GatedDeltaNet + 6 gated attention layers, hidden 1024, 248,094-token vocabulary, untied output head — and a 27-layer ViT, hidden 1152, patch 14, that merges 2×2 windows inside the encoder after layer 6) into the three files of [litert-community/MiniCPM-V-4.7-1B](https://huggingface.co/litert-community/MiniCPM-V-4.7-1B): `MiniCPM-V-4.7-1B-VL_int8.litertlm` (2,210,907,248 B: decoder int8 + int8 embedder + vision encoder and adapter fp16), `MiniCPM-V-4.7-1B-VL_int8vis.litertlm` (1,698,202,736 B: the vision encoder int8) and `MiniCPM-V-4.7-1B_int8.litertlm` (1,221,623,648 B: the text decoder alone). Scripts: [`minicpmv47_work/`](minicpmv47_work/), plus the Qwen3.5 rail in `qwen35_work/` and `qwen35vl_work/`.

```bash
PY_EXPORT=/path/to/py312-litert-torch-0.9.2/bin/python \
PY_VISION=/path/to/py314-litert-torch-0.9.3/bin/python \
PY_HF=/path/to/py312-transformers-5.19.0/bin/python \
LITERT_LM=/path/to/litert-lm-0.18.0/bin/litert-lm \
PY_SWAP=/path/to/litert-lm-0.17.1/bin/python SWAP_CLI=/path/to/litert-lm-0.17.1/bin/litert-lm \
FIXTURE_SRC=/path/to/three/512x512/photos \
  bash scripts/reproduce_vlm.sh minicpm-v-4.7-1b        # -> out/minicpmv47/*.litertlm
```

The script runs the commands the published files were built with, in this order: `make_text_ckpt.py` → `convert_qwen35_hybrid.py` with `QWEN35_FLOAT_FC_REGEX='Linear_in_proj_(a|b);'` → `set_activation_type.py --type fp32` → the dual template swap → `strip_start_token.py` (text file); `vision_fixtures.py` → `vision_ref.py` → `vision_parity_eager.py` → `vision_export.py --stage all` (vision); `export_qwen35vl_decoder.py` with `PREFILL=1024,256,64,16,4,1 CACHE=4096` and the same regex → `build_mv47_bundle.py --build` → `add_executor_metadata.py` (VL files); and last, `swap_template.py` with the shipped templates. It has not yet been run end to end from this repository's layout, so no byte-identity claim is made here. What the published files carry, from the swap reports (packed by litert-lm-builder 0.17.1, container format 1.6.0):

| file | section | bytes | sha256 |
|---|---|---|---|
| `MiniCPM-V-4.7-1B-VL_int8.litertlm` | LlmMetadataProto (template 1,364 B, `05f70447…`) | 1,529 | `8585c72ce768d0a70ceb2f7d2259ef12a482763ba0d66c6d57e29547d41de9d8` |
|  | TFLiteModel embedder | 257,027,584 | `49b596eebc65954ec33c36c3ce26a95f017a62e947f752cee57c7f41f956dd25` |
|  | TFLiteModel prefill_decode (`prefer_activation_type fp32`) | 854,453,360 | `9e334f2bdfcc9b0ceba160f149d06144b4647a8e7c3162834e1cda65839afe09` |
|  | TFLiteModel vision_encoder (fp16) | 1,043,472,832 | `739c5bdefb5158139ab202be209e098269277fd5cfa24edec3ad14609128e61e` |
|  | TFLiteModel vision_adapter (fp16) | 51,971,184 | `b06bdd1c2c832ab6dd39e13a3364a3a96e7e46a9ecd283911b432fe23db1165a` |
| `MiniCPM-V-4.7-1B-VL_int8vis.litertlm` | the same, with TFLiteModel vision_encoder (int8) | 530,775,664 | `100879f77932bc0a3330490cb3149d4d6cd6fa0cbc49332311702042a7c9243a` |
| `MiniCPM-V-4.7-1B_int8.litertlm` | LlmMetadataProto (template 866 B, `a07bb777…`) | 894 | `efc73fa3c1113a400070cae5846ce4893fa131f4da429f63c9f1239f31606135` |
|  | TFLiteModel prefill_decode (12 signatures, `prefer_activation_type fp32`) | 1,217,658,720 | `1f49e71a05bd9a983ccfe193d8f46ddfe0fd64c052a00a44c970da0351df46de` |

- **The text decoder comes out of the VLM checkpoint as its own model.** `make_text_ckpt.py` keeps the 320 `model.language_model.*` tensors as `model.*` plus the untied `lm_head`, drops the vision tower and the merger, and writes the text config as `Qwen3_5ForCausalLM` / `qwen3_5_text`; the checkpoint has no MTP weights to drop. On the same 48 text tokens the derived model and the full `MiniCPMV4_7ForConditionalGeneration` (fp32, eager) give identical logits (max |Δ| 0.0, argmax 48/48), because a text-only input takes the full model's 1-D position path. The float graph then matches upstream at all 48 teacher-forced positions (Pearson 0.99999999999, mean KL 1.2e-10); the int8 graph keeps 47 top-1s (the miss sits on a 0.37-logit margin) and all top-5s.
- **The 36 GatedDeltaNet decay/gate projections stay float.** The CPU backend quantizes the input of every int8 fully connected layer at run time; `QWEN35_FLOAT_FC_REGEX='Linear_in_proj_(a|b);'` keeps `in_proj_a` / `in_proj_b` (16×1024 each) float, as for the Qwen3.5 files of 2026-10-11/12; 151 fully connected layers, the untied head included, are int8.
- **The tokenizer declares `<|im_start|>` as its BOS, and the exporter writes it as the start token**, so the runtime would prepend one before the template's own `<|im_start|>`. `strip_start_token.py` removes it; the VL builder never writes one.
- **The vision tower is re-authored in plain torch (`mv47_vision.py`) so that it converts without GATHER, FLEX or CUSTOM ops**: the in-encoder window merger stays in raster order and attends through a block-diagonal constant mask (no token permutation), each 2×2 merge is four strided slices and a concat, and LayerNorm is fp16-safe — its input is divided by a power of two before the statistics (`ln_scales.json`: all 58 LayerNorms need a scale above 1, up to 256 from layer 10 on, where the residual stream reaches about 1,300). Against upstream `get_image_features` on four 448×448 images the fp32 TFLite pair correlates at 0.99999999996; with fp16 weights at 1.000000 (worst per-token cosine 0.999996); with the int8 encoder at 0.995 (worst token 0.911).
- **Two contract costs, measured on upstream fp32 before any conversion.** The bundle takes one 448×448 image (64 image tokens) where upstream keeps the aspect ratio and slices larger images, and the runtime feeds one-dimensional positions where upstream uses its canvas M-RoPE. On 3 images × 3 questions, native vs square 448 and canvas M-RoPE vs 1-D positions each gave 0 identical answers, 0 answers with changed content and 9 paraphrases, so no in-graph M-RoPE was added.
- **The template ends an assistant turn at `<|im_end|>` and opens the next turn with its newline.** The runtime prefills only the new part of each render and keeps the previous answer's stop token in the KV cache, so with the newline at the end of the assistant turn the second turn's context was the upstream 2-turn prompt minus that one newline (47 of 48 ids). With the shipped templates a full render is the same characters as upstream's (9 of 9 template cases against the transformers 5.19.0 processor, image expansion included), the second turn's prefill is 23 tokens, and the context equals upstream's 48 ids in 6 of 6 runtime runs (Python API 0.18.0: the VL int8 file on CPU ×3 and GPU, the int8vis and text files on CPU). The device numbers below were measured before this template swap; the swap changed no other section.
- **GPU sampling with a 248,094-token vocabulary.** The GPU sampler logs `TopK requires src tensor C dimension to be divisible by 4.` and samples over the first 248,092 ids; the two left out are `<timestamp>` / `</timestamp>`, which no chat prompt uses. Padding the logits to 248,096 inside the graph removed the line and did not change Mac GPU decode (157.47 vs 158.39 / 158.16 tok/s), so the files keep 248,094.
- Gates (litert-lm 0.18.0, Mac M4 Max, greedy, one process per question): the 8-question check 8/8 on CPU and GPU (a question passes on its regex or when the answer is the upstream fp32 text: upstream itself answers the rhyme with "white"), a 40-prompt length sweep (BANANA, fill 12–51) 40/40, and 9 image questions on 3 photos answered with the photo's content on CPU and on GPU. On the shipped files: 8/8 on CPU, the capital question on GPU, 3 of 3 images on CPU, and the card's commands on CPU and GPU; the VL and text files also answer on the litert-lm 0.17.1 and 0.16.0 CPU.
- Measured: Mac M4 Max (litert-lm 0.18.0, `benchmark -p 256 -d 256 --runs 3 --cache no`) GPU 2038.66 / 146.72 tok/s prefill / decode, CPU 624.72 / 42.65; Galaxy S26 (prebuilt `litert_lm_advanced_main` 2026-09-18, prefill 1024 / decode 256) CPU 476.75 / 36.46 (a sustained run; the CPU clock was capped in 14 of 16 samples), GPU 663.61 / 23.56, image to first token 0.93 s with the decoder and the vision encoder on the GPU; Pixel 8a, int8vis file on the CPU: image to first token 2.17–3.03 s; iPhone 18 Pro (CLiteRTLM.xcframework v0.16.0): text on Metal 72.85 tok/s decode, image to first token on the CPU 1.38 s. The Pixel 8a GPU engine did not fit (our harness stops a run below 400,000 kB of free memory; engine creation crossed it at the 4th of 7 signatures), and the Galaxy S26 NPU path rejected the 60 DynamicUpdateSlice ops of the KV-cache update (the first of 37 partitions did not compile within 150 s).
- Public pieces from this conversion (2026-10-14): `tools/vlm_text_ckpt/` (the derived text checkpoint with its logits proof against the full model), `docs/vision-tower-reauthor.md` (the tower pattern), `tools/chat_template_parts/dual/a07bb77727e4.jinja` with `two_turn_increment.py` (the second turn's newline), `docs/android-gpu.md` "Image prompts and timing with the LiteRT-LM Android binary".

## 2026-10-15 — EmbeddingGemma 2 Transcribe: EmbeddingGemma 2's audio encoder and a word reader as LiteRT graphs, with RMSNorms that run at fp16

`scripts/eg2_transcribe/` converts the audio tower and `embed_audio` of [google/embeddinggemma-2](https://huggingface.co/google/embeddinggemma-2) at revision `914f7f89` (Apache-2.0), and reader v1 of [mlboydaisuke/EmbeddingGemma-2-Transcribe-Reader](https://huggingface.co/mlboydaisuke/EmbeddingGemma-2-Transcribe-Reader) (a 4-layer transformer of 13.3 M parameters that reads each 40 ms soft token as one of 200,000 words or a blank), into the four `.tflite` files of [mlboydaisuke/EmbeddingGemma-2-Transcribe-LiteRT](https://huggingface.co/mlboydaisuke/EmbeddingGemma-2-Transcribe-LiteRT): `eg2_audio_f16safe_fp16w_w30.tflite` (589,273,696 B), `eg2_audio_f16safe_fp16w_w10.tflite` (589,015,376 B), `reader_w30.tflite` (361,494,720 B) and `reader_w10.tflite` (361,238,720 B). The Android app that runs them is [eg2-transcribe](https://github.com/john-rocky/eg2-transcribe) (`android/`). Inputs, environment and every command: [`scripts/eg2_transcribe/README.md`](scripts/eg2_transcribe/README.md).

```sh
python scripts/eg2_transcribe/export_encoder.py export --model hf/embeddinggemma-2 --fixtures eg2-transcribe/reader/fixtures --out out/eg2
python scripts/eg2_transcribe/export_reader.py export --reader-dir hf/reader --reader-code eg2-transcribe/reader/train --fixtures eg2-transcribe/reader/fixtures --out out/eg2
```

- **The encoder is loaded inside litert-torch's Gemma 4 patch.** litert-torch main (`0a81bf6`) rewrites the Gemma 4 audio classes for the GPU; the patch is a context manager registered for the Gemma 4 model types, so for `embedding_gemma2` the script wraps the load in it and checks the tower's classes. A load of the same model earlier in the process leaves the unpatched tower in place. The model is loaded with the default attention (sdpa): with transformers 5.19.0 the eager path hands the audio attention a float mask where it expects a boolean one.
- **Every RMSNorm is rewritten to run at fp16, exact in float32** ([docs/fp16-safe-norms.md](docs/fp16-safe-norms.md)). 76 of the 111 norm sites have a row sum of x² over 65,504. Each norm's input is scaled by 2^-k (k per site, `kmap.json`) and its eps by 4^-k, the rsqrt is clamped for padding rows, the attention's mask constant is −30000 instead of −1e9, and the output mask is computed from the input mask by float ops. In float32 the rewritten module is bit-equal to the original on every window, and the converted file to the unrewritten one on the CPU.
- **One window per file.** The CompiledModel API prepares every signature of a file, so a file with both windows compiles both on the phone. The NPU (Qualcomm HTP, compiled on the phone) takes the 10 s file: preparing the 30 s graph needed 6.8 GB on a Galaxy S26.
- **The reader's table stays as released.** The 200,000 × 768 word table is the release's float16 rows, stored as float16 with an explicit DEQUANTIZE, so nothing is lost; the reader's body stays float32. The table's scale and row norms are one constant MUL after its fully connected layer, and the argmax and the max run in the graph, so no 200,001-wide output leaves it. To cast only the table, the quantizer's op scope is the op's output tensor names joined by ";" with a trailing ";": a regex without the anchor on that ";" matches nothing and the file comes out unchanged.
- Device numbers are on the model card: on a Pixel 8a (LiteRT 2.2.0, GPU at fp32 precision) the 54 min 17 s LibriVox recording takes 276.8 s with WER 5.784 %, the PyTorch reader's WER on the same windows; on a Galaxy S26 the encoder takes 290.3 ms per 30 s window on the GPU at fp32 precision and 94.7 ms per 10 s window on the NPU.
- **Rebuilt on 2026-10-15 from the public inputs** (the model at `914f7f89`, the Reader repository at `1af9c5f5`, eg2-transcribe at `40ce307`, litert-torch main `0a81bf6` editable): the four files are byte for byte the published ones, their sha256 equal to the model repository's `FILES.sha256`. On the CPU the encoder files give the float32 module's output at a cosine of 0.9999999937 or more per token on the 20 test utterances, and the reader files give the PyTorch reader's word or blank on every token (5,036 and 4,426), with 20 of 20 and 11 of 11 transcripts equal to the reader's release notes.

## Vela-1.0-Encoder-307M-Embedding — a ModernBERT sentence embedder as static-window graphs: fp16 weights only, pooling the GPU delegate accepts, the Galaxy S26 HTP at 6.8 ms per text (2026-10-18)

Published: [litert-community/Vela-1.0-Encoder-307M-Embedding-LiteRT](https://huggingface.co/litert-community/Vela-1.0-Encoder-307M-Embedding-LiteRT) with `vela_L{64,128,256,512}_fp16.tflite` (one file: `input_ids` + `attention_mask` → the normalized 768-d embedding; fp16 FC weights and fp16 token table, CPU and HTP), `vela_L{64,128,256,512}_embeds_fp16.tflite` + `tok_embeddings_fp16.bin` (the host looks the token rows up in the float16 table; GPU at FP32 arithmetic and HTP), `Vela-1.0-Encoder-307M-Embedding_fp16.litertlm` (LiteRT-LM EmbeddingEngine: fp16 embedder + four-window encoder + the repo's tokenizer), `tokenizer.json`.

- **The reference is the publisher's own stack and it reproduces the card.** sentence-transformers 5.1.2 + transformers 4.57.6, CPU fp32, SDPA math kernel, on 4,184 texts (PAWS-X test en / es / ja / zh, STS-B v2 test, concatenated long texts; by dataset id + revision + row index): STS-B v2 Spearman 0.811867 vs the card's 0.811864. Every gate compares to that run row by row: cosine ≥ 0.999 on every text, top-1 retrieval agreement 100 % on non-near-tie queries, pair Spearman ≥ 0.999, the mask live, every output finite.
- **The graph is a direct trace with the publisher's YaRN RoPE tables as constants, LayerNorms of layers 12–21 and the final norm on x·2⁻⁶** (their sum of squares reaches 4e7, past float16), mean pooling and normalization in the graph. fp32 export: 0.99999977 to the reference (the converter's noise floor).
- **fp16 weights pass (0.99977 on every window); the int8 table fails the per-row bar and cannot be patched.** A per-row symmetric int8 token table rounds every row by 0.75 % on average, the control tokens by 2–3.6 %; 19 of 3,918 texts fall to 0.993–0.999, the PyTorch graph with the simulated table reproduces the LiteRT number, and keeping the 16 / 256 / 2,048 worst rows exact leaves the floor at 0.99516. int8 FC weights: 0.964. Everything shipped is float16.
- **The GPU delegate (LiteRT 2.2.0 on Metal, Adreno and Mali) refuses `torch.where` pooling and an fp16 token table in the graph.** The first is `SELECT_V2` with a broadcast condition (the Metal log names it; the phones only say "Shader compilation failed"), and the mixed graph fails to build; the second is `EMBEDDING_LOOKUP: Empty quantization params`. Pooling written as `(h·m).sum / max(m.sum, ε)` and `x · rsqrt(max(Σx², ε²))` is bit-identical on the 0 / 1 mask and compiles everywhere; the GPU form takes the looked-up rows from the host. The Qualcomm HTP runs both forms whole (1,689 / 1,689 ops, the fp16 table included).
- **GPU fp16 arithmetic misses the per-row floor on all three GPUs** (Metal 0.991–0.993, Adreno 0.9983, Mali 0.9960–0.9978 — about 1 % of the texts, every top-1 and the pair ranking kept), so the GPU rows ship at FP32 arithmetic; FP16_WITH_FP32_ACCUM passes the full bar on L128–L256 on both phones. The HTP's fp16 arithmetic passes everywhere (0.9994–0.9996).
- **Galaxy S26 (SM-S942Q, LiteRT 2.2.0), one text per call, median of 20 after 5 warm-ups on a rested phone:** HTP (C-API runner, JIT) 6.8 / 13.1 / 28.0 / 68.4 ms at L64 / 128 / 256 / 512 (JIT compile 4–40 s per window, cache it), GPU FP32 16.5 / 24.0 / 47 / 120 ms, CPU 4 threads 19.8 / 40 / 87 / 224 ms. Pixel 8a (Mali-G715): GPU FP32 76 / 92 / 181 / 408 ms, CPU 181 / 473 / 936 / 2,045 ms. Mac M4 Max: Metal FP32 6.1 / 7.3 / 11.0 / 20.2 ms, CPU 8 threads 22 / 30 / 45 / 79 ms. The time is the window's, not the text's.
- **The EmbeddingEngine bundle** passes the full set through `compute_embedding` (0.99977, top-1 1,873 / 1,873); with `insert_special_tokens` off the floor is 0.308 — the bos 2 / eos 1 metadata is the sentence-transformers contract.

Scripts, the check set, the gates and the card: [`vela_work/`](vela_work/) (`README.md`, `REPRODUCE.md`, `conversion/`, `card/`).

## granite-3.0-1b-a400m-instruct — a 32-expert MoE with SiLU experts: every expert for every token, each layer's experts as two fully connected layers, the decode logits padded for the GPU sampler, fp32 GPU activations declared (2026-10-21)

`granitemoe_work/convert_granite30_1b_a400m.sh` converts [ibm-granite/granite-3.0-1b-a400m-instruct](https://huggingface.co/ibm-granite/granite-3.0-1b-a400m-instruct) at commit `ffec3c35` — "an 1B parameter model finetuned from *Granite-3.0-1B-A400M-Base*"; 24 layers, hidden 1024, 16 attention heads / 8 KV heads, 32 SwiGLU experts per layer with 8 chosen per token, a 49,155-token vocabulary, tied embeddings — into the file of [litert-community/granite-3.0-1b-a400m-instruct](https://huggingface.co/litert-community/granite-3.0-1b-a400m-instruct). Export environment: Python 3.12 with `granitemoe_work/requirements-export.txt` (litert-torch at git commit `0d3869c6`, transformers 5.20.1, torch 2.14.0, ai-edge-litert-nightly 2.4.0.dev20261018, ai-edge-quantizer-nightly 0.10.0.dev20261019, litert-converter 0.6.0.dev20261016, litert-lm-builder 0.18.0); litert-lm 0.18.0 for the activation-type step and for running the file.

```bash
EXPORT_PY=/path/to/python-with-requirements-export \
RUN_PY=/path/to/python-with-litert-lm-0.18.0 \
LITERT_LM=/path/to/litert-lm \
MODEL_DIR=/path/to/granite-3.0-1b-a400m-instruct \
REFERENCE=/path/to/granite-3.0-1b-a400m-instruct_int8.litertlm \
  bash granitemoe_work/convert_granite30_1b_a400m.sh out_granite30_1b_a400m
```

`MODEL_DIR` is a local copy of the pinned commit, for example from `hf download ibm-granite/granite-3.0-1b-a400m-instruct --revision ffec3c35bdfd97a06f0b4cd5fcc92cd9b1584445 --local-dir DIR`; `check_pinned.py` checks every file against the Hub's record for the commit before anything is exported. `REFERENCE` is optional: the published file, which `sections.py` then compares with the result section by section (exit 1 if any section differs).

| file | section | bytes | sha256 |
|---|---|---|---|
| `granite-3.0-1b-a400m-instruct_int8.litertlm` (1,372,745,360 B) | LlmMetadataProto | 3,423 | `313fa47a32798372ea550084a22e424cd2c28188ad5ba7199c78ffee47ccf4b1` |
|  | ExecutorMetadataProto | 3,238 | `f99346f85243cedb0ba32532ec01f6f448e1e92ae6deac7042d5d6a19f2e491e` |
|  | HF_Tokenizer_Zlib | 676,317 | `3492a684fd405181ade688039be1245601bcb69c921a52f033a2c1d0afccb5d9` |
|  | TFLiteModel (`prefer_activation_type` fp32) | 1,372,008,080 | `64fd19fe2655595c2a89c203110120d0bb9860d2e2da555ecc9bcaa5ba860cb2` |

Run as written above from a copy of this repository's layout on 2026-10-21 (`MODEL_DIR` = a local copy of the pinned commit, all 11 files equal to the Hub's record), the script rebuilt every one of these sections byte for byte, with the same header items, in 127 s on a Mac M4 Max; the file itself differs from the published one only in the 41 header bytes of the packer's `uuid` and `creation_timestamp` (rebuilt sha256 `cbcc79ecd0a74efaf0f5261bfc4fa8e838c69faf39e5a1022ca6df8b6a557435`, published `b0b786a5672c25d4aeb8ed90b7363294703809a1f95a5dfb8fb3dcf5db11d39f`, both 1,372,745,360 B), so compare sections.

- **Every expert for every token.** litert-torch's sparse `litert_moe` export is out for two reasons: it computes its experts with GELU-tanh (`generative/layers/moe.py` L207 at `0d3869c6`) while this model's experts use SiLU, and LiteRT-LM 0.18.0's GPU refuses the `moe` op it writes (`moe only supports activation='gelu'.` on a two-layer test model). The `litert_moe_sequential` route computes all 32 experts of each layer for every token with the model's SiLU (the routing weight is zero outside the token's top 8): each token costs the compute of all 1.33B parameters.
- **The experts as two fully connected layers.** `fused_experts.py` registers its forward as `litert_moe_sequential` in transformers' experts interface and builds, once before the trace, gate_up [32768, 1024] and down [1024, 16384] per layer; each expert's routing weight multiplies its slice of the activations before the down projection. `export_fused.py` installs it and runs `export_one.py`. On a two-layer granitemoe model with random weights (hidden 128, 8 experts, top 2) the fused export in fp32 matched PyTorch's logits with the SiLU experts at 20 of 20 positions (largest difference 5.1e-7 of the largest logit, against 5.2e-3 for a GELU-tanh reference). The decode signature has 1,266 nodes against 10,889 for the per-expert export; on a Mac M4 Max in one measurement window the fused graph decoded at 153.1 tok/s on the GPU against 63.6–63.8, and at 42.7 and 53.6 tok/s on the CPU against 30.5–30.6.
- **The decode logits padded to 49,156 columns.** `pad_logits.py` wraps litert-torch's decode module and appends one column of -1e4 to its logits (the vocabulary is 49,155). Without it every GPU run printed "TopK requires src tensor C dimension to be divisible by 4." and, where the CPU began a reply with `<|tool_call|>` (id 49154), the GPU began with other text; with it the real 49,155 columns are bit-identical on the CPU (8 prompts, 239 positions) and the GPU begins those replies with `<|tool_call|>`.
- **IBM's template, extended for litert-lm 0.18, and the checkpoint's tokenizer.json.** `granite30_tools.jinja` is IBM's chat template (`tokenizer_config.json` of the pinned commit) with three additions: message content as a string or a list of text / tool-response parts, the tools block written by a macro that gives the bytes of `tojson(indent=4)` (the 0.18 runtime's `tojson` takes no arguments, and IBM's template fails there with `too many arguments (in template:16)`), and a `tool` role rendered as IBM's `tool_response` turn. On litert-lm 0.18.0 its renders equal IBM's template in transformers for 15 of 15 conversations without tools and 20 of 20 with tools (two further test conversations with a float in a tool schema differed only in its spelling, `1e-05` in transformers against `0.00001` in the runtime). `--tokenizer-verbatim` packs the checkpoint's `tokenizer.json` byte for byte (the unpacked section has the Hub file's git blob sha1, `11f3f2c6`).
- **No start token; stops; tool calls as text.** bos = eos = `<|end_of_text|>` with `add_bos_token` false, so the bundle declares no start token; the stops are id 0 and 14 strings the exporter derives from the template's turn ending. The model type is `generic_model`, which has no tool-call parser: a call arrives as reply text, `<|tool_call|>` followed by a JSON list of calls.
- **int8 = `dynamic_wi8_afp32`.** Blockwise int4 (OCTAV, block 32 and block 64) on the per-expert export was tried and not shipped: the first eight tokens of its answers equalled bf16's in 3 and 4 of 8, against 7 of 8 for int8 on the same graph.
- **fp32 GPU activations, declared in the file.** `scripts/set_activation_type.py --type fp32` adds `prefer_activation_type = "fp32"` to the TFLite section's header items; every section's bytes stay the same. GSM8K (questions 0–99, step by step, greedy, at most 1024 output tokens, a fresh process per question): bf16 34 in transformers; on the Mac GPU 35 / 35 / 35 with the declaration and 28 / 29 / 29 with the runtime's default fp16 activations (the third pass scored 28 before one question, whose process had logged a WebGPU validation error, was re-run); 39 / 35 on the Mac CPU. With the declaration the Mac GPU decodes 5.4 % slower and prefills 17.7 % slower.
- **`PYTHONHASHSEED=0`.** The exporter collects the stop tokens in a Python set and writes them in its iteration order, so without a fixed seed the LlmMetadata section's bytes change from run to run (the order of the stops).
- Measured on a Mac M4 Max (prefill 1024 / decode 256 / context 2048, litert-lm 0.18.0, two benchmark processes per backend): GPU (WebGPU) 3245.07 / 146.53 and 3228.70 / 146.15 tok/s, CPU 617.89 / 53.11 and 617.85 / 52.99; the 8-question gate 8/8 on both backends (bf16 8/8).
- Measured on a Galaxy S26 (Adreno 840) and a Pixel 8a (Mali-G715) through LiteRT-LM v0.18.0's `litert_lm_advanced_main`, built from source at the tag (`--benchmark_prefill_tokens=1024 --benchmark_decode_tokens=256 --max_num_tokens=2048 --num_iterations=3`, each value the mean of iterations 2 and 3 of a process): Galaxy S26 GPU prefill / decode 351.305 / 27.430, 352.895 / 27.105 and 346.705 / 27.285 tok/s (every run with the GPU clock lowered by heat partway through), CPU 355.190 / 22.785 and 368.095 / 23.725; Pixel 8a CPU 132.295 / 13.580 and 129.405 / 12.055, GPU 128.405 / 12.630 and 128.295 / 12.535. The GPU is the faster backend on the Galaxy S26, the CPU (by 1.6 % in time) on the Pixel 8a. The 8-question gate: 8/8 on both phones' GPUs, 7/8 on both CPUs. Against the runtime's default fp16 activations in the same session, the fp32 declaration costs prefill × 0.330 and decode × 0.841 on the Galaxy S26 GPU (64 decoded tokens) and × 0.496 and × 0.873 on the Pixel 8a GPU.

## clef-flash (Cloudflare) — a decision model that answers typed questions about text, JSON values and a picture with probabilities and never generates text, as raw LiteRT graphs for desktops: row graphs at five lengths and a picture tower at two tile sizes (2026-10-22)

Published: [litert-community/clef-flash-LiteRT](https://huggingface.co/litert-community/clef-flash-LiteRT) with five row graphs `clef-flash_rowprefill_embeds_L{256,512,1024,2048,4096}_fp16fc.tflite` (13,847,841,008 to 13,936,871,792 B), two towers `clef-flash_vision_tower_g{256,448}_fp16fc.tflite` (907,587,008 and 907,891,136 B), the tables the host reads (the checkpoint's bfloat16 `embed_tokens` and `lm_head` rows, 2,034,237,544 and 2,034,237,536 B, and the towers' position rows in float32), the checkpoint's joint schema head and `tokenizer.json`, `contract.json`, a Python host without PyTorch, the test fixtures and the model card: 75.5 GB in 133 files with the card (`SHA256SUMS` lists the other 132). The source is `Cloudflare/clef-flash` at `f0b7d3cd65296abfe459e6c228b618f5d8b26d2f` (Apache-2.0), post-trained from `Qwen/Qwen3.5-9B`. Given a state (text or a JSON value, and optionally one picture) and typed questions (noul, choice, score), it returns a probability per option in the response shape of the provider's `systemone()`. The reference is the provider's own code in float32 on the CPU: 214 fixture runs, 40 held-out runs and three control records. A file passes when it gives the reference's most likely option on every question that is not a near tie (the two most likely options 0.02 or less apart), with max |Δp| ≤ 0.02 and a mean of run means ≤ 0.002, and when each control moves some option by more than 0.02. On an Apple M4 Max (macOS 27.0, ai-edge-litert 2.2.0), the published folder answered all 243 runs it can make (text, and pictures at the 256 and 448 px tiles) through its own host on Metal at float32, with every node on the delegate in one partition: 545/545 plus 5/5 near ties, max |Δp| 1.65e-3 (a picture through the float16-weight tower; 1.08e-5 on the text runs), mean 7.3e-6, and the three controls moved 0.907, 0.036 and 0.975. On the CPU (XNNPACK, 8 threads), the first 20 runs of each length and the controls (69 runs) give 207/207 plus 3/3, max |Δp| 1.62e-3 and mean 1.3e-5.

- **The row graphs take embeddings and the rotary tables as inputs.** `embeds` float32 [1, L, 4096] (the host writes the checkpoint's bfloat16 rows as float32, and the tower's rows at the picture's positions), `valid` [1, L] and `cos` / `sin` [1, L, 64] (the host computes them from the three M-RoPE position planes) go in; the final-norm hidden state of every position comes out. The joint schema head (121.8M parameters) runs in NumPy on the host. Text and picture records use the same row graphs.
- **One file per length, and a call computes every position.** The host runs a request on the smallest present length that holds it and refuses a request longer than 4,096 tokens (the provider's code reads up to 16,384). One call on Metal at float32 takes 344.1, 654.4, 1,308.5, 2,707.0 and 5,570.7 ms (L256 to L4096); one decision request with the host's work takes 360.7, 673.3, 1,346.9, 2,742.9 and 5,687.2 ms (236 to 4,083 tokens), of which the head takes 16.6 to 91.1 ms; a request with a 960 × 1280 photo takes 1,410.7 ms at the 256 px tile and 1,493.8 ms at 448 px. One call on the CPU (XNNPACK, 8 threads) takes 1,288.9, 2,407.3, 4,744.5, 9,570.8 and 20,090.4 ms (the last on a busy machine); one decision request on the CPU takes 1,276.5 ms at 236 tokens and 4,726.2 ms at 1,007, also on a busy machine. A compiled row graph held 37.8 to 38.9 GB on Metal (68.1 to 69.6 GB at the peak of the compile), and closing it gave the memory back.
- **Run the GPU at float32 precision.** At Metal's default precision (float16 storage), the norms' sums of squares overflow: the L256 file gave 15,968 all-zero rows and moved the probabilities by up to 0.253, flipping three answers. With the input of every norm that overflowed scaled down by a power of two, it stayed finite and still moved them by 0.0486. The towers give wrong picture rows at the default precision (cos 0.34 to 0.36 against the provider's).
- **The towers take their position rows as an input.** ai-edge-litert 2.2.0 on Metal computed a FULLY_CONNECTED output plus a constant (the position rows) wrongly while it delegated every node (cos 0.016 to 0.024 against the expected values); with the position rows as the graph's second input `pos`, the tower's output rows match the provider's (rel. RMS 7.3e-4 and 9.8e-4 with float16 weights).
- **Dynamic int8 weights broke the calibration.** The L256 graph with int8 FULLY_CONNECTED weights moved the probabilities by up to 0.398 on the CPU (three flips) and 0.101 on Metal, so the files keep float16 weights (13,837,541,376 B per row graph).
- **No phone ran these graphs.** One row graph holds 13.8 GB of float16 weights, more than the memory of a 12 GB phone.
- **Reproduce it on a Mac with three Python environments.** The provider's reference runs on Python 3.12.11 with torch 2.9.0 and transformers 5.17.0; the conversion, quantization and graph checks on Python 3.14.6 with litert-torch 0.9.4, litert-converter 0.4.0, torch 2.13.0, transformers 5.14.1, ai-edge-litert 2.2.0 and ai-edge-quantizer 0.9.0; the host and the end-to-end checks on Python 3.14.6 with `host/requirements-host.txt`. `conversion/README.md` gives the commands in order; the scripts are the ones that ran, with their comments and a few lines of code that named paths of the machine the conversion ran on changed (listed there).
- **Check a download with the example.** `examples/run_example.py --check` (Metal at float32) and `--check --accelerator cpu` answer four public fixture requests (on the 256- and 512-token graphs, one with a drawn picture at the 256 px tile) and send twelve requests the host refuses. Both printed PASS: max |Δp| 8.71e-5 on Metal and 8.75e-5 on the CPU against the provider's float32 answers, and every refused request named its error. `--request examples/request.json --image fixtures/images/clock_3oclock_256x240.png --tile 256` returned the same response as the provider's code, value for value; a request that is refused exits with code 3.

Scripts, the host, the test fixtures, an example and the reproduction guide: [`clef_flash_work/`](clef_flash_work/) (`REPRODUCE.md` with the sources, the steps and the published names; `conversion/README.md` with the commands in order). The graphs, the tables, the head and the tokenizer are on the Hub only.

## parakeet-redux (TDT speech recognition, ternary encoder → LiteRT: int4 for the CPU, fp16 for the GPU and the NPU)

[`parakeet_redux_work/`](parakeet_redux_work/) converts [moondream/parakeet-redux](https://huggingface.co/moondream/parakeet-redux) (NVIDIA's parakeet-tdt-0.6b-v3 with a ternary encoder) into plain LiteRT `.tflite` graphs: a log-mel front end and an encoder for each window length (10, 20 and 30 s) and one TDT decoder (`predict` / `joint`), with the greedy TDT loop on the host. No NeMo: the reference is transformers' `ParakeetForTDT` with the ternary weights expanded. Published: [litert-community/parakeet-redux](https://huggingface.co/litert-community/parakeet-redux) (CPU set 1.14 GB, GPU / NPU set 3.68 GB, with an Android runner).

```sh
cd parakeet_redux_work
# three Python 3.12.11 environments at the versions that made the published files (scripts/requirements-venv-*.txt = uv pip freeze, macOS arm64)
uv venv -p 3.12.11 .venv-ref && uv pip install -p .venv-ref -r scripts/requirements-venv-ref.txt
uv venv -p 3.12.11 .venv-lt && uv pip install -p .venv-lt -r scripts/requirements-venv-lt.txt
uv venv -p 3.12.11 .venv-q && uv pip install -p .venv-q -r scripts/requirements-venv-q.txt
# inputs at pinned revisions (scripts/dl_targets.tsv: URL, local path, bytes, sha256 of the weights and the test sets)
.venv-ref/bin/hf download moondream/parakeet-redux --revision 2bf128600aac4b16946f7ed8372e56117fe5e23b --local-dir hf/redux
.venv-ref/bin/hf download nvidia/parakeet-tdt-0.6b-v3 --revision 541d1f99c6b0c3cd0b11a95167540bb8edefd82b --include "*.json" --local-dir hf/orig
curl --create-dirs -L -o data/librispeech/test.clean-00000-of-00001.parquet \
  https://huggingface.co/datasets/hf-audio/open-asr-leaderboard/resolve/de79e57adf9983993bcc72586d7313460825f5eb/librispeech/test.clean-00000-of-00001.parquet
.venv-ref/bin/python scripts/prep_librispeech.py data/librispeech/test.clean-00000-of-00001.parquet data/sets/librispeech-test-clean
.venv-ref/bin/python scripts/make_subset.py data/sets/librispeech-test-clean --every 32     # S32: 82 utterances
# the PyTorch reference on S32: 30 s windows + valid-length mask (out/s32_ii/*.npz: mel, encoder output, tokens)
.venv-ref/bin/python scripts/s32_modes.py
# 30 s graphs (litert-torch 0.9.4): mel30, the fp32 encoder, the decoder (and out/r2/enc30_r1.tflite, an earlier wrap of the encoder from scripts/enc_smoke.py kept as the comparison base of the rewrite, not shipped); then the rank-3 attention matmuls lifted to rank 4
.venv-lt/bin/python scripts/r2_build.py
.venv-q/bin/python scripts/r2_fb.py lift out/r2/enc30_fp32_raw.tflite out/r2/enc30_fp32.tflite
.venv-q/bin/python scripts/r2_pack4.py out/r2/enc30_fp32.tflite out/r2/enc30_int4x.tflite --json results/r2_int4x.json
# 10 / 20 s buckets and the LayerNorm-rescaled encoders (results/r4/fix.json, shipped here: the k of each of the 120 LayerNorms, x · 2^-k; scripts/r4_diag.py makes it from a range pass)
.venv-lt/bin/python scripts/r4_build.py
for n in enc10_fp32 enc20_fp32 enc10_fp32s enc20_fp32s enc30_fp32s; do
  .venv-q/bin/python scripts/r2_fb.py lift out/r4/${n}_raw.tflite out/r4/$n.tflite --json results/r4/lift_$n.json
done
for b in 10 20; do .venv-q/bin/python scripts/r4_pack4.py out/r4/enc${b}_fp32.tflite out/r4/enc${b}_int4x.tflite --json results/r4/int4x_$b.json; done
for b in 10 20 30; do .venv-q/bin/python scripts/r2_half.py out/r4/enc${b}_fp32s.tflite out/r4/enc${b}_fp16ws.tflite --json results/r4/fp16ws_$b.json; done
# the decoder with int8 weights (ai-edge-quantizer dynamic_wi8_afp32)
.venv-q/bin/python scripts/r4_dec8.py make
# signatures read back from the files, then the token check on S32 (fp32 buckets: every window equal to the reference)
for n in mel10 mel20 enc10_fp32 enc20_fp32 enc10_int4x enc20_int4x enc10_fp32s enc20_fp32s enc30_fp32s enc10_fp16ws enc20_fp16ws enc30_fp16ws; do
  .venv-q/bin/python scripts/r2_fb.py census out/r4/$n.tflite --json results/r4/census_$n.json
done
.venv-q/bin/python scripts/r4_contract.py
.venv-q/bin/python scripts/r4_p0.py --form fp32 --buckets 10,20,30
```

Run as printed in a clean checkout on 2026-10-21 (macOS 27 on an M4 Max; the commands from `s32_modes.py` onward took 20 min, peak memory 15.1 GB in the token check): the ten published `.tflite` files and `tokenizer.json` came out byte for byte (sha256 equal to the model repo's), and the token check (`r4_p0.py`) gave the reference's tokens on all 83 S32 windows.

Published names: `out/r4/enc{10,20}_int4x` + `out/r2/enc30_int4x` → `parakeet_redux_encoder_{10,20,30}s_i4.tflite`, `out/r4/enc{10,20,30}_fp16ws` → `parakeet_redux_encoder_{10,20,30}s_f16.tflite`, `out/r4/mel10` / `mel20` + `out/r2/mel30` → `parakeet_redux_mel_{10,20,30}s.tflite`, `out/r4/dec_int8` → `parakeet_redux_decoder_i8.tflite`; `tokenizer.json` is the original's. The Android runner (`runner/redux_asr_runner_b.cc`, `runner/build_android.sh`, `runner/tdt_text.py`) is the one in the model repo. The docstrings of `scripts/r2_build.py` and `scripts/r4_build.py` name two helpers that are not in this copy (`r2_lift.py`, `r4_post.sh`); the commands above do those steps (`r2_fb.py lift`, `r4_pack4.py`, `r2_half.py`).

Env (`scripts/requirements-venv-*.txt`, every package pinned): `.venv-ref` = torch 2.14.1 + transformers 5.20.2 + tokenizers 0.23.3 + pyarrow + soundfile (+ jiwer 4.0.0, kaldialign 0.12.0, num2words for the scoring); `.venv-lt` = litert-torch 0.9.4 + torch 2.13.0 + transformers 5.20.2 + litert-converter 0.4.0; `.venv-q` = ai-edge-litert 2.3.0 + ai-edge-quantizer 0.10.0 + soundfile. Python 3.12.11, uv. The scoring (`scripts/wer.py`) imports the Open ASR Leaderboard's `normalizer/` at huggingface/open_asr_leaderboard b2e6f0491796: copy that folder next to the scripts.

**Ternary → int4 with the model's own codes and scales.** The checkpoint stores each ternary weight as a code 0 / 1 / 2 (−1 / 0 / +1) with an fp16 scale per 128 weights (`scripts/unpack3.py` expands them). The 240 ternary linear layers go through ai-edge-quantizer's blockwise-128 int4 with the payload written as code − 1 and the model's fp16 scales written back over the quantizer's (it rounds blockwise scales through bf16), each times the factor the converter folded into that weight: 1 in 168 layers, 0.5 in 48 (exact) and the attention scale 0.2973 in the 24 `k_proj`, rounded to fp16. `r2_pack4.py` checks every layer: payload × scale equals the expanded weight exactly in 216 layers and within 4.9e-4 relative in the 24 `k_proj`. The 24 relative-position projections fold into constant position tables and stay float32.

**Float16 needs a LayerNorm rescale.** The first LayerNorm of block 0 sees inputs up to about ±8,300, and its squared deviation reaches 6.8e7 over the 55 windows of the range pass, beyond fp16's largest value 65,504; with fp16 arithmetic the encoder returned only blanks on a Mac GPU in our test. Each of the 120 LayerNorms takes x · 2^-k and eps · 4^-k (k = 3–9 from a range pass, `r4_diag.py` → `results/r4/fix.json`); in fp32 the output is bit-identical to the unscaled norm's, and every square stays under 65,504; the GPU (fp16 arithmetic) and the Hexagon NPU run these `fp16ws` files.

**Masks as float arithmetic.** The encoder takes the mel graph's `mel_mask` and builds the subsampling masks by strided slices and MUL, the attention key bias as (mask − 1) × 30,000, so the graph has no SELECT_V2 and no integer op; the rank-3 attention BATCH_MATMULs are lifted to rank 4 in the flatbuffer (`r2_fb.py lift`) because the converter folds a batch-1 rank-4 matmul back to rank 3.

**Decoding is TDT, not CTC.** The text of a window is `tokenizers` `Tokenizer.decode(ids, skip_special_tokens=True)` on the window's token ids, repeated ids kept; transformers' `ParakeetTokenizer._decode` by default groups repeated ids (the CTC rule) and drops real repeats ("1966" → "196").

## iris-3b Turbo (speridlabs) — text-to-image as 20 LiteRT graphs: a 3B pixel-space DiT and the Qwen3-VL-4B text encoder with float16 weights and float32 activations, every graph on the phone's GPU at FP32 (2026-10-22)

[`iris3b_work/`](iris3b_work/) converts [speridlabs/iris-3b](https://huggingface.co/speridlabs/iris-3b) Turbo (Apache-2.0; 4 steps, no classifier-free guidance, no VAE) and the language stack of [Qwen/Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct) into the files of [litert-community/iris-3b-Turbo-LiteRT](https://huggingface.co/litert-community/iris-3b-Turbo-LiteRT) for 768 × 768 images: 12 text-encoder graphs (one per hidden state the DiT reads), the text adapter, 6 trunk chunks and the pixel stage, 12,869,097,872 bytes with every FULLY_CONNECTED weight float16 behind a DEQUANTIZE, plus the tokenizer files, the bfloat16 embedding table and the host tensors. The Android app that runs them on the phone's GPU with LiteRT 2.2.0: [LiteRT-Models `iris3b/`](https://github.com/john-rocky/LiteRT-Models/tree/main/iris3b).

```sh
cd iris3b_work
OUT=out   # every file goes under OUT: about 55 GB (checkpoints 15 GB, float32 graphs 25.7 GB, the model folder 13.7 GB)
# Python 3.12.11 with every package pinned (macOS arm64)
uv venv -p 3.12.11 .venv && uv pip install -p .venv -r requirements.txt
# inputs at pinned revisions: the checkpoint, the text encoder, the model repo's run configuration and sums, the upstream code
.venv/bin/hf download speridlabs/iris-3b turbo/config.yaml turbo/distill.yaml turbo/model.safetensors --revision adc05b909fb05d4276d26aae5a52b36406b2ef26 --local-dir "$OUT/weights/iris-3b"
.venv/bin/hf download Qwen/Qwen3-VL-4B-Instruct --include "*.json" --include "*.txt" --include "*.safetensors" --revision ebb281ec70b05090aa6165b016eac8ec08e71b17 --local-dir "$OUT/weights/Qwen3-VL-4B-Instruct"
.venv/bin/hf download litert-community/iris-3b-Turbo-LiteRT model/iris3b_config.json model/SHA256SUMS --revision 743fd20acefd7f9ddcf3585f88bcda3257f7213a --local-dir "$OUT/published"
git clone -q https://github.com/speridlabs/iris-3b "$OUT/upstream" && git -C "$OUT/upstream" checkout -q 13fcf6a4f7fcb36561eac3daa5ad30d0b0a1822e
(cd "$OUT/weights" && shasum -a 256 -c) < inputs.sha256
# float32 graphs (text encoder, DiT), checked against SHA256SUMS.fp32, and the DiT's host tables
.venv/bin/python export_te.py --out "$OUT"
.venv/bin/python export_dit.py --out "$OUT"
(cd "$OUT/fp32" && shasum -a 256 -c) < SHA256SUMS.fp32
# float16 weights, the host files and the run configuration -> OUT/model, checked against the model repo's sums
.venv/bin/python fp16_weights.py --out "$OUT"
.venv/bin/python make_assets.py --out "$OUT"
(cd "$OUT/model" && shasum -a 256 -c SHA256SUMS)
```

Run as printed on 2026-10-22 (macOS 27.0 on an Apple M4 Max with 128 GB, OUT on an external USB disk): the 23 files the scripts write came out byte for byte (sha256 equal to the model repo's `SHA256SUMS`; `iris3b_config.json` is copied from it), and the 20 float32 graphs matched `SHA256SUMS.fp32` (16 lines from the original conversion; those for the four 1.8 GB DiT graphs, whose float32 bytes depend on the process, from three earlier processes of `export_dit.py`). The commands after the environment took 1,162 s (downloads 396 s, conversion 645 s, checks 121 s), and `export_dit.py` peaked at 22.7 GB of memory.

The non-obvious parts (more in [`iris3b_work/README.md`](iris3b_work/README.md)):

- **The tensor names carry the Python module and class names** (`te_modules.TEChunk/te_modules.TELayer_0`, `dit_modules.DualChunk/...`): renaming a module file, a class or a submodule attribute changes the bytes, not the computation, and so does the order of operations in `forward` (normalizing and transposing q, then k, against normalizing both first: the same numbers, two different files). The trace values do not enter the file: zeros here, the activations of a real prompt in the published export, the same bytes.
- **The DiT is re-authored as chunk modules whose every op the GPU delegate takes**, each rewrite exact in real arithmetic: the interleaved RoPE as a half-split rotation with the de-interleave baked into the q / k rows, the 20 / 5 GQA heads reordered into plain 5-head attentions, max-normalized RMSNorm, unfold / fold as host permutations. What depends only on t (the timestep embedding, every block's adaLN rows at the four fixed sigmas) is a host table, and the text adapter runs once per prompt.
- **Float16 weights, float32 activations.** Float16 holds the bfloat16 weights exactly except values below 2^-17 in magnitude (largest change 2.98e-8) and the text adapter's attention-query rows, which carry the folded softmax scale (largest change 1.50e-5); the graphs are half of float32 (12,869,097,872 against 25,733,449,508 bytes). In the conversion, int8 weights in either trunk group and float16 compute in any one group of trunk or pixel graphs missed 35 dB PSNR / 0.98 SSIM against the all-float32 PyTorch image on a prompt with numbers and punctuation (upstream's own bfloat16 text encoder: 30.3 dB), so the weights are float16 and the graphs run at FP32 precision.
- **litert-torch 0.9.4 lowers the text adapter's attention output projections to a rank-2 FULLY_CONNECTED and a mixed-rank ADD** (`[1, 1, 3600, 2560] + [3600, 2560]`), with or without their bias (its MLP output projections, which also feed a lone ADD, stay rank 4); with int8 weights the GPU delegate of LiteRT 2.2.0 refused the adapter (`Unable to parse bc coord for BATCH axis`, on a Mac GPU and a Galaxy S26).
- **The run configuration is copied from the model repo, not generated**: its `source` / `source_sha256` fields record the files of the original conversion, and neither the app nor the Python host reads them; `make_assets.py` checks its host offsets and shapes against the `iris_host.f32` it writes and its graph signatures against the graph files in `OUT/model`.

Measured on the phones and published in [LiteRT-Models `iris3b/README.md`](https://github.com/john-rocky/LiteRT-Models/tree/main/iris3b) (the app's first image after the install, GPU at FP32, 768 × 768): Galaxy S26 212.6 s, Pixel 8a 468.8 s; 70.92 to 106.84 dB PSNR against the all-float32 PyTorch pipeline with the same noise on three prompts.

## NVIDIA-Nemotron-Labs-Dory1.0-4B — a looped nGPT model: the loop unrolled with a KV cache per pass, GPTQ int4 on the data-free scales, an MLP normalization that fits fp16, and GPU composites in the int4 graph

`dory_work/convert_dory_4b.sh` converts [nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16](https://huggingface.co/nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16) at commit `b366b87f` — "a looped (latent-recurrent) large language model (LLM) trained by NVIDIA"; 72 blocks (36 attention, 36 MLP), of which the middle 40 run twice per token, hidden 2560, 16 query / 4 KV heads of 256, 27 sliding-window and 9 full-attention layers, a 131,072-token vocabulary, untied embeddings, OpenMDW-1.1 — into the two files of [litert-community/NVIDIA-Nemotron-Labs-Dory1.0-4B](https://huggingface.co/litert-community/NVIDIA-Nemotron-Labs-Dory1.0-4B): the int4 file for the GPU and the int8 file for the CPU; `dory_work/convert_dory_4b_1loop.sh` makes this repository's third file, the one-loop int4 file, from the checkpoint and the published int4 file (the end of this section). Environments: this repository's pinned toolchain (uv.lock: litert-torch 0.9.4, ai-edge-quantizer 0.9.0, litert-converter 0.4.0, ai-edge-litert 2.2.0, torch 2.13.0, transformers 5.14.1, litert-lm-builder 0.18.0) for the export, the quantization, the GPU composites and the pack; `dory_work/requirements-gptq.txt` (torch 2.11.0, numpy 2.2.6, transformers 4.57.6) under Python 3.12 on Apple silicon for the calibration render, the Hessians and GPTQ; a Python with transformers and datasets for g9v3_work's calibration-set builder; litert-lm 0.18.0 for running the files.

```bash
# once: the GPTQ environment (Apple silicon; GPTQ runs on the GPU through MPS)
python3.12 -m venv /path/to/py312-dory-gptq && /path/to/py312-dory-gptq/bin/pip install -r dory_work/requirements-gptq.txt
EPY=/path/to/python-with-litert-torch-0.9.4 \
GPY=/path/to/py312-dory-gptq/bin/python \
CPY=/path/to/python-with-transformers-and-datasets \
  bash dory_work/convert_dory_4b.sh out_dory
```

| file | section | bytes | sha256 |
|---|---|---|---|
| `NVIDIA-Nemotron-Labs-Dory1.0-4B_int4.litertlm` (2,403,851,584 B) | LlmMetadataProto | 10,786 | `18b8cd341757ddeed554f683531fca03179783fae4f57a86194a6d92a2cd83d0` |
|  | ExecutorMetadataProto | 7,654 | `92cdc12b0ffd6ede3f9cbe73cdd82f3b7452e300beb85c93fabceaac2af1d8ba` |
|  | HF_Tokenizer_Zlib | 2,806,488 | `86d8891c7665ef9237a81ad2debf94159389f38af2798e9dacdbcc4352d2bdc1` |
|  | TFLiteModel (prefill / decode) | 2,059,135,920 | `8c6b0004db5445d189a9e77022322495f07f655e57fb22bd02fae363b8442543` |
|  | TFLiteModel (embedder) | 341,843,264 | `6401ef0e5389496a0ce7e6ff7175cdeedaf91b333f140767c6c945637d53c566` |
| `NVIDIA-Nemotron-Labs-Dory1.0-4B_int8.litertlm` (4,046,675,264 B) | LlmMetadataProto | 10,786 | `18b8cd341757ddeed554f683531fca03179783fae4f57a86194a6d92a2cd83d0` |
|  | ExecutorMetadataProto | 7,654 | `92cdc12b0ffd6ede3f9cbe73cdd82f3b7452e300beb85c93fabceaac2af1d8ba` |
|  | HF_Tokenizer_Zlib | 2,806,488 | `86d8891c7665ef9237a81ad2debf94159389f38af2798e9dacdbcc4352d2bdc1` |
|  | TFLiteModel (prefill / decode) | 3,701,950,880 | `16fb48a5ea5c42175c5e2c962ed7670ad4006d906489a86014761afd03ce76f9` |
|  | TFLiteModel (embedder) | 341,843,264 | `6401ef0e5389496a0ce7e6ff7175cdeedaf91b333f140767c6c945637d53c566` |

Run as published on 2026-10-22 from a copy of this repository's layout, in an empty directory (the checkpoint and the G9v3 tokenizer files downloaded by `fetch_pinned.sh` with the script's own arguments and checked against the Hub's record by the fetch step, the G9v3 calibration set built from the datasets on the way; after the 8 GB download the steps took 16 minutes on a 128 GB M4 Max, peak 58.7 GB in the export), this script rebuilt the ExecutorMetadata, tokenizer and embedder sections of both files byte for byte, and every weight inside their decoder sections: all 289 int4 code buffers (3,638,558,720 codes, the GPTQ codes included; the 320 calibration samples were the same token ids) and all 289 int8 weight buffers.

The decoder sections themselves came out as the same computation in another order: the two RoPE inverse-frequency constants (sliding and full attention) are stored the other way round, and the operators of the four signature subgraphs follow that order (1,874 bytes in each file); `tools/tfl_compare/tfl_struct.py`, which compares two TFLite sections ignoring tensor, buffer and operator indices and names, finds the four signatures equal and tells such a graph apart from the same graph without the GPU composites and from one without the fp16 rewrite, and the 666 composite subgraphs of the int4 file are identical. The LlmMetadata section holds the same 16 stop tokens in another order. Both orders come from Python's per-process string hashing: `dory_modeling.py` builds the two RoPE tables in set order, and litert-torch 0.9.4 writes the stop tokens in set order; the published files were exported without a fixed seed, and the script runs under `PYTHONHASHSEED=0`.

The script's own sections, where they differ from the table: LlmMetadataProto `d2f7da482bca6cd6448568f940736718f1c09db866abfc1587944f39f220673b` (both files), int4 decoder `b49746834689cbe03c66caac80d2fe9234db9a2e41953862c80a50da6620cc66`, int8 decoder `6520dfcc986e577fec318591d119f08228676d3708f9610e95a055cceeca646a`. The `gpu` step alone, run on the int4 graph that the published int4 file was made from, gave the published int4 decoder section byte for byte. A second run of the `gpu` and `pack` steps gave every section of both files byte for byte again, and a run of the whole script earlier the same day, before the `gpu` step was added, gave the same int8 sections and the same int4 decoder as this run's before its `gpu` step; the GPTQ codes in the published files were computed the day before.

The file sha256 differs from run to run (the packer writes a new uuid and creation timestamp), so compare sections; `python scripts/section_identity.py <published> <rebuilt>` names the sections that moved.

- **Files of this repo the script calls** besides `dory_work/`: `g9v3_work/gptq_quantize.py` with `--calib-only` (sha256 `2a29567c873e608d272a284e25c5445067fc807f762bf4f47366ee07facfdbc5`), which builds the G9v3 entry's calibration set from three datasets at pinned revisions and checks it against `26e9cc95…`; it needs the G9v3 tokenizer files, which the fetch step downloads (5 files, no weights). `tools/gpu_graph/gpu_graph_retrofit.py` (sha256 `15a04736751f1d7e219fd2e3fa6cdca91b563145146577dd479b4a7a38b30c91`), which the `gpu` step loads by path.
- **A model definition of its own.** transformers has no Dory class, so `dory_modeling.py` writes the forward for litert-torch's `export_hf` from the formulas of NVIDIA's vLLM pull request (vllm-project/vllm#60149 at 55076950): no RMSNorm, every hidden vector, query and key L2-normalized (`x · rsqrt(Σx² + 1e-6)`), the residual as a learned step between two normalized vectors, q and k normalized after RoPE (θ 1e4 sliding, 1e7 full), a sigmoid output gate on the attention, SwiGLU, logits scaled per vocabulary entry. The module must be importable as `dory_modeling`: its name is part of every tensor name in the graph. A cache hook gives each (layer, loop pass) its own KV slot (56) and tells the executor metadata how those slots are laid out; the vendor tokenizer adds no BOS and the template writes none, so the bundle has no start token. Checked before any quantization against a PyTorch implementation of the same formulas (`dory_ref.py`), itself checked against NVIDIA's llama.cpp branch of the model: the largest difference between the two references among the top-20 first-step logits of 9 prompts (26 to 1,911 tokens) is 2.04e-4, and a float export with fp32 activations produced the reference's 64-token texts 9 of 9 on the Mac GPU and CPU.
- **The loop, unrolled.** The 20 recurrent attention/MLP pairs appear twice in the graph and share their weights; the loop input is added back at the start of each pass, as in NVIDIA's code. Every pass of every attention layer keeps its own KV cache (NVIDIA's per-loop mode, 56 caches). The last-loop mode (36 caches, each pass overwriting the chunk's K/V) passed the same checks and decoded at the per-loop rate on the GPU but slower on the CPU (× 0.89 on the Mac, × 0.84 on a Galaxy S26), so the files keep per-loop caches.
- **An MLP normalization that fits fp16.** On a GPU the runtime stores activations in fp16 by default. The sum of squares of the 7,168-wide SwiGLU output overflows fp16, the normalized row comes out as zero, and the int4 weights scored 83 on GSM8K with fp16 activations against 94 with fp32. `export_dory.py` rewrites the normalization of the 11 MLP blocks in `fp16_shift.json` as `ms = m · 2^-k; y = ms · min(rsqrt(Σms² + eps · 4^-k), 2^k / √eps)` — the same numbers in fp32 (powers of two are exact and the cap is never exceeded), a sum 4^k smaller in fp16 — with k (1 to 3) per block from the block's largest sum of squares and largest value over 25 prompts (rule in the JSON). The int4 weights then scored 91 with fp16 activations (90 with the GPU composites below), and the int4 file declares no activation type: on a Galaxy S26 GPU fp16 activations ran int4 weights 4.3× faster in prefill and 1.5× in decode than fp32.
- **int4 = GPTQ codes on the data-free scales.** Data-free int4 (blockwise-32 min-max, `recipe_int4_b32.json`, block scales floored at 2^-14 as XNNPACK requires) scored 85 on GSM8K against 92 for int8 and 95 for the reference, with the loss spread over the network. `fc_map.py` maps the 289 FULLY_CONNECTED weights of the data-free file to the checkpoint (the converter folds the per-channel scales that follow g_proj, gate_proj and up_proj into those weights) and writes their codes and block scales out; `gptq_hessians.py` collects a Hessian per weight group from `dory_ref.py` in fp32 on MPS over 320 samples (150,059 tokens): the 256 G9v3 calibration samples rendered again with Dory's template and the 64 GSM8K training questions with a thinking-on prompt followed by the model's own greedy continuation (`calib_think_ids.jsonl`, made once with llama.cpp's Dory branch on the BF16 weights, two loops, up to 1,024 tokens per sample); `gptq_dory.py` chooses the codes with the file's own block scales fixed (group 32, damping 0.01, natural column order, the Cholesky factors in fp32 on MPS: torch 2.11's CPU Cholesky returned wrong factors on the Mac used here); `write_codes.py` writes the codes over the data-free codes and checks that no other byte changed. No GSM8K test question is in the calibration set.
- **GPU composites in the int4 graph.** The `gpu` step runs `dory_work/retrofit_gpu.py`, which loads `tools/gpu_graph/gpu_graph_retrofit.py` by path, unchanged, and adds one thing to its mask rewrite: Dory's graph derives the sliding-window mask from the mask input with an ADD (the band), and that derived mask gets the broadcast-add view the tool gives the mask input. In each of the four signatures the 56 KV-cache writes become `odml.cache_update` composites and the attention matmuls `odml.runtime_bmm` composites (112 in decode, 110 in each prefill signature, where the last attention block has no consumer; 666 in all). Every weight buffer stays byte for byte; the embedder and both metadata files are copied from `int4/` unchanged. The int8 file has no such step. On the Mac GPU (fp16 activations) the composites made decode × 1.275 and prefill × 1.021, on a Galaxy S26 GPU decode × 1.17 (one run in each order) with 0.7 GB more GPU memory, and on the Mac CPU decode × 0.967; with fp32 activations the 8-question text is unchanged, and with fp16 activations GSM8K went from 91 to 90.
- **Two runs give the same metadata.** litert-torch 0.9.4 writes the stop tokens in the iteration order of a set of strings, so the script runs under `PYTHONHASHSEED=0`, as `scripts/convert.py` does; the packer still writes a new uuid and creation timestamp into the header, so compare sections, not files.
- **int8 = `dynamic_wi8_afp32`** on the same float export; the CPU file. The int4 weights on a CPU answer some questions differently from process to process and from CPU to CPU (the rhyme question of the 8-question check, without the GPU composites: "purple" in 3 of 4 fresh processes on the Mac CPU and 4 of 4 on a Pixel 8a), and their CPU prefill is a quarter to a third of the int8 file's on the Mac (80.7 against 302.0 tok/s) and on a Galaxy S26 (29.0 against 92.1 tok/s, an earlier build); on a Pixel 8a it is × 0.86 (10.8 against 12.5 tok/s), where the int4 weights decode 2.47× faster but answer the rhyme question wrong.
- **The template.** NVIDIA's template with three edits for the runtime's template engine (`dory_runtime.jinja`): a macro that reads message content as a string or as text parts, that macro used at the seven places the template reads content, and the two history lines rewritten so that every earlier assistant turn is rendered as `<think></think>` plus the answer and each turn's prompt starts with the previous turn's (the runtime raises otherwise). Its renders equal transformers' render of NVIDIA's template in 8 of 8 cases (4 conversations × thinking on / off). Thinking is on unless `enable_thinking` is false.
- Measured: see the [model card](https://huggingface.co/litert-community/NVIDIA-Nemotron-Labs-Dory1.0-4B). GSM8K (questions 0–99, thinking off, greedy): int4 90 on the Mac GPU with fp16 activations, int8 93 on the Mac CPU, the BF16 reference in llama.cpp 95. Prefill 1024 / decode 256 tokens (tok/s): int4 on the Mac M4 Max GPU 1033.9 / 57.71, on the Galaxy S26 GPU 284.0 / 12.09; int8 on the Mac M4 Max CPU 302.0 / 15.70, on the Galaxy S26 CPU 83.0 / 5.06, on the Pixel 8a CPU 12.5 / 1.18.

**The one-loop file.** `dory_work/convert_dory_4b_1loop.sh` makes `NVIDIA-Nemotron-Labs-Dory1.0-4B_int4_1loop.litertlm`, the int4 file with the 20 recurrent attention/MLP pairs applied once per token instead of twice — the trade the source card calls "Elastic Inference Compute": "By changing how many times the shared block of layers is applied, you can trade inference compute for accuracy to suit each task, with no retraining and no model swap." It runs no GPTQ and needs only the first environment:

```bash
EPY=/path/to/python-with-litert-torch-0.9.4 bash dory_work/convert_dory_4b_1loop.sh out_dory_1loop
```

- **Steps.** `fetch` downloads the 7 checkpoint files and the published int4 file (litert-community/NVIDIA-Nemotron-Labs-Dory1.0-4B at commit `61cb6bbe`) with `fetch_pinned.sh` and checks each against the Hub's record of its commit. `loops` makes a model directory whose `config.json` differs from the checkpoint's in one line, `n_recurrent_loops` 2 → 1 (`set_loops.py`; `dory_modeling.py` reads the loop count from the config, and the other six files are links to the snapshot), so the export unrolls one pass and lays out 36 KV caches. `export` and `quantize` are the int4 file's: `export_dory.py` with the same arguments (the fp16 rewrite of the same 11 blocks), then `recipe_int4_b32.json` with the 2^-14 floor and the runtime template. `copy` (`copy_int4.py`) writes the published int4 file's GPTQ codes into the new data-free graph weight by weight: the tensor names of a one-loop graph are not those of the two-loop graph, so each weight is found by its fp16 block scales, which the data-free quantization reproduces (289 of 289 matched one to one, 82 to 91 % of each weight's 4-bit codes already equal before the write), and every byte outside the written codes is checked unchanged; it also cuts the published file's metadata and embedder sections out for the pack. `gpu` runs `retrofit_gpu.py` with `tools/gpu_graph/gpu_graph_retrofit.py` as in the two-loop script (144 `odml.cache_update` and 282 `odml.runtime_bmm` composites in the four signatures), and `pack` packs the file with this export's executor metadata.

| file | section | bytes | sha256 |
|---|---|---|---|
| `NVIDIA-Nemotron-Labs-Dory1.0-4B_int4_1loop.litertlm` (2,400,558,400 B) | LlmMetadataProto | 10,786 | `18b8cd341757ddeed554f683531fca03179783fae4f57a86194a6d92a2cd83d0` |
|  | ExecutorMetadataProto | 4,894 | `4c228aed9053fd7ec7a2e82246f6f0dcb0d698f4391014ae06b6d0ab025f3d87` |
|  | HF_Tokenizer_Zlib | 2,806,488 | `86d8891c7665ef9237a81ad2debf94159389f38af2798e9dacdbcc4352d2bdc1` |
|  | TFLiteModel (prefill / decode) | 2,055,837,232 | `20fcd8d70faa6a6bbc6198f0ac5165eeae38eee34b521ffff337e1edce1d39c6` |
|  | TFLiteModel (embedder) | 341,843,264 | `6401ef0e5389496a0ce7e6ff7175cdeedaf91b333f140767c6c945637d53c566` |

Run as published on 2026-10-23 from a copy of this repository's layout, in an empty directory (the 7 checkpoint files copied in from an earlier download that day by the same `fetch_pinned.sh` at the same commit, the published int4 file downloaded by the script's own `fetch_pinned.sh` command, and every file then checked by the fetch step against the Hub's record; after the downloads the steps took about 6 minutes on a 128 GB M4 Max, peak 57.6 GB in the export), the script rebuilt all five sections of the published one-loop file byte for byte. A second run of every step in a second empty directory (the downloaded files copied in) gave the same five sections, and the same graph as the first run before and after the `copy` step. The file sha256 differs from run to run (the packer writes a new uuid and creation timestamp), so compare sections, as for the two-loop files.

- The metadata, tokenizer and embedder sections are the int4 file's (the same sha256 as in the first table). The one-loop export's own metadata (`d2f7da482bca6cd6448568f940736718f1c09db866abfc1587944f39f220673b`) holds the same 16 stop tokens in the order of the fixed seed, and its own embedder equals the int4 file's byte for byte; the script packs the int4 file's metadata section, so the two int4 files carry the same one.
- Measured (the model card): GSM8K (questions 0–99, thinking off, greedy) 86 on the Mac GPU with fp16 activations, against 90 for the int4 file. Prefill 1024 / decode 256 tokens (tok/s), each against the int4 file in runs that alternated the two: on the Mac M4 Max GPU 1619.6 / 88.18 (the int4 file 1034.8 / 58.46), on the Galaxy S26 GPU 439.6 / 17.32 (the int4 file 274.5 / 11.08).

## WeVisOne-VL-2B — Qwen3-VL's DeepStack carried exactly in an 8,196-wide image embedding, M-RoPE positions computed in the graph from the 1-D position and an image flag, float16 weights (2026-10-25)

`scripts/ship_wevisone.sh` (key `wevisone-vl-2b` of `scripts/reproduce_vlm.sh`) converts [tencent/WeVisOne-VL-2B](https://huggingface.co/tencent/WeVisOne-VL-2B) at revision `3dad9ce8` (apache-2.0; "a unified vision-language model developed by WeChat Vision, Tencent Inc. Built upon Qwen3-VL"; 2,441,744,384 parameters in `model.safetensors`: a 24-block ViT, hidden 1024, patch 16, DeepStack taps after blocks 5 / 11 / 17, and a 28-layer text decoder, hidden 2048, vocabulary 152,680, untied output head) into `WeVisOne-VL-2B_f16.litertlm` of [litert-community/WeVisOne-VL-2B](https://huggingface.co/litert-community/WeVisOne-VL-2B) (4,899,179,008 B, sha256 `97f29a6a1b672ad09afe33ee8c41589f62bac19a46092458bc4c96ef3f26adcb`). The 9.5M-parameter pixel decoder (`mask_decoder.pt`, segmentation and matting) is not part of this file. Scripts: [`wevisone_work/`](wevisone_work/); the method: [docs/deepstack-wide-embedding.md](docs/deepstack-wide-embedding.md).

```bash
PY_REF=/path/to/py312-torch-2.11.0-transformers-5.12.1/bin/python \
PY_EXPORT=/path/to/py312-litert-torch-0.10.0/bin/python \
LITERT_LM=/path/to/litert-lm-0.18.1/bin/litert-lm \
  bash scripts/reproduce_vlm.sh wevisone-vl-2b      # -> wevisone_work/out/r2/bundle/WeVisOne-VL-2B_f16.litertlm
```

`PY_REF`: Python 3.12 with torch 2.11.0, transformers 5.12.1, safetensors 0.8.0, numpy 2.2.6, Pillow 12.2.0 and tokenizers (the source package's own pins). `PY_EXPORT`: Python 3.12 with litert-torch 0.10.0, litert-converter 0.5.0, ai-edge-litert 2.3.0, ai-edge-quantizer 0.10.0, litert-lm-builder 0.18.1, torch 2.14.1, transformers 5.14.1 and jinja2. The script runs, in order: the drawn test pictures and the NASA photos (`fixtures/fetch_public.sh`), the text-only checkpoint (`tools/text_ckpt.sh` = `tools/vlm_text_ckpt/make_text_ckpt.py` with the M-RoPE keys dropped from the config), the vision tower (`tools/vision_taps.py consts`, `tools/wv_vision_r2.py export` and `quant` with `LN_SCALES=ln_scales_512.json`), the decoder (`tools/wv_export_decoder.py`, then `tools/wv_quant_decoder.py fp16`), the file (`tools/build_wv_bundle.py --build`, `scripts/add_executor_metadata.py`, `tools/wv_bundle_check.py`). It has not been run end to end from this repository's layout yet, so no byte-identity claim is made here. The published file's TFLite and tokenizer sections, from the check of the shipped file:

| section | bytes | sha256 |
|---|---:|---|
| TFLiteModel embedder (fp16 table, rows [1, 1, 8196]) | 625,380,064 | `7825c0ff84389436c4f149953301830979b6f7880d2c1ee2e9e64a562bd8629a` |
| TFLiteModel prefill_decode (fp16 weights, prefill 1024 / 256 / 64 / 16 / 4 / 1 + decode, KV 2048, no activation type) | 3,458,271,600 | `f07f9a4c35653092fffd3047512dc85d771396d40ad5161906296cb7775a220e` |
| TFLiteModel vision_encoder (fp16, [1, 512, 512, 3] → [1, 1024, 4096]) | 611,814,624 | `47d4089213cfdb6da006263ac92cfee92e8e2ff36c4e59027e4fe16dce2e837a` |
| TFLiteModel vision_adapter (fp16, → [1, 256, 8196]) | 201,460,224 | `d0c8dc73b2a5a358a3bb06438d643cadbc454245f011d89853d184d9d646c530` |
| HF tokenizer (unpacked) | 11,607,393 | `3bb172429b952469b51c64783f52fa01c26d1b30f1f31a4693e385a43f9cfc5e` (= the repo's `tokenizer.json`) |

- **DeepStack in full.** Adding the three DeepStack features into the one `fast_vlm` image embedding changed the float32 model's answers on 29 test questions at 512 × 512 (boxes 41/55, counts 4/6, points 17/44, OCR character error 0.472 against the unchanged model), and with the runtime's 1-D positions as well the coordinate answers fell apart (boxes 0/55). The adapter therefore emits 8,196 values per image token (the main embedding, the three DeepStack features, an image flag, 3 zeros), the embedder pads text rows with zeros, and the decoder adds each DeepStack part after layers 0, 1 and 2 (`tools/wv_patch.py`, installed in the export process only). A float32 build of these graphs gives the unchanged model's tokens on all 29 questions, on the Mac CPU and GPU.
- **Positions.** The decoder derives Qwen3-VL's (t, h, w) from the runtime's 1-D position and the image flag (equal to transformers' `get_rope_index` on 85,542 positions). The system turn is fixed in the template (15 tokens before the image), and in a conversation without a picture every position after token 270 is 240 lower than the model's own (a 353-token summary request broke down after 7 tokens).
- **The vision tower** is re-authored in plain PyTorch for 512 × 512 (`tools/vision_taps.py`): constant positions and rotary tables, raster order through the encoder, the 2 × 2 merges as strided slices and a concat (no GATHER_ND), and LayerNorm inputs divided by powers of two (all 48 encoder norms, 2 to 64). `ln_scales_512.json` carries the published scales: they were calibrated on four pictures, one of them a crop of a figure of the source package that is not redistributed here. Against the model's own image features the fp32 encoder + adapter pair gives a worst per-token cosine of 0.9999997, the fp16 pair 0.999995 (17 pictures).
- **Float16 weights, no activation type declared.** On the Galaxy S26 GPU the int8 decoder builds were faster (22.56 against 13.87 tok/s of text decode) but reproduced fewer of the float32 model's answers (points 28 of 44, OCR character error 0.226), and our harness stopped the build that declared float32 GPU activations when free memory fell to 479,412 kB (its limit is 700,000 kB). On the Galaxy S26 CPU this file decoded 0.14 tok/s in one process while the phone swapped (the CPU expands the fp16 weights to a 6.9 GB float32 cache); the Pixel 8a CPU was not started.
- **The float32 form for desktop CPUs (`FP32=1`).** With `FP32=1` the script also packs the float32 graphs of steps 3 and 4 into `WeVisOne-VL-2B_fp32.litertlm`, with the decoder section's `prefer_activation_type` set to fp32. The conversion's own build of it (9,774,433,860 bytes, sha256 `89a91b1896d525decf425232282e8b5deac7ec0bfaad184ddfa122343c852062`) is the fastest of the builds that met the agreement thresholds on the Mac M4 Max CPU (int8 and int4 builds decoded faster there and missed them): a 512 × 512 picture and a question to first token 3.32 s, decode 14.1 tok/s, points 44/44, peak resident memory 50.9 GB without a cache. It is not in the model repository, and this step has not been run from this repository's layout.
- **Template.** The system turn "You are a helpful assistant." is always rendered and a passed system message is not; only the first image item of the first user turn is rendered; an assistant turn ends at `<|im_end|>` and the next turn opens with its newline, so when the runtime prefills only the new turn, the token ids match a fresh render of the two turns.
- **A runtime defect on every file with an embedder section.** From the second message, LiteRT-LM passes the previous answer's stop token to the model with a zero embedding (`runtime/executor/llm_litert_compiled_model_executor.cc` L1326–1341, L577–584, L413–447 at v0.18.1; the same at v0.16.0 and on main of 2026-10-23). `wevisone_work/tools/r6_twoturn_probe.py` replays a file's graphs with the stop token's row embedded, zeroed, or written as the runtime writes it; on `litert-community/Qwen2.5-Coder-1.5B-Instruct` the runtime's second answer equals the zero-row replay ("I asked you to remember the number 4182.") and not the right one ("You asked me to remember the number 4182."). Details: [docs/deepstack-wide-embedding.md](docs/deepstack-wide-embedding.md).
- Agreement with the reference (litert-lm 0.18.1, greedy, the picture resized to 512 × 512 with PIL BICUBIC; the reference = the float32 CPU run of the source package at `40f735b8`, `tools/run_arms.py` run B): on 29 questions over 17 pictures this file gives boxes 55/55, counts 6/6, points 42 / 39 / 35 of 44 and OCR character error 0 on the Mac CPU / Mac GPU / Galaxy S26 GPU (the model itself in bfloat16: 55/55, 5/6, 38/44, 0), and on 10 text questions the float32 model's pass / fail on every backend. `fixtures/` holds the 18 public questions (two drawn pictures and four NASA photos, `fetch_public.sh`); the other 11 used crops of the source package's example figures, which are not redistributed. Scoring: `tools/score_arms.py`, `tools/wv_parse.py` (the source package's parser and counting metric), `tools/judge_r2.py`; the runtime suite: `tools/wv_suite.py`.
- Measured (the model card): Galaxy S26 GPU (the Android build of LiteRT-LM v0.18.1) a 512 × 512 picture and a question to first token 1.53 s, decode 12.97 tok/s, text prefill 1,024 / decode 256 at 1,125.93 / 13.87 tok/s; Mac M4 Max GPU (WebGPU) 0.30 s, 81.4 tok/s, text 3,675 / 83.0 tok/s; Mac CPU 4.85 s, 12.6 tok/s, text 506 / 14.9 tok/s.

## NVIDIA-Nemotron-3.5-Bolt-E4B — NVIDIA's post-trained Gemma 4 E4B as a text-only int4 file with a 2,048-token cache and prefill signatures of 128, 512 and 1,024 tokens: int4 block scales floored at 2^-14, GPU composites with the options that flag would add passed off, and a Snapdragon SM8850 NPU file (2026-10-26)

`nemotron_bolt_work/convert_bolt_e4b.sh` converts [nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-BF16](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-BF16) at commit `49843eed` — "a small language model (SLM) further post-trained from Gemma-4-E4B-it by NVIDIA", OpenMDW-1.1; the Gemma 4 E4B architecture: 42 decoder layers (35 sliding-window 512, 7 full-attention), hidden 2560, 8 query / 2 KV heads of 256 (512 on the full-attention layers), the last 18 layers reading the keys and values of earlier layers, per-layer embeddings, a 262,144-token vocabulary — into the int4 file of [litert-community/NVIDIA-Nemotron-3.5-Bolt-E4B](https://huggingface.co/litert-community/NVIDIA-Nemotron-3.5-Bolt-E4B), `NVIDIA-Nemotron-3.5-Bolt-E4B_int4.litertlm`. Text only: NVIDIA's card gives text as the model's input and output, and the file carries no vision or audio encoder.

```bash
PY=/path/to/python-with-litert-torch-0.10.0-at-0a81bf6f \
  bash nemotron_bolt_work/convert_bolt_e4b.sh out_bolt                    # fetch shim int4 sections
PY=... MODEL_DIR=/path/to/a/local/copy/of/49843eed bash nemotron_bolt_work/convert_bolt_e4b.sh out_bolt   # no download
```

`PY`: Python 3.12 with litert-torch 0.10.0 installed editable from a checkout at [`0a81bf6f`](https://github.com/google-ai-edge/litert-torch/commit/0a81bf6f3243a6f7c846d05d62afeff4008f2542), torch 2.12.0, transformers 5.19.0, ai-edge-quantizer-nightly 0.10.0.dev20261013, ai-edge-litert-nightly 2.4.0.dev20261011, litert-converter 0.5.0 and litert-lm-builder 0.18.0 (the environment of the run below, not this repository's uv.lock, which pins litert-torch 0.9.4). The `fetch` step downloads the 9 files the export reads with `dory_work/fetch_pinned.sh` and checks them against the Hub's record of the commit with `dory_work/verify_pinned.py`; with `MODEL_DIR` it links a local copy instead and checks it the same way. The export peaked at 73.6 GB of memory footprint and 87 GB resident on a 128 GB Mac: run one at a time.

| file | section | bytes | sha256 |
|---|---|---|---|
| `NVIDIA-Nemotron-3.5-Bolt-E4B_int4.litertlm` (4,404,768,064 B) | LlmMetadataProto | 20,387 | `c01dfd2d6048539553fce1a69b8cacde0a32a4887fc9e0c3c0431268b00707ef` |
|  | ExecutorMetadataProto | 3,238 | `f99346f85243cedb0ba32532ec01f6f448e1e92ae6deac7042d5d6a19f2e491e` |
|  | HF_Tokenizer_Zlib | 5,545,536 | `cc794c4d0b150ff5bd629d8c3e2cf32d57b110afce7024c5daad4fa4cd48ea9a` |
|  | TFLiteModel (tf_lite_prefill_decode) | 2,629,113,552 | `34b57dfa1069e04f36bebf4cc4d123f7965feeabb41e88a4001f9ec2fec86599` |
|  | TFLiteModel (tf_lite_embedder) | 348,137,568 | `aaef31b6e1b4017e2dc961b7eec71ac9c8de0a9ca0110d6192f007613e7b9d5c` |
|  | TFLiteModel (tf_lite_per_layer_embedder) | 1,421,880,640 | `431c49a9376384545bb934f685452cb79a647f5d955678ecb0a0821be9035d84` |

Run as published on 2026-10-26 with `MODEL_DIR` set (the 9 files checked against the Hub's record of the commit; the download path of the `fetch` step was not run that day), the script rebuilt the file with all six sections byte for byte: `fetch shim int4` in 245 s on a 128 GB Mac. The file sha256 differs from run to run (the packer writes a new uuid and creation timestamp into the header: 39 bytes, all before the end of the 816-byte header), so compare sections.

- **The 18 layers that share keys and values.** `model-frozen-shared-kv.safetensors` holds the key / value projections and key norms of layers 24 to 41 (54 tensors, the same bytes as in google/gemma-4-E4B-it); those layers read the keys and values of earlier layers, transformers drops the 54 tensors when it loads the model, and the graphs have no key or value projection after layer 23.
- **The template.** NVIDIA's `chat_template.jinja` ("Nemotron updated Gemma4 chat template") with its 27 `.get('…')` lookups written as `['…']`, which `export_hf` does itself; the runtime's renders equal transformers' render of NVIDIA's template byte for byte for one user message, a user / assistant / user conversation, a system and a user message given as text parts, and one user message with `enable_thinking` true and false. With a system message the runtime passes the content as text parts and the template puts a space after each part, so the prompt has one more space after the system text than transformers' render with string content. Thinking is off unless `enable_thinking` is true; on, the template adds `<|think|>` to the system turn and the model opens its thought (`<|channel>thought` … `<channel|>`) itself.
- **Tokenizer section = the repo's `tokenizer.json`**, byte for byte (32,169,626 bytes, sha256 `cc8d3a0c…bfe0f`).
- **A lower bound on the int4 block scales.** `recipe_int4_b32_emb4cw.json` is int4 blockwise-32 min-max on every FULLY_CONNECTED weight and int4 channelwise on the two embedding tables. Plain, it writes 13 block scales below 2^-14 (in 4 of the 344 int4 weights), which XNNPACK refuses; `export_clamped_hf.py` floors them before the weights are rounded (`CLAMP_TOTAL calls=344 tensors=4 scales=13`), and every scale of the result is normal.
- **Lightweight conversion.** `--experimental_lightweight_conversion=True` with `RESOURCE_CONSTANT_NUMEL_THRESHOLD=30000000` (bytes: float32 and int32 constants of 30 MB or more go to the converter as buffer references instead of being copied into its IR), to lower the export's memory peak; the default path was not run for this model.
- **GPU composites from the export flag.** At `0a81bf6f`, `--apply_gpu_composites=True` also turns on every GPU option that is not given (`core/exportable_module_config.py`): the q / k / v norm + rotary and the SDPA composites, fused gate / up and q / k / v projections, SwiGLU and short-conv composites, a boolean mask, and a dynamic GPU cache that rounds the cache length up to a prime (2,048 → 2,053). The file passes those False and keeps the rotary composite: its graphs have `odml.cache_update` (24 = the layers that keep a KV cache) and `odml.runtime_bmm` (84 in decode, 46 in each prefill signature) composites and no BATCH_MATMUL or DYNAMIC_UPDATE_SLICE. Without the composites (the same export without those flags, both measured with prefill 128 / 1,024 only) the Mac GPU decoded 67.78 instead of 88.56 tokens/s and the Galaxy S26 GPU 11.8 instead of 15.7; the CPU does not change.
- **Prefill signatures of 128, 512 and 1,024 tokens.** Without a 512 signature LiteRT-LM v0.18.0's GPU runs a 512-token prompt as one 1,024-token call with padding (prompts of 129 to 511 tokens as several 128-token calls: the 444- and 450-token prompts ran as four). With it, a 512-token prompt reached its first token in 0.865 s instead of 1.21 s on the Galaxy S26 GPU and in 0.214 s instead of 0.384 s on the Mac GPU, the 1,024-token benchmark stayed within 3 %, and the file grew by 9.8 MB, the S26's GPU memory by 0.23–0.26 GB and its engine start by 0.5 s. On the CPU the extra signature adds 0.10 GB on the S26 (with the XNNPACK weight cache) and 1.35 GB on the Mac without it.
- **No activation type declared.** On a GPU the runtime runs the file with fp16 activations. fp32 activations gave the same answers (the Mac's eight; one question on the S26) and decoded × 0.88 on the Mac GPU (77.6 against 88.3 tok/s, set on the engine) and × 0.78 on the Galaxy S26 GPU (`--force_f32=true`).
- Measured (prefill 1024 / decode 256 tokens, context 2048, litert-lm 0.18.0, the compile cache off): Mac M4 Max GPU (WebGPU) 2,667.3 / 89.16 tok/s, Mac CPU 193.2–193.3 / 23.88. GSM8K (questions 0–99, thinking off, greedy, at most 2,048 tokens of context) on the Mac GPU: 92, the BF16 model in transformers on the CPU 93. Phone rows: the model card.
- **A runtime defect on files with an embedder section.** From the second message of a conversation, LiteRT-LM passes the previous answer's stop token to the model with a zero embedding (`runtime/executor/llm_litert_compiled_model_executor.cc` L1326–1341, L575–586, L413–447 at v0.18.1); the NPU executor looks the embedding up first. One question per conversation avoids it.

**The NPU file** (`NVIDIA-Nemotron-3.5-Bolt-E4B_qualcomm_sm8850.litertlm`, 5,038,493,603 bytes, sha256 `86af693f6f2e59246f2a252008f3bbefc73ff10f75876b924238d4f7b0bac7a1`) is made in four steps that are not in the script yet (written from the record of 2026-10-26, not re-run for this entry):
1. litert-torch's NPU export from a checkout at [`0d3869c6`](https://github.com/google-ai-edge/litert-torch/commit/0d3869c6ff5d0eface9cf6a254f8875146c2c366) with one edit to `generative/export_hf/core/split_cache/attention.py`: in prefill (more than one query position) the mask is added to the logits by broadcasting over the 4 query heads of each KV head instead of being concatenated 4 times, which keeps the int16 mask the HTP's ADD reads at 557,056 bytes; decode keeps the concatenation: with the broadcast there, the compiler plugin stopped with SIGFPE in its "MHA optimization (fast vlm decode)" pass, whose source divides the query count it reads on axis 1 (1 in decode) by the 2 KV heads. Stage 1 = the float export with the split cache, prefill 128, cache 2048, text only, lightweight conversion, the embedding tables int4 channelwise.
2. Stage 2 = calibration on 13 prompts (the first 10 of the export's own sample prompts and 3 long multi-task prompts of 700 to 2,016 tokens, so the int16 rotary positions reach the end of the cache), rendered with the bundle's template, 32 decode steps each. Stage 3 = static quantization: int16 activations of the FULLY_CONNECTED layers, int4 channelwise weights for the MLP gate and up projections (84) and the final vocabulary projection (lm_head), int8 channelwise for the other 259 FULLY_CONNECTED weights (attention, the MLP down projection, the per-layer-embedding projections). With int4 also on the down and per-layer projections, GSM8K on the phone fell to 13 of 27 questions (the BF16 model 27); with every weight int8, decode exceeded the HTP's per-graph limit.
3. Ahead-of-time compilation for SM8850 with ai-edge-litert 2.2.0 and its Qualcomm SDK package (QAIRT 2.47.0.260601114230), the auxiliary graph (cache update, masks, rotary tables) left uncompiled: prefill_128 1,475 of 1,475 and decode 2,180 of 2,180 operators in one partition each, the decode graph estimated at 3,371,794,432 bytes (the limit 3,758,096,384), a 3,275,247,616-byte context binary.
4. Two prefill inputs that no operator reads (`kv_cache_k_23`, `kv_cache_v_23`, FLOAT32 where decode reads INT16) renamed to `dangling_k_23` / `dangling_v_23` (16 bytes; nothing else changes), so that the runtime creates layer 23's cache buffers from the decode signature, which `--use_hw_cache_update_for_npu=true` needs.
On a Galaxy S26 (SM8850) with a LiteRT-LM build from the main branch (commit `3dbb23e1`, in no release up to v0.18.1), the QAIRT 2.47 libraries and `--use_hw_cache_update_for_npu=true`, the file answered the 8-question check 8/8, the two prompts of 444 and 450 tokens that ask for a word stated once 2/2, and GSM8K questions 0–98 with 92 correct (the BF16 model 92 on the same 99).

## LFM3-1.7B — a short-convolution / sliding-window / global-attention hybrid as four text files: int4 on the MLP gate and up projections with int8 elsewhere, or int8 everywhere, each with a 2,048- or 4,096-token cache (2026-10-29)

`lfm3_work/convert_lfm3_17b.sh` converts [LiquidAI/LFM3-1.7B](https://huggingface.co/LiquidAI/LFM3-1.7B) at commit `40ebe0aa` — Liquid AI's text model for local agents, LFM Open License v1.0; 24 layers (13 gated short convolutions of width 7, 5 sliding-window attention layers with a 256-token window, 6 global attention layers with a rotary embedding on part of each head), a sigmoid output gate in every attention layer, hidden 2048, a 128,000-token vocabulary shared with the output — into the four files of [litert-community/LFM3-1.7B-LiteRT](https://huggingface.co/litert-community/LFM3-1.7B-LiteRT).

```bash
PY=/path/to/python-with-litert-torch-0.11.0-at-f61d0b79 bash lfm3_work/convert_lfm3_17b.sh out_lfm3                   # fetch template exporter float2048 float4096 pack sections
PY=... MODEL_DIR=/path/to/a/local/copy/of/40ebe0aa bash lfm3_work/convert_lfm3_17b.sh out_lfm3                     # no download
PY=... REF_DIR=/path/to/the/published/files bash lfm3_work/convert_lfm3_17b.sh out_lfm3 sections                    # compare section by section
PY=... bash lfm3_work/convert_lfm3_17b.sh out_lfm3 line1                                                           # the exporter against transformers
```

`PY`: Python 3.12 with litert-torch 0.11.0 installed editable from a checkout at [`f61d0b79`](https://github.com/google-ai-edge/litert-torch/commit/f61d0b792013300ae7a1ca215d8c2f9de5ef4adf), torch 2.14.0, transformers 5.20.2, ai-edge-quantizer-nightly 0.10.0.dev20261023, ai-edge-litert-nightly 2.4.0.dev20261022, litert-converter 0.6.0.dev20261022 and litert-lm-builder 0.18.1 (the environment of the run below, not this repository's uv.lock). The `fetch` step downloads the 11 files the export reads with `dory_work/fetch_pinned.sh` and checks them against the Hub's record of the commit with `dory_work/verify_pinned.py`; with `MODEL_DIR` it links a local copy instead and checks it the same way. A float export peaked at 23.8 GB resident on a 128 GB Mac.

| file | section | bytes | sha256 |
|---|---|---|---|
| `LFM3-1.7B_mixed_int4_ekv2048.litertlm` (1,441,620,928 B) | LlmMetadataProto | 12,043 | `f2e82ae94e2fa558bea85884aa9f634e4025f50eecc6441711eb1b5671e24654` |
|  | ExecutorMetadataProto | 2,299 | `75768530e137f9a53b2a11344ef5f6817f4f924414ac1c08ccc69d58bfd693b8` |
|  | HF_Tokenizer_Zlib | 2,574,276 | `13875e880ea3499b9c8a5dd541714f6310a33c60e7c7ef053a0be40d47d9f46c` |
|  | TFLiteModel (tf_lite_prefill_decode) | 1,438,983,104 | `7f4974b5a4a7bcc257ef281183206c17664e9801144d8ead3bda4724116d703b` |
| `LFM3-1.7B_mixed_int4_ekv4096.litertlm` (1,441,629,120 B) | LlmMetadataProto | 12,043 | `fb069a863efbb93c8b2564d132e529638835a1f226a0d9af72de2309148135c9` |
|  | ExecutorMetadataProto | 2,299 | `d1fe2ff6c7767cfa416228c594cd6c32b4474374b15354c6374d89da558923b0` |
|  | HF_Tokenizer_Zlib | 2,574,276 | `13875e880ea3499b9c8a5dd541714f6310a33c60e7c7ef053a0be40d47d9f46c` |
|  | TFLiteModel (tf_lite_prefill_decode) | 1,438,991,296 | `f099321d063094cbde2df65a927942002082065204cda9434316a0905f835b55` |
| `LFM3-1.7B_int8_ekv2048.litertlm` (1,801,549,264 B) | LlmMetadataProto | 12,043 | `f2e82ae94e2fa558bea85884aa9f634e4025f50eecc6441711eb1b5671e24654` |
|  | ExecutorMetadataProto | 2,299 | `75768530e137f9a53b2a11344ef5f6817f4f924414ac1c08ccc69d58bfd693b8` |
|  | HF_Tokenizer_Zlib | 2,574,276 | `13875e880ea3499b9c8a5dd541714f6310a33c60e7c7ef053a0be40d47d9f46c` |
|  | TFLiteModel (tf_lite_prefill_decode) | 1,798,911,440 | `a9d3edf561b8861766c106b40790d1821f5646488e652c600d02443f641f9467` |
| `LFM3-1.7B_int8_ekv4096.litertlm` (1,801,557,456 B) | LlmMetadataProto | 12,043 | `fb069a863efbb93c8b2564d132e529638835a1f226a0d9af72de2309148135c9` |
|  | ExecutorMetadataProto | 2,299 | `d1fe2ff6c7767cfa416228c594cd6c32b4474374b15354c6374d89da558923b0` |
|  | HF_Tokenizer_Zlib | 2,574,276 | `13875e880ea3499b9c8a5dd541714f6310a33c60e7c7ef053a0be40d47d9f46c` |
|  | TFLiteModel (tf_lite_prefill_decode) | 1,798,919,632 | `135fed584f05f550f93cea4915ad1607dfc965f017ca60a5535ae780600acec7` |

Run as published on 2026-10-29 with `MODEL_DIR` set (the 11 files checked against the Hub's record of the commit; the download path of the `fetch` step was not run that day), the script rebuilt all four files with every section byte for byte: `fetch template exporter float2048 float4096 pack sections` in 11 min 2 s on a 128 GB Mac. The file sha256 differs from run to run (the packer writes a new uuid and creation timestamp into the 592-byte header: 39 to 44 bytes), so compare sections.

- **The exporter (`lfm3_ext/`).** litert-torch has no LFM3 module; `lfm3_ext.export_lfm3` builds `Lfm3ForCausalLM` from the source repository's own remote code, loads the weights, and hands the model to litert-torch's `export_hf` in place of its checkpoint loader, with an extension (`lfm3_ext.extension`) for the rest: the short convolutions run as litert-torch's LFM2 short convolution (`model_ext/lfm2/short_conv.py`, width 7), the decoder layers take litert-torch's attention interface, and a padded prefill slot (position 0 after a real one) does not enter the convolution state (the guard reads positions, not token ids, so it also holds for embeddings input). The attention with its output gate, its per-head q / k norms and its rotary embedding is the source code's, unchanged. The folder name `LFM3-1.7B` and the package name `lfm3_ext` appear in the graph's tensor names (transformers names the remote-code module after the folder and a hash of the code files), so run the script as is: the `fetch` step puts the source in a folder of that name.
- **The exporter check.** `exporter` runs `lfm3_ext.tiny_parity_test`: a random 6-layer LFM3 against transformers on every path LiteRT-LM drives. `line1` runs the module on the real weights against transformers in float32 (CPU): 8 questions with thinking off and on and a 1,584-token passage of the license with a question, each extended by the bfloat16 model's greedy continuation (2,683 positions); one prefill call, token-by-token decode, 128-token prefill calls with the last one padded, and that prefill followed by decode. Run on 2026-10-29 from this repository's files: the reference was the same as on the conversion day (the same 17 token sequences, the same float32 logits), and on all four paths the module matched it within 1.91e-3 in every logit with the same top token at all 2,682 positions that are not near ties (1 of 2,683 left out); with embeddings input on the long text, 8.12e-4 and 1,647 of 1,647.
- **The chat template** (`make_template.py`): the source template with three edits. Earlier assistant turns render without their thought wherever they sit (`keep_thinking = preserve_thinking`): LiteRT-LM keeps a conversation's earlier prompt as a prefix of the next one, so a turn must render the same after a later user message. The two `{%- generation -%}` tags are removed and the 26 `.get("…")` lookups are written as `["…"]`, which keeps the rendered text and lets LiteRT-LM's template engine run the file. LiteRT-LM 0.18.1's own rendering equals transformers' rendering of the source template for one turn, three turns and a tool round trip, thinking off and on, except for the start token (the runtime adds it once from the file) and the thought of the assistant turn after the last user message in the tool round trip; the source template as it is does not render on LiteRT-LM 0.18.1.
- **Tokenizer section = the repository's `tokenizer.json`**, byte for byte (17,908,635 bytes, sha256 `7e226e34bface23ccf6e495d157bf77186c1c30a7659880e1a8994db528c3701`).
- **Quantization** (`quant_pack.py`, from the float32 graph the export keeps). `recipe_int8.json` = every weight int8 per channel, the same bytes as the quantizer's `dynamic_wi8_afp32` recipe. `recipe_q1b.json` = int4 blockwise-32 on the MLP gate and up projections (`w1` / `w3`), int8 per channel on everything else, the tied embedding / output table included; block scales are floored at 2^-14 before the weights are rounded (XNNPACK refuses smaller ones), and no scale of this model reaches the floor. More int4 than that cost accuracy: int4 on every layer projection (blocks of 16, 32 or 64) or on all but the attention or the short convolutions agreed with the float32 model's top token at 67.11–74.83 % of the test positions on a Mac GPU (this file 79.9 %), int4 also on the MLP down projection missed one of 8 check questions, and blocks of 16 on the MLP projections decoded no faster on a Galaxy S26 GPU while XNNPACK takes no int4 block under 32.
- **The sampler.** The packer writes `TOP_K` (k 50, temperature 0.1) from the source's `generation_config.json`, which LiteRT-LM 0.18.1's CPU sampler refuses (`UNIMPLEMENTED: Sampler type: 1`) and its GPU samplers read with `p` unset, that is greedily. `--sampler-top-p 1.0` makes the file declare `TOP_P` with k 50, p 1, temperature 0.1: top-k 50 at temperature 0.1 on both samplers, the source card's General setting without its repetition penalty.
- **No GPU composites.** The graph keeps attention and the caches as plain operators. LiteRT-LM 0.18.1's GPU executor passes the state between calls only for a graph without a state-parameter input (ping-pong), which keeps the short-convolution state; every signature runs whole on the Galaxy S26 and Pixel 8a GPUs.
- Measured on the phones (prefill 1,024 / decode 256 tokens, 2,048-token context, LiteRT-LM v0.18.1 `litert_lm_advanced_main`): the model card.

## MiniCPM-OCR-1.0 — an OCR VLM as a fast_vlm bundle: the 4× ViT re-authored in plain torch, the Qwen3.5-text hybrid decoder on the Qwen3.5 rail, and page tiles through one extra-context variable (2026-10-29)

One command rebuilds both files of [litert-community/MiniCPM-OCR-1.0](https://huggingface.co/litert-community/MiniCPM-OCR-1.0) from [openbmb/MiniCPM-OCR-1.0](https://huggingface.co/openbmb/MiniCPM-OCR-1.0) at revision `165531f9`: `scripts/ship_minicpm_ocr_1_0.sh` (key `minicpm-ocr-1.0` of `scripts/reproduce_vlm.sh`). The files are `MiniCPM-OCR-1.0-VL_int8.litertlm` (1,997,636,640 B: int8 decoder and embedder, fp16 vision encoder and adapter) and `MiniCPM-OCR-1.0-VL_int8vis.litertlm` (1,589,150,752 B: the vision encoder int8). The model (apache-2.0, 1,300,434,160 parameters) is MiniCPM-V 4.6's architecture in its 4× mode: a 27-layer ViT (hidden 1152, patch 14) that attends over all 1,024 patches of a 448×448 picture, a 2×2 merger to 256 tokens, and a 24-layer Qwen3.5-text hybrid decoder (18 GatedDeltaNet + 6 gated attention layers, hidden 1024, 248,100-token vocabulary, tied head). Scripts: [`minicpmocr_work/`](minicpmocr_work/), plus the Qwen3.5 rail in `qwen35_work/` and `qwen35vl_work/` and the derived text checkpoint of `tools/vlm_text_ckpt/`.

```bash
PY_EXPORT=/path/to/py312-litert-torch-0.9.2/bin/python \
PY_VISION=/path/to/py314-litert-torch-0.9.3/bin/python \
PY_HF=/path/to/py312-transformers-2df500f3/bin/python \
LITERT_LM=/path/to/litert-lm-0.18.1/bin/litert-lm \
FIXTURE_SRC=/path/to/448x448/text/pictures \
  bash scripts/reproduce_vlm.sh minicpm-ocr-1.0        # -> out/minicpmocr/*.litertlm
```

Run as written on 2026-10-29 from a clone of this repository with these files added (`HF_HUB_CACHE` pointing at a verified copy of the source revision, so no download; `PY_EXPORT` litert-torch 0.9.2 / converter 0.3.0 / quantizer 0.8.0 / ai-edge-litert 2.1.6 / litert-lm-builder 0.15.0 / transformers 5.14.1 / torch 2.12.1 on Python 3.12.13; `PY_VISION` litert-torch 0.9.3 / converter 0.4.0 / quantizer 0.9.0 / ai-edge-litert 2.2.0 / torch 2.13.0 on Python 3.14.6; `PY_HF` transformers 2df500f3 / torch 2.14.1 on Python 3.12.11; `LITERT_LM` 0.18.1): every section of both files came out as the published bytes; only the uuid and creation time the builder writes into the header differ.

| file | section | bytes | sha256 |
|---|---|---|---|
| `MiniCPM-OCR-1.0-VL_int8.litertlm` | LlmMetadataProto (template `chat_template_ocr.jinja`, 1,695 B, `aa44b450…`) | — | `2c7eb87e193a2c33a139726e7ffde86cd77d345ded9c5c51eeb4f5560abdc445` |
|  | ExecutorMetadata (48 state buffers: 36 linear-attention, 12 KV) | — | `269f02d161c3543a768df9061732cbd0942df9aa827cfa6df5ba70a8e7597219` |
|  | HF tokenizer (zlib; unpacks to the source `tokenizer.json`, `6269eb38…`) | 3,913,484 | `5ce8f4241e66c467202a33556cee6c821ca9f262d6c28c5a2ba6d8428f512c3c` |
|  | TFLiteModel embedder | 257,033,808 | `2f5839b220d6d74ce6efb12c5dd46a8c932925f7e2ed1892023ee789ec086f68` |
|  | TFLiteModel prefill_decode (`prefer_activation_type fp32`) | 854,459,584 | `d46e3cd4c75a50b3356ba2fabafa34fdb16d0ed963c3dd91eba2a484db5fd1bc` |
|  | TFLiteModel vision_encoder (fp16) | 830,174,160 | `3811402808e2a0d1f485042e80d808fc8ed6de47696b2adc956b2d6f438229ec` |
|  | TFLiteModel vision_adapter (fp16) | 51,971,104 | `218790811841d92261ca9dd6f4bcbcbfac0a3618a5ac560eb1b7c21eb20a3d34` |
| `MiniCPM-OCR-1.0-VL_int8vis.litertlm` | the same sections, with TFLiteModel vision_encoder (dynamic int8) | 421,675,088 | `e244bd9cfec4a60f00fd5b9106cdd215f4007cdb36f0aa90e9fde5140660bb49` |
(The LlmMetadata and ExecutorMetadata rows are the sha256 of `litert-lm unpack`'s text form.)

- **The model code is on transformers main only.** `minicpm_ocr` arrived on 2026-10-26 and is read here at `2df500f3` (2026-10-27); v5.21.0 does not have it, so `PY_HF` installs transformers at that commit. The upstream reference runs patch the GatedDeltaNet depthwise conv1d (`fast_dwconv.py`): torch 2.14.1's macOS CPU kernel takes about 10 s per decode step there; the patched step gives logits within 6.3e-5 of torch's and the same argmax.
- **The text decoder is its own model inside the VLM checkpoint.** `make_text_ckpt.py` keeps the 320 `model.language_model.*` tensors as `model.*` (no `lm_head` tensor: the head is tied), drops the vision tower and the merger (459 tensors; the 16 `vit_merger` tensors belong to the 16× mode and are not used in the 4× mode), and writes a `qwen3_5_text` config; on 20 text tokens the derived model and the full `MiniCPMOcrForConditionalGeneration` give identical logits (eager fp32, max |Δ| 0.0, argmax 20/20). The decoder goes through the 2026-10-13 MiniCPM-V-4.7-1B command unchanged: EMBEDDER + PREFILL_DECODE with prefill 1024/256/64/16/4/1 over a 4096-token KV, dynamic int8 on 151 fully connected layers, the 36 GatedDeltaNet `in_proj_a` / `in_proj_b` float.
- **The vision tower of the 4× mode is simpler than 4.7's.** In the 4× mode upstream's `get_image_features` skips the in-encoder window merger, so `ocr_vision.py` is 27 plain layers over the 1,024 patches (the position table gathered into a constant, LayerNorm inputs divided by the published powers of two in `ln_scales.json`: 256 for 34 LayerNorms and 512 for one, where the residual stream reaches about 2,100) and the 2×2 merger. Against upstream on 17 pictures: the fp32 TFLite pair at token cosine ≥ 0.99999989, fp16 weights ≥ 0.9998; the dynamic-int8 encoder of the `_int8vis` file falls to 0.5917 (correlation 0.9837).
- **The contract was measured on upstream fp32 first** (10 synthetic regions written for this conversion, an A4 page, two photos from the transformers model doc): a region padded to a white 448×448 square reads exactly as upstream reads the original picture (10/10; stretching a 16:1 line misread a number), a whole page needs the processor's tiling (one 448 picture: CER 0.136; one 672 or 896 picture: 0.116 / 0.123; an overview + 8 slices of 448: 0.000), and a 463×463 photo reads closer to the native run as one padded picture than as two slices.
- **Page tiles through one extra-context variable.** The Kotlin API cannot add keys to an image item, but `Conversation.sendMessage(…, extraContext)`, the Python `create_conversation(extra_context=…)` and the C API all hand a JSON object to the template, whose keys LiteRT-LM's minijinja renderer puts at the top level (`runtime/components/rust/minijinja_template.rs` at v0.18.1). `chat_template_ocr.jinja` reads `slice_cols`: without it the template renders the 4.7 dual template's characters (9 of 9 template cases equal the transformers processor's ids, in jinja2 and in minijinja 2.14.0); with it, the 2nd and later images of a user message become `<slice>…</slice>`, `slice_cols` per row, and the six tiled test pictures give the ids of upstream's tiled prompt (`check_template_v2.py`). Through the runtime (litert-lm 0.18.1 Python API) the tiled runs equal upstream's tiled runs on CPU 4/4 and GPU 2/2, and `ocr_page.py`, run from a fresh `pip install litert-lm pillow` (0.19.0), reads an A4 page, a form photo, a receipt photo, a text line and a receipt's three fields character for character like upstream.
- Gates (one process per picture, greedy): the 13 padded pictures on the Mac CPU equal upstream fp32 in 11 of 13 (the A4 page and the form photo, unreadable as one 448 picture, are where upstream itself guesses); on the phones (LiteRT-LM v0.18.1's `litert_lm_advanced_main`, Android tag build) `VL_int8` equals upstream on all 11 readable pictures on the Galaxy S26 CPU (15 runs) and the Pixel 8a CPU (14 runs), on 7 of them on the Galaxy S26 GPU (9 runs), and with only the vision encoder on the GPU on all 11 on the Galaxy S26 (15 runs) and on 10 of them on the Pixel 8a, whose Mali GPU writes the second formula with spaces (13 of 14 runs; the same with fp32 activations); `VL_int8vis` on the phones' CPUs writes the second formula as `\frac{n (n + 1) (2 n + 1)}{6}` where upstream writes `\frac{n(n+1)(2n+1)}{6}` (the same text on both phones, 3 of 3 processes on the Pixel 8a) while the Mac CPU's run of the same file writes upstream's text, and with its vision encoder on the GPU (Mac WebGPU) it answers `[Non-Text]` whenever the runtime has a cache directory and reads as upstream with the cache off (`:nocache`, `--cache no`); the fp16 encoder reads correctly with a cache directory. The A4 page as 9 unmarked pictures (the phone CLI has no extra-context flag) reads as upstream's tiled run on every form that ran it.
- Measured: one request in a fresh process (engine creation, encode, prefill, decode): a 448×448 region takes 12.9 s on the Mac M4 Max CPU with `VL_int8vis` (14.7 s with `VL_int8`), 11 s on the Galaxy S26 CPU and 32–35 s on the Pixel 8a CPU with `VL_int8`; an A4 page as 9 pictures takes 20.0 s on the Mac and 34 s on the Galaxy S26 with `VL_int8`'s vision encoder on the GPU, and 177 s on the Pixel 8a CPU (88 s with the vision encoder on its Mali GPU, which is not exact). A GPU decoder compiles its five large prefill programs again at every engine creation (their program-cache read fails with DATA_LOSS: 26–28 s on the Mac, 44–65 s on the Galaxy S26); on the Pixel 8a the GPU decoder's engine crossed our 400,000 kB free-memory line while building; on the Galaxy S26 NPU (QAIRT 2.47, JIT) the compiler plugin rejects the 60 DynamicUpdateSlice ops of the KV-cache update.

## LFM3-Audio-2B — speech and text in, text out through one `generic_model` `.litertlm` file (int4 decoder, audio encoder with fp16 weights), and speech out as LiteRT graphs for a host loop, with a Galaxy S26 NPU file for the depth decoder (2026-10-29)

`lfm3audio_work/convert_lfm3_audio.sh` converts [LiquidAI/LFM3-Audio-2B](https://huggingface.co/LiquidAI/LFM3-Audio-2B) at commit `9d49701c` — Liquid AI's speech and text model, LFM Open License v1.0: an LFM3 decoder (24 layers: 13 gated short convolutions, 5 sliding-window attention layers with a 256-token window, 6 global attention layers; hidden 2048, a 128,000-token vocabulary shared with the output), an audio encoder (16 kHz, 128 log-mel bins, an 8× downsampler, a 17-layer LFM2 encoder with hidden 768, a projector to 2048), a 3-layer depth decoder (hidden 1024) that writes 12 codebooks of 2,017 codes per 80 ms speech frame, and a 24 kHz codec — into the six files of [litert-community/LFM3-Audio-2B-LiteRT](https://huggingface.co/litert-community/LFM3-Audio-2B-LiteRT).

```bash
PY=/path/to/python-with-litert-torch-0.11.0-at-f61d0b79 PY_PARTS=/path/to/python-with-litert-torch-0.10.0 \
  bash lfm3audio_work/convert_lfm3_audio.sh out_lfm3a        # fetch template lm quant audio bundle parts npu sections
PY=... PY_PARTS=... MODEL_DIR=/path/to/a/local/copy/of/9d49701c bash lfm3audio_work/convert_lfm3_audio.sh out_lfm3a   # no download
PY=... REF_DIR=/path/to/the/published/files bash lfm3audio_work/convert_lfm3_audio.sh out_lfm3a sections              # compare
```

`PY`: Python 3.12 with litert-torch 0.11.0 installed editable from a checkout at [`f61d0b79`](https://github.com/google-ai-edge/litert-torch/commit/f61d0b792013300ae7a1ca215d8c2f9de5ef4adf), torch 2.14.0, transformers 5.20.2, ai-edge-quantizer-nightly 0.10.0.dev20261023, ai-edge-litert-nightly 2.4.0.dev20261022, litert-converter 0.6.0.dev20261022, litert-lm-builder 0.18.1, numpy 2.5.4, safetensors 0.8.0, tokenizers 0.23.3 (the decoder, its quantization and the `.litertlm` file). `PY_PARTS`: Python 3.12 with litert-torch 0.10.0, litert-converter 0.5.0, ai-edge-litert 2.3.0, ai-edge-quantizer 0.10.0, torch 2.14.0, transformers 5.20.2, numpy 2.5.4, safetensors 0.8.0 (the audio encoder and the speech-out graphs). The NPU step runs in a linux/amd64 image with ai-edge-litert 2.2.0, its Qualcomm compiler plugin and Qualcomm's QAIRT 2.47.0 SDK (`npu/Dockerfile`; QAIRT is not in this repository: it comes from Qualcomm under Qualcomm's license). The `fetch` step downloads 21 files of the source repository (the weights, the codec weights, the configs, the tokenizer, the chat template, the code files and the license) with `dory_work/fetch_pinned.sh` and checks them against the Hub's record of the commit with `dory_work/verify_pinned.py`; with `MODEL_DIR` it links a local copy instead and checks it the same way.

| file | section | bytes | sha256 |
|---|---|---|---|
| `lfm3-audio-2b_int4_b128_c2048_h30.litertlm` (1,531,232,624 B) | LlmMetadataProto | 9,331 | `ac87a26c32e74c32ebcd97bff592c31e3267917d3c3d67c522d5958b7d26e547` |
|  | ExecutorMetadataProto | 2,299 | `75768530e137f9a53b2a11344ef5f6817f4f924414ac1c08ccc69d58bfd693b8` |
|  | HF_Tokenizer_Zlib | 2,574,275 | `00b70753b3d704e0b4a50ad259049b5c3a21764bd0b956030b594640591c78df` |
|  | TFLiteModel (tf_lite_prefill_decode) | 1,031,710,368 | `86120442f62def8bb160f30281589e0b3d0a00fc2a63eeee04e2811c14a1b376` |
|  | TFLiteModel (tf_lite_embedder) | 263,682,208 | `8210f7d8ba7e4ac7fe637e6bc67045d9698b41686b1bafbe5d9587a332a10f56` |
|  | TFLiteModel (tf_lite_audio_encoder_hw) | 233,193,840 | `74d98e70d3467c22475a0319704e9ea0438cfc599a1f3035739cb2b3bccf1771` |
| `lfm3_audio_depth_proj_r4b.tflite` | | 100,713,664 | `66789aa754112d69fcc7bb857d317466af52af8b1686c1b327fa9fa11217476d` |
| `lfm3_audio_depth_pass_r4b.tflite` | | 334,104,044 | `5f684777daf8c73dc4a95b314fe559cb329f8a7976791ed7ecfdfa78ae93dfe7` |
| `lfm3_audio_frame_embed_f32.tflite` | | 198,281,752 | `d042be62b439630e55feed86bc9ae630db70674ba924d4752c4d5dfc3a44472d` |
| `lfm3_audio_codec_decoder_f12_r4b2.tflite` | | 152,498,860 | `9dbbbcf870c1be4e166b77c1948ca4e34e8e069780236c15066f1d3c8ed52cbd` |
| `lfm3_audio_depth_pass_r5b_aot.tflite` (SM8850) | | 169,584,336 | `e3bdb0ed185190ba7b0881af250e585385e04360c6a614e712801287a8f1bfd4` |

Run as published on 2026-10-29 with `MODEL_DIR` set (the 21 files checked against the Hub's record of the commit; the download path of the `fetch` step was not run that day), the script rebuilt the `.litertlm` file with every section byte for byte and the four CPU and GPU speech graphs byte for byte: `fetch template lm quant audio bundle parts npu sections` in 7 min 11 s on a 128 GB Mac (the decoder export peaked at 23.6 GB resident). The `.litertlm` file's sha256 differs from run to run (the packer writes a new id and creation time into the file header), so compare sections. The NPU file's bytes change with every compile: four compiles of the same float graph (sha256 `6263cb66…`) in the same image gave four different files. The published NPU file is therefore the one the Galaxy S26 measurements ran; the rebuilt NPU files were not compared with it on a phone.

- **One `.litertlm` file for text and audio input.** In LiteRT-LM 0.18.1 only `generic_model` carries audio input, so the file declares that type with audio on: the runtime cuts 16 kHz mono PCM into frames of 160 samples and hands up to 3,000 of them (30 s) per window to the audio section as they are (no mel spectrogram in the runtime); a longer recording is cut into 30 s windows with no overlap. The audio section computes the log-mel spectrogram in the graph with the source feature extractor's settings and returns up to 375 rows (one per 80 ms) plus a uint8 mask that tells the runtime how many rows are audio. The runtime passes no clip length, so the graph takes the last non-zero sample as the end of the clip: exact digital silence at the end of a recording counts as padding. `<|audio|>` in the chat template marks where the rows go; the source model puts no other token around them. Tool calls come back as text with this type.
- **The decoder** (`lfm3_work`'s `lfm3_ext.export_lfm3`, the exporter of the LFM3-1.7B entry above): the decoder weights under `lfm.model.`, the `lfm` block of `config.json`, embeddings in (the embedder is a section of its own, one token per call), the logits and the final-norm `hidden` out, prefill signatures 1024 / 256 / 64 / 16 / 4 / 1 and a 2,048-token KV cache. The `hidden` output is what the speech loop reads. The folder name `hf` and the package name `lfm3_ext` appear in the graph's tensor names (transformers names the remote-code module after its folder and a hash of the code files), so run the script as is: the `fetch` step puts the source in a folder of that name.
- **Quantization** (`lm_quant.py`, `recipe_int4_b128.json`): every fully connected weight of the decoder int4 blockwise-128, except the tied output head, which stays int8 per channel; the embedder's table int8 per channel. Block scales are floored at 2^-14 before the weights are rounded (XNNPACK refuses smaller ones); in this model the floor raises 720 scales in 3 tensors.
- **The audio encoder section** (`scripts/export_parts.py --part audio`, then `scripts/quant_fp16w.py`): `r4b_parts.AudioSectionR4b`, a re-authoring of the source encoder that the LiteRT GPU delegate runs whole (the query heads that share a key/value head folded into the matmul rows, the clip length found without index arithmetic, rank-4 masks), with its 105 fully connected weights cast to fp16 (the convolutions stay fp32).
- **Chat template** (`make_template.py`): the source template with three edits. Earlier assistant turns render without their thought wherever they sit, because LiteRT-LM keeps a conversation's earlier prompt as a prefix of the next one. The `{%- generation -%}` tags are removed and the 13 `.get("…")` lookups are written as `["…"]`, which keeps the rendered text and lets LiteRT-LM's template engine run the file. The file declares a thought channel, `<think>` … `</think>`.
- **Tokenizer section = the repository's `tokenizer.json`**, byte for byte (17,908,634 bytes, sha256 `41927bd314d17d7d7fadb44e10cd0919f7b71528d7741596d8cbd15c6945ec44`).
- **Speech out** is not in the `.litertlm` file: in speech mode each 80 ms frame is one decoder step whose input is the sum of the previous frame's 12 codebook embeddings (`frame_embed`, not a token id), and the step's `hidden` feeds the depth decoder: `depth_proj` once and `depth_pass` three times, each pass revealing 4 of the 12 codebooks (MaskGIT order). The codec decoder (`codec_decoder_f12`, with its streaming state as inputs and outputs) turns 12 frames (0.96 s) per call into 72 windowed frames of 1,280 samples, which the host overlap-adds (hop 320) into 24 kHz audio. The sampling, the text / speech mode switching and the overlap-add run on the host. The module names `r1b_graphs`, `r4b_parts` and `r1a_audio_section` appear in these graphs' tensor names, so the scripts keep those names.
- **The NPU file** (`scripts/export_parts.py --part depth_npu`, then `npu/aot_compile_sm8850.py` in the image of `npu/Dockerfile`): the depth decoder's pass compiled ahead of time for the Hexagon NPU of the Qualcomm SM8850 (Galaxy S26). The NPU runs float graphs in fp16; for it, the graph is traced with SiLU written as x · s(x) and the sigmoid as s(x), s(x) = exp(min(x, 0)) / (1 + exp(−|x|)), and with exp / cos / sin / log10 / GELU reading their input through a split and a concatenation, so that no such function reads a fully connected layer's output directly (the same values in float32). In the depth pass this rewrites 3 sigmoids and 3 SiLUs.
- Measured on the phones and a Mac: the model card.

## LFM3-VL-2B — a hybrid VLM as a `fast_vlm` file: the vision tower re-written for one 512 × 512 picture with the runtime's u8 input taken back exactly, and the text decoder with int4 or fp16 weights (2026-10-29)

`lfm3vl_work/convert_lfm3_vl.sh` converts [LiquidAI/LFM3-VL-2B](https://huggingface.co/LiquidAI/LFM3-VL-2B) at revision `f83d2b0a` — Liquid AI's vision-language model, LFM Open License v1.0; 1,918,181,888 BF16 parameters: a 24-layer LFM3 language model (13 gated short convolutions, 5 sliding-window attention layers, 6 global attention layers; hidden 2048, a 128,000-token vocabulary), a 14-layer vision encoder (8 gated depthwise 2D convolutions, 6 bidirectional attention layers, 16 × 16 patches, 2D rotary positions) and a projector that merges each 2 × 2 block of patches into one token — into the two files of [litert-community/LFM3-VL-2B-LiteRT](https://huggingface.co/litert-community/LFM3-VL-2B-LiteRT): `LFM3-VL-2B_int4.litertlm` (2,071,595,259 B, sha256 `8c208746b7758996c41e27eec8507b3c875a9e3bbde4b6da0e4c145d3e450c41`) and `LFM3-VL-2B_f16.litertlm` (4,439,967,995 B, sha256 `e9e15f4da346a8eb7fc5d2dc276dde9a9cecf1424d50193deeb2d098bf664877`). Both take text and one picture at a fixed 512 × 512 (256 image tokens) and think in a `<think>` block unless `enable_thinking` is false; they differ only in the text decoder's weights.

```bash
PY=/path/to/python-with-litert-torch-0.11.0-at-f61d0b79 PY_EXPORT=/path/to/python-with-litert-torch-0.10.0 \
  bash lfm3vl_work/convert_lfm3_vl.sh out_lfm3vl        # fetch template vision float quant pack sections
PY=... PY_EXPORT=... MODEL_DIR=/path/to/a/local/copy/of/f83d2b0a bash lfm3vl_work/convert_lfm3_vl.sh out_lfm3vl   # no download
PY=... PY_EXPORT=... REF_DIR=/path/to/the/published/files bash lfm3vl_work/convert_lfm3_vl.sh out_lfm3vl sections  # compare
```

`PY`: Python 3.12 with litert-torch 0.11.0 installed editable from a checkout at [`f61d0b79`](https://github.com/google-ai-edge/litert-torch/commit/f61d0b792013300ae7a1ca215d8c2f9de5ef4adf), torch 2.14.0, transformers 5.20.2, ai-edge-quantizer-nightly 0.10.0.dev20261023, ai-edge-litert-nightly 2.4.0.dev20261022, litert-converter 0.6.0.dev20261022, litert-lm-builder 0.18.1, numpy 2.5.4, safetensors 0.8.0, tokenizers 0.23.3 (the text decoder and its quantization). `PY_EXPORT`: Python 3.12 with litert-torch 0.10.0, litert-converter 0.5.0, ai-edge-litert 2.3.0, ai-edge-quantizer 0.10.0, torch 2.14.1, transformers 5.14.1, litert-lm-builder 0.18.1, numpy 2.5.4, safetensors 0.8.0, tokenizers 0.22.2 (the vision graphs, the `.litertlm` files and the section check).

| file | section | bytes | sha256 |
|---|---|---|---|
| `LFM3-VL-2B_int4.litertlm` (2,071,595,259 B) | LlmMetadataProto | 12,038 | `43b00bbd08573758762adf25be8ec4c61f4e5270e35df3f98127ab26c60fb458` |
|  | HF_Tokenizer_Zlib | 2,574,262 | `0f0890533d9a6cbc7631a5601f470e5ece22f70ce3affd5472a62e9cea881980` |
|  | TFLiteModel (tf_lite_embedder) | 263,682,208 | `4ab19094c7b57e548902f23777848d236ac4c65dd8863912e5c27d13cad73f12` |
|  | TFLiteModel (tf_lite_prefill_decode) | 1,100,474,784 | `1ba36fac842ebf73dd5db9666101b40267b5af2c7d106ab312e130d6b481eef0` |
|  | TFLiteModel (tf_lite_vision_encoder) | 654,412,748 | `39c5d96345eed03245a9436132b962c4d6dbdbaafd0a5c5e1262cc625d20e908` |
|  | TFLiteModel (tf_lite_vision_adapter) | 50,371,856 | `1be6993680953c3535a05f2c416b29de491469ef6d58e52dafb07603f64047f1` |
|  | ExecutorMetadataProto | 2,299 | `75768530e137f9a53b2a11344ef5f6817f4f924414ac1c08ccc69d58bfd693b8` |
| `LFM3-VL-2B_f16.litertlm` (4,439,967,995 B) | TFLiteModel (tf_lite_prefill_decode) | 3,468,846,624 | `793df314289523a8fa41cab708a11ec39549dc949bdbc0a73fed68eb620de798` |
|  | the other six sections | | the int4 file's, byte for byte |

Run as published on 2026-10-31 (the 17 source files fetched by the `fetch` step and checked against the Hub's record of the commit), the script rebuilt both files with every section byte for byte: `fetch template vision float quant pack sections` in about 14 minutes of running time on a 128 GB Mac (the float export peaked at 23.8 GB resident, the quantization at 34.1 GB). The files' sha256 differs from run to run (the packer writes a new id and creation time into the file header), so compare sections.

- **The vision tower at a fixed grid** (`tools/lfm3_vision.py`, converted by `tools/export_vision.py`). LiteRT-LM 0.18.1's `fast_vlm` path stretches every picture to the file's size and hands the encoder float(u8) / 255. The source processor keeps the aspect ratio (smart_resize, 256–800 image tokens on the test pictures); at a fixed 512 × 512 the source model in float32 keeps all 18 right answers of the test set and the first 32 answer tokens on 16 of 23 questions. The tower is re-written in plain PyTorch for the 512 × 512 grid: the 2D rotary tables and positions are constants, the whole tower runs in raster order and the projector's 2 × 2 merge returns the source's token order, with no GATHER, FLEX or custom operators. The graph's first step takes the u8 value back exactly (floor(x · 255 + 0.5)) and then normalises as the source processor does ((u − 127.5) / 127.5): a first form, (x − 0.5) · 2, lost precision at u8 = 128 and on pictures with flat grey regions landed 13–18 times farther from a float64 reference than the source model's own float32. Against the source model's image features, the fp32 encoder with the fp16-weight projector gives a worst per-token cosine of 0.9999978 over 12 pictures; fp16 or int8 encoder weights, or the GPU's fp16 precision, gave 0.20–0.70, so the encoder keeps fp32 weights and both vision sections declare fp32 activations. The module name `lfm3_vision` appears in these graphs' tensor names, so the script keeps it.
- **The text decoder** (`tools/vl_decoder.py float`, the exporter of the LFM3-1.7B entry above, `lfm3_work/lfm3_ext/`): the weights under `model.language_model.`, the `text_config` block of `config.json`, embeddings in (the token embedder is a section of its own, one token per call), prefill signatures 1024 / 256 / 64 / 16 / 4 / 1 and a 2,048-token cache, exported in float32 with the graph kept for the quantization step. The folder name `src` and the package name `lfm3_ext` appear in the graph's tensor names (transformers names the remote-code module after its folder and a hash of the code files: `transformers_modules.src.9cc417484026aa57`), so run the script as is: the `fetch` step puts the source in a folder of that name.
- **Quantization** (`tools/vl_decoder.py quant`, from the float32 graphs). `LFM3-VL-2B_int4.litertlm`: every fully connected weight of the decoder int4 blockwise-32 (symmetric min-max, `recipe_int4_b32_minmax.json`), except the tied output head, which stays int8 per channel; block scales are floored at 2^-14 before the weights are rounded (`lfm3_work/quant_pack.py`'s floor; XNNPACK refuses smaller ones), which in this model raises 3,517 scales in 4 tensors. `LFM3-VL-2B_f16.litertlm`: every fully connected weight of the decoder cast to fp16 (`recipe_f16.json`). Both files: the token embedder's table int8 per channel (`dynamic_wi8_afp32`), no activation type declared for the decoder (a GPU runs it at its default fp16 activations). The int4 file is the faster one on the Galaxy S26 GPU, the Pixel 8a GPU and the Mac GPU (the model card). Before any file was measured, we had asked that at least 21 of 23 picture answers keep the bfloat16 source model's first 32 tokens: the fp16 decoder does (22 of 23 with thinking off), the int4 one does not (17 of 23), so the f16 file came first; the int4 file followed after it answered right every test question with a known answer that it was given (18 of 18 on the Mac GPU and CPU, the 7 asked on the Galaxy S26 and Pixel 8a GPUs, with thinking off and on), the 8 short text questions and "Paris".
- **Chat template** (`tools/vl_template.py`): the source template with four edits. Picture parts render `<|vision_start|><image_soft_token><|vision_end|>`, where LiteRT-LM splices the projector's 256 embeddings. The two `{%- generation -%}` tags are removed and the 26 `.get("…")` lookups are written as `["…"]`, which keeps the rendered text and lets LiteRT-LM's template engine run the file. Earlier assistant turns render without their thought wherever they sit (`keep_thinking = preserve_thinking`), because LiteRT-LM keeps a conversation's earlier prompt as a prefix of the next one. On 10 test conversations the result renders the source processor's token ids, and LiteRT-LM 0.18.1's own rendering equals it on 5. The file declares a thought channel, `<think>` … `</think>`; `enable_thinking: false` in the conversation's extra context turns thinking off.
- **Tokenizer section = the repository's `tokenizer.json`**, byte for byte (17,908,258 bytes, sha256 `a39eed58008cbcbc09e5b629b81338a654269c64cd13348aaa6090f4db75c7e7`).
- Measured (the model card): on the Galaxy S26 GPU (the Android build of LiteRT-LM v0.18.1, `litert_lm_advanced_main`) the int4 file decodes 37.88 tok/s after a 1,024-token prompt against 14.96 for the f16 file in the same session (the two files alternated); on the Pixel 8a GPU it answers a 512 × 512 picture question with its first token after 6.24 s, while on the same day the f16 file's engine start was stopped there by the test's free-memory line; on the Mac M4 Max GPU (WebGPU) it decodes 179.0 tok/s. On the phones the timings and the thinking-off answers used a copy of each file whose template defaults to thinking off (`enable_thinking | default(true)` → `enable_thinking |default(false)`, 12 bytes; the Android command-line tool has no thinking switch), the thinking-on answers the int4 file itself.

## Muse-Glimmer-30B — a dense 52-layer decoder through `export_hf` behind a text-only wrapper, int4 with its KV cache shortened to 2,048 tokens in the int4 file, and a vision tower re-authored for one 448 × 448 picture and lifted to rank 4 for the GPU (2026-10-31)

[`muse_glimmer_work/`](muse_glimmer_work/) converts [meta-models/Muse-Glimmer-30B](https://huggingface.co/meta-models/Muse-Glimmer-30B) at revision `cec94fcf` — Meta's vision-language model, Apache License 2.0; a 52-layer dense text decoder (three sliding-window layers, then one global layer, repeating) and a 50-layer vision encoder — into `Muse-Glimmer-30B_q4_ekv2048.litertlm` (23,393,996,524 B) of [litert-community/Muse-Glimmer-30B-LiteRT](https://huggingface.co/litert-community/Muse-Glimmer-30B-LiteRT). [muse_glimmer_work/README.md](muse_glimmer_work/README.md) has the steps, what had to change and how each change was checked. The KV-cache rewrite and the prefill clone are [`tools/gpu_graph/cache_variant.py`](tools/gpu_graph/cache_variant.py) and [`tools/gpu_graph/prefill_bucket_clone.py`](tools/gpu_graph/prefill_bucket_clone.py); the vision encoder's GPU rewrite is [`tools/gpu_graph/tfl_rank4.py`](tools/gpu_graph/tfl_rank4.py).

| section | bytes | sha256 |
|---|---|---|
| LlmMetadataProto | 3,669 | `a92f7a7e4ebe6aa7805f066dbc02b1d6f0a8f4d037ee5f1839b6414e194ac2f6` |
| HF_Tokenizer_Zlib | 4,587,218 | `533740a6ccd9fda4a21ccd1e1a968a484c2c652c05fd873385e2d0cda139300a` |
| TFLiteModel (tf_lite_embedder) | 1,347,261,568 | `09b551b8fd6915938a607457305d4d6e712f4e667fe9d81f4d1b071adc6c1cd8` |
| TFLiteModel (tf_lite_prefill_decode) | 14,345,874,736 | `2f06229afa0061a964aeb18399e12cb2aec8e316eff7163d46f4b81a8963e51b` |
| TFLiteModel (tf_lite_vision_encoder) | 7,419,391,744 | `71cc2eddfcd7e12921c97cf041f22c93dda32ee68445d55bdd564394fc5a7ff0` |
| TFLiteModel (tf_lite_vision_adapter) | 276,827,884 | `c09d29e3f9c4afb88b22670622fdc42e17c67abfa48d3fb3ece6761003741898` |

Run on 2026-10-31 from this repository's files, with Python 3.12, ai-edge-litert 2.3.0 and flatbuffers 25.12.19 (the environment of the conversion), the two steps that turn the int4 export (a 4,096-token cache, prefill 128 and 1,024) into the published decoder gave its section above byte for byte:

```bash
python tools/gpu_graph/cache_variant.py model_wi4b128_wi8.tflite model_c2048.tflite --cache 2048 --report cache_variant.json
python tools/gpu_graph/prefill_bucket_clone.py tflite model_c2048.tflite model_p32c2048.tflite --lengths 32 --source prefill_128 --report clone.json
```

They took 405 s and 26 s on a 128 GB Mac with the int4 file on a USB disk (peak memory 0.33 and 0.56 GB). The steps before them (the fp32 copy, the export of about 3 h, the quantization, the vision tower) and the bundle were run for the conversion and not again for this entry.

- **The cache rewrite** (`cache_variant.py`) starts from the cache dimension of the `kv_cache_k_*` / `kv_cache_v_*` inputs, the mask, and the constant that lists the cache positions (litert-torch's sliding-window mask builder), and rewrites every tensor shape and every RESHAPE / SLICE constant that carries the cache length; a number that only happens to equal it is left alone. Every weight buffer is copied byte for byte and checked. It stops on STABLEHLO_COMPOSITE ops and on a fixed SLICE window of the cache larger than the new length. The real weights cut to 4 layers (3 sliding-window, 1 global), exported with a 2,048-token cache and rewritten to 1,024, gave the original file's logits bit for bit (160 decode positions, and 32 after a 128-token prefill).
- **The prefill clone** needed one more rule for this graph: a CONCATENATION that stacks copies of one tensor along the prefill length (the grouped-query mask repeat `[1,1,T,C] × g → [1,1,g·T,C]`) is now followed instead of stopping the tool. The rule only replaces a stop, so a graph the previous version accepted goes through the same code as before.
- Speeds, answers and the demo, measured on a Mac: the model card.

## Qwen2.5-VL-72B-Instruct — the 80-layer text decoder written with the LiteRT Tensor API one tensor at a time (int4, a 2,048-token cache, prefill 32 / 128 / 1,024), and the vision tower re-authored for one 448 × 448 picture and lifted to rank 4 for the GPU (2026-10-31)

The steps are in [john-rocky/tensor-api-harness: models/qwen2.5-vl-72b](https://github.com/john-rocky/tensor-api-harness/tree/main/models/qwen2.5-vl-72b). They convert [Qwen/Qwen2.5-VL-72B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-72B-Instruct) at revision `89c86200` — Qwen's vision-language model under the Qwen LICENSE AGREEMENT; an 80-layer Qwen2 text decoder and a 32-layer vision encoder — into `Qwen2.5-VL-72B-Instruct_q4_ekv2048.litertlm` (44,789,536,688 B) of [litert-community/Qwen2.5-VL-72B-Instruct-LiteRT](https://huggingface.co/litert-community/Qwen2.5-VL-72B-Instruct-LiteRT). The text decoder is written with the LiteRT Tensor API because the converter quantizes after it has written the whole fp32 graph, about 290 GB for this model; the writer quantizes one tensor at a time and never holds that graph. The vision encoder's GPU rewrite is [`tools/gpu_graph/tfl_rank4.py`](tools/gpu_graph/tfl_rank4.py). The harness README says how each part was checked.

| section | bytes | sha256 |
|---|---|---|
| LlmMetadataProto | 1,496 | `27bbe45b28b6617e5f46f385993445cf889ed667187ec588d78303a5ad260d5f` |
| HF_Tokenizer_Zlib | 1,968,066 | `fd77bdd170472306cd229e0961815ae0482e6934137cb6b4336a9caa1a6bdd0e` |
| TFLiteModel (tf_lite_embedder) | 1,247,534,800 | `2d57e3ab0f571c1706379a72d3245d34fd0953f91cfd5723b9add4bff07e7869` |
| TFLiteModel (tf_lite_prefill_decode) | 40,713,491,328 | `7100b1e61726f285714797ecec9e84f5f6784f34a21a4e66ae88c9c1d370464e` |
| TFLiteModel (tf_lite_vision_encoder) | 2,553,783,680 | `571792dcd2e980e10e7a92fbbad7d9eaa4a48ae622f891a8435c3dfd37236f2f` |
| TFLiteModel (tf_lite_vision_adapter) | 272,684,976 | `fbc056d882ff076451b75c2e12c829369ae3213abc701709212de8412cb5dcea` |

Run on 2026-10-31, the writer of that directory, built in a LiteRT checkout at `dcb657e8` (`bazel build -c opt`; the action cache already held a binary built from the same source bytes) and run as its step 3 on the packed weights, gave the decoder section above byte for byte:

```bash
qwen2_writer --weights_dir=PACK --out=PACK/model_p32c2048.tflite --prefill=32,128,1024 --cache=2048
```

It took 1,911 s on a 128 GB Mac, reading the packed weights (38.6 GB) from a USB disk that another job was reading at the same time (986 s on 2026-10-30 without it), with a peak memory of 6.7 GB. The weight pack (about 1.6 h), the token table, the vision tower and the bundle were run for the conversion and not again for this entry.

- Image tokens take 1-D positions (the source model: M-RoPE, three position axes). On three picture questions, Transformers with the bf16 weights chose the same next token as the replies at 83 of 84 positions under either scheme (the replies of an earlier build with a 4,096-token cache; this file gives the same three replies word for word).
- Speeds, answers and the demo, measured on a Mac: the model card.
