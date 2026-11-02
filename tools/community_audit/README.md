# Audit the .litertlm files of a Hugging Face org

`audit.py` reads every `.litertlm` file of an org (default `litert-community`) from its bundle header and writes one table of what each file carries. `run_check.py` runs a few LiteRT-LM checks on chosen files and sets the results against the header. The current table is [`table/audit.md`](table/audit.md) (data: `table/audit.json`, run results: `table/runs.json`).

```sh
pip install litert-lm-builder==0.19.0 jinja2       # audit.py
python3 tools/community_audit/audit.py --out-dir tools/community_audit/table
python3 tools/community_audit/audit.py --self-test --out-dir /tmp/selftest

pip install litert-lm==0.19.0                      # run_check.py
python3 tools/community_audit/run_check.py --list --top 8 --extra litert-community/MiniCPM-V-4 \
    litert-community/Qwen3.5-4B:Qwen3.5-4B_mixed_int4.litertlm
python3 tools/community_audit/run_check.py --dl-dir /tmp/lc_files --top 8 --extra litert-community/MiniCPM-V-4 \
    litert-community/Qwen3.5-4B:Qwen3.5-4B_mixed_int4.litertlm
python3 tools/community_audit/audit.py --render-only --runs tools/community_audit/table/runs.json \
    --out-dir tools/community_audit/table
```

## What the header audit reads

Per file: the bundle header and its LlmMetadata and ExecutorMetadata sections, two or three HTTP range requests and no weights, with the same reads as `manifest/make_manifest.py` (it loads that file for the capability and thinking-control logic). Per repo: the file list at the repo's commit, and the base model's `config.json`, `generation_config.json` and chat template when the card names a `base_model`. No token is sent, so a gated repo becomes a row with the reason.

Each file row has the weights form (inline, or external = TFLiteWeights sections), the ExecutorMetadata summary (state buffers, attention mask, NPU flags), the vision and audio sections, the LLM model type, whether the chat template reads `tools` and `enable_thinking`, the declared channels and how the template switches thinking, the sampler, `max_num_tokens`, and the backend constraint of the main model section. The flags and their rules are listed at the top of the table: `rebuild:` marks an LLM file that lacks a form the audit asks of every LLM file, `warn:` marks something to look at, and `source-` flags compare the file with its base model's config.

## Same table again

Every repo is read at its commit, and the table head carries the date and the sha256 of the sorted repo@commit list. `--pin table/audit.json` reads the same commits again. On 2026-11-02 the second read gave a byte-identical `audit.md` and an `audit.json` that differs only in `generated`. Download counts come from the first read and are carried over by `--pin`.

## Failure paths

`--self-test` runs five of them, and each one has to end as a row with the stage and the reason: a repo that does not exist (the Hub answers HTTP 401 for it, not 404), a gated repo (the file list is public, every range read gets HTTP 401), a server that ignores the Range header (Python's `http.server`), and a real bundle header cut to 24 B and to 4,096 B. `table/selftest.txt` holds the last result. A full run never stops on a failed repo or file. Those go to the "Failed reads" table.

## What only a run can tell

The header cannot tell whether a file builds an engine on a given backend or answers correctly. `run_check.py` takes the top LLM repos by Hub downloads (gated repos and NPU-only files left out), one file per repo (a file without a backend constraint first, then the smallest), plus files named with `--extra`. The table's run names two: MiniCPM-V-4-int8, the one LLM file whose model sections are all constrained to the CPU, and Qwen3.5-4B_mixed_int4, a generic_model file whose template reads `tools` and that has no tool_call_config (the `rebuild:tool-calls-as-text` case). It downloads each file at the audited commit, checks its size and sha256 against the Hub's LFS record, and runs each check in a new process on the CPU and the GPU: engine creation with LiteRT-LM's own `ModelInfo`, a one-word text answer, a photo question for files with a vision section, one tool-call exchange (the tool returns a value the model cannot guess), and thinking on and off for files that declare a channel or whose template reads `enable_thinking`. A run that ends in `DEADLINE_EXCEEDED` is tried once more in a new process. When a vision file builds no GPU engine, the checks run once more with the vision encoder on the CPU (`gpu_vcpu`). The notes in `runs.json` (why a tool check found no call, each failed check set against the header, each LiteRT-LM `ModelInfo` field that differs from the header) are derived from the stored results, so `--resume` gives them again without a download: it skips the checks already in `runs.json` and the download of a file whose checks are all there. The photo is used by reference (COCO val2017 `000000039769.jpg`, its sha256 is recorded in `runs.json`).
