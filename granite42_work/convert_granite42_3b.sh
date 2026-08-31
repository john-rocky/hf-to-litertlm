#!/usr/bin/env bash
# ibm-granite/granite-4.2-3b (DENSE GraniteForCausalLM, THINKING) -> .litertlm  (2026-08-31)
#
# granite-4.2-3b is IBM's compact reasoning model (3.66B dense, Aug 2026): 4.2 returns
# from the 4.0/4.1 hybrid (-h) line to full attention, so the granite-4.1-3b dense rail
# applies nearly unchanged. What 4.2 changes (all read from the checkpoint):
#   * THINKING model: ChatML roles (<|im_start|>...<|im_end|>) + <think>(100274) /
#     </think>(100275) added tokens; the official template pre-fills '<think>\n' in the
#     generation prompt. Template rail: templates/granite42_think.jinja — note it uses
#     the chatml_think SHAPE (think opener in the assistant HISTORY branch too): the
#     exporter derives the assistant prefix from a history message, so a template that
#     only opens <think> in its add_generation_prompt branch ships a bundle WITHOUT the
#     think-prefill scaffold. Verify prompt_templates.model.prefix in the built bundle.
#   * bos <s>(100283) != eos <|im_end|>(100257==pad); the tokenizer.json post_processor
#     adds NOTHING, so upstream never sees <s>. NO_START_TOKEN=1 keeps the builder from
#     writing start_token (4.1's echo-the-question lesson; different mechanism, same knob).
#   * UNTIED embeddings (tie_word_embeddings: false) — 3.66B = 4.1's 3.40B shape + a
#     separate 100352x2560 lm_head. EXTERNALIZE_EMBEDDER still splits the input table out.
#   * multipliers: attention 0.015625 only; embedding/logits/residual all 1.0.
#
# Tokenizer: the upstream tokenizer.json is embedded as-is (the exporter's default HF
# path). Do NOT set FORCE_SPM for this model — the BPE->SentencePiece conversion is lossy
# for byte-level BPE vocabularies.
#
# After export, add the thought channel: the structured-template path does not declare
# LlmMetadata.channels, and without it the runtime streams raw reasoning into the answer
# and silently ignores any thinking budget. See tools/add_thought_channel.py.
#
# Toolchain (pristine released stack, plain pip install):
#   litert-torch 0.9.3 / litert-converter 0.4.0 / ai-edge-quantizer 0.9.0 /
#   litert-lm-builder 0.16.1 / transformers 5.14.1
#
# Two builds:
#   int8 (dynamic_wi8_afp32) = quality reference / desktop build.
#   int4 (BOCTAV4)           = phone build, same recipe as granite-4.1-3b.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PY:-python3}
SRC=${SRC:-src_models/granite-4.2-3b}
OUT=granite42_work

export CACHE=${CACHE:-4096}
# 6-signature ladder, not 11: every exported signature is charged engine memory whether
# or not it is called — granite-4.1's 11-signature build was killed by iOS at Metal
# engine init, and the 6-signature build of the same weights passed. Start at 6.
export PREFILL=${PREFILL:-1024,256,64,16,4,1}
export EXTERNALIZE_EMBEDDER=${EXTERNALIZE_EMBEDDER:-1}
export NO_START_TOKEN=${NO_START_TOKEN:-1}

RECIPES=${RECIPES:-"int8 int4"}
for R in $RECIPES; do
  case "$R" in
    int4) Q=BOCTAV4 ;;
    int4b128) Q=BOCTAV4_128 ;;
    int8) Q=dynamic_wi8_afp32 ;;
    *) echo "unknown recipe $R"; exit 2 ;;
  esac
  echo "=== $R ($Q) ==="
  $PY scripts/export_simple_template.py "$SRC" "$OUT/out_$R" \
      templates/granite42_think.jinja "$Q" 2>&1 | tee "$OUT/export_$R.log"
  ls -la "$OUT/out_$R"/*.litertlm
done
echo "CONVERT_DONE"
