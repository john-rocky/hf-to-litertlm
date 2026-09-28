"""Round 4 bundle facts -> results/bundle_r4_<variant>.json (litert-lm 0.17.1 / litert-lm-builder 0.17.1,
out/venv-readout). A parametrized copy of scripts/inspect_bundle_r3.py (unchanged checks; paths per variant; the
content readout is compared with the same variant's CPU graph readout).

Reads the FINAL bundle out/bundle_r4/<v>/decider-2b-vision_<v>.litertlm (after ExecutorMetadata) and its unpacked
sections (out/bundle_r4/<v>/unpack, written by `litert-lm unpack`):
  - header: `litert-lm describe`, `peek_litertlm_file` (no dump), model.toml sections and their metadata,
    LlmMetadata (start token absent, stop ids, max tokens, fast_vlm size, prompt_templates presence, jinja),
    ExecutorMetadata state-buffer census;
  - bytes: sha256 of every unpacked tflite against the file the builder was given (out/weights_r4/<v>/*,
    out/fp16_r3/vision_*_fp16.tflite), and the zlib
    tokenizer section decompressed against the snapshot's tokenizer.json;
  - template: minijinja renders of the bundle's own jinja, parts [image, text] / [text] and the string form, with
    add_generation_prompt false and true, compared character by character with the expected prompt for every
    round-3 row (image rows: <|vision_start|><image_soft_token><|vision_end|> + question text; no-image rows: the
    text), and the bundle tokenizer's ids for that text against the oracle's ids after <|vision_end|>.

    out/venv-readout/bin/python -B scripts/inspect_bundle_r4.py --variant fp16|dyn8|v7c
"""
import argparse
import hashlib
import io
import json
import os
import re
import subprocess
import sys
import tomllib
import zlib
from collections import Counter

import minijinja
from google.protobuf import text_format
from litert_lm_builder import peek_litertlm_file
from litert_lm_builder.runtime.proto import llm_metadata_pb2
from tokenizers import Tokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import ROOT, read_json, write_json, sha256_file          # noqa: E402
from r3_rows import IMAGE_ROWS, TEXT_ROWS, IMG_RENDER, oracle_rows    # noqa: E402

VARIANT = None
NAME = BUNDLE = UNPACK = BUNDLE_DIR = None
LITERT_LM = ROOT / 'out/venv-readout/bin/litert-lm'


def set_variant(v):
    global VARIANT, NAME, BUNDLE, UNPACK, BUNDLE_DIR
    VARIANT = v
    NAME = f'decider-2b-vision_{v}'
    BUNDLE_DIR = ROOT / f'out/bundle_r4/{v}'
    BUNDLE = BUNDLE_DIR / f'{NAME}.litertlm'
    UNPACK = BUNDLE_DIR / 'unpack'


def content_bit_identity():
    """The graph readout on the tflites unpacked from the bundle (results/bundle_readout_r4_<v>.json, fresh XNNPACK
    weight cache) against the variant's CPU graph readout (results/readout_r4_<v>.json): every row's text-embedding and
    vision-embedding hashes, and every slot's full-vocabulary logits hash, letter logits, probabilities and top-1 id."""
    b, a = read_json(f'results/bundle_readout_r4_{VARIANT}.json'), read_json(f'results/readout_r4_{VARIANT}.json')
    arows = {r['row_id']: r for r in a['rows']}
    files = dict(decoder=b['decoder']['sha256'] == a['decoder']['sha256'], embedder=b['embedder']['sha256'] == a['embedder']['sha256'],
                 vision_encoder=b['vision'][VARIANT]['encoder']['sha256'] == a['vision'][VARIANT]['encoder']['sha256'],
                 vision_adapter=b['vision'][VARIANT]['adapter']['sha256'] == a['vision'][VARIANT]['adapter']['sha256'])
    n = same = 0
    diffs = []
    emb_same = vis_same = 0
    for rb in b['rows']:
        ra = arows[rb['row_id']]
        emb_same += rb['text_embeddings_sha256'] == ra['text_embeddings_sha256']
        if rb['image']:
            vis_same += rb['vision_sha256'][VARIANT] == ra['vision_sha256'][VARIANT]
        for arm, v in rb['arms'].items():
            for sb, sa in zip(v['slots'], ra['arms'][arm]['slots']):
                n += 1
                eq = (sb['logits_sha256'] == sa['logits_sha256'] and sb['probs'] == sa['probs'] and
                      sb['letter_logits'] == sa['letter_logits'] and sb['vocab_top1_id'] == sa['vocab_top1_id'])
                same += eq
                if not eq:
                    diffs.append(dict(row_id=rb['row_id'], arm=arm, k=sb['k']))
    n_img = sum(r['image'] for r in b['rows'])
    return dict(readout=f'results/bundle_readout_r4_{VARIANT}.json', readout_status=b['status'], xnn_cache=b['xnn_cache'],
                reference=f'results/readout_r4_{VARIANT}.json (arm {VARIANT} + both no-image arms)',
                reference_xnn_cache=a['xnn_cache'],
                arms=sorted({arm for r in b['rows'] for arm in r['arms']}), files_sha256_equal=files,
                rows=len(b['rows']), text_embeddings_equal=f"{emb_same}/{len(b['rows'])}", vision_embeddings_equal=f'{vis_same}/{n_img}',
                slots_compared=n, slots_bit_identical=same, differing_slots=diffs,
                bit_identical=all(files.values()) and same == n and emb_same == len(b['rows']) and vis_same == n_img and n > 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--variant', required=True, choices=('fp16', 'dyn8', 'v7c'))
    set_variant(ap.parse_args().variant)
    build = json.loads((BUNDLE_DIR / f'{NAME}_noexec.build.json').read_text())
    res = dict(status='RUNNING', bundle=dict(path=os.path.relpath(BUNDLE, ROOT), bytes=os.path.getsize(BUNDLE), sha256=sha256_file(BUNDLE)),
               pre_executor_metadata=dict(path=build['output'], bytes=build['bytes'], sha256=build['sha256']),
               versions={p: __import__('importlib.metadata').metadata.version(p) for p in ('litert-lm', 'litert-lm-builder', 'minijinja', 'tokenizers')})
    d = subprocess.run([str(LITERT_LM), 'describe', str(BUNDLE)], capture_output=True, text=True, stdin=subprocess.DEVNULL)
    res['describe'] = dict(rc=d.returncode, stdout=d.stdout)
    peek = io.StringIO()
    peek_litertlm_file(str(BUNDLE), None, peek)
    res['peek'] = peek.getvalue()

    toml = tomllib.loads((UNPACK / 'model.toml').read_text())
    sections = []
    for s in toml['section']:
        entry = dict(section_type=s['section_type'], model_type=s.get('model_type'), data_path=s['data_path'],
                     additional_metadata={m['key']: m['value'] for m in s.get('additional_metadata', [])})
        p = UNPACK / s['data_path']
        entry['bytes'] = os.path.getsize(p)
        if p.suffix == '.tflite':
            entry['sha256'] = sha256_file(p)
        sections.append(entry)
    res['sections'] = sections
    res['system_metadata'] = {e['key']: e['value'] for e in toml['system_metadata']['entries']}

    # bytes shipped == bytes the builder was given (== bytes the fp16 readout measured)
    src = {k: v for k, v in build['inputs'].items()}
    kind_of = {'embedder': 'EMBEDDER', 'prefill_decode': 'PREFILL_DECODE', 'vision_encoder': 'VISION_ENCODER', 'vision_adapter': 'VISION_ADAPTER'}
    identity = {}
    for s in sections:
        if s['section_type'] == 'TFLiteModel':
            k = kind_of[s['model_type']]
            identity[k] = dict(unpacked=s['data_path'], source=src[k]['path'], sha256_equal=s['sha256'] == src[k]['sha256'],
                               bytes_equal=s['bytes'] == src[k]['bytes'], source_sha256=src[k]['sha256'])
    res['section_byte_identity'] = identity
    tok_section = next(s for s in sections if s['section_type'] == 'HF_Tokenizer')
    raw = (UNPACK / tok_section['data_path']).read_bytes()
    tok_json = zlib.decompress(raw[8:] if raw[:2] != b'\x78\x9c' else raw)
    res['tokenizer_identity'] = dict(section=tok_section['data_path'], decompressed_sha256=hashlib.sha256(tok_json).hexdigest(),
                                     snapshot_sha256=build['tokenizer']['sha256'],
                                     equal=hashlib.sha256(tok_json).hexdigest() == build['tokenizer']['sha256'])

    md = llm_metadata_pb2.LlmMetadata()
    text_format.Parse((UNPACK / 'LlmMetadataProto.pbtext').read_text(), md)
    jinja = md.jinja_prompt_template
    res['llm_metadata'] = dict(
        pbtext=(UNPACK / 'LlmMetadataProto.pbtext').read_text(),
        start_token_present=md.HasField('start_token'), stop_token_ids=[i for t in md.stop_tokens for i in t.token_ids.ids],
        max_num_tokens=md.max_num_tokens, model_type=md.llm_model_type.WhichOneof('model_type'),
        fast_vlm=dict(image_tensor_height=md.llm_model_type.fast_vlm.image_tensor_height,
                      image_tensor_width=md.llm_model_type.fast_vlm.image_tensor_width),
        prompt_templates_present=md.HasField('prompt_templates'),
        prompt_templates={r: dict(present=md.prompt_templates.HasField(r), prefix=getattr(md.prompt_templates, r).prefix,
                                  suffix=getattr(md.prompt_templates, r).suffix) for r in ('user', 'model', 'system')},
        jinja=jinja, jinja_equals_build=jinja == build['jinja'], jinja_has_trailing_newline=jinja.endswith('\n'),
        jinja_sha256=hashlib.sha256(jinja.encode()).hexdigest())
    ex = (UNPACK / 'ExecutorMetadataProto.pbtext').read_text()
    types = Counter(re.findall(r'type: (TYPE_\w+)', ex))
    names = re.findall(r'decode_input_name: "(kv_cache_\w+)"', ex)
    res['executor_metadata'] = dict(state_buffers=len(names), by_type=dict(types),
                                    by_prefix=dict(Counter(n.split('_')[2] for n in names)),
                                    kv_max_sequence_length=sorted(set(int(v) for v in re.findall(r'maximum_sequence_length: (\d+)', ex))))

    # template renders + tokenizer ids, per round-3 row
    tk = Tokenizer.from_str(tok_json.decode())
    rows = oracle_rows(read_json)
    renders, all_ok = [], True
    for rid in IMAGE_ROWS + TEXT_ROWS:
        row = rows[rid]
        text = tk.decode(row['text_ids'], skip_special_tokens=False)
        enc = tk.encode(text, add_special_tokens=False).ids
        expect_parts = (IMG_RENDER if row['image'] else '') + text
        cases = []
        forms = [('parts_image_text', [dict(type='image'), dict(type='text', text=text)], expect_parts)] if row['image'] else []
        forms += [('parts_text', [dict(type='text', text=text)], text), ('string', text, text)]
        for form, content, expect in forms:
            for gen in (False, True):
                out = minijinja.render_str(jinja, messages=[dict(role='user', content=content)], add_generation_prompt=gen)
                cases.append(dict(form=form, add_generation_prompt=gen, equal=out == expect, rendered_len=len(out), expected_len=len(expect)))
        ok = all(c['equal'] for c in cases) and enc == row['text_ids']
        all_ok &= ok
        renders.append(dict(row_id=rid, image=row['image'], text=text, text_ids_equal=enc == row['text_ids'],
                            n_text_ids=len(row['text_ids']), n_input_ids=len(row['input_ids']), cases=cases, ok=ok))
    res['renders'] = renders
    lm = res['llm_metadata']
    checks = dict(
        describe_rc0=d.returncode == 0, describe_vision='Vision' in d.stdout,
        sections=[(s['section_type'], s['model_type']) for s in sections] == [
            ('LlmMetadata', None), ('ExecutorMetadata', None), ('HF_Tokenizer', None), ('TFLiteModel', 'embedder'),
            ('TFLiteModel', 'prefill_decode'), ('TFLiteModel', 'vision_encoder'), ('TFLiteModel', 'vision_adapter')],
        prefill_decode_fp32_activation=next(s for s in sections if s['model_type'] == 'prefill_decode')['additional_metadata'].get('prefer_activation_type') == 'fp32',
        no_start_token=not lm['start_token_present'], stop_only_248044=lm['stop_token_ids'] == [248044],
        max_num_tokens_4096=lm['max_num_tokens'] == 4096, fast_vlm_256=lm['fast_vlm'] == dict(image_tensor_height=256, image_tensor_width=256),
        prompt_templates_present_empty=lm['prompt_templates_present'] and all(v['present'] and not v['prefix'] and not v['suffix'] for v in lm['prompt_templates'].values()),
        jinja_equals_build=lm['jinja_equals_build'], states_48=res['executor_metadata']['state_buffers'] == 48,
        states_36_linear_12_kv=res['executor_metadata']['by_type'] == {'TYPE_LINEAR_ATTENTION': 36, 'TYPE_GLOBAL_KEY_CACHE': 6, 'TYPE_GLOBAL_VALUE_CACHE': 6},
        tflite_sections_byte_identical=all(v['sha256_equal'] and v['bytes_equal'] for v in identity.values()) and len(identity) == 4,
        tokenizer_identical=res['tokenizer_identity']['equal'], renders_and_ids_all_equal=all_ok)
    # per-signature op census of the shipped decoder (read against the GPU delegate's partition message)
    from tflite_scan import scan
    pd = next(s for s in sections if s['model_type'] == 'prefill_decode')
    sc = scan(UNPACK / pd['data_path'], with_sha=False)
    sig_of = {v['subgraph_index']: k for k, v in sc['signatures'].items()}
    res['decoder_signature_ops'] = {sig_of.get(g['index'], str(g['index'])): dict(
        operators=g['operator_count'], DEQUANTIZE=g['op_histogram'].get('DEQUANTIZE', 0),
        RELU_0_TO_1=g['op_histogram'].get('RELU_0_TO_1', 0), RELU=g['op_histogram'].get('RELU', 0),
        FULLY_CONNECTED=g['op_histogram'].get('FULLY_CONNECTED', 0))
        for g in sc['subgraphs']}
    if (ROOT / f'results/bundle_readout_r4_{VARIANT}.json').exists():
        res['content_readout'] = content_bit_identity()
        checks['content_readout_bit_identical_to_A'] = res['content_readout']['bit_identical']
    res['checks'] = checks
    res['status'] = 'PASS' if all(checks.values()) else 'FAIL'
    write_json(f'results/bundle_r4_{VARIANT}.json', res)
    print('BUNDLE_INSPECT', res['status'], {k: v for k, v in checks.items() if not v} or 'all checks true', flush=True)


if __name__ == '__main__':
    main()
