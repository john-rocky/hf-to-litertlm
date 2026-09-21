"""Build both quantized bundles serially from one float export."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from runtime_helpers import HERE, REPO, ROOT, save_json
from export_float import sha256


def rewrite_metadata(directory, template):
    from google.protobuf import json_format, text_format
    from litert_lm_builder.runtime.proto import llm_metadata_pb2
    path = directory / 'LlmMetadataProto.pbtext'
    meta = text_format.Parse(path.read_text(), llm_metadata_pb2.LlmMetadata())
    before = json_format.MessageToDict(meta, preserving_proto_field_name=True)
    structured = meta.prompt_templates.SerializeToString()
    assert 248044 in [i for stop in meta.stop_tokens for i in stop.token_ids.ids]
    meta.jinja_prompt_template = template
    if 248046 not in [i for stop in meta.stop_tokens for i in stop.token_ids.ids]:
        meta.stop_tokens.add().token_ids.ids.append(248046)
    meta.ClearField('sampler_params')
    text_format.Merge('sampler_params { type: TOP_P k: 20 p: 0.95 temperature: 0.85 }', meta)
    meta.max_num_tokens = 4096
    meta.ClearField('llm_model_type')
    meta.llm_model_type.generic_model.SetInParent()
    meta.ClearField('channels')
    text_format.Merge('channels { channel_name: "thought" start: "<think>\\n" end: "\\n</think>" }', meta)
    assert structured == meta.prompt_templates.SerializeToString()
    path.write_text(text_format.MessageToString(meta, as_utf8=True))
    return {'exporter_metadata': before, 'structured_model_prefix': before.get('prompt_templates', {}).get('model', {}).get('prefix'),
            'structured_templates_unchanged': True, 'presence_penalty': 'No runtime equivalent; omitted'}


def verify_bundle(path, template):
    from google.protobuf import json_format
    from bundle_header import read_header
    sections, meta = read_header(path)
    metadata = json_format.MessageToDict(meta, preserving_proto_field_name=True)
    ids = [i for stop in meta.stop_tokens for i in stop.token_ids.ids]
    sampler, channels = metadata.get('sampler_params', {}), metadata.get('channels', [])
    models = [s for s in sections if s['type'] == 'TFLiteModel']
    checks = {
        'template': meta.jinja_prompt_template == template,
        'stop_ids': set(ids) == {248044, 248046},
        'generic_model': meta.llm_model_type.HasField('generic_model'),
        'cache_4096': meta.max_num_tokens == 4096,
        'sampler': sampler.get('type') == 'TOP_P' and sampler.get('k') == 20 and abs(sampler.get('p', 0)-.95)<1e-6 and abs(sampler.get('temperature', 0)-.85)<1e-6,
        'thought': channels == [{'channel_name': 'thought', 'start': '<think>\n', 'end': '\n</think>'}],
        'fp32_activations': len(models) == 1 and models[0]['items'].get('prefer_activation_type') == 'fp32',
        'executor_metadata': any(s['type'] == 'ExecutorMetadataProto' for s in sections),
    }
    assert all(checks.values()), checks
    return {'file': path.name, 'size_bytes': path.stat().st_size, 'sha256': sha256(path),
            'stop_ids': ids, 'sampler': sampler, 'channels': channels, 'sections': sections,
            'checks': checks, 'verdict': 'PASS'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, default=Path(os.environ.get('AGENTS_A1_CHECKPOINT', 'src_models/Agents-A1-4B')))
    p.add_argument('--output', type=Path, default=ROOT)
    p.add_argument('--float-bundle', type=Path)
    p.add_argument('--python', default=os.environ.get('CONVERTER_PYTHON', sys.executable))
    p.add_argument('--template-python', default=os.environ.get('TEMPLATE_PYTHON', sys.executable))
    p.add_argument('--packager', default=os.environ.get('PACKAGER'))
    p.add_argument('--builder', default=os.environ.get('LITERT_LM_BUILDER'))
    p.add_argument('--litert-torch-dir', type=Path, default=Path(os.environ.get('LITERT_TORCH_DIR', 'qwen35_work/litert-torch-qwen35')))
    p.add_argument('--prepare', action='store_true')
    p.add_argument('--download', action='store_true')
    p.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    a = p.parse_args()
    if a.python != sys.executable and not a.worker:
        raise SystemExit(subprocess.call([a.python, '-B', str(Path(__file__).resolve()), *sys.argv[1:], '--worker']))
    output, model = a.output.resolve(), a.model.resolve()
    output.mkdir(parents=True, exist_ok=True)
    results = output / 'results'; results.mkdir(exist_ok=True)
    logs = output / 'logs'; logs.mkdir(exist_ok=True)
    packager = a.packager or str(Path(sys.executable).parent / 'litert-lm')
    builder = a.builder or str(Path(sys.executable).parent / 'litert-lm-builder')
    env = dict(os.environ, HF_HUB_DISABLE_XET='1', HF_HOME=str(output / 'cache/hf'),
               PYTHONDONTWRITEBYTECODE='1', LITERT_LM_BUILDER=builder)

    def run(argv, label, extra_env=None):
        with (logs / (label + '.log')).open('w') as log:
            subprocess.run([str(x) for x in argv], env=dict(env, **(extra_env or {})),
                           stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, check=True)

    float_bundle = a.float_bundle.resolve() if a.float_bundle else output / 'float/model.litertlm'
    if a.float_bundle is None:
        command = [a.python, '-B', HERE / 'export_float.py', '--model', model,
                   '--output', output / 'float', '--litert-torch-dir', a.litert_torch_dir]
        if a.prepare: command.append('--prepare')
        if a.download: command.append('--download')
        run(command, 'export_float')
    assert float_bundle.is_file()
    template_path = output / 'chat_template_agents_a1.jinja'
    run([a.python, '-B', HERE / 'build_chat_template.py', '--model', model,
         '--out', template_path, '--report', results / 'template_source.json'], 'build_template')
    template = template_path.read_text()
    assert template == (HERE / 'chat_template_agents_a1.jinja').read_text(), 'Pinned template changed'
    for mode, python in [('hf', a.python), ('minijinja', a.template_python)]:
        run([python, '-B', HERE / 'render_check.py', mode, '--model', model,
             '--template', template_path, '--output', results], 'render_' + mode)
    assert json.loads((results / 'render_check.json').read_text())['verdict'] == 'PASS'
    rows = []
    for variant, recipe in [('int8', 'wi8fc'), ('mixed_int4', 'wi4b32_wi8')]:
        final = output / ('Agents-A1-4B_' + variant + '.litertlm')
        assert not final.exists(), 'Choose a new output directory; final bundles are never silently overwritten.'
        with tempfile.TemporaryDirectory(prefix=variant + '_', dir=output) as temporary:
            work = Path(temporary)
            child_env = {'TMPDIR': str(work)}
            quant = work / 'quant.litertlm'
            run([a.python, '-B', REPO / 'minicpm5_work/quantize_minicpm5.py', 'apply', float_bundle,
                 quant, '--recipe', recipe, '--algo', 'minmax'], 'quantize_' + variant, child_env)
            run([a.python, '-B', REPO / 'minicpm5_work/quantize_minicpm5.py', 'inspect', quant],
                'inspect_' + variant, child_env)
            unpacked = work / 'unpacked'
            run([packager, 'unpack', quant, '--output-dir', unpacked], 'unpack_' + variant, child_env)
            save_json(results / ('metadata_' + variant + '.json'), rewrite_metadata(unpacked, template))
            repacked, executor = work / 'template.litertlm', work / 'executor.litertlm'
            run([packager, 'pack', unpacked / 'model.toml', '--output', repacked], 'pack_' + variant, child_env)
            run([a.python, '-B', REPO / 'scripts/add_executor_metadata.py', repacked, executor,
                 '--litert-lm', packager, '--python', a.python], 'executor_' + variant, child_env)
            run([a.python, '-B', REPO / 'scripts/set_activation_type.py', executor, final,
                 '--type', 'fp32', '--litert-lm', packager], 'fp32_' + variant, child_env)
            run([a.python, '-B', HERE / 'inspect_layout.py', final, '--variant', variant,
                 '--out', results / ('layout_' + variant + '.json')], 'layout_' + variant)
            layout = json.loads((results / ('layout_' + variant + '.json')).read_text())
            assert layout['verdict'] == 'PASS', 'Unexpected size/layout is a finding; keep the recipe unchanged.'
            rows.append(verify_bundle(final, template))
            save_json(results / 'ship_files.json', rows)
            print('VERIFIED', final, rows[-1]['size_bytes'], 'bytes', rows[-1]['sha256'], flush=True)
    print('DONE: both files in', output)


if __name__ == '__main__':
    main()
