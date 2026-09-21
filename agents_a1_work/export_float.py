"""Only step 1 of qwen35_work/convert_qwen35_hybrid.py, with portable inputs."""
import argparse
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import json
from runtime_helpers import HERE, REPO, ROOT, save_json

REVISION = '945c40a4aa6f534d434a353207b8d42ecf7a5293'
TORCH_REVISION = '115a13607c730c81018bb9789138a3e5e5119e3d'
REPO_ID = 'InternScience/Agents-A1-4B'
SOURCE_HASHES = {
    'model-00000-of-00002.safetensors': 'bc19e22bc1251efa260225a8c5d4a29ca7a063c8db755aa4d13d13f808bf29a3',
    'model-00001-of-00002.safetensors': 'b2376df4875ae41823039bff405cfc1f61e4205e6864c7961998e87c418789bf',
    'tokenizer.json': '87a7830d63fcf43bf241c3c5242e96e62dd3fdc29224ca26fed8ea333db72de4',
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def verify_patch(path):
    expected = json.loads((HERE / 'patch_state.json').read_text())['files']
    actual = {name: sha256(path / name) if (path / name).is_file() else None for name in expected}
    assert actual == expected, 'Patched source files differ from the measured export'


def prepare_checkout(path):
    url = 'https://github.com/google-ai-edge/litert-torch'
    fresh = not path.exists()
    if fresh:
        subprocess.run(['git', 'clone', '--no-checkout', '--depth', '1', url, str(path)], check=True)
    head = subprocess.run(['git', '-C', str(path), 'rev-parse', 'HEAD'], capture_output=True, text=True)
    patch = REPO / 'qwen35_work/qwen35_hybrid_litert_torch.patch'
    if head.stdout.strip() == TORCH_REVISION:
        try:
            verify_patch(path)
            return
        except AssertionError:
            pass
    dirty = subprocess.check_output(['git', '-C', str(path), 'status', '--porcelain'], text=True)
    if dirty.strip() and not fresh:
        raise SystemExit('Use a clean dedicated litert-torch checkout; existing changes will not be replaced.')
    subprocess.run(['git', '-C', str(path), 'fetch', '--depth', '1', url, TORCH_REVISION], check=True)
    subprocess.run(['git', '-C', str(path), 'checkout', '--detach', TORCH_REVISION], check=True)
    subprocess.run(['git', '-C', str(path), 'apply', str(patch)], check=True)
    subprocess.run(['git', '-C', str(path), 'apply', str(HERE / 'qwen35_export_compat.patch')], check=True)
    verify_patch(path)


def download_checkpoint(model, output):
    os.environ['HF_HUB_DISABLE_XET'] = '1'
    os.environ.setdefault('HF_HOME', str(output / 'cache/hf'))
    from huggingface_hub import snapshot_download
    snapshot_download(REPO_ID, revision=REVISION, local_dir=str(model), max_workers=4,
                      allow_patterns=list(SOURCE_HASHES) + ['config.json', 'tokenizer_config.json',
                                     'chat_template.jinja', 'model.safetensors.index.json'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, default=Path(os.environ.get('AGENTS_A1_CHECKPOINT', 'src_models/Agents-A1-4B')))
    parser.add_argument('--output', type=Path, default=ROOT / 'float')
    parser.add_argument('--litert-torch-dir', type=Path, default=Path(os.environ.get('LITERT_TORCH_DIR', 'qwen35_work/litert-torch-qwen35')))
    parser.add_argument('--python', default=os.environ.get('CONVERTER_PYTHON', sys.executable))
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--download', action='store_true')
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    model, output, checkout = args.model.resolve(), args.output.resolve(), args.litert_torch_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.prepare:
        prepare_checkout(checkout)
    if args.download:
        download_checkpoint(model, output.parent)
    verified = {name: sha256(model / name) for name in SOURCE_HASHES}
    if verified != SOURCE_HASHES:
        raise SystemExit('Checkpoint SHA256 mismatch; keep/resume the download and investigate.')
    actual = subprocess.check_output(['git', '-C', str(checkout), 'rev-parse', 'HEAD'], text=True).strip()
    assert actual == TORCH_REVISION, actual
    verify_patch(checkout)
    save_json(output / 'source.json', {'repo': REPO_ID, 'revision': REVISION, 'sha256': verified,
                                     'litert_torch_revision': actual})
    if args.prepare_only:
        return
    assert not (output / 'model.litertlm').exists(), 'Use --float-bundle in the build step to reuse an export.'
    env = dict(os.environ, PYTHONPATH=str(checkout), QWEN35_PREFILL_LADDER='1024,256,64,16,4,1',
               HF_HUB_DISABLE_XET='1', HF_HOME=str(output.parent / 'cache/hf'),
               TORCH_HOME=str(output.parent / 'cache/torch'), PYTHONDONTWRITEBYTECODE='1')
    # The same CLI entry point and arguments as the existing hybrid rail, without
    # executing its later generic-template or int8-only steps.
    command = [args.python, '-B', '-c', 'from litert_torch.cli import main; raise SystemExit(main())',
               'export_hf', '--model', str(model), '--output_dir', str(output),
               '--prefill_lengths', '1024,256,64,16,4,1', '--cache_length', '4096',
               '--quantization_recipe', '']
    with (output / 'export.log').open('w') as log:
        subprocess.run(command, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, check=True)
    assert (output / 'model.litertlm').is_file()


if __name__ == '__main__':
    main()
