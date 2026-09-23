"""Identity-template repack using the owner's step-3 string escaping."""
import argparse
import re
import subprocess
import sys
from pathlib import Path
from google.protobuf import text_format
from litert_lm_builder.runtime.proto import llm_metadata_pb2
from common import ROOT, write_json, file_fact
from identity_template import IDENTITY


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('src')
    ap.add_argument('dst')
    ap.add_argument('--unpack', required=True)
    args = ap.parse_args()
    dst, unpack = Path(args.dst), Path(args.unpack)
    assert not dst.exists() and not unpack.exists(), 'Refusing to reuse a previous artifact'
    cli = str(ROOT / 'venv-export/bin/litert-lm')
    subprocess.run([sys.executable, '-B', cli, 'unpack', args.src, '--output-dir', str(unpack)], check=True)
    path = unpack / 'LlmMetadataProto.pbtext'
    original = path.read_text()
    meta = llm_metadata_pb2.LlmMetadata()
    text_format.Parse(original, meta)
    assert not meta.HasField('start_token')
    assert [v for t in meta.stop_tokens for v in t.token_ids.ids] == [248044]
    escaped = IDENTITY.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n')
    updated, count = re.subn(r'jinja_prompt_template: "(?:[^"\\]|\\.)*"',
                            lambda _: 'jinja_prompt_template: "' + escaped + '"', original)
    assert count == 1
    parsed = llm_metadata_pb2.LlmMetadata()
    text_format.Parse(updated, parsed)
    assert parsed.jinja_prompt_template == IDENTITY
    meta.jinja_prompt_template = IDENTITY
    assert meta == parsed, 'Metadata changed beyond the template'
    path.write_text(updated)
    subprocess.run([sys.executable, '-B', cli, 'pack', str(unpack / 'model.toml'), '--output', str(dst)], check=True)
    assert dst.is_file()
    write_json(f'results/{dst.stem}_template_repack.json', dict(
        status='PASS', source=str(Path(args.src).relative_to(ROOT)), bundle=file_fact(dst),
        metadata_changed_only_template=True, start_token_present=False, stop_token_ids=[248044],
        template=IDENTITY, template_has_trailing_newline=IDENTITY.endswith('\n')))


if __name__ == '__main__':
    main()
