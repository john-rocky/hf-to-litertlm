#!/usr/bin/env python3
"""Replace only a bundle's chat template (LlmMetadata.jinja_prompt_template),
then prove every other section survived byte- or content-identical.

Same unpack / edit pbtext / pack / verify method as tools/add_thought_channel.py.
No conversion, no runtime execution: the weights, tokenizer and executor
metadata of the output are the input's bytes.

Usage (a venv with litert-lm 0.17.1+, which brings litert-lm-builder and the
`litert-lm pack/unpack` CLI):
  python swap_template.py IN.litertlm OUT.litertlm --jinja dual/SHA12.jinja \
      [--cli /path/to/litert-lm] [--report swap_reports/NAME.json]

Any supported bundle with exactly one LlmMetadataProto section is accepted.
Unknown non-metadata section types must survive as raw bytes. Existing output,
report and evidence paths are never overwritten. Failed parity returns nonzero;
the local output is retained for inspection and must not be treated as verified.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
import shutil
import shlex
import struct
import subprocess
import tempfile
import time
import traceback
import zlib
from datetime import datetime, timezone
from pathlib import Path

from google.protobuf import text_format
from litert_lm_builder import litertlm_core
from litert_lm_builder import litertlm_header_schema_py_generated as schema
from litert_lm_builder import litertlm_peek
from litert_lm_builder.runtime.proto import llm_metadata_pb2, executor_metadata_pb2, embedding_metadata_pb2

ROOT = Path.cwd()
CHUNK = 4 << 20
PROTO_CLASSES = {'LlmMetadataProto': llm_metadata_pb2.LlmMetadata,
                 'ExecutorMetadataProto': executor_metadata_pb2.ExecutorMetadata,
                 'EmbeddingMetadataProto': embedding_metadata_pb2.EmbeddingMetadata}


def inside(path):
    return Path(path).expanduser().resolve()


def rel(path):
    p = Path(path)
    return str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(CHUNK), b''):
            h.update(chunk)
    return h.hexdigest()


def read_section(path, section):
    with path.open('rb') as stream:
        stream.seek(section['begin'])
        data = stream.read(section['bytes'])
    if len(data) != section['bytes']:
        raise IOError('Short section read')
    return data


def decode_hf_tokenizer(data):
    # Builder 0.17.1 add_hf_tokenizer writes uint64 LE decoded size, then zlib.
    # Decode the actual container format (size prefix + zlib) rather than
    # comparing raw bytes only, so a re-serialized section is still proven equal.
    if len(data) < 8:
        raise ValueError('HF tokenizer section lacks its 8-byte size prefix')
    declared = int.from_bytes(data[:8], 'little')
    decoded = zlib.decompress(data[8:])
    if len(decoded) != declared:
        raise ValueError('HF tokenizer decoded size differs from its prefix')
    return decoded


def bundle_sections(path):
    """Read exact header offsets, validate bounds, retain ordered section labels."""
    size = path.stat().st_size
    with path.open('rb') as stream:
        prefix = stream.read(32)
        if prefix[:8] != litertlm_core.HEADER_MAGIC_BYTES:
            raise ValueError('Not a .litertlm bundle')
        header_end = int.from_bytes(prefix[litertlm_core.HEADER_END_LOCATION_BYTE_OFFSET:
                                           litertlm_core.HEADER_END_LOCATION_BYTE_OFFSET+8], 'little')
        if not 32 < header_end <= size:
            raise ValueError('Invalid header bounds')
        stream.seek(0)
        head = stream.read(header_end)
    metadata = schema.LiteRTLMMetaData.GetRootAs(head[litertlm_core.HEADER_BEGIN_BYTE_OFFSET:], 0)
    sections = []
    last_end = header_end
    for i in range(metadata.SectionMetadata().ObjectsLength()):
        section = metadata.SectionMetadata().Objects(i)
        begin, end = section.BeginOffset(), section.EndOffset()
        if not last_end <= begin <= end <= size:
            raise ValueError(f'Invalid/overlapping section bounds: {i}')
        last_end = end
        try:
            kind = litertlm_core.any_section_data_type_to_string(section.DataType())
        except ValueError:
            kind = 'Unknown:' + str(section.DataType())
        sections.append({'index':i, 'type':kind, 'data_type':section.DataType(),
                         'begin':begin, 'end':end, 'bytes':end-begin,
                         'metadata':[litertlm_peek.kvp_to_dict(section.Items(j)) for j in range(section.ItemsLength())]})
    system = metadata.SystemMetadata()
    system_items = [litertlm_peek.kvp_to_dict(system.Entries(i)) for i in range(system.EntriesLength())] if system else []
    return {'file_bytes':size, 'format_version':list(struct.unpack('<III',prefix[8:20])),
            'header_end':header_end, 'header_sha256':digest(head),
            'system_metadata':system_items, 'sections':sections}, head


def compare_raw_sections(src, dst, a, b):
    """Streaming exact byte comparison, plus independent section hashes."""
    ha, hb = hashlib.sha256(), hashlib.sha256()
    equal = a['bytes'] == b['bytes']
    with src.open('rb') as fa, dst.open('rb') as fb:
        fa.seek(a['begin']); fb.seek(b['begin'])
        left_a, left_b = a['bytes'], b['bytes']
        while left_a or left_b:
            ca, cb = fa.read(min(CHUNK,left_a)), fb.read(min(CHUNK,left_b))
            if (left_a and not ca) or (left_b and not cb):
                raise IOError('Short read during section comparison')
            ha.update(ca); hb.update(cb)
            if ca != cb:
                equal = False
            left_a -= len(ca); left_b -= len(cb)
    return equal, ha.hexdigest(), hb.hexdigest()


def verify_sections(src, dst, template, artifacts):
    a, head_a = bundle_sections(src)
    b, head_b = bundle_sections(dst)
    (artifacts/'original.header.bin').write_bytes(head_a)
    (artifacts/'new.header.bin').write_bytes(head_b)
    ordered_equal = [s['data_type'] for s in a['sections']] == [s['data_type'] for s in b['sections']]
    result = {'original':a, 'new':b, 'section_list_unchanged':ordered_equal, 'sections':[], 'failures':[]}
    if not ordered_equal:
        result['failures'].append('Section list/order changed')
        result['PASS'] = False
        return result
    for sa,sb in zip(a['sections'], b['sections']):
        same, ha, hb = compare_raw_sections(src,dst,sa,sb)
        kind = sa['type']
        row = {'index':sa['index'], 'type':kind, 'original_bytes':sa['bytes'], 'new_bytes':sb['bytes'],
               'original_sha256':ha, 'new_sha256':hb, 'raw_bytes_identical':same,
               'section_metadata_unchanged':sa['metadata'] == sb['metadata']}
        if kind in PROTO_CLASSES:
            old = PROTO_CLASSES[kind](); new = PROTO_CLASSES[kind]()
            old.ParseFromString(read_section(src,sa)); new.ParseFromString(read_section(dst,sb))
            if kind == 'LlmMetadataProto':
                (artifacts/'original.LlmMetadataProto.bin').write_bytes(read_section(src,sa))
                (artifacts/'new.LlmMetadataProto.bin').write_bytes(read_section(dst,sb))
                row.update(original_template_sha256=digest(old.jinja_prompt_template.encode()),
                           new_template_sha256=digest(new.jinja_prompt_template.encode()),
                           requested_template_sha256=digest(template.encode()),
                           new_template_exact=new.jinja_prompt_template == template)
                old.ClearField('jinja_prompt_template'); new.ClearField('jinja_prompt_template')
                row['comparison'] = 'parsed protobuf after clearing only jinja_prompt_template'
            else:
                row['comparison'] = 'parsed protobuf content'
            row['decoded_content_identical'] = old == new
            row['original_decoded_sha256'] = digest(old.SerializeToString(deterministic=True))
            row['new_decoded_sha256'] = digest(new.SerializeToString(deterministic=True))
            row['PASS'] = old == new and row.get('new_template_exact',True)
        elif kind == 'HF_Tokenizer_Zlib':
            da = decode_hf_tokenizer(read_section(src,sa)); db = decode_hf_tokenizer(read_section(dst,sb))
            row.update(comparison='uint64-size-prefix + zlib-decoded bytes', decoded_content_identical=da==db,
                       original_decoded_bytes=len(da), new_decoded_bytes=len(db),
                       original_decoded_sha256=digest(da), new_decoded_sha256=digest(db), PASS=da==db)
        else:
            row.update(comparison='raw bytes', PASS=same)
        # Per-section labels/constraints must also survive the round trip.
        row['PASS'] = row['PASS'] and row['section_metadata_unchanged']
        if not row['PASS']:
            result['failures'].append(f'Section {sa["index"]} {kind} content/metadata mismatch')
        result['sections'].append(row)
    result['llm_metadata_only_template_changed'] = all(r['PASS'] for r in result['sections'] if r['type']=='LlmMetadataProto')
    result['tflite_weights_byte_identical'] = all(r['raw_bytes_identical'] for r in result['sections'] if r['type'] in ('TFLiteModel','TFLiteWeights'))
    result['system_metadata_unchanged'] = a['system_metadata'] == b['system_metadata']
    result['format_version_unchanged'] = a['format_version'] == b['format_version']
    result['PASS'] = not result['failures']
    return result


def edit_pbtext(path, template, artifacts):
    before = path.read_text()
    old = text_format.Parse(before, llm_metadata_pb2.LlmMetadata())
    replacement = text_format.MessageToString(llm_metadata_pb2.LlmMetadata(jinja_prompt_template=template), as_utf8=True)
    if len(replacement.splitlines()) != 1:
        raise ValueError('Expected single-line escaped protobuf string field')
    lines = before.splitlines(keepends=True)
    found = [i for i,line in enumerate(lines) if line.startswith('jinja_prompt_template:')]
    if len(found)>1:
        raise ValueError('Repeated Jinja field in unpacked metadata')
    if found:
        lines[found[0]] = replacement
    else:
        if before and not before.endswith('\n'):
            lines.append('\n')
        lines.append(replacement)
    after = ''.join(lines)
    parsed = text_format.Parse(after, llm_metadata_pb2.LlmMetadata())
    want = llm_metadata_pb2.LlmMetadata(); want.CopyFrom(old)
    want.jinja_prompt_template = template
    if parsed != want or parsed.jinja_prompt_template != template:
        raise ValueError('Pbtext changed something besides the Jinja field')
    (artifacts/'original.LlmMetadataProto.pbtext').write_text(before, newline='')
    (artifacts/'new.LlmMetadataProto.pbtext').write_text(after, newline='')
    (artifacts/'jinja.pbtext-fragment').write_text(replacement, newline='')
    path.write_text(after, newline='')
    return {'PASS':True, 'only_field_replaced':'jinja_prompt_template',
            'field_previously_present':bool(found), 'original_template_sha256':digest(old.jinja_prompt_template.encode()),
            'parsed_template_sha256':digest(parsed.jinja_prompt_template.encode()),
            'original_pbtext_sha256':digest(before.encode()), 'new_pbtext_sha256':digest(after.encode())}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('source'); parser.add_argument('output')
    parser.add_argument('--jinja',required=True)
    parser.add_argument('--cli',default=shutil.which('litert-lm') or 'litert-lm')
    parser.add_argument('--report',help='JSON evidence path; default swap_reports/<output-name>.json')
    parser.add_argument('--timeout-seconds',type=int,default=600,help='Hard timeout for each pack/unpack process (needs `timeout`/`gtimeout` on PATH; otherwise unlimited)')
    args = parser.parse_args()
    src, dst, jinja, cli = map(inside,[args.source,args.output,args.jinja,args.cli])
    report = inside(args.report or ROOT/'swap_reports'/f'{dst.name}.json')
    artifacts = report.with_suffix('.evidence')
    if src.suffix != '.litertlm' or dst.suffix != '.litertlm' or src == dst:
        parser.error('Distinct input/output .litertlm paths are required')
    if not src.is_file() or not jinja.is_file() or not (cli.is_file() or shutil.which(str(cli))):
        parser.error('Input, Jinja or litert-lm CLI is missing')
    if dst.exists() or report.exists() or artifacts.exists():
        parser.error('Refusing to overwrite output/report/evidence')
    if args.timeout_seconds <= 0:
        parser.error('Timeout must be positive')
    template = jinja.read_bytes().decode('utf-8')
    if not template:
        parser.error('Refusing an empty template')
    dst.parent.mkdir(parents=True,exist_ok=True)
    artifacts.mkdir(parents=True)
    scratch = Path(tempfile.mkdtemp(prefix='swap_template_'))
    started = time.monotonic()
    record = {'source':rel(src), 'output':rel(dst), 'jinja':rel(jinja), 'status':'RUNNING',
              'start_utc':datetime.now(timezone.utc).isoformat(), 'timing_purpose':'informational effort per file, not benchmark',
              'builder_version':importlib.metadata.version('litert-lm-builder'),
              'protobuf_version':importlib.metadata.version('protobuf'),
              'cli':rel(cli), 'timings_seconds':{}, 'commands':[], 'scratch':rel(scratch)}
    def save():
        report.write_text(json.dumps(record,indent=2,ensure_ascii=False)+'\n')
    def run(name, argv):
        overrides = {'PYTHONDONTWRITEBYTECODE':'1','PYTHONUNBUFFERED':'1',
                     'LITERT_LM_DIR':str(scratch/'cli'), 'XDG_CACHE_HOME':str(scratch/'xdg'),
                     'TMPDIR':str(scratch/'tmp')}
        for key in ('LITERT_LM_DIR','XDG_CACHE_HOME','TMPDIR'):
            Path(overrides[key]).mkdir(exist_ok=True)
        timeout_bin = shutil.which('timeout') or shutil.which('gtimeout')
        command = ([timeout_bin,'--signal=TERM','--kill-after=10s',str(args.timeout_seconds)+'s'] if timeout_bin else []) + [str(cli),*argv]
        log = artifacts/(name+'.log')
        begin = time.monotonic()
        with log.open('wb') as stream:
            result = subprocess.run(command,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,
                                    env={**os.environ,**overrides})
        record['timings_seconds'][name] = time.monotonic()-begin
        record['commands'].append({'name':name,'command':shlex.join(['env',*[f'{k}={v}' for k,v in overrides.items()],*command])+' < /dev/null',
                                   'exit_code':result.returncode,'log':rel(log)})
        save()
        text = log.read_text(errors='replace')
        if result.returncode != 0 or 'Error ' in text or 'Traceback (most recent call last)' in text:
            raise RuntimeError(f'{name} failed (exit {result.returncode}); see {rel(log)}')
    try:
        begin = time.monotonic()
        sections,_ = bundle_sections(src)
        if sum(s['type']=='LlmMetadataProto' for s in sections['sections']) != 1:
            raise ValueError('Exactly one LlmMetadataProto section is required')
        record['original_size_bytes'] = src.stat().st_size
        record['original_sha256'] = file_sha(src)
        record['timings_seconds']['input_hash_and_header'] = time.monotonic()-begin
        save()
        unpack = scratch/'unpacked'
        run('unpack',[ 'unpack',str(src),'--output-dir',str(unpack)])
        pbtext = unpack/'LlmMetadataProto.pbtext'
        if not pbtext.is_file() or not (unpack/'model.toml').is_file():
            raise RuntimeError('Unpack did not produce required metadata/config')
        begin = time.monotonic()
        record['pbtext_edit'] = edit_pbtext(pbtext,template,artifacts)
        (artifacts/'model.toml').write_bytes((unpack/'model.toml').read_bytes())
        record['timings_seconds']['edit_and_parse_back'] = time.monotonic()-begin
        run('pack',['pack',str(unpack),'--output',str(dst)])
        if not dst.is_file():
            raise RuntimeError('Pack produced no output bundle')
        begin = time.monotonic()
        record['verification'] = verify_sections(src,dst,template,artifacts)
        record['new_size_bytes'] = dst.stat().st_size
        record['new_sha256'] = file_sha(dst)
        record['input_unchanged_after_pack'] = file_sha(src) == record['original_sha256']
        record['requested_template_sha256'] = digest(template.encode())
        record['timings_seconds']['section_verification_and_hashes'] = time.monotonic()-begin
        record['status'] = 'PASS' if record['verification']['PASS'] and record['input_unchanged_after_pack'] else 'FAIL'
        print(f'{record["status"]}: original {record["original_size_bytes"]} B; new {record["new_size_bytes"]} B',flush=True)
        print(f'original sha256 {record["original_sha256"]}\nnew sha256 {record["new_sha256"]}',flush=True)
        for row in record['verification']['sections']:
            print(f'{row["index"]} {row["type"]}: {row["original_bytes"]} -> {row["new_bytes"]} B; {row["comparison"]}; PASS={row["PASS"]}',flush=True)
        return 0 if record['status']=='PASS' else 1
    except Exception as exc:
        record['status'] = 'ERROR'
        record['error'] = {'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc()}
        raise
    finally:
        begin = time.monotonic()
        shutil.rmtree(scratch)
        record['timings_seconds']['scratch_cleanup'] = time.monotonic()-begin
        record['scratch_removed'] = not scratch.exists()
        record['wall_seconds'] = time.monotonic()-started
        record['end_utc'] = datetime.now(timezone.utc).isoformat()
        save()


if __name__=='__main__':
    raise SystemExit(main())
