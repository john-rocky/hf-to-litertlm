#!/usr/bin/env python3
"""Replace a .litertlm bundle's whole LlmMetadata, then prove every other section survived.

Unpack with the litert-lm CLI, put a new LlmMetadataProto.pbtext in place, pack, verify section by section.

Usage (the verifier must import the same litert_lm_builder the CLI packs with; litert-lm 0.18.0 here):
  uv run --no-project --with litert-lm==0.18.0 python -I repack_metadata.py IN.litertlm OUT.litertlm \
      --pbtext NEW.pbtext --report REPORT.json --expect-changed FIELD[,FIELD...]
  uv run --no-project --with litert-lm==0.18.0 python -I repack_metadata.py IN.litertlm OUT.litertlm \
      --identity --report REPORT.json     # self-test: the unpacked pbtext as is, 0 changed fields expected

PASS (exit 0) requires all of:
  - the unpacked pbtext parses to exactly IN's LlmMetadata (nothing lost by the text form)
  - the set of changed top-level LlmMetadata fields (IN vs OUT, parsed) == --expect-changed, both ways
  - OUT's LlmMetadata parses to exactly NEW.pbtext
  - section list unchanged: count, order, data type, per-section metadata items
  - TFLiteModel / TFLiteWeights / SP_Tokenizer / GenericBinaryData / unknown types: raw bytes identical
  - HF_Tokenizer_Zlib: uint64-LE size prefix + zlib; decoded bytes identical (raw identity recorded)
  - ExecutorMetadataProto / EmbeddingMetadataProto: parsed content identical
  - system metadata entries other than uuid / creation_timestamp unchanged
  - IN's sha256 unchanged after the run
Every pack writes a new uuid and creation_timestamp and the builder's header format version:
recorded, not failed. Exit 1 = a check failed (OUT is kept for inspection, not verified); exit 2 =
refused before writing anything. Nothing is written in IN's directory; OUT, REPORT and REPORT's
.evidence directory are never overwritten; the unpack scratch directory is always removed.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
import re
import shlex
import shutil
import struct
import subprocess
import sys
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
from litert_lm_builder.runtime.proto import embedding_metadata_pb2, executor_metadata_pb2, llm_metadata_pb2

CHUNK = 4 << 20
DEFAULT_CLI = 'uvx --from litert-lm==0.18.0 litert-lm'
PROTO_CLASSES = {'LlmMetadataProto': llm_metadata_pb2.LlmMetadata,
                 'ExecutorMetadataProto': executor_metadata_pb2.ExecutorMetadata,
                 'EmbeddingMetadataProto': embedding_metadata_pb2.EmbeddingMetadata}
RAW_ONLY_NOTE = 'raw bytes'
VOLATILE_SYSTEM_KEYS = ('uuid', 'creation_timestamp')


def _r(x, n):
    """x to n decimal places, as a float."""
    return float(f"{x:.{n}f}")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(CHUNK), b''):
            h.update(chunk)
    return h.hexdigest()


def canon(msg):
    """Byte form used for equality: deterministic serialization keeps unknown fields too."""
    return msg.SerializeToString(deterministic=True)


def unknown_field_count(msg):
    try:
        from google.protobuf import unknown_fields
        return len(unknown_fields.UnknownFieldSet(msg))
    except Exception:  # noqa: BLE001 - older protobuf: report as not measured
        return None


def read_section(path, section):
    with open(path, 'rb') as stream:
        stream.seek(section['begin'])
        data = stream.read(section['bytes'])
    if len(data) != section['bytes']:
        raise IOError('Short section read')
    return data


def decode_hf_tokenizer(data):
    # The builder writes the decoded size as uint64 little-endian, then the zlib stream.
    if len(data) < 8:
        raise ValueError('HF tokenizer section lacks its 8-byte size prefix')
    declared = int.from_bytes(data[:8], 'little')
    decoded = zlib.decompress(data[8:])
    if len(decoded) != declared:
        raise ValueError('HF tokenizer decoded size differs from its prefix')
    return decoded


def bundle_sections(path):
    """Header offsets with bounds checks, ordered section labels and system metadata."""
    size = os.path.getsize(path)
    with open(path, 'rb') as stream:
        prefix = stream.read(32)
        if prefix[:8] != litertlm_core.HEADER_MAGIC_BYTES:
            raise ValueError(f'Not a .litertlm bundle: {path}')
        header_end = int.from_bytes(prefix[litertlm_core.HEADER_END_LOCATION_BYTE_OFFSET:
                                           litertlm_core.HEADER_END_LOCATION_BYTE_OFFSET + 8], 'little')
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
        sections.append({'index': i, 'type': kind, 'data_type': section.DataType(),
                         'begin': begin, 'end': end, 'bytes': end - begin,
                         'metadata': [litertlm_peek.kvp_to_dict(section.Items(j)) for j in range(section.ItemsLength())]})
    system = metadata.SystemMetadata()
    system_items = [litertlm_peek.kvp_to_dict(system.Entries(i)) for i in range(system.EntriesLength())] if system else []
    return {'file_bytes': size, 'format_version': list(struct.unpack('<III', prefix[8:20])),
            'header_end': header_end, 'header_sha256': digest(head),
            'system_metadata': system_items, 'sections': sections}, head


def compare_raw_sections(src, dst, a, b):
    """Streaming byte comparison plus an independent sha256 of each side."""
    ha, hb = hashlib.sha256(), hashlib.sha256()
    equal = a['bytes'] == b['bytes']
    with open(src, 'rb') as fa, open(dst, 'rb') as fb:
        fa.seek(a['begin']); fb.seek(b['begin'])
        left_a, left_b = a['bytes'], b['bytes']
        while left_a or left_b:
            ca, cb = fa.read(min(CHUNK, left_a)), fb.read(min(CHUNK, left_b))
            if (left_a and not ca) or (left_b and not cb):
                raise IOError('Short read during section comparison')
            ha.update(ca); hb.update(cb)
            if ca != cb:
                equal = False
            left_a -= len(ca); left_b -= len(cb)
    return equal, ha.hexdigest(), hb.hexdigest()


def field_state(msg, field):
    """Comparable value of one top-level field: presence matters for optional scalars and messages."""
    value = getattr(msg, field.name)
    if field.is_repeated:
        return ('repeated', [canon(v) if hasattr(v, 'SerializeToString') else v for v in value])
    present = msg.HasField(field.name) if field.has_presence else None
    if field.message_type is not None:
        return ('message', present, canon(value) if present else None)
    return ('scalar', present, value)


def describe_value(msg, field):
    """Human-readable summary of one field for the report."""
    value = getattr(msg, field.name)
    if field.is_repeated:
        items = [text_format.MessageToString(v, as_one_line=True, as_utf8=True) if hasattr(v, 'ListFields') else repr(v)
                 for v in value]
        return {'count': len(items), 'items': items}
    if field.has_presence and not msg.HasField(field.name):
        return 'unset'
    if field.name == 'jinja_prompt_template':
        data = value.encode('utf-8')
        return {'bytes': len(data), 'sha256': digest(data)}
    if field.message_type is not None:
        text = text_format.MessageToString(value, as_one_line=True, as_utf8=True)
        return text if len(text) <= 600 else text[:600] + ' ...'
    return value


def changed_fields(before, after):
    rows = []
    for field in sorted(llm_metadata_pb2.LlmMetadata.DESCRIPTOR.fields, key=lambda f: f.number):
        if field_state(before, field) != field_state(after, field):
            rows.append({'field': field.name, 'number': field.number,
                         'before': describe_value(before, field), 'after': describe_value(after, field)})
    return rows


def parse_pbtext(text):
    return text_format.Parse(text, llm_metadata_pb2.LlmMetadata())


def same_dir(a, b):
    return Path(a).resolve() == Path(b).resolve()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('source', help='input .litertlm (read only)')
    parser.add_argument('output', help='output .litertlm (must not exist)')
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--pbtext', help='the complete new LlmMetadataProto.pbtext')
    mode.add_argument('--identity', action='store_true', help='self-test: pack the unpacked pbtext unchanged')
    parser.add_argument('--report', required=True, help='JSON report path (must not exist); evidence goes to REPORT.evidence/')
    parser.add_argument('--expect-changed', default=None,
                        help='comma-separated LlmMetadata field names that must differ, and nothing else ("" = none)')
    parser.add_argument('--cli', default=DEFAULT_CLI, help=f'litert-lm CLI command prefix (default: {DEFAULT_CLI})')
    parser.add_argument('--scratch-parent', default=None, help='where the temporary unpack dir goes (default: OUT\'s directory)')
    parser.add_argument('--timeout-seconds', type=int, default=1200, help='hard timeout per CLI call')
    args = parser.parse_args()

    src_given = Path(args.source).expanduser().absolute()
    src = src_given.resolve()
    dst = Path(args.output).expanduser().absolute()
    report = Path(args.report).expanduser().absolute()
    evidence = report.with_suffix('.evidence')
    new_pbtext = Path(args.pbtext).expanduser().absolute() if args.pbtext else None
    scratch_parent = Path(args.scratch_parent).expanduser().absolute() if args.scratch_parent else dst.parent

    if args.identity:
        if args.expect_changed not in (None, ''):
            parser.error('--identity expects no changed field')
        expected = []
    else:
        if args.expect_changed is None:
            parser.error('--expect-changed is required with --pbtext')
        expected = [x.strip() for x in args.expect_changed.split(',') if x.strip()]
    known = {f.name for f in llm_metadata_pb2.LlmMetadata.DESCRIPTOR.fields}
    unknown_names = sorted(set(expected) - known)
    if unknown_names:
        parser.error(f'not LlmMetadata fields: {unknown_names}')
    if src.suffix != '.litertlm' or dst.suffix != '.litertlm':
        parser.error('IN and OUT must be .litertlm files')
    if not src.is_file():
        parser.error(f'input missing: {src_given}')
    if new_pbtext is not None and not new_pbtext.is_file():
        parser.error(f'pbtext missing: {new_pbtext}')
    for path in (dst, report, evidence):
        if path.exists():
            parser.error(f'refusing to overwrite {path}')
    for path in (dst.parent, report.parent, scratch_parent):
        if same_dir(path, src_given.parent) or same_dir(path, src.parent):
            parser.error(f'refusing to write next to the input: {path}')
    if args.timeout_seconds <= 0:
        parser.error('timeout must be positive')
    cli = shlex.split(args.cli)

    dst.parent.mkdir(parents=True, exist_ok=True)
    report.parent.mkdir(parents=True, exist_ok=True)
    scratch_parent.mkdir(parents=True, exist_ok=True)
    evidence.mkdir(parents=True)
    scratch = Path(tempfile.mkdtemp(prefix='.repack_scratch_', dir=scratch_parent))
    started = time.monotonic()
    record = {'tool': 'repack_metadata.py', 'status': 'RUNNING',
              'source': str(src_given), 'source_realpath': str(src), 'output': str(dst),
              'mode': 'identity' if args.identity else 'pbtext',
              'new_pbtext': str(new_pbtext) if new_pbtext else None,
              'expected_changed_fields': expected,
              'start_utc': datetime.now(timezone.utc).isoformat(),
              'verifier': {'python': sys.version.split()[0],
                           'litert_lm_builder': importlib.metadata.version('litert-lm-builder'),
                           'protobuf': importlib.metadata.version('protobuf')},
              'cli': args.cli, 'scratch': str(scratch), 'evidence': str(evidence),
              'timings_seconds': {}, 'commands': [], 'checks': {}}

    def save():
        report.write_text(json.dumps(record, indent=2, ensure_ascii=False) + '\n')

    def run(name, argv):
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1', HF_HUB_OFFLINE='1',
                   TMPDIR=str(scratch / 'tmp'))
        (scratch / 'tmp').mkdir(exist_ok=True)
        log = evidence / f'{name}.log'
        begin = time.monotonic()
        with log.open('wb') as stream:
            try:
                result = subprocess.run([*cli, *argv], stdin=subprocess.DEVNULL, stdout=stream,
                                        stderr=subprocess.STDOUT, env=env, timeout=args.timeout_seconds)
                code = result.returncode
            except subprocess.TimeoutExpired:
                code = 'timeout'
        record['timings_seconds'][name] = _r(time.monotonic() - begin, 2)
        record['commands'].append({'name': name, 'argv': [*cli, *argv], 'stdin': '/dev/null',
                                   'exit_code': code, 'log': str(log)})
        save()
        text = log.read_text(errors='replace')
        # The CLI prints its errors and still exits 0: require the success line too.
        if code != 0 or re.search(r'\bError\b', text) or 'Traceback (most recent call last)' in text \
                or 'Successfully' not in text:
            raise RuntimeError(f'{name} failed (exit {code}); see {log}')
        return text

    try:
        begin = time.monotonic()
        original, head_a = bundle_sections(src)
        if sum(s['type'] == 'LlmMetadataProto' for s in original['sections']) != 1:
            raise ValueError('Exactly one LlmMetadataProto section is required')
        record['original_size_bytes'] = original['file_bytes']
        record['original_sha256'] = file_sha(src)
        record['timings_seconds']['input_hash_and_header'] = _r(time.monotonic() - begin, 2)
        version = subprocess.run([*cli, '--version'], stdin=subprocess.DEVNULL, capture_output=True, text=True,
                                 timeout=args.timeout_seconds)
        record['cli_version'] = (version.stdout + version.stderr).strip().splitlines()[-1:] or ['?']
        save()

        unpacked = scratch / 'unpacked'
        run('unpack', ['unpack', str(src_given), '--output-dir', str(unpacked)])
        pbtext_path = unpacked / 'LlmMetadataProto.pbtext'
        toml_path = unpacked / 'model.toml'
        if not pbtext_path.is_file() or not toml_path.is_file():
            raise RuntimeError('unpack did not write LlmMetadataProto.pbtext and model.toml')
        toml_before = toml_path.read_bytes()
        (evidence / 'model.toml').write_bytes(toml_before)

        # 1. The text form the edit starts from must be the input's metadata, field for field.
        begin = time.monotonic()
        llm_section = next(s for s in original['sections'] if s['type'] == 'LlmMetadataProto')
        original_bin = read_section(src, llm_section)
        original_msg = llm_metadata_pb2.LlmMetadata()
        original_msg.ParseFromString(original_bin)
        before_text = pbtext_path.read_text()
        before_msg = parse_pbtext(before_text)
        (evidence / 'original.LlmMetadataProto.pbtext').write_text(before_text, newline='')
        (evidence / 'original.LlmMetadataProto.bin').write_bytes(original_bin)
        record['checks']['unpacked_pbtext_equals_input_metadata'] = canon(before_msg) == canon(original_msg)
        record['checks']['input_metadata_unknown_fields'] = unknown_field_count(original_msg)

        # 2. Put the new metadata in place.
        if args.identity:
            after_text = before_text
        else:
            after_text = new_pbtext.read_text()
            pbtext_path.write_text(after_text, newline='')
        after_msg = parse_pbtext(after_text)
        (evidence / 'new.LlmMetadataProto.pbtext').write_text(after_text, newline='')
        record['pbtext'] = {'original_sha256': digest(before_text.encode()), 'new_sha256': digest(after_text.encode()),
                            'original_bytes': len(before_text.encode()), 'new_bytes': len(after_text.encode())}
        diff = changed_fields(before_msg, after_msg)
        record['changed_fields'] = diff
        names = [row['field'] for row in diff]
        record['checks']['changed_fields_equal_expected'] = sorted(names) == sorted(expected)
        record['checks']['unexpected_changes'] = sorted(set(names) - set(expected))
        record['checks']['expected_but_unchanged'] = sorted(set(expected) - set(names))
        record['timings_seconds']['edit_and_parse'] = _r(time.monotonic() - begin, 2)
        save()

        # 3. Pack.
        run('pack', ['pack', str(unpacked), '--output', str(dst)])
        if not dst.is_file():
            raise RuntimeError('pack wrote no output bundle')
        record['checks']['model_toml_untouched_by_edit'] = toml_path.read_bytes() == toml_before

        # 4. Verify section by section.
        begin = time.monotonic()
        new, head_b = bundle_sections(dst)
        (evidence / 'original.header.bin').write_bytes(head_a)
        (evidence / 'new.header.bin').write_bytes(head_b)
        failures = []
        type_list_same = [s['data_type'] for s in original['sections']] == [s['data_type'] for s in new['sections']]
        record['checks']['section_list_unchanged'] = type_list_same
        if not type_list_same:
            failures.append('section count/order/type changed')
        rows = []
        if type_list_same:
            for sa, sb in zip(original['sections'], new['sections']):
                same, ha, hb = compare_raw_sections(src, dst, sa, sb)
                kind = sa['type']
                row = {'index': sa['index'], 'type': kind, 'original_bytes': sa['bytes'], 'new_bytes': sb['bytes'],
                       'original_sha256': ha, 'new_sha256': hb, 'raw_bytes_identical': same,
                       'section_metadata': sa['metadata'],
                       'section_metadata_unchanged': sa['metadata'] == sb['metadata']}
                if kind == 'LlmMetadataProto':
                    new_bin = read_section(dst, sb)
                    (evidence / 'new.LlmMetadataProto.bin').write_bytes(new_bin)
                    packed = llm_metadata_pb2.LlmMetadata()
                    packed.ParseFromString(new_bin)
                    packed_names = [r['field'] for r in changed_fields(original_msg, packed)]
                    row.update(comparison='parsed: packed == requested pbtext; input vs packed differ only in the expected fields',
                               packed_equals_requested=canon(packed) == canon(after_msg),
                               changed_fields_input_vs_packed=packed_names,
                               PASS=canon(packed) == canon(after_msg) and sorted(packed_names) == sorted(expected)
                               and record['checks']['unpacked_pbtext_equals_input_metadata'])
                elif kind in PROTO_CLASSES:
                    old_msg, new_msg = PROTO_CLASSES[kind](), PROTO_CLASSES[kind]()
                    old_msg.ParseFromString(read_section(src, sa)); new_msg.ParseFromString(read_section(dst, sb))
                    row.update(comparison='parsed protobuf content', decoded_content_identical=canon(old_msg) == canon(new_msg),
                               PASS=canon(old_msg) == canon(new_msg))
                elif kind == 'HF_Tokenizer_Zlib':
                    da, db = decode_hf_tokenizer(read_section(src, sa)), decode_hf_tokenizer(read_section(dst, sb))
                    row.update(comparison='uint64-LE size prefix + zlib-decoded bytes', decoded_content_identical=da == db,
                               original_decoded_bytes=len(da), new_decoded_bytes=len(db),
                               original_decoded_sha256=digest(da), new_decoded_sha256=digest(db), PASS=da == db)
                else:
                    row.update(comparison=RAW_ONLY_NOTE, PASS=same)
                row['PASS'] = bool(row['PASS'] and row['section_metadata_unchanged'])
                if not row['PASS']:
                    failures.append(f'section {sa["index"]} {kind}: content or section metadata mismatch')
                rows.append(row)
        record['sections'] = rows
        record['checks']['tflite_sections_byte_identical'] = all(
            r['raw_bytes_identical'] for r in rows if r['type'] in ('TFLiteModel', 'TFLiteWeights'))
        if not record['checks']['unpacked_pbtext_equals_input_metadata']:
            failures.append('unpacked pbtext does not reproduce the input LlmMetadata')
        if not record['checks']['changed_fields_equal_expected']:
            failures.append(f'changed fields {sorted(names)} != expected {sorted(expected)}')
        if not record['checks']['model_toml_untouched_by_edit']:
            failures.append('model.toml changed between unpack and pack')

        def stable(items):
            return [m for m in items if m['key'] not in VOLATILE_SYSTEM_KEYS]
        record['system_metadata'] = {'original': original['system_metadata'], 'new': new['system_metadata']}
        record['checks']['system_metadata_unchanged_except_uuid_timestamp'] = \
            stable(original['system_metadata']) == stable(new['system_metadata'])
        if not record['checks']['system_metadata_unchanged_except_uuid_timestamp']:
            failures.append('system metadata changed beyond uuid / creation_timestamp')
        record['format_version'] = {'original': original['format_version'], 'new': new['format_version']}
        record['header_bytes'] = {'original': original['header_end'], 'new': new['header_end']}
        record['new_size_bytes'] = new['file_bytes']
        record['new_sha256'] = file_sha(dst)
        record['input_sha256_after'] = file_sha(src)
        record['checks']['input_unchanged'] = record['input_sha256_after'] == record['original_sha256']
        if not record['checks']['input_unchanged']:
            failures.append('input file changed during the run')
        record['timings_seconds']['verification_and_hashes'] = _r(time.monotonic() - begin, 2)
        record['failures'] = failures
        record['status'] = 'PASS' if not failures else 'FAIL'

        print(f'{record["status"]}: {src_given.name} {record["original_size_bytes"]} B -> {dst} {record["new_size_bytes"]} B')
        print(f'  input  sha256 {record["original_sha256"]} (after the run: {record["input_sha256_after"]})')
        print(f'  output sha256 {record["new_sha256"]}')
        print(f'  changed fields: {names or "none"} (expected: {expected or "none"})')
        for row in rows:
            print(f'  [{row["index"]}] {row["type"]}: {row["original_bytes"]} -> {row["new_bytes"]} B, '
                  f'raw identical {row["raw_bytes_identical"]}, {row["comparison"]}: PASS={row["PASS"]}')
        for failure in failures:
            print(f'  FAIL: {failure}')
        return 0 if not failures else 1
    except Exception as exc:  # noqa: BLE001 - every error lands in the report
        record['status'] = 'ERROR'
        record['error'] = {'type': type(exc).__name__, 'message': str(exc), 'traceback': traceback.format_exc()}
        print(f'ERROR: {type(exc).__name__}: {exc}', file=sys.stderr)
        return 1
    finally:
        begin = time.monotonic()
        shutil.rmtree(scratch, ignore_errors=True)
        record['scratch_removed'] = not scratch.exists()
        record['timings_seconds']['scratch_cleanup'] = _r(time.monotonic() - begin, 2)
        record['wall_seconds'] = _r(time.monotonic() - started, 2)
        record['end_utc'] = datetime.now(timezone.utc).isoformat()
        save()


if __name__ == '__main__':
    raise SystemExit(main())
