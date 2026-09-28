"""Read the sections of a .litertlm bundle without the LiteRT-LM package, and keep them in a local cache.

The file layout is the one LiteRT-LM reads (schema/core/litertlm_header_schema.fbs, litertlm_read.cc):
  bytes 0..7    b'LITERTLM'
  bytes 8..19   major, minor, patch version (uint32 little endian); major must be 1
  bytes 20..23  padding
  bytes 24..31  header end offset (uint64)
  bytes 32..end a FlatBuffer LiteRTLMMetaData: section_metadata.objects[i] = (items, begin_offset, end_offset, data_type)
The TFLite sections are copied out byte for byte; the HF tokenizer section (8-byte uncompressed size + zlib stream)
is decompressed to tokenizer.json. The cache folder is named by the bundle's SHA-256 and is written once.
"""
import hashlib
import json
import os
import struct
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE = ROOT / '.cache/readout'
DATA_TYPES = {0: 'NONE', 1: 'GenericBinaryData', 2: 'Deprecated', 3: 'TFLiteModel', 4: 'SP_Tokenizer',
              5: 'LlmMetadataProto', 6: 'HF_Tokenizer_Zlib', 7: 'TFLiteWeights', 8: 'EmbeddingMetadataProto',
              9: 'ExecutorMetadataProto'}
STRING_VALUE = 9                        # VData union index of StringValue


def sha256(path):
    with open(path, 'rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


class _Table:
    """Minimal FlatBuffer table reader (little endian, offsets relative to the buffer start)."""

    def __init__(self, buf, pos):
        self.buf, self.pos = buf, pos
        vtable = pos - struct.unpack_from('<i', buf, pos)[0]
        self.vt_len = struct.unpack_from('<H', buf, vtable)[0]
        self.vtable = vtable

    def _field(self, index):
        entry = 4 + 2 * index
        if entry >= self.vt_len:
            return 0
        return struct.unpack_from('<H', self.buf, self.vtable + entry)[0]

    def scalar(self, index, fmt, default=0):
        off = self._field(index)
        return default if off == 0 else struct.unpack_from('<' + fmt, self.buf, self.pos + off)[0]

    def _indirect(self, index):
        off = self._field(index)
        if off == 0:
            return None
        at = self.pos + off
        return at + struct.unpack_from('<I', self.buf, at)[0]

    def table(self, index):
        at = self._indirect(index)
        return None if at is None else _Table(self.buf, at)

    def string(self, index):
        at = self._indirect(index)
        if at is None:
            return None
        n = struct.unpack_from('<I', self.buf, at)[0]
        return bytes(self.buf[at + 4:at + 4 + n]).decode('utf-8')

    def tables(self, index):
        at = self._indirect(index)
        if at is None:
            return []
        n = struct.unpack_from('<I', self.buf, at)[0]
        out = []
        for i in range(n):
            el = at + 4 + 4 * i
            out.append(_Table(self.buf, el + struct.unpack_from('<I', self.buf, el)[0]))
        return out


def _string_items(obj):
    items = {}
    for kv in obj.tables(0):                                   # SectionObject.items: [KeyValuePair]
        key = kv.string(0)
        if kv.scalar(1, 'B') == STRING_VALUE:                  # KeyValuePair.value_type
            items[key] = kv.table(2).string(0)                 # StringValue.value
    return items


def read_sections(bundle):
    """[(index, data_type, begin, end, string items)] from the bundle header."""
    with open(bundle, 'rb') as f:
        head = f.read(32)
        if head[:8] != b'LITERTLM':
            raise ValueError(f'{bundle}: not a .litertlm file')
        major, minor, patch = struct.unpack_from('<III', head, 8)
        if major != 1:
            raise ValueError(f'{bundle}: unsupported .litertlm major version {major}')
        header_end = struct.unpack_from('<Q', head, 24)[0]
        buf = f.read(header_end - 32)
    root = _Table(buf, struct.unpack_from('<I', buf, 0)[0])    # LiteRTLMMetaData
    objects = root.table(1).tables(0)                          # section_metadata.objects
    return (major, minor, patch), [
        (i, DATA_TYPES.get(o.scalar(3, 'B'), 'UNKNOWN'), o.scalar(1, 'Q'), o.scalar(2, 'Q'), _string_items(o))
        for i, o in enumerate(objects)]


def _copy_range(src, dst, begin, end, chunk=64 << 20):
    with open(src, 'rb') as f, open(dst, 'wb') as g:
        f.seek(begin)
        left = end - begin
        while left:
            data = f.read(min(chunk, left))
            if not data:
                raise IOError(f'{src}: truncated section')
            g.write(data)
            left -= len(data)


def read_tokenizer_json(bundle):
    """The bundle's HF tokenizer.json as text, read straight from its section (no cache folder)."""
    _, sections = read_sections(bundle)
    (_, _, begin, end, _), = [s for s in sections if s[1] == 'HF_Tokenizer_Zlib']
    with open(bundle, 'rb') as f:
        f.seek(begin)
        raw = f.read(end - begin)
    data = zlib.decompress(raw[8:])
    assert len(data) == struct.unpack_from('<Q', raw, 0)[0]
    return data.decode('utf-8')


def unpack_bundle(bundle, cache_dir=None):
    """Extract once; returns (folder, {'embedder'|'prefill_decode'|'vision_encoder'|'vision_adapter': path,
    'tokenizer': path}, record)."""
    bundle = Path(bundle).resolve()
    digest = sha256(bundle)
    folder = Path(cache_dir or os.environ.get('DECIDER_LITERT_CACHE') or DEFAULT_CACHE) / digest
    marker = folder / 'complete.json'
    if not marker.exists():
        if folder.exists() and any(folder.iterdir()):
            raise RuntimeError(f'Incomplete bundle cache, remove it and retry: {folder}')
        folder.mkdir(parents=True, exist_ok=True)
        version, sections = read_sections(bundle)
        files = {}
        for i, kind, begin, end, items in sections:
            if kind == 'TFLiteModel':
                name = items.get('model_type', f'section{i}').removeprefix('tf_lite_')
                path = folder / f'{name}.tflite'
                _copy_range(bundle, path, begin, end)
            elif kind == 'HF_Tokenizer_Zlib':
                name, path = 'tokenizer', folder / 'tokenizer.json'
                with open(bundle, 'rb') as f:
                    f.seek(begin)
                    raw = f.read(end - begin)
                size = struct.unpack_from('<Q', raw, 0)[0]
                data = zlib.decompress(raw[8:])
                assert len(data) == size, (len(data), size)
                path.write_bytes(data)
            else:
                continue
            files[name] = dict(file=path.name, section=i, data_type=kind, begin=begin, end=end, bytes=path.stat().st_size,
                               sha256=sha256(path))
        record = dict(bundle=bundle.name, bundle_bytes=bundle.stat().st_size, bundle_sha256=digest,
                      litertlm_version='.'.join(map(str, version)), files=files)
        marker.write_text(json.dumps(record, indent=1) + '\n')
    record = json.loads(marker.read_text())
    paths = {k: folder / v['file'] for k, v in record['files'].items()}
    missing = {'embedder', 'prefill_decode', 'vision_encoder', 'vision_adapter', 'tokenizer'} - set(paths)
    if missing:
        raise ValueError(f'{bundle.name}: sections missing: {sorted(missing)}')
    return folder, paths, record


if __name__ == '__main__':
    import sys
    version, sections = read_sections(sys.argv[1])
    print('litertlm', '.'.join(map(str, version)))
    for i, kind, begin, end, items in sections:
        print(i, kind, begin, end, end - begin, items)
