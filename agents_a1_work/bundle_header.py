"""Read local bundle sections, read without loading model weights."""
from pathlib import Path
from litert_lm_builder import litertlm_core
from litert_lm_builder import litertlm_header_schema_py_generated as schema
from litert_lm_builder.runtime.proto import llm_metadata_pb2

def read_header(path):
    with Path(path).open("rb") as stream:
        head = stream.read(4096)
        assert head[:8] == litertlm_core.HEADER_MAGIC_BYTES
        loc = litertlm_core.HEADER_END_LOCATION_BYTE_OFFSET
        end = int.from_bytes(head[loc:loc + 8], "little")
        if end > len(head):
            stream.seek(0)
            head = stream.read(end + 65)
        meta = schema.LiteRTLMMetaData.GetRootAs(bytearray(head[litertlm_core.HEADER_BEGIN_BYTE_OFFSET:end]), 0)
        sections, llm = [], None
        for i in range(meta.SectionMetadata().ObjectsLength()):
            section = meta.SectionMetadata().Objects(i)
            typ = litertlm_core.any_section_data_type_to_string(section.DataType())
            row = {"type": typ, "begin": section.BeginOffset(), "end": section.EndOffset(), "size_bytes": section.EndOffset() - section.BeginOffset(), "items": {}}
            for j in range(section.ItemsLength()):
                item = section.Items(j)
                key = item.Key().decode() if item.Key() else None
                value = item.Value()
                if value is not None and item.ValueType() == schema.VData.StringValue:
                    val = schema.StringValue()
                    val.Init(value.Bytes, value.Pos)
                    row["items"][key] = val.Value().decode() if val.Value() else ""
            sections.append(row)
            if typ == "LlmMetadataProto":
                stream.seek(section.BeginOffset())
                llm = llm_metadata_pb2.LlmMetadata()
                llm.ParseFromString(stream.read(row["size_bytes"]))
        return sections, llm
