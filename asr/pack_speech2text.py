#!/usr/bin/env python3
"""Pack a standalone ASR .tflite, its tokenizer and a Speech2TextMetadata into one .litertlm (the form of litert-torch
PR #1426's package_speech2text_model).

    uv run --frozen --project asr python asr/pack_speech2text.py <asr_model.tflite> <tokenizer.json|.model>
        <speech2text_metadata.json> <out.litertlm> [--model-type encoder_decoder|audio_encoder]
        [--uuid U --timestamp T]

Runs in the asr/ environment (litert-lm-builder 0.19.0: 0.18.0 has no Speech2TextMetadata section). Sections, in
order: the Speech2TextMetadata proto (from the JSON, json_format.ParseDict), the tokenizer (a tokenizer.json as
HF_Tokenizer_Zlib, else SentencePiece), the .tflite with model_type tf_lite_encoder_decoder (encode + decode
signatures) or tf_lite_audio_encoder (an encoder-only CTC graph). System metadata: Authors "" (this repo's hand-built
bundles; PR #1426 writes "ODML", which the runtime does not read) and, with --uuid / --timestamp, a fixed uuid and
creation_timestamp instead of the builder's uuid4() / now(), so a rebuild gives the same bytes.
Prints one JSON line: the output path, its sha256 and each section's data type.
"""
import argparse
import hashlib
import json
import sys

from google.protobuf import json_format
from litert_lm_builder import litertlm_builder as lb
from litert_lm_builder.runtime.proto import speech2text_metadata_pb2 as s2t

MODEL_TYPES = {"encoder_decoder": s2t.Speech2TextMetadata.TF_LITE_ENCODER_DECODER,
               "audio_encoder": s2t.Speech2TextMetadata.TF_LITE_AUDIO_ENCODER}


def pin_system_metadata(uuid_value, timestamp):
    """litertlm_builder.populate_system_metadata appends uuid4() and now(); replace them in this process."""
    def populate_system_metadata(system_metadata):
        kept = [m for m in system_metadata if m.key not in ("uuid", "creation_timestamp")]
        return kept + [lb.Metadata(key="uuid", value=uuid_value, dtype=lb.DType.STRING),
                       lb.Metadata(key="creation_timestamp", value=timestamp, dtype=lb.DType.STRING)]
    lb.populate_system_metadata = populate_system_metadata


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("tflite")
    ap.add_argument("tokenizer")
    ap.add_argument("metadata_json")
    ap.add_argument("out")
    ap.add_argument("--model-type", choices=sorted(MODEL_TYPES), default="encoder_decoder")
    ap.add_argument("--uuid")
    ap.add_argument("--timestamp")
    a = ap.parse_args()
    if bool(a.uuid) != bool(a.timestamp):
        sys.exit("FAIL bundle: --uuid and --timestamp go together -> next: pass both or neither")
    if a.uuid:
        pin_system_metadata(a.uuid, a.timestamp)

    meta = json_format.ParseDict(json.load(open(a.metadata_json)), s2t.Speech2TextMetadata())
    pb_path = a.out + ".speech2text_metadata.pb"
    with open(pb_path, "wb") as f:
        f.write(meta.SerializeToString())

    b = lb.LitertLmFileBuilder()
    b.add_system_metadata(lb.Metadata(key="Authors", value="", dtype=lb.DType.STRING))
    b.add_speech2text_metadata(pb_path)
    if a.tokenizer.endswith(".json"):
        b.add_hf_tokenizer(a.tokenizer)
    else:
        b.add_sentencepiece_tokenizer(a.tokenizer)
    b.add_tflite_model(a.tflite, MODEL_TYPES[a.model_type])
    with open(a.out, "wb") as f:
        b.build(f)

    h = hashlib.sha256()
    with open(a.out, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    import importlib.metadata as md
    import platform
    pkgs = {d.metadata["Name"].lower(): d.version for d in md.distributions() if d.metadata["Name"]}
    print(json.dumps({"out": a.out, "sha256": h.hexdigest(), "metadata": json_format.MessageToDict(meta),
                      "model_type": a.model_type, "tokenizer": a.tokenizer,
                      "toolchain": {"python": platform.python_version(), "packages": dict(sorted(pkgs.items()))}}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
