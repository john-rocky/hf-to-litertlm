"""clef-flash on LiteRT: the host in plain Python.

Dependencies: NumPy, `tokenizers` and Pillow, and ai-edge-litert for the graphs (requirements-host.txt; the
safetensors format is read here, the `safetensors` library only checks the table files). No PyTorch, no transformers
and none of the provider's code at run time. The host answers a request of the SystemOne-compatible request shape the way
the checkpoint's own `systemone()` does (Cloudflare/clef-flash at f0b7d3cd, joint_schema_model.py sha256 0e304cf7...),
around two kinds of LiteRT graph that are the caller's: a row graph per length (the Qwen3.5-9B text stack, embeds in,
final-norm hidden states out) and a tower graph per picture tile. The vocabulary tables and the joint schema head stay
on the host.

1. Request: {"model": str, "state": any JSON value, "questions": {id: {"type": "noul" | "choice" | "score",
   "instructions": ..., "criteria": ...}}, "images": [one picture] (optional)}. Choice criteria are an object
   {option id: description}, score criteria a list of level descriptions (lowest first), noul criteria an optional
   object overriding the "true" / "false" descriptions. Checked as `systemone()` checks it (request.validate_request;
   the named refusals of errors.py carry the author's messages).
2. Ids (request.encode_record): the author's encode_record on the repository's tokenizer.json, every piece tokenized
   on its own without special tokens; with a picture, its block (<|vision_start|>, N x <|image_pad|>, <|vision_end|>,
   "\\n") right after the 36-id prefix. usage.input_tokens = the number of ids.
3. Picture (pixels.py): one per request, read at a fixed square tile, 448 (default, 14 x 14 = 196 picture tokens) or
   256 (8 x 8 = 64): RGB, Pillow BICUBIC to the tile, (x/255 - 0.5) / 0.5, merge-block-major patches [4 G^2, 1536]
   -> the tower graph of that tile -> [G^2, 4096] picture rows. More pictures, videos and media_kwargs are refused
   (`UnsupportedMedia`); the author's native resolution is not a tower graph.
4. Row graph inputs (host.ClefHost.row_inputs, at L = the smallest of 256 / 512 / 1024 / 2048 / 4096 that holds the
   record; longer records are refused, `RequestTooLong`, never cut):
       embeds float32 [1, L, 4096]  the bf16 embed_tokens rows of the ids widened to float32, the picture rows at the
                                    <|image_pad|> positions, zeros after the record
       valid  float32 [1, L]        1 on the record, 0 after it
       cos    float32 [1, L, 64]    the M-RoPE tables of the record's three position planes (positions.py)
       sin    float32 [1, L, 64]
   -> hidden float32 [1, L, 4096] (every position, after the final RMSNorm); the host reads the first T rows.
5. Head (head.py): the author's JointSchemaHead in NumPy float32 on those T rows, the ids, the spans and the bf16
   lm_head rows of the option spans -> one logit per option.
6. Response (response.py): per question the float32 softmax, then the author's systemone_answer ->
   {"model", "answers": {id: answer}, "usage": {"input_tokens", "output_tokens": 0}}.

Files: tokenizer.json (the repository's, sha256 06b95093...), embed_tokens.safetensors and lm_head.safetensors (one BF16
[248320, 4096] tensor each, the checkpoint's bytes, no metadata), joint_head.safetensors and joint_head_config.json
(the checkpoint's own). contract.json next to this package records names, shapes, files and checks.

The graphs (runtime.py, ai-edge-litert's CompiledModel; imported only when a graph is compiled):
    with ClefLiteRT.from_dir("<folder>", accelerator="gpu") as clef:     # "gpu" = float32 precision, "cpu" = XNNPACK
        body = clef.decide(request, tile=448)"""
from .errors import (BadQuestionType, EmptyCriteria, EmptyRecord, MissingModelOrState, NoQuestions, RequestError,
                     RequestTooLong, SchemaTooLong, UnsupportedMedia)
from .head import JointHead
from .host import FILES, HIDDEN, ClefHost, Encoded
from .pixels import TILES, tile_patches
from .positions import BUCKETS, mrope_tables, padded_positions, pick_length, record_positions
from .request import EncodedQuestion, EncodedRecord, encode_record, question_options, render, validate_request
from .response import question_probs, response, systemone_answer
from .runtime import FOLDER, ClefGraphs, ClefLiteRT, LiteRTGraph, load_f32_tensor
from .tables import Bf16Table, copy_bf16_tensor, load_bf16_tensors

__version__ = "0.2.0"
__all__ = [
    "BUCKETS", "FILES", "FOLDER", "HIDDEN", "TILES", "BadQuestionType", "Bf16Table", "ClefGraphs", "ClefHost",
    "ClefLiteRT", "EmptyCriteria", "EmptyRecord", "Encoded", "EncodedQuestion", "EncodedRecord", "JointHead",
    "LiteRTGraph", "MissingModelOrState", "NoQuestions", "RequestError", "RequestTooLong", "SchemaTooLong",
    "UnsupportedMedia", "copy_bf16_tensor", "encode_record", "load_bf16_tensors", "load_f32_tensor", "mrope_tables",
    "padded_positions", "pick_length", "question_options", "question_probs", "record_positions", "render", "response",
    "systemone_answer", "tile_patches", "validate_request",
]
