"""Export-time (native, PT2E) dynamic int8 for the codec decoder -- the A/B partner of the post-hoc
DRQ file (house rule 2026-07-22: conv-heavy graphs get both and the numbers decide).

  T=64 ~/venvs/lt094dev/bin/python3 audio8_tts_work/export_codec_native_i8.py
"""
import os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
import arktts_port as P
import litert_torch
from litert_torch.quantize import pt2e_quantizer
from litert_torch.quantize.quant_config import QuantConfig
from torchao.quantization.pt2e.quantize_pt2e import prepare_pt2e, convert_pt2e

T = int(os.environ.get("T", "64"))
OUT = os.path.join(C.OUT, "codec"); os.makedirs(OUT, exist_ok=True)
path = os.path.join(OUT, f"codec_decoder_i8native_T{T}{os.environ.get('SUFFIX', '')}.tflite")
codec, mod = P.load_codec()
dec = P.CodecDecoder(codec).eval()


class Embed(torch.nn.Module):
    """codes -> summed RVQ latent [1,1024,T] (fp32; TFLite EMBEDDING_LOOKUP needs symmetric int8, PT2E gives asymmetric)."""
    def __init__(self, dec):
        super().__init__(); self.q = dec.q
    def forward(self, codes):
        idx0 = codes[:, :1].clamp(0, self.q.semantic_quantizer.codebook_size - 1)
        idx1 = codes[:, 1:].clamp(0, self.q.quantizer.codebook_size - 1)
        return self.q.semantic_quantizer.from_codes(idx0) + self.q.quantizer.from_codes(idx1)


class Body(torch.nn.Module):
    """latent -> wav: post transformer + upsample + SEANet decoder (the part that gets native int8)."""
    def __init__(self, dec):
        super().__init__(); self.q = dec.q; self.dec = dec.dec
    def forward(self, z):
        return self.dec(self.q.upsample(self.q.post_module(z)))


class Full(torch.nn.Module):
    def __init__(self, emb, body):
        super().__init__(); self.emb = emb; self.body = body
    def forward(self, codes):
        return {"wav": self.body(self.emb(codes))}


d = np.load(f"{C.OUT}/oracle/en_ref_0.npz")
rep = np.zeros((1, C.NUM_CB, T), np.int32); n = min(T, d["codes"].shape[1]); rep[0, :, :n] = d["codes"][:, :n]
codes_t = torch.tensor(rep)
emb, body = Embed(dec).eval(), Body(dec).eval()
with torch.no_grad():
    z = emb(codes_t); body(z)  # rope cache
t0 = time.time()
config = pt2e_quantizer.get_symmetric_quantization_config(is_per_channel=True, is_dynamic=True)
q = pt2e_quantizer.PT2EQuantizer().set_global(config)
exported = torch.export.export(body, (z,)).module()
observed = prepare_pt2e(exported, q)
with torch.no_grad():
    observed(z)
quantized = convert_pt2e(observed, fold_quantize=False)
full = Full(emb, quantized)  # no .eval(): exported GraphModules refuse train/eval toggles
model = litert_torch.signature("decode", full, sample_kwargs={"codes": codes_t}).convert(quant_config=QuantConfig(pt2e_quantizer=q))
model.export(path)
print(f"exported {path} {os.path.getsize(path)/1e6:.0f} MB in {time.time()-t0:.0f}s", flush=True)
print("CODEC_NATIVE_I8_DONE")
