import sys, torch
import transformers.modeling_utils as _mu
if not hasattr(_mu.PreTrainedModel,"all_tied_weights_keys"): _mu.PreTrainedModel.all_tied_weights_keys={}
if not hasattr(_mu.PreTrainedModel,"is_parallelizable"): _mu.PreTrainedModel.is_parallelizable=False
from transformers import AutoModelForCausalLM
MID="AIDC-AI/Ovis2.5-2B"
def load(): return AutoModelForCausalLM.from_pretrained(MID,trust_remote_code=True,torch_dtype=torch.float32,low_cpu_mem_usage=True).eval()
try: m=load()
except TypeError as e:
    if "tie_weights" not in str(e): raise
    for n,mod in list(sys.modules.items()):
        if "modeling_ovis2_5" in n and hasattr(mod,"Ovis2_5"):
            c=mod.Ovis2_5;_o=c.tie_weights;c.tie_weights=lambda s,*a,**k:_o(s)
    m=load()
sys.path.insert(0,"ovis_work"); import ovis_static
N=ovis_static.install_static_vision(m)
vt,vit,vm=ovis_static._find_vision(m)
grid=torch.tensor([[1,32,32]],dtype=torch.long)
C=vm.config.num_channels*vm.config.temporal_patch_size*vm.config.patch_size**2
pv=torch.randn(N*... if False else 1024, C)  # N patches
# wrap: pixel_values -> vit features (the part that had .item/.tolist/loops)
class VitWrap(torch.nn.Module):
    def __init__(s,vt): super().__init__(); s.vt=vt
    def forward(s,pv): return s.vt._encode(pv, torch.tensor([[1,32,32]],dtype=torch.long))
w=VitWrap(vt).eval()
print("torch.export static vit (pixel_values -> features)...")
ep=torch.export.export(w,(pv,))
print("EXPORT_OK n_nodes=", sum(1 for _ in ep.graph.nodes))
