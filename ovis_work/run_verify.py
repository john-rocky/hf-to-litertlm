import sys, torch
import transformers.modeling_utils as _mu
if not hasattr(_mu.PreTrainedModel, "all_tied_weights_keys"): _mu.PreTrainedModel.all_tied_weights_keys = {}
if not hasattr(_mu.PreTrainedModel, "is_parallelizable"): _mu.PreTrainedModel.is_parallelizable = False
from transformers import AutoModelForCausalLM
MID="AIDC-AI/Ovis2.5-2B"
def load():
    return AutoModelForCausalLM.from_pretrained(MID, trust_remote_code=True, torch_dtype=torch.float32, low_cpu_mem_usage=True).eval()
try:
    m=load()
except TypeError as e:
    if "tie_weights" not in str(e): raise
    for name,mod in list(sys.modules.items()):
        if "modeling_ovis2_5" in name and hasattr(mod,"Ovis2_5"):
            cls=mod.Ovis2_5; _o=cls.tie_weights; cls.tie_weights=lambda self,*a,**k:_o(self)
    m=load()
sys.path.insert(0, "ovis_work")
import ovis_static
ovis_static.verify(m)
print("VERIFY_DONE")
