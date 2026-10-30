"""Text-only causal LM view of meta-models/Muse-Glimmer-30B for litert-torch export_hf.

transformers ships MuseGlimmerTextModel (the decoder) and MuseGlimmerForConditionalGeneration
(the VLM) but no text-only *ForCausalLM, so export_hf's text_generation task cannot load the
decoder by itself. This class wraps the decoder and the lm_head with the same logits head as
MuseGlimmerForConditionalGeneration.forward (output_multiplier, then the tanh softcap).

Parameter names match the VLM checkpoint ("model.language_model.*", "lm_head.weight"), so the
original safetensors load unchanged and the vision tower's tensors are simply not read.

register() makes the class reachable from AutoModelForCausalLM and as
transformers.MuseGlimmerTextForCausalLM (export_hf's auto_model_override).
"""

import torch
from torch import nn
import transformers
from transformers import AutoModelForCausalLM, GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.muse_glimmer.configuration_muse_glimmer import MuseGlimmerTextConfig
from transformers.models.muse_glimmer.modeling_muse_glimmer import (
        MuseGlimmerPreTrainedModel,
        MuseGlimmerTextModel,
)


class _LanguageModelHolder(nn.Module):
    """Holds the decoder as `language_model`, as MuseGlimmerModel does."""

    def __init__(self, config):
        super().__init__()
        self.language_model = MuseGlimmerTextModel(config)


class MuseGlimmerTextForCausalLM(MuseGlimmerPreTrainedModel, GenerationMixin):
    config_class = MuseGlimmerTextConfig
    config: MuseGlimmerTextConfig
    base_model_prefix = "model"
    _tied_weights_keys = {}
    _can_record_outputs = None

    def __init__(self, config: MuseGlimmerTextConfig):
        super().__init__(config)
        self.model = _LanguageModelHolder(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        model = super().from_pretrained(*args, **kwargs)
        factor = getattr(model.config, "lm_head_prescale_factor", None)
        if getattr(model.config, "lm_head_prescaled", False):
            # W' = W * output_multiplier / final_logit_softcapping, applied once after loading (a separate file
            # in the index does not work: transformers 5.x also reads the lm_head.weight of the shard that holds it).
            with torch.no_grad():
                model.lm_head.weight = nn.Parameter(model.lm_head.weight.float() * torch.tensor(factor, dtype=torch.float32))
        return model

    def get_input_embeddings(self):
        return self.model.language_model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.language_model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def get_decoder(self):
        return self.model.language_model

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            position_ids=None,
            past_key_values=None,
            inputs_embeds=None,
            use_cache=None,
            logits_to_keep=0,
            **kwargs,
    ):
        outputs = self.model.language_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                **kwargs,
        )
        hidden_states = outputs.last_hidden_state
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        # Same head as MuseGlimmerForConditionalGeneration.forward: T * tanh(logits * mult / T).
        cap = self.config.final_logit_softcapping
        if getattr(self.config, "lm_head_prescaled", False):
            # lm_head.weight already holds W * mult / T (muse_textdir.py --prescale-lm-head): nothing scalar
            # sits between the FC and the tanh, so the converter has no multiply to fold into a lazy weight.
            logits = torch.tanh(logits) * cap
        else:
            logits = logits * self.config.output_multiplier
            logits = torch.tanh(logits / cap) * cap
        return CausalLMOutputWithPast(
                logits=logits,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
        )


def register():
    # exist_ok=True: transformers 5.x silently skips register() for a config class defined inside transformers
    # unless exist_ok is set (auto_factory._LazyAutoMapping.register, measured on 5.20.2).
    AutoModelForCausalLM.register(MuseGlimmerTextConfig, MuseGlimmerTextForCausalLM, exist_ok=True)
    transformers.MuseGlimmerTextForCausalLM = MuseGlimmerTextForCausalLM
