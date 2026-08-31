from typing import Callable
import torch
import torch.nn as nn
from transformers import PreTrainedModel, Qwen3Model
from transformers.cache_utils import Cache
from transformers.masking_utils import create_causal_mask
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs


# From modeling_t5gemma.py
def bidirectional_mask_function(attention_mask: torch.Tensor | None) -> Callable:
    """
    This creates bidirectional attention mask.
    """

    def inner_mask(batch_idx: int, head_idx: int, q_idx: int, kv_idx: int) -> bool:
        if attention_mask is None:
            return torch.ones((), dtype=torch.bool)
        return attention_mask[batch_idx, kv_idx].to(torch.bool)

    return inner_mask


class Qwen3BidirectionalModel(PreTrainedModel):
    _supports_flash_attn = True
    _supports_sdpa = True

    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen3Model(config)
        self.linear = nn.Linear(config.hidden_size, config.num_labels, bias=False)
        self.post_init()

    def post_init(self):
        super().post_init()
        # Override to set all layers to non-causal attention. This'll work with attn_implementation="flash_attention_2" or "sdpa"
        for layer in self.model.layers:
            layer.self_attn.is_causal = False

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPooling:
        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(input_ids)
            input_ids = None

        # We construct a dummy tensor imitating initial positions
        dummy_cache_position = torch.arange(
            inputs_embeds.shape[1], device=inputs_embeds.device, dtype=torch.long
        )
        attention_mask = {
            "full_attention": create_causal_mask(
                config=self.config,
                input_embeds=inputs_embeds,
                attention_mask=attention_mask,
                cache_position=dummy_cache_position,
                past_key_values=None,
                position_ids=position_ids,
                or_mask_function=bidirectional_mask_function(attention_mask),
            )
        }

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )
        outputs.last_hidden_state = self.linear(outputs.last_hidden_state)
        return outputs
