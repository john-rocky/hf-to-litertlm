"""A small random ParakeetForTDT for the r4 dry runs outside the window (not the model: no redux weights read).
Encoder hidden 64, 2 layers, 2 heads, the real subsampling shape (128 mel bins, factor 8); decoder hidden 640 and
vocab 8,193 so that its encoder output [1, T, 640] feeds r2's real dec.tflite (the joint takes [1, 640]).
"""
import torch


def small_model_640():
    from transformers import ParakeetForTDT, ParakeetTDTConfig
    cfg = ParakeetTDTConfig(
        encoder_config={"hidden_size": 64, "num_attention_heads": 2, "num_key_value_heads": 2,
                        "intermediate_size": 128, "num_hidden_layers": 2, "conv_kernel_size": 9,
                        "subsampling_conv_channels": 16, "subsampling_conv_kernel_size": 3,
                        "subsampling_conv_stride": 2, "subsampling_factor": 8, "num_mel_bins": 128,
                        "attention_bias": False, "convolution_bias": False, "scale_input": False,
                        "hidden_act": "silu", "dropout": 0.0, "dropout_positions": 0.0, "layerdrop": 0.0,
                        "activation_dropout": 0.0, "attention_dropout": 0.0},
        decoder_hidden_size=640, num_decoder_layers=2, vocab_size=8193, durations=[0, 1, 2, 3, 4],
        blank_token_id=8192, hidden_act="relu")
    torch.manual_seed(0)
    m = ParakeetForTDT(cfg).eval()
    with torch.no_grad():
        for mod in m.modules():
            if isinstance(mod, torch.nn.BatchNorm1d):
                mod.running_mean.normal_()
                mod.running_var.uniform_(0.5, 2.0)
            if isinstance(mod, torch.nn.LayerNorm):
                mod.weight.normal_()
                mod.bias.normal_()
    return m
