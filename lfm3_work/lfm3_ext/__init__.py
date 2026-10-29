"""LFM3 decoder exporter for LiteRT / LiteRT-LM (litert-torch `export_hf` extension).

Entry points: `export_lfm3.export_lfm3` (any LFM3 checkpoint, any key prefix),
`extension.Lfm3Extension` (`--extension_modules=lfm3_ext.extension`),
`loader.load_lfm3_causal_lm` (the PyTorch module the export traces).
"""
