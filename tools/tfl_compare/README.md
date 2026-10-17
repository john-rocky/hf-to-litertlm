# tfl_compare — are two `.litertlm` graph sections the same computation?

A recipe re-run can rebuild a bundle whose TFLite section has the same size and a different sha256.
These three read-only tools say what differs, in order of depth; each writes a JSON report.

```sh
python tools/tfl_compare/tfl_diff.py      A.litertlm B.litertlm SECTION diff.json       # where the bytes differ: which buffers (tensors), or the flatbuffer's non-buffer part
python tools/tfl_compare/tfl_swapcheck.py A.litertlm B.litertlm SECTION swap.json       # is it only an order difference? buffers and tensor / operator tables compared as tables
python tools/tfl_compare/tfl_struct.py    A.litertlm B.litertlm SECTION struct.json [N] # the same computation in a different order? structural signatures per subgraph (default the first 7 = the signatures)
```

`SECTION` is the index of the TFLiteModel section in the bundle header (`litert-lm`'s header lists the sections; a
text-only LLM bundle usually has one, a bundle with an externalized embedder has two). They need the `ai-edge-litert` and
`litert-lm-builder` packages. Always run a must-differ control beside the pair you care about (two different files must
come out `DIFFERENT`).

Case (Olmo-3-7B-Think int4, 2026-10-16): a re-run of the published recipe produced a main graph with 17,917 differing bytes
in the seven signature subgraphs — the two RoPE inverse-frequency constants (sliding default / full-layer yarn) and their
operator branches came out in the other order; the 879 decomposition subgraphs and every weight were identical, and
`tfl_struct.py` reported the two sections structurally equal, while the int4 file against its 2,048-token-cache twin
reported `DIFFERENT`. REPRODUCE.md's Olmo-3-7B-Think entry records both sha256 values as "the same computation".
