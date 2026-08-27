import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { declaredChannels, parseManifest, resolve, thinkingMarkers } from "../dist/index.js";

const here = dirname(fileURLToPath(import.meta.url));
const examples = join(here, "..", "..", "..", "manifest", "examples");
const lfm = parseManifest(readFileSync(join(examples, "litert-community__LFM2.5-1.2B-Instruct.json"), "utf8"));
const qwen = parseManifest(readFileSync(join(examples, "litert-community__Qwen3-4B-Thinking-2507.json"), "utf8"));

test("LFM android midrange -> GPU re-export on gpu", () => {
  const r = resolve(lfm, { platform: "android", deviceClass: "midrange-2023+" });
  assert.equal(r.file, "LFM2.5-1.2B-Instruct_int4_gpu.litertlm");
  assert.equal(r.backend, "gpu");
});

test("LFM ios -> cpu, never Metal", () => {
  const r = resolve(lfm, { platform: "ios" });
  assert.equal(r.backend, "cpu");
});

test("LFM explicit gpu request survives the variant pick (android has a cpu-variant recommendation too)", () => {
  const r = resolve(lfm, { platform: "android", backend: "gpu" });
  assert.equal(r.file, "LFM2.5-1.2B-Instruct_int4_gpu.litertlm");
  assert.equal(r.backend, "gpu");
});

test("LFM explicit backend wins over the platform recommendation; caveats surface in notes", () => {
  const r = resolve(lfm, { platform: "ios", backend: "gpu" });
  assert.equal(r.file, "LFM2.5-1.2B-Instruct_int4_gpu.litertlm");
  assert.equal(r.backend, "gpu");
  assert.ok(r.variant.backends.includes(r.backend));
  assert.ok(r.notes.some((n) => /Metal/.test(n)));
});

test("LFM explicit backend counts only recommendations naming it", () => {
  // _int4_gpu carries a macos/gpu recommendation; for a cpu request the
  // macos/cpu-recommended int8 must win, not the gpu file on cpu.
  const r = resolve(lfm, { platform: "macos", backend: "cpu" });
  assert.equal(r.file, "LFM2.5-1.2B-Instruct_int8.litertlm");
  assert.equal(r.backend, "cpu");
});

test("LFM backend no variant lists resolves to null, never a substitute", () => {
  assert.equal(resolve(lfm, { backend: "npu" }), null);
  assert.equal(resolve(lfm, { platform: "android", backend: "npu" }), null);
});

test("LFM gpu-only request without platform picks a gpu-capable variant", () => {
  const r = resolve(lfm, { backend: "gpu" });
  assert.equal(r.file, "LFM2.5-1.2B-Instruct_int4_gpu.litertlm");
  assert.equal(r.backend, "gpu");
});

test("deviceClass with no matching entry falls back with a note in reason", () => {
  const r = resolve(lfm, { platform: "android", deviceClass: "budget-2019" });
  assert.match(r.reason, /no budget-2019 entry; using the midrange recommendation/);
});

test("resolution URLs follow the fetched revision", () => {
  assert.ok(resolve({ ...lfm, revision: "abc123" }, {}).url.includes("/resolve/abc123/"));
  assert.ok(resolve(lfm, { revision: "deadbeef" }).url.includes("/resolve/deadbeef/"));
  assert.ok(resolve(lfm, {}).url.includes("/resolve/main/"));
});

test("parse rejects a variant with no backends (schema minItems: 1)", () => {
  const base = { manifest_schema: "0.1.0", repo: "t/x", generated: "2026-08-27", model: { display_name: "X" } };
  assert.throws(
    () => parseManifest({ ...base, variants: [{ file: "a.litertlm", quantization: "q", backends: [] }] }),
    /no backends/,
  );
  assert.throws(
    () => parseManifest({ ...base, variants: [{ file: "a.litertlm", quantization: "q" }] }),
    /no backends/,
  );
});

test("Qwen thinking markers keep exact whitespace", () => {
  const t = thinkingMarkers(qwen);
  assert.deepEqual(t, { start: "<think>\n", end: "\n</think>" });
});

test("Qwen session defaults carry the 2048 output budget", () => {
  const r = resolve(qwen, { platform: "macos" });
  assert.equal(r.sessionDefaults.max_output_tokens_min, 2048);
});

test("Qwen ios recommendation picks the block-128 file", () => {
  const r = resolve(qwen, { platform: "ios" });
  assert.equal(r.file, "model.litertlm");
});

test("0.1.1 declared channel set flows through, tool-call included; absent -> empty", () => {
  const m = parseManifest({
    manifest_schema: "0.1.1",
    repo: "test/channels",
    generated: "2026-08-27",
    model: {
      display_name: "Channels",
      capabilities: {
        thinking: { declared: true, channel: { start: "<think>", end: "</think>" } },
        channels: [
          { name: "thought", start: "<think>", end: "</think>", is_reasoning: true },
          { name: "tool_call", start: "<tool_call>", end: "</tool_call>" },
        ],
      },
    },
    variants: [{ file: "a.litertlm", quantization: "q", backends: ["cpu"] }],
  });
  const chans = declaredChannels(m);
  assert.equal(chans.length, 2);
  assert.equal(chans[1].name, "tool_call");
  assert.deepEqual(declaredChannels(lfm), []);
});

test("recommended row naming an unverified backend is ignored", () => {
  const m = parseManifest({
    manifest_schema: "0.1.0",
    repo: "test/malformed",
    generated: "2026-08-26",
    model: { display_name: "Malformed" },
    variants: [{
      file: "m.litertlm",
      quantization: "int8",
      backends: ["cpu"],
      default_backend: "cpu",
      recommended: [{ platform: "android", backend: "gpu" }],
    }],
  });
  const r = resolve(m, { platform: "android" });
  assert.equal(r.backend, "cpu");
});

test("download URL points at the repo file", () => {
  const r = resolve(lfm, { platform: "macos" });
  assert.ok(r.url.startsWith("https://huggingface.co/litert-community/LFM2.5-1.2B-Instruct/resolve/main/"));
});

test("identity fields survive", () => {
  const r = resolve(lfm, { platform: "android" });
  assert.match(r.variant.sha256, /^[0-9a-f]{64}$/);
  assert.ok(r.variant.size_bytes > 0);
});

test("nested variant paths keep their structure in the url", () => {
  const m = parseManifest({
    manifest_schema: "0.1.0",
    repo: "test/nested",
    generated: "2026-08-27",
    model: { display_name: "Nested" },
    variants: [
      { file: "int4/model v2.litertlm", quantization: "int4", backends: ["cpu"], default_backend: "cpu" },
    ],
  });
  assert.equal(
    resolve(m).url,
    "https://huggingface.co/test/nested/resolve/main/int4/model%20v2.litertlm",
  );
});
