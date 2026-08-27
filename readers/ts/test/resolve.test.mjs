import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { parseManifest, resolve, thinkingMarkers } from "../dist/index.js";

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
