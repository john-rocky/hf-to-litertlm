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

test("LFM requested backend outside variant's verified list is never returned", () => {
  const r = resolve(lfm, { platform: "ios", backend: "gpu" });
  assert.notEqual(r.backend === "gpu" && !r.variant.backends.includes("gpu"), true);
  assert.ok(r.variant.backends.includes(r.backend));
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

test("download URL points at the repo file", () => {
  const r = resolve(lfm, { platform: "macos" });
  assert.ok(r.url.startsWith("https://huggingface.co/litert-community/LFM2.5-1.2B-Instruct/resolve/main/"));
});

test("identity fields survive", () => {
  const r = resolve(lfm, { platform: "android" });
  assert.match(r.variant.sha256, /^[0-9a-f]{64}$/);
  assert.ok(r.variant.size_bytes > 0);
});
