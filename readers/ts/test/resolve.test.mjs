import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { declaredChannels, fetchManifest, parseManifest, resolve, thinkingMarkers } from "../dist/index.js";

const here = dirname(fileURLToPath(import.meta.url));
const examples = join(here, "..", "..", "..", "manifest", "examples");
const lfm = parseManifest(readFileSync(join(examples, "litert-community__LFM2.5-1.2B-Instruct.json"), "utf8"));
const qwen = parseManifest(readFileSync(join(examples, "litert-community__Qwen3-4B-Thinking-2507.json"), "utf8"));

const fixture = () => ({
  manifest_schema: "0.1.0", repo: "fixture/origin", generated: "2026-09-08",
  model: { display_name: "Fixture" },
  variants: [{ file: "model.litertlm", quantization: "int4", backends: ["cpu"] }],
});

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

test("parse rejects missing, empty or non-string repo before resolution", () => {
  for (const repo of [undefined, null, "", "  ", 42, false, [], {}]) {
    const input = fixture();
    if (repo === undefined) delete input.repo;
    else input.repo = repo;
    for (const value of [input, JSON.stringify(input)]) {
      assert.throws(() => parseManifest(value), {
        name: "Error", message: /manifest\.repo must be a non-empty string/,
      });
    }
  }
});

test("parse rejects missing, empty or non-string file before resolution", () => {
  for (const file of [undefined, null, "", "  ", 42, false, [], {}]) {
    const input = fixture();
    if (file === undefined) delete input.variants[0].file;
    else input.variants[0].file = file;
    for (const value of [input, JSON.stringify(input)]) {
      assert.throws(() => parseManifest(value), {
        name: "Error", message: /variants\[0\]\.file must be a non-empty string/,
      });
    }
  }
});

test("fetch uses the source repo and revision for a copied manifest", async (t) => {
  const input = fixture();
  input.variants[0].file = "int4/model v2.litertlm";
  input.revision = "stale-revision";
  const requests = [];
  t.mock.method(globalThis, "fetch", async (url) => {
    requests.push(url);
    return new Response(JSON.stringify(input), { status: 200 });
  });
  const manifest = await fetchManifest("fixture/fork", "refs/pr/12");
  assert.equal(requests[0], "https://huggingface.co/fixture/fork/resolve/refs%2Fpr%2F12/litertlm_manifest.json");
  assert.equal(manifest.repo, "fixture/fork");
  assert.equal(manifest.revision, "refs/pr/12");
  assert.equal(resolve(manifest).url, "https://huggingface.co/fixture/fork/resolve/refs%2Fpr%2F12/int4/model%20v2.litertlm");
  assert.equal(resolve(manifest, { revision: "release/v2" }).url, "https://huggingface.co/fixture/fork/resolve/release%2Fv2/int4/model%20v2.litertlm");
  assert.equal(resolve(manifest).backend, "cpu");
  assert.equal((await fetchManifest("fixture/fork")).revision, "main");
  assert.equal(requests[1], "https://huggingface.co/fixture/fork/resolve/main/litertlm_manifest.json");
  assert.equal(parseManifest(input).repo, "fixture/origin");
});

test("fetch still rejects HTTP and parse errors", async (t) => {
  t.mock.method(globalThis, "fetch", async () => new Response("missing", { status: 404 }));
  await assert.rejects(fetchManifest("fixture/fork"), /HTTP 404/);
  for (const field of ["repo", "file"]) {
    const input = fixture();
    if (field === "repo") delete input.repo;
    else delete input.variants[0].file;
    t.mock.method(globalThis, "fetch", async () => new Response(JSON.stringify(input)));
    await assert.rejects(fetchManifest("fixture/fork"), new RegExp(`${field} must be a non-empty string`));
  }
  await assert.rejects(fetchManifest(""), /source repo must be a non-empty string/);
});

test("resolve excludes hand-built empty backends, including defaults and recommendations", () => {
  const input = fixture();
  Object.assign(input.variants[0], {
    backends: [], size_bytes: 1, default_backend: "cpu",
    recommended: [{ platform: "android", backend: "cpu" }],
  });
  for (const opts of [{}, { platform: "android" }, { backend: "cpu" }, { backend: "gpu" }]) {
    assert.equal(resolve(input, opts), null);
  }
  input.variants.push({ file: "gpu.litertlm", quantization: "int4", backends: ["gpu"], size_bytes: 2 });
  for (const opts of [{}, { platform: "android" }, { backend: "gpu" }]) {
    const r = resolve(input, opts);
    assert.equal(r.file, "gpu.litertlm");
    assert.equal(r.backend, "gpu");
    assert.ok(r.variant.backends.includes(r.backend));
  }
  assert.equal(resolve(input, { backend: "cpu" }), null);
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

test("parse checks string-list elements eagerly: a non-string in backends, platform_notes or known_issues fails at parse", () => {
  const withVariant = (v) => ({
    manifest_schema: "0.1.0",
    repo: "t/x",
    generated: "2026-08-29",
    model: { display_name: "X" },
    variants: [v],
  });
  for (const v of [
    { file: "a.litertlm", quantization: "q", backends: ["cpu", 42] },
    { file: "a.litertlm", quantization: "q", backends: ["cpu"], requirements: { platform_notes: ["ok", 42] } },
    { file: "a.litertlm", quantization: "q", backends: ["cpu"], known_issues: ["ok", 42] },
  ]) {
    assert.throws(() => parseManifest(withVariant(v)), /must be a list of strings/, JSON.stringify(v));
  }
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
  // "absent" is checked on an inline manifest: the fixtures are real shipped manifests and may
  // declare channels (LFM2.5-1.2B-Instruct does since its 0.1.2 re-ship on 2026-09-05).
  const bare = parseManifest({
    manifest_schema: "0.1.0",
    repo: "test/no-channels",
    generated: "2026-08-27",
    model: { display_name: "Bare" },
    variants: [{ file: "b.litertlm", quantization: "q", backends: ["cpu"] }],
  });
  assert.deepEqual(declaredChannels(bare), []);
  assert.equal(thinkingMarkers(bare), undefined);
  assert.equal(resolve(bare).thinkingChannel, undefined);
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
