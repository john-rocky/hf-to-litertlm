/**
 * litertlm-manifest — reference reader for `litertlm_manifest.json`
 * (the deployment manifest shipped at the root of .litertlm model repos).
 *
 * Core is IO-free: pass a parsed manifest (or JSON string). A convenience
 * `fetchManifest()` uses global fetch. No dependencies.
 *
 * Spec: https://github.com/john-rocky/hf-to-litertlm/blob/main/manifest/SCHEMA.md
 */

export type Backend = "cpu" | "gpu" | "npu";
export type Platform = "android" | "ios" | "macos" | "windows" | "linux";

export interface ThinkingChannel {
  start: string;
  end: string;
}

export interface Capabilities {
  vision?: boolean;
  audio?: boolean;
  thinking?: { declared: boolean; channel?: ThinkingChannel };
}

export interface Recommendation {
  platform: Platform;
  device_class?: string;
  backend: Backend;
  reason?: string;
}

export interface MeasuredRow {
  device: string;
  os?: string;
  backend: Backend;
  runtime: string;
  prompt_tokens?: number;
  decode_tokens?: number;
  prefill_tps?: number | string;
  decode_tps?: number | string;
  ttft_s?: number | string;
  max_num_tokens?: number;
  cache?: string;
  runs?: number;
  date: string;
  source: string;
}

export interface Variant {
  file: string;
  sha256?: string;
  size_bytes?: number;
  quantization: string;
  backends: Backend[];
  default_backend?: Backend;
  min_runtime_version?: string;
  recommended?: Recommendation[];
  requirements?: { peak_ram_mb?: number; platform_notes?: string[] };
  measured?: MeasuredRow[];
  known_issues?: string[];
  sections?: { type: string; size_bytes?: number; model_type?: string; backend_constraint?: string }[];
}

export interface Manifest {
  manifest_schema: string;
  repo: string;
  generated: string;
  model: {
    display_name: string;
    base_model?: string;
    architecture?: string;
    parameters_b?: number;
    license?: string;
    context_length?: number;
    capabilities?: Capabilities;
    session_defaults?: Record<string, unknown>;
  };
  variants: Variant[];
}

export interface ResolveOptions {
  platform?: Platform;
  /** Caller's backend preference; the resolver never picks a backend the variant doesn't list. */
  backend?: Backend;
  deviceClass?: string;
}

export interface Resolution {
  /** File name inside the repo — download as https://huggingface.co/<repo>/resolve/main/<file> */
  file: string;
  url: string;
  backend: Backend;
  variant: Variant;
  sessionDefaults?: Record<string, unknown>;
  capabilities?: Capabilities;
  thinkingChannel?: ThinkingChannel;
  contextLength?: number;
  /** platform_notes + known_issues of the chosen variant — surface these to the app developer. */
  notes: string[];
  /** Why this variant/backend was chosen (for logs). */
  reason: string;
}

export function parseManifest(input: string | object): Manifest {
  const m = (typeof input === "string" ? JSON.parse(input) : input) as Manifest;
  if (!m || typeof m !== "object" || !m.manifest_schema || !Array.isArray(m.variants) || m.variants.length === 0) {
    throw new Error("not a litertlm_manifest.json (missing manifest_schema or variants)");
  }
  if (!/^0\.1\./.test(m.manifest_schema)) {
    throw new Error(`unsupported manifest_schema ${m.manifest_schema} (reader supports 0.1.x)`);
  }
  return m;
}

/** Fetch <repo>'s manifest from the Hugging Face Hub. */
export async function fetchManifest(repo: string, revision = "main"): Promise<Manifest> {
  const url = `https://huggingface.co/${repo}/resolve/${revision}/litertlm_manifest.json`;
  const res = await fetch(url);
  if (!res.ok) throw new Error(`no litertlm_manifest.json at ${url} (HTTP ${res.status})`);
  return parseManifest(await res.text());
}

interface Scored {
  variant: Variant;
  backend: Backend;
  score: number;
  reason: string;
}

/**
 * Pick the variant + backend for a device. v0.1 algorithm, deterministic:
 *
 * 1. A variant with a `recommended` entry matching the requested platform wins;
 *    matching device_class too ranks higher. The recommendation's backend is used
 *    unless the caller requested one the variant also lists.
 * 2. Otherwise a variant listing the requested backend wins.
 * 3. Otherwise the smallest variant (by size_bytes), on its default_backend (else "cpu").
 *
 * Ties break toward the smaller file. The resolver never returns a backend
 * absent from the variant's verified `backends` list — recommendations naming
 * an unlisted backend are ignored.
 */
export function resolve(manifest: Manifest, opts: ResolveOptions = {}): Resolution {
  const scored: Scored[] = manifest.variants.map((v) => {
    let score = 0;
    let backend: Backend | undefined;
    let reason = "fallback: first variant";
    if (opts.platform && v.recommended) {
      const recs = v.recommended.filter((r) => r.platform === opts.platform && v.backends.includes(r.backend));
      const classRec = opts.deviceClass ? recs.find((r) => r.device_class === opts.deviceClass) : undefined;
      const rec = classRec ?? recs.find((r) => !r.device_class || !opts.deviceClass) ?? recs[0];
      if (rec) {
        score = classRec ? 300 : 200;
        backend = rec.backend;
        reason = `recommended for ${opts.platform}${classRec ? `/${opts.deviceClass}` : ""}: ${rec.reason ?? ""}`.trim();
      }
    }
    if (score === 0 && opts.backend && v.backends.includes(opts.backend)) {
      score = 100;
      backend = opts.backend;
      reason = `supports requested backend ${opts.backend}`;
    }
    if (opts.backend && v.backends.includes(opts.backend)) backend = opts.backend;
    if (!backend) backend = v.default_backend && v.backends.includes(v.default_backend) ? v.default_backend : v.backends[0] ?? "cpu";
    return { variant: v, backend, score, reason };
  });

  scored.sort(
    (a, b) => b.score - a.score || (a.variant.size_bytes ?? Infinity) - (b.variant.size_bytes ?? Infinity),
  );
  const best = scored[0];
  const v = best.variant;
  const caps = manifest.model.capabilities;
  return {
    file: v.file,
    url: `https://huggingface.co/${manifest.repo}/resolve/main/${encodeURIComponent(v.file)}`,
    backend: best.backend,
    variant: v,
    sessionDefaults: manifest.model.session_defaults,
    capabilities: caps,
    thinkingChannel: caps?.thinking?.declared ? caps.thinking.channel : undefined,
    contextLength: manifest.model.context_length,
    notes: [...(v.requirements?.platform_notes ?? []), ...(v.known_issues ?? [])],
    reason: best.reason,
  };
}

/**
 * The model's declared thinking-channel markers (exact strings, whitespace
 * included) — e.g. for wiring a streaming parser. Undefined when the model
 * declares no thinking channel.
 */
export function thinkingMarkers(manifest: Manifest): ThinkingChannel | undefined {
  const t = manifest.model.capabilities?.thinking;
  return t?.declared ? t.channel : undefined;
}
