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

/** One entry of the bundle's declared channel set (manifest 0.1.1+). */
export interface DeclaredChannel {
  name: string;
  start: string;
  end: string;
  is_reasoning?: boolean;
}

export interface Capabilities {
  vision?: boolean;
  audio?: boolean;
  thinking?: { declared: boolean; channel?: ThinkingChannel };
  /** Full bundle-declared channel set (0.1.1+); `thinking` mirrors the first entry. */
  channels?: DeclaredChannel[];
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
  /**
   * Not part of the file: the revision the manifest was fetched at, stamped by
   * fetchManifest() so resolve() URLs follow a pinned fetch. Defaults to "main".
   */
  revision?: string;
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
  /**
   * Explicit backend request. A filter, not a preference: only variants listing
   * it are considered, and resolve() returns null when none does.
   */
  backend?: Backend;
  deviceClass?: string;
  /** Repo revision for the download URL; overrides the manifest's fetched revision (default "main"). */
  revision?: string;
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
  for (const v of m.variants) {
    if (!Array.isArray(v.backends) || v.backends.length === 0) {
      throw new Error(`variant ${v?.file ?? "?"} lists no backends (schema requires minItems: 1)`);
    }
  }
  return m;
}

/** Fetch <repo>'s manifest from the Hugging Face Hub. resolve() URLs follow the revision fetched here. */
export async function fetchManifest(repo: string, revision = "main"): Promise<Manifest> {
  const url = `https://huggingface.co/${repo}/resolve/${encodeURIComponent(revision)}/litertlm_manifest.json`;
  const res = await fetch(url);
  if (!res.ok) throw new Error(`no litertlm_manifest.json at ${url} (HTTP ${res.status})`);
  const m = parseManifest(await res.text());
  m.revision = revision;
  return m;
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
 * 1. An explicit `backend` request is a filter, not a score: only variants
 *    listing it compete, the result keeps that backend, and resolve() returns
 *    null when no variant lists it — it never substitutes another backend.
 * 2. A variant with a `recommended` entry matching the requested platform wins;
 *    matching device_class too ranks higher. When a backend was requested,
 *    only recommendations naming that backend count.
 * 3. Otherwise the smallest variant (by size_bytes), on the requested backend
 *    (else its default_backend, else the first listed backend).
 *
 * Ties break toward the smaller file. The resolver never returns a backend
 * absent from the variant's verified `backends` list — recommendations naming
 * an unlisted backend are ignored.
 */
export function resolve(manifest: Manifest, opts: ResolveOptions = {}): Resolution | null {
  const requested = opts.backend;
  const candidates = requested
    ? manifest.variants.filter((v) => v.backends.includes(requested))
    : manifest.variants;
  if (candidates.length === 0) return null;

  const scored: Scored[] = candidates.map((v) => {
    let score = 0;
    let backend: Backend | undefined = requested;
    let reason = requested ? `supports requested backend ${requested}` : "fallback: smallest variant";
    if (opts.platform && v.recommended) {
      const recs = v.recommended.filter(
        (r) => r.platform === opts.platform && v.backends.includes(r.backend) && (!requested || r.backend === requested),
      );
      const classRec = opts.deviceClass ? recs.find((r) => r.device_class === opts.deviceClass) : undefined;
      const rec = classRec ?? recs.find((r) => !r.device_class || !opts.deviceClass) ?? recs[0];
      if (rec) {
        score = classRec ? 300 : 200;
        backend = requested ?? rec.backend;
        const classNote =
          !classRec && opts.deviceClass && rec.device_class
            ? ` (no ${opts.deviceClass} entry; using the ${rec.device_class} recommendation)`
            : "";
        reason = `recommended for ${opts.platform}${classRec ? `/${opts.deviceClass}` : ""}${classNote}: ${rec.reason ?? ""}`.trim();
      }
    }
    if (!backend) backend = v.default_backend && v.backends.includes(v.default_backend) ? v.default_backend : v.backends[0] ?? "cpu";
    return { variant: v, backend, score, reason };
  });

  scored.sort(
    (a, b) => b.score - a.score || (a.variant.size_bytes ?? Infinity) - (b.variant.size_bytes ?? Infinity),
  );
  const best = scored[0];
  const v = best.variant;
  const caps = manifest.model.capabilities;
  const revision = encodeURIComponent(opts.revision ?? manifest.revision ?? "main");
  return {
    file: v.file,
    // Encode per path segment so a repo that nests variants in subfolders
    // (file containing '/') keeps its structure — '%2F' would 404.
    url: `https://huggingface.co/${manifest.repo}/resolve/${revision}/${v.file.split("/").map(encodeURIComponent).join("/")}`,
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

/**
 * The bundle's full declared channel set (manifest 0.1.1+) — thinking,
 * tool-call, or anything else a model declares. Empty for 0.1.0 manifests;
 * fall back to thinkingMarkers() plus your runtime's default channels.
 */
export function declaredChannels(manifest: Manifest): DeclaredChannel[] {
  return manifest.model.capabilities?.channels ?? [];
}
