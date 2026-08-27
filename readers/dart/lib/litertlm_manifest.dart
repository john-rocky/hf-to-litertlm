/// litertlm_manifest — reference reader for `litertlm_manifest.json`
/// (the deployment manifest shipped at the root of .litertlm model repos).
///
/// Core is IO-free: pass a parsed manifest map (or JSON string); do the HTTP
/// with whatever stack your app already uses. No dependencies beyond dart:convert.
///
/// Spec: https://github.com/john-rocky/hf-to-litertlm/blob/main/manifest/SCHEMA.md
library litertlm_manifest;

import 'dart:convert';

class ThinkingChannel {
  final String start;
  final String end;
  const ThinkingChannel(this.start, this.end);
}

/// One entry of the bundle's declared channel set (manifest 0.1.1+).
class DeclaredChannel {
  final String name;
  final String start;
  final String end;
  final bool isReasoning;
  const DeclaredChannel(this.name, this.start, this.end, {this.isReasoning = false});
}

class Capabilities {
  final bool vision;
  final bool audio;
  final bool thinkingDeclared;
  final ThinkingChannel? thinkingChannel;

  /// Full bundle-declared channel set (0.1.1+) — thinking, tool-call, or
  /// anything else a model declares. Empty for 0.1.0 manifests; fall back to
  /// [thinkingChannel] plus your runtime's default channels.
  final List<DeclaredChannel> channels;

  const Capabilities({
    this.vision = false,
    this.audio = false,
    this.thinkingDeclared = false,
    this.thinkingChannel,
    this.channels = const [],
  });

  factory Capabilities.fromJson(Map<String, dynamic>? j) {
    if (j == null) return const Capabilities();
    final t = j['thinking'] as Map<String, dynamic>?;
    final ch = t?['channel'] as Map<String, dynamic>?;
    return Capabilities(
      vision: j['vision'] == true,
      audio: j['audio'] == true,
      thinkingDeclared: t?['declared'] == true,
      thinkingChannel: ch == null
          ? null
          : ThinkingChannel(ch['start'] as String? ?? '', ch['end'] as String? ?? ''),
      channels: ((j['channels'] as List?) ?? const [])
          .map((e) => e as Map<String, dynamic>)
          .map((e) => DeclaredChannel(
                e['name'] as String? ?? '',
                e['start'] as String? ?? '',
                e['end'] as String? ?? '',
                isReasoning: e['is_reasoning'] == true,
              ))
          .toList(),
    );
  }
}

class Recommendation {
  final String platform;
  final String? deviceClass;
  final String backend;
  final String? reason;
  const Recommendation(this.platform, this.deviceClass, this.backend, this.reason);

  factory Recommendation.fromJson(Map<String, dynamic> j) => Recommendation(
        j['platform'] as String,
        j['device_class'] as String?,
        j['backend'] as String,
        j['reason'] as String?,
      );
}

class Variant {
  final String file;
  final String? sha256;
  final int? sizeBytes;
  final String quantization;
  final List<String> backends;
  final String? defaultBackend;
  final String? minRuntimeVersion;
  final List<Recommendation> recommended;
  final List<String> platformNotes;
  final List<String> knownIssues;
  final Map<String, dynamic> raw;

  Variant.fromJson(Map<String, dynamic> j)
      : file = j['file'] as String,
        sha256 = j['sha256'] as String?,
        sizeBytes = j['size_bytes'] as int?,
        quantization = j['quantization'] as String? ?? '',
        backends = _requireBackends(j),
        defaultBackend = j['default_backend'] as String?,
        minRuntimeVersion = j['min_runtime_version'] as String?,
        recommended = ((j['recommended'] as List?) ?? const [])
            .map((e) => Recommendation.fromJson(e as Map<String, dynamic>))
            .toList(),
        platformNotes = (((j['requirements'] as Map<String, dynamic>?)?['platform_notes'])
                    as List?)
                ?.cast<String>() ??
            const [],
        knownIssues = (j['known_issues'] as List?)?.cast<String>() ?? const [],
        raw = j;

  static List<String> _requireBackends(Map<String, dynamic> j) {
    final b = (j['backends'] as List?)?.cast<String>();
    if (b == null || b.isEmpty) {
      throw FormatException(
          'variant ${j['file']} lists no backends (schema requires minItems: 1)');
    }
    return b;
  }
}

class LitertlmManifest {
  final String schemaVersion;
  final String repo;
  final String displayName;
  final int? contextLength;
  final Capabilities capabilities;
  final Map<String, dynamic>? sessionDefaults;
  final List<Variant> variants;

  /// Not part of the file: the revision the manifest was fetched at (the app
  /// does its own HTTP), so resolve() URLs follow a pinned fetch. Defaults to
  /// "main".
  final String? revision;

  LitertlmManifest._(this.schemaVersion, this.repo, this.displayName, this.contextLength,
      this.capabilities, this.sessionDefaults, this.variants, this.revision);

  factory LitertlmManifest.fromJson(dynamic input, {String? revision}) {
    final m = (input is String ? jsonDecode(input) : input) as Map<String, dynamic>;
    final schema = m['manifest_schema'] as String?;
    final vs = m['variants'] as List?;
    if (schema == null || vs == null || vs.isEmpty) {
      throw const FormatException(
          'not a litertlm_manifest.json (missing manifest_schema or variants)');
    }
    if (!schema.startsWith('0.1.')) {
      throw FormatException('unsupported manifest_schema $schema (reader supports 0.1.x)');
    }
    final model = m['model'] as Map<String, dynamic>? ?? const {};
    return LitertlmManifest._(
      schema,
      m['repo'] as String,
      model['display_name'] as String? ?? m['repo'] as String,
      model['context_length'] as int?,
      Capabilities.fromJson(model['capabilities'] as Map<String, dynamic>?),
      model['session_defaults'] as Map<String, dynamic>?,
      vs.map((e) => Variant.fromJson(e as Map<String, dynamic>)).toList(),
      revision,
    );
  }

  /// The model's declared thinking markers (exact strings, whitespace included).
  ThinkingChannel? get thinkingMarkers =>
      capabilities.thinkingDeclared ? capabilities.thinkingChannel : null;

  /// The bundle's full declared channel set (manifest 0.1.1+). Empty for
  /// 0.1.0 manifests — fall back to [thinkingMarkers] plus your runtime's
  /// default channels.
  List<DeclaredChannel> get declaredChannels => capabilities.channels;

  /// Pick the variant + backend for a device. v0.1 algorithm, deterministic —
  /// identical to the TypeScript reference:
  ///
  /// 1. An explicit [backend] request is a filter, not a score: only variants
  ///    listing it compete, the result keeps that backend, and resolve()
  ///    returns null when no variant lists it — it never substitutes another
  ///    backend.
  /// 2. A variant with a `recommended` entry matching [platform] wins;
  ///    matching [deviceClass] too ranks higher. When a backend was requested,
  ///    only recommendations naming that backend count.
  /// 3. Otherwise the smallest variant (by size_bytes), on the requested
  ///    backend (else its default_backend, else the first listed backend).
  ///
  /// Ties break toward the smaller file. Never returns a backend absent from
  /// the variant's verified `backends` list — recommendations naming an
  /// unlisted backend are ignored.
  Resolution? resolve(
      {String? platform, String? backend, String? deviceClass, String? revision}) {
    final candidates = backend != null
        ? variants.where((v) => v.backends.contains(backend)).toList()
        : variants;
    if (candidates.isEmpty) return null;

    final scored = <_Scored>[];
    for (final v in candidates) {
      var score = 0;
      String? chosen = backend;
      var reason = backend != null
          ? 'supports requested backend $backend'
          : 'fallback: smallest variant';
      if (platform != null && v.recommended.isNotEmpty) {
        final recs = v.recommended
            .where((r) =>
                r.platform == platform &&
                v.backends.contains(r.backend) &&
                (backend == null || r.backend == backend))
            .toList();
        Recommendation? classRec;
        if (deviceClass != null) {
          for (final r in recs) {
            if (r.deviceClass == deviceClass) {
              classRec = r;
              break;
            }
          }
        }
        Recommendation? classFree;
        for (final r in recs) {
          if (r.deviceClass == null || deviceClass == null) {
            classFree = r;
            break;
          }
        }
        final rec = classRec ?? classFree ?? (recs.isNotEmpty ? recs.first : null);
        if (rec != null) {
          score = classRec != null ? 300 : 200;
          chosen = backend ?? rec.backend;
          final classNote = classRec == null && deviceClass != null && rec.deviceClass != null
              ? ' (no $deviceClass entry; using the ${rec.deviceClass} recommendation)'
              : '';
          reason = 'recommended for $platform'
              '${classRec != null ? "/$deviceClass" : ""}$classNote: ${rec.reason ?? ""}';
        }
      }
      chosen ??= (v.defaultBackend != null && v.backends.contains(v.defaultBackend))
          ? v.defaultBackend!
          : (v.backends.isNotEmpty ? v.backends.first : 'cpu');
      scored.add(_Scored(v, chosen, score, reason.trim()));
    }
    scored.sort((a, b) {
      final s = b.score.compareTo(a.score);
      if (s != 0) return s;
      return (a.variant.sizeBytes ?? 1 << 62).compareTo(b.variant.sizeBytes ?? 1 << 62);
    });
    final best = scored.first;
    final v = best.variant;
    final rev = Uri.encodeComponent(revision ?? this.revision ?? 'main');
    return Resolution(
      file: v.file,
      // Encode per path segment so a repo that nests variants in subfolders
      // (file containing '/') keeps its structure — '%2F' would 404.
      url:
          'https://huggingface.co/$repo/resolve/$rev/${v.file.split('/').map(Uri.encodeComponent).join('/')}',
      backend: best.backend,
      variant: v,
      sessionDefaults: sessionDefaults,
      capabilities: capabilities,
      thinkingChannel: thinkingMarkers,
      contextLength: contextLength,
      notes: [...v.platformNotes, ...v.knownIssues],
      reason: best.reason,
    );
  }
}

class _Scored {
  final Variant variant;
  final String backend;
  final int score;
  final String reason;
  _Scored(this.variant, this.backend, this.score, this.reason);
}

class Resolution {
  final String file;
  final String url;
  final String backend;
  final Variant variant;
  final Map<String, dynamic>? sessionDefaults;
  final Capabilities capabilities;
  final ThinkingChannel? thinkingChannel;
  final int? contextLength;
  final List<String> notes;
  final String reason;
  const Resolution({
    required this.file,
    required this.url,
    required this.backend,
    required this.variant,
    this.sessionDefaults,
    required this.capabilities,
    this.thinkingChannel,
    this.contextLength,
    required this.notes,
    required this.reason,
  });
}
