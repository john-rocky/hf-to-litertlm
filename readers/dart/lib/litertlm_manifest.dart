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

class Capabilities {
  final bool vision;
  final bool audio;
  final bool thinkingDeclared;
  final ThinkingChannel? thinkingChannel;
  const Capabilities({
    this.vision = false,
    this.audio = false,
    this.thinkingDeclared = false,
    this.thinkingChannel,
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
        backends = (j['backends'] as List?)?.cast<String>() ?? const ['cpu'],
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
}

class LitertlmManifest {
  final String schemaVersion;
  final String repo;
  final String displayName;
  final int? contextLength;
  final Capabilities capabilities;
  final Map<String, dynamic>? sessionDefaults;
  final List<Variant> variants;

  LitertlmManifest._(this.schemaVersion, this.repo, this.displayName, this.contextLength,
      this.capabilities, this.sessionDefaults, this.variants);

  factory LitertlmManifest.fromJson(dynamic input) {
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
    );
  }

  /// The model's declared thinking markers (exact strings, whitespace included).
  ThinkingChannel? get thinkingMarkers =>
      capabilities.thinkingDeclared ? capabilities.thinkingChannel : null;

  /// Pick the variant + backend for a device. v0.1 algorithm, deterministic —
  /// identical to the TypeScript reference:
  ///
  /// 1. A variant with a `recommended` entry matching [platform] wins; matching
  ///    [deviceClass] too ranks higher. The recommendation's backend is used
  ///    unless the caller requested one the variant also lists.
  /// 2. Otherwise a variant listing the requested [backend] wins.
  /// 3. Otherwise the first variant, on its default_backend (else "cpu").
  ///
  /// Ties break toward the smaller file. Never returns a backend absent from
  /// the variant's verified `backends` list.
  Resolution resolve({String? platform, String? backend, String? deviceClass}) {
    _Scored best = _Scored(variants.first, 'cpu', -1, 'fallback: first variant');
    final scored = <_Scored>[];
    for (final v in variants) {
      var score = 0;
      String? chosen;
      var reason = 'fallback: first variant';
      if (platform != null && v.recommended.isNotEmpty) {
        final recs = v.recommended.where((r) => r.platform == platform).toList();
        Recommendation? classRec;
        if (deviceClass != null) {
          for (final r in recs) {
            if (r.deviceClass == deviceClass) classRec = r;
          }
        }
        final rec = classRec ?? (recs.isNotEmpty ? recs.first : null);
        if (rec != null) {
          score = classRec != null ? 300 : 200;
          chosen = rec.backend;
          reason = 'recommended for $platform'
              '${classRec != null ? "/$deviceClass" : ""}: ${rec.reason ?? ""}';
        }
      }
      if (score == 0 && backend != null && v.backends.contains(backend)) {
        score = 100;
        chosen = backend;
        reason = 'supports requested backend $backend';
      }
      if (backend != null && v.backends.contains(backend)) chosen = backend;
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
    best = scored.first;
    final v = best.variant;
    return Resolution(
      file: v.file,
      url: 'https://huggingface.co/$repo/resolve/main/${Uri.encodeComponent(v.file)}',
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
