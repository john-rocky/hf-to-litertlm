import 'dart:convert';
import 'dart:io';

import 'package:litertlm_manifest/litertlm_manifest.dart';
import 'package:test/test.dart';

// Fixtures are the real shipped manifests in ../../manifest/examples/.
LitertlmManifest load(String name) {
  final path = '../../manifest/examples/$name';
  return LitertlmManifest.fromJson(
      jsonDecode(File(path).readAsStringSync()) as Map<String, dynamic>);
}

void main() {
  final lfm = load('litert-community__LFM2.5-1.2B-Instruct.json');
  final qwen = load('litert-community__Qwen3-4B-Thinking-2507.json');

  test('LFM android midrange -> GPU re-export on gpu', () {
    final r = lfm.resolve(platform: 'android', deviceClass: 'midrange-2023+')!;
    expect(r.file, 'LFM2.5-1.2B-Instruct_int4_gpu.litertlm');
    expect(r.backend, 'gpu');
  });

  test('LFM ios -> cpu, never Metal', () {
    final r = lfm.resolve(platform: 'ios')!;
    expect(r.backend, 'cpu');
  });

  test('explicit gpu request survives the variant pick', () {
    final r = lfm.resolve(platform: 'android', backend: 'gpu')!;
    expect(r.file, 'LFM2.5-1.2B-Instruct_int4_gpu.litertlm');
    expect(r.backend, 'gpu');
  });

  test('explicit backend wins over the platform recommendation; caveats surface in notes', () {
    final r = lfm.resolve(platform: 'ios', backend: 'gpu')!;
    expect(r.file, 'LFM2.5-1.2B-Instruct_int4_gpu.litertlm');
    expect(r.backend, 'gpu');
    expect(r.variant.backends, contains(r.backend));
    expect(r.notes.any((n) => n.contains('Metal')), isTrue);
  });

  test('explicit backend counts only recommendations naming it', () {
    final r = lfm.resolve(platform: 'macos', backend: 'cpu')!;
    expect(r.file, 'LFM2.5-1.2B-Instruct_int8.litertlm');
    expect(r.backend, 'cpu');
  });

  test('backend no variant lists resolves to null, never a substitute', () {
    expect(lfm.resolve(backend: 'npu'), isNull);
    expect(lfm.resolve(platform: 'android', backend: 'npu'), isNull);
  });

  test('gpu-only request picks a gpu-capable variant', () {
    final r = lfm.resolve(backend: 'gpu')!;
    expect(r.file, 'LFM2.5-1.2B-Instruct_int4_gpu.litertlm');
    expect(r.backend, 'gpu');
  });

  test('deviceClass with no matching entry falls back with a note in reason', () {
    final r = lfm.resolve(platform: 'android', deviceClass: 'budget-2019')!;
    expect(r.reason, contains('no budget-2019 entry; using the midrange recommendation'));
  });

  test('resolution URLs follow the fetched revision', () {
    final base = {
      'manifest_schema': '0.1.0',
      'repo': 'test/rev',
      'generated': '2026-08-27',
      'model': {'display_name': 'Rev'},
      'variants': [
        {
          'file': 'a.litertlm',
          'quantization': 'q',
          'backends': ['cpu'],
        },
      ],
    };
    final pinned = LitertlmManifest.fromJson(base, revision: 'abc123');
    expect(pinned.resolve()!.url, contains('/resolve/abc123/'));
    expect(pinned.resolve(revision: 'deadbeef')!.url, contains('/resolve/deadbeef/'));
    expect(lfm.resolve()!.url, contains('/resolve/main/'));
  });

  test('parse rejects a variant with no backends (schema minItems: 1)', () {
    Map<String, dynamic> withVariants(List<Map<String, dynamic>> vs) => {
          'manifest_schema': '0.1.0',
          'repo': 't/x',
          'generated': '2026-08-27',
          'model': {'display_name': 'X'},
          'variants': vs,
        };
    expect(
        () => LitertlmManifest.fromJson(withVariants([
              {'file': 'a.litertlm', 'quantization': 'q', 'backends': <String>[]},
            ])),
        throwsFormatException);
    expect(
        () => LitertlmManifest.fromJson(withVariants([
              {'file': 'a.litertlm', 'quantization': 'q'},
            ])),
        throwsFormatException);
  });

  test('parse checks string-list elements eagerly: a non-string in backends, '
      'platform_notes or known_issues fails at parse, not inside resolve()', () {
    Map<String, dynamic> withVariant(Map<String, dynamic> v) => {
          'manifest_schema': '0.1.0',
          'repo': 't/x',
          'generated': '2026-08-29',
          'model': {'display_name': 'X'},
          'variants': [v],
        };
    for (final v in <Map<String, dynamic>>[
      {'file': 'a.litertlm', 'quantization': 'q', 'backends': ['cpu', 42]},
      {
        'file': 'a.litertlm',
        'quantization': 'q',
        'backends': ['cpu'],
        'requirements': {'platform_notes': ['ok', 42]},
      },
      {
        'file': 'a.litertlm',
        'quantization': 'q',
        'backends': ['cpu'],
        'known_issues': ['ok', 42],
      },
    ]) {
      expect(() => LitertlmManifest.fromJson(withVariant(v)), throwsA(isA<TypeError>()),
          reason: jsonEncode(v));
    }
  });

  test('Qwen thinking markers keep exact whitespace', () {
    final t = qwen.thinkingMarkers;
    expect(t, isNotNull);
    expect(t!.start, '<think>\n');
    expect(t.end, '\n</think>');
  });

  test('Qwen session defaults carry the 2048 output budget', () {
    final r = qwen.resolve(platform: 'macos')!;
    expect(r.sessionDefaults?['max_output_tokens_min'], 2048);
  });

  test('Qwen ios recommendation picks the block-128 file', () {
    final r = qwen.resolve(platform: 'ios')!;
    expect(r.file, 'model.litertlm');
  });

  test('0.1.1 declared channel set flows through, tool-call included; absent -> empty', () {
    final m = LitertlmManifest.fromJson({
      'manifest_schema': '0.1.1',
      'repo': 'test/channels',
      'generated': '2026-08-27',
      'model': {
        'display_name': 'Channels',
        'capabilities': {
          'thinking': {
            'declared': true,
            'channel': {'start': '<think>', 'end': '</think>'},
          },
          'channels': [
            {'name': 'thought', 'start': '<think>', 'end': '</think>', 'is_reasoning': true},
            {'name': 'tool_call', 'start': '<tool_call>', 'end': '</tool_call>'},
          ],
        },
      },
      'variants': [
        {
          'file': 'a.litertlm',
          'quantization': 'q',
          'backends': ['cpu'],
        },
      ],
    });
    expect(m.declaredChannels.length, 2);
    expect(m.declaredChannels[1].name, 'tool_call');
    expect(m.declaredChannels[0].isReasoning, isTrue);
    expect(lfm.declaredChannels, isEmpty);
  });

  test('recommended row naming an unverified backend is ignored', () {
    final m = LitertlmManifest.fromJson({
      'manifest_schema': '0.1.0',
      'repo': 'test/malformed',
      'generated': '2026-08-26',
      'model': {'display_name': 'Malformed'},
      'variants': [
        {
          'file': 'm.litertlm',
          'quantization': 'int8',
          'backends': ['cpu'],
          'default_backend': 'cpu',
          'recommended': [
            {'platform': 'android', 'backend': 'gpu'},
          ],
        },
      ],
    });
    final r = m.resolve(platform: 'android')!;
    expect(r.backend, 'cpu');
  });

  test('nested variant paths keep their structure in the url', () {
    final m = LitertlmManifest.fromJson({
      'manifest_schema': '0.1.0',
      'repo': 'test/nested',
      'generated': '2026-08-27',
      'model': {'display_name': 'Nested'},
      'variants': [
        {
          'file': 'int4/model v2.litertlm',
          'quantization': 'int4',
          'backends': ['cpu'],
          'default_backend': 'cpu',
        },
      ],
    });
    expect(
        m.resolve()!.url,
        'https://huggingface.co/test/nested/resolve/main/'
        'int4/model%20v2.litertlm');
  });

  test('identity fields survive', () {
    final r = lfm.resolve(platform: 'android')!;
    expect(r.variant.sha256, matches(RegExp(r'^[0-9a-f]{64}$')));
    expect(r.variant.sizeBytes, greaterThan(0));
  });
}
