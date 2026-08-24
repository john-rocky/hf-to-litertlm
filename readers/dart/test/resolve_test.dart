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
    final r = lfm.resolve(platform: 'android', deviceClass: 'midrange-2023+');
    expect(r.file, 'LFM2.5-1.2B-Instruct_int4_gpu.litertlm');
    expect(r.backend, 'gpu');
  });

  test('LFM ios -> cpu, never Metal', () {
    final r = lfm.resolve(platform: 'ios');
    expect(r.backend, 'cpu');
  });

  test('resolved backend is always in the verified backends list', () {
    final r = lfm.resolve(platform: 'ios', backend: 'gpu');
    expect(r.variant.backends, contains(r.backend));
  });

  test('gpu-only request picks a gpu-capable variant', () {
    final r = lfm.resolve(backend: 'gpu');
    expect(r.file, 'LFM2.5-1.2B-Instruct_int4_gpu.litertlm');
    expect(r.backend, 'gpu');
  });

  test('Qwen thinking markers keep exact whitespace', () {
    final t = qwen.thinkingMarkers;
    expect(t, isNotNull);
    expect(t!.start, '<think>\n');
    expect(t.end, '\n</think>');
  });

  test('Qwen session defaults carry the 2048 output budget', () {
    final r = qwen.resolve(platform: 'macos');
    expect(r.sessionDefaults?['max_output_tokens_min'], 2048);
  });

  test('Qwen ios recommendation picks the block-128 file', () {
    final r = qwen.resolve(platform: 'ios');
    expect(r.file, 'model.litertlm');
  });

  test('identity fields survive', () {
    final r = lfm.resolve(platform: 'android');
    expect(r.variant.sha256, matches(RegExp(r'^[0-9a-f]{64}$')));
    expect(r.variant.sizeBytes, greaterThan(0));
  });
}
