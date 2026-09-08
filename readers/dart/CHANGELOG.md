# Changelog

## 0.2.2 — 2026-09-08 (source release)

- Report missing, empty or non-string `repo` and variant `file` values with descriptive `FormatException`s.
- Add optional `sourceRepo` to `LitertlmManifest.fromJson` so forked manifests resolve downloads from their fetch source. Existing calls still use the manifest's `repo`; revision precedence is unchanged.
- Exclude backend lists emptied after parsing; return `null` if no candidates remain.
- Document local path installation and include the package license and third-party notice.

Adapted from Hugh (@hung-yueh)'s [manifest resolver hardening](https://github.com/hung-yueh/react-native-litert-lm/commit/09f44d348014a213593d40f63ff784e32b70dde3), released in [react-native-litert-lm 0.7.0](https://github.com/hung-yueh/react-native-litert-lm/releases/tag/v0.7.0) under MIT. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Thinking markers and declared channels retain their existing behavior.
