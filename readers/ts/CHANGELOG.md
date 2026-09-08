# Changelog

## 0.2.2 — 2026-09-08 (source release)

- Reject missing, empty or non-string `repo` and variant `file` values with descriptive parse errors.
- Use the source repo passed to `fetchManifest` for download URLs, preserving fetched revisions and per-resolution revision overrides.
- Exclude empty backend lists from resolution, including hand-built manifests; return `null` if no candidates remain.
- Build JS and declarations before `npm pack`; include this changelog, license and third-party notice in the archive.

Adapted from Hugh (@hung-yueh)'s [manifest resolver hardening](https://github.com/hung-yueh/react-native-litert-lm/commit/09f44d348014a213593d40f63ff784e32b70dde3), released in [react-native-litert-lm 0.7.0](https://github.com/hung-yueh/react-native-litert-lm/releases/tag/v0.7.0) under MIT. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). The reference fetch error and thinking/channel APIs retain their existing behavior.
