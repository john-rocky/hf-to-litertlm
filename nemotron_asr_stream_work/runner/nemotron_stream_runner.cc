// nemotron_stream_runner: the cache-aware streaming runner of the nvidia/nemotron-3.5-asr-streaming-0.6b LiteRT graphs
// (320 ms chunks at lookahead 3, the host's greedy RNN-T loop) for Android (and macOS), written against the LiteRT 2.3.0
// C API (CompiledModel). The I/O helpers, the accelerator options, the fixture reading and the stdout shape follow
// our runner for another ASR conversion; the chunking and the decoder loop are this model's (the processor's
// streaming rule, the right-aligned KV window with its additive mask, pure RNN-T).
// Graphs: seven roles, model[0..6]; a file with one signature serves its role whatever the signature's name, a file with
// several serves each role by the signature named as the role (mel.tflite = mel_first + mel, pre.tflite =
// pre_first + pre, dec_*.tflite = predict + joint). Two roles naming one file on one accelerator share one compiled model.
//   model[0] conformer (--model or --conformer, on --accel): embeds [1,Q,1024] + neg_mask [1,1,Q,S+Q] + k / v cache
//            [L,H,S,D] + conv cache [L,1024,K-1] + one_hot [1,P] -> enc_proj [1,Q,640] + the three caches
//   model[1] mel_first (--mel-first): PCM [A0] -> log-mel [F0,128]          (A0 = 4040, F0 = 25 at lookahead 3)
//   model[2] mel       (--mel):       PCM [A] -> log-mel [F,128]            (A = 5520, F = 32)
//   model[3] pre_first (--pre-first): log-mel [F0,128] -> embeds [Q,1024] + 3 subsampling caches
//   model[4] pre       (--pre):       log-mel [F,128] + 3 caches -> embeds + 3 caches
//   model[5] predict   (--predict):   token [1,1] (i32 or i64) + h / c [2,1,640] -> dec_out [640] + h / c
//   model[6] joint     (--joint):     dec_out [640] + enc_frame [640] -> logits [V] (blank = V - 1 unless --blank)
// model[0] is compiled first (the Environment's first accelerated load is the conformer); the others get --mel-accel /
// --pre-accel / --dec-accel (default cpu). Tensors are bound by name (the contract's names and their spellings), else by
// a unique element count, else (two tensors of one size, such as k / v or h / c) by litert-torch's positional names
// args_<n> / output_<n> in the contract's order; every binding is printed.
// Chunking (the processor's streaming rule): an utterance of N samples gets A zeros appended
// (--tail pad; --tail none = the card snippet's loop without them); chunk 0 = samples [0, A0) through mel_first and
// pre_first; chunk k >= 1 = samples [s_k, s_k + A) with s_k = (F0 + F (k - 1)) hop - n_fft / 2, through mel and pre;
// chunk k runs while s_k + A < N + A (pad) or < N (none). The chunk geometry is read from the graphs (A0, A, F0, F, Q, S)
// and checked against hop 160, n_fft 512, win 400. Per chunk: mel -> pre -> conformer (neg_mask: the slots
// [0, S - min(S, Q k)) of every query row = --mask-neg, default -30000 = graphs.py's NEG; the other slots 0; the
// caches move to the next chunk) -> the RNN-T loop
// on its Q frames. Per utterance the caches are zeroed and the decoder starts from predict(blank, h = c = 0); the
// decoder state (dec_out, h, c, the last token) runs on across chunks and frames. Loop: logits = joint(dec_out,
// enc[t]); token = argmax (the first maximum; a NaN counts as the maximum); blank -> next frame; else the token is
// emitted, predict(token) runs and the frame stays, --max-symbols (10) symbols on one frame -> next frame.
// --state copy (default): each cache output is copied into its input buffer after the call (both locked, one memcpy);
// --state swap: two buffer sets of each conformer cache from the joined input / output requirements, alternated per
// chunk (no copy).
//
// usage: nemotron_stream_runner --accel cpu|gpu|gpu+cpu|npu|npu+cpu [--gpu-precision fp16|fp32|fp16acc32|default] [--burst 0|1]
//            --libdir DIR [--cache-dir DIR] --fixtures DIR --count N [--warmup W] [--rounds R] [--ab-fixture ID]
//            [--dump full|none|enc] --out DIR --model CONFORMER.tflite --mel-first F --mel F --pre-first F --pre F
//            --predict F --joint F [--mel-accel A] [--pre-accel A] [--dec-accel A] [--threads N] [--mel-threads N]
//            [--pre-threads N] [--dec-threads N] [--lang LOCALE] [--prompt-id N] [--state copy|swap] [--tail pad|none]
//            [--max-symbols 10] [--blank ID] [--mask-neg V] [--hop 160] [--n-fft 512] [--win 400]
//            [--conf-xnn-flags U32] [--dec-xnn-flags U32] [--htp-opt-level 0|1|2] [--conf-mode split|fused]
// CPU and NPU options: --conf-xnn-flags / --dec-xnn-flags add `flags = N` to the `xnnpack`
// TOML of the conformer's / the decoder's (predict + joint) compiled model (LiteRT 2.3.0 litert_cpu_options.cc: the key
// overwrites TfLiteXNNPackDelegateOptionsDefault()'s flags, so N carries the defaults too: QS8 | QU8 = 3 in the v2.3.0
// BUILD defaults, FORCE_FP16 = 4); --htp-opt-level adds `optimization_level = N` to the `qualcomm` TOML of an NPU model
// (0 = kHtpOptimizeForInference = QNN's finalize level 2, 1 = Prepare = 1, 2 = InferenceO3 = 3 = LiteRT 2.3.0's
// default). Without these flags neither key is written; with any of them stdout gets
// an `r3_options` line and each model[k]= line its `xnn_flags=` / `htp_opt_level=`.
// Fused mode: --conf-mode fused = model[0] is the fused pre-encode + conformer graph (scripts/export_r4_fused.py,
// signature conformer_pre: mel [1,F,128] + cache0 / 1 / 2 + pre_m0 [1,1,R0,1] + pre_m1 [1,1,R1,1] + the conformer's
// inputs but x -> enc_proj + the conformer's caches + cache0_out / 1_out / 2_out); --pre-first / --pre are not read and
// no pre graph is loaded (roles 3 / 4 print `fused_into=model[0]`). Per chunk: mel_first / mel (as before) -> the fused
// graph; chunk 0 = mel_first's F0 rows behind F - F0 zero rows, the pre-encode caches zero, pre_m0 / pre_m1 zero on their
// first R0 - (F0 + 1) / 2 / R1 - (F0 + 3) / 4 rows (the rows graphs.PreFirst pads; scripts/contract_r4.py), 1 elsewhere
// and on every later chunk; the three pre-encode caches move with the conformer's (--state copy / swap, six pairs).
// pre_ms = 0 (the pre-encode runs inside conf_ms). The default, --conf-mode split, runs pre and the conformer as two
// graphs; on stdout, fused adds an `r4_options conf_mode=fused` line, a
// `binding fused ...` line and a `state pre caches ...` line.
// --fixtures DIR: DIR/index.tsv (one line per utterance: id3 <TAB> name <TAB> file [<TAB> locale [<TAB> anything]];
// id3 = 000, 001, ...; # starts a comment line) and the 16 kHz audio files it names (WAV PCM 16-bit / IEEE float 32-bit,
// channels averaged, or raw float32). The locale column, when present, picks the utterance's language prompt; else
// --lang; --prompt-id overrides both. --count N streams 000 .. N-1; 000 is streamed once more first as a warm-up.
// stdout: pid= first; per graph its signature tensors, compile_start / compile_end model[k] wall_ms= and a model[k]=
// line; the bindings; the chunk geometry; one JSON line per utterance ({"kind":"utt"}: the token ids "tok" with their
// encoder frame "frame" (chunk x Q + frame in the chunk) and symbol index on that frame "sym", the chunk count, and per
// chunk the wall ms "chunk_ms" (from the PCM write of the chunk to the end of its RNN-T loop) and the stage ms
// "mel_ms" / "pre_ms" / "conf_ms" / "dec_ms" (each = input writes + run + output reads, the conformer's with the cache
// move), each rounded to 0.001 ms); then {"kind":"stream_summary"} over the counted utterances (the warm-up and the
// loads excluded: chunk_ms p50 / p90 (linear interpolation, numpy's default, on the rounded values), rtf_stream =
// sum chunk_ms / 1000 / sum audio seconds, the stage sums and medians), a parity_summary line, the latency phase
// (--rounds R > 0 on fixture --ab-fixture: per graph W + R calls on the inputs of one steady chunk, `round r model[k]`
// lines and `summary model[k]` lines) and done wall_ms=. FATAL lines on stdout; LiteRT's own log on stderr.
// --dump full writes <out>/model0/<id3>.f32 per utterance: n_chunks, n_tokens, truncated (tokens past 1024), the encoder
// frames, then (token, frame, sym) x 1024, -1 padded, as float32; --dump enc also writes <out>/enc/<id3>.f32 = the
// utterance's enc_proj frames [chunks x Q, 640] as float32.
#include <unistd.h>
#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <limits>
#include <string>
#include <utility>
#include <vector>
#include <sys/stat.h>
#include <time.h>
#include "litert/c/litert_common.h"
#include "litert/c/litert_compiled_model.h"
#include "litert/c/litert_environment.h"
#include "litert/c/litert_environment_options.h"
#include "litert/c/litert_layout.h"
#include "litert/c/litert_model.h"
#include "litert/c/litert_model_types.h"
#include "litert/c/litert_opaque_options.h"
#include "litert/c/litert_options.h"
#include "litert/c/litert_tensor_buffer.h"
#include "litert/c/litert_tensor_buffer_requirements.h"
#include "litert/c/litert_tensor_buffer_types.h"
namespace {
using Clock = std::chrono::steady_clock;
#define LRT_CHECK(expr) do { LiteRtStatus _st = (expr); if (_st != kLiteRtStatusOk) { std::fprintf(stderr, "ERROR: %s failed: %d (%s)\n", #expr, (int)_st, LiteRtGetStatusString(_st)); std::printf("FATAL %s status=%d\n", #expr, (int)_st); std::fflush(stdout); std::exit(2); } } while (0)
const char* kContract = "v1";        // the file contract this build follows (the graphs' roles, names and shapes above)
constexpr int kMel = 128;            // mel bins
constexpr int kHid = 1024;           // conformer width (embeds)
constexpr int kEnc = 640;            // enc_proj / dec_out width
constexpr int kDecState = 2 * 640;   // h / c of the 2-layer LSTM
constexpr int kMaxRecTokens = 1024;  // tokens kept per utterance in the dump record
constexpr int kRecElems = 4 + 3 * kMaxRecTokens;

void DeleteCStringPayload(void* p) { delete[] static_cast<char*>(p); }
double Ms(Clock::time_point a, Clock::time_point b) { return std::chrono::duration<double, std::milli>(b - a).count(); }
double R3(double v) { return std::round(v * 1000.0) / 1000.0; }  // the 0.001 ms the stdout carries
double Median(std::vector<double> v) { std::sort(v.begin(), v.end()); size_t n = v.size(); return n ? ((n % 2) ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2])) : 0.0; }
double Pct(std::vector<double> v, double q) {  // numpy.percentile(v, q), method "linear"
  if (v.empty()) return 0.0; std::sort(v.begin(), v.end());
  double pos = q / 100.0 * (double)(v.size() - 1); size_t lo = (size_t)std::floor(pos), hi = std::min(lo + 1, v.size() - 1);
  return v[lo] + (pos - (double)lo) * (v[hi] - v[lo]);
}
long long WallMs() { struct timespec ts; clock_gettime(CLOCK_REALTIME, &ts); return (long long)ts.tv_sec * 1000 + ts.tv_nsec / 1000000; }
bool ReadFile(const std::string& path, std::vector<uint8_t>& b) {
  FILE* f = std::fopen(path.c_str(), "rb"); if (!f) return false;
  std::fseek(f, 0, SEEK_END); long sz = std::ftell(f); std::fseek(f, 0, SEEK_SET);
  b.resize(sz > 0 ? (size_t)sz : 0); bool ok = b.empty() || std::fread(b.data(), 1, b.size(), f) == b.size(); std::fclose(f); return ok;
}
bool WriteFloatFile(const std::string& path, const float* v, size_t n) {
  FILE* f = std::fopen(path.c_str(), "wb"); if (!f) return false;
  bool ok = n == 0 || std::fwrite(v, 4, n, f) == n; std::fclose(f); return ok;
}
uint16_t U16(const uint8_t* p) { return (uint16_t)(p[0] | (p[1] << 8)); }
uint32_t U32(const uint8_t* p) { return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24); }
// 16 kHz mono audio -> float32 samples. WAV: PCM 16-bit (/ 32768) or IEEE float 32-bit (also WAVE_FORMAT_EXTENSIBLE);
// more than one channel is averaged. Any other file: raw little-endian float32.
bool ReadAudio(const std::string& path, std::vector<float>& x, std::string& fmt) {
  std::vector<uint8_t> b; if (!ReadFile(path, b)) { fmt = "unreadable"; return false; }
  if (b.size() < 12 || std::memcmp(b.data(), "RIFF", 4) != 0 || std::memcmp(b.data() + 8, "WAVE", 4) != 0) {
    if (b.size() % 4) { fmt = "raw size not a multiple of 4"; return false; }
    x.resize(b.size() / 4); std::memcpy(x.data(), b.data(), b.size()); fmt = "raw_f32"; return true;
  }
  int tag = -1, ch = 0, bits = 0; uint32_t rate = 0; const uint8_t* data = nullptr; size_t dlen = 0;
  for (size_t p = 12; p + 8 <= b.size();) {
    uint32_t len = U32(b.data() + p + 4); const uint8_t* c = b.data() + p + 8;
    if (p + 8 + (size_t)len > b.size()) len = (uint32_t)(b.size() - p - 8);
    if (std::memcmp(b.data() + p, "fmt ", 4) == 0 && len >= 16) {
      tag = U16(c); ch = U16(c + 2); rate = U32(c + 4); bits = U16(c + 14);
      if (tag == 0xFFFE && len >= 26) tag = U16(c + 24);  // the sub-format GUID's first two bytes
    } else if (std::memcmp(b.data() + p, "data", 4) == 0) { data = c; dlen = len; }
    p += 8 + (size_t)len + (len & 1);
  }
  if (!data || ch < 1 || rate != 16000) { fmt = "wav without data / not 16 kHz (rate " + std::to_string(rate) + ")"; return false; }
  size_t frames;
  if (tag == 1 && bits == 16) {
    frames = dlen / (2 * (size_t)ch); x.assign(frames, 0.f);
    for (size_t i = 0; i < frames; ++i) { float s = 0; for (int k = 0; k < ch; ++k) s += (float)(int16_t)U16(data + 2 * (i * ch + k)) / 32768.0f; x[i] = ch == 1 ? s : s / (float)ch; }
    fmt = "wav_pcm16";
  } else if (tag == 3 && bits == 32) {
    frames = dlen / (4 * (size_t)ch); x.assign(frames, 0.f);
    for (size_t i = 0; i < frames; ++i) { float s = 0; for (int k = 0; k < ch; ++k) { float v; std::memcpy(&v, data + 4 * (i * ch + k), 4); s += v; } x[i] = ch == 1 ? s : s / (float)ch; }
    fmt = "wav_f32";
  } else { fmt = "wav format " + std::to_string(tag) + " / " + std::to_string(bits) + " bit"; return false; }
  if (ch > 1) fmt += "_mean" + std::to_string(ch);
  return true;
}
// IEEE 754 binary16 <-> binary32 (round to nearest even), for graphs whose float I/O is float16
uint16_t ToHalf(float f) {
  uint32_t x; std::memcpy(&x, &f, 4); uint32_t sign = (x >> 16) & 0x8000u; int32_t exp = (int32_t)((x >> 23) & 0xff) - 127 + 15; uint32_t man = x & 0x7fffffu;
  if (((x >> 23) & 0xff) == 0xff) return (uint16_t)(sign | 0x7c00u | (man ? 0x200u : 0u));
  if (exp >= 31) return (uint16_t)(sign | 0x7c00u);
  if (exp <= 0) {
    if (exp < -10) return (uint16_t)sign;
    man |= 0x800000u; uint32_t shift = (uint32_t)(14 - exp); uint32_t h = man >> shift; uint32_t rem = man & ((1u << shift) - 1u), half = 1u << (shift - 1);
    if (rem > half || (rem == half && (h & 1u))) ++h; return (uint16_t)(sign | h);
  }
  uint32_t h = ((uint32_t)exp << 10) | (man >> 13); uint32_t rem = man & 0x1fffu;
  if (rem > 0x1000u || (rem == 0x1000u && (h & 1u))) ++h; return (uint16_t)(sign | h);
}
float FromHalf(uint16_t h) {
  uint32_t sign = (uint32_t)(h & 0x8000u) << 16, exp = (h >> 10) & 0x1fu, man = h & 0x3ffu, x;
  if (exp == 0) {
    if (man == 0) x = sign;
    else { int e = -1; do { ++e; man <<= 1; } while (!(man & 0x400u)); x = sign | ((uint32_t)(127 - 15 - e) << 23) | ((man & 0x3ffu) << 13); }
  } else if (exp == 31) x = sign | 0x7f800000u | (man << 13);
  else x = sign | ((exp - 15 + 127) << 23) | (man << 13);
  float f; std::memcpy(&f, &x, 4); return f;
}
void AddToml(LiteRtOptions options, const char* identifier, const std::string& toml) {
  char* payload = new char[toml.size() + 1]; std::memcpy(payload, toml.data(), toml.size()); payload[toml.size()] = '\0';
  LiteRtOpaqueOptions opaque = nullptr; LRT_CHECK(LiteRtCreateOpaqueOptions(identifier, payload, DeleteCStringPayload, &opaque)); LRT_CHECK(LiteRtAddOpaqueOptions(options, opaque));
}
std::string Dims(const LiteRtRankedTensorType& t) {
  std::string s = "["; for (unsigned i = 0; i < t.layout.rank; ++i) { if (i) s += ","; s += std::to_string(t.layout.dimensions[i]); } return s + "]";
}
const char* TypeName(LiteRtElementType e) {
  return e == kLiteRtElementTypeFloat32 ? "f32" : e == kLiteRtElementTypeFloat16 ? "f16" : e == kLiteRtElementTypeInt32 ? "i32" : e == kLiteRtElementTypeInt64 ? "i64" : "other";
}
size_t ElemBytes(LiteRtElementType e) { return e == kLiteRtElementTypeFloat16 ? 2 : e == kLiteRtElementTypeInt64 ? 8 : 4; }
// the language prompts of the checkpoint's processor_config.json (ea30d66, sha256 ec47870f1091ea4f..., 121 entries)
const std::pair<const char*, int> kPrompts[] = {
  {"af-ZA", 54}, {"am-ET", 49}, {"ar", 7}, {"ar-AR", 7}, {"auto", 101}, {"ay-BO", 81}, {"az-AZ", 66}, {"bg", 30}, {"bg-BG", 30}, {"bn-IN", 36}, {"cs", 22}, {"cs-CZ", 22},
  {"da", 25}, {"da-DK", 25}, {"de", 9}, {"de-DE", 9}, {"el", 21}, {"el-GR", 21}, {"en", 0}, {"en-GB", 1}, {"en-US", 0}, {"enGB", 1}, {"es", 3}, {"es-ES", 2}, {"es-US", 3},
  {"esES", 2}, {"et", 60}, {"et-EE", 60}, {"fa-IR", 38}, {"fi", 26}, {"fi-FI", 26}, {"fr", 8}, {"fr-CA", 100}, {"fr-FR", 8}, {"gn-PY", 82}, {"gu-IN", 42}, {"ha-NG", 50},
  {"haw-US", 97}, {"he-IL", 64}, {"hi", 6}, {"hi-HI", 6}, {"hi-IN", 6}, {"hr", 29}, {"hr-HR", 29}, {"hu", 23}, {"hu-HU", 23}, {"hy-AM", 68}, {"id-ID", 34}, {"ig-NG", 53},
  {"it", 15}, {"it-IT", 15}, {"ja-JA", 10}, {"ja-JP", 10}, {"ka-GE", 67}, {"km-KH", 47}, {"kn-IN", 43}, {"ko", 14}, {"ko-KO", 14}, {"ko-KR", 14}, {"ku-TR", 65}, {"ky-KG", 71},
  {"ln-CD", 58}, {"lt", 31}, {"lt-LT", 31}, {"lv", 61}, {"lv-LV", 61}, {"mi-NZ", 96}, {"ml-IN", 44}, {"mr-IN", 41}, {"ms-MY", 35}, {"mt-MT", 102}, {"nah-MX", 83}, {"nb", 103},
  {"nb-NO", 103}, {"ne-NP", 46}, {"nl", 16}, {"nl-NL", 16}, {"nn", 104}, {"nn-NO", 104}, {"no", 27}, {"no-NO", 27}, {"ny-MW", 57}, {"or-KE", 59}, {"pl", 17}, {"pl-PL", 17},
  {"pt", 13}, {"pt-BR", 12}, {"pt-PT", 13}, {"qu-PE", 80}, {"ro", 20}, {"ro-RO", 20}, {"ru", 11}, {"ru-RU", 11}, {"rw-RW", 55}, {"si-LK", 45}, {"sk", 28}, {"sk-SK", 28},
  {"sl", 62}, {"sl-SI", 62}, {"sm-WS", 98}, {"so-SO", 56}, {"sv", 24}, {"sv-SE", 24}, {"sw-KE", 48}, {"ta-IN", 39}, {"te-IN", 40}, {"tg-TJ", 70}, {"th-TH", 32}, {"to-TO", 99},
  {"tr", 18}, {"tr-TR", 18}, {"uk", 19}, {"uk-UA", 19}, {"ur-PK", 37}, {"uz-UZ", 69}, {"vi-VN", 33}, {"yo-NG", 52}, {"zh-CN", 4}, {"zh-TW", 5}, {"zh-ZH", 4}, {"zu-ZA", 51},
};
int PromptOf(const std::string& locale) { for (auto& p : kPrompts) if (locale == p.first) return p.second; return -1; }

struct Sig {  // one signature of a compiled model: its tensors in signature order and their managed buffers
  LiteRtParamIndex index = 0; std::string key;
  std::vector<std::string> in_names, out_names; std::vector<LiteRtRankedTensorType> in_types, out_types;
  std::vector<size_t> in_elems, out_elems; std::vector<LiteRtTensorBuffer> ins, outs;
};
struct M {  // one loaded file: its compiled model, its signatures and their managed buffers (shared by the roles it serves)
  std::string path, accel; int threads = 0; long long xflags = -1; LiteRtModel model = nullptr; LiteRtOptions opts = nullptr; LiteRtCompiledModel cm = nullptr;
  std::vector<Sig> sigs; double compile_ms = 0; bool fully = false, non_cpu_fully = false;
  Sig& S(const std::string& key) {  // the signature named key; the only one when the file has one
    for (auto& s : sigs) if (s.key == key) return s;
    if (sigs.size() == 1) return sigs[0];
    std::printf("FATAL %s: no signature '%s' (has", path.c_str(), key.c_str()); for (auto& s : sigs) std::printf(" %s", s.key.c_str()); std::printf(")\n"); std::fflush(stdout); std::exit(2);
  }
};
// litert-torch's positional names: args_<n> (inputs) / output_<n> (outputs) -> n, else -1
int PosName(const std::string& s) {
  const char* pre[] = {"args_", "output_"};
  for (const char* p : pre) { size_t n = std::strlen(p); if (s.size() > n && s.compare(0, n, p) == 0 && s.find_first_not_of("0123456789", n) == std::string::npos) return std::atoi(s.c_str() + n); }
  return -1;
}
// one tensor of a signature: by the contract's names (several spellings; a name counts only with that element count);
// else the only tensor with that element count;
// else, when every tensor of that count has a positional name, the rank-th of them by position (the contract's order)
int Pick(const Sig& s, bool input, const std::vector<std::string>& names, size_t elems, const char* what, int rank = 0) {
  const auto& nm = input ? s.in_names : s.out_names; const auto& e = input ? s.in_elems : s.out_elems;
  int i = -1; const char* how = "name";
  for (auto& n : names) { for (size_t k = 0; k < nm.size() && i < 0; ++k) if (nm[k] == n && e[k] == elems) i = (int)k; if (i >= 0) break; }
  if (i < 0) {
    std::vector<int> hit; for (size_t k = 0; k < e.size(); ++k) if (e[k] == elems) hit.push_back((int)k);
    if (hit.size() == 1) { i = hit[0]; how = "size"; }
    else if (hit.size() > 1) {
      bool all_pos = true; for (int k : hit) all_pos = all_pos && PosName(nm[k]) >= 0;
      if (all_pos && rank < (int)hit.size()) { std::sort(hit.begin(), hit.end(), [&](int a, int b) { return PosName(nm[a]) < PosName(nm[b]); }); i = hit[rank]; how = "position"; }
    }
  }
  if (i < 0 || e[i] != elems) {
    std::printf("FATAL signature %s: %s %s (%zu elements) not found by name, a unique element count or position; tensors:", s.key.c_str(), input ? "input" : "output", what, elems);
    for (size_t k = 0; k < nm.size(); ++k) std::printf(" %s(%zu)", nm[k].c_str(), e[k]);
    std::printf("\n"); std::fflush(stdout); std::exit(2);
  }
  if (std::strcmp(how, "name") != 0) std::printf("binding note: %s %s %s = %s (by %s)\n", s.key.c_str(), input ? "input" : "output", what, nm[i].c_str(), how);
  return i;
}
void Write(LiteRtTensorBuffer b, const void* src, size_t bytes) { void* h = nullptr; LRT_CHECK(LiteRtLockTensorBuffer(b, &h, kLiteRtTensorBufferLockModeWrite)); std::memcpy(h, src, bytes); LRT_CHECK(LiteRtUnlockTensorBuffer(b)); }
void Read(LiteRtTensorBuffer b, void* dst, size_t bytes) { void* h = nullptr; LRT_CHECK(LiteRtLockTensorBuffer(b, &h, kLiteRtTensorBufferLockModeRead)); std::memcpy(dst, h, bytes); LRT_CHECK(LiteRtUnlockTensorBuffer(b)); }
void Zero(LiteRtTensorBuffer b, size_t bytes) { void* h = nullptr; LRT_CHECK(LiteRtLockTensorBuffer(b, &h, kLiteRtTensorBufferLockModeWrite)); std::memset(h, 0, bytes); LRT_CHECK(LiteRtUnlockTensorBuffer(b)); }
void Copy(LiteRtTensorBuffer dst, LiteRtTensorBuffer src, size_t bytes) {  // one buffer into another: both locked, one memcpy
  void* s = nullptr; void* d = nullptr;
  LRT_CHECK(LiteRtLockTensorBuffer(src, &s, kLiteRtTensorBufferLockModeRead)); LRT_CHECK(LiteRtLockTensorBuffer(dst, &d, kLiteRtTensorBufferLockModeWrite));
  std::memcpy(d, s, bytes); LRT_CHECK(LiteRtUnlockTensorBuffer(dst)); LRT_CHECK(LiteRtUnlockTensorBuffer(src));
}
// float32 data into / out of a float32 or float16 tensor
void WriteF(LiteRtTensorBuffer b, LiteRtElementType t, const float* v, size_t n) {
  if (t == kLiteRtElementTypeFloat32) { Write(b, v, n * 4); return; }
  std::vector<uint16_t> h(n); for (size_t i = 0; i < n; ++i) h[i] = ToHalf(v[i]); Write(b, h.data(), n * 2);
}
void ReadF(LiteRtTensorBuffer b, LiteRtElementType t, float* v, size_t n) {
  if (t == kLiteRtElementTypeFloat32) { Read(b, v, n * 4); return; }
  std::vector<uint16_t> h(n); Read(b, h.data(), n * 2); for (size_t i = 0; i < n; ++i) v[i] = FromHalf(h[i]);
}
double Run(M& m, Sig& s) { auto t = Clock::now(); LRT_CHECK(LiteRtRunCompiledModel(m.cm, s.index, s.ins.size(), s.ins.data(), s.outs.size(), s.outs.data())); return Ms(t, Clock::now()); }
std::vector<std::string> Split(const std::string& s, char sep) { std::vector<std::string> r; size_t p = 0; while (p <= s.size()) { size_t q = s.find(sep, p); if (q == std::string::npos) q = s.size(); r.push_back(s.substr(p, q - p)); p = q + 1; } return r; }
std::string JsonStr(const std::string& s) { std::string o = "\""; for (char c : s) { if (c == '"' || c == '\\') { o += '\\'; o += c; } else if ((unsigned char)c < 0x20) { char b[8]; std::snprintf(b, sizeof b, "\\u%04x", c); o += b; } else o += c; } return o + "\""; }
template <class T> std::string JsonArr(const std::vector<T>& v) { std::string o = "["; for (size_t i = 0; i < v.size(); ++i) { if (i) o += ","; o += std::to_string(v[i]); } return o + "]"; }
std::string F(double v, int prec = 3) { char b[64]; std::snprintf(b, sizeof b, "%.*f", prec, v); return b; }
std::string JsonMs(const std::vector<double>& v) { std::string o = "["; for (size_t i = 0; i < v.size(); ++i) { if (i) o += ","; o += F(v[i]); } return o + "]"; }
bool AccelOk(const std::string& a) { return a == "cpu" || a == "gpu" || a == "gpu+cpu" || a == "npu" || a == "npu+cpu"; }
LiteRtHwAcceleratorSet Hw(const std::string& a) {
  return a == "npu" ? kLiteRtHwAcceleratorNpu : a == "npu+cpu" ? (LiteRtHwAcceleratorSet)(kLiteRtHwAcceleratorNpu | kLiteRtHwAcceleratorCpu)
       : a == "gpu" ? kLiteRtHwAcceleratorGpu : a == "gpu+cpu" ? (LiteRtHwAcceleratorSet)(kLiteRtHwAcceleratorGpu | kLiteRtHwAcceleratorCpu) : kLiteRtHwAcceleratorCpu;
}
}  // namespace

int main(int argc, char** argv) {
  std::string accel = "cpu", mel_accel = "cpu", pre_accel = "cpu", dec_accel = "cpu", gpu_precision = "fp16", libdir, cache_dir, fixtures, out_dir, dump = "full";
  std::string conf_path, mel0_path, mel_path, pre0_path, pre_path, pred_path, joint_path, ab_fixture = "000", lang, state = "copy", tail = "pad";
  std::string conf_mode = "split";                    // split (pre and the conformer as two graphs) | fused (pre-encode inside model[0])
  int burst = 1, threads = 4, mel_threads = 0, pre_threads = 0, dec_threads = 0, prompt_id = -1;
  long warmup = 2, rounds = 0, count = 0, max_symbols = 10, blank = -1, hop = 160, n_fft = 512, win_len = 400;
  long long conf_xnn_flags = -1, dec_xnn_flags = -1;  // -1 = not given (the xnnpack TOML carries num_threads only)
  int htp_opt_level = -1;                             // -1 = not given (no optimization_level key)
  float mask_neg = -30000.f;
  auto parse_u32 = [](const char* what, const char* s) -> long long {
    char* end = nullptr; errno = 0; unsigned long long v = std::strtoull(s, &end, 0);
    if (errno || !end || *end || s[0] == '-' || v > 0xffffffffull) { std::fprintf(stderr, "%s takes an unsigned 32-bit value, got %s\n", what, s); std::exit(1); }
    return (long long)v;
  };
  for (int i = 1; i < argc; ++i) {
    std::string f = argv[i];
    auto next = [&](const char* what) -> const char* { if (i + 1 >= argc) { std::fprintf(stderr, "missing value for %s\n", what); std::exit(1); } return argv[++i]; };
    if (f == "--accel") accel = next("--accel");
    else if (f == "--mel-accel") mel_accel = next("--mel-accel");
    else if (f == "--pre-accel") pre_accel = next("--pre-accel");
    else if (f == "--dec-accel") dec_accel = next("--dec-accel");
    else if (f == "--gpu-precision") gpu_precision = next("--gpu-precision");
    else if (f == "--burst") burst = std::atoi(next("--burst"));
    else if (f == "--libdir") libdir = next("--libdir");
    else if (f == "--cache-dir") cache_dir = next("--cache-dir");
    else if (f == "--fixtures") fixtures = next("--fixtures");
    else if (f == "--count") count = std::strtol(next("--count"), nullptr, 10);
    else if (f == "--warmup") warmup = std::strtol(next("--warmup"), nullptr, 10);
    else if (f == "--rounds") rounds = std::strtol(next("--rounds"), nullptr, 10);
    else if (f == "--ab-fixture") ab_fixture = next("--ab-fixture");
    else if (f == "--out") out_dir = next("--out");
    else if (f == "--dump") dump = next("--dump");
    else if (f == "--model" || f == "--conformer") conf_path = next("--model");
    else if (f == "--mel-first") mel0_path = next("--mel-first");
    else if (f == "--mel") mel_path = next("--mel");
    else if (f == "--pre-first") pre0_path = next("--pre-first");
    else if (f == "--pre") pre_path = next("--pre");
    else if (f == "--predict") pred_path = next("--predict");
    else if (f == "--joint") joint_path = next("--joint");
    else if (f == "--threads") threads = std::atoi(next("--threads"));
    else if (f == "--mel-threads") mel_threads = std::atoi(next("--mel-threads"));
    else if (f == "--pre-threads") pre_threads = std::atoi(next("--pre-threads"));
    else if (f == "--dec-threads") dec_threads = std::atoi(next("--dec-threads"));
    else if (f == "--lang") lang = next("--lang");
    else if (f == "--prompt-id") prompt_id = std::atoi(next("--prompt-id"));
    else if (f == "--state") state = next("--state");
    else if (f == "--tail") tail = next("--tail");
    else if (f == "--max-symbols") max_symbols = std::strtol(next("--max-symbols"), nullptr, 10);
    else if (f == "--blank") blank = std::strtol(next("--blank"), nullptr, 10);
    else if (f == "--mask-neg") mask_neg = std::strtof(next("--mask-neg"), nullptr);
    else if (f == "--hop") hop = std::strtol(next("--hop"), nullptr, 10);
    else if (f == "--n-fft") n_fft = std::strtol(next("--n-fft"), nullptr, 10);
    else if (f == "--win") win_len = std::strtol(next("--win"), nullptr, 10);
    else if (f == "--conf-xnn-flags") conf_xnn_flags = parse_u32("--conf-xnn-flags", next("--conf-xnn-flags"));
    else if (f == "--dec-xnn-flags") dec_xnn_flags = parse_u32("--dec-xnn-flags", next("--dec-xnn-flags"));
    else if (f == "--htp-opt-level") {
      const char* v = next("--htp-opt-level");
      if (std::strcmp(v, "0") && std::strcmp(v, "1") && std::strcmp(v, "2")) { std::fprintf(stderr, "--htp-opt-level takes 0, 1 or 2\n"); return 1; }
      htp_opt_level = std::atoi(v);
    }
    else if (f == "--conf-mode") conf_mode = next("--conf-mode");
    else { std::fprintf(stderr, "unknown flag %s\n", f.c_str()); return 1; }
  }
  if (conf_mode != "split" && conf_mode != "fused") { std::fprintf(stderr, "--conf-mode takes split or fused\n"); return 1; }
  const bool fused = conf_mode == "fused";
  if (conf_path.empty() || mel0_path.empty() || mel_path.empty() || (!fused && (pre0_path.empty() || pre_path.empty())) || pred_path.empty() || joint_path.empty() ||
      libdir.empty() || fixtures.empty() || out_dir.empty() || count < 1) {
    std::fprintf(stderr, "need --model, --mel-first, --mel, --pre-first, --pre, --predict, --joint, --libdir, --fixtures, --count >= 1, --out\n"); return 1;
  }
  if (!AccelOk(accel) || !AccelOk(mel_accel) || !AccelOk(pre_accel) || !AccelOk(dec_accel)) { std::fprintf(stderr, "an accelerator is not cpu | gpu | gpu+cpu | npu | npu+cpu\n"); return 1; }
  if (state != "copy" && state != "swap") { std::fprintf(stderr, "--state takes copy or swap\n"); return 1; }
  if (tail != "pad" && tail != "none") { std::fprintf(stderr, "--tail takes pad or none\n"); return 1; }
  if (dump != "full" && dump != "none" && dump != "enc") { std::fprintf(stderr, "--dump takes full, none or enc\n"); return 1; }
  if (gpu_precision != "default" && gpu_precision != "fp16" && gpu_precision != "fp32" && gpu_precision != "fp16acc32") { std::fprintf(stderr, "unknown --gpu-precision %s\n", gpu_precision.c_str()); return 1; }
  if (mel_threads <= 0) mel_threads = threads;
  if (pre_threads <= 0) pre_threads = threads;
  if (dec_threads <= 0) dec_threads = threads;
  ::mkdir(out_dir.c_str(), 0775);
  std::printf("pid=%d accel=%s gpu_precision=%s burst=%d mel_accel=%s pre_accel=%s dec_accel=%s fixtures=%ld warmup=%ld rounds=%ld ab_fixture=%s libdir=%s cache_dir=%s dump=%s threads=%d mel_threads=%d pre_threads=%d dec_threads=%d "
              "lang=%s prompt_id=%d state=%s tail=%s max_symbols=%ld blank=%ld mask_neg=%g hop=%ld n_fft=%ld win=%ld contract=%s wall_ms=%lld\n",
              (int)getpid(), accel.c_str(), gpu_precision.c_str(), burst, mel_accel.c_str(), pre_accel.c_str(), dec_accel.c_str(), count, warmup, rounds, ab_fixture.c_str(), libdir.c_str(), cache_dir.c_str(), dump.c_str(),
              threads, mel_threads, pre_threads, dec_threads, lang.empty() ? "-" : lang.c_str(), prompt_id, state.c_str(), tail.c_str(), max_symbols, blank, (double)mask_neg, hop, n_fft, win_len, kContract, WallMs());
  const bool r3_opts = conf_xnn_flags >= 0 || dec_xnn_flags >= 0 || htp_opt_level >= 0;
  auto opt_str = [](long long v) { return v < 0 ? std::string("-") : std::to_string(v); };
  if (r3_opts) std::printf("r3_options conf_xnn_flags=%s dec_xnn_flags=%s htp_opt_level=%s\n", opt_str(conf_xnn_flags).c_str(), opt_str(dec_xnn_flags).c_str(), opt_str(htp_opt_level).c_str());
  if (fused) std::printf("r4_options conf_mode=fused\n");
  std::fflush(stdout);
  // --- environment: runtime lib dir (GPU accelerator), dispatch + compiler plugin dirs (NPU), JIT cache dir ---
  std::vector<LiteRtEnvOption> env_opts;
  auto add_env = [&](LiteRtEnvOptionTag tag, const std::string& v) { LiteRtEnvOption o{}; o.tag = tag; o.value.type = kLiteRtAnyTypeString; o.value.str_value = v.c_str(); env_opts.push_back(o); };
  add_env(kLiteRtEnvOptionTagRuntimeLibraryDir, libdir);
  add_env(kLiteRtEnvOptionTagDispatchLibraryDir, libdir);
  add_env(kLiteRtEnvOptionTagCompilerPluginLibraryDir, libdir);
  if (!cache_dir.empty()) { ::mkdir(cache_dir.c_str(), 0775); add_env(kLiteRtEnvOptionTagCompilerCacheDir, cache_dir); }
  LiteRtEnvironment env = nullptr;
  LRT_CHECK(LiteRtCreateEnvironment((int)env_opts.size(), env_opts.data(), &env));
  const int prec = gpu_precision == "default" ? (int)kLiteRtDelegatePrecisionDefault : gpu_precision == "fp16" ? (int)kLiteRtDelegatePrecisionFp16
                 : gpu_precision == "fp16acc32" ? (int)kLiteRtDelegatePrecisionFp16WithFp32Accum : (int)kLiteRtDelegatePrecisionFp32;
  // the roles model[0] conformer, [1] mel_first, [2] mel, [3] pre_first, [4] pre, [5] predict, [6] joint, in this order
  // (the conformer first); one compiled model per distinct (file, accelerator, threads)
  const char* kRole[7] = {"conformer", "mel_first", "mel", "pre_first", "pre", "predict", "joint"};
  const std::string rpath[7] = {conf_path, mel0_path, mel_path, pre0_path, pre_path, pred_path, joint_path};
  const std::string racc[7] = {accel, mel_accel, mel_accel, pre_accel, pre_accel, dec_accel, dec_accel};
  const int rthr[7] = {threads, mel_threads, mel_threads, pre_threads, pre_threads, dec_threads, dec_threads};
  const long long rxf[7] = {conf_xnn_flags, -1, -1, -1, -1, dec_xnn_flags, dec_xnn_flags};  // per-role XNNPACK flags
  std::vector<M> ms; ms.reserve(7);  // at most 7 files: the references below stay valid
  int rfile[7], rfirst[7]; Sig* rsig[7];
  for (int k = 0; k < 7; ++k) {
    if (fused && (k == 3 || k == 4)) { rfile[k] = rfile[0]; rsig[k] = rsig[0]; continue; }  // fused: no pre graph (never read)
    int fi = -1;
    for (size_t j = 0; j < ms.size(); ++j) if (ms[j].path == rpath[k] && ms[j].accel == racc[k] && ms[j].threads == rthr[k] && ms[j].xflags == rxf[k]) fi = (int)j;
    if (fi >= 0) { rfile[k] = fi; rsig[k] = &ms[fi].S(kRole[k]); continue; }
    ms.emplace_back(); fi = (int)ms.size() - 1; rfile[k] = fi; rfirst[fi] = k;
    M& m = ms[fi]; m.path = rpath[k]; m.accel = racc[k]; m.threads = rthr[k]; m.xflags = rxf[k];
    const LiteRtHwAcceleratorSet hw = Hw(m.accel);
    const int nt = m.threads;
    const char* role = kRole[k];
    LRT_CHECK(LiteRtCreateModelFromFile(env, m.path.c_str(), &m.model));
    LiteRtParamIndex n_sig = 0; LRT_CHECK(LiteRtGetNumModelSignatures(m.model, &n_sig));
    for (LiteRtParamIndex si = 0; si < n_sig; ++si) {
      Sig s; s.index = si; LiteRtSignature sig = nullptr; LRT_CHECK(LiteRtGetModelSignature(m.model, si, &sig));
      const char* key = nullptr; LRT_CHECK(LiteRtGetSignatureKey(sig, &key)); s.key = key ? key : "";
      LiteRtParamIndex n_in = 0, n_out = 0; LRT_CHECK(LiteRtGetNumSignatureInputs(sig, &n_in)); LRT_CHECK(LiteRtGetNumSignatureOutputs(sig, &n_out));
      s.in_types.resize(n_in); s.out_types.resize(n_out); s.in_elems.assign(n_in, 0); s.out_elems.assign(n_out, 0);
      for (LiteRtParamIndex i = 0; i < n_in; ++i) {
        LiteRtTensor t = nullptr; LRT_CHECK(LiteRtGetSignatureInputTensorByIndex(sig, i, &t)); LRT_CHECK(LiteRtGetRankedTensorType(t, &s.in_types[i])); LRT_CHECK(LiteRtGetNumLayoutElements(&s.in_types[i].layout, &s.in_elems[i]));
        const char* nm = nullptr; LRT_CHECK(LiteRtGetSignatureInputName(sig, i, &nm)); s.in_names.push_back(nm ? nm : "");
        std::printf("model[%d] %s sig[%u]=%s input[%u] name=%s type=%s dims=%s elems=%zu\n", k, role, (unsigned)si, s.key.c_str(), (unsigned)i, s.in_names.back().c_str(), TypeName(s.in_types[i].element_type), Dims(s.in_types[i]).c_str(), s.in_elems[i]);
      }
      for (LiteRtParamIndex o = 0; o < n_out; ++o) {
        LiteRtTensor t = nullptr; LRT_CHECK(LiteRtGetSignatureOutputTensorByIndex(sig, o, &t)); LRT_CHECK(LiteRtGetRankedTensorType(t, &s.out_types[o])); LRT_CHECK(LiteRtGetNumLayoutElements(&s.out_types[o].layout, &s.out_elems[o]));
        const char* nm = nullptr; LRT_CHECK(LiteRtGetSignatureOutputName(sig, o, &nm)); s.out_names.push_back(nm ? nm : "");
        std::printf("model[%d] %s sig[%u]=%s output[%u] name=%s type=%s dims=%s elems=%zu\n", k, role, (unsigned)si, s.key.c_str(), (unsigned)o, s.out_names.back().c_str(), TypeName(s.out_types[o].element_type), Dims(s.out_types[o]).c_str(), s.out_elems[o]);
      }
      m.sigs.push_back(s);
    }
    LRT_CHECK(LiteRtCreateOptions(&m.opts));
    LRT_CHECK(LiteRtSetOptionsHardwareAccelerators(m.opts, hw));
    AddToml(m.opts, "runtime_options_string", "enable_profiling = false\nerror_reporter_mode = 1\n");
    if (hw & kLiteRtHwAcceleratorCpu) AddToml(m.opts, "xnnpack", "num_threads = " + std::to_string(nt) + "\n" + (m.xflags >= 0 ? "flags = " + std::to_string(m.xflags) + "\n" : std::string()));
    if (hw & kLiteRtHwAcceleratorGpu) AddToml(m.opts, "gpu_options", "precision = " + std::to_string(prec) + "\n");
    {  // qualcomm: htp_performance_mode = 2 = kLiteRtQualcommHtpPerformanceModeBurst (--burst 1); --htp-opt-level's optimization_level
      std::string q = burst ? "htp_performance_mode = 2\n" : "";
      if (htp_opt_level >= 0) q += "optimization_level = " + std::to_string(htp_opt_level) + "\n";
      if ((hw & kLiteRtHwAcceleratorNpu) && !q.empty()) AddToml(m.opts, "qualcomm", q);
    }
    std::printf("compile_start model[%d] wall_ms=%lld\n", k, WallMs()); std::fflush(stdout);
    auto c0 = Clock::now(); LRT_CHECK(LiteRtCreateCompiledModel(env, m.model, m.opts, &m.cm)); m.compile_ms = Ms(c0, Clock::now());
    std::printf("compile_end model[%d] wall_ms=%lld\n", k, WallMs()); std::fflush(stdout);
    LiteRtStatus fs = LiteRtCompiledModelIsFullyAccelerated(m.cm, &m.fully);  // informational only
    LiteRtStatus ns = LiteRtCompiledModelIsNonCpuFullyAccelerated(m.cm, &m.non_cpu_fully);
    if (fs != kLiteRtStatusOk || ns != kLiteRtStatusOk) std::printf("model[%d] accelerated query status fully=%d non_cpu=%d\n", k, (int)fs, (int)ns);
    for (auto& s : m.sigs) {
      s.ins.assign(s.in_types.size(), nullptr); s.outs.assign(s.out_types.size(), nullptr);
      for (size_t i = 0; i < s.ins.size(); ++i) { LiteRtTensorBufferRequirements r = nullptr; LRT_CHECK(LiteRtGetCompiledModelInputBufferRequirements(m.cm, s.index, i, &r)); LRT_CHECK(LiteRtCreateManagedTensorBufferFromRequirements(env, &s.in_types[i], r, &s.ins[i])); }
      for (size_t o = 0; o < s.outs.size(); ++o) { LiteRtTensorBufferRequirements r = nullptr; LRT_CHECK(LiteRtGetCompiledModelOutputBufferRequirements(m.cm, s.index, o, &r)); LRT_CHECK(LiteRtCreateManagedTensorBufferFromRequirements(env, &s.out_types[o], r, &s.outs[o])); }
    }
    rsig[k] = &m.S(role);
  }
  for (int k = 0; k < 7; ++k) {  // one line per role: its file, its signature and the role that loaded the file
    if (fused && (k == 3 || k == 4)) { std::printf("model[%d]=- role=%s fused_into=model[0]\n", k, kRole[k]); continue; }
    const M& m = ms[rfile[k]]; const int first = rfirst[rfile[k]];
    const std::string r3s = !r3_opts ? std::string() : " xnn_flags=" + opt_str(m.xflags) + " htp_opt_level=" + ((Hw(m.accel) & kLiteRtHwAcceleratorNpu) ? opt_str(htp_opt_level) : std::string("-"));
    std::printf("model[%d]=%s role=%s accel=%s signature=%s signatures=%zu compile_ms=%.1f fully_accelerated=%s non_cpu_fully_accelerated=%s threads=%d%s%s\n", k, m.path.c_str(), kRole[k],
                Hw(m.accel) == kLiteRtHwAcceleratorCpu ? "cpu" : m.accel.c_str(), rsig[k]->key.c_str(), m.sigs.size(), m.compile_ms, m.fully ? "true" : "false", m.non_cpu_fully ? "true" : "false", m.threads,
                first == k ? "" : (" shared_with=model[" + std::to_string(first) + "]").c_str(), r3s.c_str());
  }
  std::fflush(stdout);
  M& Mconf = ms[rfile[0]]; M& Mmel0 = ms[rfile[1]]; M& Mmel = ms[rfile[2]]; M& Mpre0 = ms[rfile[3]]; M& Mpre = ms[rfile[4]]; M& Mpred = ms[rfile[5]]; M& Mjoint = ms[rfile[6]];
  Sig& sc = *rsig[0]; Sig& sm0 = *rsig[1]; Sig& sm = *rsig[2]; Sig& sp0 = *rsig[3]; Sig& sp = *rsig[4]; Sig& sd = *rsig[5]; Sig& sj = *rsig[6];
  if (&sm0 == &sm || (!fused && &sp0 == &sp) || &sd == &sj) { std::printf("FATAL one signature serves two roles (mel_first / mel, pre_first / pre or predict / joint): give each role its own signature\n"); return 2; }
  const int k_joint = 6;
  // --- bindings and the chunk geometry ---
  // mel graphs: the one float input (PCM) -> the one output (log-mel)
  auto only = [&](const Sig& s, bool input, const char* what) -> int {
    const auto& n = input ? s.in_names : s.out_names;
    if (n.size() != 1) { std::printf("FATAL signature %s has %zu %s (want 1: %s)\n", s.key.c_str(), n.size(), input ? "inputs" : "outputs", what); std::fflush(stdout); std::exit(2); }
    return 0;
  };
  const int m0_in = only(sm0, true, "PCM"), m0_out = only(sm0, false, "log-mel"), m_in = only(sm, true, "PCM"), m_out = only(sm, false, "log-mel");
  const long A0 = (long)sm0.in_elems[m0_in], A = (long)sm.in_elems[m_in];
  const int F0 = (int)(sm0.out_elems[m0_out] / kMel), Fn = (int)(sm.out_elems[m_out] / kMel);
  if (!fused && (sc.out_elems.size() != 4 || sc.in_elems.size() != 6)) { std::printf("FATAL conformer: %zu inputs / %zu outputs (want 6 / 4)\n", sc.in_elems.size(), sc.out_elems.size()); return 2; }
  if (fused && (sc.out_elems.size() != 7 || sc.in_elems.size() != 11)) { std::printf("FATAL fused conformer: %zu inputs / %zu outputs (want 11 / 7)\n", sc.in_elems.size(), sc.out_elems.size()); return 2; }
  // a fused graph's tensors by their names only (its smallest output is cache0_out, not enc_proj)
  auto named = [&](const Sig& s, bool input, const char* n) -> int {
    const auto& nm = input ? s.in_names : s.out_names;
    for (size_t k = 0; k < nm.size(); ++k) if (nm[k] == n) return (int)k;
    std::printf("FATAL fused signature %s: no %s named %s\n", s.key.c_str(), input ? "input" : "output", n); std::fflush(stdout); std::exit(2);
    return -1;
  };
  const int ce = fused ? named(sc, false, "enc_proj")
                       : Pick(sc, false, {"enc_proj", "enc", "output_0"}, *std::min_element(sc.out_elems.begin(), sc.out_elems.end()), "enc_proj");  // the smallest output
  const int Q = (int)(sc.out_elems[ce] / kEnc);
  if (Q < 1 || sc.out_elems[ce] % kEnc) { std::printf("FATAL conformer output %s has %zu elements (not Q x %d)\n", sc.out_names[ce].c_str(), sc.out_elems[ce], kEnc); return 2; }
  // pre graphs: embeds [Q,1024] + caches; the cache sizes are read from pre_first's outputs (unique element counts)
  const char* cname[3] = {"cache0", "cache1", "cache2"};
  const char* cname_out[3] = {"cache0_out", "cache1_out", "cache2_out"};
  int p0_mel = -1, p0_emb = -1, p_mel = -1, p_emb = -1, p_cin[3] = {-1, -1, -1}, p_cout[3] = {-1, -1, -1}, p0_c[3] = {-1, -1, -1};
  if (!fused) {
  p0_mel = Pick(sp0, true, {"mel", "features", "args_0"}, (size_t)F0 * kMel, "mel");
  p0_emb = Pick(sp0, false, {"embeds", "output_0"}, (size_t)Q * kHid, "embeds");
  std::vector<int> p0_cache; for (size_t o = 0; o < sp0.outs.size(); ++o) if ((int)o != p0_emb) p0_cache.push_back((int)o);
  if (p0_cache.size() != 3) { std::printf("FATAL pre_first has %zu cache outputs (want 3)\n", p0_cache.size()); return 2; }
  {  // positional output names (litert-torch's output_<n> of a tuple): the tuple's order, whatever order the signature lists
    bool all_pos = true; for (int x : p0_cache) all_pos = all_pos && PosName(sp0.out_names[x]) >= 0;
    if (all_pos) std::sort(p0_cache.begin(), p0_cache.end(), [&](int a, int b) { return PosName(sp0.out_names[a]) < PosName(sp0.out_names[b]); });
  }
  p_mel = Pick(sp, true, {"mel", "features", "args_0"}, (size_t)Fn * kMel, "mel"); p_emb = Pick(sp, false, {"embeds", "output_0"}, (size_t)Q * kHid, "embeds");
  for (int c = 0; c < 3; ++c) {
    // pre_first's outputs in the contract's order cache0, cache1, cache2: by name, else the order of the remaining outputs
    int o = -1; for (int x : p0_cache) if (sp0.out_names[x] == cname[c] || sp0.out_names[x] == cname_out[c]) o = x;
    if (o < 0) o = p0_cache[c];
    p0_c[c] = o; const size_t n = sp0.out_elems[o];
    p_cin[c] = Pick(sp, true, {cname[c], std::string(cname[c]) + "_in", "args_" + std::to_string(c + 1)}, n, cname[c]);
    p_cout[c] = Pick(sp, false, {cname_out[c], cname[c], "output_" + std::to_string(c + 1)}, n, cname_out[c]);
    if (sp0.out_types[o].element_type != sp.in_types[p_cin[c]].element_type || sp.out_types[p_cout[c]].element_type != sp.in_types[p_cin[c]].element_type) {
      std::printf("FATAL pre cache %d element types differ (pre_first out %s, pre in %s, pre out %s)\n", c, TypeName(sp0.out_types[o].element_type), TypeName(sp.in_types[p_cin[c]].element_type), TypeName(sp.out_types[p_cout[c]].element_type)); return 2;
    }
  }
  }  // !fused
  // the fused graph's pre-encode tensors (by name) and the chunk-0 rows of its two masks. Exact first chunk
  // (scripts/graphs_r4.py): P = F - F0 zero mel rows (odd), conv_in's rows Z0 = (P - 1) / 2 in front of PreFirst's
  // (F0 + 1) / 2 (Z0 odd), stage 1's Z1 = (Z0 - 1) / 2 = 1 in front of its rows, stage 2 none (its Q rows = PreFirst's)
  int f_mel = -1, f_m0 = -1, f_m1 = -1, f_cin[3] = {-1, -1, -1}, f_cout[3] = {-1, -1, -1}; long R0 = 0, R1 = 0, Z0 = 0, Z1 = 0;
  if (fused) {
    f_mel = named(sc, true, "mel"); f_m0 = named(sc, true, "pre_m0"); f_m1 = named(sc, true, "pre_m1");
    for (int c = 0; c < 3; ++c) {
      f_cin[c] = named(sc, true, cname[c]); f_cout[c] = named(sc, false, cname_out[c]);
      if (sc.in_elems[f_cin[c]] != sc.out_elems[f_cout[c]] || sc.in_types[f_cin[c]].element_type != sc.out_types[f_cout[c]].element_type) {
        std::printf("FATAL fused pre cache %s: in %s(%zu) / out %s(%zu) differ\n", cname[c], TypeName(sc.in_types[f_cin[c]].element_type), sc.in_elems[f_cin[c]], TypeName(sc.out_types[f_cout[c]].element_type), sc.out_elems[f_cout[c]]); return 2;
      }
    }
    R0 = (long)sc.in_elems[f_m0]; R1 = (long)sc.in_elems[f_m1];
    const long P = (long)Fn - F0, PF0 = ((long)F0 + 1) / 2, PF1 = (PF0 + 1) / 2;
    Z0 = R0 - PF0; Z1 = R1 - PF1;
    if (sc.in_elems[f_mel] != (size_t)Fn * kMel || P <= 0 || P % 2 == 0 || Z0 != (P - 1) / 2 || Z0 % 2 == 0 || Z1 != (Z0 - 1) / 2 || Z1 != 1) {
      std::printf("FATAL fused geometry: mel %zu (want %d x %d), zero mel rows %ld, pre_m0 %ld rows (zero %ld), pre_m1 %ld rows (zero %ld): not the round-4 first-chunk rule\n", sc.in_elems[f_mel], Fn, kMel, P, R0, Z0, R1, Z1); return 2;
    }
    for (int i : {f_mel, f_m0, f_m1}) if (sc.in_types[i].element_type != kLiteRtElementTypeFloat32 && sc.in_types[i].element_type != kLiteRtElementTypeFloat16) { std::printf("FATAL fused input %s is %s (want f32 / f16)\n", sc.in_names[i].c_str(), TypeName(sc.in_types[i].element_type)); return 2; }
  }
  // conformer: embeds, neg_mask, k / v cache, conv cache, one_hot -> enc_proj + caches
  // conformer inputs: by name; else x = Q x 1024, k / v = the two largest (equal) by position, the conv cache = the other
  // multiple of 1024, the one-hot = the num_prompts (128) input, the mask = the one left ([1,1,Q,S+Q])
  const int c_x = fused ? -1 : Pick(sc, true, {"embeds", "x", "args_0"}, (size_t)Q * kHid, "embeds");
  size_t kv_elems = 0; for (size_t i = 0; i < sc.in_elems.size(); ++i) kv_elems = std::max(kv_elems, sc.in_elems[i]);
  const int c_k = Pick(sc, true, {"k_cache", "key_cache", "k_cache_in"}, kv_elems, "k_cache", 0);
  const int c_v = Pick(sc, true, {"v_cache", "value_cache", "v_cache_in"}, kv_elems, "v_cache", 1);
  int c_mask = -1, c_oh = -1, c_conv = -1;
  for (size_t i = 0; i < sc.in_names.size(); ++i) {
    const std::string& n = sc.in_names[i];
    if (n == "neg_mask" || n == "mask" || n == "attn_mask") c_mask = (int)i;
    if (n == "one_hot" || n == "prompt_one_hot" || n == "lang_one_hot") c_oh = (int)i;
    if (n == "conv_cache" || n == "conv_cache_in") c_conv = (int)i;
  }
  auto taken = [&](int i) { return i == c_x || i == c_k || i == c_v || i == c_mask || i == c_oh || i == c_conv || i == f_mel || i == f_m0 || i == f_m1 || i == f_cin[0] || i == f_cin[1] || i == f_cin[2]; };
  if (c_conv < 0) for (size_t i = 0; i < sc.in_elems.size(); ++i) if (!taken((int)i) && sc.in_elems[i] % kHid == 0) { c_conv = (int)i; break; }
  if (c_oh < 0) for (size_t i = 0; i < sc.in_elems.size(); ++i) if (!taken((int)i) && sc.in_elems[i] == 128) { c_oh = (int)i; break; }
  if (c_mask < 0) for (size_t i = 0; i < sc.in_elems.size(); ++i) if (!taken((int)i)) { c_mask = (int)i; break; }
  if (c_conv < 0 || c_mask < 0 || c_oh < 0 || c_k == c_v) { std::printf("FATAL conformer inputs: want embeds, neg_mask, k_cache, v_cache, conv_cache, one_hot (conv=%d mask=%d one_hot=%d)\n", c_conv, c_mask, c_oh); return 2; }
  if (sc.in_elems[c_mask] % (size_t)Q) { std::printf("FATAL conformer mask %s has %zu elements (not a multiple of Q %d)\n", sc.in_names[c_mask].c_str(), sc.in_elems[c_mask], Q); return 2; }
  for (int i : {c_x, c_mask, c_oh}) if (i >= 0 && sc.in_types[i].element_type != kLiteRtElementTypeFloat32 && sc.in_types[i].element_type != kLiteRtElementTypeFloat16) { std::printf("FATAL conformer input %s is %s (want f32 / f16)\n", sc.in_names[i].c_str(), TypeName(sc.in_types[i].element_type)); return 2; }
  if (!fused)
  std::printf("binding conformer inputs x=%s(%zu) mask=%s(%zu) k=%s v=%s(%zu each) conv=%s(%zu) one_hot=%s(%zu)\n", sc.in_names[c_x].c_str(), sc.in_elems[c_x], sc.in_names[c_mask].c_str(), sc.in_elems[c_mask],
              sc.in_names[c_k].c_str(), sc.in_names[c_v].c_str(), kv_elems, sc.in_names[c_conv].c_str(), sc.in_elems[c_conv], sc.in_names[c_oh].c_str(), sc.in_elems[c_oh]);
  else
  std::printf("binding fused inputs mel=%s(%zu) pre_m0=%s(%ld rows, %ld zero at chunk 0) pre_m1=%s(%ld rows, %ld zero at chunk 0) pre caches %s(%zu) %s(%zu) %s(%zu) mask=%s(%zu) k=%s v=%s(%zu each) conv=%s(%zu) one_hot=%s(%zu)\n",
              sc.in_names[f_mel].c_str(), sc.in_elems[f_mel], sc.in_names[f_m0].c_str(), R0, Z0, sc.in_names[f_m1].c_str(), R1, Z1,
              sc.in_names[f_cin[0]].c_str(), sc.in_elems[f_cin[0]], sc.in_names[f_cin[1]].c_str(), sc.in_elems[f_cin[1]], sc.in_names[f_cin[2]].c_str(), sc.in_elems[f_cin[2]],
              sc.in_names[c_mask].c_str(), sc.in_elems[c_mask], sc.in_names[c_k].c_str(), sc.in_names[c_v].c_str(), kv_elems, sc.in_names[c_conv].c_str(), sc.in_elems[c_conv], sc.in_names[c_oh].c_str(), sc.in_elems[c_oh]);
  const int KV = (int)(sc.in_elems[c_mask] / Q), S = KV - Q, P = (int)sc.in_elems[c_oh];
  const int co_k = Pick(sc, false, {"k_out", "k_cache_out", "new_k_cache"}, kv_elems, "k_out", 0);
  const int co_v = Pick(sc, false, {"v_out", "v_cache_out", "new_v_cache"}, kv_elems, "v_out", 1);
  const int co_conv = Pick(sc, false, {"conv_out", "conv_cache_out", "new_conv_cache"}, sc.in_elems[c_conv], "conv_out");
  for (auto pr : {std::make_pair(c_k, co_k), std::make_pair(c_v, co_v), std::make_pair(c_conv, co_conv)})
    if (sc.in_types[pr.first].element_type != sc.out_types[pr.second].element_type) { std::printf("FATAL conformer cache %s in %s / out %s element types differ\n", sc.in_names[pr.first].c_str(), TypeName(sc.in_types[pr.first].element_type), TypeName(sc.out_types[pr.second].element_type)); return 2; }
  if (c_k == c_v || co_k == co_v) { std::printf("FATAL ambiguous k / v binding\n"); return 2; }
  // predict and joint
  const int d_tok = Pick(sd, true, {"token", "args_0"}, 1, "token");
  const int d_h = Pick(sd, true, {"h", "h_in", "args_1"}, kDecState, "h", 0);
  const int d_c = Pick(sd, true, {"c", "c_in", "args_2"}, kDecState, "c", 1);
  const int do_dec = Pick(sd, false, {"dec_out", "output_0"}, kEnc, "dec_out");
  const int do_h = Pick(sd, false, {"h_out", "h1", "new_h", "output_1"}, kDecState, "h_out", 0);
  const int do_c = Pick(sd, false, {"c_out", "c1", "new_c", "output_2"}, kDecState, "c_out", 1);
  const int j_dec = Pick(sj, true, {"dec_out", "args_0"}, kEnc, "dec_out", 0);
  const int j_enc = Pick(sj, true, {"enc_frame", "args_1"}, kEnc, "enc_frame", 1);
  if (sj.out_names.size() != 1) { std::printf("FATAL joint has %zu outputs (want 1: logits)\n", sj.out_names.size()); return 2; }
  const int jo = 0; const int V = (int)sj.out_elems[jo];
  if (blank < 0) blank = V - 1;
  if (d_h == d_c || do_h == do_c || j_dec == j_enc) { std::printf("FATAL ambiguous decoder binding\n"); return 2; }
  const LiteRtElementType tok_t = sd.in_types[d_tok].element_type;
  if (tok_t != kLiteRtElementTypeInt32 && tok_t != kLiteRtElementTypeInt64) { std::printf("FATAL the token input must be int32 or int64 (got %s)\n", TypeName(tok_t)); return 2; }
  // the chunk geometry: A0 = (F0 - 1) hop + win / 2, A = F hop + win, s_k = (F0 + F (k - 1)) hop - n_fft / 2
  if (A0 != (F0 - 1) * hop + win_len / 2 || A != (long)Fn * hop + win_len || S < 0 || KV <= 0 || Fn % Q != 0) {
    std::printf("FATAL chunk geometry: mel_first PCM %ld / %d frames, mel PCM %ld / %d frames, Q %d, KV %d (want A0 = (F0-1) x %ld + %ld, A = F x %ld + %ld, F a multiple of Q)\n", A0, F0, A, Fn, Q, KV, hop, win_len / 2, hop, win_len);
    return 2;
  }
  if (prompt_id >= P) { std::printf("FATAL --prompt-id %d >= one_hot size %d\n", prompt_id, P); return 2; }
  if (fused)
  std::printf("binding mel_first %s(%ld)->%s(%dx%d) | mel %s(%ld)->%s(%dx%d) | %s mel=%s + %s,%s,%s -> %s,%s,%s (chunk 0: %ld zero mel rows, then mel_first's)\n",
              sm0.in_names[m0_in].c_str(), A0, sm0.out_names[m0_out].c_str(), F0, kMel, sm.in_names[m_in].c_str(), A, sm.out_names[m_out].c_str(), Fn, kMel, sc.key.c_str(), sc.in_names[f_mel].c_str(),
              sc.in_names[f_cin[0]].c_str(), sc.in_names[f_cin[1]].c_str(), sc.in_names[f_cin[2]].c_str(), sc.out_names[f_cout[0]].c_str(), sc.out_names[f_cout[1]].c_str(), sc.out_names[f_cout[2]].c_str(), (long)Fn - F0);
  else
  std::printf("binding mel_first %s(%ld)->%s(%dx%d) | mel %s(%ld)->%s(%dx%d) | pre_first %s->%s + %s,%s,%s | pre %s + %s,%s,%s -> %s + %s,%s,%s\n",
              sm0.in_names[m0_in].c_str(), A0, sm0.out_names[m0_out].c_str(), F0, kMel, sm.in_names[m_in].c_str(), A, sm.out_names[m_out].c_str(), Fn, kMel,
              sp0.in_names[p0_mel].c_str(), sp0.out_names[p0_emb].c_str(), sp0.out_names[p0_c[0]].c_str(), sp0.out_names[p0_c[1]].c_str(), sp0.out_names[p0_c[2]].c_str(),
              sp.in_names[p_mel].c_str(), sp.in_names[p_cin[0]].c_str(), sp.in_names[p_cin[1]].c_str(), sp.in_names[p_cin[2]].c_str(),
              sp.out_names[p_emb].c_str(), sp.out_names[p_cout[0]].c_str(), sp.out_names[p_cout[1]].c_str(), sp.out_names[p_cout[2]].c_str());
  std::printf("binding conformer x=%s mask=%s k=%s v=%s conv=%s one_hot=%s -> enc=%s k=%s v=%s conv=%s | predict token=%s(%s) h=%s c=%s -> dec=%s h=%s c=%s | joint dec=%s enc=%s -> %s(%d)\n",
              fused ? "-" : sc.in_names[c_x].c_str(), sc.in_names[c_mask].c_str(), sc.in_names[c_k].c_str(), sc.in_names[c_v].c_str(), sc.in_names[c_conv].c_str(), sc.in_names[c_oh].c_str(),
              sc.out_names[ce].c_str(), sc.out_names[co_k].c_str(), sc.out_names[co_v].c_str(), sc.out_names[co_conv].c_str(),
              sd.in_names[d_tok].c_str(), TypeName(tok_t), sd.in_names[d_h].c_str(), sd.in_names[d_c].c_str(), sd.out_names[do_dec].c_str(), sd.out_names[do_h].c_str(), sd.out_names[do_c].c_str(),
              sj.in_names[j_dec].c_str(), sj.in_names[j_enc].c_str(), sj.out_names[jo].c_str(), V);
  std::printf("chunking first_samples=%ld next_samples=%ld advance=%ld first_mel=%d next_mel=%d frames_per_chunk=%d kv_slots=%d window=%d one_hot=%d vocab=%d blank=%ld tail=%s state=%s\n",
              A0, A, (long)Fn * hop, F0, Fn, Q, S, KV, P, V, blank, tail.c_str(), state.c_str());
  std::fflush(stdout);
  // --- buffers of the conformer caches: --state swap = a second set from the joined requirements, alternated per chunk ---
  // a fused graph carries the pre-encode caches as cache sets 3..5, moved like the conformer's
  const int NC = fused ? 6 : 3;
  const int cache_in[6] = {c_k, c_v, c_conv, f_cin[0], f_cin[1], f_cin[2]}, cache_out[6] = {co_k, co_v, co_conv, f_cout[0], f_cout[1], f_cout[2]};
  size_t cache_bytes[6] = {0, 0, 0, 0, 0, 0}; for (int c = 0; c < NC; ++c) cache_bytes[c] = sc.in_elems[cache_in[c]] * ElemBytes(sc.in_types[cache_in[c]].element_type);
  LiteRtTensorBuffer setA[6], setB[6]; bool swap = state == "swap";
  for (int c = 0; c < NC; ++c) { setA[c] = sc.ins[cache_in[c]]; setB[c] = sc.outs[cache_out[c]]; }
  std::vector<LiteRtTensorBuffer> own_bufs;
  if (swap) {
    for (int c = 0; c < NC && swap; ++c) {
      LiteRtTensorBufferRequirements ri = nullptr, ro = nullptr, rj = nullptr;
      LRT_CHECK(LiteRtGetCompiledModelInputBufferRequirements(Mconf.cm, sc.index, cache_in[c], &ri));
      LRT_CHECK(LiteRtGetCompiledModelOutputBufferRequirements(Mconf.cm, sc.index, cache_out[c], &ro));
      if (LiteRtJoinTensorBufferRequirements(ri, ro, &rj) != kLiteRtStatusOk) { std::printf("state: the input / output requirements of %s do not join -> --state copy\n", sc.in_names[cache_in[c]].c_str()); swap = false; break; }
      LiteRtTensorBuffer a = nullptr, b = nullptr;
      LRT_CHECK(LiteRtCreateManagedTensorBufferFromRequirements(env, &sc.in_types[cache_in[c]], rj, &a));
      LRT_CHECK(LiteRtCreateManagedTensorBufferFromRequirements(env, &sc.in_types[cache_in[c]], rj, &b));
      LiteRtDestroyTensorBufferRequirements(rj);
      setA[c] = a; setB[c] = b; own_bufs.push_back(a); own_bufs.push_back(b);
    }
    if (!swap) { for (auto b : own_bufs) LiteRtDestroyTensorBuffer(b); own_bufs.clear(); for (int c = 0; c < NC; ++c) { setA[c] = sc.ins[cache_in[c]]; setB[c] = sc.outs[cache_out[c]]; } }
  }
  std::printf("state %s (conformer caches %zu + %zu + %zu B per chunk)\n", swap ? "swap" : "copy", cache_bytes[0], cache_bytes[1], cache_bytes[2]);
  if (fused) std::printf("state pre caches %zu + %zu + %zu B per chunk (inside the fused graph, moved with the conformer caches)\n", cache_bytes[3], cache_bytes[4], cache_bytes[5]);
  std::fflush(stdout);
  // the conformer's input / output arrays per parity of the chunk (swap: even chunks read set A and write set B)
  std::vector<LiteRtTensorBuffer> cin_even(sc.ins), cout_even(sc.outs), cin_odd(sc.ins), cout_odd(sc.outs);
  for (int c = 0; c < NC; ++c) {
    cin_even[cache_in[c]] = setA[c]; cout_even[cache_out[c]] = setB[c];
    cin_odd[cache_in[c]] = swap ? setB[c] : setA[c]; cout_odd[cache_out[c]] = swap ? setA[c] : setB[c];
  }
  // --- fixtures: id3 -> (utterance id, file, locale) ---
  std::vector<std::string> fx_id, fx_file, fx_lang; {
    std::vector<uint8_t> b; if (!ReadFile(fixtures + "/index.tsv", b)) { std::printf("FIXTURE_ERROR %s/index.tsv\n", fixtures.c_str()); return 3; }
    for (auto& line : Split(std::string(b.begin(), b.end()), '\n')) {
      if (line.empty() || line[0] == '#') continue; std::string l = line; if (!l.empty() && l.back() == '\r') l.pop_back();
      auto c = Split(l, '\t'); if (c.size() < 3) continue;
      long i = std::strtol(c[0].c_str(), nullptr, 10); if ((long)fx_id.size() <= i) { fx_id.resize(i + 1); fx_file.resize(i + 1); fx_lang.resize(i + 1); }
      fx_id[i] = c[1]; fx_file[i] = c[2]; fx_lang[i] = c.size() > 3 ? c[3] : "";
    }
  }
  auto id3 = [](long i) { char idbuf[24]; std::snprintf(idbuf, sizeof idbuf, "%03ld", i); return std::string(idbuf); };
  const std::string mdir = out_dir + "/model0", edir = out_dir + "/enc"; ::mkdir(mdir.c_str(), 0775); if (dump == "enc") ::mkdir(edir.c_str(), 0775);
  // --- host state ---
  const LiteRtElementType x_t = fused ? kLiteRtElementTypeFloat32 : sc.in_types[c_x].element_type, mask_t = sc.in_types[c_mask].element_type, oh_t = sc.in_types[c_oh].element_type, enc_t = sc.out_types[ce].element_type;
  const LiteRtElementType m0_t = sm0.in_types[m0_in].element_type, m_t = sm.in_types[m_in].element_type;
  const LiteRtElementType dh_t = sd.in_types[d_h].element_type, dc_t = sd.in_types[d_c].element_type, ddo_t = sd.out_types[do_dec].element_type, dho_t = sd.out_types[do_h].element_type, dco_t = sd.out_types[do_c].element_type;
  const LiteRtElementType jd_t = sj.in_types[j_dec].element_type, je_t = sj.in_types[j_enc].element_type, jo_t = sj.out_types[jo].element_type;
  std::vector<float> enc((size_t)Q * kEnc), dec_out(kEnc), h(kDecState), c(kDecState), logits(V), mask((size_t)Q * KV), onehot(P);
  // mel -> pre and pre -> conformer move inside the buffers when the types match (one memcpy), else through the host
  const bool mel0_direct = !fused && sm0.out_types[m0_out].element_type == sp0.in_types[p0_mel].element_type, mel_direct = !fused && sm.out_types[m_out].element_type == sp.in_types[p_mel].element_type;
  const bool emb0_direct = !fused && sp0.out_types[p0_emb].element_type == x_t, emb_direct = !fused && sp.out_types[p_emb].element_type == x_t;
  // fused: the steady mel moves into the conformer's mel input; chunk 0 goes through the host (zero rows first)
  const LiteRtElementType fmel_t = fused ? sc.in_types[f_mel].element_type : kLiteRtElementTypeFloat32;
  const LiteRtElementType fm0_t = fused ? sc.in_types[f_m0].element_type : kLiteRtElementTypeFloat32, fm1_t = fused ? sc.in_types[f_m1].element_type : kLiteRtElementTypeFloat32;
  const bool fmel_direct = fused && sm.out_types[m_out].element_type == fmel_t;
  std::vector<float> fmel0(fused ? (size_t)Fn * kMel : 0, 0.f), fmask0(fused ? (size_t)R0 : 0, 1.f), fmask1(fused ? (size_t)R1 : 0, 1.f), fones(fused ? (size_t)std::max(R0, R1) : 0, 1.f);
  for (long i = 0; i < Z0 && fused; ++i) fmask0[i] = 0.f;
  for (long i = 0; i < Z1 && fused; ++i) fmask1[i] = 0.f;
  std::vector<float> hostbuf((size_t)std::max(F0, Fn) * kMel + (size_t)Q * kHid);
  auto move_f = [&](LiteRtTensorBuffer dst, LiteRtElementType dt, LiteRtTensorBuffer src, LiteRtElementType st, size_t n, bool direct) {
    if (direct) { Copy(dst, src, n * ElemBytes(st)); return; }
    ReadF(src, st, hostbuf.data(), n); WriteF(dst, dt, hostbuf.data(), n);
  };
  int64_t tok64 = 0; int32_t tok32 = 0;
  double dec_run_acc = 0;  // the decoder runs of the current utterance's loop (predict + joint)
  auto predict = [&](long token) {
    if (tok_t == kLiteRtElementTypeInt32) { tok32 = (int32_t)token; Write(sd.ins[d_tok], &tok32, 4); } else { tok64 = token; Write(sd.ins[d_tok], &tok64, 8); }
    WriteF(sd.ins[d_h], dh_t, h.data(), kDecState); WriteF(sd.ins[d_c], dc_t, c.data(), kDecState);
    dec_run_acc += Run(Mpred, sd);
    ReadF(sd.outs[do_dec], ddo_t, dec_out.data(), kEnc); ReadF(sd.outs[do_h], dho_t, h.data(), kDecState); ReadF(sd.outs[do_c], dco_t, c.data(), kDecState);
    WriteF(sj.ins[j_dec], jd_t, dec_out.data(), kEnc);   // the joint's decoder input changes only here
  };
  // argmax as torch / numpy: the first maximum, a NaN counts as the maximum (the first NaN wins); margin = top-1 - top-2
  auto argmax = [&](double& margin) {
    int bi = 0; float m1 = logits[0], m2 = -std::numeric_limits<float>::infinity();
    if (!std::isnan(m1)) for (int i = 1; i < V; ++i) { float v = logits[i]; if (std::isnan(v)) { bi = i; m1 = v; break; } if (v > m1) { m2 = m1; m1 = v; bi = i; } else if (v > m2) m2 = v; }
    margin = (std::isnan(m1) || std::isnan(m2)) ? -1.0 : (double)m1 - (double)m2; return bi;
  };
  struct Utt {
    std::string id, fmt, lang; int prompt = 0; long samples = 0, chunks = 0;
    std::vector<int> tok, frame, sym; std::vector<double> chunk_ms, mel_ms, pre_ms, conf_ms, dec_ms;
    int steps = 0, cap_hits = 0, max_run = 0; size_t nonfinite = 0; double min_margin = 1e30, init_ms = 0; int mm_frame = -1, mm_sym = -1; bool truncated = false;
    double run_mel = 0, run_pre = 0, run_conf = 0, run_dec = 0;  // the runs alone (LiteRtRunCompiledModel), summed over the chunks
    std::vector<float> encs;
  };
  long cur_prompt = -1;
  auto set_prompt = [&](int p) {
    if (p == cur_prompt) return;
    std::fill(onehot.begin(), onehot.end(), 0.f); onehot[p] = 1.f; WriteF(sc.ins[c_oh], oh_t, onehot.data(), P); cur_prompt = p;
  };
  // the utterance's language prompt: --prompt-id, else the index's locale, else --lang
  auto prompt_of = [&](long fx, std::string& loc) -> int {
    if (prompt_id >= 0) { loc = "prompt_id"; return prompt_id; }
    loc = !fx_lang[fx].empty() ? fx_lang[fx] : lang;
    if (loc.empty()) { std::printf("FATAL fixture %s has no locale column and no --lang / --prompt-id\n", id3(fx).c_str()); std::fflush(stdout); std::exit(2); }
    int p = PromptOf(loc); if (p < 0 || p >= P) { std::printf("FATAL unknown locale %s (fixture %s)\n", loc.c_str(), id3(fx).c_str()); std::fflush(stdout); std::exit(2); }
    return p;
  };
  // chunk k of an utterance: mel -> pre -> conformer (+ the cache move) -> the RNN-T loop on its Q frames
  int cur_frame_syms = 0; std::vector<float> pcm_chunk((size_t)std::max(A0, A));
  double mel_run = 0, pre_run = 0, conf_run = 0;
  auto do_chunk = [&](const std::vector<float>& x, long N, long k, Utt& u) {
    const auto t0 = Clock::now();
    // the PCM slice (zeros past the end of the utterance = the appended tail)
    const long len = k == 0 ? A0 : A, s0 = k == 0 ? 0 : ((long)F0 + (long)Fn * (k - 1)) * hop - n_fft / 2;
    for (long i = 0; i < len; ++i) { long j = s0 + i; pcm_chunk[i] = (j >= 0 && j < N) ? x[j] : 0.f; }
    if (k == 0) { WriteF(sm0.ins[m0_in], m0_t, pcm_chunk.data(), A0); mel_run = Run(Mmel0, sm0); }
    else { WriteF(sm.ins[m_in], m_t, pcm_chunk.data(), A); mel_run = Run(Mmel, sm); }
    const auto t1 = Clock::now();
    if (fused) {
      pre_run = 0;  // fused: no pre graph; the mel move into the fused graph counts in the conformer stage (t2 = t1)
    } else if (k == 0) {
      move_f(sp0.ins[p0_mel], sp0.in_types[p0_mel].element_type, sm0.outs[m0_out], sm0.out_types[m0_out].element_type, (size_t)F0 * kMel, mel0_direct);
      pre_run = Run(Mpre0, sp0);
      move_f(sc.ins[c_x], x_t, sp0.outs[p0_emb], sp0.out_types[p0_emb].element_type, (size_t)Q * kHid, emb0_direct);
      for (int cc = 0; cc < 3; ++cc) Copy(sp.ins[p_cin[cc]], sp0.outs[p0_c[cc]], sp0.out_elems[p0_c[cc]] * ElemBytes(sp0.out_types[p0_c[cc]].element_type));
    } else {
      move_f(sp.ins[p_mel], sp.in_types[p_mel].element_type, sm.outs[m_out], sm.out_types[m_out].element_type, (size_t)Fn * kMel, mel_direct);
      pre_run = Run(Mpre, sp);
      move_f(sc.ins[c_x], x_t, sp.outs[p_emb], sp.out_types[p_emb].element_type, (size_t)Q * kHid, emb_direct);
      for (int cc = 0; cc < 3; ++cc) Copy(sp.ins[p_cin[cc]], sp.outs[p_cout[cc]], sp.out_elems[p_cout[cc]] * ElemBytes(sp.out_types[p_cout[cc]].element_type));
    }
    const auto t2 = fused ? t1 : Clock::now();
    if (fused) {  // chunk 0: Fn - F0 zero rows, then mel_first's F0 rows, masks with Z0 / Z1 zero rows; chunk 1 sets the masks to 1
      if (k == 0) {
        ReadF(sm0.outs[m0_out], sm0.out_types[m0_out].element_type, fmel0.data() + (size_t)(Fn - F0) * kMel, (size_t)F0 * kMel);
        WriteF(sc.ins[f_mel], fmel_t, fmel0.data(), fmel0.size());
        WriteF(sc.ins[f_m0], fm0_t, fmask0.data(), (size_t)R0); WriteF(sc.ins[f_m1], fm1_t, fmask1.data(), (size_t)R1);
      } else {
        move_f(sc.ins[f_mel], fmel_t, sm.outs[m_out], sm.out_types[m_out].element_type, (size_t)Fn * kMel, fmel_direct);
        if (k == 1) { WriteF(sc.ins[f_m0], fm0_t, fones.data(), (size_t)R0); WriteF(sc.ins[f_m1], fm1_t, fones.data(), (size_t)R1); }
      }
    }
    // the mask of chunk k: slots [0, S - min(S, Q k)) of every query row
    const long valid = std::min<long>(S, (long)Q * k);
    for (int q = 0; q < Q; ++q) for (int j = 0; j < KV; ++j) mask[(size_t)q * KV + j] = j < S - valid ? mask_neg : 0.f;
    WriteF(sc.ins[c_mask], mask_t, mask.data(), mask.size());
    auto& ins = (k % 2 == 0) ? cin_even : cin_odd; auto& outs = (k % 2 == 0) ? cout_even : cout_odd;
    if (c_x >= 0) ins[c_x] = sc.ins[c_x];
    ins[c_mask] = sc.ins[c_mask]; ins[c_oh] = sc.ins[c_oh];
    if (fused) { ins[f_mel] = sc.ins[f_mel]; ins[f_m0] = sc.ins[f_m0]; ins[f_m1] = sc.ins[f_m1]; }
    { auto tr = Clock::now(); LRT_CHECK(LiteRtRunCompiledModel(Mconf.cm, sc.index, ins.size(), ins.data(), outs.size(), outs.data())); conf_run = Ms(tr, Clock::now()); }
    ReadF(outs[ce], enc_t, enc.data(), enc.size());
    if (!swap) for (int cc = 0; cc < NC; ++cc) Copy(setA[cc], setB[cc], cache_bytes[cc]);
    const auto t3 = Clock::now();
    for (float v : enc) if (!std::isfinite(v)) ++u.nonfinite;
    if (dump == "enc") u.encs.insert(u.encs.end(), enc.begin(), enc.end());
    u.run_mel += mel_run; u.run_pre += pre_run; u.run_conf += conf_run;
    // the RNN-T loop on the chunk's Q frames; the decoder state runs on from the previous chunk
    const long max_steps = (max_symbols + 1) * (long)Q * (u.chunks + 1) + 16;
    dec_run_acc = 0;
    for (int f = 0; f < Q; ++f) {
      const int t = (int)k * Q + f;
      WriteF(sj.ins[j_enc], je_t, enc.data() + (size_t)f * kEnc, kEnc);
      cur_frame_syms = 0;
      while (u.steps < max_steps) {
        ++u.steps; dec_run_acc += Run(Mjoint, sj); ReadF(sj.outs[jo], jo_t, logits.data(), V);
        double margin = 0; const int tokv = argmax(margin);
        if (margin >= 0 && margin < u.min_margin) { u.min_margin = margin; u.mm_frame = t; u.mm_sym = cur_frame_syms; }
        if (tokv == blank) break;
        u.tok.push_back(tokv); u.frame.push_back(t); u.sym.push_back(cur_frame_syms);
        predict(tokv); ++cur_frame_syms; u.max_run = std::max(u.max_run, cur_frame_syms);
        if (max_symbols > 0 && cur_frame_syms >= max_symbols) { ++u.cap_hits; break; }
      }
      if (u.steps >= max_steps) { u.truncated = true; break; }
    }
    const auto t4 = Clock::now();
    u.run_dec += dec_run_acc;
    u.chunk_ms.push_back(R3(Ms(t0, t4))); u.mel_ms.push_back(R3(Ms(t0, t1))); u.pre_ms.push_back(R3(Ms(t1, t2))); u.conf_ms.push_back(R3(Ms(t2, t3))); u.dec_ms.push_back(R3(Ms(t3, t4)));
    ++u.chunks;
  };
  auto n_chunks_of = [&](long N) -> long {  // chunk 0, then every k >= 1 with s_k + A < N + A (pad) or < N (none)
    const long padded = tail == "pad" ? N + A : N; long n = 1;
    while (((long)F0 + (long)Fn * (n - 1)) * hop - n_fft / 2 + A < padded) ++n;
    return n;
  };
  auto reset = [&]() {  // a new utterance: zeroed caches (set A is read first), decoder from predict(blank, h = c = 0)
    for (int cc = 0; cc < NC; ++cc) Zero(setA[cc], cache_bytes[cc]);
    std::fill(h.begin(), h.end(), 0.f); std::fill(c.begin(), c.end(), 0.f);
    predict(blank);
  };
  auto load_utt = [&](long fx, Utt& u, std::vector<float>& x) -> bool {
    if (fx >= (long)fx_id.size() || fx_id[fx].empty()) { std::printf("FIXTURE_ERROR %s/index.tsv has no line %s\n", fixtures.c_str(), id3(fx).c_str()); return false; }
    std::string fmt;
    if (!ReadAudio(fixtures + "/" + fx_file[fx], x, fmt)) { std::printf("FIXTURE_ERROR %s/%s (%s)\n", fixtures.c_str(), fx_file[fx].c_str(), fmt.c_str()); return false; }
    u = Utt(); u.id = fx_id[fx]; u.samples = (long)x.size(); u.fmt = fmt; u.prompt = prompt_of(fx, u.lang);
    return true;
  };
  auto run_utt = [&](long fx, Utt& u) -> bool {
    std::vector<float> x; if (!load_utt(fx, u, x)) return false;
    set_prompt(u.prompt);
    auto ti = Clock::now(); reset(); u.init_ms = Ms(ti, Clock::now());
    const long n = n_chunks_of(u.samples);
    for (long k = 0; k < n && !u.truncated; ++k) do_chunk(x, u.samples, k, u);
    return true;
  };
  // --- the streaming pass: one warm-up run of fixture 000 (not counted), then every fixture once ---
  Utt u; double first = 0; { auto t0 = Clock::now(); if (!run_utt(0, u)) return 3; first = Ms(t0, Clock::now()); }
  std::printf("warmup fixture 000 id=%s chunks=%ld total_ms=%.3f\n", u.id.c_str(), u.chunks, first); std::fflush(stdout);
  std::vector<double> all_chunk, all_mel, all_pre, all_conf, all_dec, steady_chunk, first_chunk;
  double s_audio = 0, s_chunk = 0, s_mel = 0, s_pre = 0, s_conf = 0, s_dec = 0; long n_chunks_all = 0, n_empty = 0, n_cap = 0, n_trunc = 0; size_t nf_all = 0; int max_run_all = 0; double min_margin_all = 1e30;
  std::vector<float> rec(kRecElems);
  for (long fx = 0; fx < count; ++fx) {
    if (!run_utt(fx, u)) return 3;
    const std::string id = id3(fx); const double sec = u.samples / 16000.0;
    double tot = 0; for (double v : u.chunk_ms) tot += v;
    for (size_t k = 0; k < u.chunk_ms.size(); ++k) {
      all_chunk.push_back(u.chunk_ms[k]); all_mel.push_back(u.mel_ms[k]); all_pre.push_back(u.pre_ms[k]); all_conf.push_back(u.conf_ms[k]); all_dec.push_back(u.dec_ms[k]);
      (k ? steady_chunk : first_chunk).push_back(u.chunk_ms[k]);
      s_mel += u.mel_ms[k]; s_pre += u.pre_ms[k]; s_conf += u.conf_ms[k]; s_dec += u.dec_ms[k];
    }
    s_audio += sec; s_chunk += tot; n_chunks_all += u.chunks; n_empty += u.tok.empty(); n_cap += u.cap_hits; n_trunc += u.truncated; nf_all += u.nonfinite; max_run_all = std::max(max_run_all, u.max_run);
    if (u.min_margin < min_margin_all) min_margin_all = u.min_margin;
    std::printf("{\"kind\":\"utt\",\"fx\":\"%s\",\"id\":%s,\"fmt\":\"%s\",\"lang\":%s,\"prompt\":%d,\"samples\":%ld,\"audio_s\":%.4f,\"chunks\":%ld,\"frames\":%ld,\"n_tok\":%zu,\"steps\":%d,\"cap_hits\":%d,\"max_run\":%d,"
                "\"truncated\":%s,\"nonfinite_enc\":%zu,\"min_margin\":%.6g,\"min_margin_at\":[%d,%d],\"init_ms\":%.3f,\"total_ms\":%.3f,\"run_ms\":{\"mel\":%.3f,\"pre\":%.3f,\"conf\":%.3f,\"dec\":%.3f},\"tok\":%s,\"frame\":%s,\"sym\":%s,"
                "\"chunk_ms\":%s,\"mel_ms\":%s,\"pre_ms\":%s,\"conf_ms\":%s,\"dec_ms\":%s}\n",
                id.c_str(), JsonStr(u.id).c_str(), u.fmt.c_str(), JsonStr(u.lang).c_str(), u.prompt, u.samples, sec, u.chunks, u.chunks * Q, u.tok.size(), u.steps, u.cap_hits, u.max_run,
                u.truncated ? "true" : "false", u.nonfinite, u.min_margin > 1e29 ? -1.0 : u.min_margin, u.mm_frame, u.mm_sym, u.init_ms, tot, u.run_mel, u.run_pre, u.run_conf, u.run_dec, JsonArr(u.tok).c_str(), JsonArr(u.frame).c_str(), JsonArr(u.sym).c_str(),
                JsonMs(u.chunk_ms).c_str(), JsonMs(u.mel_ms).c_str(), JsonMs(u.pre_ms).c_str(), JsonMs(u.conf_ms).c_str(), JsonMs(u.dec_ms).c_str());
    std::fflush(stdout);
    if (dump != "none") {
      std::fill(rec.begin(), rec.end(), -1.f);
      rec[0] = (float)u.chunks; rec[1] = (float)u.tok.size(); rec[2] = u.tok.size() > (size_t)kMaxRecTokens ? 1.f : 0.f; rec[3] = (float)(u.chunks * Q);
      for (size_t j = 0; j < u.tok.size() && j < (size_t)kMaxRecTokens; ++j) { rec[4 + 3 * j] = (float)u.tok[j]; rec[5 + 3 * j] = (float)u.frame[j]; rec[6 + 3 * j] = (float)u.sym[j]; }
      if (!WriteFloatFile(mdir + "/" + id + ".f32", rec.data(), rec.size())) { std::printf("WRITE_ERROR %s/%s.f32\n", mdir.c_str(), id.c_str()); return 3; }
      if (dump == "enc" && !WriteFloatFile(edir + "/" + id + ".f32", u.encs.data(), u.encs.size())) { std::printf("WRITE_ERROR %s/%s.f32\n", edir.c_str(), id.c_str()); return 3; }
    }
  }
  const double p50 = Pct(all_chunk, 50), p90 = Pct(all_chunk, 90);
  std::printf("{\"kind\":\"stream_summary\",\"utterances\":%ld,\"chunks\":%ld,\"audio_s\":%.4f,\"chunk_ms_p50\":%.6f,\"chunk_ms_p90\":%.6f,\"chunk_ms_mean\":%.6f,\"chunk_ms_max\":%.3f,"
              "\"rtf_stream\":%.6f,\"sum_chunk_ms\":%.3f,\"sum_mel_ms\":%.3f,\"sum_pre_ms\":%.3f,\"sum_conf_ms\":%.3f,\"sum_dec_ms\":%.3f,\"mel_ms_p50\":%.6f,\"pre_ms_p50\":%.6f,\"conf_ms_p50\":%.6f,\"dec_ms_p50\":%.6f,"
              "\"first_chunk_ms_p50\":%.6f,\"steady_chunk_ms_p50\":%.6f,\"steady_chunk_ms_p90\":%.6f,\"warmup_fx\":\"000\",\"warmup_ms\":%.3f,\"empty_hyp\":%ld,\"cap_hits\":%ld,\"max_run\":%d,\"truncated\":%ld,"
              "\"nonfinite_enc\":%zu,\"min_margin\":%.6g,\"threads\":%d,\"mel_threads\":%d,\"pre_threads\":%d,\"dec_threads\":%d,\"accel\":\"%s\",\"gpu_precision\":\"%s\",\"state\":\"%s\",\"tail\":\"%s\"}\n",
              count, n_chunks_all, s_audio, p50, p90, n_chunks_all ? s_chunk / n_chunks_all : 0.0, all_chunk.empty() ? 0.0 : *std::max_element(all_chunk.begin(), all_chunk.end()),
              s_chunk / 1000.0 / s_audio, s_chunk, s_mel, s_pre, s_conf, s_dec, Pct(all_mel, 50), Pct(all_pre, 50), Pct(all_conf, 50), Pct(all_dec, 50),
              Pct(first_chunk, 50), Pct(steady_chunk, 50), Pct(steady_chunk, 90), first, n_empty, n_cap, max_run_all, n_trunc, nf_all, min_margin_all > 1e29 ? -1.0 : min_margin_all,
              threads, mel_threads, pre_threads, dec_threads, accel.c_str(), accel.rfind("gpu", 0) == 0 ? gpu_precision.c_str() : "-", swap ? "swap" : "copy", tail.c_str());
  std::printf("parity_summary model[0] %s fixtures=%ld nonfinite_values=%zu first_run_ms=%.1f mean_ms=%.3f rtf_stream=%.6f chunk_ms_p50=%.3f outdir=%s\n", Mconf.path.c_str(), count, nf_all, first, s_chunk / count, s_chunk / 1000.0 / s_audio, p50, mdir.c_str());
  std::fflush(stdout);
  // --- latency phase on fixture --ab-fixture: stream it up to a steady chunk c (every KV slot valid: c = S / Q + 2, or its
  // last chunk), then W + R calls of each graph on chunk c's inputs; no call moves the state on (the caches stay chunk c's
  // inputs). Order: mel (chunk c's PCM) -> pre (that mel) -> conformer (those embeds, chunk c's mask), then mel_first /
  // pre_first on the utterance's first PCM, predict on the last token, joint on frame 0 of the conformer's output ---
  if (rounds > 0) {
    long ab = std::strtol(ab_fixture.c_str(), nullptr, 10); std::vector<float> x;
    if (!load_utt(ab, u, x)) { std::printf("FIXTURE_ERROR ab fixture %s\n", ab_fixture.c_str()); return 3; }
    set_prompt(u.prompt); reset();
    const long n = n_chunks_of(u.samples), target = std::max<long>(1, std::min<long>(n - 1, S / Q + 2));
    for (long k = 0; k < target && k < n; ++k) do_chunk(x, u.samples, k, u);
    std::printf("latency_start wall_ms=%lld ab_fixture=%s id=%s chunk=%ld of %ld\n", WallMs(), ab_fixture.c_str(), u.id.c_str(), target, n); std::fflush(stdout);
    auto pcm_of = [&](long k, std::vector<float>& v) {  // chunk k's PCM slice, zeros past the utterance
      const long len = k == 0 ? A0 : A, s0 = k == 0 ? 0 : ((long)F0 + (long)Fn * (k - 1)) * hop - n_fft / 2; v.assign(len, 0.f);
      for (long i = 0; i < len; ++i) { long j = s0 + i; if (j >= 0 && j < u.samples) v[i] = x[j]; }
    };
    std::vector<float> pcm_c, pcm_0, mel_c((size_t)Fn * kMel), mel_0((size_t)F0 * kMel), emb_c((size_t)Q * kHid), emb_0((size_t)Q * kHid), tmp(kDecState);
    pcm_of(target, pcm_c); pcm_of(0, pcm_0);
    auto timed = [&](int k, M& m, const std::function<double()>& one, const char* what) {
      std::vector<double> v; size_t nfr = 0; double run_ms = 0;
      for (long i = 0; i < warmup; ++i) { auto t0 = Clock::now(); run_ms = one(); std::printf("warmup %ld model[%d] %.3f ms run_ms=%.3f\n", i, k, R3(Ms(t0, Clock::now())), run_ms); }
      for (long r = 0; r < rounds; ++r) {
        auto t0 = Clock::now(); run_ms = one(); const double t = R3(Ms(t0, Clock::now())); v.push_back(t);
        if (k == 0) for (float e : enc) if (!std::isfinite(e)) { ++nfr; break; }
        std::printf("round %ld model[%d] write+run+readback_ms=%.3f run_ms=%.3f\n", r, k, t, run_ms); std::fflush(stdout);
      }
      std::printf("summary model[%d] %s n=%zu median=%.3f min=%.3f max=%.3f nonfinite_rounds=%zu compile_ms=%.1f fully_accelerated=%s call=%s\n", k, m.path.c_str(), v.size(), Median(v),
                  *std::min_element(v.begin(), v.end()), *std::max_element(v.begin(), v.end()), nfr, m.compile_ms, m.fully ? "true" : "false", what);
      std::fflush(stdout);
    };
    timed(2, Mmel, [&] { WriteF(sm.ins[m_in], m_t, pcm_c.data(), A); double r = Run(Mmel, sm); ReadF(sm.outs[m_out], sm.out_types[m_out].element_type, mel_c.data(), mel_c.size()); return r; }, "pcm,run,mel");
    if (!fused) timed(4, Mpre, [&] { WriteF(sp.ins[p_mel], sp.in_types[p_mel].element_type, mel_c.data(), mel_c.size()); double r = Run(Mpre, sp); ReadF(sp.outs[p_emb], sp.out_types[p_emb].element_type, emb_c.data(), emb_c.size()); return r; }, "mel,run,embeds");
    {  // the conformer on chunk c: embeds + mask + one_hot written, run, enc_proj read
      const long valid = std::min<long>(S, (long)Q * target);
      for (int q = 0; q < Q; ++q) for (int j = 0; j < KV; ++j) mask[(size_t)q * KV + j] = j < S - valid ? mask_neg : 0.f;
      auto& ins = (target % 2 == 0) ? cin_even : cin_odd; auto& outs = (target % 2 == 0) ? cout_even : cout_odd;
      if (fused) { WriteF(sc.ins[f_m0], fm0_t, fones.data(), (size_t)R0); WriteF(sc.ins[f_m1], fm1_t, fones.data(), (size_t)R1); }  // chunk c >= 1
      timed(0, Mconf, [&] {
        if (fused) WriteF(sc.ins[f_mel], fmel_t, mel_c.data(), mel_c.size()); else WriteF(sc.ins[c_x], x_t, emb_c.data(), emb_c.size());
        WriteF(sc.ins[c_mask], mask_t, mask.data(), mask.size()); WriteF(sc.ins[c_oh], oh_t, onehot.data(), P);
        auto tr = Clock::now(); LRT_CHECK(LiteRtRunCompiledModel(Mconf.cm, sc.index, ins.size(), ins.data(), outs.size(), outs.data())); double r = Ms(tr, Clock::now());
        ReadF(outs[ce], enc_t, enc.data(), enc.size()); return r; }, fused ? "mel+mask+one_hot,run,enc_proj" : "embeds+mask+one_hot,run,enc_proj");
      if (!swap) {  // the cache move of --state copy (three memcpy between the locked buffers) into a scratch copy of the inputs
        std::vector<double> cv;
        for (long r = 0; r < rounds; ++r) { auto t0 = Clock::now(); for (int cc = 0; cc < NC; ++cc) Copy(setB[cc], setA[cc], cache_bytes[cc]); cv.push_back(R3(Ms(t0, Clock::now()))); }
        size_t cb = 0; for (int cc = 0; cc < NC; ++cc) cb += cache_bytes[cc];
        std::printf("cache_copy n=%zu median=%.3f min=%.3f max=%.3f bytes=%zu\n", cv.size(), Median(cv), *std::min_element(cv.begin(), cv.end()), *std::max_element(cv.begin(), cv.end()), cb);
        std::fflush(stdout);
      }
    }
    timed(1, Mmel0, [&] { WriteF(sm0.ins[m0_in], m0_t, pcm_0.data(), A0); double r = Run(Mmel0, sm0); ReadF(sm0.outs[m0_out], sm0.out_types[m0_out].element_type, mel_0.data(), mel_0.size()); return r; }, "pcm,run,mel");
    if (!fused) timed(3, Mpre0, [&] { WriteF(sp0.ins[p0_mel], sp0.in_types[p0_mel].element_type, mel_0.data(), mel_0.size()); double r = Run(Mpre0, sp0); ReadF(sp0.outs[p0_emb], sp0.out_types[p0_emb].element_type, emb_0.data(), emb_0.size()); return r; }, "mel,run,embeds");
    const long last_tok = u.tok.empty() ? blank : u.tok.back();
    timed(5, Mpred, [&] {
      if (tok_t == kLiteRtElementTypeInt32) { tok32 = (int32_t)last_tok; Write(sd.ins[d_tok], &tok32, 4); } else { tok64 = last_tok; Write(sd.ins[d_tok], &tok64, 8); }
      WriteF(sd.ins[d_h], dh_t, h.data(), kDecState); WriteF(sd.ins[d_c], dc_t, c.data(), kDecState); double r = Run(Mpred, sd);
      ReadF(sd.outs[do_dec], ddo_t, tmp.data(), kEnc); ReadF(sd.outs[do_h], dho_t, tmp.data(), kDecState); ReadF(sd.outs[do_c], dco_t, tmp.data(), kDecState); return r; }, "token+h+c,run,dec_out+h+c");
    timed(k_joint, Mjoint, [&] {
      WriteF(sj.ins[j_dec], jd_t, dec_out.data(), kEnc); WriteF(sj.ins[j_enc], je_t, enc.data(), kEnc); double r = Run(Mjoint, sj);
      ReadF(sj.outs[jo], jo_t, logits.data(), V); double mg; (void)argmax(mg); return r; }, "dec_out+enc_frame,run,logits+argmax");
  }
  for (auto b : own_bufs) LiteRtDestroyTensorBuffer(b);
  for (size_t k = 0; k < ms.size(); ++k) { M& m = ms[k]; for (auto& s : m.sigs) { for (auto b : s.outs) LiteRtDestroyTensorBuffer(b); for (auto b : s.ins) LiteRtDestroyTensorBuffer(b); } LiteRtDestroyCompiledModel(m.cm); LiteRtDestroyOptions(m.opts); LiteRtDestroyModel(m.model); }
  LiteRtDestroyEnvironment(env);
  std::printf("done wall_ms=%lld\n", WallMs());
  std::fflush(stdout);
  return 0;
}
