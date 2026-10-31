// ultra_asr_runner_p: redux_asr_runner_b (the parakeet-redux card's runner) with a pipelined burst pass, for the
// parakeet-ultra LiteRT files on Android (and macOS), written against the LiteRT 2.3.0 C API (CompiledModel).
// --pipeline 0 (default) is redux_asr_runner_b's serial burst pass: per window mel -> encoder -> TDT loop, one thread.
// --pipeline 1 splits the burst pass into two threads joined by an in-order queue of encoded windows: stage A (a second
// thread) reads each file, cuts its windows and runs mel + encoder; stage B (the main thread) runs the TDT loop of each
// window in the same order and prints the file's line once its last window is decoded. Stage A runs up to
// --pipeline-depth (1 or 2) encoded windows ahead of stage B, across file boundaries (file i+1's mel + encoder while
// file i's last windows decode). Each window's TDT loop starts from blank with a zero state in both modes, so the
// tokens do not depend on the mode. The warm-up run and the latency phase stay serial.
// The burst summary also gives burst_wall_ms (from the first window's mel start to the last window's loop end) and
// rtf_wall = burst_wall_ms / 1000 / the summed audio seconds: with --pipeline 1 the per-stage ms overlap and rtf_burst
// (their sum) counts the overlap twice, so a pipelined run is read by rtf_wall.
// One process loads the graphs of up to three window buckets and the decoder:
//   model[0] = the 30 s encoder (--model; on the accelerator of --accel: cpu | gpu | gpu+cpu | npu | npu+cpu),
//   model[1] = the 30 s mel front end (--mel; CPU), model[2] = the decoder (--dec; CPU; signatures predict and joint),
//   model[3] / model[4] = the 10 s encoder / mel pair (--model10 / --mel10, optional),
//   model[5] / model[6] = the 20 s pair (--model20 / --mel20, optional; model[3] / model[4] when there is no 10 s pair).
// The 30 s encoder is compiled first, so the Environment's first accelerated load is an encoder's; the mel graphs and
// the decoder get kLiteRtHwAcceleratorCpu.
// Each file is cut into 30 s windows of --window-samples (479,840 samples; the last one shorter; a window under
// --min-window-samples = 320 samples is skipped and has no text). Each window goes to the smallest bucket whose mel
// audio input holds it (10 s: n <= 160,000 samples, 20 s: n <= 320,000, else the 30 s pair), then mel -> encoder -> the
// greedy TDT loop on the host: the predictor starts from blank (8192); token = argmax of the first 8193 logits, duration
// = argmax of the last 5 (durations 0..4); blank with duration 0 advances one frame; --max-symbols non-blank symbols on
// one frame advance one frame; token 3 (eos) stops the window (--eos -1: never); the loop stops at the valid encoder
// frames of the mel graph's frame_mask, or after 10 x 375 steps.
//
// usage: ultra_asr_runner_p --accel cpu|gpu|gpu+cpu|npu|npu+cpu [--gpu-precision default|fp16|fp32|fp16acc32] [--burst 0|1]
//            [--pipeline 0|1] [--pipeline-depth 1|2] (default 0 / 2)
//            --libdir DIR [--cache-dir DIR] --fixtures DIR --count N [--warmup W] [--rounds R] [--ab-fixture ID]
//            [--dump full|none] --out DIR --model ENC30.tflite --mel MEL30.tflite --dec DEC.tflite
//            [--model10 ENC10.tflite --mel10 MEL10.tflite] [--model20 ENC20.tflite --mel20 MEL20.tflite]
//            [--threads N] [--mel-threads N] [--dec-threads N] (XNNPACK threads of the encoder's CPU part, the mel
//            graphs and the decoder; defaults 4, --threads, --threads)
//            [--window-samples 479840] [--min-window-samples 320] [--max-symbols 10] [--eos 3|-1]
//        ultra_asr_runner_p ... --probe SIGNATURE (one graph: --model's SIGNATURE on DIR/<id3>_<input name>.f32)
// --burst 1 (default) asks the Qualcomm HTP for its burst performance mode (npu); it does not switch the burst pass.
// --libdir: the directory of the LiteRT runtime libraries (libLiteRt.so; libLiteRtClGlAccelerator.so for gpu; the
// Qualcomm dispatch and compiler plugin libraries and the QNN HTP libraries for npu+cpu, which compiles the encoder for
// the HTP on the phone the first time and keeps the result in --cache-dir).
// --fixtures DIR: DIR/index.tsv (one line per file: id3 <TAB> name <TAB> file [<TAB> anything]; id3 = 000, 001, ...;
// a line starting with # is skipped) and the audio files it names: 16 kHz WAV, PCM 16-bit (value / 32768) or IEEE
// float 32-bit (several channels are averaged), or raw little-endian float32 samples. --count N reads 000 .. N-1; file
// 000 is run once more first as a warm-up (not counted).
// stdout: one JSON line per file ({"kind":"utt"}: the token ids "tok", their encoder frame "frame" = window x 375 + the
// frame in the window, their duration "dur", per-window details "win" with the stage stamps t_mel0 / t_enc1 / t_dec0 /
// t_dec1 = ms since the first window's mel start, and the stage times in ms), then {"kind":"burst_summary"} (rtf_burst
// = the summed stage ms / the summed audio seconds, the loads excluded; burst_wall_ms, rtf_wall) and a
// parity_summary line. Stage ms = input write + run + output read back (mel, encoder) and the whole loop (every predict
// / joint call with its writes and reads, the argmax). Text: tdt_text.py decodes each window's tokens with the
// tokenizer and joins the non-empty window texts with one space.
// --dump full also writes <out>/model0/<id3>.f32 per file: n_windows, n_tokens, truncated, valid encoder frames, then
// (token, window x 375 + frame, duration) x 1024, -1 padded, as float32.
// Latency phase (--rounds > 0) on file --ab-fixture: per bucket, the encoder W + R calls and the mel graph R calls on its
// first window, and the whole loop on the 30 s window R times; `summary model[k]` lines.
// Also printed: pid= first, compile_start / compile_end model[k] wall_ms= around each compile, FATAL lines on stdout,
// LiteRT's own log on stderr, done wall_ms= at the end.
#include <unistd.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <mutex>
#include <string>
#include <thread>
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
constexpr int kVocab = 8193;          // token logits (blank = 8192 included)
constexpr int kBlank = 8192;
constexpr int kDurations = 5;         // duration logits; durations [0, 1, 2, 3, 4] = the index
constexpr int kMaxRecTokens = 1024;   // tokens kept per fixture in the dump record
constexpr int kRecElems = 4 + 3 * kMaxRecTokens;
const char* kContract = "v3";         // the file contract this build follows (v3: the 10 / 20 / 30 s bucket graphs)
constexpr int kT30 = 375;             // encoder frames of a 30 s window: the dump's frame = window * kT30 + frame

void DeleteCStringPayload(void* p) { delete[] static_cast<char*>(p); }
double Ms(Clock::time_point a, Clock::time_point b) { return std::chrono::duration<double, std::milli>(b - a).count(); }
double Median(std::vector<double> v) { std::sort(v.begin(), v.end()); size_t n = v.size(); return n ? ((n % 2) ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2])) : 0.0; }
long long WallMs() { struct timespec ts; clock_gettime(CLOCK_REALTIME, &ts); return (long long)ts.tv_sec * 1000 + ts.tv_nsec / 1000000; }
bool ReadFile(const std::string& path, std::vector<uint8_t>& b) {
  FILE* f = std::fopen(path.c_str(), "rb"); if (!f) return false;
  std::fseek(f, 0, SEEK_END); long sz = std::ftell(f); std::fseek(f, 0, SEEK_SET);
  b.resize(sz > 0 ? (size_t)sz : 0); bool ok = b.empty() || std::fread(b.data(), 1, b.size(), f) == b.size(); std::fclose(f); return ok;
}
bool WriteFloatFile(const std::string& path, const float* v, size_t n) {
  FILE* f = std::fopen(path.c_str(), "wb"); if (!f) return false;
  bool ok = std::fwrite(v, 4, n, f) == n; std::fclose(f); return ok;
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
void AddToml(LiteRtOptions options, const char* identifier, const std::string& toml) {
  char* payload = new char[toml.size() + 1]; std::memcpy(payload, toml.data(), toml.size()); payload[toml.size()] = '\0';
  LiteRtOpaqueOptions opaque = nullptr; LRT_CHECK(LiteRtCreateOpaqueOptions(identifier, payload, DeleteCStringPayload, &opaque)); LRT_CHECK(LiteRtAddOpaqueOptions(options, opaque));
}
std::string Dims(const LiteRtRankedTensorType& t) {
  std::string s = "["; for (unsigned i = 0; i < t.layout.rank; ++i) { if (i) s += ","; s += std::to_string(t.layout.dimensions[i]); } return s + "]";
}
const char* TypeName(LiteRtElementType e) { return e == kLiteRtElementTypeFloat32 ? "f32" : e == kLiteRtElementTypeInt32 ? "i32" : e == kLiteRtElementTypeInt64 ? "i64" : "other"; }

struct Sig {  // one signature of a compiled model: its tensors in signature order and their managed buffers
  LiteRtParamIndex index = 0; std::string key;
  std::vector<std::string> in_names, out_names; std::vector<LiteRtRankedTensorType> in_types, out_types;
  std::vector<size_t> in_elems, out_elems; std::vector<LiteRtTensorBuffer> ins, outs;
  int In(const std::vector<std::string>& names) const { for (auto& n : names) for (size_t i = 0; i < in_names.size(); ++i) if (in_names[i] == n) return (int)i; return -1; }
  int Out(const std::vector<std::string>& names) const { for (auto& n : names) for (size_t i = 0; i < out_names.size(); ++i) if (out_names[i] == n) return (int)i; return -1; }
};
struct M {
  std::string path, role; LiteRtModel model = nullptr; LiteRtOptions opts = nullptr; LiteRtCompiledModel cm = nullptr;
  std::vector<Sig> sigs; double compile_ms = 0; bool fully = false, non_cpu_fully = false;
  Sig& S(const std::string& key) {
    for (auto& s : sigs) if (s.key == key) return s;
    if (key.empty() && sigs.size() == 1) return sigs[0];
    std::printf("FATAL %s: no signature '%s' (has", role.c_str(), key.c_str()); for (auto& s : sigs) std::printf(" %s", s.key.c_str()); std::printf(")\n"); std::fflush(stdout); std::exit(2);
  }
};
// the tensor index by the contract's name (several spellings accepted); else the only tensor with that element count
int Pick(const Sig& s, bool input, const std::vector<std::string>& names, size_t elems, const char* what) {
  int i = input ? s.In(names) : s.Out(names);
  if (i < 0) {
    const auto& e = input ? s.in_elems : s.out_elems; int hit = -1, n = 0;
    for (size_t k = 0; k < e.size(); ++k) if (e[k] == elems) { hit = (int)k; ++n; }
    if (n == 1) i = hit;
  }
  if (i < 0 || (input ? s.in_elems[i] : s.out_elems[i]) != elems) {
    std::printf("FATAL signature %s: %s %s (%zu elements) not found by name or a unique element count; tensors:", s.key.c_str(), input ? "input" : "output", what, elems);
    const auto& nm = input ? s.in_names : s.out_names; const auto& e = input ? s.in_elems : s.out_elems;
    for (size_t k = 0; k < nm.size(); ++k) std::printf(" %s(%zu)", nm[k].c_str(), e[k]);
    std::printf("\n"); std::fflush(stdout); std::exit(2);
  }
  return i;
}
void Write(LiteRtTensorBuffer b, const void* src, size_t bytes) { void* h = nullptr; LRT_CHECK(LiteRtLockTensorBuffer(b, &h, kLiteRtTensorBufferLockModeWrite)); std::memcpy(h, src, bytes); LRT_CHECK(LiteRtUnlockTensorBuffer(b)); }
void Read(LiteRtTensorBuffer b, void* dst, size_t bytes) { void* h = nullptr; LRT_CHECK(LiteRtLockTensorBuffer(b, &h, kLiteRtTensorBufferLockModeRead)); std::memcpy(dst, h, bytes); LRT_CHECK(LiteRtUnlockTensorBuffer(b)); }
double Run(M& m, Sig& s) { auto t = Clock::now(); LRT_CHECK(LiteRtRunCompiledModel(m.cm, s.index, s.ins.size(), s.ins.data(), s.outs.size(), s.outs.data())); return Ms(t, Clock::now()); }
std::vector<std::string> Split(const std::string& s, char sep) { std::vector<std::string> r; size_t p = 0; while (p <= s.size()) { size_t q = s.find(sep, p); if (q == std::string::npos) q = s.size(); r.push_back(s.substr(p, q - p)); p = q + 1; } return r; }
std::string JsonStr(const std::string& s) { std::string o = "\""; for (char c : s) { if (c == '"' || c == '\\') { o += '\\'; o += c; } else if ((unsigned char)c < 0x20) { char b[8]; std::snprintf(b, sizeof b, "\\u%04x", c); o += b; } else o += c; } return o + "\""; }
template <class T> std::string JsonArr(const std::vector<T>& v) { std::string o = "["; for (size_t i = 0; i < v.size(); ++i) { if (i) o += ","; o += std::to_string(v[i]); } return o + "]"; }
std::string F(double v, int prec = 3) { char b[64]; std::snprintf(b, sizeof b, "%.*f", prec, v); return b; }
}  // namespace

int main(int argc, char** argv) {
  std::string accel = "cpu", gpu_precision = "fp32", libdir, cache_dir, fixtures, out_dir, dump = "full", probe;
  std::string enc_path, mel_path, dec_path, ab_fixture = "000", enc10_path, mel10_path, enc20_path, mel20_path;
  int burst = 1, threads = 4, mel_threads = 0, dec_threads = 0, pipeline = 0, pipeline_depth = 2; long warmup = 2, rounds = 8, count = 0, win = 479840, min_win = 320, max_symbols = 10, eos = 3;
  for (int i = 1; i < argc; ++i) {
    std::string f = argv[i];
    auto next = [&](const char* what) -> const char* { if (i + 1 >= argc) { std::fprintf(stderr, "missing value for %s\n", what); std::exit(1); } return argv[++i]; };
    if (f == "--accel") accel = next("--accel");
    else if (f == "--gpu-precision") gpu_precision = next("--gpu-precision");
    else if (f == "--burst") burst = std::atoi(next("--burst"));
    else if (f == "--pipeline") pipeline = std::atoi(next("--pipeline"));
    else if (f == "--pipeline-depth") pipeline_depth = std::atoi(next("--pipeline-depth"));
    else if (f == "--libdir") libdir = next("--libdir");
    else if (f == "--cache-dir") cache_dir = next("--cache-dir");
    else if (f == "--fixtures") fixtures = next("--fixtures");
    else if (f == "--count") count = std::strtol(next("--count"), nullptr, 10);
    else if (f == "--warmup") warmup = std::strtol(next("--warmup"), nullptr, 10);
    else if (f == "--rounds") rounds = std::strtol(next("--rounds"), nullptr, 10);
    else if (f == "--ab-fixture") ab_fixture = next("--ab-fixture");
    else if (f == "--out") out_dir = next("--out");
    else if (f == "--dump") dump = next("--dump");
    else if (f == "--model") enc_path = next("--model");
    else if (f == "--mel") mel_path = next("--mel");
    else if (f == "--dec") dec_path = next("--dec");
    else if (f == "--model10") enc10_path = next("--model10");
    else if (f == "--mel10") mel10_path = next("--mel10");
    else if (f == "--model20") enc20_path = next("--model20");
    else if (f == "--mel20") mel20_path = next("--mel20");
    else if (f == "--threads") threads = std::atoi(next("--threads"));
    else if (f == "--mel-threads") mel_threads = std::atoi(next("--mel-threads"));
    else if (f == "--dec-threads") dec_threads = std::atoi(next("--dec-threads"));
    else if (f == "--window-samples") win = std::strtol(next("--window-samples"), nullptr, 10);
    else if (f == "--min-window-samples") min_win = std::strtol(next("--min-window-samples"), nullptr, 10);
    else if (f == "--max-symbols") max_symbols = std::strtol(next("--max-symbols"), nullptr, 10);
    else if (f == "--eos") eos = std::strtol(next("--eos"), nullptr, 10);
    else if (f == "--probe") probe = next("--probe");
    else { std::fprintf(stderr, "unknown flag %s\n", f.c_str()); return 1; }
  }
  const bool asr = probe.empty();
  if (enc_path.empty() || libdir.empty() || fixtures.empty() || out_dir.empty() || count < 1 || (asr && (mel_path.empty() || dec_path.empty()))) {
    std::fprintf(stderr, "need --model, --libdir, --fixtures, --count, --out (and --mel, --dec unless --probe)\n"); return 1;
  }
  if (accel != "cpu" && accel != "gpu" && accel != "gpu+cpu" && accel != "npu" && accel != "npu+cpu") { std::fprintf(stderr, "unknown --accel %s\n", accel.c_str()); return 1; }
  if (enc10_path.empty() != mel10_path.empty() || enc20_path.empty() != mel20_path.empty()) { std::fprintf(stderr, "--model10 needs --mel10 and --model20 needs --mel20 (and back)\n"); return 1; }
  if (!asr && !(enc10_path.empty() && enc20_path.empty())) { std::fprintf(stderr, "--probe takes no bucket graphs\n"); return 1; }
  if ((pipeline != 0 && pipeline != 1) || pipeline_depth < 1 || pipeline_depth > 2) { std::fprintf(stderr, "--pipeline takes 0 or 1, --pipeline-depth 1 or 2\n"); return 1; }
  const int n_models = !asr ? 1 : 3 + (enc10_path.empty() ? 0 : 2) + (enc20_path.empty() ? 0 : 2);
  if (mel_threads <= 0) mel_threads = threads;
  if (dec_threads <= 0) dec_threads = threads;
  ::mkdir(out_dir.c_str(), 0775);
  std::printf("pid=%d accel=%s gpu_precision=%s burst=%d models=%d fixtures=%ld warmup=%ld rounds=%ld ab_fixture=%s libdir=%s cache_dir=%s dump=%s threads=%d mel_threads=%d dec_threads=%d mode=%s window_samples=%ld min_window_samples=%ld max_symbols=%ld eos=%ld pipeline=%d pipeline_depth=%d contract=%s wall_ms=%lld\n",
              (int)getpid(), accel.c_str(), gpu_precision.c_str(), burst, n_models, count, warmup, rounds, ab_fixture.c_str(), libdir.c_str(), cache_dir.c_str(), dump.c_str(), threads, mel_threads, dec_threads,
              asr ? "asr" : ("probe:" + probe).c_str(), win, min_win, max_symbols, eos, pipeline, pipeline_depth, kContract, WallMs());
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
  const LiteRtHwAcceleratorSet hw_enc = accel == "npu" ? kLiteRtHwAcceleratorNpu : accel == "npu+cpu" ? (LiteRtHwAcceleratorSet)(kLiteRtHwAcceleratorNpu | kLiteRtHwAcceleratorCpu)
                                       : accel == "gpu" ? kLiteRtHwAcceleratorGpu : accel == "gpu+cpu" ? (LiteRtHwAcceleratorSet)(kLiteRtHwAcceleratorGpu | kLiteRtHwAcceleratorCpu)
                                       : kLiteRtHwAcceleratorCpu;
  const int prec = gpu_precision == "default" ? (int)kLiteRtDelegatePrecisionDefault : gpu_precision == "fp16" ? (int)kLiteRtDelegatePrecisionFp16
                 : gpu_precision == "fp16acc32" ? (int)kLiteRtDelegatePrecisionFp16WithFp32Accum : (int)kLiteRtDelegatePrecisionFp32;
  std::vector<M> ms(n_models);
  ms[0].path = enc_path; ms[0].role = asr ? "enc" : "probe";
  if (asr) { ms[1].path = mel_path; ms[1].role = "mel"; ms[2].path = dec_path; ms[2].role = "dec"; }
  {  // the bucket graphs: model[3] / [4] = enc10 / mel10, then enc20 / mel20 (model[3] / [4] when there is no 10 s pair)
    size_t k = 3;
    if (!enc10_path.empty()) { ms[k].path = enc10_path; ms[k].role = "enc10"; ms[k + 1].path = mel10_path; ms[k + 1].role = "mel10"; k += 2; }
    if (!enc20_path.empty()) { ms[k].path = enc20_path; ms[k].role = "enc20"; ms[k + 1].path = mel20_path; ms[k + 1].role = "mel20"; }
  }
  for (size_t k = 0; k < ms.size(); ++k) {  // model[0] (the accelerated one) first
    M& m = ms[k]; const bool is_enc = k == 0 || m.role.rfind("enc", 0) == 0;
    const LiteRtHwAcceleratorSet hw = is_enc ? hw_enc : kLiteRtHwAcceleratorCpu;
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
        std::printf("model[%zu] %s sig[%u]=%s input[%u] name=%s type=%s dims=%s elems=%zu\n", k, m.role.c_str(), (unsigned)si, s.key.c_str(), (unsigned)i, s.in_names.back().c_str(), TypeName(s.in_types[i].element_type), Dims(s.in_types[i]).c_str(), s.in_elems[i]);
      }
      for (LiteRtParamIndex o = 0; o < n_out; ++o) {
        LiteRtTensor t = nullptr; LRT_CHECK(LiteRtGetSignatureOutputTensorByIndex(sig, o, &t)); LRT_CHECK(LiteRtGetRankedTensorType(t, &s.out_types[o])); LRT_CHECK(LiteRtGetNumLayoutElements(&s.out_types[o].layout, &s.out_elems[o]));
        const char* nm = nullptr; LRT_CHECK(LiteRtGetSignatureOutputName(sig, o, &nm)); s.out_names.push_back(nm ? nm : "");
        std::printf("model[%zu] %s sig[%u]=%s output[%u] name=%s type=%s dims=%s elems=%zu\n", k, m.role.c_str(), (unsigned)si, s.key.c_str(), (unsigned)o, s.out_names.back().c_str(), TypeName(s.out_types[o].element_type), Dims(s.out_types[o]).c_str(), s.out_elems[o]);
      }
      m.sigs.push_back(s);
    }
    LRT_CHECK(LiteRtCreateOptions(&m.opts));
    LRT_CHECK(LiteRtSetOptionsHardwareAccelerators(m.opts, hw));
    AddToml(m.opts, "runtime_options_string", "enable_profiling = false\nerror_reporter_mode = 1\n");
    const int nt = is_enc ? threads : m.role.rfind("mel", 0) == 0 ? mel_threads : dec_threads;
    if (hw & kLiteRtHwAcceleratorCpu) AddToml(m.opts, "xnnpack", "num_threads = " + std::to_string(nt) + "\n");
    if (hw & kLiteRtHwAcceleratorGpu) AddToml(m.opts, "gpu_options", "precision = " + std::to_string(prec) + "\n");
    if ((hw & kLiteRtHwAcceleratorNpu) && burst) AddToml(m.opts, "qualcomm", "htp_performance_mode = 2\n");  // kLiteRtQualcommHtpPerformanceModeBurst
    std::printf("compile_start model[%zu] wall_ms=%lld\n", k, WallMs()); std::fflush(stdout);
    auto c0 = Clock::now(); LRT_CHECK(LiteRtCreateCompiledModel(env, m.model, m.opts, &m.cm)); m.compile_ms = Ms(c0, Clock::now());
    std::printf("compile_end model[%zu] wall_ms=%lld\n", k, WallMs()); std::fflush(stdout);
    // informational only: a status other than ok is printed, not fatal
    LiteRtStatus fs = LiteRtCompiledModelIsFullyAccelerated(m.cm, &m.fully);
    LiteRtStatus ns = LiteRtCompiledModelIsNonCpuFullyAccelerated(m.cm, &m.non_cpu_fully);
    if (fs != kLiteRtStatusOk || ns != kLiteRtStatusOk) std::printf("model[%zu] accelerated query status fully=%d non_cpu=%d\n", k, (int)fs, (int)ns);
    for (auto& s : m.sigs) {
      s.ins.assign(s.in_types.size(), nullptr); s.outs.assign(s.out_types.size(), nullptr);
      for (size_t i = 0; i < s.ins.size(); ++i) { LiteRtTensorBufferRequirements r = nullptr; LRT_CHECK(LiteRtGetCompiledModelInputBufferRequirements(m.cm, s.index, i, &r)); LRT_CHECK(LiteRtCreateManagedTensorBufferFromRequirements(env, &s.in_types[i], r, &s.ins[i])); }
      for (size_t o = 0; o < s.outs.size(); ++o) { LiteRtTensorBufferRequirements r = nullptr; LRT_CHECK(LiteRtGetCompiledModelOutputBufferRequirements(m.cm, s.index, o, &r)); LRT_CHECK(LiteRtCreateManagedTensorBufferFromRequirements(env, &s.out_types[o], r, &s.outs[o])); }
    }
    std::printf("model[%zu]=%s role=%s accel=%s signatures=%zu compile_ms=%.1f fully_accelerated=%s non_cpu_fully_accelerated=%s\n", k, m.path.c_str(), m.role.c_str(),
                hw == kLiteRtHwAcceleratorCpu ? "cpu" : accel.c_str(), m.sigs.size(), m.compile_ms, m.fully ? "true" : "false", m.non_cpu_fully ? "true" : "false");
    std::fflush(stdout);
  }
  const std::string mdir = out_dir + "/model0"; ::mkdir(mdir.c_str(), 0775);
  auto id3 = [](long i) { char idbuf[24]; std::snprintf(idbuf, sizeof idbuf, "%03ld", i); return std::string(idbuf); };  // both pipeline stages call it

  if (!asr) {  // --- probe: one signature of one graph on raw float32 input files ---
    M& m = ms[0]; Sig& s = m.S(probe);
    std::vector<std::vector<float>> in(s.ins.size());
    auto load = [&](const std::string& id) -> bool {
      for (size_t i = 0; i < s.ins.size(); ++i) {
        std::vector<uint8_t> b; std::string p = fixtures + "/" + id + "_" + s.in_names[i] + ".f32";
        if (!ReadFile(p, b) || b.size() != s.in_elems[i] * 4) { std::printf("FIXTURE_ERROR %s (%zu bytes, want %zu)\n", p.c_str(), b.size(), s.in_elems[i] * 4); std::fflush(stdout); return false; }
        in[i].resize(s.in_elems[i]); std::memcpy(in[i].data(), b.data(), b.size());
      }
      return true;
    };
    std::vector<float> out0(s.out_elems[0]); double last_run = 0;
    auto run_one = [&]() -> double {
      auto t0 = Clock::now(); for (size_t i = 0; i < s.ins.size(); ++i) Write(s.ins[i], in[i].data(), s.in_elems[i] * 4);
      last_run = Run(m, s); Read(s.outs[0], out0.data(), out0.size() * 4); return Ms(t0, Clock::now());
    };
    size_t nf_all = 0; double first = -1, sum = 0;
    for (long fx = 0; fx < count; ++fx) {
      std::string id = id3(fx); if (!load(id)) return 3;
      if (fx == 0) first = run_one();
      double t = run_one(); sum += t; size_t nf = 0; for (float v : out0) if (!std::isfinite(v)) ++nf; nf_all += nf;
      if (dump == "full" && !WriteFloatFile(mdir + "/" + id + ".f32", out0.data(), out0.size())) { std::printf("WRITE_ERROR %s/%s.f32\n", mdir.c_str(), id.c_str()); return 3; }
      std::printf("parity model[0] fixture %s ms=%.3f run_ms=%.3f nonfinite=%zu of=%zu out0[0..3]=%.6g,%.6g,%.6g,%.6g\n", id.c_str(), t, last_run, nf, out0.size(),
                  out0.size() > 0 ? out0[0] : 0.f, out0.size() > 1 ? out0[1] : 0.f, out0.size() > 2 ? out0[2] : 0.f, out0.size() > 3 ? out0[3] : 0.f);
      std::fflush(stdout);
    }
    std::printf("parity_summary model[0] %s fixtures=%ld nonfinite_values=%zu first_run_ms=%.1f mean_ms=%.3f outdir=%s\n", m.path.c_str(), count, nf_all, first, sum / count, mdir.c_str());
    if (rounds > 0) {
      if (!load(ab_fixture)) return 3;
      std::printf("latency_start wall_ms=%lld\n", WallMs());
      std::vector<double> v; size_t nfr = 0;
      for (long w = 0; w < warmup; ++w) { double t = run_one(); std::printf("warmup %ld model[0] %.3f ms run_ms=%.3f\n", w, t, last_run); }
      for (long r = 0; r < rounds; ++r) { double t = run_one(); v.push_back(t); for (float x : out0) if (!std::isfinite(x)) { ++nfr; break; } std::printf("round %ld model[0] write+run+readback_ms=%.3f run_ms=%.3f\n", r, t, last_run); std::fflush(stdout); }
      std::printf("summary model[0] %s n=%zu median=%.3f min=%.3f max=%.3f nonfinite_rounds=%zu compile_ms=%.1f fully_accelerated=%s\n", m.path.c_str(), v.size(), Median(v),
                  *std::min_element(v.begin(), v.end()), *std::max_element(v.begin(), v.end()), nfr, m.compile_ms, m.fully ? "true" : "false");
    }
  } else {  // --- ASR: mel<b> -> encoder<b> -> TDT loop per 30 s window, b = the window's bucket ---
    M& dec = ms[2];
    Sig& sp = dec.S("predict"); Sig& sj = dec.S("joint");
    const int pi_tok = Pick(sp, true, {"token", "args_0"}, 1, "token");
    const int pi_h = Pick(sp, true, {"h", "args_1"}, 1280, "h");
    const int pi_c = Pick(sp, true, {"c", "args_2"}, 1280, "c");
    const int po_dec = Pick(sp, false, {"dec_out", "output_0"}, 640, "dec_out");
    const int po_h = Pick(sp, false, {"h_out", "h1", "new_h", "output_1"}, 1280, "h'");
    const int po_c = Pick(sp, false, {"c_out", "c1", "new_c", "output_2"}, 1280, "c'");
    const int ji_enc = Pick(sj, true, {"enc_frame", "args_0"}, 640, "enc_frame");
    const int ji_dec = Pick(sj, true, {"dec_out", "args_1"}, 640, "dec_out");
    const int jo_log = Pick(sj, false, {"logits", "output_0"}, kVocab + kDurations, "logits");
    if (pi_h == pi_c || po_h == po_c || ji_enc == ji_dec) { std::printf("FATAL ambiguous tensor binding (names not in the contract)\n"); return 2; }
    if (sp.in_types[pi_tok].element_type != kLiteRtElementTypeInt32) { std::printf("FATAL the token input must be int32\n"); return 2; }
    // one bucket = an encoder and its mel graph: [0] = the 30 s pair (model[0] / model[1]), then the 10 s / 20 s pairs
    struct Bucket {
      int enc_k = 0, mel_k = 1, secs = 30; M* enc = nullptr; M* mel = nullptr; Sig* se = nullptr; Sig* sm = nullptr;
      int mi_audio = -1, mi_valid = -1, mo_feat = -1, mo_melmask = -1, mo_framemask = -1, ei_feat = -1, ei_melmask = -1, eo_enc = -1;
      size_t A = 0, T = 0, Tm = 0; std::vector<float> audio_win, feat, melmask, framemask, encv;
    };
    std::vector<Bucket> bk;
    auto add_bucket = [&](int ek, int mk) -> bool {
      Bucket b; b.enc_k = ek; b.mel_k = mk; b.enc = &ms[ek]; b.mel = &ms[mk]; b.se = &b.enc->S(""); b.sm = &b.mel->S("");
      Sig& sm = *b.sm; Sig& se = *b.se;
      // mel<b>: audio [1, 160 x Tm] f32 + valid samples [1] i32 -> features [1,128,Tm] + mel_mask [1,Tm] + frame_mask [1,T]
      b.mi_audio = sm.In({"audio", "args_0"});  // else the largest float32 input
      if (b.mi_audio < 0) for (size_t i = 0; i < sm.in_elems.size(); ++i) if (sm.in_types[i].element_type == kLiteRtElementTypeFloat32 && (b.mi_audio < 0 || sm.in_elems[i] > sm.in_elems[b.mi_audio])) b.mi_audio = (int)i;
      if (b.mi_audio < 0) { std::printf("FATAL %s has no float32 audio input\n", b.mel->role.c_str()); return false; }
      b.A = sm.in_elems[b.mi_audio];
      b.T = se.out_elems.empty() ? 0 : se.out_elems[0] / 640; b.Tm = 8 * b.T; b.secs = (int)(b.Tm / 100);
      b.mi_valid = Pick(sm, true, {"n_valid", "valid", "num_samples", "args_1"}, 1, "valid samples");
      b.mo_feat = Pick(sm, false, {"features"}, 128 * b.Tm, "features");
      b.mo_melmask = Pick(sm, false, {"mel_mask"}, b.Tm, "mel_mask");
      b.mo_framemask = Pick(sm, false, {"frame_mask"}, b.T, "frame_mask");
      b.ei_feat = Pick(se, true, {"features", "args_0"}, 128 * b.Tm, "features");
      b.ei_melmask = Pick(se, true, {"mel_mask", "args_1"}, b.Tm, "mel_mask");
      b.eo_enc = Pick(se, false, {"enc", "output_0"}, b.T * 640, "enc");
      if (b.mi_audio == b.mi_valid || b.ei_feat == b.ei_melmask) { std::printf("FATAL ambiguous tensor binding (names not in the contract)\n"); return false; }
      if (sm.in_types[b.mi_valid].element_type != kLiteRtElementTypeInt32) { std::printf("FATAL the valid-samples input must be int32\n"); return false; }
      if (b.T == 0 || b.A < 160 * b.Tm) { std::printf("FATAL %s audio input %zu < 160 x %zu mel frames, or no encoder frames\n", b.mel->role.c_str(), b.A, b.Tm); return false; }
      b.audio_win.assign(b.A, 0.f); b.feat.assign(128 * b.Tm, 0.f); b.melmask.assign(b.Tm, 0.f); b.framemask.assign(b.T, 0.f); b.encv.assign(b.T * 640, 0.f);
      std::printf("binding bucket %ds model[%d] %s[%s] audio=%s(%zu) valid=%s -> features=%s mel_mask=%s frame_mask=%s | model[%d] %s[%s] features=%s mel_mask=%s -> enc=%s T=%zu\n",
                  b.secs, mk, b.mel->role.c_str(), sm.key.c_str(), sm.in_names[b.mi_audio].c_str(), b.A, sm.in_names[b.mi_valid].c_str(), sm.out_names[b.mo_feat].c_str(),
                  sm.out_names[b.mo_melmask].c_str(), sm.out_names[b.mo_framemask].c_str(), ek, b.enc->role.c_str(), se.key.c_str(), se.in_names[b.ei_feat].c_str(),
                  se.in_names[b.ei_melmask].c_str(), se.out_names[b.eo_enc].c_str(), b.T);
      bk.push_back(std::move(b)); return true;
    };
    if (!add_bucket(0, 1)) return 2;
    for (size_t k = 3; k + 1 < ms.size(); k += 2) if (!add_bucket((int)k, (int)k + 1)) return 2;
    if (bk[0].A < (size_t)win) { std::printf("FATAL mel30 audio input %zu < window %ld\n", bk[0].A, win); return 2; }
    std::vector<size_t> by_size(bk.size()); for (size_t i = 0; i < bk.size(); ++i) by_size[i] = i;
    std::sort(by_size.begin(), by_size.end(), [&](size_t a, size_t b) { return bk[a].A < bk[b].A; });
    for (size_t i = 1; i < by_size.size(); ++i) if (bk[by_size[i]].A == bk[by_size[i - 1]].A) { std::printf("FATAL two buckets with the same audio input %zu\n", bk[by_size[i]].A); return 2; }
    // the smallest bucket whose mel audio input holds the window (the 30 s pair holds every window)
    auto pick_bucket = [&](long n) -> size_t { for (size_t i : by_size) if ((long)bk[i].A >= n) return i; return 0; };
    std::printf("binding predict token=%s h=%s c=%s -> dec_out=%s h'=%s c'=%s | joint enc_frame=%s dec_out=%s -> logits=%s | buckets=%zu\n",
                sp.in_names[pi_tok].c_str(), sp.in_names[pi_h].c_str(), sp.in_names[pi_c].c_str(), sp.out_names[po_dec].c_str(), sp.out_names[po_h].c_str(), sp.out_names[po_c].c_str(),
                sj.in_names[ji_enc].c_str(), sj.in_names[ji_dec].c_str(), sj.out_names[jo_log].c_str(), bk.size());
    std::fflush(stdout);
    // fixture index: id3 -> (utterance id, file)
    std::vector<std::string> fx_id, fx_file; {
      std::vector<uint8_t> b; if (!ReadFile(fixtures + "/index.tsv", b)) { std::printf("FIXTURE_ERROR %s/index.tsv\n", fixtures.c_str()); return 3; }
      for (auto& line : Split(std::string(b.begin(), b.end()), '\n')) {
        if (line.empty() || line[0] == '#') continue; auto c = Split(line, '\t'); if (c.size() < 3) continue;
        long i = std::strtol(c[0].c_str(), nullptr, 10); if ((long)fx_id.size() <= i) { fx_id.resize(i + 1); fx_file.resize(i + 1); }
        fx_id[i] = c[1]; fx_file[i] = c[2];
      }
    }
    std::vector<float> dec_out(640), h(1280), c(1280), logits(kVocab + kDurations);
    double mel_run = 0, enc_run = 0, loop_run = 0;
    // one window: mel<b> (+ timing) -> encoder<b> (+ timing) -> loop; returns the window's record pieces
    struct Win { long samples = 0; int bucket = 30, T = kT30, mel_valid = 0, enc_valid = 0, steps = 0, cap_hits = 0, max_run = 0; bool eos = false; size_t nonfinite = 0, nonfinite_all = 0;
                 double mel_ms = 0, enc_ms = 0, dec_ms = 0, mel_run_ms = 0, enc_run_ms = 0, dec_run_ms = 0;
                 Clock::time_point t_mel0, t_enc1, t_dec0, t_dec1;  // mel start, encoder end, loop start, loop end
                 std::vector<int> tok, frame, dur; };
    auto do_mel = [&](Bucket& b, const float* x, long n) -> double {
      auto t0 = Clock::now(); std::fill(b.audio_win.begin(), b.audio_win.end(), 0.f); std::memcpy(b.audio_win.data(), x, n * 4);
      int32_t nv = (int32_t)n; Write(b.sm->ins[b.mi_audio], b.audio_win.data(), b.A * 4); Write(b.sm->ins[b.mi_valid], &nv, 4);
      mel_run = Run(*b.mel, *b.sm);
      Read(b.sm->outs[b.mo_feat], b.feat.data(), b.feat.size() * 4); Read(b.sm->outs[b.mo_melmask], b.melmask.data(), b.melmask.size() * 4);
      Read(b.sm->outs[b.mo_framemask], b.framemask.data(), b.framemask.size() * 4);
      return Ms(t0, Clock::now());
    };
    // the encoder output goes to dst (b.encv.size() floats): the bucket's own encv in the serial pass, the queued window's
    // copy in the pipelined one (stage A writes the next window of the same bucket while stage B reads this one)
    auto do_enc = [&](Bucket& b, float* dst) -> double {
      auto t0 = Clock::now(); Write(b.se->ins[b.ei_feat], b.feat.data(), b.feat.size() * 4); Write(b.se->ins[b.ei_melmask], b.melmask.data(), b.melmask.size() * 4);
      enc_run = Run(*b.enc, *b.se); Read(b.se->outs[b.eo_enc], dst, b.encv.size() * 4); return Ms(t0, Clock::now());
    };
    auto predict = [&](int32_t tok) {
      Write(sp.ins[pi_tok], &tok, 4); Write(sp.ins[pi_h], h.data(), 1280 * 4); Write(sp.ins[pi_c], c.data(), 1280 * 4);
      loop_run += Run(dec, sp); Read(sp.outs[po_dec], dec_out.data(), 640 * 4); Read(sp.outs[po_h], h.data(), 1280 * 4); Read(sp.outs[po_c], c.data(), 1280 * 4);
    };
    auto do_loop = [&](const float* encv, size_t T, int valid_T, Win& w) -> double {  // encv = [T, 640] of the window
      auto t0 = Clock::now(); loop_run = 0; w.tok.clear(); w.frame.clear(); w.dur.clear(); w.steps = 0; w.cap_hits = 0; w.max_run = 0; w.eos = false;
      std::fill(h.begin(), h.end(), 0.f); std::fill(c.begin(), c.end(), 0.f); predict(kBlank);
      int t = 0, on_frame = 0; const long max_steps = 10L * (long)kT30;  // max_symbols_per_step (10) x 375 frames, in every bucket
      // argmax as torch / numpy: the first maximum, a NaN counts as the maximum (the first NaN wins)
      auto argmax = [&](int lo, int n) { int bi = 0; for (int i = 1; i < n; ++i) { float v = logits[lo + i], m = logits[lo + bi]; if (std::isnan(m)) break; if (std::isnan(v) || v > m) bi = i; } return bi; };
      while (t < valid_T && w.steps < max_steps) {
        ++w.steps; const int tt = std::min(t, (int)T - 1);
        Write(sj.ins[ji_enc], encv + (size_t)tt * 640, 640 * 4); Write(sj.ins[ji_dec], dec_out.data(), 640 * 4);
        loop_run += Run(dec, sj); Read(sj.outs[jo_log], logits.data(), logits.size() * 4);
        int tok = argmax(0, kVocab);
        int d = argmax(kVocab, kDurations);
        if (tok == kBlank && d == 0) d = 1;
        if (tok != kBlank) {
          w.tok.push_back(tok); w.frame.push_back(t); w.dur.push_back(d);
          if (eos >= 0 && tok == eos) { w.eos = true; break; }
          predict(tok);
        }
        if (d == 0) { on_frame += tok != kBlank ? 1 : 0; w.max_run = std::max(w.max_run, on_frame); if (max_symbols > 0 && on_frame >= max_symbols) { ++w.cap_hits; d = 1; } }
        if (d > 0) on_frame = 0;
        t += d;
      }
      w.dec_run_ms = loop_run; return Ms(t0, Clock::now());
    };
    // stage A of one window: mel<b> -> encoder<b> into encv (b.encv.size() floats), the valid frames and the non-finite count
    auto mel_enc = [&](Bucket& b, const float* x, long n, Win& w, float* encv) {
      w.t_mel0 = Clock::now();
      w.samples = n; w.bucket = b.secs; w.T = (int)b.T; w.mel_ms = do_mel(b, x, n); w.mel_run_ms = mel_run;
      w.mel_valid = 0; for (float v : b.melmask) w.mel_valid += v > 0.5f; w.enc_valid = 0; for (float v : b.framemask) w.enc_valid += v > 0.5f;
      w.enc_ms = do_enc(b, encv); w.enc_run_ms = enc_run;
      w.nonfinite = 0; w.nonfinite_all = 0;
      for (size_t i = 0; i < b.encv.size(); ++i) if (!std::isfinite(encv[i])) { ++w.nonfinite_all; if (i / 640 < (size_t)w.enc_valid) ++w.nonfinite; }
      w.t_enc1 = Clock::now();
    };
    // stage B of one window: the TDT loop over its valid encoder frames
    auto dec_win = [&](const float* encv, size_t T, Win& w) { w.t_dec0 = Clock::now(); w.dec_ms = do_loop(encv, T, w.enc_valid, w); w.t_dec1 = Clock::now(); };
    auto run_window = [&](const float* x, long n, Win& w) {
      Bucket& b = bk[pick_bucket(n)];
      mel_enc(b, x, n, w, b.encv.data());
      dec_win(b.encv.data(), b.T, w);
    };
    struct Utt { std::string id; long samples = 0; std::string fmt; std::vector<Win> wins; int skipped = 0; double mel = 0, encm = 0, decm = 0; };
    auto load_utt = [&](long fx, Utt& u, std::vector<float>& x) -> bool {
      std::string fmt;
      if (fx >= (long)fx_id.size() || fx_id[fx].empty()) { std::printf("FIXTURE_ERROR %s/index.tsv has no line %s\n", fixtures.c_str(), id3(fx).c_str()); return false; }
      if (!ReadAudio(fixtures + "/" + fx_file[fx], x, fmt)) { std::printf("FIXTURE_ERROR %s/%s (%s)\n", fixtures.c_str(), fx_file[fx].c_str(), fmt.c_str()); return false; }
      u = Utt(); u.id = fx_id[fx]; u.samples = (long)x.size(); u.fmt = fmt;
      return true;
    };
    auto run_utt = [&](long fx, Utt& u) -> bool {
      std::vector<float> x;
      if (!load_utt(fx, u, x)) return false;
      for (long s0 = 0; s0 < (long)x.size(); s0 += win) {
        long n = std::min(win, (long)x.size() - s0);
        if (n < min_win) { ++u.skipped; continue; }
        Win w; run_window(x.data() + s0, n, w); u.mel += w.mel_ms; u.encm += w.enc_ms; u.decm += w.dec_ms; u.wins.push_back(w);
      }
      return true;
    };
    // --- the burst pass: one warm-up run of fixture 000 (not counted), then every fixture once ---
    Utt u; double first = 0; { auto t0 = Clock::now(); if (!run_utt(0, u)) return 3; first = Ms(t0, Clock::now()); }
    std::printf("warmup fixture 000 id=%s total_ms=%.3f\n", u.id.c_str(), first); std::fflush(stdout);
    double s_audio = 0, s_mel = 0, s_enc = 0, s_dec = 0; long n_win = 0, n_skip = 0, n_cap = 0, n_empty = 0, n_eos = 0; size_t nf_all = 0; int max_run_all = 0;
    long per_bucket[4] = {0, 0, 0, 0};  // windows in the 10 / 20 / 30 s graphs (index secs / 10)
    std::vector<float> rec(kRecElems);
    Clock::time_point burst0{}, burst1{}; bool burst_started = false;  // the first window's mel start, the last window's loop end
    auto rel = [&](Clock::time_point t) { return F(Ms(burst0, t), 1); };
    // one finished file: its stdout line, the burst sums and its dump record (stage B in the pipelined pass)
    auto emit = [&](long fx, const Utt& u) -> bool {
      const std::string id = id3(fx);
      for (const Win& w : u.wins) { if (!burst_started) { burst0 = w.t_mel0; burst_started = true; } burst1 = std::max(burst1, w.t_dec1); }
      std::vector<int> tok, frame, dur; int enc_valid = 0; size_t nf = 0; int caps = 0, maxrun = 0, steps = 0; bool e = false;
      std::string wj = "[";
      for (size_t k = 0; k < u.wins.size(); ++k) {
        const Win& w = u.wins[k];
        for (size_t j = 0; j < w.tok.size(); ++j) { tok.push_back(w.tok[j]); frame.push_back((int)(k * kT30) + w.frame[j]); dur.push_back(w.dur[j]); }
        enc_valid += w.enc_valid; nf += w.nonfinite; caps += w.cap_hits; maxrun = std::max(maxrun, w.max_run); steps += w.steps; e = e || w.eos;
        if (w.bucket / 10 >= 1 && w.bucket / 10 <= 3) ++per_bucket[w.bucket / 10];
        if (k) wj += ",";
        wj += "{\"w\":" + std::to_string(k) + ",\"bucket\":" + std::to_string(w.bucket) + ",\"T\":" + std::to_string(w.T) + ",\"samples\":" + std::to_string(w.samples) +
              ",\"mel_valid\":" + std::to_string(w.mel_valid) + ",\"enc_valid\":" + std::to_string(w.enc_valid) +
              ",\"n_tok\":" + std::to_string(w.tok.size()) + ",\"steps\":" + std::to_string(w.steps) + ",\"cap_hits\":" + std::to_string(w.cap_hits) + ",\"max_run\":" + std::to_string(w.max_run) +
              ",\"eos\":" + (w.eos ? "true" : "false") + ",\"nonfinite_enc_valid\":" + std::to_string(w.nonfinite) + ",\"nonfinite_enc_all\":" + std::to_string(w.nonfinite_all) +
              ",\"mel_ms\":" + F(w.mel_ms) + ",\"enc_ms\":" + F(w.enc_ms) + ",\"dec_ms\":" + F(w.dec_ms) + ",\"mel_run_ms\":" + F(w.mel_run_ms) + ",\"enc_run_ms\":" + F(w.enc_run_ms) + ",\"dec_run_ms\":" + F(w.dec_run_ms) +
              ",\"t_mel0\":" + rel(w.t_mel0) + ",\"t_enc1\":" + rel(w.t_enc1) + ",\"t_dec0\":" + rel(w.t_dec0) + ",\"t_dec1\":" + rel(w.t_dec1) + "}";
      }
      wj += "]";
      const double sec = u.samples / 16000.0, tot = u.mel + u.encm + u.decm;
      s_audio += sec; s_mel += u.mel; s_enc += u.encm; s_dec += u.decm; n_win += (long)u.wins.size(); n_skip += u.skipped; n_cap += caps; n_empty += tok.empty(); n_eos += e; nf_all += nf; max_run_all = std::max(max_run_all, maxrun);
      std::printf("{\"kind\":\"utt\",\"fx\":\"%s\",\"id\":%s,\"fmt\":\"%s\",\"samples\":%ld,\"audio_s\":%.4f,\"windows\":%zu,\"skipped_windows\":%d,\"enc_valid\":%d,\"n_tok\":%zu,\"steps\":%d,\"cap_hits\":%d,\"max_run\":%d,\"eos\":%s,\"nonfinite_enc_valid\":%zu,"
                  "\"mel_ms\":%.3f,\"enc_ms\":%.3f,\"dec_ms\":%.3f,\"total_ms\":%.3f,\"tok\":%s,\"frame\":%s,\"dur\":%s,\"win\":%s}\n",
                  id.c_str(), JsonStr(u.id).c_str(), u.fmt.c_str(), u.samples, sec, u.wins.size(), u.skipped, enc_valid, tok.size(), steps, caps, maxrun, e ? "true" : "false", nf,
                  u.mel, u.encm, u.decm, tot, JsonArr(tok).c_str(), JsonArr(frame).c_str(), JsonArr(dur).c_str(), wj.c_str());
      std::fflush(stdout);
      if (dump == "full") {
        std::fill(rec.begin(), rec.end(), -1.f);
        rec[0] = (float)u.wins.size(); rec[1] = (float)tok.size(); rec[2] = tok.size() > (size_t)kMaxRecTokens ? 1.f : 0.f; rec[3] = (float)enc_valid;
        for (size_t j = 0; j < tok.size() && j < (size_t)kMaxRecTokens; ++j) { rec[4 + 3 * j] = (float)tok[j]; rec[5 + 3 * j] = (float)frame[j]; rec[6 + 3 * j] = (float)dur[j]; }
        if (!WriteFloatFile(mdir + "/" + id + ".f32", rec.data(), rec.size())) { std::printf("WRITE_ERROR %s/%s.f32\n", mdir.c_str(), id.c_str()); return false; }
      }
      return true;
    };
    if (!pipeline) {  // serial: per window mel -> encoder -> loop on this thread
      for (long fx = 0; fx < count; ++fx) {
        if (!run_utt(fx, u)) return 3;
        if (!emit(fx, u)) return 3;
      }
    } else {
      // stage A (a second thread) -> an in-order queue -> stage B (this thread). An item is an encoded window (with its
      // own copy of the encoder output), the end of a file (its id, format, sample count, skipped windows), a stop
      // (stage A could not read a file; its FIXTURE_ERROR line is printed) or the end of the pass. Stage A waits while
      // --pipeline-depth encoded windows are queued; the other items do not count.
      enum Kind { kWindow, kFileEnd, kStop, kDone };
      struct Item { Kind kind = kWindow; long fx = 0; size_t T = 0; Win w; std::vector<float> encv; Utt u; };
      std::mutex mu; std::condition_variable cv; std::deque<Item> q; int queued_windows = 0;
      auto push = [&](Item&& it) {
        std::unique_lock<std::mutex> lk(mu);
        if (it.kind == kWindow) { cv.wait(lk, [&] { return queued_windows < pipeline_depth; }); ++queued_windows; }
        q.push_back(std::move(it)); cv.notify_all();
      };
      auto pop = [&]() -> Item {
        std::unique_lock<std::mutex> lk(mu); cv.wait(lk, [&] { return !q.empty(); });
        Item it = std::move(q.front()); q.pop_front(); if (it.kind == kWindow) --queued_windows;
        cv.notify_all(); return it;
      };
      std::thread stage_a([&]() {
        for (long fx = 0; fx < count; ++fx) {
          Item end; end.kind = kFileEnd; end.fx = fx; std::vector<float> x;
          if (!load_utt(fx, end.u, x)) { Item stop; stop.kind = kStop; push(std::move(stop)); return; }
          for (long s0 = 0; s0 < (long)x.size(); s0 += win) {
            long n = std::min(win, (long)x.size() - s0);
            if (n < min_win) { ++end.u.skipped; continue; }
            Bucket& b = bk[pick_bucket(n)];
            Item it; it.kind = kWindow; it.fx = fx; it.T = b.T; it.encv.resize(b.encv.size());
            mel_enc(b, x.data() + s0, n, it.w, it.encv.data());
            push(std::move(it));
          }
          push(std::move(end));
        }
        Item done; done.kind = kDone; push(std::move(done));
      });
      bool ok = true; Utt cur;
      for (bool more = true; more;) {
        Item it = pop();
        switch (it.kind) {
          case kWindow:
            dec_win(it.encv.data(), it.T, it.w);
            cur.mel += it.w.mel_ms; cur.encm += it.w.enc_ms; cur.decm += it.w.dec_ms; cur.wins.push_back(std::move(it.w));
            break;
          case kFileEnd:
            it.u.wins = std::move(cur.wins); it.u.mel = cur.mel; it.u.encm = cur.encm; it.u.decm = cur.decm; cur = Utt();
            if (!emit(it.fx, it.u)) { std::fflush(stdout); std::_Exit(3); }  // stage A may be waiting on the queue
            break;
          case kStop: ok = false; more = false; break;
          case kDone: more = false; break;
        }
      }
      stage_a.join();
      if (!ok) return 3;
    }
    const double s_tot = s_mel + s_enc + s_dec;
    const double burst_wall = burst_started ? Ms(burst0, burst1) : 0.0;
    std::printf("{\"kind\":\"burst_summary\",\"utterances\":%ld,\"windows\":%ld,\"skipped_windows\":%ld,\"audio_s\":%.4f,\"mel_ms\":%.3f,\"enc_ms\":%.3f,\"dec_ms\":%.3f,\"total_ms\":%.3f,\"rtf_burst\":%.6f,"
                "\"rtf_mel\":%.6f,\"rtf_enc\":%.6f,\"rtf_dec\":%.6f,\"warmup_fx\":\"000\",\"warmup_ms\":%.3f,\"empty_hyp\":%ld,\"cap_hits\":%ld,\"max_run\":%d,\"eos\":%ld,\"nonfinite_enc_valid\":%zu,\"threads\":%d,\"mel_threads\":%d,\"dec_threads\":%d,\"accel\":\"%s\",\"gpu_precision\":\"%s\","
                "\"buckets\":%zu,\"windows_per_bucket\":{\"10\":%ld,\"20\":%ld,\"30\":%ld},\"pipeline\":%d,\"pipeline_depth\":%d,\"burst_wall_ms\":%.3f,\"rtf_wall\":%.6f}\n",
                count, n_win, n_skip, s_audio, s_mel, s_enc, s_dec, s_tot, s_tot / 1000.0 / s_audio, s_mel / 1000.0 / s_audio, s_enc / 1000.0 / s_audio, s_dec / 1000.0 / s_audio,
                first, n_empty, n_cap, max_run_all, n_eos, nf_all, threads, mel_threads, dec_threads, accel.c_str(), accel.rfind("gpu", 0) == 0 ? gpu_precision.c_str() : "-",
                bk.size(), per_bucket[1], per_bucket[2], per_bucket[3], pipeline, pipeline_depth, burst_wall, burst_wall / 1000.0 / s_audio);
    std::printf("parity_summary model[0] %s fixtures=%ld nonfinite_values=%zu first_run_ms=%.1f mean_ms=%.3f rtf_burst=%.6f outdir=%s\n", bk[0].enc->path.c_str(), count, nf_all, first, s_tot / count, s_tot / 1000.0 / s_audio, mdir.c_str());
    std::fflush(stdout);
    // --- latency phase on fixture --ab-fixture: the 30 s pair on its first window (model[0] / [1] / [2]),
    // then each bucket encoder / mel graph on its first 10 s / 20 s (model[3..6]) ---
    if (rounds > 0) {
      long ab = std::strtol(ab_fixture.c_str(), nullptr, 10); std::vector<float> x; std::string fmt;
      if (ab >= (long)fx_id.size() || fx_id[ab].empty() || !ReadAudio(fixtures + "/" + fx_file[ab], x, fmt) || (long)x.size() < min_win) { std::printf("FIXTURE_ERROR ab fixture %s\n", ab_fixture.c_str()); return 3; }
      for (size_t bi = 0; bi < bk.size(); ++bi) {
        Bucket& b = bk[bi];
        const long n = std::min(bi == 0 ? win : (long)b.A, (long)x.size());
        std::printf("latency_start wall_ms=%lld ab_fixture=%s id=%s window_samples=%ld bucket=%d\n", WallMs(), ab_fixture.c_str(), fx_id[ab].c_str(), n, b.secs);
        do_mel(b, x.data(), n); int ev = 0; for (float v : b.framemask) ev += v > 0.5f;
        std::vector<double> ve, vm, vl; size_t nfr = 0; Win w;
        for (long i = 0; i < warmup; ++i) { double t = do_enc(b, b.encv.data()); std::printf("warmup %ld model[%d] %.3f ms run_ms=%.3f\n", i, b.enc_k, t, enc_run); }
        for (long r = 0; r < rounds; ++r) { double t = do_enc(b, b.encv.data()); ve.push_back(t); for (float v : b.encv) if (!std::isfinite(v)) { ++nfr; break; } std::printf("round %ld model[%d] write+run+readback_ms=%.3f run_ms=%.3f\n", r, b.enc_k, t, enc_run); std::fflush(stdout); }
        for (long i = 0; i < warmup; ++i) do_mel(b, x.data(), n);
        for (long r = 0; r < rounds; ++r) { double t = do_mel(b, x.data(), n); vm.push_back(t); std::printf("round %ld model[%d] write+run+readback_ms=%.3f run_ms=%.3f\n", r, b.mel_k, t, mel_run); }
        std::printf("summary model[%d] %s n=%zu median=%.3f min=%.3f max=%.3f nonfinite_rounds=%zu compile_ms=%.1f fully_accelerated=%s\n", b.enc_k, b.enc->path.c_str(), ve.size(), Median(ve),
                    *std::min_element(ve.begin(), ve.end()), *std::max_element(ve.begin(), ve.end()), nfr, b.enc->compile_ms, b.enc->fully ? "true" : "false");
        std::printf("summary model[%d] %s n=%zu median=%.3f min=%.3f max=%.3f nonfinite_rounds=0 compile_ms=%.1f fully_accelerated=%s\n", b.mel_k, b.mel->path.c_str(), vm.size(), Median(vm),
                    *std::min_element(vm.begin(), vm.end()), *std::max_element(vm.begin(), vm.end()), b.mel->compile_ms, b.mel->fully ? "true" : "false");
        if (bi == 0) {  // the whole loop on the 30 s window (the model[2] line)
          do_enc(b, b.encv.data());
          for (long i = 0; i < warmup; ++i) do_loop(b.encv.data(), b.T, ev, w);
          for (long r = 0; r < rounds; ++r) { double t = do_loop(b.encv.data(), b.T, ev, w); vl.push_back(t); std::printf("round %ld model[2] write+run+readback_ms=%.3f run_ms=%.3f steps=%d tokens=%zu\n", r, t, w.dec_run_ms, w.steps, w.tok.size()); }
          std::printf("summary model[2] %s n=%zu median=%.3f min=%.3f max=%.3f nonfinite_rounds=0 compile_ms=%.1f fully_accelerated=%s loop=whole_window enc_valid=%d\n", dec.path.c_str(), vl.size(), Median(vl),
                      *std::min_element(vl.begin(), vl.end()), *std::max_element(vl.begin(), vl.end()), dec.compile_ms, dec.fully ? "true" : "false", ev);
        }
        std::fflush(stdout);
      }
    }
  }
  for (auto& m : ms) { for (auto& s : m.sigs) { for (auto b : s.outs) LiteRtDestroyTensorBuffer(b); for (auto b : s.ins) LiteRtDestroyTensorBuffer(b); } LiteRtDestroyCompiledModel(m.cm); LiteRtDestroyOptions(m.opts); LiteRtDestroyModel(m.model); }
  LiteRtDestroyEnvironment(env);
  std::printf("done wall_ms=%lld\n", WallMs());
  std::fflush(stdout);
  return 0;
}
