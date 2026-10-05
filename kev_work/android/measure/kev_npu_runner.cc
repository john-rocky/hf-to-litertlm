// kev_npu_runner: a small C++ runner on the LiteRT 2.2.0 C API. It produced the NPU numbers of REPRODUCE.md on the
// Galaxy S26, and the GPU number measured beside them. It loads one or more graphs with the same accelerator and
// options, runs a parity pass over K fixtures per graph (one warm-up call on fixture 000, then every fixture once),
// then paired latency rounds on one fixture, the graphs alternating in every round.
//   --accel gpu|npu|cpu|npu+cpu  npu+cpu = kLiteRtHwAcceleratorNpu | kLiteRtHwAcceleratorCpu: a graph whose NPU
//                                partition leaves an op on the CPU (the int8 EMBEDDING_LOOKUP) fails
//                                LiteRtCreateCompiledModel with 504 when the set is NPU only
//   --gpu-precision default|fp16|fp32|fp16acc32    --burst 0|1 (NPU: the Qualcomm htp_performance_mode burst)
//   --libdir DIR     the runtime library, dispatch library and compiler plugin directory of the environment
//   --cache-dir DIR  the JIT compiler cache directory of the environment
//   --fixtures DIR --count K [--inputs name0,name1,...]  DIR/<id3>_<name>.f32 = the bytes of input <name> of fixture
//                    <id3> (000, 001, ...), copied byte for byte (the ids file holds int32 bytes); the names default to
//                    the signature's input names, in signature order
//   --sel-dir DIR    DIR/<id3>_sel.i32 (int32 positions [decide, *opts]) + DIR/<id3>_n.i32 (real-token count): the
//                    selected rows of the first output are appended to <out>/model<k>/hsel.f32 (little-endian float32,
//                    --hidden-dim floats per row) and the non-finite counts at the real / selected positions are printed
//   --dump full|none|all  full (default) = each fixture's first output to <out>/model<k>/<id3>.f32, all = every output
//                    to <out>/model<k>/<id3>_o<j>.f32
//   --warmup W --rounds R --ab-fixture ID  latency: W warm-up calls, then R timed calls per graph on fixture ID
//   --out DIR --model A.tflite [--model B.tflite ...] [--hidden-dim D (default 1024)]
// A call's ms = write the inputs + run + read back the first output; the run-only ms is printed beside it. Wall-clock
// stamps (CLOCK_REALTIME, epoch ms) are printed before and after each compile and at the end. Every graph in the process
// gets the same accelerator and options (the environment fixes the dispatch options at the first load, so runs on
// different accelerators belong in separate processes).
#include <unistd.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
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
#define LRT_CHECK(expr) do { LiteRtStatus _st = (expr); if (_st != kLiteRtStatusOk) { std::fprintf(stderr, "ERROR: %s failed: %d (%s)\n", #expr, (int)_st, LiteRtGetStatusString(_st)); std::printf("FATAL %s status=%d\n", #expr, (int)_st); std::exit(2); } } while (0)
void DeleteCStringPayload(void* p) { delete[] static_cast<char*>(p); }
bool ReadFloatFile(const std::string& path, std::vector<float>& v) {
  FILE* f = std::fopen(path.c_str(), "rb"); if (!f) return false;
  std::fseek(f, 0, SEEK_END); long sz = std::ftell(f); std::fseek(f, 0, SEEK_SET);
  v.assign(sz / 4, 0.f); bool ok = v.empty() || std::fread(v.data(), 4, v.size(), f) == v.size(); std::fclose(f); return ok;
}
bool WriteFloatFile(const std::string& path, const std::vector<float>& v) {
  FILE* f = std::fopen(path.c_str(), "wb"); if (!f) return false;
  bool ok = std::fwrite(v.data(), 4, v.size(), f) == v.size(); std::fclose(f); return ok;
}
double Median(std::vector<double> v) { std::sort(v.begin(), v.end()); size_t n = v.size(); return n ? ((n % 2) ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2])) : 0.0; }
double Ms(Clock::time_point a, Clock::time_point b) { return std::chrono::duration<double, std::milli>(b - a).count(); }
long long WallMs() { struct timespec ts; clock_gettime(CLOCK_REALTIME, &ts); return (long long)ts.tv_sec * 1000 + ts.tv_nsec / 1000000; }
bool ReadIntFile(const std::string& path, std::vector<int32_t>& v) {
  FILE* f = std::fopen(path.c_str(), "rb"); if (!f) return false;
  std::fseek(f, 0, SEEK_END); long sz = std::ftell(f); std::fseek(f, 0, SEEK_SET);
  v.assign(sz / 4, 0); bool ok = v.empty() || std::fread(v.data(), 4, v.size(), f) == v.size(); std::fclose(f); return ok;
}
void AddToml(LiteRtOptions options, const char* identifier, const std::string& toml) {
  char* payload = new char[toml.size() + 1]; std::memcpy(payload, toml.data(), toml.size()); payload[toml.size()] = '\0';
  LiteRtOpaqueOptions opaque = nullptr; LRT_CHECK(LiteRtCreateOpaqueOptions(identifier, payload, DeleteCStringPayload, &opaque)); LRT_CHECK(LiteRtAddOpaqueOptions(options, opaque));
}
struct M {
  std::string path; LiteRtModel model = nullptr; LiteRtOptions opts = nullptr; LiteRtCompiledModel cm = nullptr;
  std::vector<LiteRtTensorBuffer> ins, outs; std::vector<size_t> in_elems, out_elems; std::vector<std::string> in_names;
  std::vector<std::vector<float>> in_data; std::vector<double> ms; double compile_ms = 0; bool fully = false; size_t nonfinite_rounds = 0;
};
std::vector<std::string> Split(const std::string& s) { std::vector<std::string> r; size_t p = 0; while (p <= s.size()) { size_t q = s.find(',', p); if (q == std::string::npos) q = s.size(); r.push_back(s.substr(p, q - p)); p = q + 1; } return r; }
}  // namespace
int main(int argc, char** argv) {
  std::string accel = "gpu", gpu_precision = "fp32", libdir, cache_dir, fixtures, out_dir, inputs_arg; int burst = 1; long warmup = 2, rounds = 8, count = 0; std::string ab_fixture = "000";
  std::string sel_dir, dump = "full"; long hidden_dim = 1024;
  std::vector<M> ms_;
  for (int i = 1; i < argc; ++i) {
    std::string f = argv[i];
    auto next = [&](const char* what) -> const char* { if (i + 1 >= argc) { std::fprintf(stderr, "missing value for %s\n", what); std::exit(1); } return argv[++i]; };
    if (f == "--accel") accel = next("--accel");
    else if (f == "--gpu-precision") gpu_precision = next("--gpu-precision");
    else if (f == "--burst") burst = std::atoi(next("--burst"));
    else if (f == "--libdir") libdir = next("--libdir");
    else if (f == "--cache-dir") cache_dir = next("--cache-dir");
    else if (f == "--fixtures") fixtures = next("--fixtures");
    else if (f == "--count") count = std::strtol(next("--count"), nullptr, 10);
    else if (f == "--inputs") inputs_arg = next("--inputs");
    else if (f == "--warmup") warmup = std::strtol(next("--warmup"), nullptr, 10);
    else if (f == "--rounds") rounds = std::strtol(next("--rounds"), nullptr, 10);
    else if (f == "--ab-fixture") ab_fixture = next("--ab-fixture");
    else if (f == "--out") out_dir = next("--out");
    else if (f == "--model") { M m; m.path = next("--model"); ms_.push_back(m); }
    else if (f == "--sel-dir") sel_dir = next("--sel-dir");
    else if (f == "--dump") dump = next("--dump");
    else if (f == "--hidden-dim") hidden_dim = std::strtol(next("--hidden-dim"), nullptr, 10);
    else { std::fprintf(stderr, "unknown flag %s\n", f.c_str()); return 1; }
  }
  if (ms_.empty() || libdir.empty() || fixtures.empty() || out_dir.empty() || count < 1) { std::fprintf(stderr, "need --model, --libdir, --fixtures, --count, --out\n"); return 1; }
  ::mkdir(out_dir.c_str(), 0775);
  std::printf("pid=%d accel=%s gpu_precision=%s burst=%d models=%zu fixtures=%ld warmup=%ld rounds=%ld ab_fixture=%s libdir=%s cache_dir=%s sel_dir=%s dump=%s hidden_dim=%ld wall_ms=%lld\n",
              (int)getpid(), accel.c_str(), gpu_precision.c_str(), burst, ms_.size(), count, warmup, rounds, ab_fixture.c_str(), libdir.c_str(), cache_dir.c_str(), sel_dir.c_str(), dump.c_str(), hidden_dim, WallMs());
  std::fflush(stdout);
  // --- environment: runtime lib dir (GPU accelerator .so), dispatch + compiler plugin dirs (NPU), JIT cache dir ---
  std::vector<LiteRtEnvOption> env_opts;
  auto add_env = [&](LiteRtEnvOptionTag tag, const std::string& v) { LiteRtEnvOption o{}; o.tag = tag; o.value.type = kLiteRtAnyTypeString; o.value.str_value = v.c_str(); env_opts.push_back(o); };
  add_env(kLiteRtEnvOptionTagRuntimeLibraryDir, libdir);
  add_env(kLiteRtEnvOptionTagDispatchLibraryDir, libdir);
  add_env(kLiteRtEnvOptionTagCompilerPluginLibraryDir, libdir);
  if (!cache_dir.empty()) { ::mkdir(cache_dir.c_str(), 0775); add_env(kLiteRtEnvOptionTagCompilerCacheDir, cache_dir); }
  LiteRtEnvironment env = nullptr;
  LRT_CHECK(LiteRtCreateEnvironment(env_opts.size(), env_opts.data(), &env));
  LiteRtHwAcceleratorSet hw = accel == "npu" ? kLiteRtHwAcceleratorNpu : accel == "npu+cpu" ? (LiteRtHwAcceleratorSet)(kLiteRtHwAcceleratorNpu | kLiteRtHwAcceleratorCpu) : accel == "cpu" ? kLiteRtHwAcceleratorCpu : kLiteRtHwAcceleratorGpu;
  const bool npu = (hw & kLiteRtHwAcceleratorNpu) != 0;
  int prec = gpu_precision == "default" ? (int)kLiteRtDelegatePrecisionDefault : gpu_precision == "fp16" ? (int)kLiteRtDelegatePrecisionFp16 : gpu_precision == "fp16acc32" ? (int)kLiteRtDelegatePrecisionFp16WithFp32Accum : (int)kLiteRtDelegatePrecisionFp32;
  std::vector<std::string> logical_names = inputs_arg.empty() ? std::vector<std::string>() : Split(inputs_arg);
  for (size_t k = 0; k < ms_.size(); ++k) {
    M& m = ms_[k];
    LRT_CHECK(LiteRtCreateModelFromFile(env, m.path.c_str(), &m.model));
    LiteRtSignature sig = nullptr; LRT_CHECK(LiteRtGetModelSignature(m.model, 0, &sig));
    LiteRtParamIndex n_in = 0, n_out = 0; LRT_CHECK(LiteRtGetNumSignatureInputs(sig, &n_in)); LRT_CHECK(LiteRtGetNumSignatureOutputs(sig, &n_out));
    std::vector<LiteRtRankedTensorType> in_types(n_in), out_types(n_out); m.in_elems.assign(n_in, 0); m.out_elems.assign(n_out, 0);
    for (LiteRtParamIndex i = 0; i < n_in; ++i) {
      LiteRtTensor t = nullptr; LRT_CHECK(LiteRtGetSignatureInputTensorByIndex(sig, i, &t)); LRT_CHECK(LiteRtGetRankedTensorType(t, &in_types[i])); LRT_CHECK(LiteRtGetNumLayoutElements(&in_types[i].layout, &m.in_elems[i]));
      const char* nm = nullptr; LRT_CHECK(LiteRtGetSignatureInputName(sig, i, &nm));
      m.in_names.push_back(!logical_names.empty() && i < logical_names.size() ? logical_names[i] : std::string(nm));
      std::printf("model[%zu] input[%zu] sig_name=%s logical=%s elems=%zu rank=%u\n", k, (size_t)i, nm, m.in_names.back().c_str(), m.in_elems[i], in_types[i].layout.rank);
    }
    for (LiteRtParamIndex o = 0; o < n_out; ++o) { LiteRtTensor to = nullptr; LRT_CHECK(LiteRtGetSignatureOutputTensorByIndex(sig, o, &to)); LRT_CHECK(LiteRtGetRankedTensorType(to, &out_types[o])); LRT_CHECK(LiteRtGetNumLayoutElements(&out_types[o].layout, &m.out_elems[o])); }
    LRT_CHECK(LiteRtCreateOptions(&m.opts));
    LRT_CHECK(LiteRtSetOptionsHardwareAccelerators(m.opts, hw));
    AddToml(m.opts, "runtime_options_string", "enable_profiling = false\nerror_reporter_mode = 1\n");
    if (hw == kLiteRtHwAcceleratorGpu) AddToml(m.opts, "gpu_options", "precision = " + std::to_string(prec) + "\n");
    if (npu && burst) AddToml(m.opts, "qualcomm", "htp_performance_mode = 2\n");  // kLiteRtQualcommHtpPerformanceModeBurst
    std::printf("compile_start model[%zu] wall_ms=%lld\n", k, WallMs()); std::fflush(stdout);
    auto c0 = Clock::now(); LRT_CHECK(LiteRtCreateCompiledModel(env, m.model, m.opts, &m.cm)); m.compile_ms = Ms(c0, Clock::now());
    std::printf("compile_end model[%zu] wall_ms=%lld\n", k, WallMs()); std::fflush(stdout);
    LRT_CHECK(LiteRtCompiledModelIsFullyAccelerated(m.cm, &m.fully));
    m.ins.assign(n_in, nullptr); m.outs.assign(n_out, nullptr);
    for (LiteRtParamIndex i = 0; i < n_in; ++i) { LiteRtTensorBufferRequirements req = nullptr; LRT_CHECK(LiteRtGetCompiledModelInputBufferRequirements(m.cm, 0, i, &req)); LRT_CHECK(LiteRtCreateManagedTensorBufferFromRequirements(env, &in_types[i], req, &m.ins[i])); }
    for (LiteRtParamIndex o = 0; o < n_out; ++o) { LiteRtTensorBufferRequirements ro = nullptr; LRT_CHECK(LiteRtGetCompiledModelOutputBufferRequirements(m.cm, 0, o, &ro)); LRT_CHECK(LiteRtCreateManagedTensorBufferFromRequirements(env, &out_types[o], ro, &m.outs[o])); }
    std::printf("model[%zu]=%s inputs=%zu outputs=%zu out0_elems=%zu compile_ms=%.1f fully_accelerated=%s\n", k, m.path.c_str(), (size_t)n_in, (size_t)n_out, m.out_elems[0], m.compile_ms, m.fully ? "true" : "false");
    std::fflush(stdout);
  }
  auto load_fixture = [&](M& m, const std::string& id) -> bool {
    m.in_data.assign(m.ins.size(), {});
    for (size_t i = 0; i < m.ins.size(); ++i) {
      std::string p = fixtures + "/" + id + "_" + m.in_names[i] + ".f32";
      if (!ReadFloatFile(p, m.in_data[i]) || m.in_data[i].size() != m.in_elems[i]) { std::printf("FIXTURE_ERROR %s (%zu floats, want %zu)\n", p.c_str(), m.in_data[i].size(), m.in_elems[i]); return false; }
    }
    return true;
  };
  double last_run_ms = 0;  // run-only ms of the latest run_one (LiteRtRunCompiledModel alone)
  auto dump_all = [&](M& m, const std::string& mdir, const std::string& id) -> bool {
    for (size_t o = 0; o < m.outs.size(); ++o) {
      std::vector<float> tmp(m.out_elems[o]); void* oh = nullptr;
      LRT_CHECK(LiteRtLockTensorBuffer(m.outs[o], &oh, kLiteRtTensorBufferLockModeRead)); std::memcpy(tmp.data(), oh, m.out_elems[o] * 4); LRT_CHECK(LiteRtUnlockTensorBuffer(m.outs[o]));
      if (!WriteFloatFile(mdir + "/" + id + "_o" + std::to_string(o) + ".f32", tmp)) return false;
    }
    return true;
  };
  auto run_one = [&](M& m, std::vector<float>* out0) -> double {
    auto t0 = Clock::now();  // one call: write + run + read back
    for (size_t i = 0; i < m.ins.size(); ++i) { void* host = nullptr; LRT_CHECK(LiteRtLockTensorBuffer(m.ins[i], &host, kLiteRtTensorBufferLockModeWrite)); std::memcpy(host, m.in_data[i].data(), m.in_elems[i] * 4); LRT_CHECK(LiteRtUnlockTensorBuffer(m.ins[i])); }
    auto tr = Clock::now();
    LRT_CHECK(LiteRtRunCompiledModel(m.cm, 0, m.ins.size(), m.ins.data(), m.outs.size(), m.outs.data()));
    last_run_ms = Ms(tr, Clock::now());
    std::vector<float> tmp(m.out_elems[0]);
    void* oh = nullptr; LRT_CHECK(LiteRtLockTensorBuffer(m.outs[0], &oh, kLiteRtTensorBufferLockModeRead)); std::memcpy(tmp.data(), oh, m.out_elems[0] * 4); LRT_CHECK(LiteRtUnlockTensorBuffer(m.outs[0]));
    double ms = Ms(t0, Clock::now());
    if (out0) *out0 = tmp;
    return ms;
  };
  // --- phase 1: parity pass, every fixture once per model (after one warm-up run on fixture 0) ---
  char idbuf[16];
  for (size_t k = 0; k < ms_.size(); ++k) {
    M& m = ms_[k]; std::string mdir = out_dir + "/model" + std::to_string(k); ::mkdir(mdir.c_str(), 0775);
    size_t n_nonfinite = 0; double first_ms = -1, sum = 0;
    FILE* hsel = nullptr;  // --sel-dir output
    if (!sel_dir.empty()) { hsel = std::fopen((mdir + "/hsel.f32").c_str(), "wb"); if (!hsel) { std::printf("WRITE_ERROR %s/hsel.f32\n", mdir.c_str()); return 3; } }
    for (long fx = 0; fx < count; ++fx) {
      std::snprintf(idbuf, sizeof idbuf, "%03ld", fx); std::string id = idbuf;
      if (!load_fixture(m, id)) return 3;
      if (fx == 0) { std::vector<float> w; first_ms = run_one(m, &w); }
      std::vector<float> out; double t = run_one(m, &out); double run_ms = last_run_ms; sum += t;
      size_t nf = 0; for (float v : out) if (!std::isfinite(v)) ++nf; n_nonfinite += nf;
      if (dump == "full" && !WriteFloatFile(mdir + "/" + id + ".f32", out)) { std::printf("WRITE_ERROR %s\n", (mdir + "/" + id + ".f32").c_str()); return 3; }
      if (dump == "all" && !dump_all(m, mdir, id)) { std::printf("WRITE_ERROR %s/%s_o*.f32\n", mdir.c_str(), id.c_str()); return 3; }
      long sel_n = -1, nf_real = -1, nf_sel = -1, k_opts = -1;
      if (hsel) {
        std::vector<int32_t> pos, nvec;
        if (!ReadIntFile(sel_dir + "/" + id + "_sel.i32", pos) || pos.empty() || !ReadIntFile(sel_dir + "/" + id + "_n.i32", nvec) || nvec.size() != 1) { std::printf("FIXTURE_ERROR %s/%s_sel.i32 or _n.i32\n", sel_dir.c_str(), id.c_str()); return 3; }
        sel_n = nvec[0]; k_opts = (long)pos.size() - 1; nf_real = 0; nf_sel = 0;
        for (long i = 0; i < sel_n * hidden_dim && i < (long)out.size(); ++i) if (!std::isfinite(out[i])) ++nf_real;
        for (int32_t p : pos) {
          if (p < 0 || (size_t)(p + 1) * hidden_dim > out.size()) { std::printf("FIXTURE_ERROR position %d outside the output\n", p); return 3; }
          const float* row = out.data() + (size_t)p * hidden_dim;
          for (long i = 0; i < hidden_dim; ++i) if (!std::isfinite(row[i])) ++nf_sel;
          if (std::fwrite(row, 4, hidden_dim, hsel) != (size_t)hidden_dim) { std::printf("WRITE_ERROR hsel\n"); return 3; }
        }
      }
      std::printf("parity model[%zu] fixture %s ms=%.3f run_ms=%.3f nonfinite=%zu n=%ld k=%ld nonfinite_real=%ld nonfinite_sel=%ld out0[0..3]=%.6g,%.6g,%.6g,%.6g\n", k, id.c_str(), t, run_ms, nf, sel_n, k_opts, nf_real, nf_sel, out.size() > 0 ? out[0] : 0.f, out.size() > 1 ? out[1] : 0.f, out.size() > 2 ? out[2] : 0.f, out.size() > 3 ? out[3] : 0.f);
    }
    if (hsel) std::fclose(hsel);
    std::printf("parity_summary model[%zu] %s fixtures=%ld nonfinite_values=%zu first_run_ms=%.1f mean_ms=%.3f outdir=%s\n", k, m.path.c_str(), count, n_nonfinite, first_ms, sum / count, mdir.c_str());
    std::fflush(stdout);
  }
  // --- phase 2: paired latency on one fixture, models alternating in every round ---
  if (rounds > 0) {
    for (auto& m : ms_) if (!load_fixture(m, ab_fixture)) return 3;
    std::printf("latency_start wall_ms=%lld\n", WallMs());
    for (long w = 0; w < warmup; ++w) for (size_t k = 0; k < ms_.size(); ++k) { double t = run_one(ms_[k], nullptr); std::printf("warmup %ld model[%zu] %.3f ms run_ms=%.3f\n", w, k, t, last_run_ms); }
    for (long r = 0; r < rounds; ++r) for (size_t k = 0; k < ms_.size(); ++k) { std::vector<float> out; double t = run_one(ms_[k], &out); ms_[k].ms.push_back(t); for (float v : out) if (!std::isfinite(v)) { ++ms_[k].nonfinite_rounds; break; } std::printf("round %ld model[%zu] write+run+readback_ms=%.3f run_ms=%.3f\n", r, k, t, last_run_ms); std::fflush(stdout); }
    for (size_t k = 0; k < ms_.size(); ++k) { M& m = ms_[k]; std::printf("summary model[%zu] %s n=%zu median=%.3f min=%.3f max=%.3f nonfinite_rounds=%zu compile_ms=%.1f fully_accelerated=%s\n", k, m.path.c_str(), m.ms.size(), Median(m.ms), *std::min_element(m.ms.begin(), m.ms.end()), *std::max_element(m.ms.begin(), m.ms.end()), m.nonfinite_rounds, m.compile_ms, m.fully ? "true" : "false"); }
  }
  for (auto& m : ms_) { for (auto b : m.outs) LiteRtDestroyTensorBuffer(b); for (auto b : m.ins) LiteRtDestroyTensorBuffer(b); LiteRtDestroyCompiledModel(m.cm); LiteRtDestroyOptions(m.opts); LiteRtDestroyModel(m.model); }
  LiteRtDestroyEnvironment(env);
  std::printf("done wall_ms=%lld\n", WallMs());
  return 0;
}
