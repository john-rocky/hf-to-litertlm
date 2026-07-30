// macOS smoke test for the Bonsai Mac GPU pipeline: LiteRT CompiledModel C API
// against the prebuilt macos_arm64 pair (libLiteRt + libLiteRtMetalAccelerator,
// both from the same LiteRT main revision — mixing generations makes the
// accelerator reject the serialized TOML options).
//
// Stage A  textenc int4  CPU (opaque "xnnpack",  "num_threads = 6")  vs embeds fixture
// Stage B  DiT int4b32   GPU (opaque "gpu_options", "precision = 2") vs per-step fixtures
// Stage C  VAE fp32      CPU                                          vs z_vae fixture -> PNG
//
// The Metal accelerator is not linked: the environment's RuntimeLibraryDir
// option points at the prebuilt dir and the registry scans it at
// LiteRtCreateEnvironment time (no RegisterGpuAccelerator call on macOS).
// fp32 precision is REQUIRED for the DiT: default fp16 overflows this model's
// activation range and corrupts the output (measured cos ~ -0.02 on this Mac).

#import <Foundation/Foundation.h>
#import <CoreGraphics/CoreGraphics.h>
#import <ImageIO/ImageIO.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "litert/c/litert_common.h"
#include "litert/c/litert_any.h"
#include "litert/c/litert_environment.h"
#include "litert/c/litert_environment_options.h"
#include "litert/c/litert_model.h"
#include "litert/c/litert_model_types.h"
#include "litert/c/litert_options.h"
#include "litert/c/litert_opaque_options.h"
#include "litert/c/litert_compiled_model.h"
#include "litert/c/litert_tensor_buffer.h"
#include "litert/c/litert_tensor_buffer_requirements.h"
#include "litert/c/litert_tensor_buffer_types.h"

using clock_t_ = std::chrono::high_resolution_clock;
static double secs(clock_t_::time_point a, clock_t_::time_point b) {
  return std::chrono::duration<double>(b - a).count();
}

#define CHECK_OK(expr)                                              \
  do {                                                              \
    LiteRtStatus s_ = (expr);                                       \
    if (s_ != kLiteRtStatusOk) {                                    \
      printf("FAILED %s -> status %d (%s:%d)\n", #expr, s_,         \
             __FILE__, __LINE__);                                   \
      exit(1);                                                      \
    }                                                               \
  } while (0)

static std::vector<uint8_t> LoadBytes(const std::string& path, size_t expect) {
  NSData* d = [NSData dataWithContentsOfFile:
      [NSString stringWithUTF8String:path.c_str()]];
  if (!d || (expect && d.length != expect)) {
    printf("fixture %s missing or wrong size (%lu, want %zu)\n", path.c_str(),
           (unsigned long)(d ? d.length : 0), expect);
    exit(1);
  }
  std::vector<uint8_t> v(d.length);
  memcpy(v.data(), d.bytes, d.length);
  return v;
}

static double Cos(const float* a, const float* b, size_t n, double* maxd) {
  double dot = 0, na = 0, nb = 0, md = 0;
  for (size_t i = 0; i < n; ++i) {
    dot += (double)a[i] * b[i];
    na += (double)a[i] * a[i];
    nb += (double)b[i] * b[i];
    md = std::max(md, (double)std::fabs(a[i] - b[i]));
  }
  if (maxd) *maxd = md;
  return dot / (std::sqrt(na) * std::sqrt(nb) + 1e-30);
}

// One fixed-shape graph over CompiledModel. Inputs are bound positionally to
// the signature but written by args_<n> index parsed from the input name, so
// host order is always lat/embeds/sigma/... regardless of signature order.
struct Graph {
  LiteRtModel model = nullptr;
  LiteRtOptions options = nullptr;
  LiteRtCompiledModel compiled = nullptr;
  std::vector<LiteRtTensorBuffer> in;
  LiteRtTensorBuffer out = nullptr;
  std::vector<int> argpos;  // signature position -> host input index
  double loadSecs = 0, compileSecs = 0;

  Graph(LiteRtEnvironment env, const std::string& path, bool gpu,
        const std::vector<LiteRtElementType>& inTypes,
        LiteRtElementType outType) {
    auto t0 = clock_t_::now();
    CHECK_OK(LiteRtCreateModelFromFile(env, path.c_str(), &model));
    loadSecs = secs(t0, clock_t_::now());

    CHECK_OK(LiteRtCreateOptions(&options));
    CHECK_OK(LiteRtSetOptionsHardwareAccelerators(
        options, gpu ? (kLiteRtHwAcceleratorGpu | kLiteRtHwAcceleratorCpu)
                     : kLiteRtHwAcceleratorCpu));
    // Hand-built TOML payloads: the prebuilt libLiteRt does not export the
    // Lrt*Options helpers, but the accelerators parse the payload as TOML.
    const char* ident = gpu ? "gpu_options" : "xnnpack";
    const char* toml = gpu ? "precision = 2\n" : "num_threads = 6\n";
    LiteRtOpaqueOptions opaque = nullptr;
    CHECK_OK(LiteRtCreateOpaqueOptions(ident, strdup(toml), free, &opaque));
    CHECK_OK(LiteRtAddOpaqueOptions(options, opaque));

    t0 = clock_t_::now();
    CHECK_OK(LiteRtCreateCompiledModel(env, model, options, &compiled));
    compileSecs = secs(t0, clock_t_::now());

    bool full = false;
    LiteRtCompiledModelIsFullyAccelerated(compiled, &full);
    printf("  %s: load %.2fs compile %.1fs fully_accelerated=%d\n",
           path.substr(path.rfind('/') + 1).c_str(), loadSecs, compileSecs,
           full);

    // args_<n> mapping from signature input names.
    LiteRtSignature sig = nullptr;
    CHECK_OK(LiteRtGetModelSignature(model, 0, &sig));
    LiteRtParamIndex nIn = 0;
    CHECK_OK(LiteRtGetNumSignatureInputs(sig, &nIn));
    if (nIn != inTypes.size()) {
      printf("FAILED: signature has %d inputs, host expects %zu\n", (int)nIn,
             inTypes.size());
      exit(1);
    }
    for (LiteRtParamIndex i = 0; i < nIn; ++i) {
      const char* name = nullptr;
      CHECK_OK(LiteRtGetSignatureInputName(sig, i, &name));
      const char* p = strstr(name, "args_");
      argpos.push_back(p ? atoi(p + 5) : (int)i);
    }

    in.resize(nIn, nullptr);
    for (LiteRtParamIndex i = 0; i < nIn; ++i) {
      LiteRtLayout layout;
      CHECK_OK(LiteRtGetCompiledModelInputTensorLayout(compiled, 0, i, &layout));
      LiteRtRankedTensorType type;
      type.element_type = inTypes[argpos[i]];
      type.layout = layout;
      LiteRtTensorBufferRequirements req = nullptr;
      CHECK_OK(LiteRtGetCompiledModelInputBufferRequirements(compiled, 0, i, &req));
      CHECK_OK(LiteRtCreateManagedTensorBufferFromRequirements(env, &type, req,
                                                               &in[i]));
    }
    LiteRtLayout layout;
    CHECK_OK(LiteRtGetCompiledModelOutputTensorLayouts(compiled, 0, 1, &layout,
                                                       false));
    LiteRtRankedTensorType type;
    type.element_type = outType;
    type.layout = layout;
    LiteRtTensorBufferRequirements req = nullptr;
    CHECK_OK(LiteRtGetCompiledModelOutputBufferRequirements(compiled, 0, 0, &req));
    CHECK_OK(LiteRtCreateManagedTensorBufferFromRequirements(env, &type, req,
                                                             &out));
  }

  // hostInputs[k] feeds args_k. Returns wall seconds for the run call.
  double Run(const std::vector<std::pair<const void*, size_t>>& hostInputs,
             void* outData, size_t outBytes) {
    for (size_t i = 0; i < in.size(); ++i) {
      const auto& src = hostInputs[argpos[i]];
      void* host = nullptr;
      CHECK_OK(LiteRtLockTensorBuffer(in[i], &host,
                                      kLiteRtTensorBufferLockModeWrite));
      memcpy(host, src.first, src.second);
      CHECK_OK(LiteRtUnlockTensorBuffer(in[i]));
    }
    auto t0 = clock_t_::now();
    CHECK_OK(LiteRtRunCompiledModel(compiled, 0, in.size(), in.data(), 1, &out));
    double dt = secs(t0, clock_t_::now());
    void* host = nullptr;
    CHECK_OK(LiteRtLockTensorBuffer(out, &host, kLiteRtTensorBufferLockModeRead));
    memcpy(outData, host, outBytes);
    CHECK_OK(LiteRtUnlockTensorBuffer(out));
    return dt;
  }

  ~Graph() {
    for (auto b : in)
      if (b) LiteRtDestroyTensorBuffer(b);
    if (out) LiteRtDestroyTensorBuffer(out);
    if (compiled) LiteRtDestroyCompiledModel(compiled);
    if (options) LiteRtDestroyOptions(options);
    if (model) LiteRtDestroyModel(model);
  }
};

static const int kTokens = 1024, kSeq = 256, kLatDim = 128, kEmbDim = 7680;
static const float kSigmas[5] = {1.0f, 0.9580853581f, 0.8839818835f,
                                 0.7174965739f, 0.0f};

int main(int argc, char** argv) {
  std::string stage = argc > 1 ? argv[1] : "all";
  const char* home = getenv("HOME");
  std::string models = std::string(home) + "/models/bonsai-image-4b-tflite";
  std::string fix = models + "/device_fixtures";
  const char* rtdir = getenv("BONSAI_RUNTIME_DIR");
  std::string prebuilt =
      rtdir ? rtdir : std::string(home) + "/models/litert-prebuilt/macos_arm64";
  printf("runtime dir: %s\n", prebuilt.c_str());

  std::vector<LiteRtEnvOption> opts;
  LiteRtEnvOption rt;
  rt.tag = kLiteRtEnvOptionTagRuntimeLibraryDir;
  rt.value.type = kLiteRtAnyTypeString;
  rt.value.str_value = prebuilt.c_str();
  opts.push_back(rt);
  const char* cache = getenv("BONSAI_CACHE_DIR");
  LiteRtEnvOption co;
  if (cache) {
    co.tag = kLiteRtEnvOptionTagCompilerCacheDir;
    co.value.type = kLiteRtAnyTypeString;
    co.value.str_value = cache;
    opts.push_back(co);
    printf("compiler cache dir: %s\n", cache);
  }
  LiteRtEnvironment env = nullptr;
  CHECK_OK(LiteRtCreateEnvironment((int)opts.size(), opts.data(), &env));
  bool hasGpu = false;
  LiteRtEnvironmentHasGpuEnvironment(env, &hasGpu);
  printf("env created, has_gpu_environment=%d\n", hasGpu);

  std::vector<float> embeds((size_t)kSeq * kEmbDim);

  // ---- Stage A: text encoder on CPU ---------------------------------------
  if (stage == "all" || stage == "text") {
    printf("== textenc int4 CPU ==\n");
    auto ids = LoadBytes(fix + "/ids_i32.bin", kSeq * 4);
    auto mask = LoadBytes(fix + "/mask_i32.bin", kSeq * 4);
    auto ref = LoadBytes(fix + "/embeds_f32.bin", embeds.size() * 4);
    Graph te(env, models + "/hf_upload/textenc_int4.tflite", false,
             {kLiteRtElementTypeInt32, kLiteRtElementTypeInt32},
             kLiteRtElementTypeFloat32);
    double dt = te.Run({{ids.data(), ids.size()}, {mask.data(), mask.size()}},
                       embeds.data(), embeds.size() * 4);
    double maxd = 0;
    double c = Cos(embeds.data(), (const float*)ref.data(), embeds.size(), &maxd);
    printf("  run %.2fs  vs CPU fixture: cos=%.6f max|d|=%.3e\n", dt, c, maxd);
  } else {
    auto ref = LoadBytes(fix + "/embeds_f32.bin", embeds.size() * 4);
    memcpy(embeds.data(), ref.data(), ref.size());
  }

  // ---- Stage B: DiT on Metal, fp32 ----------------------------------------
  std::vector<float> lat((size_t)kTokens * kLatDim);
  if (stage == "all" || stage == "dit") {
    printf("== DiT int4b32 GPU fp32 ==\n");
    {
      auto l0 = LoadBytes(fix + "/lat0_f32.bin", lat.size() * 4);
      memcpy(lat.data(), l0.data(), l0.size());
    }
    auto imgIds = LoadBytes(fix + "/img_ids_f32.bin", kTokens * 4 * 4);
    auto txtIds = LoadBytes(fix + "/txt_ids_f32.bin", kSeq * 4 * 4);
    // Use the FIXTURE embeds in the Euler loop so per-step refs stay valid.
    auto fixEmb = LoadBytes(fix + "/embeds_f32.bin", embeds.size() * 4);
    Graph dit(env, models + "/gpu_work/dit_gpu_int4b32.tflite", true,
              {kLiteRtElementTypeFloat32, kLiteRtElementTypeFloat32,
               kLiteRtElementTypeFloat32, kLiteRtElementTypeFloat32,
               kLiteRtElementTypeFloat32},
              kLiteRtElementTypeFloat32);
    std::vector<float> v(lat.size());
    for (int k = 0; k < 4; ++k) {
      float sigma = kSigmas[k];
      double dt = dit.Run({{lat.data(), lat.size() * 4},
                           {fixEmb.data(), fixEmb.size()},
                           {&sigma, 4},
                           {imgIds.data(), imgIds.size()},
                           {txtIds.data(), txtIds.size()}},
                          v.data(), v.size() * 4);
      auto ref = LoadBytes(fix + "/dit_out_" + std::to_string(k) + "_f32.bin",
                           v.size() * 4);
      double maxd = 0;
      double c = Cos(v.data(), (const float*)ref.data(), v.size(), &maxd);
      printf("  step %d  %.2fs  vs CPU fixture: cos=%.6f max|d|=%.3e\n", k + 1,
             dt, c, maxd);
      float ds = kSigmas[k + 1] - kSigmas[k];
      for (size_t i = 0; i < lat.size(); ++i) lat[i] += ds * v[i];
    }
  }

  // ---- Stage C: VAE on CPU -> PNG -----------------------------------------
  if (stage == "all" || stage == "vae") {
    printf("== VAE fp32 CPU ==\n");
    auto z = LoadBytes(fix + "/z_vae_f32.bin", 32 * 64 * 64 * 4);
    Graph vae(env, models + "/hf_upload/vae_dec_fp32.tflite", false,
              {kLiteRtElementTypeFloat32}, kLiteRtElementTypeFloat32);
    std::vector<float> y((size_t)3 * 512 * 512);
    double dt = vae.Run({{z.data(), z.size()}}, y.data(), y.size() * 4);
    printf("  run %.2fs\n", dt);
    std::vector<uint8_t> rgba((size_t)512 * 512 * 4, 255);
    for (int c = 0; c < 3; ++c)
      for (int p = 0; p < 512 * 512; ++p) {
        float f = (y[(size_t)c * 262144 + p] / 2 + 0.5f) * 255.0f;
        rgba[(size_t)p * 4 + c] = (uint8_t)std::max(0.f, std::min(255.f, std::round(f)));
      }
    CGContextRef ctx = CGBitmapContextCreate(
        rgba.data(), 512, 512, 8, 512 * 4, CGColorSpaceCreateDeviceRGB(),
        kCGImageAlphaNoneSkipLast);
    CGImageRef img = CGBitmapContextCreateImage(ctx);
    NSURL* url = [NSURL fileURLWithPath:@"/tmp/bonsai_smoke_vae.png"];
    CGImageDestinationRef dest = CGImageDestinationCreateWithURL(
        (__bridge CFURLRef)url, CFSTR("public.png"), 1, nullptr);
    CGImageDestinationAddImage(dest, img, nullptr);
    CGImageDestinationFinalize(dest);
    printf("  wrote /tmp/bonsai_smoke_vae.png (compare vs %s/expected.png)\n",
           fix.c_str());
  }

  printf("SMOKE_DONE\n");
  return 0;
}
