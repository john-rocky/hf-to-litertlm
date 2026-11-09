// Writes a NemotronH decoder (the architecture of nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B: Mamba-2, MoE with relu2
// experts, attention without positional encoding) as a .tflite with the LiteRT Tensor API, in the signature layout
// LiteRT-LM's stock executor runs with an ExecutorMetadata section (the layout litert-torch export_hf writes for
// Nemotron-H with the hybrid cache):
//   prefill_<N> : tokens [1,N] i32, input_pos [N] i32, mask [1,1,N,C] f32, and per layer l (original index)
//                 Mamba-2: kv_cache_mc_<l> [1,Cd,K] f32 (the last K conv inputs), kv_cache_mr_<l> [1,H,P,S] f32
//                 attention: kv_cache_k_<l> [1,KV,C,hd] f32, kv_cache_v_<l> [1,KV,hd,C] f32 (V stored transposed)
//                 -> the same state tensors
//   decode      : the same with N = 1 -> the states and logits [1,1,V] f32
// Block: x = x + mixer(RMSNorm(x)).
//   M (Mamba-2, NVIDIA's arithmetic): in_proj -> z | xBC | dt; causal depthwise conv over [state, xBC] (+ bias), SiLU;
//     x | B | C; dt = softplus(dt + dt_bias) (NVIDIA's time_step_limit (0, inf): no clamp); A = -exp(A_log) (packed);
//     state' = state * exp(dt A) + dt x (outer) B (head h reads group h / (H / G)); y = state' . C + D x;
//     y * silu(z) -> RMS per group of I / G -> * norm weight -> out_proj.
//     decode: the recurrence; prefill: the chunked SSD scan in the folded rank <= 3 form of the Nemotron-H litert-torch
//     patch (granite's _folded_torch_forward: CUMSUM, exp of the masked segment differences, batched matmuls), with
//     the prefill pad guard (tokens == 0 are the runtime's right padding: the mixer input and the conv output are
//     zeroed there and dt is multiplied by the valid mask after softplus, so a pad step is an exact identity; the new
//     conv state is the window of the last K valid inputs, picked by a one-hot matmul).
//   * (attention): q / k / v, grouped-query attention over the cache (the runtime's mask hides unwritten slots), o_proj.
//   E (MoE): router FC [E, D] (fp32) -> sigmoid; top-k on sigmoid + e_score_correction_bias; weights = the plain sigmoid
//     values of the chosen experts (picked by a one-hot built from the indices) / (their sum + 1e-20) * routed scale;
//     routed experts (relu2 MLPs) + the shared expert (relu2 MLP).
//     --relu2 gelu_limit (default): the "moe" custom op with gate = s * up, ff1 = up, linear = down, per-expert scale
//       1/s, activation gelu_tanh = relu2 in the limit (pack.json and nh_pack.py document the bound).
//     --relu2 gelu_in: the "moe" custom op with gate = ff1 = up (one tensor), the op's input scaled by s and the
//       per-expert scale 1/s^2: act(s u) (s u) / s^2 = gelu(s u) u / s, the same limit without a second weight copy.
//     --relu2 dense: no custom op; all experts as two FCs per layer, the routing weights spread to [N, E] in between.
//     --relu2 relu2op: the custom op with gate = ff1 = up and activation "relu2" (for a runtime that implements it).
// Weights come from nh_pack.py (weights.bin, index.tsv, config.tsv, moe_options.bin), read through mmap; each
// constant tensor is created once and shared by every signature.

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/flags/flag.h"  // from @com_google_absl
#include "absl/flags/parse.h"  // from @com_google_absl
#include "absl/log/absl_check.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/strings/numbers.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/str_join.h"  // from @com_google_absl
#include "absl/strings/str_split.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "tensor/arithmetic.h"
#include "tensor/backends/tflite/arithmetic_tflite.h"
#include "tensor/backends/tflite/tflite_flatbuffer_conversion.h"
#include "tensor/buffer.h"
#include "tensor/datatypes.h"
#include "tensor/tensor.h"

ABSL_FLAG(std::string, weights_dir, "",
          "Directory written by nh_pack.py (weights.bin, index.tsv, config.tsv, moe_options.bin).");
ABSL_FLAG(std::string, out, "", "Output .tflite path.");
ABSL_FLAG(std::string, prefill, "128", "Prefill lengths, comma separated (empty: decode only).");
ABSL_FLAG(int, cache, 4096, "KV cache length.");
ABSL_FLAG(bool, moe_fusion_guard, true,
          "Give the moe output a second consumer (moe + shared + moe * 0) so the GPU delegate cannot fuse the next "
          "elementwise op into the moe kernel (moe_experts_kernel.cc writes args.dst.Write(Convert(sum)) and a fused "
          "op then emits invalid shader code, found with the Qwen3-30B-A3B writer). Numerically exact (+0).");
ABSL_FLAG(bool, debug_outputs, false,
          "Also write a signature decode_debug: decode plus per-layer residual outputs hidden_<l> [1,1,D].");

namespace {

using ::litert::tensor::BlockwiseQuantization;
using ::litert::tensor::ModelFactory;
using ::litert::tensor::OwningCpuBuffer;
using ::litert::tensor::PerChannelAffineQuantization;
using ::litert::tensor::SpanCpuBuffer;
using ::litert::tensor::TensorHandle;
using ::litert::tensor::TfLiteMixinTag;
using ::litert::tensor::Type;
using T = ::litert::tensor::Tensor<TfLiteMixinTag>;

struct Config {
  int hidden = 0, heads = 0, kv_heads = 0, head_dim = 0, m_heads = 0, m_head_dim = 0, m_state = 0, m_groups = 0,
      m_conv = 0, m_chunk = 0, experts = 0, top_k = 0, moe_inter = 0, shared_inter = 0, vocab = 0, norm_topk = 1;
  float eps = 0, routed_scale = 0, gelu_s = 0;
  std::string pattern, relu2, expert_kind;
  std::vector<int> layer_ids;
  bool quant_experts = false;
};

std::vector<std::string> ReadLines(const std::string& path) {
  std::ifstream f(path);
  ABSL_CHECK(f.good()) << "cannot read " << path;
  std::vector<std::string> lines;
  for (std::string line; std::getline(f, line);) {
    if (!line.empty()) lines.push_back(line);
  }
  return lines;
}

Config ReadConfig(const std::string& path) {
  absl::flat_hash_map<std::string, std::string> kv;
  for (const std::string& line : ReadLines(path)) {
    std::vector<std::string> p = absl::StrSplit(line, '\t');
    ABSL_CHECK_EQ(p.size(), 2u) << line;
    kv[p[0]] = p[1];
  }
  auto i = [&](const char* k) {
    int v = 0;
    ABSL_CHECK(absl::SimpleAtoi(kv.at(k), &v)) << k;
    return v;
  };
  auto f = [&](const char* k) {
    double v = 0;
    ABSL_CHECK(absl::SimpleAtod(kv.at(k), &v)) << k;
    return static_cast<float>(v);
  };
  Config c;
  c.hidden = i("hidden");
  c.heads = i("heads");
  c.kv_heads = i("kv_heads");
  c.head_dim = i("head_dim");
  c.m_heads = i("m_heads");
  c.m_head_dim = i("m_head_dim");
  c.m_state = i("m_state");
  c.m_groups = i("m_groups");
  c.m_conv = i("m_conv");
  c.m_chunk = i("m_chunk");
  c.experts = i("experts");
  c.top_k = i("top_k");
  c.moe_inter = i("moe_inter");
  c.shared_inter = i("shared_inter");
  c.vocab = i("vocab");
  c.norm_topk = i("norm_topk");
  c.eps = f("eps");
  c.routed_scale = f("routed_scale");
  c.gelu_s = f("gelu_s");
  c.pattern = kv.at("pattern");
  c.relu2 = kv.at("relu2");
  c.expert_kind = kv.at("expert_kind");
  for (absl::string_view s : absl::StrSplit(kv.at("layer_ids"), ',')) {
    int v = 0;
    ABSL_CHECK(absl::SimpleAtoi(s, &v));
    c.layer_ids.push_back(v);
  }
  ABSL_CHECK_EQ(c.layer_ids.size(), c.pattern.size());
  c.quant_experts = c.expert_kind == "xb4" || c.expert_kind == "xb8";
  return c;
}

// The packed weights, mapped read-only. Each constant tensor is created once and reused by every signature.
class Weights {
 public:
  explicit Weights(const std::string& dir) {
    const std::string bin = dir + "/weights.bin";
    int fd = open(bin.c_str(), O_RDONLY);
    ABSL_CHECK_GE(fd, 0) << "cannot open " << bin;
    struct stat st;
    ABSL_CHECK_EQ(fstat(fd, &st), 0);
    size_ = static_cast<size_t>(st.st_size);
    void* p = mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd, 0);
    ABSL_CHECK(p != MAP_FAILED) << "mmap failed for " << bin;
    close(fd);
    base_ = static_cast<const std::byte*>(p);
    for (const std::string& line : ReadLines(dir + "/index.tsv")) {
      std::vector<std::string> f = absl::StrSplit(line, '\t');
      ABSL_CHECK_EQ(f.size(), 8u) << line;
      Entry e;
      e.kind = f[1];
      for (absl::string_view d : absl::StrSplit(f[2], ',')) {
        int v = 0;
        ABSL_CHECK(absl::SimpleAtoi(d, &v));
        e.shape.push_back(v);
      }
      ABSL_CHECK(absl::SimpleAtoi(f[3], &e.off));
      ABSL_CHECK(absl::SimpleAtoi(f[4], &e.bytes));
      ABSL_CHECK(absl::SimpleAtoi(f[5], &e.scale_off));
      ABSL_CHECK(absl::SimpleAtoi(f[6], &e.scale_count));
      ABSL_CHECK(absl::SimpleAtoi(f[7], &e.block));
      ABSL_CHECK_LE(e.off + e.bytes, size_) << f[0];
      entries_[f[0]] = e;
    }
  }

  bool Has(const std::string& name) const { return entries_.contains(name); }

  // fp32 values of an fp32 entry (for constants the writer rebuilds, e.g. the conv taps).
  std::vector<float> Floats(const std::string& name) const {
    auto it = entries_.find(name);
    ABSL_CHECK(it != entries_.end()) << "missing weight " << name;
    ABSL_CHECK_EQ(it->second.kind, "fp32") << name;
    std::vector<float> v(it->second.bytes / sizeof(float));
    std::memcpy(v.data(), base_ + it->second.off, it->second.bytes);
    return v;
  }

  std::vector<int> Shape(const std::string& name) const { return entries_.at(name).shape; }

  // An fp32 entry as a constant of another shape with the same element count, made once per (name, shape) and
  // shared by every signature. The GPU delegate refuses a RESHAPE whose only input is a constant
  // ("RESHAPE: Expected 1 runtime input tensor(s), but node has 0", LiteRT-LM 0.19.1 on Mac), so constants are
  // written in the shape their consumer needs.
  const T& Shaped(const std::string& name, const std::vector<int>& shape) {
    const std::string key = absl::StrCat(name, "@", absl::StrJoin(shape, ","));
    auto it = cache_.find(key);
    if (it != cache_.end()) return it->second;
    std::vector<float> v = Floats(name);
    size_t n = 1;
    for (int d : shape) n *= static_cast<size_t>(d);
    ABSL_CHECK_EQ(n, v.size()) << name;
    T t({.name = key, .type = Type::kFP32, .shape = shape, .buffer = OwningCpuBuffer::Copy<Type::kFP32>(v)});
    return cache_.emplace(key, t).first->second;
  }

  const T& Get(const std::string& name) {
    auto it = cache_.find(name);
    if (it != cache_.end()) return it->second;
    auto eit = entries_.find(name);
    ABSL_CHECK(eit != entries_.end()) << "missing weight " << name;
    const Entry& e = eit->second;
    // One buffer object per (offset, size): entries that alias the same bytes (the gelu_limit gate = up's codes)
    // then share one TFLite buffer instead of writing the bytes twice (the serializer keys buffers by object).
    std::shared_ptr<SpanCpuBuffer>& buffer = spans_[std::make_pair(e.off, e.bytes)];
    if (!buffer) buffer = std::make_shared<SpanCpuBuffer>(base_ + e.off, e.bytes);
    T t;
    if (e.kind == "fp32") {
      t = T({.name = name, .type = Type::kFP32, .shape = e.shape, .buffer = buffer});
    } else if (e.kind == "bw4") {
      auto q = std::make_shared<BlockwiseQuantization>(Scales(e), std::vector<int64_t>{}, e.block, 0);
      t = T({.name = name, .type = Type::kI4, .shape = e.shape, .buffer = buffer, .quantization = q});
    } else if (e.kind == "pc8") {
      auto q = std::make_shared<PerChannelAffineQuantization>(
          Scales(e), std::vector<int64_t>(e.scale_count, 0), 0);
      t = T({.name = name, .type = Type::kI8, .shape = e.shape, .buffer = buffer, .quantization = q});
    } else if (e.kind == "xb8" || e.kind == "xb4") {
      auto q = std::make_shared<BlockwiseQuantization>(std::vector<float>{}, std::vector<int64_t>{}, e.block, 0);
      q->block_shape = {1, 1, 1, e.block};
      q->scales_tensor_name = name + "_scales";
      t = T({.name = name,
             .type = e.kind == "xb4" ? Type::kI4 : Type::kI8,
             .shape = e.shape,
             .buffer = buffer,
             .quantization = q});
    } else {
      ABSL_LOG(FATAL) << "unknown kind " << e.kind << " for " << name;
    }
    return cache_.emplace(name, t).first->second;
  }

 private:
  struct Entry {
    std::string kind;
    std::vector<int> shape;
    uint64_t off = 0, bytes = 0, scale_off = 0, scale_count = 0;
    int block = 0;
  };

  std::vector<float> Scales(const Entry& e) const {
    std::vector<float> s(e.scale_count);
    std::memcpy(s.data(), base_ + e.scale_off, e.scale_count * sizeof(float));
    return s;
  }

  const std::byte* base_ = nullptr;
  size_t size_ = 0;
  absl::flat_hash_map<std::string, Entry> entries_;
  absl::flat_hash_map<std::string, T> cache_;
  absl::flat_hash_map<std::pair<uint64_t, uint64_t>, std::shared_ptr<SpanCpuBuffer>> spans_;
};

std::vector<uint8_t> ReadBytes(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  ABSL_CHECK(f.good()) << "cannot read " << path;
  return std::vector<uint8_t>(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
}

T ConstF(std::vector<float> v, std::vector<int> shape) {
  return T({.type = Type::kFP32, .shape = std::move(shape), .buffer = OwningCpuBuffer::Copy<Type::kFP32>(v)});
}

T ConstI(std::vector<int32_t> v, std::vector<int> shape) {
  return T({.type = Type::kI32, .shape = std::move(shape), .buffer = OwningCpuBuffer::Copy<Type::kI32>(v)});
}

T Cat(std::vector<T> parts, int axis) { return Concatenation(absl::MakeSpan(parts), axis); }

std::vector<float> Iota(int n) {
  std::vector<float> v(n);
  for (int i = 0; i < n; ++i) v[i] = static_cast<float>(i);
  return v;
}

// RMSNorm over the last axis: x * rsqrt(mean(x^2) + eps) * w.
T RmsNorm(const T& x, const T& w, float eps) {
  const int last = static_cast<int>(x.GetShape().size()) - 1;
  T ms = Mean(Mul(x, x), {last}, /*keep_dims=*/true);
  return Mul(Mul(x, Rsqrt(Add(ms, ConstF({eps}, {1})))), w);
}

T Silu(const T& x) { return Mul(x, Logistic(x)); }

// softplus(x) = max(x, 0) + log(1 + exp(-|x|)) (no overflow for large |x|).
T Softplus(const T& x) {
  return Add(Maximum(x, ConstF({0.0f}, {1})), Log(Add(Exp(Neg(Abs(x))), ConstF({1.0f}, {1}))));
}

T Relu2(const T& x) {
  T r = Relu(x);
  return Mul(r, r);
}

// exp(segment_sum) from the inclusive cumulative sum cs [B, L]: out[b, i, j] = exp(cs[i] - cs[j]) for j <= i, else 0.
// The difference is masked to 0 before the exp, so nothing infinite reaches it (granite's helper).
T ExpSegSum(const T& cs, int L) {
  const int B = cs.GetShape()[0];
  T diff = Sub(Reshape(cs, {B, L, 1}), Reshape(cs, {B, 1, L}));
  std::vector<float> keep(static_cast<size_t>(L) * L, 0.0f);
  for (int i = 0; i < L; ++i) {
    for (int j = 0; j <= i; ++j) keep[static_cast<size_t>(i) * L + j] = 1.0f;
  }
  T k = ConstF(keep, {1, L, L});
  return Mul(Exp(Mul(diff, k)), k);
}

// [.., G, ..] group axis repeated rep times without BROADCAST_TO: x [A, G, 1, S] -> concat rep copies on axis 2.
T RepeatGroups(const T& x4, int rep) {
  std::vector<T> parts(rep, x4);
  return Concatenation(absl::MakeSpan(parts), 2);
}

struct Signature {
  std::vector<TensorHandle> inputs, outputs;
};

struct Ctx {
  Ctx(const Config& cfg, Weights& wts, const std::vector<uint8_t>& opts) : c(cfg), w(wts), moe_options(opts) {}
  const Config& c;
  Weights& w;
  const std::vector<uint8_t>& moe_options;
  int seq = 1, cache = 0;
  bool decode = true;
  T tokens, valid2, valid3;  // valid masks (prefill only): [1, seq], [1, seq, 1]
};

// Mamba-2 mixer on the normed input hm [1, seq, D]; reads / returns the conv and SSM states.
T Mamba(Ctx& x, int l, T hm, const T& mc_in, const T& mr_in, T* mc_out, T* mr_out) {
  const Config& c = x.c;
  const int seq = x.seq, H = c.m_heads, P = c.m_head_dim, S = c.m_state, G = c.m_groups, K = c.m_conv;
  const int I = H * P, Cd = I + 2 * G * S, rep = H / G;
  const std::string p = absl::StrCat("backbone.layers.", l, ".mixer.");
  if (!x.decode) hm = Mul(hm, x.valid3);
  T proj = FullyConnected(hm, x.w.Get(p + "in_proj.weight"));  // [1, seq, I + Cd + H]
  T z = Slice(proj, {0, 0, 0}, {1, seq, I});
  T xbc = Slice(proj, {0, 0, I}, {1, seq, Cd});
  T dtr = Slice(proj, {0, 0, I + Cd}, {1, seq, H});

  // Causal depthwise conv over [state (K), x (seq)]: out[t] = sum_k w[:, k] * cat[:, t + 1 + k] + b.
  T cat = Cat({mc_in, Transpose(xbc, {0, 2, 1})}, 2);  // [1, Cd, K + seq]
  const std::vector<float> cw = x.w.Floats(p + "conv1d.weight2d");  // [Cd, K]
  const std::vector<float> cb = x.w.Floats(p + "conv1d.bias");      // [Cd]
  T acc;
  for (int k = 0; k < K; ++k) {
    std::vector<float> tap(Cd);
    for (int ch = 0; ch < Cd; ++ch) tap[ch] = cw[static_cast<size_t>(ch) * K + k];
    T term = Mul(Slice(cat, {0, 0, 1 + k}, {1, Cd, seq}), ConstF(tap, {1, Cd, 1}));
    acc = k == 0 ? term : Add(acc, term);
  }
  acc = Add(acc, ConstF(cb, {1, Cd, 1}));
  T xc = Transpose(Silu(acc), {0, 2, 1});  // [1, seq, Cd]
  if (x.decode) {
    *mc_out = Slice(cat, {0, 0, 1}, {1, Cd, K});
  } else {
    xc = Mul(xc, x.valid3);
    // New conv state = cat[:, :, vlen + j], j < K: a one-hot [K + seq, K] matmul (relu(1 - |p - (vlen + j)|)).
    T vlen = Reshape(Sum(x.valid2, {1}, /*keep_dims=*/true), {1, 1, 1});
    T target = Add(vlen, ConstF(Iota(K), {1, 1, K}));                      // [1, 1, K]
    T pos = ConstF(Iota(K + seq), {1, K + seq, 1});                        // [1, K + seq, 1]
    T sel = Relu(Sub(ConstF({1.0f}, {1}), Abs(Sub(pos, target))));        // [1, K + seq, K]
    *mc_out = BatchMatMul(cat, sel);                                        // [1, Cd, K]
  }
  T xs = Slice(xc, {0, 0, 0}, {1, seq, I});
  T B = Slice(xc, {0, 0, I}, {1, seq, G * S});
  T C = Slice(xc, {0, 0, I + G * S}, {1, seq, G * S});

  T dt = Softplus(Add(dtr, x.w.Get(p + "dt_bias")));  // [1, seq, H]
  if (!x.decode) dt = Mul(dt, x.valid3);
  // A = -exp(A_log) and D, [H] in the pack, as constants in the shape each use needs (no RESHAPE of a constant).

  T y;
  if (x.decode) {
    T dt2 = Reshape(dt, {1, H});
    T dA = Exp(Mul(dt2, x.w.Get(p + "A")));                                                 // [1, H]
    T xh = Reshape(xs, {1, H, P});
    T Bh = Reshape(RepeatGroups(Reshape(B, {1, G, 1, S}), rep), {1, H, S});
    T Ch = Reshape(RepeatGroups(Reshape(C, {1, G, 1, S}), rep), {1, H, S});
    T dx = Mul(xh, Reshape(dt2, {1, H, 1}));                                 // [1, H, P]
    T upd = Mul(Reshape(dx, {1, H, P, 1}), Reshape(Bh, {1, H, 1, S}));       // [1, H, P, S]
    T st = Add(Mul(mr_in, Reshape(dA, {1, H, 1, 1})), upd);
    *mr_out = st;
    T yy = BatchMatMul(Reshape(st, {H, P, S}), Reshape(Ch, {H, S, 1}));      // [H, P, 1]
    y = Add(Reshape(yy, {1, H, P}), Mul(xh, x.w.Shaped(p + "D", {1, H, 1})));
    y = Reshape(y, {1, 1, I});
  } else {
    const int cl = std::min(c.m_chunk, seq);
    ABSL_CHECK_EQ(seq % cl, 0) << "prefill length " << seq << " is not a multiple of the chunk " << cl;
    const int nc = seq / cl, Hc = H * nc;
    T xs4 = Reshape(xs, {1, seq, H, P});
    T B4 = Reshape(RepeatGroups(Reshape(B, {seq, G, 1, S}), rep), {1, seq, H, S});
    T C4 = Reshape(RepeatGroups(Reshape(C, {seq, G, 1, S}), rep), {1, seq, H, S});
    T dres = Mul(xs4, x.w.Shaped(p + "D", {1, 1, H, 1}));
    T xdt = Mul(xs4, Reshape(dt, {1, seq, H, 1}));
    T adt = Mul(dt, x.w.Shaped(p + "A", {1, 1, H}));                                   // [1, seq, H]
    auto fold = [&](const T& t, int last) {                                    // [1, seq, H, last] -> [Hc, cl, last]
      return Reshape(Transpose(t, {0, 2, 1, 3}), {Hc, cl, last});
    };
    T xf = fold(xdt, P), bf = fold(B4, S), cf = fold(C4, S);
    T af = Reshape(Transpose(adt, {0, 2, 1}), {Hc, cl});
    T acs = Cumsum(af, 1);                                                      // [Hc, cl]
    T Lf = ExpSegSum(acs, cl);                                                  // [Hc, cl, cl]
    T Gm = BatchMatMul(cf, bf, /*adj_x=*/false, /*adj_y=*/true);                // [Hc, cl, cl]
    T Ydiag = BatchMatMul(Mul(Gm, Lf), xf);                                     // [Hc, cl, P]
    T last = Slice(acs, {0, cl - 1}, {Hc, 1});                                  // [Hc, 1]
    T decay = Exp(Sub(last, acs));                                              // [Hc, cl]
    T states = BatchMatMul(Mul(bf, Reshape(decay, {Hc, cl, 1})), xf, /*adj_x=*/true);  // [Hc, S, P]
    T prev = Reshape(Transpose(mr_in, {0, 1, 3, 2}), {H, 1, S * P});            // [H, 1, S*P]
    T sbh = Cat({prev, Reshape(states, {H, nc, S * P})}, 1);         // [H, nc+1, S*P]
    T tail = Cat({ConstF(std::vector<float>(H, 0.0f), {H, 1}), Reshape(last, {H, nc})}, 1);
    T dchunk = ExpSegSum(Cumsum(tail, 1), nc + 1);                              // [H, nc+1, nc+1]
    T nst = BatchMatMul(dchunk, sbh);                                           // [H, nc+1, S*P]
    T sout = Slice(nst, {0, 0, 0}, {H, nc, S * P});
    T fin = Slice(nst, {0, nc, 0}, {H, 1, S * P});
    T Yoff = Mul(BatchMatMul(cf, Reshape(sout, {Hc, S, P})), Reshape(Exp(acs), {Hc, cl, 1}));
    T yh = Transpose(Reshape(Add(Ydiag, Yoff), {1, H, seq, P}), {0, 2, 1, 3});  // [1, seq, H, P]
    y = Reshape(Add(yh, dres), {1, seq, I});
    *mr_out = Transpose(Reshape(fin, {1, H, S, P}), {0, 1, 3, 2});              // [1, H, P, S]
  }
  // Gated RMSNorm (norm after the gate), per group of I / G.
  T g = Mul(y, Silu(z));
  const int gs = I / G;
  T g4 = Reshape(g, {1, seq, G, gs});
  T ms = Mean(Mul(g4, g4), {3}, /*keep_dims=*/true);
  T gn = Reshape(Mul(g4, Rsqrt(Add(ms, ConstF({c.eps}, {1})))), {1, seq, I});
  gn = Mul(gn, x.w.Get(p + "norm.weight"));
  return FullyConnected(gn, x.w.Get(p + "out_proj.weight"));
}

T Attention(Ctx& x, int l, const T& h, const T& kin, const T& vin, const T& mask, const T& kstart, const T& vstart,
            T* kout, T* vout) {
  const Config& c = x.c;
  const int seq = x.seq, H = c.heads, KV = c.kv_heads, hd = c.head_dim, g = H / KV, C = x.cache;
  const std::string p = absl::StrCat("backbone.layers.", l, ".mixer.");
  T q = Transpose(Reshape(FullyConnected(h, x.w.Get(p + "q_proj.weight")), {1, seq, H, hd}), {0, 2, 1, 3});
  T k = Transpose(Reshape(FullyConnected(h, x.w.Get(p + "k_proj.weight")), {1, seq, KV, hd}), {0, 2, 1, 3});
  T v = Transpose(Reshape(FullyConnected(h, x.w.Get(p + "v_proj.weight")), {1, seq, KV, hd}), {0, 2, 3, 1});
  T kc = DynamicUpdateSlice(kin, k, kstart);
  T vc = DynamicUpdateSlice(vin, v, vstart);
  kc.SetName(absl::StrCat("kv_cache_k_", l));
  vc.SetName(absl::StrCat("kv_cache_v_", l));
  *kout = kc;
  *vout = vc;
  const T scale = ConstF({1.0f / std::sqrt(static_cast<float>(hd))}, {1});
  T scores = BatchMatMul(Reshape(q, {KV, g * seq, hd}), Reshape(kc, {KV, C, hd}), /*adj_x=*/false, /*adj_y=*/true);
  scores = Add(Mul(Reshape(scores, {1, H, seq, C}), scale), mask);
  T probs = Reshape(Softmax(scores), {KV, g * seq, C});
  T o = BatchMatMul(probs, Reshape(vc, {KV, hd, C}), /*adj_x=*/false, /*adj_y=*/true);  // [KV, g*seq, hd]
  o = Reshape(Transpose(Reshape(o, {1, H, seq, hd}), {0, 2, 1, 3}), {1, seq, H * hd});
  return FullyConnected(o, x.w.Get(p + "o_proj.weight"));
}

T Moe(Ctx& x, int l, const T& h) {
  const Config& c = x.c;
  const int seq = x.seq, D = c.hidden, E = c.experts, K = c.top_k, I = c.moe_inter;
  const std::string p = absl::StrCat("backbone.layers.", l, ".mixer.");
  T logits = FullyConnected(h, x.w.Get(p + "gate.weight"), ::litert::tensor::kActNone,
                            /*keep_num_dims=*/false);  // [seq, E]
  T scores = Logistic(logits);
  T choice = Add(scores, x.w.Shaped(p + "gate.e_score_correction_bias", {1, E}));
  std::vector<T> top = TopK(choice, K);  // values [seq, K], indices [seq, K] i32
  // One-hot of the chosen experts: relu(1 - |idx - e|) (exact for integers), then the plain sigmoid values.
  T idxf = Reshape(Cast(top[1], Type::kFP32), {seq, K, 1});
  T oh = Relu(Sub(ConstF({1.0f}, {1}), Abs(Sub(idxf, ConstF(Iota(E), {1, 1, E})))));  // [seq, K, E]
  T wts = Sum(Mul(oh, Reshape(scores, {seq, 1, E})), {2}, /*keep_dims=*/false);       // [seq, K]
  if (c.norm_topk) wts = Div(wts, Add(Sum(wts, {1}, /*keep_dims=*/true), ConstF({1e-20f}, {1})));
  wts = Mul(wts, ConstF({c.routed_scale}, {1}));

  T shared = FullyConnected(Relu2(FullyConnected(h, x.w.Get(p + "shared_experts.up_proj.weight"))),
                            x.w.Get(p + "shared_experts.down_proj.weight"));
  if (c.relu2 == "dense") {
    T dense_w = Sum(Mul(oh, Reshape(wts, {seq, K, 1})), {1}, /*keep_dims=*/false);  // [seq, E]
    T u = Relu2(FullyConnected(h, x.w.Get(p + "experts.up_all")));                  // [1, seq, E*I]
    u = Reshape(Mul(Reshape(u, {seq, E, I}), Reshape(dense_w, {seq, E, 1})), {1, seq, E * I});
    return Add(FullyConnected(u, x.w.Get(p + "experts.down_all")), shared);
  }
  const T src = c.relu2 == "gelu_in" ? Mul(h, ConstF({c.gelu_s}, {1})) : h;
  std::vector<T> ins = {src, Reshape(wts, {1, seq, K}), Reshape(top[1], {1, seq, K})};
  auto add_w = [&](const std::string& n) {
    ins.push_back(x.w.Get(p + "experts." + n));
    if (c.quant_experts) ins.push_back(x.w.Get(p + "experts." + n + "_scales"));
  };
  float per_expert = 1.0f;
  if (c.relu2 == "gelu_limit") {
    add_w("gate");
    add_w("up");
    add_w("down");
    per_expert = 1.0f / c.gelu_s;
  } else if (c.relu2 == "gelu_in") {
    add_w("up");
    add_w("up");
    add_w("down");
    per_expert = 1.0f / (c.gelu_s * c.gelu_s);
  } else {  // relu2op: gate = ff1 = up
    ABSL_CHECK_EQ(c.relu2, "relu2op");
    add_w("up");
    add_w("up");
    add_w("down");
  }
  ins.push_back(ConstF(std::vector<float>(E, per_expert), {1, 1, 1, E}));
  std::vector<T> moe = Custom(absl::MakeSpan(ins), "moe", x.moe_options, {{1, seq, D}}, {Type::kFP32});
  if (absl::GetFlag(FLAGS_moe_fusion_guard)) {
    return Add(Add(moe[0], shared), Mul(moe[0], ConstF({0.0f}, {1})));
  }
  return Add(moe[0], shared);
}

Signature Build(const Config& c, Weights& w, const std::vector<uint8_t>& moe_options, int seq, int cache, bool decode,
                bool debug = false) {
  Ctx x(c, w, moe_options);
  x.seq = seq;
  x.cache = cache;
  x.decode = decode;
  const int D = c.hidden, H = c.m_heads, P = c.m_head_dim, S = c.m_state, G = c.m_groups, K = c.m_conv;
  const int Cd = H * P + 2 * G * S;
  T tokens({.name = "tokens", .type = Type::kI32, .shape = {1, seq}});
  T pos({.name = "input_pos", .type = Type::kI32, .shape = {seq}});
  T mask({.name = "mask", .type = Type::kFP32, .shape = {1, 1, seq, cache}});
  if (!decode) {
    x.valid2 = Cast(NotEqual(tokens, ConstI({0}, {1})), Type::kFP32);  // [1, seq]
    x.valid3 = Reshape(x.valid2, {1, seq, 1});
  }
  const int nl = static_cast<int>(c.layer_ids.size());
  std::vector<T> sin(2 * nl), sout(2 * nl);
  std::vector<std::string> snames(2 * nl);
  for (int i = 0; i < nl; ++i) {
    const int l = c.layer_ids[i];
    const char kd = c.pattern[i];
    if (kd == 'M') {
      snames[2 * i] = absl::StrCat("kv_cache_mc_", l);
      snames[2 * i + 1] = absl::StrCat("kv_cache_mr_", l);
      sin[2 * i] = T({.name = snames[2 * i], .type = Type::kFP32, .shape = {1, Cd, K}});
      sin[2 * i + 1] = T({.name = snames[2 * i + 1], .type = Type::kFP32, .shape = {1, H, P, S}});
    } else if (kd == '*') {
      snames[2 * i] = absl::StrCat("kv_cache_k_", l);
      snames[2 * i + 1] = absl::StrCat("kv_cache_v_", l);
      sin[2 * i] = T({.name = snames[2 * i], .type = Type::kFP32, .shape = {1, c.kv_heads, cache, c.head_dim}});
      sin[2 * i + 1] = T({.name = snames[2 * i + 1], .type = Type::kFP32, .shape = {1, c.kv_heads, c.head_dim, cache}});
    }
  }
  T pos0 = Slice(pos, {0}, {1});
  T zero = ConstI({0}, {1});
  T kstart = Cat({zero, zero, pos0, zero}, 0);
  T vstart = Cat({zero, zero, zero, pos0}, 0);

  T h = Reshape(EmbeddingLookup(Reshape(tokens, {seq}), w.Get("backbone.embeddings.weight")), {1, seq, D});
  std::vector<T> hidden;
  for (int i = 0; i < nl; ++i) {
    const int l = c.layer_ids[i];
    const char kd = c.pattern[i];
    T hn = RmsNorm(h, w.Get(absl::StrCat("backbone.layers.", l, ".norm.weight")), c.eps);
    T out;
    if (kd == 'M') {
      out = Mamba(x, l, hn, sin[2 * i], sin[2 * i + 1], &sout[2 * i], &sout[2 * i + 1]);
    } else if (kd == '*') {
      out = Attention(x, l, hn, sin[2 * i], sin[2 * i + 1], mask, kstart, vstart, &sout[2 * i], &sout[2 * i + 1]);
    } else if (kd == 'E') {
      out = Moe(x, l, hn);
    } else {
      ABSL_LOG(FATAL) << "unknown layer kind " << kd;
    }
    h = Add(h, out);
    if (debug) hidden.push_back(h);
  }

  Signature s;
  s.inputs = {tokens, pos, mask};
  for (int i = 0; i < 2 * nl; ++i) {
    if (!snames[i].empty()) s.inputs.push_back(sin[i]);
  }
  for (int i = 0; i < 2 * nl; ++i) {
    if (snames[i].empty()) continue;
    sout[i].SetName(snames[i]);
    s.outputs.push_back(sout[i]);
  }
  if (decode) {
    T logits = FullyConnected(RmsNorm(h, w.Get("backbone.norm_f.weight"), c.eps), w.Get("lm_head.weight"));
    logits.SetName("logits");
    s.outputs.push_back(logits);
  }
  for (int i = 0; i < static_cast<int>(hidden.size()); ++i) {
    hidden[i].SetName(absl::StrCat("hidden_", c.layer_ids[i]));
    s.outputs.push_back(hidden[i]);
  }
  return s;
}

}  // namespace

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);
  const std::string dir = absl::GetFlag(FLAGS_weights_dir);
  const std::string out = absl::GetFlag(FLAGS_out);
  ABSL_CHECK(!dir.empty() && !out.empty()) << "--weights_dir and --out are required";
  const int cache = absl::GetFlag(FLAGS_cache);
  const Config c = ReadConfig(dir + "/config.tsv");
  const std::vector<uint8_t> moe_options = ReadBytes(dir + "/moe_options.bin");
  std::cout << "config: hidden " << c.hidden << " pattern " << c.pattern << " mamba " << c.m_heads << "x"
            << c.m_head_dim << " state " << c.m_state << " groups " << c.m_groups << " attn " << c.heads << "/"
            << c.kv_heads << "x" << c.head_dim << " experts " << c.experts << " top_k " << c.top_k << " inter "
            << c.moe_inter << " shared " << c.shared_inter << " vocab " << c.vocab << " cache " << cache
            << " relu2 " << c.relu2 << " s " << c.gelu_s << " experts " << c.expert_kind << " guard "
            << absl::GetFlag(FLAGS_moe_fusion_guard) << std::endl;
  Weights w(dir);
  ModelFactory factory;
  for (absl::string_view p : absl::StrSplit(absl::GetFlag(FLAGS_prefill), ',', absl::SkipEmpty())) {
    int n = 0;
    ABSL_CHECK(absl::SimpleAtoi(p, &n));
    Signature s = Build(c, w, moe_options, n, cache, /*decode=*/false);
    absl::Status st = factory.AddSignature(s.inputs, s.outputs, absl::StrCat("prefill_", n));
    ABSL_CHECK(st.ok()) << st;
    std::cout << "signature prefill_" << n << std::endl;
  }
  Signature d = Build(c, w, moe_options, 1, cache, /*decode=*/true);
  absl::Status st = factory.AddSignature(d.inputs, d.outputs, "decode");
  ABSL_CHECK(st.ok()) << st;
  std::cout << "signature decode" << std::endl;
  if (absl::GetFlag(FLAGS_debug_outputs)) {
    Signature dd = Build(c, w, moe_options, 1, cache, /*decode=*/true, /*debug=*/true);
    st = factory.AddSignature(dd.inputs, dd.outputs, "decode_debug");
    ABSL_CHECK(st.ok()) << st;
    std::cout << "signature decode_debug (" << dd.outputs.size() << " outputs)" << std::endl;
  }
  st = factory.Save(out);
  ABSL_CHECK(st.ok()) << st;
  std::cout << "WROTE " << out << std::endl;
  return 0;
}
