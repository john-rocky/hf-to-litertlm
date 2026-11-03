// Writes a Qwen3-MoE decoder (the architecture of Qwen/Qwen3-30B-A3B) as a .tflite with the LiteRT Tensor API, in
// the signature layout LiteRT-LM's stock executor runs (token ids in the graph, as litert-torch export_hf writes it):
//   prefill_<N> : tokens [1,N] i32, input_pos [N] i32, mask [1,1,N,C] f32,
//                 kv_cache_k_<i> [1,KV,C,hd] f32, kv_cache_v_<i> [1,KV,hd,C] f32 (V stored transposed)
//                 -> kv_cache_k_<i>, kv_cache_v_<i>
//   decode      : the same with N = 1 -> kv_cache_k_<i>, kv_cache_v_<i>, logits [1,1,V] f32
// Per layer: RMSNorm -> q / k / v projections (no bias) -> per-head q_norm / k_norm (RMSNorm over head_dim) -> RoPE
// -> grouped-query attention over the cache (the runtime's mask hides unwritten slots) -> o_proj -> residual;
// RMSNorm -> router FC [E, D] (always fp32) -> softmax -> top-k -> divide by the top-k sum (norm_topk_prob) ->
// custom op "moe" (SiLU experts) -> residual. The "moe" inputs follow the op's parsers in LiteRT (the XNNPACK
// kernel and the GPU delegate's moe_experts parser): src [1,N,D], top weights [1,N,k] f32, top indices [1,N,k] i32,
// then for fp32 experts gate [I,E,1,D], up [I,E,1,D], down [D,E,1,I], per-expert scale [1,1,1,E] (7 inputs); for
// int8 / int4 experts each weight is followed by its float32 scale tensor [out,E,1,blocks] (10 inputs). Custom
// options = the flexbuffer map the pack wrote (moe_options.bin). Weights come from tapi_pack_moe.py (weights.bin,
// index.tsv, config.tsv), read through mmap; each constant tensor is created once and shared by every signature.
// --debug_topk adds a fourth signature decode_topk (decode + per-layer topk_idx_<l> / topk_w_<l> outputs) for a
// routing check against Transformers; the prefill_<N> and decode signatures are built the same with or without it.
//
// Expert weight kinds (index.tsv): fp32; xa8 = int8 with per-tensor-row affine params (qdim 0, one scale per output
// row shared by the experts, so the params are the true dequantization) and the same scales broadcast in
// <name>_scales [out,E,1,1]; xb8 / xb4 = int8 / int4 with BlockwiseQuantization{block_shape [1,1,1,B], block_size B,
// zero point -1} whose scales tensor IS the op's <name>_scales input (ai-edge-quantizer's multi-axis form, LiteRT
// #11309; needs patches/blockwise_named_scales.diff in the LiteRT checkout, see overlay.sh).

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/flags/flag.h"  // from @com_google_absl
#include "absl/flags/parse.h"  // from @com_google_absl
#include "absl/log/absl_check.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/strings/numbers.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/str_split.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "tensor/arithmetic.h"
#include "tensor/backends/tflite/arithmetic_tflite.h"
#include "tensor/backends/tflite/tflite_flatbuffer_conversion.h"
#include "tensor/buffer.h"
#include "tensor/datatypes.h"
#include "tensor/tensor.h"

ABSL_FLAG(std::string, weights_dir, "",
          "Directory written by tapi_pack_moe.py (weights.bin, index.tsv, config.tsv, moe_options.bin).");
ABSL_FLAG(std::string, out, "", "Output .tflite path.");
ABSL_FLAG(std::string, prefill, "32,128", "Prefill lengths, comma separated (empty: decode only).");
ABSL_FLAG(int, cache, 4096, "KV cache length.");
ABSL_FLAG(bool, debug_topk, false,
          "Also write a signature decode_topk: the decode graph plus, per layer l, the outputs topk_idx_<l> [1,1,K] "
          "i32 (the experts the router picked, in TOPK_V2 order) and topk_w_<l> [1,1,K] f32 (their renormalized "
          "weights, as fed to the moe op). For a routing check only; prefill_<N> and decode are built exactly "
          "as without the flag.");
ABSL_FLAG(bool, moe_fusion_guard, false,
          "Give the moe output a second consumer (x + moe + moe * 0) so the GPU delegate cannot fuse the residual "
          "ADD into the moe kernel: the GPU kernel (ml_drift_delegate/delegate/composite/moe_experts_kernel.cc:92 "
          "in LiteRT dcb657e8 + #11192) writes args.dst.Write(ucl::Convert<args.dst::type>(sum), ...) and a fused "
          "elementwise op then emits invalid WGSL (`var vec4 : vec4<f32><f32>(sum)_final;`). Numerically exact "
          "(+0).");

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
  int hidden = 0, layers = 0, heads = 0, kv_heads = 0, head_dim = 0, moe_inter = 0, experts = 0, top_k = 0,
      vocab = 0, norm_topk = 1;
  float eps = 0, rope_theta = 0;
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
  c.layers = i("layers");
  c.heads = i("heads");
  c.kv_heads = i("kv_heads");
  c.head_dim = i("head_dim");
  c.moe_inter = i("moe_inter");
  c.experts = i("experts");
  c.top_k = i("top_k");
  c.vocab = i("vocab");
  c.norm_topk = i("norm_topk");
  c.eps = f("eps");
  c.rope_theta = f("rope_theta");
  c.quant_experts = kv.at("expert_kind") != "fp32";
  return c;
}

// The packed weights, mapped read-only. Each constant tensor is created once and reused by every signature, so
// ModelFactory stores its buffer once.
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

  const T& Get(const std::string& name) {
    auto it = cache_.find(name);
    if (it != cache_.end()) return it->second;
    auto eit = entries_.find(name);
    ABSL_CHECK(eit != entries_.end()) << "missing weight " << name;
    const Entry& e = eit->second;
    auto buffer = std::make_shared<SpanCpuBuffer>(base_ + e.off, e.bytes);
    T t;
    if (e.kind == "fp32") {
      t = T({.name = name, .type = Type::kFP32, .shape = e.shape, .buffer = buffer});
    } else if (e.kind == "bw4") {
      // 2-D FC weight: ai-edge-quantizer's blockwise int4 (fp16 scales written by ModelFactory, no zero point).
      auto q = std::make_shared<BlockwiseQuantization>(Scales(e), std::vector<int64_t>{}, e.block, 0);
      t = T({.name = name, .type = Type::kI4, .shape = e.shape, .buffer = buffer, .quantization = q});
    } else if (e.kind == "pc8") {
      auto q = std::make_shared<PerChannelAffineQuantization>(
          Scales(e), std::vector<int64_t>(e.scale_count, 0), 0);
      t = T({.name = name, .type = Type::kI8, .shape = e.shape, .buffer = buffer, .quantization = q});
    } else if (e.kind == "xa8") {
      // Expert int8 with affine params along dim 0 (one scale per output row, shared by the experts).
      auto q = std::make_shared<PerChannelAffineQuantization>(
          Scales(e), std::vector<int64_t>(e.scale_count, 0), 0);
      t = T({.name = name, .type = Type::kI8, .shape = e.shape, .buffer = buffer, .quantization = q});
    } else if (e.kind == "xb8" || e.kind == "xb4") {
      // Expert int8 / int4, multi-axis blockwise: one scale per (output row, expert, block of B inputs), held in
      // the float32 tensor <name>_scales that is also the op's scale input.
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

// RMSNorm over the last axis: x * rsqrt(mean(x^2) + eps) * w.
T RmsNorm(const T& x, const T& w, float eps) {
  const int last = static_cast<int>(x.GetShape().size()) - 1;
  T ms = Mean(Mul(x, x), {last}, /*keep_dims=*/true);
  return Mul(Mul(x, Rsqrt(Add(ms, ConstF({eps}, {1})))), w);
}

// x: [1, n, seq, hd]; cos / sin: [1, 1, seq, hd]. HF rotate_half.
T Rope(const T& x, const T& cos, const T& sin) {
  const auto& s = x.GetShape();
  const int half = s[3] / 2;
  T x1 = Slice(x, {0, 0, 0, 0}, {s[0], s[1], s[2], half});
  T x2 = Slice(x, {0, 0, 0, half}, {s[0], s[1], s[2], half});
  std::vector<T> parts = {Mul(x2, ConstF({-1.0f}, {1})), x1};
  T rotated = Concatenation(absl::MakeSpan(parts), 3);
  return Add(Mul(x, cos), Mul(rotated, sin));
}

struct Signature {
  std::vector<TensorHandle> inputs, outputs;
};

Signature Build(const Config& c, Weights& w, const std::vector<uint8_t>& moe_options, int seq, int cache,
                bool decode, bool debug_topk = false) {
  const int D = c.hidden, H = c.heads, KV = c.kv_heads, hd = c.head_dim, g = H / KV, E = c.experts, K = c.top_k;
  T tokens({.name = "tokens", .type = Type::kI32, .shape = {1, seq}});
  T pos({.name = "input_pos", .type = Type::kI32, .shape = {seq}});
  T mask({.name = "mask", .type = Type::kFP32, .shape = {1, 1, seq, cache}});
  std::vector<T> kin, vin;
  for (int i = 0; i < c.layers; ++i) {
    kin.push_back(T({.name = absl::StrCat("kv_cache_k_", i), .type = Type::kFP32, .shape = {1, KV, cache, hd}}));
    vin.push_back(T({.name = absl::StrCat("kv_cache_v_", i), .type = Type::kFP32, .shape = {1, KV, hd, cache}}));
  }

  // RoPE tables from the positions, in fp32 as HF: freqs = pos * inv_freq, emb = [freqs, freqs].
  std::vector<float> inv(hd / 2);
  for (int j = 0; j < hd / 2; ++j) {
    inv[j] = 1.0f / std::pow(c.rope_theta, static_cast<float>(2 * j) / static_cast<float>(hd));
  }
  T freqs = Mul(Reshape(Cast(pos, Type::kFP32), {seq, 1}), ConstF(inv, {1, hd / 2}));
  std::vector<T> ff = {freqs, freqs};
  T angles = Concatenation(absl::MakeSpan(ff), 1);
  T cos = Reshape(Cos(angles), {1, 1, seq, hd});
  T sin = Reshape(Sin(angles), {1, 1, seq, hd});

  // Cache write position = input_pos[0] (the runtime's positions are contiguous within a call).
  T pos0 = Slice(pos, {0}, {1});
  T zero = ConstI({0}, {1});
  std::vector<T> ks = {zero, zero, pos0, zero};
  std::vector<T> vs = {zero, zero, zero, pos0};
  T k_start = Concatenation(absl::MakeSpan(ks), 0);
  T v_start = Concatenation(absl::MakeSpan(vs), 0);
  const T scale = ConstF({1.0f / std::sqrt(static_cast<float>(hd))}, {1});
  // Qwen3-MoE has no per-expert output scale: ones.
  const T expert_scale = ConstF(std::vector<float>(E, 1.0f), {1, 1, 1, E});

  T x = Reshape(EmbeddingLookup(Reshape(tokens, {seq}), w.Get("model.embed_tokens.weight")), {1, seq, D});
  std::vector<T> kout, vout, top_idx, top_wts;
  for (int l = 0; l < c.layers; ++l) {
    const std::string p = absl::StrCat("model.layers.", l, ".");
    T h = RmsNorm(x, w.Get(p + "input_layernorm.weight"), c.eps);
    T q = FullyConnected(h, w.Get(p + "self_attn.q_proj.weight"));
    T k = FullyConnected(h, w.Get(p + "self_attn.k_proj.weight"));
    T v = FullyConnected(h, w.Get(p + "self_attn.v_proj.weight"));
    q = RmsNorm(Reshape(q, {1, seq, H, hd}), w.Get(p + "self_attn.q_norm.weight"), c.eps);
    k = RmsNorm(Reshape(k, {1, seq, KV, hd}), w.Get(p + "self_attn.k_norm.weight"), c.eps);
    q = Rope(Transpose(q, {0, 2, 1, 3}), cos, sin);                // [1, H, seq, hd]
    k = Rope(Transpose(k, {0, 2, 1, 3}), cos, sin);                // [1, KV, seq, hd]
    v = Transpose(Reshape(v, {1, seq, KV, hd}), {0, 2, 3, 1});     // [1, KV, hd, seq]
    T kc = DynamicUpdateSlice(kin[l], k, k_start);
    T vc = DynamicUpdateSlice(vin[l], v, v_start);
    kc.SetName(absl::StrCat("kv_cache_k_", l));
    vc.SetName(absl::StrCat("kv_cache_v_", l));
    kout.push_back(kc);
    vout.push_back(vc);

    T scores = BatchMatMul(Reshape(q, {KV, g * seq, hd}), Reshape(kc, {KV, cache, hd}),
                           /*adj_x=*/false, /*adj_y=*/true);  // [KV, g*seq, cache]
    scores = Add(Mul(Reshape(scores, {1, H, seq, cache}), scale), mask);
    T probs = Reshape(Softmax(scores), {KV, g * seq, cache});
    T o = BatchMatMul(probs, Reshape(vc, {KV, hd, cache}), /*adj_x=*/false, /*adj_y=*/true);  // [KV, g*seq, hd]
    o = Reshape(Transpose(Reshape(o, {1, H, seq, hd}), {0, 2, 1, 3}), {1, seq, H * hd});
    x = Add(x, FullyConnected(o, w.Get(p + "self_attn.o_proj.weight")));

    // Sparse MoE block (HF Qwen3MoeSparseMoeBlock): softmax over all experts, top-k, renormalize.
    T h2 = RmsNorm(x, w.Get(p + "post_attention_layernorm.weight"), c.eps);
    T router = FullyConnected(h2, w.Get(p + "mlp.gate.weight"), ::litert::tensor::kActNone,
                              /*keep_num_dims=*/false);  // [seq, E]
    std::vector<T> top = TopK(Softmax(router), K);       // [seq, K] values, [seq, K] i32 indices
    T top_w = top[0];
    if (c.norm_topk) top_w = Div(top_w, Sum(top_w, std::vector<int>{1}, /*keep_dims=*/true));
    T top_w3 = Reshape(top_w, {1, seq, K});
    T top_i3 = Reshape(top[1], {1, seq, K});
    if (debug_topk) {
      top_idx.push_back(top_i3);
      top_wts.push_back(top_w3);
    }
    std::vector<T> ins = {h2, top_w3, top_i3};
    for (const char* m : {"gate", "up", "down"}) {
      const std::string n = absl::StrCat(p, "mlp.experts.", m);
      ins.push_back(w.Get(n));
      if (c.quant_experts) ins.push_back(w.Get(n + "_scales"));
    }
    ins.push_back(expert_scale);
    std::vector<T> moe = Custom(absl::MakeSpan(ins), "moe", moe_options, {{1, seq, D}}, {Type::kFP32});
    if (absl::GetFlag(FLAGS_moe_fusion_guard)) {
      x = Add(Add(x, moe[0]), Mul(moe[0], ConstF({0.0f}, {1})));
    } else {
      x = Add(x, moe[0]);
    }
  }

  Signature s;
  s.inputs = {tokens, pos, mask};
  for (int l = 0; l < c.layers; ++l) {
    s.inputs.push_back(kin[l]);
    s.inputs.push_back(vin[l]);
  }
  for (int l = 0; l < c.layers; ++l) {
    s.outputs.push_back(kout[l]);
    s.outputs.push_back(vout[l]);
  }
  if (decode) {
    T logits = FullyConnected(RmsNorm(x, w.Get("model.norm.weight"), c.eps), w.Get("lm_head.weight"));  // [1,seq,V]
    logits.SetName("logits");
    s.outputs.push_back(logits);
  }
  for (int l = 0; l < static_cast<int>(top_idx.size()); ++l) {
    top_idx[l].SetName(absl::StrCat("topk_idx_", l));
    top_wts[l].SetName(absl::StrCat("topk_w_", l));
    s.outputs.push_back(top_idx[l]);
    s.outputs.push_back(top_wts[l]);
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
  std::cout << "config: hidden " << c.hidden << " layers " << c.layers << " heads " << c.heads << "/" << c.kv_heads
            << " head_dim " << c.head_dim << " moe_inter " << c.moe_inter << " experts " << c.experts << " top_k "
            << c.top_k << " vocab " << c.vocab << " cache " << cache << " quant_experts " << c.quant_experts
            << " moe_options " << moe_options.size() << " B guard " << absl::GetFlag(FLAGS_moe_fusion_guard)
            << " debug_topk " << absl::GetFlag(FLAGS_debug_topk) << std::endl;
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
  if (absl::GetFlag(FLAGS_debug_topk)) {
    Signature dt = Build(c, w, moe_options, 1, cache, /*decode=*/true, /*debug_topk=*/true);
    st = factory.AddSignature(dt.inputs, dt.outputs, "decode_topk");
    ABSL_CHECK(st.ok()) << st;
    std::cout << "signature decode_topk (" << dt.outputs.size() << " outputs)" << std::endl;
  }
  st = factory.Save(out);
  ABSL_CHECK(st.ok()) << st;
  std::cout << "WROTE " << out << std::endl;
  return 0;
}
