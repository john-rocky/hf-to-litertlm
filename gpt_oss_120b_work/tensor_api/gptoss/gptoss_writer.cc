// Writes a gpt-oss decoder (the architecture of openai/gpt-oss-120b / -20b) as a .tflite with the LiteRT Tensor
// API, in the signature layout LiteRT-LM's stock executor runs (token ids in the graph):
//   prefill_<N> : tokens [1,N] i32, input_pos [N] i32, mask [1,1,N,C] f32,
//                 kv_cache_k_<i> [1,KV,C,hd] f32, kv_cache_v_<i> [1,KV,hd,C] f32 (V stored transposed)
//                 -> kv_cache_k_<i>, kv_cache_v_<i>
//   decode      : the same with N = 1 -> kv_cache_k_<i>, kv_cache_v_<i>, logits [1,1,V] f32
// Per layer (transformers' GptOssDecoderLayer): RMSNorm -> q / k / v projections with bias -> RoPE (cos / sin rows
// gathered by input_pos from the pack's tables = transformers' yarn GptOssRotaryEmbedding, attention scaling
// folded in) -> grouped-query attention over the cache: scores * hd^-0.5 + mask (+ the sliding band on sliding
// layers: key slots at or before input_pos - window get -15000, rows gathered from a [C, C] table) -> the layer's
// learned sink logit concatenated as one more column per head -> softmax -> the sink column dropped -> . V ->
// o_proj with bias -> residual; RMSNorm -> router FC [E, D] with bias (fp32) -> top-k on the logits -> softmax over
// the k values -> custom op "moe" on x' = [x, 1/S, 0 x 31] (the bias block, S = 2^10; tools/pack_gptoss.py folds the expert
// biases, alpha and the +1 of the up side into the SiLU form, per_expert_scale = 1/alpha) -> the first D outputs
// -> residual. The gpt-oss clamps (gate <= limit, |up| <= limit) are not in this form (the op has none).
// Weights come from tools/pack_gptoss.py (weights.bin, index.tsv, config.tsv, moe_options.bin), read through mmap;
// each constant tensor is created once and shared by every signature.
// --debug_topk adds a signature decode_topk (decode + per-layer topk_idx_<l> / topk_w_<l> outputs).

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
#include <optional>
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
          "Directory written by tools/pack_gptoss.py (weights.bin, index.tsv, config.tsv, moe_options.bin).");
ABSL_FLAG(std::string, out, "", "Output .tflite path.");
ABSL_FLAG(std::string, prefill, "128,1024", "Prefill lengths, comma separated (empty: decode only).");
ABSL_FLAG(int, cache, 4096, "KV cache length.");
ABSL_FLAG(bool, debug_topk, false,
          "Also write a signature decode_topk: the decode graph plus, per layer l, the outputs topk_idx_<l> [1,1,K] "
          "i32 and topk_w_<l> [1,1,K] f32 (the softmax over the top-k logits, as fed to the moe op).");
ABSL_FLAG(bool, moe_fusion_guard, false,
          "Give the moe output a second consumer (x + y + y * 0) so a GPU delegate cannot fuse the residual ADD "
          "into the moe kernel (a WebGPU code-generation defect when that add is fused). Exact (+0).");
ABSL_FLAG(float, band_value, -15000.0f,
          "Added to the mask on sliding layers for key slots at or before input_pos - sliding_window. Finite in "
          "fp16 together with the runtime's fp16 fill (-45824).");

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
  int hidden = 0, hidden_p = 0, inter = 0, inter_p = 0, layers = 0, heads = 0, kv_heads = 0, head_dim = 0,
      experts = 0, top_k = 0, vocab = 0, sliding_window = 0, rope_max = 0;
  float eps = 0, alpha = 0, bias_in = 0, expert_scale = 0;
  std::string layer_types;
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
  c.hidden_p = i("hidden_p");
  c.inter = i("inter");
  c.inter_p = i("inter_p");
  c.layers = i("layers");
  c.heads = i("heads");
  c.kv_heads = i("kv_heads");
  c.head_dim = i("head_dim");
  c.experts = i("experts");
  c.top_k = i("top_k");
  c.vocab = i("vocab");
  c.sliding_window = i("sliding_window");
  c.rope_max = i("rope_max");
  c.eps = f("eps");
  c.alpha = f("alpha");
  c.bias_in = f("bias_in");
  // per_expert_scale: 1/alpha when alpha is folded into the SiLU gate rows (packs before 20:15 have no key = that form),
  // 1 for the gelu_tanh form.
  c.expert_scale = kv.contains("expert_scale") ? f("expert_scale") : 1.0f / c.alpha;
  c.layer_types = kv.at("layer_types");
  ABSL_CHECK_EQ(static_cast<int>(c.layer_types.size()), c.layers);
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
      // 2-D FC weight: int4 blockwise (scales written by ModelFactory, no zero point).
      auto q = std::make_shared<BlockwiseQuantization>(Scales(e), std::vector<int64_t>{}, e.block, 0);
      t = T({.name = name, .type = Type::kI4, .shape = e.shape, .buffer = buffer, .quantization = q});
    } else if (e.kind == "pc8") {
      auto q = std::make_shared<PerChannelAffineQuantization>(
          Scales(e), std::vector<int64_t>(e.scale_count, 0), 0);
      t = T({.name = name, .type = Type::kI8, .shape = e.shape, .buffer = buffer, .quantization = q});
    } else if (e.kind == "xb8" || e.kind == "xb4") {
      // Expert int8 / int4, multi-axis blockwise: one scale per (output row, expert, block of B inputs), held in
      // the float32 tensor <name>_scales that is also the op's scale input (the multi-axis blockwise form).
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

// x: [1, n, seq, hd]; cos / sin: [1, 1, seq, hd] ([cos, cos] / [sin, sin] halves). HF rotate_half.
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

// Constants shared by every signature (created once).
struct Shared {
  T band_table;     // [C, C]: 0, or band_value where key slot j <= query position p - window
  T expert_scale;   // [1, 1, 1, E] = 1 / alpha (silu form) or 1 (gelu_tanh form)
};

Signature Build(const Config& c, Weights& w, Shared& sh, const std::vector<uint8_t>& moe_options, int seq,
                int cache, bool decode, bool debug_topk = false) {
  const int D = c.hidden, Dp = c.hidden_p, H = c.heads, KV = c.kv_heads, hd = c.head_dim, g = H / KV,
            K = c.top_k;
  T tokens({.name = "tokens", .type = Type::kI32, .shape = {1, seq}});
  T pos({.name = "input_pos", .type = Type::kI32, .shape = {seq}});
  T mask({.name = "mask", .type = Type::kFP32, .shape = {1, 1, seq, cache}});
  std::vector<T> kin, vin;
  for (int i = 0; i < c.layers; ++i) {
    kin.push_back(T({.name = absl::StrCat("kv_cache_k_", i), .type = Type::kFP32, .shape = {1, KV, cache, hd}}));
    vin.push_back(T({.name = absl::StrCat("kv_cache_v_", i), .type = Type::kFP32, .shape = {1, KV, hd, cache}}));
  }

  // RoPE rows and the sliding band for these positions.
  T cos = Reshape(EmbeddingLookup(pos, w.Get("rope.cos")), {1, 1, seq, hd});
  T sin = Reshape(EmbeddingLookup(pos, w.Get("rope.sin")), {1, 1, seq, hd});
  T local_mask = Add(mask, Reshape(EmbeddingLookup(pos, sh.band_table), {1, 1, seq, cache}));

  // Cache write position = input_pos[0] (the runtime's positions are contiguous within a call).
  T pos0 = Slice(pos, {0}, {1});
  T zero = ConstI({0}, {1});
  std::vector<T> ks = {zero, zero, pos0, zero};
  std::vector<T> vs = {zero, zero, zero, pos0};
  T k_start = Concatenation(absl::MakeSpan(ks), 0);
  T v_start = Concatenation(absl::MakeSpan(vs), 0);
  const T scale = ConstF({1.0f / std::sqrt(static_cast<float>(hd))}, {1});
  // x' = [x, 1/S, 0 x 31] (bias_in = 1/S = 2^-10): the bias block of the folded experts.
  std::vector<float> pad(static_cast<size_t>(seq) * (Dp - D), 0.0f);
  for (int t = 0; t < seq; ++t) pad[static_cast<size_t>(t) * (Dp - D)] = c.bias_in;
  const T bias_pad = ConstF(pad, {1, seq, Dp - D});

  T x = Reshape(EmbeddingLookup(Reshape(tokens, {seq}), w.Get("model.embed_tokens.weight")), {1, seq, D});
  std::vector<T> kout, vout, top_idx, top_wts;
  for (int l = 0; l < c.layers; ++l) {
    const std::string p = absl::StrCat("model.layers.", l, ".");
    const bool sliding = c.layer_types[l] == 'S';
    T h = RmsNorm(x, w.Get(p + "input_layernorm.weight"), c.eps);
    T q = FullyConnected(h, w.Get(p + "self_attn.q_proj.weight"), w.Get(p + "self_attn.q_proj.bias"));
    T k = FullyConnected(h, w.Get(p + "self_attn.k_proj.weight"), w.Get(p + "self_attn.k_proj.bias"));
    T v = FullyConnected(h, w.Get(p + "self_attn.v_proj.weight"), w.Get(p + "self_attn.v_proj.bias"));
    q = Rope(Transpose(Reshape(q, {1, seq, H, hd}), {0, 2, 1, 3}), cos, sin);    // [1, H, seq, hd]
    k = Rope(Transpose(Reshape(k, {1, seq, KV, hd}), {0, 2, 1, 3}), cos, sin);   // [1, KV, seq, hd]
    v = Transpose(Reshape(v, {1, seq, KV, hd}), {0, 2, 3, 1});                   // [1, KV, hd, seq]
    T kc = DynamicUpdateSlice(kin[l], k, k_start);
    T vc = DynamicUpdateSlice(vin[l], v, v_start);
    kc.SetName(absl::StrCat("kv_cache_k_", l));
    vc.SetName(absl::StrCat("kv_cache_v_", l));
    kout.push_back(kc);
    vout.push_back(vc);

    T scores = BatchMatMul(Reshape(q, {KV, g * seq, hd}), Reshape(kc, {KV, cache, hd}),
                           /*adj_x=*/false, /*adj_y=*/true);  // [KV, g*seq, cache]
    scores = Add(Mul(Reshape(scores, {1, H, seq, cache}), scale), sliding ? local_mask : mask);
    // The sink: one more logit per head in the softmax, dropped after it.
    T sink_col = Tile(Reshape(w.Get(p + "self_attn.sinks"), {1, H, 1, 1}), std::vector<int>{1, 1, seq, 1});
    std::vector<T> sc = {scores, sink_col};
    T probs = Softmax(Concatenation(absl::MakeSpan(sc), 3));  // [1, H, seq, cache + 1]
    probs = Reshape(Slice(probs, {0, 0, 0, 0}, {1, H, seq, cache}), {KV, g * seq, cache});
    T o = BatchMatMul(probs, Reshape(vc, {KV, hd, cache}), /*adj_x=*/false, /*adj_y=*/true);  // [KV, g*seq, hd]
    o = Reshape(Transpose(Reshape(o, {1, H, seq, hd}), {0, 2, 1, 3}), {1, seq, H * hd});
    x = Add(x, FullyConnected(o, w.Get(p + "self_attn.o_proj.weight"), w.Get(p + "self_attn.o_proj.bias")));

    // MoE block (HF GptOssMLP): top-k on the router logits, softmax over the k values.
    T h2 = RmsNorm(x, w.Get(p + "post_attention_layernorm.weight"), c.eps);
    T router = FullyConnected(h2, w.Get(p + "mlp.router.weight"), w.Get(p + "mlp.router.bias"),
                              ::litert::tensor::kActNone, /*keep_num_dims=*/false);  // [seq, E]
    std::vector<T> top = TopK(router, K);  // [seq, K] values, [seq, K] i32 indices
    T top_w3 = Reshape(Softmax(top[0]), {1, seq, K});
    T top_i3 = Reshape(top[1], {1, seq, K});
    if (debug_topk) {
      top_idx.push_back(top_i3);
      top_wts.push_back(top_w3);
    }
    std::vector<T> xp_parts = {h2, bias_pad};
    T xp = Concatenation(absl::MakeSpan(xp_parts), 2);  // [1, seq, Dp]
    std::vector<T> ins = {xp, top_w3, top_i3};
    for (const char* m : {"gate", "up", "down"}) {
      const std::string n = absl::StrCat(p, "mlp.experts.", m);
      ins.push_back(w.Get(n));
      ins.push_back(w.Get(n + "_scales"));
    }
    ins.push_back(sh.expert_scale);
    std::vector<T> moe = Custom(absl::MakeSpan(ins), "moe", moe_options, {{1, seq, Dp}}, {Type::kFP32});
    T y = Slice(moe[0], {0, 0, 0}, {1, seq, D});
    if (absl::GetFlag(FLAGS_moe_fusion_guard)) {
      x = Add(Add(x, y), Mul(y, ConstF({0.0f}, {1})));
    } else {
      x = Add(x, y);
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
    T logits = FullyConnected(RmsNorm(x, w.Get("model.norm.weight"), c.eps), w.Get("lm_head.weight"));
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
  ABSL_CHECK_LE(cache, c.rope_max) << "the pack's RoPE tables hold " << c.rope_max << " positions";
  const std::vector<uint8_t> moe_options = ReadBytes(dir + "/moe_options.bin");
  std::cout << "config: hidden " << c.hidden << "/" << c.hidden_p << " inter " << c.inter << "/" << c.inter_p
            << " layers " << c.layers << " (" << c.layer_types << ") heads " << c.heads << "/" << c.kv_heads
            << " head_dim " << c.head_dim << " experts " << c.experts << " top_k " << c.top_k << " vocab "
            << c.vocab << " window " << c.sliding_window << " cache " << cache << " alpha " << c.alpha << " expert_scale " << c.expert_scale
            << " moe_options " << moe_options.size() << " B guard " << absl::GetFlag(FLAGS_moe_fusion_guard)
            << " debug_topk " << absl::GetFlag(FLAGS_debug_topk) << std::endl;
  Weights w(dir);
  Shared sh;
  {
    const float band = absl::GetFlag(FLAGS_band_value);
    std::vector<float> table(static_cast<size_t>(cache) * cache, 0.0f);
    for (int p = 0; p < cache; ++p) {
      for (int j = 0; j <= p - c.sliding_window; ++j) table[static_cast<size_t>(p) * cache + j] = band;
    }
    sh.band_table = ConstF(table, {cache, cache});
    sh.expert_scale = ConstF(std::vector<float>(c.experts, c.expert_scale), {1, 1, 1, c.experts});
  }
  ModelFactory factory;
  for (absl::string_view p : absl::StrSplit(absl::GetFlag(FLAGS_prefill), ',', absl::SkipEmpty())) {
    int n = 0;
    ABSL_CHECK(absl::SimpleAtoi(p, &n));
    Signature s = Build(c, w, sh, moe_options, n, cache, /*decode=*/false);
    absl::Status st = factory.AddSignature(s.inputs, s.outputs, absl::StrCat("prefill_", n));
    ABSL_CHECK(st.ok()) << st;
    std::cout << "signature prefill_" << n << std::endl;
  }
  Signature d = Build(c, w, sh, moe_options, 1, cache, /*decode=*/true);
  absl::Status st = factory.AddSignature(d.inputs, d.outputs, "decode");
  ABSL_CHECK(st.ok()) << st;
  std::cout << "signature decode" << std::endl;
  if (absl::GetFlag(FLAGS_debug_topk)) {
    Signature dt = Build(c, w, sh, moe_options, 1, cache, /*decode=*/true, /*debug_topk=*/true);
    st = factory.AddSignature(dt.inputs, dt.outputs, "decode_topk");
    ABSL_CHECK(st.ok()) << st;
    std::cout << "signature decode_topk (" << dt.outputs.size() << " outputs)" << std::endl;
  }
  st = factory.Save(out);
  ABSL_CHECK(st.ok()) << st;
  std::cout << "WROTE " << out << std::endl;
  return 0;
}
