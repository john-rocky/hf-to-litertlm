// Writes the text decoder of Qwen/Qwen3.5-122B-A10B (Qwen3_5MoeForConditionalGeneration: Gated DeltaNet linear-attention
// layers interleaved with gated full-attention layers, every layer a 256-expert top-8 MoE with a gated shared expert) as a
// .tflite with the LiteRT Tensor API, in the signature layout LiteRT-LM's stock executor runs (token ids in the graph):
//   prefill_<N> : tokens [1,N] i32, input_pos [N] i32, mask [1,1,N,C] f32, and per packed layer i
//                 full:   kv_cache_k_<i> [1,KV,C,hd], kv_cache_v_<i> [1,KV,hd,C] (V stored transposed)
//                 linear: kv_cache_c_<i> [1,K-1,Cc] (conv state = the last K-1 conv inputs),
//                         kv_cache_r_<i> [1,Hv,Dk,Dv] (recurrent state)
//                 -> the same cache tensors
//   decode      : the same with N = 1 -> the caches and logits [1,1,V] f32
// LiteRT-LM (dab2e708 runtime/executor/litert_compiled_model_executor_utils.cc IsLinearAttentionStateName) treats
// kv_cache_c_* / kv_cache_r_* as TYPE_LINEAR_ATTENTION state buffers (no dynamic dim, cleared before a conversation).
//
// Linear-attention layer (HF Qwen3_5MoeGatedDeltaNet): RMSNorm -> FC qkv / FC z / FC b / FC a -> causal depthwise conv
// over [conv state; qkv] (4 taps, MUL / ADD) -> silu -> split q / k / v -> q, k l2-normalized (q times 1/sqrt(Dk)) ->
// beta = sigmoid(b), g = -exp(A_log) * softplus(a + dt_bias) -> CUSTOM gated_delta_update (6 inputs: q [1,Hk,N,Dk],
// k, v [1,Hv,N,Dv], beta [1,Hv,N], g [1,Hv,N], state; 2 outputs: out [1,Hv,N,Dv], new state; GQA inside the op, the
// v head h reads k head h / (Hv/Hk) = HF's repeat_interleave) -> gated RMSNorm (w * x_hat * silu(z)) -> FC out.
// Prefill pads: the executor writes input_pos 0 into pad slots (llm_litert_compiled_model_executor.cc 660-787), so a slot
// j is real iff input_pos[j] == input_pos[0] + j; pad slots get beta 0 and g 0 (the op's sentinels: no decay, no update)
// and the new conv state is rows [M, M+K-1) of [conv state; qkv] (M = number of real slots), picked by a one-hot
// BATCH_MATMUL. Token ids are not used for the pad test (id 0 is a real token).
// Full-attention layer (HF Qwen3_5MoeAttention): q_proj split per head into q / gate (pack), q / k RMSNorm per head,
// partial RoPE on the first rotary_dim dims (text positions: mrope's 3 rows are equal = 1-D RoPE), grouped-query
// attention over the cache (the runtime's mask hides unwritten slots), output times sigmoid(gate) -> FC o.
// MoE (HF Qwen3_5MoeSparseMoeBlock): router FC (fp32) -> softmax -> top-k -> divide by the top-k sum -> CUSTOM moe (SiLU
// experts, int4 blockwise in the form of LiteRT pull request #11309) + shared expert SwiGLU * sigmoid(FC gate).
// Every RMSNorm weight of the zero-centered Qwen3_5MoeRMSNorm arrives as 1 + w (pack). Weights from
// tapi_pack_q35moe.py (weights.bin, index.tsv, config.tsv, moe_options.bin, gdn_options.bin; or per-layer parts
// listed in parts.txt), read through mmap;
// each constant tensor is created once and shared by every signature.
// --moe_fusion_guard: x + moe + moe * 0 (the moe output gets a second consumer so the GPU delegate cannot fuse the next
// ADD into the moe kernel).
// --gdn plain (default): the Gated DeltaNet recurrence in BATCH_MATMUL / elementwise ops (GdnStep, GdnChunked), so every
// state is an ordinary input -> output pair: LiteRT-LM's GPU executor binds a graph without param_tensor with two
// state banks (kPingPong, dab2e708 runtime/executor/litert/state.cc) and refuses a state whose input the GPU
// delegate dropped, which is what the in-place GPU kernel of CUSTOM gated_delta_update (--gdn custom) does.
// --debug_hidden: adds a signature decode_dbg = decode plus hidden_<i> [1,1,D] (the residual stream after packed layer i)
// for the layer-wise check; prefill_<N> and decode are built the same with or without it.

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
#include "absl/strings/str_split.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "tensor/arithmetic.h"
#include "tensor/backends/tflite/arithmetic_tflite.h"
#include "tensor/backends/tflite/tflite_flatbuffer_conversion.h"
#include "tensor/buffer.h"
#include "tensor/datatypes.h"
#include "tensor/tensor.h"

ABSL_FLAG(std::string, weights_dir, "",
          "Directory written by tapi_pack_q35moe.py (weights.bin, index.tsv, config.tsv, moe_options.bin, "
          "gdn_options.bin).");
ABSL_FLAG(std::string, out, "", "Output .tflite path.");
ABSL_FLAG(int, split_embed_cols, 1,
          "Hold the fp32 embedding table as N tables of hidden / N columns each, looked up with the same ids and "
          "concatenated: the ML Drift that LiteRT-LM dab2e708 builds keeps a GPU buffer's byte size in an int "
          "(BufferDescriptor::size), so the one table [248320, 3072] = 3,051,356,160 B became a negative size and the "
          "WebGPU engine crashed while it was created; two tables of 1,525,678,080 B stay under 2^31.");
ABSL_FLAG(bool, external_embedder, false,
          "The prefill / decode signatures take 'embeddings' [1, N, hidden] float32 in place of 'tokens', and the graph "
          "holds no embedding table: LiteRT-LM's external-embedder form (a TF_LITE_EMBEDDER section does the lookup on "
          "the CPU; the executor binds the input named 'embeddings', runtime/executor/litert_compiled_model_executor_utils.cc "
          "kEmbeddingNames at dab2e708). The fp32 table [248320, 3072] is 3.05 GB, past what the WebGPU delegate took. "
          "Write the embedder with --embedder_out.");
ABSL_FLAG(std::string, embedder_out, "",
          "Also write the embedder model to this path: one signature 'embedder', token_ids [1, 1] int32 -> embeddings "
          "[1, 1, hidden] float32 = EMBEDDING_LOOKUP of the pack's embed table (LiteRT-LM's EmbeddingLookupText runs its "
          "single signature once per token, and passes a negative id's default vector itself).");
ABSL_FLAG(bool, embedder_only, false,
          "Write only the --embedder_out model and stop (no --out): the in-place bundle's plan needs the embedder's size "
          "before the decoder graph is written at its offset (make_bundle.py --plan-offset --embedder).");
ABSL_FLAG(uint64_t, out_offset, 0,
          "Write the model into --out from this byte on (ModelFactory::SaveAt, tensor_api/patches/save_at_offset.diff): "
          "the TFLite section's offset in a .litertlm bundle (make_bundle.py --plan-offset), whose header, "
          "metadata and tokenizer sections make_bundle.py --head-only then writes in front. 0 = a plain .tflite.");
ABSL_FLAG(std::string, prefill, "128", "Prefill lengths, comma separated (empty: decode only).");
ABSL_FLAG(int, cache, 2048, "KV cache length (full-attention layers).");
ABSL_FLAG(bool, moe_fusion_guard, true, "x + moe + moe * 0 (see the header).");
ABSL_FLAG(bool, debug_hidden, false, "Also write a signature decode_dbg with hidden_<i> outputs (see the header).");
ABSL_FLAG(std::string, gdn, "plain",
          "Gated DeltaNet recurrence: plain = BATCH_MATMUL and elementwise ops (decode: the recurrent step; prefill: the "
          "chunked form, see GdnChunked) or custom = one CUSTOM gated_delta_update per layer (its GPU kernel reads and "
          "writes the state in place, which LiteRT-LM binds only for a graph with a param_tensor input).");
ABSL_FLAG(bool, debug_prefill, false, "Diagnostics: the prefill signatures also output dbgp_* (valid, m, onehot) and, per "
          "linear layer, the conv input rows and the new conv state.");
ABSL_FLAG(int, gdn_chunk, 64, "Chunk length of the plain prefill form (min(chunk, N) must divide N).");

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
  int hidden = 0, layers = 0, heads = 0, kv_heads = 0, head_dim = 0, rotary_dim = 0, lin_k_heads = 0,
      lin_v_heads = 0, lin_k_dim = 0, lin_v_dim = 0, conv_k = 0, moe_inter = 0, experts = 0, top_k = 0,
      shared_inter = 0, vocab = 0;
  float eps = 0, rope_theta = 0;
  bool quant_experts = false;
  std::vector<bool> is_linear;
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
  c.rotary_dim = i("rotary_dim");
  c.lin_k_heads = i("lin_k_heads");
  c.lin_v_heads = i("lin_v_heads");
  c.lin_k_dim = i("lin_k_dim");
  c.lin_v_dim = i("lin_v_dim");
  c.conv_k = i("conv_k");
  c.moe_inter = i("moe_inter");
  c.experts = i("experts");
  c.top_k = i("top_k");
  c.shared_inter = i("shared_inter");
  c.vocab = i("vocab");
  c.eps = f("eps");
  c.rope_theta = f("rope_theta");
  c.quant_experts = kv.at("expert_kind") != "fp32";
  for (absl::string_view t : absl::StrSplit(kv.at("layer_types"), ',')) {
    ABSL_CHECK(t == "linear" || t == "full") << t;
    c.is_linear.push_back(t == "linear");
  }
  ABSL_CHECK_EQ(static_cast<int>(c.is_linear.size()), c.layers);
  return c;
}

// The packed weights, mapped read-only. Each constant tensor is created once and reused by every signature, so
// ModelFactory stores its buffer once. Same index.tsv format and kinds as qwen3_30b_a3b_work's qwen3moe writer.
// A pack is one weights.bin + index.tsv, or (dir/parts.txt, tapi_pack_q35moe.py --part) one per listed part
// directory; every entry remembers its part's mapping.
class Weights {
 public:
  explicit Weights(const std::string& dir) {
    std::vector<std::string> parts;
    struct stat pst;
    if (stat((dir + "/parts.txt").c_str(), &pst) == 0) {
      for (const std::string& line : ReadLines(dir + "/parts.txt")) {
        if (!line.empty()) parts.push_back(dir + "/" + line);
      }
      ABSL_CHECK(!parts.empty()) << dir << "/parts.txt lists no part";
    } else {
      parts.push_back(dir);
    }
    for (const std::string& part : parts) MapPart(part);
  }

  const T& Get(const std::string& name) {
    auto it = cache_.find(name);
    if (it != cache_.end()) return it->second;
    auto eit = entries_.find(name);
    ABSL_CHECK(eit != entries_.end()) << "missing weight " << name;
    const Entry& e = eit->second;
    auto buffer = std::make_shared<SpanCpuBuffer>(maps_[e.part] + e.off, e.bytes);
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
      // Expert int8 / int4, multi-axis blockwise: one scale per (output row, expert, block of B inputs), held in the
      // float32 tensor <name>_scales that is also the op's scale input (needs tensor_api/patches/
      // blockwise_named_scales.diff in the LiteRT checkout).
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

  // The fp32 entry `name` [rows, cols] as n constant tensors [rows, cols / n] (column blocks, copied out of the mapped
  // pack once and shared by every signature).
  const std::vector<T>& SplitCols(const std::string& name, int n) {
    auto it = split_cache_.find(name);
    if (it != split_cache_.end()) return it->second;
    auto eit = entries_.find(name);
    ABSL_CHECK(eit != entries_.end()) << "missing weight " << name;
    const Entry& e = eit->second;
    ABSL_CHECK(e.kind == "fp32" && e.shape.size() == 2 && n > 1 && e.shape[1] % n == 0)
        << name << ": --split_embed_cols needs an fp32 [rows, cols] entry with cols divisible by " << n;
    const size_t rows = e.shape[0], cols = e.shape[1], c = cols / n;
    const float* src = reinterpret_cast<const float*>(maps_[e.part] + e.off);
    std::vector<T> parts;
    for (int k = 0; k < n; ++k) {
      auto buf = OwningCpuBuffer::Allocate<Type::kFP32>(rows * c);
      float* dst = reinterpret_cast<float*>(buf->begin());
      for (size_t r = 0; r < rows; ++r) std::memcpy(dst + r * c, src + r * cols + k * c, c * sizeof(float));
      parts.push_back(T({.name = absl::StrCat(name, "_c", k),
                         .type = Type::kFP32,
                         .shape = {static_cast<int>(rows), static_cast<int>(c)},
                         .buffer = buf}));
    }
    return split_cache_.emplace(name, std::move(parts)).first->second;
  }

 private:
  struct Entry {
    int part = 0;
    std::string kind;
    std::vector<int> shape;
    uint64_t off = 0, bytes = 0, scale_off = 0, scale_count = 0;
    int block = 0;
  };

  void MapPart(const std::string& part_dir) {
    const std::string bin = part_dir + "/weights.bin";
    int fd = open(bin.c_str(), O_RDONLY);
    ABSL_CHECK_GE(fd, 0) << "cannot open " << bin;
    struct stat st;
    ABSL_CHECK_EQ(fstat(fd, &st), 0);
    const size_t size = static_cast<size_t>(st.st_size);
    void* p = mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
    ABSL_CHECK(p != MAP_FAILED) << "mmap failed for " << bin;
    close(fd);
    const int part = static_cast<int>(maps_.size());
    maps_.push_back(static_cast<const std::byte*>(p));
    for (const std::string& line : ReadLines(part_dir + "/index.tsv")) {
      std::vector<std::string> f = absl::StrSplit(line, '\t');
      ABSL_CHECK_EQ(f.size(), 8u) << line;
      Entry e;
      e.part = part;
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
      ABSL_CHECK_LE(e.off + e.bytes, size) << f[0];
      ABSL_CHECK_LE(e.scale_off + e.scale_count * sizeof(float), size) << f[0];
      ABSL_CHECK(entries_.emplace(f[0], e).second) << f[0] << " is in two parts";
    }
  }

  std::vector<float> Scales(const Entry& e) const {
    std::vector<float> s(e.scale_count);
    std::memcpy(s.data(), maps_[e.part] + e.scale_off, e.scale_count * sizeof(float));
    return s;
  }

  std::vector<const std::byte*> maps_;
  absl::flat_hash_map<std::string, Entry> entries_;
  absl::flat_hash_map<std::string, T> cache_;
  absl::flat_hash_map<std::string, std::vector<T>> split_cache_;
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

int Last(const T& x) { return static_cast<int>(x.GetShape().size()) - 1; }

// x * rsqrt(mean(x^2) + eps) * w over the last axis (w = 1 + w_hf for the zero-centered norms, from the pack).
T RmsNorm(const T& x, const T& w, float eps) {
  T ms = Mean(Mul(x, x), {Last(x)}, /*keep_dims=*/true);
  return Mul(Mul(x, Rsqrt(Add(ms, ConstF({eps}, {1})))), w);
}

T Silu(const T& x) { return Mul(x, Logistic(x)); }

// HF l2norm: x * rsqrt(sum(x^2) + eps) over the last axis, times scale.
T L2Norm(const T& x, float eps, float scale) {
  T ss = Sum(Mul(x, x), std::vector<int>{Last(x)}, /*keep_dims=*/true);
  T y = Mul(x, Rsqrt(Add(ss, ConstF({eps}, {1}))));
  return scale == 1.0f ? y : Mul(y, ConstF({scale}, {1}));
}

// softplus(x) = max(x, 0) + log(1 + exp(-|x|)) (HF F.softplus with threshold 20 differs by < 2.1e-9 above 20).
T Softplus(const T& x) {
  return Add(Relu(x), Log(Add(Exp(Neg(Abs(x))), ConstF({1.0f}, {1}))));
}

// x: [1, n, seq, hd]; cos / sin: [1, 1, seq, rd] with rd <= hd. HF apply_rotary_pos_emb on the first rd dims.
T PartialRope(const T& x, const T& cos, const T& sin, int rd) {
  const auto& s = x.GetShape();
  const int half = rd / 2;
  T x1 = Slice(x, {0, 0, 0, 0}, {s[0], s[1], s[2], half});
  T x2 = Slice(x, {0, 0, 0, half}, {s[0], s[1], s[2], half});
  std::vector<T> parts = {Mul(x2, ConstF({-1.0f}, {1})), x1};
  T rotated = Concatenation(absl::MakeSpan(parts), 3);
  T xr = Slice(x, {0, 0, 0, 0}, {s[0], s[1], s[2], rd});
  T roped = Add(Mul(xr, cos), Mul(rotated, sin));
  if (rd == s[3]) return roped;
  T pass = Slice(x, {0, 0, 0, rd}, {s[0], s[1], s[2], s[3] - rd});
  std::vector<T> both = {roped, pass};
  return Concatenation(absl::MakeSpan(both), 3);
}

// [1, Hk, N, D] -> [Hk * rep, N, D]; head h reads source head h / rep (HF repeat_interleave on the head axis).
T ExpandHeads(const T& x, int hk, int rep, int n, int d) {
  T r = Reshape(x, {hk, 1, n * d});
  if (rep == 1) return Reshape(r, {hk, n, d});
  std::vector<T> parts(rep, r);
  return Reshape(Concatenation(absl::MakeSpan(parts), 1), {hk * rep, n, d});
}

// One gated delta step (decode), HF torch_recurrent_gated_delta_rule: q, k [H, 1, Dk] (q already l2-normalized and
// scaled), v [H, 1, Dv], beta, g [H, 1, 1], state [H, Dk, Dv] -> out [H, 1, Dv], new state.
std::pair<T, T> GdnStep(const T& q, const T& k, const T& v, const T& beta, const T& g, const T& state) {
  T sd = Mul(state, Exp(g));
  T kv = BatchMatMul(k, sd);                                  // [H, 1, Dv]
  T delta = Mul(Sub(v, kv), beta);                            // [H, 1, Dv]
  T sn = Add(sd, BatchMatMul(k, delta, /*adj_x=*/true));      // [H, Dk, 1] @ [H, 1, Dv]
  return {BatchMatMul(q, sn), sn};
}

// The chunked gated delta rule (HF torch_chunk_gated_delta_rule) for a prefill of N = n * C slots: q, k [H, N, Dk],
// v [H, N, Dv], beta, g [H, N] (pad slots carry beta 0 and g 0: they neither decay nor update the state and do not
// touch the real slots before them), state [H, Dk, Dv] -> out [H, N, Dv], new state. Inside a chunk everything is
// batched over (chunk, head) = B rows, chunk-major; only the state carry runs chunk by chunk.
// T = (I + A)^-1 (A = strictly lower (k_beta k^T) * decay, what HF's row loop computes) by block recursion:
// T_2b = T_b - T_b (A * M_b) T_b, M_b = the lower-left b x b block of every 2b x 2b diagonal block, T_1 = I. Unlike a
// Neumann product it never forms high powers of A (whose entries grow combinatorially for near-equal keys).
std::pair<T, T> GdnChunked(const T& q, const T& k, const T& v, const T& beta, const T& g, const T& state, int H,
                           int N, int Dk, int Dv, int C) {
  const int n = N / C, B = n * H;
  auto chunks = [&](const T& x, int d) {  // [H, N, d] -> [n * H, C, d], row b = chunk * H + head
    return Reshape(Transpose(Reshape(x, {H, n, C, d}), {1, 0, 2, 3}), {B, C, d});
  };
  std::vector<float> tri(C * C), eye(C * C);
  for (int i = 0; i < C; ++i) {
    for (int j = 0; j < C; ++j) {
      tri[i * C + j] = j <= i ? 1.0f : 0.0f;
      eye[i * C + j] = i == j ? 1.0f : 0.0f;
    }
  }
  T qc = chunks(q, Dk), kc = chunks(k, Dk), vc = chunks(v, Dv);
  T bc = chunks(Reshape(beta, {H, N, 1}), 1);                                    // [B, C, 1]
  T g2 = Reshape(chunks(Reshape(g, {H, N, 1}), 1), {B, C});
  T gcs = FullyConnected(g2, ConstF(tri, {C, C}));                               // within-chunk cumsum [B, C]
  T gcol = Reshape(gcs, {B, C, 1});
  T glast = Slice(gcs, {0, C - 1}, {B, 1});                                       // [B, 1]
  // diff[i][j] = gc[i] - gc[j] as a K = 2 BATCH_MATMUL ([gc_i, 1] @ [1; -gc_j]); exp of min(diff, 0) never overflows.
  T ones = Add(Mul(gcol, ConstF({0.0f}, {1})), ConstF({1.0f}, {1}));             // [B, C, 1]
  std::vector<T> lp = {gcol, ones};
  std::vector<T> rp = {Reshape(ones, {B, 1, C}), Neg(Reshape(gcs, {B, 1, C}))};
  T diff = BatchMatMul(Concatenation(absl::MakeSpan(lp), 2), Concatenation(absl::MakeSpan(rp), 1));  // [B, C, C]
  T decay = Mul(Reshape(Exp(Minimum(diff, ConstF({0.0f}, {1}))), {1, B, C, C}), ConstF(tri, {1, 1, C, C}));
  T kb = Mul(kc, bc), vb = Mul(vc, bc);
  T kk = Reshape(BatchMatMul(kb, kc, /*adj_x=*/false, /*adj_y=*/true), {1, B, C, C});
  std::vector<float> strict(C * C);
  for (int i = 0; i < C; ++i) {
    for (int j = 0; j < C; ++j) strict[i * C + j] = j < i ? 1.0f : 0.0f;
  }
  T a = Mul(Mul(kk, decay), ConstF(strict, {1, 1, C, C}));                       // [1, B, C, C]
  T tm;  // T_b, [1, B, C, C]
  for (int b = 1; b < C; b *= 2) {
    std::vector<float> mb(C * C);
    for (int i = 0; i < C; ++i) {
      for (int j = 0; j < C; ++j) {
        mb[i * C + j] = (i / (2 * b) == j / (2 * b) && i % (2 * b) >= b && j % (2 * b) < b) ? 1.0f : 0.0f;
      }
    }
    T off = Mul(a, ConstF(mb, {1, 1, C, C}));
    if (b == 1) {
      tm = Sub(ConstF(eye, {1, 1, C, C}), off);  // T_1 = I: T_2 = I - A * M_1
    } else {
      T t3 = Reshape(tm, {B, C, C});
      tm = Sub(tm, Reshape(BatchMatMul(BatchMatMul(t3, Reshape(off, {B, C, C})), t3), {1, B, C, C}));
    }
  }
  T t3 = Reshape(tm, {B, C, C});
  T egc = Exp(gcol);                                                               // [B, C, 1]
  T value = BatchMatMul(t3, vb);                                                   // [B, C, Dv]
  T kcd = BatchMatMul(t3, Mul(kb, egc));                                           // [B, C, Dk]
  T aqk = Reshape(Mul(Reshape(BatchMatMul(qc, kc, false, true), {1, B, C, C}), decay), {B, C, C});
  T qg = Mul(qc, egc);                                                             // [B, C, Dk]
  T kd = Mul(kc, Reshape(Exp(Sub(glast, gcs)), {B, C, 1}));                        // [B, C, Dk]
  T dl = Reshape(Exp(glast), {B, 1, 1});
  T st = state;
  std::vector<T> outs;
  for (int i = 0; i < n; ++i) {
    auto row = [&](const T& x, int d1, int d2) { return Slice(x, {i * H, 0, 0}, {H, d1, d2}); };
    T vn = Sub(row(value, C, Dv), BatchMatMul(row(kcd, C, Dk), st));             // [H, C, Dv]
    outs.push_back(Add(BatchMatMul(row(qg, C, Dk), st), BatchMatMul(row(aqk, C, C), vn)));
    st = Add(Mul(st, row(dl, 1, 1)), BatchMatMul(row(kd, C, Dk), vn, /*adj_x=*/true));
  }
  T out = n == 1 ? outs[0] : Concatenation(absl::MakeSpan(outs), 1);              // [H, N, Dv]
  return {out, st};
}

struct Signature {
  std::vector<TensorHandle> inputs, outputs;
};

Signature Build(const Config& c, Weights& w, const std::vector<uint8_t>& moe_options,
                const std::vector<uint8_t>& gdn_options, int seq, int cache, bool decode, bool debug_hidden) {
  const int D = c.hidden, H = c.heads, KV = c.kv_heads, hd = c.head_dim, g = H / KV, E = c.experts, K = c.top_k;
  const int Hk = c.lin_k_heads, Hv = c.lin_v_heads, Dk = c.lin_k_dim, Dv = c.lin_v_dim, CK = c.conv_k;
  const int Kd = Hk * Dk, Vd = Hv * Dv, Cc = 2 * Kd + Vd;
  const int rd = c.rotary_dim;
  const std::string gdn_mode = absl::GetFlag(FLAGS_gdn);
  ABSL_CHECK(gdn_mode == "plain" || gdn_mode == "custom") << "--gdn " << gdn_mode;
  const bool gdn_plain = gdn_mode == "plain";
  const bool ext_emb = absl::GetFlag(FLAGS_external_embedder);
  // the token input: ids (the table in the graph) or, --external_embedder, the embeddings the runtime looked up
  T tokens = ext_emb ? T({.name = "embeddings", .type = Type::kFP32, .shape = {1, seq, D}})
                     : T({.name = "tokens", .type = Type::kI32, .shape = {1, seq}});
  T pos({.name = "input_pos", .type = Type::kI32, .shape = {seq}});
  T mask({.name = "mask", .type = Type::kFP32, .shape = {1, 1, seq, cache}});
  std::vector<T> sin_state(c.layers), sout(c.layers), rin(c.layers), rout(c.layers);  // c / k and r / v per layer
  for (int i = 0; i < c.layers; ++i) {
    if (c.is_linear[i]) {
      sin_state[i] = T({.name = absl::StrCat("kv_cache_c_", i), .type = Type::kFP32, .shape = {1, CK - 1, Cc}});
      rin[i] = T({.name = absl::StrCat("kv_cache_r_", i), .type = Type::kFP32, .shape = {1, Hv, Dk, Dv}});
    } else {
      sin_state[i] = T({.name = absl::StrCat("kv_cache_k_", i), .type = Type::kFP32, .shape = {1, KV, cache, hd}});
      rin[i] = T({.name = absl::StrCat("kv_cache_v_", i), .type = Type::kFP32, .shape = {1, KV, hd, cache}});
    }
  }

  // RoPE tables in fp32 as HF: freqs = pos * inv_freq (inv_freq over the rotary dims), emb = [freqs, freqs].
  std::vector<float> inv(rd / 2);
  for (int j = 0; j < rd / 2; ++j) {
    inv[j] = 1.0f / std::pow(c.rope_theta, static_cast<float>(2 * j) / static_cast<float>(rd));
  }
  T freqs = Mul(Reshape(Cast(pos, Type::kFP32), {seq, 1}), ConstF(inv, {1, rd / 2}));
  std::vector<T> ff = {freqs, freqs};
  T angles = Concatenation(absl::MakeSpan(ff), 1);
  T cos = Reshape(Cos(angles), {1, 1, seq, rd});
  T sin = Reshape(Sin(angles), {1, 1, seq, rd});

  // KV write position = input_pos[0] (contiguous positions within a call).
  T pos0 = Slice(pos, {0}, {1});
  T zero = ConstI({0}, {1});
  std::vector<T> ks = {zero, zero, pos0, zero};
  std::vector<T> vs = {zero, zero, zero, pos0};
  T k_start = Concatenation(absl::MakeSpan(ks), 0);
  T v_start = Concatenation(absl::MakeSpan(vs), 0);
  const T scale = ConstF({1.0f / std::sqrt(static_cast<float>(hd))}, {1});
  const T expert_scale = ConstF(std::vector<float>(E, 1.0f), {1, 1, 1, E});

  std::vector<std::pair<std::string, T>> dbg;  // debug outputs (decode_dbg; prefill with --debug_prefill)
  const bool debug_prefill = absl::GetFlag(FLAGS_debug_prefill) && seq > 1;
  // Real slots of a prefill call: input_pos[j] == input_pos[0] + j (pads carry 0). valid [1, seq, 1] f32 0 / 1;
  // conv-state picker: one-hot rows (M + r) of [K-1 + seq] where M = number of real slots.
  T valid, conv_pick;
  if (seq > 1) {
    std::vector<int32_t> iota(seq);
    for (int j = 0; j < seq; ++j) iota[j] = j;
    T expect = Add(pos0, ConstI(iota, {seq}));
    valid = Reshape(Cast(Equal(pos, expect), Type::kFP32), {1, seq, 1});
    T m = Sum(valid, std::vector<int>{1}, /*keep_dims=*/false);  // [1, 1]
    // onehot[r][j] = 1 iff j - r == M, as relu(1 - |(j - r) - M|) (integer-valued floats), built at rank 3 with a
    // leading 1. The WebGPU delegate (LiteRT-LM dab2e708) read a rank-2 constant [K-1, K-1+seq] in an elementwise op as its first
    // row only (tiny check 2026-11-09: every one-hot row picked column M, so the conv state was 3 copies of row M; the
    // CPU was exact); Equal([1, K-1+seq], [K-1, 1]) went the same way.
    std::vector<float> jr((CK - 1) * (CK - 1 + seq));
    for (int r = 0; r < CK - 1; ++r) {
      for (int j = 0; j < CK - 1 + seq; ++j) jr[r * (CK - 1 + seq) + j] = static_cast<float>(j - r);
    }
    conv_pick = Relu(Sub(ConstF({1.0f}, {1}),
                         Abs(Sub(ConstF(jr, {1, CK - 1, CK - 1 + seq}), Reshape(m, {1, 1, 1})))));  // [1, K-1, K-1+seq]
    if (debug_prefill) {
      dbg.emplace_back("dbgp_valid", valid);
      dbg.emplace_back("dbgp_m", m);
      dbg.emplace_back("dbgp_onehot", conv_pick);
    }
  }

  const int split = absl::GetFlag(FLAGS_split_embed_cols);
  T x;
  if (ext_emb) {
    x = tokens;
  } else if (split > 1) {
    T ids = Reshape(tokens, {seq});
    std::vector<T> cols;
    for (const T& part : w.SplitCols("embed", split)) {
      // [1, seq, D / split], rank 3 like the hidden states (the WebGPU delegate read a rank-2 constant by its first row
      // only), joined on the last axis
      cols.push_back(Reshape(EmbeddingLookup(ids, part), {1, seq, D / split}));
    }
    x = Concatenation(absl::MakeSpan(cols), 2);
  } else {
    x = Reshape(EmbeddingLookup(Reshape(tokens, {seq}), w.Get("embed")), {1, seq, D});
  }
  std::vector<T> hidden;
  auto tap = [&](const std::string& name, const T& t) {
    if (debug_hidden) dbg.emplace_back(name, t);
  };
  for (int l = 0; l < c.layers; ++l) {
    const std::string p = absl::StrCat("L", l, ".");
    T h = RmsNorm(x, w.Get(p + "in_norm"), c.eps);
    if (c.is_linear[l]) {
      T qkv = FullyConnected(h, w.Get(p + "qkv"));  // [1, seq, Cc]
      T z = FullyConnected(h, w.Get(p + "z"));      // [1, seq, Vd]
      T b = FullyConnected(h, w.Get(p + "b"));      // [1, seq, Hv]
      T a = FullyConnected(h, w.Get(p + "a"));      // [1, seq, Hv]
      // Causal depthwise conv over [state; qkv]: out[t] = sum_j w_j * cat[t + j].
      std::vector<T> cat_parts = {sin_state[l], qkv};
      T cat = Concatenation(absl::MakeSpan(cat_parts), 1);  // [1, K-1+seq, Cc]
      T conv;
      for (int j = 0; j < CK; ++j) {
        T term = Mul(Slice(cat, {0, j, 0}, {1, seq, Cc}), w.Get(absl::StrCat(p, "conv", j)));
        conv = j == 0 ? term : Add(conv, term);
      }
      conv = Silu(conv);
      T new_conv = seq > 1 ? BatchMatMul(conv_pick, cat) : Slice(cat, {0, 1, 0}, {1, CK - 1, Cc});
      if (debug_prefill) dbg.emplace_back(absl::StrCat("dbgp_cat_", l), cat);
      new_conv.SetName(absl::StrCat("kv_cache_c_", l));
      sout[l] = new_conv;
      T q = Reshape(Slice(conv, {0, 0, 0}, {1, seq, Kd}), {1, seq, Hk, Dk});
      T k = Reshape(Slice(conv, {0, 0, Kd}, {1, seq, Kd}), {1, seq, Hk, Dk});
      T v = Reshape(Slice(conv, {0, 0, 2 * Kd}, {1, seq, Vd}), {1, seq, Hv, Dv});
      q = Transpose(L2Norm(q, 1e-6f, 1.0f / std::sqrt(static_cast<float>(Dk))), {0, 2, 1, 3});  // [1, Hk, seq, Dk]
      k = Transpose(L2Norm(k, 1e-6f, 1.0f), {0, 2, 1, 3});
      v = Transpose(v, {0, 2, 1, 3});                                                          // [1, Hv, seq, Dv]
      T beta = Logistic(b);                                                                    // [1, seq, Hv]
      T gg = Mul(w.Get(p + "negA"), Softplus(Add(a, w.Get(p + "dt_bias"))));
      if (seq > 1) {
        beta = Mul(beta, valid);
        gg = Mul(gg, valid);
      }
      beta = Transpose(beta, {0, 2, 1});  // [1, Hv, seq]
      gg = Transpose(gg, {0, 2, 1});
      T core;
      if (gdn_plain) {
        const int rep = Hv / Hk;
        T qe = ExpandHeads(q, Hk, rep, seq, Dk), ke = ExpandHeads(k, Hk, rep, seq, Dk);
        T ve = Reshape(v, {Hv, seq, Dv});
        T s0 = Reshape(rin[l], {Hv, Dk, Dv});
        std::pair<T, T> r;
        if (seq == 1) {
          r = GdnStep(qe, ke, ve, Reshape(beta, {Hv, 1, 1}), Reshape(gg, {Hv, 1, 1}), s0);
        } else {
          const int chunk = std::min(absl::GetFlag(FLAGS_gdn_chunk), seq);
          ABSL_CHECK_EQ(seq % chunk, 0) << "prefill " << seq << " is not a multiple of the GDN chunk " << chunk;
          r = GdnChunked(qe, ke, ve, Reshape(beta, {Hv, seq}), Reshape(gg, {Hv, seq}), s0, Hv, seq, Dk, Dv, chunk);
        }
        T rnew = Reshape(r.second, {1, Hv, Dk, Dv});
        rnew.SetName(absl::StrCat("kv_cache_r_", l));
        rout[l] = rnew;
        core = Transpose(Reshape(r.first, {1, Hv, seq, Dv}), {0, 2, 1, 3});  // [1, seq, Hv, Dv]
      } else {
        std::vector<T> gin = {q, k, v, beta, gg, rin[l]};
        std::vector<T> gout = Custom(absl::MakeSpan(gin), "gated_delta_update", gdn_options,
                                     {{1, Hv, seq, Dv}, {1, Hv, Dk, Dv}}, {Type::kFP32, Type::kFP32});
        gout[1].SetName(absl::StrCat("kv_cache_r_", l));
        rout[l] = gout[1];
        core = Transpose(gout[0], {0, 2, 1, 3});  // [1, seq, Hv, Dv]
      }
      T gated = Mul(RmsNorm(core, w.Get(p + "gnorm"), c.eps), Silu(Reshape(z, {1, seq, Hv, Dv})));
      x = Add(x, FullyConnected(Reshape(gated, {1, seq, Vd}), w.Get(p + "out")));
    } else {
      T q = FullyConnected(h, w.Get(p + "q"));
      T gate = FullyConnected(h, w.Get(p + "qgate"));
      T k = FullyConnected(h, w.Get(p + "k"));
      T v = FullyConnected(h, w.Get(p + "v"));
      q = RmsNorm(Reshape(q, {1, seq, H, hd}), w.Get(p + "qn"), c.eps);
      k = RmsNorm(Reshape(k, {1, seq, KV, hd}), w.Get(p + "kn"), c.eps);
      tap(absl::StrCat("dbg", l, "_qn"), q);
      tap(absl::StrCat("dbg", l, "_kn"), k);
      q = PartialRope(Transpose(q, {0, 2, 1, 3}), cos, sin, rd);   // [1, H, seq, hd]
      k = PartialRope(Transpose(k, {0, 2, 1, 3}), cos, sin, rd);   // [1, KV, seq, hd]
      tap(absl::StrCat("dbg", l, "_qr"), q);
      tap(absl::StrCat("dbg", l, "_kr"), k);
      v = Transpose(Reshape(v, {1, seq, KV, hd}), {0, 2, 3, 1});   // [1, KV, hd, seq]
      T kc = DynamicUpdateSlice(sin_state[l], k, k_start);
      T vc = DynamicUpdateSlice(rin[l], v, v_start);
      kc.SetName(absl::StrCat("kv_cache_k_", l));
      vc.SetName(absl::StrCat("kv_cache_v_", l));
      sout[l] = kc;
      rout[l] = vc;
      T scores = BatchMatMul(Reshape(q, {KV, g * seq, hd}), Reshape(kc, {KV, cache, hd}),
                             /*adj_x=*/false, /*adj_y=*/true);  // [KV, g*seq, cache]
      scores = Add(Mul(Reshape(scores, {1, H, seq, cache}), scale), mask);
      tap(absl::StrCat("dbg", l, "_scores"), scores);
      T probs = Reshape(Softmax(scores), {KV, g * seq, cache});
      T o = BatchMatMul(probs, Reshape(vc, {KV, hd, cache}), /*adj_x=*/false, /*adj_y=*/true);  // [KV, g*seq, hd]
      o = Reshape(Transpose(Reshape(o, {1, H, seq, hd}), {0, 2, 1, 3}), {1, seq, H * hd});
      tap(absl::StrCat("dbg", l, "_o"), o);
      o = Mul(o, Logistic(gate));
      tap(absl::StrCat("dbg", l, "_og"), o);
      x = Add(x, FullyConnected(o, w.Get(p + "o")));
      tap(absl::StrCat("dbg", l, "_xa"), x);
    }

    // Sparse MoE block: softmax over all experts, top-k, renormalize; plus the gated shared expert.
    T h2 = RmsNorm(x, w.Get(p + "post_norm"), c.eps);
    T router = FullyConnected(h2, w.Get(p + "router"), ::litert::tensor::kActNone,
                              /*keep_num_dims=*/false);  // [seq, E]
    std::vector<T> top = TopK(Softmax(router), K);       // [seq, K] values, [seq, K] i32 indices
    T top_w = Div(top[0], Sum(top[0], std::vector<int>{1}, /*keep_dims=*/true));
    std::vector<T> ins = {h2, Reshape(top_w, {1, seq, K}), Reshape(top[1], {1, seq, K})};
    for (const char* m : {"ex_gate", "ex_up", "ex_down"}) {
      const std::string n = p + m;
      ins.push_back(w.Get(n));
      if (c.quant_experts) ins.push_back(w.Get(n + "_scales"));
    }
    ins.push_back(expert_scale);
    std::vector<T> moe = Custom(absl::MakeSpan(ins), "moe", moe_options, {{1, seq, D}}, {Type::kFP32});
    T shared = FullyConnected(Mul(Silu(FullyConnected(h2, w.Get(p + "sh_gate"))),
                                  FullyConnected(h2, w.Get(p + "sh_up"))),
                              w.Get(p + "sh_down"));
    shared = Mul(shared, Logistic(FullyConnected(h2, w.Get(p + "sh_g"))));  // [1, seq, D] * [1, seq, 1]
    T y = Add(x, moe[0]);
    if (absl::GetFlag(FLAGS_moe_fusion_guard)) y = Add(y, Mul(moe[0], ConstF({0.0f}, {1})));
    x = Add(y, shared);
    if (debug_hidden) hidden.push_back(x);
  }

  Signature s;
  s.inputs = {tokens, pos, mask};
  for (int l = 0; l < c.layers; ++l) {
    s.inputs.push_back(sin_state[l]);
    s.inputs.push_back(rin[l]);
  }
  for (int l = 0; l < c.layers; ++l) {
    s.outputs.push_back(sout[l]);
    s.outputs.push_back(rout[l]);
  }
  if (decode) {
    T logits = FullyConnected(RmsNorm(x, w.Get("final_norm"), c.eps), w.Get("lm_head"));  // [1, seq, V]
    logits.SetName("logits");
    s.outputs.push_back(logits);
  }
  for (int l = 0; l < static_cast<int>(hidden.size()); ++l) {
    hidden[l].SetName(absl::StrCat("hidden_", l));
    s.outputs.push_back(hidden[l]);
  }
  for (auto& [name, t] : dbg) {
    t.SetName(name);
    s.outputs.push_back(t);
  }
  return s;
}

}  // namespace

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);
  const std::string dir = absl::GetFlag(FLAGS_weights_dir);
  const std::string out = absl::GetFlag(FLAGS_out);
  const bool embedder_only = absl::GetFlag(FLAGS_embedder_only);
  ABSL_CHECK(!dir.empty() && (!out.empty() || embedder_only)) << "--weights_dir and --out are required";
  ABSL_CHECK(!embedder_only || !absl::GetFlag(FLAGS_embedder_out).empty()) << "--embedder_only needs --embedder_out";
  const int cache = absl::GetFlag(FLAGS_cache);
  const Config c = ReadConfig(dir + "/config.tsv");
  const std::vector<uint8_t> moe_options = ReadBytes(dir + "/moe_options.bin");
  const std::vector<uint8_t> gdn_options = ReadBytes(dir + "/gdn_options.bin");
  int n_lin = 0;
  for (bool b : c.is_linear) n_lin += b;
  std::cout << "config: hidden " << c.hidden << " layers " << c.layers << " (linear " << n_lin << ", full "
            << c.layers - n_lin << ") heads " << c.heads << "/" << c.kv_heads << " head_dim " << c.head_dim
            << " rotary " << c.rotary_dim << " gdn " << c.lin_k_heads << "x" << c.lin_k_dim << " / "
            << c.lin_v_heads << "x" << c.lin_v_dim << " conv " << c.conv_k << " experts " << c.experts << " top_k "
            << c.top_k << " moe_inter " << c.moe_inter << " shared " << c.shared_inter << " vocab " << c.vocab
            << " cache " << cache << " gdn " << absl::GetFlag(FLAGS_gdn) << "/" << absl::GetFlag(FLAGS_gdn_chunk)
            << " quant_experts " << c.quant_experts << " split_embed_cols " << absl::GetFlag(FLAGS_split_embed_cols)
            << " guard "
            << absl::GetFlag(FLAGS_moe_fusion_guard) << " debug_hidden " << absl::GetFlag(FLAGS_debug_hidden)
            << std::endl;
  Weights w(dir);
  const std::string emb_out = absl::GetFlag(FLAGS_embedder_out);
  if (!emb_out.empty()) {
    ModelFactory ef;
    T ids({.name = "token_ids", .type = Type::kI32, .shape = {1, 1}});
    T e = Reshape(EmbeddingLookup(Reshape(ids, {1}), w.Get("embed")), {1, 1, c.hidden});
    e.SetName("embeddings");
    std::vector<TensorHandle> ein = {ids}, eout = {e};
    absl::Status est = ef.AddSignature(ein, eout, "embedder");
    ABSL_CHECK(est.ok()) << est;
    est = ef.Save(emb_out);
    ABSL_CHECK(est.ok()) << est;
    std::cout << "WROTE embedder " << emb_out << std::endl;
    if (embedder_only) return 0;
  }
  ModelFactory factory;
  for (absl::string_view p : absl::StrSplit(absl::GetFlag(FLAGS_prefill), ',', absl::SkipEmpty())) {
    int n = 0;
    ABSL_CHECK(absl::SimpleAtoi(p, &n));
    ABSL_CHECK_GT(n, 1) << "prefill lengths must be > 1";
    Signature s = Build(c, w, moe_options, gdn_options, n, cache, /*decode=*/false, /*debug_hidden=*/false);
    absl::Status st = factory.AddSignature(s.inputs, s.outputs, absl::StrCat("prefill_", n));
    ABSL_CHECK(st.ok()) << st;
    std::cout << "signature prefill_" << n << std::endl;
  }
  Signature d = Build(c, w, moe_options, gdn_options, 1, cache, /*decode=*/true, /*debug_hidden=*/false);
  absl::Status st = factory.AddSignature(d.inputs, d.outputs, "decode");
  ABSL_CHECK(st.ok()) << st;
  std::cout << "signature decode" << std::endl;
  if (absl::GetFlag(FLAGS_debug_hidden)) {
    Signature dd = Build(c, w, moe_options, gdn_options, 1, cache, /*decode=*/true, /*debug_hidden=*/true);
    st = factory.AddSignature(dd.inputs, dd.outputs, "decode_dbg");
    ABSL_CHECK(st.ok()) << st;
    std::cout << "signature decode_dbg (" << dd.outputs.size() << " outputs)" << std::endl;
  }
  const uint64_t out_offset = absl::GetFlag(FLAGS_out_offset);
  st = out_offset > 0 ? factory.SaveAt(out, out_offset) : factory.Save(out);
  ABSL_CHECK(st.ok()) << st;
  std::cout << "WROTE " << out << (out_offset > 0 ? " at offset " + std::to_string(out_offset) : std::string())
            << std::endl;
  return 0;
}
