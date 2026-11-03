# Tensor API writer for a Qwen3-MoE decoder (Qwen/Qwen3-30B-A3B architecture) with the "moe" custom op, in
# LiteRT-LM's stock signature layout. Linked into a LiteRT checkout as tensor/examples/qwen3moe by ../overlay.sh.
load("@rules_cc//cc:cc_binary.bzl", "cc_binary")

package(default_visibility = ["//visibility:public"])

cc_binary(
    name = "qwen3moe_writer",
    srcs = ["qwen3moe_writer.cc"],
    deps = [
        "//tensor",
        "//tensor:arithmetic",
        "//tensor:buffer",
        "//tensor:datatypes",
        "//tensor/backends/tflite:arithmetic_tflite",
        "//tensor/backends/tflite:tflite_flatbuffer_conversion",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)
