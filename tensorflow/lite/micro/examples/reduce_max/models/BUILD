load("//tensorflow/lite/micro:build_def.bzl", "generate_cc_arrays")

package(
    default_visibility = ["//visibility:public"],
)

exports_files(
    srcs = [
        "reduce_max.tflite",
    ],
    visibility = ["//tensorflow/lite/micro/examples/reduce_max:__subpackages__"],
)

generate_cc_arrays(
    name = "generated_reduce_max_model_cc",
    src = "reduce_max.tflite",
    out = "reduce_max_model_data.cc",
)

generate_cc_arrays(
    name = "generated_reduce_max_model_hdr",
    src = "reduce_max.tflite",
    out = "reduce_max_model_data.h",
)
