load("//tensorflow/lite/micro:build_def.bzl", "generate_cc_arrays")

package(
    default_visibility = ["//visibility:public"],
)

exports_files(
    srcs = [
        "bug_report_conv2d.tflite",
    ],
    visibility = ["//tensorflow/lite/micro/examples/bug_report_conv2d:__subpackages__"],
)

generate_cc_arrays(
    name = "generated_bug_report_conv2d_model_cc",
    src = "bug_report_conv2d.tflite",
    out = "bug_report_conv2d_model_data.cc",
)

generate_cc_arrays(
    name = "generated_bug_report_conv2d_model_hdr",
    src = "bug_report_conv2d.tflite",
    out = "bug_report_conv2d_model_data.h",
)
