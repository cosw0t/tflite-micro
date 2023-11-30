# Description:
#   TensorFlow Lite for Microcontrollers "hello world" example.
load("@rules_python//python:defs.bzl", "py_binary")
load("@tflm_pip_deps//:requirements.bzl", "requirement")
load(
    "//tensorflow/lite/micro:build_def.bzl",
    "micro_copts",
)

package(
    # Disabling layering_check because of http://b/177257332
    features = ["-layering_check"],
    licenses = ["notice"],
)

cc_library(
    name = "model",
    srcs = [
        "//tensorflow/lite/micro/examples/bug_report_conv2d/models:generated_bug_report_conv2d_model_cc",
    ],
    hdrs = [
        "//tensorflow/lite/micro/examples/bug_report_conv2d/models:generated_bug_report_conv2d_model_hdr",
    ],
    copts = micro_copts(),
)

cc_test(
    name = "bug_report_conv2d_test",
    srcs = [
        "bug_report_conv2d_test.cc",
    ],
    deps = [
        ":model",
        "//tensorflow/lite/micro:micro_framework",
        "//tensorflow/lite/micro:micro_log",
        "//tensorflow/lite/micro:micro_profiler",
        "//tensorflow/lite/micro:op_resolvers",
        "//tensorflow/lite/micro:recording_allocators",
        "//tensorflow/lite/micro/testing:micro_test",
        "//tensorflow/lite/schema:schema_fbs",
    ],
)

