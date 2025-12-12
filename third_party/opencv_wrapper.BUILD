# Wrapper to properly link all OpenCV libraries together

cc_library(
    name = "opencv_core",
    srcs = ["@opencv//:opencv"],
    linkopts = [
        "-lpthread",
        "-ldl",
        "-lz",
    ],
    visibility = ["//visibility:public"],
)
