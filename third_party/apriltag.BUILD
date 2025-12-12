# third_party/apriltag.BUILD
package(default_visibility = ["//visibility:public"])

cc_library(
    name = "apriltag",
    srcs = glob(
        [
            "*.c",
            "common/*.c",
        ],
        exclude = [
            "apriltag_pywrap.c",
            "**/example*.c",
            "**/test*.c",
        ],
    ),
    hdrs = glob([
        "*.h",
        "common/*.h",
    ]),
    include_prefix = "apriltag",
    strip_include_prefix = "",
    copts = [
        "-std=gnu99",
        "-fPIC",
        "-Wall",
        "-Wno-unused-parameter",
        "-Wno-unused-function",
        "-Wno-unused-variable",
        "-O3",
    ],
    linkopts = ["-lpthread", "-lm"],
)
