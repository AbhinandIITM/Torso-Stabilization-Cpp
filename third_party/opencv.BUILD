load("@rules_foreign_cc//foreign_cc:defs.bzl", "cmake")

filegroup(
    name = "all_srcs",
    srcs = glob(["**"]),
    visibility = ["//visibility:public"],
)

cmake(
    name = "opencv_build",
    lib_source = ":all_srcs",
    out_static_libs = [
        "libopencv_calib3d.a",
        "libopencv_core.a",
        "libopencv_features2d.a",
        "libopencv_flann.a",
        "libopencv_highgui.a",
        "libopencv_imgcodecs.a",
        "libopencv_imgproc.a",
        "libopencv_video.a",
        "libopencv_videoio.a",
    ],
    cache_entries = {
        "CMAKE_BUILD_TYPE": "Release",
        "BUILD_SHARED_LIBS": "OFF",
        "BUILD_LIST": "core,imgproc,imgcodecs,highgui,video,videoio,features2d,flann,calib3d",
        "WITH_JPEG": "ON",
        "WITH_PNG": "ON",
        "WITH_TIFF": "OFF",
        "WITH_WEBP": "OFF",
        "WITH_JASPER": "OFF",
        "WITH_OPENEXR": "OFF",
        "BUILD_JPEG": "ON",
        "BUILD_PNG": "ON",
        "BUILD_ZLIB": "ON",
        "BUILD_TESTS": "OFF",
        "BUILD_PERF_TESTS": "OFF",
        "BUILD_EXAMPLES": "OFF",
        "BUILD_opencv_apps": "OFF",
        "BUILD_opencv_python2": "OFF",
        "BUILD_opencv_python3": "OFF",
        "WITH_ITT": "OFF",
        "WITH_IPP": "OFF",
        "WITH_LAPACK": "OFF",
        "ENABLE_PIC": "ON",
    },
    visibility = ["//visibility:public"],
)

# Wrapper library that properly orders all OpenCV static libs
cc_library(
    name = "opencv",
    deps = [":opencv_build"],
    linkopts = [
        "-Wl,--start-group",
        "-Wl,--end-group",
        "-lpthread",
        "-ldl",
        "-lz",
    ],
    visibility = ["//visibility:public"],
)
