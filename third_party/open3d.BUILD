# third_party/open3d.BUILD

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "open3d",
    srcs = [
        "lib/libOpen3D.so",
    ],
    hdrs = glob([
        "include/**/*.h",
        "include/**/*.hpp",
    ]),
    includes = ["include"],
    linkopts = [
        "-lglfw",
        "-lGL",
        "-lGLU",
    ],
    visibility = ["//visibility:public"],
)
