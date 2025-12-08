// MiDaSDepth.hpp
#pragma once

#include <torch/torch.h>
#include <torch/script.h>
#include <opencv2/opencv.hpp>

class MiDaSDepth {
public:
    MiDaSDepth(const std::string& model_path, bool use_cuda = true);
    cv::Mat getDepthMap(const cv::Mat& frame);

private:
    torch::jit::script::Module model_;
    torch::Device device_;
    bool use_cuda_;

    torch::Tensor preprocess(const cv::Mat& frame);
    cv::Mat postprocess(const torch::Tensor& prediction, const cv::Size& orig_size);
};
