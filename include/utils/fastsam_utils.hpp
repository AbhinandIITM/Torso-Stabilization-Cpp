// fastsam_utils.hpp
#pragma once

#include <torch/torch.h>
#include <torch/script.h>
#include <opencv2/opencv.hpp>
#include <vector>

struct FastSAMResult {
    std::vector<cv::Mat> masks;           // Segmentation masks
    std::vector<cv::Rect> boxes;          // Bounding boxes
    std::vector<float> scores;            // Confidence scores
};

class FastSAMSegmenter {
public:
    FastSAMSegmenter(const std::string& model_path, bool use_cuda = true, 
                     int input_size = 640, float conf_threshold = 0.25, 
                     float iou_threshold = 0.45);
    
    FastSAMResult segment(const cv::Mat& frame);
    cv::Mat visualize(const cv::Mat& frame, const FastSAMResult& result);

private:
    torch::jit::script::Module model_;
    torch::Device device_;
    bool use_cuda_;
    int input_size_;
    float conf_threshold_;
    float iou_threshold_;

    torch::Tensor preprocess(const cv::Mat& frame);
    FastSAMResult postprocess(const torch::Tensor& detection_output, 
                             const torch::Tensor& proto_output,
                             const cv::Size& orig_size);
    std::vector<int> nms(const std::vector<cv::Rect>& boxes, 
                        const std::vector<float>& scores, 
                        float iou_threshold);
};
