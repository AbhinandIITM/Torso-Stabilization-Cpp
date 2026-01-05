#pragma once

#include <torch/torch.h>
#include <torch/script.h>
#include <opencv2/opencv.hpp>
#include <vector>

struct FastSAMResult
{
    // Direct model outputs (no scene-level filtering)
    std::vector<cv::Mat> masks;   // Binary segmentation masks
    std::vector<cv::Rect> boxes;  // Bounding boxes
    std::vector<float> scores;    // Confidence scores
};

class FastSAMSegmenter
{
public:
    FastSAMSegmenter(const std::string &model_path,
                     bool use_cuda = true,
                     int input_size = 640,
                     float conf_threshold = 0.25f,
                     float iou_threshold = 0.45f);

    // Run full inference on a frame and return raw detections
    FastSAMResult segment(const cv::Mat &frame);

    // Simple visualization utility (optional)
    cv::Mat visualize(const cv::Mat &frame, const FastSAMResult &result);

private:
    torch::jit::script::Module model;
    torch::Device device;
    bool use_cuda;
    int input_size;
    float conf_threshold;
    float iou_threshold;

    torch::Tensor preprocess(const cv::Mat &frame);

    // Postprocess converts raw model tensors to masks / boxes / scores,
    // but does NOT perform any task/scene-specific filtering (area, top-K, etc.).
    FastSAMResult postprocess(const torch::Tensor &detection_output,
                              const torch::Tensor &proto_output,
                              const cv::Size &orig_size);

    // Helper NMS used only inside postprocess to remove overlapping boxes
    std::vector<int> nms(const std::vector<cv::Rect> &boxes,
                         const std::vector<float> &scores,
                         float iou_threshold);
};
