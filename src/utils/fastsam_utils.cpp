#include "utils/fastsam_utils.hpp"
#include <iostream>
#include <algorithm>

FastSAMSegmenter::FastSAMSegmenter(const std::string& model_path, bool use_cuda,
                                   int input_size, float conf_threshold, 
                                   float iou_threshold)
    : use_cuda_(use_cuda),
      device_((use_cuda && torch::cuda::is_available()) ? torch::kCUDA : torch::kCPU),
      input_size_(input_size),
      conf_threshold_(conf_threshold),
      iou_threshold_(iou_threshold) {
    try {
        model_ = torch::jit::load(model_path);
        model_.to(device_);
        model_.eval();
        std::cout << "Loaded FastSAM model from " << model_path << std::endl;
    }
    catch (const c10::Error& e) {
        std::cerr << "Error loading the FastSAM model: " << e.what() << std::endl;
        throw;
    }
}

torch::Tensor FastSAMSegmenter::preprocess(const cv::Mat& frame) {
    // Convert BGR to RGB
    cv::Mat rgb_img;
    cv::cvtColor(frame, rgb_img, cv::COLOR_BGR2RGB);
    
    // Resize to input size (640x640 by default)
    cv::Mat resized_img;
    cv::resize(rgb_img, resized_img, cv::Size(input_size_, input_size_), 0, 0, cv::INTER_LINEAR);

    // Convert to float tensor and normalize to [0, 1]
    torch::Tensor tensor_image = torch::from_blob(
        resized_img.data, 
        {1, resized_img.rows, resized_img.cols, 3}, 
        torch::kUInt8
    );
    
    // Permute to NCHW format and normalize
    tensor_image = tensor_image.permute({0, 3, 1, 2}).to(torch::kFloat32).div(255.0);
    
    return tensor_image.to(device_);
}

FastSAMResult FastSAMSegmenter::postprocess(const torch::Tensor& detection_output,
                                           const torch::Tensor& proto_output,
                                           const cv::Size& orig_size) {
    FastSAMResult result;
    
    // detection_output shape: [1, 37, 8400]
    // proto_output shape: [1, 32, 160, 160]
    
    auto det = detection_output.squeeze(0).transpose(0, 1); // [8400, 37]
    auto protos = proto_output.squeeze(0).cpu(); // [32, 160, 160]
    
    // Extract boxes (first 4 channels), confidence (5th channel), and mask coeffs (last 32)
    auto boxes = det.slice(1, 0, 4);  // [8400, 4] - xywh format
    auto scores = det.slice(1, 4, 5).squeeze(1);  // [8400]
    auto mask_coeffs = det.slice(1, 5, 37);  // [8400, 32]
    
    // Filter by confidence threshold
    auto conf_mask = scores > conf_threshold_;
    auto indices = torch::nonzero(conf_mask).squeeze(1);
    
    if (indices.numel() == 0) {
        return result;  // No detections
    }
    
    auto filtered_boxes = boxes.index_select(0, indices);
    auto filtered_scores = scores.index_select(0, indices);
    auto filtered_coeffs = mask_coeffs.index_select(0, indices);
    
    // Convert to CPU for OpenCV processing
    filtered_boxes = filtered_boxes.cpu();
    filtered_scores = filtered_scores.cpu();
    filtered_coeffs = filtered_coeffs.cpu();
    
    // Convert boxes from normalized xywh to pixel xyxy format
    float scale_x = static_cast<float>(orig_size.width) / input_size_;
    float scale_y = static_cast<float>(orig_size.height) / input_size_;
    
    std::vector<cv::Rect> box_vec;
    std::vector<float> score_vec;
    
    auto boxes_accessor = filtered_boxes.accessor<float, 2>();
    auto scores_accessor = filtered_scores.accessor<float, 1>();
    
    for (int i = 0; i < filtered_boxes.size(0); ++i) {
        float cx = boxes_accessor[i][0] * scale_x;
        float cy = boxes_accessor[i][1] * scale_y;
        float w = boxes_accessor[i][2] * scale_x;
        float h = boxes_accessor[i][3] * scale_y;
        
        int x1 = static_cast<int>(cx - w / 2);
        int y1 = static_cast<int>(cy - h / 2);
        int x2 = static_cast<int>(cx + w / 2);
        int y2 = static_cast<int>(cy + h / 2);
        
        box_vec.push_back(cv::Rect(x1, y1, x2 - x1, y2 - y1));
        score_vec.push_back(scores_accessor[i]);
    }
    
    // Apply NMS
    auto keep_indices = nms(box_vec, score_vec, iou_threshold_);
    
    // Filter by minimum area and sort by score
    std::vector<int> keep_filtered;
    int min_area = 5000;  // Minimum area in pixels (adjust as needed)
    
    for (int idx : keep_indices) {
        int box_area = box_vec[idx].area();
        if (box_area > min_area) {
            keep_filtered.push_back(idx);
        }
    }
    
    // Sort by score to keep top detections
    std::sort(keep_filtered.begin(), keep_filtered.end(), 
              [&score_vec](int i1, int i2) {
                  return score_vec[i1] > score_vec[i2];
              });
    
    // Limit to top 5 detections
    int max_detections = 5;
    if (keep_filtered.size() > max_detections) {
        keep_filtered.resize(max_detections);
    }
    
    // Generate masks for filtered detections
    for (int idx : keep_filtered) {
        result.boxes.push_back(box_vec[idx]);
        result.scores.push_back(score_vec[idx]);
        
        // Generate mask: mask = sigmoid(coeffs @ protos)
        auto coeff = filtered_coeffs[idx].unsqueeze(0);  // [1, 32]
        auto proto_flat = protos.view({32, -1});  // [32, 160*160]
        auto mask_pred = torch::matmul(coeff, proto_flat);  // [1, 160*160]
        mask_pred = mask_pred.view({1, 160, 160});
        mask_pred = torch::sigmoid(mask_pred);
        
        // Resize mask to original image size
        auto mask_resized = torch::nn::functional::interpolate(
            mask_pred.unsqueeze(0),
            torch::nn::functional::InterpolateFuncOptions()
                .size(std::vector<int64_t>{orig_size.height, orig_size.width})
                .mode(torch::kBilinear)
                .align_corners(false)
        ).squeeze();
        
        // Convert to binary mask (threshold at 0.5)
        auto binary_mask = (mask_resized > 0.5).to(torch::kUInt8).mul(255).cpu();
        
        // Convert to OpenCV Mat
        cv::Mat mask_mat(orig_size.height, orig_size.width, CV_8UC1);
        std::memcpy(mask_mat.data, binary_mask.data_ptr(), 
                   sizeof(uint8_t) * orig_size.height * orig_size.width);
        
        result.masks.push_back(mask_mat);
    }
    
    return result;
}

std::vector<int> FastSAMSegmenter::nms(const std::vector<cv::Rect>& boxes,
                                       const std::vector<float>& scores,
                                       float iou_threshold) {
    std::vector<int> indices(boxes.size());
    std::iota(indices.begin(), indices.end(), 0);
    
    // Sort by scores descending
    std::sort(indices.begin(), indices.end(), [&scores](int i1, int i2) {
        return scores[i1] > scores[i2];
    });
    
    std::vector<int> keep;
    std::vector<bool> suppressed(boxes.size(), false);
    
    for (size_t i = 0; i < indices.size(); ++i) {
        int idx = indices[i];
        if (suppressed[idx]) continue;
        
        keep.push_back(idx);
        
        for (size_t j = i + 1; j < indices.size(); ++j) {
            int idx2 = indices[j];
            if (suppressed[idx2]) continue;
            
            // Calculate IoU
            cv::Rect inter = boxes[idx] & boxes[idx2];
            float inter_area = inter.area();
            float union_area = boxes[idx].area() + boxes[idx2].area() - inter_area;
            float iou = inter_area / (union_area + 1e-6);
            
            if (iou > iou_threshold) {
                suppressed[idx2] = true;
            }
        }
    }
    
    return keep;
}

FastSAMResult FastSAMSegmenter::segment(const cv::Mat& frame) {
    torch::NoGradGuard no_grad;
    
    torch::Tensor input_tensor = preprocess(frame);
    
    // Forward pass - returns tuple (detection, proto_masks)
    auto output = model_.forward({input_tensor}).toTuple();
    torch::Tensor detection_output = output->elements()[0].toTensor();
    torch::Tensor proto_output = output->elements()[1].toTensor();
    
    FastSAMResult result = postprocess(detection_output, proto_output, frame.size());
    
    return result;
}

cv::Mat FastSAMSegmenter::visualize(const cv::Mat& frame, const FastSAMResult& result) {
    cv::Mat vis_frame = frame.clone();
    
    // Generate random colors for each mask
    std::vector<cv::Scalar> colors;
    for (size_t i = 0; i < result.masks.size(); ++i) {
        colors.push_back(cv::Scalar(rand() % 256, rand() % 256, rand() % 256));
    }
    
    // Overlay masks
    for (size_t i = 0; i < result.masks.size(); ++i) {
        cv::Mat colored_mask;
        cv::cvtColor(result.masks[i], colored_mask, cv::COLOR_GRAY2BGR);
        colored_mask.setTo(colors[i], result.masks[i]);
        
        // Blend with original image
        cv::addWeighted(vis_frame, 1.0, colored_mask, 0.5, 0, vis_frame);
        
        // Draw bounding box
        cv::rectangle(vis_frame, result.boxes[i], colors[i], 2);
        
        // Draw confidence score
        std::string label = cv::format("%.2f", result.scores[i]);
        cv::putText(vis_frame, label, 
                   cv::Point(result.boxes[i].x, result.boxes[i].y - 5),
                   cv::FONT_HERSHEY_SIMPLEX, 0.5, colors[i], 2);
    }
    
    return vis_frame;
}
