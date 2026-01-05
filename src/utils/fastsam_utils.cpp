#include "utils/fastsam_utils.hpp"
#include <iostream>
#include <algorithm>
#include <numeric>

FastSAMSegmenter::FastSAMSegmenter(const std::string &model_path,
                                   bool use_cuda,
                                   int input_size,
                                   float conf_threshold,
                                   float iou_threshold)
    : use_cuda(use_cuda),
      device(use_cuda && torch::cuda::is_available() ? torch::kCUDA : torch::kCPU),
      input_size(input_size),
      conf_threshold(conf_threshold),
      iou_threshold(iou_threshold)
{
    try
    {
        model = torch::jit::load(model_path);
        model.to(device);
        model.eval();
        std::cout << "Loaded FastSAM model from " << model_path << std::endl;
    }
    catch (const c10::Error &e)
    {
        std::cerr << "Error loading the FastSAM model: " << e.what() << std::endl;
        throw;
    }
}

torch::Tensor FastSAMSegmenter::preprocess(const cv::Mat &frame)
{
    // BGR -> RGB
    cv::Mat rgb_img;
    cv::cvtColor(frame, rgb_img, cv::COLOR_BGR2RGB);

    // Resize to input size
    cv::Mat resized_img;
    cv::resize(rgb_img, resized_img, cv::Size(input_size, input_size), 0, 0, cv::INTER_LINEAR);

    // HWC uint8 -> NCHW float32 in [0,1]
    torch::Tensor tensor_image = torch::from_blob(
        resized_img.data,
        {1, resized_img.rows, resized_img.cols, 3},
        torch::kUInt8);

    tensor_image = tensor_image.permute({0, 3, 1, 2}).to(torch::kFloat32).div(255.0f);
    return tensor_image.to(device);
}

FastSAMResult FastSAMSegmenter::postprocess(const torch::Tensor &detection_output,
                                            const torch::Tensor &proto_output,
                                            const cv::Size &orig_size)
{
    FastSAMResult result;

    // Expected shapes:
    // detection_output: [1, 37, 8400]
    // proto_output:     [1, 32, 160, 160]
    auto det = detection_output.squeeze(0).transpose(0, 1);  // [8400, 37]
    auto protos = proto_output.squeeze(0).cpu();             // [32, 160, 160]

    // boxes xywh, scores, mask coeffs
    auto boxes = det.slice(1, 0, 4);            // [8400, 4]
    auto scores = det.slice(1, 4, 5).squeeze(1); // [8400]
    auto mask_coeffs = det.slice(1, 5, 37);    // [8400, 32]

    // Confidence threshold
    auto conf_mask = scores > conf_threshold;
    auto indices = torch::nonzero(conf_mask).squeeze(1);
    if (indices.numel() == 0)
    {
        // No detections after confidence filtering
        return result;
    }

    auto filtered_boxes = boxes.index_select(0, indices);
    auto filtered_scores = scores.index_select(0, indices);
    auto filtered_coeffs = mask_coeffs.index_select(0, indices);

    // Move to CPU for OpenCV
    filtered_boxes = filtered_boxes.cpu();
    filtered_scores = filtered_scores.cpu();
    filtered_coeffs = filtered_coeffs.cpu();

    // Convert boxes from normalized xywh to pixel xyxy
    float scale_x = static_cast<float>(orig_size.width) / static_cast<float>(input_size);
    float scale_y = static_cast<float>(orig_size.height) / static_cast<float>(input_size);

    std::vector<cv::Rect> box_vec;
    std::vector<float> score_vec;

    auto boxes_accessor = filtered_boxes.accessor<float, 2>(); // [N, 4]
    auto scores_accessor = filtered_scores.accessor<float, 1>(); // [N]

    for (int i = 0; i < filtered_boxes.size(0); ++i)
    {
        float cx = boxes_accessor[i][0] * scale_x;
        float cy = boxes_accessor[i][1] * scale_y;
        float w  = boxes_accessor[i][2] * scale_x;
        float h  = boxes_accessor[i][3] * scale_y;

        int x1 = static_cast<int>(cx - w / 2.0f);
        int y1 = static_cast<int>(cy - h / 2.0f);
        int x2 = static_cast<int>(cx + w / 2.0f);
        int y2 = static_cast<int>(cy + h / 2.0f);

        box_vec.emplace_back(x1, y1, x2 - x1, y2 - y1);
        score_vec.emplace_back(scores_accessor[i]);
    }

    // NMS to remove overlapping boxes (model-agnostic)
    std::vector<int> keep_indices = nms(box_vec, score_vec, iou_threshold);

    if (keep_indices.empty())
    {
        return result;
    }

    // Generate masks for all kept detections (no additional scene filtering)
    auto protos_flat = protos.view({32, -1}); // [32, 160*160]

    for (int idx : keep_indices)
    {
        // Save box and score
        result.boxes.push_back(box_vec[idx]);
        result.scores.push_back(score_vec[idx]);

        // Mask:
        auto coeff = filtered_coeffs[idx].unsqueeze(0); // [1, 32]
        auto mask_pred = torch::matmul(coeff, protos_flat); // [1, 160*160]
        mask_pred = mask_pred.view({1, 160, 160});
        mask_pred = torch::sigmoid(mask_pred);

        // Resize to original image
        auto mask_resized = torch::nn::functional::interpolate(
            mask_pred.unsqueeze(0),
            torch::nn::functional::InterpolateFuncOptions()
                .size(std::vector<int64_t>{orig_size.height, orig_size.width})
                .mode(torch::kBilinear)
                .align_corners(false))
            .squeeze(0); // [1, H, W]

        // Binary mask at 0.5
        auto binary_mask = (mask_resized > 0.5f).to(torch::kUInt8).mul(255).cpu();

        cv::Mat mask_mat(orig_size.height, orig_size.width, CV_8UC1);
        std::memcpy(
            mask_mat.data,
            binary_mask.data_ptr<uint8_t>(),
            sizeof(uint8_t) * orig_size.height * orig_size.width);

        result.masks.push_back(mask_mat);
    }

    return result;
}

std::vector<int> FastSAMSegmenter::nms(const std::vector<cv::Rect> &boxes,
                                       const std::vector<float> &scores,
                                       float iou_threshold)
{
    std::vector<int> indices(boxes.size());
    std::iota(indices.begin(), indices.end(), 0);

    // Sort by score descending
    std::sort(indices.begin(), indices.end(),
              [&](int i1, int i2) { return scores[i1] > scores[i2]; });

    std::vector<int> keep;
    std::vector<bool> suppressed(boxes.size(), false);

    for (size_t i = 0; i < indices.size(); ++i)
    {
        int idx = indices[i];
        if (suppressed[idx]) continue;

        keep.push_back(idx);

        for (size_t j = i + 1; j < indices.size(); ++j)
        {
            int idx2 = indices[j];
            if (suppressed[idx2]) continue;

            cv::Rect inter = boxes[idx] & boxes[idx2];
            float inter_area = static_cast<float>(inter.area());
            float union_area = static_cast<float>(boxes[idx].area() + boxes[idx2].area()) - inter_area;
            float iou = inter_area / (union_area + 1e-6f);

            if (iou > iou_threshold)
                suppressed[idx2] = true;
        }
    }

    return keep;
}

FastSAMResult FastSAMSegmenter::segment(const cv::Mat &frame)
{
    torch::NoGradGuard no_grad;

    torch::Tensor input_tensor = preprocess(frame);
    auto output = model.forward({input_tensor}).toTuple();

    torch::Tensor detection_output = output->elements()[0].toTensor();
    torch::Tensor proto_output     = output->elements()[1].toTensor();

    return postprocess(detection_output, proto_output, frame.size());
}

cv::Mat FastSAMSegmenter::visualize(const cv::Mat &frame, const FastSAMResult &result)
{
    cv::Mat vis_frame = frame.clone();
    std::vector<cv::Scalar> colors;

    for (size_t i = 0; i < result.masks.size(); ++i)
    {
        colors.emplace_back(rand() % 256, rand() % 256, rand() % 256);
    }

    for (size_t i = 0; i < result.masks.size(); ++i)
    {
        cv::Mat colored_mask;
        cv::cvtColor(result.masks[i], colored_mask, cv::COLOR_GRAY2BGR);
        colored_mask.setTo(colors[i], result.masks[i]);

        cv::addWeighted(vis_frame, 1.0, colored_mask, 0.5, 0, vis_frame);
        cv::rectangle(vis_frame, result.boxes[i], colors[i], 2);

#if CV_VERSION_MAJOR >= 4
        std::string label = cv::format("%.2f", result.scores[i]);
#else
        char buf[32];
        std::snprintf(buf, sizeof(buf), "%.2f", result.scores[i]);
        std::string label(buf);
#endif
        cv::putText(vis_frame, label,
                    cv::Point(result.boxes[i].x, result.boxes[i].y - 5),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, colors[i], 2);
    }

    return vis_frame;
}
