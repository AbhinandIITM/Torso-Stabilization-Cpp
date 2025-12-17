#include "utils/fastsam_utils.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>
#include "tools/cpp/runfiles/runfiles.h"

using bazel::tools::cpp::runfiles::Runfiles;

int main(int argc, char** argv) {
    std::string error;
    std::unique_ptr<Runfiles> runfiles(Runfiles::Create(argv[0], &error));
    if (!runfiles) {
        std::cerr << "Runfiles error: " << error << std::endl;
        return 1;
    }

    // Load FastSAM model from runfiles
    const std::string model_path =
        runfiles->Rlocation("mediapipe/src/models/FastSAM-x.torchscript");

    cv::VideoCapture cap(2);
    if (!cap.isOpened()) {
        std::cerr << "[ERROR] Failed to open webcam" << std::endl;
        return 1;
    }
    cap.set(cv::CAP_PROP_BUFFERSIZE, 1);

    FastSAMSegmenter fastsam(model_path, /*use_cuda=*/true,
                            /*input_size=*/640, 
                            /*conf_threshold=*/0.5,   // Increase from 0.25 to 0.5
                            /*iou_threshold=*/0.7);   // Increase from 0.45 to 0.7

    std::cout << "[INFO] FastSAM initialized with model: " << model_path << std::endl;
    std::cout << "[INFO] Press ESC to exit, 's' to save screenshot" << std::endl;

    cv::Mat frame;
    int frame_count = 0;
    
    while (true) {
        cap >> frame;
        if (frame.empty()) {
            std::cerr << "[WARN] Empty frame received from webcam" << std::endl;
            break;
        }

        // Run segmentation
        FastSAMResult result = fastsam.segment(frame);
        
        if (result.masks.empty()) {
            std::cout << "[INFO] No segments detected in frame " << frame_count << std::endl;
            cv::imshow("Webcam", frame);
        } else {
            std::cout << "[INFO] Frame " << frame_count << ": Found " 
                      << result.masks.size() << " segments" << std::endl;
            
            // Visualize results
            cv::Mat vis_frame = fastsam.visualize(frame, result);
            
            // Display original and segmented frames
            cv::imshow("Webcam", frame);
            cv::imshow("FastSAM Segmentation", vis_frame);
            
            // Print details for each segment
            for (size_t i = 0; i < result.boxes.size(); ++i) {
                std::cout << "  Segment " << i << ": "
                          << "Box [" << result.boxes[i].x << ", " << result.boxes[i].y 
                          << ", " << result.boxes[i].width << ", " << result.boxes[i].height 
                          << "], Score: " << result.scores[i] << std::endl;
            }
        }

        int key = cv::waitKey(1);
        if (key == 27) break;  // ESC to exit
        
        // Save screenshot on 's' key
        if (key == 's' || key == 'S') {
            if (!result.masks.empty()) {
                cv::Mat vis_frame = fastsam.visualize(frame, result);
                std::string filename = "fastsam_output_" + std::to_string(frame_count) + ".jpg";
                cv::imwrite(filename, vis_frame);
                std::cout << "[INFO] Saved screenshot: " << filename << std::endl;
            }
        }
        
        frame_count++;
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}
