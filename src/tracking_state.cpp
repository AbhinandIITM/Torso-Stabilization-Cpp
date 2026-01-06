#include "include/tracking_state.hpp"
#include <iostream>
#include <chrono>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <iomanip>

TrackingState::TrackingState(StateCommand state_command, cv::VideoCapture cap)
    : state_command(state_command),
      cap(cap),
      frame_count(0),
      tracking_active(false),
      frames_without_detection(0)
{
    std::cout << "TrackingState: Constructor called" << std::endl;

    // Get references to shared components from state_command
    midas_utils = state_command.midas_utils;
    apriltag_utils = state_command.apriltag_utils;
    mediapipe_utils = state_command.mediapipe_utils;
    camera_matrix = state_command.camera_matrix;
    dist_coeffs = state_command.dist_coeffs;

    // Verify components are initialized
    if (!midas_utils || !apriltag_utils) {
        throw std::runtime_error("TrackingState: Required components not initialized!");
    }

    std::cout << "TrackingState: All components verified" << std::endl;
}

SystemState TrackingState::run()
{
    std::cout << "TrackingState: Starting tracking loop..." << std::endl;
    std::cout << "Place AprilTag in view to begin tracking" << std::endl;
    std::cout << "Press 'i' to return to IDLE, 'q' or ESC to quit" << std::endl;

    auto starttime = std::chrono::steady_clock::now();

    while (true) {
        // Capture frame
        if (!cap.read(current_frame)) {
            std::cerr << "Error: Failed to capture frame" << std::endl;
            state_command.error_message = "Camera capture failed";
            return SystemState::ERROR;
        }
        frame_count++;

        // Process frame
        processFrame();

        // Visualize
        visualizeTracking();

        // Display
        cv::imshow("Torso Stabilization - Tracking State", current_frame);

        // Calculate and display FPS every 30 frames
        if (frame_count % 30 == 0) {
            auto currenttime = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(currenttime - starttime).count();
            float fps = 30.0f / (elapsed / 1000.0f);

            std::cout << "Frame " << frame_count
                      << " - FPS: " << fps
                      << " - Tracking: " << (tracking_active ? "ACTIVE" : "SEARCHING")
                      << " - AprilTag: " << (state_command.apriltag_detected ? "DETECTED" : "NOT FOUND");

            if (state_command.apriltag_detected) {
                std::cout << " (ID: " << state_command.apriltag_id
                          << ", Depth: " << state_command.apriltag_position.z() << "m)";
            }

            std::cout << " - Hands: " << hand_landmarks.size();

            // Print top scored object if smoothed_index_tip is valid
            if (!scored_objects.empty() && smoothed_index_tip.has_value()) {
                auto& topobj = scored_objects[0];
                std::cout << " - Target: Obj" << topobj.object_id
                          << " Score=" << topobj.reach_score
                          << ", Dist=" << topobj.distance_to_tip_meters << "m";
            }

            std::cout << std::endl;
            starttime = currenttime;
        }

        // Check for state transitions
        char key = cv::waitKey(1);
        if (key == 'i' || key == 'I') {
            std::cout << "Returning to IDLE state..." << std::endl;
            return SystemState::IDLE;
        } else if (key == 'q' || key == 'Q' || key == 27) {
            std::cout << "Shutdown requested" << std::endl;
            return SystemState::SHUTDOWN;
        }
    }

    return SystemState::IDLE;
}
void TrackingState::processFrame() {
    using namespace std::chrono;
    
    // Start total timer
    auto frame_start = steady_clock::now();

    // 1. AprilTag Detection
    auto t1 = steady_clock::now();
    detectAprilTag();
    auto d_tag = duration_cast<milliseconds>(steady_clock::now() - t1).count();

    // 2. Hand Tracking (MediaPipe)
    auto t2 = steady_clock::now();
    detectHands();
    auto d_hand = duration_cast<milliseconds>(steady_clock::now() - t2).count();

    // 3. Heavy Tasks: Segmentation & Depth (Conditional)
    long long d_sam = 0, d_depth = 0;
    if (frame_count % 10 == 0 || fastsam_result.boxes.empty()) {
        auto tsam = steady_clock::now();
        segmentObjects();
        d_sam = duration_cast<milliseconds>(steady_clock::now() - tsam).count();

        auto tdepth = steady_clock::now();
        estimateDepth();
        d_depth = duration_cast<milliseconds>(steady_clock::now() - tdepth).count();
    }

    // 4. Scoring & State Update
    auto t3 = steady_clock::now();
    calculateObjectScores();
    updateTracking();
    auto d_score = duration_cast<milliseconds>(steady_clock::now() - t3).count();

    // 5. Profiling Report
    if (frame_count % 30 == 0) {
        auto total = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - frame_start).count();
        float actual_ai_ms = mediapipe_utils->GetLatestInferenceMs();

        std::cout << "\n--- Performance Profile (Frame " << frame_count << ") ---" << std::endl;
        std::cout << "  AprilTag:  " << std::setw(3) << d_tag   << " ms" << std::endl;
        std::cout << "  MediaPipe: " << std::fixed << std::setprecision(1) << actual_ai_ms << " ms (Actual AI Latency)" << std::endl;
        std::cout << "  FastSAM:   " << std::setw(3) << d_sam   << " ms" << std::endl;
        std::cout << "  MiDaS:     " << std::setw(3) << d_depth << " ms" << std::endl;
        std::cout << "  TOTAL:     " << std::setw(3) << total   << " ms" << std::endl;
        std::cout << "------------------------------------------\n" << std::endl;
    }
}

bool TrackingState::detectAprilTag()
{
    try {
        std::vector<TagDetection> detections = apriltag_utils->get_tags(current_frame);

        if (!detections.empty()) {
            const TagDetection& detection = detections[0];

            state_command.apriltag_detected = true;
            state_command.apriltag_id = detection.id;
            state_command.apriltag_position = Eigen::Vector3f(
                detection.pose_t.at<double>(0, 0),
                detection.pose_t.at<double>(1, 0),
                detection.pose_t.at<double>(2, 0)
            );

            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    state_command.apriltag_rotation(i, j) = detection.pose_R.at<double>(i, j);
                }
            }

            // Calculate scaling factor from AprilTag
            state_command.scaling_factor = 0.05 / detection.depth;  // 15cm tag / measured distance

            frames_without_detection = 0;
            return true;
        } else {
            state_command.apriltag_detected = false;
            frames_without_detection++;
            return false;
        }
    } catch (const std::exception& e) {
        std::cerr << "AprilTag detection error: " << e.what() << std::endl;
        state_command.apriltag_detected = false;
        return false;
    }
}

void TrackingState::estimateDepth()
{
    try {
        state_command.depth_map = midas_utils->getDepthMap(current_frame);
    } catch (const std::exception& e) {
        std::cerr << "Depth estimation error: " << e.what() << std::endl;
        state_command.depth_map = cv::Mat();
    }
}

void TrackingState::segmentObjects()
{
    try {
        if (!state_command.fastsam_utils) {
            return;
        }

        // 1) Get direct inference outputs from FastSAM (already NMS + conf-thresholded,
        //    but no scene/task-specific filtering).
        fastsam_result = state_command.fastsam_utils->segment(current_frame);

        // 2) Apply tracking-state–specific filtering here.

        // Example filters (adapt as needed):
        const int    MINAREA = 5000;  // minimum pixel area
        const size_t MAXOBJS = 5;     // maximum objects to keep

        std::vector<cv::Mat>   filteredmasks;
        std::vector<cv::Rect>  filteredboxes;
        std::vector<float>     filteredscores;

        for (size_t i = 0; i < fastsam_result.boxes.size(); ++i) {
            const cv::Rect& box = fastsam_result.boxes[i];

            // Area filter
            if (box.area() < MINAREA)
                continue;

            filteredboxes.push_back(box);
            filteredscores.push_back(fastsam_result.scores[i]);
            filteredmasks.push_back(fastsam_result.masks[i]);
        }

        // Sort by score descending and keep top-K
        std::vector<size_t> indices(filteredboxes.size());
        std::iota(indices.begin(), indices.end(), 0);

        std::sort(indices.begin(), indices.end(),
                  [&](size_t i1, size_t i2) {
                      return filteredscores[i1] > filteredscores[i2];
                  });

        if (indices.size() > MAXOBJS)
            indices.resize(MAXOBJS);

        FastSAMResult final_result;
        for (size_t idx : indices) {
            final_result.boxes.push_back(filteredboxes[idx]);
            final_result.scores.push_back(filteredscores[idx]);
            final_result.masks.push_back(filteredmasks[idx]);
        }

        fastsam_result = std::move(final_result);

    } catch (const std::exception& e) {
        std::cerr << "FastSAM segmentation error: " << e.what() << std::endl;
        fastsam_result.masks.clear();
        fastsam_result.boxes.clear();
        fastsam_result.scores.clear();
    }
}

void TrackingState::detectHands()
{
    try {
        if (mediapipe_utils) {
            // 1. Send the current frame to the background AI thread
            mediapipe_utils->Update(current_frame);

            // 2. Retrieve the most recent landmarks without blocking the loop
            hand_landmarks = mediapipe_utils->GetLatestLandmarks();

            // 3. Get the smoothed index tip using the new width/height signature
            smoothed_index_tip = mediapipe_utils->GetSmoothedIndexTip(current_frame.cols, current_frame.rows);
        }
    } catch (const std::exception& e) {
        std::cerr << "MediaPipe hand detection error: " << e.what() << std::endl;
        hand_landmarks.clear();
        smoothed_index_tip = std::nullopt;
    }
}

Eigen::Vector3f TrackingState::pixel2DTo3D(const cv::Point2f& pixel, float depth_meters)
{
    // Get camera intrinsics
    float fx = camera_matrix.at<double>(0, 0);
    float fy = camera_matrix.at<double>(1, 1);
    float cx = camera_matrix.at<double>(0, 2);
    float cy = camera_matrix.at<double>(1, 2);

    // Convert pixel to 3D using pinhole camera model
    // x = (u - cx) * Z / fx
    // y = (v - cy) * Z / fy
    // z = Z
    float x = (pixel.x - cx) * depth_meters / fx;
    float y = (pixel.y - cy) * depth_meters / fy;
    float z = depth_meters;

    return Eigen::Vector3f(x, y, z);
}

void TrackingState::calculateObjectScores() {
    scored_objects.clear();
    
    // Need depth map, index tip, and objects
    if (state_command.depth_map.empty() || 
        !smoothed_index_tip.has_value() || 
        fastsam_result.boxes.empty()) {
        return;
    }
    
    // Check if we have valid scaling factor from AprilTag
    if (state_command.scaling_factor <= 0.0f) {
        std::cerr << "Warning: Invalid scaling factor, skipping score calculation" << std::endl;
        return;
    }
    
    // Verify depth map is CV_8U
    if (state_command.depth_map.type() != CV_8U) {
        std::cerr << "Warning: Depth map is not CV_8U, got type " << state_command.depth_map.type() << std::endl;
        return;
    }
    
    // Get AprilTag bounding box if detected
    cv::Rect apriltag_bbox;
    bool has_apriltag = false;
    if (state_command.apriltag_detected) {
        try {
            std::vector<TagDetection> detections = apriltag_utils->get_tags(current_frame);
            if (!detections.empty()) {
                const auto& detection = detections[0];
                if (detection.corners.size() == 4) {
                    float min_x = detection.corners[0].x;
                    float max_x = detection.corners[0].x;
                    float min_y = detection.corners[0].y;
                    float max_y = detection.corners[0].y;
                    
                    for (const auto& corner : detection.corners) {
                        min_x = std::min(min_x, corner.x);
                        max_x = std::max(max_x, corner.x);
                        min_y = std::min(min_y, corner.y);
                        max_y = std::max(max_y, corner.y);
                    }
                    
                    int margin = 20;
                    apriltag_bbox = cv::Rect(
                        std::max(0, int(min_x - margin)),
                        std::max(0, int(min_y - margin)),
                        int(max_x - min_x + 2 * margin),
                        int(max_y - min_y + 2 * margin)
                    );
                    has_apriltag = true;
                }
            }
        } catch (const std::exception& e) {
            std::cerr << "Error getting AprilTag bbox: " << e.what() << std::endl;
        } catch (...) {
            std::cerr << "Unknown error getting AprilTag bbox" << std::endl;
        }
    }
    
    // Get index fingertip position
    cv::Point tip_pixel = smoothed_index_tip.value();
    int tip_x = std::max(0, std::min(tip_pixel.x, state_command.depth_map.cols - 1));
    int tip_y = std::max(0, std::min(tip_pixel.y, state_command.depth_map.rows - 1));
    
    float tip_depth_meters = 0.0f;
    try {
        // Read as uint8 (0-255 relative depth)
        uint8_t tip_depth_uint8 = state_command.depth_map.at<uint8_t>(tip_y, tip_x);
        float tip_depth_normalized = static_cast<float>(tip_depth_uint8) / 255.0f;
        
        // Convert inverse depth to metric depth
        // MiDaS outputs: output = (1.0 / pred) normalized to [0-255]
        // So: pred_value = 1.0 / (normalized + epsilon)
        tip_depth_meters = (1.0f / (tip_depth_normalized + 1e-6f)) * state_command.scaling_factor;
        
    } catch (const cv::Exception& e) {
        std::cerr << "Error accessing depth map at fingertip: " << e.what() << std::endl;
        return;
    }
    
    // First pass: Calculate raw distances
    std::vector<float> raw_distances;
    std::vector<ScoredObject> temp_objects;
    
    for (size_t i = 0; i < fastsam_result.boxes.size(); ++i) {
        try {
            const auto& box = fastsam_result.boxes[i];
            
            // Skip if overlaps with AprilTag
            if (has_apriltag) {
                cv::Rect intersection = box & apriltag_bbox;
                float overlap_ratio = float(intersection.area()) / float(box.area());
                if (overlap_ratio > 0.3f) {
                    continue;
                }
            }
            
            // Calculate object center
            cv::Point2f center_2d(box.x + box.width / 2.0f, box.y + box.height / 2.0f);
            int obj_x = std::max(0, std::min(int(center_2d.x), state_command.depth_map.cols - 1));
            int obj_y = std::max(0, std::min(int(center_2d.y), state_command.depth_map.rows - 1));
            
            float obj_depth_meters = 0.0f;
            try {
                // Read as uint8 (0-255 relative depth)
                uint8_t obj_depth_uint8 = state_command.depth_map.at<uint8_t>(obj_y, obj_x);
                float obj_depth_normalized = static_cast<float>(obj_depth_uint8) / 255.0f;
                
                // Convert inverse depth to metric depth
                obj_depth_meters = (1.0f / (obj_depth_normalized + 1e-6f)) * state_command.scaling_factor;
                
            } catch (const cv::Exception& e) {
                std::cerr << "Error accessing depth at object " << i << ": " << e.what() << std::endl;
                continue;
            }
            
            
            float fx = camera_matrix.at<double>(0, 0);
            float fy = camera_matrix.at<double>(1, 1);
            float cx = camera_matrix.at<double>(0, 2);
            float cy = camera_matrix.at<double>(1, 2);

            // Convert to 3D coordinates
            float tip_x_3d = (tip_pixel.x - cx) * tip_depth_meters / fx;
            float tip_y_3d = (tip_pixel.y - cy) * tip_depth_meters / fy;
            float tip_z_3d = tip_depth_meters;

            float obj_x_3d = (center_2d.x - cx) * obj_depth_meters / fx;
            float obj_y_3d = (center_2d.y - cy) * obj_depth_meters / fy;
            float obj_z_3d = obj_depth_meters;

            float dx = tip_x_3d - obj_x_3d;
            float dy = tip_y_3d - obj_y_3d;
            float dz = tip_z_3d - obj_z_3d;

            // Always use 3D distance - no penalty, no special cases
            float dist_meters = std::sqrt(dx * dx + dy * dy + dz * dz);

                        
            
            // Create temporary scored object
            ScoredObject scored_obj;
            scored_obj.object_id = i;
            scored_obj.bbox = box;
            scored_obj.center_2d = center_2d;
            scored_obj.depth_meters = obj_depth_meters;
            scored_obj.position_3d = Eigen::Vector3f(
                (center_2d.x - camera_matrix.at<double>(0, 2)) * obj_depth_meters / camera_matrix.at<double>(0, 0),
                (center_2d.y - camera_matrix.at<double>(1, 2)) * obj_depth_meters / camera_matrix.at<double>(1, 1),
                obj_depth_meters
            );
            scored_obj.distance_to_tip_meters = dist_meters;
            
            temp_objects.push_back(scored_obj);
            raw_distances.push_back(-dist_meters);
            
        } catch (const std::exception& e) {
            std::cerr << "Error processing object " << i << ": " << e.what() << std::endl;
            continue;
        }
    }
    
    if (temp_objects.empty()) {
        return;
    }
    
    // SOFTMAX NORMALIZATION
    try {
        std::vector<float> exp_scores;
        float sum_exp = 0.0f;
        
        float max_dist = *std::max_element(raw_distances.begin(), raw_distances.end());
        const float TEMPERATURE = 10.0f;
        
        for (float dist : raw_distances) {
            float exp_score = std::exp((dist - max_dist) / TEMPERATURE);
            exp_scores.push_back(exp_score);
            sum_exp += exp_score;
        }
        
        for (size_t i = 0; i < temp_objects.size(); ++i) {
            temp_objects[i].reach_score = exp_scores[i] / sum_exp;
            scored_objects.push_back(temp_objects[i]);
        }
        
        std::sort(scored_objects.begin(), scored_objects.end(),
                  [](const ScoredObject& a, const ScoredObject& b) {
                      return a.reach_score > b.reach_score;
                  });
                  
    } catch (const std::exception& e) {
        std::cerr << "Error during softmax normalization: " << e.what() << std::endl;
        scored_objects.clear();
        return;
    }
    
    // Debug output
    if (!scored_objects.empty() && frame_count % 30 == 0) {
        try {
            std::cout << "Object Reach Scores (Softmax):" << std::endl;
            std::cout << "  Tip pos: (" << tip_pixel.x << "," << tip_pixel.y 
                      << ") depth=" << std::fixed << std::setprecision(2) << tip_depth_meters << "m" << std::endl;
            if (has_apriltag) {
                std::cout << "  AprilTag bbox: " << apriltag_bbox << " (excluded)" << std::endl;
            }
            
            float score_sum = 0.0f;
            for (size_t i = 0; i < std::min(scored_objects.size(), size_t(5)); ++i) {
                const auto& obj = scored_objects[i];
                std::cout << "  Obj" << obj.object_id 
                          << ": Score=" << std::fixed << std::setprecision(3) << obj.reach_score
                          << ", Dist=" << std::setprecision(2) << obj.distance_to_tip_meters << "m"
                          << ", Depth=" << obj.depth_meters << "m"
                          << std::endl;
                score_sum += obj.reach_score;
            }
            std::cout << "  Score sum: " << std::setprecision(3) << score_sum << std::endl;
        } catch (...) {
            // Silently ignore debug output errors
        }
    }
}


void TrackingState::updateTracking()
{
    if (state_command.apriltag_detected) {
        if (!tracking_active) {
            std::cout << "Tracking activated! Tag ID: " << state_command.apriltag_id << std::endl;
        }
        tracking_active = true;
        frames_without_detection = 0;
    } else if (frames_without_detection > MAX_FRAMES_WITHOUT_DETECTION) {
        if (tracking_active) {
            std::cout << "Tracking lost after " << frames_without_detection << " frames" << std::endl;
        }
        tracking_active = false;
    }
}

void TrackingState::drawHandSkeleton(const std::vector<utils::HandPoint>& landmarks)
{
    int framewidth = current_frame.cols;
    int frameheight = current_frame.rows;

    // Define hand connections
    const std::vector<std::pair<int, int>> connections = {
        {0, 1}, {1, 2}, {2, 3}, {3, 4},     // Thumb
        {0, 5}, {5, 6}, {6, 7}, {7, 8},     // Index
        {0, 9}, {9, 10}, {10, 11}, {11, 12}, // Middle
        {0, 13}, {13, 14}, {14, 15}, {15, 16}, // Ring
        {0, 17}, {17, 18}, {18, 19}, {19, 20}, // Pinky
        {5, 9}, {9, 13}, {13, 17}           // Palm
    };

    // Draw connections
    for (const auto& connection : connections) {
        int idx1 = connection.first;
        int idx2 = connection.second;

        if (idx1 < landmarks.size() && idx2 < landmarks.size()) {
            cv::Point pt1(landmarks[idx1].x * framewidth, landmarks[idx1].y * frameheight);
            cv::Point pt2(landmarks[idx2].x * framewidth, landmarks[idx2].y * frameheight);
            cv::line(current_frame, pt1, pt2, cv::Scalar(0, 255, 0), 2);
        }
    }

    // Draw landmarks
    for (size_t i = 0; i < landmarks.size(); ++i) {
        cv::Point pt(landmarks[i].x * framewidth, landmarks[i].y * frameheight);

        // Color code by finger
        cv::Scalar color;
        if (i == 0) {
            color = cv::Scalar(255, 0, 0);  // Wrist - blue
        } else if (i >= 1 && i <= 4) {
            color = cv::Scalar(255, 255, 0);  // Thumb - cyan
        } else if (i >= 5 && i <= 8) {
            color = cv::Scalar(0, 255, 255);  // Index - yellow
        } else if (i >= 9 && i <= 12) {
            color = cv::Scalar(255, 0, 255);  // Middle - magenta
        } else if (i >= 13 && i <= 16) {
            color = cv::Scalar(128, 0, 255);  // Ring - purple
        } else {
            color = cv::Scalar(0, 128, 255);  // Pinky - orange
        }

        cv::circle(current_frame, pt, 5, color, -1);
        cv::circle(current_frame, pt, 6, cv::Scalar(255, 255, 255), 1);
    }
}

void TrackingState::visualizeTracking()
{
    cv::Mat overlay = current_frame.clone();

    // Draw FastSAM segmentation masks with scores
    if (!fastsam_result.masks.empty()) {
        static std::vector<cv::Scalar> colors;

        if (colors.size() != fastsam_result.masks.size()) {
            colors.clear();
            for (size_t i = 0; i < fastsam_result.masks.size(); ++i) {
                cv::Scalar color(rand() % 200 + 55, rand() % 200 + 55, rand() % 200 + 55);
                colors.push_back(color);
            }
        }

        for (size_t i = 0; i < fastsam_result.masks.size(); ++i) {
            const auto& mask = fastsam_result.masks[i];
            const auto& box = fastsam_result.boxes[i];

            // Create colored mask
            // cv::Mat coloredmask = cv::Mat::zeros(mask.size(), CV_8UC3);
            // coloredmask.setTo(colors[i % colors.size()], mask);
            // cv::addWeighted(overlay, 1.0, coloredmask, 0.3, 0, overlay);

            // Highlight target object (highest score)
            cv::Scalar boxcolor = colors[i % colors.size()];
            int thickness = 2;

            if (!scored_objects.empty() && smoothed_index_tip.has_value()) {
                if (scored_objects[0].object_id == i) {
                    boxcolor = cv::Scalar(0, 255, 0);  // Green for target
                    thickness = 4;
                }
            }

            cv::rectangle(overlay, box, boxcolor, thickness);

            // Draw label with score
            std::string label;
            if (!scored_objects.empty() && smoothed_index_tip.has_value()) {
                // Find this object in scored list
                auto it = std::find_if(scored_objects.begin(), scored_objects.end(),
                                       [i](const ScoredObject& obj) { return obj.object_id == i; });

                if (it != scored_objects.end()) {
                    label = cv::format("Obj%zu %.2f S%.2f", i, fastsam_result.scores[i], it->reach_score);
                } else {
                    label = cv::format("Obj%zu %.2f", i, fastsam_result.scores[i]);
                }
            } else {
                label = cv::format("Obj%zu %.2f", i, fastsam_result.scores[i]);
            }

            int baseline = 0;
            cv::Size textsize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
            cv::rectangle(overlay,
                          cv::Point(box.x, box.y - textsize.height - 5),
                          cv::Point(box.x + textsize.width, box.y),
                          boxcolor, -1);
            cv::putText(overlay, label, cv::Point(box.x, box.y - 5),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
        }

        cv::addWeighted(current_frame, 0.6, overlay, 0.4, 0, current_frame);
    }

    // Draw hand skeletons
    for (const auto& [handid, landmarks] : hand_landmarks) {
        drawHandSkeleton(landmarks);
    }

    // Draw smoothed index fingertip with line to target
    if (smoothed_index_tip.has_value()) {
        cv::Point tip = smoothed_index_tip.value();

        // Draw line to target object
        if (!scored_objects.empty()) {
            const auto& target = scored_objects[0];
            cv::line(current_frame, tip,
                     cv::Point(target.center_2d.x, target.center_2d.y),
                     cv::Scalar(0, 255, 0), 2);
        }

        // Draw fingertip marker
        cv::circle(current_frame, tip, 12, cv::Scalar(0, 0, 255), -1);
        cv::circle(current_frame, tip, 14, cv::Scalar(255, 255, 255), 2);

        // Draw crosshair
        int crosssize = 20;
        cv::line(current_frame,
                 cv::Point(tip.x - crosssize, tip.y),
                 cv::Point(tip.x + crosssize, tip.y),
                 cv::Scalar(0, 255, 0), 2);
        cv::line(current_frame,
                 cv::Point(tip.x, tip.y - crosssize),
                 cv::Point(tip.x, tip.y + crosssize),
                 cv::Scalar(0, 255, 0), 2);

        cv::putText(current_frame, "Index Tip",
                    cv::Point(tip.x + 20, tip.y - 20),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 255), 2);
    }

    // Draw status info
    std::string status = tracking_active ? "TRACKING ACTIVE" : "SEARCHING FOR TAG";
    cv::Scalar statuscolor = tracking_active ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 165, 255);
    cv::putText(current_frame, status, cv::Point(10, 30),
                cv::FONT_HERSHEY_SIMPLEX, 1.0, statuscolor, 2);

    int yoffset = 70;

    if (state_command.apriltag_detected) {
        std::string taginfo = cv::format("Tag ID: %d | SF: %.3f",
                                          state_command.apriltag_id,
                                          state_command.scaling_factor);
        cv::putText(current_frame, taginfo, cv::Point(10, yoffset),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);
        yoffset += 30;
    }

    if (!scored_objects.empty() && smoothed_index_tip.has_value()) {
        const auto& target = scored_objects[0];
        std::string targetinfo = cv::format("Target: Obj%d | Score: %.2f | Dist: %.2fm",
                                              target.object_id,
                                              target.reach_score,
                                              target.distance_to_tip_meters);
        cv::putText(current_frame, targetinfo, cv::Point(10, yoffset),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);
        yoffset += 30;
    }

    if (!fastsam_result.masks.empty()) {
        std::string objcount = cv::format("Objects: %zu", fastsam_result.masks.size());
        cv::putText(current_frame, objcount, cv::Point(10, yoffset),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 200, 0), 2);
        yoffset += 30;
    }

    if (!hand_landmarks.empty()) {
        std::string handcount = cv::format("Hands: %zu", hand_landmarks.size());
        cv::putText(current_frame, handcount, cv::Point(10, yoffset),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 255), 2);
    }

    // Draw depth map preview
    if (!state_command.depth_map.empty()) {
        cv::Mat depthvis;
        cv::normalize(state_command.depth_map, depthvis, 0, 255, cv::NORM_MINMAX);
        depthvis.convertTo(depthvis, CV_8U);
        cv::applyColorMap(depthvis, depthvis, cv::COLORMAP_JET);

        int previewsize = 200;
        int margin = 10;

        if (current_frame.cols > previewsize + margin &&
            current_frame.rows > previewsize + margin) {
            cv::Mat roi = current_frame(cv::Rect(
                current_frame.cols - previewsize - margin,
                current_frame.rows - previewsize - margin,
                previewsize, previewsize));

            cv::Mat resizeddepth;
            cv::resize(depthvis, resizeddepth, roi.size());
            resizeddepth.copyTo(roi);

            cv::putText(current_frame, "Depth Map",
                        cv::Point(current_frame.cols - previewsize - margin,
                                  current_frame.rows - previewsize - margin - 10),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
        }
    }

    // Draw tracking indicator
    if (tracking_active) {
        cv::circle(current_frame, cv::Point(current_frame.cols - 30, 30),
                   15, cv::Scalar(0, 255, 0), -1);
        cv::putText(current_frame, "LOCK",
                    cv::Point(current_frame.cols - 100, 35),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);
    } else {
        cv::circle(current_frame, cv::Point(current_frame.cols - 30, 30),
                   15, cv::Scalar(0, 165, 255), -1);
        cv::putText(current_frame, "SEARCH",
                    cv::Point(current_frame.cols - 120, 35),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 165, 255), 2);
    }

    // Instructions
    cv::putText(current_frame, "Press 'i' for IDLE | 'q' to QUIT",
                cv::Point(10, current_frame.rows - 20),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
}
