#ifndef TRACKINGSTATE_HPP
#define TRACKINGSTATE_HPP

#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <memory>
#include <string>
#include <optional>

#include "state_command.hpp"
#include "utils/mp_utils.hpp"
#include "utils/midas_utils.hpp"
#include "utils/apriltag_utils.hpp"
#include "utils/fastsam_utils.hpp"

/// @brief Struct to hold object information with reach score
struct ScoredObject
{
    int object_id;
    cv::Rect bbox;
    cv::Point2f center_2d;
    float depth_meters;
    Eigen::Vector3f position_3d;
    float distance_to_tip_meters;
    float reach_score; // 10^(-x) score
};

/**
 * @brief TrackingState - Active tracking using AprilTag, MiDaS, FastSAM and MediaPipe
 * 
 * Responsibilities:
 * - Continuously track AprilTag for pose estimation
 * - Use MiDaS for depth estimation
 * - Segment objects with FastSAM
 * - Detect and visualize hand skeletons with MediaPipe
 * - Calculate reach scores for objects
 * - Update 3D position of tracked objects
 * - Provide real-time tracking visualization
 */
class TrackingState
{
public:
    TrackingState(StateCommand state_command, cv::VideoCapture cap);

    /// Run tracking state - returns next state to transition to
    SystemState run();

private:
    StateCommand state_command;
    cv::VideoCapture cap;

    // References to shared components from StateCommand
    MiDaSDepth* midas_utils;
    ApriltagUtils* apriltag_utils;
    utils::HandLandmarkerMP* mediapipe_utils;
    cv::Mat camera_matrix;
    cv::Mat dist_coeffs;

    // Frame data
    cv::Mat current_frame;
    int frame_count; // Frame counter as member

    // FastSAM result
    FastSAMResult fastsam_result;

    // Hand landmarks from MediaPipe
    utils::HandLandmarks hand_landmarks;

    // Smoothed index fingertip
    std::optional<cv::Point> smoothed_index_tip;

    // Scored objects
    std::vector<ScoredObject> scored_objects;

    // Tracking state
    bool tracking_active;
    int frames_without_detection;
    static constexpr int MAX_FRAMES_WITHOUT_DETECTION = 30;

    // Helper methods
    void processFrame();
    bool detectAprilTag();
    void estimateDepth();
    void segmentObjects();
    void detectHands();
    void calculateObjectScores();
    Eigen::Vector3f pixel2DTo3D(const cv::Point2f& pixel, float depth_meters);
    void updateTracking();
    void visualizeTracking();
    void drawHandSkeleton(const std::vector<utils::HandPoint>& landmarks);
    bool checkExitCondition();
};

#endif // TRACKINGSTATE_HPP
