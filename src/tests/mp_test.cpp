#include <iostream>
#include <chrono>
#include <opencv2/opencv.hpp>
#include "utils/mp_utils.hpp"

int main() {
  // ---------- Initialize MediaPipe hand landmarker ----------
  // Now defaults to LIVE_STREAM and GPU internally
  utils::HandLandmarkerMP hand_mp;

  // ---------- Open camera ----------
  cv::VideoCapture cap(2, cv::CAP_V4L2);
  if (!cap.isOpened()) {
    std::cerr << "ERROR: Failed to open camera\n";
    return -1;
  }

  // Set resolution to 640x480 for better performance
  cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
  cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);

  cv::namedWindow("MP Test", cv::WINDOW_AUTOSIZE);

  auto p_time = std::chrono::high_resolution_clock::now();

  // ---------- Main loop ----------
  while (true) {
    auto start_time = std::chrono::high_resolution_clock::now();

    cv::Mat frame;
    cap >> frame;
    if (frame.empty()) {
      std::cerr << "ERROR: Empty frame\n";
      break;
    }

    // 1. Send frame to MediaPipe (Async - returns immediately)
    hand_mp.Update(frame);

    // 2. Get the latest available landmarks (Non-blocking)
    utils::HandLandmarks hands = hand_mp.GetLatestLandmarks();

    // Draw landmarks
    for (const auto& [hand_id, landmarks] : hands) {
      for (const auto& pt : landmarks) {
        int x = static_cast<int>(pt.x * frame.cols);
        int y = static_cast<int>(pt.y * frame.rows);
        cv::circle(frame, {x, y}, 3, {0, 255, 0}, -1);
      }
    }

    // 3. Get smoothed tip (Pass width and height as required by new header)
    auto tip = hand_mp.GetSmoothedIndexTip(frame.cols, frame.rows);
    if (tip) {
      cv::circle(frame, *tip, 6, {0, 0, 255}, -1);
    }

    // Timing calculation
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> frame_duration = end_time - start_time;
    
    // FPS based on loop frequency
    std::chrono::duration<double> total_duration = end_time - p_time;
    double fps = 1.0 / total_duration.count();
    p_time = end_time;

    // Overlay stats
    std::string stats = cv::format("Loop: %.2f ms | FPS: %.1f", frame_duration.count(), fps);
    cv::putText(frame, stats, {10, 30}, cv::FONT_HERSHEY_SIMPLEX, 0.7, {255, 255, 255}, 2);
    cv::putText(frame, stats, {10, 30}, cv::FONT_HERSHEY_SIMPLEX, 0.7, {0, 0, 0}, 1);

    cv::imshow("MP Test", frame);

    int key = cv::waitKey(1);
    if (key == 27 || key == 'q' || key == 'Q') {
      break;
    }
  }

  return 0;
}