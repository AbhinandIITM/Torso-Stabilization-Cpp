#include <iostream>

#include <opencv2/opencv.hpp>

#include "utils/mp_utils.hpp"



int main() {
  // ---------- Initialize MediaPipe hand landmarker ----------
  utils::HandLandmarkerMP hand_mp(
      "mediapipe/models/hand_landmarker.task",
      /*max_num_hands=*/2);

  // ---------- Open camera ----------
  cv::VideoCapture cap(0);
  if (!cap.isOpened()) {
    std::cerr << "ERROR: Failed to open camera\n";
    return -1;
  }

  cv::namedWindow("MP Test", cv::WINDOW_AUTOSIZE);

  // ---------- Main loop ----------
  while (true) {
    cv::Mat frame;
    cap >> frame;
    if (frame.empty()) {
      std::cerr << "ERROR: Empty frame\n";
      break;
    }

    // Run MediaPipe
    utils::HandLandmarks hands = hand_mp.Detect(frame);

    // Draw landmarks (simple visualization)
    for (const auto& [hand_id, landmarks] : hands) {
      for (const auto& pt : landmarks) {
        int x = static_cast<int>(pt.x * frame.cols);
        int y = static_cast<int>(pt.y * frame.rows);
        cv::circle(frame, {x, y}, 3, {0, 255, 0}, -1);
      }
    }
    auto tip = hand_mp.GetSmoothedIndexTip(frame);

    if (tip) {
    cv::circle(frame, *tip, 6, {0, 0, 255}, -1);
    // tip->x, tip->y are stable pixel coords
    }

    cv::imshow("MP Test", frame);

    int key = cv::waitKey(1);
    if (key == 27 || key == 'q' || key == 'Q') {
      break;
    }
  }

  return 0;
}
