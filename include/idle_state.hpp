#pragma once

#include <opencv2/opencv.hpp>
#include <open3d/Open3D.h>
#include <memory>
#include <vector>
#include <thread>
#include <tuple>

#include "MediapipeUtils.hpp"
#include "MidasUtils.hpp"
#include "ApriltagUtils.hpp"  
#include "IMUServer.hpp"      


class StateCommand; 

class IdleState {
public:
    IdleState(StateCommand& statecommand, cv::VideoCapture& cap);
    ~IdleState(); // Destructor is crucial for clean shutdown

    void run();

private:
    void updateOpen3DTrajectory();

    // References to external objects
    StateCommand& SC;
    cv::VideoCapture& cap;

    // Visualization members
    std::shared_ptr<open3d::visualization::Visualizer> vis;
    std::shared_ptr<open3d::geometry::LineSet> trajectory;
    std::vector<std::vector<double>> positions;
    int frame_count;
    int plot_interval;

    // Camera calibration members
    double fx, fy, cx, cy;
    std::tuple<double, double, double, double> cam_intrinsics;

    // Utility class members
    std::unique_ptr<MediapipeUtils> pipe_utils;
    std::unique_ptr<MiDaSDepth> depth_estimator; // Corrected type name
    std::unique_ptr<ApriltagUtils> apriltag;

    // IMU Server and its dedicated thread
    std::unique_ptr<IMUServer> imu_server;
    std::thread server_thread;
};
