#include "idle_state.hpp"
#include "state_command.hpp"
#include "utils/IMU_server.hpp"
#include "utils/IMU_tracker.hpp"
#include "MediapipeUtils.hpp"
#include "MidasUtils.hpp"
#include "ApriltagUtils.hpp"

#include <iostream>

IdleState::IdleState(StateCommand& statecommand, 
                     cv::VideoCapture& cap,
                     const std::string& calib_path,
                     const std::string& midas_model_path)
    : SC(statecommand),
      cap(cap),
      frame_count(0),
      plot_interval(30)
{
    std::cout << "Initializing IdleState...\n";

    // --- Load Calibration ---
    cv::FileStorage fs(calib_path, cv::FileStorage::READ);
    if (!fs.isOpened()) {
        throw std::runtime_error("Failed to open calibration file: " + calib_path);
    }
    
    cv::Mat camMatrix;
    fs["camMatrix"] >> camMatrix;
    fx = camMatrix.at<double>(0,0);
    fy = camMatrix.at<double>(1,1);
    cx = camMatrix.at<double>(0,2);
    cy = camMatrix.at<double>(1,2);
    fs.release();
    cam_intrinsics = std::make_tuple(fx, fy, cx, cy);
    
    std::cout << "  Camera intrinsics loaded: fx=" << fx << ", fy=" << fy 
              << ", cx=" << cx << ", cy=" << cy << "\n";

    // --- Initialize Utilities ---
    pipe_utils = std::make_unique<MediapipeUtils>();
    std::cout << "  MediaPipe initialized\n";

    // Initialize depth estimator
    bool use_cuda = true;
    depth_estimator = std::make_unique<MiDaSDepth>(midas_model_path, use_cuda);
    std::cout << "  MiDaS depth estimator initialized\n";

    // Initialize AprilTag detector
    apriltag = std::make_unique<ApriltagUtils>(calib_path, "tag36h11", 0.05);
    std::cout << "  AprilTag detector initialized\n";

    // --- Initialize IMU Server ---
    imu_server = std::make_unique<utils::IMUServer>(8001);
    imu_server->start();
    std::cout << "  IMU Server started on port 8001\n";

    // --- Initialize Open3D Visualization ---
    vis = std::make_shared<open3d::visualization::Visualizer>();
    vis->CreateVisualizerWindow("3D IMU Trajectory", 800, 600);
    trajectory = std::make_shared<open3d::geometry::LineSet>();
    std::cout << "  Open3D visualization initialized\n";
    
    std::cout << "IdleState initialization complete!\n\n";
}

IdleState::~IdleState() {
    std::cout << "\nCleaning up IdleState...\n";
    
    if (imu_server) {
        imu_server->stop();
        std::cout << "  IMU Server stopped\n";
    }
    
    if (vis) {
        vis->DestroyVisualizerWindow();
        std::cout << "  Visualization window closed\n";
    }
    
    std::cout << "IdleState cleanup complete\n";
}

void IdleState::run() {
    std::cout << "Starting main processing loop...\n";
    std::cout << "Press 'q' to quit\n\n";
    
    while (cap.isOpened()) {
        cv::Mat frame;
        if (!cap.read(frame)) {
            std::cerr << "Failed to read frame\n";
            continue;
        }

        // Convert to RGB
        cv::Mat rgb_frame;
        cv::cvtColor(frame, rgb_frame, cv::COLOR_BGR2RGB);
        SC.RGBframe = rgb_frame;

        // Depth Estimation
        cv::Mat depth_map = depth_estimator->getDepthMap(rgb_frame);
        cv::resize(depth_map, depth_map, frame.size());
        SC.depth_map = depth_map;

        // AprilTag Detection and Scaling
        auto tags = apriltag->get_tags(frame);
        if (!tags.empty()) {
            double scaling_factor;
            apriltag->get_scaling_factor(tags, frame, depth_map, scaling_factor);
            SC.scaling_factor = scaling_factor;
            depth_map *= scaling_factor;
            SC.depth_map = depth_map;
        } else {
            SC.scaling_factor = -1;
        }

        // Get latest IMU transform (thread-safe)
        Eigen::Matrix4f imu_transform = imu_server->getTransform();
        SC.imu_transform = imu_transform;

        // Extract position from transform matrix
        double x = imu_transform(0, 3);
        double y = imu_transform(1, 3);
        double z = imu_transform(2, 3);
        
        positions.push_back({x, y, z});
        frame_count++;

        // Update 3D trajectory visualization
        if (frame_count % plot_interval == 0) {
            updateOpen3DTrajectory();
        }

        // Display frames
        cv::Mat depth_display;
        cv::normalize(depth_map, depth_display, 0, 255, cv::NORM_MINMAX, CV_8U);
        cv::applyColorMap(depth_display, depth_display, cv::COLORMAP_JET);
        
        cv::imshow("Depth Map", depth_display);
        cv::imshow("Camera Stream", frame);
        
        // Check for quit
        if (cv::waitKey(1) == 'q') {
            std::cout << "\nQuit requested by user\n";
            break;
        }

        // Update Open3D visualization
        vis->PollEvents();
        vis->UpdateRender();
    }
    
    cv::destroyAllWindows();
    std::cout << "Processing loop ended\n";
}

void IdleState::updateOpen3DTrajectory() {
    if (positions.empty()) return;

    trajectory->points_.clear();
    trajectory->lines_.clear();
    trajectory->colors_.clear();

    // Add all positions as points
    for (const auto& pos : positions) {
        trajectory->points_.push_back(Eigen::Vector3d(pos[0], pos[1], pos[2]));
    }

    // Connect consecutive points with lines
    for (size_t i = 1; i < positions.size(); i++) {
        trajectory->lines_.push_back(Eigen::Vector2i(i - 1, i));
        trajectory->colors_.push_back(Eigen::Vector3d(1.0, 0.0, 0.0)); // Red
    }

    // Add or update geometry
    static bool geometry_added = false;
    if (!geometry_added) {
        vis->AddGeometry(trajectory);
        geometry_added = true;
    } else {
        vis->UpdateGeometry(trajectory);
    }
}
