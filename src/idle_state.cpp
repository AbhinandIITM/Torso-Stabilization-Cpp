#include "idle_state.hpp"
#include "utils/mp_utils.h"
#include "utils/midas_utils.hpp" // Assuming this is the correct header for MidasUtils
#include "utils/apriltag_utils.hpp"
#include "utils/IMU_server_utils.hpp"

#include <iostream>
#include <nlohmann/json.hpp>


IdleState::IdleState(StateCommand& statecommand,
                   cv::VideoCapture& cap,
                   const std::string& calib_path,
                   const std::string& midas_model_path)
    : SC(statecommand),
      cap(cap),
      frame_count(0),
      plot_interval(30),
      // Initialize apriltag directly in the initializer list
      apriltag(calib_path, "tag36h11", 0.05)
{
    // --- Load Calibration ---
    cv::FileStorage fs(calib_path, cv::FileStorage::READ);
    if (!fs.isOpened()) {
        std::cerr << "Failed to open calibration file at: " << calib_path << std::endl;
        // You should probably throw an exception here
    }
    cv::Mat camMatrix;
    fs["camMatrix"] >> camMatrix;
    fx = camMatrix.at<double>(0,0);
    fy = camMatrix.at<double>(1,1);
    cx = camMatrix.at<double>(0,2);
    cy = camMatrix.at<double>(1,2);
    fs.release();
    cam_intrinsics = {fx, fy, cx, cy};

    // --- Initialize Utilities ---
    pipe_utils = MediapipeUtils();

    // Initialize the depth estimator (assuming MidasUtils is the class name
    // and it takes (path, use_cuda) in its constructor)
    bool use_cuda = true;
    depth_estimator = std::make_unique<MidasUtils>(midas_model_path, use_cuda);

    // --- Initialize and run IMU Server in a separate thread ---
    imu_server = std::make_unique<IMUServer>(8001);
    server_thread = std::thread(&IMUServer::run, imu_server.get());

    // --- Initialize Open3D Visualization ---
    vis = std::make_shared<open3d::visualization::Visualizer>();
    vis->CreateVisualizerWindow("3D IMU Trajectory", 800, 600);
    trajectory = std::make_shared<open3d::geometry::LineSet>();
}

// --- DESTRUCTOR ---
IdleState::~IdleState() {
    // This destructor ensures a clean shutdown.
    if (imu_server) {
        imu_server->stop();
    }
    if (server_thread.joinable()) {
        server_thread.join();
    }
    if (vis) {
        vis->DestroyVisualizerWindow();
    }
}

// --- RUN METHOD ---
void IdleState::run() {
    while (cap.isOpened()) {
        cv::Mat frame;
        if (!cap.read(frame)) continue;

        cv::Mat rgb_frame;
        cv::cvtColor(frame, rgb_frame, cv::COLOR_BGR2RGB);
        SC.RGBframe = rgb_frame;

        // Depth Estimation
        // Use ->getDepthMap() since depth_estimator is std::unique_ptr
        cv::Mat depth_map = depth_estimator->getDepthMap(rgb_frame);
        cv::resize(depth_map, depth_map, frame.size());
        SC.depth_map = depth_map;

        // Apriltag Detection and Scaling
        // Use ->get_tags() since apriltag is now a value type
        auto tags = apriltag.get_tags(frame);
        if (!tags.empty()) {
            double scaling_factor;
            apriltag.get_scaling_factor(tags, frame, depth_map, scaling_factor);
            SC.scaling_factor = scaling_factor;
            depth_map *= scaling_factor;
            SC.depth_map = depth_map;
        } else {
            SC.scaling_factor = -1;
        }

        // Get latest transform data from IMU server (thread-safe)
        nlohmann::json imu_json_data = imu_server->get_latest_transform();
        SC.imu_tf = ImuTransform(imu_json_data); // Use the new constructor

        // Check if the transform data is valid before using
        if (imu_json_data.contains("transform") && imu_json_data["transform"].contains("position")) {
            const auto& pos = imu_json_data["transform"]["position"];
            double x = pos.value("x", 0.0);
            double y = pos.value("y", 0.0);
            double z = pos.value("z", 0.0);
            positions.push_back({x, y, z});
            frame_count++;

            if (frame_count % plot_interval == 0) {
                updateOpen3DTrajectory();
            }
        }

        cv::imshow("depth-map", depth_map);
        cv::imshow("stream", frame);
        if (cv::waitKey(1) == 'q') break;

        vis->PollEvents();
        vis->UpdateRender();
    }
}

// --- UPDATE 3D TRAJECTORY ---
void IdleState::updateOpen3DTrajectory() {
    if (positions.empty()) return;

    trajectory->points_.clear();
    trajectory->lines_.clear();

    for (size_t i = 0; i < positions.size(); i++) {
        trajectory->points_.push_back(Eigen::Vector3d(positions[i][0], positions[i][1], positions[i][2]));
        if (i > 0) {
            trajectory->lines_.push_back(Eigen::Vector2i(i - 1, i));
        }
    }
    trajectory->colors_.clear();
    for (size_t i = 0; i < trajectory->lines_.size(); i++) {
        trajectory->colors_.push_back(Eigen::Vector3d(1.0, 0.0, 0.0)); // Red lines
    }

    // This logic is safer: try adding first, then just update.
    static bool geometry_added = false;
    if (!geometry_added) {
        vis->AddGeometry(trajectory);
        geometry_added = true;
    } else {
        vis->UpdateGeometry(trajectory);
    }
}