#include "StateCommand.hpp"

StateCommand state_command;
// Constructor
StateCommand::StateCommand()
  : RGBframe(),
    depth_map(),
    imu_tf(),
    camera_pos(Eigen::Matrix4f::Identity()),
    saved_3d_pos(Eigen::Matrix4f::Identity()),
    track_3d_pos(Eigen::Matrix4f::Identity())
{
    // state and scaling_factor are initialized in the header
}