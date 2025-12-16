// src/tracking_state.cpp

#include "include/tracking_state.hpp" 

#include <algorithm>
#include <cmath>

namespace torso_stabilization {

// ====================== KalmanFilterTrajectory ======================

KalmanFilterTrajectory::KalmanFilterTrajectory()
    : state_(Eigen::VectorXf::Zero(6)),
      covariance_(Eigen::MatrixXf::Identity(6, 6)),
      process_noise_(Eigen::MatrixXf::Identity(6, 6)),
      F_(Eigen::MatrixXf::Identity(6, 6)),
      H_(Eigen::MatrixXf::Zero(3, 6)),
      is_initialized_(false) {
  // Measurement matrix: observe position only (x, y, z)
  H_(0, 0) = 1.0f;
  H_(1, 1) = 1.0f;
  H_(2, 2) = 1.0f;

  // Simple process noise: small for position, larger for velocity
  const float q_pos = 0.01f;
  const float q_vel = 0.1f;
  process_noise_.setZero();
  process_noise_.block<3,3>(0,0) = q_pos * Eigen::Matrix3f::Identity();
  process_noise_.block<3,3>(3,3) = q_vel * Eigen::Matrix3f::Identity();
}

void KalmanFilterTrajectory::Initialize(const Eigen::Vector3f& initial_position,
                                        float dt) {
  state_.setZero();
  state_.head<3>() = initial_position;

  covariance_.setIdentity();
  covariance_.block<3,3>(0,0) *= 0.1f;  // position uncertainty
  covariance_.block<3,3>(3,3) *= 0.5f;  // velocity uncertainty

  F_.setIdentity();
  F_.block<3,3>(0,3) = dt * Eigen::Matrix3f::Identity();

  is_initialized_ = true;
}

void KalmanFilterTrajectory::Predict(float dt) {
  if (!is_initialized_) return;

  F_.setIdentity();
  F_.block<3,3>(0,3) = dt * Eigen::Matrix3f::Identity();

  state_ = F_ * state_;
  covariance_ = F_ * covariance_ * F_.transpose() + process_noise_;
}

void KalmanFilterTrajectory::Update(const Eigen::Vector3f& measurement,
                                    float measurement_noise) {
  if (!is_initialized_) return;

  Eigen::Vector3f z = measurement;
  Eigen::Vector3f y = z - H_ * state_;

  Eigen::Matrix3f R = measurement_noise * Eigen::Matrix3f::Identity();
  Eigen::Matrix3f S = H_ * covariance_ * H_.transpose() + R;
  Eigen::Matrix<float, 6, 3> K = covariance_ * H_.transpose() * S.inverse();

  state_ = state_ + K * y;

  Eigen::MatrixXf I = Eigen::MatrixXf::Identity(6, 6);
  covariance_ = (I - K * H_) * covariance_;
}

// ====================== ObjectScoringEngine ======================

ObjectScoringEngine::ObjectScoringEngine() = default;

ObjectScoringEngine::ObjectScoringEngine(const ScoringConfig& config)
    : config_(config) {}

std::vector<float> ObjectScoringEngine::ScoreObjects(
    const Eigen::Vector3f& hand_position,
    const Eigen::Vector3f& reach_direction,
    const std::vector<DetectedObject>& objects) const {
  std::vector<float> scores;
  scores.reserve(objects.size());
  for (const auto& obj : objects) {
    scores.push_back(ScoreSingleObject(hand_position, reach_direction, obj));
  }
  return scores;
}

float ObjectScoringEngine::ScoreSingleObject(
    const Eigen::Vector3f& hand_position,
    const Eigen::Vector3f& reach_direction,
    const DetectedObject& object) const {
  Eigen::Vector3f hand_to_object = object.center_3d - hand_position;
  float distance = hand_to_object.norm();
  float angle_score = ComputeAngleScore(reach_direction, hand_to_object);
  float distance_score = ComputeDistanceScore(distance);

  float score = config_.angle_weight * angle_score +
                config_.distance_weight * distance_score;
  return std::clamp(score, 0.0f, 1.0f);
}

float ObjectScoringEngine::ComputeAngleScore(
    const Eigen::Vector3f& reach_direction,
    const Eigen::Vector3f& hand_to_object) const {
  if (hand_to_object.norm() < 1e-6f) return 0.0f;

  Eigen::Vector3f dir_obj = hand_to_object.normalized();
  float cos_angle = reach_direction.dot(dir_obj);
  cos_angle = std::clamp(cos_angle, -1.0f, 1.0f);
  float angle_rad = std::acos(cos_angle);
  float angle_deg = angle_rad * 180.0f / static_cast<float>(M_PI);

  float cone = config_.reach_direction_angle_deg;
  if (angle_deg <= cone) {
    // Linear falloff inside cone
    return 1.0f - angle_deg / cone;
  }
  // Outside cone, quickly drop to 0
  return std::max(0.0f, 1.0f - (angle_deg - cone) / 90.0f);
}

float ObjectScoringEngine::ComputeDistanceScore(float distance_m) const {
  if (distance_m <= 0.0f) return 1.0f;
  if (distance_m >= config_.max_reach_distance_m) return 0.0f;

  float x = distance_m / config_.max_reach_distance_m;
  float s = 1.0f - x * x;  // quadratic falloff
  return std::clamp(s, 0.0f, 1.0f);
}

// ====================== TrajectoryAnalyzer ======================

TrajectoryAnalyzer::TrajectoryAnalyzer() = default;

TrajectoryAnalyzer::TrajectoryAnalyzer(const TrajectoryConfig& config)
    : config_(config) {}

void TrajectoryAnalyzer::AddHandPosition(const Eigen::Vector3f& position,
                                         int64_t timestamp_ns) {
  position_history_.push_back({position, timestamp_ns});
  PruneOldSamples(timestamp_ns);
}

bool TrajectoryAnalyzer::ComputeTrajectoryDirection(
    Eigen::Vector3f& out_direction) const {
  if (position_history_.size() < 2) return false;

  const auto& start = position_history_.front();
  const auto& end   = position_history_.back();
  Eigen::Vector3f disp = end.position - start.position;
  float dist_cm = disp.norm() * 100.0f;
  if (dist_cm < config_.min_movement_threshold_cm) return false;

  int64_t dt_ns = end.timestamp_ns - start.timestamp_ns;
  if (dt_ns <= 0) return false;
  float dt_s = static_cast<float>(dt_ns) / 1e9f;

  Eigen::Vector3f vel = disp / dt_s;
  float vel_cm_s = vel.norm() * 100.0f;
  if (vel_cm_s < config_.velocity_magnitude_threshold_cm_per_s) return false;

  out_direction = vel.normalized();
  return true;
}

Eigen::Vector3f TrajectoryAnalyzer::GetHandVelocity() const {
  if (position_history_.size() < 2) return Eigen::Vector3f::Zero();

  const auto& start = position_history_.front();
  const auto& end   = position_history_.back();
  int64_t dt_ns = end.timestamp_ns - start.timestamp_ns;
  if (dt_ns <= 0) return Eigen::Vector3f::Zero();

  float dt_s = static_cast<float>(dt_ns) / 1e9f;
  Eigen::Vector3f disp = end.position - start.position;
  return disp / dt_s;  // m/s
}

float TrajectoryAnalyzer::GetTotalMovement() const {
  if (position_history_.size() < 2) return 0.0f;
  float total = 0.0f;
  for (size_t i = 1; i < position_history_.size(); ++i) {
    total += (position_history_[i].position -
              position_history_[i - 1].position).norm();
  }
  return total * 100.0f;  // cm
}

void TrajectoryAnalyzer::PruneOldSamples(int64_t current_time_ns) {
  const int64_t window_ns =
      static_cast<int64_t>(config_.trajectory_window_ms * 1e6);
  while (!position_history_.empty() &&
         current_time_ns - position_history_.front().timestamp_ns > window_ns) {
    position_history_.pop_front();
  }
}

// ====================== TrackingState ======================

TrackingState::TrackingState()
    : current_state_(State::WAITING),
      kalman_filter_(std::make_unique<KalmanFilterTrajectory>()),
      trajectory_analyzer_(std::make_unique<TrajectoryAnalyzer>()),
      scoring_engine_(std::make_unique<ObjectScoringEngine>()),
      target_object_index_(-1),
      finalization_frame_count_(0),
      finalization_start_timestamp_ns_(0),
      last_focus_angle_deg_(0.0f) {}

TrackingState::TrackingState(const TrackingConfig& config)
    : current_state_(State::WAITING),
      config_(config),
      kalman_filter_(std::make_unique<KalmanFilterTrajectory>()),
      trajectory_analyzer_(
          std::make_unique<TrajectoryAnalyzer>(config.trajectory_config)),
      scoring_engine_(
          std::make_unique<ObjectScoringEngine>(config.scoring_config)),
      target_object_index_(-1),
      finalization_frame_count_(0),
      finalization_start_timestamp_ns_(0),
      last_focus_angle_deg_(0.0f) {}

TrackingState::State TrackingState::ProcessFrame(
    const HandFrame& hand_frame,
    const std::vector<DetectedObject>& objects,
    int64_t current_timestamp_ns) {
  switch (current_state_) {
    case State::WAITING:
      HandleWaiting(hand_frame);
      break;

    case State::TRACKING:
      HandleTracking(hand_frame, objects);
      break;

    case State::OBJECT_FINALIZING:
      HandleFinalizing(hand_frame, objects, current_timestamp_ns);
      break;

    case State::OBJECT_FINALIZED:
      // Remain finalized until external reset
      break;

    case State::RESET:
      Reset();
      break;
  }
  return current_state_;
}

void TrackingState::HandleWaiting(const HandFrame& hand_frame) {
  if (!hand_frame.detected || hand_frame.landmarks.empty()) return;

  const Eigen::Vector3f& wrist = hand_frame.landmarks[0].position;
  kalman_filter_->Initialize(wrist, 1.0f / 30.0f);
  trajectory_analyzer_->Reset();
  trajectory_analyzer_->AddHandPosition(wrist, hand_frame.timestamp_ns);

  current_state_ = State::TRACKING;
}

void TrackingState::HandleTracking(const HandFrame& hand_frame,
                                   const std::vector<DetectedObject>& objects) {
  if (!hand_frame.detected || hand_frame.landmarks.empty()) {
    current_state_ = State::WAITING;
    return;
  }

  const Eigen::Vector3f& wrist = hand_frame.landmarks[0].position;
  kalman_filter_->Predict(1.0f / 30.0f);
  kalman_filter_->Update(wrist);

  trajectory_analyzer_->AddHandPosition(wrist, hand_frame.timestamp_ns);

  Eigen::Vector3f dir;
  if (!trajectory_analyzer_->ComputeTrajectoryDirection(dir)) {
    // Not enough movement yet
    return;
  }
  current_reach_direction_ = dir;

  Eigen::Vector3f hand_pos = kalman_filter_->GetPosition();
  current_object_scores_ =
      scoring_engine_->ScoreObjects(hand_pos, current_reach_direction_, objects);

  int best_idx = FindBestScoredObject();
  if (best_idx >= 0 && current_object_scores_[best_idx] > config_.min_score_to_start_finalization) {
    target_object_index_ = best_idx;
    finalization_frame_count_ = 1;
    finalization_start_timestamp_ns_ = hand_frame.timestamp_ns;
    last_focus_angle_deg_ = 0.0f;
    current_state_ = State::OBJECT_FINALIZING;
  }
}

void TrackingState::HandleFinalizing(const HandFrame& hand_frame,
                                     const std::vector<DetectedObject>& objects,
                                     int64_t timestamp_ns) {
  if (!hand_frame.detected || hand_frame.landmarks.empty()) {
    // Lost hand → back to tracking
    current_state_ = State::TRACKING;
    target_object_index_ = -1;
    finalization_frame_count_ = 0;
    return;
  }

  const Eigen::Vector3f& wrist = hand_frame.landmarks[0].position;
  kalman_filter_->Predict(1.0f / 30.0f);
  kalman_filter_->Update(wrist);
  trajectory_analyzer_->AddHandPosition(wrist, hand_frame.timestamp_ns);

  Eigen::Vector3f dir;
  if (trajectory_analyzer_->ComputeTrajectoryDirection(dir)) {
    if (!IsTrajectoryConsistentWithTarget(dir, objects)) {
      // User changed direction → restart
      current_state_ = State::TRACKING;
      target_object_index_ = -1;
      finalization_frame_count_ = 0;
      return;
    }
    current_reach_direction_ = dir;
  }

  ++finalization_frame_count_;

  int64_t dt_ns = timestamp_ns - finalization_start_timestamp_ns_;
  float dt_ms = static_cast<float>(dt_ns) / 1e6f;

  if (finalization_frame_count_ >= config_.min_focus_frames &&
      dt_ms >= config_.object_finalization_duration_ms) {
    current_state_ = State::OBJECT_FINALIZED;
  }
}

int TrackingState::FindBestScoredObject() const {
  if (current_object_scores_.empty()) return -1;
  int best_idx = -1;
  float best_score = -1.0f;
  for (size_t i = 0; i < current_object_scores_.size(); ++i) {
    if (current_object_scores_[i] > best_score) {
      best_score = current_object_scores_[i];
      best_idx = static_cast<int>(i);
    }
  }
  return best_idx;
}

bool TrackingState::IsTrajectoryConsistentWithTarget(
    const Eigen::Vector3f& new_dir,
    const std::vector<DetectedObject>& objects) const {
  if (target_object_index_ < 0 ||
      target_object_index_ >= static_cast<int>(objects.size())) {
    return false;
  }

  float cos_angle = current_reach_direction_.dot(new_dir);
  cos_angle = std::clamp(cos_angle, -1.0f, 1.0f);
  float angle_deg =
      std::acos(cos_angle) * 180.0f / static_cast<float>(M_PI);

  return angle_deg <= config_.focus_angle_threshold_deg;
}

float TrackingState::GetFinalizationProgress() const {
  if (current_state_ != State::OBJECT_FINALIZING) return 0.0f;

  // Progress based on frames and time; use the more conservative.
  float frame_progress =
      static_cast<float>(finalization_frame_count_) /
      static_cast<float>(std::max(1, config_.min_focus_frames));
  return std::clamp(frame_progress, 0.0f, 1.0f);
}

Eigen::Vector3f TrackingState::GetSmoothedHandPosition() const {
  if (!kalman_filter_ || !kalman_filter_->IsInitialized())
    return Eigen::Vector3f::Zero();
  return kalman_filter_->GetPosition();
}

Eigen::Vector3f TrackingState::GetReachDirection() const {
  return current_reach_direction_;
}

const DetectedObject* TrackingState::GetTargetObject(
    const std::vector<DetectedObject>& objects) const {
  if (current_state_ != State::OBJECT_FINALIZED) return nullptr;
  if (target_object_index_ < 0 ||
      target_object_index_ >= static_cast<int>(objects.size())) {
    return nullptr;
  }
  return &objects[target_object_index_];
}

void TrackingState::Reset() {
  current_state_ = State::WAITING;
  target_object_index_ = -1;
  finalization_frame_count_ = 0;
  finalization_start_timestamp_ns_ = 0;
  last_focus_angle_deg_ = 0.0f;
  current_object_scores_.clear();
  current_reach_direction_.setZero();
  if (trajectory_analyzer_) trajectory_analyzer_->Reset();
}

}  // namespace torso_stabilization
