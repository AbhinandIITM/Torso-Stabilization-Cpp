#ifndef TORSO_STABILIZATION_TRACKING_STATE_HPP_
#define TORSO_STABILIZATION_TRACKING_STATE_HPP_

#include <Eigen/Dense>
#include <deque>
#include <memory>
#include <vector>

namespace torso_stabilization {

/**
 * @brief Represents a single hand landmark from MediaPipe
 */
struct HandLandmark {
  Eigen::Vector3f position;  // 3D position (x, y, z) in meters
  float visibility;          // Confidence score [0, 1]

  HandLandmark() : position(Eigen::Vector3f::Zero()), visibility(0.0f) {}
  HandLandmark(const Eigen::Vector3f& pos, float vis)
      : position(pos), visibility(vis) {}
};

/**
 * @brief Hand frame containing all landmarks from MediaPipe
 */
struct HandFrame {
  std::vector<HandLandmark> landmarks;  // 21 landmarks (MediaPipe format)
  float handedness;                      // Right hand probability [0, 1]
  bool detected;                         // Detection success flag
  int64_t timestamp_ns;                  // Frame timestamp in nanoseconds

  HandFrame()
      : handedness(0.0f), detected(false), timestamp_ns(0) {}
};

/**
 * @brief Detected object from FastSAM + depth estimation
 */
struct DetectedObject {
  int object_id;                    // Unique identifier
  Eigen::Vector3f center_3d;        // 3D world position (meters)
  Eigen::Vector2f center_2d;        // 2D image position (pixels)
  float depth;                       // Depth from camera (meters)
  float bounding_box_area;           // Bounding box area (pixels²)
  float segmentation_mask_area;      // Segmentation mask area (pixels²)

  DetectedObject()
      : object_id(-1),
        center_3d(Eigen::Vector3f::Zero()),
        center_2d(Eigen::Vector2f::Zero()),
        depth(0.0f),
        bounding_box_area(0.0f),
        segmentation_mask_area(0.0f) {}
};

/**
 * @brief Kalman filter for 3D hand trajectory smoothing
 * 
 * 6-state Kalman filter: [x, y, z, vx, vy, vz]
 * Provides smoothed position and velocity estimates
 */
class KalmanFilterTrajectory {
 public:
  KalmanFilterTrajectory();

  void Initialize(const Eigen::Vector3f& initial_position, float dt);
  void Predict(float dt);
  void Update(const Eigen::Vector3f& measurement,
              float measurement_noise = 0.01f);

  Eigen::Vector3f GetPosition() const { return state_.head<3>(); }
  Eigen::Vector3f GetVelocity() const { return state_.tail<3>(); }
  bool IsInitialized() const { return is_initialized_; }
  void Reset() { is_initialized_ = false; }

 private:
  Eigen::VectorXf state_;           // [x, y, z, vx, vy, vz]
  Eigen::MatrixXf covariance_;      // 6x6 covariance matrix
  Eigen::MatrixXf process_noise_;   // 6x6 process noise Q
  Eigen::MatrixXf F_;               // 6x6 state transition matrix
  Eigen::MatrixXf H_;               // 3x6 measurement matrix
  bool is_initialized_;
};

/**
 * @brief Object scoring engine for reach target selection
 * 
 * Scores objects based on:
 * - Angle between reach direction and hand-to-object vector
 * - Distance from hand to object
 */
class ObjectScoringEngine {
 public:
  struct ScoringConfig {
    float angle_weight = 0.6f;                      // Weight for angle score
    float distance_weight = 0.4f;                   // Weight for distance score
    float max_reach_distance_m = 0.8f;              // Maximum reach distance
    float reach_direction_angle_deg = 45.0f;        // Reach cone angle
    float min_hand_movement_cm = 2.0f;              // Minimum movement for direction
  };

  ObjectScoringEngine();
  explicit ObjectScoringEngine(const ScoringConfig& config);

  std::vector<float> ScoreObjects(
      const Eigen::Vector3f& hand_position,
      const Eigen::Vector3f& reach_direction,
      const std::vector<DetectedObject>& objects) const;

  float ScoreSingleObject(const Eigen::Vector3f& hand_position,
                          const Eigen::Vector3f& reach_direction,
                          const DetectedObject& object) const;

 private:
  float ComputeAngleScore(const Eigen::Vector3f& reach_direction,
                          const Eigen::Vector3f& hand_to_object) const;
  float ComputeDistanceScore(float distance_m) const;

  ScoringConfig config_;
};

/**
 * @brief Trajectory analyzer for computing reach direction
 * 
 * Analyzes hand movement over time to determine reaching intent
 */
class TrajectoryAnalyzer {
 public:
  struct TrajectoryConfig {
    int trajectory_window_frames = 10;              // Lookback window (frames)
    float trajectory_window_ms = 500.0f;            // Lookback window (milliseconds)
    float min_movement_threshold_cm = 2.0f;         // Ignore small movements
    float velocity_magnitude_threshold_cm_per_s = 5.0f;  // Minimum velocity
  };

  TrajectoryAnalyzer();
  explicit TrajectoryAnalyzer(const TrajectoryConfig& config);

  void AddHandPosition(const Eigen::Vector3f& position, int64_t timestamp_ns);
  bool ComputeTrajectoryDirection(Eigen::Vector3f& out_direction) const;
  Eigen::Vector3f GetHandVelocity() const;
  float GetTotalMovement() const;
  void Reset() { position_history_.clear(); }

 private:
  struct PositionSample {
    Eigen::Vector3f position;
    int64_t timestamp_ns;
  };

  void PruneOldSamples(int64_t current_time_ns);

  TrajectoryConfig config_;
  std::deque<PositionSample> position_history_;
};

/**
 * @brief Main tracking state machine
 * 
 * Orchestrates hand tracking, trajectory analysis, object scoring,
 * and target finalization for the torso stabilization system.
 */
class TrackingState {
 public:
  enum class State {
    WAITING,            // Waiting for hand detection
    TRACKING,           // Tracking hand, analyzing trajectory
    OBJECT_FINALIZING,  // Confirming target object (temporal confirmation)
    OBJECT_FINALIZED,   // Target object confirmed
    RESET               // Transitional reset state
  };

  struct TrackingConfig {
    TrajectoryAnalyzer::TrajectoryConfig trajectory_config;
    ObjectScoringEngine::ScoringConfig scoring_config;
    float object_finalization_duration_ms = 350.0f;  // Time to hold focus
    float focus_angle_threshold_deg = 20.0f;         // Max angle change during finalization
    int min_focus_frames = 8;                        // Minimum frames for finalization
    float min_score_to_start_finalization = 0.5f;    // Minimum score to start finalizing
  };

  TrackingState();
  explicit TrackingState(const TrackingConfig& config);

  // Main processing function - call each frame
  State ProcessFrame(const HandFrame& hand_frame,
                     const std::vector<DetectedObject>& objects,
                     int64_t current_timestamp_ns);

  // Getters
  State GetCurrentState() const { return current_state_; }
  Eigen::Vector3f GetSmoothedHandPosition() const;
  Eigen::Vector3f GetReachDirection() const;
  const std::vector<float>& GetObjectScores() const { return current_object_scores_; }
  float GetFinalizationProgress() const;
  const DetectedObject* GetTargetObject(const std::vector<DetectedObject>& objects) const;

  // Reset to initial state
  void Reset();

 private:
  void HandleWaiting(const HandFrame& hand_frame);
  void HandleTracking(const HandFrame& hand_frame,
                      const std::vector<DetectedObject>& objects);
  void HandleFinalizing(const HandFrame& hand_frame,
                        const std::vector<DetectedObject>& objects,
                        int64_t timestamp_ns);

  int FindBestScoredObject() const;
  bool IsTrajectoryConsistentWithTarget(
      const Eigen::Vector3f& new_dir,
      const std::vector<DetectedObject>& objects) const;

  State current_state_;
  TrackingConfig config_;

  std::unique_ptr<KalmanFilterTrajectory> kalman_filter_;
  std::unique_ptr<TrajectoryAnalyzer> trajectory_analyzer_;
  std::unique_ptr<ObjectScoringEngine> scoring_engine_;

  Eigen::Vector3f current_reach_direction_;
  std::vector<float> current_object_scores_;
  int target_object_index_;
  int finalization_frame_count_;
  int64_t finalization_start_timestamp_ns_;
  float last_focus_angle_deg_;
};

}  // namespace torso_stabilization

#endif  // TORSO_STABILIZATION_TRACKING_STATE_HPP_
