import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time

def main():
    # Global variables to track state
    latest_result = None
    actual_inference_ms = 0.0
    # Map to track start times for specific timestamps
    timestamp_map = {}

    def save_result(result, output_image, timestamp_ms):
        nonlocal latest_result, actual_inference_ms
        latest_result = result
        
        # Calculate actual inference latency
        if timestamp_ms in timestamp_map:
            start_time = timestamp_map.pop(timestamp_ms)
            actual_inference_ms = (time.perf_counter() - start_time) * 1000

    # 1. Setup the Hand Landmarker Task
    base_options = python.BaseOptions(model_asset_path='mediapipe/models/hand_landmarker.task')
    
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.LIVE_STREAM,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        result_callback=save_result
    )

    detector = vision.HandLandmarker.create_from_options(options)
    cap = cv2.VideoCapture(2) # Changed to 0 for default camera
        # In Python
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    p_time = 0

    print("Press 'q' to quit.")

    while cap.isOpened():
        success, frame = cap.read()
        if not success: break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # 2. Run detection with precise timing
        # We use perf_counter for high precision measurement
        frame_timestamp_ms = int(time.time() * 1000)
        timestamp_map[frame_timestamp_ms] = time.perf_counter()
        
        detector.detect_async(mp_image, frame_timestamp_ms)

        # 3. Visualization
        if latest_result and latest_result.hand_landmarks:
            for landmarks in latest_result.hand_landmarks:
                for lm in landmarks:
                    h, w, _ = frame.shape
                    x, y = int(lm.x * w), int(lm.y * h)
                    cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

        # 4. Loop Timing & FPS
        c_time = time.perf_counter()
        loop_time_ms = (c_time - p_time) * 1000
        fps = 1 / (c_time - p_time) if (c_time - p_time) > 0 else 0
        p_time = c_time

        # Display Loop Time (How fast the UI is running)
        cv2.putText(frame, f"Loop: {loop_time_ms:.1f}ms (FPS: {int(fps)})", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Display Actual Inference Time (The "True" processing time)
        cv2.putText(frame, f"Inference: {actual_inference_ms:.1f}ms", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow('MediaPipe Tasks - Hand Tracking', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    detector.close()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()