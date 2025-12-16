import cv2
import numpy as np
from dt_apriltags import Detector

# Load your camera calibration
fs = cv2.FileStorage("camera_calibration_charuco.yaml", cv2.FILE_STORAGE_READ)
camera_matrix = fs.getNode("camera_matrix").mat()
dist_coeffs = fs.getNode("distortion_coefficients").mat()
fs.release()

# Extract camera parameters
fx = camera_matrix[0, 0]
fy = camera_matrix[1, 1]
cx = camera_matrix[0, 2]
cy = camera_matrix[1, 2]

# AprilTag configuration
TAG_SIZE = 0.05  # 5 cm in meters
TAG_FAMILY = 'tag36h11'  # Good default choice

# Initialize detector
detector = Detector(
    families=TAG_FAMILY,
    nthreads=4,
    quad_decimate=2.0,
    quad_sigma=0.0,
    refine_edges=1,
    decode_sharpening=0.25,
    debug=0
)

# Open camera
cap = cv2.VideoCapture(2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print(f"[INFO] Using {TAG_FAMILY} with {TAG_SIZE}m tag size")
print("[INFO] Press 'q' to quit\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert to grayscale for detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect AprilTags
    detections = detector.detect(
        gray,
        estimate_tag_pose=True,
        camera_params=[fx, fy, cx, cy],
        tag_size=TAG_SIZE
    )
    
    # Process each detected tag
    for detection in detections:
        tag_id = detection.tag_id
        
        # Extract pose (rotation and translation)
        pose_R = detection.pose_R  # 3x3 rotation matrix
        pose_t = detection.pose_t  # 3x1 translation vector (x, y, z in meters)
        
        # Distance is the Z component (depth)
        distance = pose_t[2][0]  # in meters
        
        # Also compute Euclidean distance from camera origin
        distance_3d = np.linalg.norm(pose_t)
        
        # Draw detection on frame
        corners = detection.corners.astype(int)
        cv2.polylines(frame, [corners], True, (0, 255, 0), 2)
        
        # Draw center
        center = tuple(detection.center.astype(int))
        cv2.circle(frame, center, 5, (0, 0, 255), -1)
        
        # Draw coordinate axes (X=red, Y=green, Z=blue)
        axis_length = TAG_SIZE / 2
        rvec, _ = cv2.Rodrigues(pose_R)
        cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, pose_t, axis_length, 3)
        
        # Display info
        info_text = f"ID:{tag_id} Z:{distance:.2f}m ({distance*100:.1f}cm)"
        cv2.putText(frame, info_text, (center[0] - 50, center[1] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        
        # Print to console
        x, y, z = pose_t[0][0], pose_t[1][0], pose_t[2][0]
        print(f"Tag ID {tag_id}: Distance Z={z:.3f}m, 3D={distance_3d:.3f}m, XYZ=({x:.3f}, {y:.3f}, {z:.3f})")
    
    # Show frame
    cv2.imshow('AprilTag Distance Estimation', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
