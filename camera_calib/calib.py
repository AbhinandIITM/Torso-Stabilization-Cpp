import cv2
import numpy as np
import glob
import os

# Configuration
IMAGE_DIR = "camera_calib"
SQUARES_X = 7
SQUARES_Y = 5
SQUARE_LENGTH = 0.03      # ← VERIFY THIS IS CORRECT
MARKER_LENGTH = 0.015     # ← VERIFY THIS IS CORRECT
ARUCO_DICT = cv2.aruco.DICT_6X6_250
MIN_CHARUCO_CORNERS = 15

# Prepare board
aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
board = cv2.aruco.CharucoBoard(
    (SQUARES_X, SQUARES_Y),
    SQUARE_LENGTH,
    MARKER_LENGTH,
    aruco_dict
)

# Load images
image_paths = sorted(glob.glob(os.path.join(IMAGE_DIR, "captured_*.jpg")))
assert len(image_paths) > 0, "No calibration images found"
print(f"[INFO] Found {len(image_paths)} images, processing...")

all_charuco_corners = []
all_charuco_ids = []
image_size = None
used_images = 0

# Detection loop
for img_path in image_paths:
    img = cv2.imread(img_path)
    if img is None:
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if image_size is None:
        image_size = gray.shape[::-1]

    corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict)
    if ids is None:
        print(f"  [SKIP] No markers in {os.path.basename(img_path)}")
        continue

    retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
        corners, ids, gray, board
    )

    if retval is None or retval < MIN_CHARUCO_CORNERS:
        print(f"  [SKIP] {retval or 0} corners in {os.path.basename(img_path)}")
        continue

    all_charuco_corners.append(charuco_corners)
    all_charuco_ids.append(charuco_ids)
    used_images += 1
    print(f"  [OK] {os.path.basename(img_path)}: {retval} corners")

assert used_images >= 5, f"Only {used_images} valid views (need ≥5)"
print(f"\n[INFO] Using {used_images} valid images for calibration\n")

# Initialize camera matrix
camera_matrix = np.array([
    [1000, 0, image_size[0]/2],
    [0, 1000, image_size[1]/2],
    [0, 0, 1]
], dtype=np.float32)
dist_coeffs = np.zeros((5, 1))

# Calibrate
rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
    charucoCorners=all_charuco_corners,
    charucoIds=all_charuco_ids,
    board=board,
    imageSize=image_size,
    cameraMatrix=camera_matrix,
    distCoeffs=dist_coeffs
)

# Print results
print(f"[RESULT] RMS Reprojection Error: {rms:.4f} pixels")
print(f"[RESULT] Camera Matrix:\n{camera_matrix}")
print(f"[RESULT] Focal lengths: fx={camera_matrix[0,0]:.2f}, fy={camera_matrix[1,1]:.2f}")
print(f"[RESULT] Principal point: cx={camera_matrix[0,2]:.2f}, cy={camera_matrix[1,2]:.2f}")
print(f"[RESULT] Distortion coefficients: {dist_coeffs.flatten()}\n")

# Validate
if rms > 1.0:
    print("[WARNING] High RMS error - consider more/better calibration images")

# Save to YAML
fs = cv2.FileStorage("camera_calibration_charuco.yaml", cv2.FILE_STORAGE_WRITE)
fs.write("image_width", image_size[0])
fs.write("image_height", image_size[1])
fs.write("square_length", SQUARE_LENGTH)
fs.write("marker_length", MARKER_LENGTH)
fs.write("camera_matrix", camera_matrix)
fs.write("distortion_coefficients", dist_coeffs)
fs.write("reprojection_error", rms)
fs.release()

print("[INFO] ✓ Calibration saved to camera_calibration_charuco.yaml")
