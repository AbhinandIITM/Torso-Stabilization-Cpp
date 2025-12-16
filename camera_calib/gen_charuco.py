import cv2 as cv
import numpy as np

# --- Define Parameters ---
# Number of squares in X and Y direction of the chessboard
squaresX = 7
squaresY = 5
# Real-world measurements (e.g., in millimeters or meters)
squareLength = 0.03 # 40 mm
markerLength = 0.015 # 30 mm (must be less than squareLength)
# Image size in pixels
imageSize = (800, 600)

# --- Generate Board ---
dictionary = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_6X6_100)
board = cv.aruco.CharucoBoard((squaresX, squaresY), squareLength, markerLength, dictionary)

# Generate and save the image
boardImage = board.generateImage(imageSize, marginSize=10)
cv.imwrite("ChArUco_Board_6x6_250.png", boardImage)

print(f"Generated ChArUco board with DICT_6X6_250 to ChArUco_Board_6x6_250.png")
