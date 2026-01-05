from ultralytics import FastSAM
import cv2
import logging
logging.getLogger('ultralytics').setLevel(logging.ERROR)
model = FastSAM('FastSAM-x.pt')
device = 'cuda'

cap = cv2.VideoCapture(2)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Use the 'texts' parameter to isolate human faces
    results = model.track(
        frame,
        device=device,
        retina_masks=True,
        imgsz=960,
        conf=0.4,
        iou=0.9,
        texts="bowl",  # This enables CLIP-based text prompting
        boxes = False
    )

    # Plot the results
    vis = results[0].plot()
    #print(len(results[0].boxes.conf))
    
    cv2.imshow("FastSAM - Human Face Segmentation", vis)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()