from ultralytics import FastSAM
import cv2
import torch
import numpy as np
from ultralytics.models.sam import SAM2VideoPredictor
model = FastSAM('FastSAM-x.pt')
DEVICE = 'cuda'
args = {'imgsz': 640, 'conf': 0.4, 'iou': 0.9}

MAX_AREA_THRESHOLD = (640 * 640) * 0.1

results = model.predict(source='2', device=DEVICE, stream=True, **args, texts=['human hand'])
print(results)
for result in results:
    frame = result.orig_img.copy()
    h, w = frame.shape[:2]
    
    if result.masks is not None:
        masks = result.masks.data  # (N, Hm, Wm) e.g., 384x640

        # Area filter
        areas = masks.sum(dim=(1, 2))
        keep_indices = (areas < MAX_AREA_THRESHOLD).nonzero(as_tuple=True)[0]

        if len(keep_indices) > 0:
            masks = masks[keep_indices]
        else:
            masks = []

        overlay = frame.copy()

        for mask in masks:
            mask = mask.cpu().numpy().astype(np.uint8)

            # ✅ Resize mask to original frame size
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            mask = mask.astype(bool)

            # Unique color per segment
            color = np.random.randint(0, 255, size=3, dtype=np.uint8)

            overlay[mask] = color

        alpha = 0.6
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    cv2.imshow('FastSAM Filtered - Multi Color', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()