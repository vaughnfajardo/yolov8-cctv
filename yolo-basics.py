from ultralytics import YOLO
import cv2

model = YOLO('yolov8x.pt')
results = model("test-image/1.png", show=True)
cv2.waitKey(0)