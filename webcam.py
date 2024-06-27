import cv2
from ultralytics import YOLO
import cvzone
import math

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

model = YOLO("../weights/yolov8x.pt")

classNames = ["person", "bicycle", "car", "motorbike", "bus", "truck"]

while True:
    success, img = cap.read()
    results = model(img, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:

            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2-x1, y2-y1
            cvzone.cornerRect(img,(x1,y1,w,h))

            # Confidence
            conf = math.ceil((box.conf[0]*1000))/1000


            # Class
            cls = int(box.cls[0])

            # Label for Bounding Box
            cvzone.putTextRect(img,f'{classNames[cls]} {conf}',(max(0,x1),max(0,y1)), scale=1, thickness=1)
            


        cv2.imshow("Image", img)
        cv2.waitKey(1)