import cv2
from ultralytics import YOLO
import numpy as np


cap = cv2.VideoCapture("crowd.mp4")

model = YOLO("yolov8m.pt")

with open("class.txt", "r") as file:
    class_labels = [line.strip() for line in file.readlines()]

while True: 
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, device="mps")
    result = results[0]
    bboxes = np.array(result.boxes.xyxy.cpu(), dtype="int")
    classes = np.array(result.boxes.cls.cpu(), dtype="int")

    for cls, bbox in zip(classes, bboxes):
        x1, y1, x2, y2 = bbox

        class_name = class_labels[cls]

        if class_name == "person":
            class_name = "intruder"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{class_name}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Img", frame)
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()