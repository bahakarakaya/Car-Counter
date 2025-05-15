import cv2
import cvzone
import numpy as np
import time
from math import ceil
from collections import deque
from ultralytics import YOLO
from sort import *

# cap = cv2.VideoCapture(0)
# cap.set(3, 1280)
# cap.set(4, 720)
cap = cv2.VideoCapture("data/cars.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)
cars_mask = cv2.imread("data/cars-mask.png")

model = YOLO("models/yolo11s.pt")

class_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
               'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
               'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
               'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
               'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
               'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
               'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
               'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
               'toothbrush']

#TRACKING
tracker = Sort(max_age=20, min_hits=2, iou_threshold=0.3)

limits_south = [30, 380, 550, 380]
limits_north = [659, 295, 1000, 295]

totalCount_south = deque(maxlen=30)
totalCount_north = deque(maxlen=30)

MAX_COUNT = 50

while True:
    start_time = time.time()
    success, frame = cap.read()
    frameRegion = cv2.bitwise_and(frame, cars_mask)

    results = model(frameRegion)   # consider using stream=True for memory-efficient processing (e.g on edge devices or real-time video)

    detections = np.empty((0, 5))

    for r in results:
        boxes = r.boxes
        for box in boxes:
            #BOUNDING BOX
            x1, y1, x2, y2 = box.xyxy[0]                                  #with opencv
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)           #with opencv
            # cv2.rectangle(frame, (x1, y1), (x2, y2), (255,0,255), 3)    #with opencv

            w, h = x2-x1, y2-y1

            #CONFIDENCE
            conf = ceil(box.conf[0] * 100) / 100

            #CLASS NAME
            cls = int(box.cls[0])
            currentClass = class_names[cls]

            # searching is O(1) in sets, O(n) in lists
            if currentClass in {"truck", "car", "motorbike"} \
                    and conf > 0.3:
                # cvzone.putTextRect(frame, f'{class_names[cls]} {conf}', (max(0, x1), max(40, y1 - 20)),
                #                    scale=1, thickness=1, offset=3)
                cvzone.cornerRect(frame, (x1, y1, w, h), l=9, t=3, rt=5)

                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray)) #numpy makes a stack instead of .append

    resultsTracker = tracker.update(detections)
    cv2.line(frame, (limits_south[0], limits_south[1]), (limits_south[2], limits_south[3]), (0, 0, 255), thickness=5)
    cv2.line(frame, (limits_north[0], limits_north[1]), (limits_north[2], limits_north[3]), (0, 0, 255), thickness=5)

    for result in resultsTracker:
        x1, y1, x2, y2, id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        w, h = x2 - x1, y2 - y1

        center_x, center_y = x1 + w//2, y1 + h//2
        cv2.circle(frame, (center_x, center_y), radius=5, color=(255, 0, 255), thickness=cv2.FILLED)

        if limits_south[0] < center_x < limits_south[2] and limits_south[1]-13 < center_y < limits_south[3]+13:
            if id not in totalCount_south:
                totalCount_south.append(id)
                cv2.line(frame, (limits_south[0], limits_south[1]), (limits_south[2], limits_south[3]), (0, 255, 0), thickness=5)

        if limits_north[0] < center_x < limits_north[2] and limits_north[1]-13 < center_y < limits_north[3]+13:
            if id not in totalCount_north:
                totalCount_north.append(id)
                cv2.line(frame, (limits_north[0], limits_north[1]), (limits_north[2], limits_north[3]), (0, 255, 0), thickness=5)

        cvzone.putTextRect(frame, f'{int(id)}', (max(0, x1), max(40, y1 - 20)),
                                   scale=1, thickness=1, offset=3)


    cvzone.putTextRect(frame, f'To North: {len(totalCount_north)}', (30, 40), font=cv2.FONT_HERSHEY_COMPLEX_SMALL,
                       colorR=(255, 0, 0), scale=1, thickness=1, offset=6)
    cvzone.putTextRect(frame, f'To South: {len(totalCount_south)}', (30, 70), font=cv2.FONT_HERSHEY_COMPLEX_SMALL,
                       colorR=(255, 0, 0), scale=1, thickness=1, offset=6)

    cv2.imshow("Frame", frame)
    # cv2.imshow("FrameRegion", frameRegion)

    # How many seconds did it take to process this frame?
    elapsed_time = (time.time() - start_time) * 1000    #ms
    wait_time = max(int(1000 / fps - elapsed_time), 1)

    print(f"Frame time: {elapsed_time:.2f} ms, Target wait: {wait_time} ms")

    if cv2.waitKey(wait_time) & 0xFF == ord('q'):
        break
