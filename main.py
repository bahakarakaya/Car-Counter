import cv2
import cvzone
from ultralytics import YOLO
from math import ceil


# cap = cv2.VideoCapture(0)
# cap.set(3, 1280)
# cap.set(4, 720)
cap = cv2.VideoCapture("data/cars.mp4")

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


while True:
    success, frame = cap.read()

    results = model(frame, stream=True)   #stream=True using generators

    for r in results:
        boxes = r.boxes
        for box in boxes:
            #BOUNDING BOX
            x1, y1, x2, y2 = box.xyxy[0]                                  #with opencv
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)           #with opencv
            # cv2.rectangle(frame, (x1, y1), (x2, y2), (255,0,255), 3)      #with opencv

            w, h = x2-x1, y2-y1

            #CONFIDENCE
            conf = ceil(box.conf[0] * 100) / 100

            #CLASS NAME
            cls = int(box.cls[0])
            current_cls = class_names[cls]

            # searching is O(1) in sets, O(n) in lists
            if current_cls in {"truck", "car", "motorbike", "bus"} \
                and conf > 0.3:
                cvzone.putTextRect(frame, f'{class_names[cls]} {conf}', (max(0, x1), max(40, y1 - 20)),
                                   scale=1, thickness=1, offset=3)
                cvzone.cornerRect(frame, (x1, y1, w, h), l=9, t=3)





    cv2.imshow("appFrame", frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
