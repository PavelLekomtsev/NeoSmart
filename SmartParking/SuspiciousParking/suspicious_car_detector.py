import math
import time

import cv2
import cvzone
import numpy as np
from ultralytics import YOLO

from neosmart.config import get_settings
from neosmart.tracking.sort import Sort

_settings = get_settings()

cameraId = 4
model_path = str(_settings.paths.resolve(_settings.paths.car_detector))
confidence = 0.75
class_names = ["car"]
cam_width, cam_height = 1280, 720
suspicious_time = 30

cap = cv2.VideoCapture(cameraId)
cap.set(3, cam_width)
cap.set(4, cam_height)
model = YOLO(model_path)

mot_tracker = Sort()
ids_dictionary = {}


def get_object_list_yolo(_model, _img, _class_names, _confidence=0.5, draw=True):
    _results = _model(_img, stream=False, verbose=False)
    _object_list = []
    for r in _results:
        boxes = r.boxes
        for box in boxes:
            conf = math.ceil((box.conf[0] * 100)) / 100
            if conf > _confidence:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                w, h = x2 - x1, y2 - y1
                center = x1 + (w // 2), y1 + (h // 2)
                class_name = _class_names[int(box.cls[0])]
                _object_list.append({"bbox": (x1, y1, w, h), "center": center, "conf": conf, "class": class_name})
                if draw:
                    cvzone.cornerRect(_img, (x1, y1, w, h))
                    cvzone.putTextRect(_img, f'{class_name} {conf}', (max(0, x1), max(35, y1)), scale=1, thickness=1)
    return _object_list


def convert_format_for_sort_tracking(_object_list):
    detections = np.empty((0, 5))
    for obj in _object_list:
        x1, y1, w, h = obj["bbox"]
        x2, y2 = x1 + w, y1 + h
        detections = np.vstack([detections, [x1, y1, x2, y2, obj["conf"]]])
    return detections


def update_time(_track_ids, _ids_dictionary, _img, _max_time=30):
    for x1, y1, x2, y2, id in _track_ids:
        x1, y1, x2, y2, id = int(x1), int(y1), int(x2), int(y2), int(id)
        current_time = time.time()
        if id in _ids_dictionary:
            time_difference = round((current_time - _ids_dictionary[id]), 2)
        else:
            time_difference = current_time
            _ids_dictionary[id] = current_time

        if time_difference < _max_time:
            color = (255, 0, 0)
        else:
            color = (0, 0, 255)
        cvzone.cornerRect(_img, (x1, y1, x2 - x1, y2 - y1))
        cvzone.putTextRect(_img, f'{time_difference}s', (max(0, x1), max(35, y1)), scale=2, thickness=2, colorR=color)


while True:
    success, img = cap.read()
    object_list = get_object_list_yolo(model, img, class_names, confidence, draw=False)
    object_list_tracking_format = convert_format_for_sort_tracking(object_list)
    track_ids = mot_tracker.update(object_list_tracking_format)
    update_time(track_ids, ids_dictionary, img, suspicious_time)
    cv2.imshow("Image", img)
    cv2.waitKey(1)
