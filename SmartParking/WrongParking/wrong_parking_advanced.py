# Import necessary libraries
import math
import pickle
import cv2
import cvzone
import numpy as np
from ultralytics import YOLO

# Initialize camera and model settings
cameraId = 0
model_path = "Models\Car_Detector.pt"
polygon_file_path = 'camera1_parkings.p'
confidence = 0.6
class_names = ["car"]
cam_width, cam_height = 1280, 720

# Initialize video capture from camera
cap = cv2.VideoCapture(cameraId)
cap.set(3, cam_width)
cap.set(4, cam_height)

# Load the YOLO model
model = YOLO(model_path)

# Load the Regions of Interest (ROIs) polygons from a file
file_obj = open(polygon_file_path, 'rb')
rois = pickle.load(file_obj)
file_obj.close()


# Function to get a list of objects detected by YOLO in an image
def get_object_list_yolo(_model, _img, _class_names, _confidence=0.5, draw=True):
    """
        Detect objects using YOLO model.

        Parameters:
        - _model: YOLO model for object detection.
        - _img: Input image for object detection.
        - _class_names: List of class names to detect.
        - _confidence: Confidence threshold for object detection.
        - draw: Whether to draw bounding boxes on the image.

        Returns:
        - _object_list: List of dictionaries containing information about detected objects.
        """
    # Run YOLO on the input image
    _results = _model(img, stream=False, verbose=False)
    _object_list = []

    # Iterate through the detected results
    for r in _results:
        boxes = r.boxes
        for box in boxes:
            # Extract information about the detected object
            conf = math.ceil((box.conf[0] * 100)) / 100
            if conf > _confidence:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                w, h = x2 - x1, y2 - y1
                center = x1 + (w // 2), y1 + (h // 2)
                class_name = _class_names[int(box.cls[0])]

                # Append information to the object list
                _object_list.append({"bbox": (x1, y1, w, h),
                                     "center": center,
                                     "conf": conf,
                                     "class": class_name})

                # Draw bounding box and class label on the image if specified
                if draw:
                    cvzone.cornerRect(_img, (x1, y1, w, h))
                    cvzone.putTextRect(_img, f'{class_name} {conf}',
                                       (max(0, x1), max(35, y1)), scale=1, thickness=1)
    return _object_list



def find_polygon_index(point, polygons):
    """
    Find the index of the polygon containing a given point.

    Parameters:
    - point: (x, y) coordinates of the point.
    - polygons: List of polygons defined by their vertices.

    Returns:
    - index: Index of the polygon containing the point, or -1 if not found.
    """
    for idx, polygon_pts in enumerate(polygons):
        polygon_pts = np.array(polygon_pts, np.int32)
        polygon_pts = polygon_pts.reshape((-1, 1, 2))
        result = cv2.pointPolygonTest(polygon_pts, point, False)
        if result >= 0:
            return idx
    return -1


def percentage_of_rectangle_outside_polygon(polygon_pts, _bbox, _img, _index):
    """
    Calculate the percentage of a rectangle outside a given polygon.

    Parameters:
    - polygon_pts: Vertices of the polygon.
    - _bbox: Bounding box coordinates (x, y, w, h).
    - _img: Input image.
    - _index: Index of the polygon.

    Returns:
    - percentage_outside: Percentage of the rectangle outside the polygon.
    """
    # Area of Parking Spot
    height, width, _ = _img.shape
    image = np.zeros((height, width), dtype=np.uint8)
    polygon_pts = np.array(polygon_pts, dtype=np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(image, [polygon_pts], 255)
    cv2.imshow("Area of Parking Spot",image)

    # Area of Car
    rectangle_mask = np.zeros((height, width), dtype=np.uint8)
    x, y, w, h = _bbox
    cv2.rectangle(rectangle_mask, (x, y), (x + w, y + h), 255, -1)
    cv2.imshow("Area of Car",rectangle_mask)
    # Compute the total area of the rectangle
    total_rectangle_area = cv2.countNonZero(rectangle_mask)

    # Find the intersection between the parking spot and the car
    intersection = cv2.bitwise_and(image, rectangle_mask)
    cv2.imshow("intersection",intersection)
    # Compute the area of the intersection
    intersection_area = cv2.countNonZero(intersection)


    # Calculate the area of the rectangle that is outside the polygon
    # area of the car - intersection
    rectangle_outside_polygon_area = total_rectangle_area - intersection_area

    # Compute the percentage of the rectangle that is outside the polygon
    percentage_outside = (rectangle_outside_polygon_area / total_rectangle_area) * 100

    if _index in [0, 5, 6, 11]:
        threshold = 32
    else:
        threshold = 20

    if percentage_outside > threshold:
        color = (0, 0, 255)
        text = "Wrong"
    else:
        color = (0, 255, 0)
        text = "Correct"
    if y < 200:
        y = y + h + 25

    cvzone.putTextRect(_img, text, (x, y), scale=2, colorR=color)

    return percentage_outside


def detect_based_on_roi(_object_list, _img, _rois):
    """
    Detect if each object is parked correctly based on predefined ROIs.

    Parameters:
    - _object_list: List of detected objects.
    - _img: Input image.
    - _rois: List of Regions of Interest (polygons).

    Returns:
    - None
    """
    for obj in _object_list:
        center = obj["center"]
        index = find_polygon_index(center, _rois)
        bbox = obj["bbox"]
        percentage_outside = percentage_of_rectangle_outside_polygon(_rois[index], bbox, _img, index)
        print(percentage_outside)


# Main loop for capturing video and processing YOLO results
while True:
    success, img = cap.read()
    object_list = get_object_list_yolo(model, img, class_names, 0.5, draw=True)
    detect_based_on_roi(object_list, img, rois)

    cv2.imshow("Image", img)
    cv2.waitKey(1)

