import cv2 
import numpy as np
import math
from ultralytics import YOLO

model_dir = "../yolo_models"
model_type = "l"

model = YOLO(f"{model_dir}/yolov8{model_type}.pt")

name = model.names

# load video
cap = cv2.VideoCapture("../data/videos/trucks.mp4")

#  set width and height
frame_width = int(cap.get(3))
frame_height = int(cap.get(4)) 
size = (frame_width, frame_height)

# create  Videowriter
result = cv2.VideoWriter('../figures/track.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 10, size)

# init
count = 0
center_points_prev_frame = []
tracking_objects = {}
track_id = 0
max_miss = 10
misses = {}

while True:
    ret, frame = cap.read()
    count += 1
    if not ret:
        break
    
    detections = model(frame, conf=0.35, imgsz=960)[0]
    data = detections.boxes.data.cpu().numpy()

    # init current frame points
    center_points_cur_frame = []

    # detect boxes
    xyxy = data[:, :4].astype(int)
    scores = data[:, 4]
    class_ids = data[:, 5].astype(int)
    boxes = [(x1, y1, x2 - x1, y2 - y1) for (x1, y1, x2, y2) in xyxy]

    for i, box in enumerate(boxes):
        (x, y, w, h) = box
        cx = int((x + x + w)/ 2)
        cy = int((y + y + h)/ 2)
        center_points_cur_frame.append((cx, cy))
        cv2.rectangle(frame, (x,y), (x + w, y + h), (0, 255, 0), 2)
        class_info = f"{name[class_ids[i]]} {scores[i]:.2f}"
        cv2.putText(frame, class_info, (x, y),
                    cv2.FONT_HERSHEY_COMPLEX, 0.75, (250, 250, 0), 2)
        
    # first two frames
    if count <= 2:
        for pt in center_points_cur_frame:
            for pt2 in center_points_prev_frame:
                distance = math.hypot(pt2[0] -pt[0], pt2[1] - pt[1])

                if distance < 45:
                    tracking_objects[track_id] = pt
                    misses[track_id] = 0
                    track_id += 1
    else:
        tracking_objects_copy = tracking_objects.copy()
        center_points_cur_frame_copy = center_points_cur_frame.copy()

        for object_id, pt2 in tracking_objects_copy.items():
            object_exists = False
            for pt in center_points_cur_frame_copy: 
                distance = math.hypot(pt2[0] - pt[0], pt2[1] - pt[1])

                # Threshold
                if distance < 45:
                    if pt in center_points_cur_frame:
                        center_points_cur_frame.remove(pt)
                        tracking_objects[object_id] = pt
                        object_exists = True
                        misses[track_id] = 0
                    continue

            # remove id's that are lost
            if not object_exists:
                misses[object_id] = misses.get(object_id, 0) + 1
                if misses[object_id] > max_miss:
                    tracking_objects.pop(object_id, None)
                    misses.pop(object_id, None)

        # add new id's
        for pt in center_points_cur_frame:
            tracking_objects[track_id] = pt
            misses[track_id] = 0
            track_id += 1

    # label boxes with ids
    for object_id, pt in tracking_objects.items():
        cv2.circle(frame, pt, 5, (0, 0, 5), -1)
        cv2.putText(frame, str(object_id), (pt[0], pt[1] - 7), 0, 1, (0, 0, 255), 2)

    # show frame
    result.write(frame)
    cv2.imshow("Frame", frame)

    # copy cur to prev
    # for first two frames
    center_points_prev_frame = center_points_cur_frame.copy()

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
