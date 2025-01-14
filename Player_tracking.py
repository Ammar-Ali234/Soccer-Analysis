from ultralytics import YOLO
import cv2
import numpy as np
from sort import Sort

model = YOLO("yolo11n.pt")
video_path = "input_video.mp4"
cap = cv2.VideoCapture(video_path)

Tracker = Sort(max_age=25, min_hits=3, iou_threshold=0.3)

if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

output_path = "output_video.mp4"
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)

    detections = np.empty((0, 5))  # Initialize empty array for detections

    for result in results:
        for box in result.boxes:
            if int(box.cls) == 0:  # Assuming class ID 0 is for "person"
                x1, y1, x2, y2 = map(int, box.xyxy[0][:4])
                conf = float(box.conf[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, "Player", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                current_array = np.array([[x1, y1, x2, y2, conf]])
                detections = np.vstack((detections, current_array))

    resultsTracker = Tracker.update(detections)

    for result in resultsTracker:
        x1, y1, x2, y2, track_id = map(int, result)
        cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    out.write(frame)
    cv2.imshow("YOLO Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
