from ultralytics import YOLO
import cv2
import numpy as np
from Distance_calculation import update_distances, save_to_csv
from sort import Sort

model = YOLO("yolo11n.pt")
video_path = "Test_2.mp4"
cap = cv2.VideoCapture(video_path)

#fps = int(cap.get(cv2.CAP_PROP_FPS))  # Extract frame rate from the video
pixel_to_meter = 0.05  # Example: 1 pixel = 0.05 meters (adjust based on your setup)

Tracker = Sort(max_age=1000, min_hits=8, iou_threshold=0.025)

if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

output_path = "output.mp4"
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, (640, 480))
    results = model(frame)

    detections = np.empty((0, 5))  

    for result in results:
        for box in result.boxes:
            if int(box.cls) == 0:  
                x1, y1, x2, y2 = map(int, box.xyxy[0][:4])
                conf = float(box.conf[0])
                if(conf>0.5):
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 4)
                    cv2.putText(frame, "Player", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4)
                    current_array = np.array([[x1, y1, x2, y2, conf]])
                    detections = np.vstack((detections, current_array))
                

    resultsTracker = Tracker.update(detections)
    update_distances(resultsTracker, pixel_to_meter)
    for result in resultsTracker:
        x1, y1, x2, y2, track_id = map(int, result)
        cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 4)

    out.write(frame)
    cv2.imshow("YOLO Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

save_to_csv(fps, pixel_to_meter, "data.csv")
cap.release()
out.release()
cv2.destroyAllWindows()
