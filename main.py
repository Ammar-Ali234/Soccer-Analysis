# main.py
from ultralytics import YOLO
import cv2
import numpy as np
from Distance_calculation import update_distances, save_to_csv
from sort import Sort
from heatmap import FootballHeatmap
import pandas as pd

def main():
    model = YOLO("yolo11n.pt")
    video_path = "Test_2.mp4"
    cap = cv2.VideoCapture(video_path)

    pixel_to_meter = 0.05  # Example: 1 pixel = 0.05 meters
    Tracker = Sort(max_age=1000, min_hits=8, iou_threshold=0.00125)

    if not cap.isOpened():
        print("Error: Could not open video file.")
        exit()

    output_path = "output.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, fourcc, fps, (640, 480))

    # Initialize tracking data storage
    tracking_data = []

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame = cv2.resize(frame, (640, 480))
            results = model(frame)
            detections = np.empty((0, 5))

            for result in results:
                for box in result.boxes:
                    if int(box.cls) == 0:  # Class 0 for players
                        x1, y1, x2, y2 = map(int, box.xyxy[0][:4])
                        conf = float(box.conf[0])
                        if conf > 0.5:
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 4)
                            cv2.putText(frame, "Player", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4)
                            current_array = np.array([[x1, y1, x2, y2, conf]])
                            detections = np.vstack((detections, current_array))

            resultsTracker = Tracker.update(detections)
            update_distances(resultsTracker, pixel_to_meter)
            
            # Store tracking data for heatmap
            for result in resultsTracker:
                x1, y1, x2, y2, track_id = map(int, result)
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                
                tracking_data.append({
                    'player_id': track_id,
                    'x': center_x,
                    'y': center_y,
                    'frame': len(tracking_data)
                })
                
                cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 4)

            out.write(frame)
            cv2.imshow("YOLO Detection", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        # Save tracking data and metrics
        save_to_csv(fps, pixel_to_meter, "player_tracking_data.csv")
        
        # Convert tracking data to DataFrame and save
        df_tracking = pd.DataFrame(tracking_data)
        df_tracking.to_csv("heatmap_tracking_data.csv", index=False)
        
        # Generate heatmaps
        print("Generating heatmaps...")
        heatmap_gen = FootballHeatmap()
        
        # Generate team heatmap
        heatmap_gen.generate_heatmap(df_tracking, 'team_heatmap.png')
        
        # Generate individual player heatmaps
        for player_id in df_tracking['player_id'].unique():
            heatmap_gen.generate_heatmap(
                df_tracking,
                f'player_{player_id}_heatmap.png',
                player_id=player_id
            )
        
        cap.release()
        out.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()