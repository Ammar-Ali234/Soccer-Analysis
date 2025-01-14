import csv
import numpy as np
from scipy.spatial.distance import euclidean
from collections import defaultdict

# Initialize data structures to store player tracking information
player_positions = defaultdict(list)  # Tracks positions for each player ID
player_distances = defaultdict(float)  # Accumulated distances for each player ID

def update_distances(resultsTracker, pixel_to_meter):
    """
    Update distances for each tracked player ID using the current frame's tracker results.

    Args:
        resultsTracker (list): List of tracked object data [x1, y1, x2, y2, track_id].
        pixel_to_meter (float): Conversion factor from pixels to meters.
    """
    global player_positions, player_distances
    
    for result in resultsTracker:
        x1, y1, x2, y2, track_id = map(int, result)
        center = ((x1 + x2) // 2, (y1 + y2) // 2)  # Calculate the center of the bounding box
        
        # Append the current center to the player's position history
        if player_positions[track_id]:
            prev_center = player_positions[track_id][-1]
            distance = euclidean(prev_center, center) * pixel_to_meter  # Convert distance to meters
            player_distances[track_id] += distance
        
        player_positions[track_id].append(center)

def save_to_csv(fps, pixel_to_meter, output_csv="player_data.csv"):
    """
    Save player distances and calculated speed to a CSV file.

    Args:
        fps (float): Frames per second of the video.
        pixel_to_meter (float): Conversion factor from pixels to meters.
        output_csv (str): The path to the output CSV file.
    """
    global player_positions, player_distances
    
    player_data = []
    
    for track_id, positions in player_positions.items():
        total_distance = player_distances[track_id]  # Already in meters
        frames_tracked = len(positions)
        avg_speed = (total_distance / frames_tracked) * fps if frames_tracked > 0 else 0  # Speed in m/s
        player_data.append([track_id, total_distance, avg_speed])
    
    # Write data to CSV
    with open(output_csv, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Player ID", "Total Distance (m)", "Average Speed (m/s)"])
        writer.writerows(player_data)

    print(f"Player data saved to {output_csv}")
