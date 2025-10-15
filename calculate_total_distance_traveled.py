import os
import pandas as pd
import numpy as np
import argparse

directory = "C:/Users/ssene/Desktop/foot_fault_videos"

video_labels_paths = [
    os.path.join(directory, f)
    for f in os.listdir(directory)
    if f.endswith(".csv")
]
videos_paths = [
    os.path.join(directory, f)
    for f in os.listdir(directory)
    if f.endswith(".mp4")
]

video_label_dfs = [pd.read_csv(file, header=[0, 1, 2]) for file in video_labels_paths]

def extract_coordinates(video_name):
    """Extracts and sorts x and y coordinates for a given video."""
    for i, video_path in enumerate(videos_paths):
        if os.path.splitext(os.path.basename(video_path))[0] in video_name:
            video_index = i
            # Print the CSV being accessed
            print(f"Accessing CSV: {video_path}")
            break
    else:
        raise ValueError(f"No matching video found for {video_name}")

    df = video_label_dfs[video_index]
    x_coords = df.xs('x', axis=1, level=2)
    y_coords = df.xs('y', axis=1, level=2)

    # Sort columns by body part name (level 1)
    sorted_columns = sorted(x_coords.columns, key=lambda col: col[1])
    x_coords = x_coords[sorted_columns]
    y_coords = y_coords[sorted_columns]
    body_parts = [col[1] for col in sorted_columns]

    return x_coords, y_coords, body_parts

def calculate_total_distance_traveled(x_coords, y_coords, body_parts):
    """Calculates total distance traveled for each body part."""
    total_distances = {}

    for i, body_part in enumerate(body_parts):
        x = x_coords.iloc[:, i].values
        y = y_coords.iloc[:, i].values

        # Ignore NaNs by linear interpolation
        x = pd.Series(x).interpolate(limit_direction='both').values
        y = pd.Series(y).interpolate(limit_direction='both').values

        deltas = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
        total_distances[body_part] = np.nansum(deltas)

    return total_distances
if __name__ == "__main__":
    # Parse the video filename as a command-line argument
    parser = argparse.ArgumentParser(description="Analyze total distance traveled in foot fault videos")
    parser.add_argument("video_name", type=str, help="Name of the video file to analyze")
    args = parser.parse_args()

    x_coords, y_coords, body_parts = extract_coordinates(args.video_name)
    total_distances = calculate_total_distance_traveled(x_coords, y_coords, body_parts)

    print(f"\nTotal distance traveled per body part in {args.video_name}:")
    for bp, dist in total_distances.items():
        print(f"  {bp}: {dist:.2f} pixels")
