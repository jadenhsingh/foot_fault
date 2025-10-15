import os
import streamlit as st
import pandas as pd
import numpy as np
import subprocess

# --- Server directories ---
VIDEO_DIR = "/home/jhs8cue/wythe_lab/videos"
os.makedirs(VIDEO_DIR, exist_ok=True)

# Path to your batch script
BATCH_SCRIPT = "/home/jhs8cue/wythe_lab/run_video_analysis_gpu.sh"

# --- Helper functions ---
def extract_coordinates(csv_file_path):
    df = pd.read_csv(csv_file_path, header=[0,1,2])
    x_coords = df.xs("x", axis=1, level=2)
    y_coords = df.xs("y", axis=1, level=2)
    sorted_columns = sorted(x_coords.columns, key=lambda col: col[1])
    x_coords = x_coords[sorted_columns]
    y_coords = y_coords[sorted_columns]
    body_parts = [col[1] for col in sorted_columns]
    return x_coords, y_coords, body_parts

def calculate_total_distance_traveled(x_coords, y_coords, body_parts):
    total_distances = {}
    for i, body_part in enumerate(body_parts):
        x = pd.Series(x_coords.iloc[:, i]).interpolate(limit_direction="both").values
        y = pd.Series(y_coords.iloc[:, i]).interpolate(limit_direction="both").values
        deltas = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
        total_distances[body_part] = np.nansum(deltas)
    return total_distances

def run_batch_script(video_path):
    """Run your existing GPU batch script on the uploaded video."""
    cmd = [BATCH_SCRIPT, video_path]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result

# --- Streamlit app ---
st.title("Foot Fault Analysis via Batch Script")

uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    video_path = os.path.join(VIDEO_DIR, uploaded_file.name)
    with open(video_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success(f"Saved video to {video_path}")

    st.info("Running batch analysis on GPU...")
    result = run_batch_script(video_path)

    if result.returncode != 0:
        st.error(f"Batch script failed:\n{result.stderr}")
    else:
        st.success("Batch script completed successfully!")

        # Assume your batch script generates a CSV in the same folder with a predictable name
        csv_name = os.path.splitext(uploaded_file.name)[0] + "DeepCut_resnet50_projected.csv"
        csv_path = os.path.join(VIDEO_DIR, csv_name)

        if not os.path.exists(csv_path):
            st.error(f"CSV file not found: {csv_path}")
        else:
            x_coords, y_coords, body_parts = extract_coordinates(csv_path)
            total_distances = calculate_total_distance_traveled(x_coords, y_coords, body_parts)

            st.subheader("Total distance traveled per body part (pixels):")
            for bp, dist in total_distances.items():
                st.write(f"**{bp}:** {dist:.2f}")

            # Optionally allow downloads
            st.download_button("Download CSV", data=open(csv_path, "rb"), file_name=csv_name)
            st.download_button("Download video", data=open(video_path, "rb"), file_name=uploaded_file.name)
