import os
import streamlit as st
import pandas as pd
import numpy as np
import time
import subprocess

# --- Server directories ---
VIDEO_DIR = "/home/jhs8cue/wythe_lab/videos"
os.makedirs(VIDEO_DIR, exist_ok=True)

# Path to your batch script
BATCH_SCRIPT = "/home/jhs8cue/wythe_lab/run_video_analysis_gpu"

# --- Helper functions ---
def extract_coordinates(csv_file_path):
    df = pd.read_csv(csv_file_path, header=[0, 1, 2])
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
    """Submit the GPU batch script to SLURM for the specified video and return the job ID."""
    cmd = ["sbatch", BATCH_SCRIPT, video_path]
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"Batch script failed:\n{result.stderr}")

    # Example output: "Submitted batch job 123456"
    stdout = result.stdout.strip()
    job_id = stdout.split()[-1]  # get the last word
    return job_id


def check_job_done(job_id):
    """Return True if the SLURM job is done."""
    result = subprocess.run(["squeue", "-j", job_id], capture_output=True, text=True)
    return job_id not in result.stdout

# --- Streamlit App ---
st.title("Foot Fault Distance Analyzer 🐁")

uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])

if uploaded_file:
    # Save uploaded video to server
    video_path = os.path.join(VIDEO_DIR, uploaded_file.name)
    with open(video_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success(f"Uploaded video saved to: {video_path}")

    # Submit batch job to SLURM
    st.info("Submitting video for GPU batch analysis...")
    job_id = run_batch_script(video_path)
    st.success(f"Submitted batch job ID: {job_id}")

    # Polling for completion
    with st.spinner("Running analysis on GPU (this may take a while)..."):
        while not check_job_done(job_id):
            time.sleep(30)

    st.success("🎉 Job completed! Processing results...")

    # Automatically find the CSV output
    import glob
    base_name = os.path.splitext(uploaded_file.name)[0]
    csv_matches = glob.glob(os.path.join(VIDEO_DIR, f"{base_name}*DLC*.csv"))

    if not csv_matches:
        st.error("No CSV output found after analysis.")
    else:
        # Pick the most recent CSV
        csv_path = max(csv_matches, key=os.path.getmtime)
        st.success(f"Found CSV: {os.path.basename(csv_path)}")

        # Extract coordinates and calculate distances
        x_coords, y_coords, body_parts = extract_coordinates(csv_path)
        total_distances = calculate_total_distance_traveled(x_coords, y_coords, body_parts)

        st.subheader("Total Distance Traveled (pixels)")
        for bp, dist in total_distances.items():
            st.write(f"**{bp}:** {dist:.2f}")

        # Allow downloads
        st.download_button("⬇ Download CSV", data=open(csv_path, "rb"), file_name=os.path.basename(csv_path))
        st.download_button("⬇ Download Video", data=open(video_path, "rb"), file_name=uploaded_file.name)

    # Optional: remove uploaded video to save space
    try:
        os.remove(video_path)
        st.info(f"Removed uploaded video: {video_path}")
    except Exception as e:
        st.warning(f"Could not delete uploaded video: {e}")
