import os
import streamlit as st
import pandas as pd
import numpy as np
import time
import subprocess
import glob
import shutil
import zipfile

# --- Server directories ---
VIDEO_DIR = "/home/jhs8cue/wythe_lab/videos"
os.makedirs(VIDEO_DIR, exist_ok=True)

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
    cmd = ["sbatch", BATCH_SCRIPT, video_path]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Batch script failed:\n{result.stderr}")
    stdout = result.stdout.strip()
    job_id = stdout.split()[-1]
    return job_id

def check_job_done(job_id):
    result = subprocess.run(["squeue", "-j", job_id], capture_output=True, text=True)
    return job_id not in result.stdout

# --- Streamlit App ---
st.title("Foot Fault Distance Analyzer 🐁")

# Initialize session state
for key in ["job_id", "video_path", "generated_files", "base_name"]:
    if key not in st.session_state:
        st.session_state[key] = None

uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])

# --- Handle upload ---
if uploaded_file and st.session_state.video_path is None:
    video_path = os.path.join(VIDEO_DIR, uploaded_file.name)
    with open(video_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.session_state.video_path = video_path
    st.session_state.base_name = os.path.splitext(uploaded_file.name)[0]
    st.success(f"Uploaded video saved to: {video_path}")

# --- Run analysis ---
if st.session_state.video_path and st.session_state.job_id is None and st.session_state.generated_files is None:
    st.info("Submitting video for DeepLabCut video analysis...")
    job_id = run_batch_script(st.session_state.video_path)
    st.session_state.job_id = job_id
    st.success(f"Submitted batch job ID: {job_id}")

# --- Polling for job completion ---
if st.session_state.job_id and st.session_state.generated_files is None:
    job_id = st.session_state.job_id
    with st.spinner("Running video analysis on GPU (this may take a while)..."):
        while not check_job_done(job_id):
            time.sleep(30)

    # Once job completes
    st.session_state.job_id = None

    base_name = st.session_state.base_name
    generated_files = glob.glob(os.path.join(VIDEO_DIR, f"{base_name}*"))

    # Zip the plot-poses directory if it exists
    plot_dir = os.path.join(VIDEO_DIR, "plot-poses")
    if os.path.isdir(plot_dir):
        zip_path = os.path.join(VIDEO_DIR, "plot-poses.zip")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(plot_dir):
                for file in files:
                    abs_path = os.path.join(root, file)
                    rel_path = os.path.relpath(abs_path, VIDEO_DIR)
                    zipf.write(abs_path, rel_path)
        generated_files.append(zip_path)

    st.session_state.generated_files = generated_files
    st.success("Job completed! Results ready for download.")

# --- Display results if available ---
if st.session_state.generated_files:
    generated_files = st.session_state.generated_files
    csv_files = [f for f in generated_files if f.endswith(".csv")]

    if csv_files:
        csv_path = max(csv_files, key=os.path.getmtime)
        st.success(f"Found CSV: {os.path.basename(csv_path)}")
        x_coords, y_coords, body_parts = extract_coordinates(csv_path)
        total_distances = calculate_total_distance_traveled(x_coords, y_coords, body_parts)

        distances_df = pd.DataFrame({
            "Body Part": list(total_distances.keys()),
            "Total Distance (pixels)": [f"{v:.2f}" for v in total_distances.values()]
        })
        st.subheader("Total Distance Traveled")
        st.table(distances_df)

    # Download section
    st.subheader("Download Results")
    for file_path in generated_files:
        if os.path.isfile(file_path):
            with open(file_path, "rb") as f:
                st.download_button(
                    label=f"Download {os.path.basename(file_path)}",
                    data=f,
                    file_name=os.path.basename(file_path),
                    mime="application/octet-stream"
                )

    # Cleanup button
    if st.button("Reset"):
        for file_path in generated_files:
            try:
                os.remove(file_path)
            except Exception as e:
                st.warning(f"Could not delete file {file_path}: {e}")
        plot_dir = os.path.join(VIDEO_DIR, "plot-poses")
        if os.path.isdir(plot_dir):
            shutil.rmtree(plot_dir, ignore_errors=True)
        for key in ["video_path", "generated_files", "base_name"]:
            st.session_state[key] = None
        st.success("Cleanup complete. Ready for a new upload.")
