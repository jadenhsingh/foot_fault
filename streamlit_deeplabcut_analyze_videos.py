import streamlit as st
import deeplabcut
import os
import time
from pathlib import Path

# --- Configuration ---
PROJECT_PATH = "/data/wythe_lab/DLC_project/config.yaml"  # adjust to your project
UPLOAD_DIR = "/data/wythe_lab/uploads"

os.makedirs(UPLOAD_DIR, exist_ok=True)

st.title("Wythe Lab DeepLabCut Video Analyzer 🎥🐁")
st.write("Upload a video file to analyze with DeepLabCut.")

uploaded_video = st.file_uploader("Upload your video", type=["mp4", "avi", "mov"])

if uploaded_video is not None:
    # Save uploaded file
    video_path = Path(UPLOAD_DIR) / uploaded_video.name
    with open(video_path, "wb") as f:
        f.write(uploaded_video.read())

    st.success(f"Video uploaded to server: {video_path}")

    if st.button("Run DeepLabCut Analysis"):
        with st.spinner("Analyzing video... This may take a few minutes ⏳"):
            try:
                # Step 1: Analyze video
                deeplabcut.analyze_videos(PROJECT_PATH, [str(video_path)], save_as_csv=True)

                # Step 2: Create labeled video
                deeplabcut.create_labeled_video(PROJECT_PATH, [str(video_path)], draw_skeleton=True)

                st.success("Analysis complete!")

                # Step 3: Offer downloads
                video_dir = video_path.parent
                base_name = video_path.stem
                h5_file = list(video_dir.glob(f"{base_name}*.h5"))
                csv_file = list(video_dir.glob(f"{base_name}*filtered.csv"))
                labeled_video = list(video_dir.glob(f"{base_name}*labeled.mp4"))

                if csv_file:
                    with open(csv_file[0], "rb") as f:
                        st.download_button("Download CSV", data=f, file_name=csv_file[0].name)
                if h5_file:
                    with open(h5_file[0], "rb") as f:
                        st.download_button("Download H5 file", data=f, file_name=h5_file[0].name)
                if labeled_video:
                    with open(labeled_video[0], "rb") as f:
                        st.download_button("Download Annotated Video", data=f, file_name=labeled_video[0].name)

            except Exception as e:
                st.error(f"Error during analysis: {e}")
