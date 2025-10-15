import streamlit as st
import pandas as pd
import numpy as np
import os
from calculate_total_distance_traveled import extract_coordinates, calculate_total_distance_traveled

# --- Title and description ---
st.title("Foot Fault Distance Analyzer 🐁")
st.write("Upload your DeepLabCut CSV to calculate total distance traveled per body part.")

# --- File upload ---
uploaded_file = st.file_uploader("Upload a corrected CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, header=[0, 1, 2])
    st.write("File loaded successfully!")
    
    # Extract coordinates
    x_coords = df.xs('x', axis=1, level=2)
    y_coords = df.xs('y', axis=1, level=2)
    sorted_columns = sorted(x_coords.columns, key=lambda col: col[1])
    x_coords = x_coords[sorted_columns]
    y_coords = y_coords[sorted_columns]
    body_parts = [col[1] for col in sorted_columns]

    # Calculate distances
    total_distances = {}
    for i, body_part in enumerate(body_parts):
        x = x_coords.iloc[:, i].interpolate(limit_direction='both').values
        y = y_coords.iloc[:, i].interpolate(limit_direction='both').values
        deltas = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
        total_distances[body_part] = np.nansum(deltas)
    
    # Display results
    st.subheader("Total Distance Traveled (in pixels)")
    st.dataframe(pd.DataFrame.from_dict(total_distances, orient='index', columns=['Distance (px)']))

    # Option to download
    out_df = pd.DataFrame.from_dict(total_distances, orient='index', columns=['Distance (px)'])
    csv = out_df.to_csv().encode('utf-8')
    st.download_button("Download results as CSV", data=csv, file_name='distances.csv', mime='text/csv')
