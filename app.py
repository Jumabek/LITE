import streamlit as st
import pandas as pd
import numpy as np
import cv2

from pathlib import Path
import os



st.title('My first app')

st.write("Here's our first attempt at using data to create a table:")

# Video upload place 
uploaded_video = st.file_uploader("Choose a video...", type=["mp4", "avi", "mov", "mkv"])

if uploaded_video is not None:
    st.video(uploaded_video)
    st.write("Video uploaded successfully")
else:
    st.write("Please upload a video")

# Process the video make it dataset
cap = cv2.VideoCapture(uploaded_video)
# Get the name of the uploaded video
video_name = uploaded_video.name
datasets_path = Path('datasets/custom_videos')

frame_index = 1
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Save the frame to the datasets folder
    os.makedirs(os.path.join(datasets_path, video_name, 'train', 'img1'), exist_ok=True)
    
    cv2.imwrite(os.path.join(datasets_path, video_name, 'train', 'img1', f'{frame_index:06}.jpg'), frame)


    


