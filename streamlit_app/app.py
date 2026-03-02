import streamlit as st
import os, sys
sys.path.append(os.path.abspath("Streamlit"))
from test import stream_video



Upload_folder = '/home/oybek/LITE/Streamlit/Uploaded_video'
# Add a title
st.title("Trackers Comparison")
st.write("This is a simple web app to compare different trackers on a video.")


# Create a selectbox (dropdown) for toggling between different options
col1, col2 = st.columns(2)

with col1:
    tracker1 = st.selectbox(
        'Choose a first tracker:',
        ["LITEStrongSORT", "LITEDeepOCSORT", "LITEDeepSORT", "LITEBoTSORT", "StrongSORT", "DeepOCSORT", 
         "DeepSORT", "BoTSORT", "OCSORT", "DeepSORT", "SORT", "Bytetrack"]
    )

# Place the second selectbox in the second column
with col2:
    tracker2 = st.selectbox(
        'Choose a second tracker:',
        ["LITEDeepOCSORT","LITEStrongSORT", "LITEDeepSORT", "LITEBoTSORT", "StrongSORT", "DeepOCSORT", 
         "DeepSORT", "BoTSORT", "OCSORT", "DeepSORT", "SORT", "Bytetrack"]
    )


if tracker1 == tracker2:
    st.error("Please select a different tracker for the second column.")

model = st.selectbox(
    'Choose a Model:',
    ['yolo11m','yolov8l','yolov8m','yolov8n','yolov8s','yolov8x','ablation_17l','ablation_17n','ablation_17s','ablation_17x']
)

conf_lvl = st.text_input("Enter confidance level: ")

if conf_lvl:
    try:
        conf_lvl = float(conf_lvl)
        if 0 <= conf_lvl <= 1:
            pass
        else:
            st.error("Please enter a value between 0 and 1.")
    except ValueError:
        st.error("Invalid input. Please enter a valid float value between 0 and 1.")

tracker1 = tracker1.lower()
model = model+".pt"

uploaded_file = st.file_uploader("Please upload or drag a video", type=["mp4", "avi", "mov", "mkv"])
# Check if a file has been uploaded
if uploaded_file :
    if uploaded_file is not None:
        save_path = os.path.join(Upload_folder, uploaded_file.name)
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.write("Uploaded video")
            
    else:
        st.write("No video uploaded yet.")

if st.button("Process Video"):
        #try:
            col1, col2 = st.columns(2)
            with col1:
                output_video_path, proc_time = stream_video( tracker1, conf_lvl, save_path, model)
                st.write(f"Processing time of {tracker1}:" )
                st.write( f"{proc_time:.2f} seconds")
                with open(output_video_path, "rb") as file:
                    btn = st.download_button(
                        label="Download Processed Video",
                        data=file,
                        file_name=output_video_path.split("/")[-1],
                        mime="video/mp4"
                    )
            with col2:
                output_video_path, proc_time = stream_video( tracker2, conf_lvl, save_path, model)
                st.write(f"Processing time of {tracker2}:")
                st.write(f"{proc_time:.2f} seconds")
                with open(output_video_path, "rb") as file:
                    btn = st.download_button(
                        label="Download Processed Video",
                        data=file,
                        file_name=output_video_path.split("/")[-1],
                        mime="video/mp4"
                     )
        # except:
        #     st.write("Check the inputs and try again.")

def display_image(image):
    st.image(image, channels="BGR", use_column_width=True)