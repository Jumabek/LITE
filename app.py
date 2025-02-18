from __future__ import division, print_function, absolute_import
import logging
logging.getLogger().setLevel(logging.ERROR)

import sys
import os
sys.path.append(os.path.join(os.getcwd(), 'yolo_tracking'))

from pathlib import Path

from yolo_tracking.boxmot import DeepOCSORT, BoTSORT, OCSORT, BYTETracker

from deep_sort.tracker import Tracker
from deep_sort import nn_matching
import cv2
import time
from ultralytics import YOLO
from opts import opt

from reid_modules import DeepSORT, StrongSORT

from deep_sort.detection import Detection

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import tempfile

boxmot_trackers = ['BoTSORT', 'OCSORT', 'ByteTrack',
                       'DeepOCSORT', 'LITEBoTSORT', 'LITEDeepOCSORT']

def process_uploaded_video(video_file):
    with tempfile.NamedTemporaryFile(delete=False) as tfile:
        tfile.write(video_file.read())
        return tfile.name


def create_detections(image, model, tracker_name, reid_model=None, imgsz=1280, 
                      conf=0.25, appearance_feature_layer=None):
    global boxmot_trackers
    
    detection_list = []

    # Custom YOLO detections
    yolo_results = model.predict(image, verbose=False, imgsz=imgsz, classes=0,
    conf=conf, appearance_feature_layer=appearance_feature_layer, return_feature_map=False)
    
    boxes = yolo_results[0].boxes.data.cpu().numpy()

    appearance_features = None
    if tracker_name.startswith('LITE'):
        assert appearance_feature_layer is not None, "Appearance features are not extracted"
        # Lite do not need to extract appearance features again for boxes
        appearance_features = yolo_results[0].appearance_features.cpu().numpy()
    
    if tracker_name in boxmot_trackers:
        return boxes, appearance_features

    else:
        if tracker_name == 'SORT': # SORT does not need appearance features 
            appearance_features =  [None] * len(boxes)
        else:
            appearance_features = reid_model.extract_appearance_features(image, boxes)
    

    for box, feature in zip(boxes, appearance_features):
        xmin, ymin, xmax, ymax, conf, _ = box
        conf = float(conf)
        x_tl, y_tl = map(int, (xmin, ymin))
        width, height = map(int, (xmax - xmin, ymax - ymin))
        bbox = (x_tl, y_tl, width, height)
        detection = Detection(bbox, conf, feature)
        detection_list.append(detection)

    return detection_list

def run(tracker_name, yolo_model, video_path,
    nn_budget, device, appearance_feature_layer=None, visualize=True,
    max_cosine_distance=0.7, max_age=30):

    stframe = st.empty()
    
    metric = nn_matching.NearestNeighborDistanceMetric(
        'cosine',
        max_cosine_distance,
        nn_budget
    )

    tracker = Tracker(metric, max_age=max_age)

    tick = time.time()

    # Load the detection YOLO model    
    model_path = yolo_model + '.pt'
    model = YOLO(model_path)
    model.to(device)

    reid_model = None
    if 'StrongSORT' in tracker_name:
        opt.NSA = True
        opt.BoT = True 
        opt.EMA = True
        opt.MC = True
        opt.woC = True

        if tracker_name == 'StrongSORT':
            reid_model = StrongSORT(device=device)

    elif tracker_name == 'DeepSORT':
        reid_model = DeepSORT(device=device)

    elif 'BoTSORT' in tracker_name:
        if tracker_name == 'LITEBoTSORT':
            assert appearance_feature_layer is not None, "Please provide appearance feature layer for LITEBoTSORT"

        tracker = BoTSORT(
            model_weights=Path('osnet_x0_25_msmt17.pt'),
            device=device,
            fp16=False,
            appearance_feature_layer=appearance_feature_layer)
    
    elif tracker_name == 'OCSORT':
        tracker = OCSORT()
    
    elif tracker_name == 'ByteTrack':
        tracker = BYTETracker()
    
    elif 'DeepOCSORT' in tracker_name:
        if tracker_name == 'LITEDeepOCSORT':
            assert appearance_feature_layer is not None, "Please provide appearance feature layer for LITEDeepOCSORT"
        
        tracker = DeepOCSORT(
            model_weights=Path('osnet_x0_25_msmt17.pt'),
            device=device,
            fp16=False,
            appearance_feature_layer=appearance_feature_layer)
        
    # Open the video
    cap = cv2.VideoCapture(video_path)
    frame_idx = 0

    while cap.isOpened():

        ret, frame = cap.read()
        if not ret:
            break

        tick = time.time()

        # Process frame
        detections = create_detections(frame, model, tracker_name, reid_model, appearance_feature_layer=appearance_feature_layer)

        if isinstance(detections, tuple):
            boxes, appearance_features = detections
            if tracker_name.startswith('LITE'):
                tracks = tracker.update(boxes, frame, appearance_features)
            else:
                tracks = tracker.update(boxes, frame)

            for track in tracks:
                x1, y1, x2, y2, track_id, _, _, _ = track  
                color = (0, 255, 0)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.putText(frame, f"ID {int(track_id)}", (int(x1), int(y1) - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
        else:
            tracker.predict()
            tracker.update(detections)
            tracks = tracker.tracks

            # Store results
            for track in tracks:
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue
                bbox = track.to_tlwh()
                
                # Draw the bounding box and ID
                if visualize:
                    x, y, w, h = map(int, bbox)
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(frame, str(track.track_id), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        tock = time.time()

        fps = 1 / (tock - tick)
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Frame: {frame_idx}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
        stframe.image(frame, channels="BGR")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_idx += 1
    
    cap.release()
    cv2.destroyAllWindows()

# Session state initialization
def init_session_state():
    session_keys = ['video_path']
    default_values = {'video_path': None}
    for key, default_value in default_values.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

init_session_state()

if __name__ == '__main__':
    st.title("Real-Time Object Tracking")
    init_session_state()

    tracker_name = st.selectbox('Select Tracker', ['DeepSORT', 'StrongSORT', 'BoTSORT', 'OCSORT', 'ByteTrack', 
                                                   'DeepOCSORT', 'LITEBoTSORT', 'LITEDeepOCSORT',
                                                   'SORT', 'LITEStrongSORT', 'LITEDeepSORT'])
    yolo_model = st.selectbox('YOLO Model', ['yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x', 
                                             'yolo11n', 'yolo11s', 'yolo11m', 'yolo11l', 'yolo11x',])
    
    video_file = st.file_uploader('Upload Video', type=['mp4', 'avi', 'mov'])
    nn_budget = 100
    device = st.selectbox('Device', ['cuda:0', 'cpu'])
    appearance_feature_layer = st.text_input('Appearance Feature Layer', '')

    if video_file:
        st.session_state.video_path =  process_uploaded_video(video_file)

    if st.button('Run'):
        run(
            tracker_name=tracker_name,
            yolo_model=yolo_model,
            video_path=st.session_state.video_path,
            nn_budget=nn_budget,
            device=device,
            appearance_feature_layer=appearance_feature_layer if appearance_feature_layer else None,
            visualize=True
        )