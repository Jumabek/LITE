import cv2
import tempfile
from deep_sort.detection import Detection
from reid_modules import DeepSORT, StrongSORT
from opts import opt
from ultralytics import YOLO
import time
import sys
from deep_sort import nn_matching
from deep_sort.tracker import Tracker
from ultralytics import YOLO
import numpy as np
from pathlib import Path
from boxmot import DeepOCSORT, BoTSORT, StrongSORT, OCSORT, BYTETracker
import time
import logging
import tempfile
import warnings
import os
sys.path.append('/home/oybek/LITE/')
sys.path.append('/home/oybek/LITE/yolo_tracking')
logging.getLogger().setLevel(logging.ERROR)



sys.path.append(os.path.join(os.getcwd(), 'yolo_tracking'))


warnings.filterwarnings("ignore")


# Define your tracker list as before
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
    yolo_results = model.predict(
        image, verbose=False, imgsz=imgsz, classes=0,
        conf=conf, appearance_feature_layer=appearance_feature_layer, return_feature_map=False
    )

    boxes = yolo_results[0].boxes.data.cpu().numpy()

    appearance_features = None
    if tracker_name.startswith('LITE'):
        assert appearance_feature_layer is not None, "Appearance features are not extracted"
        # LITE trackers do not need to extract appearance features again for boxes
        appearance_features = yolo_results[0].appearance_features.cpu().numpy()

    if tracker_name in boxmot_trackers:
        return boxes, appearance_features

    else:
        if tracker_name == 'SORT':  # SORT does not need appearance features
            appearance_features = [None] * len(boxes)
        elif tracker_name.startswith('LITE'):
            assert appearance_feature_layer is not None, "Appearance features are not extracted"
            # LITE trackers do not need to extract appearance features again for boxes
            appearance_features = yolo_results[0].appearance_features.cpu(
            ).numpy()
        else:
            appearance_features = reid_model.extract_appearance_features(
                image, boxes)

    for box, feature in zip(boxes, appearance_features):
        xmin, ymin, xmax, ymax, conf, _ = box
        conf = float(conf)
        x_tl, y_tl = map(int, (xmin, ymin))
        width, height = map(int, (xmax - xmin, ymax - ymin))
        bbox = (x_tl, y_tl, width, height)
        detection = Detection(bbox, conf, feature)
        detection_list.append(detection)

    return detection_list


def run_tracker(tracker_name, yolo_model, video_path,
                nn_budget, device, appearance_feature_layer, conf=0.25,
                max_cosine_distance=0.7, max_age=30):
    """
    This function runs the tracker and pushes processed frames into out_queue.
    Note: It does not call any Streamlit functions.
    """
    metric = nn_matching.NearestNeighborDistanceMetric(
        'cosine',
        max_cosine_distance,
        nn_budget
    )

    # Initialize default tracker for DeepSORT-like methods.
    tracker = Tracker(metric, max_age=max_age)

    # Load YOLO detection model
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
        else:
            appearance_feature_layer = None

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
        else:
            appearance_feature_layer = None

        tracker = DeepOCSORT(
            model_weights=Path('osnet_x0_25_msmt17.pt'),
            device=device,
            fp16=False,
            appearance_feature_layer=appearance_feature_layer)

    # Open the video file
    output_path = f"/home/oybek/LITE/streamlit_app/tracked_video/{tracker_name}_{conf}_mot20-5.mp4"  # Change this to your desired output path
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    frame_idx = 0
    ttick = time.time()
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        tick = time.time()

        # Process frame
        detections = create_detections(frame, model, tracker_name, reid_model,
                                       appearance_feature_layer=appearance_feature_layer, conf=conf)
        if isinstance(detections, tuple):
            boxes, appearance_features = detections
            if tracker_name.startswith('LITE'):
                tracks = tracker.update(boxes, frame, appearance_features)
            else:
                tracks = tracker.update(boxes, frame)

            for track in tracks:
                x1, y1, x2, y2, track_id, _, _, _ = track
                color = (0, 255, 0)
                cv2.rectangle(frame, (int(x1), int(y1)),
                              (int(x2), int(y2)), color, 2)
                cv2.putText(frame, f"ID {int(track_id)}", (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        else:
            tracker.predict()
            tracker.update(detections)
            tracks = tracker.tracks

            for track in tracks:
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue
                bbox = track.to_tlwh()
                x, y, w, h = map(int, bbox)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, str(track.track_id), (x, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        tock = time.time()
        fps = 1 / (tock - tick)
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Frame: {frame_idx}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Tracker: {tracker_name}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        out.write(frame)


        # Instead of updating a Streamlit placeholder here, push the frame to the queue.

        frame_idx += 1
    ttock = time.time()
    ttime = ttock - ttick
    print(f"Total time taken: {ttime:.2f} seconds")

    cap.release()
    out.release()
    cv2.destroyAllWindows()


run_tracker('ByteTrack', 'yolov8m', '/home/oybek/LITE/datasets/MOT20/train/MOT20-05/MOT20-05.mp4', 100, 'cuda:0', 'layer14', conf=0.25)















