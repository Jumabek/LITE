from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from deep_sort import nn_matching
from ultralytics import YOLO
import numpy as np
import cv2
from opts import opt

def process_video(video_path):
    # Open the video file or camera
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    model = YOLO("yolo11m.pt")
    print(model.info(verbose=True))

    metric = nn_matching.NearestNeighborDistanceMetric(
        'cosine',
        0.3,
        100
    )

    tracker = Tracker(metric)

    frame_number = 0  

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_number += 1 

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        font_color = (255, 255, 255) 
        font_thickness = 2
        frame_with_text = frame.copy()
        cv2.putText(frame_with_text, f"Frame: {frame_number}", (
            20, 50), font, font_scale, font_color, font_thickness)

        yolo_results = model.predict(
            frame_with_text, classes=[0], verbose=False, imgsz=1280, appearance_feature_layer='layer0', conf=.25)

        boxes = yolo_results[0].boxes.data.cpu().numpy()
        appearance_features = yolo_results[0].appearance_features.cpu().numpy()
        
        detections = []
        for box, feature in zip(boxes, appearance_features):
            xmin, ymin, xmax, ymax, conf, _ = box
            x_tl = xmin
            y_tl = ymin
            width = xmax - xmin
            height = ymax - ymin
            bbox = (x_tl, y_tl, width, height)
            detection = Detection(bbox, conf, feature)

            detections.append(detection)

        # Run non-maxima suppression.
        boxes = np.array([d.tlwh for d in detections])
        tracker.predict()
        tracker.update(detections)

        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue

            bbox = track.to_tlwh()
            x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

            cv2.rectangle(frame_with_text, (x, y),
                          (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame_with_text, str(track.track_id), (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.75, (0, 255, 0), 2)

        cv2.imshow('Frame', frame_with_text)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    process_video(opt.source)


if __name__ == "__main__":
    main()
