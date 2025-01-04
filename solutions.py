from deep_sort.detection import Detection
from application_util import preprocessing
from deep_sort.tracker import Tracker
from deep_sort import nn_matching
import numpy as np
import cv2
from opts import opt
import os
from ultralytics import YOLO, solutions

def process_video(video_path):

    # Open the video file or camera
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    model = YOLO("yolov8m.pt")
    print(model.info(verbose=True))

    nms_max_overlap = 1.0
    metric = nn_matching.NearestNeighborDistanceMetric(
        'cosine',
        0.3,
        100
    )

    tracker = Tracker(metric)

    frame_number = 0  # Initialize frame number

    w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

    # Define region points
    # region_points = [(750, 700), (1200, 700), (1250, 400), (1150, 300)]
    region_points = [(850, 700), (1250, 300)]

    # Path to json file, that created with above point selection app
    polygon_json_path = "demo/videos/bounding_boxes.json"

    output_dir = "demo_output_videos"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{opt.solution}.mp4")

    video_writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    counter = solutions.ObjectCounter(
        view_img=True,
        reg_pts=region_points,
        names=model.names,
        draw_tracks=True,
        line_thickness=2,
    )

    heatmap_obj = solutions.Heatmap(
    colormap=cv2.COLORMAP_JET,
    view_img=True,
    shape="circle",
    names=model.names,
    )

    management = solutions.ParkingManagement(
        model_path="yolov8m.pt"
    )


    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_number += 1  

        # Process each frame
        classes = [2] if opt.solution == "parking_management" else [0]
        yolo_results = model.predict(
            frame, classes=classes, verbose=False, imgsz=1280, appearance_feature_layer='layer0', conf=.25)

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
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(
            boxes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        tracker.predict()
        tracker.update(detections)

        if opt.solution == "object_counter":
            frame = counter.start_counting(frame, tracker.tracks)
        elif opt.solution == "heatmap":
            frame = heatmap_obj.generate_heatmap(frame, tracker.tracks)
        elif opt.solution == "parking_management":
            json_data = management.parking_regions_extraction(polygon_json_path)
            management.process_data(json_data, frame, tracker.tracks)
            management.display_frames(frame)

        video_writer.write(frame)

        # Press Q on keyboard to exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()


def main():
    process_video(opt.source)


if __name__ == "__main__":
    main()
