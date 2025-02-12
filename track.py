from __future__ import division, print_function, absolute_import
import logging
logging.getLogger().setLevel(logging.ERROR)

from opts import opt
from deep_sort.tracker import Tracker
from deep_sort import nn_matching
from application_util import visualization
import cv2
import time
from ultralytics import YOLO
from tqdm import tqdm

from reid_modules import LITE, DeepSORT, StrongSORT, GFN

import torch
from opts import opt
from deep_sort.detection import Detection
import numpy as np
import cv2
import os

import warnings
warnings.filterwarnings("ignore")

def get_mot_detections(seq_dir, frame_index, reid_model, image):
    det_path = f'datasets/{opt.dataset}/train/{os.path.basename(seq_dir)}/det/det.txt'
    
    frcnn_boxes = []

    with open(det_path, 'r') as f:
        for line in f:
            parts = line.split(',')
            frame = int(parts[0])
            x, y, w, h = map(float, parts[2:6])
            x2, y2 = x + w, y + h 
            boxes = [x, y, x2, y2, 1, -1]
            
            if frame == frame_index:
                frcnn_boxes.append(boxes)
                
    appearance_features = get_apperance_features(image, frcnn_boxes, reid_model)

    frcnn_boxes = torch.tensor(frcnn_boxes).int()

    return frcnn_boxes, appearance_features

def gather_sequence_info(sequence_dir):
    """Gather sequence information, such as image filenames, detections,
    groundtruth (if available).

    Parameters
    ----------
    sequence_dir : str
        Path to the MOTChallenge sequence directory.
    detection_file : str
        Path to the detection file.

    Returns
    -------
    Dict
        A dictionary of the following sequence information:

        * sequence_name: Name of the sequence
        * image_filenames: A dictionary that maps frame indices to image
          filenames.
        * detections: A numpy array of detections in MOTChallenge format.
        * groundtruth: A numpy array of ground truth in MOTChallenge format.
        * image_size: Image size (height, width).
        * min_frame_idx: Index of the first frame.
        * max_frame_idx: Index of the last frame.

    """
    image_dir = os.path.join(sequence_dir, "img1")
    image_filenames = {
        int(os.path.splitext(f)[0]): os.path.join(image_dir, f)
        for f in os.listdir(image_dir)}
    groundtruth_file = os.path.join(sequence_dir, "gt/gt.txt")

    detections = None
    # if detection_file is not None:
    #     detections = np.load(detection_file)
    groundtruth = None
    if os.path.exists(groundtruth_file):
        groundtruth = np.loadtxt(groundtruth_file, delimiter=',')

    if len(image_filenames) > 0:
        image = cv2.imread(next(iter(image_filenames.values())),
                           cv2.IMREAD_GRAYSCALE)
        image_size = image.shape
    else:
        image_size = None

    if len(image_filenames) > 0:
        min_frame_idx = min(image_filenames.keys())
        max_frame_idx = max(image_filenames.keys())
    else:
        min_frame_idx = int(detections[:, 0].min())
        max_frame_idx = int(detections[:, 0].max())

    info_filename = os.path.join(sequence_dir, "seqinfo.ini")
    if os.path.exists(info_filename):
        with open(info_filename, "r") as f:
            line_splits = [l.split('=') for l in f.read().splitlines()[1:]]
            info_dict = dict(
                s for s in line_splits if isinstance(s, list) and len(s) == 2)

        update_ms = 1000 / int(info_dict["frameRate"])
    else:
        update_ms = None

    feature_dim = detections.shape[1] - 10 if detections is not None else 0
    seq_info = {
        "sequence_name": os.path.basename(sequence_dir),
        "image_filenames": image_filenames,
        "detections": detections,
        "groundtruth": groundtruth,
        "image_size": image_size,
        "min_frame_idx": min_frame_idx,
        "max_frame_idx": max_frame_idx,
        "feature_dim": feature_dim,
        "update_ms": update_ms
    }
    return seq_info

def get_apperance_features(image, boxes, reid_model):
    if opt.tracker_name == 'SORT': # SORT does not need appearance features 
        return [None] * len(boxes)

    else:
        appearance_features = reid_model.extract_appearance_features(image, boxes)
        return appearance_features


def create_detections(seq_dir, frame_index, model, reid_model=None):
    detection_list = []
    
    ext = '.jpg' if opt.dataset in [
        'MOT17', 'MOT20', 'PersonPath22', 'VIRAT-S', 'DanceTrack'] else '.png' # KITTI has png extension
    
    # assuming frame names are like 000001.jpg, 000002.jpg, ...
    if opt.dataset == 'DanceTrack':
        img_path = os.path.join(seq_dir, 'img1', f'{frame_index:08}{ext}')
    else:
        img_path = os.path.join(seq_dir, 'img1', f'{frame_index:06}{ext}')

    if not os.path.exists(img_path):
        raise ValueError(f"Image path {img_path} doesn't exist.")

    # Load and predict
    image = cv2.imread(img_path)

    # Eval MOT challenge
    if opt.eval_mot:
        boxes, appearance_features = get_mot_detections(seq_dir, frame_index, reid_model, image)
    
    # GFN detector
    elif opt.tracker_name == 'GFN':
        boxes, appearance_features = reid_model.get_detections(image)

    else:
        # Custom YOLO detections
        yolo_results = model.predict(image, classes=opt.classes, verbose=False, imgsz=opt.input_resolution,
        conf=opt.min_confidence, appearance_feature_layer=opt.appearance_feature_layer, return_feature_map=False)
        
        boxes = yolo_results[0].boxes.data.cpu().numpy()
        if opt.tracker_name.startswith('LITE'):
            # lite do not need to extract appearance features again for boxes
            appearance_features = yolo_results[0].appearance_features.cpu().numpy()
        else:
            appearance_features = get_apperance_features(image, boxes, reid_model)
        

    for box, feature in zip(boxes, appearance_features):
        xmin, ymin, xmax, ymax, conf, _ = box
        conf = float(conf)
        x_tl, y_tl = map(int, (xmin, ymin))
        width, height = map(int, (xmax - xmin, ymax - ymin))
        bbox = (x_tl, y_tl, width, height)
        detection = Detection(bbox, conf, feature)
        detection_list.append(detection)

    return detection_list

def run(sequence_dir, output_file, 
    nn_budget, device, verbose=True, visualize=False):
    """Run multi-target tracker on a particular sequence.

    Parameters
    ----------
    sequence_dir : str
        Path to the MOTChallenge sequence directory.
    detection_file : str
        Path to the detections file.
    output_file : str
        Path to the tracking output file. This file will contain the tracking
        results on completion.
    min_confidence : float
        Detection confidence threshold. Disregard all detections that have
        a confidence lower than this value.
    max_cosine_distance : float
        Gating threshold for cosine distance metric (object appearance).
    nn_budget : Optional[int]
        Maximum size of the appearance descriptor gallery. If None, no budget
        is enforced.
    display : bool
        If True, show visualization of intermediate tracking results.

    """
    # Evaluate ReID if opt.reid is True
    seq_info = gather_sequence_info(sequence_dir)
    metric = nn_matching.NearestNeighborDistanceMetric(
        'cosine',
        opt.max_cosine_distance,
        nn_budget
    )
    tracker = Tracker(metric, max_age=opt.max_age)

    tick = time.time()

    results = []

    # Load the detection YOLO model    
    model_path = opt.yolo_model + '.pt'
    model = YOLO(model_path)
    model.to(device)
    
    
    if opt.eval_mot:
        tqdm.write('Evaluating on MOT challenge...')


    reid_model = None
    if opt.tracker_name == 'StrongSORT':
        reid_model = StrongSORT(device=device)

    elif opt.tracker_name == 'DeepSORT':
        reid_model = DeepSORT(device=device)

    elif opt.tracker_name == 'GFN':
        reid_model = GFN(device=device)

    elif opt.tracker_name.startswith('LITE'):
        reid_model = LITE(model=model, appearance_feature_layer=opt.appearance_feature_layer, device=device)


    def frame_callback(vis, frame_idx):
        # Process frame
        detections = create_detections(sequence_dir, frame_idx, model, reid_model)
        tracker.predict()
        tracker.update(detections) 

        # Update visualization
        if visualize:
            image = cv2.imread(seq_info["image_filenames"][frame_idx], cv2.IMREAD_COLOR)
            vis.set_image(image.copy())
            vis.draw_trackers(tracker.tracks)
            # vis.draw_detections(detections)
            # vis.put_metadata()
            # vis.save_visualization()

        # Store results
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlwh()
            results.append([
                frame_idx, track.track_id, bbox[0], bbox[1], bbox[2], bbox[3], track.scores[0]])

    # Run tracker.
    if visualize:
        try:
            visualizer = visualization.Visualization(
                seq_info, update_ms=5, dir_save=opt.dir_save)
        except cv2.error as e:
            print(f"OpenCV error: {e}. Disabling visualization.")
            visualize = False

    if not visualize:
        visualizer = visualization.NoVisualization(seq_info)
    
    visualizer.run(frame_callback)

    if verbose:
        print(f"Storing predicted tracking results to \033[1m{output_file}\033[0m")
    if opt.dataset in ['MOT17', 'MOT20', 'PersonPath22', 'VIRAT-S', 'DanceTrack']:
       
        f = open(output_file, 'w')

        for row in results:
            print('%d,%d,%.2f,%.2f,%.2f,%.2f,%.2f,-1,-1,-1,-1' % (
            row[0], row[1], row[2], row[3], row[4], row[5], row[6]), file=f)
            
    elif opt.dataset == 'KITTI':
        with open(output_file, 'w') as f:
            for row in results:
                if 7 in opt.classes:
                    object_type = 'car'
                else:
                    object_type = "pedestrian"

                truncated = -1  
                occluded = -1 
                alpha = -10  
                dimensions = (-1, -1, -1)
                location = (-1000, -1000, -1000)

                f.write(f"{row[0]} {row[1]} {object_type} {truncated} {occluded} {alpha:.2f} "
                        f"{row[2]:.2f} {row[3]:.2f} {(row[2]+row[4]):.2f} {(row[3]+row[5]):.2f} "
                        f"{' '.join(map(lambda l: f'{l:.2f}', location))} "
                        f"{' '.join(map(lambda d: f'{d:.2f}', dimensions))} \n"
                        )
    if not verbose:
        return

    tock = time.time()

    time_spent_for_the_sequence = tock - tick
    time_info_s = f'time: {time_spent_for_the_sequence:.0f}s'

    num_frames = (seq_info["max_frame_idx"] - seq_info["min_frame_idx"])
    avg_time_per_frame = (time_spent_for_the_sequence) / num_frames

    print(f'Avg. processing speed: {1000*avg_time_per_frame:.0f} millisecond per frame')
    print(f'{time_info_s} | Avg FPS: {1/avg_time_per_frame:.1f}')
    print(f'Finished sequence \033[32m{seq_info["sequence_name"]}\033[0m')

