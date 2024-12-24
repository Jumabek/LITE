import torch
from opts import opt
from deep_sort.detection import Detection
import numpy as np
import cv2
import os

def get_mot_detections(seq_dir, frame_index, reid_model, image):
    det_path = f'datasets/{opt.dataset}/train/{os.path.basename(seq_dir)}/det/det.txt'
    
    frcnn_boxes = []

    with open(det_path, 'r') as f:
        for line in f:
            parts = line.split(',')
            frame = int(parts[0])
            x, y, w, h = map(float, parts[2:6]) 
            boxes = [x, y, w, h]
            
            if frame == frame_index:
                frcnn_boxes.append(boxes)
                
    appearance_features = get_apperance_features(image, frcnn_boxes, reid_model)

    frcnn_boxes = torch.tensor(frcnn_boxes).int()
    classes = torch.zeros(len(frcnn_boxes))
    confs = torch.ones(len(frcnn_boxes))

    return frcnn_boxes, appearance_features, classes, confs

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
        boxes, appearance_features, classes, confs = get_mot_detections(seq_dir, frame_index, reid_model, image)
    
    else:
        # Custom YOLO detections
        yolo_results = model.predict(image, classes=opt.classes, verbose=False, imgsz=opt.input_resolution,
        conf=opt.min_confidence, appearance_feature_layer='layer0')
        
        boxes = yolo_results[0].boxes.xywh.cpu().numpy()
        confs = yolo_results[0].boxes.conf.cpu().numpy()
        classes = yolo_results[0].boxes.cls.cpu().numpy()

        if opt.tracker_name.startswith('LITE'):
            # lite do not need to extract appearance features again for boxes
            appearance_features = yolo_results[0].appearance_features.cpu().numpy()
        else:
            appearance_features = get_apperance_features(image, boxes, reid_model)
        

    for box, conf, feature, cls in zip(boxes, confs, appearance_features, classes):
        xmin, ymin, width, height = box
        x_tl = xmin
        y_tl = ymin

        bbox = (x_tl, y_tl, width, height)
        detection = Detection(bbox, conf, feature, cls)
        detection_list.append(detection)

    return detection_list