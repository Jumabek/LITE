from __future__ import division, print_function, absolute_import
import logging
from fastreid.utils.checkpoint import Checkpointer
from fastreid.config import get_cfg
from torchvision import transforms
import torch
from PIL import Image
from opts import opt
from deep_sort.tracker import Tracker
from deep_sort.detection import Detection
from deep_sort import nn_matching
from application_util import visualization
import numpy as np
import cv2
import time
import os
from fastreid.modeling import build_model
from ultralytics import YOLO

logging.getLogger().setLevel(logging.ERROR)


def extract_lite_features(boxes, image, model):
    """
    Function to extract appearance features from FasterRCNN detections using LITE

    Parameters
    ----------
    model : YOLO
        YOLO model object.
        image : np.ndarray
        boxes : np.ndarray # Bboxes that is given by MOT Challenge
    
    Returns
    -------
    np.ndarray
        A NumPy array of appearance features.
    """
    features_list = []
    org_h, org_w = image.shape[:2]

    results = model.predict(image, classes=[0], verbose=False, 
    imgsz=opt.input_resolution, appearance_feature_layer=opt.appearance_feature_layer, conf=opt.min_confidence)

    appearance_feature_map = results[0].appearance_feature_map

    for box in boxes:
        x1, y1, x2, y2 = map(int, box[:4])

        h_map, w_map = appearance_feature_map.shape[1:]

        x1, x2, y1, y2 = map(
            int, [x1 * w_map / org_w, x2 * w_map / org_w, y1 * h_map / org_h, y2 * h_map / org_h])

        cropped_feature_map = appearance_feature_map[:, y1:y2, x1:x2]
        embedding = torch.mean(cropped_feature_map, dim=(1, 2)).unsqueeze(0)

        features_list.append(embedding)

    # Concatenate all embeddings into a single tensor
    if len(features_list) == 0:
        return np.array([])
    features_tensor = torch.cat(features_list, dim=0)

    # Convert the tensor to a NumPy array
    features_array = features_tensor.cpu().numpy()

    return features_array

def get_mot_detections(seq_dir, frame_index, reid_model, image):
    sequence_name = os.path.basename(seq_dir)
    dataset = opt.dataset
    det_path = f'datasets/{dataset}/train/{sequence_name}/det/det.txt'
    
    frcnn_boxes = []

    with open(det_path, 'r') as f:
        for line in f:
            parts = line.split(',')
            frame = int(parts[0])
            x, y, w, h = map(float, parts[2:6]) 
            x2 = x + w
            y2 = y + h
            conf = 1.0 # in FRCNN detections, confidence is not provided
            boxes = [x, y, x2, y2, conf, -1]
            
            if frame == frame_index:
                frcnn_boxes.append(boxes)
    
    appearance_features = get_apperance_features(frcnn_boxes, image, reid_model)
    frcnn_boxes = torch.tensor(frcnn_boxes).float()
    classes = torch.zeros(len(frcnn_boxes))

    return frcnn_boxes, appearance_features, classes



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


# This can be defined outside of the main function to avoid redefinition
def get_transform(size=(256, 128)):
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor()
    ])
    return transform


def get_apperance_features(boxes, image, reid_model):
    if opt.tracker_name == 'SORT':
        return [None] * len(boxes)

    if opt.tracker_name.startswith('LITE'):
        # it is called when only eval mot is True
        return extract_lite_features(boxes, image, reid_model)

    if opt.tracker_name == 'StrongSORT':
        # Convert the image to PIL format
        img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        transform = get_transform((256, 128))
        features_list = []

        crops = [
            transform(img_pil.crop((int(box[0]), int(box[1]), int(
                box[0]) + int(box[2]), int(box[1]) + int(box[3]))))
            * 255.0
            for box in boxes
        ]

        batch_size = 16 # batch size for feature extraction
        for i in range(0, len(crops), batch_size):
            batch = torch.stack(crops[i:i + batch_size]).cuda()
            features = reid_model(batch).detach().cpu().numpy()
            features_list.extend(features)

        return features_list

    elif opt.tracker_name == 'DeepSORT':
        features_list = []

        for box in boxes:
            x1, y1, x2, y2 = map(int, box[:4])
            crop = image[y1:y2, x1:x2]
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

            # Extract features
            feature = reid_model([crop_rgb]).detach().cpu().numpy().squeeze()

            features_list.append(feature)

        return features_list


def create_detections(seq_dir, frame_index, model, reid_model=None):
    detection_list = []

    # Get the specific image frame path
    # assuming frame names are like 000001.jpg, 000002.jpg, ...
    # KITTI has png extension

    ext = '.jpg' if opt.dataset in [
        'MOT17', 'MOT20', 'PersonPath22', 'VIRAT-S', 'DanceTrack'] else '.png'
    if opt.dataset == 'DanceTrack':
        img_path = os.path.join(seq_dir, 'img1', f'{frame_index:08}{ext}')
    else:
        img_path = os.path.join(seq_dir, 'img1', f'{frame_index:06}{ext}')

    if not os.path.exists(img_path):
        raise ValueError(f"Image path {img_path} doesn't exist.")

    # Load and predict
    image = cv2.imread(img_path)

    if opt.tracker_name.startswith('LITE'):
        assert opt.appearance_feature_layer is not None, "Please provide the appearance feature layer in order to use LITEDeepSORT"

    # Eval MOT challenge
    if opt.eval_mot:
        boxes, appearance_features, classes = get_mot_detections(seq_dir, frame_index, reid_model, image)
    else:
        # Custom YOLO detections
        yolo_results = model.predict(image, classes=opt.classes, verbose=False, imgsz=opt.input_resolution,
        conf=opt.min_confidence, appearance_feature_layer=opt.appearance_feature_layer)
        
        boxes = yolo_results[0].boxes.data.cpu().numpy()
        classes = yolo_results[0].boxes.cls.cpu().numpy()    
        if opt.tracker_name.startswith('LITE'):
            appearance_features = yolo_results[0].appearance_features.cpu().numpy()
        else:
            appearance_features = get_apperance_features(boxes, image, reid_model)

    for box, feature, cls in zip(boxes, appearance_features, classes):
        xmin, ymin, xmax, ymax, conf, _ = box
        x_tl = xmin
        y_tl = ymin
        width = xmax - xmin
        height = ymax - ymin
        bbox = (x_tl, y_tl, width, height)
   
        detection = Detection(bbox, conf, feature, cls)
        detection_list.append(detection)

    return detection_list


def load_strong_sort_reid_model(device='cuda:0'):
    cfg_path = 'checkpoints/FastReID/bagtricks_S50.yml'
    model_weights = 'checkpoints/FastReID/DukeMTMC_BoT-S50.pth'
    print("Loading ReID model on device", device)
    cfg = get_cfg()
    cfg.merge_from_file(cfg_path)
    cfg.MODEL.BACKBONE.PRETRAIN = False
    cfg.MODEL.WEIGHTS = model_weights
    model = build_model(cfg)  # Use build_model directly
    model.eval()
    Checkpointer(model).load(cfg.MODEL.WEIGHTS)

    return model


def load_deep_sort_model(device='cuda:0'):
    print("Loading DeepSORT model on device", device)
    from deep_appearance import DeepSORTApperanceExtractor
    model = DeepSORTApperanceExtractor(
    "checkpoints/FastReID/deepsort/original_ckpt.t7", device)

    return model


def run(sequence_dir, output_file, min_confidence, 
    nn_budget, display, device, verbose=False, visualize=False):
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
    logging.debug(f"min_confidence = {min_confidence}")

    seq_info = gather_sequence_info(sequence_dir)
    metric = nn_matching.NearestNeighborDistanceMetric(
        'cosine',
        opt.max_cosine_distance,
        nn_budget
    )

    tick = time.time()

    tracker = Tracker(metric, max_age=opt.max_age)
    results = []

    # Load the detection YOLO model
    model_path = opt.yolo_model + '.pt'
    model = YOLO(model_path)
    model.to(device)
    print(f'Loaded YOLO model: {model_path}')

    if opt.eval_mot:
        print('FRCNN Detections is being used.')

    reid_model = None
    if opt.tracker_name == 'StrongSORT':
        reid_model = load_strong_sort_reid_model(device)

    elif opt.tracker_name == 'DeepSORT':
        reid_model = load_deep_sort_model(device)

    elif opt.tracker_name.startswith('LITE'):
        reid_model = model  # ReID features are extracted for free from detector itself in LITE:Trackers

    def frame_callback(vis, frame_idx):
        detections = create_detections(sequence_dir, frame_idx, model, reid_model)

        if opt.ECC:
            tracker.camera_update(sequence_dir.split('/')[-1], frame_idx)

        tracker.predict()
        tracker.update(detections)

        # Update visualization.
        if visualize:
            image = cv2.imread(seq_info["image_filenames"][frame_idx], cv2.IMREAD_COLOR)
            vis.set_image(image.copy())
            vis.draw_trackers(tracker.tracks)
            # vis.draw_detections(detections)
            # vis.put_metadata()
            # vis.save_visualization()

        # Store results for evaluation.
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlwh()
            results.append([
                frame_idx, track.track_id, bbox[0], bbox[1], bbox[2], bbox[3], track.scores[0]])

    # Run tracker.
    if visualize:
        visualizer = visualization.Visualization(
            seq_info, update_ms=5, dir_save=opt.dir_save, display=display)
    else:
        visualizer = visualization.NoVisualization(seq_info)
    visualizer.run(frame_callback)

    #
    if verbose:
        print(f"storing predicted tracking results to {output_file}")
    if opt.dataset in ['MOT17', 'MOT20', 'PersonPath22', 'VIRAT-S', 'DanceTrack']:
        f = open(output_file, 'w')
        for row in results:
            # need to add conf score
            print('%d,%d,%.2f,%.2f,%.2f,%.2f,%.2f,-1,-1,-1,-1' % (
                row[0], row[1], row[2], row[3], row[4], row[5], row[6]), file=f)
    elif opt.dataset == 'KITTI':
        with open(output_file, 'w') as f:
            for row in results:
                # Set default values for fields that might not be available in your tracker
                if 7 in opt.classes:
                    object_type = 'car'
                else:
                    object_type = "pedestrian"  # or your default

                truncated = -1  # default, as this info might not be available
                occluded = -1  # default, as this info might not be available
                alpha = -10  # default, as this info might not be available
                dimensions = (-1, -1, -1)
                location = (-1000, -1000, -1000)

                # rotation_y = -10  # default, as this info might not be available
                # score = row[5]  # if you have confidence score
                # Write formatted data with maximum 2 decimal places for floating-point values
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
