from __future__ import division, print_function, absolute_import
from opts import opt
import argparse
from deep_sort.tracker import Tracker
from deep_sort import nn_matching
from application_util import visualization

import time
from ultralytics import YOLO
from tqdm import tqdm

from trackers import LITE, DeepSORT, StrongSORT
from utils import gather_sequence_info, create_detections

import warnings
warnings.filterwarnings("ignore")

def run(sequence_dir, output_file, 
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
    # Evaluate ReID if opt.reid is True
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
    
    reid_model = None

    if opt.eval_mot:
        print('Evaluating on MOT challenge...')

    if opt.tracker_name == 'StrongSORT':
        reid_model = StrongSORT(device=device)

    elif opt.tracker_name == 'DeepSORT':
        reid_model = DeepSORT(device=device)

    elif opt.tracker_name.startswith('LITE'):
        reid_model = LITE(model=model, appearance_feature_layer=opt.appearance_feature_layer, device=device)

    def frame_callback(vis, frame_idx):
        detections = create_detections(sequence_dir, frame_idx, model, reid_model)

        if opt.ECC:
            tracker.camera_update(sequence_dir.split('/')[-1], frame_idx)

        tracker.predict()
        tracker.update(detections)

        # Update visualization.if __name__ == "__main__":
    run(opt.sequence_dir, opt.output_file, opt.nn_budget, opt.display, opt.device, opt.verbose, opt.show)
    if visualize:
        visualizer = visualization.Visualization(
            seq_info, update_ms=5, dir_save=opt.dir_save, display=display)
    else:
        visualizer = visualization.NoVisualization(seq_info)

    num_frames = seq_info["max_frame_idx"] - seq_info["min_frame_idx"] + 1
    for frame_idx in tqdm(range(seq_info["min_frame_idx"], seq_info["max_frame_idx"] + 1), desc=f"Processing {seq_info['sequence_name']}"):
        frame_callback(visualizer, frame_idx)

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
