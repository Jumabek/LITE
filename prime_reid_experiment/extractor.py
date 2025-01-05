from tqdm import tqdm
import pandas as pd
import cv2
from os.path import join
import os
from ultralytics import YOLO
from reid_modules import LITE, StrongSORT, DeepSORT, OSNet
from reid_modules.gfn import GFN

class AppearanceExtractor:
    def __init__(self, tracker, dataset, sequence, split, output, device='cuda:0', appearance_feature_layer=None):
        self.tracker = tracker
        self.seq_dir = join('datasets', dataset, split, sequence)
        self.img_path = self.get_image_path()
        self.gt_path = self.get_gt_path()
        self.output = output
        self.device = device
        self.appearance_feature_layer = appearance_feature_layer
        self.model = self.load_model()

    def load_model(self):
        if self.tracker == 'LITE':
            model = YOLO('yolov8m.pt') # can be changed to from yolov8 to yolo11
            if self.appearance_feature_layer is None:
                 raise ValueError("Appearance feature layer is not provided. LITE model requires it.")
            # may change conf and imgsz by just passing them as arguments e.g. conf=0.25, imgsz=1280 # this is optional
            print(f"Appearance feature layer: {self.appearance_feature_layer}")
            return LITE(model=model, appearance_feature_layer=self.appearance_feature_layer)

        elif self.tracker == 'StrongSORT':
            return StrongSORT(device=self.device)

        elif self.tracker == 'DeepSORT':
            return DeepSORT(device=self.device)

        elif self.tracker == 'OSNet':
            return OSNet(device=self.device)
        
        elif self.tracker == 'GFN':
            return GFN(device=self.device)

        else:
            raise ValueError(f"Tracker {self.tracker} not supported.")

    def get_image_path(self):
        img_path = os.path.join(self.seq_dir, 'img1')

        if not os.path.exists(img_path):
            raise ValueError(f"Image path {img_path} doesn't exist.")

        return img_path

    def get_gt_path(self):
        gt_path = os.path.join(self.seq_dir, 'gt', 'gt.txt')

        if not os.path.exists(gt_path):
            raise ValueError(f"Ground truth path {gt_path} doesn't exist.")

        return gt_path

    def get_image(self, frame_index):
        img_file = os.path.join(self.img_path, f'{frame_index:06}.jpg')

        if not os.path.exists(img_file):
            raise ValueError(f"Image file {img_file} doesn't exist.")

        return cv2.imread(img_file)

    def extract_features(self):
        gt = pd.read_csv(self.gt_path, sep=',', header=None)
        features = []

        frame_indices = sorted(gt[0].unique())
        for frame_index in tqdm(frame_indices, desc="Extracting features"):
            image = self.get_image(frame_index)
            bbox = gt[(gt[0] == frame_index) & (
                gt[7] == 1)][[2, 3, 4, 5]].values
            # convert bbox to x1, y1, x2, y2 format
            bbox[:, 2] += bbox[:, 0]
            bbox[:, 3] += bbox[:, 1]
            track_ids = gt[(gt[0] == frame_index) & (gt[7] == 1)][1].values

            assert len(bbox) == len(track_ids)

            appearances = self.model.extract_appearance_features(image, bbox)

            frame_features = pd.DataFrame({'id': track_ids, 'features': [
                                          appearance.flatten() for appearance in appearances]})
            frame_features['frame'] = frame_index
            features.append(frame_features)

        return pd.concat(features).reset_index(drop=True)
