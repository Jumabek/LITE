import numpy as np
import sys
sys.path.append('yolo_tracking/')
from boxmot.appearance.reid_auto_backend import ReidAutoBackend


# This reid extractor is used in BoTSORT, DeepOC-SORT
class OSNet:
    def __init__(self, device='cuda:0', fp16=False):
        self.device = device
        self.fp16 = fp16
        self.model = self.load_model()

    def load_model(self):
        rab = ReidAutoBackend(device=self.device, half=self.fp16)
        return rab.get_backend()

    def extract_appearance_features(self, image, bbox):
        features_list = []
        full_boxes = []
        for box in bbox:
            x1, y1, x2, y2 = map(int, box[:4])

            full_boxes.append([x1, y1, x2, y2])

        features = self.model.get_features(np.array(full_boxes), image)

        for i in range(features.shape[0]):
            feat = features[i, :]
            features_list.append(feat)
            
        return features_list
