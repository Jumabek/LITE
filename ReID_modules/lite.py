import torch
import numpy as np


class LITE:
    def __init__(self, model, appearance_feature_layer, imgsz=1280, conf=0.25, device='cuda:0'):
        self.model = model  # reid model and detection model of th LITE tracker is same
        self.appearance_feature_layer = appearance_feature_layer
        self.device = device
        self.imgsz = imgsz
        self.conf = conf

    def extract_appearance_features(self, image, boxes):
        """
        Executed only when evaluating MOT challenge using FRNN detections (not yolo detections) and 
        evaluting the strength of reid features using ground truth detections

        Args:
            image (np.ndarray): image frame
            boxes (np.ndarray): bounding boxes

        Returns:
            np.ndarray: appearance features
        """
        features_list = []
        org_h, org_w = image.shape[:2]

        results = self.model.predict(image, classes=[0], verbose=False,
        imgsz=self.imgsz, appearance_feature_layer=self.appearance_feature_layer, conf=self.conf, return_feature_map=True)

        appearance_feature_map = results[0].appearance_feature_map

        for box in boxes:
            x1, y1, x2, y2 = map(int, box[:4])

            h_map, w_map = appearance_feature_map.shape[1:]

            x1, x2, y1, y2 = map(int, [x1 * w_map / org_w - 1, x2 * w_map /
                                 org_w + 1, y1 * h_map / org_h - 1, y2 * h_map / org_h+1])
            # ensure the box is within the image (padding)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w_map, x2), min(h_map, y2)

            cropped_feature_map = appearance_feature_map[:, y1:y2, x1:x2]

            # embedding = torch.mean(cropped_feature_map, dim=(1, 2)).unsqueeze(0)

            # spatial_size reduction operation: converts 100x150x48 to fixed embd dimenstion of 1x48 for each bbox
            feature_mean = torch.mean(
                cropped_feature_map, dim=(1, 2))
            normalized_feature = feature_mean / \
                feature_mean.norm(p=2, dim=0, keepdim=True).unsqueeze(0)
            # END spatial_size reduction operation

            features_list.append(normalized_feature)

        if len(features_list) == 0:
            return np.array([])

        features_tensor = torch.cat(features_list, dim=0)
        features_array = features_tensor.cpu().numpy()

        return features_array
