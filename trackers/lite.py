import torch
import numpy as np

class LITE:
    def __init__(self, model, appearance_feature_layer, imgsz=1280, conf=0.25, device='cuda:0'):
        self.model = model # reid model and detection model of th LITE tracker is same
        self.appearance_feature_layer = appearance_feature_layer
        self.device = device
        self.imgsz = imgsz
        self.conf = conf
        
    def extract_appearance_features(self, image, boxes):
        features_list = []
        org_h, org_w = image.shape[:2]

        results = self.model.predict(image, classes=[0], verbose=False,
        imgsz=self.imgsz, appearance_feature_layer=self.appearance_feature_layer, conf=self.conf)

        appearance_feature_map = results[0].appearance_feature_map

        for box in boxes:
            x1, y1, w, h = map(int, box[:4])
            x2 = x1 + w
            y2 = y1 + h

            h_map, w_map = appearance_feature_map.shape[1:]

            x1, x2, y1, y2 = map(int, [x1 * w_map / org_w, x2 * w_map / org_w, y1 * h_map / org_h, y2 * h_map / org_h])

            cropped_feature_map = appearance_feature_map[:, y1:y2, x1:x2]

            embedding = torch.mean(cropped_feature_map, dim=(1, 2)).unsqueeze(0)

            features_list.append(embedding)

        if len(features_list) == 0:
            return np.array([])
        
        features_tensor = torch.cat(features_list, dim=0)
        features_array = features_tensor.cpu().numpy()

        return features_array
