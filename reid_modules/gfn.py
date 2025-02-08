# Torch libs
import torch
import gdown
import logging
logging.getLogger("torch").setLevel(logging.ERROR)


## Disable nvfuser for now
torch._C._jit_override_can_fuse_on_cpu(False)
torch._C._jit_override_can_fuse_on_gpu(False)
torch._C._jit_set_texpr_fuser_enabled(False)
torch._C._jit_set_nvfuser_enabled(False)

# Libs for data pre-processing
import cv2
import numpy as np
from albumentations.augmentations.geometric import functional as FGeometric
import torchvision.transforms.functional as TF

# Libs for loading images
import os
import ssl
## Avoid SSL error
ssl._create_default_https_context = ssl._create_unverified_context


class GFN:
    def __init__(self, device='cuda'):
        self.device = device
        self.model = self.load_model()

    def get_detections(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_tensor = self.to_tensor(image)

        with torch.no_grad():
            detections = self.model([image_tensor], inference_mode='both')
            boxes = detections[0]['det_boxes'].cpu().numpy()
            scores = detections[0]['det_scores'].cpu().numpy()
            appearances = detections[0]['det_emb'].cpu().numpy()
        
        # concat boxes, scores and add 0 for class
        detections = np.concatenate((boxes, scores[:, None], np.zeros((len(scores), 1))), axis=1)
         
        return detections, appearances
        
    def load_model(self):
        model_path = 'gfn/cuhk_final_convnext-base_e30.torchscript.pt'
        # Check if model exists
        if not os.path.exists(model_path):
            print("Model not found, downloading...")
            # Google Drive file ID from the provided link
            file_id = '1pmka6VZmmxaQnuxsxQ_qOUjXN21mbav-'
            # Use gdown to download the model
            os.makedirs('gfn', exist_ok=True)
            gdown.download(f'https://drive.google.com/uc?export=download&id={file_id}', model_path, quiet=False)      
        try:
            # Load the model
            model = torch.jit.load(model_path)
            model.eval()
        except RuntimeError as e:
            raise RuntimeError(f"Error loading model: {e}")
        
        return model
    
    def normalize(self, tensor, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        mean = torch.FloatTensor(mean).view(1, 1, 3)
        std = torch.FloatTensor(std).view(1, 1, 3)

        return tensor.div(255.0).sub(mean).div(std)

    # Denormalize image tensor using ImageNet stats
    def denormalize(self, tensor, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        mean = torch.FloatTensor(mean).view(1, 1, 3)
        std = torch.FloatTensor(std).view(1, 1, 3)
        return tensor.mul(std).add(mean)


    # Resize image (numpy array) to fit in fixed size window
    def window_resize(self, img, min_size=2000, max_size=2000, interpolation=cv2.INTER_LINEAR):
        height, width = img.shape[:2]
        image_min_size = min(width, height)
        image_max_size = max(width, height)
        scale_factor = min_size / image_min_size
        if image_max_size * scale_factor > max_size:
            return FGeometric.longest_max_size(img, max_size=max_size, interpolation=interpolation)
        else:
            return FGeometric.smallest_max_size(img, max_size=min_size, interpolation=interpolation)

    def to_tensor(self, image):
        # arr = np.array(image)
        # arr_wrs = self.window_imageimage)
        tsr = torch.FloatTensor(image)
        tsr_norm = self.normalize(tsr)
        tsr_input = tsr_norm.permute(2, 0, 1).to(self.device)

        return tsr_input

    def extract_appearance_features(self, image, boxes):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        boxes = np.array(boxes)
        boxes = boxes[:, :4]

        image_tensor = self.to_tensor(image)
        box_tensor = torch.FloatTensor(boxes).to(self.device)
        box_targets = [{'boxes': box_tensor}]
        
        with torch.no_grad():
            detections = self.model([image_tensor], box_targets, inference_mode='both')
            embs = detections[0]['gt_emb']

        features_array = embs.cpu().numpy()
        
        return features_array
