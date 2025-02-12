import torch
from torchvision import transforms

from PIL import Image
import cv2

from fastreid.utils.checkpoint import Checkpointer
from fastreid.config import get_cfg
from fastreid.modeling import build_model

class StrongSORT:
    def __init__(self, batch_size=16, device='cuda:0'):
        self.device = device
        self.batch_size = batch_size
        self.model = self.load_model()
        self.transform = self.get_transform()

    def load_model(self):
        cfg_path = 'checkpoints/FastReID/bagtricks_S50.yml'
        model_weights = 'checkpoints/FastReID/DukeMTMC_BoT-S50.pth'
        print("Loading StrongSORT model on device", self.device)
        cfg = get_cfg()
        cfg.merge_from_file(cfg_path)
        cfg.MODEL.BACKBONE.PRETRAIN = False
        cfg.MODEL.HEADS.NUM_CLASSES = 702  # Ensure the number of classes matches the checkpoint
        cfg.MODEL.WEIGHTS = model_weights
        model = build_model(cfg)
        #model.heads = build_heads(cfg, model.backbone.output_shape())
        model.eval()
        Checkpointer(model).load(cfg.MODEL.WEIGHTS)

        return model
    
    def get_transform(self, size=(256, 128)):
        transform = transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor()
        ])
        return transform

    def extract_appearance_features(self, image, boxes):
        img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        features_list = []
        # make boxes integers
        boxes = boxes.astype(int)

        crops = [
            self.transform(img_pil.crop((box[0], box[1], box[2], box[3])))
            * 255.0
            for box in boxes
        ]

        for i in range(0, len(crops), self.batch_size):
            batch = torch.stack(crops[i:i + self.batch_size]).cuda()
            features = self.model(batch).detach().cpu().numpy()
            features_list.extend(features)

        return features_list
