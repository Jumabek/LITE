import os

import torch
import cv2
import numpy as np
from PIL import Image

from facenet_pytorch import InceptionResnetV1
from sklearn.metrics.pairwise import cosine_similarity

class FaceNet:
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.resnet = (
            InceptionResnetV1(pretrained="vggface2", classify=False)
            .eval()
            .to(self.device)
        )

    def extract_appearance_features(self, image, boxes):
        faces = []
        for box in boxes:
            x1, y1, x2, y2 = map(int, box[:4])
            crop = image[y1:y2, x1:x2]
            faces.append(crop)
        
        face_embs = self.compute_embeddings(faces)

        return face_embs

    def compute_embeddings(self, faces):
        resized_faces = [cv2.resize(face, (160, 160)) for face in faces]
        rgb_faces = [cv2.cvtColor(face, cv2.COLOR_BGR2RGB) for face in resized_faces]
        
        face_tensors = torch.tensor(np.array(rgb_faces)).permute(0, 3, 1, 2).float().to(self.device) / 255.0
        
        with torch.no_grad():
            embeddings = self.resnet(face_tensors).cpu().numpy()
    
        return embeddings