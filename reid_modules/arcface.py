import onnxruntime as ort
import numpy as np
import cv2
import os

class ArcFace:
    def __init__(self, device='cuda'):
        self.device = device
        self.session = self.load_model()
        self.input_name = self.session.get_inputs()[0].name

    def load_model(self):
        model_path = '/home/oybek/.insightface/models/buffalo_l/w600k_r50.onnx'
        providers = ['CUDAExecutionProvider'] if self.device == 'cuda' else ['CPUExecutionProvider']
        return ort.InferenceSession(model_path, providers=providers)

    def preprocess(self, image):
        image = cv2.resize(image, (112, 112))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32)
        image = np.transpose(image, (2, 0, 1))  # HWC to CHW
        image = np.expand_dims(image, axis=0)   # Add batch dim
        image = (image - 127.5) / 128.0          # Normalize to [-1, 1]
        return image

    def extract_appearance_features(self, image, bboxes):
        features_list = []

        for box in bboxes:
            x1, y1, x2, y2 = map(int, box[:4])
            face = image[y1:y2, x1:x2]

            # if face.size == 0:
            #     features_list.append(np.zeros(512))
            #     continue

            face_input = self.preprocess(face)
            emb = self.session.run(None, {self.input_name: face_input})[0][0]

            norm = np.linalg.norm(emb)
            if norm == 0 or np.isnan(norm):
                emb = np.zeros(512)
            else:
                emb = emb / norm

            features_list.append(emb)

        return features_list