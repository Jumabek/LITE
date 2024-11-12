import sys
import os
import argparse
from contextlib import contextmanager
import cv2
import numpy as np
import pandas as pd
from collections import defaultdict
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pickle
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics.pairwise import cosine_similarity
from ultralytics import YOLO
from lite_deepsort_app import load_reid_model, load_deep_sort_model, get_transform
import torch
from PIL import Image
search_root = 'yolo_tracking/'
for root, dirs, files in os.walk(search_root):
    if 'boxmot' in dirs:
        sys.path.append(root)
        print(f"'boxmot' directory found. Added {root} to sys.path.")
        break
else:
    print("Error: 'boxmot' directory not found.")
from boxmot.appearance.reid_auto_backend import ReidAutoBackend # type: ignore

@contextmanager
def suppress_stdout():
    """Context manager to suppress standard output."""
    with open(os.devnull, 'w') as fnull:
        old_stdout = sys.stdout
        sys.stdout = fnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Track appearance feature extraction and similarity analysis")
    parser.add_argument("--save", action="store_true", default=True, help="Save output files (ROC curve, ReID distribution).")
    parser.add_argument("--output_path", type=str, default="reid/data", help="Path to save outputs.")
    parser.add_argument("--use_cache", action="store_true", default=True, help="Use cached features if available.")
    parser.add_argument("--dataset", type=str, default='MOT20', help="Name of the dataset.")
    parser.add_argument("--seq_name", type=str, default='MOT20-01', help="Name of the MOT sequence.")
    parser.add_argument("--split", type=str, default='train', choices=['train', 'test'], help="Specify the split of the dataset.")
    parser.add_argument("--tracker", type=str, default='OSNet', choices=['LITEDeepSORT', 'StrongSORT', 'DeepSORT', 'OSNet', 'all'], help="Specify the tracker model to use.")
    return parser.parse_args()
    # Example:
    # python reid.py --tracker DeepOCSORT --dataset MOT20 --seq_name MOT20-01 --split train --save --output_path reid/data


def load_models(tracker_name):
    """Loads the required models for different trackers."""
    if tracker_name == 'LITEDeepSORT':
        print('Loading YOLOv8m model...')
        model = YOLO('yolov8m.pt')
     
    elif tracker_name == 'StrongSORT':
        print('Loading StrongSORT model...')
        model = load_reid_model('cuda:0')
    
    elif tracker_name == 'DeepSORT':
        model = load_deep_sort_model('cuda:0')

    elif tracker_name == 'OSNet':
        print('Loading OSNet model...')
        fp16 = False
        rab = ReidAutoBackend(device='cuda:0', half=fp16)
        model = rab.get_backend()

    return model

def extract_appearance_features(image, boxes, tracker_name, model):
    """Extracts appearance features based on the specified tracker."""

    features_list = []

    if tracker_name == 'LITEDeepSORT':
        yolo_results = model.predict(image, classes=[
                                          0], verbose=False, imgsz=1280, appearance_feature_layer='layer0', conf=0.25)
        appearance_feature_map = yolo_results[0].appearance_feature_map

        for box in boxes:
            x1, y1, w, h = map(int, box[:4])
            x2, y2 = x1 + w, y1 + h

            h_map, w_map = appearance_feature_map.shape[1:]
            x1, x2, y1, y2 = map(
                int, [x1 * w_map / 1920, x2 * w_map / 1920, y1 * h_map / 1080, y2 * h_map / 1080])

            cropped_feature_map = appearance_feature_map[:, y1:y2, x1:x2]
            embedding = torch.mean(cropped_feature_map,
                                   dim=(1, 2)).unsqueeze(0)
            features_list.append(embedding.cpu().numpy())

    elif tracker_name == 'StrongSORT':
        img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        transform = get_transform((256, 128))

        batch = [img_pil.crop((int(box[0]), int(box[1]), int(box[0]) + int(box[2]), int(box[1]) + int(box[3]))) for box in boxes]
        batch = [transform(crop) * 255. for crop in batch]

        if batch:
            batch = torch.stack(batch, dim=0).cuda()
            outputs = model(batch).detach().cpu().numpy()

            for output in outputs:
                features_list.append(output)


    elif tracker_name == 'DeepSORT':
        for box in boxes:
            x1, y1, w, h = map(int, box[:4])
            x2 = x1 + w
            y2 = y1 + h

            crop_rgb = cv2.cvtColor(image[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)

            feature = model([crop_rgb]).detach().cpu().numpy().squeeze()

            features_list.append(feature)

    elif tracker_name == 'OSNet':
        full_boxes = []
        for box in boxes:
            x1, y1, w, h = map(int, box[:4])
            x2 = x1 + w
            y2 = y1 + h
            full_boxes.append([x1, y1, x2, y2])

        features = model.get_features(np.array(full_boxes), image)

        for i in range(features.shape[0]):
            feat = features[i, :]
            features_list.append(feat)
            
    return features_list


def predict_features(tracker_name, dataset, seq_name, split, use_cache, output_path):
    """Caches features for each frame or loads from cache if available."""

    seq_path = f'datasets/{dataset}/{split}/{seq_name}/img1'                
    seq_gt_path = f'datasets/{dataset}/{split}/{seq_name}/gt/gt.txt' 
    
    
    os.makedirs(output_path, exist_ok=True)

    pkl_file = os.path.join(output_path, f'{tracker_name}_{seq_name}_features.pkl')
    
    if use_cache and os.path.exists(pkl_file):
        print(f"Loading cached features from {pkl_file}")
        with open(pkl_file, 'rb') as f:
            return pickle.load(f)
    
    print('Processing similarity distribution for', tracker_name)
    gt = pd.read_csv(seq_gt_path, sep=',', header=None,
                     names=['frame', 'id', 'x', 'y', 'w', 'h', 'conf', 'class', 'visibility'])
    img_files = sorted(glob.glob(os.path.join(seq_path, '*.jpg')))
    gt_apps = defaultdict(dict)
    model = load_models(tracker_name)

    for frame_num, img_file in tqdm(enumerate(img_files), total=len(img_files), desc="Processing frames"):
        img = cv2.imread(img_file)
        bbox = gt[(gt['frame'] == frame_num + 1) & (gt['class'] == 1)][['x', 'y', 'w', 'h']].values
        track_ids = gt[(gt['frame'] == frame_num + 1) & (gt['class'] == 1)]['id'].values

        assert len(bbox) == len(track_ids)
        
        with suppress_stdout():
            appearances = extract_appearance_features(img, bbox, tracker_name, model)

        df = pd.DataFrame({'id': track_ids, 'features': appearances})
        gt_apps[frame_num + 1] = df

    features = pd.concat(gt_apps).reset_index(level=0).rename(columns={'level_0': 'frame'})

    with open(pkl_file, 'wb') as f:
        pickle.dump(features, f)
    print(f"Features cached to {pkl_file}")
    return features


def calculate_similarity(features):
    """Calculates positive and negative similarities between features."""
    pos_matches, neg_matches = [], []
    features['features'] = features['features'].apply(lambda x: x / np.linalg.norm(x))
    num_frames = len(features['frame'].unique())

    for i in tqdm(range(1, num_frames), desc="Calculating similarity"):
        current_frame_data = features[features['frame'] == i]
        track_ids = current_frame_data['id'].unique()

        for track_id in track_ids:
            feat = current_frame_data[current_frame_data['id'] == track_id].features.values[0].reshape(1, -1)
            feat_pos = features[(features['id'] == track_id) & (features['frame'] > i)].features.values
            feat_neg = current_frame_data[current_frame_data['id'] != track_id].features.values

            if feat_pos.size > 0:
                pos_sim = cosine_similarity(feat, np.vstack([f.reshape(1, -1) for f in feat_pos]))
                pos_matches.extend(pos_sim)

            if feat_neg.size > 0:
                neg_sim = cosine_similarity(feat, np.vstack([f.reshape(1, -1) for f in feat_neg]))
                neg_matches.extend(neg_sim)

    pos_matches = np.concatenate(pos_matches)
    neg_matches = np.concatenate(neg_matches)

    pos_matches = np.random.choice(pos_matches, len(neg_matches), replace=False)
    return pos_matches, neg_matches


def plot_roc_curve(tracker_name, pos_matches, neg_matches, output_path, save):
    """Plots ROC curve for the tracker based on positive and negative matches."""
    y_true = np.concatenate([np.ones(len(neg_matches)), np.zeros(len(neg_matches))])
    y_scores = np.concatenate([pos_matches[:len(neg_matches)], neg_matches])
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    auc_score = roc_auc_score(y_true, y_scores)

    plt.figure(figsize=(10, 9))
    plt.plot(fpr, tpr, color='orange', lw=5, label=f'ROC curve (AUC = {auc_score:.2f})')
    plt.xlabel('False Positive Rate', fontsize=25, weight='bold')
    plt.ylabel('True Positive Rate', fontsize=25, weight='bold')
    plt.title(f'ROC Curve for {tracker_name}', fontsize=25)
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    plt.legend(loc="lower right", fontsize=18)

    if save:
        # make plots for output path
        os.makedirs(os.path.join(output_path, 'plots'), exist_ok=True)
        plt.savefig(os.path.join(output_path, f'plots/{tracker_name}_roc_curve.png'))
        print(f"ROC curve saved to {output_path}/plots/")
    # plt.show()


def plot_reid_distribution(tracker_name, pos_matches, neg_matches, output_path, save):
    """Plots the ReID distribution histogram for the tracker."""
    plt.figure(figsize=(10, 9))
    plt.title(tracker_name, fontsize=25)
    sns.histplot(pos_matches, color='red', label='Positive Matches', alpha=0.6)
    sns.histplot(neg_matches, color='blue', label='Negative Matches', alpha=0.6)
    plt.xlabel('Similarity Score', fontsize=25, weight='bold')
    plt.ylabel('Density', fontsize=25, weight='bold')
    plt.xticks(fontsize=25)
    plt.yticks([])
    plt.legend(fontsize=18)

    if save:
        # open plots folder and save the plot joining with the output path
        plt.savefig(os.path.join(output_path, f'plots/{tracker_name}_reid.png'))
        print(f"ReID distribution saved to {output_path}/plots/")
    # plt.show()



def run_tracker_analysis(tracker, dataset, seq_name, split, output_path, save):
    features = predict_features(tracker, dataset, seq_name, split, use_cache=True, output_path=output_path)
    pos_matches, neg_matches = calculate_similarity(features)
    plot_roc_curve(tracker, pos_matches, neg_matches, output_path, save)
    plot_reid_distribution(tracker, pos_matches, neg_matches, output_path, save)

if __name__ == '__main__':
    args = parse_args()
    dataset = args.dataset
    seq_name = args.seq_name
    split = args.split

    # Determine the list of trackers to run
    if args.tracker == 'all':
        trackers = ['OSNet', 'LITEDeepSORT', 'StrongSORT', 'DeepSORT']
    else:
        trackers = [args.tracker]

    # Run analysis for each tracker
    for tracker in trackers:
        run_tracker_analysis(tracker, dataset, seq_name, split, args.output_path, args.save)

    sys.exit()

