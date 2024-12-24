import os

import numpy as np
from tqdm import tqdm

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import roc_auc_score, roc_curve

import matplotlib.pyplot as plt
import seaborn as sns


class ReIDEvaluator:
    @staticmethod
    def __call__(features):
        """Calculates positive and negative similarities between features."""
        pos_matches, neg_matches = [], []
        features['features'] = features['features'].apply(lambda x: x / np.linalg.norm(x))
        num_frames = len(features['frame'].unique())

         
        def safe_normalize(x):
            norm = np.linalg.norm(x)
            if norm == 0 or np.isnan(norm):  # Check for zero or invalid norm
                return np.zeros_like(x)  # Replace with zeros
            return x / norm

        features['features'] = features['features'].apply(lambda x: safe_normalize(x))

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
    
class Plotter:
    @staticmethod
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

    @staticmethod
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