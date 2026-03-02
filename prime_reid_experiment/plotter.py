import os
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns

class Plotter:
    def __init__(self, tracker, sequence, pos_matches, neg_matches, output_path, save=True):
        self.sequence = sequence
        self.tracker = tracker
        self.pos_matches = pos_matches
        self.neg_matches = neg_matches
        self.output_path = output_path
        self.save = save
        self.fpr, self.tpr, self.auc_score = self.calculate_roc_curve(pos_matches, neg_matches)

    def calculate_roc_curve(self, pos_matches, neg_matches):
        y_true = np.concatenate([np.ones(len(pos_matches)), np.zeros(len(neg_matches))])
        y_scores = np.concatenate([pos_matches, neg_matches])
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        auc_score = roc_auc_score(y_true, y_scores)

        return fpr, tpr, auc_score

    def plot_roc_curve(self):
        """Plots ROC curve for the tracker based on positive and negative matches."""
        plt.figure(figsize=(10, 9))
        plt.plot(self.fpr, self.tpr, color='orange', lw=5, label=f'ROC curve (AUC = {self.auc_score:.2f})')
        plt.xlabel('False Positive Rate', fontsize=25, weight='bold')
        plt.ylabel('True Positive Rate', fontsize=25, weight='bold')
        # plt.title(f'ROC Curve for {self.tracker}', fontsize=25)
        plt.xticks(fontsize=25)
        plt.yticks(fontsize=25)
        plt.legend(loc="lower right", fontsize=18)

        if self.save:
            save_dir = os.path.join(self.output_path, 'prime_reid_experiment/plots')
            os.makedirs(save_dir, exist_ok=True)
            filename = f'{self.tracker}_yolov8m-face_roc_curve_{self.sequence}.png'

            plt.savefig(os.path.join(save_dir, filename))
            print(f"ROC curve saved to {save_dir}/{filename}")

    def plot_reid_distribution(self):
        """Plots the ReID distribution histogram for the tracker."""
        plt.figure(figsize=(10, 9))
        plt.title(f"AUC Score: $\\mathbf{{{self.auc_score:.3f}}}$", fontsize=25)

        # Combine data to ensure consistent binning
        all_scores = np.concatenate([self.pos_matches, self.neg_matches])
        bins = np.histogram_bin_edges(all_scores, bins='auto')  # or set bins=50

        # Plot both histograms with the same binning and density normalization
        sns.histplot(self.pos_matches, bins=bins, color='red', label='Positive Matches',
                    alpha=0.6, kde=True, stat='density')
        sns.histplot(self.neg_matches, bins=bins, color='blue', label='Negative Matches',
                    alpha=0.6, kde=True, stat='density')

        plt.xlabel('Similarity Score', fontsize=25, weight='bold')
        plt.ylabel('Density', fontsize=25, weight='bold')
        plt.xticks(fontsize=25)
        plt.yticks([])
        plt.legend(fontsize=18)

        if self.save:
            save_dir = os.path.join(self.output_path, 'prime_reid_experiment/plots')
            os.makedirs(save_dir, exist_ok=True)
            filename = f'{self.tracker}_reid_{self.sequence}.png'
            plt.savefig(os.path.join(save_dir, filename))
            print(f"ReID distribution saved to {save_dir}/{filename}")
