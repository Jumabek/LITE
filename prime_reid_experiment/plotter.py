import os
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns

class Plotter:
    def __init__(self, tracker, pos_matches, neg_matches, output_path, save=True):
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
        plt.title(f'ROC Curve for {self.tracker}', fontsize=25)
        plt.xticks(fontsize=25)
        plt.yticks(fontsize=25)
        plt.legend(loc="lower right", fontsize=18)

        if self.save:
            save_dir = os.path.join(self.output_path, 'prime_reid_experiment/plots')
            os.makedirs(save_dir, exist_ok=True)
            filename = f'{self.tracker}_roc_curve.png'

            plt.savefig(os.path.join(save_dir, filename))
            print(f"ROC curve saved to {save_dir}/{filename}")

    def plot_reid_distribution(self):
        """Plots the ReID distribution histogram for the tracker."""
        plt.figure(figsize=(10, 9))
        plt.title(f"{self.tracker} AUC Score: $\\mathbf{{{self.auc_score:.3f}}}$", fontsize=25)
        sns.histplot(self.pos_matches, color='red', label=f'Positive Matches', alpha=0.6)
        sns.histplot(self.neg_matches, color='blue', label=f'Negative Matches', alpha=0.6)
        # plt.text(0.7, 0.7, f'AUC: {self.auc_score:.2f}', fontsize=25, weight='bold', transform=plt.gca().transAxes)
        plt.xlabel('Similarity Score', fontsize=25, weight='bold')
        plt.ylabel('Density', fontsize=25, weight='bold')
        plt.xticks(fontsize=25)
        plt.yticks([])
        plt.legend(fontsize=18)

        if self.save:
            save_dir = os.path.join(self.output_path, 'prime_reid_experiment/plots')
            os.makedirs(save_dir, exist_ok=True)
            filename = f'{self.tracker}_reid.png'
            plt.savefig(os.path.join(save_dir, filename))
            print(f"ReID distribution saved to {save_dir}/{filename}")
