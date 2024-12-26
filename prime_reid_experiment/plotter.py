import os

import numpy as np

from sklearn.metrics import roc_auc_score, roc_curve

import matplotlib.pyplot as plt
import seaborn as sns


class Plotter:
    @staticmethod
    def plot_roc_curve(tracker_name, pos_matches, neg_matches, output_path, save):
        """Plots ROC curve for the tracker based on positive and negative matches."""
        y_true = np.concatenate(
            [np.ones(len(pos_matches)), np.zeros(len(neg_matches))])
        y_scores = np.concatenate(
            [pos_matches, neg_matches])
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        auc_score = roc_auc_score(y_true, y_scores)

        plt.figure(figsize=(10, 9))
        plt.plot(fpr, tpr, color='orange', lw=5,
                 label=f'ROC curve (AUC = {auc_score:.2f})')
        plt.xlabel('False Positive Rate', fontsize=25, weight='bold')
        plt.ylabel('True Positive Rate', fontsize=25, weight='bold')
        plt.title(f'ROC Curve for {tracker_name}', fontsize=25)
        plt.xticks(fontsize=25)
        plt.yticks(fontsize=25)
        plt.legend(loc="lower right", fontsize=18)

        if save:
            # make plots for output path
            os.makedirs(os.path.join(output_path, 'plots'), exist_ok=True)
            plt.savefig(os.path.join(
                output_path, f'plots/{tracker_name}_roc_curve.png'))
            print(f"ROC curve saved to {output_path}/plots/")
        # plt.show()

    @staticmethod
    def plot_reid_distribution(tracker_name, pos_matches, neg_matches, output_path, save):
        """Plots the ReID distribution histogram for the tracker."""
        plt.figure(figsize=(10, 9))
        plt.title(tracker_name, fontsize=25)

        sns.histplot(pos_matches, color='red',
                     label=f'Positive Matches: {len(pos_matches)}', alpha=0.6)
        sns.histplot(neg_matches, color='blue',
                     label=f'Negative Matches: {len(neg_matches)}', alpha=0.6)

        # add the number

        plt.xlabel('Similarity Score', fontsize=25, weight='bold')
        plt.ylabel('Density', fontsize=25, weight='bold')
        plt.xticks(fontsize=25)
        plt.yticks([])
        plt.legend(fontsize=18)

        if save:
            # open plots folder and save the plot joining with the output path
            plt.savefig(os.path.join(
                output_path, f'plots/{tracker_name}_reid.png'))
            print(f"ReID distribution saved to {output_path}/plots/")
        # plt.show()
