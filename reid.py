import sys
import argparse
from utils import Plotter, ReIDEvaluator, AppearanceExtractor

import warnings
warnings.filterwarnings("ignore")

def run_reid_evaluator(tracker, dataset, seq_name, split, output_path, save, appearance_feature_layer=None):
    extractor = AppearanceExtractor(tracker, dataset, seq_name, split,
                                    output_path, appearance_feature_layer)
    evaluator = ReIDEvaluator()

    features = extractor.extract_features()
    pos_matches, neg_matches = evaluator(features)

    Plotter.plot_roc_curve(tracker, pos_matches, neg_matches, output_path, save)
    Plotter.plot_reid_distribution(tracker, pos_matches, neg_matches, output_path, save)

def parse_args():
    parser = argparse.ArgumentParser(description="Track appearance feature extraction and similarity analysis")
    parser.add_argument("--save", action="store_true", default=True, help="Save output files (ROC curve, ReID distribution).")
    parser.add_argument("--output_path", type=str, default="reid/data", help="Path to save outputs.")
    parser.add_argument("--use_cache", action="store_true", default=True, help="Use cached features if available.")
    parser.add_argument("--dataset", type=str, default='MOT20', help="Name of the dataset.")
    parser.add_argument("--seq_name", type=str, default='MOT20-01', help="Name of the MOT sequence.")
    parser.add_argument("--split", type=str, default='train', choices=['train', 'test'], help="Specify the split of the dataset.")
    parser.add_argument("--appearance_feature_layer", type=str, help="Specify the appearance feature layer for LITE.")
    parser.add_argument("--tracker", type=str, default='LITE', choices=['LITE', 'StrongSORT', 'DeepSORT', 'OSNet', 'all'], help="Specify the tracker model to use.")
    return parser.parse_args()

# running example: python reid.py --dataset MOT20 --seq_name MOT20-01 --split train --tracker LITE --output_path reid/data --save

if __name__ == '__main__':
    args = parse_args()
    dataset, seq_name, split = args.dataset, args.seq_name, args.split

    trackers = ['OSNet', 'LITE', 'StrongSORT', 'DeepSORT'] if args.tracker == 'all' else [args.tracker]

    for tracker in trackers:
        run_reid_evaluator(tracker, dataset, seq_name, split, args.output_path, args.save, args.appearance_feature_layer)
        
    sys.exit()

