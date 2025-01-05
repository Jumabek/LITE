import argparse
from prime_reid_experiment import Evaluator, AppearanceExtractor, Plotter

import warnings
warnings.filterwarnings("ignore")


def run_reid_evaluator(tracker, dataset, seq_name, split, output_path, save, appearance_feature_layer=None):
    extractor = AppearanceExtractor(tracker, dataset, seq_name, split,
                                    output_path, appearance_feature_layer=appearance_feature_layer)
    evaluator = Evaluator()
    
    features = extractor.extract_features() 
    pos_matches, neg_matches = evaluator(features)

    tracker_name = tracker
    if tracker == 'LITE':
        tracker_name = f'LITE_{appearance_feature_layer}' 

    plotter = Plotter(tracker_name, pos_matches, neg_matches, output_path, save)

    plotter.plot_roc_curve()
    plotter.plot_reid_distribution()

    return plotter.auc_score


def parse_args():
    parser = argparse.ArgumentParser(
        description="Track appearance feature extraction and similarity analysis")
    parser.add_argument("--save", action="store_true", default=True,
                        help="Save output files (ROC curve, ReID distribution).")
    parser.add_argument("--output_path", type=str,
                        default="", help="Path to save outputs.")
    parser.add_argument("--use_cache", action="store_true",
                        default=True, help="Use cached features if available.")
    parser.add_argument("--dataset", type=str,
                        default='MOT20', help="Name of the dataset.")
    parser.add_argument("--seq_name", type=str,
                        default='MOT20-01', help="Name of the MOT sequence.")
    parser.add_argument("--split", type=str, default='train',
                        choices=['train', 'test'], help="Specify the split of the dataset.")
    parser.add_argument("--appearance_feature_layer", type=str, default='layer14',
                        help="Specify the appearance feature layer for LITE.")
    parser.add_argument("--tracker", type=str, default='LITE', choices=[
                        'LITE', 'StrongSORT', 'DeepSORT', 'OSNet', 'GFN', 'all'], help="Specify the tracker model to use.")
    return parser.parse_args()

# running example: python reid.py --dataset MOT20 --seq_name MOT20-01 --split train --tracker LITE --output_path reid/data --save
# --appearance_feature_layer layer14

if __name__ == '__main__':
    args = parse_args()
    dataset, seq_name, split = args.dataset, args.seq_name, args.split

    trackers = ['OSNet', 'LITE', 'StrongSORT', 'DeepSORT'] if args.tracker == 'all' else [args.tracker]

    for tracker in trackers:
        run_reid_evaluator(tracker, dataset, seq_name, split,
        args.output_path, args.save, appearance_feature_layer=args.appearance_feature_layer)