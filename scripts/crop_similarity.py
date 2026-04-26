import argparse
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_class_from_file(module_name, path, class_name):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, class_name)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare crop images with StrongSORT and LITEStrongSORT ReID features."
    )
    parser.add_argument(
        "crops",
        type=Path,
        nargs="+",
        help=(
            "Crop images. For --mode identity4, pass: "
            "id1_frame1 id1_frame10 id2_frame1 id2_frame10."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=["auto", "query", "identity4"],
        default="auto",
        help=(
            "Comparison mode. 'identity4' computes positive and negative ReID matches "
            "from four crops; 'query' compares crop 0 to every later crop."
        ),
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device for feature extraction. Use 'auto' to prefer cuda:0 when available.",
    )
    parser.add_argument(
        "--yolo_model",
        default="yolov8m.pt",
        help="YOLO checkpoint used by LITEStrongSORT. Examples: yolov8m.pt, yolo11m.pt.",
    )
    parser.add_argument(
        "--appearance_feature_layer",
        default="layer14",
        help="YOLO feature layer used by LITEStrongSORT.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=1280,
        help="LITEStrongSORT YOLO inference image size.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="LITEStrongSORT YOLO confidence threshold.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON instead of a text table.",
    )
    parser.add_argument(
        "--plot",
        type=Path,
        help="Optional path to save a comparison figure, e.g. outputs/crop_similarity.png.",
    )
    return parser.parse_args()


def read_crop(path):
    import cv2

    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read crop image: {path}")
    return image


def full_image_box(image):
    import numpy as np

    height, width = image.shape[:2]
    return np.array([[0, 0, width, height]], dtype=np.float32)


def extract_one(reid_model, image):
    import numpy as np

    try:
        features = reid_model.extract_appearance_features(image, full_image_box(image))
    except SyntaxError as exc:
        message = str(exc)
        if "appearance_feature_layer" in message or "return_feature_map" in message:
            raise RuntimeError(
                "LITEStrongSORT needs the custom LITE Ultralytics fork. "
                "Install it from https://github.com/Jumabek/ultralytics.git "
                "or make that checkout importable before running this script."
            ) from exc
        raise
    if len(features) == 0:
        raise RuntimeError("Feature extractor returned no features.")
    return np.asarray(features[0], dtype=np.float32).reshape(-1)


def l2_normalize(feature):
    import numpy as np

    norm = np.linalg.norm(feature)
    if norm == 0:
        raise RuntimeError("Cannot compute cosine similarity for a zero vector.")
    return feature / norm


def cosine_similarity(feature_a, feature_b):
    import numpy as np

    if feature_a.shape != feature_b.shape:
        return None
    feature_a = l2_normalize(feature_a)
    feature_b = l2_normalize(feature_b)
    return float(np.dot(feature_a, feature_b))


def score_row(model, match_type, pair_label, path_a, path_b, feature_a, feature_b):
    score = cosine_similarity(feature_a, feature_b)
    return {
        "model": model,
        "match_type": match_type,
        "pair": pair_label,
        "crop_a": str(path_a),
        "crop_b": str(path_b),
        "similarity": score,
        "feature_a_dim": int(feature_a.shape[0]),
        "feature_b_dim": int(feature_b.shape[0]),
    }


def print_text(rows):
    import statistics

    print("model             type       pair                         similarity    dims")
    print("-" * 82)
    for row in rows:
        score = f"{row['similarity']:.6f}"
        dims = f"{row['feature_a_dim']} x {row['feature_b_dim']}"
        print(
            f"{row['model']:<17} {row['match_type']:<10} "
            f"{row['pair']:<28} {score:>10}    {dims}"
        )

    print()
    print("model             positive_mean    negative_mean    margin")
    print("-" * 62)
    models = []
    for row in rows:
        if row["model"] not in models:
            models.append(row["model"])
    for model in models:
        model_rows = [row for row in rows if row["model"] == model]
        positives = [row["similarity"] for row in model_rows if row["match_type"] == "positive"]
        negatives = [row["similarity"] for row in model_rows if row["match_type"] == "negative"]
        if positives and negatives:
            pos_mean = statistics.mean(positives)
            neg_mean = statistics.mean(negatives)
            margin = pos_mean - neg_mean
            print(f"{model:<17} {pos_mean:>13.6f} {neg_mean:>16.6f} {margin:>9.6f}")


def get_query_specs(crop_paths):
    return [
        ("candidate", f"{Path(crop_paths[0]).name} vs {Path(crop_paths[index]).name}", 0, index)
        for index in range(1, len(crop_paths))
    ]


def get_identity4_specs():
    return [
        ("positive", "id1 frame1 vs frame10", 0, 1),
        ("positive", "id2 frame1 vs frame10", 2, 3),
        ("negative", "id1 f1 vs id2 f1", 0, 2),
        ("negative", "id1 f1 vs id2 f10", 0, 3),
        ("negative", "id1 f10 vs id2 f1", 1, 2),
        ("negative", "id1 f10 vs id2 f10", 1, 3),
    ]


def build_rows_by_model(features, crop_paths, pair_specs):
    rows_by_model = {}
    for model, model_features in features.items():
        rows_by_model[model] = [
            score_row(
                model,
                match_type,
                pair_label,
                crop_paths[index_a],
                crop_paths[index_b],
                model_features[index_a],
                model_features[index_b],
            )
            for match_type, pair_label, index_a, index_b in pair_specs
        ]
    return rows_by_model


def save_plot(rows_by_model, crop_paths, output_path):
    import cv2
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    rgb_images = []
    for path in crop_paths:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Could not read crop image: {path}")
        rgb_images.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    models = list(rows_by_model.keys())
    columns = len(crop_paths)
    fig, axes = plt.subplots(
        len(models),
        columns,
        figsize=(max(8, columns * 2.2), max(4, len(models) * 3.2)),
        squeeze=False,
    )

    for row_idx, model in enumerate(models):
        model_rows = rows_by_model[model]
        scores_by_candidate = {
            Path(row["crop_b"]): row["similarity"]
            for row in model_rows
        }
        for col_idx, (path, image) in enumerate(zip(crop_paths, rgb_images)):
            ax = axes[row_idx][col_idx]
            ax.imshow(image)
            ax.set_xticks([])
            ax.set_yticks([])
            if col_idx == 0:
                ax.set_title(f"{model}\nquery", fontsize=12, fontweight="bold")
                for spine in ax.spines.values():
                    spine.set_visible(True)
                    spine.set_linewidth(4)
                    spine.set_edgecolor("#bf5af2")
            else:
                score = scores_by_candidate[path]
                ax.set_title(f"{Path(path).name}\nscore={score:.4f}", fontsize=10)
                color = "#6bd34a" if score >= 0.5 else "#ff3b30"
                for spine in ax.spines.values():
                    spine.set_visible(True)
                    spine.set_linewidth(4)
                    spine.set_edgecolor(color)

        axes[row_idx][0].set_ylabel(model, fontsize=13, fontweight="bold", rotation=90)

    fig.suptitle("Crop Similarity: Query vs Candidate Crops", fontsize=16, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_identity_plot(rows_by_model, crop_paths, pair_specs, output_path):
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    rgb_images = []
    for path in crop_paths:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Could not read crop image: {path}")
        rgb_images.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    def fit_image(image, height=170, width=95):
        src_h, src_w = image.shape[:2]
        scale = min(width / src_w, height / src_h)
        resized = cv2.resize(image, (max(1, int(src_w * scale)), max(1, int(src_h * scale))))
        canvas = np.full((height, width, 3), 255, dtype=np.uint8)
        y = (height - resized.shape[0]) // 2
        x = (width - resized.shape[1]) // 2
        canvas[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
        return canvas

    def pair_card(image_a, image_b):
        left = fit_image(image_a)
        right = fit_image(image_b)
        gap = np.full((170, 38, 3), 255, dtype=np.uint8)
        cv2.putText(gap, "vs", (4, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        return np.concatenate([left, gap, right], axis=1)

    models = list(rows_by_model.keys())
    columns = len(pair_specs)
    fig, axes = plt.subplots(
        len(models),
        columns,
        figsize=(max(16, columns * 2.8), max(6, len(models) * 3.4)),
        squeeze=False,
    )

    for row_idx, model in enumerate(models):
        for col_idx, ((match_type, pair_label, index_a, index_b), row) in enumerate(
            zip(pair_specs, rows_by_model[model])
        ):
            ax = axes[row_idx][col_idx]
            ax.imshow(pair_card(rgb_images[index_a], rgb_images[index_b]))
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"{pair_label}\nscore={row['similarity']:.4f}", fontsize=10)

            color = "#6bd34a" if match_type == "positive" else "#ff3b30"
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(4)
                spine.set_edgecolor(color)

            if row_idx == 0:
                ax.text(
                    0.5,
                    1.24,
                    match_type.title(),
                    transform=ax.transAxes,
                    ha="center",
                    va="bottom",
                    fontsize=13,
                    fontweight="bold",
                    color=color,
                )
            if col_idx == 0:
                ax.set_ylabel(model, fontsize=14, fontweight="bold")

    fig.suptitle(
        "Positive vs Negative ReID Similarities by Tracker",
        fontsize=18,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main():
    args = parse_args()

    import torch
    from ultralytics import YOLO

    StrongSORT = load_class_from_file(
        "crop_similarity_strongsort",
        ROOT / "reid_modules" / "strongsort.py",
        "StrongSORT",
    )
    LITE = load_class_from_file(
        "crop_similarity_lite",
        ROOT / "reid_modules" / "lite.py",
        "LITE",
    )

    device = args.device
    if device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    mode = args.mode
    if mode == "auto":
        mode = "identity4" if len(args.crops) == 4 else "query"

    if mode == "identity4" and len(args.crops) != 4:
        raise ValueError(
            "--mode identity4 needs exactly four crop paths: "
            "id1_frame1 id1_frame10 id2_frame1 id2_frame10."
        )
    if mode == "query" and len(args.crops) < 2:
        raise ValueError("Provide at least two crop paths: query crop plus one candidate crop.")

    crop_images = [read_crop(path) for path in args.crops]

    strongsort = StrongSORT(device=device)
    yolo = YOLO(args.yolo_model)
    yolo.to(device)
    lite_strongsort = LITE(
        model=yolo,
        appearance_feature_layer=args.appearance_feature_layer,
        imgsz=args.imgsz,
        conf=args.conf,
        device=device,
    )

    features = {
        "StrongSORT": [extract_one(strongsort, image) for image in crop_images],
        "LITEStrongSORT": [extract_one(lite_strongsort, image) for image in crop_images],
    }

    pair_specs = get_identity4_specs() if mode == "identity4" else get_query_specs(args.crops)
    rows_by_model = build_rows_by_model(features, args.crops, pair_specs)

    rows = [row for model_rows in rows_by_model.values() for row in model_rows]

    if args.plot:
        if mode == "identity4":
            save_identity_plot(rows_by_model, args.crops, pair_specs, args.plot)
        else:
            save_plot(rows_by_model, args.crops, args.plot)

    if args.json:
        print(json.dumps(rows, indent=2))
    else:
        print_text(rows)
        if args.plot:
            print()
            print(f"Saved plot: {args.plot}")


if __name__ == "__main__":
    main()
