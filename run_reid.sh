#!/bin/bash

# Define the parameters
DATASET="Face_Recognition"
SPLIT="train"
OUTPUT_PATH="reid/data"
SAVE="--save"
FEATURE_LAYER="layer14"

# Define the sequence names and trackers
SEQ_NAMES=("videoa1-4")
TRACKERS=("LITE")
#TRACKERS= ['OSNet', 'LITE', 'StrongSORT', 'FaceNet', 'DeepSORT', 'ArcFace']

# Loop through all sequence names and trackers
for SEQ in "${SEQ_NAMES[@]}"; do
    for TRACKER in "${TRACKERS[@]}"; do
        echo "Running reid.py for sequence: $SEQ with tracker: $TRACKER"
        python3 reid.py --dataset "$DATASET" --seq_name "$SEQ" --split "$SPLIT" \
            --tracker "$TRACKER" --output_path "$OUTPUT_PATH" $SAVE \
            --appearance_feature_layer "$FEATURE_LAYER"
    done
done

echo "All tasks completed."
