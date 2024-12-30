#!/bin/bash

# Define the dataset, sequence name, split, and output path
DATASET="MOT20"
SEQ_NAME="MOT20-01"
SPLIT="train"
OUTPUT_PATH=""

# Loop through all 22 layers
for LAYER in {0..22}
do
    echo "Running reid.py for layer $LAYER"
    python3 reid.py --dataset $DATASET --seq_name $SEQ_NAME --split $SPLIT --tracker LITE --output_path $OUTPUT_PATH --save --appearance_feature_layer layer$LAYER
    echo "Done running reid.py for layer $LAYER"
done
