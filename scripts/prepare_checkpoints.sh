#!/bin/bash

FILE_ID="1L4gnCbkmvGB6HbPPs1YK8O2fERBS-Xvn"
OUTPUT_FILE="checkpoints.zip"

gdown --id "$FILE_ID" -O "$OUTPUT_FILE"


if [[ -f "$OUTPUT_FILE" ]]; then
    
    DIR_NAME="${OUTPUT_FILE%.zip}"
    mkdir -p "$DIR_NAME"
    
    unzip -q "$OUTPUT_FILE" #-d "$DIR_NAME"

    rm "$OUTPUT_FILE"

    echo "Download and extraction complete."
else
    echo "Failed to download $OUTPUT_FILE"
fi
