#!/bin/bash

DATASETS_DIR="."

FOLDER_ID="1hlX2n5FVFGXOJrQMVSxnSmSNW7TM_BZ3"

cd "$DATASETS_DIR" || exit
gdown --folder --id "$FOLDER_ID"

cd "datasets" || exit
for FILE in *.zip; do
    if [[ -f "$FILE" ]]; then
     
        DIR_NAME="${FILE%.zip}"
       
        if unzip -q "$FILE" ; then
            rm "$FILE"
        else
            echo "Failed to unzip $FILE"
        fi
    fi
done

cd ..
