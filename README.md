
# LITE: A Paradigm Shift in Multi-Object Tracking with Efficient ReID Feature Integration

## Environment

We use `Python 3.10.12` 

```bash
python -m venv myenv
source myenv/bin/activate
pip install -r requirements.txt
```

## Demo

```bash
python lite_deepsort_demo.py --source demo/VIRAT_S_010204_07_000942_000989.mp4
```

## Experiments Settings

### Datasets

Download our prepared [datasets](https://drive.google.com/drive/folders/1hlX2n5FVFGXOJrQMVSxnSmSNW7TM_BZ3) and put them under LITE/datasets in the following structure:

```
datasets
   |———MOT
   |     └———train
   |     └———test
   └———PersonPath22
   |     └———test
   └———VIRAT-S
   |     └———train
   └———KITTI
         └———train
         └———test

 
```

### Checkpoints

<!-- Download [checkpoints]() -->
```bash
bash scripts/prepare_checkpoints.sh
```

### FastReID

```bash
bash scripts/setup_fastreid.sh
```

### yolo_tracking

```bash
git clone https://github.com/humblebeeintel/yolo_tracking.git
```

## Running Experiments

Use the following command to run experiments with different datasets and splits:

```bash
bash scripts/run_experiment.sh -d <DATASET> -s <SPLIT>
```

## Running FPS Experiments

Use the following command to run fps experiment with specific sequence from datasets:

```bash
bash scripts/run_fps_experiment.sh -d <DATASET> -s <SPLIT> -q <SEQUENCE>
```

# Solutions demo with LITEDeepSORT

```bash
bash demo/download_solutions_demo_videos.sh
```

```bash
python lite_deepsort_solutions_demo.py \
--source videos/shortened_enterance.mp4 \
--solution object_counter
           heatmap
```

# Multi Object Tracking made easy and accessible

Code is coming soon


