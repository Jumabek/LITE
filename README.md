
# LITE

> [**LITE: A Paradigm Shift in Multi-Object Tracking with Efficient ReID Feature Integration**](http://www.arxiv.org/abs/2409.04187v2)
> 
> Jumabek Alikhanov, Dilshod Obidov, Hakil Kim
> 
> *[arXiv 2409.04187](http://www.arxiv.org/abs/2409.04187v2)*

## Abstract
The Lightweight Integrated Tracking-Feature Extraction (LITE) paradigm is introduced as a novel multi-object tracking (MOT) approach. It enhances ReID-based trackers by eliminating inference, pre-processing, post-processing, and ReID model training costs. LITE uses real-time appearance features without compromising speed. By integrating appearance feature extraction directly into the tracking pipeline using standard CNN-based detectors such as YOLOv8m, LITE demonstrates significant performance improvements. The simplest implementation of LITE on top of classic DeepSORT achieves a HOTA score of 43.03% at 28.3 FPS on the MOT17 benchmark, making it twice as fast as DeepSORT on MOT17 and four times faster on the more crowded MOT20 dataset, while maintaining similar accuracy. Additionally, a new evaluation framework for tracking-by-detection approaches reveals that conventional trackers like DeepSORT remain competitive with modern state-of-the-art trackers when evaluated under fair conditions.

![Efficient ReID feature extraction via the LITE paradigm](assets/Fig02-6390.png)

## Environment

We use `Python 3.10.12` 

```bash
python3.10 -m venv myenv
source myenv/bin/activate
pip install -r requirements.txt
```

## ultralytics
```bash
git clone https://github.com/humblebeeintel/ultralytics.git
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

Download [checkpoints](https://drive.google.com/file/d/1L4gnCbkmvGB6HbPPs1YK8O2fERBS-Xvn) and put them under LITE/checkpoints:
```
checkpoints
└── FastReID
    ├── bagtricks_S50.yml
    ├── Base-bagtricks.yml
    ├── deepsort
    │   ├── ckpt.t7
    │   └── original_ckpt.t7
    └── DukeMTMC_BoT-S50.pth
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
### Object Counter & Heatmap

```bash
python lite_deepsort_solutions_demo.py \
--source videos/shortened_enterance.mp4 \
--solution object_counter
           heatmap
```

### Parking Management

```bash
python lite_deepsort_solutions_demo.py \
--source videos/parking.mp4 \
--solution parking_management
```

# Multi Object Tracking made easy and accessible

Code is coming soon


