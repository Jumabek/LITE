# LITE: A Paradigm Shift in Multi-Object Tracking with Efficient ReID Feature Integration

> [**LITE: A Paradigm Shift in Multi-Object Tracking with Efficient ReID Feature Integration**](http://www.arxiv.org/abs/2409.04187v2)
> 
> Jumabek Alikhanov, Dilshod Obidov, Hakil Kim
> 
> *[arXiv 2409.04187](http://www.arxiv.org/abs/2409.04187v2)*
> 
> *![Published at ICONIP2024](assets/ICONIP2024_Certificate_of_Presentation_Paper_1.pdf)*


## Overview

LITE (Lightweight Integrated Tracking-Feature Extraction) introduces a groundbreaking approach to enhance ReID-based Multi-Object Tracking (MOT) systems. By integrating appearance feature extraction directly into the detection pipeline, LITE significantly improves computational efficiency while maintaining robust performance. Utilizing CNN-based object detectors like YOLOv8 and YOLO11, LITE enables real-time tracking, making it ideal for resource-constrained environments.

---
![Efficient ReID feature extraction via the LITE paradigm](assets/Fig02-6390.png)

## Key Features

- **Efficient Integration**: Combines appearance feature extraction within the detection process.
- **Lightweight Design**: Tailored for real-time applications on resource-limited devices.
- **Performance Optimization**: Demonstrates notable FPS improvements across multiple trackers while retaining competitive accuracy.

---

## Experimental Results

We evaluated LITE using **YOLOv8m** with the following settings:

- **Confidence Threshold**: 0.25
- **Input Resolution**: 1280

| Tracker              | MOT17 HOTA ↑ | MOT17 FPS ↑ | MOT20 HOTA ↑ | MOT20 FPS ↑ |
|----------------------|------------------|----------------|------------------|----------------|
| DeepSORT            | 43.7            | 10.5           | 24.4            | 8.5            |
| StrongSORT          | 44.5            | 4.5            | 26.1            | 2.6            |
| Deep OC-SORT        | 43.7            | 10.3           | 24.9            | 8.9            |
| BoTSORT             | 40.8            | 10.6           | 21.1            | 9.4            |
| **LITE:DeepSORT**   | 43.0            | 26.7           | 25.2            | 15.9           |
| **LITE:Deep OC-SORT** | 43.4            | 34.8           | 25.4            | 19.6           |
| **LITE:BoTSORT**    | 40.8            | 38.2           | 21.1            | 31.8           |
| **LITE:StrongSORT** | 42.4            | 29.7           | 25.2            | 22.9           |

---

## Installation

Follow these steps to set up the LITE tracking system:

### 1. Clone the Repository

```bash
# Clone the LITE Tracker Repository
git clone https://github.com/Jumabek/LITE.git
cd LITE
```

### 2. Set Up Python Environment

```bash
# Create and activate a virtual environment
python3.10 -m venv myenv
source myenv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Clone Additional Dependencies

```bash
# Clone supplementary repositories
git clone https://github.com/Jumabek/ultralytics.git
git clone https://github.com/humblebeeintel/yolo_tracking.git
git clone https://github.com/humblebeeintel/TrackEval
```

### 4. Set Up FastReID

```bash
bash scripts/setup_fastreid.sh
```

---

## Dataset Preparation

Download the prepared datasets from [this link](https://drive.google.com/drive/folders/1hlX2n5FVFGXOJrQMVSxnSmSNW7TM_BZ3) and organize them as follows:

```plaintext
LITE/datasets
   |-- MOT
   |   |-- train
   |   |-- test
   |-- PersonPath22
   |   |-- test
   |-- VIRAT-S
   |   |-- train
   |-- KITTI
       |-- train
       |-- test
```

---

## Checkpoints

Download the required checkpoints from [this link](https://drive.google.com/file/d/1L4gnCbkmvGB6HbPPs1YK8O2fERBS-Xvn) and place them under `LITE/checkpoints`:

```plaintext
checkpoints
└── FastReID
    ├── bagtricks_S50.yml
    ├── Base-bagtricks.yml
    ├── deepsort
    │   ├── ckpt.t7
    │   └── original_ckpt.t7
    └── DukeMTMC_BoT-S50.pth
```

---

## Running Experiments

### 1. Run Tracking and ReID Experiments

```bash
bash scripts/run_experiment.sh -d <DATASET> -s <SPLIT> -t <TRACKER> -m <YOLO_MODEL>
# TRACKER options: "SORT", "LITEDeepSORT", "DeepSORT", "StrongSORT", "LITEStrongSORT", "OCSORT", "Bytetrack", "DeepOCSORT", "LITEDeepOCSORT", "BoTSORT", "LITEBoTSORT"

# YOLO_MODEL options: all models of YOLO from yolov8 to yolo11
```

### 2. Running the ReID Evaluator
```bash
python reid.py --dataset <DATASET> --seq_name <SEQ_NAME>  --split <SPLIT>  --tracker <ReID_MODEL> --save
```

### 2. Run FPS Experiments

```bash
bash scripts/run_fps_experiment.sh -d <DATASET> -s <SPLIT> -q <SEQUENCE>
```

---

## Demo

### Download demo videos

```
bash demo/download_solutions_demo_videos.sh
```

### Basic Tracking Demo

```bash
python demo.py --source demo/VIRAT_S_010204_07_000942_000989.mp4
```

### Object Counter & Heatmap

```bash
python solutions.py \
--source videos/shortened_enterance.mp4 \
--solution object_counter heatmap
```

### Parking Management

```bash
python solutions.py \
--source videos/parking.mp4 \
--solution parking_management
```

---

## Citation

If you use LITE in your research, please cite our work:

```bibtex
@misc{alikhanov2024liteparadigmshiftmultiobject,
      title={LITE: A Paradigm Shift in Multi-Object Tracking with Efficient ReID Feature Integration}, 
      author={Jumabek Alikhanov and Dilshod Obidov and Hakil Kim},
      year={2024},
      eprint={2409.04187},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2409.04187}, 
}
```
