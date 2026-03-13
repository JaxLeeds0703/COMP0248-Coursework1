1. Project Structure

```text
├── dataset/                        # Dataset directory (Ignored in Git, requires local download)
│   └── RGB_depth_annotations       # Contains Training and Validation Dataset
│   └── test                        # Contains Test Dataset
├── weights/                        # Trained model weights (.pth files, Ignored in Git)
├── src/                            # Core modules (Model, Dataloader, Train and Test)
│   ├── dataloader_exp.py           # Dataset & DataLoader with colour & spatial augmentations
│   ├── model_exp.py                # Multi-Task Network architecture (ResNet + U-Net)
│   ├── train_exp.py                # Training engine with static loss weighting & Cosine Annealing learning-rate schedule
│   └── test_exp.py                 # Evaluate quantitative metrics (Accuracy, IoU, Dice)
├── scripts/                        # Executable scripts(contains dataset split and visualisation)
│   ├── build_dataset_split_index.py # Splits trian/val dataset and generates .CSV indexes
│   ├── build_test_index.py          # Generates isolated test set .CSV indexes
│   └── visualise.py                # visualization script
├── requirements.txt                # Python dependencies
└── README.md                       # Project Code User Guide
```

2.Step-by-Step Usage Guide

a) Environment Setup

The python version is 3.10 (3.8+). Install the required dependencies:
```text
pip install -r requirements.txt
```

b) Data Preparation

Pleace place your raw dataset into the dataset/ directory: i) RGB_depth_annotations(Contains train and val Dataset)
ii) test(Contains Test Dataset).Then, use build_dataset_split_index.py and build_test_index.py to generate the structured
CSV index files required by the DataLoader:
```text
# 1. Generate train/validation split indexes
python scripts/build_dataset_split_index.py
# 2. Generate test set indexes
python scripts/build_test_index.py
```

c) Model Training

To training the multi-task network, please run the training script. This script will load the CSV index file, initialise the Adam optimiser, and save the optimal weights (based on validation set loss) to the weights/ directory.Hyperparameters such as batch size, learning rate, and epochs can be modified inside the train_exp.py.
```text
python src/train_exp.py
```
Ablation Study Configurations (Depth & Augmentation)
To rigorously validate the contribution of multi-modal inputs(Depth Map) and spatial augmentations, the training and evaluation scripts support toggling specific features. This allows users to reproduce our 4 distinct ablation experiments (A, B, C, and D).

You can control these via command-line arguments (or configuration variables) when running `train_exp.py` and `test_exp.py`:

- `--depth True/False`: Toggles the 4th input channel. If `True`, the model accepts RGB-D data. If `False`, it accepts 3-channel RGB only.
- `--aug True/False`: Toggles synchronous spatial augmentations (random rotations, translations) during the training phase.

Example Configurations:
```bash
# Experiment A (Baseline): RGB Only, No Augmentation
python src/train_exp.py --depth False --aug False
# Experiment B: RGB-D, No Augmentation
python src/train_exp.py --depth True --aug False
# Experiment C: RGB Only, With Spatial Augmentation
python src/train_exp.py --depth False --aug True
# Experiment D (Proposed): RGB-D + Spatial Augmentation
python src/train_exp.py --depth True --aug True
```

d) Quantitative Evaluation

This script loads the .pth weights file and outputs the Classification, Detection and Boundingboc metrics.
```text
python src/test_exp.py
```
e) Qualitative Visualization

Visualize predictions on the Test Set (5 random samples):
```text
python scripts/visualise.py --split test --n 5
```
Visualize predictions on the Validation Set:
```text
python scripts/visualise.py --split val --n 5
```
