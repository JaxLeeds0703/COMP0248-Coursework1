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

d) Quantitative Evaluation

This script loads the .pth weights file and outputs the final Accuracy, MAE, and Dice scores
```text
python src/test_exp.py
```

