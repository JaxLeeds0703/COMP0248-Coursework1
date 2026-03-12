##Project Structure

```text
├── dataset/                        # Dataset directory (Ignored in Git, requires local download)
│   └── RGB_depth_annotations       # Core Multi-Task Network architecture
│   └── test                        # Core Multi-Task Network architecture
├── weights/                        # Trained model weights (.pth files, Ignored in Git)
├── src/                            # Core modules (Model, Dataloader, Train and Test)
│   └── model_exp.py                # Core Multi-Task Network architecture
├── scripts/                        # Executable scripts(contains dataset split and visualisation)
│   └── visualise.py                # visualization script
├── requirements.txt                # Python dependencies
└── README.md                       # Project Code User Guide
