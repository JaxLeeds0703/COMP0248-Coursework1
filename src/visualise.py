"""
This code is used to visualize the model predictions on the validation and test sets.

For each visualized sample, the left column shows the ground-truth annotations, including:
- a green dashed bounding box,
- a semi-transparent green manually annotated mask,
- and the true gesture class.

The right column shows the model prediction on the validation/test sample, including:
- a red predicted bounding box,
- and a predicted segmentation mask displayed in different colors
  (cyan for the validation set and yellow for the test set).

In addition, the background color of the predicted class label is used to indicate whether the predicted gesture matches the ground truth:
- green means the prediction is correct,
- red means the prediction is incorrect.

This visualization is intended to provide an intuitive comparison between the ground truth and the model output for qualitative evaluation.

To use this code:
Visualise validation set(only):    python visualise.py --split val
Visualise testset(only):           python visualise.py --split test
"""

import os
import sys
import random
import argparse
import torch
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import torchvision.transforms.functional as TF

# Resolve the project root directory automatically based on the current working directory. 

current_dir = Path.cwd()
if (current_dir / "dataset").exists():
    BASE_DIR = current_dir
elif (current_dir.parent / "dataset").exists():
    BASE_DIR = current_dir.parent
else:
    raise RuntimeError("pls make sure to run code from the cw1 or cw1/src directory.")

sys.path.append(str(BASE_DIR / "src"))
sys.path.append(str(BASE_DIR / "scripts"))

# Import the model architecture used for multi-task gesture recognition.
try:
    from model_exp import MultiTaskGestureNet
except ImportError:
    print("model_exp.py not found. Pls make sure it's located in the /src or /scripts directory.")
    sys.exit(1)

# File paths used by this visualisation script.
WEIGHTS_PATH = BASE_DIR / "weights" / "best_model_exp_depthTrue_augTrue.pth"
TEST_CSV = BASE_DIR / "dataset" / "test_index.csv"
VAL_CSV = BASE_DIR / "dataset" / "dataset_index_split.csv"

# Mapping from numeric class indices to human-readable gesture names.
GESTURE_CLASSES_INV = {
    0: "G01_call", 1: "G02_dislike", 2: "G03_like", 3: "G04_ok",
    4: "G05_one", 5: "G06_palm", 6: "G07_peace", 7: "G08_rock",
    8: "G09_stop", 9: "G10_three"
}

def load_random_samples(csv_path, split_type=None, num_samples=5, img_size=(224, 224)):
    """
    Generic dataset loading function
    
    Parameters:
    1. csv_path : Path to the dataset index .csv file.
    2. split_type : Dataset split to load ('val' or 'test')
    """
    # Keep only samples with valid mask annotations, since mask-based visualisation and bounding-box extraction both depend on them.
    df = pd.read_csv(csv_path)
    
    mask_condition = df['has_mask'] == True
    if split_type == 'val':
        mask_condition = mask_condition & (df['split'] == 'val')
    
    # Randomly select a subset of valid samples for qualitative inspection.
    df_valid = df[mask_condition].reset_index(drop=True)
    
    if len(df_valid) < num_samples:
        raise ValueError(f"Not enough matching samples available: fewer than {num_samples} images found.")
        
    sample_indices = random.sample(range(len(df_valid)), num_samples)
    sampled_df = df_valid.iloc[sample_indices]
    
    images_tensor = []
    ground_truths = []
    
    # ImageNet-style normalisation statistics for RGB input preprocessing.
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    for _, row in sampled_df.iterrows():
        # Load the RGB image, resize it for visualisation and model input, and apply standard normalisation for the RGB channels.
        img_path = BASE_DIR / row['rgb_path'] if not Path(row['rgb_path']).is_absolute() else row['rgb_path']
        image_rgb = Image.open(img_path).convert("RGB")
        orig_img_np = np.array(image_rgb.resize(img_size, Image.BILINEAR)) 
        image_tensor_rgb = TF.normalize(TF.to_tensor(image_rgb.resize(img_size, Image.BILINEAR)), mean, std)
        
        #Load the depth map, resize it to the target resolution, and normalise valid depth values using robust percentile clipping.
        depth_path = BASE_DIR / row['depth_path'] if not Path(row['depth_path']).is_absolute() else row['depth_path']
        depth_np = np.array(Image.open(depth_path))
        if depth_np.ndim == 3: depth_np = depth_np[..., 0]
        
        depth_tensor = torch.from_numpy(depth_np.astype(np.float32)).unsqueeze(0).unsqueeze(0)
        depth_tensor = torch.nn.functional.interpolate(depth_tensor, size=img_size, mode='bilinear', align_corners=False).squeeze(0)
        
        valid_pixels = depth_tensor[depth_tensor > 0]
        if valid_pixels.numel() > 0:
            d_min, d_max = torch.quantile(valid_pixels, 0.01), torch.quantile(valid_pixels, 0.99)
            if (d_max - d_min) > 1e-6:
                depth_tensor = (depth_tensor - d_min) / (d_max - d_min)
                depth_tensor = depth_tensor.clamp(0.0, 1.0)
        image_tensor_depth = (depth_tensor - 0.5) / 0.5
        
        # Load the binary segmentation mask and derive a normalised bounding box directly from the foreground mask pixels.
        mask_path = BASE_DIR / row['mask_path'] if not Path(row['mask_path']).is_absolute() else row['mask_path']
        mask_img = Image.open(mask_path).convert("L").resize(img_size, Image.NEAREST)
        mask_np = (np.array(mask_img) > 127).astype(np.float32)
        
        y_idx, x_idx = np.where(mask_np > 0)
        if len(y_idx) > 0 and len(x_idx) > 0:
            bbox = [np.min(x_idx)/img_size[0], np.min(y_idx)/img_size[1], 
                    np.max(x_idx)/img_size[0], np.max(y_idx)/img_size[1]]
        else:
            bbox = [0, 0, 0, 0]
            
        gt_label = GESTURE_CLASSES_INV[int(row['class_label'])]
        
        # Concatenate RGB and depth channels to form a 4-channel RGB-D input.
        img_tensor = torch.cat([image_tensor_rgb, image_tensor_depth], dim=0)
        images_tensor.append(img_tensor)
        
        ground_truths.append({
            'orig_img': orig_img_np,
            'mask': mask_np,
            'bbox': bbox,
            'label': gt_label
        })
        
    return torch.stack(images_tensor), ground_truths


def run_visualization(split='test', num_samples=5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Weight: {WEIGHTS_PATH.name}")
    
    # Choose the dataset index file and figure title prefix based on the requested dataset split.
    if split == 'val':
        csv_path = VAL_CSV
        title_prefix = "Validation Set"
    else:
        csv_path = TEST_CSV
        title_prefix = "Test Set"
        
    # Build the model architecture and load the trained weights.
    model = MultiTaskGestureNet(in_channels=4, num_classes=10).to(device)
    try:
        model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
    except FileNotFoundError:
        print(f"Cannot find the weight file. Please check the path: {WEIGHTS_PATH}")
        sys.exit(1)
        
    model.eval()
    
    # Load random RGB-D samples together with their ground-truth annotations.
    input_tensors, ground_truths = load_random_samples(csv_path, split_type=split, num_samples=num_samples)
    input_tensors = input_tensors.to(device)
    
    # Run inference without gradient tracking and convert the raw model outputs into class labels, bounding boxes, and binary masks.
    with torch.no_grad():
        out_cls, out_bbox, out_seg = model(input_tensors)
        preds_cls = torch.argmax(out_cls, dim=1).cpu().numpy()
        preds_bbox = out_bbox.cpu().numpy()
        preds_seg = (torch.sigmoid(out_seg).squeeze(1) > 0.5).cpu().numpy()

    # Create a side-by-side visual comparison for each sampled image.
    fig, axes = plt.subplots(num_samples, 2, figsize=(12, 5 * num_samples))
    fig.suptitle(f"{title_prefix} Results: Ground Truth vs Prediction", fontsize=15, y=0.92)
    
    for i in range(num_samples):
        gt = ground_truths[i]
        orig_img = gt['orig_img']
        h, w = orig_img.shape[:2]
        
        # Left column: ground-truth annotation
        ax_gt = axes[i, 0]
        ax_gt.imshow(orig_img)
        
        # Overlay the ground-truth segmentation mask in semi-transparent green.
        mask_rgba_gt = np.zeros((h, w, 4))
        mask_rgba_gt[..., 1] = 1.0 # Green channel
        mask_rgba_gt[..., 3] = gt['mask'] * 0.4 # Alpha
        ax_gt.imshow(mask_rgba_gt)
        
        # Draw the ground-truth bounding box using a green dashed outline.
        xmin, ymin, xmax, ymax = np.array(gt['bbox']) * [w, h, w, h]
        rect_gt = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, 
                                    linewidth=2.5, edgecolor='lime', facecolor='none', linestyle='--')
        ax_gt.add_patch(rect_gt)
        
        # Display the ground-truth gesture label near the bounding box.
        ax_gt.text(xmin, ymin - 8, f"GT: {gt['label']}", color='lime', fontsize=12, 
                   fontweight='bold', bbox=dict(facecolor='black', alpha=0.6, edgecolor='none'))
        if i == 0: ax_gt.set_title("Ground Truth", fontsize=16)
        ax_gt.axis('off')

        # Right column: model prediction
        ax_pred = axes[i, 1]
        ax_pred.imshow(orig_img)
        
        # Overlay the predicted segmentation mask in semi-transparent yellow.
        mask_rgba_pred = np.zeros((h, w, 4))
        mask_rgba_pred[..., 0] = 1.0 # Red
        mask_rgba_pred[..., 1] = 1.0 # Green
        mask_rgba_pred[..., 3] = preds_seg[i] * 0.5 # Alpha
        ax_pred.imshow(mask_rgba_pred)
        
        # Draw the predicted bounding box using a solid red outline.
        xmin_p, ymin_p, xmax_p, ymax_p = preds_bbox[i] * [w, h, w, h]
        rect_pred = patches.Rectangle((xmin_p, ymin_p), xmax_p - xmin_p, ymax_p - ymin_p, 
                                      linewidth=2.5, edgecolor='red', facecolor='none')
        ax_pred.add_patch(rect_pred)
        
        # Highlight whether the predicted class matches the ground truth:
        # green for correct predictions, dark red for incorrect predictions.
        pred_label_name = GESTURE_CLASSES_INV[preds_cls[i]]
        text_bg_color = 'green' if pred_label_name == gt['label'] else 'darkred'
        
        ax_pred.text(xmin_p, ymin_p - 8, f"Pred: {pred_label_name}", color='white', fontsize=12, 
                     fontweight='bold', bbox=dict(facecolor=text_bg_color, alpha=0.8, edgecolor='none'))
        if i == 0: ax_pred.set_title("Model Prediction", fontsize=16)
        ax_pred.axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Task Gesture Network Visualisation")
    parser.add_argument(
        '--split', 
        type=str, 
        choices=['val', 'test'], 
        default='test',
        help="Choose which dataset split to visualize: 'val' (validation set) or 'test' (default, test set)"
    )
    parser.add_argument(
        '--n', 
        type=int, 
        default=5,
        help="Number of random images to visualize (default: 5): "
    )
    args = parser.parse_args()
    
    run_visualization(split=args.split, num_samples=args.n)