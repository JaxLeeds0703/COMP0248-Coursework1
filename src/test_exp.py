import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import seaborn as sns

from model_exp import MultiTaskGestureNet


class TestGestureDataset(Dataset):
    def __init__(self, csv_file, img_size=(224, 224), use_depth=False):
        self.df = pd.read_csv(csv_file)  
        self.img_w, self.img_h = img_size 
        self.use_depth = use_depth

        # Standard ImageNet mean and std for RGB normalization; must exactly match training
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]


    def __len__(self):
        return len(self.df)


    def _get_bbox_from_mask(self, mask_np):
        # np.where finds all pixels > 0 (foreground/hand) and returns their (y, x) coordinates
        y_indices, x_indices = np.where(mask_np > 0)
        if len(y_indices) == 0 or len(x_indices) == 0:
            return [0.0, 0.0, 0.0, 0.0] # If no hand is present, return four zeros
        
        # Find the extreme box coordinates: left (xmin), right (xmax), top (ymin), and bottom (ymax)
        xmin, xmax = np.min(x_indices), np.max(x_indices)
        ymin, ymax = np.min(y_indices), np.max(y_indices)
        h, w = mask_np.shape

        # Normalize coordinates by image width/height to convert them into relative values in [0.0, 1.0]
        return [xmin / w, ymin / h, xmax / w, ymax / h]


    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # 1. Handel RGB
        img_path = row['rgb_path']
        image_rgb = Image.open(img_path).convert("RGB")
        orig_img_tensor = TF.to_tensor(image_rgb.resize((self.img_w, self.img_h), Image.BILINEAR))
        
        # Resize -> convert to tensor -> apply ImageNet normalization with mean and std
        image_rgb = image_rgb.resize((self.img_w, self.img_h), Image.BILINEAR)
        image_tensor_rgb = TF.normalize(TF.to_tensor(image_rgb), self.mean, self.std)

        # 2. Handel Depth
        has_depth = row.get('has_depth', False)
        if has_depth and pd.notna(row['depth_path']):
            depth_np = np.array(Image.open(row['depth_path']))
            if depth_np.ndim == 3: depth_np = depth_np[..., 0] # If it has 3 channels, keep only the first one

            # Convert to a tensor and resize to 224×224
            depth_tensor = torch.from_numpy(depth_np.astype(np.float32)).unsqueeze(0).unsqueeze(0)
            depth_tensor = torch.nn.functional.interpolate(depth_tensor, size=(self.img_h, self.img_w), mode='bilinear', align_corners=False).squeeze(0)
            
            # Apply the exact same percentile-based normalization as in training, clipping the top and bottom 1% outliers
            valid_pixels = depth_tensor[depth_tensor > 0]
            if valid_pixels.numel() > 0:
                d_min, d_max = torch.quantile(valid_pixels, 0.01), torch.quantile(valid_pixels, 0.99)
                if (d_max - d_min) > 1e-6:
                    depth_tensor = (depth_tensor - d_min) / (d_max - d_min)
                    depth_tensor = depth_tensor.clamp(0.0, 1.0)
                    # Stretch depth values from [0,1] to [-1,1] to better align with the RGB feature scale
            image_tensor_depth = (depth_tensor - 0.5) / 0.5
        else:
            image_tensor_depth = torch.zeros((1, self.img_h, self.img_w), dtype=torch.float32)

        # 3. Handel Mask & Bbox
        has_mask = float(row['has_mask'])
        mask_np = np.zeros((self.img_h, self.img_w), dtype=np.float32)
        bbox = [0.0, 0.0, 0.0, 0.0]

        if has_mask > 0.5 and pd.notna(row['mask_path']):
            mask_img = Image.open(row['mask_path']).convert("L").resize((self.img_w, self.img_h), Image.NEAREST)
            mask_np = (np.array(mask_img) > 127).astype(np.float32)
            bbox = self._get_bbox_from_mask(mask_np)

        # Concatenate RGB (3,224,224) and depth (1,224,224) into a 4-channel input
        # image_tensor = torch.cat([image_tensor_rgb, image_tensor_depth], dim=0)
        if not self.use_depth:
            image_tensor = image_tensor_rgb
        else:
            image_tensor = torch.cat([image_tensor_rgb, image_tensor_depth], dim=0)

        return {
                'image': image_tensor,
                'label': torch.tensor(int(row['class_label']), dtype=torch.long),
                'bbox': torch.tensor(bbox, dtype=torch.float32),
                'mask': torch.tensor(mask_np, dtype=torch.float32),
                'has_mask': torch.tensor(has_mask, dtype=torch.float32),
                'orig_img': orig_img_tensor
                }

# ------------------------------------------------------------------------------------------
# Evaluating Metrics Computing Function

# Compute the bbox IoU (Intersection over Union)
def calculate_iou_bbox(box1, box2):
    # Intersection corners: top-left = max of mins, bottom-right = min of maxes
    xA, yA = max(box1[0], box2[0]), max(box1[1], box2[1])
    xB, yB = min(box1[2], box2[2]), min(box1[3], box2[3])
    # Compute the intersection area; if there is no overlap, max(0, ...) makes it zero
    interArea = max(0, xB - xA) * max(0, yB - yA)
    # Compute the area of each box separately
    box1Area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2Area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    # Union area = box1 area + box2 area - their overlap
    unionArea = box1Area + box2Area - interArea
    return interArea / unionArea if unionArea > 0 else 0.0

# Compute segmentation IoU and Dice score
def calculate_seg_metrics(pred_mask, gt_mask):
    pred_b = (pred_mask > 0.5).astype(np.float32)
    gt_b = (gt_mask > 0.5).astype(np.float32)
    # With binary masks, NumPy multiplication counts overlap pixels since only 1*1 = 1
    intersection = np.sum(pred_b * gt_b)
    union = np.sum(pred_b) + np.sum(gt_b) - intersection

    iou = intersection / union if union > 0 else 0.0
    # Dice = 2 × intersection / (predicted area + ground-truth area)
    dice = (2. * intersection) / (np.sum(pred_b) + np.sum(gt_b)) if (np.sum(pred_b) + np.sum(gt_b)) > 0 else 0.0
    return iou, dice

# ----------------------------------------------------------------------------------4
#  Run test
def run_testing():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f" {device}")

    TEST_USE_DEPTH = True        #*
    TEST_USE_SPATIAL_AUG = True   #*

    IN_CHANNELS = 4 if TEST_USE_DEPTH else 3

    BASE_DIR = Path(__file__).resolve().parent.parent
    TEST_CSV = BASE_DIR / "dataset" / "test_index.csv"

    # WEIGHTS_PATH = BASE_DIR / "weights" / "best_model_rgbd_v3_resnet.pth" # best weight
    WEIGHTS_NAME = f"best_model_exp_depth{TEST_USE_DEPTH}_aug{TEST_USE_SPATIAL_AUG}.pth"
    WEIGHTS_PATH = BASE_DIR / "weights" / WEIGHTS_NAME

    # OUTPUT_DIR = BASE_DIR / "results"
    # OUTPUT_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR = BASE_DIR / f"test_results_depth{TEST_USE_DEPTH}_aug{TEST_USE_SPATIAL_AUG}"
    OUTPUT_DIR.mkdir(exist_ok=True)

    # test_dataset = TestGestureDataset(TEST_CSV)
    # test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4)
    test_dataset = TestGestureDataset(TEST_CSV, use_depth=TEST_USE_DEPTH)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4)

    # model = MultiTaskGestureNet(in_channels=4, num_classes=10).to(device)
    # model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
    # #Set the model to evaluation mode to disable Dropout and BatchNorm randomness
    # model.eval()
    model = MultiTaskGestureNet(in_channels=IN_CHANNELS, num_classes=10).to(device)
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
    
    model.eval()

    all_preds_cls, all_labels_cls = [], []
    all_bbox_ious, all_seg_ious, all_seg_dices = [], [], []
    vis_count = 0

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing Model"):
            images = batch['image'].to(device)
            labels = batch['label'].numpy()
            bboxes = batch['bbox'].numpy()
            masks = batch['mask'].numpy()
            has_masks = batch['has_mask'].numpy()
            orig_imgs = batch['orig_img'].numpy()

            # Run forward pass to produce the three prediction outputs
            out_cls, out_bbox, out_seg = model(images)
            
            # # Classification: take the index (0–9) of the largest score among the 10 outputs
            preds_cls = torch.argmax(out_cls, dim=1).cpu().numpy()
            all_preds_cls.extend(preds_cls)
            all_labels_cls.extend(labels)

            # Get the predicted bbox and segmentation outputs
            preds_bbox = out_bbox.cpu().numpy()
            # Apply sigmoid to the segmentation output to obtain a 0–1 probability map,
            # then squeeze out the redundant channel dimension
            preds_seg = torch.sigmoid(out_seg).squeeze(1).cpu().numpy() # shape (B, H, W)

       
            for i in range(len(images)):
                #Only evaluate frames that have a manually annotated mask
                if has_masks[i] > 0.5:
                    # BBox Evaluation
                    iou_b = calculate_iou_bbox(preds_bbox[i], bboxes[i])
                    all_bbox_ious.append(iou_b)
                    
                    # Seg Evaluation
                    iou_s, dice_s = calculate_seg_metrics(preds_seg[i], masks[i])
                    all_seg_ious.append(iou_s)
                    all_seg_dices.append(dice_s)

                    # Visualization (Save 5 pictures into the file)
                    if vis_count < 5:
                        save_overlays(orig_imgs[i], preds_bbox[i], bboxes[i], preds_seg[i], vis_count, OUTPUT_DIR)
                        vis_count += 1

    # --------------------------------------------------------------------------------------------
    # Metrics

    # Classification
    acc = accuracy_score(all_labels_cls, all_preds_cls)
    f1 = f1_score(all_labels_cls, all_preds_cls, average='macro')
    print(f"[1. Classification Metrics]")
    print(f"  - Overall Top-1 Accuracy: {acc * 100:.2f}%")
    print(f"  - Macro-Averaged F1 Score: {f1:.4f}")

    # Detection
    bbox_ious_arr = np.array(all_bbox_ious)
    acc_05 = np.mean(bbox_ious_arr >= 0.5) * 100 if len(bbox_ious_arr) > 0 else 0.0
    mean_bbox_iou = np.mean(bbox_ious_arr) if len(bbox_ious_arr) > 0 else 0.0
    print(f"\n[2. Detection Metrics]")
    print(f"  - Detection Accuracy @ 0.5 IoU: {acc_05:.2f}%")
    print(f"  - Mean Bounding-Box IoU: {mean_bbox_iou:.4f}")

    # Segmentation
    print(f"\n[3. Segmentation Metrics]")
    print(f"  - Mean IoU (Hand vs BG): {np.mean(all_seg_ious):.4f}")
    print(f"  - Mean Dice Coefficient: {np.mean(all_seg_dices):.4f}")

    # confusion matrix
    cm = confusion_matrix(all_labels_cls, all_preds_cls)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix (Test Set)')
    plt.ylabel('Ground Truth')
    plt.xlabel('Predicted')
    cm_path = OUTPUT_DIR / 'confusion_matrix_new.png'
    plt.savefig(cm_path)
    print(f"\n Confusion matrix saved to: {cm_path}")
    print(f"Overlay visualizations saved in: {OUTPUT_DIR}")


def save_overlays(orig_img, pred_box, gt_box, pred_mask, count, out_dir):
    img = np.transpose(orig_img, (1, 2, 0)) # (3, H, W) -> (H, W, 3)
    h, w, _ = img.shape
    
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.imshow(img)
    
    mask_rgba = np.zeros((h, w, 4))
    mask_rgba[..., 0] = 1.0 # R
    mask_rgba[..., 1] = 1.0 # G
    mask_rgba[..., 3] = (pred_mask > 0.5).astype(np.float32) * 0.5 # Alpha 
    ax.imshow(mask_rgba)
    
    # Draw Ground Truth Block (Green)
    xmin_g, ymin_g, xmax_g, ymax_g = gt_box * [w, h, w, h]
    ax.add_patch(patches.Rectangle((xmin_g, ymin_g), xmax_g - xmin_g, ymax_g - ymin_g, 
                                   linewidth=2, edgecolor='lime', fill=False, linestyle='--'))
    
    # Draw Predicted Block (Red)
    xmin_p, ymin_p, xmax_p, ymax_p = pred_box * [w, h, w, h]
    ax.add_patch(patches.Rectangle((xmin_p, ymin_p), xmax_p - xmin_p, ymax_p - ymin_p, 
                                   linewidth=2, edgecolor='red', fill=False))
    
    plt.axis('off')
    plt.title("Red: Pred BBox | Green: GT BBox | Yellow: Seg Mask")
    plt.tight_layout()
    plt.savefig(out_dir / f'overlay_{count}.png', bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    run_testing()