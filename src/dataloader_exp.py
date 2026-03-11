import os
import warnings
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn.functional as nnF
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import random

from pathlib import Path

#Dataset (inherit from PyTorch's Dataset class)
class HandGestureDataset(Dataset):
    def __init__(
        self,
        csv_file,
        img_size=(224, 224),
        split='train',
        strict_depth=False,
        warn_missing_depth=True,
        use_depth=True,
        use_spatial_aug =True
    ):
        # Load the entire .csv file into memory and convert it into a DataFrame table.
        self.data_frame = pd.read_csv(csv_file).reset_index(drop=True)
        self.img_size = img_size
        self.img_w, self.img_h = img_size
        self.split = split

        self.strict_depth = strict_depth
        self.warn_missing_depth = warn_missing_depth
        self.use_depth = use_depth
        self._missing_depth_warned = set()
        self.use_spatial_aug = use_spatial_aug
    
        #-------------------------------- 1. Data Augmentation ------------------------------------------------
        # *refrain from employing spatial enhancements such as flipping; limit the process to colour dithering.
        if self.split == 'train':
            # Randomly alter brightness, contrast and saturation
            #self.color_jitter = T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)
            self.color_jitter = T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1)
        else:
            self.color_jitter = None

        #-------------------------------- 2. ImageNet Normalization ---------------------------------------------------
        # Reference: https://pytorch.org/vision/stable/models.html
        # The mean and standard deviation values used below are the standard ImageNet statistics.
        # This is standard practice in PyTorch for normalizing RGB images before feeding them into deep neural networks.
        self.normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        if 'split' in self.data_frame.columns:
            self.data_frame = self.data_frame[self.data_frame['split'] == self.split].reset_index(drop=True)

    # Obtain the total length of the dataset
    def __len__(self):
        return len(self.data_frame)
    
    #================================= Update Part ===========================================
    def _infer_depth_path(self, row, img_path):
        if 'depth_path' in row.index and pd.notna(row['depth_path']):
            return Path(row['depth_path'])
        img_path = Path(img_path)
        return img_path.parent.parent / "depth" / img_path.name

    def _warn_depth_once(self, depth_path):
        depth_path = str(depth_path)
        if self.warn_missing_depth and depth_path not in self._missing_depth_warned:
            warnings.warn(f"[HandGestureDataset] Missing depth file: {depth_path}")
            self._missing_depth_warned.add(depth_path)

    def _load_depth_tensor(self, depth_path):
        depth_path = Path(depth_path)
        if not depth_path.exists():
            if self.strict_depth:
                raise FileNotFoundError(f"Depth file not found: {depth_path}")
            self._warn_depth_once(depth_path)
            depth_tensor = torch.zeros(1, self.img_h, self.img_w, dtype=torch.float32)
            return depth_tensor, 0.0

        depth_img = Image.open(depth_path)
        depth_np = np.array(depth_img)

        if depth_np.ndim == 3:
            depth_np = depth_np[..., 0]

        depth_np = depth_np.astype(np.float32)
        depth_np[~np.isfinite(depth_np)] = 0.0

        depth_tensor = torch.from_numpy(depth_np).unsqueeze(0).unsqueeze(0)
        depth_tensor = nnF.interpolate(depth_tensor, size=(self.img_h, self.img_w), mode='bilinear', align_corners=False)
        depth_tensor = depth_tensor.squeeze(0)

        valid_pixels = depth_tensor[depth_tensor > 0] # Filter out zeros (invalid depth values or dead pixels)
        if valid_pixels.numel() == 0:
            depth_tensor = torch.zeros_like(depth_tensor)
        else:
            # Use the 1st and 99th percentiles instead of the raw min/max values
            d_min = torch.quantile(valid_pixels, 0.01)
            d_max = torch.quantile(valid_pixels, 0.99)
            if (d_max - d_min) < 1e-6:
                depth_tensor = torch.zeros_like(depth_tensor)
            else:
                # Precisely normalize the valid depth range to [0.0, 1.0]
                depth_tensor = (depth_tensor - d_min) / (d_max - d_min)
                depth_tensor = depth_tensor.clamp(0.0, 1.0)

        return depth_tensor.float(), 1.0
    #===========================================================================================================

     #make bounding box
    def _get_bbox_from_mask(self, mask_np):
        """
        Extract the minimum bounding box from a 2D mask image in numpy format.
        Parameters: mask_np (a two-dimensional array of shape H x W, where 0 -> background and > 0 -> hands)
        Returns: [xmin, ymin, xmax, ymax] representing relative coordinates normalised to the range 0.0 to 1.0.
        """
        #For all pixels in this two-dimensional array whose values exceed 0, the vertical coord (y) and horizontal coord (x)
        y_indices, x_indices = np.where(mask_np > 0) 

        # If the length is zero, it indicates that no pixels exceed 0
        # We shall return an invalid bounding box filled entirely with 0
        if len(y_indices) == 0 or len(x_indices) == 0:
            return [0.0, 0.0, 0.0, 0.0]
        
        xmin = np.min(x_indices)
        xmax = np.max(x_indices)
        ymin = np.min(y_indices)
        ymax = np.max(y_indices)

        # get the actural height(h)and Width(w) for mask
        h, w = mask_np.shape
        # Height and Width Normalization
        return [xmin / w, ymin / h, xmax / w, ymax / h]

    def __getitem__(self, idx):
        row = self.data_frame.iloc[idx]
        img_path = row['rgb_path']

        # Read RGB (PIL Image)
        image_rgb = Image.open(img_path).convert("RGB")
        image_rgb = image_rgb.resize((self.img_w, self.img_h), Image.BILINEAR)

        # Read Depth (Tensor)
        depth_path = self._infer_depth_path(row, img_path)
        image_tensor_depth, has_depth_flag = self._load_depth_tensor(depth_path)

        # Read Mask (PIL Image)
        has_mask_flag = float(row['has_mask'])
        mask_img = None
        if has_mask_flag > 0.5 and pd.notna(row['mask_path']):
            mask_path = row['mask_path']
            mask_img = Image.open(mask_path).convert("L")
            mask_img = mask_img.resize((self.img_w, self.img_h), Image.NEAREST)

        # ====================   Upgarded:Synchronous Spatial Augmentation)  ================================
        if self.split == 'train':
            # a) Apply color jitter to RGB only
            if self.color_jitter:
                image_rgb = self.color_jitter(image_rgb) #Add random brightness/contrast jitter

            # b) Randomly apply translation, rotation, and scaling (80% chance) to reduce frame-to-frame correlation
            if self.use_spatial_aug and random.random() < 0.8:
                angle = random.uniform(-15, 15)  # Random rotation: -15° to +15°
                translate = [int(random.uniform(-0.1, 0.1) * self.img_w),
                             int(random.uniform(-0.1, 0.1) * self.img_h)] # Random translation: up to ±10%
                scale = random.uniform(0.85, 1.15) # Random scaling: 85% to 115%
                shear = 0.0

                # Apply the exact same affine transform to RGB, depth, and mask
                image_rgb = TF.affine(image_rgb, angle, translate, scale, shear, interpolation=TF.InterpolationMode.BILINEAR)
                image_tensor_depth = TF.affine(image_tensor_depth, angle, translate, scale, shear, interpolation=TF.InterpolationMode.BILINEAR)

                if mask_img is not None:
                    # Masks must use NEAREST interpolation to avoid gray (non-binary) edge values
                    mask_img = TF.affine(mask_img, angle, translate, scale, shear, interpolation=TF.InterpolationMode.NEAREST)

        # =====================================================================
        # Post-processing: extract the bounding box and convert outputs to tensors

        # Handling Mask and Bounding Box
        mask_np = np.zeros((self.img_h, self.img_w), dtype=np.float32)   # Defult
        bbox = [0.0, 0.0, 0.0, 0.0]                                      # Defult

        if mask_img is not None:
            mask_np = np.array(mask_img)
            mask_np = (mask_np > 127).astype(np.float32)
            # since the mask has already been transformed, the computed bbox automatically stays aligned with it
            bbox = self._get_bbox_from_mask(mask_np)

        # Hnadel RGB Tensor
        # Convert RGB to a tensor and apply ImageNet normalization
        image_tensor_rgb = TF.to_tensor(image_rgb)
        image_tensor_rgb = self.normalize(image_tensor_rgb)

        # Align feature distributions: roughly map depth values to a range close to normalized RGB, around [-1, 1]
        image_tensor_depth = (image_tensor_depth - 0.5) / 0.5

        # Combine chanels
        if self.use_depth:
            image_tensor = torch.cat([image_tensor_rgb, image_tensor_depth], dim=0)
        else:
            # zero_depth = torch.zeros_like(image_tensor_depth)
            # image_tensor = torch.cat([image_tensor_rgb, zero_depth], dim=0)
            image_tensor = image_tensor_rgb

        # Convert to PyTorch Tensors
        label_tensor = torch.tensor(int(row['class_label']), dtype=torch.long)
        bbox_tensor = torch.tensor(bbox, dtype=torch.float32)
        mask_tensor = torch.tensor(mask_np, dtype=torch.float32).unsqueeze(0)
        has_mask_tensor = torch.tensor(has_mask_flag, dtype=torch.float32)

        return {
            'image': image_tensor,
            'label': label_tensor,
            'bbox': bbox_tensor,
            'mask': mask_tensor,
            'valid_bbox': has_mask_tensor,
            'valid_seg': has_mask_tensor,
            'has_depth': torch.tensor(has_depth_flag, dtype=torch.float32),
            'used_depth': torch.tensor(float(self.use_depth and has_depth_flag > 0.5), dtype=torch.float32),
            'img_path': str(img_path),
            'depth_path': str(depth_path)
        }

if __name__ == "__main__":
    csv_path = Path(__file__).resolve().parent.parent / "dataset" / "dataset_index_split.csv"
    if not csv_path.exists():
        print("Can;t find .csv file")
    else:
        dataset = HandGestureDataset(csv_file=csv_path, img_size=(224, 224), split='train')
        dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)
        
        for batch in dataloader:
            images = batch['image']
            bboxes = batch['bbox']
            print(f"Batch Images Shape: {images.shape}")
            print(f"BBoxes: \n{bboxes}")
            print(f"RGB min/max: {images[:, :3].min().item():.4f} / {images[:, :3].max().item():.4f}")
            print(f"Depth min/max: {images[:, 3:4].min().item():.4f} / {images[:, 3:4].max().item():.4f}")
            break
