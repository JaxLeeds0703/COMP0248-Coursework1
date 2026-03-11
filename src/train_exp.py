import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm

from dataloader_exp import HandGestureDataset
from model_exp import MultiTaskGestureNet

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")

    USE_DEPTH = True        
    USE_SPATIAL_AUG = True
    
    IN_CHANNELS = 4 if USE_DEPTH else 3

    # Hyper Parameters
    BATCH_SIZE = 32
    EPOCHS = 60 #20
    LEARNING_RATE = 1e-4 #3e-5
    
    # Path settings
    BASE_DIR = Path(__file__).resolve().parent.parent
    CSV_PATH = BASE_DIR / "dataset" / "dataset_index_split.csv"
    WEIGHTS_DIR = BASE_DIR / "weights"
    WEIGHTS_DIR.mkdir(exist_ok=True)

    # train_dataset = HandGestureDataset(csv_file=CSV_PATH, img_size=(224, 224), split='train')
    # val_dataset = HandGestureDataset(csv_file=CSV_PATH, img_size=(224, 224), split='val')
    train_dataset = HandGestureDataset(csv_file=CSV_PATH, img_size=(224, 224), split='train',
                                       use_depth=USE_DEPTH, use_spatial_aug=USE_SPATIAL_AUG)
    val_dataset = HandGestureDataset(csv_file=CSV_PATH, img_size=(224, 224), split='val',
                                     use_depth=USE_DEPTH, use_spatial_aug=USE_SPATIAL_AUG)
    
    train_size = len(train_dataset)
    val_size = len(val_dataset)
    print(f"Training size: {train_size} | Validation size: {val_size}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # model = MultiTaskGestureNet().to(device)
    model = MultiTaskGestureNet(in_channels=IN_CHANNELS, num_classes=10).to(device)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    #Define a cosine annealing learning-rate scheduler
    # T_max=EPOCHS means the LR reaches its minimum after EPOCHS epochs
    # eta_min=1e-6 sets the minimum allowed learning rate floor
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    
    # Loss functions (reduction='none' is crucial for dynamic masking)
    # # Classification: Cross Entropy Loss
    criterion_cls = nn.CrossEntropyLoss()
    
    # Loss functions
    criterion_cls = nn.CrossEntropyLoss() 
    # Bbox: L1 Loss
    criterion_bbox = nn.L1Loss(reduction='none')
    # Segmentation: BCEWithLogitsLoss (BCEWithLogitsLoss: applies sigmoid internally for numerical stability)
    criterion_seg = nn.BCEWithLogitsLoss(reduction='none') 

    best_val_loss = float('inf')

    for epoch in range(EPOCHS):
        model.train()
        train_loss_total = 0.0
        
        #print(f"\n[{epoch+1}/{EPOCHS}] Training...")
        # Wrap the dataloader with tqdm to show a progress bar
        progress_bar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{EPOCHS}] Train", leave=False)
        
        for batch in progress_bar:
            # Move all tensors in the batch dict to GPU memory
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            bboxes = batch['bbox'].to(device)
            masks = batch['mask'].to(device)
            valid_bbox = batch['valid_bbox'].to(device)
            valid_seg = batch['valid_seg'].to(device)
            
            #clear gradients from the previous iteration
            optimizer.zero_grad()

            # Forward Pass
            out_cls, out_bbox, out_seg = model(images)
            
            # 1. Classification Loss
            loss_cls = criterion_cls(out_cls, labels)
            
            # 2. BBox Loss (Gated)
            loss_bbox_raw = criterion_bbox(out_bbox, bboxes).mean(dim=1) 
            loss_bbox = (loss_bbox_raw * valid_bbox).sum() / (valid_bbox.sum() + 1e-8)
            
            # 3. Segmentation Loss (Gated)
            loss_seg_raw = criterion_seg(out_seg, masks).mean(dim=(1, 2, 3))
            loss_seg = (loss_seg_raw * valid_seg).sum() / (valid_seg.sum() + 1e-8)
            
            # 4. Combine Total Loss
            # loss = loss_cls + 10.0 * loss_bbox + 1.0 * loss_seg
            # loss = loss_cls + 30.0 * loss_bbox + 1.0 * loss_seg
            loss = loss_cls + 30.0 * loss_bbox + 10.0 * loss_seg
            
            # Backpropagation
            loss.backward() #Backpropagate the total loss to compute gradients for all ~8M parameters
            optimizer.step() # Optimizer step: update all parameters using the computed gradients
            
            # Log the total loss
            train_loss_total += loss.item()
            
            # Monitor the raw losses of all three tasks in real time (before weighting) to diagnose which one is lagging behind
            progress_bar.set_postfix({
                'Tot': f"{loss.item():.2f}",
                'Cls': f"{loss_cls.item():.2f}",
                'Box': f"{loss_bbox.item():.3f}",
                'Seg': f"{loss_seg.item():.2f}"
            })

        #============================  Validation Phase  ====================================
        avg_train_loss = train_loss_total / len(train_loader)
        
        # Disable gradient tracking for evaluation; inference only, no learning
        model.eval()
        val_loss_total = 0.0
        val_cls_tot, val_box_tot, val_seg_tot = 0.0, 0.0, 0.0
        correct_cls = 0
        total_cls = 0
        
        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(device)
                labels = batch['label'].to(device)
                bboxes = batch['bbox'].to(device)
                masks = batch['mask'].to(device)
                valid_bbox = batch['valid_bbox'].to(device)
                valid_seg = batch['valid_seg'].to(device)
                
                out_cls, out_bbox, out_seg = model(images)
                
                loss_cls = criterion_cls(out_cls, labels)
                
                #------------------------------------------------------------------------
                loss_bbox_raw = criterion_bbox(out_bbox, bboxes).mean(dim=1)    # L1 loss
                #loss_bbox_raw = generalized_iou_loss(out_bbox, bboxes)         # GIoU loss
                #------------------------------------------------------------------------

                loss_bbox = (loss_bbox_raw * valid_bbox).sum() / (valid_bbox.sum() + 1e-8)
                
                loss_seg_raw = criterion_seg(out_seg, masks).mean(dim=(1, 2, 3))
                loss_seg = (loss_seg_raw * valid_seg).sum() / (valid_seg.sum() + 1e-8)
                
                # val_loss = loss_cls + 10.0 * loss_bbox + 10.0 * loss_seg
                # val_loss = loss_cls + 30.0 * loss_bbox + 1.0 * loss_seg
                val_loss = loss_cls + 30.0 * loss_bbox + 10.0 * loss_seg
                
                val_loss_total += val_loss.item()
                val_cls_tot += loss_cls.item()
                val_box_tot += loss_bbox.item()
                val_seg_tot += loss_seg.item()
                
                # Compute classification accuracy
                _, predicted = torch.max(out_cls.data, 1)
                total_cls += labels.size(0)
                correct_cls += (predicted == labels).sum().item()
                
        # Computing Validation set Metrics
        num_val_batches = len(val_loader)
        avg_val_loss = val_loss_total / num_val_batches
        avg_val_cls = val_cls_tot / num_val_batches
        avg_val_box = val_box_tot / num_val_batches
        avg_val_seg = val_seg_tot / num_val_batches
        val_acc = 100 * correct_cls / total_cls
        
        print(f"Epoch [{epoch+1}/{EPOCHS}] | Train Loss: {avg_train_loss:.3f} | Val Loss: {avg_val_loss:.3f} | Acc: {val_acc:.2f}%")
        print(f"   -> Val Details: Cls Loss: {avg_val_cls:.3f} | Box L1: {avg_val_box:.4f} | Seg BCE: {avg_val_seg:.3f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_name = f"best_model_exp_depth{USE_DEPTH}_aug{USE_SPATIAL_AUG}.pth"
            save_path = WEIGHTS_DIR / save_name
            
            torch.save(model.state_dict(), save_path)
            print(f"Model improved!  {save_path.name}")
            # save_path = WEIGHTS_DIR / "best_model_rgbd_v3_resnet.pth"
            # torch.save(model.state_dict(), save_path)
            # print(f"   => Model improved! Saved to {save_path.name}")


        # At the end of each epoch, let the scheduler reduce the learning rate slightly
        scheduler.step()
        
        # Print learning rate
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Current Learning Rate: {current_lr:.6f}")

if __name__ == "__main__":
    train()
