import torch
import torch.nn as nn
import torch.nn.functional as F

#-----------------------      Residual Block     -----------------------------------------
# The basic unit for the backbone
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock, self).__init__()

        # First convolution layer: Feature extraction, stride = 2
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        #BatchNorm: Bring values back within a reasonable range to prevent numerical overflow
        self.bn1 = nn.BatchNorm2d(out_channels)

        # Second convolution layer: continue Feature extraction, stride = 1
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut Connection
        self.shortcut = nn.Sequential()

        # Dimension Matching:
        # Should spatial downsampling (stride ≠ 1) or channel expansion (in ≠ out) in the feature map cause mismatched input and output tensor shapes,
        # a 1x1 convolution (Projection Shortcut) is introduced to adjust the input dimensions, thereby satisfying the element-wise addition requirement for residual connections.
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        # First convolution layer -> BatchNorm -> ReLU
        out = F.relu(self.bn1(self.conv1(x)))
        # Second convolution layer -> BatchNorm
        out = self.bn2(self.conv2(out))
        # add shortcut connection(residual)
        out += self.shortcut(x) 
        #Another ReLU
        out = F.relu(out)
        return out




# ==================       MultiTaskGestureNet      ===================================
class MultiTaskGestureNet(nn.Module):

    #************************* RGBD Modified  ******************************************
    # def __init__(self, in_channels=3, num_classes=10):
    def __init__(self, in_channels=4, num_classes=10): # change input chanel form 3 to 4
    #***********************************************************************************

        super(MultiTaskGestureNet, self).__init__()
        
        # ------------- 1. Encoder (ResNet Backbone) --------------------------------------
        #Initial layer: Employing a larger convolutional kernel (7x7) and stride (stride=2) to extract coarse edges.
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        #Stacking ResBlocks: As the number of layers increases, the number of channels doubles, but the width and height halve.
        self.layer1 = ResBlock(64, 64, stride=1)   # output: e1
        self.layer2 = ResBlock(64, 128, stride=2)  # output: e2
        self.layer3 = ResBlock(128, 256, stride=2) # output: e3
        self.layer4 = ResBlock(256, 512, stride=2) # output: e4 (Bottleneck)


        # ------------- 2. Indiviual Task Head: Classification Head & Bounding box Head --------
        #Global Average Pooling(GAP): 
        #(Batch,512,H,W) -> (Batch,512,1,1) by spatial averaging
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classification Head: 
        self.cls_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),  #randomly zero half activations to avoid overfitting
            nn.Linear(256, num_classes)
        )
        
        # Bbox head: 4 coordinate [xmin, ymin, xmax, ymax]
        # self.bbox_head = nn.Sequential(
        #     nn.Linear(512, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, 4),
        #     nn.Sigmoid()      #sigmoid: constrain outputs to [0,1] to match normalized coordinates
        # )

        # Keep spatial dimensions for convolutional processing
        self.bbox_head = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            # Preserve a compact 4×4 spatial map instead of collapsing everything to 1×1
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),

            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Linear(256, 4),
            nn.Sigmoid()
        )

        # ------------ 3. Decoder (Segmentation Head with Skip Connections) ----------------------
        # Decoder in_chanels must match the concatenated channel count
        # ConvTranspose2d(deconv): upsampling spatial size (H,W) by ~2x
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        #Skip concat: upconv3(256ch) + e3(256ch) -> 512ch into ResBlock
        self.dec_conv3 = ResBlock(512, 256) 

        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec_conv2 = ResBlock(256, 128)  # upconv3(128ch) + e3(128ch) -> 256ch

        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec_conv1 = ResBlock(128, 64)  # 6upconv3(64ch) + e3(64ch) -> 128ch
        
        # Extra upsampling needed to restore original resolution (224x224)
        self.upconv0 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.final_upconv = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)

        # # Seg head: 1x1 conv reduces 32ch -> 1ch (binary palm mask)
        self.seg_head = nn.Conv2d(32, 1, kernel_size=1) 

    def forward(self, x):
        input_hw = x.shape[2:]
        # Encoder Forward: x is the input batch tensor from the DataLoader (224x224 images)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x) # Shape is now 1/4 of the original (224x224 -> 56x56)

        e1 = self.layer1(x)  # e1 shape: (B, 64, H/4, W/4)
        e2 = self.layer2(e1) # e2 shape: (B, 128, H/8, W/8)
        e3 = self.layer3(e2) # e3 shape: (B, 256, H/16, W/16)
        e4 = self.layer4(e3) # e4 shape: (B, 512, H/32, W/32) -> Bottleneck

        pooled = self.global_pool(e4) #GAP
        flattened = torch.flatten(pooled, 1) #Flatten: (B,512,1,1) -> (B,512) for the Linear layer

        # Run cls head and bbox head 
        out_cls = self.cls_head(flattened)

        # In forward: directly use the intact e4 feature map (7×7) instead of a fully collapsed representation
        # out_bbox = self.bbox_head(flattened)
        out_bbox = self.bbox_head(e4)


        # Decoder Forward (U-Net) 
        d3 = self.upconv3(e4)
        
        if d3.shape != e3.shape: d3 = F.interpolate(d3, size=e3.shape[2:])
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec_conv3(d3)

        d2 = self.upconv2(d3)
        if d2.shape != e2.shape: d2 = F.interpolate(d2, size=e2.shape[2:])
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec_conv2(d2)

        d1 = self.upconv1(d2)
        if d1.shape != e1.shape: d1 = F.interpolate(d1, size=e1.shape[2:])
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec_conv1(d1)
        
        d0 = self.upconv0(d1)
        out_seg_features = self.final_upconv(d0)
        
        # segment ouput: shape (B, 1, H, W)
        out_seg = self.seg_head(out_seg_features)
        
        # if out_seg.shape[2:] != (224, 224):
        #     out_seg = F.interpolate(out_seg, size=(224, 224), mode='bilinear', align_corners=False)
        if out_seg.shape[2:] != input_hw:
            out_seg = F.interpolate(out_seg, size=input_hw, mode='bilinear', align_corners=False)

        return out_cls, out_bbox, out_seg




if __name__ == "__main__":
    # model = MultiTaskGestureNet()

    #************************* RGBD Modified  ********************************************
    #Just for small check in Jupyter Notebook
    #dummy_input = torch.randn(4, 3, 224, 224) # simulate batch_size=4, 3 chanels, 224x224
    # dummy_input = torch.randn(4, 4, 224, 224) #  simulate batch_size=4, 4 chanels, 224x224
    # #*************************************************************************************
    
    # out_cls, out_bbox, out_seg = model(dummy_input)

    # print(f"cls shape:  {out_cls.shape}  (expected [4,10])")
    # print(f"bbox shape: {out_bbox.shape} (expected [4,4])")
    # print(f"seg shape:  {out_seg.shape}  (expected [4,1,224,224])")
    model = MultiTaskGestureNet(num_classes=10)
    dummy_input = torch.randn(4, 4, 224, 224)
    out_cls, out_bbox, out_seg = model(dummy_input)

    print("out_cls shape:", out_cls.shape)
    print("out_bbox shape:", out_bbox.shape)
    print("out_seg shape:", out_seg.shape)
