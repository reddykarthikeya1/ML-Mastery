# 11.3 Image Segmentation (U-Net, Mask R-CNN, DeepLab)

## 🎯 Quick Overview
- **Semantic Segmentation**: Classifying every pixel (what is this pixel?)
- **Instance Segmentation**: Detecting and segmenting individual objects
- **Panoptic Segmentation**: Combining semantic and instance segmentation
- **Architectures**: FCN, U-Net, Mask R-CNN, and DeepLab (Atrous convolutions)
- **Foundation for**: Medical imaging, autonomous navigation, and photo editing

---

## 1. Types of Segmentation

Unlike object detection (bounding boxes), segmentation provides **pixel-perfect** masks.

### 1.1 Semantic Segmentation
Assigns a class label to every pixel in the image.
- *Key Characteristic*: Does not distinguish between different objects of the same class (e.g., all "cars" are the same color mask).

### 1.2 Instance Segmentation
Detects individual objects and generates a mask for each.
- *Key Characteristic*: Distinguishes between "Car 1" and "Car 2."

### 1.3 Panoptic Segmentation
The "ultimate" segmentation. It segments both **Things** (countable objects like people, cars) and **Stuff** (amorphous regions like sky, grass, road).

---

## 2. Core Architectures

### 2.1 Fully Convolutional Networks (FCN)
The first end-to-end network for pixel-wise prediction. Replaced the final dense layers of a CNN with $1 \times 1$ convolutions and used **Transposed Convolutions** (Upsampling) to restore image size.

### 2.2 U-Net (The Medical Gold Standard)
An **Encoder-Decoder** architecture with **Skip Connections**.
- **Encoder (Contracting Path)**: Captures context and features.
- **Decoder (Expanding Path)**: Enables precise localization.
- **Skip Connections**: Pass fine-grained spatial information from the encoder directly to the decoder.

### 2.3 Mask R-CNN (Instance Leader)
Extends Faster R-CNN by adding a third branch for predicting segmentation masks in parallel with the classification and box regression branches.
- Introduced **RoIAlign** to preserve exact spatial locations (fixed the quantization issues of RoIPool).

### 2.4 DeepLab
Introduced **Atrous (Dilated) Convolutions**. This allows the model to expand its "receptive field" without increasing the number of parameters or losing resolution through pooling.

---

## 💻 Python Code Examples

### 1. U-Net Block (PyTorch)
```python
import torch
import torch.nn as nn

class UNetBlock(nn.Module):
    def __init__(self, in_channels, out_dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_dim, 3, padding=1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_dim, out_dim, 3, padding=1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)
```

### 2. Segmentation Inference with Torchvision
```python
from torchvision import models
from torchvision import transforms
from PIL import Image

# 1. Load pre-trained DeepLabV3
model = models.segmentation.deeplabv3_resnet101(pretrained=True).eval()

# 2. Preprocess
input_image = Image.open("street.jpg")
preprocess = transforms.Compose([
    transforms.Resize(520),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
input_tensor = preprocess(input_image).unsqueeze(0)

# 3. Predict
with torch.no_grad():
    output = model(input_tensor)['out'][0]
output_predictions = output.argmax(0) # Highest prob class per pixel
```

---

## 📊 Summary Table

| Model | Type | Key Innovation | Best For |
|-------|------|----------------|----------|
| **FCN** | Semantic | $1 \times 1$ convs + Upsampling | Historical baseline |
| **U-Net** | Semantic | Skip Connections (Concatenate) | Medical MRI/CT scans |
| **Mask R-CNN**| Instance | RoIAlign + Mask Branch | Counting people/cars |
| **DeepLabV3** | Semantic | Atrous Convolutions + ASPP | High-res scenery |

---

## 🎯 ML Applications

| Technique | ML Application |
|-----------|----------------|
| U-Net | Identifying lung nodules in CT scans |
| Mask R-CNN | Counting inventory items on store shelves |
| DeepLab | Virtual backgrounds in video calls (Zoom/Teams) |
| Panoptic Seg. | Autonomous driving road/sidewalk understanding |

---

## ❓ Quick Check Questions

1. What is the difference between Semantic and Instance segmentation?
2. Why are Skip Connections critical in the U-Net architecture?
3. What is a "Transposed Convolution," and why is it used in segmentation?
4. How does "Atrous Convolution" (Dilated Convolution) help in DeepLab?
5. What was the purpose of "RoIAlign" in Mask R-CNN?

---

## 📝 Answers to Quick Check

<details>
<summary>Click to reveal answers</summary>

1. **Semantic segmentation** treats all objects of the same class as a single entity (one mask for all cars). **Instance segmentation** identifies and separates every individual object (different masks for Car A and Car B).
2. As an image goes through the encoder, spatial information is lost due to pooling. **Skip connections** pass the high-resolution features from the encoder directly to the decoder, allowing the model to reconstruct the mask with exact boundary precision.
3. A **Transposed Convolution** (sometimes called Deconvolution) is an operation that increases the spatial dimensions of a feature map. It is used in the decoder path to restore the low-resolution feature map back to the original image size.
4. **Atrous Convolution** allows the model to look at a wider area of the image (larger receptive field) without losing resolution (pooling) and without adding more parameters. This is crucial for capturing the global context of a scene.
5. **RoIAlign** fixed the misalignment issues caused by RoIPool. It uses bilinear interpolation to map the features of a region of interest to the feature map accurately, ensuring the predicted pixel-wise mask aligns perfectly with the original object.

</details>

---

**Status:** ✅ Complete
**Next:** Image Generation (GANs, VAEs, Diffusion Models)
