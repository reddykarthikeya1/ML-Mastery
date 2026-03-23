# 11.1 CNN Architectures & Image Preprocessing

## 🎯 Quick Overview
- **Image Preprocessing**: Normalization, Augmentation, and Color Space conversion
- **Classic CNNs**: AlexNet, VGG, GoogLeNet, and the ResNet breakthrough
- **Modern CNNs**: EfficientNet, MobileNet, and ConvNeXt
- **Architecture Innovations**: Skip connections, Depthwise convolutions, and Scaling laws
- **Foundation for**: Object Detection, Segmentation, and Image Generation

---

## 1. Image Preprocessing & Augmentation

Before feeding images to a CNN, they must be standardized.

### 1.1 Essential Preprocessing
- **Resizing**: All images in a batch must have the same dimensions (e.g., $224 \times 224$).
- **Normalization**: Scaling pixel values from $[0, 255]$ to $[0, 1]$ or standardizing to mean 0 and variance 1.
- **Color Spaces**: Converting from RGB to Grayscale, HSV (helpful for lighting robustness), or LAB.

### 1.2 Data Augmentation
Artificially increasing dataset size to prevent overfitting.
- **Geometric**: Rotation, Flipping, Cropping, Zooming.
- **Photometric**: Brightness, Contrast, Color Jittering.
- **Advanced**: Mixup (blending two images), Cutout (masking regions).

---

## 2. Evolution of CNN Architectures

### 2.1 The Classics
- **AlexNet (2012)**: The model that started the Deep Learning revolution. Used ReLU and Dropout.
- **VGG (2014)**: Showed that deeper is better. Used small $3 \times 3$ filters exclusively.
- **GoogLeNet/Inception**: Introduced "Inception modules" (parallel convolutions of different sizes) to reduce parameters.

### 2.2 The Breakthrough: ResNet (2015)
Introduced **Residual (Skip) Connections**.
- **The Problem**: Very deep networks suffered from accuracy degradation (gradients vanishing/exploding).
- **The Solution**: $y = F(x) + x$. This allows the gradient to flow directly through the "highway," enabling training of 1000+ layer networks.

---

## 3. Modern Efficiency & Scaling

### 3.1 MobileNet
Designed for mobile/edge devices. Uses **Depthwise Separable Convolutions** to drastically reduce compute and parameters.

### 3.2 EfficientNet
Introduced **Compound Scaling**. Instead of just making models deeper, it scales width, depth, and resolution simultaneously based on a fixed ratio.

### 3.3 ConvNeXt
A modern "pure" CNN that incorporates design choices from Vision Transformers (like AdamW, LayerNorm, and larger kernels) to match transformer performance.

---

## 💻 Python Code Examples

### 1. Image Augmentation with Torchvision
```python
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
])
```

### 2. Loading a Pre-trained ResNet (Transfer Learning)
```python
import torch.nn as nn
from torchvision import models

# 1. Load pre-trained model
model = models.resnet50(pretrained=True)

# 2. Freeze all weights
for param in model.parameters():
    param.requires_grad = False

# 3. Replace the final FC layer for your specific task
num_classes = 10
model.fc = nn.Linear(model.fc.in_features, num_classes)

# 4. Only train the new final layer
optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001)
```

---

## 📊 Summary Table

| Architecture | Key Innovation | Best For | Complexity |
|--------------|----------------|----------|------------|
| **VGG-16** | $3 \times 3$ Convolutions | Feature extraction | High |
| **ResNet** | Skip Connections | General purpose SOTA | Medium |
| **MobileNet** | Depthwise Separable | Mobile/IoT | Very Low |
| **EfficientNet** | Compound Scaling | High Accuracy/Speed ratio | Low |
| **Vision Trans.**| Global Self-Attention | Large datasets | Very High |

---

## 🎯 ML Applications

| Technique | ML Application |
|-----------|----------------|
| Transfer Learning | Custom medical image classification |
| Data Augmentation | Training models with small datasets |
| MobileNet | Real-time face filters on smartphones |
| ResNet | Foundation for object detection backbones |

---

## ❓ Quick Check Questions

1. Why are skip connections necessary for training very deep (100+ layer) networks?
2. What is the difference between "Depthwise" and "Pointwise" convolutions in MobileNet?
3. How does Data Augmentation improve model generalization?
4. What does "Compound Scaling" in EfficientNet refer to?
5. Why is a $3 \times 3$ filter size preferred over larger sizes (like $11 \times 11$) in modern CNNs?

---

## 📝 Answers to Quick Check

<details>
<summary>Click to reveal answers</summary>

1. **Skip connections** prevent the **Vanishing Gradient Problem**. By adding the input directly to the output of a block, they provide an "identity shortcut" that allows gradients to flow backwards without being diminished by the non-linear activations and small weights of every layer.
2. **Depthwise convolution** applies a single spatial filter to each input channel independently. **Pointwise convolution** uses a $1 \times 1$ filter to combine these independent channels into a new feature map. Together, they form a **Depthwise Separable Convolution**, which is much cheaper than standard convolution.
3. It creates "synthetic" variations of the training data. This forces the model to learn **invariant features** (e.g., a cat is still a cat even if it's flipped or slightly darker) rather than memorizing specific pixel configurations, thus reducing overfitting.
4. It refers to the systematic scaling of three dimensions: **Network Depth** (number of layers), **Network Width** (number of channels), and **Image Resolution**. EfficientNet uses a coefficient to scale all three in a balanced way.
5. Multiple stacks of small $3 \times 3$ filters have the same "receptive field" as a single large filter but use **fewer parameters** and introduce **more non-linearity** (more ReLU layers), which makes the model more expressive.

</details>

---

**Status:** ✅ Complete
**Next:** Object Detection (YOLO, R-CNN, SSD)
