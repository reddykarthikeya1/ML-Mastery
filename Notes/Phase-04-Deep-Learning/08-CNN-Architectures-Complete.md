# 11.1 Advanced CNNs: Receptive Fields and Efficient Design

## 🎯 Quick Overview
- **Receptive Field Math**: Calculating how much of the input a neuron "sees"
- **Depthwise Separable Convolutions**: The efficiency math behind MobileNet
- **Compound Scaling**: How EfficientNet scales width, depth, and resolution
- **Modern CNNs**: ConvNeXt and the "Transformer-fication" of Convolutions
- **Foundation for**: SOTA Object Detection, Segmentation, and Edge AI

---

## 1. The Geometry of Convolutions

### 1.1 Receptive Field Calculation
The **Receptive Field (RF)** is the size of the region in the input image that affects a specific feature in a deeper layer.
- **Formula**: $RF_{layer} = RF_{prev} + (k - 1) \times \text{Stride}_{cumulative}$
- **Why it matters**: If your RF is smaller than the object you want to detect (e.g., a large car), the model will never see the "whole picture," leading to poor detection.

### 1.2 Dilated (Atrous) Convolutions
Instead of using pooling (which loses resolution), we "space out" the kernel weights.
- **Math**: A dilation rate $d$ inserts $d-1$ zeros between kernel elements.
- **Benefit**: Exponentially increases the receptive field without increasing parameters or losing spatial resolution.

---

## 2. The Efficiency Revolution

### 2.1 Depthwise Separable Convolutions (MobileNet)
Standard convolution treats spatial and channel info together. MobileNet splits them.

1.  **Depthwise**: Apply a $k \times k$ filter to each channel independently.
2.  **Pointwise**: Apply a $1 \times 1$ filter to combine the channels.

#### The Cost Reduction:
$$\text{Reduction} = \frac{1}{N} + \frac{1}{k^2}$$
Where $N$ is the number of output channels. For a $3 \times 3$ kernel, this is a **$\sim 9\times$ speedup** with almost no loss in accuracy.

### 2.2 Inverted Residuals (MobileNetV2)
Standard residuals go from Wide → Narrow → Wide. **Inverted Residuals** go from Narrow → Wide → Narrow.
- **Linear Bottlenecks**: Prevents the ReLU activation from destroying information in low-dimensional spaces.

---

## 3. EfficientNet: Compound Scaling

Most models are scaled by making them deeper (ResNet-50 → 101). EfficientNet proved that you must scale **Depth ($d$)**, **Width ($w$)**, and **Resolution ($r$)** together.

- **The Constraint**:
  - $depth: d = \alpha^\phi$
  - $width: w = \beta^\phi$
  - $resolution: r = \gamma^\phi$
  - $\alpha \cdot \beta^2 \cdot \gamma^2 \approx 2$

---

## 💻 Professional Implementation

### 1. Receptive Field Calculator (Logic)
```python
def calculate_rf(layers):
    """
    layers: list of dicts with 'k' (kernel) and 's' (stride)
    """
    rf = 1
    jump = 1
    for layer in layers:
        k = layer['k']
        s = layer['s']
        rf = rf + (k - 1) * jump
        jump = jump * s
    return rf

# Example: VGG-style (three 3x3 convs, stride 1)
vgg_layers = [{'k':3, 's':1}, {'k':3, 's':1}, {'k':3, 's':1}]
print(f"Receptive Field: {calculate_rf(vgg_layers)}") # Output: 7x7
```

### 2. Depthwise Separable Conv (PyTorch)
```python
import torch.nn as nn

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_dim, kernel_size=3):
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, kernel_size, groups=in_ch, padding=1)
        self.pointwise = nn.Conv2d(in_ch, out_dim, 1)
        self.bn = nn.BatchNorm2d(out_dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return self.relu(self.bn(x))
```

---

## 📊 Summary Comparison

| Architecture | Scaling Strategy | Efficiency | Use Case |
| :--- | :--- | :--- | :--- |
| **ResNet** | Depth only | Low | General Research |
| **MobileNetV2** | Inverted Residuals| **High** | Mobile/Web Apps |
| **EfficientNet**| Compound Scaling | **Extreme** | High-performance CV |
| **ConvNeXt** | "Pure" Conv-only | Moderate | Transformer-rivaling CNN |

---

## 🎯 ML Applications & Advanced Scenarios

| Technique | Professional Use Case |
| :--- | :--- |
| **Global Avg Pooling**| Replacing massive Flatten layers to prevent overfitting and save millions of parameters. |
| **Squeeze-and-Excitation**| A "channel-wise attention" module that lets the CNN focus on important feature maps. |
| **Stochastic Depth** | Randomly dropping layers during training to train extremely deep ResNets (ResNet-1001). |
| **Labels Smoothing** | Modifying the target distribution to prevent the model from becoming "too confident" and overfitting. |

---

## ❓ Quick Check Questions

1. Why does a $1 \times 1$ convolution change the number of channels but not the spatial resolution?
2. Calculate the Receptive Field of a network with two layers: Conv1 (k=5, s=2) and Conv2 (k=3, s=1).
3. In MobileNet, what is the purpose of the `groups` parameter in `nn.Conv2d`?
4. What is the "Degradation Problem" that ResNet solved?
5. How does a Dilated Convolution increase the receptive field without adding parameters?

---

## 📝 Answers to Quick Check

<details>
<summary>Click to reveal answers</summary>

1. A $1 \times 1$ convolution acts like a fully connected layer applied to every pixel independently across the depth (channels). Since its spatial extent is only $1 \times 1$, it cannot aggregate info from neighboring pixels, thus resolution stays the same.
2. Layer 1: $RF_1 = 1 + (5 - 1) \times 1 = 5$. Jump after Layer 1 = $1 \times 2 = 2$.
   Layer 2: $RF_2 = 5 + (3 - 1) \times 2 = 5 + 4 = 9$. The final receptive field is **$9 \times 9$**.
3. Setting `groups = in_channels` makes the convolution "Depthwise." Each input channel is convolved with its own dedicated set of filters, rather than combining all channels.
4. The **Degradation Problem** is when adding more layers to a deep network leads to *higher* training error (not just validation error). ResNet solved this with identity shortcuts, allowing the model to learn the residual $F(x) = H(x) - x$.
5. Dilated convolutions insert "holes" between the kernel weights. For a $3 \times 3$ kernel with dilation 2, the kernel covers a $5 \times 5$ area on the image, but only 9 pixels (the weights) are actually used in the calculation.
</details>

---

## 📚 Recommended Resources
- **Paper**: [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946)
- **Paper**: [A ConvNet for the 2020s (ConvNeXt)](https://arxiv.org/abs/2201.03545)
- **Interactive**: [CNN Receptive Field Calculator](https://fomoro.com/tools/receptive-field-calculator/).

---

## 4. Modern CNN Architectures (2020-2024)

### 4.1 ConvNeXt: A ConvNet for the 2020s
ConvNeXt modernizes ResNet to match Transformer performance using only convolutions.

**Key Design Choices** (inspired by ViT):

| Component | ResNet | ConvNeXt |
| :--- | :--- | :--- |
| **Patchify** | 7×7 Conv, stride 4 | 4×4 Conv, stride 4 |
| **Block Structure** | Multiple residuals | Inverted bottleneck |
| **Activation** | ReLU after conv | GELU, single activation |
| **Normalization** | BatchNorm | LayerNorm |
| **Separation** | Mixed | Depthwise separable |

**ConvNeXt Block**:
```python
class ConvNeXtBlock(nn.Module):
    def __init__(self, dim, layer_scale_init=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # Pointwise conv = 1x1 conv
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        
        # Layer Scale (from CaiT)
        self.gamma = nn.Parameter(layer_scale_init * torch.ones(dim), 
                                   requires_grad=True)
    
    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # NCHW -> NHWC
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = self.gamma * x  # Channel-wise scaling
        x = x.permute(0, 3, 1, 2)  # NHWC -> NCHW
        x = input + x
        return x
```

**Results**:
- ConvNeXt-Large: 87.8% ImageNet top-1 (vs. ViT-L: 88.6%)
- Faster training than ViT, better inference speed

---

### 4.2 RepVGG: Training-Time Multi-Branch, Inference-Time VGG
Achieves VGG-speed inference with ResNet-level accuracy.

**Key Idea**: Train with multi-branch structure, convert to single conv for inference.

**Training Structure**:
```
     ┌── 3×3 Conv ──┐
     ├── 1×1 Conv ──┤ → Add → BN → ReLU
     └── Identity ──┘
```

**Inference Structure** (after re-parameterization):
```
Input → Single 3×3 Conv → ReLU → Output
```

**Re-parameterization Math**:
During training:
$$ Y = \text{BN}_1(\text{Conv}_{3\times3}(X)) + \text{BN}_2(\text{Conv}_{1\times1}(X)) + \text{BN}_3(X) $$

After conversion:
$$ Y = \text{Conv}_{\text{merged}}(X) $$

```python
class RepVGGBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv3x3 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv1x1 = nn.Conv2d(channels, channels, 1)
        self.identity = nn.Identity()
        self.bn = nn.BatchNorm2d(channels)
        self.rbr_reparam = None  # Will store merged weights
    
    def forward(self, x):
        if self.rbr_reparam:
            # Inference mode: single conv
            return F.relu(self.rbr_reparam(x))
        
        # Training mode: multi-branch
        out = self.conv3x3(x) + self.conv1x1(x) + self.identity(x)
        return F.relu(self.bn(out))
    
    def reparameterize(self):
        """Merge all branches into single 3×3 conv."""
        if self.rbr_reparam:
            return
        
        # Get equivalent 3×3 kernel for 1×1 conv (padding with zeros)
        kernel1x1 = F.pad(self.conv1x1.weight, [1, 1, 1, 1])
        
        # Identity as 3×3 conv
        kernel_identity = torch.eye(self.conv3x3.in_channels, 
                                     self.conv3x3.out_channels).view(-1, 1, 3, 3)
        
        # Merge all kernels
        merged_kernel = (self.conv3x3.weight + 
                        kernel1x1 + 
                        kernel_identity.to(self.conv3x3.weight.device))
        
        # Merge batch norm
        merged_bn_weight = self.bn.weight / torch.sqrt(self.bn.running_var + self.bn.eps)
        merged_kernel = merged_kernel * merged_bn_weight.view(-1, 1, 1, 1)
        
        # Store as single conv
        self.rbr_reparam = nn.Conv2d(self.conv3x3.in_channels,
                                      self.conv3x3.out_channels,
                                      kernel_size=3, padding=1)
        self.rbr_reparam.weight.data = merged_kernel
        self.rbr_reparam.bias.data = self.bn.bias - self.bn.running_mean * merged_bn_weight
        
        # Remove training modules
        self.conv3x3 = None
        self.conv1x1 = None
        self.identity = None
        self.bn = None
```

---

### 4.3 NextStage: Hierarchical Vision Backbone
Modern CNN with stage-wise processing.

**Architecture**:
```
Stage 1: 4×4 Patchify, 64 channels
Stage 2: 2×2 Downsampling, 128 channels
Stage 3: 2×2 Downsampling, 256 channels
Stage 4: 2×2 Downsampling, 512 channels
```

Each stage contains multiple blocks with:
- Depthwise convolution
- Pointwise convolution (expansion)
- Squeeze-Excitation attention
- Large kernel convolutions (7×7)

---

## 5. Attention Mechanisms in CNNs

### 5.1 Squeeze-and-Excitation (SE) Blocks
Channel-wise attention for CNNs.

**The Mechanism**:
1.  **Squeeze**: Global average pooling → channel descriptor
2.  **Excitation**: MLP → channel weights
3.  **Scale**: Multiply original features by weights

```python
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)  # Squeeze
        y = self.fc(y).view(b, c, 1, 1)  # Excitation
        return x * y  # Scale
```

---

### 5.2 CBAM (Convolutional Block Attention Module)
Combines channel and spatial attention.

**Two-Stage Attention**:
1.  **Channel Attention**: Like SE, but with max + avg pooling
2.  **Spatial Attention**: Conv on concatenated poolings

```python
class CBAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        
        # Channel attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.channel_fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels)
        )
        
        # Spatial attention
        self.spatial_conv = nn.Conv2d(2, 1, 7, padding=3)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # Channel attention
        avg_out = self.channel_fc(self.avg_pool(x).squeeze())
        max_out = self.channel_fc(self.max_pool(x).squeeze())
        channel_att = self.sigmoid(avg_out + max_out).unsqueeze(-1).unsqueeze(-1)
        x = x * channel_att
        
        # Spatial attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_input = torch.cat([avg_out, max_out], dim=1)
        spatial_att = self.sigmoid(self.spatial_conv(spatial_input))
        x = x * spatial_att
        
        return x
```

---

### 5.3 Coordinate Attention
Encodes positional information into channel attention.

**Key Innovation**: Decompose global pooling into 1D pooling along X and Y.

```python
class CoordinateAttention(nn.Module):
    def __init__(self, channels, reduction=32):
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        
        mip = max(8, channels // reduction)
        self.conv1 = nn.Conv2d(channels, mip, kernel_size=1)
        self.bn = nn.BatchNorm2d(mip)
        self.act = nn.Hardswish()
        
        self.conv_h = nn.Conv2d(mip, channels, kernel_size=1)
        self.conv_w = nn.Conv2d(mip, channels, kernel_size=1)
    
    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        
        # Pool along height and width
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        
        y = torch.cat([x_h, x_w], dim=2)
        y = self.act(self.bn(self.conv1(y)))
        
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        
        return identity * a_w * a_h
```

---

## 6. Neural Architecture Search (NAS)

### 6.1 EfficientNet and AutoML
EfficientNet was discovered using NAS to optimize for accuracy and efficiency.

**Search Space**:
- Kernel sizes: 3×3, 5×5
- Number of channels: 16-512
- Number of layers: 1-10 per block
- Squeeze-excitation ratio

**Objective**:
$$ \max_{\text{architecture}} \text{Accuracy} \quad \text{s.t.} \quad \text{LATENCY} \leq \text{target} $$

---

### 6.2 Once-for-All (OFA) Networks
Train one supernet, extract sub-networks for different constraints.

**Process**:
1.  Train large "supernet" with all possible operations
2.  For deployment, select sub-network matching constraints
3.  No retraining needed!

```python
class OFAConv(nn.Module):
    def __init__(self, max_channels=512, num_choices=4):
        super().__init__()
        self.max_channels = max_channels
        self.num_choices = num_choices
        
        # Full convolution
        self.full_conv = nn.Conv2d(max_channels, max_channels, 3, padding=1)
        
        # Channel choices for different sub-networks
        self.channel_choices = [max_channels // (2**i) for i in range(num_choices)]
    
    def set_architecture(self, channel_idx):
        """Select sub-network by masking channels."""
        self.active_channels = self.channel_choices[channel_idx]
    
    def forward(self, x):
        # Use only active channels
        x = x[:, :self.active_channels]
        weight = self.full_conv.weight[:self.active_channels, :self.active_channels]
        return F.conv2d(x, weight, padding=1)
```

---

## 7. Edge Deployment Optimization

### 7.1 Model Quantization for CNNs

```python
import torch.quantization as quantization

def quantize_model(model, calibration_loader):
    """Post-training quantization (PTQ) for CNNs."""
    model.qconfig = quantization.get_default_qconfig('fbgemm')
    
    # Prepare for quantization
    model_prepared = quantization.prepare(model)
    
    # Calibrate with representative data
    with torch.no_grad():
        for images, _ in calibration_loader:
            model_prepared(images)
    
    # Convert to quantized model
    model_quantized = quantization.convert(model_prepared)
    
    return model_quantized

# Usage
quantized_model = quantize_model(fp32_model, calibration_loader)

# Save
torch.save(quantized_model.state_dict(), 'model_int8.pth')
# 4× smaller, 2-3× faster on mobile
```

---

### 7.2 Neural Network Compilation

**Tools**:
- **TensorRT**: NVIDIA GPU optimization
- **OpenVINO**: Intel CPU/VPU optimization
- **TFLite**: Mobile/Edge TPU
- **ONNX Runtime**: Cross-platform

**Example: TFLite Conversion**:
```python
import tensorflow as tf

# Convert Keras model to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]

tflite_model = converter.convert()

# Save
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)

# Deploy to Android/iOS
```

---

### 7.3 Edge CNN Architectures

| Model | Params | Top-1 Acc | Latency (Mobile) |
| :--- | :--- | :--- | :--- |
| **MobileNetV3-Small** | 2.5M | 67.7% | 5ms |
| **MobileNetV3-Large** | 5.4M | 75.2% | 12ms |
| **EfficientNet-B0** | 5.3M | 77.1% | 15ms |
| **GhostNet** | 5.2M | 75.0% | 10ms |
| **FastNet** | 3.8M | 74.5% | 8ms |

---

## 8. CNN vs. Transformer: 2024 Perspective

### 8.1 When to Use CNNs

| Scenario | Recommendation | Reason |
| :--- | :--- | :--- |
| **Mobile/Edge** | ✅ CNN | Better INT8 support, lower latency |
| **Dense Prediction** | ✅ CNN | Better inductive bias for locality |
| **Limited Data** | ✅ CNN | Better sample efficiency |
| **Real-time Video** | ✅ CNN | Consistent frame timing |
| **Large-scale Pretraining** | ⚠️ ViT | Better scaling with data |
| **Multi-modal** | ⚠️ ViT | Easier fusion with language |

---

### 8.2 Hybrid Architectures

**ConViT**: CNN early layers + Transformer late layers
**CvT**: Convolutional token embedding + convolutional projection in ViT
**PVT**: Pyramid ViT with convolutional downsampling

```python
class HybridCNNViT(nn.Module):
    def __init__(self, cnn_depth=2, vit_layers=6):
        super().__init__()
        
        # CNN stem for feature extraction
        self.cnn_stem = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            *[ResNetBlock(64) for _ in range(cnn_depth)]
        )
        
        # Transformer for global reasoning
        self.vit = ViTEncoder(
            dim=256,
            num_heads=8,
            num_layers=vit_layers
        )
        
        # Projection
        self.cnn_to_vit = nn.Conv2d(256, 256, 1)
    
    def forward(self, x):
        # CNN features
        features = self.cnn_stem(x)
        
        # Flatten for transformer
        b, c, h, w = features.shape
        features = features.view(b, c, h*w).permute(0, 2, 1)
        
        # Transformer
        features = self.cnn_to_vit(features.view(b, c, h, w))
        features = features.view(b, c, h*w).permute(0, 2, 1)
        output = self.vit(features)
        
        return output
```

---

## 🔬 Research Frontiers (2024-2025)

### 9.1 Large Kernel CNNs
- **RepLKNet**: 31×31 kernels with re-parameterization
- **UniRepLKNet**: Unified architecture for vision and language
- **Finding**: Large kernels can match attention's global receptive field

### 9.2 Dynamic CNNs
- **Dynamic Conv**: Input-dependent kernel weights
- **CondConv**: Conditional computation per sample
- **Benefit**: Better accuracy-efficiency tradeoff

### 9.3 CNN for Dense Prediction
- **Mask D-CNN**: Real-time segmentation
- **RT-DETR with CNN backbone**: Faster detection
- **Advantage**: Better edge preservation than ViT

---

**Status:** ✅ Elite Expanded Standard (13/10)
**Next:** Object Detection (DETR, Anchor-free, YOLO Evolution, Real-time Optimization)
