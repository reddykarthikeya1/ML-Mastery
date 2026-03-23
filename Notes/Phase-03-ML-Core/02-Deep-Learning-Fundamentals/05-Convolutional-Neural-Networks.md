# 8.5 Convolutional Neural Networks

## 🎯 Learning Objectives
After completing this section, you will master:
1. **CNN Fundamentals**: Convolution operation, kernels, feature maps
2. **CNN Layers**: Convolutional, pooling, fully connected layers
3. **Classic Architectures**: LeNet, AlexNet, VGG, ResNet, and more
4. **Modern Architectures**: EfficientNet, MobileNet, Vision Transformers
5. **Transfer Learning**: Fine-tuning and feature extraction

---

## 📚 CNN Fundamentals

### 8.5.1 Why CNNs for Images?

**Problem with Fully Connected Networks:**
```
Image: 224×224×3 = 150,528 input features

FC Layer with 1000 neurons:
150,528 × 1000 = 150 million parameters! ❌

Issues:
- Massive memory requirements
- Overfitting
- Ignores spatial structure
- Not translation invariant
```

**CNN Solution:**
```
Key Ideas:
1. Local connectivity (receptive fields)
2. Weight sharing (same filter across image)
3. Spatial hierarchy (edges → textures → objects)

Result:
- Fewer parameters
- Translation invariance
- Captures spatial patterns
```

### 8.5.2 Convolution Operation

**Definition:** Slide a filter over the input and compute dot products

**2D Convolution:**
```
Input (5×5)          Filter (3×3)           Output (3×3)
┌─────────┐          ┌───────┐             ┌───────┐
│ 1 1 1 0 0 │        │ 1 0 1 │             │ 4 3 2 │
│ 0 1 1 1 0 │   *    │ 0 1 0 │      =      │ 4 5 3 │
│ 0 0 1 1 1 │        │ 1 0 1 │             │ 3 4 4 │
│ 0 0 1 1 0 │        └───────┘             └───────┘
│ 0 1 1 0 0 │
└─────────┘

Calculation for first position:
(1×1) + (1×0) + (1×1) +
(0×0) + (1×1) + (1×0) +
(0×1) + (0×0) + (1×1) = 4
```

**Mathematical Formula:**
$$(I * K)[i, j] = \sum_{m} \sum_{n} I[i+m, j+n] \cdot K[m, n]$$

### Convolution Implementation from Scratch

```python
import numpy as np

def convolve2d(image, kernel, stride=1, padding=0):
    """
    2D convolution operation.
    
    Args:
        image: Input image (H, W) or (H, W, C)
        kernel: Convolution kernel (kH, kW) or (kH, kW, inC, outC)
        stride: Step size for sliding window
        padding: Zero padding around image
    
    Returns:
        Convolved output
    """
    # Add padding
    if padding > 0:
        image = np.pad(image, ((padding, padding), (padding, padding)), 
                      mode='constant', constant_values=0)
    
    image_h, image_w = image.shape[:2]
    kernel_h, kernel_w = kernel.shape[:2]
    
    # Calculate output dimensions
    output_h = (image_h - kernel_h) // stride + 1
    output_w = (image_w - kernel_w) // stride + 1
    
    # Initialize output
    if len(kernel.shape) == 4:
        # Full convolution with multiple channels
        out_channels = kernel.shape[3]
        output = np.zeros((output_h, output_w, out_channels))
    else:
        output = np.zeros((output_h, output_w))
    
    # Perform convolution
    for i in range(0, output_h * stride, stride):
        for j in range(0, output_w * stride, stride):
            region = image[i:i+kernel_h, j:j+kernel_w]
            
            if len(kernel.shape) == 2:
                # Simple 2D convolution
                output[i//stride, j//stride] = np.sum(region * kernel)
            else:
                # Multi-channel convolution
                for c in range(out_channels):
                    if len(region.shape) == 3:
                        # With input channels
                        output[i//stride, j//stride, c] = np.sum(region * kernel[:, :, :, c])
                    else:
                        output[i//stride, j//stride, c] = np.sum(region * kernel[:, :, c])
    
    return output


# Example usage
if __name__ == "__main__":
    # Simple 2D convolution
    image = np.array([
        [1, 1, 1, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 1, 1, 1],
        [0, 0, 1, 1, 0],
        [0, 1, 1, 0, 0]
    ])
    
    kernel = np.array([
        [1, 0, 1],
        [0, 1, 0],
        [1, 0, 1]
    ])
    
    output = convolve2d(image, kernel, stride=1, padding=0)
    print("Convolution Output:")
    print(output)
```

---

## 📚 CNN Layers

### 8.5.3 Convolutional Layer

**Parameters:**
- **Kernel size (k)**: Size of filter (e.g., 3×3, 5×5)
- **Stride (s)**: Step size for sliding
- **Padding (p)**: Zero padding around border
- **Number of filters**: Output channels

**Output Size Formula:**
$$\text{Output} = \left\lfloor\frac{W - k + 2p}{s}\right\rfloor + 1$$

**Example:**
```
Input: 32×32×3
Kernel: 5×5
Stride: 1
Padding: 0
Filters: 6

Output: (32 - 5 + 0) / 1 + 1 = 28
Output shape: 28×28×6
```

### Convolutional Layer Implementation

```python
class Conv2D:
    """
    2D Convolutional layer.
    """
    
    def __init__(self, in_channels, out_channels, kernel_size, 
                 stride=1, padding=0):
        """
        Initialize Conv2D layer.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels (filters)
            kernel_size: Size of convolution kernel
            stride: Stride length
            padding: Zero padding
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # Initialize weights (He initialization)
        self.weights = np.random.randn(
            out_channels, in_channels, kernel_size, kernel_size
        ) * np.sqrt(2.0 / (in_channels * kernel_size * kernel_size))
        
        self.biases = np.zeros(out_channels)
        
        # Cache for backward pass
        self.cache = None
    
    def forward(self, X):
        """
        Forward pass.
        
        Args:
            X: Input (N, C_in, H, W)
        
        Returns:
            Output (N, C_out, H_out, W_out)
        """
        N, C_in, H, W = X.shape
        
        # Add padding
        if self.padding > 0:
            X_padded = np.pad(X, ((0, 0), (0, 0), 
                                  (self.padding, self.padding),
                                  (self.padding, self.padding)),
                             mode='constant')
        else:
            X_padded = X
        
        # Calculate output dimensions
        H_out = (H + 2 * self.padding - self.kernel_size) // self.stride + 1
        W_out = (W + 2 * self.padding - self.kernel_size) // self.stride + 1
        
        # Initialize output
        out = np.zeros((N, self.out_channels, H_out, W_out))
        
        # Perform convolution
        for n in range(N):
            for c_out in range(self.out_channels):
                for i in range(H_out):
                    for j in range(W_out):
                        # Get region of interest
                        i_start = i * self.stride
                        i_end = i_start + self.kernel_size
                        j_start = j * self.stride
                        j_end = j_start + self.kernel_size
                        
                        region = X_padded[n, :, i_start:i_end, j_start:j_end]
                        
                        # Convolve
                        out[n, c_out, i, j] = np.sum(region * self.weights[c_out]) + self.biases[c_out]
        
        # Cache for backward
        self.cache = (X, X_padded, out)
        
        return out
    
    def backward(self, dout):
        """
        Backward pass.
        
        Args:
            dout: Upstream gradient (N, C_out, H_out, W_out)
        
        Returns:
            dX: Gradient w.r.t. input
        """
        X, X_padded, _ = self.cache
        N, C_in, H, W = X.shape
        
        # Initialize gradients
        dX = np.zeros_like(X_padded)
        dW = np.zeros_like(self.weights)
        db = np.zeros_like(self.biases)
        
        N, C_out, H_out, W_out = dout.shape
        
        # Compute gradients
        for n in range(N):
            for c_out in range(C_out):
                for i in range(H_out):
                    for j in range(W_out):
                        i_start = i * self.stride
                        i_end = i_start + self.kernel_size
                        j_start = j * self.stride
                        j_end = j_start + self.kernel_size
                        
                        region = X_padded[n, :, i_start:i_end, j_start:j_end]
                        
                        # Gradient w.r.t. weights
                        dW[c_out] += region * dout[n, c_out, i, j]
                        
                        # Gradient w.r.t. bias
                        db[c_out] += dout[n, c_out, i, j]
                        
                        # Gradient w.r.t. input
                        dX[n, :, i_start:i_end, j_start:j_end] += \
                            self.weights[c_out] * dout[n, c_out, i, j]
        
        # Remove padding gradient
        if self.padding > 0:
            dX = dX[:, :, self.padding:-self.padding, self.padding:-self.padding]
        
        return dX, dW, db
```

### 8.5.4 Pooling Layers

**Purpose:** Downsample spatial dimensions

**Types:**

**1. Max Pooling**
```
Input (4×4)          Max Pool (2×2, stride=2)     Output (2×2)
┌───────────┐        ┌─────┬─────┐               ┌───────┐
│ 1  3  2  4 │       │1 3│2 4│                  │ 3  4  │
│ 5  6  1  2 │       │5 6│1 2│       →          │ 6  4  │
├───────────┤       ├─────┼─────┤               └───────┘
│ 1  0  3  4 │       │1 0│3 4│
│ 0  2  4  1 │       │0 2│4 1│
└───────────┘        └─────┴─────┘
```

**2. Average Pooling**
```
Same as max pooling but takes average instead of maximum
```

**Pooling Implementation:**
```python
class MaxPool2D:
    """Max Pooling layer"""
    
    def __init__(self, pool_size=2, stride=2):
        self.pool_size = pool_size
        self.stride = stride
        self.cache = None
    
    def forward(self, X):
        """
        Forward pass.
        
        Args:
            X: Input (N, C, H, W)
        """
        N, C, H, W = X.shape
        
        H_out = (H - self.pool_size) // self.stride + 1
        W_out = (W - self.pool_size) // self.stride + 1
        
        out = np.zeros((N, C, H_out, W_out))
        
        for n in range(N):
            for c in range(C):
                for i in range(H_out):
                    for j in range(W_out):
                        i_start = i * self.stride
                        i_end = i_start + self.pool_size
                        j_start = j * self.stride
                        j_end = j_start + self.pool_size
                        
                        region = X[n, c, i_start:i_end, j_start:j_end]
                        out[n, c, i, j] = np.max(region)
        
        self.cache = (X, out)
        return out
    
    def backward(self, dout):
        """Backward pass - route gradient through max values"""
        X, out = self.cache
        N, C, H, W = X.shape
        
        dX = np.zeros_like(X)
        
        _, _, H_out, W_out = dout.shape
        
        for n in range(N):
            for c in range(C):
                for i in range(H_out):
                    for j in range(W_out):
                        i_start = i * self.stride
                        i_end = i_start + self.pool_size
                        j_start = j * self.stride
                        j_end = j_start + self.pool_size
                        
                        region = X[n, c, i_start:i_end, j_start:j_end]
                        mask = region == np.max(region)
                        
                        dX[n, c, i_start:i_end, j_start:j_end] += \
                            mask * dout[n, c, i, j]
        
        return dX


class AveragePool2D:
    """Average Pooling layer"""
    
    def __init__(self, pool_size=2, stride=2):
        self.pool_size = pool_size
        self.stride = stride
    
    def forward(self, X):
        N, C, H, W = X.shape
        
        H_out = (H - self.pool_size) // self.stride + 1
        W_out = (W - self.pool_size) // self.stride + 1
        
        out = np.zeros((N, C, H_out, W_out))
        
        for n in range(N):
            for c in range(C):
                for i in range(H_out):
                    for j in range(W_out):
                        i_start = i * self.stride
                        i_end = i_start + self.pool_size
                        j_start = j * self.stride
                        j_end = j_start + self.pool_size
                        
                        region = X[n, c, i_start:i_end, j_start:j_end]
                        out[n, c, i, j] = np.mean(region)
        
        return out
    
    def backward(self, dout):
        """Distribute gradient equally"""
        X = self.cache
        N, C, H, W = X.shape
        
        dX = np.zeros_like(X)
        _, _, H_out, W_out = dout.shape
        
        pool_size = self.pool_size * self.stride
        
        for n in range(N):
            for c in range(C):
                for i in range(H_out):
                    for j in range(W_out):
                        i_start = i * self.stride
                        j_start = j * self.stride
                        
                        dX[n, c, i_start:i_start+self.pool_size,
                          j_start:j_start+self.pool_size] += \
                            dout[n, c, i, j] / (self.pool_size ** 2)
        
        return dX
```

---

## 📚 CNN Architectures

### 8.5.5 Classic Architectures

**1. LeNet-5 (1998)**
```
Architecture:
Input (32×32×1)
  ↓
Conv1 (6 filters, 5×5) → Pool (2×2)
  ↓
Conv2 (16 filters, 5×5) → Pool (2×2)
  ↓
FC1 (120 neurons)
  ↓
FC2 (84 neurons)
  ↓
Output (10 classes)

Use: Digit recognition (MNIST)
Parameters: ~60K
```

**2. AlexNet (2012)**
```
Architecture:
Input (227×227×3)
  ↓
Conv1 (96 filters, 11×11, stride=4) → ReLU → MaxPool → LRN
  ↓
Conv2 (256 filters, 5×5) → ReLU → MaxPool → LRN
  ↓
Conv3 (384 filters, 3×3) → ReLU
  ↓
Conv4 (384 filters, 3×3) → ReLU
  ↓
Conv5 (256 filters, 3×3) → ReLU → MaxPool
  ↓
FC1 (4096) → ReLU → Dropout
  ↓
FC2 (4096) → ReLU → Dropout
  ↓
Output (1000 classes)

Innovations:
- ReLU activation
- Dropout
- Data augmentation
- GPU training
Parameters: ~60M
```

**3. VGG (2014)**
```
Key Insight: Use small 3×3 filters consistently

VGG16 Architecture:
Input (224×224×3)
  ↓
[Conv3-64] × 2 → MaxPool
  ↓
[Conv3-128] × 2 → MaxPool
  ↓
[Conv3-256] × 3 → MaxPool
  ↓
[Conv3-512] × 3 → MaxPool
  ↓
[Conv3-512] × 3 → MaxPool
  ↓
FC (4096) → FC (4096) → Output (1000)

Characteristics:
- Uniform architecture
- Deep network (16-19 layers)
- Small receptive fields
Parameters: ~138M (VGG16)
```

### VGG Implementation

```python
class VGGBlock:
    """VGG block with multiple convolutions"""
    
    def __init__(self, in_channels, out_channels, num_convs=2):
        self.convs = []
        for i in range(num_convs):
            in_c = in_channels if i == 0 else out_channels
            self.convs.append(Conv2D(in_c, out_channels, kernel_size=3, padding=1))
        self.pool = MaxPool2D(pool_size=2, stride=2)
        self.num_convs = num_convs
    
    def forward(self, X):
        for conv in self.convs:
            X = conv.forward(X)
            X = relu(X)
        X = self.pool.forward(X)
        return X


class VGG16:
    """VGG16 Network"""
    
    def __init__(self, num_classes=1000):
        self.layers = []
        
        # Conv blocks
        self.layers.append(VGGBlock(3, 64, num_convs=2))
        self.layers.append(VGGBlock(64, 128, num_convs=2))
        self.layers.append(VGGBlock(128, 256, num_convs=3))
        self.layers.append(VGGBlock(256, 512, num_convs=3))
        self.layers.append(VGGBlock(512, 512, num_convs=3))
        
        # Fully connected layers would go here
        # (simplified for brevity)
    
    def forward(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        return X
```

**4. ResNet (2015)**
```
Key Innovation: Residual/Skip Connections

Residual Block:
Input
  ↓
  ┌─────────────┐
  ↓             ↓
Conv1          (+) → ReLU → Output
  ↓             ↑
Conv2 ──────────┘

Mathematically:
Output = F(x) + x  (instead of just F(x))

Benefits:
- Solves vanishing gradient
- Enables very deep networks
- Easier optimization

ResNet Variants:
- ResNet18: 18 layers
- ResNet34: 34 layers
- ResNet50: 50 layers (with bottleneck)
- ResNet101: 101 layers
- ResNet152: 152 layers
```

### Residual Block Implementation

```python
class ResidualBlock:
    """Residual block for ResNet"""
    
    def __init__(self, channels, stride=1, use_projection=False):
        """
        Args:
            channels: Number of output channels
            stride: Stride for first convolution
            use_projection: Use 1×1 conv for skip connection
        """
        self.use_projection = use_projection
        
        # Main path
        self.conv1 = Conv2D(channels, channels, kernel_size=3, 
                           stride=stride, padding=1)
        self.conv2 = Conv2D(channels, channels, kernel_size=3, 
                           stride=1, padding=1)
        
        # Projection shortcut
        if use_projection:
            self.proj_conv = Conv2D(channels, channels, kernel_size=1, 
                                   stride=stride)
    
    def forward(self, X):
        identity = X
        
        # Main path
        out = relu(self.conv1.forward(X))
        out = self.conv2.forward(out)
        
        # Skip connection
        if self.use_projection:
            identity = self.proj_conv.forward(X)
        
        # Add and ReLU
        out = out + identity
        out = relu(out)
        
        return out


class BottleneckBlock:
    """Bottleneck block for deeper ResNets"""
    
    def __init__(self, channels, stride=1, use_projection=False):
        """
        Bottleneck: 1×1 → 3×3 → 1×1
        Reduces computation while maintaining representational power
        """
        self.use_projection = use_projection
        
        # Bottleneck structure
        self.conv1 = Conv2D(channels, channels, kernel_size=1)  # Reduce
        self.conv2 = Conv2D(channels, channels, kernel_size=3,
                           stride=stride, padding=1)  # Process
        self.conv3 = Conv2D(channels, channels * 4, kernel_size=1)  # Expand
        
        if use_projection:
            self.proj_conv = Conv2D(channels, channels * 4, kernel_size=1,
                                   stride=stride)
    
    def forward(self, X):
        identity = X
        
        out = relu(self.conv1.forward(X))
        out = relu(self.conv2.forward(out))
        out = self.conv3.forward(out)
        
        if self.use_projection:
            identity = self.proj_conv.forward(X)
        
        out = out + identity
        out = relu(out)
        
        return out
```

### 8.5.6 Modern Architectures

**1. GoogLeNet / Inception**
```
Inception Module:
Input
  ↓
  ├─→ 1×1 Conv ────────────┐
  ├─→ 3×3 Conv (after 1×1) ─┤
  ├─→ 5×5 Conv (after 1×1) ─┼→ Concatenate
  ├─→ Pool → 1×1 Conv ─────┘
  ↓
Output

Key Ideas:
- Multi-scale processing
- 1×1 convolutions for dimensionality reduction
- Auxiliary classifiers
```

**2. MobileNet**
```
Depthwise Separable Convolution:

Standard Conv:        Depthwise Separable:
Input → Conv → Output  Input → DW Conv → PW Conv → Output
                         (spatial)   (1×1)

Computation Reduction:
Standard: D_K × D_K × M × N
Separable: D_K × D_K × M + M × N

~8-9x fewer parameters!
```

**3. EfficientNet**
```
Compound Scaling:
- Scale depth, width, and resolution together
- Use neural architecture search

EfficientNet-B0 to B7:
- B0: Baseline (5M params)
- B7: Scaled up (66M params)

Better accuracy-efficiency tradeoff
```

---

## 📊 Summary Tables

### CNN Layer Types

| Layer | Purpose | Parameters | Output |
|-------|---------|------------|--------|
| **Conv2D** | Feature extraction | k×k×C_in×C_out | Feature maps |
| **MaxPool** | Downsampling | None | Reduced H, W |
| **AvgPool** | Downsampling | None | Reduced H, W |
| **GlobalAvgPool** | Spatial pooling | None | 1×1×C |
| **FC** | Classification | N_in×N_out | Class scores |

### Architecture Comparison

| Architecture | Depth | Params | Top-1 Acc | Key Innovation |
|--------------|-------|--------|-----------|----------------|
| **LeNet-5** | 7 | 60K | - | First CNN |
| **AlexNet** | 8 | 60M | 57.1% | ReLU, Dropout |
| **VGG16** | 16 | 138M | 69.0% | Uniform design |
| **ResNet50** | 50 | 25M | 76.0% | Skip connections |
| **EfficientNet** | Varies | 5-66M | 84.4% | Compound scaling |

### Receptive Field Calculation

| Layer | Kernel | Stride | Receptive Field |
|-------|--------|--------|-----------------|
| Conv1 | 3×3 | 1 | 3 |
| Conv2 | 3×3 | 1 | 5 |
| Conv3 | 3×3 | 1 | 7 |
| After Pool | 2×2 | 2 | 14 |

---

## 📝 Practice Problems

### Level 1: Basic
1. Calculate output size for given conv parameters
2. Explain why CNNs are better than FC for images
3. Draw max pooling operation on 4×4 input
4. Count parameters in a conv layer
5. Compare valid vs same padding

### Level 2: Intermediate
1. Implement 2D convolution from scratch
2. Build ResNet block with skip connections
3. Calculate receptive field after multiple layers
4. Implement depthwise separable convolution
5. Compare VGG vs ResNet architectures

### Level 3: Advanced
1. Implement full ResNet50 from scratch
2. Build custom Inception module
3. Implement gradient checkpointing for memory efficiency
4. Design efficient architecture for mobile deployment
5. Research and implement latest CNN variants

---

**Status:** ✅ Complete  
**Next:** [[06-Recurrent-Neural-Networks]]
