# 9.2 PyTorch

## 🎯 Learning Objectives
After completing this section, you will master:
1. **PyTorch Basics**: Tensors, autograd, computational graphs
2. **Neural Networks**: nn.Module, built-in layers, custom models
3. **Training**: Loss functions, optimizers, training loops
4. **Data Loading**: Dataset, DataLoader, transforms
5. **Advanced PyTorch**: torchscript, distributed training, profiling

---

## 📚 PyTorch Fundamentals

### 9.2.1 Tensors

**Definition:** Multi-dimensional arrays with GPU support

```python
import torch
import numpy as np

# Creating tensors
# Scalar
scalar = torch.tensor(5)

# Vector
vector = torch.tensor([1, 2, 3, 4, 5])

# Matrix
matrix = torch.tensor([[1, 2, 3],
                       [4, 5, 6]])

# 3-D tensor
tensor_3d = torch.tensor([[[1, 2], [3, 4]],
                          [[5, 6], [7, 8]]])

# Tensor properties
print(f"Shape: {tensor_3d.shape}")  # torch.Size([2, 2, 2])
print(f"Dimensions: {tensor_3d.ndim}")
print(f"Dtype: {tensor_3d.dtype}")  # torch.int64
print(f"Device: {tensor_3d.device}")  # cpu

# Special tensors
zeros = torch.zeros(3, 3)
ones = torch.ones(3, 3)
full = torch.full((3, 3), 7)
eye = torch.eye(3)  # Identity
arange = torch.arange(0, 10, 2)  # [0, 2, 4, 6, 8]
linspace = torch.linspace(0, 1, 5)  # [0, 0.25, 0.5, 0.75, 1]

# Random tensors
random = torch.rand(3, 3)  # Uniform [0, 1)
randn = torch.randn(3, 3)  # Normal (0, 1)
randint = torch.randint(0, 10, (3, 3))  # Random integers

# From numpy
np_array = np.array([[1, 2], [3, 4]])
torch_tensor = torch.from_numpy(np_array)
back_to_numpy = torch_tensor.numpy()

# Tensor operations
a = torch.tensor([[1, 2], [3, 4]])
b = torch.tensor([[5, 6], [7, 8]])

# Arithmetic
add = a + b
subtract = a - b
multiply = a * b  # Element-wise
divide = a / b

# Matrix multiplication
matmul = a @ b  # or torch.matmul(a, b)

# Reshaping
reshaped = a.reshape(4,)
view = a.view(4,)  # Similar to reshape
flattened = a.flatten()

# Transpose
transposed = a.t()  # 2-D
transposed = a.transpose(0, 1)  # N-D

# Indexing and slicing
row = a[0, :]
col = a[:, 1]
submatrix = a[0:2, 1:3]

# Boolean indexing
mask = a > 2
filtered = a[mask]

# Operations
sum_all = a.sum()
sum_axis = a.sum(dim=0)
mean = a.mean()
max_val = a.max()
min_val = a.min()
argmax = a.argmax(dim=0)

# In-place operations (end with _)
a.add_(b)  # a = a + b
a.mul_(2)  # a = a * 2
```

### GPU Tensors

```python
# Check CUDA availability
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device count: {torch.cuda.device_count()}")
print(f"Current device: {torch.cuda.current_device()}")
print(f"Device name: {torch.cuda.get_device_name(0)}")

# Move tensor to GPU
if torch.cuda.is_available():
    device = torch.device('cuda')
    a_gpu = a.to(device)
    # or
    a_gpu = a.cuda()
    
    # Operations on GPU
    result = a_gpu @ b.to(device)
    
    # Move back to CPU
    result_cpu = result.cpu()

# Best practice: define device once
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tensor = torch.randn(3, 3).to(device)
```

### 9.2.2 Autograd

**Purpose:** Automatic differentiation

```python
# Requires gradient
x = torch.tensor([2.0, 3.0], requires_grad=True)

# Or for existing tensor
x = torch.randn(3, 3, requires_grad=True)

# Build computation graph
y = x ** 2 + 2 * x + 1
z = y.sum()

# Backpropagate
z.backward()

# Access gradients
print(x.grad)  # dz/dx = 2x + 2

# Stop gradient tracking
x = torch.randn(3, requires_grad=True)
y = x ** 2

# Method 1: detach
z = y.detach()

# Method 2: no_grad context
with torch.no_grad():
    z = y ** 2

# Method 3: requires_grad_
x.requires_grad_(False)

# Higher-order gradients
x = torch.tensor([2.0], requires_grad=True)
y = x ** 3

# First derivative
dy_dx = torch.autograd.grad(y, x, create_graph=True)[0]  # 3x^2 = 12

# Second derivative
d2y_dx2 = torch.autograd.grad(dy_dx, x)[0]  # 6x = 12

# Vector-Jacobian product
x = torch.randn(3, requires_grad=True)
y = x ** 2
v = torch.tensor([1.0, 2.0, 3.0])
y.backward(v)  # Computes v^T · J
```

---

## 📚 Neural Networks with PyTorch

### 9.2.2 nn.Module

```python
import torch.nn as nn
import torch.nn.functional as F

# Simple neural network
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Usage
model = SimpleNN(784, 128, 10)

# Access parameters
for param in model.parameters():
    print(param.shape)

# Named parameters
for name, param in model.named_parameters():
    print(f"{name}: {param.shape}")

# Convolutional neural network
class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(64 * 8 * 8, 128)  # Assuming 32x32 input
        self.fc2 = nn.Linear(128, num_classes)
        
        # Regularization
        self.dropout = nn.Dropout(0.5)
        self.bn = nn.BatchNorm2d(64)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.bn(self.conv2(x))))
        x = x.view(-1, 64 * 8 * 8)  # Flatten
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

# Residual block
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.bn2 = nn.BatchNorm2d(channels)
    
    def forward(self, x):
        identity = x
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        out += identity
        out = F.relu(out)
        
        return out

# Sequential container
model = nn.Sequential(
    nn.Conv2d(3, 32, 3),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Conv2d(32, 64, 3),
    nn.ReLU(),
    nn.Flatten(),
    nn.Linear(64 * 6 * 6, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)

# ModuleList and ModuleDict
class ComplexModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        # List of layers
        self.layers = nn.ModuleList([
            nn.Linear(10, 20),
            nn.Linear(20, 30),
            nn.Linear(30, 10)
        ])
        
        # Dictionary of layers
        self.special_layers = nn.ModuleDict({
            'attention': nn.Linear(10, 10),
            'projection': nn.Linear(10, 5)
        })
    
    def forward(self, x):
        for layer in self.layers:
            x = F.relu(layer(x))
        return x
```

### Built-in Layers

```python
# Linear/Dense
linear = nn.Linear(10, 5)  # (input, output)

# Convolutional
conv2d = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
conv1d = nn.Conv1d(1, 32, kernel_size=3)
conv3d = nn.Conv3d(1, 16, kernel_size=3)

# Pooling
maxpool = nn.MaxPool2d(2, 2)
avgpool = nn.AvgPool2d(2, 2)
adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))

# Normalization
batchnorm = nn.BatchNorm2d(64)
layernorm = nn.LayerNorm(128)
instancenorm = nn.InstanceNorm2d(64)
groupnorm = nn.GroupNorm(8, 64)  # (num_groups, num_channels)

# Dropout
dropout = nn.Dropout(0.5)
dropout2d = nn.Dropout2d(0.5)

# RNN
rnn = nn.RNN(10, 20, num_layers=2, batch_first=True)
lstm = nn.LSTM(10, 20, num_layers=2, batch_first=True, dropout=0.2)
gru = nn.GRU(10, 20, num_layers=2, bidirectional=True)

# Embedding
embedding = nn.Embedding(num_embeddings=10000, embedding_dim=300)

# Attention
multihead_attn = nn.MultiheadAttention(embed_dim=128, num_heads=8)

# Activation functions (also in F)
relu = nn.ReLU()
sigmoid = nn.Sigmoid()
tanh = nn.Tanh()
softmax = nn.Softmax(dim=1)
gelu = nn.GELU()
```

---

## 📚 Training in PyTorch

### 9.2.3 Training Loop

```python
# Model, loss, optimizer
model = CNN()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10

for epoch in range(num_epochs):
    model.train()  # Set training mode
    
    for batch_idx, (data, target) in enumerate(train_loader):
        # Move to device
        data, target = data.to(device), target.to(device)
        
        # Forward pass
        optimizer.zero_grad()  # Clear gradients
        output = model(data)
        loss = criterion(output, target)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Update weights
        optimizer.step()
        
        if batch_idx % 100 == 0:
            print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}')
    
    # Validation
    model.eval()  # Set evaluation mode
    correct = 0
    total = 0
    val_loss = 0
    
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            val_loss += criterion(output, target).item()
            
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    val_accuracy = 100 * correct / total
    print(f'Epoch {epoch}, Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_accuracy:.2f}%')
```

### Learning Rate Scheduling

```python
# StepLR
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

# MultiStepLR
scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optimizer, milestones=[30, 80], gamma=0.1
)

# ReduceLROnPlateau
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=5
)

# CosineAnnealing
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=100, eta_min=0
)

# OneCycleLR
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=0.01, steps_per_epoch=len(train_loader), epochs=10
)

# Usage in training loop
for epoch in range(num_epochs):
    train(...)
    validate(...)
    scheduler.step()  # or scheduler.step(val_loss) for ReduceLROnPlateau
```

### 9.2.4 Data Loading

```python
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

# Built-in datasets
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.MNIST(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.MNIST(
    root='./data',
    train=False,
    download=True,
    transform=transform
)

# DataLoader
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,  # Parallel data loading
    pin_memory=True,  # Faster CPU→GPU transfer
    drop_last=True  # Drop incomplete batch
)

# Custom Dataset
class CustomDataset(Dataset):
    def __init__(self, data_path, transform=None):
        self.data_path = data_path
        self.transform = transform
        self.samples = self._load_samples()
    
    def _load_samples(self):
        # Load your data here
        return [...]
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample, label  # Return (data, target)

# Usage
dataset = CustomDataset('path/to/data', transform=transform)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Weighted sampling (for imbalanced data)
from torch.utils.data import WeightedRandomSampler

class_counts = [100, 500, 300]  # Samples per class
weights = [1.0 / count for count in class_counts]
sample_weights = [weights[label] for label in labels]

sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True
)

loader = DataLoader(dataset, batch_size=32, sampler=sampler)
```

### Transforms

```python
from torchvision import transforms

# Basic transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

# Data augmentation
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

# Test transform (no augmentation)
test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])
```

---

## 📚 Advanced PyTorch

### 9.2.5 torch.nn.functional

```python
import torch.nn.functional as F

# Functional API (stateless)
# vs nn.Module (stateful)

# Activations
x = F.relu(x)
x = F.sigmoid(x)
x = F.softmax(x, dim=1)
x = F.gelu(x)

# Loss functions
loss = F.cross_entropy(output, target)
loss = F.binary_cross_entropy_with_logits(output, target)
loss = F.mse_loss(output, target)

# Convolution (functional)
x = F.conv2d(x, weight, bias, stride=1, padding=0)

# Normalization (functional)
x = F.batch_norm(x, running_mean, running_var, weight, bias, training=True)

# Dropout (functional)
x = F.dropout(x, p=0.5, training=True)

# Pooling (functional)
x = F.max_pool2d(x, kernel_size=2)
x = F.adaptive_avg_pool2d(x, (7, 7))
```

### Saving and Loading

```python
# Save model state (recommended)
torch.save(model.state_dict(), 'model.pth')

# Load model state
model = CNN()
model.load_state_dict(torch.load('model.pth'))
model.eval()

# Save entire model
torch.save(model, 'model_complete.pth')

# Load entire model
model = torch.load('model_complete.pth')

# Save checkpoint
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
}
torch.save(checkpoint, 'checkpoint.pth')

# Load checkpoint
checkpoint = torch.load('checkpoint.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
```

### Mixed Precision Training

```python
from torch.cuda.amp import autocast, GradScaler

# Initialize scaler
scaler = GradScaler()

for data, target in train_loader:
    data, target = data.to(device), target.to(device)
    
    optimizer.zero_grad()
    
    # Forward with autocast
    with autocast():
        output = model(data)
        loss = criterion(output, target)
    
    # Backward with scaling
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

---

## 📊 Summary Tables

### PyTorch vs TensorFlow

| Feature | PyTorch | TensorFlow |
|---------|---------|------------|
| **Execution** | Eager (default) | Eager + Graph |
| **API Style** | Pythonic | Multiple APIs |
| **Deployment** | TorchScript | SavedModel |
| **Research** | Very popular | Popular |
| **Production** | Growing | Mature |

### Common Optimizers

| Optimizer | Usage | Best For |
|-----------|-------|----------|
| **SGD** | `optim.SGD(params, lr, momentum)` | General purpose |
| **Adam** | `optim.Adam(params, lr)` | Default choice |
| **AdamW** | `optim.AdamW(params, lr, weight_decay)` | With regularization |
| **RMSprop** | `optim.RMSprop(params, lr)` | RNNs |

### Device Management

| Operation | Code |
|-----------|------|
| Check CUDA | `torch.cuda.is_available()` |
| Get device | `torch.device('cuda' if cuda else 'cpu')` |
| Move tensor | `tensor.to(device)` |
| Move model | `model.to(device)` |

---

## 📝 Practice Problems

### Level 1: Basic
1. Create tensors and perform operations
2. Build simple nn.Module
3. Implement basic training loop
4. Use DataLoader with MNIST
5. Save and load model

### Level 2: Intermediate
1. Build CNN with residual blocks
2. Implement custom Dataset
3. Use learning rate schedulers
4. Add mixed precision training
5. Create custom loss function

### Level 3: Advanced
1. Implement distributed training
2. Create custom autograd Function
3. Build model with TorchScript
4. Implement gradient checkpointing
5. Profile and optimize training

---

**Status:** ✅ Complete  
**Next:** [[03-JAX]]
