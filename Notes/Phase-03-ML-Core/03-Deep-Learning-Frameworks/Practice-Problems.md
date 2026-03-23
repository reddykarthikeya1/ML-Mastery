# Deep Learning Frameworks - Practice Problems

## Topic 1: TensorFlow/Keras

### Level 1: Basic

**1.1** Build Sequential model:
```python
from tensorflow import keras

# Build a model for MNIST:
# - Input layer (784 features)
# - Hidden layer with 128 units, ReLU, Dropout
# - Output layer with 10 units, softmax

model = keras.Sequential([
    # Your code here
])

model.compile(
    # Your code here
)
```

**1.2** Training with callbacks:
```python
# Add callbacks:
# - EarlyStopping
# - ReduceLROnPlateau
# - ModelCheckpoint

callbacks = [
    # Your code here
]

model.fit(X_train, y_train, epochs=50, callbacks=callbacks)
```

### Level 2: Intermediate

**2.1** Functional API:
```python
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import *

# Build multi-input model:
# Input 1: Image (28x28x1)
# Input 2: Metadata (10 features)
# Output: Classification (10 classes)

# Your code here
```

**2.2** Custom layer:
```python
from tensorflow.keras.layers import Layer

class CustomDense(Layer):
    def __init__(self, units, activation='relu'):
        super().__init__()
        # Your code here
    
    def build(self, input_shape):
        # Your code here
    
    def call(self, inputs):
        # Your code here
```

### Level 3: Advanced

**2.3** Custom training loop:
```python
class CustomModel(keras.Model):
    def train_step(self, data):
        # Implement custom training step
        pass
    
    def test_step(self, data):
        # Implement custom test step
        pass
```

---

## Topic 2: PyTorch

### Level 1: Basic

**1.1** Build neural network:
```python
import torch.nn as nn

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        # Your code here
    
    def forward(self, x):
        # Your code here

model = NeuralNetwork(784, 128, 10)
```

**1.2** Training loop:
```python
# Implement complete training loop:
# - Forward pass
# - Loss computation
# - Backward pass
# - Optimizer step

for epoch in range(num_epochs):
    # Your code here
```

### Level 2: Intermediate

**2.1** Custom Dataset:
```python
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        # Your code here
    
    def __len__(self):
        # Your code here
    
    def __getitem__(self, idx):
        # Your code here
```

**2.2** Transfer Learning:
```python
from torchvision import models

# Load pre-trained ResNet
# Freeze all layers except final layer
# Replace final layer for custom classification

resnet = models.resnet50(pretrained=True)
# Your code here
```

### Level 3: Advanced

**2.3** Distributed training:
```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

# Setup distributed training
# Wrap model with DDP
# Train across multiple GPUs
```

---

## Topic 3: JAX

### Level 2: Intermediate

**3.1** JAX basics:
```python
import jax
import jax.numpy as jnp
from jax import grad, jit, vmap

# 1. JIT compile a function
# 2. Compute gradient
# 3. Vectorize with vmap

@jit
def compute(x, y):
    # Your code here

grad_fn = grad(compute)
vectorized_fn = vmap(compute)
```

**3.2** Neural network with Flax:
```python
from flax import linen as nn

class MLP(nn.Module):
    features: list
    
    @nn.compact
    def __call__(self, x):
        # Your code here

model = MLP(features=[128, 64, 10])
```

### Level 3: Advanced

**3.3** Custom training loop with JAX:
```python
from flax.training import train_state
import optax

class TrainState(train_state.TrainState):
    pass

@jax.jit
def train_step(state, batch):
    def loss_fn(params):
        # Your code here
    
    # Compute gradients
    # Update state
    # Return new state
```

---

## Topic 4: Framework Comparison

### Level 2: Intermediate

**4.1** Same model in different frameworks:
```python
# Implement same architecture in:
# 1. TensorFlow/Keras
# 2. PyTorch
# 3. JAX/Flax

# Compare:
# - Code complexity
# - Training speed
# - Final accuracy
```

**4.2** Model export and deployment:
```python
# TensorFlow: Save and load model
# PyTorch: Save and load state_dict
# Compare deployment options
```

---

## Solutions (Selected)

<details>
<summary>Click to reveal solutions</summary>

### 1.1 Sequential Model
```python
model = keras.Sequential([
    keras.layers.Dense(128, activation='relu', input_shape=(784,)),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
```

### 1.2 Callbacks
```python
callbacks = [
    keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-6
    ),
    keras.callbacks.ModelCheckpoint(
        'best_model.h5',
        monitor='val_accuracy',
        save_best_only=True
    )
]
```

### 2.1 Custom Dataset
```python
class CustomDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]
        
        if self.transform:
            x = self.transform(x)
        
        return x, y
```

### 2.2 Transfer Learning
```python
resnet = models.resnet50(pretrained=True)

# Freeze all layers
for param in resnet.parameters():
    param.requires_grad = False

# Replace final layer
num_features = resnet.fc.in_features
resnet.fc = nn.Linear(num_features, num_classes)

# Only train final layer
optimizer = optim.Adam(resnet.fc.parameters(), lr=0.001)
```

### 3.1 JAX Basics
```python
@jit
def compute(x, y):
    return jnp.sum(x * y)

def loss_fn(params, x, y):
    predictions = jnp.dot(x, params)
    return jnp.mean((predictions - y) ** 2)

grad_fn = grad(loss_fn)
gradients = grad_fn(params, x, y)

@vmap
def batch_compute(x_batch, y_batch):
    return jnp.sum(x_batch * y_batch, axis=1)
```

</details>

---

## 📝 Notes Section

### My Practice Problems:


### Mistakes to Review:


### Key Insights:


---
**Last Updated:** 2026-03-23
**Status:** ✅ Deep Learning Frameworks Complete!
