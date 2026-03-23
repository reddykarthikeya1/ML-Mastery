# 9.1-9.3 Deep Learning Frameworks

## 🎯 Quick Overview
- **TensorFlow/Keras**: Production-ready framework
- **PyTorch**: Research-friendly framework
- **JAX**: High-performance numerical computing
- **Foundation for**: Building and deploying deep learning models

---

## 1. TensorFlow/Keras

### TensorFlow Basics

```python
import tensorflow as tf
from tensorflow import keras
import numpy as np

# Check TensorFlow version
print(f"TensorFlow version: {tf.__version__}")

# Check GPU availability
print(f"GPU Available: {tf.config.list_physical_devices('GPU')}")

# Create tensors
a = tf.constant([[1, 2], [3, 4]])
b = tf.constant([[5, 6], [7, 8]])

# Tensor operations
c = tf.add(a, b)
d = tf.matmul(a, b)
e = tf.transpose(a)

print(f"Addition:\n{c}")
print(f"Matrix multiplication:\n{d}")
print(f"Transpose:\n{e}")

# Variables (trainable)
w = tf.Variable(tf.random.normal([2, 2]))
b = tf.Variable(tf.zeros([2]))

# GradientTape for automatic differentiation
x = tf.constant([[1.0, 2.0]])

with tf.GradientTape() as tape:
    y = tf.matmul(x, w) + b
    loss = tf.reduce_mean(tf.square(y))

gradients = tape.gradient(loss, [w, b])
print(f"Gradients: {gradients}")
```

### Keras Sequential API

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Activation

# Create sequential model
model = Sequential([
    Dense(128, input_shape=(784,)),
    BatchNormalization(),
    Activation('relu'),
    Dropout(0.3),
    
    Dense(64),
    BatchNormalization(),
    Activation('relu'),
    Dropout(0.3),
    
    Dense(10, activation='softmax')
])

# Compile model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Model summary
model.summary()

# Train model
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    callbacks=[
        keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3)
    ]
)

# Evaluate
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_acc:.4f}")
```

### Keras Functional API

```python
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, concatenate

# Functional API for complex architectures
input_layer = Input(shape=(28, 28, 1))

# Branch 1
conv1 = Conv2D(32, 3, activation='relu', padding='same')(input_layer)
pool1 = MaxPooling2D()(conv1)

# Branch 2
conv2 = Conv2D(64, 5, activation='relu', padding='same')(input_layer)
pool2 = MaxPooling2D()(conv2)

# Concatenate branches
concat = concatenate([pool1, pool2])

# Common layers
flatten = Flatten()(concat)
dense1 = Dense(128, activation='relu')(flatten)
dropout = Dropout(0.5)(dense1)
output = Dense(10, activation='softmax')(dropout)

# Create model
model = Model(inputs=input_layer, outputs=output)

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()
```

### Custom Layers and Models

```python
from tensorflow.keras.layers import Layer

class CustomDense(Layer):
    def __init__(self, units, activation='relu'):
        super(CustomDense, self).__init__()
        self.units = units
        self.activation = keras.activations.get(activation)
    
    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer='random_normal',
            trainable=True
        )
        self.b = self.add_weight(
            shape=(self.units,),
            initializer='zeros',
            trainable=True
        )
    
    def call(self, inputs):
        return self.activation(tf.matmul(inputs, self.w) + self.b)

# Custom Model with training loop
class CustomModel(Model):
    def __init__(self):
        super(CustomModel, self).__init__()
        self.dense1 = CustomDense(128)
        self.dropout = Dropout(0.3)
        self.dense2 = CustomDense(10)
    
    def call(self, inputs, training=False):
        x = self.dense1(inputs)
        if training:
            x = self.dropout(x, training=True)
        return self.dense2(x)
    
    def train_step(self, data):
        x, y = data
        
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compiled_loss(y, y_pred)
        
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        
        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}

# Usage
model = CustomModel()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, batch_size=32)
```

### Data Pipeline with tf.data

```python
# Create dataset
dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))

# Shuffle, batch, and prefetch
dataset = dataset.shuffle(10000).batch(32).prefetch(tf.data.AUTOTUNE)

# Create validation dataset
val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(32).prefetch(tf.data.AUTOTUNE)

# Train with dataset
model.fit(dataset, epochs=50, validation_data=val_dataset)

# Data augmentation
def augment(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.1)
    return image, label

augmented_dataset = dataset.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
```

---

## 2. PyTorch

### PyTorch Basics

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# Check PyTorch version and GPU
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

# Create tensors
a = torch.tensor([[1, 2], [3, 4]])
b = torch.tensor([[5, 6], [7, 8]])

# Tensor operations
c = torch.add(a, b)
d = torch.matmul(a, b)
e = a.t()  # Transpose

# GPU tensor
if torch.cuda.is_available():
    device = torch.device('cuda')
    a_gpu = a.to(device)
    b_gpu = b.to(device)
    c_gpu = torch.matmul(a_gpu, b_gpu)
```

### Neural Network with PyTorch

```python
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(0.3)
        self.fc3 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc3(out)
        return out

# Create model
model = NeuralNetwork(input_size=784, hidden_size=128, num_classes=10)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 50
batch_size = 32

# Create DataLoader
dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    
    for batch_x, batch_y in dataloader:
        # Forward pass
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader):.4f}")

# Evaluation
model.eval()
with torch.no_grad():
    outputs = model(torch.FloatTensor(X_test))
    _, predicted = torch.max(outputs, 1)
    accuracy = (predicted == torch.LongTensor(y_test)).float().mean()
    print(f"Test Accuracy: {accuracy:.4f}")
```

### Custom Dataset

```python
from torch.utils.data import Dataset
from PIL import Image
import os

class CustomImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = [os.path.join(root_dir, f) for f in os.listdir(root_dir) if f.endswith('.jpg')]
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        # Extract label from filename or separate file
        label = int(img_path.split('/')[-1].split('_')[0])
        
        return image, label

# Usage with transforms
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataset = CustomImageDataset(root_dir='data/images', transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
```

### Transfer Learning

```python
from torchvision import models

# Load pre-trained ResNet
resnet = models.resnet50(pretrained=True)

# Freeze all layers
for param in resnet.parameters():
    param.requires_grad = False

# Replace final layer
num_features = resnet.fc.in_features
resnet.fc = nn.Linear(num_features, num_classes)

# Train only the final layer
optimizer = optim.Adam(resnet.fc.parameters(), lr=0.001)

# Training loop
for epoch in range(num_epochs):
    resnet.train()
    for batch_x, batch_y in dataloader:
        outputs = resnet(batch_x)
        loss = criterion(outputs, batch_y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

---

## 3. JAX

### JAX Basics

```python
import jax
import jax.numpy as jnp
from jax import grad, jit, vmap, pmap

print(f"JAX version: {jax.__version__}")
print(f"Devices: {jax.devices()}")

# JAX numpy is like regular numpy but with JAX features
x = jnp.array([1.0, 2.0, 3.0])
y = jnp.array([4.0, 5.0, 6.0])

# JIT compilation for speed
@jit
def compute(x, y):
    return jnp.sum(x * y)

result = compute(x, y)

# Automatic differentiation
def loss_fn(params, x, y):
    predictions = jnp.dot(x, params)
    return jnp.mean((predictions - y) ** 2)

# Gradient
grad_fn = grad(loss_fn)
params = jnp.array([1.0, 2.0, 3.0])
gradients = grad_fn(params, x, y)

# Vectorization
@vmap
def batch_compute(x_batch, y_batch):
    return jnp.sum(x_batch * y_batch, axis=1)

x_batch = jnp.array([[1, 2, 3], [4, 5, 6]])
y_batch = jnp.array([[4, 5, 6], [7, 8, 9]])
results = batch_compute(x_batch, y_batch)
```

### Neural Network with JAX and Flax

```python
from flax import linen as nn
from flax.training import train_state
import optax

class MLP(nn.Module):
    features: list
    dropout_rate: float = 0.3
    
    @nn.compact
    def __call__(self, x, train=True):
        for features in self.features[:-1]:
            x = nn.Dense(features)(x)
            x = nn.relu(x)
            x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not train)
        
        x = nn.Dense(self.features[-1])(x)
        return x

# Create model
model = MLP(features=[128, 64, 10])
params = model.init(jax.random.PRNGKey(0), jnp.ones((1, 784)))

# Create training state
class TrainState(train_state.TrainState):
    pass

tx = optax.adam(learning_rate=0.001)
state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)

# Training step
@jit
def train_step(state, batch):
    def loss_fn(params):
        logits = model.apply(params, batch['x'])
        loss = optax.softmax_cross_entropy(logits, batch['y']).mean()
        return loss
    
    grad_fn = jax.grad(loss_fn)
    grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    
    return state

# Training loop
for epoch in range(num_epochs):
    state = train_step(state, {'x': X_train, 'y': y_train})
```

---

## 💻 Python Code Examples

```python
# === Complete Comparison: TensorFlow vs PyTorch ===

# Same model in both frameworks
input_size = 784
hidden_size = 128
num_classes = 10

# TensorFlow/Keras
tf_model = keras.Sequential([
    keras.layers.Dense(128, activation='relu', input_shape=(784,)),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(10, activation='softmax')
])

tf_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
tf_model.fit(X_train, y_train, epochs=10, validation_split=0.2, verbose=0)

# PyTorch
class TorchModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

torch_model = TorchModel()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(torch_model.parameters(), lr=0.001)

# Train PyTorch model
torch_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
torch_loader = DataLoader(torch_dataset, batch_size=32, shuffle=True)

for epoch in range(10):
    torch_model.train()
    for batch_x, batch_y in torch_loader:
        outputs = torch_model(batch_x)
        loss = criterion(outputs, batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# Compare predictions
tf_preds = tf_model.predict(X_test)
torch_model.eval()
with torch.no_grad():
    torch_preds = torch_model(torch.FloatTensor(X_test))

print(f"TensorFlow accuracy: {np.mean(np.argmax(tf_preds, axis=1) == y_test):.4f}")
print(f"PyTorch accuracy: {np.mean(torch.argmax(torch_preds, axis=1).numpy() == y_test):.4f}")
```

---

## 📊 Summary Tables

### Framework Comparison

| Feature | TensorFlow | PyTorch | JAX |
|---------|------------|---------|-----|
| API Style | Declarative | Imperative | Functional |
| Debugging | Harder | Easier | Moderate |
| Production | Excellent | Good | Growing |
| Research | Good | Excellent | Excellent |
| Deployment | TFLite, TF Serving | TorchServe | XLA |

### When to Use Each

| Framework | Use Case |
|-----------|----------|
| TensorFlow | Production deployment, mobile |
| PyTorch | Research, prototyping |
| JAX | High-performance computing, research |

---

## 🎯 ML Applications

| Framework | ML Application |
|-----------|----------------|
| TensorFlow | Production models, TFLite mobile apps |
| PyTorch | Research papers, custom architectures |
| JAX | Large-scale training, numerical computing |

---

**Status:** ✅ Complete
**Next:** Phase 4 - Specialization (NLP, Computer Vision)
