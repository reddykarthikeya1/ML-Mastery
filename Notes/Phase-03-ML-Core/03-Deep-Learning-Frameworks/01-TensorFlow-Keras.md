# 9.1 TensorFlow and Keras

## 🎯 Learning Objectives
After completing this section, you will master:
1. **TensorFlow Basics**: Tensors, operations, variables, GradientTape
2. **Keras API**: Sequential, Functional, Model subclassing
3. **Model Training**: fit(), evaluate(), predict(), callbacks
4. **Data Pipeline**: tf.data API, performance optimization
5. **Advanced TensorFlow**: Distributed training, SavedModel, deployment

---

## 📚 TensorFlow Fundamentals

### 9.1.1 TensorFlow Architecture

**Overview:**
```
TensorFlow 2.x:
- Eager execution by default
- Keras as high-level API
- tf.function for graph mode
- Comprehensive ecosystem

Components:
┌─────────────────────────────────────┐
│         Applications (Keras)        │
├─────────────────────────────────────┤
│      High-Level APIs (tf.keras)     │
├─────────────────────────────────────┤
│    Core TensorFlow (tf.Tensor)      │
├─────────────────────────────────────┤
│         XLA Compiler / GPU          │
└─────────────────────────────────────┘
```

### 9.1.2 Tensors

**Definition:** Multi-dimensional arrays

```python
import tensorflow as tf
import numpy as np

# Creating tensors
# Scalar (0-D)
scalar = tf.constant(5)

# Vector (1-D)
vector = tf.constant([1, 2, 3, 4, 5])

# Matrix (2-D)
matrix = tf.constant([[1, 2, 3],
                      [4, 5, 6]])

# 3-D tensor
tensor_3d = tf.constant([[[1, 2], [3, 4]],
                         [[5, 6], [7, 8]]])

# Tensor properties
print(f"Shape: {tensor_3d.shape}")  # (2, 2, 2)
print(f"Rank: {tensor_3d.ndim}")    # 3
print(f"Dtype: {tensor_3d.dtype}")  # <dtype: 'int32'>
print(f"Size: {tf.size(tensor_3d)}")  # 8

# Converting from numpy
np_array = np.array([[1, 2], [3, 4]])
tf_tensor = tf.convert_to_tensor(np_array)
back_to_numpy = tf_tensor.numpy()

# Special tensors
zeros = tf.zeros((3, 3))
ones = tf.ones((3, 3))
full = tf.fill((3, 3), 7)
eye = tf.eye(3)  # Identity matrix
random_normal = tf.random.normal((3, 3), mean=0, stddev=1)
random_uniform = tf.random.uniform((3, 3), minval=0, maxval=1)
```

### 9.1.3 Tensor Operations

```python
# Basic arithmetic
a = tf.constant([[1, 2], [3, 4]])
b = tf.constant([[5, 6], [7, 8]])

add = tf.add(a, b)        # or a + b
subtract = tf.subtract(a, b)  # or a - b
multiply = tf.multiply(a, b)  # or a * b (element-wise)
divide = tf.divide(a, b)  # or a / b

# Matrix multiplication
matmul = tf.matmul(a, b)  # or a @ b

# Reshaping
reshaped = tf.reshape(a, (4,))
flattened = tf.flatten(a)

# Transpose
transposed = tf.transpose(a)

# Concatenation
concat_rows = tf.concat([a, b], axis=0)  # Vertical
concat_cols = tf.concat([a, b], axis=1)  # Horizontal

# Stacking
stacked = tf.stack([a, b], axis=0)  # New axis

# Slicing
sliced = a[0, :]      # First row
sliced = a[:, 1]      # Second column
sliced = a[0:2, 1:3]  # Range

# Reduction operations
sum_all = tf.reduce_sum(a)
sum_axis0 = tf.reduce_sum(a, axis=0)
mean = tf.reduce_mean(a)
max_val = tf.reduce_max(a)
min_val = tf.reduce_min(a)
argmax = tf.argmax(a, axis=0)

# Broadcasting
vector = tf.constant([1, 2, 3])
matrix = tf.ones((3, 3))
result = vector + matrix  # Broadcasting works
```

### 9.1.4 Variables

```python
# Variables are mutable tensors
var = tf.Variable([[1, 2], [3, 4]])

# Assignment
var.assign([[5, 6], [7, 8]])
var[0, 0].assign(10)

# Operations modify in place
var.assign_add([[1, 1], [1, 1]])
var.assign_sub([[1, 1], [1, 1]])

# Variables are tracked for gradients
W = tf.Variable(tf.random.normal((3, 2)))
b = tf.Variable(tf.zeros((2,)))

# Convert to tensor (for operations)
tensor = tf.convert_to_tensor(var)
```

### 9.1.5 GradientTape

**Purpose:** Automatic differentiation

```python
# Basic gradient computation
x = tf.constant(3.0)

with tf.GradientTape() as tape:
    tape.watch(x)  # Watch non-Variable tensors
    y = x ** 2 + 2 * x + 1

dy_dx = tape.gradient(y, x)  # dy/dx = 2x + 2 = 8
print(f"Gradient: {dy_dx}")

# Multiple variables
W = tf.Variable(tf.random.normal((3, 2)))
b = tf.Variable(tf.zeros((2,)))
x = tf.constant([[1, 2, 3]])

with tf.GradientTape() as tape:
    y = tf.matmul(x, W) + b
    loss = tf.reduce_mean(y ** 2)

# Gradients w.r.t. all trainable variables
grads = tape.gradient(loss, [W, b])

# Persistent tape (multiple gradient calls)
with tf.GradientTape(persistent=True) as tape:
    x = tf.constant(3.0)
    y = x ** 2
    z = y ** 3

dy_dx = tape.gradient(y, x)  # 2x = 6
dz_dx = tape.gradient(z, x)  # 6x^5 = 1458
del tape  # Delete persistent tape

# Higher-order gradients
x = tf.constant(1.0)

with tf.GradientTape() as outer_tape:
    with tf.GradientTape() as inner_tape:
        inner_tape.watch(x)
        y = x ** 3
    dy_dx = inner_tape.gradient(y, x)  # 3x^2
d2y_dx2 = outer_tape.gradient(dy_dx, x)  # 6x = 6
```

### 9.1.6 tf.function and Graph Mode

```python
# Eager execution (default)
def eager_func(x):
    return x ** 2 + 2 * x + 1

# Graph mode with tf.function
@tf.function
def graph_func(x):
    return x ** 2 + 2 * x + 1

# Graph mode is faster for repeated execution
import time

x = tf.random.uniform((1000, 1000))

# Eager
start = time.time()
for _ in range(100):
    result = eager_func(x)
print(f"Eager time: {time.time() - start:.4f}s")

# Graph (compiled)
start = time.time()
for _ in range(100):
    result = graph_func(x)
print(f"Graph time: {time.time() - start:.4f}s")

# AutoGraph (Python → TensorFlow graph)
@tf.function
def auto_graph(x):
    if tf.reduce_mean(x) > 0:
        return x
    else:
        return -x

# Works with loops too
@tf.function
def loop_func(x):
    for i in tf.range(10):
        x = x + i
    return x

# Concrete functions (specific input signatures)
concrete_fn = graph_func.get_concrete_function(
    tf.TensorSpec(shape=[None, 100], dtype=tf.float32)
)
```

---

## 📚 Keras API

### 9.1.2 Sequential API

```python
from tensorflow import keras
from tensorflow.keras import layers

# Simple sequential model
model = keras.Sequential([
    layers.Input(shape=(784,)),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(10, activation='softmax')
])

# Alternative: add layers one by one
model = keras.Sequential()
model.add(layers.Input(shape=(784,)))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

# Model summary
model.summary()

# Compile
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train
history = model.fit(
    x_train, y_train,
    batch_size=32,
    epochs=10,
    validation_split=0.2
)

# Evaluate
test_loss, test_acc = model.evaluate(x_test, y_test)

# Predict
predictions = model.predict(x_test)
```

### Functional API

```python
# Functional API for complex architectures
inputs = keras.Input(shape=(784,))

# Shared layers
x = layers.Dense(128, activation='relu')(inputs)
x = layers.Dropout(0.2)(x)

# Multiple outputs
x1 = layers.Dense(64, activation='relu')(x)
output1 = layers.Dense(10, activation='softmax', name='class_output')(x1)

x2 = layers.Dense(32, activation='relu')(x)
output2 = layers.Dense(1, activation='linear', name='reg_output')(x2)

# Create model
model = keras.Model(inputs=inputs, outputs=[output1, output2])

# Compile with multiple losses
model.compile(
    optimizer='adam',
    loss={
        'class_output': 'sparse_categorical_crossentropy',
        'reg_output': 'mse'
    },
    loss_weights={
        'class_output': 1.0,
        'reg_output': 0.5
    },
    metrics=['accuracy']
)

# Residual connection example
def residual_block(x, filters):
    shortcut = x
    
    x = layers.Conv2D(filters, 3, activation='relu', padding='same')(x)
    x = layers.Conv2D(filters, 3, padding='same')(x)
    
    # Add shortcut (with 1x1 conv if channels differ)
    if shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, 1, padding='same')(shortcut)
    
    x = layers.Add()([x, shortcut])
    x = layers.Activation('relu')(x)
    
    return x

# Build model with residual blocks
inputs = keras.Input(shape=(32, 32, 3))
x = layers.Conv2D(64, 3, padding='same', activation='relu')(inputs)
x = residual_block(x, 64)
x = residual_block(x, 64)
outputs = layers.Dense(10, activation='softmax')(layers.GlobalAveragePooling2D()(x))

model = keras.Model(inputs, outputs)
```

### Model Subclassing

```python
class CustomModel(keras.Model):
    """Custom model with subclassing"""
    
    def __init__(self, num_classes=10):
        super().__init__()
        self.dense1 = layers.Dense(128, activation='relu')
        self.dropout = layers.Dropout(0.2)
        self.dense2 = layers.Dense(64, activation='relu')
        self.classifier = layers.Dense(num_classes, activation='softmax')
    
    def call(self, inputs, training=False):
        x = self.dense1(inputs)
        if training:
            x = self.dropout(x, training=training)
        x = self.dense2(x)
        return self.classifier(x)

# Usage
model = CustomModel(num_classes=10)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
model.fit(x_train, y_train, epochs=10)

# Custom training loop
class CustomModelWithTrainStep(keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_tracker = keras.metrics.Mean(name='loss')
    
    @property
    def metrics(self):
        return [self.loss_tracker]
    
    def train_step(self, data):
        x, y = data
        
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compiled_loss(y, y_pred)
        
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        
        self.loss_tracker.update_state(loss)
        
        return {'loss': self.loss_tracker.result()}
```

---

## 📚 Model Training

### 9.1.3 Compiling and Training

```python
# Optimizers
optimizers = {
    'sgd': keras.optimizers.SGD(learning_rate=0.01, momentum=0.9),
    'adam': keras.optimizers.Adam(learning_rate=0.001),
    'adamw': keras.optimizers.AdamW(learning_rate=0.001, weight_decay=0.01),
    'rmsprop': keras.optimizers.RMSprop(learning_rate=0.001)
}

# Loss functions
losses = {
    'binary': 'binary_crossentropy',
    'categorical': 'categorical_crossentropy',
    'sparse': 'sparse_categorical_crossentropy',
    'mse': 'mse',
    'mae': 'mae'
}

# Metrics
metrics = [
    'accuracy',
    keras.metrics.Precision(),
    keras.metrics.Recall(),
    keras.metrics.AUC(),
    keras.metrics.F1Score()
]

# Compile
model.compile(
    optimizer=optimizers['adam'],
    loss=losses['sparse'],
    metrics=['accuracy']
)

# Training with callbacks
callbacks = [
    # Early stopping
    keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    ),
    
    # Model checkpoint
    keras.callbacks.ModelCheckpoint(
        'best_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max'
    ),
    
    # Learning rate reduction
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7
    ),
    
    # TensorBoard
    keras.callbacks.TensorBoard(
        log_dir='./logs',
        histogram_freq=1
    ),
    
    # CSV Logger
    keras.callbacks.CSVLogger('training_history.csv')
]

# Fit
history = model.fit(
    x_train, y_train,
    batch_size=32,
    epochs=100,
    validation_data=(x_val, y_val),
    callbacks=callbacks,
    verbose=1
)

# Learning rate scheduling
def lr_schedule(epoch, lr):
    if epoch < 10:
        return lr
    elif epoch < 30:
        return lr * 0.5
    else:
        return lr * 0.1

lr_scheduler = keras.callbacks.LearningRateScheduler(lr_schedule)
```

---

## 📚 Data Pipeline

### 9.1.4 tf.data API

```python
# Creating datasets
# From tensors
dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))

# From numpy
dataset = tf.data.Dataset.from_tensor_slices((x_numpy, y_numpy))

# From generator
def generator():
    for i in range(100):
        yield np.random.random((32, 32, 3)), np.random.randint(0, 10)

dataset = tf.data.Dataset.from_generator(
    generator,
    output_signature=(
        tf.TensorSpec(shape=(32, 32, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.int32)
    )
)

# Transformations
dataset = (dataset
    .shuffle(buffer_size=10000)      # Shuffle
    .batch(32)                        # Batch
    .map(lambda x, y: (x / 255.0, y))  # Map (normalize)
    .prefetch(tf.data.AUTOTUNE)       # Prefetch
)

# Performance optimization
dataset = (dataset
    .cache()                          # Cache after first epoch
    .shuffle(10000)
    .batch(32)
    .prefetch(tf.data.AUTOTUNE)       # Overlap preprocessing and training
)

# Parallel map
dataset = dataset.map(
    preprocess_function,
    num_parallel_calls=tf.data.AUTOTUNE
)

# Interleave (for multiple files)
def read_file(filename):
    return tf.data.TFRecordDataset(filename)

dataset = tf.data.Dataset.list_files('*.tfrecord')
dataset = dataset.interleave(read_file, cycle_length=4)

# Batching variations
dataset = dataset.batch(32, drop_remainder=True)  # Drop incomplete batches
dataset = dataset.padded_batch(32)  # For variable-length sequences
dataset = dataset.batch(32).unbatch()  # Unbatch

# Repeat
dataset = dataset.repeat(10)  # Repeat 10 times
dataset = dataset.repeat()    # Repeat indefinitely

# Zip and concatenate
dataset1 = tf.data.Dataset.range(10)
dataset2 = tf.data.Dataset.range(10, 20)
zipped = tf.data.Dataset.zip((dataset1, dataset2))
concatenated = dataset1.concatenate(dataset2)
```

### ImageDataGenerator

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    rescale=1./255
)

# Flow from directory
train_generator = datagen.flow_from_directory(
    'data/train',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# Flow from dataframe
train_generator = datagen.flow_from_dataframe(
    dataframe=df,
    x_col='filename',
    y_col='label',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# Train with generator
model.fit(train_generator, epochs=50)
```

---

## 📊 Summary Tables

### TensorFlow Core Operations

| Category | Operations |
|----------|------------|
| **Creation** | constant, zeros, ones, fill, random.normal |
| **Math** | add, subtract, multiply, matmul, divide |
| **Array** | reshape, transpose, concat, stack, slice |
| **Reduction** | reduce_sum, reduce_mean, reduce_max, argmax |
| **NN** | nn.conv2d, nn.relu, nn.softmax, nn.dropout |

### Keras Layers

| Type | Layers |
|------|--------|
| **Dense** | Dense, Dropout |
| **Conv** | Conv1D, Conv2D, Conv3D, SeparableConv2D |
| **Pooling** | MaxPooling, AveragePooling, GlobalPooling |
| **RNN** | SimpleRNN, LSTM, GRU, Bidirectional |
| **Normalization** | BatchNormalization, LayerNormalization |
| **Embedding** | Embedding |

### Callbacks

| Callback | Purpose |
|----------|---------|
| EarlyStopping | Stop training when metric stops improving |
| ModelCheckpoint | Save model during training |
| ReduceLROnPlateau | Reduce LR when metric stalls |
| TensorBoard | Visualize training |
| CSVLogger | Log metrics to CSV |
| LearningRateScheduler | Custom LR schedule |

---

## 📝 Practice Problems

### Level 1: Basic
1. Create tensors of different ranks
2. Perform basic tensor operations
3. Build Sequential model for MNIST
4. Use GradientTape for simple gradient
5. Compile and train basic model

### Level 2: Intermediate
1. Build model with Functional API
2. Create custom model with subclassing
3. Build efficient tf.data pipeline
4. Implement custom training loop
5. Use multiple callbacks

### Level 3: Advanced
1. Create model with multiple inputs/outputs
2. Implement custom layer
3. Build distributed training pipeline
4. Create SavedModel for deployment
5. Optimize with tf.function and XLA

---

**Status:** ✅ Complete  
**Next:** [[02-PyTorch]]
