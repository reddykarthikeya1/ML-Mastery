# 9.3 JAX - Complete Guide

## 🎯 Learning Objectives
After completing this section, you will master:
1. **JAX Fundamentals**: NumPy compatibility, jit compilation, functional paradigm
2. **Transformations**: grad, vmap, pmap for automatic differentiation and parallelism
3. **Deep Learning with JAX**: Flax for neural networks, complete CNN implementation
4. **Optimization**: Optax for gradient-based optimization, complete training loops
5. **Production Ready**: Debugging, best practices, real-world projects

---

## 📚 JAX Fundamentals

### 9.3.1 JAX Philosophy

**What is JAX?**
```
JAX = NumPy + Automatic Differentiation + XLA Compilation

Key Features:
┌─────────────────────────────────────────────────────────┐
│  NumPy-compatible API    → Familiar syntax              │
│  Automatic differentiation → grad() transformation      │
│  JIT compilation         → jit() with XLA               │
│  Vectorization           → vmap() automatic batching    │
│  Parallelization         → pmap() multi-device          │
│  Functional paradigm     → Pure functions, no side effects│
└─────────────────────────────────────────────────────────┘

JAX Transformation Pipeline:
┌──────────┐    ┌──────┐    ┌──────┐    ┌──────┐
│  Python  │ →  │ grad │ →  │ jit  │ →  │ XLA  │ → Hardware
│  Code    │    │vmap  │    │compile│   │ HLO  │
└──────────┘    └──────┘    └──────┘    └──────┘
```

**Installation:**
```bash
# CPU version
pip install jax jaxlib

# GPU version (CUDA)
pip install jax[cuda] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# TPU version
pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

# Deep learning libraries
pip install flax optax orbax-checkpoint
```

### 9.3.2 NumPy Compatibility

```
JAX Arrays vs NumPy Arrays:
┌────────────────────────────────────────────────────────┐
│  Feature          │  NumPy      │  JAX                │
├────────────────────────────────────────────────────────┤
│  Mutability       │  Mutable    │  Immutable          │
│  In-place ops     │  Supported  │  Not supported      │
│  Random numbers   │  Global state│  Explicit PRNGKey  │
│  GPU/TPU support  │  No         │  Yes                │
│  JIT compilation  │  No         │  Yes                │
│  Autodiff         │  No         │  Yes                │
└────────────────────────────────────────────────────────┘
```

```python
import jax
import jax.numpy as jnp
import numpy as np
from jax import random, grad, jit, vmap, pmap

# ============================================================================
# EXAMPLE 1: Basic JAX Array Operations
# ============================================================================
print("=" * 60)
print("JAX Array Basics")
print("=" * 60)

# Array creation (similar to NumPy)
x = jnp.array([1.0, 2.0, 3.0])
y = jnp.array([4.0, 5.0, 6.0])

print(f"x = {x}")
print(f"y = {y}")

# Standard operations
print(f"\nx + y = {x + y}")           # [5., 7., 9.]
print(f"x * y = {x * y}")             # [4., 10., 18.]
print(f"x · y = {jnp.dot(x, y)}")     # 32.0
print(f"sum(x) = {jnp.sum(x)}")       # 6.0

# Array creation functions
print(f"\nzeros: {jnp.zeros(5)}")
print(f"ones: {jnp.ones((3, 3))}")
print(f"arange: {jnp.arange(10)}")
print(f"linspace: {jnp.linspace(0, 1, 5)}")

# ============================================================================
# EXAMPLE 2: Immutability in JAX
# ============================================================================
print("\n" + "=" * 60)
print("Immutability in JAX")
print("=" * 60)

x = jnp.array([1, 2, 3])
print(f"Original: {x}")

# ❌ This will fail:
# x = x.at[0].set(5)  # ✅ Correct way using .at[].set()
x_updated = x.at[0].set(5)
print(f"After .at[0].set(5): {x_updated}")
print(f"Original unchanged: {x}")

# Multiple updates
x = jnp.array([1, 2, 3, 4, 5])
x = x.at[1:3].set([10, 20])  # Slice update
x = x.at[4].add(100)          # Add to index
x = x.at[0].mul(2)            # Multiply at index

print(f"After multiple updates: {x}")  # [2, 10, 20, 4, 105]

# ============================================================================
# EXAMPLE 3: Random Number Generation
# ============================================================================
print("\n" + "=" * 60)
print("Random Number Generation")
print("=" * 60)

# JAX requires explicit random keys
key = random.PRNGKey(0)
print(f"Initial key: {key}")

# Split key for different random operations
key, subkey1 = random.split(key)
key, subkey2 = random.split(key)

# Generate random arrays
random_array1 = random.normal(subkey1, shape=(3, 3))
random_array2 = random.uniform(subkey2, shape=(2, 4))

print(f"\nNormal distribution:\n{random_array1}")
print(f"\nUniform distribution:\n{random_array2}")

# ============================================================================
# EXAMPLE 4: JAX ↔ NumPy Conversion
# ============================================================================
print("\n" + "=" * 60)
print("JAX ↔ NumPy Conversion")
print("=" * 60)

# JAX to NumPy
jax_array = jnp.array([1, 2, 3])
numpy_array = np.asarray(jax_array)
print(f"JAX → NumPy: {numpy_array} (type: {type(numpy_array)})")

# NumPy to JAX
numpy_array = np.array([4, 5, 6])
jax_array = jnp.array(numpy_array)
print(f"NumPy → JAX: {jax_array} (type: {type(jax_array)})")
```

### 9.3.3 JIT Compilation

```
JIT Compilation Flow:
┌─────────────┐
│  Python     │
│  Function   │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  JAX traces │  ← First call: trace and compile
│  function   │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  XLA HLO    │  ← High-Level Operations
│  Graph      │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Machine    │  ← Optimized for CPU/GPU/TPU
│  Code       │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Fast       │  ← Subsequent calls use cached
│  Execution  │     compiled version
└─────────────┘
```

```python
# ============================================================================
# EXAMPLE 5: JIT Compilation Basics
# ============================================================================
print("\n" + "=" * 60)
print("JIT Compilation")
print("=" * 60)

from jax import jit
import time

# Function without JIT
def slow_function(x):
    """Compute-intensive function without JIT"""
    for i in range(1000):
        x = x ** 2
    return x

# JIT-compiled version
@jit
def fast_function(x):
    """Same function with JIT compilation"""
    for i in range(1000):
        x = x ** 2
    return x

# Benchmark
x = jnp.ones((100, 100))

# First call (includes compilation)
start = time.time()
result1 = slow_function(x)
slow_time = time.time() - start
print(f"Without JIT: {slow_time:.4f}s")

# JIT warmup (compilation happens here)
_ = fast_function(x)

# Actual benchmark
start = time.time()
result2 = fast_function(x)
fast_time = time.time() - start
print(f"With JIT: {fast_time:.4f}s")
print(f"Speedup: {slow_time / fast_time:.2f}x")

# ============================================================================
# EXAMPLE 6: JIT with Multiple Arguments
# ============================================================================
print("\n" + "=" * 60)
print("JIT with Multiple Arguments")
print("=" * 60)

@jit
def multiply_add(x, y, z):
    """JIT-compiled function with multiple arguments"""
    return x * y + z

x = jnp.array([1.0, 2.0, 3.0])
y = jnp.array([4.0, 5.0, 6.0])
z = jnp.array([7.0, 8.0, 9.0])

result = multiply_add(x, y, z)
print(f"Result: {result}")  # [11., 18., 27.]

# ============================================================================
# EXAMPLE 7: JIT with Static Arguments
# ============================================================================
print("\n" + "=" * 60)
print("JIT with Static Arguments")
print("=" * 60)

@jit
def power(x, n):
    """Power function where n is static"""
    for _ in range(n):
        x = x * x
    return x

# Compile with specific n value
power_jit = jit(power, static_argnums=(1,))

print(f"2^10 = {power_jit(jnp.array(2.0), 10)}")  # 1024.0
print(f"3^5 = {power_jit(jnp.array(3.0), 5)}")    # 243.0

# ============================================================================
# EXAMPLE 8: JIT Best Practices
# ============================================================================
print("\n" + "=" * 60)
print("JIT Best Practices")
print("=" * 60)

# ✅ DO: Use pure functions (no side effects)
@jit
def pure_function(x):
    return x ** 2 + 2 * x + 1

# ❌ DON'T: Use side effects in JIT'd functions
# @jit
# def impure_function(x):
#     print(f"Processing {x}")  # Won't work!
#     return x ** 2

# ✅ DO: Use jax.lax.cond for conditionals
from jax import lax

@jit
def conditional(x):
    return lax.cond(
        x > 0,
        lambda x: x ** 2,    # if true
        lambda x: -x,        # if false
        x
    )

print(f"conditional(5) = {conditional(5.0)}")   # 25.0
print(f"conditional(-3) = {conditional(-3.0)}") # 3.0
```

### 9.3.4 grad Transformation

```
Automatic Differentiation in JAX:
┌─────────────────────────────────────────────────────────┐
│  f(x) = x² + 2x + 1                                     │
│                                                         │
│  grad(f)(x) = 2x + 2                                    │
│                                                         │
│  At x=3: f(3) = 16, f'(3) = 8                          │
└─────────────────────────────────────────────────────────┘

Higher-Order Derivatives:
┌─────────────────────────────────────────────────────────┐
│  f(x) = x³                                              │
│  f'(x) = grad(f)(x) = 3x²                              │
│  f''(x) = grad(grad(f))(x) = 6x                        │
│  f'''(x) = grad(grad(grad(f)))(x) = 6                  │
└─────────────────────────────────────────────────────────┘
```

```python
# ============================================================================
# EXAMPLE 9: Basic Gradient Computation
# ============================================================================
print("\n" + "=" * 60)
print("Basic Gradient Computation")
print("=" * 60)

from jax import grad, value_and_grad, jacobian, hessian

# Simple function
def f(x):
    return x ** 2 + 2 * x + 1

# Compute gradient
df_dx = grad(f)

x = jnp.array(3.0)
print(f"f({x}) = {f(x)}")        # 16.0
print(f"f'({x}) = {df_dx(x)}")   # 8.0

# Verify analytically: f'(x) = 2x + 2, so f'(3) = 8 ✓

# ============================================================================
# EXAMPLE 10: Value and Gradient Together
# ============================================================================
print("\n" + "=" * 60)
print("Value and Gradient")
print("=" * 60)

# More efficient: compute both in one pass
f_and_grad = value_and_grad(f)

value, gradient = f_and_grad(3.0)
print(f"f(3) = {value}, f'(3) = {gradient}")  # (16.0, 8.0)

# ============================================================================
# EXAMPLE 11: Gradient w.r.t. Specific Arguments
# ============================================================================
print("\n" + "=" * 60)
print("Gradients w.r.t. Specific Arguments")
print("=" * 60)

def multiply_add(x, y, z):
    """Function with multiple arguments"""
    return jnp.sum(x * y) + z

# Gradient w.r.t. first argument (default)
grad_x = grad(multiply_add)

# Gradient w.r.t. second argument
grad_y = grad(multiply_add, argnums=1)

# Gradient w.r.t. multiple arguments
grad_xy = grad(multiply_add, argnums=(0, 1))

x = jnp.array([1.0, 2.0, 3.0])
y = jnp.array([4.0, 5.0, 6.0])
z = jnp.array(10.0)

print(f"∂f/∂x at x={x}, y={y}: {grad_x(x, y, z)}")  # [4., 5., 6.]
print(f"∂f/∂y at x={x}, y={y}: {grad_y(x, y, z)}")  # [1., 2., 3.]
gx, gy = grad_xy(x, y, z)
print(f"Both gradients: gx={gx}, gy={gy}")

# ============================================================================
# EXAMPLE 12: Higher-Order Derivatives
# ============================================================================
print("\n" + "=" * 60)
print("Higher-Order Derivatives")
print("=" * 60)

def cubic(x):
    return x ** 3

# First derivative: 3x²
first_deriv = grad(cubic)

# Second derivative: 6x
second_deriv = grad(first_deriv)

# Third derivative: 6
third_deriv = grad(second_deriv)

x = jnp.array(5.0)
print(f"f(x) = x³")
print(f"f'({x}) = {first_deriv(x)}")    # 75.0 (3·5²)
print(f"f''({x}) = {second_deriv(x)}")  # 30.0 (6·5)
print(f"f'''({x}) = {third_deriv(x)}")  # 6.0

# ============================================================================
# EXAMPLE 13: Jacobian and Hessian
# ============================================================================
print("\n" + "=" * 60)
print("Jacobian and Hessian")
print("=" * 60)

# Jacobian for vector-valued functions
def f_vector(x):
    """Function R² → R³"""
    return jnp.array([
        x[0] ** 2,           # x₁²
        x[1] ** 3,           # x₂³
        x[0] * x[1]          # x₁·x₂
    ])

jac = jacobian(f_vector)
x = jnp.array([1.0, 2.0])
J = jac(x)

print(f"f(x) = [x₁², x₂³, x₁·x₂]")
print(f"At x = {x}")
print(f"Jacobian matrix (3×2):\n{J}")
# Expected: [[2x₁, 0], [0, 3x₂²], [x₂, x₁]] = [[2, 0], [0, 12], [2, 1]]

# Hessian for scalar-valued functions
def f_scalar(x):
    """Function R² → R"""
    return x[0] ** 2 + x[1] ** 3 + x[0] * x[1]

hess = hessian(f_scalar)
x = jnp.array([1.0, 2.0])
H = hess(x)

print(f"\nf(x) = x₁² + x₂³ + x₁·x₂")
print(f"At x = {x}")
print(f"Hessian matrix (2×2):\n{H}")
# Expected: [[2, 1], [1, 6x₂]] = [[2, 1], [1, 12]]

# ============================================================================
# EXAMPLE 14: Neural Network Gradient
# ============================================================================
print("\n" + "=" * 60)
print("Neural Network Gradient")
print("=" * 60)

def forward_pass(params, x):
    """Simple 2-layer neural network"""
    W1, b1, W2, b2 = params
    h = jnp.maximum(jnp.dot(x, W1) + b1, 0)  # ReLU hidden layer
    out = jnp.dot(h, W2) + b2
    return out

def loss_fn(params, x, y):
    """Mean squared error loss"""
    predictions = forward_pass(params, x)
    return jnp.mean((predictions - y) ** 2)

# Initialize random parameters
key = random.PRNGKey(0)
key, subkey = random.split(key)

W1 = random.normal(subkey, (10, 20))
b1 = jnp.zeros(20)
W2 = random.normal(key, (20, 1))
b2 = jnp.zeros(1)

params = (W1, b1, W2, b2)

# Batch of data
x_batch = random.normal(key, (32, 10))
y_batch = random.normal(key, (32, 1))

# Compute gradient
grad_loss = grad(loss_fn)
gradients = grad_loss(params, x_batch, y_batch)

print(f"Gradient shapes:")
print(f"  dL/dW1: {gradients[0].shape}")
print(f"  db1: {gradients[1].shape}")
print(f"  dL/dW2: {gradients[2].shape}")
print(f"  db2: {gradients[3].shape}")
```

### 9.3.5 vmap Transformation

```
vmap (Vectorizing Map) Transformation:

Without vmap (Python loop):
┌─────────────────────────────────────────┐
│  for i in range(batch_size):            │
│      output[i] = model(params, x[i])    │
│  # Slow: Python loop overhead           │
└─────────────────────────────────────────┘

With vmap (automatic vectorization):
┌─────────────────────────────────────────┐
│  batch_predict = vmap(model)            │
│  output = batch_predict(params, X)      │
│  # Fast: Compiled vectorized operation  │
└─────────────────────────────────────────┘

in_axes specification:
┌─────────────────────────────────────────────────────────┐
│  in_axes = 0     → Map over first axis (batch dim)     │
│  in_axes = 1     → Map over second axis                │
│  in_axes = None  → Don't map (broadcast)               │
│  in_axes = (0, None) → Map first arg, broadcast second │
└─────────────────────────────────────────────────────────┘
```

```python
# ============================================================================
# EXAMPLE 15: Basic vmap Usage
# ============================================================================
print("\n" + "=" * 60)
print("Basic vmap Usage")
print("=" * 60)

# Function for single example
def predict_single(params, x):
    """Make prediction for single input"""
    W1, b1, W2, b2 = params
    h = jnp.maximum(jnp.dot(x, W1) + b1, 0)
    out = jnp.dot(h, W2) + b2
    return out

# Initialize params
key = random.PRNGKey(0)
W1 = random.normal(key, (10, 20))
b1 = jnp.zeros(20)
W2 = random.normal(key, (20, 1))
b2 = jnp.zeros(1)
params = (W1, b1, W2, b2)

# Without vmap: manual loop
def predict_loop(params, X):
    """Batch prediction using Python loop"""
    predictions = []
    for i in range(X.shape[0]):
        pred = predict_single(params, X[i])
        predictions.append(pred)
    return jnp.vstack(predictions)

# With vmap: automatic vectorization
predict_batch = vmap(predict_single, in_axes=(None, 0))

# Test data
X = random.normal(key, (5, 10))  # 5 examples

# Compare results
predictions_loop = predict_loop(params, X)
predictions_vmap = predict_batch(params, X)

print(f"Predictions shape (loop): {predictions_loop.shape}")
print(f"Predictions shape (vmap): {predictions_vmap.shape}")
print(f"Results match: {jnp.allclose(predictions_loop, predictions_vmap)}")

# ============================================================================
# EXAMPLE 16: vmap with Different in_axes
# ============================================================================
print("\n" + "=" * 60)
print("vmap with Different in_axes")
print("=" * 60)

# Matrix-vector multiplication
def matmul_single(A, b):
    """A @ b for single vector b"""
    return jnp.dot(A, b)

A = jnp.array([[1, 2], [3, 4], [5, 6]])  # (3, 2)

# Case 1: Batch of vectors (map over b)
b_batch = jnp.array([[1, 0], [0, 1], [1, 1]])  # (3, 2)
matmul_batch_b = vmap(matmul_single, in_axes=(None, 0))
result1 = matmul_batch_b(A, b_batch)
print(f"Map over b only: shape {result1.shape}")  # (3, 3)

# Case 2: Batch of matrices (map over A)
A_batch = jnp.ones((4, 3, 2))  # (4, 3, 2)
b = jnp.array([1, 0])  # (2,)
matmul_batch_A = vmap(matmul_single, in_axes=(0, None))
result2 = matmul_batch_A(A_batch, b)
print(f"Map over A only: shape {result2.shape}")  # (4, 3)

# Case 3: Batch of both (map over both)
matmul_batch_both = vmap(matmul_single, in_axes=(0, 0))
result3 = matmul_batch_both(A_batch, b_batch[:4])
print(f"Map over both: shape {result3.shape}")  # (4, 3)

# ============================================================================
# EXAMPLE 17: Nested vmap
# ============================================================================
print("\n" + "=" * 60)
print("Nested vmap")
print("=" * 60)

def outer_product(x, y):
    """Compute outer product of two vectors"""
    return jnp.outer(x, y)

x = jnp.array([1, 2, 3])
y = jnp.array([4, 5])

# Single outer product
print(f"Single: {outer_product(x, y).shape}")  # (3, 2)

# Batch of x vectors
x_batch = jnp.array([[1, 2, 3], [4, 5, 6]])
batched_x = vmap(outer_product, in_axes=(0, None))
result = batched_x(x_batch, y)
print(f"Batch over x: {result.shape}")  # (2, 3, 2)

# Batch of both
y_batch = jnp.array([[4, 5], [6, 7], [8, 9]])
batched_both = vmap(outer_product, in_axes=(0, 0))
# Need matching batch sizes or use nested vmap
nested_batched = vmap(vmap(outer_product, in_axes=(0, None)), in_axes=(None, 0))
result = nested_batched(x_batch, y_batch)
print(f"Nested batch: {result.shape}")  # (2, 3, 3)

# ============================================================================
# EXAMPLE 18: vmap with JIT (Best Practice)
# ============================================================================
print("\n" + "=" * 60)
print("vmap + JIT Combination")
print("=" * 60)

@jit
@vmap
def batched_forward_pass(params, x):
    """JIT-compiled batched forward pass"""
    W1, b1, W2, b2 = params
    h = jnp.maximum(jnp.dot(x, W1) + b1, 0)
    out = jnp.dot(h, W2) + b2
    return out

# Large batch
X_large = random.normal(key, (1000, 10))

# Fast execution
predictions = batched_forward_pass(params, X_large)
print(f"Batch predictions shape: {predictions.shape}")  # (1000, 1)
```

### 9.3.6 pmap for Parallelism

```
pmap (Parallel Map) for Multi-Device:

┌─────────────────────────────────────────────────────────┐
│                    Host (CPU)                           │
│                         │                               │
│         ┌───────────────┼───────────────┐               │
│         │               │               │               │
│         ▼               ▼               ▼               │
│    ┌─────────┐    ┌─────────┐    ┌─────────┐           │
│    │ GPU 0   │    │ GPU 1   │    │ GPU 2   │  ...      │
│    │ data[0] │    │ data[1] │    │ data[2] │           │
│    │ compute │    │ compute │    │ compute │           │
│    └────┬────┘    └────┬────┘    └────┬────┘           │
│         │               │               │               │
│         └───────────────┼───────────────┘               │
│                         │                               │
│              Cross-device communication                 │
│              (lax.psum, lax.pmean, etc.)               │
└─────────────────────────────────────────────────────────┘
```

```python
# ============================================================================
# EXAMPLE 19: Basic pmap Usage
# ============================================================================
print("\n" + "=" * 60)
print("Basic pmap Usage")
print("=" * 60)

# Check available devices
print(f"Number of devices: {jax.device_count()}")
print(f"Devices: {jax.devices()}")

# Simple parallel computation
def compute_on_device(x):
    """Function to run on each device"""
    return x ** 2

# Parallel map
parallel_compute = pmap(compute_on_device)

# Input must have leading dimension = num_devices
num_devices = jax.device_count()
x = jnp.arange(num_devices * 4).reshape(num_devices, 4)

print(f"Input shape: {x.shape}")  # (num_devices, 4)

# Each device processes one row
result = parallel_compute(x)
print(f"Output shape: {result.shape}")  # (num_devices, 4)

# ============================================================================
# EXAMPLE 20: Cross-Device Communication
# ============================================================================
print("\n" + "=" * 60)
print("Cross-Device Communication")
print("=" * 60)

def sum_across_devices(x):
    """Sum values across all devices"""
    return lax.psum(x, 'devices')

parallel_sum = pmap(sum_across_devices, axis_name='devices')

# Each device has different values
x = jnp.arange(num_devices * 4).reshape(num_devices, 4)
print(f"Per-device input:\n{x}")

result = parallel_sum(x)
print(f"After psum (sum across devices):\n{result}")

# ============================================================================
# EXAMPLE 21: Data Parallel Training
# ============================================================================
print("\n" + "=" * 60)
print("Data Parallel Training Pattern")
print("=" * 60)

def create_train_step():
    """Create a data-parallel training step"""

    @pmap
    def train_step(params, x_batch, y_batch, axis_name='devices'):
        """
        Each device:
        1. Computes gradients on its local batch
        2. Averages gradients across devices
        3. Updates parameters (synchronized)
        """

        # Forward pass
        predictions = forward_pass(params, x_batch)
        loss = jnp.mean((predictions - y_batch) ** 2)

        # Compute gradients
        grads = grad(loss_fn)(params, x_batch, y_batch)

        # Average gradients across devices
        grads = lax.pmean(grads, axis_name)

        # Simple SGD update
        learning_rate = 0.01
        updated_params = tuple(
            p - learning_rate * g for p, g in zip(params, grads)
        )

        return updated_params, loss

    return train_step

# Note: This requires multiple devices to run
# On single-device system, pmap behaves like vmap
```

---

## 📚 Deep Learning with Flax

### 9.3.7 Flax Fundamentals

```
Flax Module Lifecycle:
┌─────────────────────────────────────────────────────────┐
│  1. Define Module                                       │
│     class MLP(nn.Module):                               │
│         @nn.compact                                     │
│         def __call__(self, x): ...                      │
└────────────────┬────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────┐
│  2. Initialize Parameters                               │
│     params = model.init(key, example_input)             │
│     # Creates all weights and biases                    │
└────────────────┬────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────┐
│  3. Apply Model                                         │
│     output = model.apply(params, input)                 │
│     # Forward pass with frozen params                   │
└────────────────┬────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────┐
│  4. Train (with TrainState)                             │
│     state = TrainState.create(...)                      │
│     state = state.apply_gradients(grads=grads)          │
└─────────────────────────────────────────────────────────┘
```

```python
from flax import linen as nn
from flax.training import train_state
import optax

# ============================================================================
# EXAMPLE 22: Simple MLP with Flax
# ============================================================================
print("\n" + "=" * 60)
print("Simple MLP with Flax")
print("=" * 60)

class MLP(nn.Module):
    """Multi-Layer Perceptron"""
    features: list
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, x, training=True):
        for i, feat in enumerate(self.features):
            x = nn.Dense(feat)(x)
            if i < len(self.features) - 1:  # No activation on last layer
                x = nn.relu(x)
                if training:
                    x = nn.Dropout(self.dropout_rate)(x, deterministic=False)
        return x

# Create model
model = MLP(features=[128, 64, 10], dropout_rate=0.1)

# Initialize parameters
key = random.PRNGKey(0)
x_example = random.normal(key, (1, 784))  # Single example
params = model.init(key, x_example)

print(f"Parameter shapes:")
for path, value in params.items():
    print(f"  {path}: {value.shape if hasattr(value, 'shape') else 'nested'}")

# Forward pass
logits = model.apply(params, x_example, training=False)
print(f"Output shape: {logits.shape}")  # (1, 10)

# ============================================================================
# EXAMPLE 23: Complete CNN with Flax
# ============================================================================
print("\n" + "=" * 60)
print("Complete CNN with Flax")
print("=" * 60)

class CNN(nn.Module):
    """Convolutional Neural Network for image classification"""
    num_classes: int = 10
    dropout_rate: float = 0.5

    @nn.compact
    def __call__(self, x, training=True):
        """
        Args:
            x: Input images (batch, height, width, channels)
            training: Whether in training mode (for dropout)
        """
        # Block 1: Conv → ReLU → Pool
        x = nn.Conv(features=32, kernel_size=(3, 3), padding='SAME')(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        if training:
            x = nn.Dropout(self.dropout_rate)(x, deterministic=False)

        # Block 2: Conv → ReLU → Pool
        x = nn.Conv(features=64, kernel_size=(3, 3), padding='SAME')(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        if training:
            x = nn.Dropout(self.dropout_rate)(x, deterministic=False)

        # Block 3: Conv → ReLU → Pool
        x = nn.Conv(features=128, kernel_size=(3, 3), padding='SAME')(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        if training:
            x = nn.Dropout(self.dropout_rate)(x, deterministic=False)

        # Flatten
        x = x.reshape((x.shape[0], -1))

        # Fully connected layers
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        if training:
            x = nn.Dropout(self.dropout_rate)(x, deterministic=False)

        # Output layer
        x = nn.Dense(self.num_classes)(x)

        return x

# Create and test CNN
cnn_model = CNN(num_classes=10, dropout_rate=0.5)

# Example input: batch of 4 RGB images 32x32
key = random.PRNGKey(0)
x_images = random.normal(key, (4, 32, 32, 3))

# Initialize
params = cnn_model.init(key, x_images)

# Forward pass
logits = cnn_model.apply(params, x_images, training=False)
print(f"CNN output shape: {logits.shape}")  # (4, 10)

# Count parameters
total_params = sum(
    jnp.prod(jnp.array(v.shape))
    for v in jax.tree_util.tree_leaves(params)
)
print(f"Total parameters: {total_params:,}")

# ============================================================================
# EXAMPLE 24: Residual Block with Flax
# ============================================================================
print("\n" + "=" * 60)
print("Residual Block with Flax")
print("=" * 60)

class ResidualBlock(nn.Module):
    """Residual block with skip connection"""
    features: int

    @nn.compact
    def __call__(self, x):
        """
        Residual connection:
        output = F(x) + x
        where F(x) = Conv → BN → ReLU → Conv → BN
        """
        residual = x

        # First conv
        x = nn.Conv(self.features, kernel_size=(3, 3), padding='SAME')(x)
        x = nn.BatchNorm(use_running_average=False)(x)
        x = nn.relu(x)

        # Second conv
        x = nn.Conv(self.features, kernel_size=(3, 3), padding='SAME')(x)
        x = nn.BatchNorm(use_running_average=False)(x)

        # Projection shortcut if dimensions don't match
        if residual.shape[-1] != self.features:
            residual = nn.Conv(self.features, kernel_size=(1, 1), padding='SAME')(residual)

        # Add residual and ReLU
        x = nn.relu(x + residual)

        return x

class SimpleResNet(nn.Module):
    """Simple ResNet with multiple residual blocks"""
    num_classes: int = 10
    block_features: list = (64, 128, 256)

    @nn.compact
    def __call__(self, x, training=True):
        # Initial conv
        x = nn.Conv(32, kernel_size=(3, 3), padding='SAME')(x)
        x = nn.BatchNorm(use_running_average=not training)(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))

        # Residual blocks
        for features in self.block_features:
            x = ResidualBlock(features)(x)
            x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))

        # Global average pooling
        x = jnp.mean(x, axis=(1, 2))

        # Classifier
        x = nn.Dense(self.num_classes)(x)

        return x

# Test ResNet
resnet = SimpleResNet(num_classes=10)
key = random.PRNGKey(0)
x_images = random.normal(key, (4, 32, 32, 3))
params = resnet.init(key, x_images)
logits = resnet.apply(params, x_images, training=False)
print(f"ResNet output shape: {logits.shape}")  # (4, 10)

# ============================================================================
# EXAMPLE 25: Custom Module with Multiple Outputs
# ============================================================================
print("\n" + "=" * 60)
print("Custom Module with Multiple Outputs")
print("=" * 60)

class MultiTaskModel(nn.Module):
    """Model with shared backbone and multiple task heads"""
    num_classes_task1: int = 10
    num_classes_task2: int = 5

    @nn.compact
    def __call__(self, x, training=True):
        # Shared backbone
        x = nn.Conv(32, (3, 3), padding='SAME')(x)
        x = nn.relu(x)
        x = nn.max_pool(x, (2, 2))

        x = nn.Conv(64, (3, 3), padding='SAME')(x)
        x = nn.relu(x)
        x = nn.max_pool(x, (2, 2))

        # Flatten
        x = x.reshape((x.shape[0], -1))

        # Shared FC layers
        x = nn.Dense(128)(x)
        x = nn.relu(x)

        # Task-specific heads
        task1_output = nn.Dense(self.num_classes_task1, name='task1_head')(x)
        task2_output = nn.Dense(self.num_classes_task2, name='task2_head')(x)

        return {'task1': task1_output, 'task2': task2_output}

# Test multi-task model
multi_model = MultiTaskModel()
key = random.PRNGKey(0)
x_images = random.normal(key, (4, 32, 32, 3))
params = multi_model.init(key, x_images)
outputs = multi_model.apply(params, x_images, training=False)

print(f"Task 1 output shape: {outputs['task1'].shape}")  # (4, 10)
print(f"Task 2 output shape: {outputs['task2'].shape}")  # (4, 5)
```

### 9.3.8 Complete Training Loop with Optax

```
Complete Training Pipeline:
┌─────────────────────────────────────────────────────────┐
│  1. Create Model                                        │
│     model = CNN(num_classes=10)                         │
└────────┬────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│  2. Initialize TrainState                               │
│     state = TrainState.create(                          │
│         apply_fn=model.apply,                           │
│         params=model.init(...),                         │
│         tx=optax.adam(0.001)                            │
│     )                                                   │
└────────┬────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│  3. Define Training Step                                │
│     @jit                                                │
│     def train_step(state, batch):                       │
│         loss, grads = ...                               │
│         return state.apply_gradients(grads=grads)       │
└────────┬────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│  4. Training Loop                                       │
│     for epoch in range(num_epochs):                     │
│         for batch in train_loader:                      │
│             state = train_step(state, batch)            │
└─────────────────────────────────────────────────────────┘
```

```python
# ============================================================================
# EXAMPLE 26: Complete Training Loop with Flax + Optax
# ============================================================================
print("\n" + "=" * 60)
print("Complete Training Loop")
print("=" * 60)

# Define model
class Classifier(nn.Module):
    num_classes: int = 10

    @nn.compact
    def __call__(self, x, training=True):
        x = nn.Conv(32, (3, 3), padding='SAME')(x)
        x = nn.relu(x)
        x = nn.max_pool(x, (2, 2))
        x = nn.Conv(64, (3, 3), padding='SAME')(x)
        x = nn.relu(x)
        x = nn.max_pool(x, (2, 2))
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        x = nn.Dense(self.num_classes)(x)
        return x

# Custom TrainState for BatchNorm
class TrainState(train_state.TrainState):
    batch_stats: dict

def create_train_state(
    model: nn.Module,
    learning_rate: float,
    example_inputs: jnp.ndarray
) -> TrainState:
    """Initialize training state"""
    key = random.PRNGKey(0)

    # Initialize parameters
    params = model.init(key, example_inputs)

    # Separate params for BatchNorm (if used)
    params_dict = params.unfreeze() if hasattr(params, 'unfreeze') else params
    if isinstance(params_dict, dict) and 'batch_stats' in params_dict:
        model_params = params_dict['params']
        batch_stats = params_dict['batch_stats']
    else:
        model_params = params
        batch_stats = {}

    # Create optimizer
    tx = optax.adam(learning_rate)

    return TrainState.create(
        apply_fn=model.apply,
        params=model_params,
        tx=tx,
        batch_stats=batch_stats
    )

# Training step
@jit
def train_step(state: TrainState, batch: dict, dropout_rng: jnp.ndarray):
    """Single training step"""

    def loss_fn(params):
        # Forward pass
        logits, updates = state.apply_fn(
            {'params': params, 'batch_stats': state.batch_stats},
            batch['image'],
            training=True,
            rngs={'dropout': dropout_rng},
            mutable=['batch_stats']
        )

        # Compute loss
        one_hot = jax.nn.one_hot(batch['label'], 10)
        loss = jnp.mean(optax.softmax_cross_entropy(logits, one_hot))

        return loss, updates

    # Compute gradients
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, updates), grads = grad_fn(state.params)

    # Apply gradients
    state = state.apply_gradients(grads=grads)

    # Update batch stats
    if updates:
        state = state.replace(batch_stats=updates['batch_stats'])

    return state, loss

# Evaluation step
@jit
def eval_step(state: TrainState, batch: dict):
    """Single evaluation step"""
    logits = state.apply_fn(
        {'params': state.params, 'batch_stats': state.batch_stats},
        batch['image'],
        training=False
    )
    predictions = jnp.argmax(logits, axis=-1)
    accuracy = jnp.mean(predictions == batch['label'])
    return accuracy

# Create model and state
model = Classifier(num_classes=10)
example_input = jnp.ones((1, 32, 32, 3))
state = create_train_state(model, learning_rate=0.001, example_inputs=example_input)

print(f"Initial state created")
print(f"  Optimizer: Adam")
print(f"  Learning rate: 0.001")

# Simulated training loop (single iteration for demo)
key = random.PRNGKey(0)
key, dropout_rng = random.split(key)

# Create dummy batch
dummy_batch = {
    'image': random.normal(key, (32, 32, 32, 3)),
    'label': random.randint(key, (32,), 0, 10)
}

# Single training step
state, loss = train_step(state, dummy_batch, dropout_rng)
print(f"  Loss after one step: {loss:.4f}")

# ============================================================================
# EXAMPLE 27: Learning Rate Schedules with Optax
# ============================================================================
print("\n" + "=" * 60)
print("Learning Rate Schedules")
print("=" * 60)

# Linear decay
linear_schedule = optax.linear_schedule(
    init_value=0.001,
    end_value=0.0001,
    transition_steps=10000
)

# Cosine decay
cosine_schedule = optax.cosine_decay_schedule(
    init_value=0.001,
    decay_steps=10000
)

# Exponential decay
exp_schedule = optax.exponential_decay(
    init_value=0.001,
    decay_steps=10000,
    decay_rate=0.96,
    staircase=True
)

# Warmup + cosine decay
warmup_cosine = optax.warmup_cosine_decay_schedule(
    warmup_steps=1000,
    peak_value=0.001,
    end_value=0.0001,
    transition_steps=10000
)

# Print schedule values
print(f"Learning rate schedules (first 5 steps):")
print(f"  Linear: {[linear_schedule(i) for i in range(5)]}")
print(f"  Cosine: {[cosine_schedule(i) for i in range(5)]}")
print(f"  Warmup+Cosine: {[warmup_cosine(i) for i in range(5)]}")

# ============================================================================
# EXAMPLE 28: Gradient Clipping and Accumulation
# ============================================================================
print("\n" + "=" * 60)
print("Gradient Clipping and Accumulation")
print("=" * 60)

# Gradient clipping
clipped_adam = optax.chain(
    optax.clip_by_global_norm(1.0),  # Clip gradients
    optax.adam(learning_rate=0.001)
)

# Gradient accumulation (accumulate 4 steps before updating)
accumulating_optimizer = optax.MultiSteps(
    optax.adam(0.001),
    every_k_schedule=4
)

# AdamW with weight decay
adamw = optax.adamw(
    learning_rate=0.001,
    weight_decay=0.01,
    b1=0.9,
    b2=0.999
)

print(f"Optimizers configured:")
print(f"  - Adam with gradient clipping")
print(f"  - Adam with gradient accumulation (4 steps)")
print(f"  - AdamW with weight decay")

# ============================================================================
# EXAMPLE 29: Complete Training with Metrics
# ============================================================================
print("\n" + "=" * 60)
print("Training with Metrics")
print("=" * 60)

class TrainingMetrics:
    """Track training metrics"""

    def __init__(self):
        self.loss_history = []
        self.accuracy_history = []

    def update(self, loss, accuracy):
        self.loss_history.append(loss)
        self.accuracy_history.append(accuracy)

    def summary(self):
        return {
            'final_loss': self.loss_history[-1] if self.loss_history else None,
            'final_accuracy': self.accuracy_history[-1] if self.accuracy_history else None,
            'best_accuracy': max(self.accuracy_history) if self.accuracy_history else None
        }

# Simulated training with metrics
metrics = TrainingMetrics()

# Dummy training loop (for demonstration)
num_epochs = 3
steps_per_epoch = 10

for epoch in range(num_epochs):
    epoch_loss = 0.0
    epoch_acc = 0.0

    for step in range(steps_per_epoch):
        # Simulated training step
        key, dropout_rng = random.split(key)
        state, loss = train_step(state, dummy_batch, dropout_rng)

        # Simulated evaluation
        acc = eval_step(state, dummy_batch)

        epoch_loss += loss
        epoch_acc += acc

    # Average metrics for epoch
    avg_loss = epoch_loss / steps_per_epoch
    avg_acc = epoch_acc / steps_per_epoch

    metrics.update(float(avg_loss), float(avg_acc))

    print(f"Epoch {epoch + 1}/{num_epochs}: "
          f"Loss = {avg_loss:.4f}, Accuracy = {avg_acc:.4f}")

print(f"\nTraining Summary:")
summary = metrics.summary()
print(f"  Final Loss: {summary['final_loss']:.4f}")
print(f"  Final Accuracy: {summary['final_accuracy']:.4f}")
print(f"  Best Accuracy: {summary['best_accuracy']:.4f}")
```

---

## 📚 Optax Optimizers and Loss Functions

### 9.3.9 Complete Optax Guide

```python
# ============================================================================
# EXAMPLE 30: Optax Optimizer Comparison
# ============================================================================
print("\n" + "=" * 60)
print("Optax Optimizer Comparison")
print("=" * 60)

optimizers = {
    'SGD': optax.sgd(learning_rate=0.01, momentum=0.9),
    'Adam': optax.adam(learning_rate=0.001),
    'AdamW': optax.adamw(learning_rate=0.001, weight_decay=0.01),
    'RMSprop': optax.rmsprop(learning_rate=0.01),
    'Adagrad': optax.adagrad(learning_rate=0.01),
    'Adafactor': optax.adafactor(learning_rate=0.001)
}

print("Available optimizers:")
for name, opt in optimizers.items():
    print(f"  - {name}: {type(opt).__name__}")

# ============================================================================
# EXAMPLE 31: Custom Loss Functions
# ============================================================================
print("\n" + "=" * 60)
print("Custom Loss Functions")
print("=" * 60)

# Built-in loss functions
logits = jnp.array([[2.0, 0.5, -1.0], [0.5, 2.0, -1.0]])
labels = jnp.array([0, 1])

# Cross-entropy loss
ce_loss = optax.softmax_cross_entropy(logits, jax.nn.one_hot(labels, 3))
print(f"Cross-entropy loss: {ce_loss}")

# Sigmoid cross-entropy (for multi-label)
sigmoid_logits = jnp.array([[0.8, 0.3, -0.5], [0.2, 0.9, -0.3]])
multi_labels = jnp.array([[1, 0, 0], [0, 1, 0]])
sigmoid_loss = optax.sigmoid_binary_cross_entropy(sigmoid_logits, multi_labels)
print(f"Sigmoid cross-entropy: {sigmoid_loss}")

# Huber loss (robust to outliers)
predictions = jnp.array([1.0, 2.0, 3.0])
targets = jnp.array([1.1, 1.8, 3.5])
huber = optax.huber_loss(predictions, targets, delta=1.0)
print(f"Huber loss: {huber}")

# L2 loss
l2 = optax.l2_loss(predictions, targets)
print(f"L2 loss: {l2}")

# ============================================================================
# EXAMPLE 32: Custom Loss Implementation
# ============================================================================
print("\n" + "=" * 60)
print("Custom Loss Implementation")
print("=" * 60)

def focal_loss(logits, labels, alpha=0.25, gamma=2.0):
    """
    Focal Loss for handling class imbalance.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        logits: Model predictions (unnormalized)
        labels: True labels (integer or one-hot)
        alpha: Weighting factor for rare class
        gamma: Focusing parameter
    """
    # Convert to one-hot if needed
    if labels.ndim == 1:
        labels = jax.nn.one_hot(labels, logits.shape[-1])

    # Get probabilities
    probs = jax.nn.softmax(logits)

    # Get probability of true class
    p_t = jnp.sum(probs * labels, axis=-1)

    # Compute focal weight
    focal_weight = alpha * (1 - p_t) ** gamma

    # Compute cross-entropy
    ce = -jnp.log(p_t + 1e-8)

    # Apply focal weight
    loss = focal_weight * ce

    return jnp.mean(loss)

# Test focal loss
logits = jnp.array([[2.0, 0.5, -1.0], [0.5, 2.0, -1.0]])
labels = jnp.array([0, 1])

focal = focal_loss(logits, labels)
print(f"Focal loss: {focal:.4f}")

# ============================================================================
# EXAMPLE 33: Metrics Implementation
# ============================================================================
print("\n" + "=" * 60)
print("Metrics Implementation")
print("=" * 60)

def accuracy(predictions, labels):
    """Compute classification accuracy"""
    pred_labels = jnp.argmax(predictions, axis=-1)
    return jnp.mean(pred_labels == labels)

def precision_recall(predictions, labels, pos_label=1):
    """Compute precision and recall"""
    pred_labels = jnp.argmax(predictions, axis=-1)

    # True positives, false positives, false negatives
    tp = jnp.sum((pred_labels == pos_label) & (labels == pos_label))
    fp = jnp.sum((pred_labels == pos_label) & (labels != pos_label))
    fn = jnp.sum((pred_labels != pos_label) & (labels == pos_label))

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)

    return precision, recall

def f1_score(predictions, labels, pos_label=1):
    """Compute F1 score"""
    precision, recall = precision_recall(predictions, labels, pos_label)
    return 2 * precision * recall / (precision + recall + 1e-8)

# Test metrics
predictions = jnp.array([[0.8, 0.2], [0.3, 0.7], [0.6, 0.4], [0.1, 0.9]])
labels = jnp.array([0, 1, 0, 1])

print(f"Accuracy: {accuracy(predictions, labels):.4f}")
print(f"F1 Score: {f1_score(predictions, labels):.4f}")
```

---

## 🐛 Debugging Guide

### Common JAX Errors and Solutions

```python
# ============================================================================
# ERROR 1: Using Python Side Effects in JIT Functions
# ============================================================================
"""
❌ WRONG:
@jit
def bad_function(x):
    print(f"Processing {x}")  # Error! Side effects not allowed
    return x ** 2

✅ CORRECT:
@jit
def good_function(x):
    return x ** 2

# Print outside JIT
result = good_function(x)
print(f"Result: {result}")
"""

# ============================================================================
# ERROR 2: In-Place Array Modification
# ============================================================================
"""
❌ WRONG:
x = jnp.array([1, 2, 3])
x[0] = 5  # Error! JAX arrays are immutable

✅ CORRECT:
x = jnp.array([1, 2, 3])
x = x.at[0].set(5)  # Use .at[].set()
"""

# ============================================================================
# ERROR 3: Random Key Reuse
# ============================================================================
"""
❌ WRONG:
key = random.PRNGKey(0)
random1 = random.normal(key, (10,))
random2 = random.normal(key, (10,))  # Same random numbers!

✅ CORRECT:
key = random.PRNGKey(0)
key, subkey1 = random.split(key)
key, subkey2 = random.split(key)
random1 = random.normal(subkey1, (10,))
random2 = random.normal(subkey2, (10,))
"""

# ============================================================================
# ERROR 4: Python Conditionals on Traced Values
# ============================================================================
"""
❌ WRONG:
@jit
def conditional(x):
    if x > 0:  # Error! Can't use Python if on traced values
        return x ** 2
    else:
        return -x

✅ CORRECT:
from jax import lax

@jit
def conditional(x):
    return lax.cond(
        x > 0,
        lambda x: x ** 2,
        lambda x: -x,
        x
    )
"""

# ============================================================================
# ERROR 5: Incorrect vmap in_axes
# ============================================================================
"""
❌ WRONG:
def func(x, y):
    return x + y

batched = vmap(func, in_axes=(0, 0))
result = batched(jnp.ones((5, 10)), jnp.ones((10,)))  # Shape mismatch!

✅ CORRECT:
batched = vmap(func, in_axes=(0, None))  # Broadcast y
result = batched(jnp.ones((5, 10)), jnp.ones((10,)))
"""

# ============================================================================
# ERROR 6: Forgetting to Unfreeze Flax Params
# ============================================================================
"""
❌ WRONG:
params = model.init(key, x)
params['batch_stats'] = new_stats  # Error! FrozenDict

✅ CORRECT:
params = model.init(key, x)
params = params.unfreeze()
params['batch_stats'] = new_stats
params = flax.core.freeze(params)
"""

# ============================================================================
# ERROR 7: Gradient Explosion
# ============================================================================
"""
❌ WRONG:
optimizer = optax.adam(0.001)  # No gradient clipping

✅ CORRECT:
optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),  # Clip gradients
    optax.adam(0.001)
)
"""

# ============================================================================
# ERROR 8: Memory Leak with jit
# ============================================================================
"""
❌ WRONG:
@jit
def train_step(params, x, y):
    # Creating new JIT'd function inside
    @jit
    def inner(x):
        return x ** 2
    return loss(params, x, y)

✅ CORRECT:
@jit
def inner(x):
    return x ** 2

@jit
def train_step(params, x, y):
    return loss(params, x, y)
"""
```

---

## 🏆 Real-World Project: Image Classification with Flax

### Problem Statement
Build a complete image classification system for CIFAR-10 dataset using Flax and Optax.

### Dataset
- **Name**: CIFAR-10
- **Classes**: 10 (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)
- **Size**: 60,000 images (50,000 train, 10,000 test)
- **Format**: 32x32 RGB images

### Complete Implementation

```python
# ============================================================================
# CIFAR-10 Classification with Flax
# ============================================================================
"""
Complete implementation including:
1. Data loading and preprocessing
2. Model definition (ResNet-style)
3. Training loop with metrics
4. Evaluation and visualization
"""

import jax
import jax.numpy as jnp
from jax import random, grad, jit, value_and_grad
from flax import linen as nn
from flax.training import train_state
import optax
from typing import Tuple, Dict, Any

# ============================================================================
# 1. Model Definition
# ============================================================================

class ResidualBlock(nn.Module):
    """Residual block for CIFAR-10"""
    features: int
    strides: Tuple[int, int] = (1, 1)

    @nn.compact
    def __call__(self, x, training=True):
        residual = x

        # Conv1
        x = nn.Conv(
            self.features,
            kernel_size=(3, 3),
            strides=self.strides,
            padding='SAME',
            use_bias=False
        )(x)
        x = nn.BatchNorm(use_running_average=not training)(x)
        x = nn.relu(x)

        # Conv2
        x = nn.Conv(
            self.features,
            kernel_size=(3, 3),
            padding='SAME',
            use_bias=False
        )(x)
        x = nn.BatchNorm(use_running_average=not training)(x)

        # Skip connection
        if self.strides != (1, 1) or residual.shape[-1] != self.features:
            residual = nn.Conv(
                self.features,
                kernel_size=(1, 1),
                strides=self.strides,
                use_bias=False
            )(residual)
            residual = nn.BatchNorm(use_running_average=not training)(residual)

        x = nn.relu(x + residual)
        return x


class CIFAR10ResNet(nn.Module):
    """ResNet for CIFAR-10 classification"""
    num_classes: int = 10
    block_features: Tuple[int] = (32, 64, 128)

    @nn.compact
    def __call__(self, x, training=True):
        # Initial conv
        x = nn.Conv(32, kernel_size=(3, 3), padding='SAME', use_bias=False)(x)
        x = nn.BatchNorm(use_running_average=not training)(x)
        x = nn.relu(x)

        # Residual blocks
        for i, features in enumerate(self.block_features):
            strides = (2, 2) if i > 0 else (1, 1)
            x = ResidualBlock(features, strides=strides)(x, training)
            x = ResidualBlock(features, strides=(1, 1))(x, training)

        # Global average pooling
        x = jnp.mean(x, axis=(1, 2))

        # Classifier
        x = nn.Dense(self.num_classes)(x)

        return x


# ============================================================================
# 2. Training State
# ============================================================================

class TrainState(train_state.TrainState):
    batch_stats: Dict

def create_train_state(
    model: nn.Module,
    learning_rate: float,
    example_inputs: jnp.ndarray
) -> TrainState:
    key = random.PRNGKey(0)
    params = model.init(key, example_inputs)

    params_dict = params.unfreeze()
    model_params = params_dict['params']
    batch_stats = params_dict.get('batch_stats', {})

    tx = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(
            learning_rate=learning_rate,
            weight_decay=0.0001
        )
    )

    return TrainState.create(
        apply_fn=model.apply,
        params=model_params,
        tx=tx,
        batch_stats=batch_stats
    )


# ============================================================================
# 3. Training and Evaluation Steps
# ============================================================================

@jit
def train_step(state: TrainState, batch: Dict, dropout_rng: jnp.ndarray):
    def loss_fn(params):
        logits, updates = state.apply_fn(
            {'params': params, 'batch_stats': state.batch_stats},
            batch['image'],
            training=True,
            rngs={'dropout': dropout_rng},
            mutable=['batch_stats']
        )

        one_hot = jax.nn.one_hot(batch['label'], 10)
        loss = jnp.mean(optax.softmax_cross_entropy(logits, one_hot))

        return loss, updates

    grad_fn = value_and_grad(loss_fn, has_aux=True)
    (loss, updates), grads = grad_fn(state.params)

    state = state.apply_gradients(grads=grads)

    if updates:
        state = state.replace(batch_stats=updates['batch_stats'])

    return state, loss


@jit
def eval_step(state: TrainState, batch: Dict) -> Dict:
    logits = state.apply_fn(
        {'params': state.params, 'batch_stats': state.batch_stats},
        batch['image'],
        training=False
    )

    predictions = jnp.argmax(logits, axis=-1)
    accuracy = jnp.mean(predictions == batch['label'])

    return {'accuracy': accuracy, 'loss': 0.0}


# ============================================================================
# 4. Training Loop
# ============================================================================

def train_model(
    train_data: Dict,
    val_data: Dict,
    num_epochs: int = 50,
    batch_size: int = 128,
    learning_rate: float = 0.001
):
    """Complete training loop"""

    # Create model and state
    model = CIFAR10ResNet(num_classes=10)
    example_input = jnp.ones((1, 32, 32, 3))
    state = create_train_state(model, learning_rate, example_input)

    # Training metrics
    history = {
        'train_loss': [],
        'train_accuracy': [],
        'val_accuracy': []
    }

    key = random.PRNGKey(0)

    print("Starting training...")
    print("=" * 60)

    for epoch in range(num_epochs):
        # Training
        epoch_loss = 0.0
        epoch_accuracy = 0.0
        num_batches = 0

        for batch_idx in range(len(train_data['images']) // batch_size):
            start_idx = batch_idx * batch_size
            end_idx = start_idx + batch_size

            batch = {
                'image': train_data['images'][start_idx:end_idx],
                'label': train_data['labels'][start_idx:end_idx]
            }

            key, dropout_rng = random.split(key)
            state, loss = train_step(state, batch, dropout_rng)

            epoch_loss += loss
            num_batches += 1

        # Validation
        val_metrics = {'accuracy': 0.0, 'count': 0}
        for batch_idx in range(len(val_data['images']) // batch_size):
            start_idx = batch_idx * batch_size
            end_idx = start_idx + batch_size

            batch = {
                'image': val_data['images'][start_idx:end_idx],
                'label': val_data['labels'][start_idx:end_idx]
            }

            metrics = eval_step(state, batch)
            val_metrics['accuracy'] += metrics['accuracy']
            val_metrics['count'] += 1

        # Average metrics
        avg_train_loss = epoch_loss / num_batches
        avg_val_accuracy = val_metrics['accuracy'] / val_metrics['count']

        history['train_loss'].append(float(avg_train_loss))
        history['val_accuracy'].append(float(avg_val_accuracy))

        print(f"Epoch {epoch + 1:3d}/{num_epochs}: "
              f"Loss = {avg_train_loss:.4f}, "
              f"Val Accuracy = {avg_val_accuracy:.4f}")

    print("=" * 60)
    print("Training complete!")

    return state, history


# ============================================================================
# 5. Example Usage
# ============================================================================

if __name__ == "__main__":
    # Generate synthetic data (replace with actual CIFAR-10 loading)
    key = random.PRNGKey(42)

    train_data = {
        'images': random.normal(key, (50000, 32, 32, 3)),
        'labels': random.randint(key, (50000,), 0, 10)
    }

    val_data = {
        'images': random.normal(key, (10000, 32, 32, 3)),
        'labels': random.randint(key, (10000,), 0, 10)
    }

    # Train model
    state, history = train_model(
        train_data,
        val_data,
        num_epochs=10,  # Use 50+ for actual training
        batch_size=128,
        learning_rate=0.001
    )

    # Plot training curves (using matplotlib)
    try:
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        ax1.plot(history['train_loss'])
        ax1.set_title('Training Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')

        ax2.plot(history['val_accuracy'])
        ax2.set_title('Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')

        plt.tight_layout()
        plt.savefig('training_curves.png')
        print("\nTraining curves saved to training_curves.png")

    except ImportError:
        print("\nMatplotlib not available for plotting")
```

---

## 📝 Practice Problems with Solutions

### Level 1: Basic

#### Problem 1: JAX Array Operations
**Task**: Create JAX arrays and perform basic operations.

**Solution**:
```python
import jax.numpy as jnp
from jax import random

# Create arrays
a = jnp.array([1, 2, 3, 4, 5])
b = jnp.array([5, 4, 3, 2, 1])

# Operations
print(f"Addition: {a + b}")
print(f"Multiplication: {a * b}")
print(f"Dot product: {jnp.dot(a, b)}")
print(f"Sum: {jnp.sum(a)}")
print(f"Mean: {jnp.mean(a)}")

# Array creation
zeros = jnp.zeros((3, 3))
ones = jnp.ones((2, 4))
arange = jnp.arange(0, 10, 2)
linspace = jnp.linspace(0, 1, 5)
```

#### Problem 2: Gradient Computation
**Task**: Compute derivatives using grad.

**Solution**:
```python
from jax import grad

def f(x):
    return x**3 + 2*x**2 + x + 1

# First derivative
df = grad(f)
print(f"f'(2) = {df(2.0)}")  # 3*4 + 2*4 + 1 = 21

# Second derivative
d2f = grad(grad(f))
print(f"f''(2) = {d2f(2.0)}")  # 6*2 + 4 = 16
```

#### Problem 3: JIT Compilation
**Task**: Speed up a function with jit.

**Solution**:
```python
from jax import jit
import time

def slow_sum(x):
    total = 0.0
    for i in range(len(x)):
        for j in range(len(x)):
            total += x[i] * x[j]
    return total

fast_sum = jit(slow_sum)

x = jnp.ones(1000)

# Warmup
_ = fast_sum(x)

# Benchmark
start = time.time()
result = fast_sum(x)
print(f"JIT time: {time.time() - start:.4f}s")
```

#### Problem 4: Simple Flax Model
**Task**: Build and use a Flax MLP.

**Solution**:
```python
from flax import linen as nn

class SimpleMLP(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(64)(x)
        x = nn.relu(x)
        x = nn.Dense(10)(x)
        return x

model = SimpleMLP()
key = random.PRNGKey(0)
x = random.normal(key, (1, 20))
params = model.init(key, x)
output = model.apply(params, x)
print(f"Output shape: {output.shape}")
```

#### Problem 5: Optax Optimizer
**Task**: Set up an optimizer.

**Solution**:
```python
import optax

# Create optimizer
optimizer = optax.adam(learning_rate=0.001)

# Initialize optimizer state
params = {'W': jnp.ones((10, 5)), 'b': jnp.zeros(5)}
opt_state = optimizer.init(params)

# Compute gradients
loss_fn = lambda p: jnp.sum(p['W'] ** 2)
grads = grad(loss_fn)(params)

# Update parameters
updates, new_opt_state = optimizer.update(grads, opt_state)
new_params = optax.apply_updates(params, updates)
```

### Level 2: Intermediate

#### Problem 1: vmap for Batched Operations
**Task**: Vectorize a function.

**Solution**:
```python
from jax import vmap

def apply_activation(x, func='relu'):
    if func == 'relu':
        return jnp.maximum(x, 0)
    elif func == 'sigmoid':
        return 1 / (1 + jnp.exp(-x))

# Vectorize over batch
batched_apply = vmap(apply_activation, in_axes=(0, None))

x = jnp.array([[1, -2, 3], [-1, 2, -3], [0, 0, 0]])
result = batched_apply(x, 'relu')
print(f"Batched ReLU:\n{result}")
```

#### Problem 2: Flax Training Loop
**Task**: Implement complete training.

**Solution**: See Example 26 above for complete implementation.

#### Problem 3: grad + jit Combination
**Task**: Combine transformations.

**Solution**:
```python
@jit
@grad
def loss_grad(params, x, y):
    predictions = jnp.dot(x, params['W']) + params['b']
    return jnp.mean((predictions - y) ** 2)

# Fast gradient computation
gradients = loss_grad(params, x_batch, y_batch)
```

#### Problem 4: Learning Rate Schedules
**Task**: Implement LR scheduling.

**Solution**: See Example 27 above.

#### Problem 5: CNN with Flax
**Task**: Build CNN for image classification.

**Solution**: See Example 23 above.

### Level 3: Advanced

#### Problem 1: Multi-Device Training
**Task**: Use pmap for parallel training.

**Solution**: See Examples 19-21 above.

#### Problem 2: Custom Gradient
**Task**: Implement custom JVP.

**Solution**:
```python
from jax import custom_jvp

@custom_jvp
def stable_softmax(x):
    x_max = jnp.max(x, axis=-1, keepdims=True)
    exp_x = jnp.exp(x - x_max)
    return exp_x / jnp.sum(exp_x, axis=-1, keepdims=True)

@stable_softmax.jvp
def stable_softmax_jvp(primals, tangents):
    (x,) = primals
    (x_dot,) = tangents
    sm = stable_softmax(x)
    return sm, sm * (x_dot - jnp.sum(sm * x_dot, axis=-1, keepdims=True))
```

#### Problem 3: Transformer with Flax
**Task**: Build attention model.

**Solution**:
```python
class SelfAttention(nn.Module):
    embed_dim: int
    num_heads: int = 8

    @nn.compact
    def __call__(self, x):
        batch_size, seq_len, _ = x.shape

        # Q, K, V projections
        q = nn.Dense(self.embed_dim)(x)
        k = nn.Dense(self.embed_dim)(x)
        v = nn.Dense(self.embed_dim)(x)

        # Scaled dot-product attention
        attn_weights = jnp.matmul(q, k.transpose(0, 2, 1)) / jnp.sqrt(self.embed_dim)
        attn_weights = nn.softmax(attn_weights, axis=-1)

        # Apply attention to values
        output = jnp.matmul(attn_weights, v)

        return nn.Dense(self.embed_dim)(output)
```

#### Problem 4: Profile JAX Code
**Task**: Use JAX profiler.

**Solution**:
```python
from jax import profiler
import tensorflow as tf

# Start profiling
profiler.start_profile('/tmp/jax_profile')

# Run code to profile
result = train_step(state, batch, dropout_rng)

# Stop profiling
profiler.stop_profile()

# View in TensorBoard
# tensorboard --logdir=/tmp/jax_profile
```

#### Problem 5: Memory-Efficient Training
**Task**: Implement gradient checkpointing.

**Solution**:
```python
from jax.checkpoint import checkpoint

@checkpoint
def expensive_layer(x, params):
    """Memory-efficient layer"""
    for i in range(10):  # Many operations
        x = jnp.dot(x, params[f'W{i}'])
        x = nn.relu(x)
    return x

# Recomputes during backward pass to save memory
```

---

## 📊 Summary Tables

### JAX Transformations Quick Reference

| Transformation | Purpose | Syntax | Example |
|----------------|---------|--------|---------|
| **grad** | Automatic differentiation | `grad(f)` | `df = grad(lambda x: x**2)` |
| **value_and_grad** | Value + gradient | `value_and_grad(f)` | `val, grad = value_and_grad(f)(x)` |
| **jacobian** | Jacobian matrix | `jacobian(f)` | `J = jacobian(f_vector)(x)` |
| **hessian** | Hessian matrix | `hessian(f)` | `H = hessian(f_scalar)(x)` |
| **jit** | JIT compilation | `jit(f)` | `fast_f = jit(slow_f)` |
| **vmap** | Vectorization | `vmap(f, in_axes)` | `batched = vmap(f, in_axes=0)` |
| **pmap** | Parallelization | `pmap(f, axis_name)` | `parallel = pmap(f, axis_name='devices')` |

### Flax Common Modules

| Module | Purpose | Key Parameters |
|--------|---------|----------------|
| **Dense** | Fully connected layer | features |
| **Conv** | Convolutional layer | features, kernel_size, strides |
| **BatchNorm** | Batch normalization | use_running_average |
| **LayerNorm** | Layer normalization | - |
| **Dropout** | Dropout regularization | rate |
| **Embedding** | Embedding layer | num_embeddings, features |
| **LSTM** | LSTM recurrent layer | features |
| **MultiHeadAttention** | Self-attention | num_heads, features |

### Optax Components

| Component | Purpose | Usage |
|-----------|---------|-------|
| **adam** | Adam optimizer | `optax.adam(lr)` |
| **adamw** | Adam with weight decay | `optax.adamw(lr, weight_decay)` |
| **sgd** | SGD with momentum | `optax.sgd(lr, momentum)` |
| **chain** | Combine transforms | `optax.chain(clip, adam)` |
| **clip_by_global_norm** | Gradient clipping | `optax.clip_by_global_norm(max_norm)` |
| **linear_schedule** | Linear LR decay | `optax.linear_schedule(init, end, steps)` |
| **cosine_decay** | Cosine LR schedule | `optax.cosine_decay_schedule(init, steps)` |
| **warmup_cosine** | Warmup + cosine | `optax.warmup_cosine_decay_schedule(...)` |

---

## 🔗 Related Topics
- [[01-TensorFlow-Keras]] - TensorFlow framework comparison
- [[02-PyTorch]] - PyTorch framework comparison
- [[05-Convolutional-Neural-Networks]] - CNN architectures in Flax
- [[06-Recurrent-Neural-Networks]] - RNN implementations

---

**Status:** ✅ Complete with comprehensive examples
**Next:** Review all frameworks and start Phase 4 (Specialization)
