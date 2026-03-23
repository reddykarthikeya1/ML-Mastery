# 3.2 NumPy

## 🎯 Quick Overview
- **NumPy arrays**: Efficient numerical arrays
- **Broadcasting**: Vectorized operations
- **Linear algebra**: Matrix operations
- **Foundation for**: All numerical computing in Python

---

## 1. NumPy Arrays

### Creating Arrays

```python
import numpy as np

# From lists
arr = np.array([1, 2, 3, 4, 5])
matrix = np.array([[1, 2], [3, 4]])

# Built-in functions
np.zeros((3, 4))           # Array of zeros
np.ones((2, 3))            # Array of ones
np.full((2, 2), 7)         # Array filled with value
np.eye(3)                  # Identity matrix
np.arange(0, 10, 2)        # Like range: [0, 2, 4, 6, 8]
np.linspace(0, 1, 5)       # Evenly spaced: [0, 0.25, 0.5, 0.75, 1]

# Random arrays
np.random.rand(3, 3)       # Uniform [0, 1)
np.random.randn(3, 3)      # Standard normal
np.random.randint(0, 10, (3, 3))  # Random integers
np.random.choice([1, 2, 3], 5)    # Random choice

# Special arrays
np.empty((3, 3))           # Uninitialized array
np.nan                     # Not a Number
np.inf                     # Infinity
```

### Array Attributes

```python
arr = np.array([[1, 2, 3], [4, 5, 6]])

arr.ndim      # 2 (number of dimensions)
arr.shape     # (2, 3) (dimensions)
arr.size      # 6 (total elements)
arr.dtype     # dtype('int64') (data type)
arr.itemsize  # 8 (bytes per element)
arr.nbytes    # 48 (total bytes)
arr.T         # Transpose
```

### Indexing and Slicing

```python
arr = np.array([1, 2, 3, 4, 5])

# Basic indexing
arr[0]      # 1
arr[-1]     # 5
arr[1:4]    # [2, 3, 4]

# 2D indexing
matrix = np.array([[1, 2, 3], [4, 5, 6]])
matrix[0, 1]    # 2
matrix[:, 1]    # [2, 5] (second column)
matrix[0, :]    # [1, 2, 3] (first row)

# Boolean indexing
arr[arr > 3]        # [4, 5]
arr[arr % 2 == 0]   # [2, 4]

# Fancy indexing
arr[[0, 2, 4]]  # [1, 3, 5]
```

---

## 2. Array Operations

### Element-wise Operations

```python
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

a + b      # [5, 7, 9]
a * b      # [4, 10, 18]
a ** 2     # [1, 4, 9]
np.sqrt(a) # [1, 1.414, 1.732]
np.exp(a)  # [e^1, e^2, e^3]
np.log(a)  # [ln(1), ln(2), ln(3)]
```

### Broadcasting

```python
# Broadcasting rules:
# 1. Align shapes from right
# 2. Dimensions must be equal or one must be 1

a = np.array([[1, 2, 3], [4, 5, 6]])  # Shape (2, 3)
b = np.array([10, 20, 30])             # Shape (3,)

a + b  # [[11, 22, 33], [14, 25, 36]]
# b is broadcast to [[10, 20, 30], [10, 20, 30]]

# Scalar broadcasting
a * 2  # [[2, 4, 6], [8, 10, 12]]
```

### Aggregation Functions

```python
arr = np.array([[1, 2, 3], [4, 5, 6]])

np.sum(arr)        # 21
np.mean(arr)       # 3.5
np.std(arr)        # Standard deviation
np.var(arr)        # Variance
np.min(arr)        # 1
np.max(arr)        # 6
np.argmax(arr)     # Index of max (5)
np.argmin(arr)     # Index of min (0)
np.median(arr)     # Median

# Axis parameter
np.sum(arr, axis=0)  # [5, 7, 9] (column sums)
np.sum(arr, axis=1)  # [6, 15] (row sums)
np.mean(arr, axis=0) # [2.5, 3.5, 4.5]
```

---

## 3. Array Manipulation

### Reshaping

```python
arr = np.arange(12)  # [0, 1, ..., 11]

arr.reshape(3, 4)     # Reshape to 3x4
arr.ravel()           # Flatten to 1D
arr.flatten()         # Flatten (returns copy)
arr.T                 # Transpose
arr.transpose()       # Transpose
```

### Stacking and Splitting

```python
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

# Stacking
np.vstack([a, b])    # [[1, 2, 3], [4, 5, 6]]
np.hstack([a, b])    # [1, 2, 3, 4, 5, 6]
np.stack([a, b])     # Stack along new axis
np.column_stack([a, b])  # [[1, 4], [2, 5], [3, 6]]

# Splitting
arr = np.arange(10)
np.split(arr, 5)     # Split into 5 equal parts
np.hsplit(arr, 5)    # Horizontal split
np.vsplit(arr, 2)    # Vertical split (2D)
```

### Adding/Removing Elements

```python
arr = np.array([1, 2, 3])

np.append(arr, [4, 5])      # [1, 2, 3, 4, 5]
np.insert(arr, 1, 99)       # [1, 99, 2, 3]
np.delete(arr, 1)           # [1, 3]
np.unique(arr)              # Unique values
```

---

## 4. Linear Algebra with NumPy

```python
import numpy as np

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# Matrix multiplication
A @ B           # Matrix multiply
np.dot(A, B)    # Same as @
A.dot(B)        # Method form

# Element-wise
A * B           # [[5, 12], [21, 32]]

# Linear algebra functions
np.linalg.det(A)     # Determinant
np.linalg.inv(A)     # Inverse
np.linalg.trace(A)   # Trace (sum of diagonal)
np.linalg.norm(A)    # Frobenius norm

# Eigenvalues/eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(A)

# Solve linear system: Ax = b
b = np.array([5, 11])
x = np.linalg.solve(A, b)  # [1, 2]

# SVD
U, S, Vh = np.linalg.svd(A)

# Matrix rank
np.linalg.matrix_rank(A)

# Condition number
np.linalg.cond(A)
```

---

## 5. Advanced NumPy

### Structured Arrays

```python
# Define dtype
dtype = [('name', 'U10'), ('age', 'i4'), ('weight', 'f8')]

# Create structured array
data = np.array([('Alice', 25, 55.5), ('Bob', 30, 80.0)], dtype=dtype)

# Access fields
data['name']  # ['Alice', 'Bob']
data['age']   # [25, 30]
```

### Vectorization

```python
# Without vectorization (slow)
def slow_sum(arr):
    total = 0
    for x in arr:
        total += x
    return total

# With vectorization (fast)
def fast_sum(arr):
    return np.sum(arr)

# np.vectorize for non-vectorized functions
def f(x):
    return x**2 if x > 0 else 0

vectorized_f = np.vectorize(f)
```

### Einstein Summation

```python
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

# Dot product
np.einsum('i,i->', a, b)  # 32

# Matrix multiplication
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
np.einsum('ij,jk->ik', A, B)
```

### Memory Mapping

```python
# For large files
mmap = np.memmap('large_file.dat', dtype='float32', mode='w+', shape=(1000000,))
mmap[:10] = np.arange(10)
del mmap  # Flush to disk
```

---

## 💻 Python Code Examples

```python
import numpy as np

# === Example 1: Image Processing ===

# Create synthetic image (grayscale)
image = np.random.randint(0, 256, (100, 100), dtype=np.uint8)

# Flip vertically
flipped_v = image[::-1, :]

# Flip horizontally
flipped_h = image[:, ::-1]

# Rotate 90 degrees
rotated = np.rot90(image)

# Crop center
h, w = image.shape
crop = image[h//4:3*h//4, w//4:3*w//4]

# Normalize to [0, 1]
normalized = image / 255.0

# === Example 2: Statistics ===

# Generate data
np.random.seed(42)
data = np.random.normal(100, 15, 1000)

# Statistics
mean = np.mean(data)
std = np.std(data)
median = np.median(data)
q25, q75 = np.percentile(data, [25, 75])
iqr = q75 - q25

# Z-scores
z_scores = (data - mean) / std

# Outliers (|z| > 3)
outliers = data[np.abs(z_scores) > 3]

# === Example 3: Linear Regression ===

# Generate data
np.random.seed(42)
X = np.random.rand(100, 1)
y = 2 + 3 * X + np.random.randn(100, 1)

# Add intercept
X_b = np.c_[np.ones((100, 1)), X]

# Normal equation: theta = (X^T X)^(-1) X^T y
theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

print(f"Intercept: {theta[0][0]:.2f}")
print(f"Slope: {theta[1][0]:.2f}")

# === Example 4: Broadcasting Practice ===

# Distance matrix
points = np.array([[1, 2], [3, 4], [5, 6]])

# Pairwise distances
diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
distances = np.sqrt(np.sum(diff**2, axis=2))

# === Example 5: Advanced Indexing ===

# Boolean masking
arr = np.arange(1, 21)
mask = (arr % 3 == 0) | (arr % 5 == 0)
divisible = arr[mask]  # [3, 5, 6, 9, 10, 12, 15, 18, 20]

# Where
indices = np.where(arr > 10)
values = arr[indices]

# Clip
clipped = np.clip(arr, 5, 15)  # Values between 5 and 15
```

---

## 📊 Summary Tables

### Array Creation Functions

| Function | Purpose | Example |
|----------|---------|---------|
| np.array() | Create from list | np.array([1,2,3]) |
| np.zeros() | Array of zeros | np.zeros((3,3)) |
| np.ones() | Array of ones | np.ones((2,2)) |
| np.arange() | Range array | np.arange(0,10,2) |
| np.linspace() | Evenly spaced | np.linspace(0,1,5) |
| np.random.rand() | Uniform random | np.random.rand(3,3) |
| np.random.randn() | Normal random | np.random.randn(3,3) |

### Aggregation Functions

| Function | Purpose | Axis Example |
|----------|---------|--------------|
| np.sum() | Sum | np.sum(arr, axis=0) |
| np.mean() | Mean | np.mean(arr, axis=1) |
| np.std() | Standard deviation | np.std(arr) |
| np.min() | Minimum | np.min(arr) |
| np.max() | Maximum | np.max(arr) |
| np.argmax() | Index of max | np.argmax(arr) |

### Linear Algebra

| Function | Purpose | Example |
|----------|---------|---------|
| np.dot() | Dot product | np.dot(A, B) |
| @ | Matrix multiply | A @ B |
| np.linalg.inv() | Inverse | np.linalg.inv(A) |
| np.linalg.det() | Determinant | np.linalg.det(A) |
| np.linalg.eig() | Eigenvalues | np.linalg.eig(A) |
| np.linalg.solve() | Solve Ax=b | np.linalg.solve(A, b) |
| np.linalg.svd() | SVD | np.linalg.svd(A) |

---

## 🎯 ML Applications

| NumPy Feature | ML Application |
|---------------|----------------|
| Arrays | Feature matrices |
| Broadcasting | Vectorized predictions |
| Linear algebra | Neural network layers |
| Random | Weight initialization |
| Indexing | Data slicing |

---

## ❓ Quick Check Questions

1. What's the difference between np.array and np.asarray?
2. How does broadcasting work?
3. What does axis=0 mean in np.sum(arr, axis=0)?
4. How do you find unique values in an array?
5. What's the difference between ravel() and flatten()?
6. How do you compute matrix inverse?

---

## 📝 Answers to Quick Check

<details>
<summary>Click to reveal answers</summary>

1. **np.array vs np.asarray:**
   - np.array always copies
   - np.asarray only copies if needed

2. **Broadcasting:**
   - Aligns shapes from right
   - Expands dimensions of size 1

3. **axis=0:**
   - Operates along rows (column-wise)
   - axis=1 operates along columns

4. **Unique values:**
   - np.unique(arr)

5. **ravel() vs flatten():**
   - ravel() returns view (no copy)
   - flatten() returns copy

6. **Matrix inverse:**
   - np.linalg.inv(A)

</details>
---

**Status:** ✅ Complete
**Next:** Pandas
