# 1.1.2 Matrices and Matrix Operations

## 🎯 Quick Overview
- **Matrix**: Rectangular array of numbers
- **Purpose**: Represent linear transformations, systems of equations
- **Foundation for**: Neural networks, data transformations, ML algorithms

---

## 1. Matrix Representation and Notation

### Basic Notation
```
        ⎡ a₁₁  a₁₂  ...  a₁ₙ ⎤
A =     ⎢ a₂₁  a₂₂  ...  a₂ₙ ⎥  = [aᵢⱼ]
        ⎢ ...  ...  ...  ... ⎥
        ⎣ aₘ₁  aₘ₂  ...  aₘₙ ⎦

Dimensions: m × n (m rows, n columns)
Element: aᵢⱼ (row i, column j)
```

### Special Notation
- **Aᵢⱼ**: Element at row i, column j
- **Aᵢ***: Entire row i
- **A*ⱼ**: Entire column j
- **Aᵀ**: Transpose of A

---

## 2. Matrix Types

| Type | Definition | Example |
|------|------------|---------|
| **Square** | m = n | 3×3 matrix |
| **Diagonal** | aᵢⱼ = 0 for i ≠ j | diag(1,2,3) |
| **Identity (I)** | Diagonal with 1s | I₂ = [[1,0],[0,1]] |
| **Zero (0)** | All elements zero | 0₃ₓ₃ |
| **Symmetric** | A = Aᵀ | [[1,2],[2,3]] |
| **Skew-Symmetric** | A = -Aᵀ | [[0,2],[-2,0]] |
| **Upper Triangular** | aᵢⱼ = 0 for i > j | [[1,2],[0,3]] |
| **Lower Triangular** | aᵢⱼ = 0 for i < j | [[1,0],[2,3]] |
| **Orthogonal** | AᵀA = AAᵀ = I | Rotation matrices |

### Identity Matrix Properties
```
I · A = A · I = A
I⁻¹ = I
det(I) = 1
```

---

## 3. Matrix Operations

### Addition and Subtraction
```
(A ± B)ᵢⱼ = aᵢⱼ ± bᵢⱼ
```
**Requirement:** Same dimensions

**Properties:**
- ✅ Commutative: A + B = B + A
- ✅ Associative: (A + B) + C = A + (B + C)

### Scalar Multiplication
```
(cA)ᵢⱼ = c · aᵢⱼ
```

**Properties:**
- (cd)A = c(dA)
- c(A + B) = cA + cB
- (c + d)A = cA + dA

---

## 4. Matrix Multiplication

### Definition (Row × Column)
```
C = AB  where  cᵢⱼ = Σₖ aᵢₖbₖⱼ
```

**Visual:**
```
        ⎡ b₁₁  b₁₂ ⎤
[a₁₁ a₁₂] ⎢ b₂₁  b₂₂ ⎥ = [a₁₁b₁₁ + a₁₂b₂₁,  a₁₁b₁₂ + a₁₂b₂₂]
        ⎣ b₂₁  b₂₂ ⎦
```

### Dimensions Rule
```
A (m×n) × B (n×p) = C (m×p)
     ↑      ↑
  must match!
```

### Properties
| Property | Holds? | Formula |
|----------|--------|---------|
| Associative | ✅ | (AB)C = A(BC) |
| Distributive | ✅ | A(B+C) = AB + AC |
| Commutative | ❌ | AB ≠ BA (generally) |
| Transpose | ✅ | (AB)ᵀ = BᵀAᵀ |

### Special Cases
```
AI = IA = A  (Identity)
A0 = 0A = 0  (Zero matrix)
A A⁻¹ = A⁻¹ A = I  (Inverse)
```

---

## 5. Transpose of a Matrix

### Definition
```
(Aᵀ)ᵢⱼ = aⱼᵢ
```

**Visual:**
```
    ⎡ 1  2 ⎤         ⎡ 1  3  5 ⎤
A = ⎢ 3  4 ⎥  →  Aᵀ = ⎢ 2  4  6 ⎥
    ⎣ 5  6 ⎦         ⎣         ⎦
```

### Properties
| Property | Formula |
|----------|---------|
| Double transpose | (Aᵀ)ᵀ = A |
| Sum | (A + B)ᵀ = Aᵀ + Bᵀ |
| Product | (AB)ᵀ = BᵀAᵀ |
| Scalar | (cA)ᵀ = cAᵀ |
| Symmetric | A = Aᵀ |
| Skew-symmetric | A = -Aᵀ |

---

## 6. Trace of a Matrix

### Definition
For square matrix A (n×n):
```
tr(A) = a₁₁ + a₂₂ + ... + aₙₙ = Σ aᵢᵢ
```

### Properties
| Property | Formula |
|----------|---------|
| Linearity | tr(A + B) = tr(A) + tr(B) |
| Scalar | tr(cA) = c·tr(A) |
| Cyclic | tr(AB) = tr(BA) |
| Transpose | tr(Aᵀ) = tr(A) |
| Identity | tr(I) = n |

---

## 7. Matrix Multiplication Properties

### NOT Commutative
```
AB ≠ BA  (in general)

Example:
A = [[1, 2], [3, 4]]
B = [[0, 1], [1, 0]]

AB = [[2, 1], [4, 3]]
BA = [[3, 4], [1, 2]]
```

### Associative
```
(AB)C = A(BC)
```

### Distributive
```
A(B + C) = AB + AC
(A + B)C = AC + BC
```

### Dimension Mismatch
```
A (2×3) × B (3×4) = C (2×4)  ✓
A (2×3) × B (4×3)             ✗ (incompatible)
```

---

## 8. Block Matrices

### Definition
Matrix partitioned into submatrices:
```
    ⎡ A₁₁ │ A₁₂ ⎤
A = ⎢─────┼─────⎥
    ⎣ A₂₁ │ A₂₂ ⎦
```

### Block Operations
```
Addition:
⎡ A │ B ⎤   ⎡ C │ D ⎤   ⎡ A+C │ B+D ⎤
⎢───┼───⎥ + ⎢───┼───⎥ = ⎢─────┼─────⎥
⎣ C │ D ⎦   ⎣ E │ F ⎦   ⎣ C+E │ D+F ⎦

Multiplication:
⎡ A │ B ⎤ ⎡ E │ F ⎤   ⎡ AE+BG │ AF+BH ⎤
⎢───┼───⎥ ⎢───┼───⎥ = ⎢───────┼───────⎥
⎣ C │ D ⎦ ⎣ G │ H ⎦   ⎣ CE+DG │ CF+DH ⎦
```

### Applications
- Parallel computing
- Sparse matrices
- Structured matrices

---

## 9. Elementary Matrix Operations

### Row Operations
1. **Swap**: Rᵢ ↔ Rⱼ
2. **Scale**: Rᵢ → cRᵢ
3. **Replace**: Rᵢ → Rᵢ + cRⱼ

### Elementary Matrices
Obtained by applying ONE row operation to identity matrix.

**Example:**
```
Swap R₁ and R₂ in I₃:
    ⎡ 1  0  0 ⎤         ⎡ 0  1  0 ⎤
I = ⎢ 0  1  0 ⎥  →  E = ⎢ 1  0  0 ⎥
    ⎣ 0  0  1 ⎦         ⎣ 0  0  1 ⎦

EA = Matrix with R₁ and R₂ of A swapped
```

### Properties
- Every elementary matrix is invertible
- E⁻¹ is also elementary

---

## 10. Permutation Matrices

### Definition
Square binary matrix with exactly one 1 in each row and column.

**Example (3×3):**
```
    ⎡ 0  1  0 ⎤
P = ⎢ 0  0  1 ⎥
    ⎣ 1  0  0 ⎦
```

### Properties
| Property | Formula |
|----------|---------|
| Orthogonal | Pᵀ = P⁻¹ |
| Determinant | det(P) = ±1 |
| Product | P₁P₂ is also permutation |

### Application
```
PA = Permute rows of A
APᵀ = Permute columns of A
```

---

## 11. Outer Product

### Definition
```
u vᵀ = uvᵀ (column × row = matrix)
```

**Example:**
```
⎡ 1 ⎤                 ⎡ 1·2  1·3  1·4 ⎤   ⎡ 2  3  4 ⎤
⎢ 2 ⎥ · [2  3   4] =  ⎢ 2·2  2·3  2·4 ⎥ = ⎢ 4  6  8 ⎥
⎣ 3 ⎦                 ⎣ 3·2  3·3  3·4 ⎦   ⎣ 6  9  12 ⎦
```

### vs Inner Product
| | Inner Product | Outer Product |
|-|---------------|---------------|
| **Form** | uᵀv (row × col) | uvᵀ (col × row) |
| **Result** | Scalar | Matrix |
| **Dimensions** | (1×n)(n×1) = 1×1 | (n×1)(1×m) = n×m |

### ML Application
- Rank-1 updates
- Covariance matrices
- Attention mechanisms

---

## 💻 Python Code Examples

```python
import numpy as np

# === Matrix Creation ===
A = np.array([[1, 2, 3],
              [4, 5, 6]])

B = np.array([[7, 8],
              [9, 10],
              [11, 12]])

# Special matrices
I = np.eye(3)              # Identity 3×3
Z = np.zeros((2, 3))       # Zero matrix 2×3
D = np.diag([1, 2, 3])     # Diagonal matrix

# === Matrix Operations ===
# Addition (same shape required)
C = np.array([[1, 0],
              [0, 1]])
D = np.array([[2, 3],
              [4, 5]])
print(f"C + D =\n{C + D}")

# Scalar multiplication
print(f"2 * C =\n{2 * C}")

# Matrix multiplication
print(f"A @ B =\n{A @ B}")  # or np.dot(A, B)

# === Transpose ===
print(f"Aᵀ =\n{A.T}")

# === Trace ===
print(f"tr(D) = {np.trace(D)}")

# === Special Matrices ===
# Symmetric matrix
S = np.array([[1, 2, 3],
              [2, 4, 5],
              [3, 5, 6]])
print(f"Symmetric: {np.allclose(S, S.T)}")

# Orthogonal matrix (rotation)
theta = np.pi / 4
R = np.array([[np.cos(theta), -np.sin(theta)],
              [np.sin(theta), np.cos(theta)]])
print(f"Orthogonal: {np.allclose(R @ R.T, np.eye(2))}")

# === Outer Product ===
u = np.array([1, 2, 3])
v = np.array([4, 5])
outer = np.outer(u, v)
print(f"Outer product:\n{outer}")

# === Block Matrix ===
A11 = np.array([[1, 2], [3, 4]])
A12 = np.array([[5, 6], [7, 8]])
A21 = np.array([[9, 10], [11, 12]])
A22 = np.array([[13, 14], [15, 16]])

# Stack into block matrix
top = np.hstack([A11, A12])
bottom = np.hstack([A21, A22])
block = np.vstack([top, bottom])
print(f"Block matrix:\n{block}")

# === Permutation Matrix ===
P = np.array([[0, 1, 0],
              [0, 0, 1],
              [1, 0, 0]])
A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])
print(f"PA (permuted rows):\n{P @ A}")
```

---

## 📊 Summary Table

| Operation | Symbol | Dimensions | Key Property |
|-----------|--------|------------|--------------|
| **Addition** | A + B | Same size | Commutative |
| **Scalar Mult** | cA | Same size | Distributive |
| **Matrix Mult** | AB | (m×n)(n×p)→(m×p) | NOT commutative |
| **Transpose** | Aᵀ | n×m → m×n | (AB)ᵀ = BᵀAᵀ |
| **Trace** | tr(A) | Square only | tr(AB) = tr(BA) |
| **Outer Product** | uvᵀ | (n×1)(1×m)→(n×m) | Rank-1 matrix |

---

## 🎯 ML Applications

| Application | Matrix Concept |
|-------------|----------------|
| **Neural Networks** | Weight matrices, matrix multiplication |
| **Data Representation** | Data matrix (samples × features) |
| **Covariance** | Outer product, symmetric matrices |
| **Transformations** | Rotation, scaling matrices |
| **Batch Operations** | Matrix multiplication for efficiency |
| **Attention** | Outer products, permutation |

---

## ❓ Quick Check

1. What's the difference between Aᵀ and A⁻¹?
2. When can you multiply two matrices?
3. Why isn't matrix multiplication commutative?
4. What makes a matrix symmetric?
5. What is the trace of a 3×3 identity matrix?
6. What's the result of an outer product of two 3D vectors?

---

## 📝 Answers to Quick Check

<details>
<summary>Click to reveal answers</summary>

1. **Aᵀ vs A⁻¹:**
   - **Aᵀ (transpose)**: Flips rows and columns, always exists
   - **A⁻¹ (inverse)**: Matrix such that AA⁻¹ = I, only exists for invertible matrices
   - For orthogonal matrices: Aᵀ = A⁻¹

2. **Matrix multiplication condition:**
   - A (m×n) × B (n×p) = C (m×p)
   - **Inner dimensions must match!** (columns of A = rows of B)

3. **Why AB ≠ BA?**
   - Matrix multiplication represents composition of linear transformations
   - Order of transformations matters (e.g., rotate then scale ≠ scale then rotate)
   - Also, dimensions might not even allow both products

4. **Symmetric matrix:**
   - **A = Aᵀ** (matrix equals its transpose)
   - aᵢⱼ = aⱼᵢ for all i, j
   - Example: [[1, 2], [2, 3]]

5. **Trace of 3×3 identity:**
   - tr(I₃) = 1 + 1 + 1 = **3**
   - In general: tr(Iₙ) = n

6. **Outer product of two 3D vectors:**
   - Result is a **3×3 matrix**
   - If u = [u₁, u₂, u₃]ᵀ and v = [v₁, v₂, v₃]ᵀ
   - uvᵀ = [[u₁v₁, u₁v₂, u₁v₃], [u₂v₁, u₂v₂, u₂v₃], [u₃v₁, u₃v₂, u₃v₃]]

</details>
---

**Status:** ✅ Complete  
**Next:** Systems of Linear Equations
