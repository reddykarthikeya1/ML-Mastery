# 1.1.6 Orthogonality and Projections

## 🎯 Quick Overview
- **Orthogonal**: Perpendicular vectors (dot product = 0)
- **Projection**: Shadow of one vector onto another
- **Critical for**: Least squares, PCA, signal processing

---

## 1. Orthogonal and Orthonormal Vectors

### Orthogonal Vectors
```
u ⊥ v  ⟺  u · v = 0
```

**Geometric:** 90° angle

### Orthonormal Vectors
```
u · v = 0  (orthogonal)
|u| = |v| = 1  (unit length)
```

### Orthonormal Set
{v₁, v₂, ..., vₖ} is orthonormal if:
```
vᵢ · vⱼ = δᵢⱼ = { 1 if i=j, 0 if i≠j }
```

### Example
```
Standard basis in ℝ³:
    ⎡ 1 ⎤      ⎡ 0 ⎤      ⎡ 0 ⎤
e₁ = ⎢ 0 ⎥, e₂ = ⎢ 1 ⎥, e₃ = ⎢ 0 ⎥
    ⎣ 0 ⎦      ⎣ 0 ⎦      ⎣ 1 ⎦

eᵢ · eⱼ = δᵢⱼ ✓
```

---

## 2. Orthogonal Complement

### Definition
```
W⊥ = {v : v · w = 0 for all w ∈ W}

Read as "W perp"
```

### Properties
| Property | Formula |
|----------|---------|
| **Subspace** | W⊥ is always a subspace |
| **Dimension** | dim(W) + dim(W⊥) = n |
| **Double perp** | (W⊥)⊥ = W |
| **Intersection** | W ∩ W⊥ = {0} |

### Example
```
In ℝ³, if W is a plane through origin:
W⊥ is the line perpendicular to that plane

dim(W) = 2, dim(W⊥) = 1
2 + 1 = 3 ✓
```

### Fundamental Subspaces
```
For matrix A (m×n):

Row(A)⊥ = Null(A)
Col(A)⊥ = Null(Aᵀ)

dim(Row) + dim(Null) = n
dim(Col) + dim(Null(Aᵀ)) = m
```

---

## 3. Orthogonal Projection onto a Vector

### Formula
```
proj_u(v) = (v · u / u · u) · u

        = (v · û) · û  (if u is unit vector)
```

### Components
```
v = proj_u(v) + perp_u(v)

where:
  proj_u(v) = parallel component
  perp_u(v) = perpendicular component
```

### Visual
```
     v
     ↑
     |\
     | \
     |  \
     |   \
     |    \
     •-----•
    proj_u(v)
    
    u →
```

### Properties
| Property | Description |
|----------|-------------|
| **Linearity** | proj(av + bw) = a·proj(v) + b·proj(w) |
| **Idempotent** | proj(proj(v)) = proj(v) |
| **Minimum distance** | v - proj(v) is shortest distance to line |

---

## 4. Orthogonal Projection onto a Subspace

### Matrix Formula
```
proj_W(v) = A(AᵀA)⁻¹Aᵀv

where columns of A form basis for W
```

### Projection Matrix
```
P = A(AᵀA)⁻¹Aᵀ

Properties:
  P² = P  (idempotent)
  Pᵀ = P  (symmetric)
  Pv = projection of v onto W
```

### Orthonormal Basis (Simpler!)
If columns of Q are orthonormal:
```
proj_W(v) = QQᵀv

P = QQᵀ
```

### Example
```
Project v = [1, 2, 3]ᵀ onto plane spanned by:
u₁ = [1, 0, 0]ᵀ, u₂ = [0, 1, 0]ᵀ

A = [[1, 0],
     [0, 1],
     [0, 0]]

P = A(AᵀA)⁻¹Aᵀ = A·I·Aᵀ = AAᵀ

    ⎡ 1  0  0 ⎤
P = ⎢ 0  1  0 ⎥
    ⎣ 0  0  0 ⎦

Pv = [1, 2, 0]ᵀ  ← z-component removed!
```

---

## 5. Projection Matrices and Properties

### Definition
P is a projection matrix if:
```
P² = P  (idempotent)
```

### Orthogonal Projection
P is orthogonal projection if:
```
P² = P  AND  Pᵀ = P
```

### Properties
| Property | Formula |
|----------|---------|
| **Eigenvalues** | 0 and 1 only |
| **Rank** | rank(P) = trace(P) |
| **Null space** | vectors orthogonal to subspace |
| **Range** | the subspace itself |

### Decomposition
```
I = P + P⊥

where:
  P projects onto W
  P⊥ projects onto W⊥
```

---

## 6. Gram-Schmidt Orthogonalization

### Goal
Convert basis {v₁, ..., vₙ} to orthonormal basis {u₁, ..., uₙ}

### Algorithm
```
u₁ = v₁ / |v₁|

u₂' = v₂ - proj_u₁(v₂)
u₂ = u₂' / |u₂'|

u₃' = v₃ - proj_u₁(v₃) - proj_u₂(v₃)
u₃ = u₃' / |u₃'|

Continue...
```

### Formula
```
uₖ' = vₖ - Σⱼ<ₖ proj_uⱼ(vₖ)
uₖ = uₖ' / |uₖ'|
```

### Example
```
v₁ = [1, 1, 0]ᵀ
v₂ = [1, 0, 1]ᵀ
v₃ = [0, 1, 1]ᵀ

Step 1: u₁ = v₁/|v₁| = [1/√2, 1/√2, 0]ᵀ

Step 2: u₂' = v₂ - (v₂·u₁)u₁
             = [1, 0, 1] - (1/√2)[1/√2, 1/√2, 0]
             = [1/2, -1/2, 1]
        u₂ = u₂'/|u₂'| = [1/√6, -1/√6, 2/√6]ᵀ

Step 3: Continue similarly...
```

---

## 7. QR Decomposition

### Definition
```
A = QR

where:
  Q = orthogonal matrix (columns are orthonormal)
  R = upper triangular matrix
```

### Connection to Gram-Schmidt
QR decomposition is Gram-Schmidt in matrix form!

### Computation
```
A = [v₁ v₂ ... vₙ]  (original basis)
Q = [u₁ u₂ ... uₙ]  (orthonormal basis)

R = QᵀA  (upper triangular)
```

### Properties
| Property | Description |
|----------|-------------|
| **Existence** | Always exists for any matrix A |
| **Uniqueness** | Unique if A is full rank |
| **Stability** | Numerically stable algorithm |

### Applications
- Solving least squares
- Eigenvalue algorithms (QR algorithm)
- Signal processing

---

## 8. Least Squares Approximation

### Problem
Solve Ax = b when no exact solution exists (overdetermined)

### Goal
Find x that minimizes \|Ax - b\|²

### Normal Equations
```
AᵀAx = Aᵀb

Solution:
x = (AᵀA)⁻¹Aᵀb
```

### Geometric Interpretation
```
Find x such that Ax is closest to b

Residual: r = b - Ax
Minimize: |r|² = |b - Ax|²

At minimum: r ⊥ Col(A)
```

### Visual
```
     b
     ↑
     |\
     | \
     |  \  residual r
     |   \
     |    \
     •-----•
    Ax (projection)
    
    Col(A) →
```

---

## 9. Normal Equations: AᵀAx = Aᵀb

### Derivation
```
Minimize: f(x) = |Ax - b|²

f(x) = (Ax - b)ᵀ(Ax - b)
     = xᵀAᵀAx - 2xᵀAᵀb + bᵀb

df/dx = 2AᵀAx - 2Aᵀb = 0

AᵀAx = Aᵀb  ← Normal Equations
```

### Solution
```
x = (AᵀA)⁻¹Aᵀb  (if AᵀA is invertible)
```

### Using QR
```
A = QR

x = R⁻¹Qᵀb  (more numerically stable!)
```

### Example (Linear Regression)
```
Data points: (1, 2), (2, 3), (3, 5)

Fit line: y = mx + c

⎡ 1  1 ⎤ ⎡ c ⎤   ⎡ 2 ⎤
⎢ 2  1 ⎥ ⎢   ⎥ = ⎢ 3 ⎥
⎣ 3  1 ⎦ ⎣ m ⎦   ⎣ 5 ⎦

AᵀA = [[14, 6], [6, 3]]
Aᵀb = [22, 10]ᵀ

Solve: x = [1.5, 0.33]ᵀ

Line: y = 1.5x + 0.33
```

---

## 💻 Python Code Examples

```python
import numpy as np
from scipy import linalg

# === Orthogonal Vectors ===
u = np.array([1, 0, 0])
v = np.array([0, 1, 0])

dot_product = np.dot(u, v)
print(f"u · v = {dot_product}")
print(f"Orthogonal? {np.isclose(dot_product, 0)}")

# === Orthonormal Basis ===
Q = np.array([[1, 0, 0],
              [0, 1, 0],
              [0, 0, 1]])

# Check orthonormality
QTQ = Q.T @ Q
print(f"QᵀQ = I? {np.allclose(QTQ, np.eye(3))}")

# === Projection onto Vector ===
v = np.array([3, 4, 0])
u = np.array([1, 0, 0])

# proj_u(v) = (v·u / u·u) * u
proj = (np.dot(v, u) / np.dot(u, u)) * u
print(f"proj_u(v) = {proj}")  # [3, 0, 0]

# === Projection onto Subspace ===
v = np.array([1, 2, 3])

# Plane spanned by:
u1 = np.array([1, 0, 0])
u2 = np.array([0, 1, 0])
A = np.column_stack([u1, u2])

# P = A(AᵀA)⁻¹Aᵀ
P = A @ np.linalg.inv(A.T @ A) @ A.T
proj_v = P @ v
print(f"Projection onto plane: {proj_v}")  # [1, 2, 0]

# === Gram-Schmidt ===
def gram_schmidt(vectors):
    """Gram-Schmidt orthogonalization"""
    n = len(vectors)
    Q = np.zeros_like(vectors)
    
    for i in range(n):
        q = vectors[i].copy()
        for j in range(i):
            q -= np.dot(vectors[i], Q[j]) * Q[j]
        Q[i] = q / np.linalg.norm(q)
    
    return Q

vectors = np.array([[1, 1, 0],
                    [1, 0, 1],
                    [0, 1, 1]], dtype=float)

Q = gram_schmidt(vectors)
print(f"Orthonormal basis:\n{Q}")

# Verify orthonormality
print(f"QᵀQ = I? {np.allclose(Q.T @ Q, np.eye(3))}")

# === QR Decomposition ===
A = np.array([[1, 1, 0],
              [1, 0, 1],
              [0, 1, 1]], dtype=float)

Q, R = np.linalg.qr(A)
print(f"Q (orthogonal):\n{Q}")
print(f"R (upper triangular):\n{R}")
print(f"A = QR? {np.allclose(A, Q @ R)}")

# === Least Squares ===
# Fit line to data points
x_data = np.array([1, 2, 3, 4, 5])
y_data = np.array([2.1, 3.9, 6.2, 8.1, 9.8])

# Design matrix for y = mx + c
A = np.column_stack([x_data, np.ones(len(x_data))])
b = y_data

# Normal equations: x = (AᵀA)⁻¹Aᵀb
x = np.linalg.inv(A.T @ A) @ A.T @ b
m, c = x

print(f"Fitted line: y = {m:.2f}x + {c:.2f}")

# Using QR (more stable)
Q, R = np.linalg.qr(A)
x_qr = np.linalg.solve(R, Q.T @ b)
print(f"Using QR: y = {x_qr[0]:.2f}x + {x_qr[1]:.2f}")

# Using numpy (easiest)
x_np = np.linalg.lstsq(A, b, rcond=None)[0]
print(f"Using lstsq: y = {x_np[0]:.2f}x + {x_np[1]:.2f}")

# === Projection Matrix Properties ===
P = A @ np.linalg.inv(A.T @ A) @ A.T

print(f"\nProjection matrix properties:")
print(f"P² = P? {np.allclose(P @ P, P)}")  # Idempotent
print(f"Pᵀ = P? {np.allclose(P.T, P)}")    # Symmetric
print(f"Rank(P) = Trace(P)? {np.isclose(np.linalg.matrix_rank(P), np.trace(P))}")
```

---

## 📊 Summary Table

| Concept | Formula | Key Point |
|---------|---------|-----------|
| **Orthogonal** | u · v = 0 | Perpendicular |
| **Orthonormal** | u · v = δᵢⱼ | Orthogonal + unit length |
| **Projection (vector)** | (v·u/u·u)u | Parallel component |
| **Projection (subspace)** | A(AᵀA)⁻¹Aᵀv | Matrix form |
| **Gram-Schmidt** | uₖ' = vₖ - Σproj | Orthogonalization |
| **QR** | A = QR | Orthonormal basis |
| **Least Squares** | AᵀAx = Aᵀb | Minimize \|Ax-b\|² |
| **Normal Equations** | x = (AᵀA)⁻¹Aᵀb | Solution formula |

---

## 🎯 ML Applications

| Application | How It's Used |
|-------------|---------------|
| **Linear Regression** | Least squares, normal equations |
| **PCA** | Orthogonal projections |
| **Signal Processing** | Orthogonal basis (Fourier) |
| **Neural Networks** | Orthogonal initialization |
| **Recommendation** | Low-rank approximation |

---

## ❓ Quick Check

1. What's the difference between orthogonal and orthonormal?
2. Why is P² = P for projection matrices?
3. How does Gram-Schmidt work?
4. What's the geometric meaning of least squares?
5. Why use QR instead of normal equations?
6. What is the orthogonal complement of a plane in ℝ³?

---

## 📝 Answers to Quick Check

<details>
<summary>Click to reveal answers</summary>

1. **Orthogonal vs Orthonormal:**
   - **Orthogonal**: Vectors are perpendicular (dot product = 0)
   - **Orthonormal**: Orthogonal + each vector has unit length (magnitude = 1)
   - Orthonormal is more convenient (simplifies calculations)

2. **P² = P (idempotent):**
   - Projecting a projection gives the same result
   - Once you're on the subspace, projecting again does nothing
   - Example: Shadow of a shadow is the same shadow

3. **Gram-Schmidt process:**
   - Start with basis {v₁, v₂, ..., vₙ}
   - u₁ = v₁/|v₁| (normalize first vector)
   - u₂' = v₂ - proj_u₁(v₂), then u₂ = u₂'/|u₂'| (subtract projection, normalize)
   - Continue for each vector, subtracting projections onto all previous uᵢ

4. **Least squares geometric meaning:**
   - Find x such that Ax is **closest to b** in the column space
   - Residual r = b - Ax is **perpendicular to Col(A)**
   - Minimizes the "error" (distance between b and Ax)

5. **QR over normal equations:**
   - **Numerically more stable** (avoids AᵀA which can be ill-conditioned)
   - Normal equations: condition number squares (κ(AᵀA) = κ(A)²)
   - QR preserves condition number
   - Preferred for computational work

6. **Orthogonal complement of a plane in ℝ³:**
   - A **line through the origin** perpendicular to the plane
   - dim(plane) = 2, dim(orthogonal complement) = 1
   - 2 + 1 = 3 (dimension formula holds)

</details>
---

**Status:** ✅ Complete  
**Next:** Singular Value Decomposition (SVD)
