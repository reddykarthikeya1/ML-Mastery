# 1.1.5 Eigenvalues and Eigenvectors

## 🎯 Quick Overview
- **Eigenvector**: Vector that doesn't change direction under transformation
- **Eigenvalue**: Scaling factor for eigenvector
- **Critical for**: PCA, spectral analysis, stability, quantum mechanics

---

## 1. Definition and Geometric Interpretation

### Eigenvalue Equation
```
Av = λv

where:
  A = square matrix (n×n)
  v = eigenvector (non-zero)
  λ = eigenvalue (scalar)
```

### Geometric Meaning
```
Before: v  →  After: Av = λv

The vector v doesn't change DIRECTION, only MAGNITUDE (scaled by λ)
```

### Visual
```
     Av (stretched)
       ↑
       |
       |  v (original)
       | ↗
       |/
       •
       
λ > 1: Stretches
λ = 1: No change
0 < λ < 1: Shrinks
λ < 0: Reverses direction
```

---

## 2. Characteristic Equation

### Derivation
```
Av = λv
Av - λv = 0
(A - λI)v = 0

For non-trivial solution (v ≠ 0):
det(A - λI) = 0  ← Characteristic Equation
```

### Characteristic Polynomial
```
p(λ) = det(A - λI) = 0

Roots of p(λ) are the eigenvalues!
```

### Example (2×2)
```
    [ 4  1 ]
A = [      ]
    [ 2  3 ]

A - λI = [ 4-λ   1   ]
         [  2   3-λ  ]

det(A - λI) = (4-λ)(3-λ) - 2 = 0
λ² - 7λ + 10 = 0
(λ - 5)(λ - 2) = 0

Eigenvalues: λ₁ = 5, λ₂ = 2
```

---

## 3. Computing Eigenvalues and Eigenvectors

### Step-by-Step Algorithm

**Step 1:** Solve det(A - λI) = 0 for λ  
**Step 2:** For each λ, solve (A - λI)v = 0 for v

### Complete Example
```
    [ 2  1 ]
A = [      ]
    [ 1  2 ]

STEP 1: Find eigenvalues
det(A - λI) = det([[2-λ, 1], [1, 2-λ]]) = 0
(2-λ)² - 1 = 0
λ² - 4λ + 3 = 0
(λ - 3)(λ - 1) = 0

λ₁ = 3, λ₂ = 1

STEP 2: Find eigenvectors

For λ₁ = 3:
(A - 3I)v = 0
[ -1  1 ] [ x ]   [ 0 ]
[       ] [   ] = [   ]
[  1 -1 ] [ y ]   [ 0 ]

-x + y = 0  →  y = x
v₁ = [1, 1]ᵀ (or any multiple)

For λ₂ = 1:
(A - 1I)v = 0
[ 1  1 ] [ x ]   [ 0 ]
[      ] [   ] = [   ]
[ 1  1 ] [ y ]   [ 0 ]

x + y = 0  →  y = -x
v₂ = [1, -1]ᵀ (or any multiple)
```

---

## 4. Eigenspaces

### Definition
**Eigenspace** for λ = Null space of (A - λI)

```
E_λ = {v : Av = λv} = Null(A - λI)
```

### Properties
- Eigenspace is a **subspace**
- Dimension = geometric multiplicity
- Spanned by linearly independent eigenvectors

### Example
```
    [ 3  0 ]
A = [      ]
    [ 0  3 ]

λ = 3 (repeated)

A - 3I = [[0, 0], [0, 0]]
Any vector is an eigenvector!

Eigenspace = ℝ² (entire plane)
```

---

## 5. Algebraic and Geometric Multiplicity

### Algebraic Multiplicity
Number of times λ appears as a root of characteristic polynomial.

### Geometric Multiplicity
Dimension of eigenspace = number of independent eigenvectors for λ.

### Relationship
```
1 ≤ geometric multiplicity ≤ algebraic multiplicity
```

### Example
```
    [ 3  1 ]
A = [      ]
    [ 0  3 ]

Characteristic: (λ - 3)² = 0
Algebraic multiplicity of λ=3: 2

A - 3I = [[0, 1], [0, 0]]
Only ONE independent eigenvector: [1, 0]ᵀ
Geometric multiplicity: 1

Since 1 < 2, matrix is NOT diagonalizable!
```

---

## 6. Diagonalization

### Definition
A is **diagonalizable** if:
```
A = PDP⁻¹

where:
  D = diagonal matrix of eigenvalues
  P = matrix with eigenvectors as columns
```

### Conditions
**A is diagonalizable ⟺**
- A has n linearly independent eigenvectors
- Geometric multiplicity = algebraic multiplicity for all eigenvalues

### Diagonalization Process
```
Step 1: Find eigenvalues λ₁, ..., λₙ
Step 2: Find eigenvectors v₁, ..., vₙ
Step 3: Form P = [v₁ v₂ ... vₙ]
Step 4: Form D = diag(λ₁, λ₂, ..., λₙ)
Step 5: Verify A = PDP⁻¹
```

### Example
```
    [ 2  1 ]
A = [      ]
    [ 1  2 ]

λ₁ = 3, v₁ = [1, 1]ᵀ
λ₂ = 1, v₂ = [1, -1]ᵀ

    [ 1   1 ]      [ 3  0 ]
P = [      ], D = [      ]
    [ 1  -1 ]      [ 0  1 ]

Verify: PDP⁻¹ = A ✓
```

---

## 7. Powers of Diagonalizable Matrices

### Formula
```
Aⁿ = PDⁿP⁻¹

where Dⁿ = diag(λ₁ⁿ, λ₂ⁿ, ..., λₙⁿ)
```

### Application: Fibonacci Sequence
```
Fₙ₊₂ = Fₙ₊₁ + Fₙ

Can be solved using matrix powers and eigenvalues!
```

### Example
```
A = PDP⁻¹

A¹⁰⁰ = PD¹⁰⁰P⁻¹

     [ 3¹⁰⁰   0   ]
D¹⁰⁰ = [          ]
       [ 0    1¹⁰⁰ ]

Much easier than multiplying A by itself 100 times!
```

---

## 8. Spectral Theorem for Symmetric Matrices

### Theorem
For symmetric matrix A (A = Aᵀ):
1. All eigenvalues are **real**
2. Eigenvectors are **orthogonal**
3. A is **always diagonalizable**
4. Can write: A = QΛQᵀ where Q is orthogonal

### Orthogonal Diagonalization
```
A = QΛQᵀ

where:
  Q = orthogonal matrix (QᵀQ = I)
  Λ = diagonal matrix of eigenvalues
```

### Example
```
    [ 2  1 ]
A = [      ]  (symmetric)
    [ 1  2 ]

Normalized eigenvectors:
    [ 1/√2 ]      [ 1/√2 ]
q₁ = [    ], q₂ = [      ]
    [ 1/√2 ]      [-1/√2 ]

    [ 1/√2   1/√2 ]      [ 3  0 ]
Q = [                ], Λ = [      ]
    [ 1/√2  -1/√2 ]      [ 0  1 ]

A = QΛQᵀ ✓
```

---

## 💻 Python Code Examples

```python
import numpy as np
from scipy import linalg

# === Matrix ===
A = np.array([[4, 1],
              [2, 3]])

# === Eigenvalues and Eigenvectors ===
eigenvalues, eigenvectors = np.linalg.eig(A)

print(f"Eigenvalues: {eigenvalues}")
print(f"Eigenvectors:\n{eigenvectors}")

# Verify: Av = λv
for i in range(len(eigenvalues)):
    λ = eigenvalues[i]
    v = eigenvectors[:, i]
    
    Av = A @ v
    λv = λ * v
    
    print(f"λ{i+1} = {λ:.4f}")
    print(f"Av = {Av}")
    print(f"λv = {λv}")
    print(f"Match: {np.allclose(Av, λv)}\n")

# === Characteristic Polynomial ===
# For 2×2: λ² - tr(A)λ + det(A) = 0
trace_A = np.trace(A)
det_A = np.linalg.det(A)

print(f"Characteristic polynomial: λ² - {trace_A}λ + {det_A} = 0")

# === Diagonalization ===
P = eigenvectors
D = np.diag(eigenvalues)
P_inv = np.linalg.inv(P)

A_reconstructed = P @ D @ P_inv
print(f"Original A:\n{A}")
print(f"Reconstructed PDP⁻¹:\n{A_reconstructed}")
print(f"Match: {np.allclose(A, A_reconstructed)}")

# === Matrix Powers ===
A_squared = A @ A
A_via_diag = P @ (D @ D) @ P_inv

print(f"A² (direct): {A_squared}")
print(f"A² (via diag): {A_via_diag}")
print(f"Match: {np.allclose(A_squared, A_via_diag)}")

# === Symmetric Matrix (Spectral Theorem) ===
S = np.array([[2, 1],
              [1, 2]])

eigenvalues_s, eigenvectors_s = np.linalg.eig(S)

print(f"\nSymmetric matrix eigenvalues: {eigenvalues_s}")
print(f"All real? {np.all(np.isreal(eigenvalues_s))}")

# Check orthogonality
v1 = eigenvectors_s[:, 0]
v2 = eigenvectors_s[:, 1]
dot_product = np.dot(v1, v2)
print(f"Eigenvectors orthogonal? {np.isclose(dot_product, 0)}")

# Orthogonal diagonalization
Q = eigenvectors_s
Λ = np.diag(eigenvalues_s)
S_reconstructed = Q @ Λ @ Q.T  # Q.T = Q⁻¹ for orthogonal

print(f"S = QΛQᵀ: {np.allclose(S, S_reconstructed)}")

# === Applications ===
# Fibonacci using eigenvalues
Fib_matrix = np.array([[1, 1],
                       [1, 0]])

eig_fib, _ = np.linalg.eig(Fib_matrix)
print(f"\nFibonacci matrix eigenvalues: {eig_fib}")
print(f"Golden ratio φ = {(1 + np.sqrt(5)) / 2:.4f}")

# === Stability Analysis ===
# System is stable if all |λ| < 1
A_system = np.array([[0.5, 0.2],
                     [0.1, 0.6]])

eig_system, _ = np.linalg.eig(A_system)
is_stable = np.all(np.abs(eig_system) < 1)
print(f"\nSystem stable? {is_stable}")
print(f"Eigenvalues: {eig_system}")
```

---

## 📊 Summary Table

| Concept | Formula | Key Point |
|---------|---------|-----------|
| **Eigenvalue Eq** | Av = λv | Direction unchanged |
| **Characteristic** | det(A - λI) = 0 | Find eigenvalues |
| **Eigenspace** | Null(A - λI) | All eigenvectors for λ |
| **Algebraic Mult** | Multiplicity as root | From characteristic poly |
| **Geometric Mult** | dim(Eigenspace) | # of independent eigenvectors |
| **Diagonalization** | A = PDP⁻¹ | Requires n independent eigenvectors |
| **Matrix Powers** | Aⁿ = PDⁿP⁻¹ | Efficient computation |
| **Spectral Theorem** | A = QΛQᵀ | For symmetric matrices |

---

## 🎯 ML Applications

| Application | How It's Used |
|-------------|---------------|
| **PCA** | Eigenvectors of covariance matrix |
| **Spectral Clustering** | Eigenvalues of Laplacian |
| **PageRank** | Principal eigenvector |
| **Stability Analysis** | Eigenvalues of Jacobian |
| **Quantum Mechanics** | Energy levels = eigenvalues |

---

## ❓ Quick Check

1. What does it mean if λ = 0?
2. Can a matrix have complex eigenvalues?
3. When is a matrix NOT diagonalizable?
4. What's special about symmetric matrices?
5. How do you compute A¹⁰⁰ efficiently?
6. What's the relationship between trace and eigenvalues?

---

## 📝 Answers to Quick Check

1. **λ = 0 means:**
   - Matrix is **singular** (not invertible)
   - det(A) = 0
   - There exists a non-zero vector v such that Av = 0v = 0
   - The transformation collapses at least one dimension

2. **Complex eigenvalues:**
   - **Yes!** Real matrices can have complex eigenvalues
   - They come in **conjugate pairs** (a ± bi)
   - Indicates rotation in the transformation
   - Example: Rotation matrix has eigenvalues e^(±iθ)

3. **NOT diagonalizable when:**
   - Geometric multiplicity < algebraic multiplicity
   - Not enough linearly independent eigenvectors
   - Example: [[3, 1], [0, 3]] has only one eigenvector

4. **Symmetric matrices are special:**
   - All eigenvalues are **real** (no complex)
   - Eigenvectors are **orthogonal**
   - **Always diagonalizable**
   - Can write A = QΛQᵀ with orthogonal Q

5. **Compute A¹⁰⁰ efficiently:**
   - Diagonalize: A = PDP⁻¹
   - A¹⁰⁰ = PD¹⁰⁰P⁻¹
   - D¹⁰⁰ = diag(λ₁¹⁰⁰, λ₂¹⁰⁰, ..., λₙ¹⁰⁰)
   - Much faster than 100 matrix multiplications!

6. **Trace and eigenvalues:**
   - **tr(A) = λ₁ + λ₂ + ... + λₙ** (sum of eigenvalues)
   - Also: det(A) = λ₁ · λ₂ · ... · λₙ (product of eigenvalues)
   - Useful for checking eigenvalue computations

---

**Status:** ✅ Complete  
**Next:** Orthogonality and Projections
