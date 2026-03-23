# 1.1.8 Advanced Topics

## 🎯 Quick Overview
- **Matrix Decompositions**: LU, Cholesky, QR, SVD
- **Matrix Properties**: Positive definite, norms, condition number
- **Advanced Concepts**: Pseudoinverse, Jordan form, Cayley-Hamilton

---

## 1. Positive Definite and Positive Semidefinite Matrices

### Definitions

**Positive Definite (PD):**
```
A is PD ⟺ xᵀAx > 0 for all x ≠ 0

Notation: A ≻ 0
```

**Positive Semidefinite (PSD):**
```
A is PSD ⟺ xᵀAx ≥ 0 for all x

Notation: A ≽ 0
```

### Equivalent Conditions (for Symmetric A)

| Property | PD | PSD |
|----------|-----|-----|
| **Eigenvalues** | All λᵢ > 0 | All λᵢ ≥ 0 |
| **Determinant** | det(A) > 0 | det(A) ≥ 0 |
| **Pivots** | All positive | All non-negative |
| **Cholesky** | Exists | May not exist |

### Examples
```
PD Matrix:
    [ 2  1 ]
A = [     ]  →  λ₁=3, λ₂=1 (both > 0) ✓
    [ 1  2 ]

PSD Matrix:
    [ 1  1 ]
B = [     ]  →  λ₁=2, λ₂=0 (≥ 0) ✓
    [ 1  1 ]

Indefinite:
    [ 1  0 ]
C = [     ]  →  λ₁=1, λ₂=-1 (mixed signs) ✗
    [ 0 -1 ]
```

### Properties
| Property | Formula |
|----------|---------|
| **Sum** | PD + PD = PD |
| **Inverse** | A PD ⟺ A⁻¹ PD |
| **Gram Matrix** | AᵀA is always PSD |
| **Covariance** | All covariance matrices are PSD |

### Applications
- **Optimization**: Hessian PD ⟺ local minimum
- **Statistics**: Covariance matrices are PSD
- **ML**: Kernel matrices must be PSD

---

## 2. Cholesky Decomposition

### Definition
For symmetric positive definite A:
```
A = LLᵀ

where L is lower triangular
```

### Algorithm
```
For i = 1 to n:
  Lᵢᵢ = √(Aᵢᵢ - Σₖ<ᵢ Lᵢₖ²)
  
  For j = i+1 to n:
    Lⱼᵢ = (Aⱼᵢ - Σₖ<ᵢ LⱼₖLᵢₖ) / Lᵢᵢ
```

### Example (2×2)
```
    [ 4  2 ]
A = [     ]
    [ 2  5 ]

L₁₁ = √4 = 2
L₂₁ = 2/2 = 1
L₂₂ = √(5 - 1²) = 2

    [ 2  0 ]
L = [     ]
    [ 1  2 ]

Verify: LLᵀ = A ✓
```

### Applications
- Solving Ax = b (2x faster than LU)
- Monte Carlo simulation
- Kalman filtering

---

## 3. LU Decomposition

### Definition
```
A = LU

where:
  L = lower triangular (with 1s on diagonal)
  U = upper triangular
```

### With Pivoting
```
PA = LU

where P is permutation matrix
```

### Example
```
    [ 2  1 ]
A = [     ]
    [ 4  3 ]

    [ 1  0 ]      [ 2  1 ]
L = [     ], U = [     ]
    [ 2  1 ]      [ 0  1 ]

Verify: LU = A ✓
```

### Applications
- Solving linear systems
- Computing determinants
- Matrix inversion

---

## 4. LDU Decomposition

### Definition
```
A = LDU

where:
  L = lower triangular (1s on diagonal)
  D = diagonal
  U = upper triangular (1s on diagonal)
```

### Relationship to LU
```
LU = L(DU') = (LD')U

Extract diagonal from U or L to get D
```

---

## 5. Matrix Norms

### Vector Norms (Review)
| Norm | Formula |
|------|---------|
| **L1** | ‖x‖₁ = Σ\|xᵢ\| |
| **L2** | ‖x‖₂ = √(Σxᵢ²) |
| **L∞** | ‖x‖∞ = max(\|xᵢ\|) |

### Matrix Norms

**Frobenius Norm:**
```
‖A‖_F = √(Σᵢⱼ aᵢⱼ²) = √(trace(AᵀA))

Also: ‖A‖_F = √(Σσᵢ²) where σᵢ are singular values
```

**Spectral Norm (Operator 2-norm):**
```
‖A‖₂ = max_{x≠0} ‖Ax‖₂ / ‖x‖₂ = σ_max(A)

Largest singular value!
```

**Other Matrix Norms:**
| Norm | Formula |
|------|---------|
| **‖A‖₁** | max column sum |
| **‖A‖∞** | max row sum |
| **‖A‖_F** | Frobenius (Euclidean for matrices) |

### Properties
| Property | Formula |
|----------|---------|
| **Homogeneity** | ‖cA‖ = \|c\|·‖A‖ |
| **Triangle** | ‖A + B‖ ≤ ‖A‖ + ‖B‖ |
| **Submultiplicative** | ‖AB‖ ≤ ‖A‖·‖B‖ |
| **Consistency** | ‖Ax‖ ≤ ‖A‖·‖x‖ |

---

## 6. Condition Number

### Definition
```
κ(A) = ‖A‖ · ‖A⁻¹‖

For 2-norm: κ(A) = σ_max / σ_min
```

### Interpretation
| Condition Number | Meaning |
|------------------|---------|
| **κ ≈ 1** | Well-conditioned |
| **κ >> 1** | Ill-conditioned |
| **κ = ∞** | Singular |

### Error Amplification
```
Relative error in x ≤ κ(A) × Relative error in b

for Ax = b
```

### Example
```
    [ 1    0   ]
A = [          ]  →  κ(A) = 1000/1 = 1000 (ill-conditioned!)
    [ 0  0.001 ]

Small changes in b → Large changes in x
```

### Applications
- Numerical stability analysis
- Regularization (ridge regression)
- Preconditioning

---

## 7. Pseudoinverse (Moore-Penrose)

### Definition
A⁺ is the pseudoinverse of A if:
```
1. AA⁺A = A
2. A⁺AA⁺ = A⁺
3. (AA⁺)ᵀ = AA⁺
4. (A⁺A)ᵀ = A⁺A
```

### Computation via SVD
```
A = UΣVᵀ

A⁺ = VΣ⁺Uᵀ

where Σ⁺ has 1/σᵢ for non-zero σᵢ
```

### Properties
| Property | Formula |
|----------|---------|
| **Inverse** | If A invertible: A⁺ = A⁻¹ |
| **Least Squares** | x = A⁺b minimizes ‖Ax - b‖ |
| **Minimum Norm** | x = A⁺b has minimum ‖x‖ |
| **Projection** | AA⁺ projects onto Col(A) |

### Special Cases
```
Full column rank: A⁺ = (AᵀA)⁻¹Aᵀ
Full row rank: A⁺ = Aᵀ(AAᵀ)⁻¹
```

### Applications
- Least squares solutions
- Underdetermined systems
- Regularization

---

## 8. Jordan Normal Form (Conceptual)

### Definition
For any square matrix A:
```
A = PJP⁻¹

where J is block-diagonal (Jordan form)
```

### Jordan Blocks
```
[ λ  1  0 ]
[ 0  λ  1 ]  ← Jordan block for eigenvalue λ
[ 0  0  λ ]
```

### When Needed
- When A is NOT diagonalizable
- Geometric multiplicity < algebraic multiplicity

### Example
```
    [ 3  1 ]
A = [     ]  (not diagonalizable)
    [ 0  3 ]

J = A itself (already in Jordan form)
```

---

## 9. Cayley-Hamilton Theorem

### Statement
```
Every matrix satisfies its own characteristic equation

If p(λ) = det(A - λI), then p(A) = 0
```

### Example
```
    [ 1  2 ]
A = [     ]
    [ 3  4 ]

Characteristic: λ² - 5λ - 2 = 0

Cayley-Hamilton: A² - 5A - 2I = 0

Verify:
A² = [[7, 10], [15, 22]]
5A = [[5, 10], [15, 20]]
2I = [[2, 0], [0, 2]]

A² - 5A - 2I = [[0, 0], [0, 0]] ✓
```

### Applications
- Computing matrix powers
- Matrix functions (e^A, sin(A))
- Control theory

---

## 💻 Python Code Examples

```python
import numpy as np
from scipy import linalg

# === Positive Definite Check ===
A = np.array([[2, 1],
              [1, 2]])

# Method 1: Eigenvalues
eigenvalues = np.linalg.eigvalsh(A)
is_pd = np.all(eigenvalues > 0)
print(f"Eigenvalues: {eigenvalues}")
print(f"Positive definite? {is_pd}")

# Method 2: Cholesky (will fail if not PD)
try:
    L = np.linalg.cholesky(A)
    print(f"Cholesky exists: PD ✓")
    print(f"L:\n{L}")
except np.linalg.LinAlgError:
    print("Not positive definite")

# === Cholesky Decomposition ===
A = np.array([[4, 2, 2],
              [2, 5, 1],
              [2, 1, 6]])

L = np.linalg.cholesky(A)
print(f"\nCholesky L:\n{L}")
print(f"LLᵀ = A? {np.allclose(L @ L.T, A)}")

# === LU Decomposition ===
A = np.array([[2, 1, 1],
              [4, 3, 3],
              [8, 7, 9]])

P, L, U = linalg.lu(A)
print(f"\nLU Decomposition:")
print(f"P:\n{P}")
print(f"L:\n{L}")
print(f"U:\n{U}")
print(f"PA = LU? {np.allclose(P @ A, L @ U)}")

# === Matrix Norms ===
A = np.array([[1, 2],
              [3, 4]])

norm_fro = np.linalg.norm(A, 'fro')
norm_2 = np.linalg.norm(A, 2)
norm_1 = np.linalg.norm(A, 1)
norm_inf = np.linalg.norm(A, np.inf)

print(f"\nMatrix Norms:")
print(f"Frobenius: {norm_fro:.4f}")
print(f"Spectral (2-norm): {norm_2:.4f}")
print(f"1-norm: {norm_1:.4f}")
print(f"∞-norm: {norm_inf:.4f}")

# Verify Frobenius
print(f"√(Σaᵢⱼ²) = {np.sqrt(np.sum(A**2)):.4f}")

# === Condition Number ===
A_well = np.array([[1, 0],
                   [0, 1]])

A_ill = np.array([[1, 0],
                  [0, 1e-6]])

print(f"\nCondition Numbers:")
print(f"Well-conditioned: {np.linalg.cond(A_well):.2f}")
print(f"Ill-conditioned: {np.linalg.cond(A_ill):.2e}")

# === Pseudoinverse ===
A = np.array([[1, 2],
              [3, 4],
              [5, 6]])

# Using SVD
U, S, Vt = np.linalg.svd(A, full_matrices=False)
S_pinv = np.diag(1/S)
A_pinv_svd = Vt.T @ S_pinv @ U.T

# Using numpy
A_pinv_np = np.linalg.pinv(A)

print(f"\nPseudoinverse:")
print(f"A⁺ (SVD):\n{A_pinv_svd}")
print(f"A⁺ (numpy):\n{A_pinv_np}")

# Verify properties
print(f"AA⁺A = A? {np.allclose(A @ A_pinv_np @ A, A)}")
print(f"A⁺AA⁺ = A⁺? {np.allclose(A_pinv_np @ A @ A_pinv_np, A_pinv_np)}")

# === Least Squares with Pseudoinverse ===
b = np.array([1, 2, 3])
x = A_pinv_np @ b
residual = np.linalg.norm(A @ x - b)

print(f"\nLeast Squares Solution:")
print(f"x = {x}")
print(f"Residual: {residual:.4f}")

# Compare with lstsq
x_lstsq, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
print(f"lstsq solution: {x_lstsq}")
print(f"Match? {np.allclose(x, x_lstsq)}")

# === Cayley-Hamilton Verification ===
A = np.array([[1, 2],
              [3, 4]])

# Characteristic polynomial: λ² - 5λ - 2 = 0
# Verify: A² - 5A - 2I = 0

A_squared = A @ A
result = A_squared - 5*A - 2*np.eye(2)

print(f"\nCayley-Hamilton:")
print(f"A² - 5A - 2I =\n{result}")
print(f"≈ 0? {np.allclose(result, 0)}")

# === Matrix Functions using Cayley-Hamilton ===
# Compute e^A using Taylor series
def matrix_exp(A, n_terms=20):
    """Compute matrix exponential"""
    n = A.shape[0]
    result = np.eye(n)
    term = np.eye(n)
    
    for k in range(1, n_terms):
        term = term @ A / k
        result += term
    
    return result

exp_A = matrix_exp(A)
exp_A_scipy = linalg.expm(A)

print(f"\nMatrix Exponential:")
print(f"Taylor series:\n{exp_A}")
print(f"SciPy expm:\n{exp_A_scipy}")
print(f"Match? {np.allclose(exp_A, exp_A_scipy)}")
```

---

## 📊 Summary Table

| Concept | Formula | Key Point |
|---------|---------|-----------|
| **Positive Definite** | xᵀAx > 0 | All eigenvalues > 0 |
| **Cholesky** | A = LLᵀ | For symmetric PD matrices |
| **LU** | PA = LU | General linear systems |
| **Frobenius Norm** | √(Σaᵢⱼ²) | Euclidean for matrices |
| **Spectral Norm** | σ_max | Largest singular value |
| **Condition Number** | σ_max/σ_min | Measures stability |
| **Pseudoinverse** | A⁺ = VΣ⁺Uᵀ | Generalized inverse |
| **Cayley-Hamilton** | p(A) = 0 | Matrix satisfies char. poly |

---

## 🎯 ML Applications

| Application | How It's Used |
|-------------|---------------|
| **Kernel Methods** | PSD kernel matrices |
| **Optimization** | Hessian PD test for minima |
| **Regularization** | Condition number → ridge parameter |
| **Neural Networks** | Xavier initialization (SVD) |
| **Numerical Stability** | Condition number monitoring |

---

## ❓ Quick Check

1. How do you test if a matrix is positive definite?
2. Why is Cholesky faster than LU?
3. What does a high condition number indicate?
4. When do you need pseudoinverse instead of inverse?
5. What is the Frobenius norm geometrically?
6. State the Cayley-Hamilton theorem.

---

## 📝 Answers to Quick Check

1. **Test for positive definite:**
   - All eigenvalues > 0
   - All leading principal minors (determinants) > 0
   - xᵀAx > 0 for all x ≠ 0
   - Cholesky decomposition exists (no error)
   - In Python: `np.all(np.linalg.eigvalsh(A) > 0)`

2. **Cholesky faster than LU:**
   - Exploits symmetry (only computes half the matrix)
   - **2x faster** than LU for symmetric positive definite
   - More numerically stable
   - Requires half the storage

3. **High condition number:**
   - Matrix is **ill-conditioned** (nearly singular)
   - Small changes in input → large changes in output
   - Numerical instability in computations
   - κ > 1000 is typically considered ill-conditioned

4. **Pseudoinverse needed when:**
   - Matrix is not square (no inverse exists)
   - Matrix is singular (det = 0)
   - Solving overdetermined systems (least squares)
   - Solving underdetermined systems (minimum norm solution)

5. **Frobenius norm geometrically:**
   - Euclidean distance treating matrix as a vector
   - ‖A‖_F = √(sum of all elements squared)
   - Like "length" of the matrix
   - Also: √(sum of squared singular values)

6. **Cayley-Hamilton theorem:**
   - **"Every square matrix satisfies its own characteristic equation"**
   - If p(λ) = det(A - λI), then p(A) = 0
   - Example: If λ² - 5λ + 6 = 0, then A² - 5A + 6I = 0
   - Useful for computing matrix powers and functions

---

## 🎉 Linear Algebra Complete!

You've now covered all essential linear algebra topics for AI/ML!

**Next Steps:**
1. Review all notes
2. Practice problems
3. Move to Calculus
4. Apply concepts in Python projects

---

**Status:** ✅ Complete  
**Phase 1 Progress:** 8/8 sections done!
