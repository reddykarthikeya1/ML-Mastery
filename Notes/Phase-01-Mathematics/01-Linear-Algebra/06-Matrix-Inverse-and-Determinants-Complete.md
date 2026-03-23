# 1.1.4 Matrix Inverse and Determinants

## 🎯 Quick Overview
- **Inverse**: Matrix "division" (A⁻¹ such that AA⁻¹ = I)
- **Determinant**: Scalar measuring scaling factor, invertibility
- **Key for**: Solving systems, change of variables, eigenvalues

---

## 1. Identity Matrix Properties

### Definition
```
Iₙ = n×n matrix with 1s on diagonal, 0s elsewhere

    [ 1  0  0 ]
I₃ = [ 0  1  0 ]
    [ 0  0  1 ]
```

### Properties
| Property | Formula |
|----------|---------|
| **Multiplication** | AI = IA = A |
| **Inverse** | I⁻¹ = I |
| **Determinant** | det(I) = 1 |
| **Trace** | tr(I) = n |
| **Power** | Iⁿ = I |

---

## 2. Invertible Matrices and Inverse

### Definition
A is **invertible** (non-singular) if there exists A⁻¹ such that:
```
A A⁻¹ = A⁻¹ A = I
```

### Conditions for Invertibility
For square matrix A (n×n), these are **equivalent**:
- ✅ A is invertible
- ✅ det(A) ≠ 0
- ✅ rank(A) = n
- ✅ Columns are linearly independent
- ✅ Rows are linearly independent
- ✅ Ax = 0 has only trivial solution
- ✅ Ax = b has unique solution for all b
- ✅ 0 is NOT an eigenvalue

### Non-Example (Singular Matrix)
```
    [ 1  2 ]
A = [     ]  →  det(A) = 1(4) - 2(2) = 0  →  NOT invertible ✗
    [ 2  4 ]     (row 2 = 2 × row 1, dependent!)

    [ 2  1 ]
B = [     ]  →  det(B) = 2(3) - 1(4) = 2 ≠ 0  →  Invertible ✓
    [ 4  3 ]
```

---

## 3. Computing Inverse: Gaussian Elimination

### Method: [A | I] → [I | A⁻¹]

**Algorithm:**
1. Form augmented matrix [A | I]
2. Apply row operations to get [I | ?]
3. The "?" is A⁻¹

### Example (2×2)
```
    [ 1  2 ]
A = [     ]
    [ 3  4 ]

[A | I]:
[ 1  2  │  1  0 ]
[ 3  4  │  0  1 ]

R₂ → R₂ - 3R₁:
[ 1   2  │   1   0 ]
[ 0  -2  │  -3   1 ]

R₂ → -½R₂:
[ 1   2  │    1     0  ]
[ 0   1  │  1.5  -0.5 ]

R₁ → R₁ - 2R₂:
[ 1   0  │  -2    1  ]
[ 0   1  │  1.5  -0.5 ]

         [ -2    1  ]
A⁻¹ =    [         ]
         [ 1.5  -0.5 ]

Verify: A · A⁻¹ = I ✓
```

---

## 4. Computing Inverse: Adjugate Matrix

### Formula
```
A⁻¹ = (1/det(A)) · adj(A)

where adj(A) = cofactor matrix transposed
```

### Steps
1. Find det(A)
2. Find cofactor matrix C
3. Transpose: adj(A) = Cᵀ
4. A⁻¹ = (1/det(A)) · adj(A)

### Cofactor Formula
```
Cᵢⱼ = (-1)ⁱ⁺ʲ · Mᵢⱼ

where Mᵢⱼ = minor (determinant after removing row i, column j)
```

### Example (2×2)
```
    [ a  b ]
A = [     ]
    [ c  d ]

det(A) = ad - bc

         [  d  -b ]
A⁻¹ = (1/(ad-bc)) [      ]
         [ -c   a ]
```

### Example (3×3)
```
    [ 1  2  3 ]
A = [ 0  1  4 ]
    [ 5  6  0 ]

Cofactors:
C₁₁ = +det([[1,4],[6,0]]) = -24
C₁₂ = -det([[0,4],[5,0]]) = +20
C₁₃ = +det([[0,1],[5,6]]) = -5
...

         [ -24  18  5 ]
adj(A) = [  20  -15  -4 ]ᵀ
         [  -5   4  1 ]

A⁻¹ = (1/det(A)) · adj(A)
```

---

## 5. Properties of Matrix Inverse

| Property | Formula |
|----------|---------|
| **Inverse of inverse** | (A⁻¹)⁻¹ = A |
| **Transpose** | (Aᵀ)⁻¹ = (A⁻¹)ᵀ |
| **Product** | (AB)⁻¹ = B⁻¹A⁻¹ |
| **Scalar** | (cA)⁻¹ = (1/c)A⁻¹ |
| **Determinant** | det(A⁻¹) = 1/det(A) |
| **Power** | (Aⁿ)⁻¹ = (A⁻¹)ⁿ |

### Important Notes
- ❌ (A + B)⁻¹ ≠ A⁻¹ + B⁻¹
- ❌ Inverse doesn't distribute over addition
- ✅ Only for SQUARE, NON-SINGULAR matrices

---

## 6. Determinant: Definition and Computation

### Geometric Meaning
|det(A)| = **scaling factor** of linear transformation

**2D:** Area scaling  
**3D:** Volume scaling

### 2×2 Formula
```
    [ a  b ]
A = [     ]
    [ c  d ]

det(A) = ad - bc
```

### 3×3 Formula (Sarrus' Rule)
```
    [ a  b  c ]
A = [ d  e  f ]
    [ g  h  i ]

det(A) = aei + bfg + cdh - ceg - bdi - afh
```

**Visual (Sarrus):**
```
  a  b  c │ a  b
  d  e  f │ d  e
  g  h  i │ g  h

Down-right diagonals: aei + bfg + cdh
Up-right diagonals: -ceg - bdi - afh
```

---

## 7. Cofactor Expansion (Laplace Expansion)

### Formula
```
det(A) = Σⱼ aᵢⱼCᵢⱼ  (expand along row i)
det(A) = Σᵢ aᵢⱼCᵢⱼ  (expand along column j)
```

### Example (expand along first row)
```
    [ 1  2  3 ]
A = [ 4  5  6 ]
    [ 7  8  9 ]

det(A) = 1·C₁₁ + 2·C₁₂ + 3·C₁₃

C₁₁ = +det([[5,6],[8,9]]) = 45-48 = -3
C₁₂ = -det([[4,6],[7,9]]) = -(36-42) = 6
C₁₃ = +det([[4,5],[7,8]]) = 32-35 = -3

det(A) = 1(-3) + 2(6) + 3(-3) = -3 + 12 - 9 = 0

→ A is singular!
```

### Strategy
- Expand along row/column with most zeros
- Reduces n×n to (n-1)×(n-1) determinants

---

## 8. Properties of Determinants

| Property | Effect on det(A) |
|----------|------------------|
| **Transpose** | det(Aᵀ) = det(A) |
| **Row swap** | det → -det |
| **Row scale** | det → c·det |
| **Row replace** | det unchanged |
| **Product** | det(AB) = det(A)det(B) |
| **Inverse** | det(A⁻¹) = 1/det(A) |
| **Power** | det(Aⁿ) = det(A)ⁿ |
| **Zero row/col** | det(A) = 0 |
| **Identity** | det(I) = 1 |

### Quick Tests for det(A) = 0
- ❌ Row or column of zeros
- ❌ Two identical rows/columns
- ❌ One row/column is multiple of another
- ❌ Rows/columns are linearly dependent

---

## 9. Determinant and Invertibility

### Theorem
```
A is invertible  ⟺  det(A) ≠ 0
```

### Implications
| det(A) | Invertible? | Solutions to Ax = b |
|--------|-------------|---------------------|
| ≠ 0 | ✅ Yes | Unique solution |
| = 0 | ❌ No | No solution OR infinite |

### Example
```
    [ 2  1 ]
A = [     ]  →  det(A) = 2(3) - 1(4) = 2 ≠ 0  →  Invertible ✓
    [ 4  3 ]

    [ 2  4 ]
B = [     ]  →  det(B) = 2(6) - 4(3) = 0  →  NOT invertible ✗
    [ 3  6 ]
```

---

## 10. Cramer's Rule

### For System Ax = b

If det(A) ≠ 0:
```
xᵢ = det(Aᵢ) / det(A)

where Aᵢ = A with column i replaced by b
```

### Example (2×2)
```
2x + y = 5
4x + 3y = 13

    [ 2  1 ]      [ 5 ]
A = [     ], b = [   ]
    [ 4  3 ]      [ 13 ]

det(A) = 6 - 4 = 2

     [ 5  1 ]
A₁ = [     ]  →  det(A₁) = 15 - 13 = 2
     [ 13 3 ]

     [ 2  5 ]
A₂ = [     ]  →  det(A₂) = 26 - 20 = 6
     [ 4  13 ]

x = det(A₁)/det(A) = 2/2 = 1
y = det(A₂)/det(A) = 6/2 = 3
```

### Limitations
- Only for square systems
- Computationally expensive for large n
- Mainly theoretical importance

---

## 11. Matrix Determinant Lemma

### Statement
For invertible A and vectors u, v:
```
det(A + uvᵀ) = det(A) · (1 + vᵀA⁻¹u)
```

### Special Case (Rank-1 Update)
```
det(I + uvᵀ) = 1 + vᵀu
```

### Application
- Efficient determinant updates
- Sherman-Morrison formula
- Statistics (covariance updates)

---

## 💻 Python Code Examples

```python
import numpy as np
from scipy import linalg

# === Matrix Creation ===
A = np.array([[1, 2, 3],
              [0, 1, 4],
              [5, 6, 0]])

# === Determinant ===
det_A = np.linalg.det(A)
print(f"det(A) = {det_A:.4f}")

# === Inverse ===
try:
    A_inv = np.linalg.inv(A)
    print(f"A⁻¹ =\n{A_inv}")
    
    # Verify
    I_check = A @ A_inv
    print(f"A · A⁻¹ =\n{I_check}")  # Should be ~I
except np.linalg.LinAlgError:
    print("Matrix is singular!")

# === Check Invertibility ===
is_invertible = det_A != 0
print(f"Is invertible: {is_invertible}")

# === 2×2 Inverse Formula ===
B = np.array([[2, 1],
              [4, 3]])

det_B = np.linalg.det(B)
B_inv_formula = (1/det_B) * np.array([[B[1,1], -B[0,1]],
                                       [-B[1,0], B[0,0]]])
B_inv_numpy = np.linalg.inv(B)

print(f"Formula: {B_inv_formula}")
print(f"NumPy: {B_inv_numpy}")

# === Properties ===
C = np.array([[1, 2],
              [3, 4]])
D = np.array([[5, 6],
              [7, 8]])

# det(AB) = det(A)det(B)
print(f"det(CD) = {np.linalg.det(C @ D):.4f}")
print(f"det(C)det(D) = {np.linalg.det(C) * np.linalg.det(D):.4f}")

# det(Aᵀ) = det(A)
print(f"det(Cᵀ) = {np.linalg.det(C.T):.4f}")
print(f"det(C) = {np.linalg.det(C):.4f}")

# (AB)⁻¹ = B⁻¹A⁻¹
CD_inv = np.linalg.inv(C @ D)
D_inv_C_inv = np.linalg.inv(D) @ np.linalg.inv(C)
print(f"(CD)⁻¹ ≈ D⁻¹C⁻¹: {np.allclose(CD_inv, D_inv_C_inv)}")

# === Cramer's Rule ===
def cramers_rule(A, b):
    """Solve Ax = b using Cramer's rule"""
    n = len(b)
    det_A = np.linalg.det(A)
    
    if abs(det_A) < 1e-10:
        raise ValueError("Matrix is singular")
    
    x = np.zeros(n)
    for i in range(n):
        A_i = A.copy()
        A_i[:, i] = b
        x[i] = np.linalg.det(A_i) / det_A
    
    return x

A_sys = np.array([[2, 1],
                  [4, 3]])
b_sys = np.array([5, 13])

x_cramer = cramers_rule(A_sys, b_sys)
x_numpy = np.linalg.solve(A_sys, b_sys)

print(f"Cramer's rule: {x_cramer}")
print(f"NumPy solve: {x_numpy}")

# === Singular Matrix ===
singular = np.array([[1, 2],
                     [2, 4]])

print(f"det(singular) = {np.linalg.det(singular):.10f}")
try:
    singular_inv = np.linalg.inv(singular)
except np.linalg.LinAlgError:
    print("Cannot invert singular matrix!")

# === Rank-1 Update (Determinant Lemma) ===
A = np.array([[1, 2],
              [3, 4]])
u = np.array([1, 0])
v = np.array([0, 1])

# det(A + uvᵀ)
lhs = np.linalg.det(A + np.outer(u, v))
rhs = np.linalg.det(A) * (1 + v @ np.linalg.inv(A) @ u)

print(f"det(A + uvᵀ) = {lhs:.4f}")
print(f"det(A)(1 + vᵀA⁻¹u) = {rhs:.4f}")
```

---

## 📊 Summary Table

| Concept | Formula/Property | Key Point |
|---------|------------------|-----------|
| **Inverse** | AA⁻¹ = A⁻¹A = I | Only for square, non-singular |
| **Determinant (2×2)** | ad - bc | Scaling factor |
| **Determinant (3×3)** | aei+bfg+cdh-ceg-bdi-afh | Sarrus' rule |
| **Cofactor** | Cᵢⱼ = (-1)ⁱ⁺ʲMᵢⱼ | For expansion, adjugate |
| **Invertibility** | det(A) ≠ 0 ⟺ invertible | Key test |
| **Cramer's Rule** | xᵢ = det(Aᵢ)/det(A) | Theoretical importance |
| **det(AB)** | det(A)det(B) | Multiplicative |
| **det(A⁻¹)** | 1/det(A) | Reciprocal |

---

## 🎯 ML Applications

| Application | How It's Used |
|-------------|---------------|
| **Linear Regression** | Normal equations: (XᵀX)⁻¹Xᵀy |
| **Gaussian Processes** | Covariance matrix inverse |
| **Change of Variables** | Jacobian determinant |
| **PCA** | Eigendecomposition requires invertibility |
| **Optimization** | Hessian determinant (second-order tests) |

---

## ❓ Quick Check

1. When is a matrix NOT invertible?
2. What does det(A) = 0 mean geometrically?
3. How do you compute inverse using Gaussian elimination?
4. What's the relationship between det(A) and det(A⁻¹)?
5. Why is Cramer's rule not used in practice?
6. What happens to determinant when you swap two rows?

---

## 📝 Answers to Quick Check

1. **Matrix NOT invertible when:**
   - det(A) = 0 (singular)
   - Rows/columns are linearly dependent
   - rank(A) < n (not full rank)
   - 0 is an eigenvalue
   - Ax = 0 has non-trivial solutions

2. **det(A) = 0 geometrically:**
   - Transformation **collapses space** to lower dimension
   - Area/volume becomes zero
   - Vectors squish into a line or plane
   - Information is lost (not reversible)

3. **Inverse via Gaussian elimination:**
   - Form augmented matrix [A | I]
   - Row reduce to [I | A⁻¹]
   - The right half is the inverse

4. **det(A) and det(A⁻¹):**
   - **det(A⁻¹) = 1/det(A)**
   - If det(A) = 5, then det(A⁻¹) = 1/5
   - Follows from det(AB) = det(A)det(B) and AA⁻¹ = I

5. **Cramer's rule limitations:**
   - **Computationally expensive**: O(n!) for large n
   - Requires computing n+1 determinants
   - Gaussian elimination is O(n³) - much faster
   - Mainly theoretical/educational value

6. **Swapping two rows:**
   - **det → -det** (sign flips)
   - Magnitude stays the same
   - Two swaps → back to original

---

**Status:** ✅ Complete  
**Next:** Eigenvalues and Eigenvectors
