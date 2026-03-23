# 1.1.3 Systems of Linear Equations

## 🎯 Quick Overview
- **Goal**: Solve for variables in linear equations
- **Form**: Ax = b
- **Methods**: Gaussian elimination, matrix inverse
- **Applications**: Linear regression, circuit analysis, optimization

---

## 1. Representation as Ax = b

### System of Equations
```
2x + 3y -  z =  5
4x + y +  2z =  8
x - 2y +  3z = -1
```

### Matrix Form
```
⎡ 2   3  -1 ⎤ ⎡ x ⎤   ⎡  5 ⎤
⎢ 4   1   2 ⎥ ⎢ y ⎥ = ⎢  8 ⎥
⎣ 1  -2   3 ⎦ ⎣ z ⎦   ⎣ -1 ⎦

     A         x       b
```

### Augmented Matrix
```
⎡ 2   3  -1  │  5 ⎤
⎢ 4   1   2  │  8 ⎥
⎣ 1  -2   3  │ -1 ⎦
```

---

## 2. Gaussian Elimination

### Goal
Transform to **Row Echelon Form (REF)** using row operations.

### Row Operations
1. **Swap**: Rᵢ ↔ Rⱼ
2. **Scale**: Rᵢ → cRᵢ (c ≠ 0)
3. **Replace**: Rᵢ → Rᵢ + cRⱼ

### Algorithm
```
For each column k:
  1. Find pivot (largest element in column k)
  2. Swap rows if needed
  3. Eliminate below: Rᵢ → Rᵢ - (aᵢₖ/aₖₖ)Rₖ
```

### Example
```
Initial:
⎡ 2   3  -1  │  5 ⎤
⎢ 4   1   2  │  8 ⎥
⎣ 1  -2   3  │ -1 ⎦

R₂ → R₂ - 2R₁:
⎡ 2   3  -1  │   5  ⎤
⎢ 0  -5   4  │  -2  ⎥
⎣ 1  -2   3  │  -1  ⎦

R₃ → R₃ - 0.5R₁:
⎡ 2   3  -1  │   5   ⎤
⎢ 0  -5   4  │  -2   ⎥
⎣ 0  -3.5 3.5│ -3.5  ⎦

Continue until upper triangular...
```

---

## 3. Gauss-Jordan Elimination

### Goal
Transform to **Reduced Row Echelon Form (RREF)** - diagonal of 1s, zeros above and below.

### Additional Step
After Gaussian elimination:
- Scale each row to make pivot = 1
- Eliminate ABOVE each pivot too

### Example (continuing from above)
```
After Gaussian elimination, continue:

Scale rows:
⎡ 1  1.5 -0.5 │  2.5 ⎤
⎢ 0   1  -0.8 │  0.4 ⎥
⎣ 0   0    1  │  1.0 ⎦

Back substitution (eliminate above):
⎡ 1   0   0  │  2 ⎤
⎢ 0   1   0  │  1 ⎥
⎣ 0   0   1  │  1 ⎦

Solution: x=2, y=1, z=1
```

---

## 4. Row Echelon Form (REF)

### Properties
1. All non-zero rows are above zero rows
2. Each pivot is to the right of the pivot above
3. All entries below pivots are zero

### Example REF
```
⎡ 2   3  -1   5 ⎤
⎢ 0  -5   4  -2 ⎥  ← Pivots: 2, -5, 7
⎣ 0   0   7   3 ⎦
```

### Non-Example
```
⎡ 2   3  -1   5 ⎤
⎢ 0   0   4  -2 ⎥  ✗ Pivot not strictly right
⎣ 0  -5   7   3 ⎦
```

---

## 5. Reduced Row Echelon Form (RREF)

### Properties (REF +)
4. All pivots = 1
5. All entries above AND below pivots = 0

### Example RREF
```
⎡ 1   0   0   2 ⎤
⎢ 0   1   0   1 ⎥  ← Identity-like
⎣ 0   0   1   1 ⎦
```

### Unique Solution
RREF gives the **unique** solution directly!

---

## 6. Pivot Positions and Columns

### Pivot Position
Location of leading entry in each row (after REF).

### Pivot Column
Column containing a pivot.

### Example
```
⎡ 2   3  -1   5 ⎤
⎢ 0  -5   4  -2 ⎥  ← Pivots at (1,1), (2,2), (3,3)
⎣ 0   0   7   3 ⎦

Pivot columns: 1, 2, 3
```

### Significance
- **Rank** = Number of pivots
- **Linear independence**: Pivot columns are independent
- **Basis**: Pivot columns form basis for column space

---

## 7. Free Variables and Basic Variables

### Basic Variables
Variables corresponding to pivot columns.

### Free Variables
Variables without pivots - can take ANY value.

### Example
```
RREF:
⎡ 1   2   0   3   0   5 ⎤
⎢ 0   0   1   4   0   6 ⎥
⎣ 0   0   0   0   1   7 ⎦

Variables: x₁, x₂, x₃, x₄, x₅
Pivot columns: 1, 3, 5

Basic variables: x₁, x₃, x₅
Free variables: x₂, x₄

Solution:
x₁ = 5 - 2x₂ - 3x₄
x₃ = 6 - 4x₄
x₅ = 7
x₂, x₄ are free
```

---

## 8. Existence and Uniqueness

### Existence Theorem
Ax = b has a solution **iff** rank(A) = rank([A|b])

### Cases

| Case | Condition | Solutions |
|------|-----------|-----------|
| **Unique** | rank(A) = n (full column rank) | Exactly one |
| **Infinite** | rank(A) < n, consistent | Infinitely many |
| **None** | rank(A) < rank([A|b]) | No solution |

### Visual (2 variables)
```
Unique:        Infinite:      None:
  \              ≡ (same line)   ∥ (parallel)
   \                            
    \                          
     • (one point)
```

---

## 9. Homogeneous Systems (Ax = 0)

### Properties
- **Always consistent** (x = 0 is always a solution)
- Solution set forms a **subspace**

### Trivial Solution
x = 0 (zero vector)

### Non-trivial Solutions
Exist **iff** A has free variables (rank < n)

### Example
```
x + 2y - z = 0
2x + 4y - 2z = 0  ← This is 2×(first equation)

RREF:
⎡ 1   2  -1   0 ⎤
⎢ 0   0   0   0 ⎥

Solution:
x = -2s + t
y = s  (free)
z = t  (free)

Infinite solutions!
```

---

## 10. Particular and General Solutions

### For Ax = b (non-homogeneous)

**General Solution = Particular Solution + Homogeneous Solution**

```
x_general = x_particular + x_homogeneous
```

### Example
```
System:
x + y = 3
2x + 2y = 6

Particular solution (find one):
x_p = [3, 0]ᵀ

Homogeneous (Ax = 0):
x + y = 0  →  y = -x
x_h = t[1, -1]ᵀ

General solution:
x = [3, 0]ᵀ + t[1, -1]ᵀ
```

---

## 💻 Python Code Examples

```python
import numpy as np
from scipy import linalg

# === System: Ax = b ===
A = np.array([[2, 3, -1],
              [4, 1,  2],
              [1, -2, 3]])

b = np.array([5, 8, -1])

# Method 1: Direct solve (uses LU decomposition)
x = linalg.solve(A, b)
print(f"Solution: {x}")  # [2. 1. 1.]

# Method 2: Matrix inverse
x = np.linalg.inv(A) @ b
print(f"Solution (inverse): {x}")

# Method 3: Least squares (for overdetermined)
x, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
print(f"Least squares solution: {x}")

# === Augmented Matrix ===
augmented = np.column_stack([A, b])
print(f"Augmented matrix:\n{augmented}")

# === Row Echelon Form (manual) ===
def gaussian_elimination(A, b):
    """Simple Gaussian elimination"""
    n = len(b)
    Aug = np.column_stack([A.astype(float), b.astype(float)])
    
    # Forward elimination
    for i in range(n):
        # Find pivot
        max_row = i + np.argmax(np.abs(Aug[i:, i]))
        Aug[[i, max_row]] = Aug[[max_row, i]]
        
        # Eliminate below
        for j in range(i + 1, n):
            factor = Aug[j, i] / Aug[i, i]
            Aug[j] -= factor * Aug[i]
    
    # Back substitution
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (Aug[i, -1] - np.dot(Aug[i, :n], x)) / Aug[i, i]
    
    return x, Aug

x, ref = gaussian_elimination(A, b)
print(f"REF:\n{ref}")
print(f"Solution: {x}")

# === Check for solutions ===
rank_A = np.linalg.matrix_rank(A)
rank_Ab = np.linalg.matrix_rank(np.column_stack([A, b]))

if rank_A == rank_Ab == len(b):
    print("Unique solution")
elif rank_A == rank_Ab:
    print("Infinite solutions")
else:
    print("No solution")

# === Homogeneous system ===
b_zero = np.zeros(3)
x_homo = linalg.solve(A, b_zero)
print(f"Homogeneous solution: {x_homo}")  # [0. 0. 0.]

# === Null space ===
U, S, Vh = np.linalg.svd(A)
null_space = Vh[np.linalg.matrix_rank(A):]
print(f"Null space basis:\n{null_space}")

# === Overdetermined system ===
A_over = np.array([[1, 1],
                   [2, 2],
                   [3, 3]])
b_over = np.array([2, 4, 6])

x_ls, _, _, _ = np.linalg.lstsq(A_over, b_over, rcond=None)
print(f"Least squares: {x_ls}")
```

---

## 📊 Summary Table

| Concept | Description | Key Point |
|---------|-------------|-----------|
| **Ax = b** | Matrix form | Compact representation |
| **Gaussian Elimination** | Forward elimination | Produces REF |
| **Gauss-Jordan** | Full elimination | Produces RREF |
| **REF** | Row Echelon Form | Zeros below pivots |
| **RREF** | Reduced REF | Zeros above & below |
| **Pivot** | Leading entry | Determines rank |
| **Free Variable** | No pivot | Creates infinite solutions |
| **Homogeneous** | Ax = 0 | Always has x = 0 |
| **Particular** | One solution to Ax = b | Part of general solution |

---

## 🎯 ML Applications

| Application | How It's Used |
|-------------|---------------|
| **Linear Regression** | Normal equations: AᵀAx = Aᵀb |
| **Deep Learning** | Backpropagation solves linear systems |
| **Circuit Analysis** | Kirchhoff's laws → Ax = b |
| **Optimization** | KKT conditions → linear systems |
| **Computer Graphics** | Transformations, projections |

---

## ❓ Quick Check

1. What's the difference between REF and RREF?
2. When does Ax = b have no solution?
3. What is a free variable?
4. Why is Ax = 0 always consistent?
5. How do you find the rank of a matrix?
6. What's the relationship between pivots and linear independence?

---

## 📝 Answers to Quick Check

<details>
<summary>Click to reveal answers</summary>

1. **REF vs RREF:**
   - **REF (Row Echelon Form)**: Zeros below pivots, pivots to the right
   - **RREF (Reduced REF)**: REF + pivots = 1 + zeros above AND below pivots
   - RREF is unique; REF is not

2. **No solution when:**
   - rank(A) < rank([A|b]) (inconsistent system)
   - Geometrically: Lines/planes are parallel and don't intersect
   - In RREF: Row like [0 0 ... 0 | c] where c ≠ 0

3. **Free variable:**
   - Variable corresponding to a column **without a pivot**
   - Can take ANY value (parameter)
   - Creates infinitely many solutions

4. **Ax = 0 always consistent because:**
   - **x = 0** (zero vector) is always a solution
   - Called the "trivial solution"
   - Solution set forms a subspace (null space)

5. **Finding rank:**
   - Count the number of **pivots** in REF/RREF
   - Or: rank(A) = number of linearly independent columns
   - In Python: `np.linalg.matrix_rank(A)`

6. **Pivots and linear independence:**
   - **Pivot columns are linearly independent**
   - Number of pivots = dimension of column space
   - If every column has a pivot → columns are independent

</details>
---

**Status:** ✅ Complete  
**Next:** Matrix Inverse and Determinants
