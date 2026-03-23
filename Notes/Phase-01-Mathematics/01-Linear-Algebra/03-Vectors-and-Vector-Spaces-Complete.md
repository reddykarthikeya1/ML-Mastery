# 1.1.1 Vectors and Vector Spaces

## 🎯 Quick Overview
- **Vector**: Object with magnitude and direction
- **Vector Space**: Collection of vectors with defined operations
- **Foundation for**: All linear algebra, ML feature spaces, embeddings

---

## 1. Vector Definition and Representation

### Geometric vs Algebraic

| Aspect | Geometric | Algebraic |
|--------|-----------|-----------|
| **Representation** | Arrow in space | Ordered list of numbers |
| **Visual** | → (direction + length) | **v** = (v₁, v₂, ..., vₙ) |
| **Use Case** | Physics, graphics | Computation, ML |

**Column Vector (Standard):**
```
    [ v₁ ]
v = [ v₂ ]  ∈ ℝⁿ
    [ ... ]
    [ vₙ ]
```

**Row Vector:**
```
v = [v₁  v₂  ...  vₙ]  ∈ ℝⁿ
```

---

## 2. Vector Operations

### Addition & Subtraction
```
    [ a₁ ]   [ b₁ ]   [ a₁ ± b₁ ]
a ± b = [ a₂ ] ± [ b₂ ] = [ a₂ ± b₂ ]
    [ ... ]   [ ... ]   [   ...   ]
    [ aₙ ]   [ bₙ ]   [ aₙ ± bₙ ]
```

### Scalar Multiplication
```
        [ a₁ ]   [ c·a₁ ]
c · a = c · [ a₂ ] = [ c·a₂ ]
        [ ... ]   [  ... ]
        [ aₙ ]   [ c·aₙ ]
```

**Properties:**
- ✅ Commutative: **a** + **b** = **b** + **a**
- ✅ Associative: (**a** + **b**) + **c** = **a** + (**b** + **c**)
- ✅ Distributive: c(**a** + **b**) = c**a** + c**b**

---

## 3. Dot Product (Inner Product)

### Definition
```
a · b = a₁b₁ + a₂b₂ + ... + aₙbₙ = Σ aᵢbᵢ
```

### Geometric Interpretation
```
a · b = |a| |b| cos(θ)
```
where θ is the angle between vectors

**Key Insights:**
- **a · b > 0**: Angle < 90° (similar direction)
- **a · b = 0**: Orthogonal (perpendicular)
- **a · b < 0**: Angle > 90° (opposite direction)

### Properties
| Property | Formula |
|----------|---------|
| Commutative | **a** · **b** = **b** · **a** |
| Distributive | **a** · (**b** + **c**) = **a**·**b** + **a**·**c** |
| Scalar | (c**a**) · **b** = c(**a** · **b**) |
| Self-dot | **a** · **a** = \|**a**\|² |

### Applications in ML
- **Cosine similarity**: sim(a,b) = (a·b)/(\|a\|\|b\|)
- **Projection**: proj_b(a) = (a·b/\|b\|²) **b**
- **Neural networks**: Weighted sum = w · x

---

## 4. Cross Product (3D Only)

### Definition
```
        [ a₂b₃ - a₃b₂ ]
a × b = [ a₃b₁ - a₁b₃ ]
        [ a₁b₂ - a₂b₁ ]
```

### Geometric Interpretation
- **Result**: Vector perpendicular to both **a** and **b**
- **Magnitude**: \|**a** × **b**\| = \|**a**\| \|**b**\| sin(θ)
- **Direction**: Right-hand rule

**Properties:**
- ❌ NOT commutative: **a** × **b** = -(**b** × **a**)
- ✅ Distributive: **a** × (**b** + **c**) = **a**×**b** + **a**×**c**
- **a** × **a** = **0**

### Applications
- Computing normal vectors (computer graphics)
- Torque and angular momentum (physics)
- Area of parallelogram: Area = \|**a** × **b**\|

---

## 5. Vector Magnitude (Norm) and Unit Vectors

### Euclidean Norm (L2 Norm)
```
\|v\| = √(v₁² + v₂² + ... + vₙ²) = √(Σ vᵢ²)
```

### Other Norms
| Norm | Formula | Use Case |
|------|---------|----------|
| **L1 Norm** | \|v\|₁ = Σ\|vᵢ\| | Sparse solutions |
| **L2 Norm** | \|v\|₂ = √(Σ vᵢ²) | Default (Euclidean) |
| **L∞ Norm** | \|v\|∞ = max(\|vᵢ\|) | Worst-case analysis |

### Unit Vector
```
unit(v) = v / \|v\|
```
**Properties:**
- \|unit(v)\| = 1
- Preserves direction, normalizes magnitude

---

## 6. Linear Combinations

### Definition
```
c₁v₁ + c₂v₂ + ... + cₖvₖ
```
where cᵢ are scalars and vᵢ are vectors

### Example
```
    [ 1 ]       [ 0 ]   [ 2 ]
2 · [ 0 ] + 3 · [ 1 ] = [ 3 ]
    [ 0 ]       [ 0 ]   [ 0 ]
```

**ML Application:** Neural network output = linear combination of inputs

---

## 7. Linear Independence and Dependence

### Linear Independence
Vectors {**v₁**, **v₂**, ..., **vₖ**} are **linearly independent** if:
```
c₁v₁ + c₂v₂ + ... + cₖvₖ = 0  implies  c₁ = c₂ = ... = cₖ = 0
```

### Linear Dependence
Vectors are **linearly dependent** if at least one vector can be written as a linear combination of others.

**Test:**
```
If det([v₁ v₂ ... vₖ]) ≠ 0  →  Independent
If det([v₁ v₂ ... vₖ]) = 0   →  Dependent
```

**Visual Test (2D):**
- Independent: Vectors point in different directions
- Dependent: Vectors are collinear (parallel)

---

## 8. Span of a Set of Vectors

### Definition
**Span{v₁, v₂, ..., vₖ}** = Set of ALL linear combinations of these vectors

```
Span{v₁, ..., vₖ} = {c₁v₁ + ... + cₖvₖ : cᵢ ∈ ℝ}
```

### Examples
| Vectors | Span |
|---------|------|
| Single non-zero vector in ℝ² | Line through origin |
| Two non-collinear vectors in ℝ² | Entire ℝ² |
| Two non-collinear vectors in ℝ³ | Plane through origin |
| Three non-coplanar vectors in ℝ³ | Entire ℝ³ |

---

## 9. Basis Vectors and Coordinate Systems

### Basis
A **basis** for a vector space is:
1. Linearly independent set
2. Spans the entire space

### Standard Basis for ℝⁿ
```
    [ 1 ]      [ 0 ]         [ 0 ]
e₁ = [ 0 ], e₂ = [ 1 ], ..., eₙ = [ 0 ]
    [ 0 ]      [ 0 ]         [ 1 ]
    [ 0 ]      [ 0 ]         [ 0 ]
```

### Coordinates
Any vector **v** can be uniquely written as:
```
v = c₁e₁ + c₂e₂ + ... + cₙeₙ
```
where cᵢ are the coordinates

---

## 10. Vector Spaces and Subspaces

### Vector Space (ℝⁿ)
A set closed under:
1. Vector addition
2. Scalar multiplication

**Must satisfy:**
- Contains zero vector
- Closed under addition: **u**, **v** ∈ V → **u** + **v** ∈ V
- Closed under scalar multiplication: **v** ∈ V, c ∈ ℝ → c**v** ∈ V

### Subspace
A subset of a vector space that is itself a vector space.

**Common Subspaces:**
- Line through origin
- Plane through origin
- Any span of vectors

---

## 11. Null Space, Column Space, Row Space

For matrix A (m×n):

### Column Space (Range)
```
Col(A) = {Ax : x ∈ ℝⁿ} = Span{columns of A}
```
- Subspace of ℝᵐ
- All possible outputs of Ax

### Row Space
```
Row(A) = Span{rows of A} = Col(Aᵀ)
```
- Subspace of ℝⁿ

### Null Space (Kernel)
```
Null(A) = {x : Ax = 0}
```
- Subspace of ℝⁿ
- All vectors that map to zero

### Fundamental Theorem
```
dim(Col(A)) = dim(Row(A)) = rank(A)
dim(Null(A)) = n - rank(A)
dim(Col(A)) + dim(Null(A)) = n
```

---

## 💻 Python Code Examples

```python
import numpy as np

# === Vector Operations ===
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

# Addition, Subtraction
print(f"a + b = {a + b}")
print(f"a - b = {a - b}")

# Scalar multiplication
print(f"3 * a = {3 * a}")

# === Dot Product ===
dot_product = np.dot(a, b)
print(f"a · b = {dot_product}")

# Angle between vectors
cos_theta = dot_product / (np.linalg.norm(a) * np.linalg.norm(b))
angle = np.arccos(cos_theta)
print(f"Angle: {np.degrees(angle):.2f}°")

# === Cross Product (3D) ===
cross = np.cross(a, b)
print(f"a × b = {cross}")

# === Norms ===
print(f"L2 norm: {np.linalg.norm(a)}")
print(f"L1 norm: {np.linalg.norm(a, ord=1)}")
print(f"L∞ norm: {np.linalg.norm(a, ord=np.inf)}")

# === Unit Vector ===
unit_a = a / np.linalg.norm(a)
print(f"Unit vector: {unit_a}")

# === Linear Independence Test ===
v1 = np.array([1, 0, 0])
v2 = np.array([0, 1, 0])
v3 = np.array([0, 0, 1])
matrix = np.column_stack([v1, v2, v3])
det = np.linalg.det(matrix)
print(f"Determinant: {det:.2f} (≠0 → Independent)")

# === Column Space, Null Space ===
A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

# Rank
rank = np.linalg.matrix_rank(A)
print(f"Rank: {rank}")

# Null space (using SVD)
U, S, Vh = np.linalg.svd(A)
null_space = Vh[rank:]
print(f"Null space basis: {null_space}")
```

---

## 📊 Summary Table

| Concept | Formula/Definition | Key Property |
|---------|-------------------|--------------|
| **Dot Product** | a·b = Σaᵢbᵢ | a·b = \|a\|\|b\|cos(θ) |
| **Cross Product** | a×b (3D only) | Perpendicular to a,b |
| **L2 Norm** | \|v\| = √(Σvᵢ²) | Euclidean length |
| **Unit Vector** | v/\|v\| | Magnitude = 1 |
| **Linear Independence** | Σcᵢvᵢ = 0 → all cᵢ = 0 | No redundancy |
| **Span** | All linear combinations | Reachable space |
| **Basis** | Independent + spans | Minimal spanning set |
| **Column Space** | {Ax : x ∈ ℝⁿ} | Range of A |
| **Null Space** | {x : Ax = 0} | Kernel of A |

---

## 🎯 ML Applications

| Application | Linear Algebra Concept |
|-------------|----------------------|
| **PCA** | Eigenvectors, orthogonal projection |
| **Neural Networks** | Linear combinations, dot products |
| **Word Embeddings** | Vector spaces, cosine similarity |
| **Recommendation Systems** | Matrix factorization, SVD |
| **Linear Regression** | Normal equations: AᵀAx = Aᵀb |
| **Feature Engineering** | Basis transformations |

---

## ❓ Quick Check Questions

1. What's the difference between geometric and algebraic vector representation?
2. When is the dot product zero?
3. Can you take a cross product in 4D? Why/why not?
4. What does linear independence mean intuitively?
5. How do you test if vectors form a basis?
6. What is the relationship between rank and null space dimension?

---

## 📝 Answers to Quick Check

1. **Geometric vs Algebraic:**
   - **Geometric**: Arrow with direction and magnitude (visual)
   - **Algebraic**: Ordered list of numbers (computational)
   - Both represent the same mathematical object

2. **Dot product is zero when:**
   - Vectors are **orthogonal** (perpendicular, θ = 90°)
   - OR at least one vector is the zero vector

3. **Cross product in 4D?**
   - **No!** Cross product is only defined in ℝ³ (and ℝ⁷)
   - In 4D, use wedge product or exterior algebra instead

4. **Linear independence (intuitive):**
   - No vector is redundant
   - Each vector adds a new "direction"
   - Cannot write any vector as combination of others

5. **Test for basis:**
   - Check if vectors are **linearly independent** (det ≠ 0)
   - Check if vectors **span** the space (n vectors in ℝⁿ)
   - Form matrix and check if rank = n

6. **Rank-Nullity Theorem:**
   - **rank(A) + dim(Null(A)) = n** (number of columns)
   - Dimension of column space + dimension of null space = total dimensions

---

**Status:** ✅ Complete  
**Next:** Matrices and Matrix Operations
