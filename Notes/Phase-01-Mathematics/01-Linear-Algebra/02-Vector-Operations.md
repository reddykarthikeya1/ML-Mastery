# Vector Operations

## 🎯 Key Concepts
- Vector addition and subtraction
- Scalar multiplication
- Geometric interpretation of operations
- Properties of vector operations
- Linear combinations

---

## 📚 Theory

### 1. Vector Addition

**Algebraic Definition:**
To add two vectors, add their corresponding components:

```
    [ a₁ ]   [ b₁ ]   [ a₁ + b₁ ]
a + [ a₂ ] + [ b₂ ] = [ a₂ + b₂ ]
    [ ...]   [ ...]   [   ...   ]
    [ aₙ ]   [ bₙ ]   [ aₙ + bₙ ]
```

**Example:**
```
    [ 1 ]   [ 4 ]   [ 5 ]
a + [ 2 ] + [ 5 ] = [ 7 ]
    [ 3 ]   [ 6 ]   [ 9 ]
```

**Geometric Interpretation (Parallelogram Rule):**
```
     b
     ↑
     |    a + b
     |   ↗
     |  /
     | /
     |/
     •------→
        a

To add b to a: Place tail of b at head of a
Result: Vector from tail of a to head of b
```

**Properties:**
- **Commutative**: **a** + **b** = **b** + **a**
- **Associative**: (**a** + **b**) + **c** = **a** + (**b** + **c**)
- **Identity**: **a** + **0** = **a**
- **Inverse**: **a** + (-**a**) = **0**

---

### 2. Vector Subtraction

**Algebraic Definition:**
Subtract corresponding components:

```
    [ a₁ ]   [ b₁ ]   [ a₁ - b₁ ]
a - [ a₂ ] - [ b₂ ] = [ a₂ - b₂ ]
    [ ...]   [ ...]   [   ...   ]
    [ aₙ ]   [ bₙ ]   [ aₙ - bₙ ]
```

**Example:**
```
    [ 5 ]   [ 2 ]   [ 3 ]
a - [ 7 ] - [ 3 ] = [ 4 ]
    [ 9 ]   [ 4 ]   [ 5 ]
```

**Geometric Interpretation:**
```
     b      a - b
     ↑     ↙
     |    /
     |   /
     |  /
     | /
     •------→
        a

a - b points from head of b to head of a
```

**Note:** **a** - **b** = **a** + (-**b**)

---

### 3. Scalar Multiplication

**Definition:**
Multiply every component by the scalar:

```
      [ a₁ ]   [ c·a₁ ]
c · a = [ a₂ ] = [ c·a₂ ]
      [ ...]   [  ... ]
      [ aₙ ]   [ c·aₙ ]
```

**Example:**
```
      [ 2 ]   [ 6  ]
3 · a = [ 1 ] = [ 3  ]
      [ 4 ]   [ 12 ]
```

**Geometric Interpretation:**
```
     a         2a        -a
     ↑         ↑         ↓
     |         |         |
     |         |         |
     |         |         |
     •--→      •----→    •
                (2× longer, same direction)
                         (same length, opposite direction)
```

**Effects:**
- **c > 1**: Stretches the vector
- **0 < c < 1**: Shrinks the vector
- **c < 0**: Reverses direction
- **c = 0**: Results in zero vector

**Properties:**
- **Distributive**: c(**a** + **b**) = c**a** + c**b**
- **Associative**: (cd)**a** = c(d**a**)
- **Identity**: 1 · **a** = **a**
- **Zero**: 0 · **a** = **0**

---

### 4. Linear Combinations

**Definition:**
A linear combination of vectors **v₁**, **v₂**, ..., **vₙ** with scalars c₁, c₂, ..., cₙ is:

```
c₁v₁ + c₂v₂ + ... + cₙvₙ
```

**Example:**
```
    [ 1 ]       [ 0 ]   [ 2 ]
2 · [ 0 ] + 3 · [ 1 ] = [ 3 ]
    [ 0 ]       [ 0 ]   [ 0 ]
```

**Importance in ML:**
- Weighted sum of features
- Neural network layers compute linear combinations
- Principal components are linear combinations of original features

---

## 💻 Code Examples (Python with NumPy)

```python
import numpy as np

# Define vectors
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

# Vector Addition
addition = a + b
print(f"a + b = {addition}")  # [5 7 9]

# Vector Subtraction
subtraction = a - b
print(f"a - b = {subtraction}")  # [-3 -3 -3]

# Scalar Multiplication
scalar = 3
scaled = scalar * a
print(f"3 * a = {scaled}")  # [3 6 9]

# Linear Combination: 2a + 3b
linear_comb = 2 * a + 3 * b
print(f"2a + 3b = {linear_comb}")  # [14 19 24]

# Magnitude after scaling
v = np.array([3, 4])
print(f"|v| = {np.linalg.norm(v)}")      # 5.0
print(f"|2v| = {np.linalg.norm(2*v)}")   # 10.0 (doubled!)

# Unit vector
unit_v = v / np.linalg.norm(v)
print(f"Unit vector: {unit_v}")          # [0.6 0.8]
print(f"|unit vector| = {np.linalg.norm(unit_v)}")  # 1.0
```

**Output:**
```
a + b = [5 7 9]
a - b = [-3 -3 -3]
3 * a = [3 6 9]
2a + 3b = [14 19 24]
|v| = 5.0
|2v| = 10.0
Unit vector: [0.6 0.8]
|unit vector| = 1.0
```

---

## 🖼 Visual Summary

```
Vector Addition (Parallelogram Rule):

  b
  ↑
  |    a + b (diagonal)
  |   ↗
  |  /|
  | / |
  |/  |
  •---+---→
      a

Vector Subtraction:

  b      a - b
  ↑     ↙
  |    /
  |   /
  |  /
  | /
  •------→
     a

Scalar Multiplication:

a:  →
2a: ----→  (doubled)
-a: ←      (reversed)
0.5a: →    (halved)
```

---

## 🔗 Related Topics
- [[Vectors-Definition-Representation]]
- [[Dot-Product]]
- [[Matrix-Operations]]
- [[Linear-Combinations]]

---

## ❓ Practice Questions

### Level 1: Conceptual
1. Is vector addition commutative? Show with an example.
2. What happens to a vector when multiplied by -1?
3. Can the sum of two non-zero vectors be zero? When?
4. What is a linear combination?
5. Explain the geometric interpretation of scalar multiplication.

### Level 2: Basic Computations
*Practice vector addition, subtraction, and scalar multiplication*

1. Given **a** = (2, 3) and **b** = (1, -1), find:
   - **a** + **b**
   - **a** - **b**
   - 3**a**
   - -2**b**

2. Given **u** = [1, 0, 2]ᵀ and **v** = [3, -1, 1]ᵀ, find:
   - **u** + **v**
   - 4**u** - **v**
   - 2(**u** + 3**v**)

### Level 3: Linear Combinations
1. Find scalars c₁ and c₂ such that:
   c₁(1, 2) + c₂(3, 1) = (5, 4)

2. Express **w** = (7, 5) as a linear combination of:
   **u** = (1, 1) and **v** = (2, 1)

### Level 4: Challenge Problems
1. If **v** = (3, 4), verify that |2**v**| = 2|**v**|
2. Verify the triangle inequality: |**a** + **b**| ≤ |**a**| + |**b**| for **a** = (1, 2), **b** = (3, 1)
3. Prove that for any vector **v** and scalar c: |c**v**| = |c|·|**v**|

### Level 5: Python Practice
```python
import numpy as np

# Given vectors:
u = np.array([1, 2, 3])
v = np.array([4, 0, -2])
w = np.array([2, 1, 1])

# 1. Compute u + v + w
print(u + v + w)  # [7, 3, 2]

# 2. Compute 2u - 3v + w
print(2*u - 3*v + w)  # [-8, 1, 13]

# 3. Find the unit vector of u (review from previous topic)
unit_u = u / np.linalg.norm(u)  # [0.267, 0.535, 0.802]

# 4. Verify: |u + v| ≤ |u| + |v| (Triangle Inequality)
print(np.linalg.norm(u + v) <= np.linalg.norm(u) + np.linalg.norm(v))  # True

# 5. Create a linear combination: 0.5u + 0.3v + 0.2w
print(0.5*u + 0.3*v + 0.2*w)  # [2.1, 1.3, 1.1]

# Bonus: Verify commutative property: u + v = v + u
print(np.array_equal(u + v, v + u))  # True
```

---

## 📝 Answers to Practice Questions

### Level 1: Conceptual
1. **Is vector addition commutative? Show with an example.**
   - **Yes!** a + b = b + a
   - Example: (1, 2) + (3, 4) = (4, 6) = (3, 4) + (1, 2)

2. **What happens to a vector when multiplied by -1?**
   - Direction is **reversed** (180° rotation)
   - Magnitude stays the same

3. **Can the sum of two non-zero vectors be zero? When?**
   - **Yes!** When they are equal in magnitude but opposite in direction
   - Example: (2, 3) + (-2, -3) = (0, 0)

4. **What is a linear combination?**
   - A sum of vectors, each multiplied by a scalar: c₁v₁ + c₂v₂ + ... + cₖvₖ

5. **Explain the geometric interpretation of scalar multiplication:**
   - **c > 1**: Stretches the vector
   - **0 < c < 1**: Shrinks the vector
   - **c < 0**: Reverses direction and scales

### Level 2: Basic Computations
1. **Given a = (2, 3) and b = (1, -1):**
   - **a + b** = (2+1, 3+(-1)) = **(3, 2)**
   - **a - b** = (2-1, 3-(-1)) = **(1, 4)**
   - **3a** = (6, 9)
   - **-2b** = (-2, 2)

2. **Given u = [1, 0, 2]ᵀ and v = [3, -1, 1]ᵀ:**
   - **u + v** = [4, -1, 3]ᵀ
   - **4u - v** = [4, 0, 8]ᵀ - [3, -1, 1]ᵀ = [1, 1, 7]ᵀ
   - **2(u + 3v)** = 2([1, 0, 2]ᵀ + [9, -3, 3]ᵀ) = 2[10, -3, 5]ᵀ = **[20, -6, 10]ᵀ**

### Level 3: Linear Combinations
1. **Find scalars c₁ and c₂ such that: c₁(1, 2) + c₂(3, 1) = (5, 4)**
   - System: c₁ + 3c₂ = 5  and  2c₁ + c₂ = 4
   - Solution: **c₁ = 7/5 = 1.4, c₂ = 6/5 = 1.2**
   - Check: 1.4(1, 2) + 1.2(3, 1) = (1.4, 2.8) + (3.6, 1.2) = (5, 4) ✓

2. **Express w = (7, 5) as a linear combination of u = (1, 1) and v = (2, 1):**
   - System: c₁ + 2c₂ = 7  and  c₁ + c₂ = 5
   - Solution: **c₁ = 3, c₂ = 2**
   - Check: 3(1, 1) + 2(2, 1) = (3, 3) + (4, 2) = (7, 5) ✓

### Level 4: Challenge Problems
1. **If v = (3, 4), verify that |2v| = 2|v|:**
   - |v| = √(9 + 16) = 5
   - 2v = (6, 8)
   - |2v| = √(36 + 64) = √100 = 10
   - 2|v| = 2(5) = 10
   - **10 = 10** ✓

2. **Verify the triangle inequality for a = (1, 2), b = (3, 1):**
   - |a| = √(1 + 4) = √5 ≈ 2.236
   - |b| = √(9 + 1) = √10 ≈ 3.162
   - a + b = (4, 3)
   - |a + b| = √(16 + 9) = 5
   - |a| + |b| ≈ 5.398
   - **5 ≤ 5.398** ✓

3. **Prove that for any vector v and scalar c: |cv| = |c|·|v|:**
   - |cv| = √((cv₁)² + (cv₂)² + ... + (cvₙ)²)
   - = √(c²v₁² + c²v₂² + ... + c²vₙ²)
   - = √(c²(v₁² + v₂² + ... + vₙ²))
   - = |c|√(v₁² + v₂² + ... + vₙ²)
   - = **|c|·|v|** ✓

---

## 📌 Summary

| Operation | Formula | Geometric Meaning |
|-----------|---------|-------------------|
| **Addition** | (a₁+b₁, a₂+b₂, ...) | Parallelogram diagonal |
| **Subtraction** | (a₁-b₁, a₂-b₂, ...) | Vector from b to a |
| **Scalar Mult.** | (ca₁, ca₂, ...) | Stretch/shrink/reverse |
| **Linear Comb.** | c₁v₁ + c₂v₂ + ... | Weighted sum |

**Key Properties:**
- ✅ Commutative: **a** + **b** = **b** + **a**
- ✅ Associative: (**a** + **b**) + **c** = **a** + (**b** + **c**)
- ✅ Distributive: c(**a** + **b**) = c**a** + c**b**
- ✅ Identity: **a** + **0** = **a**
- ✅ Inverse: **a** + (-**a**) = **0**

**ML Applications:**
- Feature scaling (scalar multiplication)
- Weighted sums in neural networks (linear combinations)
- Data augmentation (vector operations)
- Gradient updates (vector addition)

---

**Created:** 2026-03-22
**Last Updated:** 2026-03-23
**Status:** ✅ Complete
