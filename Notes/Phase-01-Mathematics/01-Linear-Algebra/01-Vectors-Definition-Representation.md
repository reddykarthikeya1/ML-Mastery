# Vectors: Definition and Representation

## 🎯 Key Concepts
- Vector as a mathematical object with magnitude and direction
- Geometric representation (arrows in space)
- Algebraic representation (ordered lists of numbers)
- Dimension of a vector
- Column vectors vs row vectors

---

## 📚 Theory

### What is a Vector?

A **vector** is a mathematical object that has both:
1. **Magnitude** (length/size)
2. **Direction**

Vectors are fundamental to linear algebra and are used extensively in machine learning to represent:
- Data points (features)
- Model parameters (weights)
- Gradients for optimization
- Embeddings (word vectors, etc.)

---

### Geometric Representation

In geometry, a vector is represented as an **arrow** in space:

```
      ↑
      |  vector v
      |
      •------→
     origin
```

**Key Properties:**
- **Tail**: Starting point (usually origin)
- **Head**: Ending point (arrow tip)
- **Length**: Magnitude of the vector
- **Direction**: Where the arrow points

**In 2D Space:**
- A vector from origin (0,0) to point (3,2) is written as **v** = (3, 2)

**In 3D Space:**
- A vector from origin (0,0,0) to point (1,2,3) is written as **v** = (1, 2, 3)

---

### Algebraic Representation

Algebraically, a vector is an **ordered list of numbers**:

**Notation:**
- Bold lowercase letters: **v**, **u**, **w**
- Arrow notation: v⃗ (or **v** with arrow)
- Component form: **v** = (v₁, v₂, ..., vₙ)

**Column Vector (most common in ML):**
```
    [ v₁ ]
v = [ v₂ ]
    [ ...]
    [ vₙ ]
```

**Row Vector:**
```
v = [v₁  v₂  ...  vₙ]
```

**Example:**
```
    [ 3 ]
v = [ 1 ]    is a 3-dimensional column vector
    [ 4 ]
```

---

### Dimension

The **dimension** of a vector is the number of components:
- **2D vector**: **v** = (x, y) ∈ ℝ²
- **3D vector**: **v** = (x, y, z) ∈ ℝ³
- **nD vector**: **v** = (v₁, v₂, ..., vₙ) ∈ ℝⁿ

**In Machine Learning:**
- A data point with 100 features = 100-dimensional vector
- An image (28×28 pixels) = 784-dimensional vector
- A word embedding = typically 50-300 dimensional vector

---

## 💻 Code Examples (Python with NumPy)

```python
import numpy as np

# Creating vectors
v1 = np.array([3, 2])              # 2D vector
v2 = np.array([1, 2, 3])           # 3D vector
v3 = np.array([[3], [2], [1]])     # Column vector (3×1)

print(f"Vector v1: {v1}")
print(f"Vector v2: {v2}")
print(f"Vector v3 shape: {v3.shape}")

# Vector dimension
print(f"Dimension of v2: {len(v2)}")

# Vector magnitude (norm)
magnitude = np.linalg.norm(v2)
print(f"Magnitude of v2: {magnitude:.4f}")

# Unit vector (direction only)
unit_v2 = v2 / magnitude
print(f"Unit vector of v2: {unit_v2}")
```

**Output:**
```
Vector v1: [3 2]
Vector v2: [1 2 3]
Vector v3 shape: (3, 1)
Dimension of v2: 3
Magnitude of v2: 3.7417
Unit vector of v2: [0.2673 0.5345 0.8018]
```

---

## 🖼 Visual Representation

```
2D Vector v = (3, 2)

     y
     ↑
   2 +     * (3,2)
     |    /|
     |   / |
     |  /  |
     | /   |
     |/    |
   0 +-----+-----→ x
     0     3

Geometric: Arrow from (0,0) to (3,2)
Algebraic: v = [3, 2]ᵀ
Magnitude: √(3² + 2²) = √13 ≈ 3.61
```

---

## 🔗 Related Topics
- [[Matrix-Operations]]
- [[Vector-Operations]]
- [[Python-NumPy]]
- [[Dot-Product]]

---

## ❓ Practice Questions

### Level 1: Basic (Conceptual)
1. What's the difference between a point and a vector?
2. Can two vectors be equal if they start at different points?
3. What does the dimension of a vector tell us?
4. Why do we prefer column vectors in machine learning?
5. What are the two key properties of a vector?

### Level 2: Representation
1. Represent the point (5, -2, 1) as a column vector
2. What is the dimension of a vector with 100 components?
3. Write a 3D vector with components 2, -1, 4 in both row and column form
4. Convert the row vector [1, 2, 3] to a column vector

### Level 3: Computational (Magnitude & Unit Vectors)
*Note: These use concepts from the code examples above*
1. Find the magnitude of vector **v** = (4, 3)
2. Find the magnitude of vector **u** = (1, 2, 2)
3. Find the unit vector of **v** = (3, 4)
4. Verify that the unit vector of **u** = (1, 1, 1) has magnitude 1

### Level 4: Python Practice
```python
import numpy as np

# 1. Create a 4-dimensional vector [2, 4, 1, 3]
v = np.array([2, 4, 1, 3])

# 2. Calculate its magnitude using np.linalg.norm()
magnitude = np.linalg.norm(v)  # √(4+16+1+9) = √30 ≈ 5.477

# 3. Convert it to a unit vector
unit_v = v / magnitude

# 4. Verify the unit vector has magnitude 1
print(np.linalg.norm(unit_v))  # 1.0

# 5. Create a column vector (shape should be (4, 1))
col_v = v.reshape(4, 1)

# Bonus: Verify triangle inequality
v1 = np.array([1, 2, 3])
v2 = np.array([4, 5, 6])
print(np.linalg.norm(v1 + v2) <= np.linalg.norm(v1) + np.linalg.norm(v2))  # True
```

---

## 📝 Answers to Practice Questions

### Level 1: Basic (Conceptual)
1. **What's the difference between a point and a vector?**
   - A **point** represents a location in space (e.g., (3, 4))
   - A **vector** represents direction and magnitude (e.g., [3, 4]ᵀ)
   - Points are fixed; vectors can be moved as long as direction and magnitude stay the same

2. **Can two vectors be equal if they start at different points?**
   - **Yes!** Vectors are equal if they have the same direction and magnitude, regardless of starting point

3. **What does the dimension of a vector tell us?**
   - The number of components/features in the vector
   - The space the vector lives in (ℝⁿ)

4. **Why do we prefer column vectors in machine learning?**
   - Consistent with matrix multiplication conventions (Ax = b)
   - Standard in linear algebra notation
   - Works naturally with NumPy and ML libraries

5. **What are the two key properties of a vector?**
   - **Magnitude** (length/size)
   - **Direction**

### Level 2: Representation
1. **Represent the point (5, -2, 1) as a column vector:**
   ```
   [  5 ]
   [ -2 ]
   [  1 ]
   ```

2. **What is the dimension of a vector with 100 components?**
   - **100** (it's in ℝ¹⁰⁰)

3. **Write a 3D vector with components 2, -1, 4 in both row and column form:**
   - Row: [2, -1, 4]
   - Column:
   ```
   [  2 ]
   [ -1 ]
   [  4 ]
   ```

4. **Convert the row vector [1, 2, 3] to a column vector:**
   ```
   [ 1 ]
   [ 2 ]
   [ 3 ]
   ```

### Level 3: Computational (Magnitude & Unit Vectors)
1. **Find the magnitude of vector v = (4, 3):**
   - |v| = √(4² + 3²) = √(16 + 9) = √25 = **5**

2. **Find the magnitude of vector u = (1, 2, 2):**
   - |u| = √(1² + 2² + 2²) = √(1 + 4 + 4) = √9 = **3**

3. **Find the unit vector of v = (3, 4):**
   - |v| = √(9 + 16) = 5
   - unit(v) = (3/5, 4/5) = **(0.6, 0.8)**

4. **Verify that the unit vector of u = (1, 1, 1) has magnitude 1:**
   - |u| = √(1 + 1 + 1) = √3
   - unit(u) = (1/√3, 1/√3, 1/√3)
   - |unit(u)| = √(1/3 + 1/3 + 1/3) = √1 = **1** ✓

---

## 📌 Summary

| Aspect | Description |
|--------|-------------|
| **Definition** | Object with magnitude and direction |
| **Geometric** | Arrow in space (has direction and length) |
| **Algebraic** | Ordered list of numbers |
| **Notation** | Bold lowercase: **v**, or arrow: v⃗ |
| **Dimension** | Number of components (n) |
| **Column Vector** | Vertical arrangement (n×1 matrix) |
| **Row Vector** | Horizontal arrangement (1×n matrix) |
| **Magnitude** | Length: \|**v**\| = √(v₁² + v₂² + ... + vₙ²) |

**Key Takeaways:**
- Vectors bridge geometry and algebra
- Essential for representing data in ML
- Can be visualized geometrically or manipulated algebraically
- Column vectors are standard in linear algebra

---

**Created:** 2026-03-22  
**Last Updated:** 2026-03-22  
**Status:** ✅ Complete
