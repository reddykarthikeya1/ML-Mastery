# Linear Algebra - Practice Problems

## Topic 1: Vectors - Definition and Representation

### Level 1: Basic

**1.1** Represent the following points as column vectors:
- a) (3, 5)
- b) (1, -2, 4)
- c) (-1, 0, 2, 3)

**1.2** Find the magnitude of these vectors:
- a) **v** = (3, 4)
- b) **u** = (1, 2, 2)
- c) **w** = (5, 0, 0, 0)

**1.3** Convert these vectors to unit vectors:
- a) **v** = (1, 0)
- b) **u** = (2, 2)
- c) **w** = (1, 1, 1)

---

### Level 2: Intermediate

**1.4** A vector **v** = (x, y) has magnitude 10 and points in the same direction as **u** = (3, 4). Find **v**.

**1.5** In a 2D plane, a vector starts at point A(1, 2) and ends at point B(4, 6). 
- a) Represent this as a vector
- b) Find its magnitude
- c) Find its unit vector

**1.6** Python Practice:
```python
import numpy as np

# Create the following vectors:
v1 = np.array([1, 2, 3, 4, 5])
v2 = np.array([5, 4, 3, 2, 1])

# 1. Find the magnitude of each
# 2. Create unit vectors
# 3. Verify: |unit vector| = 1
```

---

## Topic 2: Vector Operations

### Level 1: Basic

**2.1** Given **a** = (2, 3) and **b** = (1, -1), compute:
- a) **a** + **b**
- b) **a** - **b**
- c) 3**a**
- d) -2**b**
- e) 2**a** + 3**b**

**2.2** Given **u** = [1, 0, 2]ᵀ and **v** = [3, -1, 1]ᵀ, find:
- a) **u** + **v**
- b) 4**u** - **v**
- c) 2(**u** + 3**v**)

**2.3** Verify the commutative property for:
- **a** = (1, 2, 3)
- **b** = (4, 5, 6)
Show that **a** + **b** = **b** + **a**

---

### Level 2: Intermediate

**2.4** Solve for **x**:
- a) **x** + (2, 3) = (5, 1)
- b) 3**x** - (1, 2) = (2, 4)
- c) 2**x** + 3(1, -1) = (5, 1)

**2.5** Find scalars c₁ and c₂ such that:
c₁(1, 2) + c₂(3, 1) = (5, 4)

**2.6** Prove that for any vector **v**:
- a) |2**v**| = 2|**v**|
- b) |-**v**| = |**v**|
- c) |c**v**| = |c|·|**v**|

**2.7** Triangle Inequality:
Verify that |**a** + **b**| ≤ |**a**| + |**b**| for:
- **a** = (3, 4)
- **b** = (1, 2)

---

### Level 3: Advanced

**2.8** Linear Combination:
Express **w** = (7, 5, 9) as a linear combination of:
- **u** = (1, 1, 2)
- **v** = (2, 1, 1)

Find c₁ and c₂ such that: c₁**u** + c₂**v** = **w**

**2.9** Vector Geometry:
Three points in space:
- A = (1, 2, 3)
- B = (4, 5, 6)
- C = (7, 8, 9)

- a) Find vectors AB and BC
- b) Show that A, B, C are collinear (lie on same line)

**2.10** Python Challenge:
```python
import numpy as np
import matplotlib.pyplot as plt

# Create two 2D vectors
a = np.array([3, 1])
b = np.array([1, 2])

# 1. Plot vectors a, b, and a+b
# 2. Show the parallelogram visually
# 3. Verify triangle inequality numerically
```

---

## Solutions (Attempt First!)

<details>
<summary>Click to reveal solutions</summary>

### 1.1
```
a) [3, 5]ᵀ    b) [1, -2, 4]ᵀ    c) [-1, 0, 2, 3]ᵀ
```

### 1.2
```
a) |v| = √(9+16) = 5
b) |u| = √(1+4+4) = 3
c) |w| = 5
```

### 1.3
```
a) (1, 0)
b) (2/√8, 2/√8) = (1/√2, 1/√2)
c) (1/√3, 1/√3, 1/√3)
```

### 2.1
```
a) (3, 2)    b) (1, 4)    c) (6, 9)    d) (-2, 2)    e) (7, 3)
```

### 2.2
```
a) [4, -1, 3]ᵀ    b) [1, 1, 7]ᵀ    c) [20, -2, 10]ᵀ
```

### 2.3
```
a + b = (5, 7, 9)
b + a = (5, 7, 9)  ✓
```

### 2.4
```
a) x = (3, -2)
b) x = (1, 2)
c) x = (1, 2)
```

### 2.5
```
c₁ = 1, c₂ = 2
Check: 1(1,2) + 2(3,1) = (1,2) + (6,2) = (7,4) ✗
Correct: c₁ = 7/5, c₂ = 6/5
```

### 2.7
```
|a + b| = |(4, 6)| = √52 ≈ 7.21
|a| + |b| = 5 + √5 ≈ 7.24
7.21 ≤ 7.24  ✓
```

</details>

---

## 📝 Notes Section

Use this space for additional problems you encounter:

### My Practice Problems:


### Mistakes to Review:


### Key Insights:


---
**Last Updated:** 2026-03-22
