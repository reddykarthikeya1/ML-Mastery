# 1.4.5 Combinatorics

## 🎯 Quick Overview
- **Counting**: Systematic enumeration
- **Permutations**: Ordered arrangements
- **Combinations**: Unordered selections
- **Foundation for**: Probability, algorithms, complexity analysis

---

## 1. Sum and Product Rules

### Sum Rule (OR)

```
If tasks are mutually exclusive:
Total ways = n₁ + n₂ + ... + nₖ

Example: Choose 1 from 5 shirts OR 4 pants = 9 choices
```

### Product Rule (AND)

```
If tasks are sequential:
Total ways = n₁ × n₂ × ... × nₖ

Example: Choose 1 from 5 shirts AND 4 pants = 20 outfits
```

---

## 2. Permutations

### Definition

**Ordered arrangement** of r elements from n distinct elements

### Formula

```
P(n, r) = n! / (n-r)!

Number of ways to arrange r items from n (order matters)
```

### Examples

```
P(5, 3) = 5! / 2! = 120 / 2 = 60

Arrange 3 books from 5:
5 × 4 × 3 = 60 ways
```

### With Repetition

```
nʳ ways to arrange r items from n with repetition

Example: 3-digit codes from {0-9}: 10³ = 1000
```

### With Identical Items

```
n! / (n₁! × n₂! × ... × nₖ!)

Example: MISSISSIPPI
11! / (4! × 4! × 2! × 1!) = 34,650
```

---

## 3. Combinations

### Definition

**Unordered selection** of r elements from n distinct elements

### Formula

```
C(n, r) = n! / (r! × (n-r)!)

Also written as: ₙCᵣ, (n choose r), binomial coefficient
```

### Examples

```
C(5, 3) = 5! / (3! × 2!) = 120 / 12 = 10

Choose 3 books from 5: 10 ways
```

### Properties

```
C(n, r) = C(n, n-r)
C(n, 0) = C(n, n) = 1
C(n, 1) = n
```

---

## 4. Binomial Theorem

### Formula

```
(x + y)ⁿ = Σₖ₌₀ⁿ C(n,k) xⁿ⁻ᵏ yᵏ

= C(n,0)xⁿ + C(n,1)xⁿ⁻¹y + ... + C(n,n)yⁿ
```

### Pascal's Triangle

```
     1        n=0
    1 1       n=1
   1 2 1      n=2
  1 3 3 1     n=3
 1 4 6 4 1    n=4

Each entry = sum of two above
Row n gives C(n, k) for k = 0 to n
```

### Pascal's Identity

```
C(n+1, k) = C(n, k-1) + C(n, k)
```

---

## 5. Inclusion-Exclusion Principle

### Two Sets

```
|A ∪ B| = |A| + |B| - |A ∩ B|
```

### Three Sets

```
|A ∪ B ∪ C| = |A| + |B| + |C| 
              - |A∩B| - |A∩C| - |B∩C| 
              + |A∩B∩C|
```

### General Form

```
|A₁ ∪ A₂ ∪ ... ∪ Aₙ| = 
  Σ|Aᵢ| - Σ|Aᵢ∩Aⱼ| + Σ|Aᵢ∩Aⱼ∩Aₖ| - ... + (-1)ⁿ⁻¹|A₁∩...∩Aₙ|
```

---

## 6. Pigeonhole Principle

### Basic Form

```
If n items are placed in m containers and n > m,
then at least one container has more than one item
```

### Generalized Form

```
At least one container has ⌈n/m⌉ items
```

### Examples

```
- In a group of 13 people, at least 2 have birthdays in same month
- In any group of 367 people, at least 2 share a birthday
- Among 10 numbers from 1-50, at least 2 have same last digit
```

---

## 7. Recurrence Relations

### Definition

**Equation defining sequence recursively**

### Examples

```
Fibonacci: F(n) = F(n-1) + F(n-2), F(0)=0, F(1)=1

Factorial: n! = n × (n-1)!, 0! = 1

Tower of Hanoi: T(n) = 2T(n-1) + 1, T(1) = 1
```

---

## 💻 Python Code Examples

```python
import numpy as np
from itertools import permutations, combinations

# === Factorial ===

def factorial(n):
    """Calculate factorial"""
    if n <= 1:
        return 1
    return n * factorial(n - 1)

print("Factorials:")
for n in range(1, 11):
    print(f"{n}! = {factorial(n)}")

# === Permutations ===

def permutation(n, r):
    """P(n, r)"""
    return factorial(n) // factorial(n - r)

print("\nPermutations:")
print(f"P(5, 3) = {permutation(5, 3)}")
print(f"P(10, 4) = {permutation(10, 4)}")

# Generate actual permutations
items = ['A', 'B', 'C', 'D']
print(f"\nPermutations of {items} taken 2 at a time:")
for p in permutations(items, 2):
    print(p)

# === Combinations ===

def combination(n, r):
    """C(n, r)"""
    return factorial(n) // (factorial(r) * factorial(n - r))

print("\nCombinations:")
print(f"C(5, 3) = {combination(5, 3)}")
print(f"C(10, 4) = {combination(10, 4)}")

# Generate actual combinations
print(f"\nCombinations of {items} taken 2 at a time:")
for c in combinations(items, 2):
    print(c)

# === Pascal's Triangle ===

def pascals_triangle(n_rows):
    """Generate Pascal's triangle"""
    triangle = []
    
    for n in range(n_rows):
        row = [combination(n, k) for k in range(n + 1)]
        triangle.append(row)
    
    return triangle

print("\nPascal's Triangle:")
triangle = pascals_triangle(10)
for i, row in enumerate(triangle):
    print(f"n={i}: {row}")

# === Inclusion-Exclusion ===

def inclusion_exclusion(sets):
    """Calculate union size using inclusion-exclusion"""
    from itertools import combinations
    
    n = len(sets)
    total = 0
    
    for r in range(1, n + 1):
        for subset in combinations(sets, r):
            intersection = set.intersection(*subset)
            term = len(intersection)
            
            if r % 2 == 1:
                total += term
            else:
                total -= term
    
    return total

# Example
A = {1, 2, 3, 4, 5}
B = {4, 5, 6, 7, 8}
C = {6, 7, 8, 9, 10}

print(f"\nInclusion-Exclusion:")
print(f"|A ∪ B ∪ C| = {inclusion_exclusion([A, B, C])}")
print(f"Direct: |A ∪ B ∪ C| = {len(A | B | C)}")

# === Pigeonhole Principle Demo ===

def pigeonhole_demo(n_items, m_containers):
    """Demonstrate pigeonhole principle"""
    import random
    
    # Randomly distribute items
    distribution = [0] * m_containers
    for _ in range(n_items):
        container = random.randint(0, m_containers - 1)
        distribution[container] += 1
    
    max_count = max(distribution)
    min_guaranteed = (n_items + m_containers - 1) // m_containers
    
    print(f"\nPigeonhole Demo:")
    print(f"Items: {n_items}, Containers: {m_containers}")
    print(f"Distribution: {distribution}")
    print(f"Max in any container: {max_count}")
    print(f"Guaranteed minimum max: {min_guaranteed}")
    
    return max_count >= min_guaranteed

pigeonhole_demo(13, 12)  # 13 people, 12 months
pigeonhole_demo(10, 3)   # 10 items, 3 containers

# === Binomial Coefficient Verification ===

def verify_binomial_theorem(n):
    """Verify (x+y)^n = Σ C(n,k) x^(n-k) y^k"""
    
    x, y = 2, 3
    
    # Direct calculation
    direct = (x + y) ** n
    
    # Binomial expansion
    expansion = sum(combination(n, k) * (x ** (n-k)) * (y ** k) 
                    for k in range(n + 1))
    
    print(f"\nBinomial Theorem Verification (n={n}, x={x}, y={y}):")
    print(f"Direct: ({x}+{y})^{n} = {direct}")
    print(f"Expansion: {expansion}")
    print(f"Match: {direct == expansion}")

verify_binomial_theorem(5)

# === Counting Problems ===

def counting_examples():
    """Various counting examples"""
    
    print("\nCounting Examples:")
    print("=" * 40)
    
    # Password counting
    print("\nPasswords (8 chars, alphanumeric):")
    print(f"  With repetition: {62**8:,}")
    print(f"  Without repetition: {permutation(62, 8):,}")
    
    # Committee selection
    print("\nCommittee of 4 from 10 people:")
    print(f"  Any 4: {combination(10, 4)}")
    print(f"  2 men, 2 women (5 each): {combination(5, 2) * combination(5, 2)}")
    
    # Arrangements
    print("\nArrangements of 'BOOK':")
    print(f"  With repeated O: {factorial(4) // factorial(2)}")
    
    # Subsets
    print(f"\nSubsets of {{1,2,3,4}}:")
    print(f"  Total subsets: {2**4}")
    print(f"  Non-empty: {2**4 - 1}")

counting_examples()
```

---

## 📊 Summary Tables

### Counting Formulas

| Scenario | Formula | Example |
|----------|---------|---------|
| Permutation | P(n,r) = n!/(n-r)! | Arrange 3 from 5 |
| Combination | C(n,r) = n!/(r!(n-r)!) | Choose 3 from 5 |
| With repetition | nʳ | 3 digits from 10 |
| Identical items | n!/(n₁!n₂!...nₖ!) | MISSISSIPPI |

### Key Principles

| Principle | When to Use |
|-----------|-------------|
| Sum Rule | Mutually exclusive cases |
| Product Rule | Sequential choices |
| Inclusion-Exclusion | Overlapping sets |
| Pigeonhole | Proving existence |

---

## 🎯 ML Applications

| Application | Combinatorics Concept |
|-------------|----------------------|
| **Feature Selection** | Combinations of features |
| **Hyperparameter Tuning** | Grid search (product rule) |
| **Ensemble Methods** | Combinations of models |
| **Decision Trees** | Feature splits (permutations) |

---

**Status:** ✅ Complete
**Next:** Graph Theory
