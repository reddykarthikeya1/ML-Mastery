# 1.4.2 Set Theory

## 🎯 Quick Overview
- **Sets**: Collections of objects
- **Set operations**: Union, intersection, complement
- **Foundation for**: Data structures, databases, probability

---

## 1. Sets and Set Notation

### Definition

**Set:** Well-defined collection of distinct objects

**Notation:**
```
A = {1, 2, 3}          # Explicit listing
B = {x : x > 0}        # Set builder notation
x ∈ A                  # x is element of A
x ∉ A                  # x is not element of A
```

### Special Sets

| Symbol | Set | Elements |
|--------|-----|----------|
| **∅** or **{}** | Empty set | No elements |
| **ℕ** | Natural numbers | {1, 2, 3, ...} |
| **ℤ** | Integers | {..., -2, -1, 0, 1, 2, ...} |
| **ℚ** | Rational numbers | Fractions |
| **ℝ** | Real numbers | All points on number line |

### Cardinality

```
|A| = number of elements in A

Example: |{1, 2, 3}| = 3
         |∅| = 0
```

---

## 2. Set Operations

### Basic Operations

| Operation | Notation | Definition | Venn Diagram |
|-----------|----------|------------|--------------|
| **Union** | A ∪ B | {x : x ∈ A or x ∈ B} | Combined area |
| **Intersection** | A ∩ B | {x : x ∈ A and x ∈ B} | Overlapping area |
| **Complement** | A' or Aᶜ | {x : x ∉ A} | Outside A |
| **Difference** | A - B | {x : x ∈ A and x ∉ B} | A minus overlap |
| **Subset** | A ⊆ B | All elements of A are in B | A inside B |

### Examples

```
A = {1, 2, 3, 4}
B = {3, 4, 5, 6}

A ∪ B = {1, 2, 3, 4, 5, 6}
A ∩ B = {3, 4}
A - B = {1, 2}
B - A = {5, 6}
```

---

## 3. Venn Diagrams

### Visual Representation

```
        ┌─────────────────┐
        │      U          │
        │   ┌───┐         │
        │ A │   │         │
        │ ┌─┼───┼─┐       │
        │ │ │ ∩ │ │       │  A ∩ B = overlap
        │ └─┼───┼─┘       │
        │   │   │ B       │
        │   └───┘         │
        └─────────────────┘
```

### Two-Set Regions

```
Region I: Only in A
Region II: In both A and B
Region III: Only in B
Region IV: In neither (complement of A ∪ B)
```

---

## 4. Power Sets

### Definition

```
P(A) = set of all subsets of A

Including: ∅ and A itself
```

### Examples

```
A = {1, 2}

P(A) = {∅, {1}, {2}, {1, 2}}

|P(A)| = 2^|A| = 2² = 4
```

### Properties

```
If |A| = n, then |P(A)| = 2ⁿ

For infinite sets: |P(A)| > |A|
```

---

## 5. Cartesian Products

### Definition

```
A × B = {(a, b) : a ∈ A, b ∈ B}

Set of all ordered pairs
```

### Examples

```
A = {1, 2}
B = {x, y}

A × B = {(1,x), (1,y), (2,x), (2,y)}

|A × B| = |A| × |B| = 2 × 2 = 4
```

### Applications

```
- Coordinate planes: ℝ × ℝ = ℝ²
- Database relations
- Function domains
```

---

## 6. Set Identities

### Basic Laws

| Law | Formula |
|-----|---------|
| **Identity** | A ∪ ∅ = A, A ∩ U = A |
| **Domination** | A ∪ U = U, A ∩ ∅ = ∅ |
| **Idempotent** | A ∪ A = A, A ∩ A = A |
| **Complement** | A ∪ A' = U, A ∩ A' = ∅ |
| **Commutative** | A ∪ B = B ∪ A, A ∩ B = B ∩ A |
| **Associative** | (A ∪ B) ∪ C = A ∪ (B ∪ C) |
| **Distributive** | A ∩ (B ∪ C) = (A ∩ B) ∪ (A ∩ C) |
| **De Morgan's** | (A ∪ B)' = A' ∩ B', (A ∩ B)' = A' ∪ B' |

---

## 7. Partitions

### Definition

**Partition of A:** Collection of non-empty, disjoint subsets whose union is A

```
{A₁, A₂, ..., Aₙ} is a partition of A if:
1. Aᵢ ≠ ∅ for all i
2. Aᵢ ∩ Aⱼ = ∅ for i ≠ j
3. A₁ ∪ A₂ ∪ ... ∪ Aₙ = A
```

### Examples

```
A = {1, 2, 3, 4, 5}

Partition 1: {{1, 2}, {3, 4}, {5}}
Partition 2: {{1}, {2, 3, 4, 5}}
Partition 3: {{1, 3, 5}, {2, 4}}  (odd/even)
```

---

## 8. Russell's Paradox

### The Paradox

```
Let S = {x : x ∉ x}

Question: Is S ∈ S?

If S ∈ S, then by definition S ∉ S
If S ∉ S, then by definition S ∈ S

Contradiction!
```

### Resolution

```
Modern set theory uses axioms to prevent such paradoxes:
- Zermelo-Fraenkel set theory (ZF)
- Type theory
- Restricted comprehension
```

---

## 💻 Python Code Examples

```python
# === Set Operations ===

A = {1, 2, 3, 4}
B = {3, 4, 5, 6}

print("Set Operations")
print("=" * 40)
print(f"A = {A}")
print(f"B = {B}")
print(f"A ∪ B = {A | B}")
print(f"A ∩ B = {A & B}")
print(f"A - B = {A - B}")
print(f"B - A = {B - A}")
print(f"A Δ B (symmetric diff) = {A ^ B}")

# === Power Set Generator ===

def power_set(s):
    """Generate all subsets of a set"""
    from itertools import combinations
    
    s = list(s)
    result = []
    
    for i in range(len(s) + 1):
        for subset in combinations(s, i):
            result.append(set(subset))
    
    return result

# Example
A = {1, 2, 3}
print(f"\nPower Set of {A}:")
for subset in power_set(A):
    print(subset)

print(f"\n|P(A)| = {len(power_set(A))} = 2^{len(A)}")

# === Cartesian Product ===

def cartesian_product(A, B):
    """Compute A × B"""
    return {(a, b) for a in A for b in B}

A = {1, 2}
B = {'x', 'y'}

print(f"\nCartesian Product:")
print(f"A × B = {cartesian_product(A, B)}")

# === Set Identities Verification ===

def verify_set_identities():
    """Verify set identities"""
    U = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
    A = {1, 2, 3, 4, 5}
    B = {4, 5, 6, 7, 8}
    C = {7, 8, 9, 10}
    
    print("\nVerifying Set Identities")
    print("=" * 40)
    
    # Commutative
    print(f"A ∪ B = B ∪ A: {A | B == B | A} ✓")
    print(f"A ∩ B = B ∩ A: {A & B == B & A} ✓")
    
    # Associative
    print(f"(A ∪ B) ∪ C = A ∪ (B ∪ C): {(A | B) | C == A | (B | C)} ✓")
    
    # Distributive
    print(f"A ∩ (B ∪ C) = (A ∩ B) ∪ (A ∩ C): {A & (B | C) == (A & B) | (A & C)} ✓")
    
    # De Morgan's
    print(f"(A ∪ B)' = A' ∩ B': {U - (A | B) == (U - A) & (U - B)} ✓")
    print(f"(A ∩ B)' = A' ∪ B': {U - (A & B) == (U - A) | (U - B)} ✓")

verify_set_identities()

# === Partition Checker ===

def is_partition(partition, universal_set):
    """Check if collection is a valid partition"""
    
    # Check non-empty
    for subset in partition:
        if len(subset) == 0:
            return False, "Contains empty set"
    
    # Check disjoint
    for i, s1 in enumerate(partition):
        for j, s2 in enumerate(partition):
            if i != j and len(s1 & s2) > 0:
                return False, f"Sets {s1} and {s2} are not disjoint"
    
    # Check union equals universal set
    union = set()
    for subset in partition:
        union |= subset
    
    if union != universal_set:
        return False, f"Union {union} ≠ {universal_set}"
    
    return True, "Valid partition"

# Test
U = {1, 2, 3, 4, 5}
partition1 = [{1, 2}, {3, 4}, {5}]
partition2 = [{1, 2}, {2, 3}, {4, 5}]  # Not disjoint
partition3 = [{1, 2}, {3, 4}]  # Missing 5

print("\nPartition Checker")
print("=" * 40)
print(f"Partition 1: {is_partition(partition1, U)}")
print(f"Partition 2: {is_partition(partition2, U)}")
print(f"Partition 3: {is_partition(partition3, U)}")

# === Venn Diagram Visualization ===

def visualize_venn():
    """Simple text-based Venn diagram"""
    
    A = {1, 2, 3, 4, 5}
    B = {4, 5, 6, 7, 8}
    
    only_A = A - B
    only_B = B - A
    both = A & B
    neither = "elements outside A and B"
    
    print("\nVenn Diagram (Text Representation)")
    print("=" * 40)
    print(f"Only in A: {only_A}")
    print(f"Only in B: {only_B}")
    print(f"In both (A ∩ B): {both}")
    print(f"In A ∪ B: {A | B}")

visualize_venn()
```

---

## 📊 Summary Tables

### Set Operations

| Operation | Symbol | Python | Example |
|-----------|--------|--------|---------|
| Union | A ∪ B | `A \| B` | {1,2} ∪ {2,3} = {1,2,3} |
| Intersection | A ∩ B | `A & B` | {1,2} ∩ {2,3} = {2} |
| Complement | A' | `U - A` | {1,2,3,4}' = depends on U |
| Difference | A - B | `A - B` | {1,2,3} - {2} = {1,3} |
| Subset | A ⊆ B | `A <= B` | {1,2} ⊆ {1,2,3} |

### Set Identities

| Identity | Formula |
|----------|---------|
| Identity | A ∪ ∅ = A |
| Domination | A ∪ U = U |
| Idempotent | A ∪ A = A |
| Complement | A ∪ A' = U |
| De Morgan's | (A ∪ B)' = A' ∩ B' |

---

## 🎯 ML Applications

| Application | Set Theory Concept |
|-------------|-------------------|
| **Data Preprocessing** | Set operations for filtering |
| **Evaluation Metrics** | Intersection over Union (IoU) |
| **Database Queries** | Set-based operations |
| **Feature Selection** | Set operations on feature sets |
| **Clustering** | Partitions of data points |

---

**Status:** ✅ Complete
**Next:** Functions and Relations
