# 1.4.3 Functions and Relations

## 🎯 Quick Overview
- **Functions**: Mappings between sets
- **Relations**: Relationships between elements
- **Foundation for**: Algorithms, databases, programming

---

## 1. Functions

### Definition

**Function f: A → B:** Assignment of each element of A to exactly one element of B

```
f(x) = y means f maps x to y

A = Domain (input set)
B = Codomain (output set)
Range = {f(x) : x ∈ A} (actual outputs)
```

### Examples

```
f: ℝ → ℝ, f(x) = x²
- Domain: ℝ
- Codomain: ℝ
- Range: [0, ∞)

g: ℤ → ℤ, g(x) = x + 1
- Domain: ℤ
- Codomain: ℤ
- Range: ℤ
```

---

## 2. Types of Functions

### Injective (One-to-One)

```
f is injective if f(x₁) = f(x₂) → x₁ = x₂

Different inputs → Different outputs

Test: Horizontal line test (passes at most once)
```

### Surjective (Onto)

```
f is surjective if every element in codomain is mapped to

Range = Codomain

Test: Every element in B has a preimage
```

### Bijective

```
f is bijective if both injective and surjective

One-to-one correspondence

Has an inverse function!
```

### Examples

| Function | Injective? | Surjective? | Bijective? |
|----------|------------|-------------|------------|
| f(x) = x² on ℝ | ❌ | ❌ | ❌ |
| f(x) = x² on ℝ⁺ | ✅ | ✅ | ✅ |
| f(x) = x + 1 on ℤ | ✅ | ✅ | ✅ |
| f(x) = eˣ on ℝ | ✅ | ❌ | ❌ |

---

## 3. Inverse Functions

### Definition

```
If f: A → B is bijective, then f⁻¹: B → A exists

f⁻¹(f(x)) = x for all x ∈ A
f(f⁻¹(y)) = y for all y ∈ B
```

### Finding Inverse

```
Steps:
1. Write y = f(x)
2. Swap x and y
3. Solve for y
4. That's f⁻¹(x)

Example: f(x) = 2x + 3
1. y = 2x + 3
2. x = 2y + 3
3. y = (x - 3)/2
4. f⁻¹(x) = (x - 3)/2
```

---

## 4. Composition of Functions

### Definition

```
(f ∘ g)(x) = f(g(x))

Apply g first, then f
```

### Properties

```
- NOT commutative: (f ∘ g) ≠ (g ∘ f) in general
- Associative: (f ∘ g) ∘ h = f ∘ (g ∘ h)
- Identity: f ∘ id = f = id ∘ f
```

### Examples

```
f(x) = x + 1
g(x) = x²

(f ∘ g)(x) = f(g(x)) = f(x²) = x² + 1
(g ∘ f)(x) = g(f(x)) = g(x + 1) = (x + 1)²

Note: (f ∘ g) ≠ (g ∘ f)!
```

---

## 5. Floor and Ceiling Functions

### Definitions

```
⌊x⌋ = floor(x) = greatest integer ≤ x
⌈x⌉ = ceiling(x) = least integer ≥ x
```

### Examples

| x | ⌊x⌋ | ⌈x⌉ |
|---|-----|-----|
| 3.7 | 3 | 4 |
| -2.3 | -3 | -2 |
| 5 | 5 | 5 |
| π | 3 | 4 |

### Properties

```
⌊x⌋ ≤ x < ⌊x⌋ + 1
⌈x⌉ - 1 < x ≤ ⌈x⌉
⌊-x⌋ = -⌈x⌉
```

---

## 6. Relations

### Definition

**Relation R from A to B:** Subset of A × B

```
(a, b) ∈ R means "a is related to b"

Notation: a R b
```

### Examples

```
A = {1, 2, 3}
B = {x, y}

R = {(1, x), (2, y), (3, x)}

1 R x, 2 R y, 3 R x
```

---

## 7. Properties of Relations

### On a Set A

| Property | Definition | Example |
|----------|------------|---------|
| **Reflexive** | ∀a, (a,a) ∈ R | ≤, = |
| **Symmetric** | (a,b) ∈ R → (b,a) ∈ R | =, ≠ |
| **Antisymmetric** | (a,b) ∈ R and (b,a) ∈ R → a = b | ≤, < |
| **Transitive** | (a,b) ∈ R and (b,c) ∈ R → (a,c) ∈ R | ≤, <, = |

### Examples

```
Relation ≤ on ℝ:
- Reflexive: x ≤ x ✓
- Antisymmetric: x ≤ y and y ≤ x → x = y ✓
- Transitive: x ≤ y and y ≤ z → x ≤ z ✓

Relation "divides" on ℤ⁺:
- Reflexive: a | a ✓
- Antisymmetric: a|b and b|a → a = b ✓
- Transitive: a|b and b|c → a|c ✓
```

---

## 8. Equivalence Relations

### Definition

**Equivalence Relation:** Reflexive + Symmetric + Transitive

```
Partitions the set into equivalence classes
```

### Equivalence Classes

```
[a] = {x : x ~ a}

All elements equivalent to a
```

### Examples

```
Congruence modulo n:
a ≡ b (mod n) if n | (a - b)

This is an equivalence relation:
- Reflexive: a ≡ a (mod n)
- Symmetric: a ≡ b → b ≡ a
- Transitive: a ≡ b and b ≡ c → a ≡ c

Equivalence classes: [0], [1], ..., [n-1]
```

---

## 9. Partial Orderings

### Definition

**Partial Order:** Reflexive + Antisymmetric + Transitive

**Poset:** Set with partial order (A, ≤)

### Hasse Diagrams

```
Visual representation of poset

Rules:
- Draw elements as points
- If a < b, draw a below b
- Connect if immediate predecessor
- Remove loops and transitive edges
```

### Example: Divisibility on {1, 2, 3, 4, 6, 12}

```
        12
       /  \
      4    6
      |   / \
      2  3   |
       \ |  /
         1
```

---

## 10. Lattices

### Definition

**Lattice:** Poset where every pair has:
- Least upper bound (join, ∨)
- Greatest lower bound (meet, ∧)

### Examples

```
(Power set, ⊆) is a lattice:
- A ∨ B = A ∪ B (join)
- A ∧ B = A ∩ B (meet)

(ℤ, ≤) is a lattice:
- a ∨ b = max(a, b)
- a ∧ b = min(a, b)
```

---

## 💻 Python Code Examples

```python
# === Function Type Checker ===

def check_function_type(domain, codomain, mapping):
    """Check if function is injective, surjective, bijective"""
    
    # Check injective (one-to-one)
    outputs = list(mapping.values())
    is_injective = len(outputs) == len(set(outputs))
    
    # Check surjective (onto)
    is_surjective = set(outputs) == set(codomain)
    
    # Check bijective
    is_bijective = is_injective and is_surjective
    
    return is_injective, is_surjective, is_bijective

# Example 1: f(x) = x² on {1, 2, 3, 4}
domain = [1, 2, 3, 4]
codomain = [1, 4, 9, 16]
mapping = {x: x**2 for x in domain}

inj, surj, bij = check_function_type(domain, codomain, mapping)
print(f"f(x) = x² on {{1,2,3,4}}:")
print(f"  Injective: {inj}, Surjective: {surj}, Bijective: {bij}")

# Example 2: f(x) = x + 1 on {1, 2, 3}
domain = [1, 2, 3]
codomain = [2, 3, 4]
mapping = {x: x + 1 for x in domain}

inj, surj, bij = check_function_type(domain, codomain, mapping)
print(f"\nf(x) = x + 1 on {{1,2,3}}:")
print(f"  Injective: {inj}, Surjective: {surj}, Bijective: {bij}")

# === Inverse Function ===

def find_inverse(mapping):
    """Find inverse of a bijective function"""
    return {v: k for k, v in mapping.items()}

# Example
f = {1: 'a', 2: 'b', 3: 'c'}
f_inv = find_inverse(f)

print(f"\nFunction: {f}")
print(f"Inverse: {f_inv}")

# Verify
for x in f:
    print(f"f⁻¹(f({x})) = f⁻¹({f[x]}) = {f_inv[f[x]]}")

# === Function Composition ===

def compose(f, g):
    """Compute f ∘ g"""
    return {x: f[g[x]] for x in g if g[x] in f}

# Example
f = {'a': 1, 'b': 2, 'c': 3}
g = {1: 'x', 2: 'y', 3: 'z'}

fog = compose(f, g)
print(f"\nf ∘ g = {fog}")

# === Relation Properties Checker ===

def check_relation_properties(elements, relation):
    """Check properties of a relation"""
    
    # Reflexive
    is_reflexive = all((a, a) in relation for a in elements)
    
    # Symmetric
    is_symmetric = all((b, a) in relation for (a, b) in relation)
    
    # Antisymmetric
    is_antisymmetric = all(
        a == b for (a, b) in relation if (b, a) in relation
    )
    
    # Transitive
    is_transitive = True
    for (a, b) in relation:
        for (c, d) in relation:
            if b == c and (a, d) not in relation:
                is_transitive = False
                break
    
    return is_reflexive, is_symmetric, is_antisymmetric, is_transitive

# Example: ≤ on {1, 2, 3}
elements = [1, 2, 3]
relation = {(1,1), (2,2), (3,3), (1,2), (1,3), (2,3)}

ref, sym, anti, trans = check_relation_properties(elements, relation)
print("\nRelation ≤ on {1, 2, 3}:")
print(f"  Reflexive: {ref}")
print(f"  Symmetric: {sym}")
print(f"  Antisymmetric: {anti}")
print(f"  Transitive: {trans}")
print(f"  Is Partial Order: {ref and anti and trans}")

# === Equivalence Class Generator ===

def equivalence_classes(elements, relation):
    """Generate equivalence classes"""
    
    classes = []
    remaining = set(elements)
    
    while remaining:
        a = remaining.pop()
        eq_class = {a}
        
        for b in list(remaining):
            if (a, b) in relation or (b, a) in relation:
                eq_class.add(b)
                remaining.remove(b)
        
        classes.append(eq_class)
    
    return classes

# Example: Congruence modulo 3 on {0, 1, 2, 3, 4, 5, 6}
elements = list(range(7))
relation = {(a, b) for a in elements for b in elements if a % 3 == b % 3}

classes = equivalence_classes(elements, relation)
print(f"\nEquivalence classes (mod 3):")
for i, eq_class in enumerate(classes):
    print(f"  [{min(eq_class)}] = {eq_class}")

# === Hasse Diagram (Text) ===

def draw_hasse_diagram(elements, relation):
    """Draw simple text-based Hasse diagram"""
    
    # Find levels
    levels = {}
    for elem in elements:
        # Count how many elements are below this one
        below = sum(1 for (a, b) in relation if b == elem and a != b)
        levels[elem] = below
    
    # Sort by level
    sorted_elements = sorted(elements, key=lambda x: levels[x], reverse=True)
    
    print("\nHasse Diagram (levels):")
    current_level = -1
    for elem in sorted_elements:
        if levels[elem] != current_level:
            current_level = levels[elem]
            print(f"\nLevel {current_level}:")
        print(f"  {elem}", end="")
    
    print()

# Example: Divisibility on {1, 2, 3, 4, 6, 12}
elements = [1, 2, 3, 4, 6, 12]
relation = set()
for a in elements:
    for b in elements:
        if b % a == 0:
            relation.add((a, b))

draw_hasse_diagram(elements, relation)

# === Floor and Ceiling ===

def floor(x):
    """Floor function"""
    return int(x) if x >= 0 else int(x) - (1 if x != int(x) else 0)

def ceiling(x):
    """Ceiling function"""
    return int(x) + (1 if x != int(x) else 0)

print("\nFloor and Ceiling:")
for x in [3.7, -2.3, 5, 3.14159]:
    print(f"  ⌊{x}⌋ = {floor(x)}, ⌈{x}⌉ = {ceiling(x)}")
```

---

## 📊 Summary Tables

### Function Types

| Type | Definition | Test |
|------|------------|------|
| **Injective** | f(x₁) = f(x₂) → x₁ = x₂ | Horizontal line ≤ 1 |
| **Surjective** | Range = Codomain | All outputs covered |
| **Bijective** | Both injective and surjective | Has inverse |

### Relation Properties

| Property | Definition | Example |
|----------|------------|---------|
| **Reflexive** | ∀a, (a,a) ∈ R | ≤, = |
| **Symmetric** | (a,b) ∈ R → (b,a) ∈ R | = |
| **Antisymmetric** | (a,b),(b,a) ∈ R → a=b | ≤ |
| **Transitive** | (a,b),(b,c) ∈ R → (a,c) ∈ R | ≤, = |

### Special Relations

| Type | Properties | Example |
|------|------------|---------|
| **Equivalence** | Reflexive, Symmetric, Transitive | ≡ (mod n) |
| **Partial Order** | Reflexive, Antisymmetric, Transitive | ≤, ⊆ |
| **Total Order** | Partial order + comparability | ≤ on ℝ |

---

## 🎯 ML Applications

| Application | Function/Relation Concept |
|-------------|--------------------------|
| **Activation Functions** | Bijective functions (reversible) |
| **Database Joins** | Relations, equivalence |
| **Clustering** | Equivalence classes |
| **Ordering** | Partial orders (preferences) |
| **Normalization** | Floor/ceiling for binning |

---

**Status:** ✅ Complete
**Next:** Number Theory Basics
