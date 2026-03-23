# Discrete Mathematics - Practice Problems

## Topic 1: Logic and Proofs

### Level 1: Basic

**1.1** Construct truth tables for:
- a) p → (q → p)
- b) (p → q) ↔ (¬q → ¬p)
- c) (p ∧ q) → (p ∨ q)

**1.2** Determine if the following are tautologies:
- a) p ∨ ¬p
- b) p → (q → p)
- c) (p → q) ∧ p ∧ ¬q

**1.3** Negate the following statements:
- a) All students passed the exam
- b) Some students passed the exam
- c) No students passed the exam

---

### Level 2: Intermediate

**1.4** Prove by mathematical induction:
```
1 + 3 + 5 + ... + (2n-1) = n²
```

**1.5** Prove by contradiction:
```
If n² is even, then n is even
```

**1.6** Python Practice - Truth Table Generator:
```python
def generate_truth_table(expression):
    """
    Generate truth table for a logical expression.
    Support variables p, q, r and operators &, |, ~, ->
    """
    # Your code here
    pass

# Test with: p & q, p | q, p -> q, ~(p & q)
```

---

## Topic 2: Set Theory

### Level 1: Basic

**2.1** Given A = {1, 2, 3, 4}, B = {3, 4, 5, 6}, find:
- a) A ∪ B
- b) A ∩ B
- c) A - B
- d) B - A
- e) A × B (first 5 elements)

**2.2** List all subsets of {a, b, c}

**2.3** Verify De Morgan's Law for:
```
A = {1, 2, 3}, B = {3, 4, 5}, U = {1, 2, 3, 4, 5, 6}
```

---

### Level 2: Intermediate

**2.4** Prove that for any sets A, B, C:
```
A ∩ (B ∪ C) = (A ∩ B) ∪ (A ∩ C)
```

**2.5** How many elements are in P(A) if |A| = 5?

**2.6** Python Practice - Set Operations:
```python
def power_set(s):
    """Generate all subsets of a set"""
    # Your code here
    pass

def is_partition(partition, universal_set):
    """Check if collection is a valid partition"""
    # Your code here
    pass

# Test with various sets
```

---

## Topic 3: Functions and Relations

### Level 1: Basic

**3.1** Determine if each function is injective, surjective, bijective:
- a) f(x) = 2x + 1, f: ℝ → ℝ
- b) f(x) = x², f: ℝ → ℝ
- c) f(x) = x², f: ℝ⁺ → ℝ⁺

**3.2** For the relation R = {(1,1), (2,2), (3,3), (1,2), (2,1)} on {1,2,3}:
- a) Is R reflexive?
- b) Is R symmetric?
- c) Is R transitive?
- d) Is R an equivalence relation?

**3.3** Find f⁻¹(x) for f(x) = 3x - 5

---

### Level 2: Intermediate

**3.4** Prove that the composition of two injective functions is injective.

**3.5** Find all equivalence classes for congruence modulo 4 on {0, 1, 2, ..., 11}

**3.6** Python Practice - Function Types:
```python
def analyze_function(domain, codomain, mapping):
    """
    Determine if function is injective, surjective, bijective.
    Return (is_injective, is_surjective, is_bijective)
    """
    # Your code here
    pass

# Test with various functions
```

---

## Topic 4: Number Theory

### Level 1: Basic

**4.1** Find gcd and lcm:
- a) gcd(48, 18)
- b) gcd(100, 35)
- c) lcm(12, 18)
- d) lcm(15, 25)

**4.2** Find the prime factorization:
- a) 84
- b) 100
- c) 97

**4.3** Compute:
- a) 17 mod 5
- b) -7 mod 4
- c) 3¹⁰ mod 7

---

### Level 2: Intermediate

**4.4** Use Extended Euclidean Algorithm to find x, y such that:
```
48x + 18y = gcd(48, 18)
```

**4.5** Find the modular inverse:
- a) 3⁻¹ mod 11
- b) 5⁻¹ mod 7
- c) 7⁻¹ mod 26

**4.6** Python Practice - GCD and Primes:
```python
def gcd(a, b):
    """Euclidean algorithm"""
    # Your code here
    pass

def sieve_of_eratosthenes(n):
    """Generate all primes up to n"""
    # Your code here
    pass

def mod_inverse(a, m):
    """Find modular inverse using extended GCD"""
    # Your code here
    pass

# Test with various inputs
```

---

## Topic 5: Combinatorics

### Level 1: Basic

**5.1** Calculate:
- a) P(8, 3)
- b) C(10, 4)
- c) C(15, 11)
- d) 7!

**5.2** How many ways to:
- a) Arrange 5 books on a shelf?
- b) Choose 3 students from 10?
- c) Form a committee of 4 from 6 men and 4 women (2 of each)?

**5.3** Expand using binomial theorem:
- a) (x + y)⁴
- b) (x + 2)⁵

---

### Level 2: Intermediate

**5.4** How many 4-digit numbers (no leading zero):
- a) Total?
- b) With no repeated digits?
- c) That are even?

**5.5** Prove using combinatorial argument:
```
C(n, k) = C(n, n-k)
```

**5.6** Python Practice - Counting:
```python
def count_passwords(length, charset_size, with_repetition=True):
    """Count possible passwords"""
    # Your code here
    pass

def count_committees(n_men, n_women, committee_size, min_men=0, min_women=0):
    """Count ways to form committee with constraints"""
    # Your code here
    pass

# Test with various parameters
```

---

## Topic 6: Graph Theory

### Level 1: Basic

**6.1** For the graph with V = {A, B, C, D} and E = {(A,B), (A,C), (B,C), (C,D)}:
- a) Draw the graph
- b) Find degree of each vertex
- c) Verify handshaking lemma

**6.2** How many edges in:
- a) K₅ (complete graph on 5 vertices)?
- b) A tree with 10 vertices?
- c) A cycle C₆?

**6.3** Determine if graphs are isomorphic (same structure)

---

### Level 2: Intermediate

**6.4** Find MST using Kruskal's algorithm:
```
Vertices: {0, 1, 2, 3, 4}
Edges: (0,1,4), (0,2,3), (1,2,1), (1,3,2), (2,3,4), (3,4,2)
```

**6.5** Color the graph with minimum colors:
```
Vertices: {0, 1, 2, 3, 4}
Edges: (0,1), (0,2), (1,2), (1,3), (2,3), (3,4)
```

**6.6** Python Practice - Graph Algorithms:
```python
def bfs(graph, start):
    """Breadth-first search"""
    # Your code here
    pass

def kruskal_mst(n, edges):
    """Kruskal's MST algorithm"""
    # Your code here
    pass

def greedy_coloring(n, edges):
    """Greedy graph coloring"""
    # Your code here
    pass

# Test with various graphs
```

---

## Topic 7: Boolean Algebra

### Level 1: Basic

**7.1** Construct truth tables:
- a) xy + x'y'
- b) (x + y)'
- c) x ⊕ y

**7.2** Simplify:
- a) x + xy
- b) x(x' + y)
- c) xy + xy'

**7.3** Apply De Morgan's Law:
- a) (x + yz)'
- b) (xy + z)'

---

### Level 2: Intermediate

**7.4** Simplify to minimum SOP form:
```
F = xy + x'y + xy'
```

**7.5** Draw logic circuit for:
```
F = (x + y)z + x'y
```

**7.6** Python Practice - Boolean Simplifier:
```python
def simplify_boolean(expression):
    """
    Simplify Boolean expression using identities.
    Support +, *, ~ for OR, AND, NOT
    """
    # Your code here
    pass

# Test with: x + xy, xy + xy', (x + y)'
```

---

## Solutions (Selected Problems)

<details>
<summary>Click to reveal solutions</summary>

### 1.1
```
a) p → (q → p) is a tautology
b) (p → q) ↔ (¬q → ¬p) is a tautology (contrapositive)
c) (p ∧ q) → (p ∨ q) is a tautology
```

### 1.4
```
Proof by induction:
Base case (n=1): 1 = 1² ✓
Inductive step: Assume 1+3+...+(2k-1) = k²
Add (2k+1): k² + (2k+1) = (k+1)² ✓
```

### 2.1
```
a) A ∪ B = {1, 2, 3, 4, 5, 6}
b) A ∩ B = {3, 4}
c) A - B = {1, 2}
d) B - A = {5, 6}
```

### 3.1
```
a) Injective ✓, Surjective ✓, Bijective ✓
b) Injective ✗, Surjective ✗, Bijective ✗
c) Injective ✓, Surjective ✓, Bijective ✓
```

### 4.1
```
a) gcd(48, 18) = 6
b) gcd(100, 35) = 5
c) lcm(12, 18) = 36
d) lcm(15, 25) = 75
```

### 5.1
```
a) P(8, 3) = 8×7×6 = 336
b) C(10, 4) = 210
c) C(15, 11) = C(15, 4) = 1365
d) 7! = 5040
```

### 6.2
```
a) K₅: 5(4)/2 = 10 edges
b) Tree with 10 vertices: 9 edges
c) Cycle C₆: 6 edges
```

### 7.2
```
a) x + xy = x(1 + y) = x
b) x(x' + y) = xx' + xy = 0 + xy = xy
c) xy + xy' = x(y + y') = x(1) = x
```

</details>

---

## 📝 Notes Section

### My Practice Problems:


### Mistakes to Review:


### Key Insights:


---
**Last Updated:** 2026-03-23
**Status:** ✅ Discrete Mathematics Complete!
