# 1.4.1 Logic and Proofs

## 🎯 Quick Overview
- **Propositional logic**: Statements that are true or false
- **Predicates and quantifiers**: Logic with variables
- **Proof methods**: Ways to establish truth
- **Foundation for**: Algorithms, programming, AI reasoning

---

## 1. Propositional Logic

### Propositions

**Proposition:** A statement that is either true or false

**Examples:**
```
✓ "2 + 2 = 4" (True)
✓ "Paris is in Germany" (False)
✗ "What time is it?" (Not a proposition)
✗ "x + 1 = 2" (Depends on x)
```

### Logical Operators

| Operator | Symbol | Meaning | Truth Table |
|----------|--------|---------|-------------|
| **Negation** | ¬p | not p | opposite of p |
| **Conjunction** | p ∧ q | p and q | T only if both T |
| **Disjunction** | p ∨ q | p or q | F only if both F |
| **Implication** | p → q | if p then q | F only when p=T, q=F |
| **Biconditional** | p ↔ q | p iff q | T when same value |

### Truth Table

```
p | q | ¬p | p∧q | p∨q | p→q | p↔q
--+---+----+-----+-----+-----+-----
T | T |  F |  T  |  T  |  T  |  T
T | F |  F |  F  |  T  |  F  |  F
F | T |  T |  F  |  T  |  T  |  F
F | F |  T |  F  |  F  |  T  |  T
```

---

## 2. Logical Equivalences

### Basic Laws

| Law | Formula |
|-----|---------|
| **Identity** | p ∧ T ≡ p, p ∨ F ≡ p |
| **Domination** | p ∨ T ≡ T, p ∧ F ≡ F |
| **Idempotent** | p ∨ p ≡ p, p ∧ p ≡ p |
| **Double Negation** | ¬(¬p) ≡ p |
| **Commutative** | p ∨ q ≡ q ∨ p, p ∧ q ≡ q ∧ p |
| **Associative** | (p ∨ q) ∨ r ≡ p ∨ (q ∨ r) |
| **Distributive** | p ∧ (q ∨ r) ≡ (p ∧ q) ∨ (p ∧ r) |
| **De Morgan's** | ¬(p ∧ q) ≡ ¬p ∨ ¬q, ¬(p ∨ q) ≡ ¬p ∧ ¬q |

### Tautologies and Contradictions

```
Tautology: Always true (e.g., p ∨ ¬p)
Contradiction: Always false (e.g., p ∧ ¬p)
Contingency: Sometimes true, sometimes false
```

---

## 3. Predicates and Quantifiers

### Predicates

**Predicate:** Statement containing variables

```
P(x): "x > 0"
Q(x, y): "x + y = 5"

Becomes proposition when values assigned
```

### Quantifiers

**Universal (∀):**
```
∀x P(x) means "For all x, P(x) is true"

Example: ∀x ∈ ℝ, x² ≥ 0
```

**Existential (∃):**
```
∃x P(x) means "There exists x such that P(x) is true"

Example: ∃x ∈ ℝ, x² = 4
```

### Negation of Quantifiers

```
¬(∀x P(x)) ≡ ∃x ¬P(x)
¬(∃x P(x)) ≡ ∀x ¬P(x)

Examples:
¬("All students passed") ≡ "Some student failed"
¬("Some student passed") ≡ "No students passed"
```

---

## 4. Rules of Inference

### Basic Rules

| Rule | Form | Example |
|------|------|---------|
| **Modus Ponens** | p, p→q ⊢ q | If it rains, ground wet. It rains. ∴ Ground wet. |
| **Modus Tollens** | ¬q, p→q ⊢ ¬p | If it rains, ground wet. Ground not wet. ∴ Didn't rain. |
| **Hypothetical Syllogism** | p→q, q→r ⊢ p→r | If A then B, if B then C ∴ If A then C |
| **Disjunctive Syllogism** | p∨q, ¬p ⊢ q | A or B. Not A. ∴ B |
| **Addition** | p ⊢ p∨q | It rains ∴ It rains or snows |
| **Simplification** | p∧q ⊢ p | It rains and cold ∴ It rains |
| **Conjunction** | p, q ⊢ p∧q | It rains. It cold. ∴ It rains and cold. |

---

## 5. Proof Methods

### Direct Proof

```
To prove p → q:
1. Assume p is true
2. Show q follows

Example: Prove "If n is even, then n² is even"
Proof: Let n be even. Then n = 2k for some integer k.
       n² = (2k)² = 4k² = 2(2k²)
       Since 2k² is an integer, n² is even. □
```

### Proof by Contraposition

```
To prove p → q:
Prove ¬q → ¬p instead (logically equivalent)

Example: Prove "If n² is odd, then n is odd"
Proof: (Contraposition) Assume n is even.
       Then n = 2k, so n² = 4k² = 2(2k²), which is even.
       So if n² is odd, n must be odd. □
```

### Proof by Contradiction

```
To prove p:
1. Assume ¬p
2. Derive a contradiction
3. Therefore p must be true

Example: Prove "√2 is irrational"
Proof: Assume √2 is rational. Then √2 = a/b in lowest terms.
       2 = a²/b², so a² = 2b²
       This means a² is even, so a is even. Let a = 2k.
       Then 4k² = 2b², so b² = 2k², so b is even.
       But this contradicts "lowest terms"! □
```

### Proof by Cases

```
To prove p → q:
Break into cases that cover all possibilities

Example: Prove |x| ≥ 0 for all real x
Proof: Case 1: x ≥ 0. Then |x| = x ≥ 0. ✓
       Case 2: x < 0. Then |x| = -x > 0. ✓
       Both cases give |x| ≥ 0. □
```

### Mathematical Induction

```
To prove P(n) for all n ≥ 1:

1. Base case: Prove P(1)
2. Inductive step: Assume P(k), prove P(k+1)
3. Conclusion: P(n) true for all n ≥ 1

Example: Prove 1 + 2 + ... + n = n(n+1)/2

Base case (n=1): 1 = 1(2)/2 = 1 ✓

Inductive step: Assume 1+2+...+k = k(k+1)/2
Prove for k+1:
1+2+...+k+(k+1) = k(k+1)/2 + (k+1)
                = (k+1)(k/2 + 1)
                = (k+1)(k+2)/2 ✓

Therefore true for all n ≥ 1. □
```

---

## 💻 Python Code Examples

```python
# === Truth Table Generator ===

def truth_table():
    """Generate truth table for logical operations"""
    print("Truth Table for Logical Operators")
    print("=" * 60)
    print(f"{'p':<5} {'q':<5} {'¬p':<5} {'p∧q':<5} {'p∨q':<5} {'p→q':<5} {'p↔q':<5}")
    print("-" * 60)
    
    for p in [True, False]:
        for q in [True, False]:
            not_p = not p
            p_and_q = p and q
            p_or_q = p or q
            p_implies_q = (not p) or q
            p_iff_q = (p == q)
            
            print(f"{str(p):<5} {str(q):<5} {str(not_p):<5} {str(p_and_q):<5} {str(p_or_q):<5} {str(p_implies_q):<5} {str(p_iff_q):<5}")

truth_table()

# === Logical Equivalence Checker ===

def check_de_morgan():
    """Verify De Morgan's Laws"""
    print("\nVerifying De Morgan's Laws")
    print("=" * 60)
    
    print("¬(p ∧ q) ≡ ¬p ∨ ¬q")
    print("¬(p ∨ q) ≡ ¬p ∧ ¬q")
    print()
    
    for p in [True, False]:
        for q in [True, False]:
            left1 = not (p and q)
            right1 = (not p) or (not q)
            
            left2 = not (p or q)
            right2 = (not p) and (not q)
            
            print(f"p={p}, q={q}:")
            print(f"  ¬(p∧q)={left1}, ¬p∨¬q={right1}, Equal: {left1==right2}")
            print(f"  ¬(p∨q)={left2}, ¬p∧¬q={right2}, Equal: {left2==right2}")

check_de_morgan()

# === Quantifier Simulator ===

def simulate_quantifiers():
    """Simulate universal and existential quantifiers"""
    
    # Domain: integers 1 to 10
    domain = list(range(1, 11))
    
    # Predicate: P(x) = "x is even"
    def P(x):
        return x % 2 == 0
    
    # Universal: ∀x P(x)
    universal = all(P(x) for x in domain)
    
    # Existential: ∃x P(x)
    existential = any(P(x) for x in domain)
    
    print("\nQuantifier Simulation")
    print("=" * 60)
    print(f"Domain: {domain}")
    print(f"Predicate P(x): 'x is even'")
    print(f"∀x P(x) (all even): {universal}")
    print(f"∃x P(x) (some even): {existential}")
    
    # Negation
    print(f"\n¬(∀x P(x)) ≡ ∃x ¬P(x): {not universal == any(not P(x) for x in domain)}")
    print(f"¬(∃x P(x)) ≡ ∀x ¬P(x): {not existential == all(not P(x) for x in domain)}")

simulate_quantifiers()

# === Proof by Induction Verifier ===

def verify_sum_formula(n_max=10):
    """Verify sum formula 1+2+...+n = n(n+1)/2"""
    
    print("\nVerifying Sum Formula by Induction")
    print("=" * 60)
    print("Formula: 1 + 2 + ... + n = n(n+1)/2\n")
    
    for n in range(1, n_max + 1):
        left_side = sum(range(1, n + 1))
        right_side = n * (n + 1) // 2
        
        match = "✓" if left_side == right_side else "✗"
        print(f"n={n}: {left_side} = {right_side} {match}")

verify_sum_formula()

# === Valid Argument Checker ===

def check_valid_argument(premises, conclusion, variables):
    """
    Check if an argument is valid using truth table
    
    premises: list of functions that take variable assignments
    conclusion: function that takes variable assignments
    variables: list of variable names
    """
    from itertools import product
    
    print("\nArgument Validity Check")
    print("=" * 60)
    
    n_vars = len(variables)
    is_valid = True
    
    for assignment in product([False, True], repeat=n_vars):
        var_dict = dict(zip(variables, assignment))
        
        # Check if all premises are true
        all_premises_true = all(p(**var_dict) for p in premises)
        
        if all_premises_true:
            conclusion_true = conclusion(**var_dict)
            
            if not conclusion_true:
                is_valid = False
                print(f"Counterexample: {var_dict}")
    
    if is_valid:
        print("Argument is VALID - no counterexamples found")
    else:
        print("Argument is INVALID - counterexamples exist")

# Example: Modus Ponens
# Premises: p, p→q
# Conclusion: q
check_valid_argument(
    premises=[
        lambda p, q: p,
        lambda p, q: (not p) or q
    ],
    conclusion=lambda p, q: q,
    variables=['p', 'q']
)
```

---

## 📊 Summary Tables

### Logical Operators

| p | q | ¬p | p∧q | p∨q | p→q | p↔q |
|---|---|----|----|----|----|----|
| T | T | F | T | T | T | T |
| T | F | F | F | T | F | F |
| F | T | T | F | T | T | F |
| F | F | T | F | F | T | T |

### Proof Methods

| Method | When to Use |
|--------|-------------|
| Direct | Standard implications |
| Contraposition | When ¬q → ¬p is easier |
| Contradiction | Existence proofs, impossibility |
| Cases | Natural case divisions |
| Induction | Statements about natural numbers |

---

## 🎯 ML Applications

| Application | Logic Concept |
|-------------|---------------|
| **Logic Programming** | Propositional logic |
| **Knowledge Representation** | Predicates, quantifiers |
| **Automated Theorem Proving** | Proof methods |
| **Rule-Based Systems** | Logical inference |

---

**Status:** ✅ Complete
**Next:** Set Theory
