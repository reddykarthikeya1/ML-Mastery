# 1.3.1 Foundations of Probability

## 🎯 Quick Overview
- **Probability**: Measure of uncertainty
- **Sample space**: All possible outcomes
- **Bayes' theorem**: Update beliefs with evidence
- **Foundation for**: Statistical inference, Bayesian ML, decision making

---

## 1. Random Experiments and Sample Spaces

### Definitions

**Random Experiment:** Process with uncertain outcome that can be repeated

**Sample Space (S or Ω):** Set of all possible outcomes

**Event:** Subset of the sample space

### Examples

| Experiment | Sample Space | Example Event |
|------------|--------------|---------------|
| Coin flip | S = {H, T} | A = {H} (heads) |
| Die roll | S = {1, 2, 3, 4, 5, 6} | B = {2, 4, 6} (even) |
| Two coin flips | S = {HH, HT, TH, TT} | C = {HT, TH} (one each) |
| Continuous measurement | S = [0, ∞) | D = [0, 10] |

### Types of Events

**Simple Event:** Single outcome
```
Rolling a 3: {3}
```

**Compound Event:** Multiple outcomes
```
Rolling even: {2, 4, 6}
```

**Complementary Event (A' or Aᶜ):** All outcomes NOT in A
```
If A = {1, 2, 3}, then A' = {4, 5, 6}
```

**Mutually Exclusive (Disjoint):** A ∩ B = ∅
```
A = {1, 2}, B = {3, 4} → A ∩ B = ∅
```

---

## 2. Event Operations

### Set Operations on Events

| Operation | Notation | Meaning | Venn Diagram |
|-----------|----------|---------|--------------|
| **Union** | A ∪ B | A OR B (or both) | Combined area |
| **Intersection** | A ∩ B | A AND B | Overlapping area |
| **Complement** | A' or Aᶜ | NOT A | Outside A |
| **Difference** | A - B | A but not B | A minus overlap |

### Event Algebra Properties

| Property | Formula |
|----------|---------|
| **Commutative** | A ∪ B = B ∪ A; A ∩ B = B ∩ A |
| **Associative** | (A ∪ B) ∪ C = A ∪ (B ∪ C) |
| **Distributive** | A ∩ (B ∪ C) = (A ∩ B) ∪ (A ∩ C) |
| **De Morgan's Laws** | (A ∪ B)' = A' ∩ B'; (A ∩ B)' = A' ∪ B' |

### Example

```
Roll a die: S = {1, 2, 3, 4, 5, 6}

A = {1, 2, 3} (roll ≤ 3)
B = {3, 4, 5} (3 ≤ roll ≤ 5)

A ∪ B = {1, 2, 3, 4, 5}
A ∩ B = {3}
A' = {4, 5, 6}
A - B = {1, 2}
```

---

## 3. Axioms of Probability (Kolmogorov)

### The Three Axioms

**Axiom 1 (Non-negativity):**
```
P(A) ≥ 0 for any event A
```

**Axiom 2 (Normalization):**
```
P(S) = 1
```

**Axiom 3 (Additivity):**
```
If A₁, A₂, ..., Aₙ are mutually exclusive:
P(A₁ ∪ A₂ ∪ ... ∪ Aₙ) = P(A₁) + P(A₂) + ... + P(Aₙ)
```

### Important Consequences

**C1: Probability of empty set**
```
P(∅) = 0
```

**C2: Complement rule**
```
P(A') = 1 - P(A)
```

**C3: General addition rule**
```
P(A ∪ B) = P(A) + P(B) - P(A ∩ B)
```

**C4: Subset rule**
```
If A ⊆ B, then P(A) ≤ P(B)
```

**C5: Bounds**
```
0 ≤ P(A) ≤ 1 for any event A
```

### Example

```
In a deck of 52 cards:

P(Heart) = 13/52 = 1/4
P(King) = 4/52 = 1/12
P(Heart ∩ King) = P(King of Hearts) = 1/52

P(Heart ∪ King) = P(Heart) + P(King) - P(Heart ∩ King)
                = 13/52 + 4/52 - 1/52
                = 16/52 = 4/13
```

---

## 4. Classical, Empirical, and Subjective Probability

### Classical (Theoretical) Probability

**When:** All outcomes equally likely

**Formula:**
```
P(A) = Number of favorable outcomes / Total outcomes
     = n(A) / n(S)
```

**Example:**
```
Fair die roll:
P(rolling 4) = 1/6

Two dice sum to 7:
Favorable: (1,6), (2,5), (3,4), (4,3), (5,2), (6,1) = 6 outcomes
Total: 36 outcomes
P(sum = 7) = 6/36 = 1/6
```

### Empirical (Experimental) Probability

**Based on observed data:**
```
P(A) ≈ Relative frequency = (times A occurred) / (total trials)
```

**Law of Large Numbers:** As trials → ∞, empirical → theoretical

**Example:**
```
Flip coin 1000 times, get 512 heads:
P(H) ≈ 512/1000 = 0.512

With more flips, approaches 0.5
```

### Subjective Probability

**Based on:** Personal belief, expert opinion, available information

**Example:**
```
"What's the probability it rains tomorrow?"
"Probability team X wins the championship"
```

---

## 5. Counting Principles

### Multiplication Rule (Fundamental Counting Principle)

**If task 1 can be done in n₁ ways and task 2 in n₂ ways:**
```
Total ways to do both = n₁ × n₂
```

**Example:**
```
Outfit combinations:
3 shirts × 4 pants × 2 shoes = 24 outfits
```

### Permutations

**Ordered arrangements:**
```
P(n, r) = n! / (n-r)!

Number of ways to arrange r items from n (order matters)
```

**Example:**
```
Arrange 3 books from 5:
P(5, 3) = 5! / (5-3)! = 5! / 2! = 120 / 2 = 60 ways
```

### Combinations

**Unordered selections:**
```
C(n, r) = n! / (r! × (n-r)!)

Number of ways to choose r items from n (order doesn't matter)
```

**Notation:** C(n,r), ₙCᵣ, (n choose r), binomial coefficient

**Example:**
```
Choose 3 books from 5:
C(5, 3) = 5! / (3! × 2!) = 120 / (6 × 2) = 10 ways
```

### Key Distinction

| Scenario | Order Matters? | Formula |
|----------|---------------|---------|
| **Permutation** | Yes | P(n,r) = n!/(n-r)! |
| **Combination** | No | C(n,r) = n!/(r!(n-r)!) |

**Example:**
```
Lock combination 1-2-3:
- If order matters (permutation): 1-2-3 ≠ 3-2-1
- If order doesn't matter (combination): {1,2,3} = {3,2,1}
```

---

## 6. Conditional Probability

### Definition

**Probability of A given that B has occurred:**
```
P(A|B) = P(A ∩ B) / P(B),  provided P(B) > 0
```

**Read as:** "Probability of A given B" or "P of A given B"

### Interpretation

```
Original sample space: S
Reduced sample space (given B): B
Favorable outcomes: A ∩ B

P(A|B) = (A ∩ B) / B
```

### Multiplication Rule

**From conditional probability definition:**
```
P(A ∩ B) = P(A|B) × P(B)

Also: P(A ∩ B) = P(B|A) × P(A)
```

**Extended to multiple events:**
```
P(A ∩ B ∩ C) = P(A) × P(B|A) × P(C|A ∩ B)
```

### Examples

**Example 1: Cards**
```
Draw 2 cards without replacement:

P(both Aces) = P(Ace on 1st) × P(Ace on 2nd | Ace on 1st)
             = (4/52) × (3/51)
             = 12/2652 = 1/221 ≈ 0.45%
```

**Example 2: Urn Problem**
```
Urn has 3 red and 2 blue balls.

Draw 2 balls without replacement:

P(2nd is red | 1st is red) = 2/4 = 0.5

P(both red) = (3/5) × (2/4) = 6/20 = 0.3
```

---

## 7. Law of Total Probability

### Partition

A set of events {B₁, B₂, ..., Bₙ} forms a **partition** of S if:
1. Mutually exclusive: Bᵢ ∩ Bⱼ = ∅ for i ≠ j
2. Exhaustive: B₁ ∪ B₂ ∪ ... ∪ Bₙ = S

### Theorem

**For any event A and partition {B₁, ..., Bₙ}:**
```
P(A) = Σ P(A|Bᵢ) × P(Bᵢ)
     = P(A|B₁)P(B₁) + P(A|B₂)P(B₂) + ... + P(A|Bₙ)P(Bₙ)
```

### Example

```
Factory has 3 machines:
- Machine A: 50% of production, 2% defective
- Machine B: 30% of production, 3% defective  
- Machine C: 20% of production, 5% defective

P(defective) = P(D|A)P(A) + P(D|B)P(B) + P(D|C)P(C)
             = 0.02×0.50 + 0.03×0.30 + 0.05×0.20
             = 0.01 + 0.009 + 0.01
             = 0.029 = 2.9%
```

---

## 8. Bayes' Theorem ⭐ CRITICAL

### Statement

**For events A and B with P(B) > 0:**
```
P(A|B) = P(B|A) × P(A) / P(B)
```

### Expanded Form (using Law of Total Probability)

```
                    P(B|A) × P(A)
P(A|B) = ──────────────────────────────────
         P(B|A)×P(A) + P(B|A')×P(A')
```

**For multiple hypotheses {H₁, ..., Hₙ}:**
```
                        P(B|Hᵢ) × P(Hᵢ)
P(Hᵢ|B) = ─────────────────────────────────────
          Σⱼ [P(B|Hⱼ) × P(Hⱼ)]
```

### Terminology

| Term | Meaning |
|------|---------|
| **P(Hᵢ)** | Prior probability (before evidence) |
| **P(Hᵢ\|B)** | Posterior probability (after evidence) |
| **P(B\|Hᵢ)** | Likelihood (probability of evidence given hypothesis) |
| **P(B)** | Marginal likelihood (total probability of evidence) |

### Example 1: Medical Testing

```
Disease prevalence: P(D) = 0.01 (1%)
Test sensitivity: P(+|D) = 0.99 (99% true positive)
Test specificity: P(-|no D) = 0.95 (95% true negative)

Question: If test is positive, what's P(D|+)?

P(D|+) = P(+|D)×P(D) / [P(+|D)×P(D) + P(+|no D)×P(no D)]
       = 0.99×0.01 / [0.99×0.01 + 0.05×0.99]
       = 0.0099 / [0.0099 + 0.0495]
       = 0.0099 / 0.0594
       = 0.167 ≈ 16.7%

Despite 99% accurate test, positive result means only 16.7% disease chance!

Why? Base rate is low (1%), so false positives overwhelm true positives.
```

### Example 2: Spam Filter

```
Email classification:
- 30% of emails are spam: P(S) = 0.30
- "FREE" appears in 80% of spam: P(FREE|S) = 0.80
- "FREE" appears in 5% of non-spam: P(FREE|not S) = 0.05

Question: If email contains "FREE", what's P(S|FREE)?

P(S|FREE) = P(FREE|S)×P(S) / [P(FREE|S)×P(S) + P(FREE|not S)×P(not S)]
          = 0.80×0.30 / [0.80×0.30 + 0.05×0.70]
          = 0.24 / [0.24 + 0.035]
          = 0.24 / 0.275
          = 0.873 ≈ 87.3%

Email with "FREE" has 87.3% chance of being spam
```

---

## 9. Independent Events

### Definition

**Events A and B are independent if:**
```
P(A ∩ B) = P(A) × P(B)

Equivalently: P(A|B) = P(A)  (when P(B) > 0)
```

**Meaning:** Occurrence of B doesn't affect probability of A

### Properties

| Property | Formula |
|----------|---------|
| **Symmetry** | If A independent of B, then B independent of A |
| **Complement** | If A, B independent, then A', B also independent |
| **Multiple events** | A₁, ..., Aₙ independent if P(∩Aᵢ) = ΠP(Aᵢ) |

### Examples

**Independent:**
```
Flip coin twice:
A = "1st is Heads"
B = "2nd is Heads"

P(A) = 0.5, P(B) = 0.5
P(A ∩ B) = P(HH) = 0.25 = P(A) × P(B) ✓
```

**Dependent:**
```
Draw 2 cards without replacement:
A = "1st is Ace"
B = "2nd is Ace"

P(A) = 4/52
P(B|A) = 3/51 ≠ P(B) = 4/52

NOT independent!
```

### Independent vs Mutually Exclusive

| Property | Independent | Mutually Exclusive |
|----------|-------------|-------------------|
| **Definition** | P(A ∩ B) = P(A)P(B) | A ∩ B = ∅ |
| **Can both occur?** | Yes | No |
| **P(A ∩ B)** | P(A) × P(B) | 0 |
| **Relationship** | P(A\|B) = P(A) | P(A\|B) = 0 |

**Key insight:** Mutually exclusive events (with P > 0) CANNOT be independent!

---

## 💻 Python Code Examples

```python
import numpy as np
from itertools import combinations, permutations
import matplotlib.pyplot as plt

# === Counting Principles ===

def factorial(n):
    """Calculate factorial"""
    if n <= 1:
        return 1
    return n * factorial(n - 1)

def permutation(n, r):
    """P(n, r) = n! / (n-r)!"""
    return factorial(n) // factorial(n - r)

def combination(n, r):
    """C(n, r) = n! / (r!(n-r)!)"""
    return factorial(n) // (factorial(r) * factorial(n - r))

# Examples
print("Counting Examples:")
print(f"P(5, 3) = {permutation(5, 3)}")  # Arrange 3 from 5
print(f"C(5, 3) = {combination(5, 3)}")  # Choose 3 from 5

# === Probability Simulations ===

def simulate_coin_flips(n_flips=1000):
    """Simulate coin flips and track empirical probability"""
    outcomes = np.random.choice(['H', 'T'], size=n_flips)
    
    heads_count = np.sum(outcomes == 'H')
    empirical_prob = heads_count / n_flips
    
    print(f"\nCoin Flip Simulation ({n_flips} flips):")
    print(f"Heads: {heads_count}")
    print(f"Empirical P(H): {empirical_prob:.4f}")
    print(f"Theoretical P(H): 0.5000")
    
    return empirical_prob

simulate_coin_flips(1000)

def simulate_die_rolls(n_rolls=10000):
    """Simulate die rolls"""
    rolls = np.random.randint(1, 7, size=n_rolls)
    
    # Check various events
    even_count = np.sum(rolls % 2 == 0)
    six_count = np.sum(rolls == 6)
    high_count = np.sum(rolls >= 4)
    
    print(f"\nDie Roll Simulation ({n_rolls} rolls):")
    print(f"P(even) = {even_count/n_rolls:.4f} (theoretical: 0.5000)")
    print(f"P(6) = {six_count/n_rolls:.4f} (theoretical: 0.1667)")
    print(f"P(≥4) = {high_count/n_rolls:.4f} (theoretical: 0.5000)")

simulate_die_rolls(10000)

# === Bayes' Theorem Calculator ===

def bayes_theorem(prior, likelihood, marginal):
    """
    Calculate posterior using Bayes' theorem
    
    P(H|E) = P(E|H) × P(H) / P(E)
    
    Args:
        prior: P(H) - prior probability
        likelihood: P(E|H) - probability of evidence given hypothesis
        marginal: P(E) - total probability of evidence
    
    Returns:
        posterior: P(H|E)
    """
    return likelihood * prior / marginal

# Medical testing example
print("\n" + "="*50)
print("BAYES' THEOREM: Medical Testing")
print("="*50)

p_disease = 0.01  # Prior: disease prevalence
p_positive_given_disease = 0.99  # Sensitivity
p_negative_given_healthy = 0.95  # Specificity

# Calculate marginal P(positive)
p_positive = (p_positive_given_disease * p_disease + 
              (1 - p_negative_given_healthy) * (1 - p_disease))

# Calculate posterior
p_disease_given_positive = bayes_theorem(
    p_disease, 
    p_positive_given_disease, 
    p_positive
)

print(f"Disease prevalence (prior): {p_disease:.2%}")
print(f"Test sensitivity: {p_positive_given_disease:.2%}")
print(f"Test specificity: {p_negative_given_healthy:.2%}")
print(f"\nP(Positive test): {p_positive:.4f}")
print(f"\nP(Disease | Positive): {p_disease_given_positive:.4f}")
print(f"Only {p_disease_given_positive:.1%} chance of disease despite positive test!")

# === Monty Hall Problem Simulation ===

def monty_hall_simulation(n_trials=10000, switch=True):
    """
    Simulate Monty Hall problem
    
    3 doors: 1 car, 2 goats
    Contestant picks door, host opens another door with goat
    Contestant can switch or stay
    
    Args:
        n_trials: Number of simulations
        switch: Whether contestant switches doors
    
    Returns:
        Win rate
    """
    wins = 0
    
    for _ in range(n_trials):
        # Place car randomly
        car_door = np.random.randint(1, 4)
        
        # Contestant's initial choice
        initial_choice = np.random.randint(1, 4)
        
        if switch:
            # If switch, win if initial choice was wrong
            if initial_choice != car_door:
                wins += 1
        else:
            # If stay, win if initial choice was correct
            if initial_choice == car_door:
                wins += 1
    
    return wins / n_trials

print("\n" + "="*50)
print("MONTY HALL PROBLEM")
print("="*50)

win_rate_switch = monty_hall_simulation(10000, switch=True)
win_rate_stay = monty_hall_simulation(10000, switch=False)

print(f"Win rate when SWITCHING: {win_rate_switch:.4f} (theoretical: 2/3)")
print(f"Win rate when STAYING: {win_rate_stay:.4f} (theoretical: 1/3)")
print(f"\nSwitching doubles your chances of winning!")

# === Conditional Probability Visualization ===

def visualize_conditional_probability():
    """Visualize conditional probability with Venn diagram"""
    from matplotlib.patches import Circle
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create circles for events A and B
    circle_A = Circle((3, 4), 3, alpha=0.4, color='blue', label='Event A')
    circle_B = Circle((5, 4), 3, alpha=0.4, color='red', label='Event B')
    
    ax.add_patch(circle_A)
    ax.add_patch(circle_B)
    
    # Label regions
    ax.text(2, 4, 'A only', fontsize=12, ha='center')
    ax.text(6, 4, 'B only', fontsize=12, ha='center')
    ax.text(4, 4, 'A ∩ B', fontsize=12, ha='center', fontweight='bold')
    ax.text(0, 7, 'S (Sample Space)', fontsize=14)
    
    ax.set_xlim(-1, 9)
    ax.set_ylim(0, 8)
    ax.set_aspect('equal')
    ax.legend()
    ax.set_title('Conditional Probability: P(A|B) = P(A ∩ B) / P(B)')
    
    plt.show()

visualize_conditional_probability()

# === Independence Check ===

def check_independence(p_a, p_b, p_a_intersect_b, tolerance=1e-6):
    """
    Check if events A and B are independent
    
    Independent if P(A ∩ B) = P(A) × P(B)
    """
    product = p_a * p_b
    is_independent = abs(p_a_intersect_b - product) < tolerance
    
    print(f"\nP(A) = {p_a}")
    print(f"P(B) = {p_b}")
    print(f"P(A ∩ B) = {p_a_intersect_b}")
    print(f"P(A) × P(B) = {product}")
    print(f"Independent: {is_independent}")
    
    return is_independent

# Example 1: Independent events (coin flips)
print("\n" + "="*50)
print("INDEPENDENCE CHECK: Coin Flips")
print("="*50)
check_independence(0.5, 0.5, 0.25)

# Example 2: Dependent events (cards without replacement)
print("\n" + "="*50)
print("INDEPENDENCE CHECK: Cards Without Replacement")
print("="*50)
p_first_ace = 4/52
p_second_ace = 3/51  # Given first was ace
p_both_aces = p_first_ace * p_second_ace
check_independence(p_first_ace, 3/51, p_both_aces)
```

---

## 📊 Summary Table

| Concept | Formula | Key Point |
|---------|---------|-----------|
| **Classical Probability** | n(A)/n(S) | Equally likely outcomes |
| **Complement** | P(A') = 1 - P(A) | Useful for "at least" problems |
| **Union** | P(A ∪ B) = P(A) + P(B) - P(A ∩ B) | Avoid double counting |
| **Conditional** | P(A\|B) = P(A ∩ B)/P(B) | Reduced sample space |
| **Multiplication** | P(A ∩ B) = P(A\|B)P(B) | Sequential events |
| **Total Probability** | P(A) = Σ P(A\|Bᵢ)P(Bᵢ) | Partition sample space |
| **Bayes' Theorem** | P(A\|B) = P(B\|A)P(A)/P(B) | Update beliefs |
| **Independence** | P(A ∩ B) = P(A)P(B) | No influence |

---

## 🎯 ML Applications

| Application | Probability Concept |
|-------------|-------------------|
| **Naive Bayes Classifier** | Bayes' theorem, independence |
| **Bayesian Networks** | Conditional probability, chain rule |
| **Hidden Markov Models** | Conditional probability, Markov property |
| **A/B Testing** | Hypothesis testing, p-values |
| **Uncertainty Quantification** | Probability distributions |
| **Decision Trees** | Conditional probability, information gain |

---

## ❓ Quick Check Questions

1. What is the difference between mutually exclusive and independent events?
2. State Bayes' theorem and explain each term.
3. Why does the medical testing example give counterintuitive results?
4. How do you calculate P(A ∪ B) when A and B are not disjoint?
5. What is the multiplication rule for independent events?
6. Explain the Monty Hall problem solution.

---

## 📝 Answers to Quick Check

<details>
<summary>Click to reveal answers</summary>

1. **Mutually exclusive vs Independent:**
   - Mutually exclusive: A ∩ B = ∅, cannot both occur
   - Independent: P(A|B) = P(A), occurrence doesn't affect probability
   - Mutually exclusive events (with P > 0) CANNOT be independent

2. **Bayes' theorem:**
   - P(A|B) = P(B|A)P(A)/P(B)
   - P(A|B): Posterior (after evidence)
   - P(B|A): Likelihood
   - P(A): Prior (before evidence)
   - P(B): Marginal likelihood

3. **Medical testing counterintuitive:**
   - Low base rate (1%) means few true cases
   - Even with 99% accuracy, false positives from 99% healthy > true positives from 1% sick
   - Base rate neglect is common cognitive bias

4. **P(A ∪ B) for non-disjoint:**
   - P(A ∪ B) = P(A) + P(B) - P(A ∩ B)
   - Subtract intersection to avoid double counting

5. **Multiplication for independent:**
   - P(A ∩ B) = P(A) × P(B)
   - Extends to n events: P(∩Aᵢ) = ΠP(Aᵢ)

6. **Monty Hall:**
   - Switching wins 2/3, staying wins 1/3
   - Initial choice is wrong 2/3 of time
   - Host's action provides information
   - Switching capitalizes on initial wrong choice

</details>
---

**Status:** ✅ Complete
**Next:** Random Variables
