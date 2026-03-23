# 1.3.11 Information Theory

## 🎯 Quick Overview
- **Entropy**: Measure of uncertainty
- **Cross-entropy**: Average bits needed with wrong distribution
- **KL Divergence**: Distance between distributions
- **Foundation for**: Decision trees, neural network loss functions, variational inference

---

## 1. Entropy

### Definition (Discrete)

**Shannon Entropy:**
```
H(X) = -Σ p(x) log₂ p(x)

Units: bits (log base 2) or nats (natural log)
```

### Interpretation

| Perspective | Meaning |
|-------------|---------|
| **Information** | Expected information content |
| **Uncertainty** | Measure of randomness |
| **Coding** | Minimum bits to encode outcome |

### Examples

**Fair Coin:**
```
H = -[0.5·log₂(0.5) + 0.5·log₂(0.5)] = 1 bit
```

**Biased Coin (p = 0.9):**
```
H = -[0.9·log₂(0.9) + 0.1·log₂(0.1)] = 0.469 bits
```

---

## 2. Cross-Entropy

### Definition

```
H(p, q) = -Σ p(x) log q(x)

Average bits needed to encode X ~ p using code optimized for q
```

### ML Application: Classification Loss

```
Binary cross-entropy:
L = -[y·log(ŷ) + (1-y)·log(1-ŷ)]

Minimizing CE = Maximizing likelihood!
```

---

## 3. KL Divergence

### Definition

```
D_KL(p || q) = Σ p(x) log(p(x)/q(x))
             = H(p, q) - H(p)
```

### Properties

| Property | Value |
|----------|-------|
| **Non-negative** | D_KL(p \|\| q) ≥ 0 |
| **Zero iff** | p = q |
| **Not symmetric** | D_KL(p \|\| q) ≠ D_KL(q \|\| p) |

---

## 4. Mutual Information

### Definition

```
I(X; Y) = Σ Σ p(x, y) log(p(x, y) / (p(x)·p(y)))
```

### Properties

- I(X; Y) ≥ 0
- I(X; Y) = I(Y; X) (symmetric)
- I(X; Y) = 0 iff X and Y independent

### ML Applications

- Feature selection
- Decision trees (information gain)
- Variational inference

---

## 💻 Python Code Examples

```python
import numpy as np

def entropy(p):
    """Calculate entropy"""
    p = np.array(p)
    p = p[p > 0]
    return -np.sum(p * np.log2(p))

def kl_divergence(p, q):
    """Calculate KL divergence"""
    p = np.array(p)
    q = np.array(q)
    return np.sum(p * np.log2(p / q))

# Examples
print(f"Fair coin entropy: {entropy([0.5, 0.5]):.4f} bits")
print(f"Biased coin entropy: {entropy([0.9, 0.1]):.4f} bits")
```

---

## 🎯 ML Applications

| Application | Information Theory Concept |
|-------------|---------------------------|
| **Decision Trees** | Information gain |
| **Neural Networks** | Cross-entropy loss |
| **VAEs** | KL divergence |
| **Feature Selection** | Mutual information |

---

**Status:** ✅ Complete
**Next:** Practice Problems
