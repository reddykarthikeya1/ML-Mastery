# 1.3.6 Joint Distributions

## 🎯 Quick Overview
- **Joint PMF/PDF**: Probability of multiple RVs together
- **Marginal**: Distribution of one RV from joint
- **Conditional**: Distribution given another RV
- **Foundation for**: Correlation, dependence, Bayesian inference

---

## 1. Joint PMF and Joint PDF

### For Discrete RVs

**Joint PMF:**
```
p(x, y) = P(X = x, Y = y)

Properties:
1. p(x, y) ≥ 0
2. Σₓ Σᵧ p(x, y) = 1
3. P((X,Y) ∈ A) = ΣΣ_{(x,y)∈A} p(x, y)
```

### For Continuous RVs

**Joint PDF:**
```
P((X,Y) ∈ R) = ∬_R f(x, y) dx dy

Properties:
1. f(x, y) ≥ 0
2. ∫∫ f(x, y) dx dy = 1
```

---

## 2. Marginal Distributions

### From Joint PMF

```
P(X = x) = Σᵧ P(X = x, Y = y)

P(Y = y) = Σₓ P(X = x, Y = y)
```

### From Joint PDF

```
f_X(x) = ∫ f(x, y) dy

f_Y(y) = ∫ f(x, y) dx
```

---

## 3. Conditional Distributions

### Conditional PMF

```
P(X = x | Y = y) = P(X = x, Y = y) / P(Y = y)

                 = p(x, y) / p_Y(y)
```

### Conditional PDF

```
f_{X|Y}(x|y) = f(x, y) / f_Y(y)
```

### Conditional Expectation

```
E[X | Y = y] = Σₓ x · P(X = x | Y = y)  (discrete)
             = ∫ x · f_{X|Y}(x|y) dx    (continuous)
```

---

## 4. Independent Random Variables

### Definition

```
X and Y are independent if:

P(X = x, Y = y) = P(X = x) · P(Y = y)

For all x, y
```

### Equivalent Conditions

```
F(x, y) = F_X(x) · F_Y(y)

f(x, y) = f_X(x) · f_Y(y)

P(X ∈ A, Y ∈ B) = P(X ∈ A) · P(Y ∈ B)
```

### Consequences

```
If X, Y independent:

E[XY] = E[X] · E[Y]

Cov(X, Y) = 0

Var(X + Y) = Var(X) + Var(Y)
```

---

## 5. Covariance and Correlation

### Covariance

```
Cov(X, Y) = E[(X - μₓ)(Y - μᵧ)]
          = E[XY] - E[X]E[Y]
```

### Correlation

```
ρ(X, Y) = Corr(X, Y) = Cov(X, Y) / (σₓ · σᵧ)

-1 ≤ ρ ≤ 1
```

### Properties

| Property | Covariance | Correlation |
|----------|------------|-------------|
| **Symmetry** | Cov(X,Y) = Cov(Y,X) | ρ(X,Y) = ρ(Y,X) |
| **Scaling** | Cov(aX, bY) = ab·Cov(X,Y) | ρ(aX, bY) = ±ρ(X,Y) |
| **Independence** | Cov = 0 | ρ = 0 |
| **Zero Cov ≠ Independent** | Yes | Yes |

---

## 6. Bivariate Normal Distribution

### Definition

```
(X, Y) ~ Bivariate Normal(μₓ, μᵧ, σₓ², σᵧ², ρ)

Parameters:
- μₓ, μᵧ: means
- σₓ², σᵧ²: variances
- ρ: correlation coefficient
```

### Properties

```
Marginal distributions:
X ~ N(μₓ, σₓ²)
Y ~ N(μᵧ, σᵧ²)

Conditional distribution:
X | Y = y ~ N(μₓ + ρ(σₓ/σᵧ)(y - μᵧ), σₓ²(1 - ρ²))
```

---

## 💻 Python Code Examples

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# === Joint Distribution Example ===

# Discrete joint distribution
# X = number of heads, Y = number of tails in 3 coin flips

print("Joint Distribution: 3 Coin Flips")
print("=" * 50)

# Create joint PMF table
outcomes = ['HHH', 'HHT', 'HTH', 'HTT', 'THH', 'THT', 'TTH', 'TTT']
X_vals = [s.count('H') for s in outcomes]
Y_vals = [s.count('T') for s in outcomes]

# Joint PMF
joint_pmf = {}
for x, y in zip(X_vals, Y_vals):
    joint_pmf[(x, y)] = joint_pmf.get((x, y), 0) + 1/8

print("Joint PMF:")
for (x, y), p in sorted(joint_pmf.items()):
    print(f"P(X={x}, Y={y}) = {p:.3f}")

# Marginal of X
marginal_X = {}
for (x, y), p in joint_pmf.items():
    marginal_X[x] = marginal_X.get(x, 0) + p

print("\nMarginal PMF of X:")
for x, p in sorted(marginal_X.items()):
    print(f"P(X={x}) = {p:.3f}")

# === Continuous Joint Distribution ===

print("\n" + "=" * 50)
print("Continuous Joint Distribution")
print("=" * 50)

# Bivariate normal
mu_x, mu_y = 0, 0
sigma_x, sigma_y = 1, 1
rho = 0.7

mean = [mu_x, mu_y]
cov = [[sigma_x**2, rho*sigma_x*sigma_y], 
       [rho*sigma_x*sigma_y, sigma_y**2]]

bvn = stats.multivariate_normal(mean, cov)

# Generate samples
samples = bvn.rvs(10000)

print(f"Sample correlation: {np.corrcoef(samples[:, 0], samples[:, 1])[0, 1]:.4f}")
print(f"Theoretical correlation: {rho}")

# === Visualization ===

def plot_joint_distribution():
    """Visualize joint distribution"""
    
    fig = plt.figure(figsize=(14, 10))
    
    # 3D surface
    ax1 = fig.add_subplot(221, projection='3d')
    
    x = np.linspace(-3, 3, 100)
    y = np.linspace(-3, 3, 100)
    X, Y = np.meshgrid(x, y)
    pos = np.dstack([X, Y])
    Z = bvn.pdf(pos)
    
    ax1.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
    ax1.set_title('Joint PDF - 3D Surface')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('f(x,y)')
    
    # Contour plot
    ax2 = fig.add_subplot(222)
    contour = ax2.contourf(X, Y, Z, levels=20, cmap='viridis', alpha=0.7)
    ax2.set_title('Joint PDF - Contour Plot')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_aspect('equal')
    plt.colorbar(contour, ax=ax2)
    
    # Scatter of samples
    ax3 = fig.add_subplot(223)
    ax3.scatter(samples[:, 0], samples[:, 1], alpha=0.1, s=10)
    ax3.set_title('Samples from Joint Distribution')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_aspect('equal')
    ax3.grid(True, alpha=0.3)
    
    # Marginal distributions
    ax4 = fig.add_subplot(224)
    ax4.hist(samples[:, 0], bins=50, density=True, alpha=0.5, label='X marginal')
    ax4.hist(samples[:, 1], bins=50, density=True, alpha=0.5, label='Y marginal')
    ax4.plot(x, stats.norm(mu_x, sigma_x).pdf(x), 'b-', linewidth=2)
    ax4.plot(y, stats.norm(mu_y, sigma_y).pdf(y), 'r-', linewidth=2)
    ax4.set_title('Marginal Distributions')
    ax4.set_xlabel('Value')
    ax4.set_ylabel('Density')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

plot_joint_distribution()

# === Covariance and Correlation Demo ===

def covariance_correlation_demo():
    """Demonstrate covariance and correlation"""
    
    np.random.seed(42)
    n = 1000
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    rhos = [0.9, 0.5, 0, -0.7]
    
    for ax, rho in zip(axes.flatten(), rhos):
        # Generate correlated data
        cov_matrix = [[1, rho], [rho, 1]]
        data = np.random.multivariate_normal([0, 0], cov_matrix, n)
        
        X = data[:, 0]
        Y = data[:, 1]
        
        # Calculate sample correlation
        sample_corr = np.corrcoef(X, Y)[0, 1]
        
        ax.scatter(X, Y, alpha=0.3, s=20)
        ax.set_title(f'ρ = {rho}\nSample ρ = {sample_corr:.3f}')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

covariance_correlation_demo()
```

---

**Status:** ✅ Complete
**Next:** Limit Theorems
