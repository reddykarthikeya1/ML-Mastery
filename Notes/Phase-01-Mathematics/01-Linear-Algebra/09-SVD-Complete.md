# 1.1.7 Singular Value Decomposition (SVD)

## 🎯 Quick Overview
- **SVD**: Factorizes ANY matrix (not just square!)
- **Form**: A = UΣVᵀ
- **Critical for**: PCA, recommendation systems, image compression, NLP

---

## 1. Singular Values and Singular Vectors

### Definition
For matrix A (m×n), SVD is:
```
A = UΣVᵀ

where:
  U (m×m) = orthogonal matrix (left singular vectors)
  Σ (m×n) = diagonal matrix (singular values)
  V (n×n) = orthogonal matrix (right singular vectors)
```

### Relationship to Eigenvalues
```
Left singular vectors (U): Eigenvectors of AAᵀ
Right singular vectors (V): Eigenvectors of AᵀA
Singular values (σ): √(eigenvalues of AᵀA)
```

### Key Insight
```
Every matrix has an SVD!
(Unlike eigendecomposition which requires square matrices)
```

---

## 2. Full SVD vs Reduced SVD

### Full SVD
```
A = UΣVᵀ

U: m×m (full orthogonal)
Σ: m×n (full diagonal with zeros)
V: n×n (full orthogonal)
```

**Example (3×2 matrix):**
```
        [ σ₁  0  ]
        [        ]
A = U   [ 0   σ₂ ]  Vᵀ
        [        ]
        [ 0   0  ]

U: 3×3, Σ: 3×2, V: 2×2
```

### Reduced (Economy) SVD
```
A = UᵣΣᵣVᵣᵀ

Uᵣ: m×r (only r columns)
Σᵣ: r×r (square diagonal)
Vᵣ: n×r (only r columns)

where r = rank(A)
```

**Same example:**
```
        [ σ₁  0  ]
A = Uᵣ  [        ]  Vᵣᵀ
        [ 0   σ₂ ]

Uᵣ: 3×2, Σᵣ: 2×2, Vᵣ: 2×2
```

### When to Use
| Type | Use Case |
|------|----------|
| **Full SVD** | Theoretical analysis |
| **Reduced SVD** | Computation, storage efficiency |

---

## 3. Computing SVD

### Algorithm Overview
```
Step 1: Compute AᵀA (n×n)
Step 2: Find eigenvalues λᵢ and eigenvectors vᵢ of AᵀA
Step 3: Singular values: σᵢ = √λᵢ
Step 4: V = [v₁ v₂ ... vₙ]
Step 5: Compute uᵢ = (1/σᵢ)Avᵢ for each i
Step 6: U = [u₁ u₂ ... uₘ]
Step 7: Form Σ with σᵢ on diagonal
```

### Example (2×2)
```
    [ 3  0 ]
A = [     ]
    [ 4  5 ]

Step 1: AᵀA = [[25, 20], [20, 25]]

Step 2: Eigenvalues of AᵀA:
λ₁ = 45, λ₂ = 5

Step 3: Singular values:
σ₁ = √45 ≈ 6.71
σ₂ = √5 ≈ 2.24

Step 4-7: Compute U, Σ, V

    [ 0.447  -0.894 ]      [ 6.71  0    ]      [ 0.707  -0.707 ]ᵀ
A = [                 ]      [            ]      [               ]
    [ 0.894   0.447 ]      [ 0     2.24 ]      [ 0.707   0.707 ]
```

---

## 4. Relationship to Eigenvalues

### Comparison
| | Eigendecomposition | SVD |
|-|-------------------|-----|
| **Matrix** | Square only | Any (m×n) |
| **Form** | A = PDP⁻¹ | A = UΣVᵀ |
| **Vectors** | Eigenvectors | Singular vectors |
| **Values** | Eigenvalues (can be complex) | Singular values (always real, ≥0) |

### Connection
```
For symmetric positive definite A:
  SVD = Eigendecomposition
  U = V = eigenvectors
  Σ = |eigenvalues|

For general A:
  Singular values = √(eigenvalues of AᵀA)
  Left singular vectors = eigenvectors of AAᵀ
  Right singular vectors = eigenvectors of AᵀA
```

---

## 5. Geometric Interpretation

### SVD as Transformation
```
A = UΣVᵀ

Interpretation:
1. Vᵀ: Rotate/reflect (in ℝⁿ)
2. Σ: Scale along axes (by σᵢ)
3. U: Rotate/reflect (in ℝᵐ)
```

### Visual (2D → 2D)
```
Unit circle --Vᵀ--> Rotated circle --Σ--> Ellipse --U--> Rotated ellipse

The singular values σ₁, σ₂ are the semi-axes of the ellipse!
```

### Key Insight
```
Any linear transformation = Rotation + Scaling + Rotation
```

---

## 6. Low-Rank Approximations

### Best Rank-k Approximation
```
A ≈ Aₖ = UₖΣₖVₖᵀ

where:
  Uₖ: first k columns of U
  Σₖ: top-left k×k of Σ
  Vₖ: first k columns of V
```

### Eckart-Young Theorem
```
Aₖ is the BEST rank-k approximation to A

Error: ‖A - Aₖ‖₂ = σₖ₊₁

(in spectral norm)
```

### Truncated SVD
```
Keep only largest singular values:

A ≈ σ₁u₁v₁ᵀ + σ₂u₂v₂ᵀ + ... + σₖuₖvₖᵀ
```

### Example (Image Compression)
```
Original image: 1000×1000 pixels = 1M values

Rank-50 approximation:
- Store: 50×(1000 + 1000 + 1) = 100,050 values
- Compression: 10x reduction!
```

---

## 7. Applications

### Image Compression
```
Grayscale image = matrix A

SVD: A = UΣVᵀ

Keep top k singular values:
Aₖ = UₖΣₖVₖᵀ

Result: Compressed image with minimal quality loss
```

### Noise Reduction
```
Noisy data = Signal + Noise

SVD separates:
- Large σᵢ: Signal
- Small σᵢ: Noise

Truncate small singular values → Denoised data
```

### Principal Component Analysis (PCA)
```
Data matrix X (centered)

SVD: X = UΣVᵀ

Principal components = columns of V
Variance explained = σᵢ²
```

### Recommendation Systems
```
User-Item matrix R

SVD: R ≈ UΣVᵀ

Latent factors = columns of U, V
Predict ratings: Rᵢⱼ ≈ uᵢᵀΣvⱼ
```

---

## 💻 Python Code Examples

```python
import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt

# === Matrix ===
A = np.array([[3, 0],
              [4, 5]])

# === Full SVD ===
U, S, Vt = np.linalg.svd(A, full_matrices=True)
Σ = np.zeros_like(A, dtype=float)
np.fill_diagonal(Σ, S)

print("Full SVD:")
print(f"U:\n{U}")
print(f"Σ:\n{Σ}")
print(f"Vᵀ:\n{Vt}")

# Verify
A_reconstructed = U @ Σ @ Vt
print(f"A = UΣVᵀ? {np.allclose(A, A_reconstructed)}")

# === Reduced SVD ===
U_r, S_r, Vt_r = np.linalg.svd(A, full_matrices=False)
Σ_r = np.diag(S_r)

print(f"\nReduced SVD:")
print(f"U shape: {U_r.shape}")
print(f"Σ shape: {Σ_r.shape}")
print(f"Vᵀ shape: {Vt_r.shape}")

# === Singular Values ===
print(f"\nSingular values: {S}")
print(f"σ₁ = {S[0]:.4f}, σ₂ = {S[1]:.4f}")

# === Relationship to Eigenvalues ===
AtA = A.T @ A
eigenvalues_AtA = np.linalg.eigvalsh(AtA)
singular_values_from_eig = np.sqrt(np.sort(eigenvalues_AtA)[::-1])

print(f"\n√(eigenvalues of AᵀA): {singular_values_from_eig}")
print(f"Match SVD? {np.allclose(S, singular_values_from_eig)}")

# === Low-Rank Approximation ===
# Create a larger matrix for demonstration
np.random.seed(42)
A_large = np.random.randn(10, 8)

U, S, Vt = np.linalg.svd(A_large, full_matrices=False)

# Rank-k approximations
for k in [1, 3, 5, 8]:
    U_k = U[:, :k]
    Σ_k = np.diag(S[:k])
    Vt_k = Vt[:k, :]
    
    A_k = U_k @ Σ_k @ Vt_k
    error = np.linalg.norm(A_large - A_k, 'fro')
    
    print(f"\nRank-{k} approximation:")
    print(f"Error (Frobenius): {error:.4f}")
    print(f"Next singular value: σ_{k+1} = {S[k] if k < len(S) else 0:.4f}")

# === Image Compression Example ===
def compress_image_svd(image, k):
    """Compress image using SVD"""
    U, S, Vt = np.linalg.svd(image, full_matrices=False)
    
    # Keep top k components
    U_k = U[:, :k]
    Σ_k = np.diag(S[:k])
    Vt_k = Vt[:k, :]
    
    # Reconstruct
    compressed = U_k @ Σ_k @ Vt_k
    return compressed, U_k, S[:k], Vt_k

# Create a sample "image" (gradient pattern)
x = np.linspace(0, 1, 100)
y = np.linspace(0, 1, 100)
X, Y = np.meshgrid(x, y)
image = np.sin(2*np.pi*X) * np.cos(2*np.pi*Y)

# Compress with different ranks
for k in [5, 10, 20, 50]:
    compressed, _, S_k, _ = compress_image_svd(image, k)
    
    original_size = image.size
    compressed_size = k * (image.shape[0] + image.shape[1] + 1)
    compression_ratio = original_size / compressed_size
    
    mse = np.mean((image - compressed) ** 2)
    
    print(f"\nk={k}:")
    print(f"  Compression ratio: {compression_ratio:.1f}x")
    print(f"  MSE: {mse:.6f}")

# === PCA via SVD ===
# Generate sample data
np.random.seed(0)
n_samples = 100
X = np.random.randn(n_samples, 5)
X[:, 0] = X[:, 1] * 2 + np.random.randn(n_samples) * 0.1  # Correlated
X[:, 2] = X[:, 1] * 0.5 + np.random.randn(n_samples) * 0.1

# Center the data
X_centered = X - np.mean(X, axis=0)

# SVD
U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)

# Principal components
PCs = Vt
variance_explained = (S ** 2) / (n_samples - 1)
variance_ratio = variance_explained / np.sum(variance_explained)

print(f"\nPCA via SVD:")
print(f"Principal components shape: {PCs.shape}")
print(f"Variance explained: {variance_explained}")
print(f"Variance ratio: {variance_ratio}")
print(f"Cumulative ratio: {np.cumsum(variance_ratio)}")

# === Pseudoinverse via SVD ===
A = np.array([[1, 2],
              [3, 4],
              [5, 6]])

U, S, Vt = np.linalg.svd(A, full_matrices=False)

# A⁺ = VΣ⁺Uᵀ
Σ_pinv = np.diag(1/S)
A_pinv = Vt.T @ Σ_pinv @ U.T

print(f"\nPseudoinverse:")
print(f"A⁺:\n{A_pinv}")

# Verify: AA⁺A = A
print(f"AA⁺A = A? {np.allclose(A @ A_pinv @ A, A)}")
```

---

## 📊 Summary Table

| Concept | Formula | Key Point |
|---------|---------|-----------|
| **SVD** | A = UΣVᵀ | Works for ANY matrix |
| **Full SVD** | U: m×m, Σ: m×n, V: n×n | Complete decomposition |
| **Reduced SVD** | U: m×r, Σ: r×r, V: n×r | Efficient storage |
| **Singular Values** | σᵢ = √λᵢ(AᵀA) | Always real, ≥ 0 |
| **Best Rank-k** | Aₖ = UₖΣₖVₖᵀ | Eckart-Young theorem |
| **PCA** | PCs = V from SVD(X) | Variance = σᵢ² |
| **Pseudoinverse** | A⁺ = VΣ⁺Uᵀ | Generalized inverse |

---

## 🎯 ML Applications

| Application | How It's Used |
|-------------|---------------|
| **PCA** | SVD of centered data matrix |
| **Recommendation** | Matrix factorization (Netflix prize) |
| **NLP (LSA)** | SVD of term-document matrix |
| **Image Compression** | Low-rank approximation |
| **Denoising** | Truncate small singular values |
| **Pseudoinverse** | Solve overdetermined systems |

---

## ❓ Quick Check

1. Why does SVD work for non-square matrices?
2. What's the difference between full and reduced SVD?
3. How are singular values related to eigenvalues?
4. What does the Eckart-Young theorem guarantee?
5. How is PCA related to SVD?
6. Why are singular values always non-negative?

---

## 📝 Answers to Quick Check

<details>
<summary>Click to reveal answers</summary>

1. **SVD for non-square matrices:**
   - SVD uses AᵀA (n×n) and AAᵀ (m×m) which are always square
   - Eigenvectors of these give V and U
   - Works for ANY matrix, unlike eigendecomposition

2. **Full vs Reduced SVD:**
   - **Full SVD**: U is m×m, Σ is m×n, V is n×n (complete orthogonal bases)
   - **Reduced SVD**: U is m×r, Σ is r×r, V is n×r (r = rank)
   - Reduced is more efficient for computation and storage

3. **Singular values vs eigenvalues:**
   - σᵢ = √(λᵢ) where λᵢ are eigenvalues of AᵀA
   - Singular values are always real and ≥ 0
   - For symmetric positive definite A: singular values = eigenvalues

4. **Eckart-Young theorem:**
   - Truncated SVD gives the **best rank-k approximation**
   - Minimum error in spectral norm: ‖A - Aₖ‖₂ = σₖ₊₁
   - Optimal for compression, denoising, low-rank approximation

5. **PCA and SVD:**
   - PCA on data matrix X = SVD of centered X
   - Principal components = right singular vectors (columns of V)
   - Variance explained = σᵢ²/(n-1)
   - SVD is the computational method for PCA

6. **Singular values non-negative:**
   - σᵢ = √(eigenvalue of AᵀA)
   - AᵀA is positive semidefinite → eigenvalues ≥ 0
   - Square root of non-negative is non-negative
   - Represents "stretching factors" which can't be negative

</details>
---

**Status:** ✅ Complete  
**Next:** Advanced Topics
