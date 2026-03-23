# 11.4 Advanced Image Generation: VAEs, GANs, and Diffusion

## 🎯 Quick Overview
- **VAE Math**: Deriving the Evidence Lower Bound (ELBO) and KL-Divergence
- **GAN Stability**: Wasserstein GAN (WGAN) and Gradient Penalty (GP)
- **Diffusion Theory**: Forward/Reverse SDEs and the Denoising objective
- **Latent Space Manipulation**: StyleGAN and ControlNet logic
- **Foundation for**: Modern creative AI, Synthetic data generation, and Medical imaging

---

## 1. Variational Autoencoders (VAEs)

VAEs learn a continuous latent space by mapping inputs to a distribution.

### 1.1 The ELBO Derivation
The goal is to maximize the log-likelihood of the data $p(x)$. Since this is intractable, we maximize the **Evidence Lower Bound (ELBO)**:
$$ \mathcal{L}(\theta, \phi; x) = \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] - D_{KL}(q_\phi(z|x) \| p(z)) $$
1.  **Reconstruction Term**: How well the decoder reconstructs $x$ from $z$.
2.  **Regularization Term**: Forces the encoded distribution $q(z|x)$ to be close to a standard normal prior $p(z) = N(0, I)$.

---

## 2. Generative Adversarial Networks (GANs)

Standard GANs use Jensen-Shannon divergence, which leads to vanishing gradients.

### 2.1 Wasserstein GAN (WGAN)
Uses the **Earth Mover's Distance** (Wasserstein distance) to measure the gap between distributions.
- **Math**: The critic (discriminator) must be **1-Lipschitz continuous**. 
- **WGAN-GP**: Enforces the Lipschitz constraint by adding a **Gradient Penalty** to the loss function.
- **Benefit**: Virtually eliminates Mode Collapse and provides a meaningful loss metric that correlates with image quality.

---

## 3. Diffusion Models: The State of the Art

Diffusion models decompose image generation into a series of small denoising steps.

### 3.1 The Forward Process ($q$)
Adds Gaussian noise to image $x_0$ over $T$ steps:
$$ x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon $$
Where $\epsilon \sim N(0, I)$.

### 3.2 The Reverse Process ($p$)
A U-Net is trained to predict the noise $\epsilon$ that was added at time $t$.
- **Objective**: $\mathcal{L}_{simple} = \mathbb{E}_{t, x_0, \epsilon} [\| \epsilon - \epsilon_\theta(x_t, t) \|^2]$
- **Guidance**: **Classifier-Free Guidance (CFG)** allows the model to follow text prompts by blending the conditional and unconditional predictions.

---

## 💻 Professional Implementation

### 1. The Reparameterization Trick (NumPy/PyTorch)
```python
import torch

def reparameterize(mu, logvar):
    """
    Sample z = mu + std * epsilon
    Allows gradients to flow through the stochastic sampling.
    """
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std
```

### 2. WGAN-GP Gradient Penalty Logic
```python
def compute_gradient_penalty(critic, real_samples, fake_images):
    # Random weight term for interpolation between real and fake
    alpha = torch.rand((real_samples.size(0), 1, 1, 1))
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_images)).requires_grad_(True)
    
    d_interpolates = critic(interpolates)
    
    # Get gradient w.r.t. interpolates
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(d_interpolates),
        create_graph=True,
        retain_graph=True,
    )[0]
    
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty
```

---

## 📊 Summary Comparison

| Feature | VAE | GAN | Diffusion |
| :--- | :--- | :--- | :--- |
| **Training** | Stable | Unstable (needs tuning)| Very Stable |
| **Diversity** | High | Low (Mode collapse) | **Very High** |
| **Quality** | Blurry | **Sharp** | **Photo-realistic**|
| **Inference** | Fast | Fast | Slow (Iterative) |

---

## 🎯 ML Applications & Advanced Scenarios

| Technique | Professional Use Case |
| :--- | :--- |
| **ControlNet** | Adding specific conditions (e.g., human pose, depth map) to Stable Diffusion. |
| **StyleGAN3** | Generating aliasing-free high-res portraits for VFX. |
| **DreamBooth** | Fine-tuning a diffusion model on a specific subject (e.g., your dog). |
| **Inpainting** | Automatically filling in missing parts of an image (e.g., removing a person). |

---

## ❓ Quick Check Questions

1. Why is the KL-divergence term necessary in the VAE loss function?
2. What is the "1-Lipschitz" constraint in WGAN, and how is it enforced?
3. In Diffusion models, why is it easier to predict the *noise* rather than the original *image*?
4. Explain the difference between DDPM and DDIM (Inference speed).
5. What does the "Latent" in Latent Diffusion (Stable Diffusion) refer to?

---

## 📝 Answers to Quick Check

<details>
<summary>Click to reveal answers</summary>

1. Without **KL-divergence**, the encoder would just map each input to a single, distant point in the latent space (like a standard autoencoder). The KL term forces the space to be a dense, continuous "ball" centered at zero, ensuring that points between two encoded samples generate meaningful "blended" images.
2. The **Lipschitz constraint** ensures that the discriminator (critic) doesn't have local gradients that explode, which stabilizes the adversarial game. It is enforced using **Gradient Penalty (GP)**, which penalizes the model if the norm of the gradient of the critic output with respect to the input is not close to 1.
3. Predicting the original image from noise is a massive, multi-modal jump. Predicting the **noise** added at a specific step is a much smaller, better-defined Gaussian problem. By repeating this small prediction 50-100 times, the model can generate a complex image accurately.
4. **DDPM** (Probabilistic) requires many stochastic steps (e.g., 1000) to generate an image. **DDIM** (Implicit) uses a deterministic mathematical path that allows for much faster sampling (e.g., 20-50 steps) while maintaining similar quality.
5. Standard diffusion happens in **Pixel Space** ($512 \times 512 \times 3 = 786k$ values). Stable Diffusion happens in the **Latent Space** ($64 \times 64 \times 4 = 16k$ values) of a pre-trained VAE. This $48\times$ reduction in dimensionality is why Stable Diffusion is so fast.

</details>

---

## 📚 Recommended Resources
- **Paper**: [Denoising Diffusion Probabilistic Models (DDPM)](https://arxiv.org/abs/2006.11239)
- **Paper**: [High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752)
- **Blog**: [Lilian Weng: What are Diffusion Models?](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/) - *Mathematical Masterpiece*.

---

**Status:** ✅ Expanded Standard (10/10)
**Next:** Vision Transformers & Multimodal AI (ViT, CLIP, LLaVA math)
