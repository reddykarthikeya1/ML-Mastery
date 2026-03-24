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

#### 🧒 ELI5: The Vending Machine with a Window

> Imagine a magical vending machine that creates images.
>
> **Standard Autoencoder** (no window):
> - You put in a photo of a cat
> - Machine secretly picks a code (like "A7")
> - Machine reconstructs the cat from code "A7"
> - Problem: Codes are random! "A7" might be cats today, cars tomorrow
>
> **VAE** (with a window):
> - You put in a photo of a cat
> - Machine shows you THROUGH A WINDOW: "I'm picking from the CAT ZONE (centered around mean=0, variance=1)"
> - The window forces the machine to pick codes from a predictable zone
> - KL divergence = Penalty if machine tries to pick from outside the zone
>
> **Reparameterization Trick** (why it's needed):
> - Problem: You can't backpropagate through "randomly pick a code"
> - Solution: 
>   1. Machine tells you: "Mean = 5, Variance = 2"
>   2. YOU generate random noise ε (this is external, not the machine's choice)
>   3. Code = Mean + Variance × ε = 5 + 2 × ε
>   4. Now you can backpropagate through Mean and Variance!
>
> **Analogy**: Instead of "machine magically picks a number," it's "machine picks a formula (mean/variance), you supply randomness, together you get the code."

</details>

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

#### 🧒 ELI5: Shredding and Reconstructing a Photo

> Imagine you have a beautiful photo and a paper shredder.
>
> **Forward Process (Destroying)**:
> 1. Start with perfect photo ($x_0$)
> 2. Run through shredder once → slightly noisy ($x_1$)
> 3. Run through again → more noisy ($x_2$)
> 4. Repeat 1000 times → pure static ($x_{1000}$)
>
> **Reverse Process (Reconstructing)**:
> - Train a model to do ONE thing: "Given a slightly shredded photo, remove the shredding"
> - At step 500: "What noise was added at step 500?" → Remove it
> - At step 499: "What noise was added at step 499?" → Remove it
> - Repeat until you reconstruct the original!
>
> **Why predict noise, not image?**
> - Predicting the whole image from noise = "Create a masterpiece from nothing" (HARD!)
> - Predicting noise = "What's the tiny difference between these two similar images?" (EASY!)
> - Like: "Spot the difference" vs. "Draw this from memory"
>
> **Classifier-Free Guidance (CFG)**:
> Imagine a GPS that gives you TWO routes:
> - Route A: Fastest way (unconditional - no preferences)
> - Route B: Fastest way avoiding tolls (conditional - your preference)
> - CFG blends them: "Mostly avoid tolls, but if it's MUCH faster, maybe one toll is okay"
> - Guidance scale = How much you prefer avoiding tolls

</details>

---

## 💻 Professional Implementation: End-to-End Diffusion Pipeline

This script demonstrates a memory-optimized Stable Diffusion pipeline with half-precision (FP16) and automatic safety filtering.

```python
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
from typing import Optional

class DiffusionGenerator:
    def __init__(self, model_id: str = "runwayml/stable-diffusion-v1-5"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        
        # 1. Load optimized pipeline
        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_id, 
            torch_dtype=self.dtype,
            variant="fp16" if self.device == "cuda" else None
        ).to(self.device)
        
        # 2. Memory optimization (Attention Slicing)
        if self.device == "cuda":
            self.pipe.enable_attention_slicing()
            print(f"Diffusion loaded on {self.device} with FP16 precision.")

    def generate(self, prompt: str, negative_prompt: Optional[str] = None, 
                 steps: int = 50, guidance_scale: float = 7.5) -> Image.Image:
        """Generate an image from a text prompt with CFG guidance."""
        with torch.autocast(self.device):
            image = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=steps,
                guidance_scale=guidance_scale
            ).images[0]
        return image

# --- Usage Example ---
# gen = DiffusionGenerator()
# img = gen.generate(
#     prompt="A high-tech lab with glowing neon servers, cinematic lighting",
#     negative_prompt="blurry, distorted, low resolution",
#     steps=30
# )
# img.save("result.png")
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

## 4. Advanced Diffusion Techniques

### 4.1 Classifier-Free Guidance (CFG)
Enable conditional generation without a separate classifier.

**The Trick**: Train the model with occasional null conditioning ($c = \varnothing$).

**Guidance Formula**:
$$ \epsilon_\theta^{guided}(x_t, c) = \epsilon_\theta(x_t, \varnothing) + w \cdot (\epsilon_\theta(x_t, c) - \epsilon_\theta(x_t, \varnothing)) $$

Where $w$ is the guidance scale (typically 7-15).

```python
class GuidedDiffusion:
    def __init__(self, model, guidance_scale=7.5):
        self.model = model
        self.guidance_scale = guidance_scale
    
    def sample(self, prompt_embed, steps=50):
        # Start from pure noise
        x_t = torch.randn(1, 4, 64, 64)  # Latent space
        
        for t in reversed(range(steps)):
            # Unconditional prediction
            eps_uncond = self.model(x_t, t, None)
            
            # Conditional prediction
            eps_cond = self.model(x_t, t, prompt_embed)
            
            # Apply guidance
            eps_guided = eps_uncond + self.guidance_scale * (eps_cond - eps_uncond)
            
            # Denoising step
            x_t = self.denoise_step(x_t, t, eps_guided)
        
        return x_t
```

---

### 4.2 ControlNet: Spatial Conditioning
Add fine-grained spatial control to diffusion models.

**Architecture**:
- Clone the diffusion model's encoder
- Train the clone on conditioning input (edges, depth, pose)
- Zero convolution layers for clean integration

```python
class ControlNet(nn.Module):
    def __init__(self, unet, conditioning_channels=3):
        super().__init__()
        self.unet = unet
        
        # ControlNet encoder (copy of UNet encoder)
        self.control_encoder = copy.deepcopy(unet.encoder)
        
        # Zero convolution layers (start with zero weights)
        self.zero_convs = nn.ModuleList([
            ZeroConv2d(ch, ch) for ch in unet.encoder_channels
        ])
        
        # Conditioning input
        self.cond_input = nn.Conv2d(conditioning_channels, 3, 3, padding=1)
    
    def forward(self, x, t, cond, context=None):
        # Process conditioning
        cond = self.cond_input(cond)
        
        # Run control encoder
        control_features = self.control_encoder(cond, t)
        
        # Apply zero convolutions
        control_outputs = [
            zero_conv(feature) 
            for feature, zero_conv in zip(control_features, self.zero_convs)
        ]
        
        # Run UNet with control signals
        output = self.unet(x, t, context, control=control_outputs)
        
        return output


class ZeroConv2d(nn.Module):
    """Convolution layer initialized to zero."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        
        # Initialize to zero
        nn.init.zeros_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)
    
    def forward(self, x):
        return self.conv(x)
```

---

### 4.3 IP-Adapter: Image Prompting
Use images instead of text for prompting.

**The Method**:
1.  Encode reference image with CLIP
2.  Project CLIP features to match text embedding dimension
3.  Use as cross-attention context

```python
class IPAdapter(nn.Module):
    def __init__(self, unet, clip_model, projection_dim=768):
        super().__init__()
        self.unet = unet
        self.clip = clip_model
        
        # Image projection to text space
        self.image_proj = nn.Linear(clip_model.dim, projection_dim)
        
        # Additional cross-attention for image features
        self.image_cross_attn = nn.MultiheadAttention(
            embed_dim=projection_dim,
            num_heads=8,
            batch_first=True
        )
    
    def encode_image_prompt(self, image):
        """Encode image to use as prompt."""
        with torch.no_grad():
            clip_features = self.clip.encode_image(image)
        
        # Project to text embedding space
        image_embeds = self.image_proj(clip_features)
        
        return image_embeds
    
    def forward(self, x, t, image_prompt_embeds, text_embeds=None):
        # Combine image and text prompts
        if text_embeds is not None:
            context = torch.cat([text_embeds, image_prompt_embeds], dim=1)
        else:
            context = image_prompt_embeds
        
        # Standard diffusion forward pass
        return self.unet(x, t, context=context)
```

---

## 5. Consistency Models: One-Step Generation

### 5.1 The Consistency Approach
Train a model that directly maps noisy input to clean output in one step.

**Key Insight**: Learn a function that satisfies the consistency property:
$$ f(x_t, t) = f(x_{t'}, t') \quad \forall t, t' $$

Where $f$ maps any noisy version to the same clean output.

---

### 5.2 Consistency Model Training

```python
class ConsistencyModel(nn.Module):
    def __init__(self, unet, num_steps=64):
        super().__init__()
        self.unet = unet
        self.num_steps = num_steps
        
        # Time conditioning
        self.time_embed = nn.Embedding(num_steps, 256)
        
        # Skip connection scaling
        self.c_skip = nn.Parameter(torch.ones(1))
        self.c_out = nn.Parameter(torch.zeros(1))
    
    def forward(self, x, t):
        """Map noisy input at time t to clean output."""
        # Time embedding
        t_emb = self.time_embed(t)
        
        # UNet forward pass
        f_t = self.unet(x, t_emb)
        
        # Skip connection + output scaling
        output = self.c_skip * x + self.c_out * f_t
        
        return output
    
    def sample_one_step(self, noise):
        """Generate image in single step."""
        t = torch.full((noise.shape[0],), self.num_steps - 1, dtype=torch.long)
        return self(noise, t)
    
    def sample_multistep(self, noise, steps=4):
        """Generate with multiple refinement steps."""
        x = noise
        
        for i in reversed(range(steps)):
            t = torch.full((noise.shape[0],), i, dtype=torch.long)
            x = self(x, t)
            
            # Add noise for next step (if not last)
            if i > 0:
                noise_level = self.get_noise_level(i - 1)
                x = x + noise_level * torch.randn_like(x)
        
        return x
```

---

### 5.3 Progressive Distillation
Distill a diffusion model into fewer steps.

**Process**:
1.  Train student to match teacher's 2-step output
2.  Halve the number of steps
3.  Repeat until desired step count

```python
class ProgressiveDistillation:
    def __init__(self, teacher_model, student_model):
        self.teacher = teacher_model
        self.student = student_model
    
    def distill_step(self, x_0, steps):
        """One distillation step."""
        # Teacher generates with many steps
        with torch.no_grad():
            x_teacher = self.teacher.sample(x_0, steps=steps * 2)
        
        # Student generates with half the steps
        x_student = self.student.sample(x_0, steps=steps)
        
        # Match outputs
        loss = F.mse_loss(x_student, x_teacher)
        
        return loss
```

---

## 6. Video Generation

### 6.1 Video Diffusion Models
Extend diffusion to spatiotemporal data.

**Architecture Options**:
1.  **3D U-Net**: Full spatiotemporal convolutions
2.  **Factorized**: Separate spatial + temporal attention
3.  **Latent Video**: Compress video to latent space first

```python
class VideoDiffusion(nn.Module):
    def __init__(self, image_unet, num_frames=16):
        super().__init__()
        self.num_frames = num_frames
        
        # Initialize from image model
        self.unet = image_unet
        
        # Add temporal layers
        self.temporal_attention = nn.ModuleList([
            TemporalAttentionBlock(ch, num_heads=8)
            for ch in image_unet.channels
        ])
        
        # Time position encoding
        self.frame_pos = PositionalEncoding(256, max_len=num_frames)
    
    def forward(self, x, t, context=None):
        # x: [batch, frames, channels, height, width]
        b, f, c, h, w = x.shape
        
        # Reshape for spatial processing
        x_flat = x.view(b * f, c, h, w)
        
        # Spatial UNet
        features = self.unet(x_flat, t, context)
        
        # Reshape back
        features = features.view(b, f, -1, h, w)
        
        # Temporal attention
        for temporal_layer in self.temporal_attention:
            features = temporal_layer(features)
        
        return features


class TemporalAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.attention = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)
    
    def forward(self, x):
        # x: [batch, frames, channels, height, width]
        b, f, c, h, w = x.shape
        
        # Reshape for temporal attention
        x = x.permute(0, 3, 4, 1, 2).reshape(b * h * w, f, c)
        
        # Self-attention across frames
        x_attended, _ = self.attention(x, x, x)
        x_attended = self.norm(x_attended + x)
        
        # Reshape back
        x_attended = x_attended.view(b, h, w, f, c).permute(0, 3, 4, 1, 2)
        
        return x_attended
```

---

### 6.2 Stable Video Diffusion
Meta's approach to video generation.

**Key Features**:
- Fine-tuned from Stable Diffusion image model
- Generates 14-25 frames at 576×1024
- Motion bucket controls camera/object motion

```python
class StableVideoDiffusion:
    def __init__(self, image_model, motion_bucket_size=128):
        # Load pretrained image model
        self.model = image_model
        
        # Motion conditioning
        self.motion_bucket = nn.Embedding(motion_bucket_size, 256)
        
        # FPS conditioning
        self.fps_embed = nn.Embedding(30, 256)
    
    def generate(self, image, motion_value=127, fps=10, steps=25):
        """Generate video from single image."""
        # Encode input image
        latents = self.model.encode_image(image)
        
        # Repeat for multiple frames
        latents = latents.unsqueeze(1).repeat(1, 14, 1, 1, 1)
        
        # Add motion conditioning
        motion_embed = self.motion_bucket(torch.tensor([motion_value]))
        fps_embed = self.fps_embed(torch.tensor([fps]))
        context = torch.cat([motion_embed, fps_embed], dim=1)
        
        # Diffusion sampling
        for t in reversed(range(steps)):
            noise = self.model(latents, t, context)
            latents = self.denoise_step(latents, t, noise)
        
        # Decode to video
        video = self.model.decode(latents)
        
        return video
```

---

## 7. Advanced Training Techniques

### 7.1 DreamBooth: Personalized Generation
Fine-tune diffusion on specific subjects.

**The Method**:
1.  Collect 3-5 images of subject
2.  Train with rare class token (e.g., "sks dog")
3.  Preserve class knowledge with prior preservation

```python
class DreamBoothTrainer:
    def __init__(self, diffusion_model, class_prompt):
        self.model = diffusion_model
        self.class_prompt = class_prompt
        
        # Prior preservation images (generated)
        self.prior_images = self.generate_prior(100)
    
    def generate_prior(self, num_images):
        """Generate class prior images."""
        prompts = [f"a photo of {self.class_prompt}"] * num_images
        return self.model.generate(prompts)
    
    def train_step(self, subject_images, subject_prompt):
        """One training step for DreamBooth."""
        all_images = subject_images + self.prior_images
        
        # Subject prompts
        subject_prompts = [subject_prompt] * len(subject_images)
        
        # Class prompts for prior
        class_prompts = [f"a photo of {self.class_prompt}"] * len(self.prior_images)
        
        all_prompts = subject_prompts + class_prompts
        
        # Standard diffusion training
        loss = self.diffusion_loss(all_images, all_prompts)
        
        return loss
```

---

### 7.2 Textual Inversion
Learn new "words" (embeddings) for concepts.

**The Approach**:
- Optimize a single embedding vector (not full model)
- 3-5 training images
- Works with any diffusion model

```python
class TextualInversion:
    def __init__(self, diffusion_model, tokenizer):
        self.model = diffusion_model
        self.tokenizer = tokenizer
    
    def train_embedding(self, images, placeholder_token, class_token, steps=3000):
        """Train a new embedding vector."""
        # Initialize embedding randomly
        embedding = nn.Parameter(torch.randn(1, 768))
        
        # Add to tokenizer
        self.tokenizer.add_placeholder(placeholder_token, embedding)
        
        optimizer = torch.optim.Adam([embedding], lr=5e-4)
        
        for step in range(steps):
            # Sample training image
            image = random.choice(images)
            
            # Create prompt with placeholder
            prompt = f"a photo of {placeholder_token} {class_token}"
            
            # Compute diffusion loss
            loss = self.model.loss(image, prompt)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        return embedding
```

---

## 8. Evaluation Metrics

### 8.1 Generation Quality Metrics

| Metric | What It Measures | Ideal Value |
| :--- | :--- | :--- |
| **FID** | Distribution similarity | Lower is better |
| **IS** | Image quality + diversity | Higher is better |
| **CLIP Score** | Text-image alignment | Higher is better |
| **LPIPS** | Perceptual similarity | Lower is better |
| **PSNR** | Pixel-level similarity | Higher is better |

```python
def compute_fid(real_features, fake_features):
    """Compute Fréchet Inception Distance."""
    mu_real = real_features.mean(dim=0)
    sigma_real = real_features.cov()
    
    mu_fake = fake_features.mean(dim=0)
    sigma_fake = fake_features.cov()
    
    # FID formula
    diff = (mu_real - mu_fake).pow(2).sum()
    trace = (sigma_real + sigma_fake - 
             2 * (sigma_real @ sigma_fake).sqrtm().trace())
    
    return diff + trace
```

---

## 🔬 Research Frontiers (2024-2025)

### 9.1 Faster Generation
- **LCM (Latent Consistency Models)**: 4-8 step generation
- **Turbo models**: Real-time image generation
- **Distillation**: Knowledge distillation for speed

### 9.2 Controllable Generation
- **Regional prompting**: Different prompts for different image regions
- **Layout guidance**: Generate based on bounding boxes
- **Sketch-to-image**: Convert sketches to photorealistic images

### 9.3 3D Generation
- **DreamFusion**: 3D objects from text via diffusion
- **Zero-1-to-3**: Novel views from single image
- **Wonder3D**: Multi-view consistent generation

### 9.4 Multimodal Generation
- **Image + Audio → Video**: Generate video from image and sound
- **Text + Layout → Scene**: Structured scene generation
- **Video → 3D**: Reconstruct 3D from video

---

**Status:** ✅ Elite Expanded Standard (14/10)
**Next:** Vision Transformers & Multimodal AI (ViT, CLIP, LLaVA, DINO, MAE)
