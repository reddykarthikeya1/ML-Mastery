# 11.4 Image Generation (GANs, VAEs, Diffusion)

## 🎯 Quick Overview
- **Generative AI**: Moving from "recognizing" to "creating" new data
- **VAEs (Variational Autoencoders)**: Learning structured latent spaces
- **GANs (Generative Adversarial Networks)**: The game-theory approach to image generation
- **Diffusion Models**: The current SOTA (DALL-E 3, Midjourney, Stable Diffusion)
- **Foundation for**: Deepfakes, Art generation, Data augmentation, and Image restoration

---

## 1. Variational Autoencoders (VAEs)

Unlike a standard autoencoder, a **VAE** learns a **probability distribution** ($mean$ and $variance$) of the data in the latent space.
- **Goal**: Ensure the latent space is continuous, allowing us to sample new points and generate never-before-seen images.
- **Key Math**: Uses the **Reparameterization Trick** to allow backpropagation through random sampling.

---

## 2. Generative Adversarial Networks (GANs)

GANs consist of two neural networks competing against each other:
1. **The Generator**: Tries to create realistic fake images from random noise.
2. **The Discriminator**: Tries to distinguish between real training images and fake ones from the generator.

- **The Game**: It's a Zero-Sum game. As the discriminator gets better at catching fakes, the generator must get better at creating them.
- **Problem**: GANs are notoriously hard to train (Mode Collapse, Instability).

---

## 3. Diffusion Models (The New King)

Diffusion models work by slowly destroying data with noise and then learning to **reverse** that process.

### 3.1 Forward Diffusion
Gradually adding Gaussian noise to an image until it becomes pure white noise.

### 3.2 Reverse Diffusion (The Generative Part)
A neural network (usually a **U-Net**) is trained to predict and remove the noise at each step, "recovering" an image from pure noise.

### 3.3 Stable Diffusion
Uses **Latent Diffusion**. Instead of diffusing pixels (slow/expensive), it diffuses the "latent representation" of the image using an autoencoder, making it much faster and capable of running on consumer GPUs.

---

## 💻 Python Code Examples

### 1. Generating with Stable Diffusion (HuggingFace Diffusers)
```python
from diffusers import StableDiffusionPipeline
import torch

# 1. Load pipeline
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", 
    torch_dtype=torch.float16
)
pipe = pipe.to("cuda")

# 2. Generate
prompt = "A futuristic city in the style of Van Gogh"
image = pipe(prompt).images[0]

# 3. Save
image.save("futuristic_city.png")
```

### 2. GAN Loss Logic (Conceptual)
```python
# Discriminator Loss
d_loss_real = binary_cross_entropy(discriminator(real_images), 1)
d_loss_fake = binary_cross_entropy(discriminator(generator(noise)), 0)
d_loss = d_loss_real + d_loss_fake

# Generator Loss (wants the discriminator to think its fakes are real)
g_loss = binary_cross_entropy(discriminator(generator(noise)), 1)
```

---

## 📊 Summary Table

| Model | Mechanism | Pros | Cons |
|-------|-----------|------|------|
| **VAE** | Latent Distribution | Stable training, Continuous space | Often generates blurry images |
| **GAN** | Adversarial Game | Sharp, high-quality images | Unstable, Mode collapse |
| **Diffusion**| Iterative Denoising | SOTA quality, High diversity | Slow generation (many steps) |

---

## 🎯 ML Applications

| Technique | ML Application |
|-----------|----------------|
| CycleGAN | Changing day-time photos to night-time |
| Stable Diffusion | Concept art and UI/UX design prototyping |
| Denoising Diffusion | Super-resolution (enhancing low-res images) |
| Deepfakes | Hollywood de-aging and dubbing |

---

## ❓ Quick Check Questions

1. Why is the "Reparameterization Trick" necessary in VAEs?
2. What is "Mode Collapse" in GAN training?
3. How does Diffusion differ from GANs in terms of the training process?
4. What is the difference between Pixel-space Diffusion and Latent Diffusion?
5. What role does the "U-Net" play in a Diffusion model?

---

## 📝 Answers to Quick Check

<details>
<summary>Click to reveal answers</summary>

1. You cannot backpropagate through a random sampling operation. The **reparameterization trick** rewrites the sample as $z = \mu + \sigma \cdot \epsilon$ (where $\epsilon \sim N(0,1)$), making the parameters $\mu$ and $\sigma$ differentiable.
2. **Mode Collapse** is a GAN failure where the generator discovers a single "type" of image that always fools the discriminator and stops trying to create anything else, leading to a lack of variety in the output.
3. GANs are trained using **Adversarial loss** (competing with another network). Diffusion models are trained using a **Mean Squared Error (MSE) loss** to predict the specific noise added at a given step, which is much more stable.
4. **Pixel-space diffusion** operates directly on image pixels (e.g., $1024 \times 1024 \times 3$), which is computationally massive. **Latent Diffusion** operates on a compressed version of the image (e.g., $64 \times 64$), making high-res generation much faster.
5. The **U-Net** is the "brain" of the diffusion process. At each step, it takes the noisy image and the current time-step as input and predicts the **noise component** that needs to be subtracted to get closer to the clean image.

</details>

---

**Status:** ✅ Complete
**Next:** Vision Transformers (ViT) & Multimodal AI (CLIP, LLaVA)
