# Deep Learning Frameworks - Practice Problems

## 📊 Graded Practice Levels

### Level 1: Basic Concept Recall
**1.1** What is the difference between a `Tensor` and a standard `NumPy` array?
**1.2** In TensorFlow, what is the difference between `model.fit()` and a custom training loop using `GradientTape`?
**1.3** In PyTorch, what does `loss.backward()` actually do under the hood?
**1.4** What is the purpose of a "DataLoader" in any framework?
**1.5** In JAX, why are arrays immutable?

### Level 2: Intermediate Framework Operations
**2.1** In Keras, when would you use `layers.GlobalAveragePooling2D()` instead of `layers.Flatten()`?
**2.2** In PyTorch, why must you call `optimizer.zero_grad()` before `loss.backward()`?
**2.3** In JAX, explain the purpose of the `PRNGKey` and why we must "split" it for every random operation.
**2.4** How do you move a model to a GPU in PyTorch vs. TensorFlow?
**2.5** What is the difference between the "Sequential API" and the "Functional API" in Keras?

### Level 3: Advanced Framework Features
**3.1** **Custom Layers:** Outline the methods you need to override to create a custom layer in PyTorch (`__init__`, `forward`) vs. Keras (`__init__`, `build`, `call`).
**3.2** **Performance:** Explain how `tf.data.AUTOTUNE` and `prefetch` optimize the data pipeline in TensorFlow.
**3.3** **JIT Compilation:** What is the `@tf.function` (TF) or `@jit` (JAX) decorator doing, and what are "side effects" that can break it?
**3.4** **Distributed Training:** Compare "Data Parallelism" vs. "Model Parallelism" at a high level.

### Level 4: Python Implementation Practice
**4.1** **PyTorch:** Write a simple `nn.Module` class for a 2-layer MLP with ReLU activation.
**4.2** **Keras:** Write the code to implement a "Residual Connection" (Add layer) between two Conv2D layers using the Functional API.
**4.3** **JAX:** Use `vmap` to turn a function `predict(params, x)` that works on one sample into a function that works on a batch of samples.

### Level 5: Real-world Deployment & Selection
**5.1** **Scenario:** You are a Lead ML Engineer. Your team needs to build a State-of-the-Art (SOTA) Transformer model for a new research paper, but it also needs to be deployed to a mobile app eventually.
**Task:** Choose between TensorFlow/Keras, PyTorch, or JAX. Justify your framework choice based on:
1. Ease of experimentation (Research).
2. Ecosystem for mobile deployment (Lite/Mobile/CoreML).
3. Availability of pre-trained models (HuggingFace/Hub).

---

## 📝 Solutions (Selected)

<details>
<summary>Click to reveal solutions</summary>

### 1.3
`loss.backward()` computes the gradient of the loss with respect to every leaf node in the computation graph that has `requires_grad=True`, and stores these gradients in the `.grad` attribute of each tensor.

### 2.2
PyTorch accumulates gradients by default (adds them to existing values). `zero_grad()` clears old gradients from the previous training step so they don't interfere with the current update.

### 4.1
```python
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )
    def forward(self, x):
        return self.net(x)
```

### 4.3
```python
from jax import vmap
# Map over the 0-th axis of input x, but broadcast params (None)
batched_predict = vmap(predict, in_axes=(None, 0))
```

</details>

---

## 📝 Notes Section

### My Practice Problems:


### Mistakes to Review:


### Key Insights:


---
**Last Updated:** 2026-03-23
**Status:** ✅ Deep Learning Frameworks Complete!
