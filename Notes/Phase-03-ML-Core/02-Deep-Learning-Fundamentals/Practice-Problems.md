# Deep Learning Fundamentals - Practice Problems

## 📊 Graded Practice Levels

### Level 1: Basic Concept Recall
**1.1** What is the "Universal Approximation Theorem"?
**1.2** Define "Backpropagation" in one sentence.
**1.3** What is the main problem with the Sigmoid activation function in very deep networks?
**1.4** Explain the difference between "Weight Sharing" in CNNs and standard connections in an MLP.
**1.5** What is the role of the "Forget Gate" in an LSTM?

### Level 2: Intermediate Operations & Tuning
**2.1** How does "He Initialization" differ from "Xavier Initialization," and why was it specifically designed for ReLU?
**2.2** In the Adam optimizer, what is the purpose of "Bias Correction"?
**2.3** Explain how "Dropout" acts as a regularizer during training and how it changes during inference.
**2.4** Given a $224 \times 224 \times 3$ image, what is the output shape after a $3 \times 3$ convolution with 64 filters, stride 2, and padding 1?
**2.5** Why do we use "Gradient Clipping" specifically for RNNs?

### Level 3: Advanced Architectural Analysis
**3.1** Compare "Batch Normalization" and "Layer Normalization." Why is LayerNorm preferred for Transformers and RNNs?
**3.2** Derive why "Residual Connections" (Skip Connections) prevent the vanishing gradient problem using the chain rule.
**3.3** Explain the "Inverted Dropout" implementation and why it's more efficient than scaling during testing.
**3.4** Compare LSTMs and GRUs. What is the specific architectural simplification in GRUs?

### Level 4: Implementation Practice (Pseudo-code or Python)
**4.1** Write the code for a single "Forward Step" of a vanilla RNN cell: $h_t = \tanh(W_{hh} h_{t-1} + W_{xh} x_t + b)$.
**4.2** Implement a "Max Pooling" function from scratch using NumPy for a single channel $4 \times 4$ matrix with a $2 \times 2$ window and stride 2.
**4.3** Write a Python function to implement the "Softmax" activation with numerical stability (subtracting the max).

### Level 5: Real-world Model Design
**5.1** **Scenario:** You are designing a system to classify 10-second audio clips of bird songs.
- The input is a Spectrogram (an image-like representation of frequency over time).
- Audio patterns are sequential and can happen at any time in the clip.
**Task:** Propose a hybrid architecture (e.g., CRNN). Explain which part of the network handles spatial features, which part handles temporal dependencies, and what specific layers (Conv2D, LSTM/GRU, etc.) you would use.

---

## 📝 Solutions (Selected)

<details>
<summary>Click to reveal solutions</summary>

### 1.2
Backpropagation is the efficient calculation of the gradient of the loss function with respect to all weights in the network by applying the **Chain Rule** from the output layer back to the input.

### 2.2
Bias correction compensates for the fact that the first and second moment estimates in Adam are initialized to zero, which makes them biased towards zero during the initial steps of training.

### 2.4
Formula: $\lfloor(W - K + 2P) / S\rfloor + 1$
$\lfloor(224 - 3 + 2(1)) / 2\rfloor + 1 = \lfloor 223 / 2 \rfloor + 1 = 111 + 1 = 112$.
Output Shape: **$112 \times 112 \times 64$**.

### 3.1
**Batch Normalization** normalizes across the batch dimension; it depends on other samples in the batch and is hard to use with variable-length sequences. **Layer Normalization** normalizes across the feature dimension for each sample independently, making it ideal for RNNs and Transformers.

### 4.3
```python
def stable_softmax(x):
    z = x - np.max(x) # Stability
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z)
```

</details>

---

## 📝 Notes Section

### My Practice Problems:


### Mistakes to Review:


### Key Insights:


---
**Last Updated:** 2026-03-23
**Status:** ✅ Deep Learning Fundamentals Complete!
