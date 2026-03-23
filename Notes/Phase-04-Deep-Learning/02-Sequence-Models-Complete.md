# 10.2 Sequence Models: Temporal Dependencies & Memory

## 🎯 Quick Overview
- **RNN Mechanics**: The math of hidden states and recurrent loops
- **Vanishing Gradients**: Deriving why standard RNNs "forget"
- **LSTMs & GRUs**: Master the gate logic (Forget, Input, Output)
- **Seq2Seq Architecture**: Encoder-Decoder and the context bottleneck
- **Foundation for**: Machine Translation, Speech Recognition, and Time-Series Forecasting

---

## 1. Recurrent Neural Networks (RNNs)

RNNs process data sequentially, maintaining a hidden state $h_t$ that carries information from previous steps.

### 1.1 The Forward Pass
At time $t$:
1.  **Hidden State**: $h_t = \sigma(W_{hh}h_{t-1} + W_{xh}x_t + b_h)$
2.  **Output**: $y_t = \text{softmax}(W_{hy}h_t + b_y)$

### 1.2 Backpropagation Through Time (BPTT)
To train an RNN, we "unroll" it through time. The loss $L$ is the sum of losses at each time step.
**The Chain Rule Challenge**:
$$\frac{\partial L_t}{\partial W_{hh}} = \sum_{k=1}^t \frac{\partial L_t}{\partial y_t} \frac{\partial y_t}{\partial h_t} \left( \prod_{j=k+1}^t \frac{\partial h_j}{\partial h_{j-1}} \right) \frac{\partial h_k}{\partial W_{hh}}$$

**The Vanishing Gradient**: If the weights in $W_{hh}$ are small, the product $\prod \frac{\partial h_j}{\partial h_{j-1}}$ shrinks to zero exponentially as $(t-k)$ increases. This is why standard RNNs cannot learn long-term dependencies (>10-20 steps).

---

## 2. Gated Architectures (LSTM & GRU)

LSTMs and GRUs use "gates" to protect and control the flow of information.

### 2.1 LSTM: Long Short-Term Memory
Introduces the **Cell State** ($C_t$), a "conveyor belt" that carries info through time with minimal interaction.

#### The 4-Step Math:
1.  **Forget Gate ($f_t$)**: Decides what to discard.
    $f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$
2.  **Input Gate ($i_t \text{ and } \tilde{C}_t$)**: Decides what to add.
    $i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$
    $\tilde{C}_t = \tanh(W_C \cdot [h_{t-1}, x_t] + b_C)$
3.  **Update Cell State**:
    $C_t = f_t * C_{t-1} + i_t * \tilde{C}_t$
4.  **Output Gate ($o_t$)**: Decides what to show.
    $o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$
    $h_t = o_t * \tanh(C_t)$

---

## 3. Sequence-to-Sequence (Seq2Seq)

Used for mapping a variable-length input to a variable-length output.

### 3.1 The Encoder-Decoder Logic
```
Input: "How are you?" 
Encoder -> [h1, h2, h3] -> final state 'z' (Context Vector)
Decoder -> 'z' + <START> -> "Comment"
Decoder -> h_dec1 + "Comment" -> "allez"
Decoder -> h_dec2 + "allez" -> "vous?"
```

### 3.2 The Bottleneck Problem
The Encoder must compress the entire meaning of a 100-word paragraph into a single vector 'z'. This leads to massive information loss. **Attention** (Phase 10.3) was invented to fix this by allowing the decoder to look at *all* encoder states.

---

## 💻 Professional Implementation

### 1. Manual LSTM Cell Step (NumPy)
```python
import numpy as np

def sigmoid(x): return 1 / (1 + np.exp(-x))

def lstm_step(x_t, h_prev, c_prev, params):
    # Concatenate hidden state and input
    concat = np.concatenate((h_prev, x_t), axis=0)
    
    # Gates
    f = sigmoid(np.dot(params['Wf'], concat) + params['bf'])
    i = sigmoid(np.dot(params['Wi'], concat) + params['bi'])
    o = sigmoid(np.dot(params['Wo'], concat) + params['bo'])
    c_tilde = np.tanh(np.dot(params['Wc'], concat) + params['bc'])
    
    # State updates
    c_next = f * c_prev + i * c_tilde
    h_next = o * np.tanh(c_next)
    
    return h_next, c_next
```

### 2. Bi-Directional LSTM (PyTorch)
```python
import torch.nn as nn

class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        # bidirectional=True doubles the hidden dim for the final layer
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, 1)
        
    def forward(self, x):
        x = self.embedding(x)
        output, (hn, cn) = self.lstm(x)
        # Concatenate final hidden states from both directions
        fwd_last = hn[-2,:,:]
        rev_last = hn[-1,:,:]
        merged = torch.cat((fwd_last, rev_last), dim=1)
        return self.fc(merged)
```

---

## 📊 Summary Comparison

| Metric | Simple RNN | LSTM | GRU |
| :--- | :--- | :--- | :--- |
| **Gating** | None | 3 Gates (F, I, O) | 2 Gates (Reset, Update) |
| **Memory Type** | Hidden State | Cell + Hidden State | Hidden State |
| **Training Speed** | Fast | Slow | Moderate |
| **Long sequences**| Very Poor | Excellent | Good |

---

## 🎯 ML Applications & Advanced Scenarios

| Technique | Professional Use Case |
| :--- | :--- |
| **Stacked LSTMs** | Complex signal processing (e.g., EEG data analysis). |
| **Bi-LSTMs** | NER where context from both "left" and "right" is required. |
| **Beam Search** | Optimizing Decoder output in Seq2Seq translation. |
| **Teacher Forcing** | Accelerating training of RNN decoders by providing ground-truth inputs. |

---

## ❓ Quick Check Questions

1. Derive why the derivative of the Cell State $C_t$ with respect to $C_{t-1}$ prevents vanishing gradients in LSTMs.
2. In a Bi-directional RNN, how many hidden states does the model maintain at time step $t$?
3. What is the difference between "Hard" and "Soft" Attention (conceptual)?
4. Why is "Teacher Forcing" used during training but not during inference?
5. How does a GRU merge the functions of the Forget and Input gates?

---

## 📝 Answers to Quick Check

<details>
<summary>Click to reveal answers</summary>

1. In an LSTM, the gradient $\frac{\partial C_t}{\partial C_{t-1}} = f_t + (\text{small terms})$. Because $f_t$ (the forget gate) can be close to 1, the gradient is **added** rather than multiplied, allowing it to flow back through thousands of steps without vanishing.
2. It maintains **two** hidden states: one from the forward pass ($h \to$) and one from the backward pass ($\leftarrow h$). These are typically concatenated.
3. **Soft Attention** (standard) computes a weighted average of all input features (differentiable). **Hard Attention** picks exactly one input feature to look at (stochastic, requires Reinforcement Learning to train).
4. **Teacher Forcing** feeds the *correct* previous token into the decoder during training to prevent the model from drifting off-track early on. During inference, we don't have the "correct" labels, so the model must use its own predicted token from the previous step.
5. A GRU uses an **Update Gate** ($z_t$). It adds $z_t \times (\text{new info})$ and $(1 - z_t) \times (\text{old info})$. This ensures that whenever new info is added, a corresponding amount of old info is forgotten.

</details>

---

## 📚 Recommended Resources
- **Paper**: [Long Short-Term Memory (Hochreiter & Schmidhuber)](https://www.bioinf.jku.at/publications/older/2604.pdf)
- **Blog**: [Understanding LSTMs (Colah's Blog)](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) - *Highly Recommended*.
- **Course**: CS224N: Natural Language Processing with Deep Learning (Stanford).

---

**Status:** ✅ Expanded Standard (10/10)
**Next:** The Transformer Revolution (Self-Attention, Multi-Head, Positional Encoding)
