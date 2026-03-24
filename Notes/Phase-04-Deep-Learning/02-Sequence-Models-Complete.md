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

#### 🧒 ELI5: The Telephone Game (Vanishing Gradients)

> Remember the children's game where a message passes through 20 kids?
>
> **Forward pass** (making predictions):
> - Kid 1 whispers to Kid 2, who whispers to Kid 3...
> - Message: "The cat sat on the mat"
>
> **Backward pass** (BPTT - learning from mistakes):
> - Kid 20 got it wrong! We need to tell Kid 1 what went wrong
> - Kid 20 tells Kid 19: "You whispered too quietly"
> - Kid 19 tells Kid 18... and the correction gets quieter each time
> - By the time Kid 5 hears the correction, it's a barely audible whisper
> - Kid 1 never learns what they did wrong!
>
> **Why it vanishes**: Each kid (layer) whispers a bit quieter (multiplies by < 1). After 20 kids, the correction is 0.5²⁰ = 0.000001 of the original. Gone!
>
> **LSTM solution**: Instead of whispering, kids pass a written note (cell state). The note stays clear even through 100 kids!

</details>

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

#### 🧒 ELI5: Airport Security Checkpoint

> Imagine you're going through airport security with your carry-on bag (Cell State).
>
> **1. Forget Gate** = Security makes you throw away liquids:
> - "Do you need to keep that old water bottle from 10 airports ago?" → NO
> - Some things need to be forgotten to make room for new stuff
>
> **2. Input Gate** = What new items can you add:
> - You bought a souvenir at this airport (new information)
> - Security checks if it's allowed (sigmoid decides what to accept)
>
> **3. Update Cell State** = Update your bag contents:
> - Throw away the water (forget)
> - Add the souvenir (input)
> - Your bag now has updated contents for the next airport
>
> **4. Output Gate** = What you actually take out at security:
> - Your bag might have 20 things, but you only take out laptop + liquids
> - The output gate decides what to SHOW based on current situation
>
> **Why LSTMs remember long-term**: The cell state (your bag) goes through security mostly unchanged. Only small updates at each step. Information can travel through 100+ airports without being lost!

</details>

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

## 4. Advanced RNN Variants

### 4.1 Peephole LSTM
Standard LSTMs compute gates without looking at the cell state. **Peephole connections** allow gates to "peek" at $C_t$.

#### Modified Gate Equations:
$$ f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + W_{cf} \odot C_{t-1} + b_f) $$
$$ i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + W_{ci} \odot C_{t-1} + b_i) $$
$$ o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + W_{co} \odot C_t + b_o) $$

**Impact**: Better at learning precise timing tasks (e.g., "count exactly 10 steps then fire").

---

### 4.2 ConvLSTM (Spatiotemporal Sequences)
For video or weather prediction, we need to capture both spatial and temporal dependencies.

#### Architecture:
- Replace fully connected layers with **convolutional layers**.
- All operations become element-wise convolutions ($*$) instead of matrix multiplications.

$$ i_t = \sigma(W_{xi} * x_t + W_{hi} * h_{t-1} + W_{ci} * C_{t-1} + b_i) $$

**Applications**: 
- Precipitation nowcasting (predicting rain 1 hour ahead)
- Video frame prediction
- Traffic flow forecasting

---

### 4.3 Attention-Augmented RNNs
Before Transformers, researchers combined RNNs with attention.

#### Bahdanau Attention in Seq2Seq:
1.  Encoder produces hidden states: $h_1, h_2, ..., h_T$
2.  Decoder computes **attention weights** at each step:
    $$ \alpha_{tj} = \frac{\exp(e_{tj})}{\sum_{k=1}^T \exp(e_{tk})} $$
    $$ e_{tj} = \text{score}(s_{t-1}, h_j) $$
3.  Create **context vector**: $c_t = \sum_{j=1}^T \alpha_{tj} h_j$
4.  Feed $c_t$ to the decoder RNN at each step.

**Key Insight**: This allows the decoder to "focus" on relevant parts of the input sequence dynamically.

---

### 4.4 Deep (Stacked) RNNs
Single-layer RNNs have limited capacity. **Stacked RNNs** add depth:

$$ h_t^{(1)} = \text{RNN}_1(x_t, h_{t-1}^{(1)}) $$
$$ h_t^{(2)} = \text{RNN}_2(h_t^{(1)}, h_{t-1}^{(2)}) $$
$$ h_t^{(L)} = \text{RNN}_L(h_t^{(L-1)}, h_{t-1}^{(L)}) $$

**Trade-offs**:
- **Pros**: Higher representational capacity, better for complex tasks.
- **Cons**: Slower training, more prone to vanishing gradients (mitigated by LayerNorm).

---

## 5. Training Techniques for Sequence Models

### 5.1 Gradient Clipping
RNNs are prone to **exploding gradients**. Clipping prevents this:

$$ \text{if } ||g|| > \theta: \quad g \leftarrow \frac{\theta}{||g||} \cdot g $$

**PyTorch Implementation**:
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

---

### 5.2 Variational Dropout
Standard dropout breaks temporal dependencies in RNNs. **Variational Dropout** applies the *same* dropout mask at each time step.

```python
# WRONG: Different mask at each timestep
for t in range(seq_len):
    h = rnn(x[t] * dropout_mask[t], h)

# CORRECT: Same mask across time
dropout_mask = torch.bernoulli(p * torch.ones_like(x[0]))
for t in range(seq_len):
    h = rnn(x[t] * dropout_mask, h)
```

---

### 5.3 Zoneout
A regularization technique specific to RNNs where some units are forced to **keep their previous values** with probability $p$.

$$ h_t = \text{mask} \odot h_{t-1} + (1 - \text{mask}) \odot \text{RNN}(x_t, h_{t-1}) $$

**Benefit**: Encourages the network to maintain information over time (similar to residual connections).

---

### 5.4 Truncated BPTT
For very long sequences, full BPTT is memory-intensive. **Truncated BPTT**:
1.  Split sequence into chunks of length $k$.
2.  Backpropagate only through $k$ steps.
3.  Carry the hidden state across chunks (detached from the computation graph).

**Trade-off**: Loses dependencies longer than $k$, but enables training on massive sequences.

---

## 6. Implementation Deep Dive

### 6.1 Building an LSTM from Scratch (NumPy)
```python
import numpy as np

class LSTMFromScratch:
    def __init__(self, input_dim, hidden_dim):
        self.hidden_dim = hidden_dim
        
        # Initialize weights (Xavier initialization)
        self.Wf = np.random.randn(hidden_dim, input_dim + hidden_dim) * np.sqrt(2.0 / (input_dim + hidden_dim))
        self.Wi = np.random.randn(hidden_dim, input_dim + hidden_dim) * np.sqrt(2.0 / (input_dim + hidden_dim))
        self.Wc = np.random.randn(hidden_dim, input_dim + hidden_dim) * np.sqrt(2.0 / (input_dim + hidden_dim))
        self.Wo = np.random.randn(hidden_dim, input_dim + hidden_dim) * np.sqrt(2.0 / (input_dim + hidden_dim))
        
        self.bf = np.zeros((hidden_dim, 1))
        self.bi = np.zeros((hidden_dim, 1))
        self.bc = np.zeros((hidden_dim, 1))
        self.bo = np.zeros((hidden_dim, 1))
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def forward_step(self, x_t, h_prev, c_prev):
        # Concatenate input and previous hidden state
        concat = np.vstack([x_t.reshape(-1, 1), h_prev])
        
        # Gates
        f_t = self.sigmoid(self.Wf @ concat + self.bf)
        i_t = self.sigmoid(self.Wi @ concat + self.bi)
        c_tilde = np.tanh(self.Wc @ concat + self.bc)
        o_t = self.sigmoid(self.Wo @ concat + self.bo)
        
        # Cell state and hidden state
        c_next = f_t * c_prev + i_t * c_tilde
        h_next = o_t * np.tanh(c_next)
        
        return h_next, c_next
    
    def forward_sequence(self, sequence):
        """Process a full sequence."""
        h = np.zeros((self.hidden_dim, 1))
        c = np.zeros((self.hidden_dim, 1))
        outputs = []
        
        for x_t in sequence:
            h, c = self.forward_step(x_t, h, c)
            outputs.append(h.copy())
        
        return np.hstack(outputs), h, c

# Example: Process a sequence of 10 time steps
lstm = LSTMFromScratch(input_dim=50, hidden_dim=128)
sequence = [np.random.randn(50) for _ in range(10)]
outputs, h_final, c_final = lstm.forward_sequence(sequence)
print(f"Output shape: {outputs.shape}")  # (128, 10)
```

---

### 6.2 Multi-Layer BiLSTM with PyTorch
```python
import torch
import torch.nn as nn

class DeepBiLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.layer_norm = nn.LayerNorm(hidden_dim * 2)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, 1)
    
    def forward(self, x, lengths=None):
        # x: [batch, seq_len]
        embedded = self.dropout(self.embedding(x))  # [batch, seq_len, embed_dim]
        
        # Pack sequence if lengths provided (for variable-length sequences)
        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            output, (hn, cn) = self.lstm(packed)
            output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        else:
            output, (hn, cn) = self.lstm(embedded)
        
        # Pooling: use last valid output from each direction
        # hn shape: [num_layers * 2, batch, hidden_dim]
        fwd_last = hn[-2, :, :]  # Last forward layer
        rev_last = hn[-1, :, :]  # Last backward layer
        merged = torch.cat([fwd_last, rev_last], dim=1)  # [batch, hidden_dim * 2]
        
        merged = self.layer_norm(merged)
        merged = self.dropout(merged)
        
        return self.fc(merged)

# Usage
model = DeepBiLSTM(vocab_size=30000, embed_dim=300, hidden_dim=256, num_layers=3)
x = torch.randint(0, 30000, (32, 100))  # batch=32, seq_len=100
lengths = torch.randint(50, 100, (32,))  # variable lengths
output = model(x, lengths)
print(f"Output shape: {output.shape}")  # [32, 1]
```

---

## 7. Common Pitfalls and Debugging

### 7.1 The "Exposure Bias" Problem
- **Issue**: During training, the decoder sees ground-truth tokens. During inference, it sees its own (potentially wrong) predictions.
- **Symptom**: Generated sequences degrade in quality over time.
- **Solutions**:
    - **Scheduled Sampling**: Gradually replace ground-truth with model predictions during training.
    - **Beam Search**: Maintain multiple hypotheses during inference.
    - **Reinforcement Learning**: Use rewards (e.g., BLEU score) to train the model end-to-end.

---

### 7.2 Variable-Length Sequence Handling
**Problem**: Padding tokens can corrupt hidden states.

**Solution**: Use **packed sequences** in PyTorch:
```python
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# Sort by length (required for packing)
lengths, perm_idx = lengths.sort(descending=True)
x = x[perm_idx]

# Pack
packed = pack_padded_sequence(x, lengths, batch_first=True)
output, (hn, cn) = lstm(packed)

# Unpack
output, _ = pad_packed_sequence(output, batch_first=True)
```

---

### 7.3 Hidden State Initialization
- **Zero Initialization**: Standard, but can cause "cold start" problems.
- **Learnable Initial State**: Add trainable parameters $h_0, c_0$ for each layer.
    ```python
    self.h0 = nn.Parameter(torch.zeros(num_layers * 2, batch_size, hidden_dim))
    self.c0 = nn.Parameter(torch.zeros(num_layers * 2, batch_size, hidden_dim))
    ```

---

## 8. When to Use RNNs vs. Transformers (2024 Perspective)

| Scenario | Recommended Architecture | Rationale |
| :--- | :--- | :--- |
| **Short sequences (<50 tokens)** | LSTM/GRU | Lower latency, simpler deployment |
| **Long sequences (>500 tokens)** | Transformer | Parallel training, better long-range modeling |
| **Streaming/Online inference** | RNN with state caching | Constant memory, no need to recompute history |
| **Time-series forecasting** | ConvLSTM or Transformer | Depends on spatial vs. temporal emphasis |
| **Edge devices (low power)** | Quantized LSTM | Better INT8 support, lower memory footprint |
| **Multi-modal sequences** | Transformer | Easier to add cross-attention layers |

---

## 🔬 Research Frontiers (2024-2025)

### 9.1 State Space Models (SSMs)
**Mamba** and related architectures challenge Transformers with $O(n)$ complexity while maintaining long-range modeling.

$$ h_t = \bar{A} h_{t-1} + \bar{B} x_t $$
$$ y_t = \bar{C} h_t $$

**Key Innovation**: Data-dependent parameters ($\bar{A}, \bar{B}, \bar{C}$ depend on $x_t$).

---

### 9.2 Linear Attention
Approximating self-attention with linear complexity:
- **Performer**: Uses random feature maps to approximate softmax attention.
- **Linformer**: Projects keys/values to lower-dimensional space.

---

### 9.3 Neural ODEs for Sequences
Treating RNN hidden states as continuous dynamical systems:
$$ \frac{dh(t)}{dt} = f(h(t), x(t), \theta) $$

**Benefit**: Adaptive computation time, memory-efficient backpropagation.

---

**Status:** ✅ Elite Expanded Standard (12/10)
**Next:** The Transformer Revolution (Self-Attention, Multi-Head, Positional Encoding)
