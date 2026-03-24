# 8.2 Optimization Algorithms

## 🎯 Quick Overview
- **Gradient Descent**: Foundation of optimization
- **Momentum**: Accelerate convergence
- **Adam**: Adaptive learning rates
- **Learning Rate Scheduling**: Dynamic adjustment
- **Foundation for**: Efficient neural network training

---

#### 🧒 ELI5: Gradient Descent, Backpropagation & Optimizers

> Imagine you're on a foggy mountain and need to get to the bottom.
>
> **Gradient Descent** (Finding the bottom):
> - You can't see the bottom (minimum loss)
> - But you can feel which way is DOWN (gradient)
> - Take a step downhill
> - Repeat until you reach flat ground (minimum)
>
> **Learning Rate** (Step size):
> - Too big: You might overshoot and bounce around!
> - Too small: Takes FOREVER to get down!
> - Just right: Smooth descent to bottom
>
> **Batch vs SGD vs Mini-Batch**:
>
> **Batch GD** (Measure ALL stairs before stepping):
> - Check EVERY training example
> - Calculate EXACT direction downhill
> - Take ONE precise step
> - Slow but steady
> - Like: Carefully planning each move
>
> **SGD** (Feel one stair, step immediately):
> - Check ONE training example
> - Step in that direction
> - Check NEXT example, step again
> - Fast but zig-zags a lot!
> - Like: Running down quickly but clumsily
>
> **Mini-Batch** (Check a few stairs, then step):
> - Check 32 examples (batch)
> - Calculate average direction
> - Take step
> - Best of both worlds!
> - Like: Quick but not reckless
>
> **Backpropagation** (How the network learns):
>
> Imagine a tower of 10 people passing buckets:
> - Top person (output) sees: "We're 5 gallons short!"
> - Tells person below: "You're responsible for 2 gallons"
> - That person tells person below them: "You're responsible for 1 gallon"
> - Continues down to bottom (input layer)
> - Everyone adjusts how much they carry!
>
> **Chain Rule**: Each person calculates their share of the error
> **Backward pass**: Error info flows from top to bottom
>
> **Momentum** (Rolling ball):
> - Normal GD: Only looks at current slope
> - Momentum: Remembers which way it was rolling
> - Like a ball rolling downhill - builds up speed!
> - Helps roll through small bumps (local minima)
>
> **Adam Optimizer** (Smart step adjustment):
> - Some stairs are steep, some are gentle
> - Adam: "Steep stairs → small steps, gentle stairs → bigger steps"
> - Adapts learning rate for EACH weight!
> - Like: Careful on steep parts, fast on flat parts
>
> **Why so many optimizers?**:
> - Different problems need different strategies
> - Adam: Good default choice (works for most things)
> - SGD with Momentum: Sometimes better for final accuracy
> - It's like choosing between walking, running, or biking downhill!

</details>

---

## 1. Gradient Descent Variants

### Batch Gradient Descent

```python
def batch_gradient_descent(X, y, learning_rate=0.01, epochs=1000):
    """Batch GD - uses entire dataset"""
    m, n = X.shape
    weights = np.random.randn(n)
    bias = 0
    losses = []
    
    for epoch in range(epochs):
        # Forward pass
        predictions = np.dot(X, weights) + bias
        
        # Compute loss (MSE)
        loss = np.mean((predictions - y) ** 2)
        losses.append(loss)
        
        # Backward pass (gradients)
        dw = (2/m) * np.dot(X.T, (predictions - y))
        db = (2/m) * np.sum(predictions - y)
        
        # Update weights
        weights -= learning_rate * dw
        bias -= learning_rate * db
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.4f}")
    
    return weights, bias, losses
```

### Stochastic Gradient Descent (SGD)

```python
def sgd(X, y, learning_rate=0.01, epochs=1000):
    """SGD - one sample at a time"""
    m, n = X.shape
    weights = np.random.randn(n)
    bias = 0
    losses = []
    
    for epoch in range(epochs):
        # Shuffle data
        indices = np.random.permutation(m)
        X_shuffled = X[indices]
        y_shuffled = y[indices]
        
        epoch_loss = 0
        
        for i in range(m):
            xi = X_shuffled[i:i+1]
            yi = y_shuffled[i]
            
            # Forward pass
            prediction = np.dot(xi, weights) + bias
            
            # Compute loss
            loss = (prediction - yi) ** 2
            epoch_loss += loss
            
            # Backward pass
            dw = 2 * np.dot(xi.T, (prediction - yi))
            db = 2 * (prediction - yi)
            
            # Update weights
            weights -= learning_rate * dw
            bias -= learning_rate * db
        
        losses.append(epoch_loss / m)
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {losses[-1]:.4f}")
    
    return weights, bias, losses
```

### Mini-Batch Gradient Descent

```python
def mini_batch_gd(X, y, learning_rate=0.01, epochs=1000, batch_size=32):
    """Mini-batch GD - compromise between batch and SGD"""
    m, n = X.shape
    weights = np.random.randn(n)
    bias = 0
    losses = []
    
    for epoch in range(epochs):
        # Shuffle data
        indices = np.random.permutation(m)
        X_shuffled = X[indices]
        y_shuffled = y[indices]
        
        epoch_loss = 0
        
        for i in range(0, m, batch_size):
            xi = X_shuffled[i:i+batch_size]
            yi = y_shuffled[i:i+batch_size]
            mb = len(xi)
            
            # Forward pass
            predictions = np.dot(xi, weights) + bias
            
            # Compute loss
            loss = np.mean((predictions - yi) ** 2)
            epoch_loss += loss * mb
            
            # Backward pass
            dw = (2/mb) * np.dot(xi.T, (predictions - yi))
            db = (2/mb) * np.sum(predictions - yi)
            
            # Update weights
            weights -= learning_rate * dw
            bias -= learning_rate * db
        
        losses.append(epoch_loss / m)
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {losses[-1]:.4f}")
    
    return weights, bias, losses
```

---

## 2. Momentum-Based Methods

### Momentum

```python
def momentum_gd(X, y, learning_rate=0.01, epochs=1000, momentum=0.9):
    """Gradient Descent with Momentum"""
    m, n = X.shape
    weights = np.random.randn(n)
    bias = 0
    
    # Velocity terms
    v_dw = np.zeros(n)
    v_db = 0
    
    losses = []
    
    for epoch in range(epochs):
        # Forward pass
        predictions = np.dot(X, weights) + bias
        loss = np.mean((predictions - y) ** 2)
        losses.append(loss)
        
        # Backward pass
        dw = (2/m) * np.dot(X.T, (predictions - y))
        db = (2/m) * np.sum(predictions - y)
        
        # Update velocity
        v_dw = momentum * v_dw + (1 - momentum) * dw
        v_db = momentum * v_db + (1 - momentum) * db
        
        # Update weights
        weights -= learning_rate * v_dw
        bias -= learning_rate * v_db
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.4f}")
    
    return weights, bias, losses
```

### Nesterov Accelerated Gradient (NAG)

```python
def nesterov_gd(X, y, learning_rate=0.01, epochs=1000, momentum=0.9):
    """Nesterov Accelerated Gradient"""
    m, n = X.shape
    weights = np.random.randn(n)
    bias = 0
    
    v_dw = np.zeros(n)
    v_db = 0
    
    losses = []
    
    for epoch in range(epochs):
        # Look ahead position
        weights_ahead = weights + momentum * v_dw
        bias_ahead = bias + momentum * v_db
        
        # Forward pass at look ahead position
        predictions = np.dot(X, weights_ahead) + bias_ahead
        
        # Backward pass
        dw = (2/m) * np.dot(X.T, (predictions - y))
        db = (2/m) * np.sum(predictions - y)
        
        # Update velocity
        v_dw = momentum * v_dw + (1 - momentum) * dw
        v_db = momentum * v_db + (1 - momentum) * db
        
        # Update weights
        weights -= learning_rate * v_dw
        bias -= learning_rate * v_db
        
        loss = np.mean((np.dot(X, weights) + bias - y) ** 2)
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.4f}")
    
    return weights, bias, losses
```

---

## 3. Adaptive Learning Rate Methods

### AdaGrad

```python
def adagrad(X, y, learning_rate=0.01, epochs=1000, epsilon=1e-8):
    """AdaGrad - adaptive learning rates"""
    m, n = X.shape
    weights = np.random.randn(n)
    bias = 0
    
    # Accumulate squared gradients
    G_dw = np.zeros(n)
    G_db = 0
    
    losses = []
    
    for epoch in range(epochs):
        predictions = np.dot(X, weights) + bias
        loss = np.mean((predictions - y) ** 2)
        losses.append(loss)
        
        dw = (2/m) * np.dot(X.T, (predictions - y))
        db = (2/m) * np.sum(predictions - y)
        
        # Accumulate squared gradients
        G_dw += dw ** 2
        G_db += db ** 2
        
        # Update weights with adaptive learning rate
        weights -= learning_rate * dw / (np.sqrt(G_dw) + epsilon)
        bias -= learning_rate * db / (np.sqrt(G_db) + epsilon)
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.4f}")
    
    return weights, bias, losses
```

### RMSprop

```python
def rmsprop(X, y, learning_rate=0.01, epochs=1000, decay=0.9, epsilon=1e-8):
    """RMSprop - fixes AdaGrad's learning rate decay"""
    m, n = X.shape
    weights = np.random.randn(n)
    bias = 0
    
    E_dw = np.zeros(n)
    E_db = 0
    
    losses = []
    
    for epoch in range(epochs):
        predictions = np.dot(X, weights) + bias
        loss = np.mean((predictions - y) ** 2)
        losses.append(loss)
        
        dw = (2/m) * np.dot(X.T, (predictions - y))
        db = (2/m) * np.sum(predictions - y)
        
        # Update running average of squared gradients
        E_dw = decay * E_dw + (1 - decay) * dw ** 2
        E_db = decay * E_db + (1 - decay) * db ** 2
        
        # Update weights
        weights -= learning_rate * dw / (np.sqrt(E_dw) + epsilon)
        bias -= learning_rate * db / (np.sqrt(E_db) + epsilon)
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.4f}")
    
    return weights, bias, losses
```

### Adam

```python
def adam(X, y, learning_rate=0.001, epochs=1000, beta1=0.9, beta2=0.999, epsilon=1e-8):
    """Adam - Adaptive Moment Estimation"""
    m, n = X.shape
    weights = np.random.randn(n)
    bias = 0
    
    m_dw = np.zeros(n)
    m_db = 0
    v_dw = np.zeros(n)
    v_db = 0
    
    losses = []
    
    for epoch in range(epochs):
        predictions = np.dot(X, weights) + bias
        loss = np.mean((predictions - y) ** 2)
        losses.append(loss)
        
        dw = (2/m) * np.dot(X.T, (predictions - y))
        db = (2/m) * np.sum(predictions - y)
        
        # Update biased first moment estimate
        m_dw = beta1 * m_dw + (1 - beta1) * dw
        m_db = beta1 * m_db + (1 - beta1) * db
        
        # Update biased second raw moment estimate
        v_dw = beta2 * v_dw + (1 - beta2) * (dw ** 2)
        v_db = beta2 * v_db + (1 - beta2) * (db ** 2)
        
        # Compute bias-corrected first moment estimate
        m_dw_hat = m_dw / (1 - beta1 ** (epoch + 1))
        m_db_hat = m_db / (1 - beta1 ** (epoch + 1))
        
        # Compute bias-corrected second raw moment estimate
        v_dw_hat = v_dw / (1 - beta2 ** (epoch + 1))
        v_db_hat = v_db / (1 - beta2 ** (epoch + 1))
        
        # Update weights
        weights -= learning_rate * m_dw_hat / (np.sqrt(v_dw_hat) + epsilon)
        bias -= learning_rate * m_db_hat / (np.sqrt(v_db_hat) + epsilon)
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.4f}")
    
    return weights, bias, losses
```

### AdamW

```python
def adamw(X, y, learning_rate=0.001, weight_decay=0.01, epochs=1000, 
          beta1=0.9, beta2=0.999, epsilon=1e-8):
    """AdamW - Adam with decoupled weight decay"""
    m, n = X.shape
    weights = np.random.randn(n)
    bias = 0
    
    m_dw = np.zeros(n)
    v_dw = np.zeros(n)
    
    losses = []
    
    for epoch in range(epochs):
        predictions = np.dot(X, weights) + bias
        loss = np.mean((predictions - y) ** 2)
        losses.append(loss)
        
        dw = (2/m) * np.dot(X.T, (predictions - y)) + weight_decay * weights
        db = (2/m) * np.sum(predictions - y)
        
        m_dw = beta1 * m_dw + (1 - beta1) * dw
        v_dw = beta2 * v_dw + (1 - beta2) * (dw ** 2)
        
        m_dw_hat = m_dw / (1 - beta1 ** (epoch + 1))
        v_dw_hat = v_dw / (1 - beta2 ** (epoch + 1))
        
        weights -= learning_rate * m_dw_hat / (np.sqrt(v_dw_hat) + epsilon)
        bias -= learning_rate * db
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.4f}")
    
    return weights, bias, losses
```

---

## 4. Learning Rate Scheduling

### Step Decay

```python
def step_decay(epoch, initial_lr=0.01, drop=0.5, epochs_drop=10):
    """Step decay learning rate"""
    return initial_lr * (drop ** (epoch // epochs_drop))
```

### Exponential Decay

```python
def exponential_decay(epoch, initial_lr=0.01, decay_rate=0.96):
    """Exponential decay"""
    return initial_lr * (decay_rate ** epoch)
```

### Cosine Annealing

```python
def cosine_annealing(epoch, initial_lr=0.01, min_lr=0.001, total_epochs=100):
    """Cosine annealing with warm restarts"""
    return min_lr + 0.5 * (initial_lr - min_lr) * (1 + np.cos(epoch / total_epochs * np.pi))
```

### Warm Restarts

```python
def warm_restarts(epoch, initial_lr=0.01, min_lr=0.001, restart_period=10):
    """Cosine annealing with warm restarts"""
    epoch_in_cycle = epoch % restart_period
    return min_lr + 0.5 * (initial_lr - min_lr) * (1 + np.cos(epoch_in_cycle / restart_period * np.pi))
```

---

## 💻 Python Code Examples

```python
# === Compare Optimization Algorithms ===

from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Generate data
X, y = make_regression(n_samples=1000, n_features=20, noise=10, random_state=42)

# Split and scale
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Compare optimizers
optimizers = {
    'Batch GD': lambda: batch_gradient_descent(X_train_scaled, y_train, learning_rate=0.1, epochs=500),
    'SGD': lambda: sgd(X_train_scaled, y_train, learning_rate=0.01, epochs=500),
    'Momentum': lambda: momentum_gd(X_train_scaled, y_train, learning_rate=0.1, epochs=500),
    'Adam': lambda: adam(X_train_scaled, y_train, learning_rate=0.01, epochs=500),
    'AdamW': lambda: adamw(X_train_scaled, y_train, learning_rate=0.01, epochs=500)
}

plt.figure(figsize=(15, 10))

for name, optimizer_fn in optimizers.items():
    weights, bias, losses = optimizer_fn()
    plt.plot(losses, label=name)

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Optimizer Comparison')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# === Learning Rate Scheduling ===

epochs = 100
lrs = {
    'Constant': [0.01] * epochs,
    'Step Decay': [step_decay(i) for i in range(epochs)],
    'Exponential': [exponential_decay(i) for i in range(epochs)],
    'Cosine Annealing': [cosine_annealing(i, total_epochs=epochs) for i in range(epochs)],
    'Warm Restarts': [warm_restarts(i) for i in range(epochs)]
}

plt.figure(figsize=(15, 8))

for name, lr_values in lrs.items():
    plt.plot(lr_values, label=name)

plt.xlabel('Epoch')
plt.ylabel('Learning Rate')
plt.title('Learning Rate Schedules')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

---

## 📊 Summary Tables

### Optimizer Comparison

| Optimizer | Pros | Cons | Use Case |
|-----------|------|------|----------|
| Batch GD | Stable | Slow | Small datasets |
| SGD | Fast | Noisy | Large datasets |
| Momentum | Faster convergence | May overshoot | General purpose |
| AdaGrad | Adaptive | LR decays too fast | Sparse data |
| RMSprop | Fixes AdaGrad | Another hyperparameter | RNNs |
| Adam | Best of all | May not generalize | Default choice |
| AdamW | Better generalization | Weight decay tuning | Transformers |

### Learning Rate Schedules

| Schedule | Formula | Use Case |
|----------|---------|----------|
| Constant | lr | Simple problems |
| Step Decay | lr × drop^(epoch//n) | Standard training |
| Exponential | lr × decay^epoch | Smooth decay |
| Cosine | lr_min + 0.5(lr_max-lr_min)(1+cos) | Modern training |
| Warm Restarts | Cosine with restarts | Avoid local minima |

---

## 🎯 ML Applications

| Optimizer | ML Application |
|-----------|----------------|
| SGD | Large-scale training |
| Adam | Default for most models |
| AdamW | Transformers, BERT |
| RMSprop | RNNs, LSTMs |
| Learning Rate Scheduling | Fine-tuning |

---

## ❓ Quick Check Questions

1. What is the fundamental trade-off between Batch Gradient Descent and Stochastic Gradient Descent (SGD)?
2. How does the "Momentum" term help in optimizing a neural network?
3. What is the primary problem with AdaGrad that RMSprop and Adam attempt to solve?
4. Explain the concept of "Bias Correction" in the Adam optimizer.
5. Why is "Learning Rate Scheduling" (like Cosine Annealing) often used in modern deep learning training?

---

## 📝 Answers to Quick Check

1. **Batch GD** uses the entire dataset to compute gradients, making updates stable and accurate but very slow and memory-intensive for large data. **SGD** updates weights using only one sample at a time, making it very fast and capable of escaping local minima due to its noisy updates, but it never truly "settles" at the minimum.
2. **Momentum** accumulates a "velocity" vector of past gradients. This helps accelerate the optimizer in directions where the gradient is consistent and dampens oscillations in directions where the gradient changes frequently (like in ravines), leading to faster convergence.
3. **AdaGrad** accumulates all past squared gradients in the denominator, which causes the effective learning rate to decrease monotonically and eventually become so small that the model stops learning. **RMSprop** and **Adam** use an exponentially decaying average of squared gradients to ensure the learning rate stays effective.
4. In **Adam**, the first and second moment estimates ($m_t$ and $v_t$) are initialized to zero. This biases them towards zero, especially during the early steps of training. **Bias correction** divides these estimates by $(1 - \beta^t)$ to compensate for this initialization bias.
5. **Learning Rate Scheduling** allows the model to start with a high learning rate to quickly explore the loss landscape and escape local minima, then gradually decrease it to "fine-tune" the weights and settle accurately into a deep minimum as training progresses.

---

**Status:** ✅ Complete
**Next:** Regularization Techniques
