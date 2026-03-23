# 8.3-8.6 Deep Learning Fundamentals

## 🎯 Quick Overview
- **Regularization**: Prevent overfitting
- **Training**: Weight initialization, debugging
- **CNNs**: Image processing
- **RNNs**: Sequence modeling
- **Foundation for**: Modern deep learning architectures

---

## 1. Regularization Techniques

### L1 and L2 Regularization

```python
import numpy as np

def l1_regularization(X, y, learning_rate=0.01, epochs=1000, lambda_l1=0.01):
    """L1 Regularization (Lasso)"""
    m, n = X.shape
    weights = np.random.randn(n)
    bias = 0
    
    for epoch in range(epochs):
        predictions = np.dot(X, weights) + bias
        loss = np.mean((predictions - y) ** 2)
        
        dw = (2/m) * np.dot(X.T, (predictions - y)) + lambda_l1 * np.sign(weights)
        db = (2/m) * np.sum(predictions - y)
        
        weights -= learning_rate * dw
        bias -= learning_rate * db
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.4f}, L1 penalty: {lambda_l1 * np.sum(np.abs(weights)):.4f}")
    
    return weights, bias

def l2_regularization(X, y, learning_rate=0.01, epochs=1000, lambda_l2=0.01):
    """L2 Regularization (Ridge)"""
    m, n = X.shape
    weights = np.random.randn(n)
    bias = 0
    
    for epoch in range(epochs):
        predictions = np.dot(X, weights) + bias
        loss = np.mean((predictions - y) ** 2)
        
        dw = (2/m) * np.dot(X.T, (predictions - y)) + lambda_l2 * weights
        db = (2/m) * np.sum(predictions - y)
        
        weights -= learning_rate * dw
        bias -= learning_rate * db
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.4f}, L2 penalty: {0.5 * lambda_l2 * np.sum(weights**2):.4f}")
    
    return weights, bias

def elastic_net(X, y, learning_rate=0.01, epochs=1000, lambda_l1=0.01, lambda_l2=0.01):
    """Elastic Net (L1 + L2)"""
    m, n = X.shape
    weights = np.random.randn(n)
    bias = 0
    
    for epoch in range(epochs):
        predictions = np.dot(X, weights) + bias
        loss = np.mean((predictions - y) ** 2)
        
        dw = (2/m) * np.dot(X.T, (predictions - y)) + lambda_l1 * np.sign(weights) + lambda_l2 * weights
        db = (2/m) * np.sum(predictions - y)
        
        weights -= learning_rate * dw
        bias -= learning_rate * db
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.4f}")
    
    return weights, bias
```

### Dropout

```python
class DropoutLayer:
    def __init__(self, dropout_rate=0.5):
        self.dropout_rate = dropout_rate
        self.mask = None
    
    def forward(self, X, training=True):
        if training:
            self.mask = (np.random.rand(*X.shape) > self.dropout_rate).astype(float)
            self.mask /= (1 - self.dropout_rate)  # Inverted dropout
            return X * self.mask
        else:
            return X
    
    def backward(self, dout):
        return dout * self.mask

# Example usage
class MLPWithDropout:
    def __init__(self, layer_sizes, dropout_rate=0.5):
        self.weights = []
        self.biases = []
        self.dropout_layers = []
        
        for i in range(len(layer_sizes) - 1):
            w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2/layer_sizes[i])
            b = np.zeros((1, layer_sizes[i+1]))
            self.weights.append(w)
            self.biases.append(b)
            if i < len(layer_sizes) - 2:  # No dropout on output layer
                self.dropout_layers.append(DropoutLayer(dropout_rate))
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        return (x > 0).astype(float)
    
    def forward(self, X, training=True):
        self.activations = [X]
        self.z_values = []
        
        current = X
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            z = np.dot(current, w) + b
            self.z_values.append(z)
            
            if i < len(self.weights) - 1:  # Hidden layers
                current = self.relu(z)
                if i < len(self.dropout_layers):
                    current = self.dropout_layers[i].forward(current, training)
            else:
                current = z  # Output layer
        
        self.activations.append(current)
        return current
    
    def backward(self, X, y, learning_rate=0.01):
        m = X.shape[0]
        output = self.activations[-1]
        delta = 2 * (output - y.reshape(-1, 1)) / m
        
        for i in range(len(self.weights) - 1, -1, -1):
            dw = np.dot(self.activations[i].T, delta)
            db = np.sum(delta, axis=0, keepdims=True)
            
            self.weights[i] -= learning_rate * dw
            self.biases[i] -= learning_rate * db
            
            if i > 0:
                delta = np.dot(delta, self.weights[i].T) * self.relu_derivative(self.z_values[i-1])
                if i - 1 < len(self.dropout_layers):
                    delta = self.dropout_layers[i-1].backward(delta)
```

### Batch Normalization

```python
class BatchNormalization:
    def __init__(self, num_features, epsilon=1e-5, momentum=0.1):
        self.gamma = np.ones((1, num_features))
        self.beta = np.zeros((1, num_features))
        self.epsilon = epsilon
        self.momentum = momentum
        
        # Running statistics for inference
        self.running_mean = np.zeros((1, num_features))
        self.running_var = np.ones((1, num_features))
        
        # Cache for backward pass
        self.cache = None
    
    def forward(self, X, training=True):
        if training:
            mean = np.mean(X, axis=0)
            var = np.var(X, axis=0)
            
            # Update running statistics
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
            
            # Normalize
            x_norm = (X - mean) / np.sqrt(var + self.epsilon)
            
            # Scale and shift
            out = self.gamma * x_norm + self.beta
            
            # Cache for backward
            self.cache = (X, x_norm, mean, var)
        else:
            x_norm = (X - self.running_mean) / np.sqrt(self.running_var + self.epsilon)
            out = self.gamma * x_norm + self.beta
        
        return out
    
    def backward(self, dout, learning_rate=0.01):
        X, x_norm, mean, var = self.cache
        m = X.shape[0]
        
        # Gradients for gamma and beta
        dgamma = np.sum(dout * x_norm, axis=0)
        dbeta = np.sum(dout, axis=0)
        
        # Gradient for normalized input
        dx_norm = dout * self.gamma
        
        # Gradient for variance
        dvar = np.sum(dx_norm * (X - mean) * -0.5 * (var + self.epsilon)**(-1.5), axis=0)
        
        # Gradient for mean
        dmean = np.sum(dx_norm * -1 / np.sqrt(var + self.epsilon), axis=0) + \
                dvar * np.mean(-2 * (X - mean), axis=0)
        
        # Gradient for input
        dX = dx_norm / np.sqrt(var + self.epsilon) + \
             dvar * 2 * (X - mean) / m + \
             dmean / m
        
        # Update gamma and beta
        self.gamma -= learning_rate * dgamma
        self.beta -= learning_rate * dbeta
        
        return dX
```

---

## 2. Training Deep Networks

### Weight Initialization

```python
def xavier_initialization(input_size, output_size):
    """Xavier/Glorot initialization"""
    limit = np.sqrt(6 / (input_size + output_size))
    return np.random.uniform(-limit, limit, (input_size, output_size))

def he_initialization(input_size, output_size):
    """He initialization (for ReLU)"""
    std = np.sqrt(2 / input_size)
    return np.random.normal(0, std, (input_size, output_size))

def lecun_initialization(input_size, output_size):
    """LeCun initialization"""
    std = np.sqrt(1 / input_size)
    return np.random.normal(0, std, (input_size, output_size))

def compare_initializations(layer_sizes, X, y, epochs=500):
    """Compare different initialization methods"""
    initializations = {
        'Xavier': xavier_initialization,
        'He': he_initialization,
        'LeCun': lecun_initialization,
        'Random': lambda i, o: np.random.randn(i, o)
    }
    
    losses_dict = {}
    
    for name, init_fn in initializations.items():
        weights = [init_fn(layer_sizes[i], layer_sizes[i+1]) for i in range(len(layer_sizes)-1)]
        biases = [np.zeros((1, layer_sizes[i+1])) for i in range(len(layer_sizes)-1)]
        
        losses = []
        for epoch in range(epochs):
            # Forward pass
            activations = [X]
            current = X
            for w, b in zip(weights, biases):
                z = np.dot(current, w) + b
                current = np.maximum(0, z)  # ReLU
            
            # Loss
            loss = np.mean((current - y.reshape(-1, 1)) ** 2)
            losses.append(loss)
        
        losses_dict[name] = losses
    
    return losses_dict
```

### Vanishing/Exploding Gradients

```python
def check_gradient_flow(layer_sizes, X, y):
    """Check for vanishing/exploding gradients"""
    weights = [he_initialization(layer_sizes[i], layer_sizes[i+1]) for i in range(len(layer_sizes)-1)]
    biases = [np.zeros((1, layer_sizes[i+1])) for i in range(len(layer_sizes)-1)]
    
    # Forward pass
    activations = [X]
    z_values = []
    current = X
    for w, b in zip(weights, biases):
        z = np.dot(current, w) + b
        z_values.append(z)
        current = np.maximum(0, z)  # ReLU
    
    # Backward pass
    output = activations[-1]
    delta = 2 * (output - y.reshape(-1, 1))
    
    gradient_norms = []
    for i in range(len(weights) - 1, -1, -1):
        dw = np.dot(activations[i].T, delta)
        gradient_norms.append(np.linalg.norm(dw))
        
        if i > 0:
            delta = np.dot(delta, weights[i].T) * (z_values[i-1] > 0).astype(float)
    
    gradient_norms.reverse()
    
    print("Gradient norms by layer:")
    for i, norm in enumerate(gradient_norms):
        print(f"Layer {i}: {norm:.6f}")
    
    return gradient_norms
```

---

## 3. Convolutional Neural Networks (CNNs)

### Convolution Operation

```python
def convolve_single(image, kernel):
    """2D convolution for single channel"""
    kh, kw = kernel.shape
    ih, iw = image.shape
    
    oh, ow = ih - kh + 1, iw - kw + 1
    output = np.zeros((oh, ow))
    
    for i in range(oh):
        for j in range(ow):
            output[i, j] = np.sum(image[i:i+kh, j:j+kw] * kernel)
    
    return output

def conv2d(image, kernel, padding='valid', stride=1):
    """2D convolution with padding and stride"""
    if padding == 'same':
        pad_h = kernel.shape[0] // 2
        pad_w = kernel.shape[1] // 2
        image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant')
    
    kh, kw = kernel.shape
    ih, iw = image.shape
    
    oh = (ih - kh) // stride + 1
    ow = (iw - kw) // stride + 1
    
    output = np.zeros((oh, ow))
    
    for i in range(oh):
        for j in range(ow):
            i_start = i * stride
            j_start = j * stride
            output[i, j] = np.sum(image[i_start:i_start+kh, j_start:j_start+kw] * kernel)
    
    return output

# Example kernels
kernels = {
    'Identity': np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]]),
    'Edge Detection': np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]),
    'Sharpen': np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]),
    'Blur': np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]) / 9,
    'Sobel X': np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]),
    'Sobel Y': np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
}
```

### CNN Architecture

```python
class SimpleCNN:
    def __init__(self, input_shape=(28, 28, 1), num_classes=10):
        # Conv layer 1
        self.conv1_filters = np.random.randn(5, 5, input_shape[2], 32) * 0.1
        self.conv1_bias = np.zeros(32)
        
        # Conv layer 2
        self.conv2_filters = np.random.randn(5, 5, 32, 64) * 0.1
        self.conv2_bias = np.zeros(64)
        
        # Fully connected
        self.fc1_weights = np.random.randn(7 * 7 * 64, 128) * np.sqrt(2 / (7 * 7 * 64))
        self.fc1_bias = np.zeros(128)
        self.fc2_weights = np.random.randn(128, num_classes) * np.sqrt(2 / 128)
        self.fc2_bias = np.zeros(num_classes)
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        return (x > 0).astype(float)
    
    def max_pool(self, X, pool_size=2, stride=2):
        n, h, w, c = X.shape
        oh = (h - pool_size) // stride + 1
        ow = (w - pool_size) // stride + 1
        
        output = np.zeros((n, oh, ow, c))
        
        for i in range(n):
            for j in range(oh):
                for k in range(ow):
                    output[i, j, k, :] = np.max(
                        X[i, j*stride:j*stride+pool_size, k*stride:k*stride+pool_size, :],
                        axis=(0, 1)
                    )
        
        return output
    
    def conv2d(self, X, filters, biases, stride=1, padding='valid'):
        n, h, w, c_in = X.shape
        fh, fw, c_in, c_out = filters.shape
        
        if padding == 'same':
            pad = fh // 2
            X = np.pad(X, ((0, 0), (pad, pad), (pad, pad), (0, 0)), mode='constant')
        
        oh = (h - fh) // stride + 1
        ow = (w - fw) // stride + 1
        
        output = np.zeros((n, oh, ow, c_out))
        
        for i in range(n):
            for j in range(oh):
                for k in range(ow):
                    for l in range(c_out):
                        output[i, j, k, l] = np.sum(
                            X[i, j*stride:j*stride+fh, k*stride:k*stride+fw, :] * filters[:, :, :, l]
                        ) + biases[l]
        
        return output
    
    def forward(self, X):
        # Conv1
        self.conv1_input = X
        self.conv1_out = self.conv2d(X, self.conv1_filters, self.conv1_bias)
        self.conv1_out = self.relu(self.conv1_out)
        self.pool1_out = self.max_pool(self.conv1_out)
        
        # Conv2
        self.conv2_input = self.pool1_out
        self.conv2_out = self.conv2d(self.pool1_out, self.conv2_filters, self.conv2_bias)
        self.conv2_out = self.relu(self.conv2_out)
        self.pool2_out = self.max_pool(self.conv2_out)
        
        # Flatten
        n = X.shape[0]
        self.flatten_out = self.pool2_out.reshape(n, -1)
        
        # FC1
        self.fc1_z = np.dot(self.flatten_out, self.fc1_weights) + self.fc1_bias
        self.fc1_out = self.relu(self.fc1_z)
        
        # FC2 (output)
        self.output = np.dot(self.fc1_out, self.fc2_weights) + self.fc2_bias
        
        return self.output
    
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def predict(self, X):
        output = self.forward(X)
        probs = self.softmax(output)
        return np.argmax(probs, axis=1)
```

---

## 4. Recurrent Neural Networks (RNNs)

### Vanilla RNN

```python
class VanillaRNN:
    def __init__(self, input_size, hidden_size, output_size):
        self.hidden_size = hidden_size
        
        # Weights
        self.Wxh = np.random.randn(input_size, hidden_size) * 0.01
        self.Whh = np.random.randn(hidden_size, hidden_size) * 0.01
        self.Why = np.random.randn(hidden_size, output_size) * 0.01
        
        # Biases
        self.bh = np.zeros((1, hidden_size))
        self.by = np.zeros((1, output_size))
        
        # Cache for BPTT
        self.cache = {}
    
    def tanh(self, x):
        return np.tanh(x)
    
    def tanh_derivative(self, x):
        return 1 - np.tanh(x) ** 2
    
    def forward(self, X, h_prev=None):
        """
        X: (seq_len, batch_size, input_size)
        h_prev: (batch_size, hidden_size)
        """
        seq_len, batch_size, _ = X.shape
        
        if h_prev is None:
            h_prev = np.zeros((batch_size, self.hidden_size))
        
        self.cache['X'] = X
        self.cache['h_prev'] = h_prev
        
        h = h_prev
        outputs = []
        hidden_states = [h]
        
        for t in range(seq_len):
            x_t = X[t]
            h = self.tanh(np.dot(x_t, self.Wxh) + np.dot(h, self.Whh) + self.bh)
            y_t = np.dot(h, self.Why) + self.by
            
            outputs.append(y_t)
            hidden_states.append(h)
        
        self.cache['outputs'] = outputs
        self.cache['hidden_states'] = hidden_states
        
        return np.array(outputs), h
    
    def backward(self, y_true, learning_rate=0.01):
        """Backpropagation Through Time (BPTT)"""
        X = self.cache['X']
        h_prev = self.cache['h_prev']
        outputs = self.cache['outputs']
        hidden_states = self.cache['hidden_states']
        
        seq_len, batch_size, _ = X.shape
        
        # Initialize gradients
        dWxh = np.zeros_like(self.Wxh)
        dWhh = np.zeros_like(self.Whh)
        dWhy = np.zeros_like(self.Why)
        dbh = np.zeros_like(self.bh)
        dby = np.zeros_like(self.by)
        
        dh_next = np.zeros_like(h_prev)
        
        for t in reversed(range(seq_len)):
            y_t = outputs[t]
            h_t = hidden_states[t + 1]
            h_prev_t = hidden_states[t]
            x_t = X[t]
            
            # Output layer gradient (assuming MSE loss)
            dy = 2 * (y_t - y_true[t].reshape(-1, 1)) / batch_size
            
            dWhy += np.dot(h_t.T, dy)
            dby += np.sum(dy, axis=0, keepdims=True)
            
            # Hidden layer gradient
            dh = np.dot(dy, self.Why.T) + dh_next
            dh_raw = dh * self.tanh_derivative(np.dot(x_t, self.Wxh) + np.dot(h_prev_t, self.Whh) + self.bh)
            
            dWxh += np.dot(x_t.T, dh_raw)
            dWhh += np.dot(h_prev_t.T, dh_raw)
            dbh += np.sum(dh_raw, axis=0, keepdims=True)
            
            dh_next = np.dot(dh_raw, self.Whh.T)
        
        # Gradient clipping
        for grad in [dWxh, dWhh, dWhy, dbh, dby]:
            np.clip(grad, -1, 1, out=grad)
        
        # Update weights
        self.Wxh -= learning_rate * dWxh
        self.Whh -= learning_rate * dWhh
        self.Why -= learning_rate * dWhy
        self.bh -= learning_rate * dbh
        self.by -= learning_rate * dby
```

### LSTM

```python
class LSTM:
    def __init__(self, input_size, hidden_size, output_size):
        self.hidden_size = hidden_size
        
        # Forget gate
        self.Wxf = np.random.randn(input_size, hidden_size) * 0.01
        self.Whf = np.random.randn(hidden_size, hidden_size) * 0.01
        self.bf = np.zeros((1, hidden_size))
        
        # Input gate
        self.Wxi = np.random.randn(input_size, hidden_size) * 0.01
        self.Whi = np.random.randn(hidden_size, hidden_size) * 0.01
        self.bi = np.zeros((1, hidden_size))
        
        # Cell state
        self.Wxc = np.random.randn(input_size, hidden_size) * 0.01
        self.Whc = np.random.randn(hidden_size, hidden_size) * 0.01
        self.bc = np.zeros((1, hidden_size))
        
        # Output gate
        self.Wxo = np.random.randn(input_size, hidden_size) * 0.01
        self.Who = np.random.randn(hidden_size, hidden_size) * 0.01
        self.bo = np.zeros((1, hidden_size))
        
        # Output
        self.Why = np.random.randn(hidden_size, output_size) * 0.01
        self.by = np.zeros((1, output_size))
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        s = self.sigmoid(x)
        return s * (1 - s)
    
    def tanh(self, x):
        return np.tanh(x)
    
    def tanh_derivative(self, x):
        return 1 - np.tanh(x) ** 2
    
    def forward(self, X, h_prev=None, c_prev=None):
        seq_len, batch_size, _ = X.shape
        
        if h_prev is None:
            h_prev = np.zeros((batch_size, self.hidden_size))
        if c_prev is None:
            c_prev = np.zeros((batch_size, self.hidden_size))
        
        h = h_prev
        c = c_prev
        
        outputs = []
        hidden_states = [h]
        cell_states = [c]
        
        for t in range(seq_len):
            x_t = X[t]
            
            # Forget gate
            f_t = self.sigmoid(np.dot(x_t, self.Wxf) + np.dot(h, self.Whf) + self.bf)
            
            # Input gate
            i_t = self.sigmoid(np.dot(x_t, self.Wxi) + np.dot(h, self.Whi) + self.bi)
            c_tilde = self.tanh(np.dot(x_t, self.Wxc) + np.dot(h, self.Whc) + self.bc)
            
            # Cell state
            c = f_t * c + i_t * c_tilde
            
            # Output gate
            o_t = self.sigmoid(np.dot(x_t, self.Wxo) + np.dot(h, self.Who) + self.bo)
            h = o_t * self.tanh(c)
            
            # Output
            y_t = np.dot(h, self.Why) + self.by
            
            outputs.append(y_t)
            hidden_states.append(h)
            cell_states.append(c)
        
        return np.array(outputs), h, c
```

---

## 💻 Python Code Examples

```python
# === Complete CNN Example ===

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# Load digits dataset
digits = load_digits()
X = digits.images.reshape(-1, 28, 28, 1) / 16.0  # Normalize
y = digits.target

# One-hot encode
encoder = OneHotEncoder(sparse=False)
y_encoded = encoder.fit_transform(y.reshape(-1, 1))

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Create and train CNN
cnn = SimpleCNN(input_shape=(28, 28, 1), num_classes=10)

# Training loop
epochs = 10
batch_size = 32

for epoch in range(epochs):
    # Forward pass
    output = cnn.forward(X_train)
    
    # Compute loss (cross-entropy)
    probs = cnn.softmax(output)
    loss = -np.mean(np.sum(y_train * np.log(probs + 1e-15), axis=1))
    
    print(f"Epoch {epoch+1}, Loss: {loss:.4f}")
    
    # Backward pass (simplified - full implementation requires more code)
    # ... (backpropagation code)

# Predict
predictions = cnn.predict(X_test)
accuracy = np.mean(predictions == np.argmax(y_test, axis=1))
print(f"Test Accuracy: {accuracy:.4f}")
```

---

## 📊 Summary Tables

### Regularization Techniques

| Technique | Formula | Use Case |
|-----------|---------|----------|
| L1 | λΣ\|w\| | Feature selection |
| L2 | λΣw² | General regularization |
| Dropout | Random zeroing | Deep networks |
| BatchNorm | (x-μ)/σ | Stabilize training |

### CNN Architectures

| Architecture | Layers | Use Case |
|-------------|--------|----------|
| LeNet-5 | Conv-Pool-Conv-Pool-FC | Digit recognition |
| AlexNet | 5 Conv + 3 FC | Image classification |
| VGG | Deep Conv stacks | Feature extraction |
| ResNet | Residual blocks | Very deep networks |

### RNN Variants

| Type | Pros | Cons | Use Case |
|------|------|------|----------|
| Vanilla RNN | Simple | Vanishing gradients | Short sequences |
| LSTM | Long-term memory | Complex | Long sequences |
| GRU | Simpler than LSTM | Good performance | General purpose |

---

## 🎯 ML Applications

| Architecture | ML Application |
|-------------|----------------|
| CNN | Image classification, Object detection |
| RNN | Time series, Text generation |
| LSTM | Language translation, Speech recognition |
| Dropout | Prevent overfitting in deep networks |
| BatchNorm | Faster training, stability |

---

**Status:** ✅ Complete
**Next:** Deep Learning Frameworks (TensorFlow, PyTorch)
