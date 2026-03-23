# 7.5 Naive Bayes

## 🎯 Learning Objectives
After completing this section, you will master:
1. **Bayes Theorem**: Understand conditional probability and Bayesian inference
2. **Naive Bayes Variants**: Master Gaussian, Multinomial, and Bernoulli Naive Bayes
3. **Smoothing Techniques**: Apply Laplace and Lidstone smoothing for sparse data
4. **Text Classification**: Build spam filters and sentiment analyzers
5. **Practical Implementation**: Implement Naive Bayes from scratch and apply to real problems

---

## 📚 Bayes Theorem Fundamentals

### 7.5.1 Conditional Probability Review

**Definition:** Probability of event A given that event B has occurred

$$P(A|B) = \frac{P(A \cap B)}{P(B)}$$

**Visual Representation:**
```
Sample Space (Ω)
┌─────────────────────────┐
│                         │
│    ┌─────┐              │
│    │  A  │              │
│    │  ┌──┼─────┐        │
│    │  │A∩B │      │     │
│    └──┼──┘      │        │
│       │    B    │        │
│       └─────────┘        │
│                          │
└──────────────────────────┘

P(A|B) = Area(A∩B) / Area(B)
```

**Example:**
```
Deck of 52 cards:
- P(King) = 4/52 = 1/13
- P(King | Face card) = 4/12 = 1/3
  (Given it's a face card, probability it's a King)
```

### 7.5.2 Bayes Theorem

**Formula:**
$$P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)}$$

**Components:**
- **P(A|B)**: Posterior probability (what we want to find)
- **P(B|A)**: Likelihood (probability of evidence given hypothesis)
- **P(A)**: Prior probability (initial belief)
- **P(B)**: Evidence/Marginal likelihood (normalizing constant)

**Expanded Form:**
$$P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B|A) \cdot P(A) + P(B|\neg A) \cdot P(\neg A)}$$

### Bayes Theorem Derivation

```
Step 1: By definition of conditional probability
P(A|B) = P(A ∩ B) / P(B)

Step 2: Also
P(B|A) = P(A ∩ B) / P(A)

Step 3: Rearrange Step 2
P(A ∩ B) = P(B|A) · P(A)

Step 4: Substitute into Step 1
P(A|B) = P(B|A) · P(A) / P(B)

QED
```

### 7.5.3 Naive Bayes Classifier

**Core Idea:** Apply Bayes theorem with a "naive" independence assumption

**Bayes for Classification:**
$$P(C_k|x) = \frac{P(x|C_k) \cdot P(C_k)}{P(x)}$$

Where:
- $C_k$: Class k
- $x$: Feature vector $(x_1, x_2, ..., x_n)$
- $P(C_k|x)$: Posterior probability of class given features
- $P(C_k)$: Prior probability of class
- $P(x|C_k)$: Likelihood of features given class
- $P(x)$: Evidence (same for all classes, can ignore)

**The "Naive" Assumption:**
```
Assume all features are conditionally independent given the class:

P(x|C_k) = P(x_1, x_2, ..., x_n | C_k)
         = P(x_1|C_k) · P(x_2|C_k) · ... · P(x_n|C_k)
         = ∏_{i=1}^{n} P(x_i|C_k)
```

**Final Classification Rule:**
$$\hat{y} = \arg\max_{C_k} P(C_k) \prod_{i=1}^{n} P(x_i|C_k)$$

**Why "Naive"?**
```
Real world: Features are often correlated
Example: "free" and "money" often appear together in spam

Naive assumption: Treat "free" and "money" as independent
Reality: P(free, money | spam) ≠ P(free | spam) · P(money | spam)

Despite this, Naive Bayes works surprisingly well!
```

---

## 📚 Naive Bayes Variants

### 7.5.4 Gaussian Naive Bayes

**Use Case:** Continuous features that follow normal distribution

**Assumption:** Each feature follows a Gaussian (normal) distribution within each class

$$P(x_i|C_k) = \frac{1}{\sqrt{2\pi\sigma_{ik}^2}} \exp\left(-\frac{(x_i - \mu_{ik})^2}{2\sigma_{ik}^2}\right)$$

Where:
- $\mu_{ik}$: Mean of feature i in class k
- $\sigma_{ik}^2$: Variance of feature i in class k

**Parameter Estimation:**
```python
# For each class k:
μ_ik = mean of feature i for all samples in class k
σ_ik² = variance of feature i for all samples in class k
```

**Gaussian Naive Bayes Implementation:**

```python
import numpy as np
from typing import Tuple

class GaussianNaiveBayes:
    """
    Gaussian Naive Bayes classifier for continuous features.
    Assumes features follow normal distribution within each class.
    """
    
    def __init__(self, var_smoothing: float = 1e-9):
        """
        Initialize Gaussian Naive Bayes.
        
        Args:
            var_smoothing: Portion of largest variance added to variances
                          for numerical stability
        """
        self.var_smoothing = var_smoothing
        self.classes = None
        self.class_priors = None
        self.means = None
        self.variances = None
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'GaussianNaiveBayes':
        """
        Fit Gaussian Naive Bayes model.
        
        Args:
            X: Training features (n_samples, n_features)
            y: Training labels (n_samples,)
        """
        n_samples, n_features = X.shape
        self.classes = np.unique(y)
        n_classes = len(self.classes)
        
        # Initialize parameters
        self.class_priors = np.zeros(n_classes)
        self.means = np.zeros((n_classes, n_features))
        self.variances = np.zeros((n_classes, n_features))
        
        # Calculate parameters for each class
        for idx, c in enumerate(self.classes):
            X_c = X[y == c]
            
            # Prior: P(C_k) = N_k / N
            self.class_priors[idx] = len(X_c) / n_samples
            
            # Mean: μ_ik
            self.means[idx, :] = np.mean(X_c, axis=0)
            
            # Variance: σ_ik² with smoothing
            self.variances[idx, :] = np.var(X_c, axis=0) + self.var_smoothing
        
        return self
    
    def _gaussian_probability(self, x: np.ndarray, mean: float, var: float) -> np.ndarray:
        """Calculate Gaussian probability density"""
        coeff = 1.0 / np.sqrt(2 * np.pi * var)
        exponent = np.exp(-(x - mean) ** 2 / (2 * var))
        return coeff * exponent
    
    def _joint_log_likelihood(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate joint log-likelihood for each class.
        Using log to avoid numerical underflow.
        """
        n_samples = X.shape[0]
        n_classes = len(self.classes)
        log_likelihood = np.zeros((n_samples, n_classes))
        
        for idx in range(n_classes):
            # Log prior: log(P(C_k))
            log_prior = np.log(self.class_priors[idx])
            
            # Log likelihood: Σ log(P(x_i|C_k))
            log_likelihood_features = np.sum(
                np.log(self._gaussian_probability(
                    X, 
                    self.means[idx, :], 
                    self.variances[idx, :]
                ) + 1e-300),  # Avoid log(0)
                axis=1
            )
            
            log_likelihood[:, idx] = log_prior + log_likelihood_features
        
        return log_likelihood
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels"""
        log_likelihood = self._joint_log_likelihood(X)
        return self.classes[np.argmax(log_likelihood, axis=1)]
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities"""
        log_likelihood = self._joint_log_likelihood(X)
        
        # Convert log-likelihood to probability using softmax
        # Subtract max for numerical stability
        log_likelihood_shifted = log_likelihood - np.max(log_likelihood, axis=1, keepdims=True)
        likelihood = np.exp(log_likelihood_shifted)
        probas = likelihood / np.sum(likelihood, axis=1, keepdims=True)
        
        return probas
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Calculate accuracy"""
        predictions = self.predict(X)
        return np.mean(predictions == y)


# Example usage
if __name__ == "__main__":
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    
    # Load data
    iris = load_iris()
    X, y = iris.data, iris.target
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale (optional but recommended for Gaussian NB)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train
    gnb = GaussianNaiveBayes()
    gnb.fit(X_train_scaled, y_train)
    
    # Predict
    y_pred = gnb.predict(X_test_scaled)
    accuracy = gnb.score(X_test_scaled, y_test)
    
    print(f"Gaussian NB Accuracy: {accuracy:.4f}")
    print(f"Class priors: {gnb.class_priors}")
    print(f"Feature means (class 0): {gnb.means[0, :]}")
```

### 7.5.5 Multinomial Naive Bayes

**Use Case:** Discrete count data (especially text classification)

**Assumption:** Features follow a multinomial distribution

$$P(x|C_k) = \frac{(\sum_i x_i)!}{\prod_i x_i!} \prod_{i=1}^{n} P(x_i|C_k)^{x_i}$$

**Simplified (ignoring constant term):**
$$P(x|C_k) \propto \prod_{i=1}^{n} P(x_i|C_k)^{x_i}$$

**Parameter Estimation with Laplace Smoothing:**
$$P(x_i|C_k) = \frac{N_{ik} + \alpha}{N_k + \alpha \cdot n}$$

Where:
- $N_{ik}$: Count of feature i in class k
- $N_k$: Total count of all features in class k
- $n$: Number of features
- $\alpha$: Smoothing parameter (α=1 for Laplace, α<1 for Lidstone)

**Why Smoothing?**
```
Problem: If a feature never appears in training for a class,
         P(feature|class) = 0, making entire product = 0

Solution: Add small constant (α) to all counts

Laplace smoothing (α=1): Add-one smoothing
Lidstone smoothing (0<α<1): Add-fraction smoothing
```

**Multinomial Naive Bayes Implementation:**

```python
class MultinomialNaiveBayes:
    """
    Multinomial Naive Bayes for count/frequency data.
    Commonly used for text classification.
    """
    
    def __init__(self, alpha: float = 1.0, fit_prior: bool = True):
        """
        Initialize Multinomial Naive Bayes.
        
        Args:
            alpha: Smoothing parameter (Laplace smoothing when α=1)
            fit_prior: Whether to learn class priors from data
        """
        self.alpha = alpha
        self.fit_prior = fit_prior
        self.classes = None
        self.class_log_priors = None
        self.feature_log_prob = None
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'MultinomialNaiveBayes':
        """
        Fit Multinomial Naive Bayes.
        
        Args:
            X: Training features (n_samples, n_features) - count data
            y: Training labels (n_samples,)
        """
        n_samples, n_features = X.shape
        self.classes = np.unique(y)
        n_classes = len(self.classes)
        
        # Calculate class priors
        if self.fit_prior:
            class_counts = np.array([np.sum(y == c) for c in self.classes])
            self.class_log_priors = np.log(class_counts / n_samples)
        else:
            self.class_log_priors = np.log(np.ones(n_classes) / n_classes)
        
        # Calculate feature probabilities with smoothing
        self.feature_log_prob = np.zeros((n_classes, n_features))
        
        for idx, c in enumerate(self.classes):
            X_c = X[y == c]
            
            # Count features in class
            feature_counts = np.sum(X_c, axis=0)
            total_count = np.sum(feature_counts)
            
            # Apply Laplace/Lidstone smoothing
            smoothed_counts = feature_counts + self.alpha
            smoothed_total = total_count + self.alpha * n_features
            
            # Log probability
            self.feature_log_prob[idx, :] = np.log(smoothed_counts / smoothed_total)
        
        return self
    
    def _joint_log_likelihood(self, X: np.ndarray) -> np.ndarray:
        """Calculate joint log-likelihood for each class"""
        n_samples = X.shape[0]
        n_classes = len(self.classes)
        log_likelihood = np.zeros((n_samples, n_classes))
        
        for idx in range(n_classes):
            # Log prior + log likelihood
            # log(P(C_k)) + Σ x_i · log(P(x_i|C_k))
            log_likelihood[:, idx] = (
                self.class_log_priors[idx] + 
                np.dot(X, self.feature_log_prob[idx, :].T)
            )
        
        return log_likelihood
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels"""
        log_likelihood = self._joint_log_likelihood(X)
        return self.classes[np.argmax(log_likelihood, axis=1)]
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities"""
        log_likelihood = self._joint_log_likelihood(X)
        
        # Softmax
        log_likelihood_shifted = log_likelihood - np.max(log_likelihood, axis=1, keepdims=True)
        likelihood = np.exp(log_likelihood_shifted)
        return likelihood / np.sum(likelihood, axis=1, keepdims=True)
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Calculate accuracy"""
        return np.mean(self.predict(X) == y)


# Text Classification Example
if __name__ == "__main__":
    from sklearn.feature_extraction.text import CountVectorizer
    
    # Sample text data
    texts = [
        "I love this movie",
        "This film is amazing",
        "Wonderful experience",
        "Terrible movie",
        "Waste of time",
        "Very disappointing"
    ]
    labels = [1, 1, 1, 0, 0, 0]  # 1=positive, 0=negative
    
    # Convert to word counts
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(texts).toarray()
    
    # Train
    mnb = MultinomialNaiveBayes(alpha=1.0)
    mnb.fit(X, labels)
    
    # Predict new text
    new_text = ["love this film", "terrible experience"]
    X_new = vectorizer.transform(new_text).toarray()
    
    predictions = mnb.predict(X_new)
    probas = mnb.predict_proba(X_new)
    
    for text, pred, prob in zip(new_text, predictions, probas):
        print(f"'{text}' → {'Positive' if pred==1 else 'Negative'} (conf: {prob[pred]:.2%})")
```

### 7.5.6 Bernoulli Naive Bayes

**Use Case:** Binary/boolean features (feature present/absent)

**Assumption:** Features follow a Bernoulli distribution

$$P(x|C_k) = \prod_{i=1}^{n} P(x_i|C_k)^{x_i} (1 - P(x_i|C_k))^{1-x_i}$$

Where $x_i \in \{0, 1\}$

**Parameter Estimation:**
$$P(x_i|C_k) = \frac{N_{ik} + \alpha}{N_k + 2\alpha}$$

Where:
- $N_{ik}$: Number of samples in class k where feature i is present
- $N_k$: Total samples in class k

**Bernoulli NB vs Multinomial NB:**
```
Multinomial NB:
- Counts how many times a word appears
- P(word|class) based on frequency

Bernoulli NB:
- Only cares if word is present or not
- P(word|class) based on document frequency
- Also penalizes absence of words
```

**Bernoulli Naive Bayes Implementation:**

```python
class BernoulliNaiveBayes:
    """
    Bernoulli Naive Bayes for binary features.
    Considers feature presence/absence.
    """
    
    def __init__(self, alpha: float = 1.0):
        """
        Initialize Bernoulli Naive Bayes.
        
        Args:
            alpha: Smoothing parameter
        """
        self.alpha = alpha
        self.classes = None
        self.class_log_priors = None
        self.feature_log_prob = None
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'BernoulliNaiveBayes':
        """Fit Bernoulli Naive Bayes"""
        n_samples, n_features = X.shape
        self.classes = np.unique(y)
        n_classes = len(self.classes)
        
        # Class priors
        class_counts = np.array([np.sum(y == c) for c in self.classes])
        self.class_log_priors = np.log(class_counts / n_samples)
        
        # Feature probabilities
        self.feature_log_prob = np.zeros((n_classes, n_features))
        self.feature_log_prob_complement = np.zeros((n_classes, n_features))
        
        for idx, c in enumerate(self.classes):
            X_c = X[y == c]
            n_c = len(X_c)
            
            # Count samples where feature is present
            feature_present = np.sum(X_c, axis=0)
            
            # Smoothed probabilities
            prob_present = (feature_present + self.alpha) / (n_c + 2 * self.alpha)
            prob_absent = 1 - prob_present
            
            # Log probabilities
            self.feature_log_prob[idx, :] = np.log(prob_present)
            self.feature_log_prob_complement[idx, :] = np.log(prob_absent)
        
        return self
    
    def _joint_log_likelihood(self, X: np.ndarray) -> np.ndarray:
        """Calculate joint log-likelihood"""
        n_samples = X.shape[0]
        n_classes = len(self.classes)
        log_likelihood = np.zeros((n_samples, n_classes))
        
        for idx in range(n_classes):
            # Contribution from present features
            present_contrib = np.dot(X, self.feature_log_prob[idx, :].T)
            
            # Contribution from absent features
            absent_contrib = np.dot((1 - X), self.feature_log_prob_complement[idx, :].T)
            
            log_likelihood[:, idx] = (
                self.class_log_priors[idx] + 
                present_contrib + 
                absent_contrib
            )
        
        return log_likelihood
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels"""
        log_likelihood = self._joint_log_likelihood(X)
        return self.classes[np.argmax(log_likelihood, axis=1)]
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Calculate accuracy"""
        return np.mean(self.predict(X) == y)


# Example: Binary text classification
if __name__ == "__main__":
    from sklearn.feature_extraction.text import CountVectorizer
    
    texts = [
        "I love this movie",
        "This film is amazing", 
        "Wonderful experience",
        "Terrible movie",
        "Waste of time",
        "Very disappointing"
    ]
    labels = [1, 1, 1, 0, 0, 0]
    
    # Binary features (word presence)
    vectorizer = CountVectorizer(binary=True)
    X = vectorizer.fit_transform(texts).toarray()
    
    # Train
    bnb = BernoulliNaiveBayes(alpha=1.0)
    bnb.fit(X, labels)
    
    # Predict
    new_text = ["love this", "terrible"]
    X_new = vectorizer.transform(new_text).toarray()
    
    predictions = bnb.predict(X_new)
    print(f"Predictions: {predictions}")
```

---

## 📚 Naive Bayes Applications

### 7.5.7 Text Classification

**Spam Filtering:**
```python
class SpamFilter:
    """Naive Bayes spam filter"""
    
    def __init__(self):
        self.vectorizer = CountVectorizer(stop_words='english')
        self.classifier = MultinomialNaiveBayes(alpha=1.0)
    
    def train(self, texts, labels):
        """Train spam filter"""
        X = self.vectorizer.fit_transform(texts)
        self.classifier.fit(X, labels)
    
    def predict(self, texts):
        """Predict spam or ham"""
        X = self.vectorizer.transform(texts)
        predictions = self.classifier.predict(X)
        probas = self.classifier.predict_proba(X)
        return predictions, probas
    
    def classify(self, text):
        """Classify single text"""
        pred, proba = self.predict([text])
        label = "SPAM" if pred[0] == 1 else "HAM"
        confidence = proba[0][pred[0]]
        return label, confidence


# Example
if __name__ == "__main__":
    # Training data
    spam_texts = [
        "Win money now! Click here for free cash",
        "Congratulations! You've won a lottery",
        "Free entry to win a prize",
        "Claim your reward now",
        "Limited time offer, act now"
    ]
    
    ham_texts = [
        "Hey, are we still meeting tomorrow?",
        "Thanks for the update",
        "Let's catch up soon",
        "Please review the document",
        "Meeting at 3pm today"
    ]
    
    texts = spam_texts + ham_texts
    labels = [1] * len(spam_texts) + [0] * len(ham_texts)
    
    # Train
    filter = SpamFilter()
    filter.train(texts, labels)
    
    # Test
    test_emails = [
        "Win free money now!!!",
        "Meeting scheduled for tomorrow",
        "Claim your prize today",
        "Can you send me the report?"
    ]
    
    for email in test_emails:
        label, conf = filter.classify(email)
        print(f"{email[:40]:<40} → {label} ({conf:.2%})")
```

### Sentiment Analysis

```python
class SentimentAnalyzer:
    """Simple sentiment analyzer using Naive Bayes"""
    
    def __init__(self):
        self.vectorizer = CountVectorizer(ngram_range=(1, 2), max_features=1000)
        self.classifier = MultinomialNaiveBayes(alpha=0.5)
    
    def train(self, texts, sentiments):
        """Train sentiment analyzer"""
        X = self.vectorizer.fit_transform(texts)
        self.classifier.fit(X, sentiments)
    
    def analyze(self, text):
        """Analyze sentiment of text"""
        X = self.vectorizer.transform([text])
        pred = self.classifier.predict(X)[0]
        proba = self.classifier.predict_proba(X)[0]
        
        sentiment = "Positive" if pred == 1 else "Negative"
        confidence = proba[pred]
        
        return {
            'sentiment': sentiment,
            'confidence': confidence,
            'positive_prob': proba[1],
            'negative_prob': proba[0]
        }


# Example
if __name__ == "__main__":
    # Training data
    positive = [
        "Great product, highly recommend",
        "Excellent service and quality",
        "Love it! Best purchase ever",
        "Amazing experience, will buy again",
        "Perfect, exactly what I needed"
    ]
    
    negative = [
        "Terrible quality, very disappointed",
        "Worst product ever, don't buy",
        "Poor service, waste of money",
        "Horrible experience, never again",
        "Broken on arrival, very upset"
    ]
    
    texts = positive + negative
    sentiments = [1] * len(positive) + [0] * len(negative)
    
    # Train
    analyzer = SentimentAnalyzer()
    analyzer.train(texts, sentiments)
    
    # Analyze
    reviews = [
        "Great product but slow shipping",
        "Terrible quality but good price",
        "Absolutely love it!",
        "Worst purchase ever"
    ]
    
    for review in reviews:
        result = analyzer.analyze(review)
        print(f"'{review}'")
        print(f"  → {result['sentiment']} ({result['confidence']:.2%})")
```

---

## 💻 Complete Implementation from Scratch

```python
import numpy as np
from typing import Dict, List, Optional, Union

class NaiveBayesClassifier:
    """
    Unified Naive Bayes Classifier supporting multiple variants.
    
    Variants:
    - gaussian: For continuous features
    - multinomial: For count/frequency data
    - bernoulli: For binary features
    """
    
    def __init__(self, 
                 variant: str = 'gaussian',
                 alpha: float = 1.0,
                 var_smoothing: float = 1e-9):
        """
        Initialize Naive Bayes Classifier.
        
        Args:
            variant: Type of Naive Bayes ('gaussian', 'multinomial', 'bernoulli')
            alpha: Smoothing parameter (for multinomial and bernoulli)
            var_smoothing: Variance smoothing (for gaussian)
        """
        self.variant = variant
        self.alpha = alpha
        self.var_smoothing = var_smoothing
        
        self.classes = None
        self.class_priors = None
        self.parameters = None
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'NaiveBayesClassifier':
        """Fit the classifier based on selected variant"""
        self.classes = np.unique(y)
        n_classes = len(self.classes)
        n_samples, n_features = X.shape
        
        if self.variant == 'gaussian':
            self._fit_gaussian(X, y, n_classes, n_features)
        elif self.variant == 'multinomial':
            self._fit_multinomial(X, y, n_classes, n_features)
        elif self.variant == 'bernoulli':
            self._fit_bernoulli(X, y, n_classes, n_features)
        else:
            raise ValueError(f"Unknown variant: {self.variant}")
        
        return self
    
    def _fit_gaussian(self, X: np.ndarray, y: np.ndarray, 
                      n_classes: int, n_features: int):
        """Fit Gaussian Naive Bayes"""
        self.class_priors = np.zeros(n_classes)
        self.parameters = {
            'means': np.zeros((n_classes, n_features)),
            'variances': np.zeros((n_classes, n_features))
        }
        
        for idx, c in enumerate(self.classes):
            X_c = X[y == c]
            self.class_priors[idx] = len(X_c) / len(X)
            self.parameters['means'][idx, :] = np.mean(X_c, axis=0)
            self.parameters['variances'][idx, :] = (
                np.var(X_c, axis=0) + self.var_smoothing
            )
    
    def _fit_multinomial(self, X: np.ndarray, y: np.ndarray,
                         n_classes: int, n_features: int):
        """Fit Multinomial Naive Bayes"""
        self.class_priors = np.zeros(n_classes)
        self.parameters = {'feature_log_prob': np.zeros((n_classes, n_features))}
        
        for idx, c in enumerate(self.classes):
            X_c = X[y == c]
            self.class_priors[idx] = len(X_c) / len(X)
            
            feature_counts = np.sum(X_c, axis=0)
            total_count = np.sum(feature_counts)
            
            smoothed_counts = feature_counts + self.alpha
            smoothed_total = total_count + self.alpha * n_features
            
            self.parameters['feature_log_prob'][idx, :] = np.log(
                smoothed_counts / smoothed_total
            )
    
    def _fit_bernoulli(self, X: np.ndarray, y: np.ndarray,
                       n_classes: int, n_features: int):
        """Fit Bernoulli Naive Bayes"""
        self.class_priors = np.zeros(n_classes)
        self.parameters = {
            'feature_log_prob': np.zeros((n_classes, n_features)),
            'feature_log_prob_complement': np.zeros((n_classes, n_features))
        }
        
        for idx, c in enumerate(self.classes):
            X_c = X[y == c]
            n_c = len(X_c)
            self.class_priors[idx] = n_c / len(X)
            
            feature_present = np.sum(X_c, axis=0)
            
            prob_present = (feature_present + self.alpha) / (n_c + 2 * self.alpha)
            prob_absent = 1 - prob_present
            
            self.parameters['feature_log_prob'][idx, :] = np.log(prob_present)
            self.parameters['feature_log_prob_complement'][idx, :] = np.log(prob_absent)
    
    def _gaussian_log_likelihood(self, X: np.ndarray) -> np.ndarray:
        """Calculate Gaussian log-likelihood"""
        n_samples = X.shape[0]
        n_classes = len(self.classes)
        log_likelihood = np.zeros((n_samples, n_classes))
        
        for idx in range(n_classes):
            mean = self.parameters['means'][idx, :]
            var = self.parameters['variances'][idx, :]
            
            # Log of Gaussian PDF
            log_prob = -0.5 * np.sum(
                np.log(2 * np.pi * var) + 
                (X - mean) ** 2 / var,
                axis=1
            )
            
            log_likelihood[:, idx] = np.log(self.class_priors[idx]) + log_prob
        
        return log_likelihood
    
    def _multinomial_log_likelihood(self, X: np.ndarray) -> np.ndarray:
        """Calculate Multinomial log-likelihood"""
        n_samples = X.shape[0]
        n_classes = len(self.classes)
        log_likelihood = np.zeros((n_samples, n_classes))
        
        for idx in range(n_classes):
            log_likelihood[:, idx] = (
                np.log(self.class_priors[idx]) +
                np.dot(X, self.parameters['feature_log_prob'][idx, :].T)
            )
        
        return log_likelihood
    
    def _bernoulli_log_likelihood(self, X: np.ndarray) -> np.ndarray:
        """Calculate Bernoulli log-likelihood"""
        n_samples = X.shape[0]
        n_classes = len(self.classes)
        log_likelihood = np.zeros((n_samples, n_classes))
        
        for idx in range(n_classes):
            present_contrib = np.dot(
                X, 
                self.parameters['feature_log_prob'][idx, :].T
            )
            absent_contrib = np.dot(
                (1 - X),
                self.parameters['feature_log_prob_complement'][idx, :].T
            )
            
            log_likelihood[:, idx] = (
                np.log(self.class_priors[idx]) +
                present_contrib + absent_contrib
            )
        
        return log_likelihood
    
    def _joint_log_likelihood(self, X: np.ndarray) -> np.ndarray:
        """Calculate joint log-likelihood based on variant"""
        if self.variant == 'gaussian':
            return self._gaussian_log_likelihood(X)
        elif self.variant == 'multinomial':
            return self._multinomial_log_likelihood(X)
        elif self.variant == 'bernoulli':
            return self._bernoulli_log_likelihood(X)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels"""
        log_likelihood = self._joint_log_likelihood(X)
        return self.classes[np.argmax(log_likelihood, axis=1)]
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities"""
        log_likelihood = self._joint_log_likelihood(X)
        
        # Softmax
        log_likelihood_shifted = log_likelihood - np.max(log_likelihood, axis=1, keepdims=True)
        likelihood = np.exp(log_likelihood_shifted)
        return likelihood / np.sum(likelihood, axis=1, keepdims=True)
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Calculate accuracy"""
        return np.mean(self.predict(X) == y)


# Comprehensive Example
if __name__ == "__main__":
    from sklearn.datasets import load_iris, fetch_20newsgroups
    from sklearn.model_selection import train_test_split
    from sklearn.feature_extraction.text import CountVectorizer
    
    print("=" * 60)
    print("Naive Bayes Classifier - Comprehensive Demo")
    print("=" * 60)
    
    # 1. Gaussian NB on Iris dataset
    print("\n1. Gaussian Naive Bayes (Iris Dataset)")
    print("-" * 40)
    
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.2, random_state=42
    )
    
    gnb = NaiveBayesClassifier(variant='gaussian')
    gnb.fit(X_train, y_train)
    print(f"Accuracy: {gnb.score(X_test, y_test):.4f}")
    
    # 2. Multinomial NB on text data
    print("\n2. Multinomial Naive Bayes (Text Classification)")
    print("-" * 40)
    
    # Simple text classification
    train_texts = [
        "I love this product", "Amazing quality", "Highly recommend",
        "Terrible experience", "Waste of money", "Very disappointed"
    ]
    train_labels = [1, 1, 1, 0, 0, 0]
    
    vectorizer = CountVectorizer()
    X_train_text = vectorizer.fit_transform(train_texts).toarray()
    
    mnb = NaiveBayesClassifier(variant='multinomial', alpha=1.0)
    mnb.fit(X_train_text, train_labels)
    
    test_texts = ["love it", "terrible quality"]
    X_test_text = vectorizer.transform(test_texts).toarray()
    
    predictions = mnb.predict(X_test_text)
    for text, pred in zip(test_texts, predictions):
        print(f"'{text}' → {'Positive' if pred==1 else 'Negative'}")
    
    # 3. Bernoulli NB
    print("\n3. Bernoulli Naive Bayes (Binary Features)")
    print("-" * 40)
    
    vectorizer_binary = CountVectorizer(binary=True)
    X_train_binary = vectorizer_binary.fit_transform(train_texts).toarray()
    
    bnb = NaiveBayesClassifier(variant='bernoulli', alpha=1.0)
    bnb.fit(X_train_binary, train_labels)
    
    X_test_binary = vectorizer_binary.transform(test_texts).toarray()
    predictions = bnb.predict(X_test_binary)
    
    for text, pred in zip(test_texts, predictions):
        print(f"'{text}' → {'Positive' if pred==1 else 'Negative'}")
```

---

## 📊 Summary Tables

### Naive Bayes Variants Comparison

| Variant | Distribution | Input Type | Use Case | Formula |
|---------|-------------|------------|----------|---------|
| **Gaussian** | Normal | Continuous | Numeric features | $P(x_i\|C) = \mathcal{N}(\mu, \sigma^2)$ |
| **Multinomial** | Multinomial | Counts | Text, word counts | $P(x_i\|C) = \frac{N_{ik}+\alpha}{N_k + \alpha n}$ |
| **Bernoulli** | Bernoulli | Binary | Word presence | $P(x_i\|C) = \frac{N_{ik}+\alpha}{N_k + 2\alpha}$ |

### Smoothing Techniques

| Technique | α Value | Effect | Use Case |
|-----------|---------|--------|----------|
| **No smoothing** | α = 0 | Raw probabilities | Large datasets, no zeros |
| **Laplace** | α = 1 | Add-one smoothing | Default choice |
| **Lidstone** | 0 < α < 1 | Add-fraction smoothing | Sparse data, text |
| **Good-Turing** | Variable | Frequency-based | Advanced NLP |

### Naive Bayes Pros and Cons

| Advantages | Disadvantages |
|------------|---------------|
| ✅ Fast training (single pass) | ❌ Naive independence assumption |
| ✅ Fast prediction | ❌ Poor probability calibration |
| ✅ Works with small data | ❌ Zero frequency problem (needs smoothing) |
| ✅ Handles high dimensions | ❌ Can't capture feature interactions |
| ✅ Multi-class support | ❌ Sensitive to irrelevant features |
| ✅ Interpretable | ❌ Assumes specific distribution |

---

## 🎯 ML Applications

| Application | Variant | Description |
|-------------|---------|-------------|
| **Spam Filtering** | Multinomial | Classify emails as spam/ham |
| **Sentiment Analysis** | Multinomial/Bernoulli | Detect positive/negative sentiment |
| **Document Classification** | Multinomial | Categorize news articles |
| **Medical Diagnosis** | Gaussian | Predict disease from symptoms |
| **Recommendation Systems** | Bernoulli | User preference prediction |
| **Anomaly Detection** | Gaussian | Detect outliers in data |

---

## 📝 Practice Problems

### Level 1: Basic

1. **Conceptual**: Explain Bayes theorem in your own words with an example
2. **Calculation**: Given P(A)=0.3, P(B\|A)=0.8, P(B)=0.5, calculate P(A\|B)
3. **Understanding**: Why is Naive Bayes called "naive"?
4. **Code**: Implement Laplace smoothing for a simple probability calculation
5. **Analysis**: When would you use Gaussian NB vs Multinomial NB?

### Level 2: Intermediate

1. **Implementation**: Build a complete Naive Bayes classifier from scratch supporting all three variants
2. **Experiment**: Compare the three variants on the same text classification task
3. **Analysis**: Investigate the effect of different alpha values on classification accuracy
4. **Application**: Build a spam filter using Multinomial Naive Bayes
5. **Debugging**: Handle the zero-frequency problem and demonstrate its impact

### Level 3: Advanced

1. **Research**: Implement Complement Naive Bayes and compare with standard NB
2. **Optimization**: Add parallel processing for faster training on large datasets
3. **Extension**: Implement semi-supervised Naive Bayes using EM algorithm
4. **Project**: Build a complete text classification pipeline with feature selection
5. **Analysis**: Investigate why Naive Bayes works well despite the independence assumption

---

## 🔗 Related Topics
- [[04-Support-Vector-Machines]] - Alternative classification approach
- [[06-Clustering-Algorithms]] - Unsupervised learning comparison
- [[01-Linear-Models]] - Logistic regression comparison
- [[03-ML-Fundamentals]] - Probability theory foundations

---

**Status:** ✅ Complete  
**Next:** [[06-Clustering-Algorithms]]
