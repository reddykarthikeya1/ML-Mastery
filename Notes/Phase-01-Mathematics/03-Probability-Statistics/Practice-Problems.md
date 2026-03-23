# Probability & Statistics - Practice Problems

## 📊 Graded Practice Levels

### Level 1: Basic Probability & Distributions
**1.1** A fair die is rolled. Find:
- a) P(rolling a 4)
- b) P(rolling even)
- c) P(rolling ≥ 3)
**1.2** Two coins are flipped. What is the probability of getting at least one head?
**1.3** If $X \sim \text{Bernoulli}(p = 0.7)$, find the expected value $E[X]$ and variance $Var(X)$.
**1.4** For a normal distribution $N(\mu = 100, \sigma = 15)$, find $P(X > 115)$ using Z-scores.

### Level 2: Intermediate Probability & Inference
**2.1** **Bayes' Theorem:** Disease prevalence is 1%, test sensitivity is 99%, and specificity is 95%. If a person tests positive, what is the probability they actually have the disease?
**2.2** **Joint Distributions:** Given a joint PMF table, how do you verify if two random variables $X$ and $Y$ are independent?
**2.3** **Central Limit Theorem:** If you take a sample of size $n=100$ from an exponential distribution with $\lambda=1$, what is the approximate distribution of the sample mean?
**2.4** **Confidence Intervals:** A sample of $n=100$ has a mean $\bar{X}=50$ and standard deviation $s=10$. Find the 95% confidence interval for the population mean.

### Level 3: Advanced Theory & Analysis
**3.1** **Chebyshev's Inequality:** For a distribution with $\mu=50$ and $\sigma=10$, find the lower bound for the probability $P(30 < X < 70)$.
**3.2** **Information Theory:** Calculate the Entropy (in bits) of a fair 6-sided die.
**3.3** **KL Divergence:** Given two distributions $p = [0.5, 0.5]$ and $q = [0.8, 0.2]$, calculate $D_{KL}(p || q)$. Is $D_{KL}(p || q) = D_{KL}(q || p)$?
**3.4** **Estimation Theory:** Derive the Maximum Likelihood Estimator (MLE) for the parameter $\lambda$ of an Exponential distribution given data $x_1, \dots, x_n$.

### Level 4: Python Implementation Practice
**4.1** Write a Python function using `numpy` to simulate the **Monty Hall problem** over 10,000 trials and verify that switching doors wins ~2/3 of the time.
**4.2** Use `scipy.stats` to plot the Probability Density Function (PDF) of a Normal distribution $N(0, 1)$ and shade the area representing $P(-1 < X < 1)$.
**4.3** Implement a function to calculate the **Shannon Entropy** of a given probability vector.

### Level 5: Real-world Statistical Design
**5.1** **Scenario:** You are running an A/B test for a new website feature.
- Version A (Current) has a 10% conversion rate.
- Version B (New) has an 11% conversion rate based on 1,000 visitors.
**Task:** Formulate the Null and Alternative hypotheses. Describe which statistical test (Z-test vs T-test) you would use, and calculate the p-value. Would you recommend the change at $\alpha = 0.05$?

---

## 📝 Solutions (Selected)

<details>
<summary>Click to reveal solutions</summary>

### 1.1
a) 1/6, b) 3/6 = 1/2, c) 4/6 = 2/3

### 2.1
$P(D|+) = \frac{P(+|D)P(D)}{P(+|D)P(D) + P(+|\neg D)P(\neg D)} = \frac{0.99 \times 0.01}{0.99 \times 0.01 + 0.05 \times 0.99} \approx 0.167$ (16.7%)

### 3.1
$P(30 < X < 70) = P(|X - 50| < 2\sigma)$. By Chebyshev: $P \geq 1 - 1/2^2 = 0.75$.

### 3.2
$H(X) = \sum p_i \log_2(1/p_i) = \log_2(6) \approx 2.585$ bits.

### 4.1
```python
def monty_hall(n=10000, switch=True):
    car = np.random.randint(0, 3, n)
    pick = np.random.randint(0, 3, n)
    if switch:
        return np.mean(pick != car)
    return np.mean(pick == car)
```

</details>

---

## 📝 Notes Section

### My Practice Problems:


### Mistakes to Review:


### Key Insights:


---
**Last Updated:** 2026-03-23
**Status:** ✅ Probability & Statistics Complete!
