# ML Fundamentals Summary - Practice Problems

## 📊 Graded Practice Levels

### Level 1: Basic Concept Recall
**1.1** What is the "universal" goal of any Machine Learning algorithm?
**1.2** Explain the difference between "Supervised" and "Unsupervised" learning in one sentence each.
**1.3** Define "Feature Scaling" and name two common methods.
**1.4** What is the "Train-Test Split" and why is it used?

### Level 2: Intermediate Algorithm Logic
**2.1** **Metrics:** In a classification task, if your classes are highly imbalanced (e.g., 99% vs 1%), why is "Accuracy" a poor metric? Which metrics would you use instead?
**2.2** **Clustering:** How does the "Elbow Method" help determine the number of clusters in K-Means?
**2.3** **Dimensionality Reduction:** Why would you use PCA before training a model on a dataset with 500 features?
**2.4** **Validation:** Explain the concept of a "Validation Set" vs. a "Test Set."

### Level 3: Advanced Pipeline Analysis
**3.1** **Data Leakage:** Describe a scenario where data leakage could occur during the feature scaling process. How does a Scikit-Learn `Pipeline` prevent this?
**3.2** **Bias-Variance:** If a model has high training error and high validation error, is it suffering from High Bias or High Variance? What are two ways to fix it?
**3.3** **Feature Selection:** Compare "Filter" and "Wrapper" methods for feature selection. Which one is generally more computationally expensive?

### Level 4: Python Implementation Practice
**4.1** Write a Python snippet using Scikit-Learn to:
1. Load a dataset.
2. Create a `Pipeline` containing `StandardScaler` and `LogisticRegression`.
3. Use `cross_val_score` to evaluate the pipeline with 5-fold CV.

**4.2** Implement a manual "Train-Test Split" using only NumPy (assume you have an array `X` and labels `y`).

### Level 5: Real-world Design Scenario
**5.1** **Scenario:** You are building a system to predict whether a loan application should be approved.
- The data includes `income`, `credit_score`, `employment_history` (categorical), and `loan_amount`.
**Task:** Describe your preprocessing pipeline. Which encoding would you use for `employment_history`? Which scaling would you use for `income`? How would you handle potential outliers in `loan_amount`?

---

## 📝 Solutions (Selected)

<details>
<summary>Click to reveal solutions</summary>

### 1.2
Supervised learning uses labeled data to learn a mapping from inputs to outputs. Unsupervised learning uses unlabeled data to find hidden patterns or structures.

### 2.1
Accuracy would be 99% even if the model predicts the majority class every time. Use **Precision**, **Recall**, **F1-Score**, or **AUC-ROC** instead to better capture the model's performance on the minority class.

### 3.1
Leakage occurs if you `fit` the scaler on the *entire* dataset before splitting. This allows the training set to "know" the mean and variance of the test set. A `Pipeline` ensures the scaler is only `fit` on the training folds during cross-validation.

### 4.1
```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression())
])
scores = cross_val_score(pipe, X, y, cv=5)
```

</details>

---

## 📝 Notes Section

### My Practice Problems:


### Mistakes to Review:


### Key Insights:


---
**Last Updated:** 2026-03-23
**Status:** ✅ ML Fundamentals Complete!
