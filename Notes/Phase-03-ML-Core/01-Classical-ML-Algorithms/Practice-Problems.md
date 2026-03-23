# Classical ML Algorithms - Practice Problems

## 📊 Graded Practice Levels

### Level 1: Basic Concept Recall
**1.1** In Linear Regression, what is the geometric interpretation of the "Residuals"?
**1.2** Explain why a single Decision Tree is highly prone to overfitting and name one ensemble method that solves this.
**1.3** Define "Support Vectors" in the context of SVM.
**1.4** What is the fundamental "Naive" assumption in Naive Bayes?
**1.5** What is the difference between "Inertia" and "Silhouette Score" in Clustering?

### Level 2: Intermediate Operations & Tuning
**2.1** Given a model with high variance (overfitting), should you increase or decrease the $\alpha$ (alpha) parameter in Ridge/Lasso regression? Why?
**2.2** In Random Forest, how does the `max_features` hyperparameter help in reducing the correlation between individual trees?
**2.3** Compare the "Kernel Trick" in SVM to explicit feature transformation. What is the computational advantage?
**2.4** When using K-Means, why is the choice of initial centroid placement (e.g., K-Means++) so critical?
**2.5** How does DBSCAN handle "Noise" (outliers) differently compared to K-Means?

### Level 3: Advanced Algorithm Analysis
**3.1** Derive the gradient update rule for a simple Linear Regression model with L2 regularization (Ridge).
**3.2** Explain the concept of "Information Gain" and how it relates to Shannon Entropy in Decision Trees.
**3.3** Prove why Naive Bayes becomes a linear classifier when the features are binary (Bernoulli Naive Bayes).
**3.4** Compare Gaussian Mixture Models (GMM) and K-Means. In what scenario would GMM perform significantly better?

### Level 4: Python Implementation Practice
**4.1** Write a Scikit-Learn pipeline that:
1. Scales data using `StandardScaler`.
2. Selects the top 5 features using `SelectKBest`.
3. Fits a `RandomForestClassifier`.
4. Performs a 5-fold cross-validation.

**4.2** Implement a simple version of the K-Means "Assignment" step from scratch using only NumPy (given data points $X$ and centroids $C$).

### Level 5: Real-world System Design
**5.1** **Scenario:** You are building a Credit Card Fraud Detection system.
- The dataset is highly imbalanced (0.1% fraud cases).
- False Negatives (missing a fraud) are extremely costly.
- You need a model that is both accurate and somewhat interpretable for auditing.
**Task:** Propose an algorithm choice (e.g., Random Forest vs. SVM vs. Logistic Regression). Explain your choice, how you would handle the class imbalance, and which evaluation metric (Precision, Recall, F1, or AUC-ROC) you would prioritize.

---

## 📝 Solutions (Selected)

<details>
<summary>Click to reveal solutions</summary>

### 1.1
Residuals represent the vertical distance between the actual data point and the fitted regression line ($y - \hat{y}$). Geometrically, the goal of OLS is to minimize the sum of the squares of these distances.

### 2.1
You should **increase $\alpha$**. Increasing the regularization strength increases the penalty on large coefficients, which simplifies the model, reduces variance (overfitting), but slightly increases bias.

### 3.4
**GMM** provides "soft clustering" (probabilities of membership) and can model elliptical clusters because it accounts for both the mean and the covariance of the data. **K-Means** is "hard clustering" and assumes clusters are spherical and have similar variance. GMM is better when clusters have overlapping boundaries or non-spherical shapes.

### 4.1
```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('selector', SelectKBest(k=5)),
    ('clf', RandomForestClassifier())
])

scores = cross_val_score(pipeline, X, y, cv=5)
```

### 5.1
**Choice:** Random Forest or Gradient Boosting (XGBoost).
**Reasoning:** They handle non-linearities well, provide "Feature Importance" for interpretability, and are generally robust.
**Imbalance:** Use **SMOTE** (Oversampling) or set `class_weight='balanced'`.
**Metric:** Prioritize **Recall** (to catch as many frauds as possible) while maintaining a reasonable **Precision-Recall AUC**.

</details>

---

## 📝 Notes Section

### My Practice Problems:


### Mistakes to Review:


### Key Insights:


---
**Last Updated:** 2026-03-23
**Status:** ✅ Classical ML Complete!
