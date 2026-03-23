# ML Fundamentals - Practice Problems

## 📊 Graded Practice Levels

### Level 1: Basic Concept Recall
**1.1** Name the three main types of machine learning and give a one-sentence definition for each.
**1.2** What is the difference between "Features" and "Labels" in a dataset?
**1.3** Define "Overfitting" and "Underfitting" in terms of model performance on training vs. test data.
**1.4** List three common distance metrics used in unsupervised clustering.
**1.5** What is the "Bias-Variance Tradeoff"?

### Level 2: Intermediate Algorithm Logic
**2.1** In a binary classification problem, what information does a "Confusion Matrix" provide? Define Precision and Recall using TP, FP, TN, and FN.
**2.2** How does the "Elbow Method" help in choosing the optimal number of clusters for K-Means?
**2.3** Why is "Feature Scaling" (like Standardization) considered mandatory for algorithms like KNN and PCA, but optional for Decision Trees?
**2.4** Explain the purpose of "Cross-Validation." Why is it more reliable than a single train-test split?
**2.5** What is the "Kernel Trick" in Support Vector Machines, and why is it useful?

### Level 3: Advanced Analysis & Diagnostics
**3.1** Given a Learning Curve where the Training Error is very low but the Validation Error is very high, what is the most likely diagnosis, and what are two potential solutions?
**3.2** Compare PCA and LDA. Which one is "supervised," and what is the difference in their optimization objectives?
**3.3** Explain the difference between "L1 (Lasso)" and "L2 (Ridge)" regularization. Which one produces sparse models?
**3.4** How does the "Silhouette Score" differ from "Inertia" as a metric for evaluating clusters?

### Level 4: Python Implementation Practice
**4.1** Write a Scikit-Learn script to:
1. Load the `Iris` dataset.
2. Split it into 80% training and 20% testing.
3. Scale the features using `StandardScaler`.
4. Train a `LogisticRegression` model.
5. Print the `classification_report`.

**4.2** Use `GridSearchCV` to find the best `max_depth` (test values: 3, 5, 10) for a `DecisionTreeClassifier`.

### Level 5: Real-world ML Workflow
**5.1** **Scenario:** You are tasked with predicting whether a customer will churn (leave) a subscription service.
- The dataset has 100,000 rows.
- 95% of customers stayed (Class 0), only 5% churned (Class 1).
- Missing values exist in the `income` and `last_login_date` columns.
**Task:** Outline your complete ML workflow. Specify how you would handle the missing data, how you would handle the class imbalance, and which evaluation metric you would prioritize to ensure the business catches as many churners as possible.

---

## 📝 Solutions (Selected)

<details>
<summary>Click to reveal solutions</summary>

### 1.1
1. **Supervised:** Learning from labeled data to predict outputs.
2. **Unsupervised:** Finding hidden patterns in unlabeled data.
3. **Reinforcement:** Learning through trial and error to maximize rewards.

### 2.1
- **Precision** = TP / (TP + FP): Of all predicted positives, how many were actually positive?
- **Recall** = TP / (TP + FN): Of all actual positives, how many did we correctly identify?

### 3.1
**Diagnosis:** Overfitting (High Variance).
**Solutions:** 1. Get more training data. 2. Simplify the model (pruning, fewer features). 3. Add regularization.

### 4.1
```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LogisticRegression().fit(X_train, y_train)
print(classification_report(y_test, model.predict(X_test)))
```

### 5.1
1. **Imputation:** Use Median for `income`, and transform `last_login_date` into "days since last login" (filling missing with a large value or mean).
2. **Imbalance:** Use **SMOTE** (oversampling) or set `class_weight='balanced'`.
3. **Metric:** Prioritize **Recall** (to catch all potential churners), while monitoring the F1-score to maintain reasonable precision.

</details>

---

## 📝 Notes Section

### My Practice Problems:


### Mistakes to Review:


### Key Insights:


---
**Last Updated:** 2026-03-23
**Status:** ✅ ML Fundamentals Complete!
