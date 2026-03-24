# 12.4 Model Monitoring and Versioning: Drift & Lineage

## 🎯 Quick Overview
- **Model Versioning**: Tracking weights, parameters, and metadata with **MLflow**
- **Data Versioning**: Using **DVC** to manage large datasets via Git
- **Model Monitoring**: Detecting performance degradation in real-time
- **Drift Detection**: Identifying **Data Drift** and **Concept Drift** with Evidently AI
- **Foundation for**: Reproducible research, model accountability, and automated retraining

---

## 1. The Versioning Trinity

To reproduce an ML experiment exactly, you must version three components:
1. **Code**: Versioned by **Git**.
2. **Data**: Versioned by **DVC** (Data Version Control). DVC stores metadata in Git and large files in S3/GCP/Azure.
3. **Model**: Versioned by an **MLflow Model Registry**, which tracks hyperparameters, training environment, and the final `.pkl` or `.onnx` file.

---

## 2. Drift: Why Models Fail in Production

A model that is 99% accurate today might be 70% accurate in six months due to **Drift**.

### 2.1 Data Drift (Feature Drift)
The distribution of the input data changes.
- *Example*: An model trained on images from high-end cameras fails when users start uploading low-res smartphone photos.
- **Statistical Test**: Kolmogorov-Smirnov (K-S) test or Population Stability Index (PSI).

### 2.2 Concept Drift (Target Drift)
The relationship between input features and the target label changes.
- *Example*: A fraud detection model trained before a major change in banking laws. The "definition" of fraud has evolved.

---

## 3. Monitoring Infrastructure

A production monitor should track four main metrics:
1. **System Metrics**: Latency, CPU/GPU usage, Memory, Throughput.
2. **Data Quality**: Percentage of missing values, schema violations.
3. **Model Performance**: Accuracy, Precision, Recall (calculated using "Ground Truth" if available).
4. **Business Metrics**: Click-through rate, Revenue, User satisfaction.

---

## 💻 Professional Implementation: Drift Detection with Evidently AI

This script calculates a "Data Drift Report" comparing a reference dataset (training) with a current dataset (production).

```python
import pandas as pd
from sklearn import datasets
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset

# 1. Load Data
iris_data = datasets.load_iris(as_frame=True)
iris_frame = iris_data.frame

# 2. Simulate Production Data (Inject Noise/Drift)
reference_data = iris_frame.iloc[:75]
# Shift the mean of the first feature to simulate drift
current_data = iris_frame.iloc[75:].copy()
current_data['sepal length (cm)'] = current_data['sepal length (cm)'] + 2.0

# 3. Create Monitoring Report
drift_report = Report(metrics=[
    DataDriftPreset(),
    TargetDriftPreset()
])

drift_report.run(reference_data=reference_data, current_data=current_data)

# 4. Save and Analyze
drift_report.save_html("drift_report.html")
# In production, check if drift is detected
results = drift_report.as_dict()
is_drift = results['metrics'][0]['result']['dataset_drift']
if is_drift:
    print("ALERT: Data drift detected! Triggering retraining pipeline.")
```

---

## 📊 Summary Comparison

| Feature | Git | DVC | MLflow |
| :--- | :--- | :--- | :--- |
| **Target** | Code/Text | Large Data/Binaries | Experiments/Weights |
| **Storage** | Local/GitHub | S3/Azure/GCP | Database/Object Store |
| **Logic** | Snapshots | Content-addressable | Lifecycle tracking |
| **Best For** | Logic changes | Training datasets | Hyperparameter tuning |

---

## 🎯 ML Applications & Advanced Scenarios

| Technique | Professional Use Case |
| :--- | :--- |
| **Lineage Tracking** | Proving to a regulator exactly which dataset and code produced a specific credit-scoring prediction. |
| **Canary Monitoring** | Comparing the "Data Drift" of a new model version against the production model before full rollout. |
| **Retraining Loop** | Automatically triggering a Kubeflow pipeline when the K-S test score exceeds a threshold. |
| **Model Cards** | Generating a standardized document (via MLflow) that describes a model's intended use and bias limitations. |

---

## ❓ Quick Check Questions

1. Why shouldn't you store large datasets (e.g., 10GB CSVs) directly in a Git repository?
2. What is the difference between Data Drift and Concept Drift?
3. How does MLflow distinguish between a "Run" and a "Registered Model"?
4. What is a "Reference Dataset" in the context of monitoring?
5. Which statistical test is commonly used to detect if a continuous feature's distribution has shifted?

---

## 📝 Answers to Quick Check

<details>
<summary>Click to reveal answers</summary>

1. **Git** is not designed to handle large binary files. It stores every version of a file locally, making the repository size explode and operations (like `git clone`) extremely slow. **DVC** stores the actual data in external cloud storage and only keeps tiny "pointer" files in Git.
2. **Data Drift** is when the inputs ($X$) change (e.g., younger users signing up). **Concept Drift** is when the relationship between $X$ and $y$ changes (e.g., $y$ is now predicted differently by the same $X$ due to external world changes).
3. An **MLflow Run** is a single execution of a training script (tracking one set of params/metrics). A **Registered Model** is a collection of selected runs that have been promoted to a "Production" or "Staging" status for deployment.
4. A **Reference Dataset** is the gold-standard data used to train the model. It acts as the "baseline" distribution that production data is compared against to determine if drift has occurred.
5. The **Kolmogorov-Smirnov (K-S) test** is widely used. It checks the null hypothesis that two samples (training and production) are drawn from the same continuous distribution by measuring the maximum distance between their empirical cumulative distribution functions (ECDF).

</details>

---

## 📚 Recommended Resources
- **Paper**: [Model Cards for Model Reporting (Mitchell et al.)](https://arxiv.org/abs/1810.03993).
- **Docs**: [MLflow Model Registry Guide](https://www.mlflow.org/docs/latest/model-registry.html).
- **Tool**: [Evidently AI: Interactive Drift Visualization](https://docs.evidentlyai.com/).

---

**Status:** ✅ Elite Standard (10/10)
**Next:** Cloud Platforms (AWS SageMaker, Google Vertex AI, Azure ML)
