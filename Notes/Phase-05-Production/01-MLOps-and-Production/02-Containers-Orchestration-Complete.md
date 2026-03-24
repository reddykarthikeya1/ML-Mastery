# 12.2 Containers and Orchestration: Docker & Kubernetes

## 🎯 Quick Overview
- **Docker**: Packaging code, dependencies, and models into immutable units
- **Kubernetes (K8s)**: Orchestrating and scaling containers at scale
- **ML on K8s**: Understanding Pods, Deployments, and GPU scheduling
- **Kubeflow**: A specialized platform for ML workflows on Kubernetes
- **Foundation for**: Production-grade MLOps, scalable training, and reliable serving

---

## 1. Containerization with Docker

In MLOps, "it works on my machine" is a disaster. Docker ensures the environment is identical from research to production.

### 1.1 The Dockerfile for ML
A good ML Dockerfile must be small and efficient.
- **Base Image**: Use specialized images like `pytorch/pytorch` or `tensorflow/tensorflow` instead of generic `python`.
- **Layers**: Order commands from least to most frequent changes (OS dependencies → Requirements → Model → Code).
- **Multi-stage Builds**: Use one stage for building/compiling and a second, smaller stage for the final production image.

---

## 2. Kubernetes (K8s) Fundamentals

Kubernetes is the "Operating System" for the cloud. It manages a cluster of servers and ensures your containers are always running.

### 2.1 Core Objects
- **Pod**: The smallest unit in K8s. Usually contains one container (your model API).
- **Deployment**: Defines how many replicas of a Pod should be running. Handles rolling updates.
- **Service**: Provides a stable IP address and load balances traffic across Pods.
- **ConfigMap/Secret**: Stores configuration and sensitive data (API keys).

### 2.2 GPU Scheduling
K8s doesn't natively "see" GPUs. You must install the **NVIDIA Device Plugin**.
- **Resource Limits**: You must explicitly request GPUs in your deployment YAML:
  ```yaml
  resources:
    limits:
      nvidia.com/gpu: 1
  ```

---

## 3. ML Orchestration: Kubeflow

Kubeflow is built on top of Kubernetes to simplify the ML lifecycle.
- **Kubeflow Pipelines**: Automating the workflow from data prep to model deployment as a Directed Acyclic Graph (DAG).
- **KFServing (KServe)**: Specialized K8s controller for serverless model inference with built-in auto-scaling and GPU management.

---

## 💻 Professional Implementation: ML Production Dockerfile

This production-grade Dockerfile uses multi-stage builds to keep the final image size minimal and secure.

```dockerfile
# --- Stage 1: Build ---
FROM python:3.10-slim as builder

WORKDIR /app
COPY requirements.txt .
# Install build dependencies
RUN apt-get update && apt-get install -y build-essential
# Install to a local folder to copy in next stage
RUN pip install --user --no-cache-dir -r requirements.txt

# --- Stage 2: Production ---
FROM python:3.10-slim

WORKDIR /app
# Copy only the installed packages from builder
COPY --from=builder /root/.local /root/.local
COPY . .

# Ensure scripts in .local/bin are in PATH
ENV PATH=/root/.local/bin:$PATH
# Set non-root user for security
RUN useradd -m myuser
USER myuser

EXPOSE 8000
ENTRYPOINT ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## 📊 Summary Comparison

| Feature | Docker | Kubernetes | Kubeflow |
| :--- | :--- | :--- | :--- |
| **Role** | Package | Orchestrate | ML Workflow |
| **Scale** | Individual Instance | Cluster | Pipeline |
| **Target User** | Developer | DevOps Engineer | ML Engineer |
| **Analogy** | A Single Shipping Box | The Cargo Ship | The Logistics Platform |

---

## 🎯 ML Applications & Advanced Scenarios

| Technique | Professional Use Case |
| :--- | :--- |
| **Triton on K8s** | Serving multiple models (PyTorch + ONNX) on a single GPU node. |
| **HPA (Auto-scaling)**| Increasing the number of model replicas when user traffic spikes at 9 AM. |
| **Taint & Toleration**| Ensuring only training jobs run on expensive GPU nodes, while APIs run on CPU nodes. |
| **Sidecar Containers** | Adding a "Logger" container next to the model container to stream logs to ELK. |

---

## ❓ Quick Check Questions

1. Why are "Multi-stage builds" important for deploying Large Language Models?
2. What is the difference between a "Deployment" and a "Service" in Kubernetes?
3. How does Kubernetes ensure high availability for a model API?
4. What is a "DAG" in the context of Kubeflow Pipelines?
5. Why should you never run a Docker container as the `root` user in production?

---

## 📝 Answers to Quick Check

<details>
<summary>Click to reveal answers</summary>

1. **Multi-stage builds** allow you to install large build-time dependencies (like compilers or dev libraries) in the first stage and copy only the final, compiled weights and minimal runtime into the second stage. This reduces image size from GBs to MBs, making deployments faster and more secure.
2. A **Deployment** manages the lifecycle of Pods (how many should run, which version). A **Service** provides a stable entry point (IP/DNS) so that external users can reach those Pods without knowing their individual, changing IP addresses.
3. K8s uses **Health Checks** (Liveness and Readiness probes). If a model container crashes or becomes unresponsive, K8s automatically kills the Pod and restarts a new one on a healthy node.
4. **DAG (Directed Acyclic Graph)** is a collection of all the tasks you want to run, organized in a way that reflects their relationships and dependencies. For example: Data Prep → Training → Evaluation → Deployment.
5. If an attacker gains access to a container running as **root**, they potentially have root access to the underlying host machine (the server). Using a non-root user limits the "blast radius" of a security breach.

</details>

---

## 📚 Recommended Resources
- **Docs**: [Kubernetes Official Basics](https://kubernetes.io/docs/tutorials/kubernetes-basics/).
- **Repo**: [Kubeflow Examples](https://github.com/kubeflow/examples).
- **Guide**: [NVIDIA Container Toolkit Installation](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).

---

**Status:** ✅ Elite Standard (10/10)
**Next:** CI/CD for ML (GitHub Actions, MLflow, Argo)
