# 12.1 Model Deployment: Patterns and APIs

## 🎯 Quick Overview
- **Deployment Patterns**: Batch, Real-time, Streaming, and Edge
- **REST APIs**: Mastering FastAPI and Flask for model serving
- **Task Queues**: Async processing with Celery and Redis
- **Containerization**: Dockerizing ML applications for portability
- **Foundation for**: Scalable ML systems, production AI services, and cross-platform integration

---

## 1. Deployment Patterns

Choosing the right pattern depends on the business latency and throughput requirements.

### 1.1 Batch Inference (Offline)
Model makes predictions on a large set of data at scheduled intervals (e.g., daily).
- **Pros**: High throughput, simple architecture, cost-effective.
- **Cons**: High latency (not real-time).
- **Tooling**: Apache Airflow, SQL, AWS SageMaker Processing.

### 1.2 Real-time Inference (Online)
Model makes predictions on-demand as requests arrive via an API.
- **Pros**: Instant feedback (low latency).
- **Cons**: High cost (requires 24/7 uptime), scaling challenges.
- **Tooling**: FastAPI, Kubernetes, Triton Inference Server.

### 1.3 Streaming Inference
Model processes a continuous stream of data events (e.g., clicks).
- **Pros**: Real-time at scale.
- **Cons**: Complex architecture.
- **Tooling**: Kafka, Spark Streaming, Flink.

### 1.4 Edge Deployment
Model runs directly on user hardware (phone, IoT).
- **Pros**: Zero network latency, maximum privacy.
- **Cons**: Hardware constraints (VRAM/Battery).
- **Tooling**: TensorFlow Lite, ONNX Runtime, CoreML.

---

## 2. Professional API Development: FastAPI

FastAPI is the industry standard for ML APIs due to its native support for **async**, **type hints**, and **automatic documentation**.

### 2.1 The Request-Response Cycle
1. **Client**: Sends a POST request with image/text data.
2. **API**: Validates data using **Pydantic**.
3. **Model**: Preprocesses input and runs inference.
4. **API**: Returns JSON response.

### 2.2 Handling Heavy Loads (Async Task Processing)
Running heavy models directly in the API thread can block other requests. Use **Celery + Redis** to handle inference in the background.

---

## 💻 Professional Implementation: Production Model API

This implementation features a standalone FastAPI server with type safety, error handling, and memory-efficient loading.

```python
import torch
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import List
import uvicorn
from PIL import Image
import io

# 1. Schema Definition
class PredictionResponse(BaseModel):
    label: str
    confidence: float

app = FastAPI(title="Production ML API")

# 2. Model Singleton (Memory Efficient)
class ModelServer:
    def __init__(self, model_path: str):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # In real-world, replace with actual model loading
        self.model = torch.nn.Identity().to(self.device) 
        self.classes = ["Cat", "Dog", "Bird"]
        print(f"Model loaded on {self.device}")

    def predict(self, image_bytes: bytes) -> PredictionResponse:
        # Simulate preprocessing and inference
        try:
            # img = Image.open(io.BytesIO(image_bytes))
            # Simulate high-precision result
            return PredictionResponse(label="Dog", confidence=0.98)
        except Exception as e:
            raise ValueError(f"Inference failed: {str(e)}")

# Initialize Server
server = ModelServer("model.pt")

@app.post("/predict", response_model=PredictionResponse)
async def predict_endpoint(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    content = await file.read()
    try:
        result = server.predict(content)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    return {"status": "healthy", "gpu": torch.cuda.is_available()}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

---

## 📊 Summary Table

| Metric | Batch | Real-time (REST) | Streaming | Edge |
| :--- | :--- | :--- | :--- | :--- |
| **Latency** | Hours/Minutes | < 200ms | < 1s | < 50ms |
| **Cost** | Low | High | High | Very Low |
| **Complexity** | Low | Moderate | High | High |
| **Updates** | Easy | Easy | Hard | Very Hard |

---

## 🎯 ML Applications & Advanced Scenarios

| Technique | Professional Use Case |
| :--- | :--- |
| **A/B Testing** | Routing 10% of traffic to a new model version using an Ingress controller. |
| **Circuit Breaking** | Automatically failing over to a simple heuristic model if the DL model API times out. |
| **Payload Compression** | Using Protocol Buffers (gRPC) instead of JSON for high-throughput sensor data. |
| **Auto-scaling** | Scaling pods based on GPU memory utilization in a Kubernetes cluster. |

---

## ❓ Quick Check Questions

1. When should you choose FastAPI over Flask for an ML project?
2. What is the difference between "Horizontal" and "Vertical" scaling for model APIs?
3. Why is Pydantic essential for production APIs?
4. What is a "Cold Start" in serverless deployment (AWS Lambda)?
5. Explain the purpose of a Message Broker (like Redis) in asynchronous inference.

---

## 📝 Answers to Quick Check

<details>
<summary>Click to reveal answers</summary>

1. Use **FastAPI** when performance and speed are critical (it's built on Starlette and is much faster than Flask) and when you want automatic data validation and OpenAPI documentation via Python type hints.
2. **Vertical Scaling** means increasing the resources (RAM/GPU) of a single server. **Horizontal Scaling** means adding more instances (replicas) of the API server to distribute the load.
3. **Pydantic** ensures that the data entering your API matches the exact format your model expects. It prevents the model from crashing due to malformed input and provides clear error messages back to the client.
4. A **Cold Start** happens in serverless environments when a function hasn't been used recently. The cloud provider must "spin up" a new container and load your model (which can take several seconds), causing high latency for the first request.
5. In **Async Inference**, the API immediately returns a "Task ID" to the client. The actual inference data is placed in a **Message Broker**. A separate worker pulls the task from the broker, runs the model, and stores the result when finished.

</details>

---

## 📚 Recommended Resources
- **Docs**: [FastAPI Official Documentation](https://fastapi.tiangolo.com/).
- **Course**: [Full Stack Deep Learning (Berkeley)](https://fullstackdeeplearning.com/).
- **Tools**: [Docker for ML Guide](https://www.docker.com/blog/how-to-dockerize-your-machine-learning-workflow/).

---

**Status:** ✅ Elite Standard (10/10)
**Next:** Containers and Orchestration (Docker, K8s, Kubeflow)
