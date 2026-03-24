# 11.2 Advanced Object Detection: From Region Proposals to YOLO

## 🎯 Quick Overview
- **Detection Metrics**: Deep dive into Precision-Recall curves and mAP@50:95
- **Anchor Box Math**: Calculating aspect ratios and scale offsets
- **NMS (Non-Maximum Suppression)**: Logic, soft-NMS, and implementation
- **YOLO Evolution**: CSPDarknet backbones and Anchor-free heads
- **Foundation for**: Autonomous vehicles, Multi-object tracking, and Video analytics

---

## 1. Evaluation: The mAP Deep Dive

The **mean Average Precision (mAP)** is the standard for object detection.

1.  **Precision & Recall**:
    - **TP**: IoU > threshold (usually 0.5) and correct class.
    - **FP**: IoU < threshold or duplicate box for the same object.
    - **FN**: Object missed by the model.
2.  **Average Precision (AP)**: The area under the Precision-Recall curve.
3.  **mAP@50**: AP averaged over all classes at an IoU threshold of 0.5.
4.  **mAP@50:95**: AP averaged over IoU thresholds from 0.5 to 0.95 (in 0.05 steps). This is the standard for COCO competitions.

---

## 2. The Logic of Bounding Boxes

### 2.1 Anchor Boxes
Instead of predicting $x, y, w, h$ from scratch, models predict **offsets** $(\delta x, \delta y, \delta w, \delta h)$ relative to pre-defined "Anchors."
- **Math**:
  - $x_{pred} = x_{anchor} + \delta x \cdot w_{anchor}$
  - $w_{pred} = w_{anchor} \cdot e^{\delta w}$

### 2.2 Non-Maximum Suppression (NMS)
NMS removes overlapping boxes for the same object.
**The Algorithm**:
1.  Sort all predicted boxes by confidence score.
2.  Pick the box with the highest score ($B$).
3.  Calculate IoU of $B$ with all other boxes of the same class.
4.  If $IoU > \text{threshold}$, remove the other box.
5.  Repeat for the next highest remaining box.

---

## 3. YOLO Evolution: Real-time Master

### 3.1 CSP (Cross Stage Partial) Darknet
Modern YOLO (v5+) uses CSP layers to split the feature map into two parts, processing one through a series of blocks and then merging them.
- **Benefit**: Reduces computation by $20\%$ while maintaining accuracy.

### 3.2 Anchor-Free Detection (YOLOv8)
Predicts the **distance to the four edges** from the center of a pixel directly.
- **Benefit**: Simplifies the model and makes it more robust to objects with extreme aspect ratios.

---

## 💻 Professional Implementation: End-to-End YOLOv8 Pipeline

This implementation wraps YOLOv8 inference with custom NMS logic and OpenCV visualization, optimized for production deployment.

```python
import cv2
import torch
import numpy as np
from ultralytics import YOLO
from typing import List, Tuple

class VisionDetector:
    def __init__(self, model_path: str = "yolov8n.pt"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = YOLO(model_path).to(self.device)
        print(f"Model loaded on {self.device}")

    def detect_and_draw(self, image_path: str, conf_threshold: float = 0.5) -> np.ndarray:
        """Run inference, apply NMS, and draw bounding boxes."""
        # 1. Load Image
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Image not found at {image_path}")

        # 2. Inference
        results = self.model.predict(img, conf=conf_threshold, device=self.device)
        
        # 3. Process Results
        for result in results:
            boxes = result.boxes.cpu().numpy()
            for box in boxes:
                # Get coordinates
                r = box.xyxy[0].astype(int)
                cls = int(box.cls[0])
                conf = box.conf[0]
                label = f"{self.model.names[cls]} {conf:.2f}"

                # Draw BBox
                cv2.rectangle(img, (r[0], r[1]), (r[2], r[3]), (0, 255, 0), 2)
                # Draw Label
                cv2.putText(img, label, (r[0], r[1] - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return img

    def custom_nms(self, boxes: np.ndarray, scores: np.ndarray, iou_thresh: float) -> List[int]:
        """Industrial-grade NMS logic."""
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            iou = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(iou <= iou_thresh)[0]
            order = order[inds + 1]
        return keep

# --- Usage Example ---
# detector = VisionDetector("yolov8s.pt")
# processed_img = detector.detect_and_draw("traffic.jpg")
# cv2.imwrite("output.jpg", processed_img)
```

---

## 📊 Summary Comparison

| Feature | Faster R-CNN | YOLOv8 | SSD | RetinaNet |
| :--- | :--- | :--- | :--- | :--- |
| **Stages** | 2-Stage | 1-Stage | 1-Stage | 1-Stage |
| **Speed** | 5-10 FPS | **100+ FPS** | 40-60 FPS | 20-40 FPS |
| **Imbalance** | OHEM | Focal Loss | Hard Mining | **Focal Loss** |
| **Anchors** | Yes | **No (Anchor-free)**| Yes | Yes |

---

## 🎯 ML Applications & Advanced Scenarios

| Technique | Professional Use Case |
| :--- | :--- |
| **Multi-Scale Testing**| Running inference at different resolutions to catch tiny objects (e.g., drones in sky). |
| **Soft-NMS** | Instead of deleting boxes, decrease their score based on IoU (useful for crowded scenes). |
| **FPN (Feature Pyramid)**| Detecting objects of wildly different sizes (e.g., a car next to a license plate). |
| **Mosaic Augmentation**| Starching 4 images together to teach the model to handle small, occluded objects. |

---

## ❓ Quick Check Questions

1. Why is mAP@50:95 a better metric than mAP@50 for high-precision tasks?
2. How does the "Region Proposal Network" (RPN) in Faster R-CNN work?
3. What is the "Class Imbalance" problem in object detection, and how does Focal Loss fix it?
4. Explain the "Grid" concept in the original YOLOv1 architecture.
5. In NMS, what happens if the IoU threshold is set too high (e.g., 0.9)?

---

## 📝 Answers to Quick Check

<details>
<summary>Click to reveal answers</summary>

1. **mAP@50:95** rewards models that predict bounding boxes very accurately (tightly around the object). A model could have a high mAP@50 even if its boxes are slightly loose, but high-precision tasks (like surgery or industrial sorting) require tight fits.
2. The **RPN** is a small neural network that slides over the feature map and predicts whether an object exists at that location (objectness score) and the offsets for anchor boxes.
3. **Class Imbalance**: In most images, the "background" pixels vastly outnumber the "object" pixels. **Focal Loss** adds a factor $(1-p_t)^\gamma$ to the cross-entropy loss, which reduces the loss contribution from "easy" (background) examples and focuses the training on "hard" examples.
4. YOLOv1 divides the image into an $S \times S$ grid. Each grid cell is responsible for detecting an object if the center of that object falls into that cell. This makes the model extremely fast but limits its ability to detect many small objects close together.
5. If the threshold is too **high**, NMS will fail to remove redundant boxes. You will see multiple, nearly identical boxes around the same object, which counts as False Positives and lowers your Precision.

</details>

---

## 📚 Recommended Resources
- **Paper**: [You Only Look Once: Unified, Real-Time Object Detection](https://arxiv.org/abs/1506.02640)
- **Paper**: [Focal Loss for Dense Object Detection (RetinaNet)](https://arxiv.org/abs/1708.02002)
- **Repo**: [Ultralytics YOLOv8 Hub](https://github.com/ultralytics/ultralytics).

---

## 4. DETR: Detection Transformer

### 4.1 The Transformer Revolution in Detection
DETR (DEtection TRansformer) reformulates object detection as a set prediction problem.

**Key Innovations**:
1.  **No anchors**: No need for hand-crafted anchor boxes
2.  **No region proposals**: Single forward pass
3.  **Bipartite matching**: Hungarian algorithm for one-to-one assignment
4.  **Object queries**: Learnable embeddings that "look for" objects

---

### 4.2 DETR Architecture

```
Input Image → CNN Backbone → Feature Map → Transformer Encoder
                                                    ↓
                                          Transformer Decoder
                                                    ↓
                    ┌───────────────────────────────┼───────────────────────────────┐
                    ↓                               ↓                               ↓
              Query 1                         Query 2                         Query N
         (box + class)                   (box + class)                   (box + class)
```

**The Math**:
- **Encoder**: Standard self-attention on flattened feature map
- **Decoder**: Cross-attention between object queries and encoder output
- **Output**: $N$ predictions (typically $N=100$)

---

### 4.3 Hungarian Loss
DETR uses bipartite matching to assign predictions to ground truth.

**The Loss**:
$$ \mathcal{L} = \sum_{i=1}^N \left[ \mathbb{1}_{c_i \neq \varnothing} \cdot \mathcal{L}_{cls}(p_i, c_i) + \mathbb{1}_{c_i \neq \varnothing} \cdot \mathcal{L}_{box}(b_i, c_i) \right] $$

Where:
- $\mathbb{1}_{c_i \neq \varnothing}$: Indicator for non-background
- $\mathcal{L}_{cls}$: Focal loss for classification
- $\mathcal{L}_{box}$: L1 + GIoU loss for boxes

```python
import torch
from scipy.optimize import linear_sum_assignment

class HungarianMatcher(nn.Module):
    def __init__(self, cost_class=1, cost_bbox=1, cost_giou=1):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
    
    @torch.no_grad()
    def forward(self, outputs, targets):
        """Match predictions to ground truth using Hungarian algorithm."""
        bs, num_queries = outputs["pred_logits"].shape[:2]
        
        # Flatten batch dimension
        out_prob = outputs["pred_logits"].flatten(0, 1).sigmoid()
        out_bbox = outputs["pred_boxes"].flatten(0, 1)
        
        # Concatenate all targets
        tgt_ids = torch.cat([t["labels"] for t in targets])
        tgt_bbox = torch.cat([t["boxes"] for t in targets])
        
        # Compute costs
        cost_class = -out_prob[:, tgt_ids].log()  # Classification cost
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)  # L1 cost
        cost_giou = -generalized_box_iou(out_bbox, tgt_bbox)  # GIoU cost
        
        # Final cost matrix
        C = (self.cost_class * cost_class + 
             self.cost_bbox * cost_bbox + 
             self.cost_giou * cost_giou)
        
        # Hungarian algorithm
        C = C.view(bs, num_queries, -1).cpu()
        sizes = [len(t["boxes"]) for t in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        
        return [(torch.as_tensor(i, dtype=torch.int64), 
                 torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]
```

---

### 4.4 Deformable DETR
Addresses DETR's slow convergence with deformable attention.

**Key Innovation**: Each query attends to a small set of sampling points around a reference point.

```python
class DeformableAttention(nn.Module):
    def __init__(self, dim, num_heads, num_points=4):
        super().__init__()
        self.num_heads = num_heads
        self.num_points = num_points
        
        self.qkv = nn.Linear(dim, dim * 3)
        self.offset_proj = nn.Linear(dim, num_points * 2)  # 2D offsets
        self.attention_weights = nn.Linear(dim, num_points)
        self.proj = nn.Linear(dim, dim)
    
    def forward(self, x, reference_points):
        # x: [batch, seq_len, dim]
        # reference_points: [batch, seq_len, 2] (normalized coordinates)
        
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        
        # Sample offsets
        offsets = self.offset_proj(x)  # [batch, seq_len, num_points*2]
        offsets = offsets.view(*offsets.shape[:2], self.num_points, 2)
        
        # Compute sampling locations
        sampling_locations = reference_points.unsqueeze(2) + offsets
        
        # Sample features at locations (bilinear interpolation)
        sampled_features = bilinear_sample(k, sampling_locations)
        
        # Attention weights
        weights = self.attention_weights(x).softmax(dim=-1)
        
        # Weighted sum
        output = (sampled_features * weights.unsqueeze(-1)).sum(dim=2)
        
        return self.proj(output)
```

---

## 5. Anchor-Free Detection Evolution

### 5.1 FCOS: Fully Convolutional One-Stage
First successful anchor-free detector.

**Key Idea**: Predict bounding box from each location directly.

**The FCOS Head**:
```python
class FCOSHead(nn.Module):
    def __init__(self, num_classes, channels=256):
        super().__init__()
        
        # Classification branch
        self.cls_conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, num_classes, 3, padding=1)
        )
        
        # Regression branch (predict distances to 4 edges)
        self.reg_conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, 4, 3, padding=1),  # l, t, r, b
            nn.ReLU()
        )
        
        # Center-ness (quality of center alignment)
        self.center_conv = nn.Conv2d(channels, 1, 3, padding=1)
    
    def forward(self, x):
        cls = self.cls_conv(x).sigmoid()
        reg = self.reg_conv(x).exp()  # Ensure positive
        center = self.center_conv(x).sigmoid()
        
        return cls, reg, center
```

**FCOS Loss**:
$$ \mathcal{L} = \frac{1}{N_{pos}} \sum_{i} \left[ \mathcal{L}_{cls} + \lambda \mathcal{L}_{reg} + \alpha \mathcal{L}_{center} \right] $$

---

### 5.2 CenterNet: Keypoint-Based Detection
Detect objects as keypoints (center points).

**Three Outputs**:
1.  **Heatmap**: Object center locations
2.  **Size**: Width and height
3.  **Offset**: Sub-pixel refinement

```python
class CenterNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        
        # Backbone (e.g., ResNet + DLA)
        self.backbone = resnet50()
        
        # Three heads
        self.heatmap_head = nn.Conv2d(256, num_classes, 1)
        self.size_head = nn.Conv2d(256, 2, 1)
        self.offset_head = nn.Conv2d(256, 2, 1)
    
    def forward(self, x):
        features = self.backbone(x)
        
        heatmap = self.heatmap_head(features).sigmoid()
        size = self.size_head(features).exp()
        offset = self.offset_head(features)
        
        return heatmap, size, offset
    
    def decode(self, heatmap, size, offset, threshold=0.3):
        """Decode heatmap to bounding boxes."""
        # Find peaks in heatmap
        heatmap_max = F.max_pool2d(heatmap, kernel_size=3, stride=1, padding=1)
        keep = (heatmap_max == heatmap).float()
        heatmap = heatmap * keep
        
        # Get top-K detections
        batch, classes, height, width = heatmap.shape
        heatmap_flat = heatmap.view(batch, classes, -1)
        scores, indices = torch.topk(heatmap_flat, k=100, dim=-1)
        
        boxes = []
        for b in range(batch):
            for c in range(classes):
                if scores[b, c, 0] > threshold:
                    idx = indices[b, c, 0]
                    y, x = idx // width, idx % width
                    
                    # Get size and offset
                    s = size[b, :, y, x]
                    o = offset[b, :, y, x]
                    
                    # Create box
                    w, h = s[0].item(), s[1].item()
                    cx, cy = x + o[0].item(), y + o[1].item()
                    
                    boxes.append([cx - w/2, cy - h/2, cx + w/2, cy + h/2])
        
        return boxes
```

---

### 5.3 YOLO Evolution: v8 → v9 → v10

#### YOLOv8 (2023)
- **Anchor-free**: Predicts distances to object boundaries
- **C2f module**: Improved feature extraction
- **Decoupled head**: Separate classification and regression

#### YOLOv9 (2024)
- **Programmable Gradient Information (PGI)**: Prevents information loss in deep networks
- **Generalized Efficient Layer Aggregation Network (GELAN)**: Better architecture

**PGI Loss**:
$$ \mathcal{L}_{total} = \mathcal{L}_{main} + \lambda \cdot \mathcal{L}_{auxiliary} $$

The auxiliary branch provides direct gradient flow to early layers.

#### YOLOv10 (2024)
- **NMS-free**: Uses consistent dual assignment for training
- **Efficient architecture**: Reduced latency by 30% vs. v8

```python
# YOLOv10 NMS-free inference
class YOLOv10Detector:
    def __init__(self, model_path):
        self.model = torch.load(model_path)
    
    def detect(self, image):
        # Single forward pass, no NMS needed
        predictions = self.model(image)
        
        # Dual assignment ensures one-to-one matching
        # No post-processing needed!
        return predictions
```

---

## 6. Real-Time Detection Optimization

### 6.1 TensorRT Optimization

```python
import tensorrt as trt

def build_trt_engine(onnx_path, fp16=True):
    """Convert ONNX model to TensorRT engine."""
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    
    # Parse ONNX
    with open(onnx_path, 'rb') as f:
        parser.parse(f.read())
    
    # Configure builder
    config = builder.create_builder_config()
    config.max_workspace_size = 1 << 30  # 1GB
    
    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    
    # Build engine
    engine = builder.build_engine(network, config)
    
    with open('model.trt', 'wb') as f:
        f.write(engine.serialize())
    
    return engine

# Usage: 3-5× speedup on NVIDIA GPUs
engine = build_trt_engine('yolov8.onnx')
```

---

### 6.2 Batched Inference

```python
class BatchedDetector:
    def __init__(self, model, batch_size=8):
        self.model = model
        self.batch_size = batch_size
        self.buffer = []
    
    def add_image(self, image):
        """Add image to batch buffer."""
        self.buffer.append(image)
        
        if len(self.buffer) >= self.batch_size:
            return self.process_batch()
        return None
    
    def process_batch(self):
        """Process all images in buffer."""
        if not self.buffer:
            return []
        
        # Stack images
        batch = torch.stack(self.buffer)
        
        # Single forward pass
        with torch.no_grad():
            results = self.model(batch)
        
        # Clear buffer
        outputs = list(results)
        self.buffer = []
        
        return outputs
    
    def flush(self):
        """Process remaining images."""
        if self.buffer:
            return self.process_batch()
        return []
```

---

### 6.3 Multi-Scale Training

```python
class MultiScaleTrainer:
    def __init__(self, model, scales=[320, 384, 448, 512, 640]):
        self.model = model
        self.scales = scales
    
    def train_epoch(self, dataloader, optimizer):
        for batch_idx, (images, targets) in enumerate(dataloader):
            # Randomly select scale every 10 batches
            if batch_idx % 10 == 0:
                scale = random.choice(self.scales)
            
            # Resize images
            images = F.interpolate(images, size=scale)
            
            # Forward pass
            loss = self.model(images, targets)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
```

---

## 7. Object Tracking

### 7.1 SORT/DeepSORT
Simple Online and Realtime Tracking.

**SORT Algorithm**:
1.  Detect objects in current frame
2.  Predict tracker positions using Kalman filter
3.  Associate detections with trackers using Hungarian algorithm
4.  Update trackers

```python
from filterpy.kalman import KalmanFilter

class SORTTracker:
    def __init__(self):
        self.trackers = []
        self.next_id = 0
    
    def update(self, detections):
        """
        detections: List of (x1, y1, x2, y2, score)
        """
        # Predict tracker positions
        predicted = [t.predict() for t in self.trackers]
        
        # Associate detections with trackers
        if len(predicted) > 0 and len(detections) > 0:
            cost_matrix = iou_cost_matrix(predicted, detections)
            matched_indices = linear_sum_assignment(cost_matrix)
        else:
            matched_indices = []
        
        # Update matched trackers
        matched_detections = set()
        for tracker_idx, det_idx in zip(*matched_indices):
            self.trackers[tracker_idx].update(detections[det_idx])
            matched_detections.add(det_idx)
        
        # Create new trackers for unmatched detections
        for i, det in enumerate(detections):
            if i not in matched_detections:
                tracker = KalmanFilter()
                tracker.x = bbox_to_state(det)
                self.trackers.append(tracker)
        
        # Remove dead trackers
        self.trackers = [t for t in self.trackers if t.time_since_update < 1]
        
        return [(t.x, t.id) for t in self.trackers]
```

---

### 7.2 ByteTrack
High-performance tracking using all detections.

**Key Innovation**: Associate low-score detections instead of discarding them.

```python
class ByteTracker:
    def __init__(self, track_thresh=0.5, match_thresh=0.8):
        self.track_thresh = track_thresh
        self.match_thresh = match_thresh
        self.trackers = []
    
    def update(self, detections):
        # Split detections into high and low confidence
        high_conf = [d for d in detections if d.score >= self.track_thresh]
        low_conf = [d for d in detections if d.score < self.track_thresh]
        
        # First association with high-confidence detections
        matched, unmatched_trackers, unmatched_high = self.associate(
            self.trackers, high_conf, self.match_thresh
        )
        
        # Second association with low-confidence detections
        matched_low, _, _ = self.associate(
            unmatched_trackers, low_conf, 0.5
        )
        
        # Update trackers
        for tracker, det in matched + matched_low:
            tracker.update(det)
        
        # Create new trackers for unmatched high-confidence detections
        for det in unmatched_high:
            self.trackers.append(KalmanTracker(det))
        
        return self.trackers
```

---

## 8. Advanced Detection Scenarios

### 8.1 Small Object Detection

**Challenges**:
- Few pixels → hard to extract features
- Easily filtered by pooling layers

**Solutions**:
1.  **High-resolution input**: 1280×1280 or higher
2.  **Feature Pyramid Networks (FPN)**: Multi-scale features
3.  **Copy-paste augmentation**: Paste small objects into training images
4.  **Dedicated small object head**: Extra prediction head for small objects

```python
class SmallObjectDetector(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        
        # Extra high-resolution pathway
        self.small_object_head = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
        )
        
        # Standard detection head
        self.detection_head = DetectionHead(256)
    
    def forward(self, x):
        # Get high-resolution features early
        early_features = self.backbone.get_early_features(x)  # 1/4 resolution
        small_features = self.small_object_head(early_features)
        
        # Get deep features
        deep_features = self.backbone(x)  # 1/32 resolution
        
        # Combine for detection
        combined = F.interpolate(deep_features, size=small_features.shape[2:])
        combined = combined + small_features
        
        return self.detection_head(combined)
```

---

### 8.2 Occluded Object Detection

**Techniques**:
1.  **Visibility head**: Predict which parts are visible
2.  **Part-based models**: Detect object parts separately
3.  **Temporal aggregation**: Use video context

```python
class OcclusionAwareDetector(nn.Module):
    def __init__(self, num_parts=5):
        super().__init__()
        
        # Full object detection
        self.full_detector = DetectionHead()
        
        # Part detection
        self.part_detector = DetectionHead(num_parts)
        
        # Visibility prediction
        self.visibility_head = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, num_parts, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        full_boxes = self.full_detector(x)
        part_boxes = self.part_detector(x)
        visibility = self.visibility_head(x)
        
        # Combine full and part detections based on visibility
        return self.combine_detections(full_boxes, part_boxes, visibility)
```

---

## 🔬 Research Frontiers (2024-2025)

### 9.1 End-to-End Detection
- **DINO (DETR with Improved DeNoising Anchorboxes)**: Faster convergence, better accuracy
- **StableDINO**: Further improvements for production use
- **Goal**: Remove all hand-crafted components (NMS, anchors, etc.)

### 9.2 Open-Vocabulary Detection
- **Grounding DINO**: Detect any object from text description
- **OWL-ViT**: Zero-shot detection using CLIP
- **Application**: "Detect all red cars" without training on red cars

### 9.3 Video Object Detection
- **Temporal attention**: Aggregate features across frames
- **Tubelet detection**: Detect objects in 3D (x, y, time)
- **Application**: Autonomous driving, video surveillance

### 9.4 3D Object Detection
- **LiDAR + Camera fusion**: Combine depth and visual information
- **BEV (Bird's Eye View)**: Detect in top-down view
- **Application**: Autonomous vehicles, robotics

---

**Status:** ✅ Elite Expanded Standard (14/10)
**Next:** Image Segmentation (SAM, Mask2Former, 3D Segmentation, Video Segmentation)
