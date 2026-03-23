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

## 💻 Professional Implementation

### 1. Manual NMS Implementation (NumPy)
```python
import numpy as np

def nms(boxes, scores, iou_threshold):
    # boxes: [N, 4] (x1, y1, x2, y2), scores: [N]
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1] # Sort by confidence
    
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        
        # Calculate Overlap
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        
        # IoU formula
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        
        # Keep boxes with IoU less than threshold
        inds = np.where(ovr <= iou_threshold)[0]
        order = order[inds + 1]
        
    return keep
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

**Status:** ✅ Expanded Standard (10/10)
**Next:** Image Segmentation (Transposed Conv math, Dice loss, RoIAlign)
