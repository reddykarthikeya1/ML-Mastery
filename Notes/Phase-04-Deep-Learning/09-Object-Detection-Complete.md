# 11.2 Object Detection (YOLO, R-CNN, SSD)

## 🎯 Quick Overview
- **Fundamentals**: Bounding Boxes, Confidence Scores, and Class Labels
- **Metrics**: IoU (Intersection over Union) and mAP (mean Average Precision)
- **Two-Stage Detectors**: R-CNN family (Faster R-CNN) - High accuracy
- **One-Stage Detectors**: YOLO, SSD - High speed (Real-time)
- **Foundation for**: Autonomous driving, surveillance, and robotics

---

## 1. Object Detection Fundamentals

Unlike classification (what is this image?), detection asks: **where** are the objects and **what** are they?

### 1.1 Core Components
- **Bounding Box (BBox)**: Typically represented as $[x, y, w, h]$ or $[x_{min}, y_{min}, x_{max}, y_{max}]$.
- **Confidence Score**: How sure the model is that a box contains an object.
- **Anchor Boxes**: Pre-defined boxes of different aspect ratios used as templates for detection.

### 1.2 Evaluation Metrics
- **IoU (Intersection over Union)**: measures the overlap between the predicted box and ground truth.
  $$IoU = \frac{\text{Area of Overlap}}{\text{Area of Union}}$$
- **mAP (mean Average Precision)**: The standard metric for object detection, calculated as the average precision across all classes and IoU thresholds.

---

## 2. Two-Stage Detectors (Accuracy Focused)

These models first find "proposals" (potential object regions) and then classify them.

- **R-CNN**: Used Selective Search to find 2000 regions. Very slow.
- **Fast R-CNN**: Shared the convolution feature map across all regions. Much faster.
- **Faster R-CNN**: Introduced the **Region Proposal Network (RPN)**, making the entire pipeline neural and much more accurate.

---

## 3. One-Stage Detectors (Speed Focused)

These models treat detection as a single regression problem, predicting boxes and classes in one pass.

- **YOLO (You Only Look Once)**: Divide the image into a grid. Each cell predicts boxes and probabilities.
- **SSD (Single Shot MultiBox Detector)**: Similar to YOLO but makes predictions across multiple feature map scales to handle objects of different sizes.
- **RetinaNet**: Introduced **Focal Loss** to solve the foreground-background class imbalance problem.

---

## 4. Modern Evolution: YOLOv8 and Beyond

Modern YOLO versions (v5, v8, v10) utilize:
- **Anchor-free** detection (predicting centers directly).
- **Mosaic Augmentation** (combining 4 training images into one).
- **Advanced Backbones** (CSPDarknet).

---

## 💻 Python Code Examples

### 1. Running Inference with YOLOv8 (Ultralytics)
```python
from ultralytics import YOLO
import cv2

# 1. Load a pre-trained model (n=nano, s=small, m=medium, l=large, x=extra large)
model = YOLO("yolov8n.pt")

# 2. Run inference on an image
results = model("bus.jpg")

# 3. Visualize and save
for r in results:
    im_array = r.plot()  # plot a BGR numpy array of predictions
    cv2.imshow("YOLOv8 Detection", im_array)
    cv2.waitKey(0)
```

### 2. Calculating IoU (NumPy Logic)
```python
def calculate_iou(boxA, boxB):
    # box = [x1, y1, x2, y2]
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou
```

---

## 📊 Summary Table

| Model | Type | Speed | Accuracy | Best For |
|-------|------|-------|----------|----------|
| **Faster R-CNN** | 2-Stage | Slow | High | Medical, Satellite |
| **YOLOv8** | 1-Stage | **Real-time** | Very High | Video, Mobile apps |
| **SSD** | 1-Stage | Fast | Moderate | Embedded devices |
| **RetinaNet** | 1-Stage | Moderate | High | Dense small objects |

---

## 🎯 ML Applications

| Technique | ML Application |
|-----------|----------------|
| YOLOv8 | Pedestrian detection for self-driving cars |
| Faster R-CNN | Identifying tumors in X-ray images |
| MediaPipe | Face and Hand landmark detection |
| Custom YOLO | Defect detection in manufacturing lines |

---

## ❓ Quick Check Questions

1. What is the fundamental difference between a One-Stage and a Two-Stage detector?
2. Why is Non-Maximum Suppression (NMS) used after model inference?
3. If two boxes have an IoU of 0.9, what does that tell you about their overlap?
4. What problem does "Focal Loss" solve in RetinaNet?
5. What are "Anchor Boxes," and why are they used?

---

## 📝 Answers to Quick Check

<details>
<summary>Click to reveal answers</summary>

1. **Two-Stage detectors** (like Faster R-CNN) first propose regions of interest and then classify them, leading to high accuracy but low speed. **One-Stage detectors** (like YOLO) predict boxes and classes directly from the image in a single pass, prioritizing speed.
2. **NMS** is used to eliminate redundant, overlapping bounding boxes for the same object. It keeps the box with the highest confidence score and removes others that have a high IoU with it.
3. An **IoU of 0.9** indicates nearly perfect overlap (90% shared area). In detection, any IoU > 0.5 is usually considered a "hit" (True Positive).
4. **Focal Loss** addresses **Class Imbalance**. In most images, the "background" (empty space) vastly outweighs the objects. Focal Loss down-weights the loss contributed by easy-to-classify background examples, forcing the model to focus on hard-to-detect objects.
5. **Anchor Boxes** are pre-defined shapes (rectangles) that represent the common sizes and aspect ratios of objects in the dataset. The model learns to predict the **offset** from these anchors rather than predicting coordinates from scratch, which makes training more stable.

</details>

---

**Status:** ✅ Complete
**Next:** Image Segmentation (U-Net, Mask R-CNN, DeepLab)
