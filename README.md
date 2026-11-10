# Fish Object Detection Using YOLOv8n

<p align="center">
  <strong>Deep Learning | Computer Vision | Object Detection</strong>
</p>

An end-to-end marine object detection project using a custom-trained YOLOv8n model to identify and localize six marine object classes in underwater images.

---

## 1. Project Overview

Underwater object detection is an important computer vision task with applications in marine research, underwater monitoring, and aquatic ecosystem observation.

This project implements an object detection system using **YOLOv8n**, trained from scratch on the **Brackish Dataset**.

The system detects marine objects and predicts their classes and locations using bounding boxes and confidence scores.

### Key Highlights

- Custom YOLOv8n model trained from scratch.
- External real-world underwater dataset.
- Six marine object detection classes.
- GPU-accelerated training using NVIDIA CUDA.
- Separate training, validation, and test datasets.
- Comprehensive model evaluation and visualization.
- Bounding box localization analysis using Intersection over Union (IoU).

---

## 2. Project Information

| Component | Details |
|---|---|
| Project Type | Object Detection |
| Domain | Deep Learning / Computer Vision |
| Dataset | Brackish Dataset |
| Model Architecture | YOLOv8n |
| Training Strategy | From Scratch |
| Framework | PyTorch |
| Detection Framework | Ultralytics |
| Programming Language | Python |
| Development Environment | Visual Studio Code / Jupyter Notebook |
| Hardware Acceleration | NVIDIA GPU with CUDA |

---

## 3. Dataset

### Brackish Dataset

The project uses the Brackish Dataset, an underwater image dataset designed for detecting marine objects in challenging aquatic environments.

The dataset contains images with different underwater conditions, including variations in visibility, lighting, and object appearance.

### Detection Classes

The model is trained to detect six classes:

| Class ID | Class Name |
|---|---|
| 0 | fish |
| 1 | small_fish |
| 2 | crab |
| 3 | shrimp |
| 4 | jellyfish |
| 5 | starfish |

### Dataset Split

The dataset is divided into three independent subsets:

| Split | Images |
|---|---:|
| Training | 11,739 |
| Validation | 1,467 |
| Testing | 1,468 |
| **Total** | **14,674** |

The test set is kept separate from model training and is used for final performance evaluation.

**Note:** The original dataset and processed image files are not included in this GitHub repository because of their large storage requirements. They must be obtained and prepared separately before running the complete notebook.

---

## 4. Model Architecture

### YOLOv8n

YOLOv8n (You Only Look Once — Nano) is a lightweight object detection architecture selected for its balance between detection performance and computational efficiency.

The model is initialized using the YOLOv8n architecture configuration and trained from scratch without loading pretrained weights.

### Model Configuration

| Parameter | Value |
|---|---|
| Model | YOLOv8n |
| Initialization | YOLOv8n YAML architecture |
| Pretrained Weights | Disabled |
| Input Image Size | 640 × 640 |
| Number of Classes | 6 |
| Parameters | 3,006,818 |
| GFLOPs | 8.1 |

---

## 5. Data Preprocessing

The data preparation pipeline includes:

1. Extracting image frames from the original dataset.
2. Organizing images into training, validation, and testing folders.
3. Converting annotations into YOLO-compatible bounding box format.
4. Organizing corresponding label files.
5. Creating the dataset configuration YAML file.
6. Validating image-label correspondence and dataset integrity.

The processed dataset follows the standard YOLO directory structure.

### Expected Dataset Structure

```text
data/
└── processed/
    ├── images/
    │   ├── train/
    │   ├── val/
    │   └── test/
    │
    └── labels/
        ├── train/
        ├── val/
        └── test/
```

---

## 6. Training Configuration

The model was trained using the following configuration:

| Hyperparameter | Value |
|---|---:|
| Epochs | 50 |
| Image Size | 640 |
| Batch Size | 4 |
| Optimizer | AdamW |
| Initial Learning Rate | 0.001 |
| Random Seed | 42 |
| Device | NVIDIA CUDA GPU |
| Workers | 2 |
| Early Stopping Patience | 10 |

### Training from Scratch

To ensure that the model learns from the project dataset rather than relying on pretrained model weights, training is initialized using the YOLOv8n architecture definition.

The training configuration uses:

```python
model = YOLO("yolov8n.yaml")

results = model.train(
    data="data.yaml",
    epochs=50,
    imgsz=640,
    batch=4,
    device=0,
    pretrained=False
)
```

---

## 7. Model Evaluation

The trained model is evaluated on the independent test dataset using standard object detection metrics.

### Evaluation Metrics

- **Precision:** Measures the proportion of predicted objects that are correct.
- **Recall:** Measures the proportion of actual objects successfully detected.
- **mAP@50:** Mean Average Precision at an Intersection over Union threshold of 0.50.
- **mAP@50-95:** Mean Average Precision averaged across IoU thresholds from 0.50 to 0.95.

### Final Test Results

| Metric | Test Performance |
|---|---:|
| Precision | 92.99% |
| Recall | 90.46% |
| mAP@50 | 94.83% |
| mAP@50-95 | 70.32% |

### Per-Class Performance

| Class | Precision | Recall | mAP@50 | mAP@50-95 |
|---|---:|---:|---:|---:|
| Fish | 96.3% | 95.6% | 98.4% | 78.4% |
| Small Fish | 84.6% | 81.4% | 86.9% | 51.5% |
| Crab | 94.1% | 97.3% | 98.8% | 77.1% |
| Shrimp | 90.9% | 88.1% | 93.3% | 61.7% |
| Jellyfish | 94.7% | 85.5% | 92.3% | 61.9% |
| Starfish | 97.4% | 94.9% | 99.3% | 91.4% |

### Results Interpretation

The model achieved strong overall detection performance on the independent test set.

Key observations:

- Starfish achieved the highest mAP@50-95 performance.
- Crab detection achieved strong precision and recall.
- Small fish remained the most challenging class.
- The difference between mAP@50 and mAP@50-95 indicates that precise bounding box localization remains an area for improvement.

---

## 8. Evaluation Visualizations

The project includes evaluation plots generated by Ultralytics YOLO.

The visualizations include:

- Confusion Matrix.
- Normalized Confusion Matrix.
- Precision-Recall Curve.
- F1-Confidence Curve.
- Precision-Confidence Curve.
- Recall-Confidence Curve.

These plots provide insights into classification performance, confidence thresholds, and detection errors.

Evaluation images are available in the `results/` directory.

---

## 9. Prediction and Localization Analysis

The trained model is tested on unseen images to visualize predicted objects.

Each prediction includes:

- Bounding boxes.
- Predicted class labels.
- Confidence scores.

Ground truth annotations are compared with model predictions to inspect detection quality.

### Intersection over Union (IoU)

IoU is used to evaluate bounding box localization by measuring the overlap between predicted and ground truth bounding boxes.

The metric is defined as:

\[
IoU = \frac{Area\ of\ Intersection}{Area\ of\ Union}
\]

Selected test images were also inspected through qualitative Ground Truth versus Prediction comparisons.

---

## 10. Repository Structure

```text
Fish-Detection/
│
├── notebooks/
│   └── Fish_Detection.ipynb
│
├── weights/
│   └── best.pt
│
├── results/
│   ├── confusion_matrix.png
│   ├── confusion_matrix_normalized.png
│   ├── BoxPR_curve.png
│   ├── BoxF1_curve.png
│   ├── BoxP_curve.png
│   └── BoxR_curve.png
│
└── .gitignore
```

### Important Files

| File | Description |
|---|---|
| `Fish_Detection.ipynb` | Complete project implementation and analysis |
| `best.pt` | Best trained YOLOv8n checkpoint |
| `results/` | Model evaluation plots |
| `.gitignore` | Excludes datasets, environments, and temporary files |

---

## 11. Installation and Usage

### Prerequisites

- Python 3.12
- NVIDIA GPU with compatible CUDA-enabled PyTorch installation (optional, but recommended for GPU training)
- Git
- Jupyter Notebook or Visual Studio Code

### Step 1: Clone the Repository

```bash
git clone https://github.com/nour-hossam7/Fish-Detection.git

cd Fish-Detection
```

### Step 2: Create a Virtual Environment

```bash
python -m venv .venv
```

Activate the environment:

**Windows:**

```powershell
.venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install ultralytics opencv-python pandas matplotlib seaborn tqdm pillow pyyaml
```

Install a PyTorch version compatible with your system and CUDA configuration. Refer to the official PyTorch installation instructions.

### Step 4: Prepare the Dataset

Download and extract the Brackish Dataset.

Prepare the images and YOLO-format annotations according to the directory structure described in Section 5.

The original and processed datasets are not included in this repository.

### Step 5: Open the Notebook

Open:

```text
notebooks/Fish_Detection.ipynb
```

Select the Python environment and run the notebook cells in order.

### Step 6: Load the Trained Model

To use the included trained checkpoint:

```python
from ultralytics import YOLO

model = YOLO("weights/best.pt")
```

### Step 7: Run Object Detection

```python
results = model.predict(
    source="path/to/your/image.jpg",
    imgsz=640,
    conf=0.25,
    save=True
)
```

The model will detect marine objects and generate predictions with bounding boxes, class labels, and confidence scores.

**Note:** Reproducing the complete training process requires the original dataset and the notebook's expected dataset configuration. The trained checkpoint can be used independently for inference on new images.

---

## 12. Limitations and Future Improvements

Although the model achieved strong overall detection performance, several improvements can be explored:

- Improve small fish detection through targeted data augmentation.
- Investigate challenging underwater images with low visibility.
- Improve bounding box localization accuracy.
- Analyze overlapping detections and potential false positives.
- Experiment with different YOLO model configurations.
- Perform further hyperparameter optimization.
- Evaluate performance on additional underwater datasets.

---

## 13. Conclusion

This project demonstrates the development of a custom-trained YOLOv8n object detection system for underwater marine object detection.

Using the Brackish Dataset, the model was trained from scratch to detect six marine object classes.

The final evaluation achieved **94.83% mAP@50** and **92.99% Precision** on the independent test dataset.

The project demonstrates the complete object detection workflow, including dataset preparation, model training, performance evaluation, prediction visualization, and bounding box analysis.

---

## 14. Technologies Used

- Python
- PyTorch
- Ultralytics YOLOv8
- OpenCV
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Jupyter Notebook
- CUDA

---

## 15. Author

**Nour Hossam**

Computer Science Student

Egyptian Chinese University

GitHub: [@nour-hossam7](https://github.com/nour-hossam7)

**Project Repository:** [Fish Detection Using YOLOv8n](https://github.com/nour-hossam7/Fish-Detection)

---

<p align="center">
  <strong>Fish Object Detection Using YOLOv8n</strong>
  <br/>
  Deep Learning | Computer Vision | Object Detection
</p>
