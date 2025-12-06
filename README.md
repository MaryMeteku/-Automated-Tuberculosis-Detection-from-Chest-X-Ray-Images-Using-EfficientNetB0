# 🫁 Automated Tuberculosis Detection from Chest X-Ray Images Using EfficientNetB0

A deep learning project for **binary classification of TB-positive vs. normal chest X-rays**, using **transfer learning (EfficientNetB0)**.  
This work demonstrates how AI can support radiological screening, especially in **low-resource or high-volume environments**.

---

## 📌 Project Overview

Tuberculosis (TB) remains a global health challenge, especially in areas where radiology resources are limited.  
This project uses **Convolutional Neural Networks (CNNs)** to automatically classify chest X-ray images as:

- **TB-Positive**
- **Normal**

The goal is not to replace clinicians, but to **assist early screening**, reduce diagnostic delays, and improve consistency.

---

## 🗂️ Dataset

**Source:** Kaggle  
**Dataset:** Tuberculosis Chest X-Rays Images (Yasser Hessein)  
**Link:** https://www.kaggle.com/datasets/yasserhessein/tuberculosis-chest-x-rays-images/data

### Final dataset after cleaning:

| Class | Count |
|---|---:|
| TB Positive | 2,494 |
| Normal | 514 |
| **Total** | **3,008** |

The dataset is **imbalanced** (5:1), reflecting real screening conditions.

---

## 🔧 Preprocessing & Augmentation

- Resize images to **224 × 224**
- Encode labels as **0 = Normal**, **1 = TB**
- Normalize pixel values

### Stratified Split:

- **70% Training**
- **15% Validation**
- **15% Testing**

### Augmentation:

- Horizontal flip
- Rotation (±10–15°)
- Zoom (±10%)
- Width/height shift (≤ 0.1)

No **vertical flips**, to preserve anatomical realism.

---

## 🧠 Model Architecture

Two models were trained:

### ❌ Baseline CNN (from scratch)

- Conv2D(32) → MaxPooling  
- Conv2D(64) → MaxPooling  
- Conv2D(128) → MaxPooling  
- Flatten → Dense(128, ReLU)  
- Dropout(0.5) → Dense(1, Sigmoid)

Result: **rapid overfitting and limited generalization**

---

### ✅ Transfer Learning Model — EfficientNetB0

EfficientNetB0 (frozen backbone)
↓
GlobalAveragePooling2D
↓
Dense(128, ReLU)
↓
Dropout(0.45)
↓
Dense(1, Sigmoid)


**Why EfficientNet?**

- Best **accuracy-per-parameter**
- Scales **depth, width, resolution**
- **Lower memory usage** → ideal for VMs

---

## ⚙️ Training Configuration

| Parameter | Value |
|---|---|
| Optimizer | Adam |
| Batch Size | 32 |
| Loss | Binary Cross-Entropy |
| Epochs | 10–15 (Early Stopping) |
| Learning Rate | 1e-4 → reduced |
| Callbacks | EarlyStopping, ReduceLROnPlateau, ModelCheckpoint |

### Overfitting prevention:

- Dropout = **0.45**
- Early stopping
- Progressive unfreezing
- Data augmentation

---

## 📊 Results

### 📈 Quantitative Performance

| Metric | Baseline CNN | EfficientNetB0 |
|---|---:|---:|
| Accuracy | 0.84 | **0.99** |
| Precision | 0.85 | **1.00** |
| Recall | 0.88 | **0.99** |
| F1-Score | 0.86 | **0.99** |
| ROC-AUC | 0.92 | **0.99** |

EfficientNetB0 **clearly outperformed** the baseline.

---

### ✔ Confusion Matrix Example

| | Pred Normal | Pred TB |
|---|---:|---:|
| **True Normal** | 507 | 7 |
| **True TB** | 11 | 2244 |

- Only **7 false positives**
- Only **11 false negatives**

False negatives are medically dangerous →  
**High recall was prioritized.**

---

### 🩺 ROC Curve

- **AUC ≈ 0.99**
- Strong discrimination between TB and normal

---

## 🔧 Technical Challenges

### Dataset Path Issues

Initial preprocessing returned:

Found normal images: 0


The issue was caused by inconsistent folder structures from Kaggle.

> Lesson: **AI fails at preprocessing, not modeling.**

---

### VM + TensorFlow Constraints

- CPU-only TensorFlow
- No CUDA
- Memory crashes for larger models

**Practical choices:**

- EfficientNet instead of VGG16
- Smaller batch size
- ModelCheckpoint for resume

---

## 🛡 Ethical Reflections

Medical AI must be **safe, responsible, and clinically supervised.**

### Risks:

- Dataset bias (age, equipment, geography)
- False negatives have consequences
- Misuse in self-diagnosis

### Safeguards:

- **Human in the loop**
- Model must not replace diagnosis
- Clinical deployment requires **FDA-level validation**

> As a healthcare professional, I’ve seen how misdiagnosis affects families and communities.  
> AI must be careful and compassionate.

---

## 🎯 Future Directions

- Add **Grad-CAM heatmaps** for explainability
- Train on **multi-institutional datasets**
- Deploy via **TensorFlow Lite** for mobile clinics
- Integrate with **EHR/FHIR dashboards**
- Explore **self-supervised learning**

---

## 📁 Repository Contents

| File | Description |
|---|---|
| `notebook.ipynb` | Training & evaluation |
| `efficientnet_model.h5` | Saved model |
| `confusion_matrix.png` | Results image |
| `roc_curve.png` | ROC plot |
| `report.pdf` | Final project report |
| `README.md` | This file |

---

## 📚 References

Full references are included in the report.

## 🙌 Acknowledgements

Kaggle dataset by Yasser Hessein

EfficientNet authors

Michigan Technological University – Intro to Big Data Analytics Course

## 🧑‍⚕️ Final Remark

AI does not treat patients — clinicians do.

This project is a tool to support screening, not replace medical judgment.

## 📬 Contact

For questions, collaboration, or feedback, please reach out:

Mary Nnipaa Meteku
Health Informatics – Michigan Technological University
📧 Email: mmeteku@mtu.edu

🌐 LinkedIn: https://www.linkedin.com/in/mary-nnipaa-meteku

