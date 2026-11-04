# Pediatric Pneumonia Detection via Chest Radiographs (Milestones 2 & 3)

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.x](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org)
[![Status](https://img.shields.io/badge/Milestone_3-Completed_&_Verified-brightgreen.svg)]()
[![License](https://img.shields.io/badge/license-MIT-blue.svg)]()

This repository contains the complete experimental code, reproduction notebook, presentation decks, and formal academic research report for **Milestone 2 (Baseline Replication)** and **Milestone 3 (Controlled Architectural Innovation & Evaluation)** in automated pediatric pneumonia detection from chest X-rays.

- **Base Paper**: Stephen, O., Sain, M., Maduh, U. J., & Jeong, D. U. (2019). *An Efficient Deep Learning Approach to Pneumonia Classification in Healthcare*. Journal of Healthcare Engineering.
- **Dataset**: Kermany Pediatric Chest X-Ray Cohort (5,856 radiographs: 1,583 Normal, 4,273 Pneumonia).
- **Split**: Exact Stephen et al. (2019) stratified split (3,722 Train / 2,134 Validation).

---

## 📌 Executive Summary of Milestone 3 Findings

In **Milestone 2**, we successfully reproduced the baseline CNN from Stephen et al. (2019), reaching **93.77% validation accuracy** (matching the published 93.73%). However, architectural auditing revealed that **97.51% of all model parameters resided in a single `Dense(512)` layer** ($18,432 \times 512 + 512 = 9,437,696$ parameters), inducing spatial overfitting and learning position-dependent noise.

In **Milestone 3**, we formulated and tested the following controlled hypothesis:
> *Replacing the large `Flatten` + `Dense(512)` classification head with `GlobalAveragePooling2D` (GAP) while keeping the convolutional feature extractor identical will reduce overfitting, leading to improved validation F1-score and/or AUC while maintaining comparable accuracy and enabling native Grad-CAM interpretability.*

### 🚀 Key Empirical Outcomes:
| Evaluation Dimension | Baseline (Stephen et al., 2019) | Proposed GAP-CNN (M3) | Impact / Verdict |
| :--- | :---: | :---: | :--- |
| **Validation Accuracy** | 93.77% | **95.55%** | **+1.78% generalization improvement** |
| **Validation F1-Score** | 0.9578 | **0.9698** | **+0.0120 balance improvement** |
| **Diagnostic Sensitivity (Recall)** | 96.98% | **98.01%** | **Fewer missed pneumonia cases (FN dropped 47 $	o$ 31)** |
| **Diagnostic Specificity** | 85.12% | **88.93%** | **+3.81% (Reduced false positive overtreatment)** |
| **Total Trainable Parameters** | 9,679,041 | **240,961** | **-97.51% massive parameter reduction** |
| **Classification Head Params** | 9,438,209 | **129** | **-99.998% classification head reduction** |
| **Model Disk Size** | 36.92 MB | **0.92 MB** | **40x smaller storage footprint** |
| **Batch Inference Latency** | 14.8 ms | **4.3 ms** | **3.4x faster real-time inference** |
| **Grad-CAM Attention** | Diffuse / Edge Artifacts | Focused Pulmonary Lesions | **Clinically trustworthy localization** |

All four pre-experimental predictions of the hypothesis were **empirically confirmed**.

---

## 📂 Repository Directory Structure

```text
├── Pneumonia_M2_M3_Stephen2019_Replication.ipynb  # Master Jupyter Notebook (Complete M2 & M3)
├── generate_materials.py                         # Master generator script for all deliverables
├── run_m3_experiments.py                         # Standalone runner for experiments & plots
├── Milestone_3_Complete_Project_Report.md        # Comprehensive academic research report
├── Milestone_2_3_Presentation_Materials.txt      # 15-Slide presentation deck & viva script
├── Milestone_3_Presentation_Deck.md              # Markdown formatted presentation deck
├── Milestone_2_3_Report_Introduction.txt         # Literature review introduction
├── outputs/                                      # Experimental outputs, CSVs & figures
│   ├── architecture_parameter_comparison.csv     # Layer-by-layer parameter breakdown
│   ├── m3_comparative_metrics.csv                # Complete comparative benchmark table
│   ├── baseline_metrics.csv / proposed_gap_metrics.csv
│   ├── baseline_history.csv / proposed_gap_history.csv
│   ├── baseline_confusion_matrix.csv / proposed_gap_confusion_matrix.csv
│   ├── m3_comparative_loss_accuracy.png          # Learning dynamics comparison
│   ├── m3_comparative_confusion_matrices.png     # Side-by-side confusion matrices
│   ├── m3_comparative_roc_pr_curves.png          # ROC and PR curves
│   ├── m3_parameter_and_complexity_comparison.png# Bar charts of params, size & latency
│   ├── m3_gradcam_comparative_interpretability.png# Grad-CAM attention comparison
│   └── m3_executive_summary.txt                  # Full text summary of M3 verification
└── README.md                                     # Project documentation
```

---

## 💻 How to Run

### Option 1: Run the Standalone M3 Pipeline
```bash
python3 run_m3_experiments.py
```
This generates all figures, comparative CSV tables, parameter breakdowns, and the executive summary inside the `outputs/` folder.

### Option 2: Run the Jupyter Notebook
Open `Pneumonia_M2_M3_Stephen2019_Replication.ipynb` in **Jupyter Lab**, **VS Code**, **Google Colab**, or **Kaggle**.
The notebook automatically detects the environment and resolves the dataset paths:
- **Colab**: Upload your dataset or mount Google Drive to `/content/chest_xray`.
- **Kaggle**: Attach the `chest-xray-pneumonia` dataset.
- **Local**: Place the dataset in `./chest_xray` or `~/Downloads/chest_xray`.

### Option 3: Regenerate All Project Deliverables
```bash
python3 generate_materials.py
```

---

## 🔬 Architectural Comparison

```
BASELINE (Stephen et al., 2019)
Input (200x200x3)
  │
[Conv2D(32) + MaxPool]
  │
[Conv2D(64) + MaxPool]
  │
[Conv2D(128) + MaxPool]
  │
[Conv2D(128) + MaxPool]  ───> Feature Map: (12x12x128)
  │
Flatten()                ───> 18,432 units
  │
Dropout(0.5)
  │
Dense(512)               ───> 9,437,696 parameters  <-- OVERFITTING BOTTLENECK
  │
Dense(1, 'sigmoid')      ───> Total: 9,679,041 params (36.92 MB)

================================================================================

PROPOSED GAP-CNN (Milestone 3)
Input (200x200x3)
  │
[Conv2D(32) + MaxPool]
  │
[Conv2D(64) + MaxPool]
  │
[Conv2D(128) + MaxPool]
  │
[Conv2D(128) + MaxPool]  ───> Feature Map: (12x12x128)
  │
GlobalAveragePooling2D() ───> 128 units (Spatial average, 0 params)
  │
Dropout(0.3)
  │
Dense(1, 'sigmoid')      ───> 129 parameters  <-- EFFICIENT & REGULARIZED
  │
Total: 240,961 params (0.92 MB)  [97.51% PARAMETER REDUCTION]
```

---

## 🔮 Roadmap for Milestone 4 (M4)
1. **Multi-Class Diagnosis**: Extend model to separate **Bacterial Pneumonia**, **Viral Pneumonia**, and **Normal** cases for antibiotic stewardship.
2. **External Validation**: Test generalization on external clinical cohorts (CheXpert and NIH ChestX-ray14).
3. **Edge Deployment**: Quantize model to **TensorFlow Lite (INT8)** to achieve a **~240 KB** footprint suitable for offline handheld medical devices.

---

## 📚 References
1. Stephen, O., Sain, M., Maduh, U. J., & Jeong, D. U. (2019). An Efficient Deep Learning Approach to Pneumonia Classification in Healthcare. *Journal of Healthcare Engineering*, 2019.
2. Kermany, D. S. et al. (2018). Identifying medical diagnoses and treatable diseases by image-based deep learning. *Cell*, 172(5), 1122-1131.
3. Lin, M., Chen, Q., & Yan, S. (2013). Network in Network. *arXiv:1312.4400*.
4. Selvaraju, R. R. et al. (2017). Grad-CAM: Visual explanations from deep networks via gradient-based localization. *ICCV 2017*.
