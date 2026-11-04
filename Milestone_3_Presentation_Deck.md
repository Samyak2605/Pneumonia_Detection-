# Milestone 2 & 3: Presentation Deck (Markdown Edition)
## Pediatric Pneumonia Detection via Chest Radiographs
### Stephen et al. (2019) Replication & Proposed GAP Architectural Innovation

> **Presenters**: Deep Learning Research Team  
> **Course / Program**: Deep Learning for Medical Healthcare Applications  
> **Repository**: [Pneumonia_Detection-](https://github.com/Samyak2605/Pneumonia_Detection-)

---

# Milestone 2 & 3: Comprehensive Presentation Deck & Viva Script
# Project: Pediatric Pneumonia Detection via Chest Radiographs
# Base Paper: Stephen et al. (2019) Replication & Architectural Innovation (GAP-CNN)

================================================================================
SLIDE 1: Title & Presentation Overview
================================================================================
Slide Title: Pediatric Pneumonia Detection: Baseline Replication & Controlled Architectural Innovation
Subtitle: From Dense Overfitting to Parameter-Efficient Global Average Pooling (Milestones 2 & 3)
Presenters: Student Research Team
Context: Deep Learning for Medical Healthcare Applications
Core Paper: Stephen et al. (2019), Journal of Healthcare Engineering

Talking Points / Speaker Notes:
- Good morning/afternoon, esteemed evaluators. Today we present our Milestone 2 and Milestone 3 outcomes on deep learning-based pneumonia classification from pediatric chest X-rays.
- We first reproduced the influential baseline paper by Stephen et al. (2019) in Milestone 2.
- In Milestone 3, we identified a critical architectural bottleneck—overfitting driven by an oversized fully connected classification head—formulated a scientific hypothesis, implemented a controlled architectural change using Global Average Pooling, and empirically verified substantial gains in generalization, parameter efficiency, and clinical interpretability.

================================================================================
SLIDE 2: Clinical Problem & Healthcare Motivation
================================================================================
Slide Title: Clinical Motivation: Pediatric Pneumonia
Bullet Points:
- Global Health Impact: Pneumonia is the single largest infectious killer of children under 5 worldwide (WHO).
- Diagnostic Modality: Chest X-rays (CXR) are standard, low-cost diagnostic modalities, but interpretation requires pediatric radiological expertise.
- Clinical Bottleneck: In resource-limited clinics, specialist shortages lead to diagnostic delays, severe complications, or antibiotic over-prescription.
- Machine Learning Role: Automated, reliable CNN screening can rapidly triage suspected radiographs and highlight suspicious pulmonary opacities.
- The Core Challenge: High benchmark accuracy often masks severe overfitting and spatial fragility.

Talking Points / Speaker Notes:
- Pediatric chest radiographs are notoriously difficult to interpret because of anatomical factors: rapid breathing motion blur, variable lung volumes, and prominent thymic silhouettes that can mimic consolidations.
- Therefore, a diagnostic deep learning model must learn invariant pulmonary pathology rather than memorizing spatial positions of bones or image borders.

================================================================================
SLIDE 3: Base Paper Review: Stephen et al. (2019)
================================================================================
Slide Title: Base Paper Overview: Stephen et al. (2019)
Bullet Points:
- Paper Title: "An Efficient Deep Learning Approach to Pneumonia Classification in Healthcare" (Journal of Healthcare Engineering, 2019).
- Architecture: 4 Convolutional layers (32, 64, 128, 128 filters) with MaxPool, followed by Flatten, Dropout(0.5), Dense(512), and Sigmoid(1).
- Input Dimension: 200 × 200 × 3 RGB chest X-ray images.
- Published Performance: Reported 95.31% training accuracy and 93.73% validation accuracy.
- Training Protocol: Real-time data augmentation (rotation, shift, zoom, shear, horizontal flip) with Adam optimizer and binary cross-entropy.

Talking Points / Speaker Notes:
- Stephen et al. proposed this model as a scratch-trained alternative to large pre-trained transfer learning backbones like VGG16 or ResNet.
- While effective, their paper utilized a very large dense layer at the end. In our M2 replication, we sought to verify their exact metrics under identical conditions.

================================================================================
SLIDE 4: Dataset Restructuring (Kermany Pediatric Cohort)
================================================================================
Slide Title: Dataset Architecture & Experimental Splitting
Bullet Points:
- Raw Dataset: Kermany et al. pediatric chest X-rays (1–5 years old), 5,856 total images (1,583 Normal, 4,273 Pneumonia).
- Original Kaggle Split: Train (5,216), Val (16), Test (624) -> Val split is statistically invalid (only 16 images!).
- Stephen et al. Restructuring: Consolidated all 5,856 images and stratified into:
  * Training Split: 3,722 images (~63.5%)
  * Validation Split: 2,134 images (~36.5%)
- Class Distribution in Validation Split:
  * 578 Normal radiographs (27.1%)
  * 1,556 Pneumonia radiographs (72.9%)
- Strict Stratification: Prevents label imbalance leakage between splits.

Talking Points / Speaker Notes:
- An essential contribution of our replication in M2 was adhering strictly to Stephen's 3,722 / 2,134 split rather than the default Kaggle split.
- A 16-image validation set cannot provide statistically meaningful confidence intervals, whereas 2,134 images provides high statistical power.

================================================================================
SLIDE 5: Milestone 2 Replication Results (Baseline CNN)
================================================================================
Slide Title: Milestone 2 Baseline Reproduction Results
Bullet Points:
- Training Accuracy: 95.31% (matches reported ~95.3%)
- Validation Accuracy: 93.77% (closely reproduces reported 93.73%)
- Validation Precision: 0.9461
- Validation Recall (Sensitivity): 0.9698
- Validation Specificity: 0.8512
- Validation F1-Score: 0.9578
- Validation ROC-AUC: 0.9997 (high discriminative separation)
- Identified Symptom: Noticeable divergence between training loss and validation loss after epoch 12, indicating beginning of overfitting.

Talking Points / Speaker Notes:
- In Milestone 2, we successfully replicated the baseline within 0.04% of the authors' published validation accuracy.
- However, our in-depth audit uncovered why the model was plateauing: an acute parameter imbalance in the classification head.

================================================================================
SLIDE 6: Identified Problem: The Dense(512) Overfitting Bottleneck
================================================================================
Slide Title: The Problem: Overfitting Bottleneck in Dense Classification Head
Bullet Points:
- Layer Dimension Analysis:
  * Final Conv Layer Output: 12 × 12 × 128 = 18,432 spatial activations per image.
  * Flatten() + Dense(512): (18,432 × 512) + 512 = 9,437,696 parameters!
  * Total Model Parameters: 9,679,041 parameters.
- Disproportionate Allocation: **97.51% of all model parameters reside in a single dense layer!**
- Pitfalls in Medical Imaging:
  1. Position-dependent spatial memorization rather than translation-invariant feature learning.
  2. Susceptibility to border artifacts, diaphragm positioning, and patient markers.
  3. Excessive model weight file size (36.9 MB) and high inference memory requirements.

Talking Points / Speaker Notes:
- Look at this parameter breakdown: the 4 convolutional layers that do the actual medical feature extraction only contain 240,000 parameters.
- But the single Dense(512) layer contains over 9.4 million parameters. This is classic architectural bloat that fosters memorization over generalization.

================================================================================
SLIDE 7: Milestone 3 Hypothesis Formulation
================================================================================
Slide Title: Milestone 3: Controlled Architectural Hypothesis
Bullet Points:
- Proposed Architectural Change:
  * Remove `Flatten()` and `Dense(512)`.
  * Introduce `GlobalAveragePooling2D()` followed by `Dropout(0.3)` and `Dense(1, sigmoid)`.
  * Convolutional feature extractor kept 100% identical.
- Formal Hypothesis:
  "We hypothesize that replacing the large Flatten + Dense(512) classification head with GlobalAveragePooling2D will reduce overfitting by substantially decreasing trainable parameters (>97%), leading to improved validation F1-score and/or AUC while maintaining comparable classification accuracy."
- Pre-Experimental Predictions:
  1. Parameter reduction >97%.
  2. Slightly lower training accuracy (less memorization).
  3. Superior validation F1-score and AUC-ROC (higher generalization).
  4. Native interpretability via Class Activation Mapping (Grad-CAM).

Talking Points / Speaker Notes:
- This is a rigorously controlled experiment. We do not change learning rates, optimizers, image sizes, or convolutional backbones.
- The ONLY independent variable is the classification head mechanism: Flatten + Dense vs. Global Average Pooling.

================================================================================
SLIDE 8: Architectural Comparison: Baseline vs. Proposed GAP-CNN
================================================================================
Slide Title: Architectural Comparison: Baseline vs. Proposed GAP-CNN
Bullet Points:
- Feature Extractor (Both): Conv32 -> Pool -> Conv64 -> Pool -> Conv128 -> Pool -> Conv128 -> Pool
  * Conv Parameters: 240,832 (Identical in both)
- Classification Head:
  * Baseline Head: Flatten(18,432) -> Dropout(0.5) -> Dense(512) -> Dense(1) = 9,438,209 params
  * Proposed Head: GlobalAveragePooling2D(128) -> Dropout(0.3) -> Dense(1) = 129 params!
- Head Parameter Reduction: **99.998% reduction in classification head!**
- Overall Model Parameter Reduction: **9,679,041 down to 240,961 (-97.51%)**
- Disk Size: 36.92 MB down to 0.92 MB (40x reduction!).

Talking Points / Speaker Notes:
- By taking the spatial average across each 12x12 feature map, GAP outputs a single vector of length 128.
- The classification layer simply requires 128 weights and 1 bias. That is just 129 parameters compared to 9.44 million!

================================================================================
SLIDE 9: Controlled Experimental Design
================================================================================
Slide Title: Controlled Experimental Protocol
Bullet Points:
- Constant Dataset: Exact same 3,722 training and 2,134 validation images.
- Constant Augmentation: Rescaling (1/255), rotation (40°), shift (0.2), shear (0.2), zoom (0.2), horizontal flip.
- Constant Resolution: 200 × 200 × 3 RGB.
- Constant Optimizer & Loss: Adam (lr = 0.001), Binary Cross-Entropy.
- Constant Training Duration: 25 epochs with EarlyStopping (patience = 10, restore best weights).
- Evaluated Metrics: Accuracy, Precision, Recall (Sensitivity), Specificity, F1-Score, ROC-AUC, Parameter Count, Disk Size, Latency.

Talking Points / Speaker Notes:
- In machine learning research, altering multiple variables simultaneously invalidates scientific causality.
- Here, every variable is locked. Any shift in performance is directly attributable to the inductive bias of Global Average Pooling.

================================================================================
SLIDE 10: Quantitative Results: Head-to-Head Comparison
================================================================================
Slide Title: Quantitative Benchmark: Baseline vs. Proposed GAP
Bullet Points:
| Metric | Baseline (Stephen 2019) | Proposed (GAP-CNN) | Delta / Gain | Hypothesis Status |
| :--- | :---: | :---: | :---: | :---: |
| Train Accuracy | 95.31% | 94.12% | -1.19% | Verified (Less Overfit) |
| Validation Accuracy | 93.77% | 95.55% | **+1.78%** | **Verified** |
| Precision | 0.9461 | 0.9597 | +0.0136 | **Verified** |
| Recall (Sensitivity) | 0.9698 | 0.9801 | **+0.0103** | **Verified** |
| Specificity | 0.8512 | 0.8893 | **+0.0381** | **Verified** |
| F1-Score | 0.9578 | 0.9698 | **+0.0120** | **Verified** |
| ROC-AUC | 0.9997 | 1.0000 | +0.0003 | **Verified** |
| Trainable Params | 9,679,041 | 240,961 | **-97.51%** | **Verified** |
| Model Size | 36.92 MB | 0.92 MB | **-97.51%** | **Verified** |
| Inference Latency | 14.8 ms | 4.3 ms | **-70.95%** | **Verified (3.4x Faster)** |

Talking Points / Speaker Notes:
- These results confirm our hypothesis across all 4 predictions.
- Training accuracy decreased slightly by 1.19%, proving the model stopped memorizing noise.
- Crucially, validation accuracy rose from 93.77% to 95.55%, and F1-score improved by over 1.2 percentage points.

================================================================================
SLIDE 11: Learning Dynamics & Generalization Gap
================================================================================
Slide Title: Training Dynamics & Overfitting Mitigation
Bullet Points:
- Baseline Behavior:
  * Rapid training loss drop to near-zero.
  * Validation loss begins increasing after epoch 12 (classic divergence / overfitting).
  * Train-validation generalization gap is ~1.54%.
- Proposed GAP-CNN Behavior:
  * Training loss and validation loss descend together in tight synchrony.
  * Validation loss remains flat and stable through epoch 25 without divergence.
  * Train-validation generalization gap is eliminated (validation accuracy actually surpasses training accuracy due to dropout and augmentation).

Talking Points / Speaker Notes:
- Notice the comparative learning curves: GAP acts as an intrinsic structural regularizer.
- Because it has no position-specific dense weights, the model cannot overfit to specific pixel coordinate patterns in the training images.

================================================================================
SLIDE 12: Confusion Matrix & ROC Curves
================================================================================
Slide Title: Diagnostic Error Distribution & ROC Analysis
Bullet Points:
- Validation Cohort (N = 2,134 images: 578 Normal, 1,556 Pneumonia):
- Baseline Confusion Matrix:
  * True Normal: 492 | False Positive: 86
  * False Negative: 47 | True Pneumonia: 1,509
- Proposed GAP Confusion Matrix:
  * True Normal: 514 (+22 correct Normal identification)
  * False Positive: 64 (-22 false positives, preventing overtreatment)
  * False Negative: 31 (**-16 missed pneumonia cases**, saving lives!)
  * True Pneumonia: 1,525 (+16 true positive detections)
- Clinical Recall Gain: Recall increased from 96.98% to 98.01%.

Talking Points / Speaker Notes:
- In clinical triage, false negatives are dangerous—missing a child with active pneumonia can lead to death.
- GAP decreased false negatives from 47 down to 31, a 34% reduction in missed cases, while also reducing false alarms by 25%.

================================================================================
SLIDE 13: Model Explainability: Grad-CAM Feature Attribution
================================================================================
Slide Title: Model Interpretability: Grad-CAM Visual Comparison
Bullet Points:
- Why Explainability Matters: Clinicians will not trust "black box" deep learning predictions without visual validation.
- Baseline Model Grad-CAM:
  * Attention is frequently scattered across rib cages, patient positioning markers, and diaphragmatic borders.
  * Dense layers allow high activations based on non-lung peripheral pixels.
- Proposed GAP-CNN Grad-CAM:
  * Attention heatmaps sharply localize directly over pulmonary infiltrates, consolidation opacities, and bronchopneumonic markings.
  * Quiescent on clear lung fields in Normal radiographs.
- Scientific Explanation: GAP establishes direct mathematical correspondence between the 128 feature maps and the final class score.

Talking Points / Speaker Notes:
- This is a critical finding for Milestone 3: GAP doesn't just improve accuracy; it fundamentally improves clinical trustworthiness.
- Clinicians can see exactly where the model is looking, verifying that it is detecting genuine pathology rather than radiograph borders.

================================================================================
SLIDE 14: Computational Efficiency & Edge Readiness
================================================================================
Slide Title: Edge Deployment & Point-of-Care Readiness
Bullet Points:
- Model Compression: From 36.92 MB to 0.92 MB (40-fold reduction in storage).
- Parameter Efficiency: 240,961 parameters easily fit in on-chip SRAM cache.
- Latency Speedup: Batch inference reduced from 14.8 ms to 4.3 ms (3.4x faster).
- Deployment Scenario:
  * Rural and remote community clinics without cloud connectivity.
  * Direct integration into battery-operated handheld X-ray / ultrasound scanners.
  * Zero recurring cloud inference API costs.

Talking Points / Speaker Notes:
- In many developing nations where pediatric pneumonia mortality is highest, internet connectivity is intermittent or absent.
- A 0.92 MB model can run locally on low-cost mobile devices or Raspberry Pi hardware without needing any internet connection.

================================================================================
SLIDE 15: Conclusions, Limitations & Milestone 4 Roadmap
================================================================================
Slide Title: Conclusions & Milestone 4 Roadmap
Bullet Points:
- Milestone 3 Summary:
  * Replicated Stephen et al. (2019) baseline with 93.77% validation accuracy.
  * Proposed GAP-CNN achieved 95.55% val accuracy, 0.9698 F1, and 98.01% recall.
  * Eliminated 97.51% of model parameters and reduced disk size 40x.
  * All 4 hypothesis predictions empirically verified.
- Project Limitations:
  * Single-center pediatric dataset (demographic and scanner homogeneity).
  * Binary classification (does not separate viral vs. bacterial etiologies).
- Milestone 4 (M4) Objectives:
  1. Multi-class differentiation: Normal vs. Bacterial vs. Viral pneumonia.
  2. External generalization: Cross-evaluation on CheXpert and NIH ChestX-ray14.
  3. Mobile deployment: TensorFlow Lite INT8 quantization (shrinking to ~240 KB).

Talking Points / Speaker Notes:
- In conclusion, Milestone 3 demonstrates that bigger is not better in medical imaging. By applying the right inductive bias with Global Average Pooling, we achieved better accuracy, higher sensitivity, superior interpretability, and 40x smaller model size.
- Thank you. We are now open for questions from the evaluation committee.
================================================================================

