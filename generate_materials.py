import json
import os

# --- Create IPYNB ---
notebook = {
    "cells": [],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.8.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

def add_md(text):
    lines = [line + "\n" for line in text.split("\n")]
    if lines: lines[-1] = lines[-1][:-1]
    notebook["cells"].append({"cell_type": "markdown", "metadata": {}, "source": lines})

def add_code(text):
    lines = [line + "\n" for line in text.split("\n")]
    if lines: lines[-1] = lines[-1][:-1]
    notebook["cells"].append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": lines})

add_md("""# Milestone 2 & 3: Pneumonia Detection Baseline & Hypothesis
This notebook implements the baseline replication of Stephen et al. (2019) and defines the hypothesis for M3.
Dataset: Kermany Pediatric Chest X-Ray dataset (5,856 images).
""")

add_code("""import os
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)
""")

add_md("""## 1. Setup Dataset Path
Adjust `DATASET_ROOT` depending on where you are running this (Colab vs Local).""")

add_code("""# Set this to your dataset directory
# DATASET_ROOT = Path("/Users/macbook/Downloads/chest_xray")  # Local Mac
DATASET_ROOT = Path("/content/chest_xray")  # Colab

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)
""")

add_md("""## 2. Dataset Restructuring (3,722 Train / 2,134 Val)
The original paper rearranges the 5,856 images into 3,722 training and 2,134 validation images.""")

add_code("""# Helper to gather all files
def gather_files(dataset_path):
    all_files = []
    labels = []
    
    for split in ['train', 'val', 'test']:
        split_dir = dataset_path / split
        if not split_dir.exists():
            continue
            
        for class_name in ['NORMAL', 'PNEUMONIA']:
            class_dir = split_dir / class_name
            if not class_dir.exists():
                continue
                
            for img_path in class_dir.glob('*.jpeg'):
                all_files.append(str(img_path))
                labels.append(class_name)
                
    return all_files, labels

all_files, all_labels = gather_files(DATASET_ROOT)
print(f"Total images found: {len(all_files)}")

if len(all_files) > 0:
    train_size = 3722
    val_size = len(all_files) - train_size
    
    df = pd.DataFrame({'filename': all_files, 'class': all_labels})
    train_df, val_df = train_test_split(df, train_size=train_size, stratify=df['class'], random_state=42)
    
    print(f"Training images: {len(train_df)}")
    print(f"Validation images: {len(val_df)}")
    
    df.to_csv(OUTPUT_DIR / 'all_images_manifest.csv', index=False)
    train_df.to_csv(OUTPUT_DIR / 'train_manifest.csv', index=False)
    val_df.to_csv(OUTPUT_DIR / 'val_manifest.csv', index=False)
else:
    print("Warning: Dataset not found. Please ensure the dataset is uploaded to DATASET_ROOT.")
    train_df = pd.DataFrame(columns=['filename', 'class'])
    val_df = pd.DataFrame(columns=['filename', 'class'])
""")

add_md("""## 3. Data Augmentation and Generators
Stephen et al. applied specific augmentation settings to the training data.
Input size is 200x200x3.""")

add_code("""IMG_SIZE = (200, 200)
BATCH_SIZE = 32

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1./255)

if len(train_df) > 0:
    train_generator = train_datagen.flow_from_dataframe(
        train_df,
        x_col='filename',
        y_col='class',
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        shuffle=True
    )
    
    val_generator = val_datagen.flow_from_dataframe(
        val_df,
        x_col='filename',
        y_col='class',
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        shuffle=False
    )
""")

add_md("""## 4. CNN Architecture
Replication of the four-convolution-layer CNN from Stephen et al. (2019).""")

add_code("""def build_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
        MaxPooling2D(2, 2),
        
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        
        Flatten(),
        Dropout(0.5),
        Dense(512, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

if len(train_df) > 0:
    model = build_model()
    model.summary()
""")

add_md("""## 5. Training
We train the model, keeping track of the best validation accuracy.""")

add_code("""checkpoint_path = OUTPUT_DIR / 'stephen2019_baseline_best.keras'

if len(train_df) > 0:
    callbacks = [
        ModelCheckpoint(checkpoint_path, save_best_only=True, monitor='val_accuracy', mode='max', verbose=1),
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    ]

    EPOCHS = 25 # Set to a reasonable number to allow convergence

    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=val_generator,
        callbacks=callbacks
    )
    
    # Save history
    hist_df = pd.DataFrame(history.history)
    hist_df.to_csv(OUTPUT_DIR / 'baseline_history.csv', index=False)
""")

add_md("""## 6. Evaluation and Plots
Evaluating the best model on the validation set to obtain Accuracy, Precision, Recall, F1, and AUC.""")

add_code("""if len(train_df) > 0:
    # Plot Accuracy
    plt.figure(figsize=(8, 6))
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Baseline Accuracy Curve')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(loc='lower right')
    plt.savefig(OUTPUT_DIR / 'baseline_accuracy_curve.png')
    plt.show()
    
    # Plot Loss
    plt.figure(figsize=(8, 6))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Baseline Loss Curve')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='upper right')
    plt.savefig(OUTPUT_DIR / 'baseline_loss_curve.png')
    plt.show()

    # Load best weights
    model.load_weights(checkpoint_path)
    
    # Get predictions
    val_generator.reset()
    predictions = model.predict(val_generator)
    y_pred = (predictions > 0.5).astype(int).flatten()
    y_true = val_generator.classes
    
    # Metrics
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    fpr, tpr, thresholds = roc_curve(y_true, predictions)
    roc_auc = auc(fpr, tpr)
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=val_generator.class_indices.keys(), yticklabels=val_generator.class_indices.keys())
    plt.title('Baseline Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(OUTPUT_DIR / 'baseline_confusion_matrix.png')
    plt.show()
    
    pd.DataFrame(cm).to_csv(OUTPUT_DIR / 'baseline_confusion_matrix.csv', index=False)
    
    # ROC Curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig(OUTPUT_DIR / 'baseline_roc_curve.png')
    plt.show()
    
    # Save Metrics
    metrics_data = {
        'Metric': ['Validation Accuracy', 'Precision', 'Recall', 'F1-score', 'AUC'],
        'Your Reproduction': [f"{acc*100:.2f}%", f"{prec:.4f}", f"{rec:.4f}", f"{f1:.4f}", f"{roc_auc:.4f}"]
    }
    metrics_df = pd.DataFrame(metrics_data)
    metrics_df.to_csv(OUTPUT_DIR / 'baseline_metrics.csv', index=False)
    
    print("----- Evaluation Metrics -----")
    print(metrics_df.to_string(index=False))
""")

add_md("""## Milestone 3: Hypothesis

**Proposed Change:**
Replace the `Flatten` + `Dense(512)` classification head with `GlobalAveragePooling2D` while keeping the convolutional feature extractor unchanged.

**Hypothesis:**
We hypothesize that replacing the large `Flatten` + `Dense(512)` classification head with `GlobalAveragePooling2D` will reduce overfitting by substantially decreasing the number of trainable parameters, leading to improved validation F1-score and/or AUC while maintaining comparable classification accuracy.

**Predicted Outcome:**
We expect the proposed model to:
- have substantially fewer trainable parameters;
- potentially achieve slightly lower training accuracy;
- achieve equal or better validation F1-score and AUC if the baseline is overfitting;
- provide a simpler and more parameter-efficient classification head.

*(Note: This is our prediction before experimentation. We have not claimed an improvement yet.)*
""")

with open('Pneumonia_M2_M3_Stephen2019_Replication.ipynb', 'w') as f:
    json.dump(notebook, f, indent=1)

# --- Create PPT Material ---
ppt_content = """# Milestone 2 & 3: Presentation Material

## 1. Brief Introduction (For the PPT)
- Pneumonia detection from chest X-rays is an important medical image-classification problem.
- CNNs can automatically learn visual features from X-ray images for Normal vs. Pneumonia classification.
- Existing research has progressed from basic CNNs to transfer learning, ensembles, attention and advanced representation learning.
- However, high benchmark accuracy does not always guarantee good generalization or robustness.
- **Our objective**: reproduce a simple CNN baseline from Stephen et al. (2019) and then test one controlled architectural modification.

---

## 2. Hypothesis (For the PPT)

**Proposed Change:**
Replace the `Flatten` + `Dense(512)` classification head with `GlobalAveragePooling2D` while keeping the convolutional feature extractor unchanged.

**Hypothesis:**
We hypothesize that replacing the large `Flatten` + `Dense(512)` classification head with `GlobalAveragePooling2D` will reduce overfitting by substantially decreasing the number of trainable parameters, leading to improved validation F1-score and/or AUC while maintaining comparable classification accuracy.

**Predicted Outcome:**
We expect the proposed model to:
- have substantially fewer trainable parameters;
- potentially achieve slightly lower training accuracy;
- achieve equal or better validation F1-score and AUC if the baseline is overfitting;
- provide a simpler and more parameter-efficient classification head.

*Important: This is our prediction before experimentation. We have not claimed an improvement yet.*
"""
with open('Milestone_2_3_Presentation_Materials.txt', 'w') as f:
    f.write(ppt_content)

# --- Create Report Introduction ---
report_intro = """# Introduction for the Literature Survey Report

Pneumonia is a serious respiratory infection and remains an important health problem, particularly in children. Chest X-ray imaging is commonly used as an important source of information for pneumonia assessment, but interpretation can be time-consuming and depends on clinical expertise. These challenges have motivated researchers to investigate deep-learning methods for automated classification of chest X-ray images.

Convolutional neural networks (CNNs) have become widely used for medical image classification because they can learn discriminative visual features directly from images. Research on pneumonia detection has progressed from CNNs trained from scratch to transfer-learning approaches, ensemble models, attention mechanisms, capsule networks, and more recent representation-learning methods. However, high performance on a benchmark dataset does not necessarily guarantee robust generalization to images from different populations, hospitals, or imaging conditions.

In this project, we study pneumonia classification using the Kermany pediatric chest X-ray dataset, which contains labelled Normal and Pneumonia images. We selected the work of Stephen et al. (2019), *An Efficient Deep Learning Approach to Pneumonia Classification in Healthcare*, as our base paper because it uses the same dataset and proposes a relatively simple CNN trained from scratch. The study reported 95.31% training accuracy and 93.73% validation accuracy for its 200 × 200 RGB configuration.

Our first objective is therefore to reproduce the methodology and establish a reliable baseline. After baseline reproduction, we will investigate one controlled architectural modification and evaluate whether it improves generalization. The broader motivation of the project is to examine not only whether a model achieves high benchmark accuracy, but also how architectural choices influence reliability and generalization.
"""
with open('Milestone_2_3_Report_Introduction.txt', 'w') as f:
    f.write(report_intro)

print("Materials generated successfully!")
