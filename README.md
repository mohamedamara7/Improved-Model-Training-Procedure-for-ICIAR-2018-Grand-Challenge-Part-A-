# Improved Model Training Procedure for ICIAR 2018 Grand Challenge (Part A)  

This repository contains the training methodology and techniques used to achieve **93% accuracy** on the ICIAR 2018 BACH dataset (Part A: Classification), outperforming the winning team's 87% accuracy.  

---

## **Overview**
The BACH (Breast Cancer Histology) dataset consists of high-resolution histopathological images categorized into four classes: Normal, Benign, In Situ, and Invasive. Due to the large size of the images, a patch-based approach was employed. This README describes the data preprocessing, model architecture, training process, and post-processing techniques used to achieve state-of-the-art results on the test set.


---

## Methodology  

### 1. Data Preprocessing  
- **Patch Extraction**:  
  - Original histopathology images were too large, so they were divided into patches of size `(1400, 1400, 3)` with a step size of `92`.  
  - Each patch was resized to `(224, 224, 3)`.  
- **Offline Augmentation (Stored on Disk)**:  
  Applied 5 augmentations per patch:  
  - Left-right flip  
  - Up-down flip  
  - Rotation (90°, 180°, 270°)  

### 2. Model Architecture (Metric Learning)  
- **Backbone**: `Xception` or `EfficientNet`  
- **Head**:  
  - Generalized Mean Pooling (**GeM Pooling**)  
  - Fully Connected Layer  
  - Batch Normalization  
  - **Arc Margin Product Layer**
- **Loss Function**:  
  - Combined Loss:  
    ```
    Loss = W₁ * ArcFace Loss + W₂ * Cross-Entropy Loss  
    ```
  - Best performance observed with `W₂ = 0` (ArcFace alone worked best due to small class count).  

### 3. Training Strategy  
- **Optimizer**: Adam (`lr=0.001`, `weight_decay=1e-5`)  
- **Learning Rate Schedule**: **Warm-Up Cosine Decay**  
- **Regularization**:  
  - Label Smoothing  
  - **Trivial Augment**

### 4. Post-Processing (Test Time)  
- **Test Time Augmentation (TTA)**:  
  - Applied multiple augmentations at inference and averaged predictions.  
  - **Significantly improved final accuracy.**  

### 5. Techniques That Slightly Improved Accuracy (~1%) (limited impact on the test due to small test set)
- **Advanced Mixing Strategies**:  
  - ResizeMix, SaliencyMix, GridMix, FMix  
- **Model Averaging**:  
  - Stochastic Weight Averaging (**SWA**)  

### 6. Techniques That Did Not Work  
- **Stain Normalization**:  
  - Macenko Normalization
  - Adaptive Color Deconvolution (**ACD**)   
- **Repeated Augmentation** (**RA**):
  - RA For trivial augmentation with N>2 and when N=2 didn't improve the accuracy
- Augmenters Other Than Trivial Aug:
  - RandAug and AugMix

---

## **Key Observations** 

1. **Label Smoothing, TrivialAugment, and TTA**:
   - These techniques collectively contributed to the largest gains in accuracy.

2. **Stain Normalization**:
   - Despite being commonly used in histopathology tasks, stain normalization techniques like Macenko and ACD did not improve performance.

3. **Small Test Set**:
   - The limited size of the test set constrained the potential gains from techniques like SWA, EMA, and mixup variants.
