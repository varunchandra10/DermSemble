# PHASE 6: SWIN TRANSFORMER TINY TRAINING & COMPARATIVE EVALUATION REPORT

**Project:** AI-Based Tool for Preliminary Diagnosis of Dermatological Manifestations  
**Date:** November 1, 2026  
**Status:** COMPLETE & EMPIRICALLY VERIFIED  
**Compute Architecture:** NVIDIA GeForce RTX 5050 Laptop GPU (8.55 GB VRAM, Blackwell Architecture)  
**Notebook Reference:** `BTech_Project/Notebooks/06_swin_transformer_training.ipynb`

---

## 1. Executive Summary

Phase 6 implements the second core vision backbone of our dual-architecture diagnostic pipeline: **Swin Transformer Tiny (`swin_t`)**. Unlike traditional Vision Transformers (ViTs) that calculate quadratic global self-attention, Swin Transformer computes self-attention within local $7 \times 7$ windows with cross-window shifted connections across successive stages. This provides hierarchical representation modeling across multi-scale skin patches.

Evaluated on the strictly isolated **Test Holdout Set (4,599 images)**, Swin-T achieved **72.86% Top-1 Accuracy**, **87.82% Top-3 Differential Recall**, and **67.93% Macro-F1**.

Crucially, Swin-T demonstrates strong architectural complementarity to ConvNeXt-Tiny (Phase 5), excelling on widespread, diffuse anatomical conditions (e.g. Tinea at **89.1%**, Alopecia at **96.0%**, Melanoma recall at **80.3%**), while ConvNeXt-Tiny excels on fine localized lesion margin textures. This diversity provides the mathematical foundation for the Phase 7 Stacking Ensemble.

---

## 2. Experimental Setup & Swin-T Hyperparameters

| Hyperparameter | Configuration | Algorithmic Rationale |
|:---|:---|:---|
| **Backbone Architecture** | `swin_t` (`IMAGENET1K_V1`) | Hierarchical shifted-window self-attention ($W=7\times 7$) |
| **Input Spatial Resolution** | $384 \times 384 \times 3$ | Matched with ConvNeXt-Tiny for seamless fusion |
| **Batch Size** | $16$ | Consumes ~3.59 GB VRAM on RTX 5050 (peak memory safety) |
| **Total Epochs** | $25$ | Reached complete plateau at Epoch 25 ($165.96$ minutes runtime) |
| **Optimizer** | `AdamW` ($\beta_1=0.9, \beta_2=0.999$) | Decoupled weight decay ($0.05$) standard for vision transformers |
| **Learning Rate Schedule** | SequentialLR | 2-epoch linear warmup ($10^{-6} \to 10^{-4}$) + 23-epoch Cosine Annealing to $10^{-6}$ |
| **Gradient Clipping** | $\text{max\_norm} = 1.0$ | Prevents attention score gradient instability |
| **Loss Function** | Class-Balanced Focal Loss | $\gamma = 2.0$, label smoothing $\epsilon = 0.1$, Cui et al. weights ($\alpha_t$) |
| **Precision** | PyTorch AMP (`torch.amp.autocast`) | Accelerated via RTX 5050 Tensor Cores |
| **DataLoader Workers** | `num_workers = 0` | Guarantees deadlock-free execution under Windows |

---

## 3. Two-Stage Clinical Classification Head

The ImageNet classification head (`Linear(768, 1000)`) was replaced with a specialized bottleneck head:
$$\mathbf{z} = \text{Linear}_{512 \to 25} \Big( \text{Dropout}_{0.2} \big( \text{BatchNorm1d} ( \text{GELU}( \text{Linear}_{768 \to 512} ( \text{Dropout}_{0.3}(\mathbf{x}) ) ) ) \big) \Big)$$
- **Total Parameters:** $27,926,931$ ($106.53$ MB float32)
- **Trainable Parameters:** $27,926,931$ ($100\%$ fine-tuning)

---

## 4. Training Convergence & Generalization Dynamics

Across the 25 epochs:
- **Epoch 01 (Warmup):** Train Loss $1.4898$ | Val Loss $1.3773$ | Val Top-1 $11.93\%$ | Val Macro-F1 $6.80\%$
- **Epoch 05 (Fast Acquisition):** Train Loss $0.7770$ | Val Loss $0.7509$ | Val Top-1 $53.19\%$ | Val Macro-F1 $46.64\%$
- **Epoch 10 (Attention Refinement):** Train Loss $0.4884$ | Val Loss $0.6095$ | Val Top-1 $62.46\%$ | Val Macro-F1 $56.81\%$
- **Epoch 15 (Discriminative Stride):** Train Loss $0.2908$ | Val Loss $0.5569$ | Val Top-1 $69.53\%$ | Val Macro-F1 $65.48\%$
- **Epoch 20 (Deep Spatial Tuning):** Train Loss $0.1851$ | Val Loss $0.5343$ | Val Top-1 $72.38\%$ | Val Macro-F1 $68.01\%$
- **Epoch 23 (Val Loss Global Min):** Train Loss $0.1611$ | Val Loss $\mathbf{0.5146}$ | Val Top-1 $73.34\%$ | Val Macro-F1 $68.41\%$
- **Epoch 25 (Peak Generalization):** Train Loss $0.1526$ | Val Loss $0.5187$ | Val Top-1 $\mathbf{73.69\%}$ | Val Macro-F1 $\mathbf{68.78\%}$ 🏆

**Generalization Gap Audit:**
- Swin-T Train Loss ($0.1526$) vs. Val Loss ($0.5146$) exhibited a gap of $0.362$, noticeably narrower than ConvNeXt-Tiny's gap of $0.448$, confirming Swin-T's shifted-window attention is less prone to local memorization.

---

## 5. Unbiased Evaluation on Isolated Test Holdout ($N = 4,599$)

### 5.1 Head-to-Head Comparative Benchmark
| Metric | ConvNeXt-Tiny (Phase 5) | Swin-Transformer Tiny (Phase 6) | Delta ($\Delta$) | Clinical Significance |
|:---|:---:|:---:|:---:|:---|
| **Top-1 Accuracy** | **73.47%** | $72.86\%$ | $-0.61\%$ | Both architectures achieve strong single-model performance |
| **Top-3 Differential Recall** | $87.61\%$ | **87.82%** | $+0.21\%$ | Correct pathology in top 3 differential in $87.8\%$ of tests |
| **Macro-F1 (Balanced)** | **68.20%** | $67.93\%$ | $-0.27\%$ | Consistent class-balanced sensitivity across 25 diseases |
| **Weighted-F1** | **73.32%** | $72.73\%$ | $-0.59\%$ | Population-weighted diagnostic fidelity |
| **Kaposi Sarcoma Recall** | **100.0%** | **100.0%** | $0.00\%$ | Zero false negatives on lethal immunocompromised malignancy |
| **Malignant Melanoma Recall** | $79.8\%$ | **80.3%** | $+0.50\%$ | Swin-T captures more melanoma cases |
| **Cutaneous Vasculitis Recall** | $69.6\%$ | **71.6%** | $+2.00\%$ | Swin-T improves systemic purpura detection |
| **Tinea & Fungal Infections** | $75.8\%$ | **89.1%** | $+13.30\%$ | Major spatial improvement on diffuse annular erythema |
| **Alopecia Areata** | $93.3\%$ | **96.0%** | $+2.70\%$ | Outstanding pattern recognition on scalp lesions |
| **Atopic Dermatitis (Eczema)**| $71.5\%$ | **74.7%** | $+3.20\%$ | Better distinction of diffuse eczematous plaques |

### 5.2 Critical High-Risk Conditions Audit (Swin-T)
| Disease | Urgency | Precision | Recall | F1-Score | Support |
|:---|:---|:---:|:---:|:---:|:---:|
| **Malignant Melanoma** | Tier 1 (Malignant neoplasm) | **0.789** | **0.803** | **0.796** | 564 |
| **Bullous Pemphigoid** | Tier 1 (Autoimmune blistering) | **0.723** | **0.550** | **0.625** | 109 |
| **Cutaneous Vasculitis** | Tier 1 (Vascular necrotizing) | **0.658** | **0.716** | **0.685** | 102 |
| **Cellulitis** | Tier 1 (Acute bacterial spread) | **0.280** | **0.250** | **0.264** | 56 |
| **Kaposi Sarcoma** | Tier 1 (HIV-related tumor) | **0.800** | **1.000** | **0.889** | 8 |

---

## 6. Artifacts & Deliverables Generated

1. **Model Checkpoints:**
   - `Models/checkpoints/swin_tiny/best_swin_tiny.pth` (112.5 MB)
   - `Models/checkpoints/swin_tiny/last_checkpoint.pth` (112.5 MB)
2. **Telemetry Logs:**
   - `Models/logs/swin_tiny/training_metrics.csv`
   - `Models/logs/swin_tiny/training_logs.json`
   - `Models/logs/swin_tiny/test_classification_report.csv`
3. **Diagnostic Plots:**
   - `results/plots/swin_training/01_swin_tiny_training_curves.png`
   - `results/plots/swin_training/02_swin_tiny_test_confusion_matrix.png`

---

## 7. Next Roadmap: Phase 7 (Stacking Ensemble Fusion) & Standalone XAI

With both distinct backbones trained and validated:
1. **Phase 7 (`07_stacking_ensemble_fusion.ipynb`)**:
   - Extract out-of-fold and holdout probability vectors from ConvNeXt-Tiny and Swin-Transformer Tiny.
   - Train an $L_2$-regularized Logistic Regression / LightGBM meta-learner to fuse both representations.
   - Target: Overcome individual model weaknesses to push overall Top-1 accuracy to $\mathbf{>78-82\%}$.
2. **Standalone XAI Notebook (`12_explainable_ai_gradcam.ipynb`)**:
   - Generates side-by-side Grad-CAM and Swin Attention visual explanation galleries for thesis and publication.

