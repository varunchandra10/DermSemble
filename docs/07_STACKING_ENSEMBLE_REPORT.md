# PHASE 7: DUAL-ARCHITECTURE STACKING ENSEMBLE FUSION REPORT

**Project:** AI-Based Tool for Preliminary Diagnosis of Dermatological Manifestations  
**Date:** November 1, 2026  
**Status:** COMPLETE & EMPIRICALLY VERIFIED  
**Compute Architecture:** NVIDIA GeForce RTX 5050 Laptop GPU (8.55 GB Dedicated VRAM)  
**Notebook Reference:** `BTech_Project/Notebooks/07_stacking_ensemble_fusion.ipynb`

---

## 1. Executive Summary

Phase 7 realizes the central hypothesis of this project: **orthogonal multi-modal vision synthesis**. Individual vision models exhibit inherent blind spots—convolutional networks excel at localized micro-textures (follicular plugging, sharp pigmented borders), while vision transformers excel at global spatial context and diffuse erythema distribution. 

By fusing **ConvNeXt-Tiny** ($323.3$ MB) and **Swin-Transformer Tiny** ($320.0$ MB) via an optimal weighted meta-learner ($w^* = 0.42 \cdot \text{ConvNeXt} + 0.58 \cdot \text{Swin-T}$), the ensemble was evaluated on the strictly isolated **Test Holdout Set (4,599 images)**.

The Champion Ensemble achieved:
- **Top-1 Diagnostic Accuracy:** **75.52%** ($+2.05\%$ gain over the best single model)
- **Top-3 Differential Recall:** **89.28%** ($+1.46\%$ gain, reaching nearly $90\%$)
- **Macro-F1 (All 25 Classes):** **70.55%** ($+2.35\%$ gain across all diagnostic categories)
- **Weighted-F1:** **75.36%** ($+2.04\%$ gain)

Crucially, the ensemble delivered significant accuracy leaps on lethal and acute clinical pathologies, elevating **Malignant Melanoma recall to 81.9%** ($F_1 = 0.811$), **Cutaneous Vasculitis recall to 74.5%**, and maintaining **100% sensitivity on Kaposi Sarcoma**.

---

## 2. Fusion Methodology & Experimental Setup

Four distinct fusion strategies were fitted strictly on the **Validation Set ($N = 3,679$)** with zero test set exposure, and benchmarked on the **Test Holdout Set ($N = 4,599$)**:

| Strategy | Formulation | Test Top-1 | Test Top-3 | Test Macro-F1 | Status |
|:---|:---|:---:|:---:|:---:|:---|
| **ConvNeXt-Tiny (Baseline 1)** | Single convolutional backbone | $73.47\%$ | $87.61\%$ | $68.20\%$ | Baseline |
| **Swin-Transformer (Baseline 2)**| Single shifted-window transformer | $72.86\%$ | $87.82\%$ | $67.93\%$ | Baseline |
| **Strategy 1: Equal Soft Voting** | $P = 0.50 P_{\text{conv}} + 0.50 P_{\text{swin}}$ | $75.43\%$ | $89.37\%$ | $70.59\%$ | Competitive |
| **Strategy 2: Optimal Weighted Voting** | $P = 0.42 P_{\text{conv}} + 0.58 P_{\text{swin}}$ | **75.52%** | **89.28%** | **70.55%** | 🏆 **Champion** |
| **Strategy 3: Probability Stacking (LR)** | $L_2$-Logistic Regression on $[P_{\text{conv}}, P_{\text{swin}}]$ | $74.76\%$ | $87.17\%$ | $65.32\%$ | Over-regularized |
| **Strategy 4: Logits Stacking (LR)** | $L_2$-Logistic Regression on $[z_{\text{conv}}, z_{\text{swin}}]$ | $74.95\%$ | $89.02\%$ | $69.73\%$ | Strong |

### Why Weighted Soft Voting Won:
Deep neural networks with label smoothing and temperature scaling produce well-calibrated posterior probabilities. Linear blending with $w^* = 0.42$ preserves the probability geometry of both models without adding redundant parametric complexity that might overfit small minority categories.

---

## 3. Clinical Acuity Tier Performance Gains

The ensemble yielded substantial improvements across both clinical urgency tiers:

| Clinical Tier | Definition & Acuity | Single Model Best Acc | Ensemble Test Acc | Net Improvement ($\Delta$) |
|:---|:---|:---:|:---:|:---:|
| **Tier 1 (High Acuity / Critical)** | Melanoma, Kaposi, Vasculitis, Cellulitis, Lupus ($N = 1,317$) | $67.73\%$ | **69.78%** | **+2.05%** |
| **Tier 2 (Chronic / Non-Urgent)** | Eczema, Psoriasis, Acne, Fungal, Warts ($N = 3,282$) | $75.78\%$ | **77.82%** | **+2.04%** |

---

## 4. Critical High-Risk Pathologies Audit

| Disease Category | Acuity | Precision | Recall (Sensitivity) | F1-Score | Support | Single Best Recall |
|:---|:---|:---:|:---:|:---:|:---:|:---:|
| **Malignant Melanoma** | Tier 1 | **0.802** | **0.819 (81.9%)** | **0.811** | 564 | $80.3\%$ ($+1.6\%$ gain) |
| **Kaposi Sarcoma** | Tier 1 | **0.889** | **1.000 (100.0%)**| **0.941** | 8 | $100.0\%$ (Zero misses) |
| **Cutaneous Vasculitis** | Tier 1 | **0.650** | **0.745 (74.5%)** | **0.694** | 102 | $71.6\%$ ($+2.9\%$ gain) |
| **Bullous Pemphigoid** | Tier 1 | **0.753** | **0.560 (56.0%)** | **0.642** | 109 | $56.9\%$ ($+0.017$ F1 gain) |
| **Cellulitis** | Tier 1 | **0.364** | **0.286 (28.6%)** | **0.320** | 56 | $25.0\%$ ($+3.6\%$ gain) |

---

## 5. Normalized Confusion Matrix Analysis

The final $25 \times 25$ confusion matrix (`results/plots/ensemble/02_champion_ensemble_confusion_matrix.png`) demonstrates exceptional diagonal dominance across the majority of conditions:

- **Top-Tier Separation ($>80\%$ Accuracy):**
  - Alopecia Areata: **$97.3\%$**
  - Melanocytic Nevi: **$90.8\%$**
  - Onychomycosis & Nail Diseases: **$82.5\%$**
  - Malignant Melanoma: **$81.9\%$**
  - Acne Vulgaris: **$81.5\%$**
- **Strong Category Separation ($70\% - 80\%$ Accuracy):**
  - Seborrheic Keratosis: **$77.0\%$**
  - Atopic Dermatitis / Eczema: **$76.1\%$**
  - Psoriasis and Lichen Planus: **$76.1\%$**
  - Tinea and Fungal Infections: **$75.8\%$**
  - Cutaneous Vasculitis: **$74.5\%$**
  - Herpes HPV and STDs: **$74.2\%$**
  - Viral Warts and Molluscum: **$73.8\%$**
  - Vascular Tumors: **$73.5\%$**
  - Rosacea: **$71.4\%$**
  - Kaposi Sarcoma: **$100.0\%$**

---

## 6. Artifacts & Deliverables Generated

1. **Champion Ensemble Checkpoint:**
   `Models/ensemble/champion_ensemble.pkl` (Contains weights $w_{\text{conv}}=0.42, w_{\text{swin}}=0.58$, winning probability vectors, and metadata)
2. **Probability Caches:**
   - `Models/ensemble/val_predictions.npz` (Compressed multi-model validation tensors)
   - `Models/ensemble/test_predictions.npz` (Compressed multi-model holdout tensors)
3. **Clinical Classification Report:**
   `Models/ensemble/champion_ensemble_classification_report.csv`
4. **Publication Plots:**
   - `results/plots/ensemble/01_ensemble_performance_comparison.png` (2-panel strategy comparison and per-class gain chart)
   - `results/plots/ensemble/02_champion_ensemble_confusion_matrix.png` (Full $25 \times 25$ normalized heatmap)

---

## 7. Next Roadmap: Phase 8 (Threshold Tuning & Clinical Abstention)

Now that the ensemble has reached **75.52% Top-1 Accuracy** and **89.28% Top-3 Recall**, we proceed to **Notebook 08 (`08_threshold_tuning_abstention.ipynb`)**:
- **Goal:** Optimize decision thresholds via Nelder-Mead on Tier 1 conditions to push Melanoma sensitivity to $\mathbf{>90\%}$, while introducing a clinical **Abstention / "Refer to Dermatologist"** uncertainty mechanism when ensemble confidence is below diagnostic certainty.

