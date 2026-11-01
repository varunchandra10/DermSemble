# Notebooks Directory Guide

This folder contains the sequential, reproducible implementation notebooks for the **DermaVision-AI: Dual-Architecture Deep Ensembling & Production Edge Deployment for Dermatological Manifestations**.

---

## Notebook Index & Execution Roadmap

| Notebook | Phase | Focus / Key Outputs | Status |
| :--- | :--- | :--- | :---: |
| **[`01_dataset_harmonization.ipynb`](./01_dataset_harmonization.ipynb)** | **Phase 1** | Ingested DERMNet & Clinical MultiSource (35,389 raw images). Applied 2-stage MD5 deduplication (purged 12,398 duplicates). Built canonical 25-class master manifest (`metadata/unified_master_manifest.csv`, 22,991 images). | ✅ Complete & Verified |
| **[`02_exploratory_data_analysis.ipynb`](./02_exploratory_data_analysis.ipynb)** | **Phase 2** | Audited long-tail imbalance ($IR = 65.5:1$), geometry clustering, Laplacian blur, RMS contrast, and algorithmic skin-tone demographics via Individual Typology Angle ($\text{ITA}^\circ$). Produced 5 publication-grade figures. | ✅ Complete & Verified |
| **[`03_clinical_preprocessig_qc.ipynb`](./03_clinical_preprocessig_qc.ipynb)** | **Phase 3** | Clinical preprocessing: Multi-directional **DullRazor** (thin-hair inpainting without lesion blur) + **Shades-of-Gray** ($p=6$ color constancy neutralizing lighting casts) + Native $384 \times 384$ clinical standardization. Generated 1.21 GB offline cache of 22,991 images in 9.36 mins. | ✅ Complete & Verified |
| **[`04_stratified_cv_partitioning.ipynb`](./04_stratified_cv_partitioning.ipynb)** | **Phase 4** | Pinned `SEED=42`. 80/20 train/test holdout split (**4,599 test images strictly isolated**) + 5-fold stratified cross-validation preserving class distribution, clinical tiers, and Fitzpatrick skin tones across all folds. | ✅ Complete & Verified |
| **[`05_convNext_tiny_training.ipynb`](./05_convNext_tiny_training.ipynb)** | **Phase 5** | Pretrained ConvNeXt-Tiny with regularized clinical head. Optimized via Class-Balanced Focal Loss ($\gamma=2.0$, $\beta=0.999$, weights $0.25\times$ to $7.61\times$). 25 epochs in 124.43 mins on RTX 5050. **Test Top-1: 73.47%, Top-3: 87.61%, Macro-F1: 68.20%**. | ✅ Complete & Verified |
| **[`06_swin_transformer_training.ipynb`](./06_swin_transformer_training.ipynb)** | **Phase 6** | Pretrained Swin-Transformer Tiny with shifted window self-attention ($W=7\times 7$, 384x384 native). 25 epochs in 165.96 mins on RTX 5050. **Test Top-1: 72.86%, Top-3: 87.82%, Macro-F1: 67.93%**. Strong on diffuse spread: Tinea (89.1%) and Alopecia (96.0%). | ✅ Complete & Verified |
| **[`07_stacking_ensemble_fusion.ipynb`](./07_stacking_ensemble_fusion.ipynb)** | **Phase 7** | Compared 4 late-fusion strategies. Champion: **Optimal Weighted Soft Voting ($0.42 \cdot \text{ConvNeXt} + 0.58 \cdot \text{Swin-T}$)**. Achieved super-additive performance: **Test Top-1: 75.52% (+2.05%), Top-3: 89.28%, Macro-F1: 70.55%, Melanoma Sensitivity: 81.9%**. | ✅ Complete & Verified |
| **[`08_threshold_tuning_abstention.ipynb`](./08_threshold_tuning_abstention.ipynb)** | **Phase 8** | Acuity-Aware Safety Tuning ($\alpha=1.85, \tau=0.541$) boosted **Melanoma Sensitivity from 81.91% to 86.17% (+24 lethal cancers caught)**. Clinical Uncertainty Abstention ($\theta=0.50$) achieved **85.11% accuracy on automated cases (+9.59% boost)** with 20.4% referred to biopsy. | ✅ Complete & Verified |
| **[`09_fairness_audit_subgroup_analysis.ipynb`](./09_fairness_audit_subgroup_analysis.ipynb)** | **Phase 9** | Comprehensive demographic audit across all 6 Fitzpatrick phototypes ($N=4,599$, Types V-VI = $39.4\%$). **Disparate Impact Ratio = 89.39%** (EEOC regulatory pass $\ge 80\%$). **Melanoma Equal Opportunity Diff = 4.80%**, cancer sensitivity delta between light and dark skin tones = **2.06%**. | ✅ Complete & Verified |
| **[`10_onnx_export_and_quantization.ipynb`](./10_onnx_export_and_quantization.ipynb)** | **Phase 10** | Exported FP32 ONNX backbones (bit-exact max error $< 1.02 \times 10^{-5}$). Applied Dynamic INT8 Quantization: Slashed storage from **643.3 MB to 57.2 MB (91.1% space saved, 11.3x compression)** while retaining **99.7% of accuracy (75.29%)**. GPU latency: **7.2 ms (139.3 FPS)**, CPU latency: **440.3 ms**. | ✅ Complete & Verified |
| **[`11_explainable_ai_gradcam.ipynb`](./11_explainable_ai_gradcam.ipynb)** | **Phase 11** | Standalone dual Grad-CAM hook engine across ConvNeXt stage 7 and Swin-T stage 7. Proved orthogonal visual specialization (ConvNeXt local texture vs Swin global context, spatial corr $r=0.0538$ during error correction). Saved 4-panel case studies and master publication grid. | ✅ Complete & Verified |

---

## Detailed Technical Documentation Index
- [Phase 1: Multi-Source Harmonization & Deduplication Report](../docs/01_DATASET_HARMONIZATION_REPORT.md)
- [Phase 2: Exploratory Data Analysis & Quality Control Report](../docs/02_EXPLORATORY_DATA_ANALYSIS_REPORT.md)
- [Phase 3: Clinical Preprocessing & Illumination Standardization Report](../docs/03_CLINICAL_PREPROCESSING_REPORT.md)
- [Phase 4: Stratified Partitioning & Cross-Validation Report](../docs/04_STRATIFIED_PARTITIONING_REPORT.md)
- [Phase 5: ConvNeXt-Tiny Deep Feature Training Report](../docs/05_CONVNEXT_TINY_TRAINING_REPORT.md)
- [Phase 6: Swin-Transformer Attention Backbone Training Report](../docs/06_SWIN_TRANSFORMER_TRAINING_REPORT.md)
- [Phase 7: Stacking Ensemble & Probability Fusion Report](../docs/07_STACKING_ENSEMBLE_REPORT.md)
- [Phase 8: Threshold Tuning & Clinical Abstention Report](../docs/08_THRESHOLD_TUNING_ABSTENTION_REPORT.md)
- [Phase 9: Algorithmic Demographic Fairness & Subgroup Audit Report](../docs/09_FAIRNESS_AUDIT_REPORT.md)
- [Phase 10: Production ONNX Export & INT8 Quantization Report](../docs/10_ONNX_EXPORT_AND_QUANTIZATION_REPORT.md)
- [Phase 11: Standalone Explainable AI & Dual Grad-CAM Report](../docs/11_EXPLAINABLE_AI_REPORT.md)
- [Comprehensive Re-Implementation Blueprint & Scientific Roadmap](../docs/RE_IMPLEMENTATION_PLAN.md)
