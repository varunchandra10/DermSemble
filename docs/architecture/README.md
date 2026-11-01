# Dermsemble Deep Learning Architecture Schematics

This directory contains standalone, reproducible Mermaid architecture diagrams for the **Dermsemble** clinical triage engine.

---

## 1. End-to-End Clinical System Pipeline
**File:** [`01_end_to_end_pipeline.mmd`](./01_end_to_end_pipeline.mmd)

```mermaid
graph TD
    classDef input fill:#eff6ff,stroke:#3b82f6,stroke-width:2px,color:#1e3a8a
    classDef prep fill:#f0fdf4,stroke:#22c55e,stroke-width:2px,color:#14532d
    classDef model fill:#faf5ff,stroke:#a855f7,stroke-width:2px,color:#581c87
    classDef fusion fill:#fff7ed,stroke:#f97316,stroke-width:2px,color:#7c2d12
    classDef safety fill:#fef2f2,stroke:#ef4444,stroke-width:2px,color:#7f1d1d
    classDef output fill:#f0fdfa,stroke:#14b8a6,stroke-width:2px,color:#134e4a

    subgraph STAGE1 ["1. Clinical Ingestion & Optical Standardization"]
        A["Patient Skin Photograph<br/>(RGB Raw Ingestion)"]:::input --> B["DullRazor Hair Inpainting<br/>(4-Dir Kernels + Telea)"]:::prep
        B --> C["Minkowski p=6 Shades-of-Gray<br/>(Color Constancy Norm)"]:::prep
        C --> D["Standardized Tensor<br/>(384 x 384 x 3 Native)"]:::prep
    end

    subgraph STAGE2 ["2. Dual INT8 ONNX Deep Learning Engines"]
        D -->|High-Frequency Texture| E["ConvNeXt-Tiny INT8<br/>(28.5 MB, 7x7 Depthwise CNN)"]:::model
        D -->|Global Plaque Asymmetry| F["Swin-Transformer INT8<br/>(28.7 MB, Shifted Window ViT)"]:::model
        E --> G["Probability Vector P_conv<br/>(25 Classes, Calibrated)"]:::model
        F --> H["Probability Vector P_swin<br/>(25 Classes, Calibrated)"]:::model
    end

    subgraph STAGE3 ["3. Decision-Theoretic Calibration Engine"]
        G --> I["Calibrated Soft Voting<br/>P_ens = 0.42 * P_conv + 0.58 * P_swin"]:::fusion
        H --> I
        I --> J["Melanoma Acuity Multiplier<br/>alpha = 1.85 on Tier 1 Cancer"]:::fusion
        J --> K{"Uncertainty Gate<br/>max(P) >= 0.50 ?"}:::safety
    end

    subgraph STAGE4 ["4. Clinical Triage & Visual Attribution"]
        K -->|Conf >= 0.50| L["Automated Triage<br/>Tier 1 Urgent vs Tier 2 Routine"]:::output
        K -->|Conf < 0.50| M["Clinical Abstention<br/>Biopsy & Dermoscopy Advised"]:::safety
        E -.->|Conv Layer 4| N["Dual-Stream Grad-CAM<br/>Cross-Attribution Service"]:::output
        F -.->|Norm Layer| N
        N --> O["Transparent Saliency Heatmap<br/>Spatial Evidence Overlay"]:::output
    end
```

---

## 2. Dual Backbone Synergy (CNN vs. Vision Transformer)
**File:** [`02_dual_backbone_synergy.mmd`](./02_dual_backbone_synergy.mmd)

```mermaid
graph LR
    classDef conv fill:#eff6ff,stroke:#2563eb,stroke-width:2px,color:#1e40af
    classDef swin fill:#faf5ff,stroke:#9333ea,stroke-width:2px,color:#6b21a8
    classDef fusion fill:#ecfdf5,stroke:#059669,stroke-width:2px,color:#065f46

    subgraph CONVNEXT ["ConvNeXt-Tiny (CNN Backbone - 42% Weight)"]
        C1["Input 384x384"]:::conv --> C2["Stem: 4x4 Patchify"]:::conv
        C2 --> C3["4 ResNet-style Stages<br/>7x7 Depthwise Convolutions"]:::conv
        C3 --> C4["Inverted Bottleneck + GELU"]:::conv
        C4 --> C5["Local Margin Texture<br/>Follicular Plugs & Networks"]:::conv
    end

    subgraph SWIN ["Swin-Transformer Tiny (ViT Backbone - 58% Weight)"]
        S1["Input 384x384"]:::swin --> S2["Patch Partition: 4x4"]:::swin
        S2 --> S3["Hierarchical Transformer Stages"]:::swin
        S3 --> S4["Shifted Window Self-Attention<br/>W=7x7 Local + Cross-Window"]:::swin
        S4 --> S5["Global Lesion Asymmetry<br/>Erythematous Halos & Plaque Spread"]:::swin
    end

    C5 --> M["Soft-Voting Ensemble<br/>0.42 * CNN + 0.58 * ViT<br/>Top-1: 75.52% | Top-3: 89.28%"]:::fusion
    S5 --> M
```

---

## 3. Decision Safety Policies & Clinical Triage Tree
**File:** [`03_decision_safety_policies.mmd`](./03_decision_safety_policies.mmd)

```mermaid
graph TD
    classDef start fill:#eff6ff,stroke:#3b82f6,stroke-width:2px,color:#1e3a8a
    classDef math fill:#fefce8,stroke:#ca8a04,stroke-width:2px,color:#713f12
    classDef tier1 fill:#fee2e2,stroke:#ef4444,stroke-width:2px,color:#991b1b
    classDef tier2 fill:#e0e7ff,stroke:#6366f1,stroke-width:2px,color:#3730a3
    classDef abstain fill:#ffedd5,stroke:#f97316,stroke-width:2px,color:#9a3412

    A["Raw Ensemble Softmax Distribution P"]:::start --> B["Check Tier 1 Oncology Classes<br/>(Melanoma, Basal Cell, Squamous Cell)"]:::math
    B --> C["Apply Acuity Multiplier: P_mel = P_mel * 1.85<br/>Re-normalize Probability Simplex"]:::math
    C --> D{"Evaluate Confidence: max(P) >= 0.50 ?"}:::math

    D -->|Above Threshold: max >= 0.50| E{"Is Predicted Class Tier 1 Oncology?"}:::math
    E -->|Yes| F["RED FLAG: Tier 1 Oncology<br/>Immediate Biopsy & Dermoscopy Protocol"]:::tier1
    E -->|No| G["BLUE FLAG: Tier 2 Outpatient<br/>Routine Clinical Management"]:::tier2

    D -->|Below Threshold: max < 0.50| H["AMBER FLAG: Clinical Abstention<br/>Uncertain Diagnostic Boundary<br/>Automated Abstention Triggered (Acc: 85.11%)"]:::abstain
```

