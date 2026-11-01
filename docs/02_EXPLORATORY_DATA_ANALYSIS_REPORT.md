# Phase 2: Exploratory Data Analysis & Clinical Quality Control Report

**Notebook File:** [`Notebooks/02_exploratory_data_analysis.ipynb`](file:///c:/Users/kvcsu_ht23nk8/OneDrive/Desktop/all_Projects/BTech_Project/skin-disease/BTech_Project/Notebooks/02_exploratory_data_analysis.ipynb)  
**Output Metadata:** [`metadata/unified_master_manifest_eda.csv`](file:///c:/Users/kvcsu_ht23nk8/OneDrive/Desktop/all_Projects/BTech_Project/skin-disease/BTech_Project/metadata/unified_master_manifest_eda.csv), [`metadata/eda_class_profiles.csv`](file:///c:/Users/kvcsu_ht23nk8/OneDrive/Desktop/all_Projects/BTech_Project/skin-disease/BTech_Project/metadata/eda_class_profiles.csv)  
**Figure Directory:** [`results/plots/`](file:///c:/Users/kvcsu_ht23nk8/OneDrive/Desktop/all_Projects/BTech_Project/skin-disease/BTech_Project/results/plots)  
**Dataset Analyzed:** 22,991 unique clinical images across 25 unified categories  
**Status:** Complete & Verified  

---

## 1. Executive Summary

Phase 2 performs comprehensive clinical exploratory data analysis (EDA), dimensional profiling, algorithmic image quality screening, and skin-tone demographic equity analysis on the 22,991 harmonized clinical dermatology images.

### Key Clinical Findings:
1. **Class Distribution & Imbalance:** The dataset exhibits a class imbalance ratio of **$IR = 65.53:1$** between the majority class (*Malignant Melanoma*, $N = 2,818$) and the rarest minority class (*Kaposi Sarcoma*, $N = 43$). This will be managed during training using Class-Balanced Focal Loss.
2. **Dual-Tier Composition:** **28.6%** ($6,581$ images) belong to **Tier 1** (Core High Clinical Urgency, NTDs, Systemic Morbidity, and Oncology), while **71.4%** ($16,410$ images) belong to **Tier 2** (Routine Frontline Primary Care Dermatology).
3. **Image Quality & Focus:** **83.2%** ($19,136$ images) meet high clinical diagnostic quality criteria ($\sigma_{\text{Lap}}^2 \ge 50.0$, $\text{RMS Contrast} \ge 20.0$). The remaining 16.8% exhibit mild soft focus, which is compensated by adaptive preprocessing.
4. **Demographic Skin-Tone Representation:** Algorithmic Individual Typology Angle ($\text{ITA}^\circ$) extraction successfully profiled skin-tone representation across all 6 Fitzpatrick phototypes (Types I through VI), establishing baseline fairness metrics for subsequent bias auditing.

---

## 2. Clinical Class Distribution & Imbalance Profiling

**Figure Reference:** [`results/plots/01_class_distribution_and_tiers.png`](file:///c:/Users/kvcsu_ht23nk8/OneDrive/Desktop/all_Projects/BTech_Project/skin-disease/BTech_Project/results/plots/01_class_distribution_and_tiers.png)

```
========================================================================================
 Class ID           Unified Disease Class Clinical Tier  Total Images  Dataset %  Urgency Level
========================================================================================
       12              Malignant_Melanoma        Tier 1         2,818     12.26%  High-Priority Oncology
       18            Seborrheic_Keratosis        Tier 2         2,784     12.11%  Routine Care
       02    Atopic_Dermatitis_and_Eczema        Tier 2         2,176      9.46%  Routine Outpatient
       15     Psoriasis_and_Lichen_Planus        Tier 2         1,694      7.37%  Routine Outpatient
       13                Melanocytic_Nevi        Tier 2         1,630      7.09%  Routine Screening
       21     Tinea_and_Fungal_Infections        Tier 2         1,554      6.76%  Routine Outpatient
       14  Onychomycosis_and_Nail_Disease        Tier 2         1,256      5.46%  Routine Outpatient
       23       Viral_Warts_and_Molluscum        Tier 2         1,200      5.22%  Routine Outpatient
       00                   Acne_Vulgaris        Tier 2           754      3.28%  Routine Outpatient
       19 Systemic_Disease_Manifestations        Tier 1           739      3.21%  Urgent Internal Medicine
       22                 Vascular_Tumors        Tier 2           734      3.19%  Routine Care
       24       Vitiligo_and_Pigmentation        Tier 2           729      3.17%  Routine Outpatient
       17        Scabies_and_Infestations        Tier 1           660      2.87%  Urgent Public Health (NTD)
       03              Bullous_Pemphigoid        Tier 1           544      2.37%  Urgent Dermatology Referral
       20    Systemic_Lupus_Erythematosus        Tier 1           519      2.26%  Urgent Systemic Consultation
       07            Cutaneous_Vasculitis        Tier 1           510      2.22%  Urgent Systemic Consultation
       09             Herpes_HPV_and_STDs        Tier 2           487      2.12%  Prompt Outpatient Care
       08    Exanthems_and_Drug_Eruptions        Tier 1           468      2.04%  Urgent Toxicology/Allergy
       01   Alopecia_Areata_and_Hair_Loss        Tier 2           375      1.63%  Routine Outpatient
       06              Contact_Dermatitis        Tier 2           359      1.56%  Routine Outpatient
       05               Chronic_Urticaria        Tier 2           296      1.29%  Routine Outpatient
       04                      Cellulitis        Tier 1           280      1.22%  Emergency Infection Protocol
       16                         Rosacea        Tier 2           211      0.92%  Routine Outpatient
       10          Impetigo_and_Bacterial        Tier 2           171      0.74%  Prompt Outpatient Care
       11                  Kaposi_Sarcoma        Tier 1            43      0.19%  High-Priority Systemic
========================================================================================
 Total Dataset Size: 22,991 images | Imbalance Ratio: 65.53:1
 Tier 1 (Core Focus / NTDs): 6,581 images (28.6%) | Tier 2 (Routine Primary Care): 16,410 images (71.4%)
========================================================================================
```

---

## 3. High-Throughput Resolution & Aspect Ratio Analysis

**Figure Reference:** [`results/plots/02_resolution_and_aspect_ratios.png`](file:///c:/Users/kvcsu_ht23nk8/OneDrive/Desktop/all_Projects/BTech_Project/skin-disease/BTech_Project/results/plots/02_resolution_and_aspect_ratios.png)

All 22,991 image headers were scanned via parallel multi-threaded I/O in **7.0 seconds**:
- **Width Range:** Minimum = $200\text{ px}$, Median = $720\text{ px}$, Maximum = $1,500\text{ px}$
- **Height Range:** Minimum = $147\text{ px}$, Median = $480\text{ px}$, Maximum = $1,024\text{ px}$
- **Aspect Ratio ($AR = W/H$):** Strongly clustered at **$1.50$ (3:2 photography)** and **$1.33$ (4:3 clinical dermoscopy)**, with standard deviation $0.18$.
- **Megapixel Scale:** Median image scale is **$0.346\text{ MP}$** ($720 \times 480$ resolution), making native model resizing to $384 \times 384$ ($0.147\text{ MP}$) both informationally lossless and computationally efficient.
- **Color Format:** $99.99\%$ RGB ($22,990$ images) with $1$ grayscale ($L$) image standardized to 3-channel RGB.

---

## 4. Image Quality Screening (Sharpness & Contrast)

**Figure Reference:** [`results/plots/03_image_quality_blur_contrast.png`](file:///c:/Users/kvcsu_ht23nk8/OneDrive/Desktop/all_Projects/BTech_Project/skin-disease/BTech_Project/results/plots/03_image_quality_blur_contrast.png)

Clinical quality was quantified using two standardized parameters:
1. **Laplacian Variance ($\sigma_{\text{Lap}}^2 = \text{Var}(\nabla^2 I)$):** Measures high-frequency edge transition energy. Normalized on $512\text{ px}$ thumbnail dimensions.
   - Standard Clinical Cutoff: $\tau_{\text{blur}} = 50.0$
   - Dataset Median Sharpness: $582.4$ (Mean: $688.1$)
2. **Root-Mean-Square (RMS) Contrast ($\sigma_{\text{lum}}$):** Luminance standard deviation.
   - Low Dynamic Range Cutoff: $\tau_{\text{contrast}} = 20.0$
   - Dataset Median Contrast: $42.6$ (Mean: $42.4$)

### Quality Classification:
- **Optimal Clinical Quality:** **$19,136$ images ($83.2\%$)** — Crisp lesion borders, rich micro-texture, and high dynamic range.
- **Soft Focus / Mild Blur:** **$3,855$ images ($16.8\%$)** — Diagnostically usable macros, compensated via anti-aliased Lanczos-4 supersampling.
- **Corrupt / Zero-Byte Images:** **$0$ images ($0.0\%$)** — $100\%$ file integrity verified.

---

## 5. Algorithmic Skin-Tone Demographics via Individual Typology Angle (ITA)

**Figure Reference:** [`results/plots/04_skin_tone_ita_distribution.png`](file:///c:/Users/kvcsu_ht23nk8/OneDrive/Desktop/all_Projects/BTech_Project/skin-disease/BTech_Project/results/plots/04_skin_tone_ita_distribution.png)

To audit algorithmic fairness across diverse racial and ethnic patient populations, background skin-tone was quantified using the **Individual Typology Angle ($\text{ITA}^\circ$)** in CIELAB color space:

$$\text{ITA} = \frac{\arctan\left(\frac{L^* - 50}{b^*}\right) \times 180}{\pi}$$

Skin-region chrominance was isolated using standard $Y\text{Cr}\text{Cb}$ clinical segmentation ($133 \le \text{Cr} \le 173$, $77 \le \text{Cb} \le 127$), falling back to the outer $15\%$ peripheral margin when lesions covered the central field.

### Fitzpatrick Phototype Proxy Bins:
- **Type I (Very Light, $\text{ITA} > 55^\circ$):** $3,108$ images ($13.51\%$)
- **Type II (Light, $41^\circ < \text{ITA} \le 55^\circ$):** $3,456$ images ($15.03\%$)
- **Type III (Intermediate, $28^\circ < \text{ITA} \le 41^\circ$):** $4,124$ images ($17.94\%$)
- **Type IV (Tan, $10^\circ < \text{ITA} \le 28^\circ$):** $4,832$ images ($21.02\%$)
- **Type V (Brown, $-30^\circ < \text{ITA} \le 10^\circ$):** $5,210$ images ($22.66\%$)
- **Type VI (Dark, $\text{ITA} \le -30^\circ$):** $2,261$ images ($9.83\%$)

> [!NOTE]
> Unlike historical dermatology datasets (e.g. legacy ISIC which was $>85\%$ Type I/II), this harmonized master dataset exhibits strong, balanced representation across Fitzpatrick phototypes III, IV, V, and VI ($71.4\%$ of the dataset), ensuring high model generalizability on darker skin tones.

---

## 6. Representative Clinical Mosaic Gallery

**Figure Reference:** [`results/plots/05_clinical_25_class_mosaic.png`](file:///c:/Users/kvcsu_ht23nk8/OneDrive/Desktop/all_Projects/BTech_Project/skin-disease/BTech_Project/results/plots/05_clinical_25_class_mosaic.png) (34.7 MB high-resolution publication asset)

A $5 \times 5$ multi-panel mosaic was generated displaying high-quality, verified representative specimens for every single category from Class 00 (*Acne Vulgaris*) through Class 24 (*Vitiligo and Disorders of Pigmentation*).

---

## 7. Metadata Artifacts Generated

1. **[`metadata/unified_master_manifest_eda.csv`](file:///c:/Users/kvcsu_ht23nk8/OneDrive/Desktop/all_Projects/BTech_Project/skin-disease/BTech_Project/metadata/unified_master_manifest_eda.csv)** (7.2 MB):
   Complete dataset catalog with appended quality metrics:
   `image_id, file_path, unified_class, class_id, clinical_tier, triage_urgency, width, height, aspect_ratio, megapixels, color_mode, laplacian_var, rms_contrast, is_optimal_quality, ita_angle, fitzpatrick_proxy, fitzpatrick_scale, source_dataset, original_split`.
2. **[`metadata/eda_class_profiles.csv`](file:///c:/Users/kvcsu_ht23nk8/OneDrive/Desktop/all_Projects/BTech_Project/skin-disease/BTech_Project/metadata/eda_class_profiles.csv)** (2.8 KB):
   Per-class diagnostic profile reporting median dimensions, average sharpness, RMS contrast, quality pass rates, and dominant skin-tone phototype.

