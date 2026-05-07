# 🧬 TCGA-BRCA Transcriptomic Subtype Discovery and Explainable AI Prognostic Modeling

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![Cancer Genomics](https://img.shields.io/badge/Domain-Computational_Oncology-purple)
![Machine Learning](https://img.shields.io/badge/AI-Explainable_ML-red)
![TCGA](https://img.shields.io/badge/Dataset-TCGA--BRCA-green)
![Status](https://img.shields.io/badge/Status-Research_Grade-success)

> A research-grade computational oncology framework for transcriptomic subtype reconstruction, differential expression analysis, survival modeling, and explainable AI-based biomarker discovery using TCGA-BRCA RNA-seq data.

---

# 📖 Overview

Breast cancer is a highly heterogeneous disease composed of multiple molecular states associated with distinct biological programs, therapeutic vulnerabilities, and clinical outcomes. Transcriptomic profiling has enabled the identification of intrinsic subtype systems such as PAM50, which remain foundational in precision oncology.

This project presents an end-to-end computational transcriptomics pipeline for:

- unsupervised molecular subtype discovery
- survival-associated stratification
- transcriptomic biomarker identification
- pathway-level biological interpretation
- explainable machine learning classification

Using RNA-seq expression data from the TCGA-BRCA cohort, the framework reconstructs biologically coherent subtype structures corresponding to Luminal A-like, Luminal B-like, and stromal/normal-like transcriptional programs.

The project integrates:
- statistical genomics
- survival analysis
- dimensionality reduction
- clustering
- explainable machine learning
- latent-space deep learning

within a fully reproducible research workflow.

---

# 🎯 Research Objectives

1. Reconstruct intrinsic breast cancer molecular subtypes directly from transcriptomic data using unsupervised learning.
2. Identify subtype-associated differential gene expression programs.
3. Evaluate survival trends across discovered transcriptomic states.
4. Develop interpretable machine learning models for subtype prediction.
5. Prioritize biologically meaningful biomarkers and therapeutic targets.
6. Establish a reproducible computational oncology framework for future multi-omics expansion.

---

# 🧪 Dataset Description

| Parameter | Description |
|---|---|
| Cohort | TCGA-BRCA |
| Source | UCSC Xena |
| Patients | 1,214 |
| Genes | 20,530 |
| Data Type | Bulk RNA-seq (HiSeqV2 log2-RSEM) |
| Clinical Data | Overall Survival (OS), Progression-Free Interval (PFI) |
| Clinical Resource | Liu et al. 2018 Pan-Cancer Survival Dataset |

---

# 🧠 Computational Pipeline

```text
TCGA-BRCA RNA-seq + Clinical Metadata
                │
                ▼
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. Data Preprocessing & QC
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Clinical-expression sample matching
• Missing value handling
• Low-variance gene filtering
• Z-score normalization
• High-dimensional matrix generation

                │
                ▼
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
2. Dimensionality Reduction
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Median Absolute Deviation (MAD) filtering
• Top 5000 variable genes
• Principal Component Analysis (PCA)

                │
                ▼
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
3. Unsupervised Clustering
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Repeated K-Means optimization
• k = 2–6 evaluation
• Silhouette scoring
• Cluster assignment

                │
                ▼
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
4. Survival Analysis
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Kaplan-Meier estimation
• Log-rank testing
• Pairwise subtype comparison
• Nelson-Aalen cumulative hazard

                │
                ▼
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
5. Differential Expression Analysis
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• One-vs-rest Welch t-tests
• Benjamini-Hochberg FDR correction
• Volcano visualization
• DEG prioritization

                │
                ▼
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
6. Biological Interpretation
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• PAM50 concordance analysis
• Hallmark pathway scoring
• Biomarker visualization
• Translational interpretation

                │
                ▼
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
7. Explainable AI Modules
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Random Forest classification
• Deep Neural Network classifier
• Variational Autoencoder (VAE)
• Permutation feature importance
