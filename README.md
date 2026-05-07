# 🧬 TCGA-BRCA Transcriptomic Subtype Discovery and Explainable AI Prognostic Modeling

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![Cancer Genomics](https://img.shields.io/badge/Domain-Computational_Oncology-purple)
![Machine Learning](https://img.shields.io/badge/AI-Explainable_ML-red)
![TCGA](https://img.shields.io/badge/Dataset-TCGA--BRCA-green)
![Status](https://img.shields.io/badge/Status-Research_Grade-success)

> A research-grade computational oncology framework for transcriptomic subtype reconstruction, differential expression analysis, survival modeling, and explainable AI-based biomarker discovery using TCGA-BRCA RNA-seq data.

---

#  Overview

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

#  Research Objectives

1. Reconstruct intrinsic breast cancer molecular subtypes directly from transcriptomic data using unsupervised learning.
2. Identify subtype-associated differential gene expression programs.
3. Evaluate survival trends across discovered transcriptomic states.
4. Develop interpretable machine learning models for subtype prediction.
5. Prioritize biologically meaningful biomarkers and therapeutic targets.
6. Establish a reproducible computational oncology framework for future multi-omics expansion.

---

#  Dataset Description

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

#  Computational Pipeline

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
#  Key Findings

## Transcriptomic Subtype Reconstruction

Unsupervised clustering identified three biologically interpretable transcriptomic states:

| Cluster | Biological Interpretation | Key Markers |
|---|---|---|
| C1 | Luminal A-like | ESR1, PGR, FOXA1 |
| C2 | Luminal B-like | MKI67, CDC20, AURKB |
| C3 | Stromal/Normal-like | IL33, SFRP1, NGFR |

The reconstructed clusters demonstrated strong concordance with established breast cancer subtype biology.

---

## Clustering Performance

- Optimal cluster number: **k = 3**
- Silhouette score: **0.2123**

The moderate silhouette value reflects the inherent biological heterogeneity of bulk tumor transcriptomics while still preserving biologically meaningful subtype structure.

---

## Differential Expression Analysis

| Metric | Result |
|---|---|
| Significant DEGs per cluster | 5,626 – 7,043 |
| Statistical Threshold | FDR < 0.05 |
| Fold Change Threshold | \|log2FC\| > 0.5 |

### Aggressive Luminal B-like Program (C2)

Strong proliferative and mitotic signatures were identified, including:

- CDC20
- MKI67
- CENPA
- ORC6L
- AURKB
- PLK1

These findings are highly consistent with aggressive proliferative breast cancer biology reported in prior TCGA and PAM50 studies.

---

## Pathway-Level Biology

### C2 (Luminal B-like)
- proliferation ↑
- DNA repair ↑
- mitotic activity ↑
- cell-cycle signaling ↑

### C1 (Luminal A-like)
- hormone receptor signaling ↑
- endocrine-associated programs ↑

### C3 (Stromal/Normal-like)
- extracellular matrix activity ↑
- stromal signaling ↑
- immune-associated signatures ↑

---

## Survival Analysis

Kaplan-Meier and cumulative hazard analyses revealed trends toward subtype-associated survival differences.

However, pairwise log-rank significance remained limited across several subtype comparisons, likely reflecting:
- cohort heterogeneity
- censoring complexity
- intrinsic limitations of bulk RNA-seq survival stratification

The project therefore emphasizes:
- biological subtype reconstruction
- transcriptomic interpretation
- predictive modeling



---

#  Explainable AI Framework

## Random Forest Classifier

A Random Forest classifier was trained to predict subtype assignments derived from transcriptomic clustering.

### Performance
- Internal cross-validation AUC ≈ 0.997

This performance reflects strong internal subtype separability within the TCGA cohort.

---

## Deep Neural Network Classifier

A multilayer perceptron classifier reproduced high predictive performance while modeling nonlinear transcriptomic relationships.

---

## Variational Autoencoder (VAE)

A custom Variational Autoencoder was implemented for nonlinear latent-space learning.

### Objectives
- transcriptomic compression
- nonlinear representation learning
- subtype manifold exploration
- dimensionality reduction beyond PCA

---

## Explainability Layer

Permutation-based feature importance analysis enabled:
- biomarker prioritization
- subtype-associated gene ranking
- interpretable AI-driven molecular attribution

---

#  Literature Concordance

The findings reproduce major observations from landmark breast cancer transcriptomics studies:

| Study | Concordance |
|---|---|
| Perou et al. (2000) | Intrinsic subtype biology reproduced |
| Parker et al. (2009) | PAM50 marker consistency validated |
| TCGA Network (2012) | Transcriptomic subtype structure recapitulated |
| Ciriello et al. (2013) | Proliferative Luminal B signatures confirmed |
| Koboldt et al. (2012) | BRCA-associated pathway biology supported |

---

# Conclusion

This project establishes a reproducible computational oncology framework for transcriptomic subtype discovery, explainable AI classification, and biomarker prioritization in breast cancer.

The pipeline successfully reconstructs biologically meaningful molecular states from high-dimensional RNA-seq data while integrating:
- survival analysis
- differential expression profiling
- pathway interpretation
- interpretable machine learning

into a unified research workflow.

The framework provides a strong foundation for future expansion toward:
- multi-omics precision oncology
- advanced AI-driven cancer genomics
- single-cell transcriptomics
- translational biomarker discovery
- next-generation explainable oncology systems
