TCGA-BRCA Transcriptomic Subtype Discovery and Explainable AI Prognostic Modeling

A reproducible computational oncology workflow for data analysis of tumour transcriptomics for subtype classification, differential gene expression and patient survival studies and for interpretable artificial intelligence-based biomarker selection using TCGA-BRCA RNA sequencing data.

Overview

Breast cancer is a highly heterogeneous disease composed of distinct biological entities with different molecular states, clinical behavior, and therapeutic options for patients. In addition to intrinsic subtype classification systems, such as the PAM50 subtypes, there are additional molecular states and corresponding biological programs that influence clinical behavior and affect therapy selection.

The project aims to develop an end-to-end computational transcriptomics pipeline for analysis of RNA-seq data.

unsupervised molecular subtype discovery,
survival-associated stratification,
transcriptomic biomarker identification,
pathway-level biological interpretation,
and explainable machine learning classification.

I applied the framework to analyze RNA-seq expression data from the TCGA-BRCA cohort. The resulting subtype structure is biologically coherent and consists of three major subtypes including Luminal A-like, Luminal B-like and stromal/normal-like transcriptional programs.

The project integrates:

statistical genomics,
survival analysis,
dimensionality reduction,
clustering,
explainable machine learning,
and latent-space deep learning approaches

within a fully reproducible research pipeline.

Research Objectives
Primary Goals
This analysis allows users to directly reconstruct intrinsic subtypes of breast cancer from transcriptomic data using unsupervised learning.
Identify subtype-associated differential gene expression programs.
Evaluate survival trends across discovered transcriptomic states.
Develop interpretable machine learning models for subtype prediction.
Prioritize biologically meaningful biomarkers and therapeutic targets.
Integration strategies and empirical analysis indicate a viable route toward a reproducible framework to support further multi-omics growth for long-term computational oncology.

Dataset Description
Parameter	Description
Cohort	TCGA-BRCA
Source	UCSC Xena
Patients	1,214
Genes	20,530
Data Type	Bulk RNA-seq (HiSeqV2 log2-RSEM)
Clinical Data	Overall Survival (OS), Progression-Free Interval (PFI)
Clinical Resource	Liu et al. 2018 Pan-Cancer Survival Dataset Key Findings: Analysis, Visualizations & downloadable CSV, JSON, and RData files.

Computational Pipeline

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

Key Findings

1. Transcriptomic Subtype Reconstruction

Unsupervised clustering identified three biologically interpretable transcriptomic states:

Cluster	Biological Interpretation	Key Markers
C1	Luminal A-like	ESR1, PGR, FOXA1
C2	Luminal B-like	MKI67, CDC20, AURKB
C3	Stromal/Normal-like	IL33, SFRP1, NGFR

The post-hierarchical clustering assignments showed good correspondence with known subtype biology for breast cancer.

2. Clustering Performance

Optimal clustering was achieved at:

k=3

with silhouette score:

s=0.2123

I find that the optimal silhouette value, which balances bulk tumor’s intrinsic biological variation with subgroup structure, to be moderate, and reflecting the natural biological variation in the tumour bulk dataset used here.

3. Differential Expression Analysis

Subtype-specific differential expression revealed large-scale transcriptional divergence:

Metric	Result
Significant DEGs per cluster	5,626 – 7,043
Statistical Threshold	FDR < 0.05
Fold Change Threshold	|log2FC| > 0.5
Aggressive Luminal B-like Program (C2)

Strong proliferative and mitotic signatures were identified, including:

CDC20
MKI67
CENPA
ORC6L
AURKB
PLK1

Our findings are highly consistent with an aggressive proliferative type of breast cancer as defined by prior TCGA and PAM50 studies.

4. Pathway-Level Biology

Hallmark pathway analysis demonstrated:

C2 (Luminal B-like)
proliferation ↑
DNA repair ↑
mitotic activity ↑
cell-cycle signaling ↑
C1 (Luminal A-like)
hormone receptor signaling ↑
endocrine-associated programs ↑
C3 (Stromal/Normal-like)
extracellular matrix activity ↑
stromal signaling ↑
immune-associated signatures ↑

5. Survival Analysis

Survival analysis by Kaplan-Meier and cumulative hazard showed trends associated with patient survival that were distributed by subtype.

However, pairwise log-rank significance remained limited across several subtype comparisons .

cohort heterogeneity,
censoring complexity,
and intrinsic limitations of bulk RNA-seq survival stratification.

The project therefore emphasizes:

biological subtype reconstruction,
transcriptomic interpretation,
and predictive modeling



Explainable AI Framework
Random Forest Classifier

I trained a Random Forest classifier to predict the subtype assignments from the transcriptomic clustering analysis.

Performance

Internal cross-validation achieved near-perfect discrimination:

AUC≈0.997

These results indicate good internal subtype separation for this TCGA cohort.

Deep Neural Network Classifier

The multilayer perceptron classifier achieved high predictive performance while modeling transcriptomic (or global) gene expression relationships that are highly nonlinear.

Variational Autoencoder (VAE)

I learned a non-linear latent space with a custom Variational Autoencoder.

Objectives
transcriptomic compression,
nonlinear representation learning,
subtype manifold exploration,
dimensionality reduction beyond PCA.



Explainability Layer

Permutation-based feature importance analysis enabled:

biomarker prioritization,
subtype-associated gene ranking,
and interpretable AI-driven molecular attribution. 

Literature Concordance

My results recreate the findings of current breast cancer transcriptomics researches:

Study	Concordance
Perou et al. ( 2000)	Intrinsic subtype biology reproduced from publicly accessible TCGA datasets.
Parker et al. ( 2009)	High consistent use of PAM50 markers.
TCGA Network (2012)	Transcriptomic subtype structure recapitulated
Ciriello et al. (2013) Proliferative Luminal B signatures confirmed
Koboldt DC, et al.	BRCA-associated pathway biology supports a multifactorial origin for ovarian cancer.

This project develops and evaluates a reproducible computational framework for Oncology, specifically for discovering transcriptomic subtypes of breast cancer, explainable AI-driven classification and subsequent prioritization of biomarkers.

I present a pipeline that learns to generate biologically meaningful molecular states directly and exclusively from high-dimensional RNA-seq data while simultaneously incorporating survival information, differential gene expression measurements, pathway level results and interpretable machine learning models into a streamlined research pipeline.

This resource provides a solid platform on which future multi-omics precision oncology and advanced high-dimensional AI-powered translational cancer genomics analyses can be conducted.
