# Reproducibility guide

## Included analyses

1. Patient-level GSE203274 module scoring, covariate-aware models, alternative module definitions, leave-one-gene/module analyses, and expression-matched patient-score permutations.
2. Discovery model-specification curves, leave-one-individual-out estimates, and regression influence diagnostics.
3. Competitive gene-level expression-matched permutations using adjusted and high-contamination-excluded differential-expression tables.
4. Fetal leave-arterial-out mapping and within-BEC spatial sensitivity analysis.
5. External whole-tissue module scoring, exploratory Hedges-g REML/modified Hartung-Knapp meta-analysis, and five-lineage marker-score adjustment.
6. Frozen-module analysis of published RNF20-loss endothelial perturbations.
7. Post hoc individual-level intermodule-coupling evaluation with individual-balanced residualization and directional equal-cell downsampling.
8. Submission-figure reconstruction from machine-readable result tables.

Random procedures use fixed seeds recorded in the scripts. The meta-analysis is exploratory because only three ventricular cohorts were available. The marker-score models are sensitivity analyses and are not substitutes for cell deconvolution. The RNF20 analysis is a perturbation-context evaluation, not an independent CHD validation.

## External cohort paths

Set these environment variables before external reconstruction:

```text
AMA_PROJECT_ROOT             repository root; defaults to the parent of scripts/
AMA_EXTERNAL_VALIDATION_DIR directory containing GSE217772, GSE36761 and GSE132176 files
AMA_GSE23959_DIR            directory containing GSE23959 and GPL5188 files
AMA_GENE_INFO               path to Homo_sapiens.gene_info.gz
AMA_GSE23959_SQLITE         optional transcript-cluster annotation SQLite
AMA_OUTPUT_DIR              output directory; defaults to results/reproduced
AMA_RNF20_SC_XLSX           optional path override for RNF20 Supplementary Data 4
AMA_RNF20_BULK_XLSX         optional path override for RNF20 Supplementary Data 5
AMA_COUNTS                  GSE203274 endothelial raw-count matrix
AMA_METADATA                GSE203274 endothelial cell metadata
AMA_PSEUDOBULK              processed pseudobulk NPZ used for order/count validation
AMA_PATIENT_SCORES          individual module-score table
AMA_COUPLING_OUT            coupling output directory; defaults to results/reproduced/coupling
```

Example in PowerShell:

```powershell
$env:AMA_EXTERNAL_VALIDATION_DIR = "D:\data\external_validation"
$env:AMA_GSE23959_DIR = "D:\data\GSE23959"
$env:AMA_GENE_INFO = "D:\data\Homo_sapiens.gene_info.gz"
$env:AMA_RNF20_SC_XLSX = "D:\data\RNF20_Supplementary_Data_4_iEC_KO_DEG.xlsx"
$env:AMA_RNF20_BULK_XLSX = "D:\data\RNF20_Supplementary_Data_5_shRNF20_EC_DEG.xlsx"
python scripts/discovery_and_external_robustness.py --external-only
python scripts/editorial_robustness_and_perturbation.py
```

Run commands from the repository root. The editorial robustness script imports the external-cohort loader and therefore requires the same GEO inputs as the external reconstruction.

For the post hoc coupling analysis, set `AMA_COUNTS`, `AMA_METADATA`, `AMA_PSEUDOBULK`, and `AMA_PATIENT_SCORES`, then run `python scripts/coupling_sensitivity.py`. Its complete-label summaries are descriptive because age and cardiac region are confounded with disease group in the available cohort.

## Expected results

Reference outputs are stored in `results/tables/` and `results/figures/`. Minor graphical differences can occur across operating systems because of font rendering. Numerical results should agree within floating-point tolerance when the same inputs and dependency versions are used.
