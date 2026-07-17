# Reproducibility guide

## Included analyses

1. Patient-level GSE203274 module scoring, covariate-aware models, alternative module definitions, leave-one-gene/module analyses, and expression-matched patient-score permutations.
2. Competitive gene-level expression-matched permutations using adjusted and high-contamination-excluded differential-expression tables.
3. Fetal leave-arterial-out mapping and within-BEC spatial sensitivity analysis.
4. External cross-cohort module scoring and exploratory Hedges-g REML/modified Hartung–Knapp meta-analysis.
5. Submission-figure reconstruction from machine-readable result tables.

Random procedures use fixed seeds recorded in the scripts. The reported meta-analysis is exploratory because only three ventricular cohorts were available.

## External cohort paths

Set these environment variables before external reconstruction:

```text
AMA_PROJECT_ROOT              repository root; defaults to the parent of scripts/
AMA_EXTERNAL_VALIDATION_DIR  directory containing GSE217772, GSE36761 and GSE132176 files
AMA_GSE23959_DIR             directory containing GSE23959 and GPL5188 files
AMA_GENE_INFO                path to Homo_sapiens.gene_info.gz
AMA_GSE23959_SQLITE          optional transcript-cluster annotation SQLite
AMA_OUTPUT_DIR               output directory; defaults to results/reproduced
```

Example in PowerShell:

```powershell
$env:AMA_EXTERNAL_VALIDATION_DIR = "D:\data\external_validation"
$env:AMA_GSE23959_DIR = "D:\data\GSE23959"
$env:AMA_GENE_INFO = "D:\data\Homo_sapiens.gene_info.gz"
python scripts/discovery_and_external_robustness.py --external-only
```

## Expected results

Reference outputs are already stored in `results/tables/` and `results/figures/`. Minor graphical differences can occur across operating systems because of font rendering. Numerical results should agree within floating-point tolerance when the same processed inputs and dependency versions are used.
