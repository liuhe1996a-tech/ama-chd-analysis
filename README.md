# AMA-CHD analysis

Reproducibility repository for **An endothelial arterial-matrix attenuation transcript framework in congenital heart disease: multi-cohort evaluation with developmental and spatial contextualization**.

This repository contains analysis code, de-identified processed public-data derivatives, machine-readable result tables, and figure source files used to evaluate the arterial-matrix attenuation (AMA) framework. Raw GEO and spatial expression matrices are not redistributed; accession numbers and download instructions are listed in [`docs/data_sources.md`](docs/data_sources.md).

## Interpretation boundary

AMA is a transcript-module framework for organizing heterogeneous endothelial remodeling signals across datasets, not a validated diagnostic score, a direct assay of pathway activity or a disease state shown to be independent of broader endothelial stress. In the discovery cohort, the unadjusted patient-level contrast was negative and imprecise, whereas the age-, sex-, and contamination-adjusted coefficient was positive. Cardiac region was not adjusted because lesion and region had near-complete overlap. Five of seven prespecified covariate specifications were positive, all leave-one-individual-out estimates remained positive, and one donor was highly influential. These findings support reporting model dependence rather than claiming a stable, independently identified CHD effect.

A post hoc coupling analysis tested whether the three modules were less coordinated in CHD. The primary structural-cohort fit gave each individual equal total weight during cell-level residualization and retained the biological individual as the group-summary unit. The CHD-minus-donor difference was small, positive, and not significant, so lower coupling was not supported in this cohort. Complete-label permutation summaries remain descriptive because age and cardiac region are confounded with group.

In ventricular bulk cohorts, adjustment for five lineage marker scores changed effect estimates and did not produce consistent cross-cohort support. The RNF20-loss analysis evaluates the frozen transcript modules in an endothelial perturbation context; it is not an independent CHD validation and must not be interpreted as proof that transcript abundance equals TGF-beta pathway activity.

The primary non-overlapping modules are:

- arterial maturation: `NOTCH1`, `DLL4`, `JAG1`, `HEY1`, `HEY2`, `EFNB2`, `GJA5`, `SOX17`;
- endothelial-matrix interface: `FN1`, `VCAN`, `COL4A1`, `COL4A2`, `LAMA4`, `ITGA5`, `ITGB1`;
- TGF-beta/plasticity transcripts: `TGFB1`, `TGFB2`, `TGFBR1`, `TGFBR2`, `SMAD2`, `SMAD3`, `SNAI1`, `SNAI2`.

## Repository structure

```text
data/processed/        De-identified processed derivatives used by included analyses
scripts/               Analysis and figure-generation scripts
results/tables/        Machine-readable numerical outputs and figure source data
results/figures/       Main figures and analysis-level figures
docs/                  Data-source, reproducibility, and result-dictionary notes
```

## Quick start

Python 3.11 or later is recommended.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

python scripts/discovery_and_external_robustness.py --discovery-only
python scripts/competitive_gene_set_permutation.py
python scripts/fetal_spatial_sensitivity.py
python scripts/build_submission_figures.py
```

The first three commands write reproduced outputs to `results/reproduced/`. External-cohort reconstruction requires the GEO files listed in [`docs/data_sources.md`](docs/data_sources.md); configure their locations as described in [`docs/reproducibility.md`](docs/reproducibility.md), then run:

```bash
python scripts/discovery_and_external_robustness.py --external-only
```

The editorial robustness and RNF20 perturbation workflow requires the external GEO files plus the two RNF20 supplementary workbooks described in the data-source guide:

```bash
python scripts/editorial_robustness_and_perturbation.py
```

The post hoc coupling workflow requires the GSE203274 endothelial raw-count matrix, its cell metadata, the processed pseudobulk NPZ, and the patient-score table. Supply these through environment variables and then run:

```bash
# PowerShell example
$env:AMA_COUNTS="path/to/GSE203274_Endothelial_snRNA_rawCount.csv.gz"
$env:AMA_METADATA="path/to/GSE203274_Endothelial_snRNA_metadata.csv.gz"
$env:AMA_PSEUDOBULK="path/to/pseudobulk_counts.npz"
$env:AMA_PATIENT_SCORES="results/tables/S_discovery_patient_module_scores.csv"
python scripts/coupling_sensitivity.py
```

Reference numerical outputs are provided in `results/tables/`, so the reported results can be audited without redistributing raw source matrices.

## Data and code availability

All analyzed source datasets are public or publicly described. The processed derivatives in this repository are limited to de-identified expression summaries and analysis inputs derived from those sources. Code is released under the MIT License. Dataset reuse remains subject to the terms of the originating repositories.

## Citation

Please cite the associated article when available. Repository citation metadata are provided in [`CITATION.cff`](CITATION.cff).
