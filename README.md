# AMA-CHD analysis

Reproducibility repository for **Endothelial arterial–matrix attenuation in congenital heart disease: a multi-cohort single-nucleus and spatial transcriptomic analysis**.

This repository contains the analysis code, processed public-data derivatives, result tables, and figure source files used to evaluate the arterial–matrix attenuation (AMA) framework. Raw GEO and spatial expression matrices are not redistributed; accession numbers and download instructions are listed in [`docs/data_sources.md`](docs/data_sources.md).

## Interpretation boundary

AMA is an exploratory, covariate-sensitive framework, not a validated diagnostic score. The discovery contrast contains 4 donors and 6 structural-CHD individuals and has substantial age and cardiac-region non-overlap. External ventricular estimates are directionally positive but nonsignificant after multiple-testing correction, and the spatial sensitivity analysis contains three sections from one biological heart.

The primary disjoint modules are:

- arterial maturation: `NOTCH1`, `DLL4`, `JAG1`, `HEY1`, `HEY2`, `EFNB2`, `GJA5`, `SOX17`;
- endothelial–matrix interface: `FN1`, `VCAN`, `COL4A1`, `COL4A2`, `LAMA4`, `ITGA5`, `ITGB1`;
- TGF-beta/plasticity: `TGFB1`, `TGFB2`, `TGFBR1`, `TGFBR2`, `SMAD2`, `SMAD3`, `SNAI1`, `SNAI2`.

## Repository structure

```text
data/processed/        Processed public-data derivatives needed for rerunning sensitivity analyses
scripts/               Analysis and figure-generation scripts
results/tables/        Machine-readable numerical outputs and figure source data
results/figures/       Main figures and analysis-level source figures
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

The first three commands write reproducibility outputs to `results/reproduced/`. External-cohort reconstruction additionally requires the GEO files listed in `docs/data_sources.md`; configure their locations with the environment variables documented in `docs/reproducibility.md`, then run:

```bash
python scripts/discovery_and_external_robustness.py --external-only
```

## Data and code availability

All source datasets are public. The processed derivatives in this repository are limited to de-identified expression summaries and analysis inputs derived from those public datasets. Code is released under the MIT License. Dataset reuse remains subject to the terms of the originating repositories.

## Citation

Please cite the associated article when available. Repository citation metadata are provided in [`CITATION.cff`](CITATION.cff).
