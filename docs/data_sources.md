# Public data sources

| Resource | Role | Required local file(s) |
|---|---|---|
| GSE203274 | Single-nucleus endothelial discovery | A processed patient-level pseudobulk derivative is included in `data/processed/` |
| GSE217772 | TOF right-ventricular external evaluation | `GSE217772_mRNA_count.txt.gz`, `GSE217772_series_matrix.txt.gz` |
| GSE36761 | TOF ventricular external evaluation | `GSE36761_gene_expression_levels_normalized.txt.gz`, `GSE36761_series_matrix.txt.gz` |
| GSE23959 | HLHS ventricular external evaluation | `GSE23959_series_matrix.txt.gz`, `GPL5188_family.soft.gz`; optional local annotation SQLite |
| GSE132176 | Right-atrial and cardiopulmonary-bypass boundary analysis | `GSE132176_series_matrix.txt.gz`, `GPL13158_family.soft.gz` |
| NCBI gene_info | Ensembl/Entrez mapping | `Homo_sapiens.gene_info.gz` |
| Fetal endothelial atlas | Developmental reference-state analysis | Processed cell-level module-score derivative included in `data/processed/` |
| Developing-heart MERFISH | Within-BEC spatial sensitivity analysis | Processed cell-level spatial-score derivative included in `data/processed/` |

GEO series can be downloaded from <https://www.ncbi.nlm.nih.gov/geo/>. Raw source matrices are intentionally not committed to this repository.
