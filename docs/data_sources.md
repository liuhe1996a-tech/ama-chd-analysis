# Public data sources

| Resource | Analytical role | Required local file(s) or access note |
|---|---|---|
| GSE203274 | Single-nucleus endothelial discovery | A processed patient-level pseudobulk derivative is included in `data/processed/` |
| GSE217772 | TOF right-ventricular whole-tissue sensitivity analysis | `GSE217772_mRNA_count.txt.gz`, `GSE217772_series_matrix.txt.gz` |
| GSE36761 | TOF ventricular whole-tissue sensitivity analysis | `GSE36761_gene_expression_levels_normalized.txt.gz`, `GSE36761_series_matrix.txt.gz` |
| GSE23959 | HLHS ventricular whole-tissue sensitivity analysis | `GSE23959_series_matrix.txt.gz`, `GPL5188_family.soft.gz`; optional local annotation SQLite |
| GSE132176 | Right-atrial and cardiopulmonary-bypass boundary analysis | `GSE132176_series_matrix.txt.gz`, `GPL13158_family.soft.gz` |
| NCBI gene_info | Ensembl/Entrez mapping | `Homo_sapiens.gene_info.gz` |
| Fetal endothelial atlas | Developmental reference-state analysis | Processed cell-level module-score derivative included in `data/processed/` |
| Developing-heart MERFISH | Within-BEC spatial sensitivity analysis | Processed cell-level spatial-score derivative included in `data/processed/` |
| Dou et al., Nature Communications 2025 | RNF20-loss endothelial perturbation analysis | Supplementary Data 4 and 5 from <https://doi.org/10.1038/s41467-025-65291-0> |
| Li et al., Nature Cardiovascular Research 2026 | Candidate future HLHS/VAD test | Individual-level matrices were not analyzed in this release; the cited repository was access-restricted at the analysis freeze (<https://doi.org/10.5281/zenodo.18407403>) |

GEO series can be downloaded from <https://www.ncbi.nlm.nih.gov/geo/>. Raw source matrices and third-party supplementary workbooks are intentionally not committed to this repository.

For the RNF20 workflow, place the two downloaded workbooks at:

```text
data/external/RNF20_Supplementary_Data_4_iEC_KO_DEG.xlsx
data/external/RNF20_Supplementary_Data_5_shRNF20_EC_DEG.xlsx
```

Alternatively, set `AMA_RNF20_SC_XLSX` and `AMA_RNF20_BULK_XLSX` to their local paths.
