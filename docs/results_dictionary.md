# Result dictionary

- `S_discovery_sample_metadata.csv`: one row per biological individual, with age, sex, region, endothelial yield, and contamination flags.
- `S_discovery_patient_module_scores.csv`: donor-standardized component and AMA scores.
- `S_module_definition_sensitivity.csv`: adjusted estimates across alternative module definitions.
- `S_leave_one_gene_out.csv` and `S_leave_one_module_out.csv`: score-definition stability checks.
- `S_expression_matched_permutation_summary.csv`: patient-score null result; the full null distribution is gzip-compressed.
- `S_competitive_gene_set_permutation_summary.csv`: gene-level competitive null results; the full null distribution is gzip-compressed.
- `S_discovery_disease_specificity_effects.csv`: CHD, TOF, HLHS, and cardiomyopathy contrasts.
- `S_external_main_effects.csv`: cohort-specific module and AMA estimates with confidence intervals and FDR values.
- `S_external_meta_hedges_g_inputs.csv` and `S_external_meta_reml_hk.csv`: exploratory meta-analysis inputs and pooled result.
- `S_fetal_*`: leave-arterial-out developmental reference-state analyses.
- `S_spatial_*`: within-BEC outer–inner spatial sensitivity analyses.
