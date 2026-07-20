from __future__ import annotations

import importlib.util
import json
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


V7_TABLES = ROOT / "results" / "tables"
V8 = ROOT / "results"
TABLES = ROOT / "results" / "tables"
FIGURES = ROOT / "results" / "figures"
TABLES.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)

RNF20_SC = Path(
    os.environ.get(
        "AMA_RNF20_SC_XLSX",
        ROOT / "data" / "external" / "RNF20_Supplementary_Data_4_iEC_KO_DEG.xlsx",
    )
)
RNF20_BULK = Path(
    os.environ.get(
        "AMA_RNF20_BULK_XLSX",
        ROOT / "data" / "external" / "RNF20_Supplementary_Data_5_shRNF20_EC_DEG.xlsx",
    )
)
EXTERNAL_SCRIPT = ROOT / "scripts" / "external_cross_cohort_analysis.py"

ARTERIAL = ["NOTCH1", "DLL4", "JAG1", "HEY1", "HEY2", "EFNB2", "GJA5", "SOX17"]
MATRIX = ["FN1", "VCAN", "COL4A1", "COL4A2", "LAMA4", "ITGA5", "ITGB1"]
TGFB_TRANSCRIPT = ["TGFB1", "TGFB2", "TGFBR1", "TGFBR2", "SMAD2", "SMAD3", "SNAI1", "SNAI2"]
MODULES = {
    "arterial_maturation": ARTERIAL,
    "endothelial_matrix_core": MATRIX,
    "tgfb_plasticity_transcript": TGFB_TRANSCRIPT,
}
ALL_AMA_GENES = [gene for genes in MODULES.values() for gene in genes]

COMPOSITION_MARKERS = {
    "endothelial_abundance": ["PECAM1", "CDH5", "VWF", "EMCN", "KDR", "FLT1", "TEK"],
    "fibroblast": ["COL1A1", "COL1A2", "COL3A1", "DCN", "LUM", "PDGFRA"],
    "cardiomyocyte": ["TNNT2", "ACTC1", "MYH6", "MYH7", "TNNI3", "MYL2"],
    "immune": ["PTPRC", "LST1", "TYROBP", "CD68", "FCER1G"],
    "mural_pericyte": ["RGS5", "CSPG4", "MCAM", "PDGFRB", "DES", "ACTA2", "TAGLN"],
}

plt.rcParams.update(
    {
        "font.family": "Arial",
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def bh_adjust(values: pd.Series) -> pd.Series:
    p = values.to_numpy(dtype=float)
    finite = np.isfinite(p)
    out = np.full(len(p), np.nan)
    if not finite.any():
        return pd.Series(out, index=values.index)
    pv = p[finite]
    order = np.argsort(pv)
    ranked = pv[order]
    n = len(ranked)
    adjusted = np.empty(n)
    running = 1.0
    for idx in range(n - 1, -1, -1):
        running = min(running, ranked[idx] * n / (idx + 1))
        adjusted[order[idx]] = running
    out[np.where(finite)[0]] = adjusted
    return pd.Series(out, index=values.index)


def ols_fit(data: pd.DataFrame, outcome: str, group: str, covariates: list[str]) -> tuple[dict, np.ndarray, np.ndarray]:
    columns = [outcome, group, *covariates]
    frame = data[columns].apply(pd.to_numeric, errors="coerce").dropna()
    y = frame[outcome].to_numpy(dtype=float)
    x_names = ["intercept", *covariates, group]
    x = np.column_stack(
        [np.ones(len(frame)), *[frame[c].to_numpy(dtype=float) for c in covariates], frame[group].to_numpy(dtype=float)]
    )
    rank = int(np.linalg.matrix_rank(x))
    df = len(y) - rank
    beta = np.linalg.pinv(x) @ y
    residual = y - x @ beta
    xtx_inv = np.linalg.pinv(x.T @ x)
    idx = len(x_names) - 1
    sigma2 = float(residual @ residual / df) if df > 0 else np.nan
    cov = sigma2 * xtx_inv if np.isfinite(sigma2) else np.full_like(xtx_inv, np.nan)
    se = float(np.sqrt(max(cov[idx, idx], 0))) if np.isfinite(cov[idx, idx]) else np.nan
    t_value = float(beta[idx] / se) if se > 0 else np.nan
    p_value = float(2 * stats.t.sf(abs(t_value), df)) if df > 0 and np.isfinite(t_value) else np.nan
    critical = float(stats.t.ppf(0.975, df)) if df > 0 else np.nan
    condition_number = float(np.linalg.cond(x)) if len(y) else np.nan
    result = {
        "beta": float(beta[idx]),
        "se": se,
        "ci_low": float(beta[idx] - critical * se) if np.isfinite(critical) else np.nan,
        "ci_high": float(beta[idx] + critical * se) if np.isfinite(critical) else np.nan,
        "p_value": p_value,
        "t_value": t_value,
        "n": int(len(y)),
        "rank": rank,
        "df_resid": int(df),
        "condition_number": condition_number,
        "x_names": x_names,
        "row_index": frame.index.astype(str).tolist(),
        "beta_vector": beta,
        "residual": residual,
        "x": x,
        "xtx_inv": xtx_inv,
        "sigma2": sigma2,
    }
    return result, x, y


def public_ols_result(result: dict) -> dict:
    return {k: v for k, v in result.items() if k not in {"beta_vector", "residual", "x", "xtx_inv", "sigma2", "x_names", "row_index"}}


def discovery_editorial_robustness() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    scores = pd.read_csv(V7_TABLES / "S_discovery_patient_module_scores.csv")
    structural = scores[scores["condition"].isin(["Donor", "TOF", "HLHS"])].copy()
    structural["structural_chd"] = structural["condition"].isin(["TOF", "HLHS"]).astype(int)
    structural = structural.set_index("patient_id", drop=False)

    specifications = [
        ("Unadjusted OLS", []),
        ("CHD + age", ["age_years"]),
        ("CHD + sex", ["sex_male"]),
        ("CHD + contamination", ["log10_cm_fraction"]),
        ("CHD + age + contamination", ["age_years", "log10_cm_fraction"]),
        ("CHD + age + sex", ["age_years", "sex_male"]),
        ("Full: CHD + age + sex + contamination", ["age_years", "sex_male", "log10_cm_fraction"]),
    ]
    specification_rows = []
    full_result = None
    for order, (label, covariates) in enumerate(specifications, start=1):
        result, _, _ = ols_fit(structural, "AMA", "structural_chd", covariates)
        specification_rows.append(
            {
                "order": order,
                "specification": label,
                "covariates": ";".join(covariates) if covariates else "none",
                **public_ols_result(result),
                "positive_direction": bool(result["beta"] > 0),
            }
        )
        if label.startswith("Full"):
            full_result = result
    specification = pd.DataFrame(specification_rows)

    case = structural.loc[structural["structural_chd"] == 1, "AMA"].to_numpy(dtype=float)
    control = structural.loc[structural["structural_chd"] == 0, "AMA"].to_numpy(dtype=float)
    welch = stats.ttest_ind(case, control, equal_var=False)
    unadjusted_welch = {
        "delta_case_minus_control": float(np.mean(case) - np.mean(control)),
        "p_value": float(welch.pvalue),
        "n_case": int(len(case)),
        "n_control": int(len(control)),
    }

    loio_rows = []
    full_covariates = ["age_years", "sex_male", "log10_cm_fraction"]
    for patient_id in structural.index:
        subset = structural.drop(index=patient_id)
        result, _, _ = ols_fit(subset, "AMA", "structural_chd", full_covariates)
        omitted = structural.loc[patient_id]
        loio_rows.append(
            {
                "omitted_patient": patient_id,
                "omitted_condition": omitted["condition"],
                "omitted_high_contamination": bool(omitted["high_contamination"]),
                **public_ols_result(result),
                "positive_direction": bool(result["beta"] > 0),
            }
        )
    loio = pd.DataFrame(loio_rows)

    assert full_result is not None
    x = full_result["x"]
    residual = full_result["residual"]
    xtx_inv = full_result["xtx_inv"]
    n, p = x.shape
    leverage = np.einsum("ij,jk,ik->i", x, xtx_inv, x)
    mse = float(full_result["sigma2"])
    cooks = (residual**2 / (p * mse)) * leverage / np.maximum((1 - leverage) ** 2, 1e-12)
    influence_rows = []
    beta_full = float(full_result["beta"])
    group_idx = p - 1
    for row_number, patient_id in enumerate(full_result["row_index"]):
        subset = structural.drop(index=patient_id)
        delete_result, _, _ = ols_fit(subset, "AMA", "structural_chd", full_covariates)
        delete_scale = float(delete_result["sigma2"])
        standardized_denominator = math.sqrt(max(delete_scale * xtx_inv[group_idx, group_idx], 0))
        dfbeta = (
            (beta_full - float(delete_result["beta"])) / standardized_denominator
            if standardized_denominator > 0
            else np.nan
        )
        studentized = residual[row_number] / math.sqrt(max(mse * (1 - leverage[row_number]), 1e-12))
        influence_rows.append(
            {
                "patient_id": patient_id,
                "condition": structural.loc[patient_id, "condition"],
                "high_contamination": bool(structural.loc[patient_id, "high_contamination"]),
                "leverage": float(leverage[row_number]),
                "internally_studentized_residual": float(studentized),
                "cooks_distance": float(cooks[row_number]),
                "dfbeta_structural_chd": float(dfbeta),
                "deleted_beta": float(delete_result["beta"]),
                "cooks_reference_4_over_n": float(4 / n),
                "dfbeta_reference_2_over_sqrt_n": float(2 / math.sqrt(n)),
                "leverage_reference_2p_over_n": float(2 * p / n),
            }
        )
    influence = pd.DataFrame(influence_rows)
    return specification, loio, influence, unadjusted_welch


def load_external_module():
    spec = importlib.util.spec_from_file_location("external_source", EXTERNAL_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to import {EXTERNAL_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    # The source loader intentionally maps only its predefined target genes in
    # GSE23959. Extend that target list before parsing GPL5188 so composition
    # sensitivity markers are not lost during targeted probe aggregation.
    module.TARGET_GENES = sorted(
        set(module.TARGET_GENES)
        | {gene for genes in COMPOSITION_MARKERS.values() for gene in genes}
        | set(ALL_AMA_GENES)
    )
    mapping = module.parse_gene_info_mapping(module.GENE_INFO)
    entrez_mapping = module.parse_gene_info_entrez_symbols(module.GENE_INFO)
    datasets = {
        "GSE36761": module.load_gse36761(mapping),
        "GSE217772": module.load_gse217772(mapping),
        "GSE23959": module.load_gse23959(entrez_mapping),
    }
    all_for_contrasts = dict(datasets)
    all_for_contrasts["GSE132176"] = module.load_gse132176()
    contrasts = module.build_contrasts(all_for_contrasts)
    wanted = {
        "GSE36761_TOF_RV_vs_healthy_RV",
        "GSE217772_TOF_RV_vs_healthy_RV",
        "GSE23959_HLHS_ventricle_vs_healthy_RV",
    }
    return datasets, [contrast for contrast in contrasts if contrast.name in wanted]


def safe_zscore(values: pd.Series, controls: list[str]) -> pd.Series:
    control = values.reindex(controls).dropna().astype(float)
    sd = float(control.std(ddof=1))
    if not np.isfinite(sd) or sd <= 0:
        return pd.Series(np.nan, index=values.index)
    return (values.astype(float) - float(control.mean())) / sd


def variance_inflation_for_group(frame: pd.DataFrame, group: str, covariates: list[str]) -> float:
    y = frame[group].to_numpy(dtype=float)
    x = np.column_stack([np.ones(len(frame)), *[frame[c].to_numpy(dtype=float) for c in covariates]])
    beta = np.linalg.pinv(x) @ y
    residual = y - x @ beta
    tss = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - float(residual @ residual) / tss if tss > 0 else np.nan
    return float(1 / (1 - r2)) if np.isfinite(r2) and r2 < 1 else np.inf


def bulk_composition_sensitivity() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    datasets, contrasts = load_external_module()
    score_rows = []
    effect_rows = []
    coverage_rows = []
    covariate_names = list(COMPOSITION_MARKERS)
    for contrast in contrasts:
        ds = datasets[contrast.dataset]
        samples = contrast.case_samples + contrast.control_samples
        expr = ds.expression.loc[:, samples]
        frame = pd.DataFrame(index=samples)
        frame["disease"] = [1 if sample in contrast.case_samples else 0 for sample in samples]

        module_z = []
        for module_name, genes in MODULES.items():
            present = [gene for gene in genes if gene in expr.index]
            raw = expr.loc[present].mean(axis=0)
            z = safe_zscore(raw, contrast.control_samples)
            frame[module_name] = z
            module_z.append(z)
            coverage_rows.append(
                {
                    "contrast": contrast.name,
                    "feature": module_name,
                    "n_present": len(present),
                    "n_total": len(genes),
                    "present_genes": ";".join(present),
                    "missing_genes": ";".join(g for g in genes if g not in present),
                }
            )
        frame["AMA"] = -pd.concat(module_z, axis=1).mean(axis=1)

        for marker_name, genes in COMPOSITION_MARKERS.items():
            present = [gene for gene in genes if gene in expr.index]
            raw = expr.loc[present].mean(axis=0)
            frame[marker_name] = safe_zscore(raw, contrast.control_samples)
            coverage_rows.append(
                {
                    "contrast": contrast.name,
                    "feature": marker_name,
                    "n_present": len(present),
                    "n_total": len(genes),
                    "present_genes": ";".join(present),
                    "missing_genes": ";".join(g for g in genes if g not in present),
                }
            )

        for sample, row in frame.iterrows():
            score_rows.append({"contrast": contrast.name, "dataset": contrast.dataset, "sample": sample, **row.to_dict()})

        model_specs = [("unadjusted", [])]
        model_specs.extend((f"adjusted_{name}_only", [name]) for name in covariate_names)
        model_specs.append(("adjusted_all_five_marker_scores", covariate_names))
        for model_name, covariates in model_specs:
            result, _, _ = ols_fit(frame, "AMA", "disease", covariates)
            effect_rows.append(
                {
                    "contrast": contrast.name,
                    "dataset": contrast.dataset,
                    "model": model_name,
                    "composition_covariates": ";".join(covariates) if covariates else "none",
                    **public_ols_result(result),
                    "disease_vif": variance_inflation_for_group(frame.dropna(), "disease", covariates) if covariates else 1.0,
                }
            )
    effects = pd.DataFrame(effect_rows)
    effects["fdr_within_model_family"] = effects.groupby("model")["p_value"].transform(bh_adjust)
    return pd.DataFrame(score_rows), effects, pd.DataFrame(coverage_rows)


def matched_mean_null(
    frame: pd.DataFrame,
    target_genes: list[str],
    effect_col: str,
    matching_col: str,
    seed: int,
    n_permutations: int = 10_000,
) -> dict:
    data = frame.copy()
    data.index = data.index.astype(str).str.upper()
    data = data[~data.index.duplicated(keep="first")]
    targets = [gene for gene in target_genes if gene in data.index and np.isfinite(data.loc[gene, effect_col])]
    universe = data.drop(index=targets, errors="ignore").dropna(subset=[effect_col, matching_col])
    combined_matching = pd.concat([universe[matching_col], data.loc[targets, matching_col]])
    bins = pd.qcut(combined_matching.rank(method="average"), q=20, duplicates="drop", labels=False)
    pools = {}
    for gene in targets:
        pool = universe.index[bins.loc[universe.index] == bins.loc[gene]].to_numpy()
        if len(pool) == 0:
            distances = (universe[matching_col] - data.loc[gene, matching_col]).abs().sort_values()
            pool = distances.index[: max(20, min(100, len(distances)))].to_numpy()
        pools[gene] = pool
    rng = np.random.default_rng(seed)
    null_values = np.empty(n_permutations, dtype=float)
    for i in range(n_permutations):
        sampled = [rng.choice(pools[gene]) for gene in targets]
        null_values[i] = float(data.loc[sampled, effect_col].mean())
    observed = float(data.loc[targets, effect_col].mean())
    return {
        "n_target_genes": len(targets),
        "observed_mean_log2fc": observed,
        "attenuation_score_negative_mean_log2fc": -observed,
        "null_mean": float(np.mean(null_values)),
        "null_sd": float(np.std(null_values, ddof=1)),
        "empirical_p_more_negative": float((1 + np.sum(null_values <= observed)) / (n_permutations + 1)),
        "n_permutations": n_permutations,
    }


def rnf20_perturbation_analysis() -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows = []
    gene_rows = []
    for sheet in ["EC-EndoC", "EndoV"]:
        data = pd.read_excel(RNF20_SC, sheet_name=sheet, skiprows=1)
        data.columns = [str(c).strip() for c in data.columns]
        data["GENE"] = data["gene"].astype(str).str.upper()
        data["avg_detection"] = (
            pd.to_numeric(data["pct.1"], errors="coerce") + pd.to_numeric(data["pct.2"], errors="coerce")
        ) / 2
        data["avg_log2FC"] = pd.to_numeric(data["avg_log2FC"], errors="coerce")
        data["p_val_adj"] = pd.to_numeric(data["p_val_adj"], errors="coerce")
        indexed = data.drop_duplicates("GENE").set_index("GENE")
        null = matched_mean_null(indexed, ALL_AMA_GENES, "avg_log2FC", "avg_detection", 65291 + len(summary_rows))
        for module, genes in MODULES.items():
            subset = indexed.reindex(genes)
            summary_rows.append(
                {
                    "perturbation": "Rnf20 iEC-KO vs control",
                    "context": sheet,
                    "module": module,
                    "n_genes_present": int(subset["avg_log2FC"].notna().sum()),
                    "mean_log2fc": float(subset["avg_log2FC"].mean()),
                    "median_log2fc": float(subset["avg_log2FC"].median()),
                    "attenuation_score_negative_mean_log2fc": float(-subset["avg_log2FC"].mean()),
                    "combined_empirical_p_more_negative": null["empirical_p_more_negative"],
                    "source": "Dou et al., Nature Communications 2025, Supplementary Data 4",
                }
            )
            for gene in genes:
                gene_rows.append(
                    {
                        "perturbation": "Rnf20 iEC-KO vs control",
                        "context": sheet,
                        "module": module,
                        "gene": gene,
                        "log2fc": indexed.loc[gene, "avg_log2FC"] if gene in indexed.index else np.nan,
                        "adjusted_p_value": indexed.loc[gene, "p_val_adj"] if gene in indexed.index else np.nan,
                    }
                )

    data = pd.read_excel(RNF20_BULK, sheet_name=0, skiprows=1)
    data.columns = [str(c).strip() for c in data.columns]
    data["GENE"] = data["SYMBOL"].astype(str).str.upper()
    data["log2FoldChange"] = pd.to_numeric(data["log2FoldChange"], errors="coerce")
    data["baseMean"] = pd.to_numeric(data["baseMean"], errors="coerce")
    data["padj"] = pd.to_numeric(data["padj"], errors="coerce")
    indexed = data.drop_duplicates("GENE").set_index("GENE")
    null = matched_mean_null(indexed, ALL_AMA_GENES, "log2FoldChange", "baseMean", 65295)
    for module, genes in MODULES.items():
        subset = indexed.reindex(genes)
        summary_rows.append(
            {
                "perturbation": "shRnf20 PECAM1-positive EC vs control",
                "context": "sorted differentiated EC bulk RNA-seq",
                "module": module,
                "n_genes_present": int(subset["log2FoldChange"].notna().sum()),
                "mean_log2fc": float(subset["log2FoldChange"].mean()),
                "median_log2fc": float(subset["log2FoldChange"].median()),
                "attenuation_score_negative_mean_log2fc": float(-subset["log2FoldChange"].mean()),
                "combined_empirical_p_more_negative": null["empirical_p_more_negative"],
                "source": "Dou et al., Nature Communications 2025, Supplementary Data 5",
            }
        )
        for gene in genes:
            gene_rows.append(
                {
                    "perturbation": "shRnf20 PECAM1-positive EC vs control",
                    "context": "sorted differentiated EC bulk RNA-seq",
                    "module": module,
                    "gene": gene,
                    "log2fc": indexed.loc[gene, "log2FoldChange"] if gene in indexed.index else np.nan,
                    "adjusted_p_value": indexed.loc[gene, "padj"] if gene in indexed.index else np.nan,
                }
            )
    return pd.DataFrame(summary_rows), pd.DataFrame(gene_rows)


def save_figure(fig: plt.Figure, stem: str) -> None:
    fig.savefig(FIGURES / f"{stem}.png", dpi=400, bbox_inches="tight", facecolor="white")
    fig.savefig(FIGURES / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_specification_and_loio(specification: pd.DataFrame, loio: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 5.1), gridspec_kw={"width_ratios": [1.15, 1]})
    ax = axes[0]
    data = specification.iloc[::-1].reset_index(drop=True)
    y = np.arange(len(data))
    colors = ["#C75146" if beta > 0 else "#3569A8" for beta in data["beta"]]
    for i, row in data.iterrows():
        ax.plot([row["ci_low"], row["ci_high"]], [i, i], color=colors[i], lw=1.5)
        ax.scatter(row["beta"], i, color=colors[i], s=34, zorder=3)
        ax.text(row["ci_high"] + 0.12, i, f"P={row['p_value']:.3f}", va="center", fontsize=7.5)
    ax.axvline(0, color="#555555", lw=0.9, ls="--")
    ax.set_yticks(y, data["specification"])
    ax.set_xlabel("Disease coefficient for AMA (95% CI)")
    ax.set_title("A  Covariate specification curve", loc="left", fontweight="bold")
    ax.grid(axis="x", color="#E8E8E8", lw=0.7)
    right = float(np.nanmax(data["ci_high"])) + 1.3
    left = float(np.nanmin(data["ci_low"])) - 0.5
    ax.set_xlim(left, right)

    ax = axes[1]
    data = loio.sort_values(["omitted_condition", "omitted_patient"]).reset_index(drop=True)
    y = np.arange(len(data))[::-1]
    condition_colors = {"Donor": "#4C78A8", "TOF": "#F28E2B", "HLHS": "#B24C8A"}
    for i, row in data.iterrows():
        yy = y[i]
        color = condition_colors.get(row["omitted_condition"], "#666666")
        ax.plot([row["ci_low"], row["ci_high"]], [yy, yy], color=color, lw=1.4)
        ax.scatter(row["beta"], yy, color=color, s=33, zorder=3)
    ax.axvline(0, color="#555555", lw=0.9, ls="--")
    ax.set_yticks(y, [f"omit {x}" for x in data["omitted_patient"]])
    ax.set_xlabel("Full-model CHD coefficient (95% CI)")
    ax.set_title("B  Leave-one-individual-out analysis", loc="left", fontweight="bold")
    ax.grid(axis="x", color="#E8E8E8", lw=0.7)
    for condition, color in condition_colors.items():
        ax.scatter([], [], color=color, label=condition)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.suptitle("Discovery effect is model- and individual-sensitive", fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    save_figure(fig, "Figure_S_editorial_model_robustness")


def plot_influence(influence: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(11.6, 3.7))
    x = np.arange(len(influence))
    labels = influence["patient_id"].tolist()
    metrics = [
        ("leverage", "Leverage", "leverage_reference_2p_over_n"),
        ("cooks_distance", "Cook's distance", "cooks_reference_4_over_n"),
        ("dfbeta_structural_chd", "DFBETA for CHD coefficient", "dfbeta_reference_2_over_sqrt_n"),
    ]
    for index, (column, ylabel, threshold_column) in enumerate(metrics):
        ax = axes[index]
        colors = ["#D55E00" if flag else "#4C78A8" for flag in influence["high_contamination"]]
        ax.bar(x, influence[column], color=colors, width=0.72)
        threshold = float(influence[threshold_column].iloc[0])
        if column == "dfbeta_structural_chd":
            ax.axhline(threshold, color="#555555", ls="--", lw=0.8)
            ax.axhline(-threshold, color="#555555", ls="--", lw=0.8)
        else:
            ax.axhline(threshold, color="#555555", ls="--", lw=0.8)
        ax.set_xticks(x, labels, rotation=55, ha="right", fontsize=7)
        ax.set_ylabel(ylabel)
        ax.set_title(chr(65 + index), loc="left", fontweight="bold")
        ax.grid(axis="y", color="#EEEEEE", lw=0.6)
    axes[0].bar([], [], color="#D55E00", label="high contamination")
    axes[0].bar([], [], color="#4C78A8", label="other")
    axes[0].legend(frameon=False, fontsize=7)
    fig.suptitle("Patient-level influence diagnostics for the full discovery model", fontsize=11.5, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, "Figure_S_editorial_influence_diagnostics")


def plot_bulk_composition(effects: pd.DataFrame) -> None:
    selected = effects[effects["model"].isin(["unadjusted", "adjusted_all_five_marker_scores"])].copy()
    contrast_labels = {
        "GSE36761_TOF_RV_vs_healthy_RV": "GSE36761\nTOF RV vs healthy RV",
        "GSE217772_TOF_RV_vs_healthy_RV": "GSE217772\nTOF RV vs healthy RV",
        "GSE23959_HLHS_ventricle_vs_healthy_RV": "GSE23959\nHLHS ventricle vs healthy RV",
    }
    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    contrasts = list(contrast_labels)
    base = np.arange(len(contrasts))
    offsets = {"unadjusted": -0.13, "adjusted_all_five_marker_scores": 0.13}
    colors = {"unadjusted": "#4C78A8", "adjusted_all_five_marker_scores": "#E45756"}
    names = {"unadjusted": "Unadjusted", "adjusted_all_five_marker_scores": "Adjusted for five marker scores"}
    for model in offsets:
        rows = selected[selected["model"] == model].set_index("contrast").reindex(contrasts)
        y = base + offsets[model]
        for i, (_, row) in enumerate(rows.iterrows()):
            ax.plot([row["ci_low"], row["ci_high"]], [y[i], y[i]], color=colors[model], lw=1.7)
            ax.scatter(row["beta"], y[i], color=colors[model], s=38, zorder=3)
        ax.scatter([], [], color=colors[model], label=names[model])
    ax.axvline(0, color="#555555", lw=0.9, ls="--")
    ax.set_yticks(base, [contrast_labels[x] for x in contrasts])
    ax.invert_yaxis()
    ax.set_xlabel("Disease coefficient for AMA (95% CI)")
    ax.set_title("Bulk tissue cell-composition marker-score sensitivity", loc="left", fontweight="bold")
    ax.grid(axis="x", color="#E8E8E8", lw=0.7)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, fontsize=8)
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    save_figure(fig, "Figure_S_bulk_composition_sensitivity")


def plot_rnf20(summary: pd.DataFrame) -> None:
    contexts = summary[["perturbation", "context"]].drop_duplicates().apply(lambda row: f"{row['context']}", axis=1).tolist()
    modules = list(MODULES)
    labels = {
        "arterial_maturation": "Arterial maturation",
        "endothelial_matrix_core": "Endothelial-matrix core",
        "tgfb_plasticity_transcript": "TGF-beta/plasticity transcripts",
    }
    colors = {modules[0]: "#4C78A8", modules[1]: "#59A14F", modules[2]: "#B24C8A"}
    fig, ax = plt.subplots(figsize=(8.2, 4.5))
    base = np.arange(len(contexts))
    width = 0.22
    for j, module in enumerate(modules):
        values = []
        for context in contexts:
            row = summary[(summary["context"] == context) & (summary["module"] == module)]
            values.append(float(row["attenuation_score_negative_mean_log2fc"].iloc[0]))
        ax.bar(base + (j - 1) * width, values, width=width, color=colors[module], label=labels[module])
    ax.axhline(0, color="#555555", lw=0.9)
    ax.set_xticks(base, contexts, rotation=15, ha="right")
    ax.set_ylabel("Attenuation score (-mean log2 fold change)")
    ax.set_title("Frozen AMA transcript modules under RNF20 loss", loc="left", fontweight="bold")
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.22), fontsize=8)
    ax.grid(axis="y", color="#EEEEEE", lw=0.6)
    fig.tight_layout()
    save_figure(fig, "Figure_S_RNF20_perturbation")


def main() -> None:
    missing_rnf20 = [path for path in (RNF20_SC, RNF20_BULK) if not path.exists()]
    if missing_rnf20:
        missing_text = "\n".join(f"  - {path}" for path in missing_rnf20)
        raise FileNotFoundError(
            "RNF20 supplementary workbooks are required for the full workflow. "
            "Download Supplementary Data 4 and 5 from doi:10.1038/s41467-025-65291-0, "
            "place them under data/external, or set AMA_RNF20_SC_XLSX and "
            f"AMA_RNF20_BULK_XLSX. Missing:\n{missing_text}"
        )
    specification, loio, influence, welch = discovery_editorial_robustness()
    bulk_scores, bulk_effects, bulk_coverage = bulk_composition_sensitivity()
    rnf20_summary, rnf20_genes = rnf20_perturbation_analysis()

    outputs = {
        "S_discovery_model_specification_curve.csv": specification,
        "S_discovery_leave_one_individual_out.csv": loio,
        "S_discovery_influence_diagnostics.csv": influence,
        "S_bulk_composition_marker_scores.csv": bulk_scores,
        "S_bulk_composition_sensitivity_effects.csv": bulk_effects,
        "S_bulk_composition_gene_coverage.csv": bulk_coverage,
        "S_RNF20_perturbation_module_summary.csv": rnf20_summary,
        "S_RNF20_perturbation_gene_effects.csv": rnf20_genes,
    }
    for filename, frame in outputs.items():
        frame.to_csv(TABLES / filename, index=False, encoding="utf-8-sig")

    plot_specification_and_loio(specification, loio)
    plot_influence(influence)
    plot_bulk_composition(bulk_effects)
    plot_rnf20(rnf20_summary)

    full = specification[specification["specification"].str.startswith("Full")].iloc[0]
    all_positive = bool(loio["positive_direction"].all())
    summary = {
        "unadjusted_welch": welch,
        "full_adjusted": full.to_dict(),
        "model_specification_positive_count": int(specification["positive_direction"].sum()),
        "model_specification_total": int(len(specification)),
        "loio_all_positive": all_positive,
        "loio_beta_min": float(loio["beta"].min()),
        "loio_beta_max": float(loio["beta"].max()),
        "max_cooks_distance": float(influence["cooks_distance"].max()),
        "max_cooks_patient": str(influence.loc[influence["cooks_distance"].idxmax(), "patient_id"]),
        "bulk_composition_full_models": bulk_effects[bulk_effects["model"] == "adjusted_all_five_marker_scores"].to_dict("records"),
        "rnf20_combined_competitive_p_by_context": rnf20_summary.groupby("context")[
            "combined_empirical_p_more_negative"
        ].first().to_dict(),
        "interpretation": (
            "The discovery coefficient is sensitive to covariate specification and individual omission. "
            "Bulk marker-score adjustment is a tissue-composition sensitivity analysis, not cell deconvolution. "
            "RNF20-loss analyses test frozen transcript modules and do not equate component abundance with TGF-beta pathway activity."
        ),
    }
    (V8 / "analysis_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
    print(f"Outputs written to {V8}")


if __name__ == "__main__":
    main()
