from __future__ import annotations

import importlib.util
import json
import math
import re
import sys
from pathlib import Path

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import optimize, stats


PROJECT = Path(os.environ.get("AMA_PROJECT_ROOT", Path(__file__).resolve().parents[1])).resolve()
DATA = Path(os.environ.get("AMA_PROCESSED_DATA_DIR", PROJECT / "data" / "processed"))
DISCOVERY_NPZ = DATA / "discovery_patient_pseudobulk_counts.npz"
DISCOVERY_META = DATA / "discovery_sample_metadata.csv"
EXTERNAL_SCRIPT = Path(__file__).with_name("external_cross_cohort_analysis.py")
OUT = Path(os.environ.get("AMA_OUTPUT_DIR", PROJECT / "results" / "reproduced"))
TABLES = OUT / "tables"
FIGURES = OUT / "figures"
TABLES.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)


ARTERIAL = ["NOTCH1", "DLL4", "JAG1", "HEY1", "HEY2", "EFNB2", "GJA5", "SOX17"]
ECM_CORE = ["FN1", "VCAN", "COL4A1", "COL4A2", "LAMA4", "ITGA5", "ITGB1"]
ECM_BROAD_STROMAL = ["POSTN", "COL1A1", "COL1A2", "COL3A1", "DCN"]
ECM_ORIGINAL = ECM_CORE + ECM_BROAD_STROMAL
TGFB_SIGNALING_CORE = ["TGFB1", "TGFB2", "TGFBR1", "TGFBR2", "SMAD2", "SMAD3", "SNAI1", "SNAI2"]
TGFB_MESENCHYMAL = ["TAGLN", "ACTA2", "VIM"]
TGFB_ORIGINAL = TGFB_SIGNALING_CORE + TGFB_MESENCHYMAL + ["FN1"]

MAIN_MODULES = {
    "arterial_maturation": ARTERIAL,
    "endothelial_matrix_core": ECM_CORE,
    "tgfb_plasticity_core": TGFB_SIGNALING_CORE,
}

MODULE_VARIANTS = {
    "main_core_disjoint": MAIN_MODULES,
    "original_overlapping": {
        "arterial_maturation": ARTERIAL,
        "ecm_integrin": ECM_ORIGINAL,
        "tgfb_endmt": TGFB_ORIGINAL,
    },
    "original_disjoint": {
        "arterial_maturation": ARTERIAL,
        "ecm_integrin": ECM_ORIGINAL,
        "tgfb_endmt": TGFB_SIGNALING_CORE + TGFB_MESENCHYMAL,
    },
    "original_disjoint_no_mesenchymal": {
        "arterial_maturation": ARTERIAL,
        "ecm_integrin": ECM_ORIGINAL,
        "tgfb_plasticity": TGFB_SIGNALING_CORE,
    },
    "core_ecm_disjoint_full_endmt": {
        "arterial_maturation": ARTERIAL,
        "endothelial_matrix_core": ECM_CORE,
        "tgfb_endmt": TGFB_SIGNALING_CORE + TGFB_MESENCHYMAL,
    },
}


def age_to_years(value: str) -> float:
    match = re.match(r"(?:(\d+)y)?(?:_(\d+)m)?(?:_(\d+)d)?$", str(value))
    if not match:
        raise ValueError(f"Cannot parse age: {value}")
    years, months, days = match.groups()
    return int(years or 0) + int(months or 0) / 12 + int(days or 0) / 365.25


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


def design_ols(y: np.ndarray, group: np.ndarray, covariates: list[np.ndarray]) -> dict[str, float]:
    y = np.asarray(y, dtype=float)
    group = np.asarray(group, dtype=float)
    x = np.column_stack([np.ones(len(y)), *[np.asarray(v, dtype=float) for v in covariates], group])
    valid = np.isfinite(y) & np.isfinite(x).all(axis=1)
    y = y[valid]
    x = x[valid]
    n = len(y)
    if n == 0:
        return {k: np.nan for k in ("beta", "se", "ci_low", "ci_high", "p_value", "df_resid", "n", "rank")}
    beta = np.linalg.pinv(x) @ y
    residual = y - x @ beta
    rank = int(np.linalg.matrix_rank(x))
    df = n - rank
    idx = x.shape[1] - 1
    if df > 0:
        sigma2 = float(residual @ residual / df)
        cov = sigma2 * np.linalg.pinv(x.T @ x)
        se = float(math.sqrt(max(cov[idx, idx], 0)))
        t_value = float(beta[idx] / se) if se > 0 else np.nan
        p_value = float(2 * stats.t.sf(abs(t_value), df)) if np.isfinite(t_value) else np.nan
        crit = float(stats.t.ppf(0.975, df))
        ci_low = float(beta[idx] - crit * se)
        ci_high = float(beta[idx] + crit * se)
    else:
        se = p_value = ci_low = ci_high = np.nan
    return {
        "beta": float(beta[idx]),
        "se": se,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "p_value": p_value,
        "df_resid": int(df),
        "n": int(n),
        "rank": rank,
    }


def welch_summary(case: np.ndarray, control: np.ndarray) -> dict[str, float]:
    case = np.asarray(case, dtype=float)
    control = np.asarray(control, dtype=float)
    case = case[np.isfinite(case)]
    control = control[np.isfinite(control)]
    n1, n0 = len(case), len(control)
    delta = float(np.mean(case) - np.mean(control))
    if n1 > 1 and n0 > 1:
        test = stats.ttest_ind(case, control, equal_var=False)
        v1 = float(np.var(case, ddof=1) / n1)
        v0 = float(np.var(control, ddof=1) / n0)
        se = math.sqrt(v1 + v0)
        df = (v1 + v0) ** 2 / (v1**2 / (n1 - 1) + v0**2 / (n0 - 1))
        crit = float(stats.t.ppf(0.975, df))
        p = float(test.pvalue)
        low, high = delta - crit * se, delta + crit * se
    else:
        p = low = high = np.nan
    return {
        "n_case": n1,
        "n_control": n0,
        "mean_case": float(np.mean(case)),
        "mean_control": float(np.mean(control)),
        "delta": delta,
        "ci_low": float(low),
        "ci_high": float(high),
        "p_value": p,
    }


def load_discovery() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    data = np.load(DISCOVERY_NPZ, allow_pickle=False)
    genes = data["genes"].astype(str)
    patients = data["patient_labels"].astype(str)
    counts = pd.DataFrame(data["patient_counts"].T.astype(np.int64), index=patients, columns=genes)
    totals = counts.sum(axis=1).replace(0, np.nan)
    logcpm = np.log2(counts.div(totals, axis=0) * 1_000_000 + 1)

    meta = pd.read_csv(DISCOVERY_META, index_col="patient_index")
    meta.index = meta.index.astype(str)
    meta["sex_male"] = meta["sex"].eq("M").astype(int)
    meta["structural_chd"] = meta["condition"].isin(["TOF", "HLHS"]).astype(int)
    meta["log10_cm_fraction"] = np.log10(meta["pseudobulk_cardiomyocyte_fraction"].astype(float) + 1e-6)
    meta["high_contamination"] = meta["high_contamination"].astype(str).str.lower().eq("true")
    return logcpm, meta, {}


def module_score_table(
    logcpm: pd.DataFrame,
    modules: dict[str, list[str]],
    controls: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = pd.DataFrame(index=logcpm.index)
    coverage_rows = []
    for module, genes in modules.items():
        present = [gene for gene in genes if gene in logcpm.columns]
        raw[module] = logcpm[present].mean(axis=1)
        coverage_rows.append(
            {
                "module": module,
                "n_present": len(present),
                "n_total": len(genes),
                "present_genes": ";".join(present),
                "missing_genes": ";".join(g for g in genes if g not in present),
            }
        )
    z = pd.DataFrame(index=raw.index)
    for module in raw.columns:
        reference = raw.loc[controls, module]
        sd = float(reference.std(ddof=1))
        z[module] = (raw[module] - float(reference.mean())) / sd if np.isfinite(sd) and sd > 0 else np.nan
    z["AMA"] = -z[list(modules)].mean(axis=1)
    raw = raw.add_prefix("raw_")
    z = z.add_prefix("z_").rename(columns={"z_AMA": "AMA"})
    return pd.concat([raw, z], axis=1), pd.DataFrame(coverage_rows)


def discovery_effects(scores: pd.DataFrame, meta: pd.DataFrame, score_name: str = "AMA") -> pd.DataFrame:
    contrast_specs = [
        ("CHD_vs_Donor", ["TOF", "HLHS"], ["Donor"]),
        ("TOF_vs_Donor", ["TOF"], ["Donor"]),
        ("HLHS_vs_Donor", ["HLHS"], ["Donor"]),
        ("Cardiomyopathy_vs_Donor", ["Cardiomyopathy"], ["Donor"]),
        ("CHD_vs_Cardiomyopathy", ["TOF", "HLHS"], ["Cardiomyopathy"]),
        ("HLHS_vs_TOF", ["HLHS"], ["TOF"]),
    ]
    joined = meta.join(scores[[score_name]])
    rows = []
    for label, case_groups, control_groups in contrast_specs:
        subset = joined[joined["condition"].isin(case_groups + control_groups)].copy()
        subset["case"] = subset["condition"].isin(case_groups).astype(int)
        unadjusted = welch_summary(
            subset.loc[subset["case"] == 1, score_name].to_numpy(),
            subset.loc[subset["case"] == 0, score_name].to_numpy(),
        )
        adjusted = design_ols(
            subset[score_name].to_numpy(),
            subset["case"].to_numpy(),
            [subset["age_years"].to_numpy(), subset["sex_male"].to_numpy(), subset["log10_cm_fraction"].to_numpy()],
        )
        rows.append({
            "contrast": label,
            "case_groups": ";".join(case_groups),
            "control_groups": ";".join(control_groups),
            **{f"unadjusted_{k}": v for k, v in unadjusted.items()},
            **{f"adjusted_{k}": v for k, v in adjusted.items()},
        })
    result = pd.DataFrame(rows)
    result["adjusted_fdr"] = bh_adjust(result["adjusted_p_value"])
    return result


def adjusted_beta_fast(y: np.ndarray, group: np.ndarray, covariate_matrix: np.ndarray) -> float:
    covariate_matrix = np.asarray(covariate_matrix, dtype=float)
    group = np.asarray(group, dtype=float)
    y = np.asarray(y, dtype=float)
    group_residual = group - covariate_matrix @ (np.linalg.pinv(covariate_matrix) @ group)
    denom = float(group_residual @ group_residual)
    return float(group_residual @ y / denom) if denom > 0 else np.nan


def expression_matched_permutation(
    logcpm: pd.DataFrame,
    meta: pd.DataFrame,
    observed_scores: pd.DataFrame,
    n_permutations: int = 10_000,
    seed: int = 203274,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    structural = meta.index[meta["condition"].isin(["Donor", "TOF", "HLHS"])].tolist()
    donor = meta.index[meta["condition"] == "Donor"].tolist()
    expr = logcpm.loc[structural]
    gene_mean = expr.mean(axis=0)
    detected = (expr > 0).sum(axis=0)
    universe = gene_mean.index[(detected >= 3) & np.isfinite(gene_mean)].tolist()
    target_genes = [gene for genes in MAIN_MODULES.values() for gene in genes]
    universe = [gene for gene in universe if gene not in target_genes]
    combined = gene_mean.loc[universe + target_genes]
    ranks = combined.rank(method="average", pct=True)
    bins = np.minimum((ranks * 20).astype(int), 19)
    candidates: dict[str, list[str]] = {}
    for gene in target_genes:
        target_bin = int(bins.loc[gene])
        same_bin = [g for g in universe if int(bins.loc[g]) == target_bin]
        candidates[gene] = same_bin

    subset_meta = meta.loc[structural]
    group = subset_meta["structural_chd"].to_numpy(dtype=float)
    covariates = np.column_stack([
        np.ones(len(subset_meta)),
        subset_meta["age_years"].to_numpy(dtype=float),
        subset_meta["sex_male"].to_numpy(dtype=float),
        subset_meta["log10_cm_fraction"].to_numpy(dtype=float),
    ])
    observed_beta = adjusted_beta_fast(observed_scores.loc[structural, "AMA"].to_numpy(), group, covariates)
    rng = np.random.default_rng(seed)
    permuted = np.empty(n_permutations, dtype=float)
    for iteration in range(n_permutations):
        used: set[str] = set()
        module_z = []
        for module_genes in MAIN_MODULES.values():
            selected = []
            for target in module_genes:
                pool = [gene for gene in candidates[target] if gene not in used]
                if not pool:
                    pool = candidates[target]
                chosen = str(rng.choice(pool))
                selected.append(chosen)
                used.add(chosen)
            raw = expr[selected].mean(axis=1)
            ref = raw.loc[donor]
            sd = float(ref.std(ddof=1))
            module_z.append((raw - float(ref.mean())) / sd if np.isfinite(sd) and sd > 0 else raw * np.nan)
        random_ama = -pd.concat(module_z, axis=1).mean(axis=1)
        permuted[iteration] = adjusted_beta_fast(random_ama.to_numpy(), group, covariates)
    p_one = (1 + int(np.sum(permuted >= observed_beta))) / (n_permutations + 1)
    p_two = (1 + int(np.sum(np.abs(permuted) >= abs(observed_beta)))) / (n_permutations + 1)
    summary = pd.DataFrame([{
        "n_permutations": n_permutations,
        "observed_adjusted_beta": observed_beta,
        "permutation_mean": float(np.mean(permuted)),
        "permutation_sd": float(np.std(permuted, ddof=1)),
        "empirical_p_one_sided": p_one,
        "empirical_p_two_sided": p_two,
        "matching": "20 expression-quantile bins; module sizes preserved; target genes excluded",
        "seed": seed,
    }])
    values = pd.DataFrame({"permutation": np.arange(1, n_permutations + 1), "adjusted_beta": permuted})
    return summary, values


def leave_out_sensitivity(logcpm: pd.DataFrame, meta: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    structural = meta.index[meta["condition"].isin(["Donor", "TOF", "HLHS"])].tolist()
    donors = meta.index[meta["condition"] == "Donor"].tolist()
    subset_meta = meta.loc[structural]
    group = subset_meta["structural_chd"].to_numpy(dtype=float)
    covariates = np.column_stack([
        np.ones(len(subset_meta)),
        subset_meta["age_years"].to_numpy(dtype=float),
        subset_meta["sex_male"].to_numpy(dtype=float),
        subset_meta["log10_cm_fraction"].to_numpy(dtype=float),
    ])
    gene_rows = []
    for module, genes in MAIN_MODULES.items():
        for omitted in genes:
            modified = {name: list(values) for name, values in MAIN_MODULES.items()}
            modified[module] = [gene for gene in genes if gene != omitted]
            score, _ = module_score_table(logcpm, modified, donors)
            beta = adjusted_beta_fast(score.loc[structural, "AMA"].to_numpy(), group, covariates)
            gene_rows.append({"module": module, "omitted_gene": omitted, "adjusted_beta": beta})
    module_rows = []
    full_score, _ = module_score_table(logcpm, MAIN_MODULES, donors)
    module_rows.append({
        "omitted_module": "none",
        "modules_retained": ";".join(MAIN_MODULES),
        "adjusted_beta": adjusted_beta_fast(full_score.loc[structural, "AMA"].to_numpy(), group, covariates),
    })
    for omitted in MAIN_MODULES:
        retained = {name: genes for name, genes in MAIN_MODULES.items() if name != omitted}
        score, _ = module_score_table(logcpm, retained, donors)
        beta = adjusted_beta_fast(score.loc[structural, "AMA"].to_numpy(), group, covariates)
        module_rows.append({
            "omitted_module": omitted,
            "modules_retained": ";".join(retained),
            "adjusted_beta": beta,
        })
    return pd.DataFrame(gene_rows), pd.DataFrame(module_rows)


def discovery_analysis() -> dict[str, pd.DataFrame]:
    logcpm, meta, summary = load_discovery()
    donors = meta.index[meta["condition"] == "Donor"].tolist()
    structural = meta.index[meta["condition"].isin(["Donor", "TOF", "HLHS"])].tolist()

    main_scores, coverage = module_score_table(logcpm, MAIN_MODULES, donors)
    joined = meta.join(main_scores)
    effects = discovery_effects(main_scores, meta)

    variant_rows = []
    for variant, modules in MODULE_VARIANTS.items():
        scores, _ = module_score_table(logcpm, modules, donors)
        subset = meta.loc[structural]
        result = design_ols(
            scores.loc[structural, "AMA"].to_numpy(),
            subset["structural_chd"].to_numpy(),
            [subset["age_years"].to_numpy(), subset["sex_male"].to_numpy(), subset["log10_cm_fraction"].to_numpy()],
        )
        clean = [patient for patient in structural if not bool(meta.loc[patient, "high_contamination"])]
        clean_meta = meta.loc[clean]
        clean_result = design_ols(
            scores.loc[clean, "AMA"].to_numpy(),
            clean_meta["structural_chd"].to_numpy(),
            [clean_meta["age_years"].to_numpy(), clean_meta["sex_male"].to_numpy()],
        )
        variant_rows.append({
            "variant": variant,
            "module_definitions": " | ".join(f"{k}:{','.join(v)}" for k, v in modules.items()),
            **{f"adjusted_{k}": v for k, v in result.items()},
            **{f"clean_{k}": v for k, v in clean_result.items()},
        })
    variants = pd.DataFrame(variant_rows)

    loo_gene, loo_module = leave_out_sensitivity(logcpm, meta)
    permutation_summary, permutation_values = expression_matched_permutation(logcpm, meta, main_scores)

    metadata_columns = [
        "patient_id", "diagnosis_original", "condition", "region", "age", "age_years", "sex",
        "n_endothelial_nuclei", "patient_level_pseudobulk_profiles", "n_endothelial_subtype_profiles",
        "n_subtype_profiles_ge20_nuclei", "pseudobulk_cardiomyocyte_fraction",
        "median_cardiomyocyte_fraction", "high_contamination", "primary_inclusion", "clean_sensitivity_status",
    ]
    metadata_table = meta[metadata_columns].copy()
    metadata_table.index.name = "patient_index"
    metadata_table = metadata_table.reset_index()

    outputs = {
        "S_discovery_sample_metadata.csv": metadata_table,
        "S_main_module_gene_coverage.csv": coverage,
        "S_discovery_patient_module_scores.csv": joined.reset_index(names="patient_index"),
        "S_discovery_disease_specificity_effects.csv": effects,
        "S_module_definition_sensitivity.csv": variants,
        "S_leave_one_gene_out.csv": loo_gene,
        "S_leave_one_module_out.csv": loo_module,
        "S_expression_matched_permutation_summary.csv": permutation_summary,
        "S_expression_matched_permutation_values.csv": permutation_values,
    }
    for filename, frame in outputs.items():
        frame.to_csv(TABLES / filename, index=False, encoding="utf-8-sig")

    make_discovery_robustness_figure(joined, variants, permutation_summary, permutation_values, loo_gene)
    return outputs


def make_discovery_robustness_figure(
    joined: pd.DataFrame,
    variants: pd.DataFrame,
    permutation_summary: pd.DataFrame,
    permutation_values: pd.DataFrame,
    loo_gene: pd.DataFrame,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    ax = axes[0, 0]
    order = ["Donor", "TOF", "HLHS", "Cardiomyopathy"]
    colors = {"Donor": "#4C78A8", "TOF": "#F2B134", "HLHS": "#D95F59", "Cardiomyopathy": "#59A14F"}
    for idx, group in enumerate(order):
        vals = joined.loc[joined["condition"] == group, "AMA"].dropna().to_numpy()
        ax.scatter(np.full(len(vals), idx) + np.linspace(-0.08, 0.08, max(len(vals), 1)), vals, s=55, color=colors[group], edgecolor="white")
        if len(vals):
            ax.hlines(np.mean(vals), idx - 0.22, idx + 0.22, color="#222222", linewidth=2)
    ax.axhline(0, color="#888888", linestyle="--", linewidth=0.8)
    ax.set_xticks(range(len(order)), order, rotation=20)
    ax.set_ylabel("Core disjoint AMA score")
    ax.set_title("A  Discovery disease groups")

    ax = axes[0, 1]
    y = np.arange(len(variants))
    ax.errorbar(variants["adjusted_beta"], y, xerr=[variants["adjusted_beta"] - variants["adjusted_ci_low"], variants["adjusted_ci_high"] - variants["adjusted_beta"]], fmt="o", color="#C44E52", capsize=3)
    ax.axvline(0, color="#777777", linewidth=0.9)
    ax.set_yticks(y, variants["variant"].str.replace("_", " "))
    ax.set_xlabel("Adjusted CHD−donor beta")
    ax.set_title("B  Module-definition sensitivity")

    ax = axes[1, 0]
    values = permutation_values["adjusted_beta"].to_numpy()
    observed = float(permutation_summary.loc[0, "observed_adjusted_beta"])
    ax.hist(values, bins=55, color="#9CBED2", edgecolor="white")
    ax.axvline(observed, color="#C44E52", linewidth=2.2, label=f"Observed={observed:.2f}")
    ax.set_xlabel("Adjusted beta from expression-matched random gene sets")
    ax.set_ylabel("Count")
    ax.set_title(f"C  10,000-gene-set null (empirical P={permutation_summary.loc[0, 'empirical_p_one_sided']:.4f})")
    ax.legend(frameon=False)

    ax = axes[1, 1]
    module_order = list(MAIN_MODULES)
    positions = {name: idx for idx, name in enumerate(module_order)}
    for module, group in loo_gene.groupby("module"):
        x = positions[module]
        vals = group["adjusted_beta"].to_numpy()
        ax.scatter(np.full(len(vals), x) + np.linspace(-0.12, 0.12, len(vals)), vals, s=36, alpha=0.85)
    ax.axhline(0, color="#777777", linewidth=0.9)
    ax.set_xticks(range(len(module_order)), [name.replace("_", " ") for name in module_order], rotation=20)
    ax.set_ylabel("Adjusted CHD−donor beta")
    ax.set_title("D  Leave-one-gene-out stability")
    fig.tight_layout()
    fig.savefig(FIGURES / "Figure_S_module_and_disease_robustness.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def load_external_module():
    spec = importlib.util.spec_from_file_location("ama_external", EXTERNAL_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {EXTERNAL_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.EXT = Path(os.environ.get("AMA_EXTERNAL_VALIDATION_DIR", PROJECT / "external_validation"))
    module.GSE23959_DIR = Path(os.environ.get("AMA_GSE23959_DIR", module.EXT / "GSE23959"))
    module.GSE23959_GPL_SOFT = module.GSE23959_DIR / "GPL5188_family.soft.gz"
    module.GENE_INFO = Path(os.environ.get("AMA_GENE_INFO", PROJECT / "resources" / "Homo_sapiens.gene_info.gz"))
    module.GSE23959_ANNOTATION_SQLITE = Path(os.environ.get(
        "AMA_GSE23959_SQLITE", PROJECT / "resources" / "GSE23959_annotation" / "huex10sttranscriptcluster.sqlite"
    ))
    return module


def hedges_g(case: np.ndarray, control: np.ndarray) -> dict[str, float]:
    case = np.asarray(case, dtype=float)
    control = np.asarray(control, dtype=float)
    case = case[np.isfinite(case)]
    control = control[np.isfinite(control)]
    n1, n0 = len(case), len(control)
    df = n1 + n0 - 2
    sd1 = float(np.std(case, ddof=1))
    sd0 = float(np.std(control, ddof=1))
    pooled = math.sqrt(((n1 - 1) * sd1**2 + (n0 - 1) * sd0**2) / df)
    d = float((np.mean(case) - np.mean(control)) / pooled)
    correction = 1 - 3 / (4 * df - 1)
    g = correction * d
    variance = (n1 + n0) / (n1 * n0) + g**2 / (2 * df)
    return {
        "n_case": n1,
        "n_control": n0,
        "hedges_g": g,
        "sampling_variance": variance,
        "standard_error": math.sqrt(variance),
        "ci_low_normal": g - 1.96 * math.sqrt(variance),
        "ci_high_normal": g + 1.96 * math.sqrt(variance),
    }


def reml_hartung_knapp(effect_table: pd.DataFrame) -> pd.DataFrame:
    yi = effect_table["hedges_g"].to_numpy(dtype=float)
    vi = effect_table["sampling_variance"].to_numpy(dtype=float)
    k = len(yi)

    def objective(tau2: float) -> float:
        total_var = vi + tau2
        weights = 1 / total_var
        mean = float(np.sum(weights * yi) / np.sum(weights))
        return 0.5 * (np.sum(np.log(total_var)) + np.log(np.sum(weights)) + np.sum(weights * (yi - mean) ** 2))

    upper = max(10.0, float(np.var(yi, ddof=1) * 20 + np.max(vi)))
    opt = optimize.minimize_scalar(objective, bounds=(0, upper), method="bounded")
    tau2 = max(0.0, float(opt.x))
    if objective(0) <= objective(tau2) + 1e-10:
        tau2 = 0.0
    weights = 1 / (vi + tau2)
    pooled = float(np.sum(weights * yi) / np.sum(weights))
    q_hk = float(np.sum(weights * (yi - pooled) ** 2) / (k - 1))
    q_modified = max(1.0, q_hk)
    se_hk = math.sqrt(q_modified / np.sum(weights))
    crit = float(stats.t.ppf(0.975, k - 1))
    ci_low = pooled - crit * se_hk
    ci_high = pooled + crit * se_hk
    t_value = pooled / se_hk
    p_value = float(2 * stats.t.sf(abs(t_value), k - 1))
    pred_crit = float(stats.t.ppf(0.975, max(k - 2, 1)))
    pred_se = math.sqrt(tau2 + se_hk**2)
    pred_low = pooled - pred_crit * pred_se
    pred_high = pooled + pred_crit * pred_se

    fixed_weights = 1 / vi
    fixed_mean = float(np.sum(fixed_weights * yi) / np.sum(fixed_weights))
    cochran_q = float(np.sum(fixed_weights * (yi - fixed_mean) ** 2))
    i2 = max(0.0, (cochran_q - (k - 1)) / cochran_q * 100) if cochran_q > 0 else 0.0
    return pd.DataFrame([{
        "k": k,
        "pooled_hedges_g": pooled,
        "hk_modified_se": se_hk,
        "ci95_low": ci_low,
        "ci95_high": ci_high,
        "p_value": p_value,
        "tau2_reml": tau2,
        "cochran_q": cochran_q,
        "i2_percent": i2,
        "prediction_interval_low": pred_low,
        "prediction_interval_high": pred_high,
        "hartung_knapp_scale": q_hk,
        "modified_hk_scale": q_modified,
        "method": "Hedges g; REML random effects; modified Hartung-Knapp CI; t-based prediction interval",
        "interpretation": "exploratory because only three cohorts were available",
    }])


def external_analysis() -> dict[str, pd.DataFrame]:
    ext = load_external_module()
    mapping = ext.parse_gene_info_mapping(ext.GENE_INFO)
    entrez_mapping = ext.parse_gene_info_entrez_symbols(ext.GENE_INFO)
    datasets = {
        "GSE36761": ext.load_gse36761(mapping),
        "GSE217772": ext.load_gse217772(mapping),
        "GSE132176": ext.load_gse132176(),
        "GSE23959": ext.load_gse23959(entrez_mapping),
    }
    contrasts = ext.build_contrasts(datasets)

    variant_effects = []
    main_scores = None
    main_effects = None
    main_contrast_scores = None
    coverage_main = None
    for variant, modules in MODULE_VARIANTS.items():
        ext.MODULES = modules
        score_frames = []
        coverage_frames = []
        for dataset in datasets.values():
            frame, coverage = ext.module_scores(dataset)
            score_frames.append(frame)
            coverage_frames.append(coverage)
        all_scores = pd.concat(score_frames, ignore_index=True)
        coverage = pd.concat(coverage_frames, ignore_index=True)
        contrast_scores, effects = ext.compute_contrast_scores(all_scores, datasets, contrasts)
        ama = effects[effects["module"] == "AMA"].copy()
        ama.insert(0, "variant", variant)
        variant_effects.append(ama)
        if variant == "main_core_disjoint":
            main_scores = all_scores
            main_effects = effects
            main_contrast_scores = contrast_scores
            coverage_main = coverage
    assert main_scores is not None and main_effects is not None and main_contrast_scores is not None and coverage_main is not None
    variant_effects_df = pd.concat(variant_effects, ignore_index=True)

    primary_names = [
        "GSE217772_TOF_RV_vs_healthy_RV",
        "GSE36761_TOF_RV_vs_healthy_RV",
        "GSE23959_HLHS_ventricle_vs_healthy_RV",
    ]
    meta_rows = []
    for contrast_name in primary_names:
        data = main_contrast_scores[
            (main_contrast_scores["contrast"] == contrast_name)
            & (main_contrast_scores["module"] == "AMA")
        ]
        effect = hedges_g(
            data.loc[data["role"] == "case", "raw_module_score"].to_numpy(),
            data.loc[data["role"] == "control", "raw_module_score"].to_numpy(),
        )
        meta_rows.append({"contrast": contrast_name, **effect})
    meta_inputs = pd.DataFrame(meta_rows)
    meta_result = reml_hartung_knapp(meta_inputs)

    outputs = {
        "S_external_main_module_scores.csv": main_scores,
        "S_external_main_contrast_scores.csv": main_contrast_scores,
        "S_external_main_effects.csv": main_effects,
        "S_external_module_coverage.csv": coverage_main,
        "S_external_module_definition_sensitivity.csv": variant_effects_df,
        "S_external_meta_hedges_g_inputs.csv": meta_inputs,
        "S_external_meta_reml_hk.csv": meta_result,
    }
    for filename, frame in outputs.items():
        frame.to_csv(TABLES / filename, index=False, encoding="utf-8-sig")
    make_external_meta_figure(meta_inputs, meta_result)
    return outputs


def make_external_meta_figure(inputs: pd.DataFrame, result: pd.DataFrame) -> None:
    labels = [
        "GSE217772: TOF RV vs healthy RV",
        "GSE36761: TOF RV vs healthy RV",
        "GSE23959: HLHS ventricle vs healthy RV",
    ]
    y = np.arange(len(inputs), 0, -1)
    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.errorbar(
        inputs["hedges_g"], y,
        xerr=[inputs["hedges_g"] - inputs["ci_low_normal"], inputs["ci_high_normal"] - inputs["hedges_g"]],
        fmt="o", color="#4C78A8", capsize=4, markersize=7,
    )
    pooled = float(result.loc[0, "pooled_hedges_g"])
    low = float(result.loc[0, "ci95_low"])
    high = float(result.loc[0, "ci95_high"])
    ax.errorbar([pooled], [0], xerr=[[pooled - low], [high - pooled]], fmt="D", color="#C44E52", capsize=5, markersize=8)
    ax.axvline(0, color="#777777", linewidth=1)
    ax.set_yticks(list(y) + [0], labels + ["REML + modified Hartung-Knapp"])
    ax.set_xlabel("Hedges g (higher = stronger AMA)")
    ax.set_title("Exploratory primary ventricular cross-cohort evaluation")
    ax.text(
        0.02, -0.22,
        f"Pooled g={pooled:.2f} (95% CI {low:.2f} to {high:.2f}); P={result.loc[0, 'p_value']:.3f}; "
        f"tau²={result.loc[0, 'tau2_reml']:.3f}; prediction interval {result.loc[0, 'prediction_interval_low']:.2f} to {result.loc[0, 'prediction_interval_high']:.2f}",
        transform=ax.transAxes, fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(FIGURES / "Figure_external_meta_Hedges_REML_HK.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_summary(discovery: dict[str, pd.DataFrame], external: dict[str, pd.DataFrame]) -> None:
    discovery_effect = discovery["S_discovery_disease_specificity_effects.csv"]
    chd = discovery_effect.loc[discovery_effect["contrast"] == "CHD_vs_Donor"].iloc[0]
    permutation = discovery["S_expression_matched_permutation_summary.csv"].iloc[0]
    meta = external["S_external_meta_reml_hk.csv"].iloc[0]
    external_effects = external["S_external_main_effects.csv"]
    g217 = external_effects[
        (external_effects["contrast"] == "GSE217772_TOF_RV_vs_healthy_RV")
        & (external_effects["module"] == "AMA")
    ].iloc[0]
    lines = [
        "# JBS v7 robustness revision: analysis summary",
        "",
        "## Discovery",
        "",
        f"- Patient-level discovery: 4 donors, 3 TOF, 3 HLHS and 3 cardiomyopathy patients (13 biological individuals).",
        f"- Core disjoint AMA adjusted CHD-versus-donor beta: {chd['adjusted_beta']:.3f} "
        f"(95% CI {chd['adjusted_ci_low']:.3f} to {chd['adjusted_ci_high']:.3f}; P={chd['adjusted_p_value']:.4g}).",
        f"- Expression-matched 10,000-set empirical one-sided P: {permutation['empirical_p_one_sided']:.4g}; "
        f"two-sided P: {permutation['empirical_p_two_sided']:.4g}.",
        "- Disease-specific and leave-one-gene/module-out results are in the accompanying CSV files.",
        "",
        "## External cross-cohort evaluation",
        "",
        f"- GSE217772 core AMA delta: {g217['delta_case_minus_control']:.3f}; nominal P={g217['p_value']:.4g}; "
        f"all-module FDR={g217['fdr_within_all_module_tests']:.4g}.",
        f"- Exploratory Hedges-g meta-analysis: pooled g={meta['pooled_hedges_g']:.3f} "
        f"(modified Hartung-Knapp 95% CI {meta['ci95_low']:.3f} to {meta['ci95_high']:.3f}; P={meta['p_value']:.4g}); "
        f"tau2={meta['tau2_reml']:.4g}; prediction interval {meta['prediction_interval_low']:.3f} to {meta['prediction_interval_high']:.3f}.",
        "",
        "## Interpretation boundary",
        "",
        "Age, chamber and lesion remain partly confounded in GSE203274, especially because TOF samples are infant RVOT and donor samples are older LV/RV tissues. Covariate adjustment and sensitivity analyses reduce but cannot eliminate this design limitation.",
    ]
    (OUT / "v7_robustness_analysis_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Reproduce AMA discovery and/or external robustness analyses.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--discovery-only", action="store_true")
    group.add_argument("--external-only", action="store_true")
    args = parser.parse_args()

    discovery = {} if args.external_only else discovery_analysis()
    external = {} if args.discovery_only else external_analysis()
    if discovery and external:
        write_summary(discovery, external)
    manifest = []
    for path in sorted(OUT.rglob("*")):
        if path.is_file():
            manifest.append({"relative_path": str(path.relative_to(OUT)).replace("\\", "/"), "bytes": path.stat().st_size})
    pd.DataFrame(manifest).to_csv(OUT / "output_manifest.csv", index=False, encoding="utf-8-sig")
    print(f"Outputs written to {OUT}")


if __name__ == "__main__":
    main()
