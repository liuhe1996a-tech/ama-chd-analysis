from __future__ import annotations

import gzip
import itertools
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
COUNTS = Path(os.environ.get("AMA_COUNTS", ROOT / "data" / "raw" / "GSE203274_Endothelial_snRNA_rawCount.csv.gz"))
METADATA = Path(os.environ.get("AMA_METADATA", ROOT / "data" / "raw" / "GSE203274_Endothelial_snRNA_metadata.csv.gz"))
PSEUDOBULK = Path(os.environ.get("AMA_PSEUDOBULK", ROOT / "data" / "processed" / "pseudobulk_counts.npz"))
PATIENT_SCORES = Path(os.environ.get(
    "AMA_PATIENT_SCORES",
    ROOT / "results" / "tables" / "S_discovery_patient_module_scores.csv",
))
OUT = Path(
    os.environ.get(
        "AMA_COUPLING_OUT",
        ROOT / "results" / "reproduced" / "coupling",
    )
)
TABLES = OUT / "tables"
FIGURES = OUT / "figures"
TABLES.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)

MODULES = {
    "arterial_maturation": ["NOTCH1", "DLL4", "JAG1", "HEY1", "HEY2", "EFNB2", "GJA5", "SOX17"],
    "endothelial_matrix_core": ["FN1", "VCAN", "COL4A1", "COL4A2", "LAMA4", "ITGA5", "ITGB1"],
    "tgfb_plasticity_transcript": ["TGFB1", "TGFB2", "TGFBR1", "TGFBR2", "SMAD2", "SMAD3", "SNAI1", "SNAI2"],
}
PAIRS = [
    ("arterial_maturation", "endothelial_matrix_core"),
    ("arterial_maturation", "tgfb_plasticity_transcript"),
    ("endothelial_matrix_core", "tgfb_plasticity_transcript"),
]
PAIR_LABELS = {
    PAIRS[0]: "Arterial–matrix",
    PAIRS[1]: "Arterial–plasticity transcripts",
    PAIRS[2]: "Matrix–plasticity transcripts",
}


def bh_adjust(values: pd.Series) -> pd.Series:
    p = values.to_numpy(dtype=float)
    order = np.argsort(p)
    ranked = p[order]
    adjusted = np.minimum.accumulate((ranked * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]
    out = np.empty(len(p), dtype=float)
    out[order] = np.clip(adjusted, 0, 1)
    return pd.Series(out, index=values.index)


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def normalize_patient_id(value: str) -> str:
    value = str(value)
    if value in {"13_198_LV", "13_198_RV"}:
        return "13_198"
    return value


def load_target_counts() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    meta = pd.read_csv(METADATA)
    with gzip.open(COUNTS, "rt", encoding="utf-8") as handle:
        cell_ids = handle.readline().rstrip("\r\n").split(",")
    if len(cell_ids) != len(meta):
        raise ValueError(f"Cell count mismatch: matrix={len(cell_ids)}, metadata={len(meta)}")
    prefix_match = np.array(
        [str(cell).startswith(f"{orig}_") for cell, orig in zip(cell_ids, meta["orig.ident"].astype(str))]
    )
    if not prefix_match.all():
        raise ValueError(f"Cell-order mismatch for {(~prefix_match).sum()} cells")

    pseudobulk = np.load(PSEUDOBULK, allow_pickle=True)
    genes = pseudobulk["genes"].astype(str)
    targets = [gene for module in MODULES.values() for gene in module]
    positions = {int(np.where(genes == gene)[0][0]): gene for gene in targets}
    if len(positions) != len(targets):
        raise ValueError("Target genes are absent or duplicated in the source gene vector")

    arrays: dict[str, np.ndarray] = {}
    row_count = 0
    with gzip.open(COUNTS, "rt", encoding="utf-8") as handle:
        next(handle)
        for row_index, line in enumerate(handle):
            row_count += 1
            gene = positions.get(row_index)
            if gene is None:
                continue
            values = np.fromstring(line.rstrip("\r\n"), dtype=np.float64, sep=",")
            if len(values) != len(cell_ids):
                raise ValueError(f"Malformed row for {gene}: {len(values)} values")
            arrays[gene] = values
    if row_count != len(genes):
        raise ValueError(f"Gene-row mismatch: matrix={row_count}, gene vector={len(genes)}")

    target_counts = pd.DataFrame(arrays, index=cell_ids)
    meta.index = cell_ids
    meta["individual_id"] = meta["patientID"].map(normalize_patient_id)

    patient_labels = pseudobulk["patient_labels"].astype(str)
    patient_counts = pseudobulk["patient_counts"]
    validation_differences = []
    for gene in targets:
        gene_index = int(np.where(genes == gene)[0][0])
        observed = target_counts[gene].groupby(meta["individual_id"]).sum()
        expected = pd.Series(patient_counts[gene_index, :], index=patient_labels, dtype=float)
        common = expected.index.intersection(observed.index)
        validation_differences.extend((observed.loc[common] - expected.loc[common]).abs().tolist())
    validation = {
        "n_cells": len(meta),
        "n_gene_rows": row_count,
        "n_target_genes": len(targets),
        "cell_order_all_matched": bool(prefix_match.all()),
        "max_target_patient_count_difference_vs_pseudobulk": float(max(validation_differences)),
    }
    return target_counts, meta, validation


def build_module_scores(target_counts: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    library_size = pd.to_numeric(meta["nCount_RNA"], errors="coerce").to_numpy(dtype=float)
    library_size = np.where(library_size > 0, library_size, np.nan)
    scores = pd.DataFrame(index=target_counts.index)
    for module, genes in MODULES.items():
        normalized = np.log1p(target_counts[genes].to_numpy(dtype=float) / library_size[:, None] * 10_000)
        scores[module] = np.nanmean(normalized, axis=1)
    return scores


def residualize(
    scores: pd.DataFrame,
    meta: pd.DataFrame,
    cell_mask: pd.Series,
    *,
    balance_individuals: bool = False,
) -> pd.DataFrame:
    idx = meta.index[cell_mask]
    m = meta.loc[idx].copy()
    y = scores.loc[idx].copy()
    continuous = pd.DataFrame(
        {
            "log1p_nCount_RNA": np.log1p(pd.to_numeric(m["nCount_RNA"], errors="coerce")),
            "log1p_nFeature_RNA": np.log1p(pd.to_numeric(m["nFeature_RNA"], errors="coerce")),
            "percent_mt": pd.to_numeric(m["percent.mt"], errors="coerce"),
        },
        index=idx,
    )
    continuous = (continuous - continuous.mean()) / continuous.std(ddof=0).replace(0, 1)
    # orig.ident and batch_indices are one-to-one in this release; retaining both
    # would create a rank-deficient duplicate encoding. Use one library/batch term.
    categories = pd.get_dummies(m[["Cluster", "batch_indices"]].astype(str), drop_first=True, dtype=float)
    x_frame = pd.concat([continuous, categories], axis=1).replace([np.inf, -np.inf], np.nan).fillna(0)
    x = np.column_stack([np.ones(len(x_frame)), x_frame.to_numpy(dtype=float)])
    if balance_individuals:
        counts = m["individual_id"].value_counts()
        weights = m["individual_id"].map(lambda value: 1.0 / counts[value]).to_numpy(dtype=float)
        weights = weights / np.mean(weights)
    else:
        weights = np.ones(len(m), dtype=float)
    root_weight = np.sqrt(weights)
    x_weighted = x * root_weight[:, None]
    residuals = pd.DataFrame(index=idx)
    for module in MODULES:
        outcome = y[module].to_numpy(dtype=float)
        beta = np.linalg.lstsq(x_weighted, outcome * root_weight, rcond=None)[0]
        residuals[module] = outcome - x @ beta
    return residuals


def fisher_z(r: float) -> float:
    return float(np.arctanh(np.clip(r, -0.999999, 0.999999)))


def patient_correlations(residuals: pd.DataFrame, meta: pd.DataFrame, subset: str) -> pd.DataFrame:
    rows = []
    for patient, patient_index in meta.loc[residuals.index].groupby("individual_id").groups.items():
        frame = residuals.loc[patient_index]
        pair_values: dict[tuple[str, str], dict[str, float]] = {}
        for left, right in PAIRS:
            pearson = float(stats.pearsonr(frame[left], frame[right]).statistic)
            spearman = float(stats.spearmanr(frame[left], frame[right]).statistic)
            pair_values[(left, right)] = {"pearson": pearson, "spearman": spearman}
            for method, value in pair_values[(left, right)].items():
                rows.append(
                    {
                        "subset": subset,
                        "patient_id": patient,
                        "n_cells": len(frame),
                        "pair": PAIR_LABELS[(left, right)],
                        "method": method,
                        "correlation": value,
                        "fisher_z": fisher_z(value),
                    }
                )
        for method in ["pearson", "spearman"]:
            zs = [fisher_z(pair_values[pair][method]) for pair in PAIRS]
            rows.append(
                {
                    "subset": subset,
                    "patient_id": patient,
                    "n_cells": len(frame),
                    "pair": "Mean three-pair coupling",
                    "method": method,
                    "correlation": float(np.tanh(np.mean(zs))),
                    "fisher_z": float(np.mean(zs)),
                }
            )
    return pd.DataFrame(rows)


def downsample_coupling(
    residuals: pd.DataFrame,
    meta: pd.DataFrame,
    subset: str,
    n_cells: int = 100,
    n_iterations: int = 500,
) -> pd.DataFrame:
    rng = np.random.default_rng(20260719)
    rows = []
    patient_groups = meta.loc[residuals.index].groupby("individual_id").groups
    for patient, patient_index in patient_groups.items():
        indices = np.asarray(list(patient_index))
        sample_n = min(n_cells, len(indices))
        iteration_z = []
        for _ in range(n_iterations):
            selected = rng.choice(indices, size=sample_n, replace=False)
            frame = residuals.loc[selected]
            pair_z = []
            for left, right in PAIRS:
                r = float(stats.spearmanr(frame[left], frame[right]).statistic)
                pair_z.append(fisher_z(r))
            iteration_z.append(float(np.mean(pair_z)))
        rows.append(
            {
                "subset": subset,
                "patient_id": patient,
                "available_cells": len(indices),
                "sampled_cells_per_iteration": sample_n,
                "n_iterations": n_iterations,
                "mean_fisher_z": float(np.mean(iteration_z)),
                "sd_fisher_z": float(np.std(iteration_z, ddof=1)),
                "mean_correlation": float(np.tanh(np.mean(iteration_z))),
            }
        )
    return pd.DataFrame(rows)


def exact_patient_permutation(values: pd.Series, labels: pd.Series) -> dict[str, float]:
    values = values.to_numpy(dtype=float)
    labels = labels.to_numpy(dtype=int)
    n_case = int(labels.sum())
    observed = float(values[labels == 1].mean() - values[labels == 0].mean())
    permuted = []
    for case_indices in itertools.combinations(range(len(values)), n_case):
        case_mask = np.zeros(len(values), dtype=bool)
        case_mask[list(case_indices)] = True
        permuted.append(float(values[case_mask].mean() - values[~case_mask].mean()))
    permuted = np.asarray(permuted)
    tolerance = 1e-12
    return {
        "difference_mean_fisher_z_chd_minus_donor": observed,
        "difference_on_correlation_scale": float(
            np.tanh(values[labels == 1].mean()) - np.tanh(values[labels == 0].mean())
        ),
        "mean_fisher_z_chd": float(values[labels == 1].mean()),
        "mean_fisher_z_donor": float(values[labels == 0].mean()),
        "mean_correlation_chd": float(np.tanh(values[labels == 1].mean())),
        "mean_correlation_donor": float(np.tanh(values[labels == 0].mean())),
        "exact_p_two_sided": float(np.mean(np.abs(permuted) >= abs(observed) - tolerance)),
        "exact_p_lower_in_chd": float(np.mean(permuted <= observed + tolerance)),
        "n_label_permutations": int(len(permuted)),
    }


def group_contrasts(correlations: pd.DataFrame, patient_meta: pd.DataFrame) -> pd.DataFrame:
    rows = []
    structural = patient_meta[
        patient_meta["primary_inclusion"].eq("included")
        & patient_meta["condition"].isin(["Donor", "TOF", "HLHS"])
    ].copy()
    structural["chd"] = structural["condition"].isin(["TOF", "HLHS"]).astype(int)
    for (subset, method, pair), group in correlations.groupby(["subset", "method", "pair"]):
        work = group.merge(structural[["patient_id", "chd", "high_contamination"]], on="patient_id", how="inner")
        for sensitivity, frame in [
            ("all_structural_individuals", work),
            ("high_contamination_excluded", work[~work["high_contamination"].astype(bool)]),
        ]:
            if frame["chd"].nunique() < 2:
                continue
            result = exact_patient_permutation(frame["fisher_z"], frame["chd"])
            rows.append(
                {
                    "subset": subset,
                    "method": method,
                    "pair": pair,
                    "sensitivity": sensitivity,
                    "n_chd": int(frame["chd"].sum()),
                    "n_donor": int((1 - frame["chd"]).sum()),
                    **result,
                }
            )
    out = pd.DataFrame(rows)
    pair_mask = out["pair"].ne("Mean three-pair coupling")
    out["fdr_q_three_pairs"] = np.nan
    out.loc[pair_mask, "fdr_q_three_pairs"] = out[pair_mask].groupby(
        ["subset", "method", "sensitivity"]
    )["exact_p_two_sided"].transform(bh_adjust)
    return out


def pca_patient_scores(patient_meta: pd.DataFrame) -> pd.DataFrame:
    structural = patient_meta[patient_meta["primary_inclusion"].eq("included")].copy()
    columns = ["raw_arterial_maturation", "raw_endothelial_matrix_core", "raw_tgfb_plasticity_core"]
    x = structural[columns].to_numpy(dtype=float)
    x = (x - x.mean(axis=0)) / x.std(axis=0, ddof=0)
    _, singular, vt = np.linalg.svd(x, full_matrices=False)
    coordinates = x @ vt.T
    explained = singular**2 / np.sum(singular**2)
    out = structural[["patient_id", "condition", "age_years", "region"]].copy()
    out["PC1"] = coordinates[:, 0]
    out["PC2"] = coordinates[:, 1]
    out["PC1_variance_fraction"] = explained[0]
    out["PC2_variance_fraction"] = explained[1]
    return out


def plot_results(correlations: pd.DataFrame, contrasts: pd.DataFrame, pca: pd.DataFrame) -> None:
    structural_ids = set(pca.loc[pca["condition"].isin(["Donor", "TOF", "HLHS"]), "patient_id"])
    primary = correlations[
        correlations["patient_id"].isin(structural_ids)
        & correlations["method"].eq("spearman")
        & correlations["pair"].eq("Mean three-pair coupling")
    ].copy()
    if "condition" not in primary.columns:
        primary = primary.merge(pca[["patient_id", "condition"]], on="patient_id", how="left")
    primary["group"] = np.where(primary["condition"].eq("Donor"), "Donor", "Structural CHD")

    fig, axes = plt.subplots(2, 2, figsize=(9.4, 7.4))
    colors = {"Donor": "#4C78A8", "Structural CHD": "#E45756"}
    for ax, subset, title in [
        (axes[0, 0], "all_endothelial_structural_balanced", "A  Individual-balanced structural cohort"),
        (axes[0, 1], "blood_endothelial_structural_balanced", "B  Blood endothelial sensitivity"),
    ]:
        frame = primary[primary["subset"].eq(subset)]
        for x, group_name in enumerate(["Donor", "Structural CHD"]):
            values = frame.loc[frame["group"].eq(group_name), "fisher_z"].to_numpy(dtype=float)
            ax.scatter(np.full(len(values), x) + np.linspace(-0.08, 0.08, len(values)), values, s=42,
                       color=colors[group_name], edgecolor="white", linewidth=0.6, zorder=3)
            ax.plot([x - 0.16, x + 0.16], [np.mean(values), np.mean(values)], color="#222222", lw=2)
        result = contrasts[
            contrasts["subset"].eq(subset)
            & contrasts["method"].eq("spearman")
            & contrasts["pair"].eq("Mean three-pair coupling")
            & contrasts["sensitivity"].eq("all_structural_individuals")
        ].iloc[0]
        ax.text(0.02, 0.96, f"nominal complete-label P={result['exact_p_two_sided']:.3f}", transform=ax.transAxes,
                ha="left", va="top", fontsize=8)
        ax.set_xticks([0, 1], ["Donor\n(n=4)", "CHD\n(n=6)"])
        ax.set_ylabel("Mean Fisher z across three pairs")
        ax.set_title(title, loc="left", fontweight="bold")
        ax.axhline(0, color="#888888", lw=0.7, ls="--")
        ax.grid(axis="y", color="#EEEEEE", lw=0.6)

    pair_frame = contrasts[
        contrasts["subset"].eq("all_endothelial_structural_balanced")
        & contrasts["method"].eq("spearman")
        & contrasts["sensitivity"].eq("all_structural_individuals")
        & contrasts["pair"].ne("Mean three-pair coupling")
    ].copy()
    pair_order = [PAIR_LABELS[p] for p in PAIRS]
    pair_frame["order"] = pair_frame["pair"].map({name: i for i, name in enumerate(pair_order)})
    pair_frame = pair_frame.sort_values("order")
    axes[1, 0].axvline(0, color="#777777", lw=0.8)
    axes[1, 0].scatter(pair_frame["difference_mean_fisher_z_chd_minus_donor"], np.arange(3), s=55, color="#6F4E7C")
    for i, row in pair_frame.reset_index(drop=True).iterrows():
        axes[1, 0].text(row["difference_mean_fisher_z_chd_minus_donor"], i + 0.18,
                        f"P={row['exact_p_two_sided']:.3f}; q={row['fdr_q_three_pairs']:.3f}",
                        ha="center", va="bottom", fontsize=7)
    axes[1, 0].set_yticks(np.arange(3), pair_order)
    # Keep extra lower margin so the final P/q annotation is not clipped.
    axes[1, 0].set_ylim(2.5, -0.5)
    axes[1, 0].set_xlabel("CHD minus donor difference in Fisher z")
    axes[1, 0].set_title("C  Pair-specific coupling contrasts", loc="left", fontweight="bold")
    axes[1, 0].grid(axis="x", color="#EEEEEE", lw=0.6)

    for condition, group in pca.groupby("condition"):
        marker = "o" if condition == "Donor" else "^" if condition in {"TOF", "HLHS"} else "s"
        color = "#4C78A8" if condition == "Donor" else "#E45756" if condition in {"TOF", "HLHS"} else "#999999"
        axes[1, 1].scatter(group["PC1"], group["PC2"], s=55, marker=marker, color=color, label=condition,
                           edgecolor="white", linewidth=0.6)
        for _, row in group.iterrows():
            axes[1, 1].text(row["PC1"] + 0.04, row["PC2"] + 0.03, row["patient_id"], fontsize=6)
    variance = pca.iloc[0]
    axes[1, 1].set_xlabel(f"PC1 ({variance['PC1_variance_fraction'] * 100:.1f}%)")
    axes[1, 1].set_ylabel(f"PC2 ({variance['PC2_variance_fraction'] * 100:.1f}%)")
    axes[1, 1].set_title("D  Descriptive patient-level component PCA", loc="left", fontweight="bold")
    axes[1, 1].legend(frameon=False, fontsize=7, ncol=2)
    axes[1, 1].grid(color="#EEEEEE", lw=0.6)

    for ax in axes.flat:
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Exploratory endothelial module-coupling analysis", x=0.06, ha="left", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(FIGURES / "Figure_S_coupling_evaluation.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURES / "Figure_S_coupling_evaluation.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    target_counts, metadata, validation = load_target_counts()
    module_scores = build_module_scores(target_counts, metadata)
    patient_meta = pd.read_csv(PATIENT_SCORES)
    structural_ids = set(
        patient_meta.loc[
            patient_meta["primary_inclusion"].eq("included")
            & patient_meta["condition"].isin(["Donor", "TOF", "HLHS"]),
            "patient_id",
        ]
    )

    all_mask = pd.Series(True, index=metadata.index)
    blood_mask = metadata["Cluster"].isin(["EC1", "EC2", "EC3", "EC4", "EC5"])
    structural_mask = metadata["individual_id"].isin(structural_ids)
    correlation_frames = []
    downsample_frames = []
    variants = [
        ("all_endothelial", all_mask, False),
        ("blood_endothelial", blood_mask, False),
        ("all_endothelial_structural_balanced", all_mask & structural_mask, True),
        ("blood_endothelial_structural_balanced", blood_mask & structural_mask, True),
    ]
    for subset, mask, balance_individuals in variants:
        residuals = residualize(module_scores, metadata, mask, balance_individuals=balance_individuals)
        correlation_frames.append(patient_correlations(residuals, metadata, subset))
        downsample_frames.append(downsample_coupling(residuals, metadata, subset))
    correlations = pd.concat(correlation_frames, ignore_index=True)
    contrasts = group_contrasts(correlations, patient_meta)
    correlations = correlations.merge(
        patient_meta[["patient_id", "condition", "primary_inclusion", "high_contamination", "age_years", "region"]],
        on="patient_id",
        how="left",
    )
    downsample = pd.concat(downsample_frames, ignore_index=True).merge(
        patient_meta[["patient_id", "condition", "primary_inclusion", "high_contamination"]],
        on="patient_id",
        how="left",
    )
    pca = pca_patient_scores(patient_meta)

    correlations.to_csv(TABLES / "S_discovery_patient_module_coupling.csv", index=False, encoding="utf-8-sig")
    contrasts.to_csv(TABLES / "S_discovery_coupling_group_contrasts.csv", index=False, encoding="utf-8-sig")
    downsample.to_csv(TABLES / "S_discovery_coupling_equal_cell_downsampling.csv", index=False, encoding="utf-8-sig")
    pca.to_csv(TABLES / "S_discovery_component_PCA.csv", index=False, encoding="utf-8-sig")
    plot_results(correlations, contrasts, pca)

    primary = contrasts[
        contrasts["subset"].eq("all_endothelial_structural_balanced")
        & contrasts["method"].eq("spearman")
        & contrasts["pair"].eq("Mean three-pair coupling")
        & contrasts["sensitivity"].eq("all_structural_individuals")
    ].iloc[0]
    clean = contrasts[
        contrasts["subset"].eq("all_endothelial_structural_balanced")
        & contrasts["method"].eq("spearman")
        & contrasts["pair"].eq("Mean three-pair coupling")
        & contrasts["sensitivity"].eq("high_contamination_excluded")
    ].iloc[0]
    downsample_primary = downsample[
        downsample["subset"].eq("all_endothelial_structural_balanced")
        & downsample["condition"].isin(["Donor", "TOF", "HLHS"])
    ].copy()
    downsample_difference = float(
        downsample_primary.loc[~downsample_primary["condition"].eq("Donor"), "mean_fisher_z"].mean()
        - downsample_primary.loc[downsample_primary["condition"].eq("Donor"), "mean_fisher_z"].mean()
    )
    module_gene_lists = list(MODULES.values())
    module_gene_overlap = len([gene for genes in module_gene_lists for gene in genes]) - len(
        set(gene for genes in module_gene_lists for gene in genes)
    )
    summary = {
        "source_validation": {
            **validation,
            "module_gene_counts": {name: len(genes) for name, genes in MODULES.items()},
            "module_gene_overlap_count": module_gene_overlap,
        },
        "primary_all_endothelial_spearman": primary.to_dict(),
        "clean_sensitivity_all_endothelial_spearman": clean.to_dict(),
        "equal_cell_downsampling_all_endothelial_spearman": {
            "subset": "all_endothelial_structural_balanced",
            "n_chd": 6,
            "n_donor": 4,
            "difference_mean_fisher_z_chd_minus_donor": downsample_difference,
            "interpretation": "Directional cell-count sensitivity only; no inferential P value was assigned.",
        },
        "interpretation_rule": (
            "A lower CHD coupling estimate with concordant sensitivity analyses would support exploratory uncoupling. "
            "A null, unstable, or positive estimate does not support recasting AMA as coupling failure. "
            "All group summaries use the biological individual as the permutation and inferential unit; "
            "age and region confounding limit label exchangeability and disease-effect interpretation."
        ),
    }
    summary = json_safe(summary)
    (OUT / "analysis_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False, default=str, allow_nan=False))
    print(f"Outputs written to {OUT}")


if __name__ == "__main__":
    main()
