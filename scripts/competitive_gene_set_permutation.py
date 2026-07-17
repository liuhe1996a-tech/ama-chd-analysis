from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT = Path(os.environ.get("AMA_PROJECT_ROOT", Path(__file__).resolve().parents[1])).resolve()
DATA = Path(os.environ.get("AMA_PROCESSED_DATA_DIR", PROJECT / "data" / "processed"))
OUT = Path(os.environ.get("AMA_OUTPUT_DIR", PROJECT / "results" / "reproduced")) / "tables"
OUT.mkdir(parents=True, exist_ok=True)


ARTERIAL = ["NOTCH1", "DLL4", "JAG1", "HEY1", "HEY2", "EFNB2", "GJA5", "SOX17"]
ECM_CORE = ["FN1", "VCAN", "COL4A1", "COL4A2", "LAMA4", "ITGA5", "ITGB1"]
TGFB_CORE = ["TGFB1", "TGFB2", "TGFBR1", "TGFBR2", "SMAD2", "SMAD3", "SNAI1", "SNAI2"]
MODULES = {
    "arterial_maturation": ARTERIAL,
    "endothelial_matrix_core": ECM_CORE,
    "tgfb_plasticity_core": TGFB_CORE,
    "combined_unique_AMA_genes": list(dict.fromkeys(ARTERIAL + ECM_CORE + TGFB_CORE)),
}

INPUTS = {
    "age_sex_contamination_adjusted": DATA / "DE_CHD_vs_Donor_adjusted_cm_fraction.csv.gz",
    "high_contamination_excluded": DATA / "DE_CHD_vs_Donor_clean_samples_only.csv.gz",
}



def matched_null(
    table: pd.DataFrame,
    genes: list[str],
    iterations: int,
    seed: int,
) -> tuple[dict[str, float | int | str], pd.DataFrame]:
    work = table.copy()
    work = work[
        np.isfinite(work["baseMean"].to_numpy(dtype=float))
        & np.isfinite(work["log2FoldChange"].to_numpy(dtype=float))
        & np.isfinite(work["lfcSE"].to_numpy(dtype=float))
        & (work["lfcSE"].to_numpy(dtype=float) > 0)
    ]
    work = work[~work.index.duplicated(keep="first")]
    present = [gene for gene in genes if gene in work.index]
    excluded = set(genes)
    universe = work.index[~work.index.isin(excluded)].tolist()
    log_base = np.log10(work["baseMean"].clip(lower=1e-8))
    pct = log_base.rank(pct=True, method="average")
    bins = np.minimum((pct * 20).astype(int), 19)

    pools: dict[str, list[str]] = {}
    for gene in present:
        gene_bin = int(bins.loc[gene])
        pools[gene] = [candidate for candidate in universe if int(bins.loc[candidate]) == gene_bin]

    observed_mean = -float(work.loc[present, "log2FoldChange"].mean())
    observed_median = -float(work.loc[present, "log2FoldChange"].median())
    observed_stouffer = -float(
        np.sum(work.loc[present, "log2FoldChange"] / work.loc[present, "lfcSE"])
        / np.sqrt(len(present))
    )

    rng = np.random.default_rng(seed)
    null_mean = np.empty(iterations)
    null_median = np.empty(iterations)
    null_stouffer = np.empty(iterations)
    for iteration in range(iterations):
        used: set[str] = set()
        selected = []
        for target in present:
            pool = [candidate for candidate in pools[target] if candidate not in used]
            if not pool:
                pool = pools[target]
            chosen = str(rng.choice(pool))
            selected.append(chosen)
            used.add(chosen)
        lfc = work.loc[selected, "log2FoldChange"].to_numpy(dtype=float)
        se = work.loc[selected, "lfcSE"].to_numpy(dtype=float)
        null_mean[iteration] = -float(np.mean(lfc))
        null_median[iteration] = -float(np.median(lfc))
        null_stouffer[iteration] = -float(np.sum(lfc / se) / np.sqrt(len(selected)))

    summary = {
        "n_genes_present": len(present),
        "genes_present": ";".join(present),
        "observed_negative_mean_lfc": observed_mean,
        "empirical_p_mean_one_sided": (1 + int(np.sum(null_mean >= observed_mean))) / (iterations + 1),
        "observed_negative_median_lfc": observed_median,
        "empirical_p_median_one_sided": (1 + int(np.sum(null_median >= observed_median))) / (iterations + 1),
        "observed_negative_stouffer_z": observed_stouffer,
        "empirical_p_stouffer_one_sided": (1 + int(np.sum(null_stouffer >= observed_stouffer))) / (iterations + 1),
        "iterations": iterations,
        "matching": "20 quantile bins of log10 baseMean; target genes excluded; no replacement within draw",
    }
    null = pd.DataFrame(
        {
            "iteration": np.arange(1, iterations + 1),
            "negative_mean_lfc": null_mean,
            "negative_median_lfc": null_median,
            "negative_stouffer_z": null_stouffer,
        }
    )
    return summary, null


def main() -> None:
    summary_rows = []
    null_frames = []
    for analysis, path in INPUTS.items():
        table = pd.read_csv(path, index_col=0)
        table.index = table.index.astype(str)
        for module_index, (module, genes) in enumerate(MODULES.items()):
            summary, null = matched_null(table, genes, iterations=10_000, seed=203274 + module_index)
            summary_rows.append({"analysis": analysis, "module": module, **summary})
            null.insert(0, "module", module)
            null.insert(0, "analysis", analysis)
            null_frames.append(null)
    summary_df = pd.DataFrame(summary_rows)
    null_df = pd.concat(null_frames, ignore_index=True)
    summary_df.to_csv(OUT / "S_competitive_gene_set_permutation_summary.csv", index=False, encoding="utf-8-sig")
    null_df.to_csv(OUT / "S_competitive_gene_set_permutation_values.csv", index=False, encoding="utf-8-sig")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
