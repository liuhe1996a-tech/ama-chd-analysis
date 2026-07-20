from __future__ import annotations

import itertools
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT = Path(os.environ.get("AMA_PROJECT_ROOT", Path(__file__).resolve().parents[1])).resolve()
DATA = Path(os.environ.get("AMA_PROCESSED_DATA_DIR", PROJECT / "data" / "processed"))
OUT = Path(os.environ.get("AMA_OUTPUT_DIR", PROJECT / "results" / "reproduced"))
TABLES = OUT / "tables"
FIGURES = OUT / "figures"
TABLES.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)
FETAL = DATA / "fetal_endothelial_cell_module_scores.csv.gz"
SPATIAL = DATA / "farah_overall_endothelial_spatial_scores.csv.gz"


POP_ORDER = [
    "BEC-Arterial",
    "BEC-Capillary",
    "BEC-Venous",
    "BEC-Proliferating",
    "LEC",
    "aEndocardial",
    "Endocardial",
    "vEndocardial",
]


def zscore(values: pd.Series) -> pd.Series:
    sd = values.std(ddof=0)
    if not np.isfinite(sd) or sd == 0:
        return pd.Series(np.zeros(len(values)), index=values.index)
    return (values - values.mean()) / sd


def exact_sign_flip(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    observed = float(values.mean())
    null = np.asarray(
        [np.mean(values * np.asarray(signs)) for signs in itertools.product([-1.0, 1.0], repeat=len(values))]
    )
    return {
        "observed_mean": observed,
        "exact_one_sided_p": float(np.mean(null >= observed - 1e-12)),
        "exact_two_sided_p": float(np.mean(np.abs(null) >= abs(observed) - 1e-12)),
        "n_technical_sections": int(len(values)),
        "n_biological_hearts": 1,
    }


def fetal_analysis() -> tuple[pd.DataFrame, pd.DataFrame]:
    cells = pd.read_csv(FETAL, index_col=0)
    cells["donor_id"] = cells["Age"].astype(str).str.zfill(2) + "_rep" + cells["Rep"].astype(str)
    cells["z_matrix"] = zscore(cells["ECM_integrin"])
    cells["z_tgfb"] = zscore(cells["EndMT_TGFb"])
    cells["z_arterial"] = zscore(cells["NOTCH_arterial"])
    cells["matrix_tgfb_attenuation"] = -(cells["z_matrix"] + cells["z_tgfb"]) / 2.0

    # Donor-population means are the analysis units; regions within a heart are pooled.
    donor_population = (
        cells.groupby(["donor_id", "Age", "Rep", "populations"], observed=True)
        .agg(
            n_cells=("matrix_tgfb_attenuation", "size"),
            matrix_tgfb_attenuation=("matrix_tgfb_attenuation", "mean"),
            arterial_identity_z=("z_arterial", "mean"),
        )
        .reset_index()
    )
    donor_population.to_csv(
        TABLES / "S_fetal_leave_arterial_out_donor_population_scores.csv", index=False, encoding="utf-8-sig"
    )

    # Residualization controls arterial identity and pairs populations within donor.
    donor_dummies = pd.get_dummies(donor_population["donor_id"], drop_first=True, dtype=float)
    design = np.column_stack(
        [
            np.ones(len(donor_population)),
            donor_population["arterial_identity_z"].to_numpy(dtype=float),
            donor_dummies.to_numpy(dtype=float),
        ]
    )
    outcome = donor_population["matrix_tgfb_attenuation"].to_numpy(dtype=float)
    coefficients = np.linalg.lstsq(design, outcome, rcond=None)[0]
    donor_population["arterial_identity_donor_adjusted_residual"] = outcome - design @ coefficients

    summary = (
        donor_population.groupby("populations", observed=True)
        .agg(
            n_donors=("donor_id", "nunique"),
            mean_leave_arterial_out_score=("matrix_tgfb_attenuation", "mean"),
            sd_leave_arterial_out_score=("matrix_tgfb_attenuation", "std"),
            mean_adjusted_residual=("arterial_identity_donor_adjusted_residual", "mean"),
            sd_adjusted_residual=("arterial_identity_donor_adjusted_residual", "std"),
        )
        .reset_index()
    )
    summary.to_csv(TABLES / "S_fetal_leave_arterial_out_population_summary.csv", index=False, encoding="utf-8-sig")

    coef_rows = []
    for pop in [p for p in POP_ORDER if p != "BEC-Arterial"]:
        paired = donor_population[donor_population["populations"].isin(["BEC-Arterial", pop])].pivot(
            index="donor_id", columns="populations", values="arterial_identity_donor_adjusted_residual"
        )
        paired = paired.dropna()
        differences = paired[pop] - paired["BEC-Arterial"]
        perm = exact_sign_flip(differences.to_numpy()) if len(differences) else {}
        coef_rows.append(
            {
                "contrast": f"{pop} minus BEC-Arterial",
                "n_donors": len(differences),
                "mean_adjusted_residual_difference": float(differences.mean()) if len(differences) else np.nan,
                "exact_two_sided_sign_flip_p": perm.get("exact_two_sided_p", np.nan),
            }
        )
    contrasts = pd.DataFrame(coef_rows)
    contrasts.to_csv(TABLES / "S_fetal_population_paired_contrasts.csv", index=False, encoding="utf-8-sig")

    plot = summary.set_index("populations").reindex(POP_ORDER).dropna(how="all")
    fig, ax = plt.subplots(figsize=(7.4, 4.7))
    y = np.arange(len(plot))
    ax.barh(y, plot["mean_adjusted_residual"], color="#4C78A8", alpha=0.88)
    ax.axvline(0, color="#333333", lw=0.8)
    ax.set_yticks(y, plot.index)
    ax.invert_yaxis()
    ax.set_xlabel("Matrix–TGFβ attenuation residual\n(adjusted for arterial identity and donor)")
    ax.set_title("Fetal endothelial mapping without the arterial module")
    fig.tight_layout()
    fig.savefig(FIGURES / "S_fetal_leave_arterial_out_population_mapping.png", dpi=600)
    fig.savefig(FIGURES / "S_fetal_leave_arterial_out_population_mapping.pdf")
    plt.close(fig)
    return summary, contrasts


def spatial_analysis() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    cells = pd.read_csv(SPATIAL, index_col=0)
    mask = cells["populations"].eq("BEC") & cells["communities"].astype(str).str.match(r"^(Outer|Inner)-(LV|RV)$")
    bec = cells.loc[mask].copy()
    means = (
        bec.groupby(["sample_id", "communities"], observed=True)
        .agg(n_cells=("coarse_attenuation_proxy", "size"), mean_coarse_attenuation=("coarse_attenuation_proxy", "mean"))
        .reset_index()
    )
    means.to_csv(TABLES / "S_spatial_within_BEC_outer_inner_means.csv", index=False, encoding="utf-8-sig")
    wide = means.pivot(index="sample_id", columns="communities", values="mean_coarse_attenuation")
    rows = []
    for sample_id in wide.index:
        for ventricle in ["LV", "RV"]:
            outer = float(wide.loc[sample_id, f"Outer-{ventricle}"])
            inner = float(wide.loc[sample_id, f"Inner-{ventricle}"])
            rows.append(
                {
                    "sample_id": sample_id,
                    "ventricle": ventricle,
                    "outer_BEC_mean": outer,
                    "inner_BEC_mean": inner,
                    "outer_minus_inner_BEC": outer - inner,
                }
            )
    contrasts = pd.DataFrame(rows)
    contrasts.to_csv(TABLES / "S_spatial_within_BEC_outer_inner_contrasts.csv", index=False, encoding="utf-8-sig")
    section = contrasts.groupby("sample_id", observed=True)["outer_minus_inner_BEC"].mean().reset_index()
    section_stats = exact_sign_flip(section["outer_minus_inner_BEC"].to_numpy())
    section_stats.update(
        {
            "analysis": "within-BEC, LV/RV averaged within each section",
            "interpretation": "technical-section sensitivity analysis from one 13-pcw heart",
        }
    )
    (TABLES / "S_spatial_within_BEC_exact_sign_flip.json").write_text(
        json.dumps(section_stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    fig, ax = plt.subplots(figsize=(6.0, 4.5))
    colors = {"LV": "#4C78A8", "RV": "#F58518"}
    for ventricle, group in contrasts.groupby("ventricle", observed=True):
        ax.scatter(
            group["inner_BEC_mean"], group["outer_BEC_mean"], s=55, color=colors[ventricle], label=ventricle
        )
        for _, row in group.iterrows():
            ax.plot(
                [row["inner_BEC_mean"], row["outer_BEC_mean"]],
                [row["inner_BEC_mean"], row["outer_BEC_mean"]],
                color=colors[ventricle], alpha=0.18,
            )
    lo = float(min(contrasts["inner_BEC_mean"].min(), contrasts["outer_BEC_mean"].min()) - 0.05)
    hi = float(max(contrasts["inner_BEC_mean"].max(), contrasts["outer_BEC_mean"].max()) + 0.05)
    ax.plot([lo, hi], [lo, hi], color="#333333", lw=0.8, ls="--")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Inner-wall BEC coarse attenuation proxy")
    ax.set_ylabel("Outer-wall BEC coarse attenuation proxy")
    ax.set_title("Outer–inner contrast after restricting to BECs")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES / "S_spatial_within_BEC_outer_inner.png", dpi=600)
    fig.savefig(FIGURES / "S_spatial_within_BEC_outer_inner.pdf")
    plt.close(fig)
    return means, contrasts, section_stats


def main() -> None:
    fetal_summary, fetal_contrasts = fetal_analysis()
    spatial_means, spatial_contrasts, spatial_stats = spatial_analysis()
    print("FETAL LEAVE-ARTERIAL-OUT SUMMARY")
    print(fetal_summary.to_string(index=False))
    print("\nFETAL PAIRED CONTRASTS")
    print(fetal_contrasts.to_string(index=False))
    print("\nSPATIAL WITHIN-BEC CONTRASTS")
    print(spatial_contrasts.to_string(index=False))
    print("\nSPATIAL SECTION-LEVEL EXACT TEST")
    print(json.dumps(spatial_stats, indent=2))


if __name__ == "__main__":
    main()
