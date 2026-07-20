from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd


DATA = ROOT / "results"
V7_TABLES = DATA / "tables"
V8_TABLES = DATA / "tables"
OUT = DATA / "figures" / "main"
PANELS = DATA / "figures" / "individual_panels"
OUT.mkdir(parents=True, exist_ok=True)
PANELS.mkdir(parents=True, exist_ok=True)

BLUE = "#4C78A8"
ORANGE = "#F58518"
GREEN = "#54A24B"
RED = "#E45756"
PURPLE = "#B24C8A"
GRAY = "#6B7280"

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


def save(fig: plt.Figure, stem: Path, dpi: int = 600) -> None:
    fig.savefig(stem.with_suffix(".png"), dpi=dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.10, 1.06, label, transform=ax.transAxes, fontsize=14, fontweight="bold", va="top")


def model_spec_panel(ax: plt.Axes, specification: pd.DataFrame) -> None:
    labels = {
        "Unadjusted OLS": "Unadjusted",
        "CHD + age": "+ age",
        "CHD + sex": "+ sex",
        "CHD + contamination": "+ contamination",
        "CHD + age + contamination": "+ age + contamination",
        "CHD + age + sex": "+ age + sex",
        "Full: CHD + age + sex + contamination": "Full model",
    }
    data = specification.iloc[::-1].reset_index(drop=True)
    y = np.arange(len(data))
    for i, row in data.iterrows():
        color = RED if row["beta"] > 0 else BLUE
        ax.plot([row["ci_low"], row["ci_high"]], [i, i], color=color, lw=1.4)
        ax.scatter(row["beta"], i, color=color, s=27, zorder=3)
    ax.axvline(0, color="#555555", lw=0.8, ls="--")
    ax.set_yticks(y, [labels[x] for x in data["specification"]], fontsize=7.5)
    ax.set_xlabel("CHD coefficient (95% CI)")
    ax.set_title("Model-specification curve", fontsize=11, fontweight="bold")
    ax.grid(axis="x", color="#E9E9E9", lw=0.6)


def loio_panel(ax: plt.Axes, loio: pd.DataFrame) -> None:
    condition_colors = {"Donor": BLUE, "TOF": ORANGE, "HLHS": PURPLE}
    data = loio.sort_values(["omitted_condition", "omitted_patient"]).reset_index(drop=True)
    y = np.arange(len(data))[::-1]
    for i, row in data.iterrows():
        color = condition_colors[row["omitted_condition"]]
        ax.plot([row["ci_low"], row["ci_high"]], [y[i], y[i]], color=color, lw=1.3)
        ax.scatter(row["beta"], y[i], color=color, s=25, zorder=3)
    ax.axvline(0, color="#555555", lw=0.8, ls="--")
    ax.set_yticks(y, [f"omit {x}" for x in data["omitted_patient"]], fontsize=7.2)
    ax.set_xlabel("Full-model CHD coefficient (95% CI)")
    ax.set_title("Leave-one-individual-out", fontsize=11, fontweight="bold")
    ax.grid(axis="x", color="#E9E9E9", lw=0.6)


def figure2() -> None:
    scores = pd.read_csv(V7_TABLES / "S_discovery_patient_module_scores.csv")
    specification = pd.read_csv(V8_TABLES / "S_discovery_model_specification_curve.csv")
    loio = pd.read_csv(V8_TABLES / "S_discovery_leave_one_individual_out.csv")
    influence = pd.read_csv(V8_TABLES / "S_discovery_influence_diagnostics.csv")
    disease = pd.read_csv(V7_TABLES / "S_discovery_disease_specificity_effects.csv")

    fig = plt.figure(figsize=(15.6, 9.2))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 0.9], wspace=0.48, hspace=0.45)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])
    ax_d = fig.add_subplot(gs[1, 0:2])
    ax_e = fig.add_subplot(gs[1, 2])

    panel_label(ax_a, "A")
    order = ["Donor", "TOF", "HLHS", "Cardiomyopathy"]
    colors = {"Donor": GRAY, "TOF": ORANGE, "HLHS": BLUE, "Cardiomyopathy": PURPLE}
    for i, group in enumerate(order):
        values = scores.loc[scores["condition"].eq(group), "AMA"].to_numpy(dtype=float)
        offsets = np.linspace(-0.09, 0.09, len(values)) if len(values) else []
        ax_a.scatter(i + offsets, values, s=52, color=colors[group], edgecolor="white", linewidth=0.6, zorder=3)
        if len(values):
            ax_a.hlines(np.mean(values), i - 0.23, i + 0.23, color="#222222", lw=1.1)
    ax_a.axhline(0, color="#888888", lw=0.8, ls="--")
    ax_a.set_xticks(range(len(order)), order, rotation=20, ha="right")
    ax_a.set_ylabel("Donor-standardized AMA")
    ax_a.set_title("Patient-level discovery scores", fontsize=11, fontweight="bold")

    panel_label(ax_b, "B")
    model_spec_panel(ax_b, specification)
    panel_label(ax_c, "C")
    loio_panel(ax_c, loio)

    panel_label(ax_d, "D")
    x = np.arange(len(influence))
    width = 0.36
    cooks_ratio = influence["cooks_distance"] / influence["cooks_reference_4_over_n"]
    dfbeta_ratio = influence["dfbeta_structural_chd"].abs() / influence["dfbeta_reference_2_over_sqrt_n"]
    ax_d.bar(x - width / 2, cooks_ratio, width=width, color=BLUE, label="Cook's D / (4/n)")
    ax_d.bar(x + width / 2, dfbeta_ratio, width=width, color=ORANGE, label="|DFBETA| / (2/sqrt(n))")
    ax_d.axhline(1, color="#555555", ls="--", lw=0.9, label="reference threshold")
    ax_d.set_xticks(x, influence["patient_id"], rotation=35, ha="right", fontsize=7.5)
    ax_d.set_ylabel("Influence relative to reference threshold")
    ax_d.set_title("Patient-level influence diagnostics", fontsize=11, fontweight="bold")
    ax_d.legend(frameon=False, fontsize=7.5, ncol=3, loc="upper left")
    ax_d.grid(axis="y", color="#EEEEEE", lw=0.6)

    panel_label(ax_e, "E")
    data = disease.iloc[::-1].reset_index(drop=True)
    y = np.arange(len(data))
    ax_e.errorbar(
        data["adjusted_beta"],
        y,
        xerr=[data["adjusted_beta"] - data["adjusted_ci_low"], data["adjusted_ci_high"] - data["adjusted_beta"]],
        fmt="o",
        color=PURPLE,
        capsize=2,
    )
    ax_e.axvline(0, color="#555555", lw=0.8)
    ax_e.set_yticks(y, data["contrast"].str.replace("_", " ", regex=False), fontsize=7.2)
    ax_e.set_xlabel("Adjusted coefficient (95% CI)")
    ax_e.set_title("Disease-context contrasts", fontsize=11, fontweight="bold")
    fig.suptitle("Discovery effect is model- and individual-sensitive", fontsize=15, fontweight="bold", y=0.99)
    save(fig, OUT / "Figure_2")

    # Export the principal new panels separately for flexible assembly.
    fig, ax = plt.subplots(figsize=(6.7, 4.2)); model_spec_panel(ax, specification); fig.tight_layout(); save(fig, PANELS / "Figure2_model_specification")
    fig, ax = plt.subplots(figsize=(6.2, 4.8)); loio_panel(ax, loio); fig.tight_layout(); save(fig, PANELS / "Figure2_leave_one_individual_out")


def forest(ax: plt.Axes, values: pd.DataFrame, beta: str, low: str, high: str, labels: list[str], color: str) -> None:
    y = np.arange(len(values))
    ax.errorbar(
        values[beta],
        y,
        xerr=[values[beta] - values[low], values[high] - values[beta]],
        fmt="o",
        color=color,
        capsize=3,
    )
    ax.axvline(0, color="#555555", lw=0.8)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.grid(axis="x", color="#EEEEEE", lw=0.6)


def figure3() -> None:
    effects = pd.read_csv(V7_TABLES / "S_external_main_effects.csv")
    meta_input = pd.read_csv(V7_TABLES / "S_external_meta_hedges_g_inputs.csv")
    meta = pd.read_csv(V7_TABLES / "S_external_meta_reml_hk.csv").iloc[0]
    composition = pd.read_csv(V8_TABLES / "S_bulk_composition_sensitivity_effects.csv")
    primary = [
        "GSE217772_TOF_RV_vs_healthy_RV",
        "GSE36761_TOF_RV_vs_healthy_RV",
        "GSE23959_HLHS_ventricle_vs_healthy_RV",
    ]
    labels = ["GSE217772 TOF RV", "GSE36761 TOF RV", "GSE23959 HLHS ventricle"]

    fig, axes = plt.subplots(2, 2, figsize=(13.8, 9.4), gridspec_kw={"hspace": 0.46, "wspace": 0.50})
    ax = axes[0, 0]; panel_label(ax, "A")
    y = np.arange(3, 0, -1)
    ax.errorbar(meta_input["hedges_g"], y,
                xerr=[meta_input["hedges_g"] - meta_input["ci_low_normal"], meta_input["ci_high_normal"] - meta_input["hedges_g"]],
                fmt="o", color=BLUE, capsize=3)
    ax.errorbar(meta["pooled_hedges_g"], 0,
                xerr=[[meta["pooled_hedges_g"] - meta["ci95_low"]], [meta["ci95_high"] - meta["pooled_hedges_g"]]],
                fmt="D", color=RED, capsize=4)
    ax.axvline(0, color="#555555", lw=0.8)
    ax.set_yticks([3, 2, 1, 0], labels + ["REML + modified HK"])
    ax.set_xlabel("Hedges g (higher = stronger AMA)")
    ax.set_title("Exploratory ventricular meta-analysis", fontsize=11, fontweight="bold")

    ama = effects[effects["module"].eq("AMA") & effects["contrast"].isin(primary)].set_index("contrast").reindex(primary).reset_index()
    ax = axes[0, 1]; panel_label(ax, "B")
    forest(ax, ama, "delta_case_minus_control", "ci95_low", "ci95_high", labels, GREEN)
    ax.set_xlabel("Unadjusted AMA delta (95% CI)")
    ax.set_title("Whole-tissue point estimates", fontsize=11, fontweight="bold")

    adjusted = composition[composition["model"].eq("adjusted_all_five_marker_scores")].set_index("contrast").reindex(primary).reset_index()
    ax = axes[1, 0]; panel_label(ax, "C")
    forest(ax, adjusted, "beta", "ci_low", "ci_high", labels, RED)
    ax.set_xlabel("Composition-adjusted disease coefficient (95% CI)")
    ax.set_title("Five-marker-score sensitivity", fontsize=11, fontweight="bold")
    for i, row in adjusted.iterrows():
        ax.text(row["ci_high"] + 0.08, i, f"P={row['p_value']:.3f}", va="center", fontsize=7.3)
    left = min(float(adjusted["ci_low"].min()) - 0.3, -2.6)
    right = max(float(adjusted["ci_high"].max()) + 0.8, 3.4)
    ax.set_xlim(left, right)

    boundary_names = [
        "GSE132176_TOF_PRE_CPB_vs_ASD_PRE_CPB",
        "GSE132176_TOF_POST_vs_PRE_CPB_paired",
        "GSE132176_ASD_POST_vs_PRE_CPB_paired",
    ]
    boundary_labels = ["TOF vs ASD pre-CPB", "TOF post vs pre-CPB", "ASD post vs pre-CPB"]
    boundary = effects[effects["module"].eq("AMA") & effects["contrast"].isin(boundary_names)].set_index("contrast").reindex(boundary_names).reset_index()
    ax = axes[1, 1]; panel_label(ax, "D")
    forest(ax, boundary, "delta_case_minus_control", "ci95_low", "ci95_high", boundary_labels, ORANGE)
    ax.set_xlabel("AMA delta (95% CI)")
    ax.set_title("Atrial and CPB context", fontsize=11, fontweight="bold")
    fig.suptitle("External tissue estimates are composition- and context-sensitive", fontsize=15, fontweight="bold", y=0.99)
    save(fig, OUT / "Figure_3")


def add_box(ax, xy, width, height, title, body, face, edge, dashed=False, fontsize=8.6):
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        facecolor=face,
        edgecolor=edge,
        linewidth=1.3,
        linestyle="--" if dashed else "-",
    )
    ax.add_patch(patch)
    ax.text(xy[0] + 0.018, xy[1] + height - 0.025, title, fontsize=fontsize + 0.8, fontweight="bold", va="top")
    ax.text(xy[0] + 0.018, xy[1] + height - 0.085, body, fontsize=fontsize, va="top", linespacing=1.18)
    return patch


def add_arrow(ax, start, end, color=GRAY, dashed=True):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=11, linewidth=1.2, color=color, linestyle="--" if dashed else "-"))


def figure4() -> None:
    fetal = pd.read_csv(V7_TABLES / "S_fetal_leave_arterial_out_population_summary.csv")
    contrasts = pd.read_csv(V7_TABLES / "S_fetal_population_paired_contrasts.csv")
    spatial = pd.read_csv(V7_TABLES / "S_spatial_within_BEC_outer_inner_contrasts.csv")
    spatial_map = DATA / "figures" / "source_analysis" / "farah_overall_coarse_attenuation_spatial_by_sample.png"

    fig = plt.figure(figsize=(18.5, 13.4))
    outer = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.08], hspace=0.36)
    top = outer[0].subgridspec(1, 2, wspace=0.38)
    bottom = outer[1].subgridspec(1, 2, width_ratios=[1.55, 0.78], wspace=0.30)
    ax_a = fig.add_subplot(top[0, 0])
    ax_b = fig.add_subplot(top[0, 1])
    ax_c = fig.add_subplot(bottom[0, 0])
    ax_d = fig.add_subplot(bottom[0, 1])

    pop_order = ["BEC-Arterial", "BEC-Capillary", "BEC-Proliferating", "BEC-Venous", "LEC", "aEndocardial", "Endocardial", "vEndocardial"]
    f = fetal.set_index("populations").reindex(pop_order)
    panel_label(ax_a, "A")
    ax_a.barh(np.arange(len(f)), f.mean_adjusted_residual, color=BLUE)
    ax_a.axvline(0, color="#444444", lw=0.8)
    ax_a.set_yticks(np.arange(len(f)), f.index)
    ax_a.invert_yaxis()
    ax_a.set_xlabel("Arterial-identity/donor-adjusted residual")
    ax_a.set_title("Leave-arterial-out fetal mapping", fontsize=12.5, fontweight="bold")

    panel_label(ax_b, "B")
    c = contrasts.iloc[::-1].reset_index(drop=True)
    ax_b.scatter(c.mean_adjusted_residual_difference, np.arange(len(c)), color="#8F6BB3", s=52)
    ax_b.axvline(0, color="#444444", lw=0.8)
    ax_b.set_yticks(np.arange(len(c)), c.contrast.str.replace(" minus BEC-Arterial", "", regex=False), fontsize=9)
    ax_b.set_xlabel("Adjusted residual difference vs arterial BEC")
    ax_b.set_title("Paired contrasts across eight donors", fontsize=12.5, fontweight="bold")

    panel_label(ax_c, "C")
    image = plt.imread(spatial_map)
    ax_c.imshow(image, interpolation="nearest")
    ax_c.axis("off")
    ax_c.set_title("Coarse proxy in three sections from one 13-pcw heart", fontsize=12, fontweight="bold", pad=10)

    panel_label(ax_d, "D")
    colors = {"LV": BLUE, "RV": ORANGE}
    for ventricle, group in spatial.groupby("ventricle"):
        ax_d.scatter(group.inner_BEC_mean, group.outer_BEC_mean, s=70, color=colors[ventricle], label=ventricle)
    lo = min(spatial.inner_BEC_mean.min(), spatial.outer_BEC_mean.min()) - 0.04
    hi = max(spatial.inner_BEC_mean.max(), spatial.outer_BEC_mean.max()) + 0.04
    ax_d.plot([lo, hi], [lo, hi], ls="--", color="#555555", lw=0.9)
    ax_d.set_xlim(lo, hi)
    ax_d.set_ylim(lo, hi)
    ax_d.set_xlabel("Inner-wall BEC proxy")
    ax_d.set_ylabel("Outer-wall BEC proxy")
    ax_d.set_title("Within-BEC outer–inner sensitivity\nsection-level exact two-sided P=0.25", fontsize=11.5, fontweight="bold")
    ax_d.legend(frameon=False)

    fig.suptitle("Developmental and spatial contextualization with circularity and composition checks", fontsize=17, fontweight="bold", y=0.985)
    save(fig, OUT / "Figure_4", dpi=700)

    # Preserve an unreduced standalone spatial panel for manual assembly.
    import shutil

    shutil.copy2(spatial_map, PANELS / "Figure4_C_spatial_overview_full_resolution.png")


def figure5() -> None:
    fig, ax = plt.subplots(figsize=(16.2, 9.8))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.text(0.5, 0.95, "Provisional AMA transcript framework and perturbation boundary", ha="center", fontsize=17, fontweight="bold")
    ax.text(0.5, 0.907, "Dashed arrows denote proposed or contextual relationships, not demonstrated causality.", ha="center", fontsize=10, color="#555555")

    add_box(ax, (0.03, 0.59), 0.22, 0.25, "CHD-associated contexts",
            "abnormal morphogenesis\npressure/volume loading\nflow and oxygen differences\nage, chamber and region confounding",
            "#FFF4DD", ORANGE, fontsize=9)
    add_box(ax, (0.31, 0.71), 0.31, 0.13, "Arterial maturation transcripts",
            "NOTCH1 / DLL4 / JAG1 / HEY1 / HEY2\nEFNB2 / GJA5 / SOX17", "#E8F1FA", BLUE)
    add_box(ax, (0.31, 0.50), 0.31, 0.13, "Endothelial-matrix interface transcripts",
            "FN1 / VCAN / COL4A1 / COL4A2\nLAMA4 / ITGA5 / ITGB1", "#EEF7EA", GREEN)
    add_box(ax, (0.31, 0.29), 0.31, 0.13, "TGF-beta/plasticity transcript module",
            "TGFB1/2 / TGFBR1/2 / SMAD2/3 / SNAI1/2\ncomponent abundance is not pathway activity", "#F3EAF8", PURPLE)
    add_box(ax, (0.70, 0.59), 0.27, 0.22, "Frozen AMA transcript score",
            "negative mean of donor-standardized modules\nadjusted discovery direction is model-sensitive\nnot CHD-specific, diagnostic or causal",
            "#F5F7FA", GRAY, fontsize=9)

    for target_y in (0.775, 0.565, 0.355):
        add_arrow(ax, (0.25, 0.715), (0.31, target_y))
    add_arrow(ax, (0.62, 0.775), (0.70, 0.72))
    add_arrow(ax, (0.62, 0.565), (0.70, 0.69))
    add_arrow(ax, (0.62, 0.355), (0.70, 0.66))

    add_box(ax, (0.03, 0.03), 0.43, 0.18, "Public RNF20-loss perturbation [38]",
            "Frozen AMA genes were competitively more negative in\nEC/EndoC (empirical P=0.0019) and EndoV (P=0.014),\nbut not in sorted EC bulk RNA-seq (P=0.136).\nPublished RNF20 loss increases TGF-beta signaling/EndMT;\ntherefore the transcript module cannot be read as activity.",
            "#FDEBEC", RED, fontsize=8.5)
    add_arrow(ax, (0.44, 0.21), (0.46, 0.29), color=RED)

    add_box(ax, (0.54, 0.03), 0.43, 0.18, "Highest-priority prospective/controlled validation",
            "independent CHD endothelial cells; age/chamber matching\nmultiple-heart CHD spatial sampling\nHLHS stage and VAD-unloading evaluation [39]\nsigned TGF-beta/SMAD and EndMT perturbation assays",
            "#FFFFFF", "#222222", dashed=True, fontsize=8.9)
    add_arrow(ax, (0.835, 0.59), (0.835, 0.21), color=RED)
    save(fig, OUT / "Figure_5")


def copy_unchanged() -> None:
    import shutil

    sources = DATA / "figures" / "source_analysis"
    for name in (
        "S_fetal_leave_arterial_out_population_mapping.png",
        "S_spatial_within_BEC_outer_inner.png",
    ):
        source = sources / name
        if source.exists():
            shutil.copy2(source, PANELS / name)


def main() -> None:
    copy_unchanged()
    figure2()
    figure3()
    figure4()
    figure5()
    print(OUT)


if __name__ == "__main__":
    main()
