from __future__ import annotations

import csv
import gzip
import math
import re
import sqlite3
import sys
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

# Keep bundled site-packages first (it contains a complete Pillow install), while
# appending the local package cache for scipy/openpyxl when needed.
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from scipy import stats


ROOT = Path(".").resolve()
EXT = ROOT / "external_validation"
GSE23959_DIR = EXT / "GSE23959"
OUT = Path("outputs") / "external_validation"
TABLES = OUT / "tables"
FIGS = OUT / "figures"
GENE_INFO = Path("resources") / "Homo_sapiens.gene_info.gz"
GSE23959_ANNOTATION_SQLITE = Path("resources") / "GSE23959_annotation" / "huex10sttranscriptcluster.sqlite"
GSE23959_GPL_SOFT = GSE23959_DIR / "GPL5188_family.soft.gz"


MODULES: dict[str, list[str]] = {
    "arterial_maturation": ["NOTCH1", "DLL4", "JAG1", "HEY1", "HEY2", "EFNB2", "GJA5", "SOX17"],
    "endothelial_matrix_core": ["FN1", "VCAN", "COL4A1", "COL4A2", "LAMA4", "ITGA5", "ITGB1"],
    "tgfb_plasticity_core": ["TGFB1", "TGFB2", "TGFBR1", "TGFBR2", "SMAD2", "SMAD3", "SNAI1", "SNAI2"],
}

SUPPORTING_GENES: dict[str, list[str]] = {
    "endothelial_identity": ["PECAM1", "CDH5", "VWF", "KDR", "FLT1", "TEK", "ENG", "EMCN", "ROBO4", "NRP1", "APLNR"],
    "flow_mechanosensing": ["KLF2", "KLF4", "NOS3", "THBD", "CAV1", "ICAM1", "VCAM1", "EDN1"],
    "hypoxia_inflammation": ["HIF1A", "EPAS1", "VEGFA", "CA9", "LDHA", "PGK1", "HMOX1", "IL6", "CXCL8", "CCL2", "TNF", "NFKB1"],
    "cardiac_stress": ["NPPA", "NPPB", "MYH6", "MYH7", "ACTC1", "TNNT2"],
}

MODULE_GENE_SUPERSET = [
    "NOTCH1", "DLL4", "JAG1", "HEY1", "HEY2", "EFNB2", "GJA5", "SOX17",
    "FN1", "POSTN", "COL1A1", "COL1A2", "COL3A1", "COL4A1", "COL4A2", "VCAN", "ITGA5", "ITGB1", "LAMA4", "DCN",
    "TGFB1", "TGFB2", "TGFBR1", "TGFBR2", "SMAD2", "SMAD3", "SNAI1", "SNAI2", "TAGLN", "ACTA2", "VIM",
]
TARGET_GENES = sorted(set(MODULE_GENE_SUPERSET + [g for genes in SUPPORTING_GENES.values() for g in genes]))


@dataclass
class Dataset:
    name: str
    expression: pd.DataFrame  # genes x samples, log-scale expression
    metadata: pd.DataFrame  # one row per sample
    notes: str


@dataclass
class Contrast:
    name: str
    dataset: str
    case_label: str
    control_label: str
    case_samples: list[str]
    control_samples: list[str]
    paired: bool = False
    pair_column: str | None = None


def ensure_dirs() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)


def strip_quotes(x: str) -> str:
    return x.strip().strip('"')


def parse_geo_sample_metadata(path: Path) -> pd.DataFrame:
    values_by_key: dict[str, list[list[str]]] = {}
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line == "!series_matrix_table_begin\n":
                break
            if not line.startswith("!Sample"):
                continue
            row = next(csv.reader([line.rstrip("\n")], delimiter="\t"))
            key = row[0].lstrip("!")
            values = [strip_quotes(v) for v in row[1:]]
            values_by_key.setdefault(key, []).append(values)

    accessions = values_by_key.get("Sample_geo_accession", [[]])[0]
    n = len(accessions)
    records: list[dict[str, str]] = []
    for i in range(n):
        rec: dict[str, str] = {"geo_accession": accessions[i] if i < len(accessions) else f"sample_{i+1}"}
        for key, value_lists in values_by_key.items():
            for repeat_idx, vals in enumerate(value_lists, start=1):
                if i >= len(vals):
                    continue
                value = vals[i]
                out_key = key if len(value_lists) == 1 else f"{key}_{repeat_idx}"
                rec[out_key] = value
                if key == "Sample_characteristics_ch1":
                    if ":" in value:
                        k, v = value.split(":", 1)
                        rec[f"char_{k.strip().lower().replace(' ', '_')}"] = v.strip()
        records.append(rec)
    return pd.DataFrame.from_records(records)


def parse_gene_info_mapping(path: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
        header = next(handle).rstrip("\n").split("\t")
        idx = {name: i for i, name in enumerate(header)}
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < len(header):
                continue
            symbol = parts[idx.get("Symbol_from_nomenclature_authority", idx["Symbol"])]
            if not symbol or symbol == "-":
                symbol = parts[idx["Symbol"]]
            dbxrefs = parts[idx["dbXrefs"]]
            for ref in dbxrefs.split("|"):
                if ref.startswith("Ensembl:"):
                    ens = ref.split(":", 1)[1].split(".")[0]
                    mapping[ens] = symbol
    return mapping


def parse_gene_info_entrez_symbols(path: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
        header = next(handle).rstrip("\n").split("\t")
        idx = {name: i for i, name in enumerate(header)}
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < len(header):
                continue
            gene_id = parts[idx["GeneID"]]
            symbol = parts[idx.get("Symbol_from_nomenclature_authority", idx["Symbol"])]
            if not symbol or symbol == "-":
                symbol = parts[idx["Symbol"]]
            if gene_id and symbol and symbol != "-":
                mapping[gene_id] = symbol
    return mapping


def collapse_ensembl_to_symbol(expr: pd.DataFrame, mapping: dict[str, str]) -> pd.DataFrame:
    gene_ids = [str(x).split(".")[0] for x in expr.index]
    symbols = [mapping.get(gid) for gid in gene_ids]
    out = expr.copy()
    out.insert(0, "gene_symbol", symbols)
    out = out[out["gene_symbol"].notna()]
    out = out[out["gene_symbol"].astype(str).str.len() > 0]
    grouped = out.groupby("gene_symbol", sort=True).mean(numeric_only=True)
    grouped.index.name = "gene_symbol"
    return grouped


def log2_if_needed(expr: pd.DataFrame, pseudocount: float = 1.0) -> pd.DataFrame:
    vals = expr.to_numpy(dtype=float)
    finite = vals[np.isfinite(vals)]
    if finite.size == 0:
        return expr.astype(float)
    q99 = np.nanpercentile(finite, 99)
    mx = np.nanmax(finite)
    if q99 > 30 or mx > 100:
        return np.log2(expr.astype(float) + pseudocount)
    return expr.astype(float)


def counts_to_logcpm(counts: pd.DataFrame) -> pd.DataFrame:
    counts = counts.astype(float)
    lib_sizes = counts.sum(axis=0)
    cpm = counts.divide(lib_sizes, axis=1) * 1_000_000.0
    return np.log2(cpm + 1.0)


def load_gse36761(mapping: dict[str, str]) -> Dataset:
    expr_path = EXT / "GSE36761_gene_expression_levels_normalized.txt.gz"
    meta_path = EXT / "GSE36761_series_matrix.txt.gz"
    raw = pd.read_csv(expr_path, sep="\t", compression="gzip")
    raw = raw.set_index(raw.columns[0])
    expr = log2_if_needed(raw)
    expr = collapse_ensembl_to_symbol(expr, mapping)

    geo_meta = parse_geo_sample_metadata(meta_path)
    title_map: dict[str, dict[str, str]] = {}
    for _, row in geo_meta.iterrows():
        title = row.get("Sample_title", "")
        m = re.match(r"^(NH-\d+|TOF-\d+)_(LV|RV)_mRNA$", title)
        if not m:
            continue
        title_map[m.group(1)] = {
            "geo_accession": row["geo_accession"],
            "title": title,
            "tissue": m.group(2),
            "disease": "healthy" if m.group(1).startswith("NH") else "TOF",
        }

    records = []
    for sample in expr.columns:
        rec = title_map.get(sample, {})
        records.append({
            "dataset": "GSE36761",
            "sample": sample,
            "geo_accession": rec.get("geo_accession", ""),
            "title": rec.get("title", sample),
            "disease": rec.get("disease", "healthy" if sample.startswith("NH") else "TOF"),
            "tissue": rec.get("tissue", "RV" if sample.startswith("TOF") else ""),
            "group": ("TOF_RV" if sample.startswith("TOF") else f"healthy_{rec.get('tissue', 'unknown')}"),
            "individual": sample,
            "time": "",
        })
    meta = pd.DataFrame.from_records(records)
    return Dataset("GSE36761", expr, meta, "Normalized mRNA-seq expression; log2(value+1) applied because values were on a positive abundance scale.")


def load_gse217772(mapping: dict[str, str]) -> Dataset:
    expr_path = EXT / "GSE217772_mRNA_count.txt.gz"
    raw = pd.read_csv(expr_path, sep="\t", compression="gzip")
    raw = raw.set_index(raw.columns[0])
    expr = counts_to_logcpm(raw)
    expr = collapse_ensembl_to_symbol(expr, mapping)

    records = []
    for col in expr.columns:
        m = re.match(r"^(control|treat)(\d+)$", col)
        condition = m.group(1) if m else ("control" if col.startswith("control") else "treat")
        idx = int(m.group(2)) if m else len(records) + 1
        is_tof = condition == "treat"
        gsm_num = 6726580 + (idx + 5 if is_tof else idx)
        records.append({
            "dataset": "GSE217772",
            "sample": col,
            "geo_accession": f"GSM{gsm_num}",
            "title": ("patient with Tetralogy of Fallot" if is_tof else "healthy unaffected individual") + f", {idx}",
            "disease": "TOF" if is_tof else "healthy",
            "tissue": "RV",
            "group": "TOF_RV" if is_tof else "healthy_RV",
            "individual": col,
            "time": "",
            "library_size": float(raw[col].sum()),
        })
    meta = pd.DataFrame.from_records(records)
    return Dataset("GSE217772", expr, meta, "Raw mRNA counts; converted to log2(CPM+1). lncRNA file was not used for AMA mRNA module scoring.")


def count_series_table_rows(path: Path) -> tuple[int, int]:
    begin_line = -1
    rows = 0
    in_table = False
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
        for i, line in enumerate(handle):
            if line.rstrip("\n") == "!series_matrix_table_begin":
                begin_line = i + 1
                in_table = True
                continue
            if line.rstrip("\n") == "!series_matrix_table_end":
                break
            if in_table:
                rows += 1
    return begin_line, rows


def parse_gpl_probe_symbols(path: Path) -> dict[str, list[str]]:
    probe_to_symbols: dict[str, list[str]] = {}
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
        in_table = False
        header: list[str] | None = None
        for line in handle:
            line = line.rstrip("\n")
            if line == "!platform_table_begin":
                in_table = True
                continue
            if line == "!platform_table_end":
                break
            if not in_table:
                continue
            row = next(csv.reader([line], delimiter="\t"))
            if header is None:
                header = row
                continue
            if len(row) < len(header):
                continue
            col = {name: i for i, name in enumerate(header)}
            probe_id = row[col["ID"]]
            raw_symbol = row[col["Gene Symbol"]]
            symbols = []
            for part in re.split(r"\s*///\s*", raw_symbol):
                part = part.strip()
                if part and part != "---":
                    symbols.append(part)
            if symbols:
                probe_to_symbols[probe_id] = sorted(set(symbols))
    return probe_to_symbols


def load_gse132176() -> Dataset:
    meta_path = EXT / "GSE132176_series_matrix.txt.gz"
    gpl_path = EXT / "GPL13158_family.soft.gz"
    meta_geo = parse_geo_sample_metadata(meta_path)

    begin_line, rows = count_series_table_rows(meta_path)
    expr = pd.read_csv(
        meta_path,
        sep="\t",
        compression="gzip",
        skiprows=begin_line,
        nrows=rows - 1,
        quotechar='"',
    )
    expr = expr.set_index(expr.columns[0]).astype(float)

    probe_symbols = parse_gpl_probe_symbols(gpl_path)
    target_rows = []
    target_index = []
    for probe_id, row in expr.iterrows():
        symbols = probe_symbols.get(str(probe_id), [])
        matched = [s for s in symbols if s in TARGET_GENES]
        for sym in matched:
            target_rows.append(row)
            target_index.append(sym)
    if not target_rows:
        raise RuntimeError("No target genes mapped from GPL13158.")
    target_expr = pd.DataFrame(target_rows, index=target_index)
    target_expr.index.name = "gene_symbol"
    gene_expr = target_expr.groupby(level=0, sort=True).mean()

    records = []
    for _, row in meta_geo.iterrows():
        sample = row["geo_accession"]
        disease = row.get("char_disease_state", "")
        time = row.get("char_time", "").replace("before CPB", "PRE_CPB").replace("after CPB", "POST_CPB")
        individual = row.get("char_individual", "").replace("patient ", "P")
        desc = row.get("Sample_description", "")
        group = f"{disease}_{time}" if disease and time else desc
        records.append({
            "dataset": "GSE132176",
            "sample": sample,
            "geo_accession": sample,
            "title": row.get("Sample_title", ""),
            "description": desc,
            "disease": disease,
            "tissue": "right atrium biopsy",
            "group": group,
            "individual": individual,
            "time": time,
        })
    meta = pd.DataFrame.from_records(records)
    return Dataset("GSE132176", gene_expr, meta, "RMA-normalized Affymetrix expression from series matrix; targeted probe-to-gene aggregation using GPL13158.")


def ensure_gse23959_sqlite() -> Path:
    if GSE23959_ANNOTATION_SQLITE.exists():
        return GSE23959_ANNOTATION_SQLITE
    GSE23959_ANNOTATION_SQLITE.parent.mkdir(parents=True, exist_ok=True)
    tar_path = GSE23959_DIR / "huex10sttranscriptcluster.db_8.8.0.tar.gz"
    member_name = "huex10sttranscriptcluster.db/inst/extdata/huex10sttranscriptcluster.sqlite"
    with tarfile.open(tar_path, "r:gz") as tar:
        member = tar.getmember(member_name)
        src = tar.extractfile(member)
        if src is None:
            raise RuntimeError(f"Could not read {member_name} from {tar_path}")
        GSE23959_ANNOTATION_SQLITE.write_bytes(src.read())
    return GSE23959_ANNOTATION_SQLITE


def symbols_from_gpl5188_gene_assignment(value: str) -> set[str]:
    symbols: set[str] = set()
    for part in re.split(r"\s*///\s*", value or ""):
        bits = [b.strip() for b in part.split("//")]
        if len(bits) >= 2:
            sym = bits[1].strip()
            if sym and sym != "---":
                symbols.add(sym)
    return symbols


def parse_gpl5188_probe_symbols(expr_ids: set[str]) -> dict[str, list[str]]:
    if not GSE23959_GPL_SOFT.exists():
        raise FileNotFoundError(
            f"Missing GPL5188 platform file needed for GSE23959 ID_REF annotation: {GSE23959_GPL_SOFT}"
        )
    probe_to_symbols: dict[str, set[str]] = {}
    target_set = set(TARGET_GENES)
    try:
        with gzip.open(GSE23959_GPL_SOFT, "rt", encoding="utf-8", errors="replace") as handle:
            in_table = False
            header: list[str] | None = None
            idx: dict[str, int] = {}
            for line in handle:
                line = line.rstrip("\n")
                if line == "!platform_table_begin":
                    in_table = True
                    continue
                if line == "!platform_table_end":
                    break
                if not in_table:
                    continue
                row = next(csv.reader([line], delimiter="\t"))
                if header is None:
                    header = row
                    idx = {name: i for i, name in enumerate(header)}
                    continue
                if len(row) <= max(idx["ID"], idx["gene_assignment"]):
                    continue
                probe_id = str(row[idx["ID"]])
                if probe_id not in expr_ids:
                    continue
                matched = symbols_from_gpl5188_gene_assignment(row[idx["gene_assignment"]]) & target_set
                if matched:
                    probe_to_symbols.setdefault(probe_id, set()).update(matched)
    except EOFError:
        # A partially downloaded SOFT is still usable if the platform table rows
        # containing target probes were already read.
        pass
    except gzip.BadGzipFile:
        pass
    return {probe: sorted(symbols) for probe, symbols in probe_to_symbols.items()}


def parse_gse23959_probe_symbols(expr_ids: set[str], entrez_to_symbol: dict[str, str]) -> dict[str, list[str]]:
    # The GSE23959 series matrix uses GPL5188 probeset-level ID_REF values.
    # The Bioconductor huex10sttranscriptcluster.db maps transcript-cluster IDs,
    # which do not match this matrix, so GPL5188's gene_assignment is preferred.
    probe_to_symbols = parse_gpl5188_probe_symbols(expr_ids)
    if probe_to_symbols:
        return probe_to_symbols

    # Fallback: retain the sqlite route for environments where matrix IDs happen
    # to match transcript clusters, but this is not expected for the present file.
    sqlite_path = ensure_gse23959_sqlite()
    con = sqlite3.connect(sqlite_path)
    try:
        rows = con.execute("select probe_id, gene_id from probes").fetchall()
    finally:
        con.close()
    target_set = set(TARGET_GENES)
    fallback: dict[str, set[str]] = {}
    for probe_id, gene_id in rows:
        probe_id = str(probe_id)
        if probe_id not in expr_ids:
            continue
        symbol = entrez_to_symbol.get(str(gene_id))
        if symbol in target_set:
            fallback.setdefault(probe_id, set()).add(symbol)
    return {probe: sorted(symbols) for probe, symbols in fallback.items()}


def load_gse23959(entrez_to_symbol: dict[str, str]) -> Dataset:
    meta_path = GSE23959_DIR / "GSE23959_series_matrix.txt.gz"
    meta_geo = parse_geo_sample_metadata(meta_path)
    begin_line, rows = count_series_table_rows(meta_path)
    expr = pd.read_csv(
        meta_path,
        sep="\t",
        compression="gzip",
        skiprows=begin_line,
        nrows=rows - 1,
        quotechar='"',
    )
    expr = expr.set_index(expr.columns[0]).astype(float)

    expr_ids = {str(x) for x in expr.index}
    probe_symbols = parse_gse23959_probe_symbols(expr_ids, entrez_to_symbol)
    target_rows = []
    target_index = []
    for probe_id, row in expr.iterrows():
        symbols = probe_symbols.get(str(probe_id), [])
        for sym in symbols:
            target_rows.append(row)
            target_index.append(sym)
    if not target_rows:
        raise RuntimeError("No target genes mapped from huex10sttranscriptcluster annotation.")
    target_expr = pd.DataFrame(target_rows, index=target_index)
    target_expr.index.name = "gene_symbol"
    gene_expr = target_expr.groupby(level=0, sort=True).mean()

    records = []
    for _, row in meta_geo.iterrows():
        sample = row["geo_accession"]
        title = row.get("Sample_title", "")
        source = row.get("Sample_source_name_ch1", "")
        disease = row.get("char_disease_status", "")
        subtype = row.get("char_tissue_subtype", "")
        if "HLHS" in disease or "HLHS" in subtype or "HLHS" in title:
            disease_clean = "HLHS"
            tissue = "RV/HLHS ventricle"
            group = "HLHS_RV"
        elif "Right ventricle" in subtype or "right ventricle" in source:
            disease_clean = "healthy"
            tissue = "RV"
            group = "healthy_RV"
        elif "Left ventricle" in subtype or "left ventricle" in source:
            disease_clean = "healthy"
            tissue = "LV"
            group = "healthy_LV"
        else:
            disease_clean = disease or "unknown"
            tissue = subtype or source
            group = f"{disease_clean}_{tissue}".replace(" ", "_")
        records.append({
            "dataset": "GSE23959",
            "sample": sample,
            "geo_accession": sample,
            "title": title,
            "description": row.get("Sample_description", ""),
            "disease": disease_clean,
            "tissue": tissue,
            "group": group,
            "individual": title,
            "time": "",
        })
    meta = pd.DataFrame.from_records(records)
    return Dataset(
        "GSE23959",
        gene_expr,
        meta,
        "GCRMA/quantile-normalized Affymetrix Human Exon 1.0 ST expression from series matrix; targeted probe-to-gene aggregation using huex10sttranscriptcluster.db.",
    )


def module_scores(dataset: Dataset) -> tuple[pd.DataFrame, pd.DataFrame]:
    records = []
    coverage = []
    expr = dataset.expression
    for module, genes in MODULES.items():
        present = [g for g in genes if g in expr.index]
        coverage.append({
            "dataset": dataset.name,
            "module": module,
            "present_genes": ";".join(present),
            "missing_genes": ";".join([g for g in genes if g not in expr.index]),
            "n_present": len(present),
            "n_total": len(genes),
            "coverage_fraction": len(present) / len(genes),
        })
        if not present:
            continue
        scores = expr.loc[present].mean(axis=0)
        for sample, score in scores.items():
            records.append({
                "dataset": dataset.name,
                "sample": sample,
                "module": module,
                "raw_module_score": float(score),
                "n_genes_present": len(present),
                "n_genes_total": len(genes),
            })
    return pd.DataFrame.from_records(records), pd.DataFrame.from_records(coverage)


def safe_sd(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=float)
    sd = float(np.nanstd(arr, ddof=1)) if arr.size > 1 else float("nan")
    if not np.isfinite(sd) or sd == 0:
        return float("nan")
    return sd


def p_adjust_bh(pvals: list[float]) -> list[float]:
    p = np.asarray([1.0 if not np.isfinite(x) else x for x in pvals], dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    adj = np.empty(n, dtype=float)
    prev = 1.0
    for i in range(n - 1, -1, -1):
        val = ranked[i] * n / (i + 1)
        prev = min(prev, val)
        adj[order[i]] = prev
    return adj.tolist()


def independent_stats(case: np.ndarray, ctrl: np.ndarray) -> dict[str, float]:
    case = np.asarray(case, dtype=float)
    ctrl = np.asarray(ctrl, dtype=float)
    case = case[np.isfinite(case)]
    ctrl = ctrl[np.isfinite(ctrl)]
    n1, n0 = len(case), len(ctrl)
    mean1 = float(np.mean(case)) if n1 else float("nan")
    mean0 = float(np.mean(ctrl)) if n0 else float("nan")
    delta = mean1 - mean0
    sd1 = float(np.std(case, ddof=1)) if n1 > 1 else float("nan")
    sd0 = float(np.std(ctrl, ddof=1)) if n0 > 1 else float("nan")
    if n1 > 1 and n0 > 1:
        res = stats.ttest_ind(case, ctrl, equal_var=False, nan_policy="omit")
        p = float(res.pvalue)
        se = math.sqrt(sd1**2 / n1 + sd0**2 / n0)
        df_num = (sd1**2 / n1 + sd0**2 / n0) ** 2
        df_den = ((sd1**2 / n1) ** 2 / (n1 - 1)) + ((sd0**2 / n0) ** 2 / (n0 - 1))
        df = df_num / df_den if df_den else float("nan")
        tcrit = float(stats.t.ppf(0.975, df)) if np.isfinite(df) else 1.96
        ci_low = delta - tcrit * se
        ci_high = delta + tcrit * se
        pooled = math.sqrt(((n1 - 1) * sd1**2 + (n0 - 1) * sd0**2) / (n1 + n0 - 2)) if n1 + n0 > 2 else float("nan")
        cohen_d = delta / pooled if pooled and np.isfinite(pooled) else float("nan")
    else:
        p = ci_low = ci_high = cohen_d = float("nan")
    return {
        "n_case": n1,
        "n_control": n0,
        "mean_case": mean1,
        "mean_control": mean0,
        "delta_case_minus_control": delta,
        "ci95_low": float(ci_low),
        "ci95_high": float(ci_high),
        "p_value": p,
        "effect_size_cohen_d": float(cohen_d),
    }


def paired_stats(case: pd.Series, ctrl: pd.Series) -> dict[str, float]:
    aligned = pd.concat([case.rename("case"), ctrl.rename("control")], axis=1).dropna()
    diff = aligned["case"].to_numpy(dtype=float) - aligned["control"].to_numpy(dtype=float)
    n = len(diff)
    mean_case = float(aligned["case"].mean()) if n else float("nan")
    mean_ctrl = float(aligned["control"].mean()) if n else float("nan")
    delta = float(np.mean(diff)) if n else float("nan")
    if n > 1:
        res = stats.ttest_rel(aligned["case"], aligned["control"], nan_policy="omit")
        p = float(res.pvalue)
        sd = float(np.std(diff, ddof=1))
        se = sd / math.sqrt(n)
        tcrit = float(stats.t.ppf(0.975, n - 1))
        ci_low = delta - tcrit * se
        ci_high = delta + tcrit * se
        dz = delta / sd if sd else float("nan")
    else:
        p = ci_low = ci_high = dz = float("nan")
    return {
        "n_case": n,
        "n_control": n,
        "mean_case": mean_case,
        "mean_control": mean_ctrl,
        "delta_case_minus_control": delta,
        "ci95_low": float(ci_low),
        "ci95_high": float(ci_high),
        "p_value": p,
        "effect_size_cohen_d": float(dz),
    }


def build_contrasts(datasets: dict[str, Dataset]) -> list[Contrast]:
    contrasts = []
    g367 = datasets["GSE36761"].metadata
    contrasts.append(Contrast(
        name="GSE36761_TOF_RV_vs_healthy_RV",
        dataset="GSE36761",
        case_label="TOF right ventricle",
        control_label="healthy right ventricle",
        case_samples=g367.loc[g367["group"] == "TOF_RV", "sample"].tolist(),
        control_samples=g367.loc[g367["group"] == "healthy_RV", "sample"].tolist(),
    ))
    contrasts.append(Contrast(
        name="GSE36761_TOF_RV_vs_all_healthy_ventricle_sensitivity",
        dataset="GSE36761",
        case_label="TOF right ventricle",
        control_label="all healthy ventricles",
        case_samples=g367.loc[g367["group"] == "TOF_RV", "sample"].tolist(),
        control_samples=g367.loc[g367["disease"] == "healthy", "sample"].tolist(),
    ))
    g217 = datasets["GSE217772"].metadata
    contrasts.append(Contrast(
        name="GSE217772_TOF_RV_vs_healthy_RV",
        dataset="GSE217772",
        case_label="TOF right ventricle",
        control_label="healthy right ventricle",
        case_samples=g217.loc[g217["group"] == "TOF_RV", "sample"].tolist(),
        control_samples=g217.loc[g217["group"] == "healthy_RV", "sample"].tolist(),
    ))
    g132 = datasets["GSE132176"].metadata
    contrasts.append(Contrast(
        name="GSE132176_TOF_PRE_CPB_vs_ASD_PRE_CPB",
        dataset="GSE132176",
        case_label="TOF pre-CPB right atrium",
        control_label="ASD pre-CPB right atrium",
        case_samples=g132.loc[(g132["disease"] == "TOF") & (g132["time"] == "PRE_CPB"), "sample"].tolist(),
        control_samples=g132.loc[(g132["disease"] == "ASD") & (g132["time"] == "PRE_CPB"), "sample"].tolist(),
    ))
    contrasts.append(Contrast(
        name="GSE132176_TOF_POST_vs_PRE_CPB_paired",
        dataset="GSE132176",
        case_label="TOF post-CPB",
        control_label="TOF pre-CPB",
        case_samples=g132.loc[(g132["disease"] == "TOF") & (g132["time"] == "POST_CPB"), "sample"].tolist(),
        control_samples=g132.loc[(g132["disease"] == "TOF") & (g132["time"] == "PRE_CPB"), "sample"].tolist(),
        paired=True,
        pair_column="individual",
    ))
    contrasts.append(Contrast(
        name="GSE132176_ASD_POST_vs_PRE_CPB_paired",
        dataset="GSE132176",
        case_label="ASD post-CPB",
        control_label="ASD pre-CPB",
        case_samples=g132.loc[(g132["disease"] == "ASD") & (g132["time"] == "POST_CPB"), "sample"].tolist(),
        control_samples=g132.loc[(g132["disease"] == "ASD") & (g132["time"] == "PRE_CPB"), "sample"].tolist(),
        paired=True,
        pair_column="individual",
    ))
    if "GSE23959" in datasets:
        g239 = datasets["GSE23959"].metadata
        contrasts.append(Contrast(
            name="GSE23959_HLHS_ventricle_vs_healthy_RV",
            dataset="GSE23959",
            case_label="HLHS ventricle",
            control_label="healthy right ventricle",
            case_samples=g239.loc[g239["group"] == "HLHS_RV", "sample"].tolist(),
            control_samples=g239.loc[g239["group"] == "healthy_RV", "sample"].tolist(),
        ))
        contrasts.append(Contrast(
            name="GSE23959_HLHS_ventricle_vs_all_healthy_ventricles_sensitivity",
            dataset="GSE23959",
            case_label="HLHS ventricle",
            control_label="all healthy ventricles",
            case_samples=g239.loc[g239["group"] == "HLHS_RV", "sample"].tolist(),
            control_samples=g239.loc[g239["disease"] == "healthy", "sample"].tolist(),
        ))
    return contrasts


def compute_contrast_scores(all_module_scores: pd.DataFrame, datasets: dict[str, Dataset], contrasts: list[Contrast]) -> tuple[pd.DataFrame, pd.DataFrame]:
    sample_score_records = []
    effect_records = []
    for contrast in contrasts:
        ds_scores = all_module_scores[all_module_scores["dataset"] == contrast.dataset]
        # module z scores relative to controls within the current contrast.
        z_by_sample: dict[str, list[float]] = {s: [] for s in contrast.case_samples + contrast.control_samples}
        for module in MODULES:
            mdat = ds_scores[ds_scores["module"] == module].set_index("sample")
            ctrl_vals = mdat.loc[[s for s in contrast.control_samples if s in mdat.index], "raw_module_score"].astype(float)
            ctrl_mean = float(ctrl_vals.mean())
            ctrl_sd = safe_sd(ctrl_vals)
            for sample in contrast.case_samples + contrast.control_samples:
                if sample not in mdat.index:
                    continue
                raw = float(mdat.at[sample, "raw_module_score"])
                z = (raw - ctrl_mean) / ctrl_sd if np.isfinite(ctrl_sd) and ctrl_sd != 0 else float("nan")
                z_by_sample.setdefault(sample, []).append(z)
                sample_score_records.append({
                    "contrast": contrast.name,
                    "dataset": contrast.dataset,
                    "sample": sample,
                    "role": "case" if sample in contrast.case_samples else "control",
                    "module": module,
                    "raw_module_score": raw,
                    "z_vs_control": z,
                })
        for sample, zs in z_by_sample.items():
            valid = [z for z in zs if np.isfinite(z)]
            ama = -float(np.mean(valid)) if valid else float("nan")
            sample_score_records.append({
                "contrast": contrast.name,
                "dataset": contrast.dataset,
                "sample": sample,
                "role": "case" if sample in contrast.case_samples else "control",
                "module": "AMA",
                "raw_module_score": ama,
                "z_vs_control": ama,
            })

        score_df = pd.DataFrame.from_records(sample_score_records)
        curr = score_df[score_df["contrast"] == contrast.name]
        for module in list(MODULES.keys()) + ["AMA"]:
            mdat = curr[curr["module"] == module].set_index("sample")
            if contrast.paired:
                meta = datasets[contrast.dataset].metadata.set_index("sample")
                case_pairs = meta.loc[[s for s in contrast.case_samples if s in meta.index], [contrast.pair_column]]
                ctrl_pairs = meta.loc[[s for s in contrast.control_samples if s in meta.index], [contrast.pair_column]]
                case_series = {}
                ctrl_series = {}
                for sample, prow in case_pairs.iterrows():
                    if sample in mdat.index:
                        case_series[str(prow[contrast.pair_column])] = float(mdat.at[sample, "raw_module_score"])
                for sample, prow in ctrl_pairs.iterrows():
                    if sample in mdat.index:
                        ctrl_series[str(prow[contrast.pair_column])] = float(mdat.at[sample, "raw_module_score"])
                stat = paired_stats(pd.Series(case_series), pd.Series(ctrl_series))
                test = "paired_t"
            else:
                case_vals = mdat.loc[[s for s in contrast.case_samples if s in mdat.index], "raw_module_score"].to_numpy(dtype=float)
                ctrl_vals = mdat.loc[[s for s in contrast.control_samples if s in mdat.index], "raw_module_score"].to_numpy(dtype=float)
                stat = independent_stats(case_vals, ctrl_vals)
                test = "welch_t"
            direction = "higher_in_case" if stat["delta_case_minus_control"] > 0 else "lower_in_case"
            supports = (module == "AMA" and stat["delta_case_minus_control"] > 0) or (module != "AMA" and stat["delta_case_minus_control"] < 0)
            effect_records.append({
                "contrast": contrast.name,
                "dataset": contrast.dataset,
                "case_label": contrast.case_label,
                "control_label": contrast.control_label,
                "module": module,
                "test": test,
                **stat,
                "direction": direction,
                "supports_ama_attenuation_direction": bool(supports),
            })
    effects = pd.DataFrame.from_records(effect_records)
    effects["fdr_within_all_module_tests"] = p_adjust_bh(effects["p_value"].tolist())
    return pd.DataFrame.from_records(sample_score_records), effects


def candidate_gene_effects(datasets: dict[str, Dataset], contrasts: list[Contrast]) -> pd.DataFrame:
    gene_to_module = {}
    for module, genes in MODULES.items():
        for g in genes:
            gene_to_module.setdefault(g, []).append(module)
    for module, genes in SUPPORTING_GENES.items():
        for g in genes:
            gene_to_module.setdefault(g, []).append(module)

    records = []
    for contrast in contrasts:
        ds = datasets[contrast.dataset]
        expr = ds.expression
        meta = ds.metadata.set_index("sample")
        for gene in TARGET_GENES:
            if gene not in expr.index:
                records.append({
                    "contrast": contrast.name,
                    "dataset": contrast.dataset,
                    "gene": gene,
                    "gene_set": ";".join(gene_to_module.get(gene, [])),
                    "present": False,
                    "n_case": len(contrast.case_samples),
                    "n_control": len(contrast.control_samples),
                    "p_value": np.nan,
                    "delta_case_minus_control": np.nan,
                })
                continue
            vals = expr.loc[gene]
            if contrast.paired:
                case_series = {}
                ctrl_series = {}
                for sample in contrast.case_samples:
                    if sample in vals.index and sample in meta.index:
                        case_series[str(meta.at[sample, contrast.pair_column])] = float(vals[sample])
                for sample in contrast.control_samples:
                    if sample in vals.index and sample in meta.index:
                        ctrl_series[str(meta.at[sample, contrast.pair_column])] = float(vals[sample])
                stat = paired_stats(pd.Series(case_series), pd.Series(ctrl_series))
                test = "paired_t"
            else:
                case_vals = vals[[s for s in contrast.case_samples if s in vals.index]].to_numpy(dtype=float)
                ctrl_vals = vals[[s for s in contrast.control_samples if s in vals.index]].to_numpy(dtype=float)
                stat = independent_stats(case_vals, ctrl_vals)
                test = "welch_t"
            records.append({
                "contrast": contrast.name,
                "dataset": contrast.dataset,
                "gene": gene,
                "gene_set": ";".join(gene_to_module.get(gene, [])),
                "present": True,
                "test": test,
                **stat,
            })
    out = pd.DataFrame.from_records(records)
    out["fdr_within_contrast"] = np.nan
    for contrast_name, idx in out.groupby("contrast").groups.items():
        pvals = out.loc[idx, "p_value"].fillna(1.0).tolist()
        out.loc[idx, "fdr_within_contrast"] = p_adjust_bh(pvals)
    return out


def draw_text(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, font: ImageFont.ImageFont, fill=(32, 39, 50)) -> None:
    draw.text(xy, text, font=font, fill=fill)


def pretty_contrast_label(name: str) -> str:
    label = name.replace("_", " ")
    replacements = {
        "all healthy ventricle sensitivity": "all healthy ventricles sens.",
        "all healthy ventricles sensitivity": "all healthy ventricles sens.",
        "GSE132176 ": "GSE132176: ",
        "GSE36761 ": "GSE36761: ",
        "GSE217772 ": "GSE217772: ",
        "GSE23959 ": "GSE23959: ",
        "PRE CPB": "pre-CPB",
        "POST": "post",
    }
    for old, new in replacements.items():
        label = label.replace(old, new)
    return label


def font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        r"C:\Windows\Fonts\arialbd.ttf" if bold else r"C:\Windows\Fonts\arial.ttf",
        r"C:\Windows\Fonts\calibrib.ttf" if bold else r"C:\Windows\Fonts\calibri.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def make_forest_plot(effects: pd.DataFrame) -> None:
    data = effects[effects["module"] == "AMA"].copy()
    data = data.sort_values("delta_case_minus_control", ascending=True)
    w = 1500
    row_h = 92
    top = 140
    left = 560
    right = 1380
    h = top + row_h * len(data) + 90
    img = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(img)
    f_title = font(34, True)
    f_axis = font(22)
    f_small = font(20)
    draw_text(d, (40, 35), "AMA score changes across independent validation contrasts", f_title)
    vals = data[["ci95_low", "ci95_high", "delta_case_minus_control"]].to_numpy(dtype=float).flatten()
    vals = vals[np.isfinite(vals)]
    vmin = min(-1.0, float(np.nanmin(vals)) if vals.size else -1.0)
    vmax = max(1.0, float(np.nanmax(vals)) if vals.size else 1.0)
    pad = (vmax - vmin) * 0.12
    vmin -= pad
    vmax += pad
    def xmap(v: float) -> int:
        return int(left + (v - vmin) / (vmax - vmin) * (right - left))
    zero_x = xmap(0)
    d.line((zero_x, top - 30, zero_x, h - 70), fill=(120, 130, 145), width=2)
    d.line((left, h - 70, right, h - 70), fill=(70, 80, 95), width=2)
    for tick in np.linspace(math.floor(vmin), math.ceil(vmax), 7):
        x = xmap(float(tick))
        d.line((x, h - 75, x, h - 65), fill=(70, 80, 95), width=2)
        draw_text(d, (x - 24, h - 55), f"{tick:.1f}", f_small, fill=(70, 80, 95))
    draw_text(d, (left + 180, h - 28), "Delta AMA score (case − control); higher = stronger attenuation", f_axis, fill=(70, 80, 95))
    for i, (_, row) in enumerate(data.iterrows()):
        y = top + i * row_h
        label = pretty_contrast_label(row["contrast"])
        draw_text(d, (40, y - 10), label[:64], f_small)
        draw_text(d, (40, y + 22), f"n={int(row['n_case'])} vs {int(row['n_control'])}; P={row['p_value']:.3g}", f_small, fill=(90, 100, 115))
        x0 = xmap(row["ci95_low"]) if np.isfinite(row["ci95_low"]) else xmap(row["delta_case_minus_control"])
        x1 = xmap(row["ci95_high"]) if np.isfinite(row["ci95_high"]) else xmap(row["delta_case_minus_control"])
        xm = xmap(row["delta_case_minus_control"])
        color = (211, 111, 56) if row["delta_case_minus_control"] > 0 else (54, 117, 181)
        d.line((x0, y + 20, x1, y + 20), fill=color, width=5)
        d.ellipse((xm - 10, y + 10, xm + 10, y + 30), fill=color, outline=(35, 35, 35))
    img.save(FIGS / "validation_AMA_forest.png", dpi=(300, 300))


def blend(c1: tuple[int, int, int], c2: tuple[int, int, int], t: float) -> tuple[int, int, int]:
    return tuple(int(c1[i] + (c2[i] - c1[i]) * t) for i in range(3))


def heat_color(v: float, vmax: float) -> tuple[int, int, int]:
    if not np.isfinite(v) or vmax == 0:
        return (230, 234, 240)
    t = min(1.0, abs(v) / vmax)
    if v < 0:
        return blend((245, 248, 252), (49, 105, 172), t)
    return blend((252, 248, 244), (212, 101, 50), t)


def make_heatmap(effects: pd.DataFrame) -> None:
    mods = ["arterial_maturation", "ecm_integrin", "tgfb_endmt", "AMA"]
    contrasts = effects["contrast"].drop_duplicates().tolist()
    cell_w = 210
    cell_h = 72
    left = 610
    top = 150
    w = left + cell_w * len(mods) + 80
    h = top + cell_h * len(contrasts) + 130
    img = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(img)
    f_title = font(32, True)
    f = font(20)
    f_small = font(17)
    draw_text(d, (40, 35), "Module-score deltas across validation contrasts", f_title)
    vals = effects["delta_case_minus_control"].to_numpy(dtype=float)
    vmax = float(np.nanpercentile(np.abs(vals[np.isfinite(vals)]), 95)) if np.isfinite(vals).any() else 1.0
    vmax = max(vmax, 0.1)
    for j, mod in enumerate(mods):
        draw_text(d, (left + j * cell_w + 8, top - 45), mod.replace("_", " "), f_small)
    for i, con in enumerate(contrasts):
        y = top + i * cell_h
        label = pretty_contrast_label(con)
        draw_text(d, (40, y + 20), label[:62], f_small)
        for j, mod in enumerate(mods):
            sub = effects[(effects["contrast"] == con) & (effects["module"] == mod)]
            v = float(sub["delta_case_minus_control"].iloc[0]) if len(sub) else float("nan")
            x = left + j * cell_w
            d.rectangle((x, y, x + cell_w - 6, y + cell_h - 6), fill=heat_color(v, vmax), outline=(255, 255, 255))
            draw_text(d, (x + 18, y + 20), f"{v:+.2f}" if np.isfinite(v) else "NA", f)
    draw_text(d, (left, h - 60), "Blue = lower module score in case/post; Orange = higher. AMA is expected to increase.", f_small, fill=(80, 90, 105))
    img.save(FIGS / "validation_module_delta_heatmap.png", dpi=(300, 300))


def make_strip_plot(sample_scores: pd.DataFrame) -> None:
    data = sample_scores[sample_scores["module"] == "AMA"].copy()
    contrasts = data["contrast"].drop_duplicates().tolist()
    w = 1500
    panel_h = 115
    top = 110
    left = 650
    right = 1400
    h = top + panel_h * len(contrasts) + 70
    img = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(img)
    f_title = font(32, True)
    f = font(20)
    f_small = font(17)
    draw_text(d, (40, 32), "Sample-level AMA scores used for validation contrasts", f_title)
    vals = data["raw_module_score"].to_numpy(dtype=float)
    vals = vals[np.isfinite(vals)]
    vmin = min(-4.0, float(np.nanmin(vals)) if vals.size else -4.0)
    vmax = max(4.0, float(np.nanmax(vals)) if vals.size else 4.0)
    def xmap(v: float) -> int:
        return int(left + (v - vmin) / (vmax - vmin) * (right - left))
    for i, con in enumerate(contrasts):
        sub = data[data["contrast"] == con]
        y = top + i * panel_h
        draw_text(d, (40, y + 38), pretty_contrast_label(con)[:70], f_small)
        d.line((left, y + 55, right, y + 55), fill=(210, 216, 224), width=2)
        d.line((xmap(0), y + 28, xmap(0), y + 82), fill=(120, 130, 145), width=2)
        for role, color, yy in [("control", (74, 127, 185), y + 42), ("case", (214, 111, 61), y + 68)]:
            vals_role = sub[sub["role"] == role]["raw_module_score"].to_numpy(dtype=float)
            for k, val in enumerate(vals_role):
                jitter = ((k % 5) - 2) * 3
                x = xmap(float(val))
                d.ellipse((x - 6, yy + jitter - 6, x + 6, yy + jitter + 6), fill=color, outline=(35, 35, 35))
        draw_text(d, (right - 230, y + 22), "control", f_small, fill=(74, 127, 185))
        draw_text(d, (right - 230, y + 58), "case/post", f_small, fill=(214, 111, 61))
    draw_text(d, (left, h - 45), "AMA score (contrast-standardized; higher = stronger attenuation)", f, fill=(70, 80, 95))
    img.save(FIGS / "validation_sample_AMA_strip.png", dpi=(300, 300))


def write_excel(tables: dict[str, pd.DataFrame]) -> None:
    xlsx = TABLES / "AMA_CHD_external_validation_results.xlsx"
    with pd.ExcelWriter(xlsx, engine="openpyxl") as writer:
        for name, df in tables.items():
            sheet = name[:31]
            df.to_excel(writer, sheet_name=sheet, index=False)
            ws = writer.book[sheet]
            ws.freeze_panes = "A2"
            for col in ws.columns:
                max_len = min(60, max(len(str(cell.value)) if cell.value is not None else 0 for cell in col) + 2)
                ws.column_dimensions[col[0].column_letter].width = max_len


def write_summary(datasets: dict[str, Dataset], contrasts: list[Contrast], effects: pd.DataFrame, coverage: pd.DataFrame) -> None:
    lines = ["# External cross-cohort evaluation analysis summary", ""]
    lines.append("## Data sets parsed")
    for ds in datasets.values():
        lines.append(f"- {ds.name}: expression matrix {ds.expression.shape[0]} genes x {ds.expression.shape[1]} samples. {ds.notes}")
    lines.append("")
    lines.append("## Validation contrasts")
    for c in contrasts:
        lines.append(f"- {c.name}: {len(c.case_samples)} case/post vs {len(c.control_samples)} control/pre; paired={c.paired}.")
    lines.append("")
    lines.append("## AMA score results")
    ama = effects[effects["module"] == "AMA"].copy()
    for _, row in ama.iterrows():
        lines.append(
            f"- {row['contrast']}: delta AMA={row['delta_case_minus_control']:+.3f} "
            f"(95% CI {row['ci95_low']:+.3f} to {row['ci95_high']:+.3f}), "
            f"P={row['p_value']:.4g}, supports direction={row['supports_ama_attenuation_direction']}."
        )
    lines.append("")
    lines.append("## Coverage caveats")
    for _, row in coverage.iterrows():
        if row["coverage_fraction"] < 0.8:
            lines.append(f"- {row['dataset']} {row['module']}: {row['n_present']}/{row['n_total']} genes covered; missing {row['missing_genes']}.")
    lines.append("")
    lines.append("## Interpretation caution")
    lines.append("These public cohorts differ in tissue, platform and clinical comparator. They should be used as independent supportive validation, not as definitive causal proof.")
    (OUT / "analysis_summary.md").write_text("\n".join(lines), encoding="utf-8")


def gse23959_probe_mapping_table() -> pd.DataFrame:
    gse = GSE23959_DIR / "GSE23959_series_matrix.txt.gz"
    begin_line, rows = count_series_table_rows(gse)
    ids = pd.read_csv(gse, sep="\t", compression="gzip", skiprows=begin_line, nrows=rows - 1, quotechar='"', usecols=[0])
    expr_ids = {str(x) for x in ids.iloc[:, 0]}
    mapping = parse_gpl5188_probe_symbols(expr_ids)
    records = []
    for probe_id, symbols in sorted(mapping.items()):
        for symbol in symbols:
            records.append({"dataset": "GSE23959", "probe_id": probe_id, "gene_symbol": symbol})
    return pd.DataFrame.from_records(records)


def main() -> None:
    ensure_dirs()
    if not GENE_INFO.exists():
        raise FileNotFoundError(f"Missing gene-info mapping: {GENE_INFO}")

    mapping = parse_gene_info_mapping(GENE_INFO)
    entrez_mapping = parse_gene_info_entrez_symbols(GENE_INFO)
    datasets = {
        "GSE36761": load_gse36761(mapping),
        "GSE217772": load_gse217772(mapping),
        "GSE132176": load_gse132176(),
        "GSE23959": load_gse23959(entrez_mapping),
    }

    all_meta = pd.concat([ds.metadata for ds in datasets.values()], ignore_index=True, sort=False)
    module_score_frames = []
    coverage_frames = []
    for ds in datasets.values():
        ms, cov = module_scores(ds)
        module_score_frames.append(ms)
        coverage_frames.append(cov)
    all_module_scores = pd.concat(module_score_frames, ignore_index=True)
    coverage = pd.concat(coverage_frames, ignore_index=True)
    contrasts = build_contrasts(datasets)
    contrast_scores, effects = compute_contrast_scores(all_module_scores, datasets, contrasts)
    gene_effects = candidate_gene_effects(datasets, contrasts)
    gse23959_probe_map = gse23959_probe_mapping_table()

    contrast_inventory = pd.DataFrame.from_records([{
        "contrast": c.name,
        "dataset": c.dataset,
        "case_label": c.case_label,
        "control_label": c.control_label,
        "n_case": len(c.case_samples),
        "n_control": len(c.control_samples),
        "paired": c.paired,
        "case_samples": ";".join(c.case_samples),
        "control_samples": ";".join(c.control_samples),
    } for c in contrasts])

    outputs = {
        "sample_metadata_all.csv": all_meta,
        "contrast_inventory.csv": contrast_inventory,
        "gene_set_coverage.csv": coverage,
        "module_scores_by_sample.csv": all_module_scores,
        "contrast_standardized_scores_by_sample.csv": contrast_scores,
        "module_effects_by_contrast.csv": effects,
        "candidate_gene_effects_by_contrast.csv": gene_effects,
        "GSE23959_target_probe_mapping.csv": gse23959_probe_map,
    }
    for filename, df in outputs.items():
        df.to_csv(TABLES / filename, index=False, encoding="utf-8-sig")
    write_excel({
        "sample_metadata": all_meta,
        "contrasts": contrast_inventory,
        "coverage": coverage,
        "module_effects": effects,
        "sample_AMA_scores": contrast_scores[contrast_scores["module"] == "AMA"],
        "candidate_gene_effects": gene_effects,
        "GSE23959_probe_map": gse23959_probe_map,
    })

    make_forest_plot(effects)
    make_heatmap(effects)
    make_strip_plot(contrast_scores)
    write_summary(datasets, contrasts, effects, coverage)

    print(f"Analysis complete: {OUT.resolve()}")
    print(effects[effects["module"] == "AMA"][["contrast", "delta_case_minus_control", "ci95_low", "ci95_high", "p_value", "supports_ama_attenuation_direction"]].to_string(index=False))


if __name__ == "__main__":
    main()
