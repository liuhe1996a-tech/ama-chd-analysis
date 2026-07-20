"""Verify target-gene mapping from the complete GPL5188 platform file.

The complete GPL5188 SOFT file is very large and should not be committed to
GitHub. Provide its local path through AMA_GPL5188_SOFT.
"""

from __future__ import annotations

import csv
import gzip
import json
import os
from pathlib import Path

import external_validation_analysis as ev


FULL_GPL = Path(os.environ.get("AMA_GPL5188_SOFT", "external_validation/GSE23959/GPL5188_family.soft.gz"))
OLD_MAP = Path(os.environ.get("AMA_GSE23959_PROBE_MAP", "results/tables/GSE23959_target_probe_mapping.csv"))
OUT = Path(os.environ.get("AMA_OUTPUT_DIR", "outputs/external_validation")) / "qc"


def load_existing_symbols() -> set[str]:
    symbols = set()
    if OLD_MAP.exists():
        with OLD_MAP.open("r", encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                symbols.add(row["gene_symbol"].upper())
    return symbols


def verify_prefix_until_all_targets(max_lines: int = 5_000_000) -> dict:
    target_set = set(ev.TARGET_GENES)
    found_symbols: set[str] = set()
    found_rows = 0
    header = None
    idx = {}
    in_table = False
    lines_read = 0
    with gzip.open(FULL_GPL, "rt", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            lines_read += 1
            line = line.rstrip("\n")
            if line == "!platform_table_begin":
                in_table = True
                continue
            if in_table and line == "!platform_table_end":
                break
            if not in_table:
                continue
            row = line.split("\t")
            if header is None:
                header = row
                idx = {name: i for i, name in enumerate(header)}
                if "ID" not in idx or "gene_assignment" not in idx:
                    raise RuntimeError("GPL5188 table lacks ID or gene_assignment columns")
                continue
            if len(row) <= max(idx["ID"], idx["gene_assignment"]):
                continue
            matched = ev.symbols_from_gpl5188_gene_assignment(row[idx["gene_assignment"]]) & target_set
            if matched:
                found_rows += 1
                found_symbols.update(matched)
                if target_set.issubset(found_symbols):
                    break
            if lines_read >= max_lines:
                break
    return {
        "full_gpl_path": str(FULL_GPL),
        "full_gpl_exists": FULL_GPL.exists(),
        "full_gpl_size_bytes": FULL_GPL.stat().st_size if FULL_GPL.exists() else None,
        "lines_read": lines_read,
        "found_target_rows_in_scan": found_rows,
        "target_genes_total": len(target_set),
        "target_genes_found_in_scan": len(found_symbols),
        "target_genes_missing_in_scan": sorted(target_set - found_symbols),
        "existing_mapping_symbols_total": len(load_existing_symbols()),
        "existing_mapping_symbols": sorted(load_existing_symbols()),
    }


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    result = verify_prefix_until_all_targets()
    (OUT / "GPL5188_full_file_target_verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
