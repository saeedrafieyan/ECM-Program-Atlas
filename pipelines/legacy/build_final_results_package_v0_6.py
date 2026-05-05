from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd


OUTPUT_ROOT = Path("outputs")
BASE_PACKAGE_DIR = OUTPUT_ROOT / "final_results_v0.5"
V06_DIR = OUTPUT_ROOT / "final_results_v0.6"

RANK_DIR = OUTPUT_ROOT / "rank_based_ecm_program_scoring"
RANK_TABLE_DIR = RANK_DIR / "tables"
RANK_FIG_HTML_DIR = RANK_DIR / "figures" / "html"
RANK_FIG_PNG_DIR = RANK_DIR / "figures" / "png"
RANK_REPORT_DIR = RANK_DIR / "reports"


RANK_TABLES = {
    "Table_25_rank_based_all_rank_vs_mean_score_correlation": RANK_TABLE_DIR / "rank_based_all_rank_vs_mean_score_correlation.csv",
    "Table_26_gtex_v11_rank_based_tissue_summary": RANK_TABLE_DIR / "gtex_v11_rank_based_tissue_summary.csv",
    "Table_27_tabula_sapiens_rank_based_compartment_summary": RANK_TABLE_DIR / "tabula_sapiens_rank_based_compartment_summary.csv",
    "Table_28_tabula_sapiens_rank_based_celltype_summary": RANK_TABLE_DIR / "tabula_sapiens_rank_based_celltype_summary.csv",
    "Supplementary_Table_gtex_v11_rank_based_tissue_detail_summary": RANK_TABLE_DIR / "gtex_v11_rank_based_tissue_detail_summary.csv",
    "Supplementary_Table_tabula_sapiens_rank_based_organ_celltype_summary": RANK_TABLE_DIR / "tabula_sapiens_rank_based_organ_celltype_summary.csv",
    "Supplementary_Table_gtex_v11_rank_based_gene_availability": RANK_TABLE_DIR / "gtex_v11_rank_based_program_gene_availability.csv",
    "Supplementary_Table_tabula_sapiens_rank_based_gene_availability": RANK_TABLE_DIR / "tabula_sapiens_rank_based_program_gene_availability.csv",
}

RANK_REPORTS = {
    "rank_based_ecm_program_scoring_summary": RANK_REPORT_DIR / "rank_based_ecm_program_scoring_summary.md",
}

RANK_FIGURES_PNG = {
    "figure_13a_rank_vs_mean_score_spearman_correlation": RANK_FIG_PNG_DIR / "rank_vs_mean_score_spearman_correlation.png",
    "figure_13b_gtex_v11_rank_based_tissue_program_heatmap": RANK_FIG_PNG_DIR / "gtex_v11_rank_based_tissue_program_heatmap.png",
    "figure_13c_tabula_sapiens_rank_based_compartment_heatmap_10X": RANK_FIG_PNG_DIR / "tabula_sapiens_rank_based_compartment_heatmap_10X.png",
    "figure_13d_tabula_sapiens_rank_based_compartment_heatmap_smartseq2": RANK_FIG_PNG_DIR / "tabula_sapiens_rank_based_compartment_heatmap_smartseq2.png",
}

RANK_FIGURES_HTML = {
    "figure_13a_rank_vs_mean_score_spearman_correlation": RANK_FIG_HTML_DIR / "rank_vs_mean_score_spearman_correlation.html",
    "figure_13b_gtex_v11_rank_based_tissue_program_heatmap": RANK_FIG_HTML_DIR / "gtex_v11_rank_based_tissue_program_heatmap.html",
    "figure_13c_tabula_sapiens_rank_based_compartment_heatmap_10X": RANK_FIG_HTML_DIR / "tabula_sapiens_rank_based_compartment_heatmap_10X.html",
    "figure_13d_tabula_sapiens_rank_based_compartment_heatmap_smartseq2": RANK_FIG_HTML_DIR / "tabula_sapiens_rank_based_compartment_heatmap_smartseq2.html",
}


def reset_v06_dir() -> None:
    if V06_DIR.exists():
        shutil.rmtree(V06_DIR)

    if not BASE_PACKAGE_DIR.exists():
        raise FileNotFoundError(
            f"Missing base package: {BASE_PACKAGE_DIR}. "
            "Run src/build_final_results_package_v0_5.py first."
        )

    shutil.copytree(BASE_PACKAGE_DIR, V06_DIR)

    for folder in [
        V06_DIR / "tables" / "rank_based_scoring",
        V06_DIR / "figures" / "rank_based_scoring" / "png",
        V06_DIR / "figures" / "rank_based_scoring" / "html",
        V06_DIR / "reports" / "rank_based_scoring",
        V06_DIR / "metadata",
    ]:
        folder.mkdir(parents=True, exist_ok=True)


def copy_file(src: Path, dst: Path) -> str:
    if not src.exists():
        print(f"[MISSING] {src}")
        return "missing"

    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    print(f"[COPIED] {src} -> {dst}")
    return "copied"


def collect_group(files: dict[str, Path], output_subdir: Path, suffix: str, file_type: str) -> list[dict]:
    records = []

    for name, src in files.items():
        dst = output_subdir / f"{name}{suffix}"
        status = copy_file(src, dst)

        records.append(
            {
                "logical_name": name,
                "type": file_type,
                "source_path": str(src),
                "final_path": str(dst) if status == "copied" else "",
                "status": status,
            }
        )

    return records


def collect_v06_outputs() -> pd.DataFrame:
    records = []

    records.extend(
        collect_group(
            RANK_TABLES,
            V06_DIR / "tables" / "rank_based_scoring",
            ".csv",
            "rank_based_table",
        )
    )

    records.extend(
        collect_group(
            RANK_REPORTS,
            V06_DIR / "reports" / "rank_based_scoring",
            ".md",
            "rank_based_report",
        )
    )

    records.extend(
        collect_group(
            RANK_FIGURES_PNG,
            V06_DIR / "figures" / "rank_based_scoring" / "png",
            ".png",
            "rank_based_figure_png",
        )
    )

    records.extend(
        collect_group(
            RANK_FIGURES_HTML,
            V06_DIR / "figures" / "rank_based_scoring" / "html",
            ".html",
            "rank_based_figure_html",
        )
    )

    manifest = pd.DataFrame(records)
    manifest.to_csv(V06_DIR / "metadata" / "rank_based_v0.6_manifest.csv", index=False)

    return manifest


def safe_read_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None

    try:
        return pd.read_csv(path)
    except Exception as exc:
        print(f"[WARNING] Could not read {path}: {exc}")
        return None


def make_key_numbers(manifest: pd.DataFrame) -> dict:
    numbers = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "project_version": "v0.6",
        "description": "v0.5 plus rank-based ECM program scoring robustness analysis",
        "rank_based_files_copied": int((manifest["status"] == "copied").sum()),
        "rank_based_files_missing": int((manifest["status"] == "missing").sum()),
    }

    corr = safe_read_csv(
        V06_DIR / "tables" / "rank_based_scoring" / "Table_25_rank_based_all_rank_vs_mean_score_correlation.csv"
    )

    if corr is not None:
        numbers["n_rank_correlation_rows"] = int(corr.shape[0])

        for dataset in sorted(corr["dataset"].dropna().unique()):
            subset = corr[corr["dataset"].eq(dataset)]
            numbers[f"{dataset}_mean_spearman_r"] = float(subset["spearman_r"].mean())
            numbers[f"{dataset}_min_spearman_r"] = float(subset["spearman_r"].min())
            numbers[f"{dataset}_max_spearman_r"] = float(subset["spearman_r"].max())

    with open(V06_DIR / "metadata" / "key_result_numbers_v0.6.json", "w", encoding="utf-8") as f:
        json.dump(numbers, f, indent=2)

    return numbers


def write_v06_report(numbers: dict) -> None:
    report = f"""# ECM Latent Space Project, Final Results Package v0.6

## Purpose

This package extends v0.5 by adding rank-based ECM program scoring robustness analysis.

## Package creation

- Created at: {numbers.get("created_at")}
- Version: {numbers.get("project_version")}
- Rank-based files copied: {numbers.get("rank_based_files_copied")}
- Rank-based files missing: {numbers.get("rank_based_files_missing")}

## What changed from v0.5 to v0.6

v0.5 established the transcriptomic/proteomic/GTEx/Tabula Sapiens ECM representation framework.

v0.6 adds method robustness by testing whether the nine curated ECM programs remain stable when scored using within-sample rank-based scoring rather than mean expression.

## Rank-based scoring methods

The analysis computed:

1. rank_percentile_score: mean within-sample percentile rank of genes in an ECM program.
2. top10_fraction_score: fraction of program genes ranked in the top 10 percent of Matrisome genes.
3. top20_fraction_score: fraction of program genes ranked in the top 20 percent of Matrisome genes.

## Key correlation numbers

- GTEx V11 mean Spearman correlation: {numbers.get("gtex_v11_mean_spearman_r", "NA")}
- GTEx V11 minimum Spearman correlation: {numbers.get("gtex_v11_min_spearman_r", "NA")}
- GTEx V11 maximum Spearman correlation: {numbers.get("gtex_v11_max_spearman_r", "NA")}

- Tabula Sapiens mean Spearman correlation: {numbers.get("tabula_sapiens_mean_spearman_r", "NA")}
- Tabula Sapiens minimum Spearman correlation: {numbers.get("tabula_sapiens_min_spearman_r", "NA")}
- Tabula Sapiens maximum Spearman correlation: {numbers.get("tabula_sapiens_max_spearman_r", "NA")}

## Main interpretation

Most ECM programs remain stable under rank-based scoring, supporting the robustness of the curated ECM program representation.

Programs such as vascular/stromal/interstitial ECM, epithelial/mucosal basement membrane ECM, immune/lymphoid remodeling ECM, stromal remodeling ECM, renal/endothelial basement membrane ECM, and reproductive-specialized ECM are especially robust.

CNS/neural, retinal/sensory, and hepatic/plasma-associated programs show more context-dependent behavior and should be interpreted more cautiously.

## Why this matters

This analysis reduces dependence on one scoring method and aligns the project conceptually with rank-based gene-set scoring approaches such as UCell, used in MatriSpace for spatial matrisome analysis.

## Suggested manuscript addition

Add this as a methodological robustness section:

“Rank-based scoring confirms stability of most curated ECM programs.”

Potential Figure 13:

- Figure 13A: Spearman correlation between mean-expression and rank-based scores.
- Figure 13B: GTEx V11 rank-based tissue program heatmap.
- Figure 13C-D: Tabula Sapiens rank-based compartment heatmaps.

## Next recommended step

Stop expanding and begin manuscript consolidation unless a focused spatial validation using MatriSpace-like analysis is explicitly required.
"""

    report_path = V06_DIR / "reports" / "final_results_v0.6_summary.md"
    report_path.write_text(report, encoding="utf-8")
    print(f"[SAVED] {report_path}")


def main() -> None:
    reset_v06_dir()
    manifest = collect_v06_outputs()
    numbers = make_key_numbers(manifest)
    write_v06_report(numbers)

    print("\n[DONE]")
    print(f"Final v0.6 package: {V06_DIR}")
    print(f"Manifest: {V06_DIR / 'metadata' / 'rank_based_v0.6_manifest.csv'}")
    print(f"Report: {V06_DIR / 'reports' / 'final_results_v0.6_summary.md'}")


if __name__ == "__main__":
    main()