from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import h5py


DEFAULT_DATASETS = {
    "breast_cancer_cytassist_ffpe": Path("data/raw/spatial_visium/breast_cancer_cytassist_ffpe"),
    "human_lymph_node": Path("data/raw/spatial_visium/human_lymph_node"),
}

DEFAULT_OUTPUT_DIR = Path("results/revision_spatial_validation")


def inspect_h5_matrix(path: Path) -> dict:
    result = {
        "matrix_h5_exists": path.exists(),
        "matrix_h5_path": str(path),
        "matrix_h5_size_mb": path.stat().st_size / (1024 ** 2) if path.exists() else None,
        "h5_top_level_keys": "",
        "h5_shape": "",
        "feature_preview": "",
        "barcode_preview": "",
    }

    if not path.exists():
        return result

    try:
        with h5py.File(path, "r") as h5:
            result["h5_top_level_keys"] = "; ".join(list(h5.keys()))

            # 10x feature-barcode H5 usually stores /matrix
            if "matrix" in h5:
                matrix = h5["matrix"]
                result["h5_shape"] = str(matrix["shape"][:].tolist())

                if "features" in matrix:
                    features = matrix["features"]
                    if "name" in features:
                        names = features["name"][:10]
                        result["feature_preview"] = "; ".join(
                            x.decode() if isinstance(x, bytes) else str(x)
                            for x in names
                        )

                if "barcodes" in matrix:
                    barcodes = matrix["barcodes"][:10]
                    result["barcode_preview"] = "; ".join(
                        x.decode() if isinstance(x, bytes) else str(x)
                        for x in barcodes
                    )

    except Exception as exc:
        result["h5_error"] = str(exc)

    return result


def find_spatial_files(folder: Path) -> dict:
    spatial = folder / "spatial"

    candidates = {
        "spatial_folder_exists": spatial.exists(),
        "tissue_positions_csv": spatial / "tissue_positions.csv",
        "tissue_positions_list_csv": spatial / "tissue_positions_list.csv",
        "scalefactors_json": spatial / "scalefactors_json.json",
        "tissue_lowres_image": spatial / "tissue_lowres_image.png",
        "tissue_hires_image": spatial / "tissue_hires_image.png",
    }

    result = {
        "spatial_folder_exists": spatial.exists(),
    }

    for key, path in candidates.items():
        if key == "spatial_folder_exists":
            continue

        result[f"{key}_exists"] = path.exists()
        result[f"{key}_path"] = str(path)
        result[f"{key}_size_mb"] = path.stat().st_size / (1024 ** 2) if path.exists() else None

    return result


def inspect_positions(folder: Path) -> dict:
    spatial = folder / "spatial"

    positions_file = None
    for name in ["tissue_positions.csv", "tissue_positions_list.csv"]:
        candidate = spatial / name
        if candidate.exists():
            positions_file = candidate
            break

    result = {
        "positions_file": str(positions_file) if positions_file else "",
        "positions_shape": "",
        "positions_columns": "",
        "n_spots_total": None,
        "n_spots_in_tissue": None,
    }

    if positions_file is None:
        return result

    try:
        # Newer Space Ranger has header, older tissue_positions_list.csv often does not.
        if positions_file.name == "tissue_positions_list.csv":
            df = pd.read_csv(
                positions_file,
                header=None,
                names=[
                    "barcode",
                    "in_tissue",
                    "array_row",
                    "array_col",
                    "pxl_row_in_fullres",
                    "pxl_col_in_fullres",
                ],
            )
        else:
            df = pd.read_csv(positions_file)

        result["positions_shape"] = str(df.shape)
        result["positions_columns"] = "; ".join(df.columns.astype(str).tolist())
        result["n_spots_total"] = int(df.shape[0])

        if "in_tissue" in df.columns:
            result["n_spots_in_tissue"] = int(pd.to_numeric(df["in_tissue"], errors="coerce").fillna(0).sum())

    except Exception as exc:
        result["positions_error"] = str(exc)

    return result


def inspect_scalefactors(folder: Path) -> dict:
    path = folder / "spatial" / "scalefactors_json.json"

    result = {
        "scalefactors_file": str(path),
        "scalefactors_exists": path.exists(),
        "scalefactors_keys": "",
        "spot_diameter_fullres": None,
        "tissue_hires_scalef": None,
        "tissue_lowres_scalef": None,
    }

    if not path.exists():
        return result

    try:
        data = json.loads(path.read_text())
        result["scalefactors_keys"] = "; ".join(data.keys())
        result["spot_diameter_fullres"] = data.get("spot_diameter_fullres")
        result["tissue_hires_scalef"] = data.get("tissue_hires_scalef")
        result["tissue_lowres_scalef"] = data.get("tissue_lowres_scalef")

    except Exception as exc:
        result["scalefactors_error"] = str(exc)

    return result


def inspect_dataset(name: str, folder: Path) -> dict:
    print("=" * 100)
    print(f"[DATASET] {name}")
    print(f"Folder: {folder}")

    matrix_path = folder / "filtered_feature_bc_matrix.h5"

    result = {
        "dataset": name,
        "folder": str(folder),
        "folder_exists": folder.exists(),
    }

    result.update(inspect_h5_matrix(matrix_path))
    result.update(find_spatial_files(folder))
    result.update(inspect_positions(folder))
    result.update(inspect_scalefactors(folder))

    for key, value in result.items():
        print(f"{key}: {value}")

    return result


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )

    parser.add_argument(
        "--breast-dir",
        type=Path,
        default=DEFAULT_DATASETS["breast_cancer_cytassist_ffpe"],
    )

    parser.add_argument(
        "--lymph-dir",
        type=Path,
        default=DEFAULT_DATASETS["human_lymph_node"],
    )

    args = parser.parse_args()

    table_dir = args.output_dir / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)

    datasets = {
        "breast_cancer_cytassist_ffpe": args.breast_dir,
        "human_lymph_node": args.lymph_dir,
    }

    records = []
    for name, folder in datasets.items():
        records.append(inspect_dataset(name, folder))

    summary = pd.DataFrame(records)
    output_path = table_dir / "spatial_visium_input_inspection.csv"
    summary.to_csv(output_path, index=False)

    print("\n[SAVED]")
    print(output_path)


if __name__ == "__main__":
    main()