from pathlib import Path
import pandas as pd


def read_table(path: str | Path, **kwargs) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    if path.suffix == ".csv":
        return pd.read_csv(path, **kwargs)
    if path.suffix in [".tsv", ".txt"]:
        return pd.read_csv(path, sep="\t", **kwargs)
    if path.suffix == ".parquet":
        return pd.read_parquet(path, **kwargs)

    raise ValueError(f"Unsupported file type: {path}")
