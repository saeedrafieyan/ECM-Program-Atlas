from pathlib import Path
import pandas as pd
from .utils import normalize_gene_symbol


def load_human_matrisome(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    df = pd.read_excel(path, header=1)
    df.columns = [str(c).strip() for c in df.columns]
    if "Gene Symbol" not in df.columns:
        raise ValueError("Expected column 'Gene Symbol' in Matrisome file.")
    df["gene_symbol_clean"] = normalize_gene_symbol(df["Gene Symbol"])
    return df.dropna(subset=["gene_symbol_clean"]).copy()
