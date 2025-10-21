import csv
import json
from pathlib import Path
import pandas as pd

from .log_utils import crash

# LLM Code

def sniff_delimiter(file_path: Path, sample_size: int = 10000):
    """
    Probe the delimiter for CSV/TSV-like files. Never assume comma/tab.
    """
    with open(file_path, "r", encoding="utf-8", newline="") as f:
        sample = f.read(sample_size)
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=[',','\t',';','|',':'])
        return dialect.delimiter
    except Exception:
        crash(f"Could not detect delimiter for file: {file_path}")

def read_table_with_sniff(file_path: Path) -> pd.DataFrame:
    delim = sniff_delimiter(file_path)
    try:
        df = pd.read_csv(file_path, delimiter=delim, dtype=str, keep_default_na=False)
        return df
    except Exception as e:
        crash(f"Failed reading table {file_path} with detected delimiter '{delim}': {e}")

def write_table(df, out_path: Path, delimiter: str = '\t'):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, sep=delimiter, index=False, encoding="utf-8")

def read_json(path: Path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        crash(f"Missing required JSON file: {path}")
    except json.JSONDecodeError as e:
        crash(f"Invalid JSON in {path}: {e}")

def write_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def read_text(path: Path) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        crash(f"Missing required text file: {path}")

def write_text(text: str, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
