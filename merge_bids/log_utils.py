import os
from pathlib import Path
from datetime import datetime

# LLM Code
# Helper functions for logging

# Base directory of these scripts
BASE_DIR = Path(__file__).resolve().parent

def get_log_path() -> Path:
    return BASE_DIR / "log.txt"

def log(message: str) -> None:
    log_path = get_log_path()
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"[{ts}] {message}\n")

def crash(message: str):
    # Print helpful message then exit with failure
    print(f"ERROR: {message}")
    raise SystemExit(1)
