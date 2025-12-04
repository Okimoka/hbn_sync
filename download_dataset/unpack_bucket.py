import os
import tarfile
import shutil
import logging
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

"""
This unpacks the tar.gzs downloaded using dl_bucket.py
It assumes the default folder name (bucket_tars).
The staging area again avoids artifacts/incomplete results.
Final results will be in bucket_tars_unpacked.
When the script aborts and is restarted, it will also clear the staging area fully before continuing.

LLM Code
"""

# --- Configuration ---
SOURCE_DIR = Path("./bucket_tars")
DEST_DIR = Path("./bucket_tars_unpacked")
STAGING_DIR = Path("./bucket_tars_unpacked.staging")
MAX_WORKERS = 4 

# --- Setup Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("unpack_log.txt"),
        logging.StreamHandler()
    ]
)

def reset_staging_area():
    """
    Completely wipes and recreates the staging directory.
    This ensures that if the script was aborted previously, 
    no garbage remains.
    """
    if STAGING_DIR.exists():
        logging.info("Cleaning up previous staging area...")
        shutil.rmtree(STAGING_DIR)
    STAGING_DIR.mkdir(parents=True, exist_ok=True)
    
    # Ensure dest exists too
    DEST_DIR.mkdir(parents=True, exist_ok=True)

def get_expected_foldername(tar_filename):
    """
    Converts 'NDAREV593YN4.tar.gz' -> 'NDAREV593YN4'
    """
    # .stem on .tar.gz usually returns .tar, so we might need to do it twice
    # or just replace the string if we know the format is consistent.
    name = tar_filename.name
    if name.endswith(".tar.gz"):
        return name[:-7]
    elif name.endswith(".tgz"):
        return name[:-4]
    else:
        return tar_filename.stem

def process_tarball(tar_path):
    tar_filename = tar_path.name
    expected_folder_name = get_expected_foldername(tar_path)
    final_dest_path = DEST_DIR / expected_folder_name

    # --- OPTIMIZATION: Check existance BEFORE extraction ---
    if final_dest_path.exists():
        return True, f"{tar_filename}: Skipped. '{expected_folder_name}' already exists in destination."

    # Unique staging path for this worker
    current_staging = STAGING_DIR / expected_folder_name
    
    try:
        if current_staging.exists():
            shutil.rmtree(current_staging)
        current_staging.mkdir()

        # Extract
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(path=current_staging)

        # Verification
        items = list(current_staging.iterdir())
        if len(items) != 1:
            return False, f"{tar_filename}: Failed. Found {len(items)} items, expected 1."
        
        extracted_item = items[0]
        if not extracted_item.is_dir():
            return False, f"{tar_filename}: Failed. Extracted item is a file, not a folder."

        # Double check: Does the extracted folder match the expected name?
        # If the tarball filename is 'A.tar.gz' but it contains folder 'B',
        # we need to handle that rename or move logic.
        actual_dest_path = DEST_DIR / extracted_item.name
        
        # Check destination AGAIN (in case the internal name differs from the filename)
        if actual_dest_path.exists():
             return False, f"{tar_filename}: Skipped. Destination '{extracted_item.name}' already exists."

        shutil.move(str(extracted_item), str(actual_dest_path))
        
        return True, f"{tar_filename}: Success -> {extracted_item.name}"

    except Exception as e:
        return False, f"{tar_filename}: Error - {str(e)}"
    
    finally:
        # Cleanup worker's staging folder
        if current_staging.exists():
            shutil.rmtree(current_staging)

def main():
    # 1. Verification and Cleanup on Start
    if not SOURCE_DIR.exists():
        logging.error(f"Source directory {SOURCE_DIR} missing.")
        return

    reset_staging_area()
    
    tar_files = list(SOURCE_DIR.glob("*.tar.gz"))
    total_files = len(tar_files)
    logging.info(f"Found {total_files} files. Processing with {MAX_WORKERS} workers...")

    success_count = 0
    skipped_count = 0
    error_count = 0

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_tarball, f): f for f in tar_files}
        
        for i, future in enumerate(as_completed(futures), 1):
            success, message = future.result()
            
            if success:
                if "Skipped" in message:
                    skipped_count += 1
                    # Log skipped at DEBUG or INFO depending on how verbose you want to be
                    logging.info(f"[{i}/{total_files}] {message}") 
                else:
                    success_count += 1
                    logging.info(f"[{i}/{total_files}] {message}")
            else:
                error_count += 1
                logging.error(f"[{i}/{total_files}] {message}")

    logging.info("--- Processing Complete ---")
    logging.info(f"Extracted: {success_count}")
    logging.info(f"Skipped (Already Existed): {skipped_count}")
    logging.info(f"Errors: {error_count}")
    
    # Final cleanup of staging root
    if STAGING_DIR.exists() and not any(STAGING_DIR.iterdir()):
        STAGING_DIR.rmdir()

if __name__ == "__main__":
    main()